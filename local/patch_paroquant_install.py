#!/usr/bin/env python3
"""Local patch (ours): install the fork's ParoQuant vLLM plugin into the serving image at container start,
so serve-mxfp4.sh (our MTP/KV-pinned launcher) can serve a `quant_method: paroquant` checkpoint. This is the
install half of paroquant/run_paroquant.sh, minus its dflash-only serve flags and minus the paroquant_mxfp4
variant (it imports radiance_mxfp4, which this image does not ship in site-packages, and an int5 checkpoint
never needs it).

GATED: only acts when RADIANCE_PAROQUANT_INSTALL=1 (pass via the deploy wrapper's EXTRA_ENV). An MXFP4 serve
never sees it.

Installs: prebuilt radiance_paroquant_kernel.so (persist/paroquant/build.sh) + radiance_paroquant.py, and the
guarded sitecustomize import that registers the `paroquant` quantization config in every process (appended to
the STDLIB sitecustomize: Ubuntu's /usr/lib/python3.12/sitecustomize.py shadows any site-packages one).

FP8 HEADS (ours): the installed copy of the plugin is source-patched so that a checkpoint whose
quantization_config carries `"fp8_heads": ["*lm_head", "mtp.*"]` routes those prefixes to Quark's FP8
per-output-channel scheme (the exact scheme prod's fp8 lm_head / fp8 MTP head use, incl. the local quark.py
ParallelLMHead patch) while the int5 decoder keeps ParoQuantLinearMethod. Without that key the plugin behaves
exactly as shipped. Tensor layout expected: <name>.weight F8_E4M3 [N,K] + <name>.weight_scale F32 [N]
(custom-quant/fp8_lmhead_sharded.py + graft_mtp.py --head-dtype fp8 write it).
"""
import os, shutil, sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_PAROQUANT_INSTALL", "0") != "1":
    print("[local patch] paroquant install: skipped (RADIANCE_PAROQUANT_INSTALL != 1)")
    raise SystemExit(0)

SP = Path(sysconfig.get_paths()["purelib"])
assert "vllm" in str(SP), f"unexpected site-packages {SP}"
P = Path("/patches")
so = P / "persist/paroquant/radiance_paroquant_kernel.so"
# vLLM 0.29 image (torch 2.11 / ROCm 7.14 from-source build, 2026-09-17): use the kernel hipcc-built inside that image
_alt = P / "persist/paroquant/radiance_paroquant_kernel-0.29.so"
if any(SP.glob("vllm-0.29*")) and _alt.exists():
    so = _alt
# local A/B (2026-09-28): RADIANCE_LOCAL_PQ_SO=<file in persist/paroquant> overrides the kernel .so (e.g. the -mr3 build)
_ov = os.environ.get("RADIANCE_LOCAL_PQ_SO", "")
if _ov:
    so = P / "persist/paroquant" / _ov
    print(f"[local patch] paroquant install: kernel override {so}")
assert so.exists(), "prebuilt paroquant kernel missing: run persist/paroquant/build.sh"
py = P / "paroquant/radiance_paroquant.py"
assert py.exists(), f"missing {py}"

def _install(src: Path, dst: Path):
    # temp + rename, never in place: on a container RESTART the sitecustomize import below is already active,
    # so this very process has the kernel .so mapped -- truncating it in place segfaulted the interpreter at exit
    # (crash-looped the container on 2026-09-16). A rename leaves the mapped inode alone.
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

_install(so, SP / "radiance_paroquant_kernel.so")
_install(py, SP / py.name)
pym = P / "paroquant/radiance_paroquant_mxfp4.py"      # rotation-MXFP4 variant (quant_method paroquant_mxfp4)
assert pym.exists(), f"missing {pym}"
_install(pym, SP / pym.name)

# ---- fp8 heads source patch on the installed copy -------------------------------------------------
def _patch_plugin(dst: Path, register_anchor: str):
    s = dst.read_text()
    HELPER = '''
# ---- local patch: fp8 heads via Quark (lm_head / MTP projections) --------------------------------
import fnmatch as _fnmatch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.utils import set_weight_attrs

def _pq_fp8_spec(dtype, dynamic, qscheme, ch_axis):
    return {"block_size": None, "ch_axis": ch_axis, "dtype": dtype, "enable_buffer_reuse": False,
            "group_size": None, "is_dynamic": dynamic, "is_scale_quant": False,
            "max_input_numel": 4194304, "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver" if qscheme == "per_channel" else "PerTensorMinMaxObserver",
            "qscheme": qscheme, "round_method": "half_even", "scale_calculation_mode": None,
            "scale_format": None, "scale_type": "float", "symmetric": True}

class _PQFp8RowEmbeddingMethod(QuantizeMethodBase):
    """local patch: embed_tokens stored as fp8 e4m3 rows + fp32 per-row scale (custom-quant/fp8_embed_sharded.py);
    dequantized at lookup, so the 2.4 GiB bf16 table becomes 1.2 GiB. Only the token embedding, never the LM head."""
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        n = sum(output_partition_sizes)
        w = torch.nn.Parameter(torch.empty(n, input_size_per_partition, dtype=torch.float8_e4m3fn), requires_grad=False)
        set_weight_attrs(w, {"input_dim": 1, "output_dim": 0}); set_weight_attrs(w, extra_weight_attrs)
        layer.register_parameter("weight", w)
        sc = torch.nn.Parameter(torch.ones(n, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(sc, {"output_dim": 0}); set_weight_attrs(sc, extra_weight_attrs)
        layer.register_parameter("weight_scale", sc)
    def process_weights_after_loading(self, layer):
        # local patch (2026-09-29, PQ_EMBED_UVA=1): keep the 1.2 GiB fp8 table in pinned host memory; the lookup below gathers
        # its rows over PCIe through a UVA view (~50 KB per decode step, ~42 MB per 8k prefill chunk) and the VRAM goes to the
        # KV pool. Deliberately not a RADIANCE_* name: where the table lives never changes a traced graph, so it must not
        # re-key the AOT compile cache (patch_aot_envkey.py hashes RADIANCE_*).
        import os
        if os.environ.get("PQ_EMBED_UVA", "0") != "1" or layer.weight.device.type == "cpu":
            return
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        host = layer.weight.data.to("cpu").pin_memory()
        layer.weight.data = get_accelerator_view_from_cpu_tensor(host)
        layer._pq_uva_host = host   # keep the pinned buffer alive whether or not the view owns it
        print(f"[paroquant] fp8 embed_tokens -> pinned host memory via UVA ({host.numel() / 2**30:.2f} GiB off the GPU)", flush=True)
    def embedding(self, layer, input_):
        rows = layer.weight.view(torch.uint8)[input_].view(torch.float8_e4m3fn)
        return rows.to(torch.bfloat16) * layer.weight_scale[input_].unsqueeze(-1).to(torch.bfloat16)
    def apply(self, layer, x, bias=None):
        raise NotImplementedError("fp8 row embedding is lookup-only")

_PQ_FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
               "weight": _pq_fp8_spec("fp8_e4m3", False, "per_channel", 0),
               "input_tensors": _pq_fp8_spec("fp8_e4m3", True, "per_tensor", -1)}

# ---- local patch (2026-09-28): MTP drafter linears in MXFP4 (RADIANCE_MTP_MXFP4=1) -------------------------------
# The drafter's layer (fc + qkv/o + gate_up/down) streams ~423 MB of fp8 weights per draft pass through hipBLASLt, 4 passes
# per decode step. Here the fp8 checkpoint weights are requantized ONCE at load to MXFP4 (e2m1 + e8m0 per 32, the scale
# exponent picked per block from {OCP floor, floor+1} by squared error) and served by the same W4A8 kernel as the body
# (radiance_mxfp4 ext): half the bytes. Drafts only -- the target verifies every token, so the output distribution is
# unchanged; only acceptance can move. Quark's create_weights is kept, so loading/sharding are untouched.
import os as _pq_os
_PQ_MTP_MXFP4 = _pq_os.environ.get("RADIANCE_MTP_MXFP4", "0") == "1"
# prefixes containing any of these comma-separated substrings stay fp8 (sensitivity A/Bs, e.g. "self_attn" or "fc,self_attn")
_PQ_MTP_MXFP4_SKIP = [t for t in _pq_os.environ.get("RADIANCE_MTP_MXFP4_SKIP", "").split(",") if t]
_PQ_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# RADIANCE_MTP_MXFP4_FILE=<.pt>: {vLLM prefix: {"codes": u8 [N, K/2], "e8": u8 [N, K/32]}} from mtp_gptq.py replaces the
# load-time round-to-nearest for the prefixes it holds (shape-checked; anything else falls back to RTN).
_PQ_MTP_MXFP4_FILE = _pq_os.environ.get("RADIANCE_MTP_MXFP4_FILE", "")
_PQ_MTP_FILE_CACHE = {}


def _pq_mtp_file():
    if "d" not in _PQ_MTP_FILE_CACHE:
        _PQ_MTP_FILE_CACHE["d"] = torch.load(_PQ_MTP_MXFP4_FILE, map_location="cpu")
    return _PQ_MTP_FILE_CACHE["d"]


def _pq_mxfp4_quant(w):
    """float32 [N, K] (K % 32 == 0) -> (codes uint8 [N, K/2], e8m0 uint8 [N, K/32]); element 2i in the low nibble."""
    N, K = w.shape
    grid = torch.tensor(_PQ_E2M1, device=w.device)
    codes = torch.empty((N, K // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((N, K // 32), dtype=torch.uint8, device=w.device)
    rows = max(1, (1 << 27) // (K * 8))
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].reshape(-1, K // 32, 32)
        amax = b.abs().amax(-1).clamp_min(2.0 ** -120)
        e_lo = torch.floor(torch.log2(amax)) - 2.0
        best_err, best_idx, best_e = None, None, None
        for de in (0.0, 1.0):
            e = (e_lo + de).clamp(-127.0, 127.0)
            q = b / torch.exp2(e).unsqueeze(-1)
            idx = (q.abs().clamp(max=6.0).unsqueeze(-1) - grid).abs().argmin(-1)
            err = ((grid[idx] * torch.sign(q) - q) * torch.exp2(e).unsqueeze(-1)).square().sum(-1)
            if best_err is None:
                best_err, best_idx, best_e = err, idx | ((q < 0).to(idx.dtype) << 3), e
            else:
                pick = err < best_err
                best_err = torch.where(pick, err, best_err)
                best_idx = torch.where(pick.unsqueeze(-1), idx | ((q < 0).to(idx.dtype) << 3), best_idx)
                best_e = torch.where(pick, e, best_e)
        nib = best_idx.reshape(b.shape[0], K).to(torch.uint8)
        codes[r0:r0 + rows] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scales[r0:r0 + rows] = (best_e + 127.0).to(torch.uint8)
    return codes, scales


def _pq_mtp_mx_linear(x: torch.Tensor, weight: torch.Tensor, ws: torch.Tensor, wref: torch.Tensor) -> torch.Tensor:
    """One opaque op for a drafter linear: per-token fp8 quant + the W4A8 MXFP4 GEMM (x [M, K] bf16 contiguous)."""
    import radiance_mxfp4 as _mx
    from vllm import _custom_ops as _ops
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    xq, xs = _ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    xs = xs.reshape(-1).float().contiguous()
    _mx._ext.launch(xq.data_ptr(), weight.data_ptr(), ws.data_ptr(), wref.data_ptr(), xs.data_ptr(),
                    out.data_ptr(), M, N, K, torch.cuda.current_stream().cuda_stream)
    return out


# both patched plugin modules carry this block: register once
if _PQ_MTP_MXFP4 and not hasattr(torch.ops.radiance, "mtp_mxfp4_linear"):
    _pq_mtp_op = torch.library.custom_op("radiance::mtp_mxfp4_linear", mutates_args=())(_pq_mtp_mx_linear)

    @_pq_mtp_op.register_fake
    def _(x, weight, ws, wref):
        return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


class _PQMtpMXFP4Method(QuantizeMethodBase):
    """Wraps the Quark fp8 per-channel method of one mtp.* linear: same weights at load, MXFP4 at run time."""
    def __init__(self, inner, prefix=""):
        self.inner = inner
        self.prefix = prefix

    def create_weights(self, layer, *args, **kwargs):
        return self.inner.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        import radiance_mxfp4 as _mx
        w8 = layer.weight.data                                  # fp8 e4m3 [N, K], Quark has not transposed it yet
        s = layer.weight_scale.data.float().reshape(-1)         # per output channel
        N, K = w8.shape
        if K % 64 or N % 16 or s.numel() != N:
            raise RuntimeError(f"RADIANCE_MTP_MXFP4: unsupported shape N={N} K={K} scales={s.numel()}")
        wf = w8.float() * s.unsqueeze(1)
        codes = None
        if _PQ_MTP_MXFP4_FILE:                                  # offline GPTQ codes (mtp_gptq.py), by vLLM prefix
            ent = _pq_mtp_file().get(self.prefix)
            if ent is not None and tuple(ent["codes"].shape) == (N, K // 2) and tuple(ent["e8"].shape) == (N, K // 32):
                # the codes are only valid for the exact fp8 weights they were fitted to: check the fingerprint
                import hashlib as _hl
                _h = _hl.sha1(w8.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                _h.update(s.float().contiguous().cpu().numpy().tobytes())
                if ent.get("src_sha1") == _h.hexdigest():
                    codes, e8, src = ent["codes"].to(wf.device), ent["e8"].to(wf.device), "gptq"
                else:
                    import sys as _s
                    _s.stderr.write(f"[radiance.mtp_mxfp4] {self.prefix}: {_PQ_MTP_MXFP4_FILE} was fitted to other weights "
                                    f"(fingerprint mismatch) -- using load-time RTN" + chr(10))
        if codes is None:
            codes, e8 = _pq_mxfp4_quant(wf)
            src = "rtn"
        # load-time report: weight error of the requantization, and the kernel against the dequantized weight
        lut = torch.tensor(_PQ_E2M1 + tuple(-v for v in _PQ_E2M1), device=wf.device)
        deq = lut[torch.stack([codes & 15, codes >> 4], -1).reshape(N, K).long()] \
            * torch.exp2(e8.float() - 127.0).repeat_interleave(32, dim=1)
        rel_w = float((deq - wf).norm() / wf.norm().clamp_min(1e-12))
        ws_t = e8.t().contiguous()
        wk = _mx.permute_w(codes, N, K) if _mx.WPERM else codes
        wref = _mx.make_row_ref(ws_t)
        for name in ("weight", "weight_scale", "input_scale"):
            if hasattr(layer, name):
                delattr(layer, name)
        layer.weight = torch.nn.Parameter(wk.contiguous(), requires_grad=False)
        layer.mx_ws = torch.nn.Parameter(ws_t, requires_grad=False)
        layer.mx_ref = torch.nn.Parameter(wref, requires_grad=False)
        x = torch.randn(4, K, device=wf.device, dtype=torch.bfloat16)
        ref = x.float() @ deq.t()
        got = self._run(layer, x).float()
        rel_k = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
        import sys as _s
        _s.stderr.write(f"[radiance.mtp_mxfp4] {self.prefix} {N}x{K} ({src}): weight relerr vs fp8 {rel_w:.4f}, kernel vs dequant "
                        f"{rel_k:.4f}, {w8.numel() >> 20} MiB fp8 -> {(wk.numel() + ws_t.numel()) >> 20} MiB" + chr(10))

    @staticmethod
    def _run(layer, x2):
        return torch.ops.radiance.mtp_mxfp4_linear(x2, layer.weight, layer.mx_ws, layer.mx_ref)

    def apply(self, layer, x, bias=None):
        out = self._run(layer, x.reshape(-1, x.shape[-1]).contiguous())
        if x.dim() != 2:
            out = out.view(*x.shape[:-1], out.shape[-1])
        return out + bias if bias is not None else out


# ---- local patch (2026-09-28): drafter input statistics for an offline GPTQ requant (RADIANCE_MTP_HCOLLECT=<dir>) ------
# Calibration run only, with --enforce-eager (the flag polling below is plain Python). The fp8 drafter runs unchanged;
# every mtp.* linear also accumulates its input second moment H = sum x^T x in fp32, split by call width (M <= 16 = the
# draft passes, else the drafter's prefill over prompt tokens). Accumulates while <dir>/ON exists; creating <dir>/SAVE
# writes <dir>/<prefix>.pt = {Hd, nd, Hp, np, w8, scale} and removes SAVE. mtp_gptq.py turns those into MXFP4 codes.
_PQ_MTP_HC = _pq_os.environ.get("RADIANCE_MTP_HCOLLECT", "")


class _PQMtpHCollect(QuantizeMethodBase):
    _all = []
    _state = {"calls": 0, "on": False}

    def __init__(self, inner, prefix=""):
        self.inner = inner
        self.prefix = prefix

    def create_weights(self, layer, *args, **kwargs):
        return self.inner.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        self.w8 = layer.weight.data.clone().cpu()                         # fp8 [N, K] as loaded (before Quark's transpose)
        self.scale = layer.weight_scale.data.float().reshape(-1).clone().cpu()
        K = self.w8.shape[1]
        self.inner.process_weights_after_loading(layer)
        dev = layer.weight.device
        self.hd = torch.zeros(K, K, dtype=torch.float32, device=dev)
        self.hp = torch.zeros(K, K, dtype=torch.float32, device=dev)
        self.nd = self.np_ = 0
        _PQMtpHCollect._all.append(self)
        import sys as _s
        _s.stderr.write(f"[radiance.mtp_hcollect] {self.prefix} {tuple(self.w8.shape)}: collecting into {_PQ_MTP_HC}" + chr(10))

    @classmethod
    def _poll(cls):
        st = cls._state
        st["on"] = _pq_os.path.exists(_pq_os.path.join(_PQ_MTP_HC, "ON"))
        sv = _pq_os.path.join(_PQ_MTP_HC, "SAVE")
        if _pq_os.path.exists(sv):
            torch.cuda.synchronize()
            for m in cls._all:
                torch.save({"Hd": m.hd.cpu(), "nd": m.nd, "Hp": m.hp.cpu(), "np": m.np_, "w8": m.w8, "scale": m.scale},
                           _pq_os.path.join(_PQ_MTP_HC, m.prefix + ".pt"))
            with open(_pq_os.path.join(_PQ_MTP_HC, "SAVED"), "w") as f:
                f.write(" ".join(f"{m.prefix}:{m.nd}/{m.np_}" for m in cls._all) + chr(10))
            _pq_os.remove(sv)

    def apply(self, layer, x, bias=None):
        st = _PQMtpHCollect._state
        st["calls"] += 1
        if st["calls"] % 32 == 0:
            _PQMtpHCollect._poll()
        if st["on"]:
            x2 = x.reshape(-1, x.shape[-1]).float()
            if x2.shape[0] <= 16:
                self.hd.addmm_(x2.t(), x2); self.nd += x2.shape[0]
            else:
                self.hp.addmm_(x2.t(), x2); self.np_ += x2.shape[0]
        return self.inner.apply(layer, x, bias)
# --------------------------------------------------------------------------------------------------

'''
    anchor = register_anchor
    assert anchor in s, "plugin register anchor not found"
    s = s.replace(anchor, HELPER + anchor, 1)

    old_init = "    def __init__(self, bits: int, group_size: int, krot: int, fp16_patterns: list[str]):\n        super().__init__()\n"
    new_init = ("    def __init__(self, bits: int, group_size: int, krot: int, fp16_patterns: list[str],\n"
                "                 fp8_heads: list[str] | None = None, pq_layers: list[str] | None = None,\n"
                "                 quark_config: dict | None = None, body_config: dict | None = None,\n"
                "                 body_layers: list[str] | None = None, fp8_embed: bool = False):\n        super().__init__()\n"
                "        self.fp8_embed = bool(fp8_embed)   # local patch: fp8 row embed_tokens\n"
                "        self.fp8_heads = list(fp8_heads or [])   # local patch: prefixes served by Quark fp8 per-channel\n"
                "        self._quark_heads_cfg = None\n"
                "        self.pq_layers = list(pq_layers or [])   # local patch (mixed): ONLY these prefixes are paroquant\n"
                "        self.quark_config = quark_config          # local patch (mixed): everything else -> this Quark config\n"
                "        self._quark_full_cfg = None\n"
                "        self.body_config = body_config             # local patch (3-way): decoder layers outside pq_layers -> this\n"
                "        self.body_layers = list(body_layers or [])   # local patch (3-way): patterns, checkpoint naming, mapped later\n"
                "        self._body_cfg = None\n")
    assert old_init in s, "plugin __init__ anchor not found"
    s = s.replace(old_init, new_init, 1)

    old_ret = "        return cls(bits, group_size, krot, pats)\n"
    new_ret = ("        fp8_heads = list(config.get(\"fp8_heads\") or [])   # local patch\n"
               "        pq_layers = list(config.get(\"pq_layers\") or [])   # local patch (mixed)\n"
               "        quark_config = config.get(\"quark_config\")          # local patch (mixed)\n"
               "        body_config = config.get(\"body_config\")            # local patch (3-way)\n"
               "        body_layers = list(config.get(\"body_layers\") or [])   # local patch (3-way)\n"
               "        fp8_embed = bool(config.get(\"fp8_embed\", False))   # local patch\n"
               "        return cls(bits, group_size, krot, pats, fp8_heads, pq_layers, quark_config, body_config, body_layers, fp8_embed)\n")
    assert old_ret in s, "plugin from_config return anchor not found"
    s = s.replace(old_ret, new_ret, 1)

    old_gqm = ("    def get_quant_method(self, layer, prefix: str):\n"
               "        if not isinstance(layer, LinearBase):\n"
               "            return None\n")
    new_gqm = '''    def _quark_heads(self):
        # local patch: an embedded QuarkConfig whose layer_quant_config maps every fp8_heads pattern to the
        # FP8 per-channel scheme. quark.py carries our ParallelLMHead branch, so the LM head resolves too.
        if self._quark_heads_cfg is None:
            from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
            qc = QuarkConfig({"quant_method": "quark", "exclude": [],
                              "layer_quant_config": {p: _PQ_FP8_CFG for p in self.fp8_heads},
                              "layer_type_quant_config": {}, "global_quant_config": None})
            qc.packed_modules_mapping = dict(self.packed_modules_mapping)
            self._quark_heads_cfg = qc
        return self._quark_heads_cfg

    def _quark_full(self):
        # local patch (mixed): the WHOLE Quark quantization_config of the MXFP4 checkpoint (global MXFP4 scheme,
        # exclude list, fp8 layer_quant_config for lm_head / mtp): every prefix outside pq_layers resolves here
        # exactly as on a plain Quark serve (patch_quark_mxfp4 + our quark.py patches apply to it).
        if self._quark_full_cfg is None:
            from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
            qc = QuarkConfig.from_config(self.quark_config)
            qc.packed_modules_mapping = dict(self.packed_modules_mapping)
            if getattr(self, "_vllm_mapper", None) is not None:
                qc.apply_vllm_mapper(self._vllm_mapper)   # exclude / layer_quant_config names -> vLLM prefixes, as on prod
            self._quark_full_cfg = qc
        return self._quark_full_cfg

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # local patch (mixed): the loader hands the MODEL quant config the hf->vLLM name mapper; our patterns are written
        # in checkpoint naming (model.language_model.layers.N.*) but get_quant_method sees vLLM prefixes
        # (language_model.model.layers.N.*), so map them, and keep the mapper for the embedded Quark config.
        self._vllm_mapper = hf_to_vllm_mapper
        self.pq_layers = list(hf_to_vllm_mapper.apply_list(self.pq_layers))
        self.fp8_heads = list(hf_to_vllm_mapper.apply_list(self.fp8_heads))
        self.body_layers = list(hf_to_vllm_mapper.apply_list(self.body_layers))
        self._quark_full_cfg = None

    def _body(self):
        # local patch (3-way): embedded ParoQuantMXFP4Config for a rotation-MXFP4 body (its own visual / in_proj_a/b
        # skips apply; None for non-Linear modules such as Attention, exactly as a plain paroquant_mxfp4 serve).
        if self._body_cfg is None:
            import radiance_paroquant_mxfp4 as _pqm
            self._body_cfg = _pqm.ParoQuantMXFP4Config.from_config(self.body_config)
            self._body_cfg.packed_modules_mapping = dict(self.packed_modules_mapping)
        return self._body_cfg

    def get_quant_method(self, layer, prefix: str):
        if self.fp8_embed and isinstance(layer, VocabParallelEmbedding) and not isinstance(layer, ParallelLMHead):
            return _PQFp8RowEmbeddingMethod()   # local patch: fp8 row embed_tokens
        if self.quark_config is not None and not any(_fnmatch.fnmatchcase(prefix, p) for p in self.pq_layers):
            return self._quark_full().get_quant_method(layer, prefix)   # local patch (mixed)
        if self.fp8_heads and any(_fnmatch.fnmatchcase(prefix, p) for p in self.fp8_heads):
            _qm = self._quark_heads().get_quant_method(layer, prefix)   # local patch: fp8 heads
            # every LinearBase the fp8-heads patterns reach is a drafter linear (the LM heads are ParallelLMHead)
            if _PQ_MTP_HC and _qm is not None and isinstance(layer, LinearBase):
                return _PQMtpHCollect(_qm, prefix)   # local patch (2026-09-28): drafter H collection (calibration run)
            if _PQ_MTP_MXFP4 and _qm is not None and isinstance(layer, LinearBase) \\
                    and not any(t in prefix for t in _PQ_MTP_MXFP4_SKIP):
                return _PQMtpMXFP4Method(_qm, prefix)   # local patch (2026-09-28): drafter linears in MXFP4
            return _qm
        if self.body_config is not None and any(_fnmatch.fnmatchcase(prefix, p) for p in self.body_layers) and not any(_fnmatch.fnmatchcase(prefix, p) for p in self.pq_layers):
            return self._body().get_quant_method(layer, prefix)   # local patch (3-way): rotation-MXFP4 body
        if not isinstance(layer, LinearBase):
            return None
'''
    assert old_gqm in s, "plugin get_quant_method anchor not found"
    s = s.replace(old_gqm, new_gqm, 1)
    dst.write_text(s)

_patch_plugin(SP / py.name, '@register_quantization_config("paroquant")' + chr(10))
_patch_plugin(SP / pym.name, '@register_quantization_config("paroquant_mxfp4")' + chr(10))

sc = Path("/usr/lib/python3.12/sitecustomize.py")
block = ("# local patch v2: plugin registration; skipped in the fork's patch_*.py helper interpreters (each import costs ~10 s of HIP/vLLM init)\n"
         "import os as _os, sys as _sys\n"
         "if not _os.path.basename(_sys.argv[0] if _sys.argv else '').startswith('patch_'):\n"
         "    try:\n"
         "        import radiance_paroquant  # registers the paroquant quantization config (local patch)\n"
         "    except Exception as e:\n"
         "        _sys.stderr.write(\"[radiance.paroquant] registration failed: %r\\n\" % (e,))\n"
         "    try:\n"
         "        import radiance_paroquant_mxfp4  # paroquant_mxfp4 variant; needs radiance_mxfp4, which serve-mxfp4.sh installs\n"
         "    except Exception:\n"
         "        pass\n")
_sc_txt = sc.read_text() if sc.exists() else ""
if "radiance_paroquant" in _sc_txt and "local patch v2" not in _sc_txt:
    # v1 block (unguarded import in every interpreter) -> drop it; our blocks are the tail of the file
    _k = _sc_txt.index("try:" + chr(10) + "    import radiance_paroquant")
    sc.write_text(_sc_txt[:_k].rstrip() + chr(10)); _sc_txt = sc.read_text()
if "local patch v2" not in _sc_txt:
    with open(sc, "a") as fh:
        fh.write("\n" + block)
print(f"[local patch] paroquant plugin installed into {SP} (kernel .so + radiance_paroquant.py [+fp8-heads patch] + sitecustomize registration)")
