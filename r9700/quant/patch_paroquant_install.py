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
    def embedding(self, layer, input_):
        rows = layer.weight.view(torch.uint8)[input_].view(torch.float8_e4m3fn)
        return rows.to(torch.bfloat16) * layer.weight_scale[input_].unsqueeze(-1).to(torch.bfloat16)
    def apply(self, layer, x, bias=None):
        raise NotImplementedError("fp8 row embedding is lookup-only")

_PQ_FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
               "weight": _pq_fp8_spec("fp8_e4m3", False, "per_channel", 0),
               "input_tensors": _pq_fp8_spec("fp8_e4m3", True, "per_tensor", -1)}
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
            return self._quark_heads().get_quant_method(layer, prefix)   # local patch: fp8 heads
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
