#!/usr/bin/env python3
"""quantize_mxfp4_qwen35.py - AWQ-scaled OCP-MXFP4 body + FP8 MTP head for a Qwen3.5/3.8-27B checkpoint,
in the exact Quark layout that vLLM-Radiance serves (amd/Qwen3.8-27B-Quark-AWQ-MXFP4 + fp8_mtp.py).

Ported from the fork's quantize_dflash_mxfp4.py (drafter) to the full decoder. Same arithmetic:
e2m1 round-half-even with a power-of-two E8M0 scale per 32 weights, an AWQ alpha search on a diagonal
error surrogate weighted by measured mean |x|, and the scale folded into whatever produces the input
(there is nowhere to store it in a Quark MXFP4 checkpoint).

What gets what (this is the census of the production checkpoint, reproduced exactly):
  full-attention layers (16): q_proj k_proj v_proj  <- AWQ, s folded into input_layernorm
                              o_proj                <- RTN
  linear-attention layers (48): in_proj_qkv in_proj_z in_proj_a in_proj_b  <- AWQ, one shared s
                                folded into input_layernorm (all four read the same normed input;
                                conv1d sits AFTER in_proj_qkv, so it is untouched)
                                out_proj            <- RTN (its input is the gated RMSNorm output,
                                whose weight is per head-dim, so a per-channel fold is not available)
  every layer: gate_proj up_proj <- AWQ, s folded into post_attention_layernorm
               down_proj         <- AWQ, s folded into up_proj's OUTPUT rows
  mtp.fc + the 7 mtp.layers.0 projections <- FP8 e4m3 per-output-channel (fp8_mtp.py recipe)
  bf16 untouched: embed_tokens, all norms, conv1d, A_log/dt_bias, lm_head, the whole vision tower

Usage:
  python3 quantize_mxfp4_qwen35.py --src /work/src --stats /work/stats.pt --out /work/out \
      [--device cuda] [--shard-layers 8] [--alpha-grid 11] [--ref-config /work/ref-config.json]
--ref-config: the production checkpoint's config.json (quantization_config is copied from it verbatim,
so exclude/layer_quant_config match byte for byte). Without it an equivalent block is synthesised.
Output: sharded safetensors + index, config.json, tokenizer/template files, quant-report.json.
"""
import argparse, json, os, re, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
GROUP = 32
FP8_MAX = 448.0
MTP = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
       "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
       "mtp.layers.0.self_attn.v_proj"]
LP = "model.language_model.layers"

# ---------------------------------------------------------------- mxfp4 arithmetic (verbatim port)
def quantize_mxfp4(w: torch.Tensor, chunk: int = 2048):
    if w.shape[0] > chunk:
        ps, es = [], []
        for i in range(0, w.shape[0], chunk):
            a, b = quantize_mxfp4(w[i:i + chunk], chunk)
            ps.append(a); es.append(b)
        return torch.cat(ps), torch.cat(es)
    n, k = w.shape
    assert k % GROUP == 0, k
    wb = w.float().reshape(n, k // GROUP, GROUP)
    amax = wb.abs().amax(-1)
    exp = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 2.0, torch.zeros_like(amax)).clamp(-127, 127)
    scale = torch.exp2(exp)
    v = wb / scale.unsqueeze(-1)
    sign = torch.signbit(v)
    mag = v.abs().clamp(max=E2M1_MAX)
    grid = E2M1.to(w.device)
    d = (mag.unsqueeze(-1) - grid).abs()
    code = d.argmin(-1).to(torch.uint8)
    tie = (d.min(-1).values.unsqueeze(-1) == d).sum(-1) > 1
    if tie.any():
        even = (code // 2) * 2
        code = torch.where(tie, even.to(torch.uint8), code)
    code = (code | (sign.to(torch.uint8) << 3)).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    e8m0 = (exp + 127).to(torch.uint8).contiguous()
    return packed, e8m0

def dequantize_mxfp4(packed, e8m0, k):
    n = packed.shape[0]
    code = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    code[:, 0::2] = packed & 0xF
    code[:, 1::2] = packed >> 4
    grid = E2M1.to(packed.device)
    mag = grid[(code & 0x7).long()]
    val = torch.where((code & 0x8) > 0, -mag, mag)
    scale = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    return (val.reshape(n, k // GROUP, GROUP) * scale).reshape(n, k)

def search_scale(W, a, grid, max_rows=2048):
    if W.shape[0] > max_rows:
        W = W[:: max(1, W.shape[0] // max_rows)][:max_rows]
    W = W.float()
    a = a.float().clamp_min(1e-8)
    an = a / a.mean()
    best = best_s = best_alpha = None
    for alpha in grid:
        s = an.pow(alpha).clamp(1e-2, 1e2)
        s = s / s.log().mean().exp()
        q, e = quantize_mxfp4(W * s)
        err = (dequantize_mxfp4(q, e, W.shape[1]) / s - W).mul(a).pow(2).sum()
        if best is None or err < best:
            best, best_s, best_alpha = err, s, alpha
    return best_s, best_alpha, float(best)

def fold_norm(w, s):
    """Qwen3.5's decoder RMSNorm is ZERO-CENTERED (weight initialised to zeros, applied as x_norm * (1 + w)),
    so folding a per-channel 1/s means (1 + w) / s - 1, not w / s. (Getting this wrong = PPL in the millions,
    found 2026-09-09 on the first pod run.) The gated GDN norm uses the plain convention but is never folded."""
    return (1.0 + w.float()) / s - 1.0

def fp8_per_channel(w):
    w = w.float()
    amax = w.abs().amax(dim=1).clamp(min=1e-12)
    s = (amax / FP8_MAX).float()
    q = (w / s.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    rel = ((q.float() * s.unsqueeze(1) - w).norm() / w.norm()).item()
    return q, s, rel

# ---------------------------------------------------------------- checkpoint access
class Src:
    def __init__(self, d):
        self.d = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            self.map = {k: "model.safetensors" for k in safe_open(os.path.join(d, "model.safetensors"), "pt").keys()}
        self.handles = {}
    def keys(self): return list(self.map)
    def get(self, name, device="cpu"):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.handles[f].get_tensor(name).to(device)

class Out:
    """Sharded writer: accumulates tensors, flushes a shard on demand, builds the index."""
    def __init__(self, d):
        self.d = d; os.makedirs(d, exist_ok=True)
        self.buf = {}; self.n = 0; self.map = {}; self.total = 0; self.files = []
    def put(self, name, t):
        t = t.contiguous().cpu()
        self.buf[name] = t; self.total += t.numel() * t.element_size()
    def flush(self):
        if not self.buf: return
        self.n += 1
        fn = f"model-{self.n:05d}.safetensors"
        save_file(self.buf, os.path.join(self.d, fn), metadata={"format": "pt"})
        for k in self.buf: self.map[k] = fn
        self.files.append(fn); self.buf = {}
    def finish(self):
        self.flush()
        # rename to -of-N and write the index
        n = len(self.files); ren = {}
        for i, fn in enumerate(self.files, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.rename(os.path.join(self.d, fn), os.path.join(self.d, new)); ren[fn] = new
        self.map = {k: ren[v] for k, v in self.map.items()}
        json.dump({"metadata": {"total_size": self.total}, "weight_map": self.map},
                  open(os.path.join(self.d, "model.safetensors.index.json"), "w"), indent=2)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-layers", type=int, default=8)
    ap.add_argument("--alpha-grid", type=int, default=11, help="alphas in [0,1], default 11 -> step 0.1")
    ap.add_argument("--ref-config", default=None, help="production checkpoint config.json to copy quantization_config from")
    args = ap.parse_args()
    dev = args.device
    grid = [i / (args.alpha_grid - 1) for i in range(args.alpha_grid)]

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tc = cfg["text_config"]
    nl = tc["num_hidden_layers"]; layer_types = tc["layer_types"]
    stats = torch.load(args.stats, map_location=dev)["absmean"]
    src = Src(args.src); out = Out(args.out)
    keys = set(src.keys())
    report = {"alpha": {}, "rel_err": {}, "fp8_rel_err": {}}
    t0 = time.time()

    def q_and_put(name, w):
        packed, e8m0 = quantize_mxfp4(w)
        deq = dequantize_mxfp4(packed, e8m0, w.shape[1])
        report["rel_err"][name] = float((deq - w.float()).norm() / w.float().norm())
        out.put(name + ".weight", packed); out.put(name + ".weight_scale", e8m0)

    def bf16(name, t): out.put(name, t.to(torch.bfloat16))

    for l in range(nl):
        P = f"{LP}.{l}"
        T = {}
        def g(n):
            if n not in T: T[n] = src.get(n, dev).float()
            return T[n]
        # ---- token mixer ----
        if layer_types[l] == "full_attention":
            names = [f"{P}.self_attn.{x}_proj" for x in "qkv"]
            a = stats[f"layers.{l}.attn_in"]
            W = torch.cat([g(n + ".weight") for n in names], 0)
            s, alpha, _ = search_scale(W, a, grid)
            report["alpha"][f"{l}.attn_qkv"] = alpha
            bf16(f"{P}.input_layernorm.weight", fold_norm(g(f"{P}.input_layernorm.weight"), s))
            for n in names: q_and_put(n, g(n + ".weight") * s)
            q_and_put(f"{P}.self_attn.o_proj", g(f"{P}.self_attn.o_proj.weight"))
            for n in [f"{P}.self_attn.q_norm.weight", f"{P}.self_attn.k_norm.weight"]:
                if n in keys: bf16(n, g(n))
            for n in names + [f"{P}.self_attn.o_proj"]:
                if n + ".bias" in keys: bf16(n + ".bias", g(n + ".bias"))
        else:
            names = [f"{P}.linear_attn.in_proj_{x}" for x in ("qkv", "z", "a", "b")]
            a = stats[f"layers.{l}.gdn_in"]
            W = torch.cat([g(n + ".weight") for n in names], 0)
            s, alpha, _ = search_scale(W, a, grid)
            report["alpha"][f"{l}.gdn_in"] = alpha
            bf16(f"{P}.input_layernorm.weight", fold_norm(g(f"{P}.input_layernorm.weight"), s))
            for n in names: q_and_put(n, g(n + ".weight") * s)
            q_and_put(f"{P}.linear_attn.out_proj", g(f"{P}.linear_attn.out_proj.weight"))
            # everything else in the GDN block stays bf16 (conv1d, norm, A_log, dt_bias, biases)
            for n in keys:
                if n.startswith(f"{P}.linear_attn.") and not any(n == m + ".weight" for m in names + [f"{P}.linear_attn.out_proj"]):
                    bf16(n, g(n))
        # ---- mlp ----
        gn, un, dn = (f"{P}.mlp.{x}_proj" for x in ("gate", "up", "down"))
        a = stats[f"layers.{l}.mlp_in"]
        W = torch.cat([g(gn + ".weight"), g(un + ".weight")], 0)
        s, alpha, _ = search_scale(W, a, grid)
        report["alpha"][f"{l}.gate_up"] = alpha
        bf16(f"{P}.post_attention_layernorm.weight", fold_norm(g(f"{P}.post_attention_layernorm.weight"), s))
        Wg = g(gn + ".weight") * s; Wu = g(un + ".weight") * s
        a = stats[f"layers.{l}.down_in"]
        s2, alpha2, _ = search_scale(g(dn + ".weight"), a, grid)
        report["alpha"][f"{l}.down"] = alpha2
        Wu = Wu / s2.unsqueeze(1)
        Wd = g(dn + ".weight") * s2
        q_and_put(gn, Wg); q_and_put(un, Wu); q_and_put(dn, Wd)
        for n in (gn, un, dn):
            if n + ".bias" in keys: bf16(n + ".bias", g(n + ".bias"))
        del T
        if (l + 1) % args.shard_layers == 0:
            out.flush()
            print(f"  layer {l+1}/{nl}  {time.time()-t0:.0f}s  shard {out.n} written", flush=True)
    out.flush()

    # ---- everything outside the decoder layers ----
    for n in sorted(keys):
        if n.startswith(LP + "."): continue
        mod = n[:-7] if n.endswith(".weight") else None
        if mod in MTP:
            q, s, rel = fp8_per_channel(src.get(n, dev))
            report["fp8_rel_err"][mod] = rel
            out.put(n, q); out.put(mod + ".weight_scale", s)
        else:
            t = src.get(n)
            out.put(n, t.to(torch.bfloat16) if t.is_floating_point() else t)
    out.finish()

    # ---- config + aux files ----
    if args.ref_config:
        qc = json.load(open(args.ref_config))["quantization_config"]
    else:
        raise SystemExit("--ref-config is required in this build (copy the production checkpoint's config.json)")
    cfg["quantization_config"] = qc
    cfg["use_cache"] = True
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        p = os.path.join(args.src, n)
        if os.path.exists(p): shutil.copy(p, os.path.join(args.out, n))

    errs = sorted(report["rel_err"].items(), key=lambda x: -x[1])
    report["summary"] = {"quantized_linears": len(report["rel_err"]), "mean_rel_err": sum(v for _, v in errs) / len(errs),
                         "worst": errs[:8], "seconds": time.time() - t0, "shards": out.n, "bytes": out.total}
    json.dump(report, open(os.path.join(args.out, "quant-report.json"), "w"), indent=2)
    print(f"quantized {len(errs)} linears, mean rel err {report['summary']['mean_rel_err']:.4f} (AMD's drafter MXFP4 was ~0.116), "
          f"{out.total/2**30:.2f} GiB in {out.n} shards, {time.time()-t0:.0f}s")
    print("worst:", *[f"{k} {v:.4f}" for k, v in errs[:5]], sep="\n  ")
    print("fp8 mtp:", {k.split('.')[-1]: round(v, 4) for k, v in report["fp8_rel_err"].items()})

if __name__ == "__main__":
    main()
