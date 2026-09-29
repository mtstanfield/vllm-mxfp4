#!/usr/bin/env python3
"""selftest_synthetic.py - exercise quantize_mxfp4_qwen35.py end to end on a tiny synthetic Qwen3.5-shaped
checkpoint (2 layers: one linear-attention, one full-attention; small dims that are multiples of 32), on CPU.

Checks: (1) mxfp4 round trip is exact on representable values, (2) AWQ fold preserves the layer function
(W x == (W s)(x / s) up to bf16 rounding), (3) the script runs, writes shards + index + config, (4) every
tensor the production census expects is present with the right dtype/shape, (5) dequantized output of a
folded+quantized MLP is close to the bf16 MLP on random inputs.

  python3 selftest_synthetic.py /tmp/qtest
"""
import json, os, subprocess, sys, tempfile
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_mxfp4_qwen35 as Q

root = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
src, out = os.path.join(root, "src"), os.path.join(root, "out")
os.makedirs(src, exist_ok=True)
H, I, NL = 256, 384, 2                     # hidden, intermediate, layers
NH, NKV, HD = 4, 2, 32                     # attention heads (q_proj emits q + gate = 2*NH*HD)
LK, LV, LKD, LVD = 4, 8, 32, 32            # gdn heads / dims
torch.manual_seed(0)
def W(n, k): return (torch.randn(n, k) * 0.05).to(torch.bfloat16)
LP = "model.language_model.layers"
T = {}
# layer 0: linear attention ; layer 1: full attention
T[f"{LP}.0.input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16) * 1.5
T[f"{LP}.0.linear_attn.in_proj_qkv.weight"] = W(LK*LKD*2 + LV*LVD, H)
T[f"{LP}.0.linear_attn.in_proj_z.weight"] = W(LV*LVD, H)
T[f"{LP}.0.linear_attn.in_proj_a.weight"] = W(LV, H)
T[f"{LP}.0.linear_attn.in_proj_b.weight"] = W(LV, H)
T[f"{LP}.0.linear_attn.out_proj.weight"] = W(H, LV*LVD)
T[f"{LP}.0.linear_attn.conv1d.weight"] = W(LK*LKD*2 + LV*LVD, 1).unsqueeze(-1).repeat(1, 1, 4)
T[f"{LP}.0.linear_attn.conv1d.bias"] = torch.zeros(LK*LKD*2 + LV*LVD, dtype=torch.bfloat16)
T[f"{LP}.0.linear_attn.norm.weight"] = torch.ones(LVD, dtype=torch.bfloat16)
T[f"{LP}.0.linear_attn.A_log"] = torch.zeros(LV, dtype=torch.float32)
T[f"{LP}.0.linear_attn.dt_bias"] = torch.zeros(LV, dtype=torch.float32)
T[f"{LP}.1.input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
T[f"{LP}.1.self_attn.q_proj.weight"] = W(NH*HD*2, H)
T[f"{LP}.1.self_attn.k_proj.weight"] = W(NKV*HD, H)
T[f"{LP}.1.self_attn.v_proj.weight"] = W(NKV*HD, H)
T[f"{LP}.1.self_attn.o_proj.weight"] = W(H, NH*HD)
T[f"{LP}.1.self_attn.q_norm.weight"] = torch.ones(HD, dtype=torch.bfloat16)
T[f"{LP}.1.self_attn.k_norm.weight"] = torch.ones(HD, dtype=torch.bfloat16)
for l in range(NL):
    T[f"{LP}.{l}.post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16) * 0.8
    T[f"{LP}.{l}.mlp.gate_proj.weight"] = W(I, H)
    T[f"{LP}.{l}.mlp.up_proj.weight"] = W(I, H)
    T[f"{LP}.{l}.mlp.down_proj.weight"] = W(H, I)
T["model.language_model.embed_tokens.weight"] = W(1000, H)
T["model.language_model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
T["lm_head.weight"] = W(1000, H)
T["model.visual.blocks.0.attn.qkv.weight"] = W(96, 32)
for m in Q.MTP:
    T[m + ".weight"] = W(H, 2*H) if m == "mtp.fc" else W(H if "down" in m or "o_proj" in m else I if "proj" in m and "mlp" in m else H, I if "down" in m else H)
T["mtp.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
save_file({k: v.contiguous() for k, v in T.items()}, os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
       "text_config": {"num_hidden_layers": NL, "layer_types": ["linear_attention", "full_attention"], "hidden_size": H, "intermediate_size": I}}
json.dump(cfg, open(os.path.join(src, "config.json"), "w"))
open(os.path.join(src, "tokenizer.json"), "w").write("{}")
# stats: skewed channel magnitudes so AWQ has something to do
stats = {}
for l in range(NL):
    stats[f"layers.{l}.attn_in" if l == 1 else f"layers.{l}.gdn_in"] = torch.rand(H) * 3 + 0.1
    stats[f"layers.{l}.mlp_in"] = torch.rand(H) * 3 + 0.1
    stats[f"layers.{l}.down_in"] = torch.rand(I) * 3 + 0.1
stats["layers.0.attn_in"] = stats.pop("layers.0.gdn_in"); stats["layers.0.gdn_in"] = stats["layers.0.attn_in"]
torch.save({"absmean": stats}, os.path.join(root, "stats.pt"))
ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-config.json")

# (1) round trip on representable values
w = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4 + [-6.0, -4.0] * 16]) * 2.0
p, e = Q.quantize_mxfp4(w); assert torch.equal(Q.dequantize_mxfp4(p, e, 64), w), "round trip"
# (2) fold identity
x = torch.randn(8, H); Wt = torch.randn(64, H); s = torch.rand(H) + 0.5
assert torch.allclose(x @ Wt.T, (x / s) @ (Wt * s).T, atol=1e-4), "fold identity"
print("arithmetic checks ok")

# (3) run the script on CPU
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantize_mxfp4_qwen35.py"),
                    "--src", src, "--stats", os.path.join(root, "stats.pt"), "--out", out, "--device", "cpu",
                    "--shard-layers", "1", "--ref-config", ref], capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:])
assert r.returncode == 0, "quantizer failed"

# (4) census of the output
from safetensors import safe_open
idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
hs = {f: safe_open(os.path.join(out, f), "pt") for f in set(idx.values())}
def info(n): sl = hs[idx[n]].get_slice(n); return sl.get_dtype(), tuple(sl.get_shape())
expect_q = [f"{LP}.0.linear_attn.in_proj_{x}" for x in ("qkv", "z", "a", "b")] + [f"{LP}.0.linear_attn.out_proj"] + \
           [f"{LP}.1.self_attn.{x}_proj" for x in "qkvo"] + [f"{LP}.{l}.mlp.{x}_proj" for l in range(NL) for x in ("gate", "up", "down")]
for m in expect_q:
    d, sh = info(m + ".weight"); ds, shs = info(m + ".weight_scale")
    K = T[m + ".weight"].shape[1]
    assert d == "U8" and sh == (T[m + ".weight"].shape[0], K // 2), (m, d, sh)
    assert ds == "U8" and shs == (T[m + ".weight"].shape[0], K // 32), (m, ds, shs)
for m in Q.MTP:
    d, sh = info(m + ".weight"); ds, shs = info(m + ".weight_scale")
    assert d == "F8_E4M3" and sh == tuple(T[m + ".weight"].shape) and ds == "F32" and shs == (sh[0],), (m, d, ds, shs)
for n in ["lm_head.weight", "model.visual.blocks.0.attn.qkv.weight", f"{LP}.0.linear_attn.conv1d.weight", f"{LP}.0.linear_attn.A_log"]:
    assert n in idx and info(n)[0] in ("BF16", "F32"), n
assert set(idx) >= set(T), f"missing tensors: {set(T) - set(idx)}"
extra = {k for k in idx if k not in T and not k.endswith(".weight_scale")}
assert not extra, f"unexpected tensors: {extra}"
print(f"census ok: {len(idx)} tensors, {len(expect_q)} mxfp4 linears, {len(Q.MTP)} fp8")

# (5) functional check of layer 1's folded MLP: bf16 reference vs dequantized quant (same random inputs)
def get(n): return hs[idx[n]].get_tensor(n)
xin = torch.randn(64, H)
def mlp(gate, up, down, pre):
    h = xin * (1.0 + pre.float())     # Qwen3.5 decoder RMSNorm is zero-centered: x_norm * (1 + w)
    return (torch.nn.functional.silu(h @ gate.float().T) * (h @ up.float().T)) @ down.float().T
ref_out = mlp(T[f"{LP}.1.mlp.gate_proj.weight"], T[f"{LP}.1.mlp.up_proj.weight"], T[f"{LP}.1.mlp.down_proj.weight"], T[f"{LP}.1.post_attention_layernorm.weight"])
dq = lambda m: Q.dequantize_mxfp4(get(m + ".weight"), get(m + ".weight_scale"), T[m + ".weight"].shape[1])
q_out = mlp(dq(f"{LP}.1.mlp.gate_proj"), dq(f"{LP}.1.mlp.up_proj"), dq(f"{LP}.1.mlp.down_proj"), get(f"{LP}.1.post_attention_layernorm.weight"))
rel = ((q_out - ref_out).norm() / ref_out.norm()).item()
print(f"folded+quantized MLP vs bf16 MLP: rel output err {rel:.4f} (expect ~0.05-0.15 for random 4-bit weights)")
assert rel < 0.3, rel
cfgo = json.load(open(os.path.join(out, "config.json")))
assert cfgo["quantization_config"]["quant_method"] == "quark" and len(cfgo["quantization_config"]["layer_quant_config"]) == 8
print("SELFTEST PASSED")
