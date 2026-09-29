#!/usr/bin/env python3
"""selftest.py - checks that must pass before spending GPU hours (run on the pod after setup; ~1 min).
  1. torch reference rotation == z-lab CUDA kernel (so run.py's math == the serving transform)
  2. mx OCP quantizer == the finalist builder's shim, byte for byte, on a real rotated weight
  3. rotate_hessian == E[x_rot^T x_rot]
  4. GPTQ lowers the layer-output proxy loss vs RTN; scale search lowers weight MSE vs OCP
  5. pack/unpack round trip
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx, rot, gptq
from safetensors import safe_open

PARO = os.environ.get("PQ2_PARO", "/workspace/models/Qwen3.8-27B-PARO")
BASE = os.environ.get("PQ2_BASE", "/workspace/models/Qwen3.8-27B-bf16")
SHIM = os.environ.get("PQ2_SHIM", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4_shim.py"))
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


torch.manual_seed(0)
rp = rot.RotParams(PARO)
key = [k for k in rp.keys() if k[1].endswith("mlp.down_proj")][3]
p = rp.get(key)
import json
wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
wname = [n for n in wmap if n.endswith(f"layers.{key[0]}.{key[1]}.weight")][0]
with safe_open(f"{BASE}/{wmap[wname]}", framework="pt") as f:
    w = f.get_tensor(wname).cuda().float()
print(f"module {key}: weight {tuple(w.shape)}, pairs {tuple(p['pairs'].shape)}, kernel {'yes' if rot._kernel else 'NO'}")

# 1
x = torch.randn(64, w.shape[1], device="cuda")
a = rot.rotate(x * p["cs"], p["pairs"], p["theta"]); b = rot.rotate(x * p["cs"], p["pairs"], p["theta"], force_ref=True)
check("rotation kernel == torch reference", (a - b).abs().max() < 1e-4, f"max diff {(a-b).abs().max():.2e}")
r = rot.unrotate(a, p["pairs"], p["theta"])
check("unrotate(rotate(x)) == x", (r - x * p["cs"]).abs().max() < 1e-4, f"max diff {(r - x*p['cs']).abs().max():.2e}")

# 2
import importlib.util
spec = importlib.util.spec_from_file_location("shim", SHIM); shim = importlib.util.module_from_spec(spec); spec.loader.exec_module(shim)
wr = rot.rotate(w * p["cs"].view(1, -1), p["pairs"], p["theta"])
sp, se = shim.quantize_to_codes(wr)
c0, s0, e0 = mx.quant_matrix(wr, "ocp")
mp, me = mx.pack(c0, s0, e0)
# The shim's tie rule ((argmin // 2) * 2) steps DOWN from an odd lower code on an exact midpoint (0.75 -> 0.0, 1.75 -> 1.0,
# 3.5 -> 2.0) instead of rounding half to even; ours rounds correctly. Allow differences only at those exact ties.
bad = []
for r, col in (sp != mp).nonzero().tolist():
    for j in (2 * col, 2 * col + 1):
        if ((sp[r, col] >> (4 * (j % 2))) & 0xF) != ((mp[r, col] >> (4 * (j % 2))) & 0xF):
            v = abs((wr[r, j] / torch.exp2(e0[r, j // 32])).item())
            if v not in (0.75, 1.75, 3.5):
                bad.append((r, j, v))
check("mx OCP == finalist shim (bytes, except the shim's exact-tie bug)", not bad and torch.equal(se, me),
      f"weight bytes differ: {(sp != mp).sum().item()} (all exact ties: {not bad}), scale bytes differ: {(se != me).sum().item()}")

# 3
xs = torch.randn(512, w.shape[1], device="cuda") * torch.linspace(0.2, 3, w.shape[1], device="cuda")
h = xs.T @ xs
hr = rot.rotate_hessian(h, p["pairs"], p["theta"], p["cs"])
xr = rot.rotate(xs / p["cs"], p["pairs"], p["theta"])
d = (hr - xr.T @ xr).abs().max() / (xr.T @ xr).abs().max()
check("rotate_hessian == E[x_rot^T x_rot]", d < 1e-4, f"rel max diff {d:.2e}")

# 4
rows = w[:1024]; wr1 = wr[:1024]
q_rtn = mx.dequant_matrix(*mx.quant_matrix(wr1, "ocp"))
q_ss = mx.dequant_matrix(*mx.quant_matrix(wr1, "search"))
q_gp = mx.dequant_matrix(*gptq.gptq_mxfp4(wr1, hr, "search"))
m_rtn, m_ss = (q_rtn - wr1).square().mean().item(), (q_ss - wr1).square().mean().item()
check("scale search MSE < OCP MSE", m_ss < m_rtn, f"{m_ss:.3e} vs {m_rtn:.3e} ({(1-m_ss/m_rtn)*100:.1f}% lower)")
l_rtn, l_ss, l_gp = (gptq.proxy_loss(wr1, q, hr) for q in (q_rtn, q_ss, q_gp))
check("GPTQ proxy loss < scale-search < OCP", l_gp < l_ss < l_rtn,
      f"OCP {l_rtn:.4g}  search {l_ss:.4g}  gptq {l_gp:.4g} ({(1-l_gp/l_rtn)*100:.1f}% lower than OCP)")

# 5
c, s, e = mx.quant_matrix(wr1, "search")
c2, s2, e2 = mx.unpack(*mx.pack(c, s, e))
check("pack/unpack round trip", torch.equal(mx.dequant_matrix(c, s, e), mx.dequant_matrix(c2, s2, e2)))
clip = (wr1.view(-1, 32).abs().amax(-1) / torch.exp2(mx.ocp_exp(wr1.view(-1, 32).abs().amax(-1))) > 6).float().mean().item()
print(f"info: {clip*100:.1f}% of blocks clip their max under the OCP rule")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
