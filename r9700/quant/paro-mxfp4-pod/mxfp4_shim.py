"""paroquant/optim/mxfp4.py - shim for the fork's build_hybrid.py (the fork dev never published this module).
OCP MXFP4: e2m1 codes, one e8m0 shared scale per 32 weights, scale = 2^(floor(log2(amax)) - 2), round-half-even on ties.
Layout = Quark's: weight [N, K/2] uint8 (even index in the LOW nibble), weight_scale [N, K/32] uint8 (E+127).
Identical to custom-quant/quantize_mxfp4_qwen35.py, which produced checkpoints that vLLM-Radiance served (2026-09-09/15)."""
import torch
GROUP = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0

def scale_rule() -> str:
    return "ocp: e8m0 = floor(log2(amax_32)) - 2, e2m1 round-half-even"

def quantize_to_codes(w: torch.Tensor, chunk: int = 2048):
    if w.shape[0] > chunk:
        ps, es = [], []
        for i in range(0, w.shape[0], chunk):
            a, b = quantize_to_codes(w[i:i + chunk], chunk); ps.append(a); es.append(b)
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
        code = torch.where(tie, ((code // 2) * 2).to(torch.uint8), code)
    code = (code | (sign.to(torch.uint8) << 3)).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    e8m0 = (exp + 127).to(torch.uint8).contiguous()
    return packed, e8m0

def dequantize_from_codes(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    n = packed.shape[0]; k = packed.shape[1] * 2
    code = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    code[:, 0::2] = packed & 0xF
    code[:, 1::2] = packed >> 4
    grid = E2M1.to(packed.device)
    mag = grid[(code & 0x7).long()]
    val = torch.where((code & 0x8) > 0, -mag, mag)
    scale = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    return (val.reshape(n, k // GROUP, GROUP) * scale).reshape(n, k)
