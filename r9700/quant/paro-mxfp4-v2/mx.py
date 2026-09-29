"""mx.py - OCP MXFP4 (e2m1 codes, one e8m0 power-of-two scale per 32 weights) quantizers, bit-compatible with
custom-quant/paro-mxfp4-pod/mxfp4_shim.py (the finalist's builder) and the Quark layout vLLM-Radiance serves.

Exponent modes:
  ocp     e = floor(log2(amax)) - 2   (the spec rule; scaled amax lands in [4, 8) so anything above 6 clips to 6)
  search  per block, the best of e, e+1, e-1 by (optionally weighted) squared error; e wins ties
"""
import torch

BLOCK = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_TH = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])      # midpoints between neighbouring grid values


def round_e2m1(v: torch.Tensor):
    """Nearest e2m1 code for already-scaled values, ties to the even code (= the shim's rule); |v| > 6 saturates."""
    th = _TH.to(v.device, v.dtype)
    mag = v.abs()
    code = torch.bucketize(mag, th)                                       # th[c-1] < mag <= th[c]
    tie = (code % 2 == 1) & (mag == th[code.clamp(max=6)])                # exactly on a midpoint above an odd code
    code = (code + tie.to(code.dtype)).to(torch.uint8)
    return code, torch.signbit(v)


def dequant(code, sign, e):
    mag = E2M1.to(code.device)[code.long()]
    return torch.where(sign, -mag, mag) * torch.exp2(e.float()).unsqueeze(-1)


def ocp_exp(amax):
    return torch.where(amax > 0, torch.floor(torch.log2(amax)) - 2.0, torch.zeros_like(amax)).clamp(-127, 127)


def quant_blocks(wb: torch.Tensor, mode: str = "ocp", weight: torch.Tensor | None = None):
    """wb [..., 32] float32 -> (code uint8, sign bool, e float) with the same leading shape. weight broadcasts to wb."""
    e0 = ocp_exp(wb.abs().amax(-1))
    cands = [e0] if mode == "ocp" else [e0, (e0 + 1).clamp(max=127), (e0 - 1).clamp(min=-127)]
    best = None
    for e in cands:
        code, sign = round_e2m1(wb / torch.exp2(e).unsqueeze(-1))
        if len(cands) == 1:
            return code, sign, e
        d = (wb - dequant(code, sign, e)).square()
        err = (d * weight).sum(-1) if weight is not None else d.sum(-1)
        if best is None:
            best = [err, code, sign, e]
        else:
            m = err < best[0]
            best[0] = torch.where(m, err, best[0])
            best[1] = torch.where(m.unsqueeze(-1), code, best[1])
            best[2] = torch.where(m.unsqueeze(-1), sign, best[2])
            best[3] = torch.where(m, e, best[3])
    return best[1], best[2], best[3]


def quant_matrix(w: torch.Tensor, mode: str = "ocp", col_weight: torch.Tensor | None = None, chunk: int = 4096):
    """w [N, K] (rotated basis) -> (code [N,K] u8, sign [N,K] bool, e [N,K/32]). col_weight [K] weights the error per column."""
    n, k = w.shape
    assert k % BLOCK == 0, k
    cw = None if col_weight is None else col_weight.float().view(1, k // BLOCK, BLOCK)
    codes, signs, exps = [], [], []
    for i in range(0, n, chunk):
        wb = w[i:i + chunk].float().view(-1, k // BLOCK, BLOCK)
        c, s, e = quant_blocks(wb, mode, cw)
        codes.append(c.view(-1, k)); signs.append(s.view(-1, k)); exps.append(e)
    return torch.cat(codes), torch.cat(signs), torch.cat(exps)


def dequant_matrix(code, sign, e):
    n, k = code.shape
    return dequant(code.view(n, k // BLOCK, BLOCK), sign.view(n, k // BLOCK, BLOCK), e).view(n, k)


def pack(code, sign, e):
    """Quark layout: weight [N, K/2] u8 (even index in the LOW nibble), weight_scale [N, K/32] u8 (E + 127)."""
    c = (code | (sign.to(torch.uint8) << 3))
    packed = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    return packed, (e + 127).to(torch.uint8).contiguous()


def unpack(packed, e8m0):
    n = packed.shape[0]; k = packed.shape[1] * 2
    c = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    c[:, 0::2] = packed & 0xF; c[:, 1::2] = packed >> 4
    return c & 0x7, (c & 0x8) > 0, e8m0.float() - 127.0


def fake_quant_ste(w: torch.Tensor, mode: str = "ocp") -> torch.Tensor:
    """Pseudo-quantize a [N, K] rotated weight with a straight-through gradient (training)."""
    with torch.no_grad():
        q = dequant_matrix(*quant_matrix(w.detach(), mode)).to(w.dtype)
    return w + (q - w).detach()
