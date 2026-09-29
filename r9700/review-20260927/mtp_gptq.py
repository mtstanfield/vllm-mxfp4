#!/usr/bin/env python3
"""mtp_gptq.py <hc dir> <out.pt> [alpha] -- GPTQ requant of the MTP drafter's fp8 linears to MXFP4 on their own inputs.

<hc dir> holds <prefix>.pt from a RADIANCE_MTP_HCOLLECT calibration run: fp8 weight + per-channel scale as loaded, and
the input second moments Hd (draft passes, M <= 16) / Hp (drafter prefill over prompt tokens). H = alpha * Hd/nd +
(1 - alpha) * Hp/np. Quantizer: gptq.gptq_mxfp4 (custom-quant/paro-mxfp4-v2, the body's v2 requant: lazy 128-column
batches, damped Cholesky, per-32 exponent chosen from the error-updated weights by diag(H)-weighted search among
e-1/e/e+1). Output {prefix: {"codes": u8 [N, K/2], "e8": u8 [N, K/32]}} for RADIANCE_MTP_MXFP4_FILE.
Logs the proxy loss tr(dW H dW^T)/N under Hd and Hp for the serve's load-time RTN and for GPTQ."""
import glob, hashlib, os, sys, time
import torch
sys.path.insert(0, os.environ.get("PQV2", "/pqv2"))
import mx, gptq

hc, out = sys.argv[1], sys.argv[2]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
dev = "cuda"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def rtn(w):
    """The serve's load-time quantizer (_pq_mxfp4_quant): e in {floor(log2 amax)-2, +1} by unweighted squared error."""
    N, K = w.shape
    grid = torch.tensor(E2M1, device=w.device)
    b = w.reshape(N, K // 32, 32)
    amax = b.abs().amax(-1).clamp_min(2.0 ** -120)
    e_lo = torch.floor(torch.log2(amax)) - 2.0
    best = None
    for de in (0.0, 1.0):
        e = (e_lo + de).clamp(-127.0, 127.0)
        q = b / torch.exp2(e).unsqueeze(-1)
        idx = (q.abs().clamp(max=6.0).unsqueeze(-1) - grid).abs().argmin(-1)
        deq = grid[idx] * torch.sign(q) * torch.exp2(e).unsqueeze(-1)
        err = (deq - b).square().sum(-1)
        if best is None:
            best = [err, deq]
        else:
            pick = err < best[0]
            best[0] = torch.where(pick, err, best[0])
            best[1] = torch.where(pick.unsqueeze(-1), deq, best[1])
    return best[1].reshape(N, K)


def src_sha1(w8, scale):
    """Fingerprint of the fp8 weight + per-channel scale the codes were derived from (the serve re-checks it at load)."""
    h = hashlib.sha1(w8.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    h.update(scale.float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


res = {}
for f in sorted(glob.glob(os.path.join(hc, "*.pt"))):
    prefix = os.path.basename(f)[:-3]
    d = torch.load(f, map_location="cpu")
    w = d["w8"].to(dev).float() * d["scale"].to(dev).float().unsqueeze(1)
    hd = d["Hd"].to(dev) / max(d["nd"], 1)
    hp = d["Hp"].to(dev) / max(d["np"], 1)
    h = alpha * hd + (1 - alpha) * hp
    t0 = time.time()
    code, sign, e = gptq.gptq_mxfp4(w, h, mode="search")
    qg = mx.dequant_matrix(code, sign, e)
    qr = rtn(w)
    packed, e8 = mx.pack(code, sign, e)
    res[prefix] = {"codes": packed.cpu(), "e8": e8.cpu(), "src_sha1": src_sha1(d["w8"], d["scale"])}
    def pl(q, H):   # tr(dW H dW^T)/N as one GEMM (torch.einsum's contraction path took ~2.5 min per call at K=10240)
        d = w - q
        return float(((d @ H) * d).sum() / w.shape[0])
    print(f"{prefix} {tuple(w.shape)} nd={d['nd']} np={d['np']} ({time.time() - t0:.0f} s)  proxy Hd: rtn {pl(qr, hd):.4e} "
          f"gptq {pl(qg, hd):.4e} ({pl(qg, hd) / pl(qr, hd):.3f}x)  Hp: rtn {pl(qr, hp):.4e} gptq {pl(qg, hp):.4e} "
          f"({pl(qg, hp) / pl(qr, hp):.3f}x)  w relerr rtn {float((qr - w).norm() / w.norm()):.4f} "
          f"gptq {float((qg - w).norm() / w.norm()):.4f}", flush=True)
    del d, w, hd, hp, h
    torch.cuda.empty_cache()
torch.save(res, out)
print("wrote", out, list(res))
