"""decode_check.py <libr4d dir> -- R4D paged DECODE attention (split-KV, fp8 KV cache) vs an fp32 reference at the
Qwen3.8 geometry (24 q heads, 4 kv heads, GQA 6, head_dim 256, 16-token blocks). Covers what the serve feeds it:
plain decode (q_len 1), the SPEC-4 verify (q_len 5, causal inside the block), the widest verify (10), two sequences of
different lengths in one launch, contexts that are not a multiple of the tile or the block, and the split count a
captured graph bakes in (max_ctx = 262,144 regardless of the real context). Logits are Gaussian with spread `sigma`;
`vmean` adds a shared component to V (lost numerator mass would show as norm ratio < 1). K/V are e4m3 as the cache
stores them and the reference uses the same dequantized values, so the error is the kernel's own arithmetic.
Physical blocks are shuffled so block-table indirection is exercised."""
import sys, torch
sys.path.insert(0, sys.argv[1])
import r4d
torch.manual_seed(1)
dev = "cuda"
HQ, HK, D, BS = 24, 4, 256, 16
G = HQ // HK
scale = D ** -0.5
st = torch.cuda.current_stream().cuda_stream


def run(ctxs, q_len, sigma, vmean=0.0, max_ctx=None, splits=0):
    ns = len(ctxs)
    maxb = max((c + BS - 1) // BS for c in ctxs)
    nb_tot = sum((c + BS - 1) // BS for c in ctxs)
    perm = torch.randperm(nb_tot, device=dev)
    kv = torch.zeros(nb_tot, HK, BS, 2 * D, dtype=torch.uint8, device=dev)
    bt = torch.zeros(ns, maxb, dtype=torch.int32, device=dev)
    q = torch.randn(ns * q_len, HQ, D, device=dev) * sigma
    qb = q.bfloat16()
    refs, used = [], 0
    for s, c in enumerate(ctxs):
        nb = (c + BS - 1) // BS
        k = torch.randn(c, HK, D, device=dev)
        v = torch.randn(c, HK, D, device=dev) + vmean
        k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        kd, vd = k8.float(), v8.float()
        phys = perm[used:used + nb]
        used += nb
        bt[s, :nb] = phys.int()
        pad = torch.zeros(nb * BS, HK, 2 * D, dtype=torch.uint8, device=dev)
        pad[:c, :, :D] = k8.view(torch.uint8)
        pad[:c, :, D:] = v8.view(torch.uint8)
        kv[phys] = pad.reshape(nb, BS, HK, 2 * D).permute(0, 2, 1, 3)
        qr = qb[s * q_len:(s + 1) * q_len].float()
        ref = torch.empty(q_len, HQ, D, device=dev)
        for h in range(HK):
            qq = qr[:, h * G:(h + 1) * G].reshape(-1, D)                        # [q_len*G, D]
            sc = (qq @ kd[:, h].T) * scale
            lim = (c - q_len + torch.arange(q_len, device=dev)).repeat_interleave(G)
            sc.masked_fill_(torch.arange(c, device=dev)[None, :] > lim[:, None], float("-inf"))
            ref[:, h * G:(h + 1) * G] = (torch.softmax(sc, -1) @ vd[:, h]).reshape(q_len, G, D)
        refs.append(ref)
    ref = torch.cat(refs)
    sl = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    mc = max_ctx or max(ctxs)
    nbytes = r4d.attn_decode_h256_gqa6_scratch_bytes(ns, q_len, HQ, HK, D, mc, splits)
    scratch = torch.full((nbytes + 4096,), 0xFF, dtype=torch.uint8, device=dev)   # NaN garbage, not zeros
    out = torch.full((ns * q_len, HQ, D), float("nan"), dtype=torch.bfloat16, device=dev)
    rc = r4d.attn_decode_h256_gqa6_fp8kv(qb.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                                         0, 0, scratch.data_ptr(), ns, q_len, HQ, HK, D, BS, maxb, kv.stride(0),
                                         kv.stride(1), scale, splits, mc, st)
    torch.cuda.synchronize()
    of = out.float()
    fin = bool(torch.isfinite(of).all())
    rel = ((of - ref).norm(dim=-1) / ref.norm(dim=-1))
    ratio = (of.norm(dim=-1) / ref.norm(dim=-1))
    tail_ok = bool((scratch[nbytes:] == 0xFF).all())                           # nothing written past the sized scratch
    print(f"ctx={str(ctxs):18s} q_len={q_len:2d} sigma={sigma} vmean={vmean} max_ctx={mc:6d} splits={splits:3d} rc={rc} "
          f"relRMSE mean {float(rel.mean()):.2e} max {float(rel.max()):.2e}  norm ratio {float(ratio.mean()):.4f}  "
          f"finite {fin}  scratch-bound {tail_ok}", flush=True)


for c in (1, 7, 16, 17, 100, 1000, 8191, 32773):
    run([c], 1, 2.0)
for c in (5, 16, 21, 8191, 32773, 131072):
    run([c], 5, 2.0)
run([4099], 10, 2.0)
run([32773, 811], 5, 2.0)
run([811, 131072], 1, 2.0)
for c in (100, 8191, 131072):
    run([c], 5, 2.0, max_ctx=262144)                                            # graph-captured split count
for sg in (1.0, 3.0):
    run([131072], 5, sg, vmean=1.0)
run([262144], 5, 2.0, vmean=1.0)
for sp in (1, 16, 128):
    run([32773], 5, 2.0, splits=sp)
