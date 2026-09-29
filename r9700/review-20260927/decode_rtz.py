"""decode_rtz.py <libr4d dir> -- is decode's small norm shrink the round-toward-zero f16 packing (v_cvt_pkrtz)?
Reference B folds the query exactly as the kernel does: q_bf16 * scale * log2(e), rounded to f16 TOWARD ZERO. If the
kernel matches B at norm ratio ~1, the shrink is that rounding (a slightly flatter softmax), not an algorithmic loss."""
import sys, math, torch
sys.path.insert(0, sys.argv[1])
import r4d
torch.manual_seed(1)
dev = "cuda"
HQ, HK, D, BS = 24, 4, 256, 16
G, scale = HQ // HK, D ** -0.5
mul = scale * 1.4426950408889634


def rtz16(x):
    y = x.to(torch.float16)
    over = y.float().abs() > x.abs()
    return torch.where(over, torch.nextafter(y, torch.zeros_like(y)), y).float()


def run(c, q_len, sigma, vmean):
    nb = (c + BS - 1) // BS
    k = torch.randn(c, HK, D, device=dev); v = torch.randn(c, HK, D, device=dev) + vmean
    k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    kd, vd = k8.float(), v8.float()
    kv = torch.zeros(nb * BS, HK, 2 * D, dtype=torch.uint8, device=dev)
    kv[:c, :, :D] = k8.view(torch.uint8); kv[:c, :, D:] = v8.view(torch.uint8)
    kv = kv.reshape(nb, BS, HK, 2 * D).permute(0, 2, 1, 3).contiguous()
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, nb)
    qb = (torch.randn(q_len, HQ, D, device=dev) * sigma).bfloat16()
    sl = torch.tensor([c], dtype=torch.int32, device=dev)
    scr = torch.empty(r4d.attn_decode_h256_gqa6_scratch_bytes(1, q_len, HQ, HK, D, c, 0), dtype=torch.uint8, device=dev)
    out = torch.empty(q_len, HQ, D, dtype=torch.bfloat16, device=dev)
    r4d.attn_decode_h256_gqa6_fp8kv(qb.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(), 0, 0,
                                    scr.data_ptr(), 1, q_len, HQ, HK, D, BS, nb, kv.stride(0), kv.stride(1), scale, 0, c,
                                    torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    res = []
    for qf in (qb.float() * mul, rtz16(qb.float() * mul)):                 # A exact fold, B kernel's RTZ fold
        ref = torch.empty(q_len, HQ, D, device=dev)
        for h in range(HK):
            s2 = qf[:, h * G:(h + 1) * G].reshape(-1, D) @ kd[:, h].T          # log2 domain
            lim = (c - q_len + torch.arange(q_len, device=dev)).repeat_interleave(G)
            s2.masked_fill_(torch.arange(c, device=dev)[None, :] > lim[:, None], float("-inf"))
            ref[:, h * G:(h + 1) * G] = (torch.softmax(s2 * math.log(2), -1) @ vd[:, h]).reshape(q_len, G, D)
        of = out.float()
        res.append((float(((of - ref).norm(dim=-1) / ref.norm(dim=-1)).mean()), float((of.norm(dim=-1) / ref.norm(dim=-1)).mean())))
    print(f"ctx={c:6d} q_len={q_len} sigma={sigma} vmean={vmean}  vs exact fold: rel {res[0][0]:.2e} ratio {res[0][1]:.5f}"
          f"   vs RTZ fold: rel {res[1][0]:.2e} ratio {res[1][1]:.5f}", flush=True)


for c, sg, vm in ((8191, 2.0, 0.0), (32773, 2.0, 0.0), (131072, 2.0, 0.0), (131072, 3.0, 1.0), (32773, 1.0, 1.0)):
    run(c, 5, sg, vm)
