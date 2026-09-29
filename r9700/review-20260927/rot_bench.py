"""rot_bench.py <paroquant .so dir> -- time the stock PARO producer pq_rotate_tokquant (channel-scale + krot Givens rounds
per 128-channel group + token amax + e4m3 encode) at prefill M, to find what bounds it. Records are realistic: every
round of every group is a random perfect matching of its 128 channels (64 pairs), random angles, fp16 cos/sin -- the
[P, krot, K/2, 4] u16 layout pq_load_group reads. Reports us/call and two byte counts: row data (read bf16 + write fp8)
and record traffic (each row's workgroup re-reads its groups' records: M x K/128 x krot x 512 B per partition)."""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k

dev = "cuda"
torch.manual_seed(0)


def mk_records(P, krot, K):
    G = K // 128
    T = torch.zeros(P, krot, K // 2, 4, dtype=torch.int16)
    for p in range(P):
        for r in range(krot):
            perm = torch.stack([torch.randperm(128) for _ in range(G)]).view(G, 64, 2)
            ij = (perm[..., 0] + perm[..., 1] * 256).to(torch.int16)
            th = torch.rand(G, 64) * 2 * math.pi
            T[p, r, :, 0] = ij.reshape(-1)
            T[p, r, :, 1] = torch.cos(th).half().view(torch.int16).reshape(-1)
            T[p, r, :, 2] = torch.sin(th).half().view(torch.int16).reshape(-1)
    return T.to(dev)


def bench(M, K, P, krot, tiled=1, iters=10):
    T = mk_records(P, krot, K)
    CS = (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    X = torch.randn(M, K, device=dev).bfloat16()
    Mt = (M + 15) // 16
    A = torch.empty(P, Mt * 16 * K if tiled else M * K, dtype=torch.uint8, device=dev)
    AS = torch.empty(P, M, dtype=torch.float32, device=dev)
    st = torch.cuda.current_stream().cuda_stream
    f = lambda: k.launch_rotate_tokquant(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A.data_ptr(), AS.data_ptr(),
                                         M, K, P, krot, st, tiled)
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / iters
    data = M * K * 2 + P * M * K                       # bf16 row read once per partition-row? (read per (m,p) WG)
    data_rd = P * M * K * 2 + P * M * K               # each (m, p) workgroup reads the row and writes its codes
    rec = P * M * (K // 128) * krot * 512
    print(f"M={M:5d} K={K:5d} P={P} krot={krot}  {us:8.1f} us   rows {data_rd / us / 1e3:6.0f} GB/s   "
          f"records {rec / us / 1e3:6.0f} GB/s (L2)", flush=True)
    return us


for K, P in ((5120, 1), (5120, 2), (6144, 1), (17408, 1)):
    for krot in (1, 4, 8):
        bench(8192, K, P, krot)
bench(512, 5120, 2, 8)
bench(5, 5120, 2, 8, tiled=0)
