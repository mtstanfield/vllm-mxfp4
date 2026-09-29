"""pq_abl.py <so dir> <label> -- time the rot3 producers at prefill M=8192 (no exactness: ablation builds)."""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
dev = "cuda"; st = lambda: torch.cuda.current_stream().cuda_stream; KROT = 8
torch.manual_seed(0)
def mk(P, K):
    G = K // 128
    T = torch.zeros(P, KROT, K // 2, 4, dtype=torch.int16)
    for p in range(P):
        for r in range(KROT):
            perm = torch.stack([torch.randperm(128) for _ in range(G)]).view(G, 64, 2)
            th = torch.rand(G, 64) * 2 * math.pi
            T[p, r, :, 0] = (perm[..., 0] + perm[..., 1] * 256).to(torch.int16).reshape(-1)
            T[p, r, :, 1] = torch.cos(th).half().view(torch.int16).reshape(-1)
            T[p, r, :, 2] = torch.sin(th).half().view(torch.int16).reshape(-1)
    R3 = torch.zeros_like(T); INIT = torch.zeros(P, G, 32, 4, dtype=torch.int16)
    assert k.build_rot3(T.data_ptr(), P, KROT, K, R3.data_ptr(), INIT.data_ptr()) == 0
    return R3.to(dev), INIT.to(dev)
def timed(f, iters=10):
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters
M = 8192; out = []
for K, P in ((5120, 2), (6144, 1)):
    R3, INIT = mk(P, K); CS = (1 + 0.1 * torch.randn(P, K, device=dev)).half()
    X = (torch.randn(M, K, device=dev) * 3).bfloat16()
    A = torch.zeros(P, M * K, dtype=torch.uint8, device=dev); S = torch.zeros(P, M, device=dev)
    t = timed(lambda: k.launch_rotate_tokquant3_mr(X.data_ptr(), R3.data_ptr(), INIT.data_ptr(), CS.data_ptr(), A.data_ptr(), S.data_ptr(), M, K, P, KROT, st(), 1))
    out.append(f"rot K{K} P{P} {t:7.1f} us ({(M * K * 2 * P + M * K * P) / t / 1e3:4.0f} GB/s)")
for mode, N in ((0, 17408), (2, 6144)):
    R3, INIT = mk(1, N); CS = (1 + 0.1 * torch.randn(1, N, device=dev)).half()
    X = (torch.randn(M, 2 * N if mode == 0 else N, device=dev) * 2).bfloat16(); ys = N + 64
    Y = torch.randn(M, ys, device=dev).bfloat16(); Wn = (1 + 0.1 * torch.randn(128, device=dev)).bfloat16()
    A = torch.zeros(M * N, dtype=torch.uint8, device=dev); S = torch.zeros(M, device=dev); H = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    t = timed(lambda: k.launch_ew_rot_tok3_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, R3.data_ptr(), INIT.data_ptr(), CS.data_ptr(), H.data_ptr(), A.data_ptr(), S.data_ptr(), M, N, KROT, st(), 1, 0))
    rb = M * N * 2 * (2 if mode == 0 else 2) + M * N
    out.append(f"ew{mode} N{N} {t:7.1f} us ({rb / t / 1e3:4.0f} GB/s)")
print(f"{sys.argv[2]:8s} " + " | ".join(out), flush=True)
