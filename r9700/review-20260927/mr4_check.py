"""mr4_check.py <kernel .so dir> -- gate for the select-free rot4 producers: codes/scales (and HS where written)
byte-identical to the stock per-token kernels AND to rot3, over M (incl. partial workgroups), K/N, P, tiled/row-major,
the three ew modes; then timing at prefill M=8192 (stock / rot3 / rot4). Records as mr_check.py."""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
dev = "cuda"; torch.manual_seed(0)
st = lambda: torch.cuda.current_stream().cuda_stream
KROT = 8


def mk_records(P, K, krot=KROT):
    G = K // 128
    T = torch.zeros(P, krot, K // 2, 4, dtype=torch.int16)
    for p in range(P):
        for r in range(krot):
            perm = torch.stack([torch.randperm(128) for _ in range(G)]).view(G, 64, 2)
            th = torch.rand(G, 64) * 2 * math.pi
            T[p, r, :, 0] = (perm[..., 0] + perm[..., 1] * 256).to(torch.int16).reshape(-1)
            T[p, r, :, 1] = torch.cos(th).half().view(torch.int16).reshape(-1)
            T[p, r, :, 2] = torch.sin(th).half().view(torch.int16).reshape(-1)
    return T.to(dev)


def tables(T, which):
    Tc = T.cpu().contiguous()
    P, krot, HK, _ = Tc.shape
    R = torch.zeros_like(Tc)
    I = torch.zeros(P, (2 * HK) // 128, 32, 4, dtype=torch.int16)
    bad = (k.build_rot3 if which == 3 else k.build_rot4)(Tc.data_ptr(), P, krot, 2 * HK, R.data_ptr(), I.data_ptr())
    assert bad == 0, f"build_rot{which} failures: {bad}"
    return R.to(dev), I.to(dev)


def timed(f, iters=10):
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters


fails = 0


def check_rot(M, K, P, tiled, time_it=False):
    global fails
    T, CS = mk_records(P, K), (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    X = (torch.randn(M, K, device=dev) * 3).bfloat16()
    n = ((M + 15) // 16) * 16 * K if tiled else M * K
    A = [torch.zeros(P, n, dtype=torch.uint8, device=dev) for _ in range(3)]
    S = [torch.zeros(P, M, device=dev) for _ in range(3)]
    R3, I3 = tables(T, 3); R4, I4 = tables(T, 4)
    f0 = lambda: k.launch_rotate_tokquant(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A[0].data_ptr(), S[0].data_ptr(), M, K, P, KROT, st(), tiled)
    f3 = lambda: k.launch_rotate_tokquant3_mr(X.data_ptr(), R3.data_ptr(), I3.data_ptr(), CS.data_ptr(), A[1].data_ptr(), S[1].data_ptr(), M, K, P, KROT, st(), tiled, 0)
    f4 = lambda: k.launch_rotate_tokquant3_mr(X.data_ptr(), R4.data_ptr(), I4.data_ptr(), CS.data_ptr(), A[2].data_ptr(), S[2].data_ptr(), M, K, P, KROT, st(), tiled, 1)
    f0(); f3(); f4(); torch.cuda.synchronize()
    ok = all(torch.equal(A[0], A[i]) and torch.equal(S[0], S[i]) for i in (1, 2))
    fails += not ok
    t = f"  stock {timed(f0):7.1f}  rot3 {timed(f3):7.1f}  rot4 {timed(f4):7.1f} us" if time_it else ""
    print(f"rot  M={M:5d} K={K:5d} P={P} tiled={tiled}  {'EXACT' if ok else 'MISMATCH A %d AS %d' % ((A[0] != A[2]).sum(), (S[0] != S[2]).sum())}{t}", flush=True)


def check_ew(mode, M, N, tiled, whs, time_it=False):
    global fails
    T, CS = mk_records(1, N), (1.0 + 0.1 * torch.randn(1, N, device=dev)).half()
    X = (torch.randn(M, 2 * N if mode == 0 else N, device=dev) * 2).bfloat16()
    ys = N + 64
    Y = torch.randn(M, ys, device=dev).bfloat16()
    Wn = (1.0 + 0.1 * torch.randn(128, device=dev)).bfloat16()
    n = ((M + 15) // 16) * 16 * N if tiled else M * N
    A = [torch.zeros(n, dtype=torch.uint8, device=dev) for _ in range(3)]
    S = [torch.zeros(M, device=dev) for _ in range(3)]
    H = [torch.zeros(M, N, dtype=torch.bfloat16, device=dev) for _ in range(3)]
    R3, I3 = tables(T, 3); R4, I4 = tables(T, 4)
    f0 = lambda: k.launch_ew_rot_tok(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                     H[0].data_ptr(), A[0].data_ptr(), S[0].data_ptr(), M, N, KROT, st(), tiled)
    f3 = lambda: k.launch_ew_rot_tok3_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, R3.data_ptr(), I3.data_ptr(),
                                         CS.data_ptr(), H[1].data_ptr(), A[1].data_ptr(), S[1].data_ptr(), M, N, KROT, st(), tiled, whs, 0)
    f4 = lambda: k.launch_ew_rot_tok3_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, R4.data_ptr(), I4.data_ptr(),
                                         CS.data_ptr(), H[2].data_ptr(), A[2].data_ptr(), S[2].data_ptr(), M, N, KROT, st(), tiled, whs, 1)
    f0(); f3(); f4(); torch.cuda.synchronize()
    ok = all(torch.equal(A[0], A[i]) and torch.equal(S[0], S[i]) and (not whs or torch.equal(H[0], H[i])) for i in (1, 2))
    fails += not ok
    t = f"  stock {timed(f0):7.1f}  rot3 {timed(f3):7.1f}  rot4 {timed(f4):7.1f} us" if time_it else ""
    print(f"ew{mode} M={M:5d} N={N:5d} tiled={tiled} whs={whs}  {'EXACT' if ok else 'MISMATCH'}{t}", flush=True)


for M in (65, 100, 513, 1000):
    for K, P in ((5120, 2), (6144, 1), (17408, 1), (5120, 3)):
        check_rot(M, K, P, tiled=int(M >= 513))
    check_ew(0, M, 17408, int(M >= 513), 1)
    check_ew(1, M, 6144, int(M >= 513), 1)
    check_ew(2, M, 6144, int(M >= 513), 1)
check_rot(100, 5120, 2, tiled=0)
check_ew(2, 100, 6144, 0, 0)
print("-- prefill M=8192 timing")
for K, P in ((5120, 1), (5120, 2), (6144, 1), (17408, 1)):
    check_rot(8192, K, P, tiled=1, time_it=True)
for mode, N in ((0, 17408), (1, 6144), (2, 6144)):
    check_ew(mode, 8192, N, 1, 0, time_it=True)
print("ALL EXACT" if fails == 0 else f"FAILURES: {fails}")
