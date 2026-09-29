"""mr_check.py <kernel .so dir> -- gate for the multi-row PARO producers (par_kernels_mr.h): byte-identical codes (A),
scales (AS) and, where written, HS against the stock kernels on the same inputs, over M (incl. partial last
workgroups), K/N, P, tiled/row-major and the three ew modes; then timing at prefill M. Records: random perfect matchings
per round, random angles (rot_bench.py's generator)."""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k

dev = "cuda"
torch.manual_seed(0)
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


def rot3_tables(T):
    Tc = T.cpu().contiguous()
    P, krot, HK, _ = Tc.shape
    R3 = torch.zeros_like(Tc)
    INIT = torch.zeros(P, (2 * HK) // 128, 32, 4, dtype=torch.int16)
    bad = k.build_rot3(Tc.data_ptr(), P, krot, 2 * HK, R3.data_ptr(), INIT.data_ptr())
    assert bad == 0, f"build_rot3 failures: {bad}"
    return R3.to(dev), INIT.to(dev)


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
    Mt = (M + 15) // 16
    n = Mt * 16 * K if tiled else M * K
    A0, A1 = torch.zeros(P, n, dtype=torch.uint8, device=dev), torch.zeros(P, n, dtype=torch.uint8, device=dev)
    S0, S1 = torch.zeros(P, M, device=dev), torch.zeros(P, M, device=dev)
    f0 = lambda: k.launch_rotate_tokquant(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, K, P, KROT, st(), tiled)
    f1 = lambda: k.launch_rotate_tokquant_mr(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A1.data_ptr(), S1.data_ptr(), M, K, P, KROT, st(), tiled)
    R3, INIT = rot3_tables(T)
    A2, S2 = torch.zeros_like(A0), torch.zeros_like(S0)
    f2 = lambda: k.launch_rotate_tokquant3_mr(X.data_ptr(), R3.data_ptr(), INIT.data_ptr(), CS.data_ptr(), A2.data_ptr(), S2.data_ptr(), M, K, P, KROT, st(), tiled)
    f0(); f1(); f2(); torch.cuda.synchronize()
    ok = torch.equal(A0, A1) and torch.equal(S0, S1) and torch.equal(A0, A2) and torch.equal(S0, S2)
    fails += not ok
    t = f" stock {timed(f0):8.1f} us  mr {timed(f1):8.1f} us  mr3 {timed(f2):8.1f} us" if time_it else ""
    print(f"rot  M={M:5d} K={K:5d} P={P} tiled={tiled}  {'EXACT' if ok else 'MISMATCH (A %d bytes, AS %d)' % ((A0 != A1).sum(), (S0 != S1).sum())}{t}", flush=True)


def check_ew(mode, M, N, tiled, whs, time_it=False):
    global fails
    T, CS = mk_records(1, N), (1.0 + 0.1 * torch.randn(1, N, device=dev)).half()
    X = (torch.randn(M, 2 * N if mode == 0 else N, device=dev) * 2).bfloat16()
    ys = N + 64
    Y = (torch.randn(M, ys, device=dev)).bfloat16()
    Wn = (1.0 + 0.1 * torch.randn(128, device=dev)).bfloat16()
    Mt = (M + 15) // 16
    n = Mt * 16 * N if tiled else M * N
    A0, A1 = torch.zeros(n, dtype=torch.uint8, device=dev), torch.zeros(n, dtype=torch.uint8, device=dev)
    S0, S1 = torch.zeros(M, device=dev), torch.zeros(M, device=dev)
    H0, H1 = torch.zeros(M, N, dtype=torch.bfloat16, device=dev), torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    f0 = lambda: k.launch_ew_rot_tok(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                     H0.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, N, KROT, st(), tiled)
    f1 = lambda: k.launch_ew_rot_tok_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                        H1.data_ptr(), A1.data_ptr(), S1.data_ptr(), M, N, KROT, st(), tiled, whs)
    R3, INIT = rot3_tables(T)
    A2, S2, H2 = torch.zeros_like(A0), torch.zeros_like(S0), torch.zeros_like(H0)
    f2 = lambda: k.launch_ew_rot_tok3_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, R3.data_ptr(), INIT.data_ptr(),
                                         CS.data_ptr(), H2.data_ptr(), A2.data_ptr(), S2.data_ptr(), M, N, KROT, st(), tiled, whs)
    f0(); f1(); f2(); torch.cuda.synchronize()
    ok = (torch.equal(A0, A1) and torch.equal(S0, S1) and (not whs or torch.equal(H0, H1))
          and torch.equal(A0, A2) and torch.equal(S0, S2) and (not whs or torch.equal(H0, H2)))
    fails += not ok
    t = f" stock {timed(f0):8.1f} us  mr {timed(f1):8.1f} us  mr3 {timed(f2):8.1f} us" if time_it else ""
    print(f"ew{mode} M={M:5d} N={N:5d} tiled={tiled} whs={whs}  {'EXACT' if ok else 'MISMATCH'}{t}", flush=True)


for M in (65, 100, 513, 1000):
    for K, P in ((5120, 2), (6144, 1), (17408, 1), (5120, 3)):
        check_rot(M, K, P, tiled=int(M >= 513))
    check_ew(0, M, 17408, int(M >= 513), 1)
    check_ew(1, M, 6144, int(M >= 513), 1)
    check_ew(2, M, 6144, int(M >= 513), 1)
check_rot(100, 5120, 2, tiled=1)
check_ew(2, 100, 6144, 0, 0)
print("-- prefill M=8192 timing")
for K, P in ((5120, 1), (5120, 2), (6144, 1), (17408, 1)):
    check_rot(8192, K, P, tiled=1, time_it=True)
for mode, N in ((0, 17408), (1, 6144), (2, 6144)):
    check_ew(mode, 8192, N, 1, 1, time_it=True)
    check_ew(mode, 8192, N, 1, 0, time_it=True)
print("ALL EXACT" if fails == 0 else f"FAILURES: {fails}")
