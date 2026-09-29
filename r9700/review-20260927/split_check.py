"""split_check.py <kernel .so dir> -- gate for the split-token decode producers (par_kernels_split.h): codes (A), scales
(AS) and HS byte-identical to the stock launch_rotate_tokquant / launch_ew_rot_tok (row-major) on the same inputs, over
decode-band M, K/N, P and the three ew modes, repeated launches (the per-token counters must reset), a strided gate
row (ys > N, as the GDN no-copy path hands it); then timing at M = 5 / 8 / 16 (the V2 verify sizes)."""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k

dev = "cuda"
torch.manual_seed(0)
st = lambda: torch.cuda.current_stream().cuda_stream
KROT = 8
MMAX, NMAX = 64, 18432
SCR = torch.empty(3 * MMAX * NMAX, dtype=torch.bfloat16, device=dev)
GAMAX = torch.zeros(3 * MMAX, dtype=torch.int32, device=dev)
GCNT = torch.zeros(3 * MMAX, dtype=torch.int32, device=dev)


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


def timed(f, iters=50):
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters


fails = 0


def counters_clean():
    return int(GAMAX.abs().sum()) == 0 and int(GCNT.abs().sum()) == 0


def check_rot(M, K, P, time_it=False):
    global fails
    T, CS = mk_records(P, K), (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    X = (torch.randn(M, K, device=dev) * 3).bfloat16()
    X[0, :7] = 0                                        # a few exact zeros / a large outlier
    X[-1, 5] = 300.0
    A0, A1 = torch.zeros(P, M * K, dtype=torch.uint8, device=dev), torch.full((P, M * K), 7, dtype=torch.uint8, device=dev)
    S0, S1 = torch.zeros(P, M, device=dev), torch.full((P, M), -1.0, device=dev)
    f0 = lambda: k.launch_rotate_tokquant(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, K, P, KROT, st(), 0)
    f1 = lambda: k.launch_rotate_tokquant_split(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A1.data_ptr(), S1.data_ptr(),
                                                SCR.data_ptr(), GAMAX.data_ptr(), GCNT.data_ptr(), M, K, P, KROT, st())
    f0(); f1(); f1(); torch.cuda.synchronize()          # twice: the second launch must see reset counters
    ok = torch.equal(A0, A1) and torch.equal(S0, S1) and counters_clean()
    fails += not ok
    t = f"  stock {timed(f0):7.2f} us  split {timed(f1):7.2f} us" if time_it else ""
    print(f"rot   M={M:3d} K={K:5d} P={P}  {'EXACT' if ok else 'MISMATCH (A %d, AS %d, clean %s)' % ((A0 != A1).sum(), (S0 != S1).sum(), counters_clean())}{t}", flush=True)


def check_ew(mode, M, N, ys_pad=64, time_it=False):
    global fails
    T, CS = mk_records(1, N), (1.0 + 0.1 * torch.randn(1, N, device=dev)).half()
    X = (torch.randn(M, 2 * N if mode == 0 else N, device=dev) * 2).bfloat16()
    ys = N + ys_pad
    Y = (torch.randn(M, ys, device=dev)).bfloat16()
    Wn = (1.0 + 0.1 * torch.randn(128, device=dev)).bfloat16()
    A0, A1 = torch.zeros(M * N, dtype=torch.uint8, device=dev), torch.full((M * N,), 7, dtype=torch.uint8, device=dev)
    S0, S1 = torch.zeros(M, device=dev), torch.full((M,), -1.0, device=dev)
    H0, H1 = torch.zeros(M, N, dtype=torch.bfloat16, device=dev), torch.ones(M, N, dtype=torch.bfloat16, device=dev)
    f0 = lambda: k.launch_ew_rot_tok(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                     H0.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, N, KROT, st(), 0)
    f1 = lambda: k.launch_ew_rot_tok_split(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(),
                                           CS.data_ptr(), H1.data_ptr(), A1.data_ptr(), S1.data_ptr(), SCR.data_ptr(),
                                           GAMAX.data_ptr(), GCNT.data_ptr(), M, N, KROT, st())
    f0(); f1(); f1(); torch.cuda.synchronize()
    ok = torch.equal(A0, A1) and torch.equal(S0, S1) and torch.equal(H0, H1) and counters_clean()
    fails += not ok
    t = f"  stock {timed(f0):7.2f} us  split {timed(f1):7.2f} us" if time_it else ""
    print(f"ew{mode}   M={M:3d} N={N:5d} ys={ys}  {'EXACT' if ok else 'MISMATCH (A %d, AS %d, HS %d)' % ((A0 != A1).sum(), (S0 != S1).sum(), (H0 != H1).sum())}{t}", flush=True)


def check_arr(M, K, P, time_it=False):
    """stream 1: residual add + Gemma RMSNorm + rotate + token quant (HS, RO, A, AS) vs launch_add_rms_rot_tok."""
    global fails
    T, CS = mk_records(P, K), (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    Y = (torch.randn(M, K, device=dev) * 2).bfloat16()
    R = (torch.randn(M, K, device=dev) * 4).bfloat16()
    Wn = (0.1 * torch.randn(K, device=dev)).bfloat16()
    outs = []
    for fill in (0, 7):
        outs.append((torch.full((M, K), fill, dtype=torch.bfloat16, device=dev), torch.full((M, K), fill, dtype=torch.bfloat16, device=dev),
                     torch.full((P, M * K), fill, dtype=torch.uint8, device=dev), torch.full((P, M), float(fill), device=dev)))
    (H0, R0, A0, S0), (H1, R1, A1, S1) = outs
    f0 = lambda: k.launch_add_rms_rot_tok(Y.data_ptr(), R.data_ptr(), Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                          H0.data_ptr(), R0.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, K, P, KROT, st(), 0)
    f1 = lambda: k.launch_add_rms_rot_tok_split(Y.data_ptr(), R.data_ptr(), Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(),
                                                H1.data_ptr(), R1.data_ptr(), A1.data_ptr(), S1.data_ptr(), SCR.data_ptr(),
                                                GAMAX.data_ptr(), GCNT.data_ptr(), M, K, P, KROT, st())
    f0(); f1(); f1(); torch.cuda.synchronize()
    ok = (torch.equal(H0, H1) and torch.equal(R0, R1) and torch.equal(A0, A1) and torch.equal(S0, S1)
          and counters_clean())
    fails += not ok
    t = f"  stock {timed(f0):7.2f} us  split {timed(f1):7.2f} us" if time_it else ""
    print(f"arr   M={M:3d} K={K:5d} P={P}  {'EXACT' if ok else 'MISMATCH (HS %d RO %d A %d AS %d)' % ((H0 != H1).sum(), (R0 != R1).sum(), (A0 != A1).sum(), (S0 != S1).sum())}{t}", flush=True)


HAS_ARR = hasattr(k, "launch_add_rms_rot_tok_split")
if HAS_ARR:
    for M in (1, 2, 3, 5, 8, 12, 16):
        for K, P in ((5120, 1), (5120, 2), (5120, 3)):
            check_arr(M, K, P)
    for M in (5, 8):
        check_arr(M, 5120, 1, time_it=True)
        check_arr(M, 5120, 2, time_it=True)

for M in (1, 2, 3, 5, 8, 12, 16, 33, 64):
    for K, P in ((5120, 1), (5120, 2), (6144, 1), (17408, 1), (5120, 3)):
        check_rot(M, K, P)
    check_ew(0, M, 17408)
    check_ew(1, M, 6144)
    check_ew(2, M, 6144)
check_ew(2, 5, 6144, ys_pad=16384 - 6144)          # the GDN no-copy gate row stride
print("-- timing (decode band)")
for M in (5, 8, 16):
    for K, P in ((5120, 1), (5120, 2), (6144, 1)):
        check_rot(M, K, P, time_it=True)
    for mode, N in ((0, 17408), (1, 6144), (2, 6144)):
        check_ew(mode, M, N, time_it=True)
print("ALL EXACT" if fails == 0 else f"FAILURES: {fails}")
