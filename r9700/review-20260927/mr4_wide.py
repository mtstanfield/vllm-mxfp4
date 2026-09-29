"""mr4_wide.py <so dir> -- wider rot4 exactness sweep: random M in the prefill band (odd/even, tails), P 1-3 at
K 5120/6144, krot 1..8, heavy-tailed inputs (outliers, exact zeros, denormals), identity/zero-angle records."""
import math, random, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
dev = "cuda"; torch.manual_seed(1); random.seed(1)
st = lambda: torch.cuda.current_stream().cuda_stream


def mk(P, K, krot, special=False):
    G = K // 128
    T = torch.zeros(P, krot, K // 2, 4, dtype=torch.int16)
    for p in range(P):
        for r in range(krot):
            perm = torch.stack([torch.randperm(128) for _ in range(G)]).view(G, 64, 2)
            th = torch.rand(G, 64) * 2 * math.pi
            if special:                                   # some exact 0 / pi/2 / pi angles and -0.0 sines
                m = torch.rand(G, 64)
                th = torch.where(m < 0.1, torch.zeros_like(th), th)
                th = torch.where((m >= 0.1) & (m < 0.2), torch.full_like(th, math.pi / 2), th)
                th = torch.where((m >= 0.2) & (m < 0.3), torch.full_like(th, math.pi), th)
            s = torch.sin(th).half()
            if special:
                s = torch.where(s == 0, torch.tensor(-0.0).half(), s)
            T[p, r, :, 0] = (perm[..., 0] + perm[..., 1] * 256).to(torch.int16).reshape(-1)
            T[p, r, :, 1] = torch.cos(th).half().view(torch.int16).reshape(-1)
            T[p, r, :, 2] = s.view(torch.int16).reshape(-1)
    return T


def tabs(Tc, which):
    P, krot, HK, _ = Tc.shape
    R = torch.zeros_like(Tc); I = torch.zeros(P, (2 * HK) // 128, 32, 4, dtype=torch.int16)
    bad = (k.build_rot3 if which == 3 else k.build_rot4)(Tc.data_ptr(), P, krot, 2 * HK, R.data_ptr(), I.data_ptr())
    return bad, R.to(dev), I.to(dev)


def xin(M, K):
    x = torch.randn(M, K, device=dev) * 3
    x[torch.rand(M, K, device=dev) < 0.01] *= 200.0                 # outliers
    x[torch.rand(M, K, device=dev) < 0.02] = 0.0                    # exact zeros
    x[random.randrange(M)] = 0.0                                    # an all-zero row
    return x.bfloat16()


fails = 0
cases = 0
for it in range(40):
    M = random.choice([random.randint(65, 9000), 8192, 8191, random.randint(513, 2048)])
    K, P = random.choice([(5120, 1), (5120, 2), (5120, 3), (6144, 1), (6144, 2), (6144, 3)])
    krot = random.choice([8, 8, 8, 7, 5, 4, 2, 1])
    special = it % 3 == 0
    T = mk(P, K, krot, special); Tc = T.contiguous()
    b3, R3, I3 = tabs(Tc, 3); b4, R4, I4 = tabs(Tc, 4)
    T = T.to(dev); CS = (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    X = xin(M, K); tiled = int(M >= 513)
    n = ((M + 15) // 16) * 16 * K if tiled else M * K
    A0, A4 = torch.zeros(P, n, dtype=torch.uint8, device=dev), torch.zeros(P, n, dtype=torch.uint8, device=dev)
    S0, S4 = torch.zeros(P, M, device=dev), torch.zeros(P, M, device=dev)
    k.launch_rotate_tokquant(X.data_ptr(), T.data_ptr(), CS.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, K, P, krot, st(), tiled)
    k.launch_rotate_tokquant3_mr(X.data_ptr(), R4.data_ptr(), I4.data_ptr(), CS.data_ptr(), A4.data_ptr(), S4.data_ptr(), M, K, P, krot, st(), tiled, 1)
    torch.cuda.synchronize()
    ok = torch.equal(A0, A4) and torch.equal(S0, S4)
    cases += 1; fails += not ok
    if not ok or it < 3:
        print(f"rot M={M} K={K} P={P} krot={krot} special={special} build3={b3} build4={b4}: {'EXACT' if ok else 'MISMATCH A %d AS %d' % ((A0 != A4).sum(), (S0 != S4).sum())}", flush=True)
for it in range(30):
    mode = it % 3
    N = 17408 if mode == 0 else 6144
    M = random.choice([random.randint(65, 9000), 8192, random.randint(513, 2048)])
    krot = random.choice([8, 8, 6, 3])
    special = it % 2 == 0
    T = mk(1, N, krot, special); Tc = T.contiguous()
    b4, R4, I4 = tabs(Tc, 4)
    T = T.to(dev); CS = (1.0 + 0.1 * torch.randn(1, N, device=dev)).half()
    X = xin(M, 2 * N if mode == 0 else N); ys = N + 64
    Y = (torch.randn(M, ys, device=dev) * 2).bfloat16(); Wn = (1.0 + 0.1 * torch.randn(128, device=dev)).bfloat16()
    tiled = int(M >= 513); n = ((M + 15) // 16) * 16 * N if tiled else M * N
    A0, A4 = torch.zeros(n, dtype=torch.uint8, device=dev), torch.zeros(n, dtype=torch.uint8, device=dev)
    S0, S4 = torch.zeros(M, device=dev), torch.zeros(M, device=dev)
    H0, H4 = torch.zeros(M, N, dtype=torch.bfloat16, device=dev), torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    k.launch_ew_rot_tok(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, T.data_ptr(), CS.data_ptr(), H0.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, N, krot, st(), tiled)
    k.launch_ew_rot_tok3_mr(mode, X.data_ptr(), Y.data_ptr(), ys, Wn.data_ptr(), 1e-6, R4.data_ptr(), I4.data_ptr(), CS.data_ptr(), H4.data_ptr(), A4.data_ptr(), S4.data_ptr(), M, N, krot, st(), tiled, 0, 1)
    torch.cuda.synchronize()
    ok = torch.equal(A0, A4) and torch.equal(S0, S4)
    cases += 1; fails += not ok
    if not ok:
        print(f"ew{mode} M={M} N={N} krot={krot} special={special} build4={b4}: MISMATCH A {(A0 != A4).sum()} AS {(S0 != S4).sum()}", flush=True)
print(f"{cases} cases, {fails} mismatches")
