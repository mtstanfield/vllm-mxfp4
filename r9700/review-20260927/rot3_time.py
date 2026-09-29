import sys, time, math, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
def mk(P, K, krot=8):
    G = K // 128
    T = torch.zeros(P, krot, K // 2, 4, dtype=torch.int16)
    for p in range(P):
        for r in range(krot):
            perm = torch.stack([torch.randperm(128) for _ in range(G)]).view(G, 64, 2)
            T[p, r, :, 0] = (perm[..., 0] + perm[..., 1] * 256).to(torch.int16).reshape(-1)
    return T
for P, K in ((2, 5120), (3, 5120), (1, 6144), (1, 17408)):
    T = mk(P, K); R3 = torch.zeros_like(T); INIT = torch.zeros(P, K // 128, 32, 4, dtype=torch.int16)
    t0 = time.time(); bad = k.build_rot3(T.data_ptr(), P, 8, K, R3.data_ptr(), INIT.data_ptr()); dt = time.time() - t0
    print(f"P={P} K={K}: {dt * 1000:.1f} ms, failures {bad}")
