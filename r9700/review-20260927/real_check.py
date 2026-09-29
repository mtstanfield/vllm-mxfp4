"""real_check.py <so dir> <dump dir> -- rot4 vs stock on the model's REAL pair records + channel scales (dumped at load
with RADIANCE_PQM_DUMP_REC): norm-fed rotate for every linear (its own P), plus the three ew modes on P=1 linears."""
import glob, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
dev = "cuda"; torch.manual_seed(2)
st = lambda: torch.cuda.current_stream().cuda_stream
fails = n = 0
shapes = {}
for f in sorted(glob.glob(sys.argv[2] + "/rec*.pt"))[:int(sys.argv[3]) if len(sys.argv) > 3 else None]:
    d = torch.load(f)
    rec, cs = d["rec"].contiguous(), d["cs"]
    P, krot, HK, _ = rec.shape
    K = 2 * HK
    shapes[(P, krot, K)] = shapes.get((P, krot, K), 0) + 1
    R4 = torch.zeros_like(rec); F4 = torch.zeros(P, K // 128, 32, 4, dtype=torch.int16)
    b4 = k.build_rot4(rec.data_ptr(), P, krot, K, R4.data_ptr(), F4.data_ptr())
    R3 = torch.zeros_like(rec); F3 = torch.zeros(P, K // 128, 32, 4, dtype=torch.int16)
    b3 = k.build_rot3(rec.data_ptr(), P, krot, K, R3.data_ptr(), F3.data_ptr())
    if b4 or b3:
        print(f"{f}: build3 {b3} build4 {b4}")
    recd, R4, F4 = rec.to(dev), R4.to(dev), F4.to(dev)
    csd = cs.to(dev).half() if cs is not None else (1.0 + 0.1 * torch.randn(P, K, device=dev)).half()
    csd = csd.reshape(P, K).contiguous()
    for M in (777, 2048):
        X = (torch.randn(M, K, device=dev) * 3).bfloat16()
        nb = ((M + 15) // 16) * 16 * K
        A0, A4 = torch.zeros(P, nb, dtype=torch.uint8, device=dev), torch.zeros(P, nb, dtype=torch.uint8, device=dev)
        S0, S4 = torch.zeros(P, M, device=dev), torch.zeros(P, M, device=dev)
        k.launch_rotate_tokquant(X.data_ptr(), recd.data_ptr(), csd.data_ptr(), A0.data_ptr(), S0.data_ptr(), M, K, P, krot, st(), 1)
        k.launch_rotate_tokquant3_mr(X.data_ptr(), R4.data_ptr(), F4.data_ptr(), csd.data_ptr(), A4.data_ptr(), S4.data_ptr(), M, K, P, krot, st(), 1, 1)
        torch.cuda.synchronize()
        ok = torch.equal(A0, A4) and torch.equal(S0, S4)
        n += 1; fails += not ok
        if not ok:
            print(f"{f} P={P} krot={krot} K={K} M={M}: rot MISMATCH A {(A0 != A4).sum().item()} AS {(S0 != S4).sum().item()}", flush=True)
        if P == 1:
            for mode in ((0,) if K > 6144 else (1, 2)):
                N = K
                Xe = (torch.randn(M, 2 * N if mode == 0 else N, device=dev) * 2).bfloat16()
                Y = torch.randn(M, N, device=dev).bfloat16(); Wn = (1 + 0.1 * torch.randn(128, device=dev)).bfloat16()
                a0, a4 = torch.zeros(nb, dtype=torch.uint8, device=dev), torch.zeros(nb, dtype=torch.uint8, device=dev)
                s0, s4 = torch.zeros(M, device=dev), torch.zeros(M, device=dev)
                h0, h4 = torch.zeros(M, N, dtype=torch.bfloat16, device=dev), torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
                k.launch_ew_rot_tok(mode, Xe.data_ptr(), Y.data_ptr(), N, Wn.data_ptr(), 1e-6, recd.data_ptr(), csd.data_ptr(), h0.data_ptr(), a0.data_ptr(), s0.data_ptr(), M, N, krot, st(), 1)
                k.launch_ew_rot_tok3_mr(mode, Xe.data_ptr(), Y.data_ptr(), N, Wn.data_ptr(), 1e-6, R4.data_ptr(), F4.data_ptr(), csd.data_ptr(), h4.data_ptr(), a4.data_ptr(), s4.data_ptr(), M, N, krot, st(), 1, 0, 1)
                torch.cuda.synchronize()
                ok = torch.equal(a0, a4) and torch.equal(s0, s4)
                n += 1; fails += not ok
                if not ok:
                    print(f"{f} K={K} krot={krot} M={M}: ew{mode} MISMATCH A {(a0 != a4).sum().item()}", flush=True)
print("shapes (P, krot, K): count", shapes)
print(f"{n} cases, {fails} mismatches")
