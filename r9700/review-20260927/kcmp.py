"""kcmp.py <so dir> <out.pt> [dump dir] -- run the serve's producer entry points on fixed inputs (real records from the
dump if given) and save every output, to diff two kernel builds byte for byte (python3 kcmp.py A a.pt; ... B b.pt)."""
import glob, math, sys, torch
sys.path.insert(0, sys.argv[1])
import radiance_paroquant_kernel as k
dev = "cuda"; st = lambda: torch.cuda.current_stream().cuda_stream
out = {}
recs = sorted(glob.glob(sys.argv[3] + "/rec*.pt"))[:256] if len(sys.argv) > 3 else []
by = {}
for f in recs:
    d = torch.load(f); r = d["rec"]; key = (r.shape[0], 2 * r.shape[2])
    by.setdefault(key, (r, d["cs"]))
SMAX, PMAX, NMAX = 16, 3, 18432
scr = torch.zeros(PMAX * SMAX * NMAX, dtype=torch.bfloat16, device=dev)
amax = torch.zeros(PMAX * SMAX, dtype=torch.int32, device=dev); cnt = torch.zeros(PMAX * SMAX, dtype=torch.int32, device=dev)
for (P, K), (rec, cs) in sorted(by.items()):
    g = torch.Generator(device="cpu").manual_seed(P * 100000 + K)
    krot = rec.shape[1]; recd = rec.contiguous().to(dev); csd = cs.to(dev).half().reshape(P, K).contiguous()
    R3 = torch.zeros_like(rec); I3 = torch.zeros(P, K // 128, 32, 4, dtype=torch.int16)
    k.build_rot3(rec.contiguous().data_ptr(), P, krot, K, R3.data_ptr(), I3.data_ptr())
    out[f"tab3_{P}_{K}"] = (R3.clone(), I3.clone())
    R3, I3 = R3.to(dev), I3.to(dev)
    for M in (5, 8, 16, 100, 777, 5337, 7762, 8192):
        X = (torch.randn(M, K, generator=g) * 3).bfloat16().to(dev)
        tiled = int(M >= 513)
        nb = ((M + 15) // 16) * 16 * K if tiled else M * K
        A = torch.zeros(P, nb, dtype=torch.uint8, device=dev); S = torch.zeros(P, M, device=dev)
        if M <= 16:
            k.launch_rotate_tokquant_split(X.data_ptr(), recd.data_ptr(), csd.data_ptr(), A.data_ptr(), S.data_ptr(),
                                           scr.data_ptr(), amax.data_ptr(), cnt.data_ptr(), M, K, P, krot, st())
        else:
            k.launch_rotate_tokquant3_mr(X.data_ptr(), R3.data_ptr(), I3.data_ptr(), csd.data_ptr(), A.data_ptr(), S.data_ptr(), M, K, P, krot, st(), tiled)
        A2 = torch.zeros_like(A); S2 = torch.zeros_like(S)
        k.launch_rotate_tokquant(X.data_ptr(), recd.data_ptr(), csd.data_ptr(), A2.data_ptr(), S2.data_ptr(), M, K, P, krot, st(), tiled)
        torch.cuda.synchronize()
        out[f"rot_{P}_{K}_{M}"] = (A.cpu(), S.cpu(), A2.cpu(), S2.cpu())
        if P == 1:
            for mode in ((0,) if K > 6144 else (1, 2)):
                N = K
                Xe = (torch.randn(M, 2 * N if mode == 0 else N, generator=g) * 2).bfloat16().to(dev)
                Y = torch.randn(M, N, generator=g).bfloat16().to(dev); Wn = (1 + 0.1 * torch.randn(128, generator=g)).bfloat16().to(dev)
                a = torch.zeros(nb, dtype=torch.uint8, device=dev); s = torch.zeros(M, device=dev); h = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
                if M <= 16:
                    k.launch_ew_rot_tok_split(mode, Xe.data_ptr(), Y.data_ptr(), N, Wn.data_ptr(), 1e-6, recd.data_ptr(), csd.data_ptr(), h.data_ptr(), a.data_ptr(), s.data_ptr(),
                                              scr.data_ptr(), amax.data_ptr(), cnt.data_ptr(), M, N, krot, st())
                else:
                    k.launch_ew_rot_tok3_mr(mode, Xe.data_ptr(), Y.data_ptr(), N, Wn.data_ptr(), 1e-6, R3.data_ptr(), I3.data_ptr(), csd.data_ptr(), h.data_ptr(), a.data_ptr(), s.data_ptr(), M, N, krot, st(), tiled, 0)
                torch.cuda.synchronize()
                out[f"ew{mode}_{K}_{M}"] = (a.cpu(), s.cpu(), h.cpu())
        # add + rms + rot (norm feed, pass 1 of the prefill path)
        if M > 16:
            y = (torch.randn(M, K, generator=g)).bfloat16().to(dev); res = (torch.randn(M, K, generator=g)).bfloat16().to(dev)
            w = (1 + 0.1 * torch.randn(K, generator=g)).bfloat16().to(dev)
            hs = torch.zeros(M, K, dtype=torch.bfloat16, device=dev); ro = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
            k.launch_add_rms_rot(y.data_ptr(), res.data_ptr(), w.data_ptr(), 1e-6, recd.data_ptr(), csd.data_ptr(), hs.data_ptr(), ro.data_ptr(), 0, 0, 0, M, K, P, krot, 0, st(), 0)
            torch.cuda.synchronize()
            out[f"arr_{P}_{K}_{M}"] = (hs.cpu(), ro.cpu())
torch.save(out, sys.argv[2])
print("saved", len(out), "entries")
