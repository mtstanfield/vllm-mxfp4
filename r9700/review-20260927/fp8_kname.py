"""fp8_kname.py -- which hipBLASLt kernel runs each MTP/lm_head shape (M=1 and M=5), eager and graph-replayed."""
import torch
from torch.profiler import profile, ProfilerActivity
cases = []
for K, N in ((17408, 5120), (10240, 5120), (6144, 5120), (5120, 14336), (5120, 34816), (5120, 248320)):
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    for M in (1, 5):
        a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        sa, sb = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
        cases.append((f"{K}x{N} M={M}", lambda a=a, w=w, sa=sa, sb=sb: torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb,
                                                                                        out_dtype=torch.bfloat16)))
for tag, f in cases:
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    print(f"{tag:14s} {names[-1][:150] if names else '?'}", flush=True)
