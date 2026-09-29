"""fp8mm_bench.py [tag] -- the MTP drafter's fp8 linears as the serve runs them (torch._scaled_mm, e4m3 x e4m3, per-token
activation scale x per-channel weight scale, bf16 out) at the draft batch sizes M=1 and M=5: median us and GB/s of
weight traffic. Run once plain (hipBLASLt heuristic) and once with PYTORCH_TUNABLEOP_ENABLED=1 TUNING=1 (exhaustive
solution search) to see what the heuristic leaves on the table."""
import sys, torch
tag = sys.argv[1] if len(sys.argv) > 1 else ""
dev = "cuda"
SHAPES = {"fc": (10240, 5120), "qkv": (5120, 14336), "o_proj": (6144, 5120), "gate_up": (5120, 34816), "down": (17408, 5120),
          "lm_head": (5120, 248320)}
MS = (1, 2, 5, 10)
tot = {m: 0.0 for m in MS}
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)     # > the 64 MB Infinity Cache: every call reads DRAM
for name, (K, N) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
    sb = torch.rand(1, N, device=dev) * 0.01 + 0.001
    for M in MS:
        a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        sa = torch.rand(M, 1, device=dev) * 0.01 + 0.001
        f = lambda: torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        for _ in range(20):
            f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(60):
            flush.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); f(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000)
        us = sorted(ts)[len(ts) // 2]
        tot[M] += us if name != "lm_head" else 0.0
        print(f"{tag:8s} {name:8s} M={M} K={K:5d} N={N:5d}  {us:8.1f} us  {N * K / us / 1e3:6.0f} GB/s", flush=True)
print(f"{tag:8s} TOTAL MTP layer per draft forward (cold cache): " + ", ".join(f"M={m} {tot[m]:.0f} us" for m in MS))
