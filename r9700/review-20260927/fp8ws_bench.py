"""fp8ws_bench.py -- M=1 torch._scaled_mm (rowwise fp8) on the MTP / lm_head shapes, warm cache: does the hipBLASLt
workspace size (HIPBLASLT_WORKSPACE_SIZE / CUBLASLT_WORKSPACE_SIZE, KiB) change the solution the heuristic picks?"""
import os, torch


def med(f, n=100):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    return sorted(ts)[n // 2]


out = []
for K, N in ((17408, 5120), (10240, 5120), (6144, 5120), (5120, 14336), (5120, 34816), (5120, 248320)):
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    a = torch.randn(1, K, device="cuda").to(torch.float8_e4m3fn)
    sa, sb = torch.ones(1, 1, device="cuda"), torch.ones(1, N, device="cuda")
    f = lambda: torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    out.append(f"{K}x{N} {med(f):.0f}")
print(f"WS={os.environ.get('HIPBLASLT_WORKSPACE_SIZE', 'unset'):>7s} TUNABLE={os.environ.get('PYTORCH_TUNABLEOP_ENABLED', '0')}  "
      + "  ".join(out), flush=True)
