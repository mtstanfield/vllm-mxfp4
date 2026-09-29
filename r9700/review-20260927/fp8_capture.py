"""fp8_capture.py -- with the tuned TunableOp table loaded: does a HIP graph that captures the FIRST-EVER call of a shape
bake the tuned solution or the default? (down 17408x5120 M=1: tuned ~166 us, default ~312 us, warm cache)."""
import torch


def med(f, n=100):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    return sorted(ts)[n // 2]


def mk(K, N, M):
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    sa, sb = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
    return lambda: torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


warm = mk(10240, 5120, 5)          # a DIFFERENT tuned shape first: the table is loaded, the context is up
warm(); torch.cuda.synchronize()
f = mk(17408, 5120, 1)             # never called eagerly before capture
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.graph(g):
    f()
torch.cuda.synchronize()
print(f"graph whose capture was the first call of the shape: {med(g.replay):.1f} us")
print(f"same shape, eager afterwards:                         {med(f):.1f} us")
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    f()
print(f"graph captured after eager calls:                     {med(g2.replay):.1f} us")
