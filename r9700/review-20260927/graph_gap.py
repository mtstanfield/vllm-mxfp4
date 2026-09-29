"""graph_gap.py -- per-kernel cost of a replayed HIP graph of N tiny dependent kernels (the decode step is ~1,300 kernels
with a ~3.5 us gap each). Env knobs are read by the HIP runtime at start, so run one process per setting.
Prints us/kernel for a chain of in-place adds on a small tensor (pure launch cost) and on a 1 MB tensor."""
import os, sys, torch
N = int(os.environ.get("GG_N", "1000"))
res = []
for numel in (256, 1 << 18):
    x = torch.zeros(numel, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(N):
            x.add_(1.0)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = 20
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    res.append(e0.elapsed_time(e1) * 1000 / reps / N)
knobs = " ".join(f"{k}={os.environ[k]}" for k in sorted(os.environ) if k.startswith(("DEBUG_HIP", "DEBUG_CLR", "ROC_", "GPU_", "AMD_", "HIP_")))
print(f"{res[0]:6.2f} us/kernel (1 KB)  {res[1]:6.2f} us/kernel (1 MB)   {knobs or 'defaults'}", flush=True)
