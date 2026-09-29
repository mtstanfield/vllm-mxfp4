"""mxdec_bench.py <dir with radiance_mxfp4_fp8.so> <label> -- the MXFP4 W4A8 decode GEMM at the production TP=1 shapes,
M = 1 and 5, DRAM-fed (weight copies rotate, > 3x the 64 MB MALL), prod knobs (WPERM=1, DECODE_NT=1, DECODE_MAX_M=64).
RADIANCE_MXFP4_DECODE_KS forces the split (whole process). Prints us/call and weight GB/s."""
import os, sys
for k, v in (("RADIANCE_MXFP4_WPERM", "1"), ("RADIANCE_MXFP4_DECODE_NT", "1"), ("RADIANCE_MXFP4_DECODE_MAX_M", "64")):
    os.environ.setdefault(k, v)
sys.path.insert(0, sys.argv[1])
import torch
import radiance_mxfp4_fp8 as ext
label = sys.argv[2]
dev = "cuda"
SH = [(5120, 6144, "out/o"), (5120, 17408, "down"), (16384, 5120, "qkvz"), (34816, 5120, "gate_up"), (5120, 10240, "mtp.fc")]
scr = torch.empty(8 * 64 * 36864, dtype=torch.float32, device=dev)
cnt = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
ext.set_decode_scratch(scr.data_ptr(), scr.numel() * 4, cnt.data_ptr())
st = torch.cuda.current_stream().cuda_stream
torch.manual_seed(0)
row = []
for N, K, name in SH:
    wb = N * K // 2 + N * K // 32 + N
    nc = max(2, -(-3 * 64 * 2**20 // wb))
    W = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev) for _ in range(nc)]
    WS = [torch.randint(118, 130, (K // 32, N), dtype=torch.uint8, device=dev) for _ in range(nc)]
    WR = [ws.max(0).values.contiguous() for ws in WS]
    for M in (1, 5):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        xs = torch.ones(M, dtype=torch.float32, device=dev)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        call = lambda i: ext.launch(x.data_ptr(), W[i % nc].data_ptr(), WS[i % nc].data_ptr(), WR[i % nc].data_ptr(),
                                    xs.data_ptr(), out.data_ptr(), M, N, K, st)
        for i in range(10):
            call(i)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 200
        e0.record()
        for i in range(reps):
            call(i)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / reps
        row.append(f"{name} M{M}: {us:6.1f} us {wb / us / 1e3:4.0f} GB/s")
print(f"{label:8s} " + " | ".join(row), flush=True)
