"""gemm_bench.py <mxfp4 ext dir> -- prefill GEMM (M=8192) at the production TP=1 shapes: the MXFP4 A-tiled kernel
(launch_at, fragment-tiled fp8 A, WPERM layout, prod knobs), hipBLASLt fp8 x fp8 (torch._scaled_mm, the vendor library
with 8-bit weights and no dequant), and bf16 torch.mm; TFLOP/s with the GPU clock (sclk) and power sampled during each
timed loop from sysfs."""
import glob, os, statistics as st, sys, threading, time
for k_, v_ in (("RADIANCE_MXFP4_WPERM", "1"), ("RADIANCE_MXFP4_DECODE_NT", "1"), ("RADIANCE_MXFP4_DECODE_MAX_M", "64"),
               ("RADIANCE_MXFP4_TN4_MIN_M", "2048"), ("RADIANCE_MXFP4_EPIFAST", "1")):
    os.environ.setdefault(k_, v_)
sys.path.insert(0, sys.argv[1])
import torch
import radiance_mxfp4_fp8 as ext
dev = "cuda"
HW = [h for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*") if os.path.exists(h + "/power1_cap")][0]


class Sampler:
    def __enter__(self):
        self.f, self.p, self.run = [], [], True
        def loop():
            while self.run:
                try:
                    self.f.append(int(open(HW + "/freq1_input").read()) / 1e6)
                    self.p.append(int(open(HW + "/power1_average").read()) / 1e6)
                except OSError:
                    pass
                time.sleep(0.05)
        self.t = threading.Thread(target=loop); self.t.start(); return self
    def __exit__(self, *a):
        self.run = False; self.t.join()
    def s(self):
        return f"{st.median(self.f):5.0f} MHz {st.median(self.p):4.0f} W" if self.f else "n/a"


def timed(fn, flop, secs=1.5):
    fn(); torch.cuda.synchronize()
    t0 = time.time(); fn(); torch.cuda.synchronize()
    reps = max(3, min(2000, int(secs / max(time.time() - t0, 1e-5))))
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with Sampler() as sm:
        e0.record()
        for _ in range(reps): fn()
        e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / reps
    return ms, flop / (ms * 1e-3) / 1e12, sm.s()


M = 8192
SH = [("gate_up", 34816, 5120), ("qkvz", 16384, 5120), ("attn_qkv", 14336, 5120), ("out/o", 5120, 6144), ("down", 5120, 17408)]
st_ = torch.cuda.current_stream().cuda_stream
tot = {"mx": 0.0, "hb": 0.0, "flop": 0.0}
for name, N, K in SH:
    flop = 2.0 * M * N * K
    A = torch.randint(0, 0x7F, (((M + 15) // 16) * 16 * K,), dtype=torch.uint8, device=dev) & 0xBF
    As = torch.ones(M, dtype=torch.float32, device=dev)
    W = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    WS = torch.randint(118, 130, (K // 32, N), dtype=torch.uint8, device=dev)
    WR = WS.max(0).values.contiguous()
    C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    ms, tf, cl = timed(lambda: ext.launch_at(A.data_ptr(), W.data_ptr(), WS.data_ptr(), WR.data_ptr(), As.data_ptr(), C.data_ptr(), M, N, K, st_), flop)
    a8 = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    b8 = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    ms2, tf2, cl2 = timed(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), flop)
    ab, bb = torch.randn(M, K, device=dev, dtype=torch.bfloat16), torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ms3, tf3, cl3 = timed(lambda: torch.mm(ab, bb), flop)
    tot["mx"] += ms; tot["hb"] += ms2; tot["flop"] += flop
    print(f"{name:9s} N={N:5d} K={K:5d}: MXFP4 A-tiled {ms:6.2f} ms {tf:5.1f} TF/s [{cl}] | hipBLASLt fp8 {ms2:6.2f} ms "
          f"{tf2:5.1f} TF/s [{cl2}] | bf16 {tf3:5.1f} TF/s [{cl3}]", flush=True)
    del A, W, WS, WR, C, a8, b8, ab, bb
    torch.cuda.empty_cache()
print(f"all five (one layer's worth, qkvz+out as GDN): MXFP4 {tot['flop'] / tot['mx'] / 1e9:.1f} TF/s, "
      f"hipBLASLt fp8 {tot['flop'] / tot['hb'] / 1e9:.1f} TF/s")
