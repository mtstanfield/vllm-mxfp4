"""attn_bench.py <variant dir> <tag> [Q] [depths...] -- time R4D paged PREFILL attention (fp8 KV, R4D_ATTN_FP8 from the
environment, production = 15) at the Qwen3.8 geometry (24 q heads, 4 kv heads, head_dim 256, 16-token blocks): one
chunked-prefill step of Q queries at the END of an N-token context (causal), for each depth N. Imports r4d from the
variant dir, so run one process per variant. Saves each output to /tmp/attnb/<tag>_<N>.pt so a driver can compare
variants bit for bit; prints ms, effective TFLOPS (QK + PV, causal pairs only)."""
import os, sys, time, torch
sys.path.insert(0, sys.argv[1])
import r4d
tag = sys.argv[2]
Q = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
depths = [int(x) for x in sys.argv[4:]] or [8192, 32768, 65536, 131072]
torch.manual_seed(1)
dev = "cuda"
HQ, HK, D, BS = 24, 4, 256, 16
scale = D ** -0.5
os.makedirs("/tmp/attnb", exist_ok=True)
NMAX = max(depths)
nbmax = (NMAX + BS - 1) // BS
# one shared KV pool (fp8 bytes), filled once; a context of N keys uses the first ceil(N/16) blocks
k = (torch.randn(nbmax * BS, HK, D, device=dev)).to(torch.float8_e4m3fn)
v = (torch.randn(nbmax * BS, HK, D, device=dev) + 0.3).to(torch.float8_e4m3fn)
kv = torch.empty(nbmax * BS, HK, 2 * D, dtype=torch.uint8, device=dev)
kv[:, :, :D] = k.view(torch.uint8)
kv[:, :, D:] = v.view(torch.uint8)
del k, v
kv = kv.reshape(nbmax, BS, HK, 2 * D).permute(0, 2, 1, 3).contiguous()        # (blocks, heads, slot, 2D)
q = (torch.randn(Q, HQ, D, device=dev) * 2.0).bfloat16()
out = torch.empty(Q, HQ, D, dtype=torch.bfloat16, device=dev)
scratch = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
st = torch.cuda.current_stream().cuda_stream
for N in depths:
    nb = (N + BS - 1) // BS
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, nb)
    sl = torch.tensor([N], dtype=torch.int32, device=dev)
    call = lambda: r4d.attn_prefill_h256_gqa6_fp8kv(q.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(),
                                                    out.data_ptr(), 0, 0, scratch.data_ptr(), 1, Q, HQ, HK, D, BS,
                                                    nb, kv.stride(0), kv.stride(1), scale, 0, N, st)
    rc = call(); torch.cuda.synchronize()
    if rc:
        print(f"{tag} N={N}: entry point returned {rc}", flush=True); continue
    torch.save(out.cpu(), f"/tmp/attnb/{tag}_{N}.pt")
    it = 3 if N <= 65536 else 2
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        call()
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / it
    pairs = Q * (N - Q) + Q * (Q + 1) / 2
    tf = pairs * HQ * D * 4 / (ms * 1e-3) / 1e12
    print(f"{tag} mode={os.environ.get('R4D_ATTN_FP8', '0')} Q={Q} N={N:6d}: {ms:8.2f} ms  {tf:6.1f} TFLOPS  "
          f"finite={bool(torch.isfinite(out.float()).all())}", flush=True)
