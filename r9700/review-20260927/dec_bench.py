"""dec_bench.py <libr4d dir> [splits,...] -- time the R4D paged DECODE attention (split-KV, fp8 KV) at the Qwen3.8
geometry (24 q / 4 kv heads, head_dim 256, 16-token blocks) the way the serve calls it: max_ctx = 262,144 (the captured
graph's bound), q_len 5 (SPEC-4 verify) and 1 (a draft pass), one sequence. KV copies rotate so every call is DRAM-fed
(> 3x the 64 MB MALL). Reports us/call (decode + combine) and effective KV GB/s; splits=0 is the library's law."""
import sys, torch
sys.path.insert(0, sys.argv[1])
import r4d
dev = "cuda"
HQ, HK, D, BS = 24, 4, 256, 16
scale = D ** -0.5
st = torch.cuda.current_stream().cuda_stream
SPLITS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,16,32,64,128,256").split(",")]
CTXS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "8192,32768,100000,200000").split(",")]
MAXC = 262144
torch.manual_seed(0)


def bench(ctx, q_len, splits, kvs, bts, reps=40):
    maxb = bts[0].shape[1]
    q = (torch.randn(q_len, HQ, D, device=dev) * 2).bfloat16()
    sl = torch.tensor([ctx], dtype=torch.int32, device=dev)
    nbytes = r4d.attn_decode_h256_gqa6_scratch_bytes(1, q_len, HQ, HK, D, MAXC, splits)
    scratch = torch.empty(nbytes + 4096, dtype=torch.uint8, device=dev)
    out = torch.empty(q_len, HQ, D, dtype=torch.bfloat16, device=dev)

    def call(i):
        kv, bt = kvs[i % len(kvs)], bts[i % len(kvs)]
        return r4d.attn_decode_h256_gqa6_fp8kv(q.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(),
                                               out.data_ptr(), 0, 0, scratch.data_ptr(), 1, q_len, HQ, HK, D, BS,
                                               maxb, kv.stride(0), kv.stride(1), scale, splits, MAXC, st)
    for i in range(6):
        call(i)            # the binding returns None; launch errors raise
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(reps):
        call(i)
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / reps
    return us, out.float().clone()


for ctx in CTXS:
    nb = (ctx + BS - 1) // BS
    per = nb * HK * BS * 2 * D
    ncopy = max(1, -(-3 * 64 * 2**20 // per))
    kvs, bts = [], []
    for _ in range(ncopy):
        kv = torch.randint(0, 0x7F, (nb, HK, BS, 2 * D), dtype=torch.uint8, device=dev)   # finite e4m3
        kv &= 0xBF                                                                          # |x| < 2 (exp <= 7)
        kvs.append(kv)
        bts.append(torch.randperm(nb, device=dev, dtype=torch.int32)[None].contiguous())
    gb = ctx * HK * 2 * D / 1e9
    for q_len in (5, 1):
        ref = None
        row = []
        for sp in SPLITS:
            us, o = bench(ctx, q_len, sp, kvs, bts)
            if ref is None:
                ref = o
            d = float((o - ref).abs().max())
            row.append(f"s{sp}: {us:7.1f} us {gb / us * 1e6:6.0f} GB/s d{d:.0e}")
        print(f"ctx {ctx:6d} x{ncopy} q{q_len}  " + " | ".join(row), flush=True)
    del kvs, bts
    torch.cuda.empty_cache()
