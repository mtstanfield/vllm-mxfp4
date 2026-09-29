"""attn_check.py <libr4d dir> <N keys> <sigma> -- R4D paged PREFILL attention (fp8 KV cache) vs an fp32
reference, at the Qwen3.8 geometry (24 q heads, 4 kv heads, GQA 6, head_dim 256, 16-token blocks). One chunked-prefill step
at depth: the last Q=256 tokens of an N-token context attend causally. Logits are Gaussian with spread `sigma` nats
(q, k ~ N(0,1) per dim, q scaled by sigma; the first tile sets the kernel's reference max, as on real data). K/V are quantized to e4m3 exactly as the cache stores them and the reference uses the SAME dequantized
values, so the difference is the kernel's own arithmetic. Run once per R4D_ATTN_FP8 mode (the library reads it once).
Prints row relRMSE and the output-norm ratio (a numerator losing mass shows up as a ratio < 1)."""
import os, sys, torch
sys.path.insert(0, sys.argv[1])
import r4d
N, sigma = int(sys.argv[2]), float(sys.argv[3])
vmean = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # shared offset in V: lost numerator mass then shrinks o
torch.manual_seed(1)
dev = "cuda"
HQ, HK, D, BS, Q = 24, 4, 256, 16, 256
G = HQ // HK
scale = D ** -0.5

k = torch.randn(N, HK, D, device=dev)
v = torch.randn(N, HK, D, device=dev) + vmean
q = torch.randn(Q, HQ, D, device=dev) * sigma
k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
kd, vd = k8.float(), v8.float()
qb = q.bfloat16()

nb = (N + BS - 1) // BS
kv = torch.zeros(nb * BS, HK, 2 * D, dtype=torch.uint8, device=dev)
kv[:N, :, :D] = k8.view(torch.uint8)
kv[:N, :, D:] = v8.view(torch.uint8)
kv = kv.reshape(nb, BS, HK, 2 * D).permute(0, 2, 1, 3).contiguous()                      # (blocks, heads, slot, 2D)
bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, nb)
sl = torch.tensor([N], dtype=torch.int32, device=dev)
out = torch.empty(Q, HQ, D, dtype=torch.bfloat16, device=dev)
scratch = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
st = torch.cuda.current_stream().cuda_stream
r4d.attn_prefill_h256_gqa6_fp8kv(qb.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(), 0, 0,
                                 scratch.data_ptr(), 1, Q, HQ, HK, D, BS, nb, kv.stride(0), kv.stride(1), scale, 0, N, st)
torch.cuda.synchronize()

# fp32 reference, causal: query t sits at position N - Q + t
ref = torch.empty(Q, HQ, D, device=dev)
qr = qb.float()
qpos = torch.arange(N - Q, N, device=dev)
for h in range(HK):
    for c0 in range(0, Q, 64):
        qq = qr[c0:c0 + 64, h * G:(h + 1) * G].reshape(-1, D)                                    # [64*G, D]
        s = (qq @ kd[:, h].T) * scale                                                              # [64*G, N]
        mask = torch.arange(N, device=dev)[None, :] > qpos[c0:c0 + 64].repeat_interleave(G)[:, None]
        s.masked_fill_(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        ref[c0:c0 + 64, h * G:(h + 1) * G] = (p @ vd[:, h]).reshape(-1, G, D)
of = out.float()
rel = ((of - ref).norm(dim=-1) / ref.norm(dim=-1)).mean()
ratio = (of.norm(dim=-1) / ref.norm(dim=-1)).mean()
print(f"mode={os.environ.get('R4D_ATTN_FP8', '0')} N={N} sigma={sigma} vmean={vmean} "
      f"row relRMSE {float(rel):.3e}  norm ratio {float(ratio):.4f}  finite {bool(torch.isfinite(of).all())}")
