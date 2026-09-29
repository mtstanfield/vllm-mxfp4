"""gdn_check.py <libr4d dir> [state]: libr4d kkt_solve + chunk_scan vs an fp32 recurrent reference of the gated delta
rule (FLA convention: h <- h*e^{g_t}; v' = beta_t (v_t - h^T k_t); h <- h + k_t v'^T; o_t = scale * h^T q_t), at the
Qwen3.8 GDN geometry (48 v heads, 16 k heads, v head h -> k head h // 3, 128 x 128 state). Heads 0-11 get per-token
log decays in [-6, 0) (chunk spans ~190, past the e^80 clamp); the rest ordinary [-0.5, 0). state: fp32|fp16|bf16."""
import sys, time, torch
sys.path.insert(0, sys.argv[1])
import r4d
state = sys.argv[2] if len(sys.argv) > 2 else "fp32"
torch.manual_seed(0)
dev = "cuda"
T, H, Hg, D, BT = 2048, 48, 16, 128, 64
scale = D ** -0.5
q = torch.nn.functional.normalize(torch.randn(T, Hg, D, device=dev), dim=-1)
k = torch.nn.functional.normalize(torch.randn(T, Hg, D, device=dev), dim=-1)
v = torch.randn(T, H, D, device=dev)
g = -torch.rand(T, H, device=dev) * 0.5
g[:, :12] = -torch.rand(T, 12, device=dev) * 6.0
beta = torch.sigmoid(torch.randn(T, H, device=dev))
h0 = torch.randn(1, H, D, D, device=dev) * 0.5            # [N, H, V, K]
qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()

# reference in fp32 on the bf16-rounded inputs (what the kernel sees)
qr, kr, vr = qb.float(), kb.float(), vb.float()
hmap = torch.arange(H, device=dev) // (H // Hg)
h = h0[0].clone()                                          # [H, V, K]
o_ref = torch.empty(T, H, D, device=dev)
for t in range(T):
    kt, qt, vt = kr[t, hmap], qr[t, hmap], vr[t]           # [H, D]
    h = h * torch.exp(g[t])[:, None, None]
    vp = beta[t][:, None] * (vt - torch.einsum("hvk,hk->hv", h, kt))
    h = h + torch.einsum("hv,hk->hvk", vp, kt)
    o_ref[t] = scale * torch.einsum("hvk,hk->hv", h, qt)
h_ref = h

gcs = torch.empty_like(g)                                  # per-chunk running sum (FLA chunk_local_cumsum)
for c in range(0, T, BT):
    gcs[c:c + BT] = torch.cumsum(g[c:c + BT], dim=0)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
A = torch.empty(T, H, BT, dtype=torch.bfloat16, device=dev)
st = torch.cuda.current_stream().cuda_stream
r4d.gdn_kkt_solve_k128_c64_bf16(kb.data_ptr(), beta.data_ptr(), gcs.data_ptr(), A.data_ptr(), cu.data_ptr(),
                                 1, T, H, Hg, D, BT, st)
sd = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[state]
name = "gdn_chunk_scan_k128_v128_c64_bf16" + {"fp32": "", "fp16": "_f16state", "bf16": "_bf16state"}[state]
fn = getattr(r4d, name)
h0s = h0.to(sd).contiguous()
o = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
ht = torch.empty_like(h0s)
run = lambda: fn(qb.data_ptr(), kb.data_ptr(), vb.data_ptr(), A.data_ptr(), gcs.data_ptr(), beta.data_ptr(),
                 h0s.data_ptr(), o.data_ptr(), ht.data_ptr(), cu.data_ptr(), 1, H, Hg, D, D, BT, scale, st)
run(); torch.cuda.synchronize()


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


of = o.float()
print(f"{sys.argv[1].rstrip('/').split('/')[-1]} state={state}  kernel={name}")
print(f"  o  NMSE  high-span heads 0-11: {nmse(of[:, :12], o_ref[:, :12]):.3e}   others: {nmse(of[:, 12:], o_ref[:, 12:]):.3e}")
print(f"  hT NMSE  high-span heads 0-11: {nmse(ht.float()[0, :12], h_ref[:12]):.3e}   others: {nmse(ht.float()[0, 12:], h_ref[12:]):.3e}")
print(f"  finite: o {bool(torch.isfinite(of).all())} hT {bool(torch.isfinite(ht.float()).all())}")
for _ in range(3):
    run()
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(20):
    run()
e1.record(); torch.cuda.synchronize()
print(f"  chunk_scan {e0.elapsed_time(e1) / 20 * 1000:.1f} us per call (T={T}, H={H})")
