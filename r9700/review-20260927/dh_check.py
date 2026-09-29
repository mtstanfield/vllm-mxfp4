"""dh_check.py -- fused exact-set draft head (RADIANCE_DRAFT_FUSED) vs the unfused vocab path, on a synthetic fp8
sub-head the size of the production draft vocab (49152 x 5120) inside a 248320-token vocab: same finite set, same
bf16 logits; plus timing of both (synchronised, eager)."""
import os, sys, time, types
os.environ["RADIANCE_DRAFT_FUSED"] = "1"
sys.path.insert(0, "/dh")
import torch
import radiance_drafthead as dh
dev = "cuda"; torch.manual_seed(0)
N, K, NFULL = 49152, 5120, 248320
w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
sc = (torch.rand(N, 1, device=dev) * 0.02 + 0.01).float()
sub = dh._SubHead(w, sc)
class LP: pass
lp = LP(); lp.head_dtype = None; lp._radiance_topk_only = True
print(dh._quantize_head_now(lp, sub))
lp._dv_nfull = NFULL
lp._dv_ids_dev = torch.sort(torch.randperm(NFULL, device=dev)[:N]).values
for m in (1, 2, 3, 5, 8, 16):
    h = (torch.randn(m, K, device=dev) * 1.5).bfloat16()
    def unfused():
        y_sub = dh._apply_head_int2(lp, sub, h, None)
        y = torch.full((m, NFULL), float("-inf"), dtype=y_sub.dtype, device=dev)
        y[..., lp._dv_ids_dev] = y_sub
        return y
    a = unfused(); b = dh._apply_vocab_fused(lp, sub, h); torch.cuda.synchronize()
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    same_set = bool(torch.equal(fa, fb))
    same_val = same_set and bool(torch.equal(a[fa], b[fb]))
    t = {}
    for name, f in (("unfused", unfused), ("fused", lambda: dh._apply_vocab_fused(lp, sub, h))):
        for _ in range(3): f()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50): f()
        torch.cuda.synchronize(); t[name] = (time.time() - t0) / 50 * 1e6
    print(f"m={m:2d}: finite {int(fa.sum())} / {int(fb.sum())}, same set {same_set}, same values {same_val}, "
          f"unfused {t['unfused']:.0f} us, fused {t['fused']:.0f} us (eager wall incl. launch)", flush=True)
