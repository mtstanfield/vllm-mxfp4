"""pod check: the GDN chunk backward (what the rotation optimizer differentiates through) runs and is finite, and its
gradients match a float32 reference on a small problem."""
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
torch.manual_seed(0)
B, T, H, K, V = 1, 512, 4, 128, 128
def inputs(dtype):
    g0 = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, T, H, K, device="cuda", generator=g0)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device="cuda", generator=g0), dim=-1)
    v = torch.randn(B, T, H, V, device="cuda", generator=g0)
    g = -torch.rand(B, T, H, device="cuda", generator=g0) * 0.5
    beta = torch.rand(B, T, H, device="cuda", generator=g0)
    return [t.to(dtype).requires_grad_() if t.dim() == 4 or dtype == torch.float32 else t.to(dtype).requires_grad_() for t in (q, k, v, g, beta)]
res = {}
for dt in (torch.bfloat16, torch.float32):
    q, k, v, g, beta = inputs(dt)
    o, _ = chunk_gated_delta_rule(q, k, v, g.float(), beta, use_qk_l2norm_in_kernel=False)
    o.float().pow(2).mean().backward()
    res[dt] = [t.grad.float() for t in (q, k, v, g, beta)]
    print(dt, "backward OK, finite:", all(torch.isfinite(x).all().item() for x in res[dt]))
for name, a, b in zip("q k v g beta".split(), res[torch.bfloat16], res[torch.float32]):
    print(f"grad {name}: rel diff bf16 vs fp32 {((a - b).norm() / b.norm()).item():.3e}")
