"""qknr_check.py -- vLLM's fused split + QK-RMSNorm + (interleaved-mRoPE) RoPE + gate Triton kernel vs the eager path
it replaces, on ROCm, at the Qwen3.8-27B attention geometry (24 q / 4 kv heads, head_dim 256, rotary 64, mrope
sections 11/11/10, theta 1e7, GemmaRMSNorm (1 + w)). Both paths use vLLM's own modules."""
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import sys; sys.path.insert(0, "/w"); import radiance_qknr; radiance_qknr.install()
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
dev = "cuda"; torch.manual_seed(0)
HQ, HK, D = 24, 4, 256
RP = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
      "rope_theta": 10000000, "rope_type": "default"}
with set_current_vllm_config(VllmConfig()):
    rope = get_rope(head_size=D, max_position=262144, rope_parameters=RP, dtype=torch.bfloat16).to(dev)
    qn, kn = GemmaRMSNorm(D, eps=1e-6).to(dev), GemmaRMSNorm(D, eps=1e-6).to(dev)
    with torch.no_grad():
        qn.weight.copy_(torch.randn(D) * 0.3); kn.weight.copy_(torch.randn(D) * 0.3)
    print(type(rope).__name__, "rotary_dim", rope.rotary_dim, "neox", rope.is_neox_style, "mrope", getattr(rope, "mrope_section", None))
    for T, spread in ((5, 3.0), (1, 1.0), (777, 30.0), (2048, 3.0)):
        qkv = (torch.randn(T, HQ * 2 * D + 2 * HK * D, device=dev) * spread).bfloat16()
        pos1 = torch.randint(0, 200000, (T,), device=dev)
        for pos in (pos1.expand(3, T).contiguous(), torch.stack([pos1, pos1 // 7, pos1 % 97])):
            q_gate, k, v = qkv.split([HQ * 2 * D, HK * D, HK * D], dim=-1)
            # eager (the ROCm path today)
            qg = q_gate.reshape(T, HQ, -1)
            q, gate = torch.chunk(qg, 2, dim=-1)
            q = q.reshape(T, -1); gate = gate.reshape(T, -1)
            qe = qn(q.reshape(-1, HQ, D)).reshape(-1, HQ * D)
            ke = kn(k.reshape(-1, HK, D)).reshape(-1, HK * D)
            qe, ke = rope(pos, qe, ke)
            qe, ke = qe.detach(), ke.detach()
            # fused
            qf, kf, gf = fused_qk_rmsnorm_rope_gate(q_gate, k, qn.weight, kn.weight, rope.cos_sin_cache, pos, 1e-6, HQ, HK,
                                                    D, rope.rotary_dim, mrope_section=rope.mrope_section, norm_beta=1.0)
            torch.cuda.synchronize()
            # fp64 reference with the upstream kernel's own storage contract: normalized values rounded to bf16
            # (the unfused path stores them), rotation exact, one final rounding
            def ref(x, w, H, rnd=True):
                x = x.reshape(T, H, D).double()
                xn = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * (w.double() + 1.0))
                if rnd:
                    xn = xn.bfloat16().double()
                cs = rope.cos_sin_cache.double()
                half = rope.rotary_dim // 2
                idx = torch.arange(half, device=dev)
                sel = torch.where((idx % 3 == 1) & (idx < 33), 1, torch.where((idx % 3 == 2) & (idx < 30), 2, 0))
                p = pos[sel, :].T                                                       # [T, half]
                cos = torch.gather(cs[:, :half], 0, p.clamp(max=cs.shape[0] - 1)) if False else cs[p, idx]
                sin = cs[p, half + idx]
                x1, x2 = xn[..., :half], xn[..., half:2 * half]
                o = xn.clone()
                o[..., :half] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
                o[..., half:2 * half] = x2 * cos[:, None, :] + x1 * sin[:, None, :]
                return o.reshape(T, H * D)
            qr0, kr0 = ref(q, qn.weight.detach(), HQ, False), ref(k, kn.weight.detach(), HK, False)
            for name, r, a, b in (("q", qr0, qe, qf), ("k", kr0, ke, kf)):
                ea = float((a.double() - r).norm() / r.norm()); eb = float((b.double() - r).norm() / r.norm())
                print(f"   vs fp64 (no intermediate rounding) {name}: eager rel {ea:.3e}  fused rel {eb:.3e}", flush=True)
            for name, a, b in (("q", qe, qf), ("k", ke, kf), ("gate", gate, gf)):
                a, b = a.float(), b.float()
                d = (a - b).abs()
                rel = float(d.norm() / a.norm().clamp_min(1e-9))
                eq = float((a == b).float().mean())
                ulp = float((d / a.abs().clamp_min(1e-3) * 128).max())  # ~bf16 ulps (7-bit mantissa)
                print(f"T={T:5d} spread={spread:4.1f} pos={'1d-as-3' if pos[1].equal(pos[0]) else 'thw'} {name:4s}: "
                      f"rel {rel:.2e} max|d| {float(d.max()):.3e} equal {eq*100:5.1f}% max ~{ulp:.1f} ulp", flush=True)
