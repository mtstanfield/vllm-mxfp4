"""radiance_qknr.py -- ROCm build of vLLM's fused split + QK-RMSNorm + (m)RoPE + gate kernel (vllm/model_executor/
layers/fused_qk_norm_rope.py). The upstream kernel selects its base pointers in a runtime `if is_k:` and uses them after
the branch; the AMD Triton backend's TritonAMDGPUCanonicalizePointers pass aborts on pointers merged out of an scf.if
("PassManager::run failed"), which is one reason vLLM gates the fused path on is_cuda(). Here each branch calls the same
@triton.jit body with its own pointers, so no pointer value crosses the branch. The math is the upstream kernel's,
statement for statement."""
import os
import torch
from vllm.triton_utils import tl, triton

# RADIANCE_QKNR_ROUND=1 keeps the upstream contract (normalized q/k rounded to bf16 before RoPE, matching EAGER
# PyTorch); the default 0 rounds once at the store, matching the inductor-compiled path production actually runs
# (p50: the upstream contract cost acceptance, .559 -> .549 on held-out omp windows, and +0.1% PPL).
_ROUND_NORM = tl.constexpr(int(os.environ.get("RADIANCE_QKNR_ROUND", "0")))   # a jit global, not a kernel arg:
# a defaulted constexpr parameter compiles standalone but broke the engine boot under torch.compile (p52)


@triton.jit
def _qknr_head(in_base, w_ptr, out_base, gate_in_base, gate_out_base, positions_ptr, cos_sin_cache_ptr, token,
               positions_stride_m, positions_stride_t, cache_stride_p,
               head_dim: tl.constexpr, rotary_dim: tl.constexpr, half_rotary: tl.constexpr, eps: tl.constexpr,
               norm_beta: tl.constexpr, INPUT_DTYPE: tl.constexpr, HEAD_BLOCK: tl.constexpr,
               ROT_HALF_BLOCK: tl.constexpr, HAS_PASS: tl.constexpr, HAS_MROPE: tl.constexpr,
               MROPE_SECTION_H: tl.constexpr, MROPE_SECTION_W: tl.constexpr, IS_Q: tl.constexpr):
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
    x_norm = x * inv_rms * w
    if _ROUND_NORM:
        x_norm = x_norm.to(INPUT_DTYPE).to(tl.float32)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    w_rot1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    w_rot2 = tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    x_rot1 = x_rot1 * inv_rms * w_rot1
    x_rot2 = x_rot2 * inv_rms * w_rot2
    if _ROUND_NORM:
        x_rot1 = x_rot1.to(INPUT_DTYPE).to(tl.float32)
        x_rot2 = x_rot2.to(INPUT_DTYPE).to(tl.float32)
    pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
    if HAS_MROPE:
        pos_h = tl.load(positions_ptr + positions_stride_m + token * positions_stride_t).to(tl.int64)
        pos_w = tl.load(positions_ptr + 2 * positions_stride_m + token * positions_stride_t).to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    else:
        pos = pos_t
    cache_offset = pos * cache_stride_p
    cos = tl.load(cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)
    if IS_Q:
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel_rocm(
    q_gate_ptr, k_ptr, q_out_ptr, k_out_ptr, gate_out_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions_ptr,
    q_gate_stride_t, k_stride_t, q_out_stride_t, k_out_stride_t, gate_out_stride_t, cache_stride_p,
    positions_stride_m, positions_stride_t,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr, head_dim: tl.constexpr, rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr, eps: tl.constexpr, norm_beta: tl.constexpr, INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr, ROT_HALF_BLOCK: tl.constexpr, HAS_PASS: tl.constexpr, HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr, MROPE_SECTION_W: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    if head >= num_q_heads:
        lh = head - num_q_heads
        _qknr_head(k_ptr + token * k_stride_t + lh * head_dim, k_weight_ptr,
                   k_out_ptr + token * k_out_stride_t + lh * head_dim, k_ptr, k_out_ptr,
                   positions_ptr, cos_sin_cache_ptr, token, positions_stride_m, positions_stride_t, cache_stride_p,
                   head_dim, rotary_dim, half_rotary, eps, norm_beta, INPUT_DTYPE, HEAD_BLOCK, ROT_HALF_BLOCK,
                   HAS_PASS, HAS_MROPE, MROPE_SECTION_H, MROPE_SECTION_W, False)
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + head * 2 * head_dim
        _qknr_head(in_base, q_weight_ptr, q_out_ptr + token * q_out_stride_t + head * head_dim, in_base + head_dim,
                   gate_out_ptr + token * gate_out_stride_t + head * head_dim,
                   positions_ptr, cos_sin_cache_ptr, token, positions_stride_m, positions_stride_t, cache_stride_p,
                   head_dim, rotary_dim, half_rotary, eps, norm_beta, INPUT_DTYPE, HEAD_BLOCK, ROT_HALF_BLOCK,
                   HAS_PASS, HAS_MROPE, MROPE_SECTION_H, MROPE_SECTION_W, True)


def install():
    """Swap the kernel object the upstream wrapper launches (same signature)."""
    import vllm.model_executor.layers.fused_qk_norm_rope as m
    m._fused_qk_rmsnorm_rope_gate_kernel = _fused_qk_rmsnorm_rope_gate_kernel_rocm
