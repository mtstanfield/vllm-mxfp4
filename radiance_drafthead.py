"""2-bit MTP draft head with an exact rerank, behind RADIANCE_FAST_DRAFT.

Off by default, in which case the drafter uses the stock bf16 head that vLLM shares with the target
model (_maybe_share_lm_head does this unconditionally for an MTP drafter, so it costs no extra
memory). That head reads 1.18 GiB/rank on every draft slot and measures ~2002 us per call.
RADIANCE_FAST_DRAFT=1 replaces it with a 2-bit head at ~473 us including the rerank, for 0.167
GiB/rank.

The head is the largest bandwidth consumer in a decode step: per rank it is x[M,5120] @
W[5120,124160], it runs once per draft slot (up to 8 times per engine step) and it is flat in M from
1 to 72, so the only lever is fewer bytes.

Weights are int2 with an asymmetric per-(row, group-of-128) scale. Four properties make that pay,
and narrower weights alone is not one of them: at group 64 the same kernel measures 781 us, slower
than a 4-bit head.

  * Group 128, not 64. The per-group scale and zero-point arithmetic on the [BLOCK_M, BLOCK_N]
    accumulator is the dominant non-memory term, so halving the group count halves it: 781 -> 417 us
    on identical bytes.
  * Quarter-split packing. Byte j carries k = j, K/4+j, K/2+j and 3K/4+j, so all four 2-bit planes
    feed contiguous k ranges and one byte load serves four contiguous dots. A contiguous-4 layout
    (byte j holding k = 4j..4j+3, one dot per group) measures 676 us instead, because four tile rows
    then share a byte and gather rather than stream.
  * Bit-pattern dequant, hoisted. 0x3F80 | ((b << (5-2q)) & 0x60) reads as the bf16 value 1 + v/4,
    one shift and one mask, with no int-to-float convert and no extract-then-reposition; the uint16
    conversion is hoisted out of the quarter loop, one per tile rather than four. 417 -> 350 us. The
    1.0 bias is exact rather than an approximation: the dot returns sum_k x_k + dot(x,v)/4, and the
    kernel already holds sum_k x_k per group for the zero point, so the contribution collapses to
    (4 s)*p - (4 s + z s)*sum_k x_k, the same two accumulator ops against premultiplied scales.
  * The scale is applied to the accumulator, never to the weight tile. Dequantising [G, BLOCK_N]
    elementwise costs ~400 us instead, 8x more elements to touch.

Accuracy comes from reranking rather than from bits. Each program already holds the maximum of its
64 columns, so it emits the top KCAND of them for free; the top RERANK of those are scored exactly
against the bf16 weight (a few hundred KB against the coarse pass's 0.167 GiB) and written back over
the coarse values. On 8192 draft-head inputs captured from a live serve this matches the exact bf16
argmax on every row, against 22 misses for a 4-bit head at KCAND=1.

KCAND is the lever, not RERANK. Selection emits the top K of each block, so at K=1 a winner that
shares a block with a stronger token is never a candidate at any R, and recall saturates. 2 bits
needs K=8; 4 bits is adequate at K=1. Selecting by block max rather than a token-level topk over the
full row is also the faster choice, 65 us against 105.

Model output cannot move as a result: the draft head decides which tokens are proposed, the target
model verifies every proposal with its own untouched bf16 head on a separate LogitsProcessor
instance, and speculative decoding is distribution-preserving, so a worse draft costs acceptance
rather than a different token. mtp.fc is deliberately left alone: the checkpoint lists it in
modules_to_not_convert alongside the norms, gates, lm_head and embed_tokens, and it is worth only
~0.5% of a decode step.
"""
import os
import sys
import types

import torch

try:
    import triton
    import triton.language as tl
except Exception:                       # pragma: no cover - triton always present in the image
    triton = None

# RADIANCE_FAST_DRAFT gates draft-head quantisation entirely.
#   0 (default): nothing here installs. The drafter uses the stock bf16 head -- which vLLM shares
#                with the target model, so it costs no extra memory but reads 1.18 GiB/rank per draft
#                slot and measures ~2002 us per call.
#   1:           2-bit head, 0.167 GiB/rank, ~473 us per call including the rerank. The rerank makes
#                it exact: on 8192 real draft-head inputs it matches the bf16 argmax on every row.
FAST = os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"

GROUP = 128        # weight-quantisation group along K; also the kernel's BLOCK_K
BLOCK_N = 64       # do_bench optimum, and the width of one block-max entry
BITS = 2
# Candidates scored exactly per row. For an ARGMAX caller (mtp) this only caps how many of the
# coarse pass's candidates get rescored, and KCAND is the recall lever -- see the docstring. For a
# TOP-K caller (DFlash2, which asks for selector_top_k=16 candidates per position) it is a HARD
# CEILING instead: _radiance_topk_only blanks every entry the rerank did not touch, so a top-16
# request draws from exactly R tokens. Measured on Qwen3.8-27B + DFlash2-FP8 at ctx 0, R=32:
# acc/draft 1.904 -> 1.804 against the bf16 head, i.e. 2R is too tight a pool for K=16.
RERANK = int(os.environ.get("RADIANCE_DRAFT_RERANK", "32"))
KCAND = 8          # candidates emitted per block; R caps the final count, K feeds it
# Launch geometry, per M band. The head changes regime across the batch sizes one serve produces:
# at M=16 it is memory-bound (427 GB/s, 68% of the DRAM roofline on its 152 MiB of int2) and at
# M=64 it is compute-bound (67 TF/s against a 207 TF/s bf16 WMMA ceiling, only 130 GB/s). The warp
# count that suits one end is wrong at the other, and the penalty is not symmetric: warps=2 is the
# optimum at M=16 (1.10x) and 0.38x at M=64, while warps=8 is the optimum at M=64 (1.07x) and 0.78x
# at M=32. num_stages>1 regresses everywhere. BLOCK_N=64 wins or ties at every M measured.
# Isolated, 124160 x 5120 (one rank's vocab shard), CUDA-graph window, us:
#     M=16   warps 2 / 4 / 8 = 372.4 / 408.8 / 425.6
#     M=32   warps 2 / 4 / 8 = 660.4 / 540.0 / 690.3
#     M=64   warps 2 / 4 / 8 = 3428.4 / 1309.4 / 1218.7
# M is padded to a power of two >= 16 before it gets here, so the bands are 16, 32, 64.
_CFG_BY_M = ((16, {"num_warps": 2, "num_stages": 1}),
             (48, {"num_warps": 4, "num_stages": 1}),
             (64, {"num_warps": 8, "num_stages": 1}))
_CFG = {"num_warps": 4, "num_stages": 1}    # fallback for an M past the table


def _cfg_for(m):
    for lim, cfg in _CFG_BY_M:
        if m <= lim:
            return cfg
    return _CFG


if triton is not None:

    @triton.jit
    def _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC: tl.constexpr, BLOCK_N: tl.constexpr):
        """Top-KC of this block, by successive max-and-mask.

        The exact winner is always the maximum of its own block, so block maxima carry the rerank
        candidates at 1/BLOCK_N the selection width of a token-level top-R; a token-level topk over
        the full row measures 105 us against 65 for this, so the cheap selection is also the fast
        one. KC matters where R does not: R caps how many candidates are finally rescored, but at
        KC=1 a winner sharing a block with a stronger token is never a candidate at any R.
        """
        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        for c in tl.static_range(KC):
            mx = tl.max(masked, axis=1)
            am = tl.argmax(masked, axis=1)
            tl.store(BM + offs_m * (NBLK * KC) + (pid * KC + c), mx)
            tl.store(BI + offs_m * (NBLK * KC) + (pid * KC + c),
                     (pid * BLOCK_N + am).to(tl.int32))
            masked = tl.where(tl.arange(0, BLOCK_N)[None, :] == am[:, None], float("-inf"), masked)

    @triton.jit
    def _draft_head_int2(X, XS, Wq, S, ZS, Y, BM, BI, K: tl.constexpr, N, stride_wq, stride_s,
                         stride_xs, NBLK, KC: tl.constexpr, G: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """2 bits/weight. Packing splits K into quarters, byte j carrying k = j, K/4+j, K/2+j and
        3K/4+j, so all four planes feed contiguous k ranges and one byte load serves four contiguous
        dots. Group 128 rather than 64 is what makes it pay: the per-group accumulator work is the
        dominant non-memory term, and halving the group count moves this kernel 781 -> 417 us."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            # hoisted out of the quarter loop: it is the same tile each time, so one convert not four
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                # 0x3F80 | ((b << (5-2q)) & 0x60) is exactly 0x3F80 | (((b >> 2q) & 3) << 5): one
                # shift and one mask instead of extract-then-reposition. Reads as bf16 1 + v/4, and
                # the 1.0 bias divides out through the group's sum of x, as in the 4-bit path.
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :]).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.load(XS + offs_m * stride_xs + gi).to(tl.float32)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mask_n[None, :])
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)

    @triton.jit
    def _rerank_exact(X, W, S, IDX, OUT, K: tl.constexpr, stride_w, R: tl.constexpr,
                      BLOCK_K: tl.constexpr, FP8: tl.constexpr):
        """Exact logit for R candidate rows per draft row, straight off the head's own weight:
        bf16 rows, or (FP8) e4m3 rows as raw bytes decoded here times the per-row fp32 scale S --
        the compressed-tensors per-channel FP8 lm_head of the NVFP4 checkpoints. The decode is
        exact bit arithmetic (no Triton fp8 cast, whose gfx12 lowering is not relied on)."""
        m = tl.program_id(0)
        j = tl.program_id(1)
        n = tl.load(IDX + m * R + j)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if FP8:
            sc = tl.load(S + n).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            if FP8:
                b = tl.load(W + n * stride_w + offs).to(tl.int32)
                e = (b >> 3) & 15
                mant = (b & 7).to(tl.float32)
                mag = tl.where(e == 0, mant * 0.001953125,
                               (1.0 + mant * 0.125) * tl.exp2((e - 7).to(tl.float32)))
                wv = tl.where((b & 128) != 0, -mag, mag) * sc
            else:
                wv = tl.load(W + n * stride_w + offs).to(tl.float32)
            acc += tl.load(X + m * K + offs).to(tl.float32) * wv
        tl.store(OUT + m * R + j, tl.sum(acc, axis=0))

    # ---- local (2026-09-28 evening): fused draft head for the exact-set vocab path (RADIANCE_DRAFT_FUSED=1) ------------
    # Same math as _draft_head_int2, minus the three launches around it: rows past the real m are masked instead of
    # zero-padded (fill + cat), the per-group sums of x come from the tile the kernel already loads instead of a separate
    # fp32 cast + reduce, and the coarse scores are not written at all (the exact set discards them).
    @triton.jit
    def _draft_head_int2_cand(X, Wq, S, ZS, BM, BI, m_rows, K: tl.constexpr, N, stride_wq, stride_s, NBLK,
                              KC: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        mask_m = offs_m < m_rows
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :],
                             mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.sum(xv.to(tl.float32), axis=1)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)

    @triton.jit
    def _rerank_scatter(X, W, S, IDX, IDS, OUT, K: tl.constexpr, stride_w, NFULL, R: tl.constexpr,
                        BLOCK_K: tl.constexpr, FP8: tl.constexpr):
        """_rerank_exact writing its bf16 logit straight into the full-vocab row at IDS[candidate] (the rows around it
        are -inf from one fill): replaces the -inf fill of the sub row, the cast, the scatter and the index_put."""
        m = tl.program_id(0)
        j = tl.program_id(1)
        n = tl.load(IDX + m * R + j)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if FP8:
            sc = tl.load(S + n).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            if FP8:
                b = tl.load(W + n * stride_w + offs).to(tl.int32)
                e = (b >> 3) & 15
                mant = (b & 7).to(tl.float32)
                mag = tl.where(e == 0, mant * 0.001953125,
                               (1.0 + mant * 0.125) * tl.exp2((e - 7).to(tl.float32)))
                wv = tl.where((b & 128) != 0, -mag, mag) * sc
            else:
                wv = tl.load(W + n * stride_w + offs).to(tl.float32)
            acc += tl.load(X + m * K + offs).to(tl.float32) * wv
        tl.store(OUT + m.to(tl.int64) * NFULL + tl.load(IDS + n), tl.sum(acc, axis=0).to(tl.bfloat16))


def _head_matrix(lm_head):
    """(rows [N, K], per-row scale [N] fp32 or None) for the head's weight. A compressed-tensors
    FP8 per-channel lm_head keeps its [N, K] e4m3 storage but exposes it TRANSPOSED ([K, N] view)
    after process_weights_after_loading, with weight_scale [N, 1]; undo the view here."""
    w = getattr(lm_head, "weight", None)
    if w is None or w.dim() != 2:
        return w, None
    if w.dtype == torch.float8_e4m3fn:
        sc = getattr(lm_head, "weight_scale", None)
        if sc is None:
            return None, None
        rows = w if w.shape[0] == sc.numel() else w.t()
        if not rows.is_contiguous() or rows.shape[0] != sc.numel():
            return None, None
        return rows, sc.detach().reshape(-1).float()
    return w, None


def _head_is_empty(rows, scale):
    if scale is None:
        return float(rows.data.abs().max()) == 0.0
    for i in range(0, rows.shape[0], 8192):          # fp8 has no abs(); bytes are enough
        if bool((rows.data[i:i + 8192].view(torch.uint8) != 0).any()):
            return False
    return True


def _pow2_at_least(m):
    """tl.arange needs a power-of-two extent, and tl.dot needs at least 16 rows."""
    n = 16
    while n < m:
        n *= 2
    return n


def _apply_head_int2(self, lm_head, hidden_states, embedding_bias):
    """Drop-in for LogitsProcessor._apply_head against the quantised draft head."""
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    m, k = x.shape
    M = _pow2_at_least(m)
    if M != m:
        x = torch.cat([x, x.new_zeros(M - m, k)])
    x = x.contiguous()

    n = self._radiance_n
    nblk = self._radiance_nblk
    ng = k // GROUP
    xs = x.reshape(M, ng, GROUP).float().sum(-1).contiguous()
    y = torch.empty(M, n, dtype=torch.bfloat16, device=x.device)
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    _draft_head_int2[(nblk,)](
        x, xs, self._radiance_wq, self._radiance_scale, self._radiance_zs, y, bm, bi,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))

    w, wsc = _head_matrix(lm_head)
    if w is not None and w.shape == (n, k) and \
            (wsc is not None or w.dtype in (torch.bfloat16, torch.float16)):
        idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
        ex = torch.empty(M, RERANK, dtype=torch.float32, device=x.device)
        if wsc is not None:
            _rerank_exact[(M, RERANK)](x, w.view(torch.uint8), wsc, idx, ex, k, w.stride(0),
                                       R=RERANK, BLOCK_K=512, FP8=True, num_warps=4)
        else:
            _rerank_exact[(M, RERANK)](x, w, x, idx, ex, k, w.stride(0), R=RERANK, BLOCK_K=512,
                                       FP8=False, num_warps=4)
        if getattr(self, "_radiance_topk_only", False):
            # A top-k caller ranks the whole row against itself, so the ~124k entries the rerank
            # did NOT touch are still coarse 2-bit values competing with exact ones -- and a
            # spuriously high coarse entry becomes a candidate. An argmax caller never noticed
            # (the true winner is a reranked block maximum); a top-16 caller sees garbage in
            # every slot the coarse pass over-scored. Make the reranked set the only eligible
            # one, so recall is bounded by RERANK rather than by 2-bit noise.
            y.fill_(float("-inf"))
        y.scatter_(1, idx.long(), ex.to(torch.bfloat16))
    elif not self._radiance_warned:
        # Without the exact weight the coarse pass stands on its own; that is a ~5% top-1 change
        # against bf16, so say so rather than let it pass as the reranked path.
        self._radiance_warned = True
        sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: no bf16 lm_head to rerank against "
                         f"({None if w is None else (tuple(w.shape), w.dtype)}); "
                         "running the coarse 2-bit head unreranked\n")
        sys.stderr.flush()

    if M != m:
        y = y[:m]
    if embedding_bias is not None:
        y = y + embedding_bias
    if self.head_dtype is not None and self.head_dtype != y.dtype:
        y = y.to(self.head_dtype)
    return y.reshape(*hidden_states.shape[:-1], -1)


def _apply_head_lazy(self, lm_head, hidden_states, embedding_bias):
    """First real call quantises, then hands over to the quantised path for good.

    Needed because the weight a drafter scores against may not exist yet when load_weights
    returns; here it is the argument, so it is guaranteed populated.
    """
    _rows, _sc = _head_matrix(lm_head)
    if _rows is None or _head_is_empty(_rows, _sc):
        # Still empty on a real call: something is wrong, but a coarse head would be silently
        # catastrophic, so fall back to the stock GEMM rather than guess.
        return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
    status = _quantize_head_now(self, lm_head)
    sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD (lazy): {status}\n")
    sys.stderr.flush()
    return self._apply_head(lm_head, hidden_states, embedding_bias)


def _quantize_draft_head(mtp, lp_attr="logits_processor"):
    """Quantise the head a drafter scores with, and rebind that LogitsProcessor's _apply_head.

    lp_attr names which LogitsProcessor to hook. MTP has exactly one; DFlash2 keeps a separate
    `candidate_logits_processor` for candidate generation, and hooking THAT is what confines the
    approximation to the draft path -- the target samples through its own instance and is
    untouched. Both share the same lm_head weight, which is why the bf16 copy has to stay: it is
    the target's, and it is also what the rerank scores against.
    """
    lm_head = getattr(mtp, "lm_head", None)
    lp = getattr(mtp, lp_attr, None)
    if lm_head is None or lp is None or not hasattr(lm_head, "weight"):
        return f"no lm_head/{lp_attr}"
    w = lm_head.weight
    rows, rsc = _head_matrix(lm_head)
    if rows is None or w.dim() != 2 or \
            (rsc is None and w.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return f"unsupported draft-head weight {tuple(w.shape)} {w.dtype}"
    lp._radiance_topk_only = lp_attr == "candidate_logits_processor"
    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target's tensor shared in
    # AFTER load_weights returns, so at this point the parameter is still allocated-but-empty.
    # Quantising that yields an all-zero head, and the failure is silent and total: the serve comes
    # up, text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --
    # which reads as a plausible accuracy verdict on the quantisation. Defer instead.
    if _head_is_empty(rows, rsc):
        lp._apply_head = types.MethodType(_apply_head_lazy, lp)
        return "lm_head empty at load_weights (shared in later); quantising on first use"
    return _quantize_head_now(lp, lm_head)


# int2 buffers keyed by the bf16 weight they were derived from. DFlash2 shares ONE lm_head between
# the drafter's candidate_logits_processor and the target's logits_processor, so when both are armed
# the second one must reuse the first's packing rather than spend another 0.167 GiB/rank -- at
# GPU_UTIL 0.98 that second copy comes straight out of the KV cache. Keyed by data_ptr because the
# two LogitsProcessors hold the same tensor object anyway; a weight that moved would get a new
# pointer and be repacked, which is the safe direction.
_HEAD_CACHE: dict = {}


def _quantize_head_now(lp, lm_head):
    w, wsc = _head_matrix(lm_head)          # [N, K] rows (bf16, or e4m3 + per-row scale)
    n, k = w.shape
    if k % (2 * GROUP):
        return f"hidden size {k} not a multiple of {2 * GROUP}"

    cached = _HEAD_CACHE.get((w.data_ptr(), n, k))
    if cached is not None:
        (lp._radiance_wq, lp._radiance_scale, lp._radiance_zs,
         lp._radiance_n, lp._radiance_nblk) = cached
        lp._radiance_warned = False
        lp._apply_head = types.MethodType(_apply_head_int2, lp)
        return (f"draft head ({n}, {k}) reusing the int{BITS} packing already built for this "
                f"lm_head, {KCAND} cand/block, rerank top-{RERANK} exact")

    # Asymmetric min/max at BITS, quarter-split packing (see the module docstring).
    # Quantise in row chunks: a whole-tensor fp32 intermediate is 2.5 GiB here, and the caching
    # allocator keeps that reservation for the rest of the process, which comes straight out of
    # the VRAM the KV cache could have used.
    per_byte = 8 // BITS
    lv = (1 << BITS) - 1
    packed = torch.empty(n, k // per_byte, dtype=torch.uint8, device=w.device)
    scale = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    zs = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    CH = 8192
    for i in range(0, n, CH):
        j = min(i + CH, n)
        wg = w.data[i:j].to(torch.float32)
        if wsc is not None:
            wg = wg * wsc[i:j, None]
        wg = wg.reshape(j - i, k // GROUP, GROUP)
        lo, hi = wg.amin(dim=2), wg.amax(dim=2)
        sc = ((hi - lo) / lv).clamp(min=1e-8)
        zp = torch.round(-lo / sc).clamp(0, lv)
        q = torch.round(wg / sc[:, :, None] + zp[:, :, None]).clamp(0, lv).to(torch.uint8)
        q = q.reshape(j - i, k)
        if BITS == 2:
            # quarter-split: byte j carries k = j, K/4+j, K/2+j, 3K/4+j
            Q = k // 4
            packed[i:j] = (q[:, :Q] | (q[:, Q:2 * Q] << 2)
                           | (q[:, 2 * Q:3 * Q] << 4) | (q[:, 3 * Q:] << 6))
        else:
            packed[i:j] = q[:, : k // 2] | (q[:, k // 2:] << 4)
        # The kernel's dot returns sum_k x_k + dot(x,v)/16 because every weight carries the bf16
        # 1.0 bias, so s*dot(x,q) - zp*s*sum_k x_k becomes (16 s)*p - (16 s + zp s)*sum_k x_k.
        # Both premultiplied here, which also saves a load in the inner loop.
        # the bf16 bias is 1 + v/(2^m) with m the mantissa slot used, so the premultiplier is
        # 2^m: 16 for the 4-bit path (v << 3), 4 for the 2-bit one (v << 5)
        bias = 4.0 if BITS == 2 else 16.0
        scale[i:j] = (bias * sc).to(torch.bfloat16)
        zs[i:j] = (bias * sc + zp * sc).to(torch.bfloat16)
        del wg, lo, hi, sc, zp, q
    lp._radiance_wq = packed
    lp._radiance_scale = scale
    lp._radiance_zs = zs
    lp._radiance_n = n
    lp._radiance_nblk = (n + BLOCK_N - 1) // BLOCK_N
    lp._radiance_warned = False
    lp._apply_head = types.MethodType(_apply_head_int2, lp)
    _HEAD_CACHE[(w.data_ptr(), n, k)] = (packed, scale, zs, n, lp._radiance_nblk)
    torch.cuda.empty_cache()

    stored = (lp._radiance_wq.numel()
              + (lp._radiance_scale.numel() + lp._radiance_zs.numel()) * 2)
    # The bf16 weight is deliberately left in place: the rerank scores against it, and leaving it
    # lets vLLM's _maybe_share_lm_head point the MTP at the target's copy instead of keeping a
    # second one. Blanking it here (as the fp8 head did) defeats that share.
    return (f"draft head ({n}, {k}) {'fp8' if wsc is not None else 'bf16'} -> int{BITS} g{GROUP} asym "
            f"({stored / 2**30:.2f} GiB/rank), {KCAND} cand/block, rerank top-{RERANK} exact")


# ---- local (2026-09-27): pruned DRAFT vocabulary --------------------------------------------------------------------
# RADIANCE_DRAFT_VOCAB=<file, one token id per line>: the MTP drafter scores only those rows -- the int2 coarse pass and
# exact rerank above, run on the sub-matrix (a 48k list reads ~1/5 of the full int2 head) -- and every other entry of the
# vocabulary row is -inf. Output cannot move: the target verifies with its own head; only acceptance can.
VOCAB_FILE = os.environ.get("RADIANCE_DRAFT_VOCAB", "")


class _SubHead:
    """The rows the draft may propose, in the shape _head_matrix reads: [n_sub, K] rows (+ [n_sub, 1] scale)."""

    def __init__(self, w, wsc):
        self.weight = w
        if wsc is not None:
            self.weight_scale = wsc


def _apply_head_vocab(self, lm_head, hidden_states, embedding_bias):
    sub = self._dv_sub
    if sub is None:
        rows, rsc = _head_matrix(lm_head)
        if rows is None or _head_is_empty(rows, rsc):
            return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
        ids = self._dv_ids.to(rows.device)
        sub = _SubHead(rows.index_select(0, ids).contiguous(),
                       rsc.index_select(0, ids).reshape(-1, 1).contiguous() if rsc is not None else None)
        status = _quantize_head_now(self, sub)          # rebinds _apply_head to the int2 path; take it back
        self._apply_head = types.MethodType(_apply_head_vocab, self)
        self._dv_sub, self._dv_ids_dev, self._dv_nfull = sub, ids, rows.shape[0]
        sys.stderr.write(f"[radiance] DRAFT_VOCAB: {ids.numel()} of {rows.shape[0]} rows -> {status}\n")
        sys.stderr.flush()
    if FUSED and getattr(self, "_radiance_topk_only", False) and embedding_bias is None:
        return _apply_vocab_fused(self, sub, hidden_states)
    y_sub = _apply_head_int2(self, sub, hidden_states, embedding_bias)
    y = torch.full((*y_sub.shape[:-1], self._dv_nfull), float("-inf"), dtype=y_sub.dtype, device=y_sub.device)
    y[..., self._dv_ids_dev] = y_sub
    return y


# RADIANCE_DRAFT_FUSED=1 (local 2026-09-28 evening): the exact-set vocab path in 6 launches per draft call instead of 15
# (int2 candidates, topk, sort, gather, one -inf fill, rerank-scatter). Output identical to the unfused path's: -inf
# everywhere except the RERANK exactly scored candidates (bf16); the candidate SET can differ only where two coarse
# scores tie to within the fp32 order of the in-kernel group sums. Drafts only.
FUSED = os.environ.get("RADIANCE_DRAFT_FUSED", "0") == "1"


def _apply_vocab_fused(self, sub, hidden_states):
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    m, k = x.shape
    M = _pow2_at_least(m)
    n, nblk = self._radiance_n, self._radiance_nblk
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    _draft_head_int2_cand[(nblk,)](
        x, self._radiance_wq, self._radiance_scale, self._radiance_zs, bm, bi, m,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), nblk, KCAND,
        G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))
    idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
    w, wsc = _head_matrix(sub)
    nfull = self._dv_nfull
    out = torch.full((m, nfull), float("-inf"), dtype=torch.bfloat16, device=x.device)
    if wsc is not None:
        _rerank_scatter[(m, RERANK)](x, w.view(torch.uint8), wsc, idx, self._dv_ids_dev, out, k, w.stride(0), nfull,
                                     R=RERANK, BLOCK_K=512, FP8=True, num_warps=4)
    else:
        _rerank_scatter[(m, RERANK)](x, w, x, idx, self._dv_ids_dev, out, k, w.stride(0), nfull, R=RERANK,
                                     BLOCK_K=512, FP8=False, num_warps=4)
    if self.head_dtype is not None and self.head_dtype != out.dtype:
        out = out.to(self.head_dtype)
    return out.reshape(*hidden_states.shape[:-1], -1)


def _install_vocab():
    ids = sorted({int(t) for t in open(VOCAB_FILE).read().split()})
    ids_t = torch.tensor(ids, dtype=torch.long)
    for mod_name, cls_name in (("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP"),
                               ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP")):
        try:
            cls = getattr(__import__(mod_name, fromlist=[cls_name]), cls_name)
        except Exception:
            continue
        if getattr(cls, "_radiance_vocab_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig):
            loaded = _orig(self, weights)
            lp = getattr(self, "logits_processor", None)
            if lp is None:
                sys.stderr.write("[radiance] DRAFT_VOCAB: drafter has no logits_processor, full head kept\n")
            else:
                lp._dv_ids, lp._dv_sub = ids_t, None
                lp._apply_head = types.MethodType(_apply_head_vocab, lp)   # sub-head built on first real call
                if EXACT_SET:
                    lp._radiance_topk_only = True
            return loaded

        cls.load_weights = wrapped
        cls._radiance_vocab_wrapped = True
    sys.stderr.write(f"[radiance] draft vocab armed: {len(ids)} ids from {VOCAB_FILE}"
                     f"{', exact-reranked set only' if EXACT_SET else ''}\n")
    sys.stderr.flush()


# ---- local (2026-09-27): draft distribution for SAMPLED requests ---------------------------------------------------
# vLLM's probabilistic drafting (draft_sample_method=probabilistic) applies only the temperature to the draft
# ("we ignore most of the sampling parameters"), while the target keeps just the request's top-k/top-p set -- so every
# unit of draft mass outside that set is a guaranteed rejection.
#   RADIANCE_DRAFT_TOPKP=1     draft rows take temperature -> top-k -> top-p (the target sampler's order) before the
#                              softmax, through the same apply_top_k_top_p the target uses.
#   RADIANCE_DRAFT_EXACTSET=1  only the exactly reranked candidates are eligible (the rest of the int2 row is coarse 2-bit
#                              scores; argmax never sees them, a sampled draft does).
# Lossless either way: the drafted token is sampled from exactly the probs tensor handed to the rejection sampler, and
# standard rejection sampling returns the target distribution for ANY such proposal. Only acceptance can move.
DRAFT_TOPKP = os.environ.get("RADIANCE_DRAFT_TOPKP", "0") == "1"
EXACT_SET = os.environ.get("RADIANCE_DRAFT_EXACTSET", "0") == "1"


def _install_draft_topkp():
    import vllm.v1.spec_decode.llm_base_proposer as lbp
    from vllm.v1.sample.ops.topk_topp_sampler import (apply_top_k_top_p, empty_exponential_noise_like,
                                                      sample_with_exponential_noise)
    from vllm.v1.sample.sampler import _SAMPLING_EPS
    if getattr(lbp, "_radiance_topkp", False):
        return
    orig = lbp.compute_probs_and_sample_next_token

    def compute_probs_and_sample_next_token(logits, sampling_metadata, use_fp64_gumbel=False):
        md = sampling_metadata
        if md.all_greedy or (md.top_k is None and md.top_p is None):
            return orig(logits, md, use_fp64_gumbel)
        rows = logits.shape[0]

        def per_row(t):      # parallel drafting has K rows per request (the caller did this for temperature only)
            return t if t is None or t.shape[0] == rows else t.repeat_interleave(rows // t.shape[0], dim=0)

        temperature = per_row(md.temperature)
        if not md.all_random:
            is_greedy = temperature < _SAMPLING_EPS
            temperature = torch.where(is_greedy, 1.0, temperature)
        logits.div_(temperature.view(-1, 1))
        logits = apply_top_k_top_p(logits, per_row(md.top_k), per_row(md.top_p),
                                   max_top_k=getattr(md, "max_top_k", 0))
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        noise = empty_exponential_noise_like(probs, use_fp64_gumbel)
        noise.exponential_()
        next_token_ids = sample_with_exponential_noise(probs.clone(), noise)
        if not md.all_random:
            next_token_ids = torch.where(is_greedy, probs.argmax(dim=-1), next_token_ids)
        return next_token_ids, probs

    lbp.compute_probs_and_sample_next_token = compute_probs_and_sample_next_token
    lbp._radiance_topkp = True
    sys.stderr.write("[radiance] draft top-k/top-p armed: sampled drafts use the request's truncation\n")
    sys.stderr.flush()


_TOPKP_KCAP = 64


def _draft_mask(logits, k, p, temp):
    """Capturable top-k/top-p over a top-64 window (fixed shapes, no host sync). Rows whose top-k is not a real cap
    <= 64, and greedy rows, pass through untouched -- any proposal is lossless, the mask only has to be sampled from."""
    vals, idx = torch.topk(logits, _TOPKP_KCAP, dim=-1)
    kk = k.to(torch.long).clamp(1, _TOPKP_KCAP)
    thr = vals.gather(1, kk.unsqueeze(1) - 1)
    kept = vals >= thr
    t = torch.where(temp > 0, temp, torch.ones_like(temp)).unsqueeze(1)
    w = torch.where(kept, vals.float() / t, torch.full_like(t, float("-inf")))
    probs = w.softmax(dim=-1)
    maskp = (probs.cumsum(dim=-1) - probs) >= p.unsqueeze(1)     # same rule as the target's composite top-p
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(1, idx, torch.where(kept & ~maskp, vals, torch.full_like(vals, float("-inf"))))
    ok = ((k <= _TOPKP_KCAP) & (temp > 0)).unsqueeze(1)
    return torch.where(ok, out, logits)


def _install_draft_topkp_v2():
    """The V2 model runner (vLLM 0.29 default) drafts in v1/worker/gpu/spec_decode: gumbel_sample on the draft logits
    at the request temperature, the PRE-temperature logits cached for the rejection sampler's q. Mask those logits
    before both, so the draw and the cached q are the same truncated distribution. Runs inside the captured draft
    graphs, so top-k/top-p are copied into speculator-owned buffers outside them (as the stock code does for
    temperature) -- the sampler's UVA arrays are re-pointed on every update and cannot be captured."""
    from vllm.v1.worker.gpu.spec_decode import speculator as spm
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu import model_runner as mrm
    S, R = spm.DraftModelSpeculator, mrm.GPUModelRunner
    if getattr(S, "_radiance_topkp", False):
        return
    orig_load, orig_copy, orig_sample = R.load_model, S._copy_request_inputs, S.sample_draft

    def load_model(self, *args, **kwargs):
        out = orig_load(self, *args, **kwargs)
        spec, smp = getattr(self, "speculator", None), getattr(self, "sampler", None)
        if isinstance(spec, S) and smp is not None and getattr(spec, "draft_logits", None) is not None:
            st_ = smp.sampling_states
            spec._rad_states = st_
            spec._rad_topk = torch.full((spec.max_num_reqs,), st_.vocab_size, dtype=torch.int32, device=spec.temperature.device)
            spec._rad_topp = torch.ones(spec.max_num_reqs, dtype=torch.float32, device=spec.temperature.device)
            sys.stderr.write("[radiance] draft top-k/top-p live on the V2 speculator\n")
            sys.stderr.flush()
        return out

    def _copy_request_inputs(self, num_reqs, idx_mapping, temperature, seeds):
        orig_copy(self, num_reqs, idx_mapping, temperature, seeds)
        st_ = getattr(self, "_rad_states", None)
        if st_ is not None:
            self._rad_topk.copy_(st_.top_k.gpu[:self.max_num_reqs])
            self._rad_topp.copy_(st_.top_p.gpu[:self.max_num_reqs])

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step,
                     draft_logits):
        if draft_logits is None or getattr(self, "_rad_states", None) is None:
            return orig_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                               draft_step, draft_logits)
        logits = self.model.compute_logits(hidden_states)
        rows = idx_mapping.clamp(min=0).long()
        logits = _draft_mask(logits, self._rad_topk[rows], self._rad_topp[rows], temperature[rows].float())
        return gumbel_sample(logits, idx_mapping, temperature, seeds, sample_src_positions, apply_temperature=True,
                             is_drafting=True, logits_cache=draft_logits, logits_cache_col=draft_step,
                             use_fp64=self.use_fp64_gumbel)

    R.load_model = load_model
    S._copy_request_inputs = _copy_request_inputs
    S.sample_draft = sample_draft
    S._radiance_topkp = True
    sys.stderr.write("[radiance] draft top-k/top-p armed (V2 runner)\n")
    sys.stderr.flush()


def _install_tunable_diag():
    """RADIANCE_TUNABLE_DIAG=1 (local, diagnostic): after the V2 runner loads the model, print TunableOp's live state in
    the ENGINE process and time the MTP down-projection shape (17408x5120 fp8 rowwise, M=1) there."""
    from vllm.v1.worker.gpu import model_runner as mrm
    R = mrm.GPUModelRunner
    orig = R.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            t = torch.cuda.tunable
            msg = (f"enabled={t.is_enabled()} tuning={t.tuning_is_enabled()} file={t.get_filename()} "
                   f"results={len(t.get_results())}")
            w = (torch.randn(5120, 17408, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            x = torch.randn(1, 17408, device="cuda").to(torch.float8_e4m3fn)
            sa, sb = torch.ones(1, 1, device="cuda"), torch.ones(1, 5120, device="cuda")
            f = lambda: torch._scaled_mm(x, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            for _ in range(5):
                f()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                f()
            e1.record()
            torch.cuda.synchronize()
            msg += f" | down 17408x5120 M=1: {e0.elapsed_time(e1) * 1000 / 20:.0f} us (tuned ~170, default ~310)"
            del w
        except Exception as e:
            msg = f"diag failed: {e!r}"
        sys.stderr.write(f"[radiance.tunablediag] {msg}\n")
        sys.stderr.flush()
        return out

    R.load_model = load_model


def install():
    if os.environ.get("RADIANCE_TUNABLE_DIAG", "0") == "1":
        try:
            _install_tunable_diag()
        except Exception as e:
            sys.stderr.write(f"[radiance] tunable diag install failed: {e!r}\n")
    if DRAFT_TOPKP:
        for f in (_install_draft_topkp, _install_draft_topkp_v2):
            try:
                f()
            except Exception as e:             # never cost the vocab/int2 install below
                sys.stderr.write(f"[radiance] {f.__name__} failed (stock draft sampling there): {e!r}\n")
    if VOCAB_FILE and triton is not None:
        _install_vocab()
        return
    if not FAST:
        # stock bf16 head; nothing patched, nothing quantised
        return
    if triton is None:
        sys.stderr.write("[radiance] quantised draft head off: no triton\n")
        return

    # (module, class, which LogitsProcessor that drafter scores candidates with)
    targets = []
    for mod_name, cls_name, lp_attr in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP", "logits_processor"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP", "logits_processor"),
        # DFlash2 reaches the head through get_top_k_tokens, which calls _apply_head like
        # everything else. Its lm_head is not in the drafter checkpoint -- it is shared in from
        # the target after load_weights -- so this one always takes the lazy path.
        ("vllm.model_executor.models.qwen3_dflash2", "DFlash2Qwen3ForCausalLM",
         "candidate_logits_processor"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            targets.append((getattr(mod, cls_name), lp_attr))
        except Exception:
            continue
    if not targets:
        sys.stderr.write("[radiance] quantised draft head off: no drafter class found\n")
        return

    for cls, lp_attr in targets:
        if getattr(cls, "_radiance_quant_head_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig, _lp=lp_attr):
            loaded = _orig(self, weights)
            try:
                status = _quantize_draft_head(self, _lp)
            except Exception as e:
                status = f"FAILED, bf16 head kept: {e!r}"
            sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: {status}\n")
            sys.stderr.flush()
            return loaded

        cls.load_weights = wrapped
        cls._radiance_quant_head_wrapped = True
    sys.stderr.write(f"[radiance] int{BITS} draft head armed (RADIANCE_FAST_DRAFT)\n")
    sys.stderr.flush()
