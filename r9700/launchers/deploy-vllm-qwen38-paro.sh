#!/bin/bash
# deploy-vllm-qwen38-paro.sh - the 2026-09-17 finalist: z-lab rotations on MXFP4 weights (paroquant_mxfp4, built on RunPod) +
# fp8 MTP head + fp8 lm_head + fp8 row embed_tokens, served through the paroquant plugin (local/patch_paroquant_install.py)
# with the fork's W5A8 knobs. Measured vs the AMD MXFP4 prod: wiki PPL 6.104 -> 5.980, code 2.341 -> 2.330, TTFT 1.16 -> 1.04 s,
# greedy decode 58.7 -> 53.4 @8k, weights 16.86 -> 16.43 GiB, KV pool 247k -> 258k (MAXLEN 253,952; a 253,011-token request served).
# Same knobs as the main wrapper (KV_MEM/MAXLEN/MAXSEQS/EXTRA/... still overridable); restore the old prod with a plain
#   ./deploy-vllm-qwen38.sh
# 2026-09-17 evening: vLLM 0.29 from-source image (ggz14 recipe, build.sh --full; the recipe verify step fails on the paroquant files so the
# intermediate image is tagged -prebake) + the dev single-GPU profile (fp16 SSM state, rx9 kernels) with lazy GDN OFF (corrupts output on 0.29
# at mamba block boundaries; port + repro in the worktree). Gates: wiki 5.9845 / code 2.333, greedy 77-78 @8k (was 54.5), 63.8 @100k, warm
# TTFT 0.48 s @8k / 1.11 s @100k, pool 273k -> MAXLEN 262,144 (a 260,174-token request served), NIAH 225k 8/8. Own caches so the
# 0.27.1 tree stays intact: SINGLE_GPU_PROFILE=0 keeps fp32 state (pool 258k, 76.5 @8k, 66 @100k); IMAGE= unset + CACHE=cache-next-paro = old tree.
P=/mnt/user/appdata/vllm-radiance/persist
export IMAGE=${IMAGE:-ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake}
export AITER_DIR=${AITER_DIR:-$P/aiter-jit-029} R4D_CACHE=${R4D_CACHE:-$P/libr4d-029}
# 2026-09-28 evening (round 6): rotation stream 1 is live (RADIANCE_FP8_STREAM=0 below). serve-mxfp4.sh appends the
# -tp1s cache suffix only when FP8_STREAM=1, so pin the production cache dir here -- without it a stream-1 boot runs
# from cache-029-paro (no TunableOp table, no GPTQ drafter file): that, not stream 1, was round 4's "5% slower".
export RADIANCE_FP8_STREAM=${RADIANCE_FP8_STREAM:-0}
if [ "$RADIANCE_FP8_STREAM" = 1 ]; then export CACHE=${CACHE:-$P/cache-029-paro}; else export CACHE=${CACHE:-$P/cache-029-paro-tp1s}; fi
# 2026-09-27 (docs/vllm-radiance-review-20260927.md, A/B on the vllm-exp rig):
#   R4D_KEY=b9e42ab-rx9x  libr4d with the EXACT-decay GDN chunk scan (vllm-patches/review-20260927/patch_rx9x.py). The stock
#                         midpoint split + e^80 clamp moved served code PPL +0.5-0.6% off the FLA reference (2.3170 vs 2.3037;
#                         wiki 5.9592 vs 5.9757); rx9x tracks the reference (2.3065 / 5.9846), no prefill speed change.
#                         R4D_KEY=b9e42ab-rx9 = the stock kernel.
#   FAST_DRAFT=0 + RADIANCE_DRAFT_VOCAB (EXTRA_ENV below): the int2 draft head on the 48k-row draft vocabulary only
#                         (radiance_drafthead.py local change) instead of on all 248k rows: +5% sampled decode (code 75.2 -> 80.2,
#                         prose 63.6 -> 68.0), +6-7% greedy, byte-identical greedy text. FAST_DRAFT=1 with the vocab env
#                         removed = the old full-vocab int2 head. SPEC 8 was tested and rejected (sampled -7%, pool -32k).
#   R4D_KEY=b9e42ab-rx9y + R4D_ATTN_FP8=15 (2026-09-27 correctness review): rx9y = rx9x + the prefill-attention fp8-P
#                         fix (consistent denominator + 4-octave headroom shift; vllm-patches/review-20260927/patch_attnden.py,
#                         patch_attnsh.py). At 32k-115k depth the stock fp8 legs (mode 3) flipped the top next token at 6%
#                         of positions vs the f16 path; mode 15 at 0%, same prefill speed (2615/2083 vs 2617/2098 t/s @48k/120k).
#                         R4D_ATTN_FP8=3 = the stock fp8 legs, 0 = f16 legs (-7%/-14% prefill).
#   R4D_KEY=b9e42ab-rx9z (2026-09-28 day, round 5): rx9y + mode-15 prefill attention re-tuned for the 8-bit legs (prefetch 8,
#                         PVDEN summed by fp8 dot4 after the PV WMMAs) -- bit-identical, attention -4.5%, prefill +1.4-1.7% at 100k+.
export R4D_KEY=${R4D_KEY:-b9e42ab-rx9z} FAST_DRAFT=${FAST_DRAFT:-0} R4D_ATTN_FP8=${R4D_ATTN_FP8:-15}
# 2026-09-17: fork commit 1e82407 (evaluation worktree vllm-radiance-next) via deploy-vllm-qwen38-next.sh: +33% greedy decode from the
# TP=1 kernel pass; the single-GPU PROFILE stays OFF (fp16 SSM state makes our GDN layers fall back to FLA; lazy GDN needs vLLM 0.29).
export SINGLE_GPU_PROFILE=${SINGLE_GPU_PROFILE:-1}
export RADIANCE_GDN_LAZY=${RADIANCE_GDN_LAZY:-0}   # host-side: serve-mxfp4.sh decides the lazy patch + rx9/rx10 from THIS env, not from the container -e
WRAPPER=${WRAPPER:-deploy-vllm-qwen38-next.sh}   # WRAPPER=deploy-vllm-qwen38.sh + CACHE=…/cache-paro-mxfp4-knobs = the pinned 92eed82 tree
export SNAP=${SNAP:-/mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4-v2-mxrot1_gptq_ss_mix}   # v2 2026-09-23 (GPTQ + MXFP4-trained rotations); old finalist: SNAP=/mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4-mtpfp8-lmfp8-embfp8
# OFFLOAD=1 (default since 2026-09-29, see the EXTRA block below): vision tower + fp8 embed table in pinned host RAM, -2.07 GiB of
# weights -> KV_MEM 10.1e9 -> 13.9e9, pool 273,333 -> 375,633 tokens. OFFLOAD=0 restores the 10.1e9 pin; splitting the knobs
# (VIS_OFFLOAD / EMBED_UVA) needs an explicit KV_MEM (vision-only: 12.6e9 = 340,467 tokens).
export OFFLOAD=${OFFLOAD:-1}
if [ "$OFFLOAD" = 1 ]; then KV_MEM_DEFAULT=13900000000; else KV_MEM_DEFAULT=10100000000; fi
# MAXSEQS 2 -> 4 (2026-09-29): 4 x 64k agents fit the 376k pool (each active seq also pins 18 GDN blocks = ~15k tokens):
# 4 x 64,000 concurrent +512 served, 0 preemptions, KV 78%, peak VRAM 31,681 / 32,624 MiB (graphs 0.68 -> 0.88 GiB); single
# stream unchanged (greedy dd828e3e, 138.4 vs 139.1 t/s @8k; sampled 90.5, noise). A 5th full 64k session would preempt.
export KV_MEM=${KV_MEM:-$KV_MEM_DEFAULT} MAXLEN=${MAXLEN:-262144} MAXSEQS=${MAXSEQS:-4}
# 2026-09-27 review round 3 (docs/vllm-radiance-review-20260927.md), both lossless:
#   TunableOp table for the skinny fp8 GEMMs (MTP drafter linears + fp8 lm_head, M=1..12,16; built by
#     vllm-patches/review-20260927/fp8_tune.py into persist/cache-029-paro-tp1s/tunableop/skinny0.csv, read-only here):
#     hipBLASLt's heuristic tile left the N=5120 draft GEMMs at 40 workgroups; sampled decode 80.4 -> 84.5 t/s, greedy
#     text identical. The table's validators drop it on an image change (defaults return) -> re-run fp8_tune.py.
#   RADIANCE_DRAFT_EXACTSET=1: sampled drafts only from the 32 exactly reranked draft-head candidates (+0.8% sampled).
TUNABLE_ENV="-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv"
# 2026-09-28 overnight round (docs/vllm-radiance-review-20260927.md "Round 4"), both BYTE-EXACT (greedy sha + sampled acceptance
# identical to the previous production; kernel gates vllm-patches/review-20260927/mr_check.py + split_check.py ALL EXACT):
#   kernel .so -split2 (persist/paroquant; built from paroquant/radiance_paroquant.hip + par_kernels_mr*.h + par_kernels_split.h)
#   RADIANCE_PQM_ROT3=1: prefill norm-fed rotate+quant on the conflict-free multi-row core (-36% kernel time) -> prefill ~+1%
#     (the ew-producer variants stay off: RADIANCE_PQM_ROT3_EW, see the fault note in radiance_paroquant_mxfp4.py)
#   RADIANCE_PQM_SPLIT=1: decode producers spread each token over ceil(G/16) workgroups (last arriver encodes) -> sampled +1.2%
#   RADIANCE_LOCAL_AOT_ENVKEY=1 (local/patch_aot_envkey.py): torch_aot_compile dirs keyed on the RADIANCE_* env too -- vLLM keys
#     only config + traced source, so an env-driven graph change silently loaded a stale artifact.
#   RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 (local/patch_gdn_warmup_sync.py): device sync before the empty_cache() vLLM 0.29 runs
#     inside the boot profile forward (GDN warm-up) -- hardening against unmapping buffers in-flight kernels still read
#     (suspected cause of the boot-time memory faults seen 2026-09-28 with the rot3 ew producers); boot-only, no numerics.
# 2026-09-28 day (round 5): RADIANCE_MTP_MXFP4=1 -- the MTP drafter's linears requantized fp8 -> MXFP4 at load and run on the
#   body's W4A8 kernel (one opaque op per linear): draft pass -27% bytes, step -5.2%, sampled 85.9 -> 88.6 t/s (+3%) at
#   slightly lower acceptance (.531/.430/.725 -> .521/.415/.682). Drafts only: the verified output distribution is exact.
#   (SPEC 5 re-tested with it: 87.3, SPEC 4 stays.) The codes come from an offline GPTQ fit on the drafter's own inputs
#   (RADIANCE_MTP_MXFP4_FILE; vllm-patches/review-20260927/mtp_gptq.py, calibrated on 48 windows of the omp traces):
#   on held-out omp-session windows acceptance fp8 .574 / RTN .547 / GPTQ .559, decode 75.3 / 76.0 / 77.9 t/s. The file
#   carries a fingerprint of the fp8 weights it was fitted to; any other checkpoint falls back to load-time RTN (logged).
#   patch_aot_envkey.py now also keys vLLM's piecewise compile cache
#   (torch_compile_cache/<hash>, loaded by piece index): the drafter's inductor assertion was stale pieces of another env's
#   graph. With that fixed, the round-4 byte-exact extras are back ON (p37,
#   greedy shas + sampled acceptance identical, 3/3 boots clean, sampled +0.3%, prefill +0.3-0.5%): the rot3 ew producers
#   (RADIANCE_PQM_ROT3_EW) + SKIP_HS, and the GDN core_attn_out zero_() skip (RADIANCE_LOCAL_GDN_NOCOPY=1 + EMPTY_OUT,
#   which serve-mxfp4.sh also reads on the HOST). The z-in-place read (RADIANCE_LOCAL_GDN_NOCOPY_Z=1) stays off: not exact,
#   and it is the prime suspect for the round-4 faults (all in the gated-norm producer = the z consumer).
# 2026-09-28 afternoon (round 5b): prefill producers on the select-free "rot4" core -- kernel .so -0.29-rot4 (built from
#   paroquant/radiance_paroquant.hip + par_kernels_mr4.h) + RADIANCE_PQM_ROT4=1. Byte-exact (mr4_check.py / real_check.py
#   on the model's own rotation records: ALL EXACT; e2e text identical to the rot3 path on the same compile), prefill
#   +4.5% @8k / +3.7% @43k / +3.1% @112k. RADIANCE_LOCAL_PQ_SO=...-split2.so without ROT4 = the round-5 kernels.
export RADIANCE_GDN_EMPTY_OUT=${RADIANCE_GDN_EMPTY_OUT:-1}
# 2026-09-28 evening (round 6, decode fusion): rotation stream 1 live (RADIANCE_FP8_STREAM=0, see CACHE above): one split
#   add+RMSNorm+rotate+quant kernel per norm site instead of inductor add+rms + rotate (128 launches/step fewer):
#   8k step 35.3 -> 34.6 ms, sampled 89.1 -> 90.5, held-out omp windows 77.8 -> 80.8 t/s, PPL 5.977/2.304 -> 5.970/2.304.
#   RADIANCE_DRAFT_FUSED=1: the exact-set draft head in 6 launches instead of 15 per drafter pass (radiance_drafthead.py
#   _apply_vocab_fused; identical drafts): sampled 90.5 -> 91.0, omp windows 80.8 -> 81.3 t/s.
#   Tried, not adopted: vLLM's fused QK-RMSNorm+mRoPE+gate kernel (local/patch_fused_qknr.py, made ROCm-compilable;
#   RADIANCE_LOCAL_FUSED_QKNR=1 [+ _TARGET_ONLY=1]): -1.8% profiled step, but drafter acceptance drops by about as much
#   (72 omp prompts: .592 -> .584; target-only .590) -> end to end within +-0.6% of not having it.
R4_ENV="-e RADIANCE_DRAFT_FUSED=1 -e RADIANCE_FP8_STREAM=$RADIANCE_FP8_STREAM -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=${MTP_GPTQ:-/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt} -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=$RADIANCE_GDN_EMPTY_OUT"
export EXTRA_ENV=${EXTRA_ENV:--e PYTHONHASHSEED=0 -e RADIANCE_PAROQUANT_INSTALL=1 -e RADIANCE_PAROQUANT=1 -e RADIANCE_PQ_I8=1 -e RADIANCE_PQ_PG=1 -e RADIANCE_PQ_ZPE=1 -e RADIANCE_PQ_ROT_STREAM=1 -e RADIANCE_PQ_ROT_STREAM2=1 -e RADIANCE_GDN_LAZY=${RADIANCE_GDN_LAZY:-0} -e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 $TUNABLE_ENV $R4_ENV}
# MTP drafting samples from the draft distribution (A/B 2026-09-17: accepted/draft-token 0.505 -> 0.548, sampled decode +5% @8k, greedy unchanged);
# the serve script only exposes DRAFT_SAMPLE for dflash, so the whole spec config is re-passed via EXTRA (last --speculative-config wins).
MTP_SPEC_DEFAULT='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
export EXTRA=${EXTRA:-$MTP_SPEC_DEFAULT}   # a JSON default written inline in ${var:-...} is cut at its first }; keep it in a variable
# VRAM -> KV pool (2026-09-29): OFFLOAD=1 keeps the vision tower (0.86 GiB bf16, stock --cpu-offload-params) and the fp8 embed
# table (1.18 GiB, PQ_EMBED_UVA in local/patch_paroquant_install.py) in pinned host RAM, read over PCIe through UVA views.
# Measured: weights 16.19 -> 14.12 GiB; greedy shas unchanged (dd828e3e 8k / 9718d93c 100k); prefill 3490 @8k; sampled 91.4
# (vision-only arm 92.0 / 91.0, prod 92.3: noise); screenshot TTFT 0.48 -> 0.72 s. Peak VRAM at KV_MEM 13.9e9 over a 255k
# prefill + 2 x 165k concurrent (0 preemptions, KV 88%): 31,585 / 32,624 MiB -- keep ~1 GiB spare, do not push further blind.
if [ "${VIS_OFFLOAD:-$OFFLOAD}" = 1 ]; then EXTRA="$EXTRA --cpu-offload-gb 1 --cpu-offload-params visual"; fi
EXTRA_ENV="$EXTRA_ENV -e PQ_EMBED_UVA=${EMBED_UVA:-$OFFLOAD}"   # appended, not in R4_ENV: callers that replace EXTRA_ENV (p7 arms) must not get the 13.9e9 pin with the table on the GPU
exec "$(dirname "$0")/$WRAPPER" "$@"
