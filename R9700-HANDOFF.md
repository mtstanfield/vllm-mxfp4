# Qwen3.8-27B on Radeon AI PRO R9700: the `r9700-tp1` branch

This branch is GGZ14's vLLM-Radiance fork (`github.com/GGZ14/vllm-mxfp4`, mirrored at `codeberg.org/ggz14/radiance-vllm-mxfp4`) at
commit **1e82407**, plus two weeks of single-card tuning for one R9700 (gfx1201, 32 GB). It serves Qwen3.8-27B as a PARO-MXFP4
checkpoint with MTP speculative decoding. **Everything here was built and validated at tensor parallel 1 on one card.** If you are
an agent setting this up on a two-card machine, read "Two cards" before you start.

`git diff 1e82407..r9700-tp1` shows every change we made to the fork. `r9700/docs/vllm-radiance-review-20260927.md` is the
engineering log (rounds 1-7): what each change does, how it was measured, and what was tried and rejected. Read it before
changing anything; most knobs have a recorded A/B.

## Results on one card (300 W, TP=1)

BetterBench 0.6.0, corpus v1.0, 2026-09-29, against the launcher defaults on this branch:

| | this branch | fork at 1e82407 with our 17 Sep config |
|---|---|---|
| decode, 1 stream (weighted) | 102.3 tok/s | 88.4 |
| decode, 1 stream (median of all runs) | 108.1 tok/s | 93.3 |
| prefill at 5.9k / 47k tokens | 3,357 / 3,000 tok/s | 3,130 / 2,802 |
| aggregate at 1 / 2 / 4 / 8 streams | 97 / 171 / 278 / 280 tok/s | 85 / 151 / 151 / 150 |
| 8 streams with `MAXSEQS=8 MAXLEN=65536` | 426 tok/s | - |

- KV cache pool 375,633 tokens (fp8), max model length 262,144, up to 4 concurrent sequences (4 x 64k agents fit without preemption).
- Output quality: served perplexity wikitext 5.970 / code 2.304; greedy output is bit-identical across every lossless change.

## What is where

| path | what |
|---|---|
| modified fork files (`git diff`) | `paroquant/*` rotation/quantize producers (rot3/rot4/split, see `paroquant/par_kernels_mr*.h`), `radiance_drafthead.py` (48k draft vocabulary, exact rerank, fused draft head), `radiance_gdn*.py`, `radiance_r4d_attn.py`, `patch_gdn_lazy.py`, `build.sh` (one-line `set -e` fix) |
| `local/*.py` | our container-start patches. The launcher's docker shim runs every `/patches/local/*.py` at boot; each is gated by its env var and prints what it did |
| `local/qwen38-draft-vocab-49152.txt` | draft vocabulary (token ids only) used by `RADIANCE_DRAFT_VOCAB` |
| `tools/stress_tok.py`, `tools/vis_lat.py` | exact-length KV/VRAM stress client; screenshot latency probe |
| `r9700/launchers/` | our deploy scripts: `deploy-vllm-qwen38-paro.sh` (all production knobs, heavily commented) -> `deploy-vllm-qwen38-next.sh` (docker shim + health wait) -> the fork's `serve-mxfp4.sh` |
| `r9700/review-20260927/` | every experiment script (`pNN-*.sh`), the libr4d patches (`patch_rx9x.py`, `patch_attnden.py`, `patch_attnsh.py`, `patch_rx9z.diff`), kernel diffs, check/bench tools (`fp8_tune.py`, `mtp_gptq.py`, `attn_bench.py`, ...) |
| `r9700/prebuilt/tunableop/` | TunableOp tables for the skinny fp8 GEMMs (per rank: `skinny%d.csv`) |
| `r9700/bench/` | `spec_sampled.py` (sampled decode), `oai_bench_greedy.py` (greedy/prefill by depth), `vllm_accept_delta.py` |
| `r9700/quant/` | how the checkpoint and the fp8 heads were built (`paro-mxfp4-v2/RUNBOOK.md` is the current recipe) |

## Not in the branch

| item | size | where it goes |
|---|---|---|
| checkpoint: [huggingface.co/MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2](https://huggingface.co/MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2) | 16 GB | `hf download MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2 --local-dir $MODELS/Qwen3.8-27B-PARO-MXFP4-v2`, then point the launcher's `SNAP` at it. Rebuilding it needs an 80 GB GPU for several hours (`r9700/quant/paro-mxfp4-v2/RUNBOOK.md`); its calibration mixed the owner's agent traces (not shipped) with wikitext. |
| MTP drafter GPTQ file: `extras/qwen38-paro-v2-mtp-gptq-a05.pt` in the same HF repo | 225 MB | `$CACHE/mtp_gptq/`. Without it, drop `RADIANCE_MTP_MXFP4*` from `R4_ENV` (fp8 drafter, ~1% slower). |
| image `ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake` (tar.gz + .sha256), **from the owner** | 3.7 GB | `docker load -i <file>`. Or build it: `./build.sh --full --jobs=12` on this branch (~1.5 h; torch 2.11 / triton 3.6 / aiter 0.1.17 / vLLM 0.29.0 / ROCm 7.14). The recipe's final verify step fails on the paroquant files; the image before that step is the one tagged `-prebake`. |

Checksums: image tarball `70f073ece45a97c88dea98767f0e761ef9f6ad27d73669f35b2a6918235d2a38`, drafter file
`943d18921d532c58898301ba3ecf6a1114864d78ed6f9c16397b376729252270`, checkpoint files in `r9700/checkpoint-SHA256SUMS`.

**Prebuilt kernels (GitHub release `r9700-tp1-20260929` on this fork):**

| asset | where it goes |
|---|---|
| `libr4d-b9e42ab-rx9z-gfx1201.tar.gz` | unpack into `$R4D_CACHE/` so that `$R4D_CACHE/b9e42ab-rx9z/r4d.so` exists (libr4d at b9e42ab + the fork's extras + our four patches, sources included) |
| `radiance_paroquant_kernel-0.29-rot4.so` | `<this checkout>/persist/paroquant/` (selected by `RADIANCE_LOCAL_PQ_SO`) |

Both are compiled for gfx1201 against the image above, so they work as-is on any R9700 running that image.

## Paths to change

The launchers are written for the owner's Unraid box. Change these (all near the top of the two launcher scripts):

| owner path | meaning |
|---|---|
| `P=/mnt/user/appdata/vllm-radiance/persist` | persistent root: `CACHE` (compile caches, TunableOp table, drafter file), `R4D_CACHE=$P/libr4d-029`, `AITER_DIR` |
| `UP=/mnt/user/appdata/vllm-radiance-next` | this checkout (mounted into the container as `/patches`) |
| `MODELS=/mnt/user/Models/vllm-radiance`, `HF_CACHE=/mnt/user/Models/hf-cache` | checkpoint dir, HF cache |
| `PATH=.../hostshim:$PATH` | Unraid has no host python3; on a normal Linux host delete this line |
| `. bin/env.sh` | llama.cpp-era variables; the wrapper only uses `JOBS_DIR` from it (where failed boot logs go) |

Two guards in `deploy-vllm-qwen38-next.sh` will stop you: it refuses to run while a container named `llama-qwen38` exists
(owner-specific, delete the check), and it refuses unless `$UP` HEAD equals `PIN_COMMIT=1e82407`. On this branch HEAD is our
commit, so set `PIN_COMMIT` to `git rev-parse --short HEAD` or delete the check.

The TunableOp table goes to `$CACHE/tunableop/skinny0.csv`. Its validators include the torch/hipBLASLt versions: with the
owner's image it applies as-is; with a self-built image it may be silently ignored (defaults return, ~5% slower sampled decode).
Re-tune with `r9700/review-20260927/fp8_tune.py` in that case.

## Reproduce one card first

1. Load the image, put the checkpoint, drafter file, kernels and TunableOp table in place, fix the paths.
2. `GPUS=0 bash r9700/launchers/deploy-vllm-qwen38-paro.sh`. The first boot compiles (Triton/inductor, several minutes);
   later boots take ~2.5 min. It prints `READY-VLLM ... kv_tokens=375633` when healthy.
3. Check the boot log for: `Total CPU offloaded parameters: 0.86`, `fp8 embed_tokens -> pinned host memory via UVA (1.18 GiB`,
   `Model loading took 14.12 GiB`, `Graph capturing finished ... took 0.88 GiB`, no `Memory Fault` or `Traceback`.
4. Measure (owner's figures at 300 W in brackets):
   - `python3 r9700/bench/spec_sampled.py --base http://HOST:1246 --runs 12 --n 600 --metrics http://HOST:1246/metrics --label x`
     [GRAND 90-92 t/s; code ~90, prose ~79, json ~110]
   - `python3 r9700/bench/oai_bench_greedy.py depth --base http://HOST:1246 --model qwen38-27b --label x --depths 8000 --runs 1 --n-predict 600 --tag dec`
     [prefill ~3,480 t/s, greedy decode ~138 t/s, sha `dd828e3e`]. The sha only matches when the compile artifacts match; a
     different sha on a fresh compile is expected and not a failure (see "compile variance" in the log). Check perplexity instead.
   - `python3 tools/stress_tok.py http://HOST:1246 64000,64000,64000,64000 512` [all ok, 0 preemptions, KV ~78%]
   - BetterBench 0.6.0 (`github.com/GGZ14/BetterBench`) with `--max-model-len 262144`.

## Two cards

**Recommended: two independent TP=1 servers, one per card.** This reproduces the single-card numbers on each card with no
untested code paths, and doubles aggregate throughput and total KV capacity.

- Run the launcher twice with `GPUS=0` and `GPUS=1`, a different `NAME`/`PORT` (edit the wrapper; it hardcodes
  `vllm-qwen38`/`1246`) and **separate `CACHE` dirs**. Two containers compiling into one cache dir at the same time can race;
  after both have booted once you can point them at copies of the same warm cache.
- Each replica pins ~2 GiB of host RAM for the offloaded vision tower and embedding table, which is nothing at 128 GB.
- Put any OpenAI-compatible router in front (LiteLLM, nginx `least_conn`). Route each agent session stickily to one replica.
  Prefix caching is per server, and an agent that bounces between replicas re-prefills its whole context every turn.

**TP=2 (one model across both cards)** is the fork's native multi-GPU path (`serve-tp2.sh`, or `TP=2` with `serve-mxfp4.sh`),
and it should give a faster single stream. None of our changes were tested there. Known issues:

- The fork auto-detects TP as the largest of 8/4/2/1 that fits. Our wrapper exports `TP=1`; if you call the fork's scripts
  directly on a two-card box you get TP=2 unless you set it.
- `SINGLE_GPU_PROFILE` (fp16 SSM state, which halves the GDN state page and the pool math behind `KV_MEM`) and
  `RADIANCE_FP8_STREAM_TP1` (fp8 residual stream without an all-reduce) are TP=1-only in the fork.
- Untested at TP=2: the rot3/rot4/split producers, the MXFP4 MTP drafter, the fused draft head (built around a single-rank
  lm_head), the UVA embedding placement (`PQ_EMBED_UVA`, vocab-parallel sharding), and the TunableOp tables (TP=2 GEMM shapes
  differ; re-run `fp8_tune.py`, which writes one `skinny<rank>.csv` per rank).
- `KV_MEM` and `MAXSEQS` were sized for one card. At TP=2 each card holds half the weights and half the KV heads, so re-derive
  `KV_MEM` from measured peak VRAM (below) rather than copying ours. Max context stays 262,144 either way (a model limit).
- Suggested order: boot the fork's defaults at TP=2 first (no `R4_ENV`, no `OFFLOAD`), confirm it serves and check perplexity,
  then enable our switches one at a time with a greedy and perplexity check after each, the same way the log does it.

## Gotchas we paid for

- **Size `KV_MEM` from measured peak VRAM, not idle.** It is pinned in bytes because vLLM's profiler drifts run to run. Measure
  with `r9700/review-20260927/p57-vram-peak.sh` (sysfs `mem_info_vram_used`, 0.1 s samples) and `tools/stress_tok.py` at your
  longest prompt and highest concurrency, then keep ~1 GiB spare. Idle readings mislead: a weight moved off the GPU after
  loading leaves its block in torch's cache, so idle VRAM barely drops even though activations can reuse the space.
- **The AOT compile cache ignores the environment.** `local/patch_aot_envkey.py` keys it on the `RADIANCE_*` variables.
  `PQ_EMBED_UVA` is deliberately not `RADIANCE_*` because it never changes a traced graph.
- **The right `CACHE` dir matters.** `serve-mxfp4.sh` appends `-tp1s` to the cache dir only when `RADIANCE_FP8_STREAM=1`; the
  launcher pins `CACHE` because the TunableOp table and drafter file live there. A boot from the wrong dir is silently ~5% slower.
- **Greedy hashes only compare within one compile.** A fresh torch.compile can change 100k-token greedy text. Compare arms
  compiled in the same session; use perplexity for anything else.
- **`oai_bench_greedy.py --depths N` is not N tokens** (100000 -> ~112k, 120000 -> ~158k). Use `tools/stress_tok.py` for exact sizes.
- **Chunked prefill stalls other streams.** While one request prefills a long uncached prompt (8192-token chunks, ~2.4 s each),
  concurrent streams advance about one step per chunk. Prefix caching keeps normal agent turns small.
- **Power.** These numbers are at the stock 300 W. The owner normally caps at 250 W (a udev rule), which costs ~1.5% decode;
  prefill GEMMs are power-bound.
- **Keep `RADIANCE_GDN_LAZY=0`** on the 0.29 image; lazy GDN corrupts output at mamba block boundaries there.
- **Read a script before running it on a busy box.** `grep` deploy scripts for `rm`, `stop` and `kill` first; a "dry run" once
  took down the owner's production container.
