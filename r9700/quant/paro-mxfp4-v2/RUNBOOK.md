# PARO-MXFP4 v2: better rounding + MXFP4-trained rotations, same bytes/kernel/speed (2026-09-23)

Goal: a drop-in replacement for the finalist `Qwen3.8-27B-PARO-MXFP4-mtpfp8-lmfp8-embfp8`. The format, tensor names,
shapes and dtypes are identical, so it has the same speed and context. Only the 4-bit codes, the block scales and
optionally the rotation angles/scales change. The finalist was built with plain round-to-nearest using the OCP scale
rule, which clips the largest value in ~40% of blocks, and with z-lab's rotations that were trained for int4.

## Variants the pipeline builds (all scored against bf16 on the same sets)
| name | what |
|---|---|
| rtn | z-lab rotations + OCP RTN = the finalist, rebuilt (sanity: wiki delta should be ~+0.8%) |
| ss | + per-block scale search (e, e+1, e-1) |
| gptq_ss | + GPTQ error-compensating rounding in the rotated basis, calibrated on the omp traces |
| mxrot_ss | rotations + weights re-trained against MXFP4 (z-lab optimizer, warm-started), scale search |
| mxrot_gptq_ss | re-trained rotations + tuned weights, then GPTQ |
| mxrot1_gptq_ss | re-trained rotations only (base weights), then GPTQ |
| *_a8 | the saved variants re-scored with the server's per-token fp8 activations emulated |
| redhat_gptq_awq | control: RedHatAI/Qwen3.8-27B-MXFP4 (public, 2026-09-18: GPTQ + AWQ smoothing, no rotations), weights-only |

Then, automatically: `pick.py` chooses the body with the lowest held-out KL (fp8-activation scores), `mtp_refit.py`
re-fits the MTP draft head to THAT body (self-distillation on the traces: the head's input is the quantized body's
hidden state and its target is the quantized body's next-next-token distribution, fp8 activations emulated), exports it
in the finalist's fp8 layout (recipe verified byte-identical on the base head), and adopts it only if its held-out
acceptance (sum min(p,q)) beats the finalist's head. lm_head/embed fp8 shards are reused: they are pure functions of the
bf16 base, independent of the body quant, so regenerating them would give the same bytes.

Eval sets: wiki (same 40,960 tokens as the earlier pod runs), code-sample.txt (30x2048), and held-out agent traces
(24x8k, 6x32k, from 18 whole subagent sessions that are not in the calibration pool). Metrics: PPL, KL vs bf16 (top-32
+ tail), top-1 agreement, and the same for assistant-written tokens only.

## Pod
1x H100 80GB, **>=128 GB system RAM** (optimizer keeps the fp16 model + layer caches on CPU), **300 GB volume** at
/workspace, template "Runpod PyTorch", SSH over exposed TCP (direct root@IP -p PORT; the ssh.runpod.io proxy can't scp).

## Upload (from Tower)
```bash
P=<port>; IP=<ip>
ssh -p $P root@$IP mkdir -p /workspace/pq2/finalist-extras
scp -P $P /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2/* root@$IP:/workspace/pq2/
F=/mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4-mtpfp8-lmfp8-embfp8
scp -P $P $F/model-mtp.safetensors $F/model-lmhead-fp8.safetensors $F/model-embed-fp8.safetensors \
    /mnt/user/appdata/vllm-radiance/custom-quant/finalist-manifest.json root@$IP:/workspace/pq2/finalist-extras/
```

## On the pod
```bash
bash /workspace/pq2/setup_pod.sh 2>&1 | tee /workspace/pq2/setup.log          # ~10-15 min (74 GB of downloads)
mkdir -p /workspace/pq2/logs
tmux new -s pq2 'bash /workspace/pq2/pipeline.sh 2>&1 | tee -a /workspace/pq2/logs/pipeline.log'
python3 /workspace/pq2/summary.py                                              # any time
```
The pipeline can be re-run after an interruption: finished steps are skipped, and the optimizer resumes per layer.
Estimated time: selftest 1 min, ref ~10, rtn/ss ~10 each, gptq ~1 h, optimizer ~1.5-3 h (TRAIN=512; watch the
per-layer time in logs/optimize.log), the three mxrot variants ~2.5 h, a8 re-scores ~10 min each. About 6-8 h total.
If the 32k set OOMs under sdpa: add `--attn kernels-community/flash-attn2` (pip install kernels) to run.py.

## Result, verification, transfer
The pipeline ends with `/workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-<WINNER>`; finish.py exits non-zero unless every
tensor name/dtype/shape matches the finalist. MTP refit numbers: `mtp_results.jsonl`. To ship a different body by hand:
```bash
cd /workspace/pq2; V=<variant>
python3 mtp_refit.py --codes $V --act-fp8
python3 package.py --variant $V --out /workspace/models/pkg-$V
python3 finish.py --pkg /workspace/models/pkg-$V --extras finalist-extras --out /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-$V \
  $( [ -f mtp/$V/model-mtp.safetensors ] && echo --mtp mtp/$V/model-mtp.safetensors )
```
From Tower (pull; the pod can't reach the LAN):
```bash
ssh root@192.168.88.89 'rsync -a --info=progress2 -e "ssh -p <port>" root@<ip>:/workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-<V>/ /mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4-v2-<V>/'
```
Also pull `/workspace/pq2/results.jsonl` and `logs/` for the record.

## Serve on Tower (same launcher, new SNAP)
```bash
cd /mnt/user/appdata/llama-gemma31b && SNAP=/mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4-v2-<V> ./deploy-vllm-qwen38-paro.sh
```
Gates (same as the finalist): served wiki/code PPL on a small-KV deploy (served_ppl.py; finalist 5.9796 / 2.3304),
greedy decode @8k (finalist 77-78) and sampled decode + vLLM's SpecDecoding acceptance lines (the MTP refit should show
here), NIAH 225k 8/8, determinism x3. Rollback: plain launcher (SNAP default = finalist).
