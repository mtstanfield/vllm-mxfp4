#!/bin/bash
# Served PPL of one vLLM config on the rig (vllm-exp :1252): a small-KV deploy (prompt_logprobs allocates a full-vocab
# fp32 matrix per prompt token -- the prod-size pool OOMs), then bench/served_ppl.py over wikitext-2 test (60 chunks)
# and the code sample (30 chunks), 2048-token chunks scored on their second half (llama-perplexity's rule).
# Usage: [env...] p7-vllm-ppl.sh <label>     (env as for p7-vllm-arm.sh: R4D_KEY, SINGLE_GPU_PROFILE, ARM_ENV, ...)
set -u
cd /mnt/user/appdata/llama-gemma31b
L=$1
export KV_MEM=${KV_MEM:-4000000000} MAXLEN=${MAXLEN:-65536}
NOBENCH=1 bash jobs/p7-vllm-arm.sh "$L" || exit 1
PY="docker exec llama-hip-dev python3"
for t in "wiki.test.raw 60" "code-sample.txt 30"; do
  set -- $t
  $PY /repo/bench/served_ppl.py --url http://192.168.88.89:1252 --model qwen38-27b --text /repo/bench/$1 --ctx 2048 \
    --chunks $2 --label "$L-$1" --json /repo/bench/results/ppl-vllm-p7.jsonl 2>&1 | tail -1
done
echo P7_PPL_DONE $L
