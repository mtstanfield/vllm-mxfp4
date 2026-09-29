#!/bin/bash
# Sampled-decode A/B of the draft distribution (radiance_drafthead.py local switches), production config otherwise
# (paro launcher defaults: rx9y, R4D_ATTN_FP8=15, FAST_DRAFT=0, SPEC 4, MAXLEN 262144, 48k draft vocab):
#   dA = production    dB = + RADIANCE_DRAFT_TOPKP=1    dC = + RADIANCE_DRAFT_EXACTSET=1 (on top of B)
# spec_sampled.py RUNS x 600 tokens x code/prose/json with the server's sampling (temp 1.0, top-p 0.95, top-k 20),
# acceptance from /metrics. dA and dC also get the greedy decode check (oai_bench_greedy 8k, 600 tokens).
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
docker stop vllm-qwen38 >/dev/null 2>&1
greedy() {
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  R=$($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $1 --depths 8000 --runs 1 --n-predict 600 --tag g$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=')
  echo "greedy 8k :: $R :: $($PY /repo/bench/vllm_accept_delta.py delta)"
}
arm() {  # <label> <extra env>
  echo "== $1"
  ARM_ENV="$VOC $2" RUNS=${RUNS:-12} bash jobs/p7-vllm-sampled.sh $1 2>&1 | grep -vE '^\s*$'
  docker logs vllm-exp 2>&1 | grep -E 'draft top-k|draft vocab armed|DRAFT_VOCAB:' | cut -c1-160 | sort -u
}
arm dA ""; greedy dA
arm dB "-e RADIANCE_DRAFT_TOPKP=1"
arm dC "-e RADIANCE_DRAFT_TOPKP=1 -e RADIANCE_DRAFT_EXACTSET=1"; greedy dC
echo P10_DONE
