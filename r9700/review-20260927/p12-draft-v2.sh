#!/bin/bash
# p10 follow-up. The serve runs vLLM 0.29's V2 model runner, so p10's draft top-k/top-p (a V1 proposer hook) was inert;
# radiance_drafthead.py now hooks the V2 speculator too. Greedy uses the FIXED prompt tag (dec) this time.
#   eA = production              : greedy only
#   eE = + EXACTSET (= p10 dC)   : sampled (must reproduce dC exactly: deterministic engine, seed 0) + greedy
#   eF = + EXACTSET + TOPKP (V2) : sampled + greedy
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
docker stop vllm-qwen38 >/dev/null 2>&1
greedy() {
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  R=$($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $1 --depths 8000 --runs 1 --n-predict 600 --tag dec 2>&1 | grep -E 'prompt=')
  echo "greedy 8k dec :: $R :: $($PY /repo/bench/vllm_accept_delta.py delta)"
}
logs() { docker logs vllm-exp 2>&1 | grep -E 'draft top-k|exact-reranked|topkp.*failed' | cut -c1-160 | sort -u; }
echo "== eA"; ARM_ENV="$VOC" NOBENCH=1 bash jobs/p7-vllm-arm.sh eA 2>&1 | grep -E 'deployed|FAILED'; greedy eA
echo "== eE"; ARM_ENV="$VOC -e RADIANCE_DRAFT_EXACTSET=1" RUNS=12 bash jobs/p7-vllm-sampled.sh eE 2>&1 | grep -E 'deployed|FAILED|TOTAL|GRAND'; logs; greedy eE
echo "== eF"; ARM_ENV="$VOC -e RADIANCE_DRAFT_EXACTSET=1 -e RADIANCE_DRAFT_TOPKP=1" RUNS=12 bash jobs/p7-vllm-sampled.sh eF 2>&1 | grep -E 'deployed|FAILED|TOTAL|GRAND'; logs; greedy eF
echo P12_DONE
