#!/bin/bash
# Long-context A/B of the R4D prefill attention fp8 legs (libr4d rx9y: R4D_ATTN_FP8 0 / 3 / 7 / 15 / 14) on the rig:
# per mode, deploy (MAXLEN 131072), cold prefill speed at 48k / 112k, then bench/lastpos_kl.py collect (next-token top-20
# at 80 cut points of 32k-112k depth). Afterwards compare each mode against mode 0 (f16 legs).
# Usage: p8-attn-e2e.sh "<modes>"
set -u
cd /mnt/user/appdata/llama-gemma31b
MODES=${1:-"0 3 7 15 14"}
R=bench/results/p8; mkdir -p $R
for M in $MODES; do
  echo "== mode $M"
  R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=$M MAXLEN=131072 NOBENCH=1 bash jobs/p7-vllm-arm.sh attn$M 2>&1 | grep -E 'deployed|FAILED'
  docker exec vllm-exp env | grep R4D_ATTN_FP8
  docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base http://192.168.88.89:1252 --model qwen38-27b \
    --label attn$M --depths 43000,100000 --runs 1 --n-predict 8 --tag a$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill /'
  docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py collect --base http://192.168.88.89:1252 --out /repo/$R/lastpos-$M.json 2>&1 | tail -1
done
for M in $MODES; do
  [ "$M" = 0 ] && continue
  docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-0.json /repo/$R/lastpos-$M.json 2>&1 | tail -9
done
echo P8_DONE
