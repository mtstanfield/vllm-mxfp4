#!/bin/bash
# Conflict-free multi-row PARO producers (par_kernels_mr3.h, radiance_paroquant_kernel-0.29-mr3.so) end to end:
# production settings, prod compile cache (the producers live inside opaque custom-op bodies, nothing traced changes).
# Arms: base (stock producers) / rot3 / rot3 + skip the unread HS write. Each: greedy 8k/100k decode (sha must match
# 55222d73 / 86cd9c47 -- decode is untouched, prefill must be byte-identical), cold prefill 8k/43k/100k twice.
set -u
cd /mnt/user/appdata/llama-gemma31b
PROD='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv'
R3='-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-mr3.so -e RADIANCE_PQM_ROT3=1'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L"
  ARM_ENV="$PROD $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
  docker logs vllm-exp 2>&1 | grep -E 'kernel override|rot3 tables failed' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | sort | uniq -c
  $PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L --depths 8000,43000,100000 --runs 1 \
    --n-predict 16 --tag q$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
}
arm base
arm rot3 $R3
arm rot3hs $R3 -e RADIANCE_PQM_SKIP_HS=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P23_DONE
