#!/bin/bash
# Sampled (production sampling) decode of one vLLM config on the rig: deploy (p7-vllm-arm.sh, NOBENCH), one warm-up
# request, then bench/spec_sampled.py -- code / prose / json prompts, RUNS runs x 600 tokens, totals + acceptance.
# Usage: [env as for p7-vllm-arm.sh] [RUNS=8] [NODEPLOY=1] p7-vllm-sampled.sh <label>
set -u
cd /mnt/user/appdata/llama-gemma31b
L=$1
[ "${NODEPLOY:-0}" = 1 ] || { NOBENCH=1 bash jobs/p7-vllm-arm.sh "$L" || exit 1; }   # NODEPLOY=1: bench the running vllm-exp
B=http://192.168.88.89:1252
docker exec llama-hip-dev python3 /repo/bench/spec_sampled.py --base $B --prompts code --runs 1 --n 64 --label warm >/dev/null 2>&1
docker exec llama-hip-dev python3 /repo/bench/spec_sampled.py --base $B --runs ${RUNS:-8} --n 600 --metrics $B/metrics \
  --label "$L" 2>&1 | grep -E 'TOTAL|GRAND'
echo P7_SAMPLED_DONE $L
