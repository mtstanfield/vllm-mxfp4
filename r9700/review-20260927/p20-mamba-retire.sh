#!/bin/bash
# vLLM #55450 (align-mode mamba states pinned across null gaps) on our 0.29 serve, production settings on the rig:
#   mL = as shipped, mF = + local/patch_mamba_retire.py (RADIANCE_LOCAL_MAMBA_RETIRE=1).
# bench/kvusage_probe.py: KV-pool usage through one 200k and one 255k-token prefill (MAXLEN 262144, pool 273k tokens).
# mF also: schedule-identical prefix pairs (must stay 12/12 bit-equal) and greedy dec 8k (sha 55222d73 expected).
set -u
cd /mnt/user/appdata/llama-gemma31b
B=http://192.168.88.89:1252
PY="docker exec llama-hip-dev python3"
PROD='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv'
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in mL mF; do
  X=""; [ $arm = mF ] && X="-e RADIANCE_LOCAL_MAMBA_RETIRE=1"
  echo "== $arm"
  ARM_ENV="$PROD $X" NOBENCH=1 bash jobs/p7-vllm-arm.sh $arm 2>&1 | grep -E 'deployed|FAILED|cold compile'
  docker logs vllm-exp 2>&1 | grep -E 'mamba-retire' | sort -u
  for N in 200000 255000; do $PY /repo/bench/kvusage_probe.py --base $B --tokens $N --every 20 2>&1 | tail -12; done
  docker logs vllm-exp 2>&1 | grep -iE 'preempt|Traceback|Error' | grep -v 'kvstats' | tail -3 | cut -c1-200
  if [ $arm = mF ]; then
    $PY /repo/bench/prefix_equiv.py --base $B --pairs 12928:12929,12929:12930,32320:32321,32321:32322,72720:72721,72721:72722 --case-salt 2>&1 | tail -1
    echo "greedy 8k dec :: $($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1 --label mF --depths 8000 --n-predict 600 --tag dec 2>&1 | grep -oE 'decode= *[0-9.]+ t/s|sha=[0-9a-f]+' | tr '\n' ' ')"
  fi
done
echo P20_DONE
