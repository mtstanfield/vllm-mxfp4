#!/bin/bash
# Round-5 candidate = round-4 prod + libr4d rx9z (bit-exact prefill attention) + MXFP4 drafter (RADIANCE_MTP_MXFP4=1).
# Greedy 8k/100k, cold prefill 8k/43k/100k, sampled 12x600, then two plain reboots (boot stability; the fp8/MXFP4 mixed
# drafter configs hit an inductor input assertion). Host load recorded.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1'
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
echo "== r5 (load: $(cut -d' ' -f1-3 /proc/loadavg))"
R4D_KEY=b9e42ab-rx9z ARM_ENV="$F" bash jobs/p7-vllm-arm.sh r5 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
docker logs vllm-exp 2>&1 | grep -cE 'radiance.mtp_mxfp4\]' | sed 's/^/   drafter layers in MXFP4: /'
$G --label r5 --depths 8000,43000,100000 --n-predict 16 --tag r$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh r5 2>&1 | grep -E "TOTAL|GRAND"
echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
for r in 1 2; do
  R4D_KEY=b9e42ab-rx9z ARM_ENV="$F" NOBENCH=1 bash jobs/p7-vllm-arm.sh r5b$r 2>&1 | grep -E 'deployed|FAILED' | head -2
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P35_DONE
