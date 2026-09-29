#!/bin/bash
# MTP drafter linears in MXFP4 (RADIANCE_MTP_MXFP4=1, local/patch_paroquant_install.py) vs the fp8 drafter, round-4
# production env otherwise. Drafts only: greedy text must stay identical, acceptance may move. Per arm: load-time
# requantization report, KV pool, greedy 8k/100k (tok/step, acceptance, sha), sampled 12x600 with host load noted.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1'
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L  (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d='
  docker logs vllm-exp 2>&1 | grep -E 'radiance.mtp_mxfp4|Traceback|Error' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-200 | sort -u | head -12
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
}
arm fp8mtp
arm mxmtp -e RADIANCE_MTP_MXFP4=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P33_DONE
