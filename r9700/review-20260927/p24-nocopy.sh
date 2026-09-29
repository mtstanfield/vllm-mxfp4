#!/bin/bash
# GDN glue removal (local/patch_gdn_nocopy.py + radiance_paroquant GDN_NOCOPY, EMPTY_OUT on) on top of rot3, plus the
# stream-1 trace probe (RADIANCE_PQ_STREAM_DEBUG=1 prints at dynamo trace time: the radiance_paroquant.py edit forces a
# re-trace on boot 1). Arms: r3 (rot3 only, reference) and nc (rot3 + nocopy): greedy 8k/100k sha + decode, cold
# prefill, then sampled 12x600 (seed-0 deterministic, so acceptance must match exactly).
set -u
cd /mnt/user/appdata/llama-gemma31b
PROD='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv'
R3='-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-mr3.so -e RADIANCE_PQM_ROT3=1'
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L"
  ARM_ENV="$PROD $R3 $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
  grep -hE 'TRACE|DEBUG|gdn-nocopy' /tmp/p7-$L-boot1.log | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | sort | uniq -c | head -20
  docker logs vllm-exp 2>&1 | grep -E 'gdn-nocopy' | sort | uniq -c
  ARM_ENV="$PROD $R3 $*" RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
}
arm r3 -e RADIANCE_PQ_STREAM_DEBUG=1
RADIANCE_GDN_EMPTY_OUT=1 arm nc -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_GDN_EMPTY_OUT=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P24_DONE
