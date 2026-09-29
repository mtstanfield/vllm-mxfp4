#!/bin/bash
# Boot one rig config and keep the full container log: dbg-boot.sh <label> <log> [extra -e flags...]
# (R4D_KEY etc. from the environment). Prints the deploy result and the error lines.
set -u
cd /mnt/user/appdata/llama-gemma31b
L=$1; LOG=$2; shift 2
F="-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 $*"
docker stop vllm-qwen38 >/dev/null 2>&1
( for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -q vllm-exp && break; sleep 1; done
  timeout 500 docker logs -f vllm-exp > $LOG 2>&1 ) &
ARM_ENV="$F" NOBENCH=1 timeout 560 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold' | head -3
sleep 3
grep -E 'mtp_mxfp4|Error|error:|Traceback|Fault|Assertion' $LOG | grep -v 'registration failed' \
  | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //; s/^ERROR [0-9-]+ [0-9:]+ \[core.py:1374\] //' | cut -c1-220 | tail -16
