#!/bin/bash
# In-engine TunableOp diagnostic (radiance_drafthead.py RADIANCE_TUNABLE_DIAG): is the tuned fp8 table live inside the
# vLLM engine process, and what does the MTP down shape run at there?
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
TUN='-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/mtp%d.csv'
docker stop vllm-qwen38 >/dev/null 2>&1
ARM_ENV="$VOC $TUN -e RADIANCE_TUNABLE_DIAG=1" NOBENCH=1 bash jobs/p7-vllm-arm.sh td 2>&1 | grep -E 'deployed|FAILED'
docker logs vllm-exp 2>&1 | grep -E 'tunablediag|reading tuning' | cut -c1-260
echo P16_DONE
