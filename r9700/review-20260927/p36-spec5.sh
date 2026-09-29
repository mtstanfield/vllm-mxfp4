#!/bin/bash
# With the MXFP4 drafter each draft pass is ~27% cheaper, so the best speculation depth may move from 4 to 5
# (round 3, fp8 drafter: SPEC 3 80.8 / SPEC 4 84.5 / SPEC 5 83.7 sampled). Round-5 candidate env + SPEC 5, sampled 12x600.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1'
MTP5='--speculative-config {"method":"mtp","num_speculative_tokens":5,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
docker stop vllm-qwen38 >/dev/null 2>&1
echo "== spec5 (load: $(cut -d' ' -f1-3 /proc/loadavg))"
R4D_KEY=b9e42ab-rx9z EXTRA="$MTP5" ARM_ENV="$F" NOBENCH=1 bash jobs/p7-vllm-arm.sh spec5 2>&1 | grep -E 'deployed|FAILED'
RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh spec5 2>&1 | grep -E "TOTAL|GRAND"
echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
docker rm -f vllm-exp >/dev/null 2>&1
echo P36_DONE
