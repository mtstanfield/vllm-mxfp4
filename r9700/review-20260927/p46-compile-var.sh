#!/bin/bash
# The rot4 .so gives 100k greedy c2b0a3bf even with ROT4 off, although every producer entry point returns identical bytes
# on both builds (kcmp.py/kdiff.py: 80 entries, 0 differ). Suspect: the env change keys a NEW compile artifact, and a
# freshly compiled artifact can round differently. Control: the unchanged production env (split2 .so) plus a dummy
# RADIANCE_ variable -> fresh artifact, same kernels. 98b2b349 = the artifact is stable; anything else = compile variance.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
docker stop vllm-qwen38 >/dev/null 2>&1
echo "== r5dummy (load: $(cut -d' ' -f1-3 /proc/loadavg))"
ARM_ENV="$F -e RADIANCE_DUMMY_KEY=1" bash jobs/p7-vllm-arm.sh r5dummy 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=" | cut -c1-240
docker rm -f vllm-exp >/dev/null 2>&1
echo P46_DONE
