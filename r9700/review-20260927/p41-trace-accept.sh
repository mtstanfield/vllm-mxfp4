#!/bin/bash
# Drafter acceptance on the user's own workload (held-out omp-session windows, traces.npz eval8k; trace_accept.py):
# fp8 drafter vs MXFP4 RTN vs MXFP4 GPTQ (p39, calibrated on the traces' calib split), all on the round-5 + x1 env.
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
V2=/mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2
S=/mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $*" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
  docker run --rm --network host --entrypoint bash -v $V2:/pqv2 -v $S:/s $IMG \
    -lc "python3 /s/trace_accept.py http://192.168.88.89:1252 $L ${NWIN:-24} 6144 256" 2>&1 | grep -v registration
}
arm fp8mtp
arm mxrtn -e RADIANCE_MTP_MXFP4=1
arm mxgptq -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_hc/mtp_gptq_a0.5.pt
docker rm -f vllm-exp >/dev/null 2>&1
echo P41_DONE
