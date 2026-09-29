#!/bin/bash
# Does the fused QK-norm/mRoPE kernel really cost drafter acceptance? (production stays DOWN) All arms = stream 1 +
# fused draft head; held-out omp windows cut at 2048/4096/6144 (72 prompts x 256 sampled tokens) + sampled 12x600.
#   s1d    no qknr (profiled step 36.47 ms)       s1qd   qknr everywhere (35.91 ms)
#   s1qtd  qknr on the target's 16 attention layers only; the MTP drafter's attention stays unfused
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
P=/mnt/user/appdata/vllm-radiance/persist
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1 -e RADIANCE_DRAFT_FUSED=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=0 CACHE=$P/cache-029-paro-tp1s
trace() { docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
  -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py http://192.168.88.89:1252 $1 24 6144 256 2048,4096,6144" 2>&1 | grep TRACE; }
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in "s1d:" "s1qd:-e RADIANCE_LOCAL_FUSED_QKNR=1" "s1qtd:-e RADIANCE_LOCAL_FUSED_QKNR=1 -e RADIANCE_LOCAL_FUSED_QKNR_TARGET_ONLY=1"; do
  L=${arm%%:*}; X=${arm#*:}
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $X" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile"
  trace $L
  [ "$L" = s1d ] || RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P55_DONE
