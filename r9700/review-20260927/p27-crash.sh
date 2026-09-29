#!/bin/bash
# Isolate the p26 fin boot fault (pq_ew_rot_tok3_mr<2,8,1,6,true,false> memory fault in the profile run): boot-only arms.
set -u
cd /mnt/user/appdata/llama-gemma31b
BASE='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=1
docker stop vllm-qwen38 >/dev/null 2>&1
boot() {
  local L=$1; shift
  (for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -q vllm-exp && break; sleep 1; done; docker logs -f vllm-exp > /tmp/p27-$L.log 2>&1) &
  ARM_ENV="$BASE $*" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED' | head -3
  sleep 2
  echo "   $L: $(grep -c 'Memory Fault' /tmp/p27-$L.log) memory faults; $(grep -oE 'kernel: [^(]*' /tmp/p27-$L.log | head -1)"
}
echo "== a: fin with the split build (= eo + dup flag)"
boot c27a -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split.so -e RADIANCE_PQM_SKIP_HS=1
echo "== b: fin (split2) without SKIP_HS"
boot c27b -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so
docker rm -f vllm-exp >/dev/null 2>&1
echo P27_DONE
