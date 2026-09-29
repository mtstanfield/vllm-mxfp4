#!/bin/bash
# Decode fusions, round 4 (production stays DOWN). All arms on stream 1 (host RADIANCE_FP8_STREAM=0, CACHE pinned):
#   s1q = + fused split+QK-RMSNorm+mRoPE+gate v2 (RADIANCE_LOCAL_FUSED_QKNR=1; the round constant is now a jit global)
#   s1d = + fused exact-set draft head (RADIANCE_DRAFT_FUSED=1; dh_check.py: identical set + logits)
# Reference s1 (p52): 8k step 34.6 ms, sampled 90.46 (.534/.416/.691), trace .5751 / 80.82 t/s, PPL 5.9699 / 2.3036.
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
P=/mnt/user/appdata/vllm-radiance/persist
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=0 CACHE=$P/cache-029-paro-tp1s
trace() { docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
  -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py http://192.168.88.89:1252 $1 24 6144 256" 2>&1 | grep TRACE; }
docker stop vllm-qwen38 >/dev/null 2>&1
full() {   # label, ppl(0/1), extra -e flags
  local L=$1 PP=$2; shift 2
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill" | cut -c1-230
  docker logs vllm-exp 2>&1 | grep -E "fused-qknr\] applied|Traceback|Error" | sort -u | head -4 | cut -c1-200
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  trace $L
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
  [ "$PP" = 1 ] && ARM_ENV="$F $*" bash jobs/p7-vllm-ppl.sh ppl-$L 2>&1 | grep -E "FINAL|FAILED"
}
full s1q 1 -e RADIANCE_LOCAL_FUSED_QKNR=1
full s1d 0 -e RADIANCE_DRAFT_FUSED=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P53_DONE
