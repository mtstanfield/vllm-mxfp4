#!/bin/bash
# MXFP4 drafter sensitivity: which drafter projections cost acceptance. Arms keep some groups fp8
# (RADIANCE_MTP_MXFP4_SKIP): attn = qkv/o fp8; fcattn = fc + qkv/o fp8 (MXFP4 on the MLP only). Round-4 prod env.
# Reference from p33 (same env, 10:30-10:50): fp8 drafter 85.87 (84.39/73.54/105.40, accept .531/.430/.725),
# all-MXFP4 88.35 (87.82/75.95/106.37, accept .521/.415/.682).
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1'
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L  (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $*" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED'
  docker logs vllm-exp 2>&1 | grep -c 'radiance.mtp_mxfp4\]' | sed 's/^/   layers in MXFP4: /'
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
}
arm attn -e RADIANCE_MTP_MXFP4_SKIP=self_attn
arm fcattn -e RADIANCE_MTP_MXFP4_SKIP=fc,self_attn
docker rm -f vllm-exp >/dev/null 2>&1
echo P34_DONE
