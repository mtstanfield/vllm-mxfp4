#!/bin/bash
# Re-test of the options shelved on 09-28 night after the piecewise-cache fix (patch_aot_envkey.py now keys
# torch_compile_cache/<hash> on the RADIANCE_* env too). Base = round-5 candidate (rx9z + MXFP4 drafter): greedy
# 8k/100k sha 57e7b168 / da44c0cc, sampled acceptance .521/.415/.682 (p35).
#   x1: + RADIANCE_PQM_ROT3_EW=1 + SKIP_HS + GDN zero_() skip (NOCOPY=1, NOCOPY_Z=0, EMPTY_OUT=1): must be exact
#   x2: x1 + GDN gate read in place (NOCOPY_Z=1): was NOT exact at 100k on 09-28 -- real or a stale-cache artifact?
# Each arm boots 3x (boot-fault check), x1 also sampled 12x600.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1'
X1='-e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1 S=$2; shift 2
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
  [ "$S" = 1 ] && RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  for r in 1 2; do
    ARM_ENV="$F $*" NOBENCH=1 bash jobs/p7-vllm-arm.sh ${L}b$r 2>&1 | grep -E 'deployed|FAILED' | head -1
  done
}
arm x1 1 $X1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0
arm x2 0 $X1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P37_DONE
