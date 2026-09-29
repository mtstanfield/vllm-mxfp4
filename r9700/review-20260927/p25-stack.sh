#!/bin/bash
# Stacked decode/prefill levers, each arm on its OWN AOT artifact (local/patch_aot_envkey.py: the torch_aot_compile
# key includes the RADIANCE_* env, new dirs seeded from the base inductor cache -> unchanged pieces keep their kernels).
# Arms (prod settings + rot3):
#   hs  : + RADIANCE_PQM_SKIP_HS=1                      (prefill producers skip the unread bf16 HS)   byte-exact
#   sp  : hs + RADIANCE_PQM_SPLIT=1 (split-token decode producers, -split.so)                          byte-exact
#   eo/zv: sp + the two halves of the GDN no-copy (p24 nc changed the 100k sha): fill skip only / z view only (greedy only)
#   s1  : sp + stream 1 live (RADIANCE_FP8_STREAM=0: radiance_arnq no longer overwrites the layer forwards) NOT exact
# Each: greedy 8k/100k (sha vs 55222d73 / 86cd9c47), cold prefill 8k/43k/100k, sampled 12x600.
set -u
cd /mnt/user/appdata/llama-gemma31b
PROD='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1'
R3='-e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SKIP_HS=1'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
docker stop vllm-qwen38 >/dev/null 2>&1
arm() {
  local L=$1; shift
  echo "== $L  (host env: EMPTY_OUT=${RADIANCE_GDN_EMPTY_OUT:-unset} FP8_STREAM=${RADIANCE_FP8_STREAM:-unset})"
  ARM_ENV="$PROD $R3 $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
  grep -hE 'aot_envkey|Compiling model again|torch.compile took|gdn-nocopy|kernel override|fp8 stream installed' /tmp/p7-$L-boot1.log \
    | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-220 | sort | uniq -c
  docker logs vllm-exp 2>&1 | grep -E 'aot_envkey|Directly load|torch.compile took' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-200 | sort | uniq -c
  [ "${QUICK:-0}" = 1 ] && return
  $PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L --depths 100000 --runs 1 \
    --n-predict 16 --tag q$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
}
SO='-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split.so'
SP='-e RADIANCE_PQM_SPLIT=1'
export RADIANCE_GDN_EMPTY_OUT=0 RADIANCE_FP8_STREAM=1
arm hs $SO
arm sp $SO $SP
# bisect the p24 nc non-exactness (100k sha): EMPTY_OUT alone, z view alone (greedy only)
export RADIANCE_GDN_EMPTY_OUT=1
QUICK=1 arm eo $SO $SP -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0
export RADIANCE_GDN_EMPTY_OUT=0
QUICK=1 arm zv $SO $SP -e RADIANCE_LOCAL_GDN_NOCOPY=1
export RADIANCE_FP8_STREAM=0
arm s1 $SO $SP
docker rm -f vllm-exp >/dev/null 2>&1
echo P25_DONE
