#!/bin/bash
# Precise decode-step A/B for the fusion candidates (production stays DOWN): torch-profiler median wall of 50 decode
# steps at 8k (step_profile.py) + kernels/step, for s1 / s1+qknr / s1+draft-fused / all three, and base (no stream 1).
set -u
cd /mnt/user/appdata/llama-gemma31b
P=/mnt/user/appdata/vllm-radiance/persist
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1 CACHE=$P/cache-029-paro-tp1s
docker stop vllm-qwen38 >/dev/null 2>&1
prof() {   # label, fp8 stream (0 = stream 1 live), extra -e
  local L=$1 FS=$2; shift 2
  local HD=$P/cache-029-paro-tp1s/tprof/$L; rm -rf $HD; mkdir -p $HD
  local PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  RADIANCE_FP8_STREAM=$FS EXTRA="$MTP $PROF" ARM_ENV="$F $*" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
  $G --label w --depths 2000 --n-predict 16 --tag w1 >/dev/null 2>&1
  $G --label fill --depths 8000 --n-predict 1 --tag p8000 >/dev/null 2>&1
  curl -s -X POST $B/start_profile >/dev/null
  echo "   $($G --label $L --depths 8000 --n-predict 150 --tag p8000 2>&1 | grep -oE 'decode= *[0-9.]+ t/s|sha=[0-9a-f]+' | tr '\n' ' ')"
  curl -s -X POST $B/stop_profile >/dev/null
  sleep 25
  local T=$(ls -tr $HD/rank0*.json* | tail -1)
  docker run --rm --entrypoint python3 -v $HD:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro \
    ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake /b/step_profile.py /t/$(basename $T) --marker _rejection_kernel --top 0 2>&1 \
    | grep -E "steps [0-9]+: wall|kernels/step" | head -2
}
prof s1 0
prof s1q 0 -e RADIANCE_LOCAL_FUSED_QKNR=1
prof s1d 0 -e RADIANCE_DRAFT_FUSED=1
prof s1qd 0 -e RADIANCE_LOCAL_FUSED_QKNR=1 -e RADIANCE_DRAFT_FUSED=1
docker rm -f vllm-exp >/dev/null 2>&1
echo P54_DONE
