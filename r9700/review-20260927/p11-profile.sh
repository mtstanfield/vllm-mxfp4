#!/bin/bash
# Decode-step kernel profile of the production config on the rig: the server runs under rocprofv3's kernel trace (the
# tool's env passed into the container -- every dispatch, HIP-graph replays included), greedy decode at 8k and 100k
# context, graceful stop so the trace flushes, then bench/step_profile.py per decode run.
# Usage: p11-profile.sh <label> [extra container env]
set -u
cd /mnt/user/appdata/llama-gemma31b
L=$1; X=${2:-}
OUT=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/rocprof/$L
rm -rf $OUT; mkdir -p $OUT
RP=/opt/rocm/core-7.14/lib
PROF="-e LD_PRELOAD=$RP/rocprofiler-sdk/librocprofiler-sdk-tool.so:$RP/librocprofiler-sdk.so -e ROCPROFILER_LIBRARY_CTOR=1 \
-e ROCPROFILER_REGISTER_LIBRARY=$RP/librocprofiler-sdk.so.1.3.2 -e ROCP_TOOL_LIBRARIES=$RP/rocprofiler-sdk/librocprofiler-sdk-tool.so \
-e ROCPROF_KERNEL_TRACE=1 -e ROCPROF_OUTPUT_FORMAT=csv -e ROCPROF_OUTPUT_PATH=/cache/rocprof/$L -e ROCPROF_OUTPUT_FILE_NAME=%pid%"
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
docker stop vllm-qwen38 >/dev/null 2>&1
ARM_ENV="$VOC $X $PROF" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label w --depths 2000 --runs 1 --n-predict 32 --tag w1 >/dev/null 2>&1
sleep 3
for D in 8000 100000; do
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  R=$($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L --depths $D --runs 1 --n-predict 300 --tag pr$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=')
  echo "decode d=$D (traced) :: $R :: $($PY /repo/bench/vllm_accept_delta.py delta)"
  sleep 3
done
docker stop -t 300 vllm-exp >/dev/null
ls -la $OUT | head; du -sh $OUT
F=$(ls -S $OUT/*kernel_trace.csv 2>/dev/null | head -1)
[ -n "$F" ] && docker run --rm --entrypoint python3 -v $OUT:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro   ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake /b/step_profile.py /t/$(basename $F) 2>&1 | grep -v paroquant || echo "NO TRACE"
echo P11_DONE $L
