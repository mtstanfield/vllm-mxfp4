#!/bin/bash
# Production-config rig boot with (1) the fp8 KV census (RADIANCE_KV_STATS, printed during one 40k prefill) and
# (2) vLLM's torch profiler: each decode run's prompt is prefilled once (cache fill), then /start_profile, the same
# prompt again (prefix hit -> only a short tail prefill) + 150 greedy tokens, /stop_profile. Traces land under the
# persist cache dir; bench/step_profile.py reads them. p11's rocprofv3 route died on a tool abort loop at shutdown.
set -u
cd /mnt/user/appdata/llama-gemma31b
L=${1:-pf2}
HOSTDIR=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/tprof/$L
rm -rf $HOSTDIR; mkdir -p $HOSTDIR
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
B=http://192.168.88.89:1252
PY="docker exec llama-hip-dev python3"
G="$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
EXTRA="$MTP $PROF" ARM_ENV="$VOC -e RADIANCE_KV_STATS=16384" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
$G --label w --depths 2000 --n-predict 16 --tag w1 >/dev/null 2>&1
echo "== KV census (40k prefill)"
$G --label kv --depths 40000 --n-predict 1 --tag kv1 2>&1 | grep -oE 'prompt= *[0-9]+|prefill= *[0-9.]+ t/s' | tr '\n' ' '; echo
sleep 2
docker logs vllm-exp 2>&1 | grep 'radiance.kvstats' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-460
for D in 8000 100000; do
  $G --label fill --depths $D --n-predict 1 --tag p$D >/dev/null 2>&1
  curl -s -X POST $B/start_profile >/dev/null
  echo "decode d=$D (profiled) :: $($G --label $L --depths $D --n-predict 150 --tag p$D 2>&1 | grep -E 'prompt=')"
  curl -s -X POST $B/stop_profile >/dev/null
  sleep 20
done
ls -la $HOSTDIR
for F in $(ls -tr $HOSTDIR/*.json* 2>/dev/null); do
  echo "=== $(basename $F)"
  docker run --rm --entrypoint python3 -v $HOSTDIR:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro \
    ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake /b/step_profile.py /t/$(basename $F) --top 30 2>&1 | grep -v paroquant
done
echo P13_DONE
