#!/bin/bash
# Decode-step profile of the round-4 production config (for the write-up): torch profiler around a prefix-cached 150-token
# greedy decode at 8k and 100k; bench/step_profile.py per-step breakdown (V2 marker _rejection_kernel). Then restore prod.
set -u
cd /mnt/user/appdata/llama-gemma31b
L=${1:-r4prof}
HOSTDIR=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/tprof/$L
rm -rf $HOSTDIR; mkdir -p $HOSTDIR
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
export RADIANCE_GDN_EMPTY_OUT=0 RADIANCE_FP8_STREAM=1
docker stop vllm-qwen38 >/dev/null 2>&1
EXTRA="$MTP $PROF" ARM_ENV="$F" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
$G --label w --depths 2000 --n-predict 16 --tag w1 >/dev/null 2>&1
for D in 8000 100000; do
  $G --label fill --depths $D --n-predict 1 --tag p$D >/dev/null 2>&1
  curl -s -X POST $B/start_profile >/dev/null
  echo "decode d=$D (profiled) :: $($G --label $L --depths $D --n-predict 150 --tag p$D 2>&1 | grep -oE 'decode= *[0-9.]+ t/s')"
  curl -s -X POST $B/stop_profile >/dev/null
  sleep 25
done
for T in $(ls -tr $HOSTDIR/rank0*.json* 2>/dev/null); do
  echo "=== $(basename $T)"
  docker run --rm --entrypoint python3 -v $HOSTDIR:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro \
    ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake /b/step_profile.py /t/$(basename $T) --marker _rejection_kernel --top 22 2>&1 \
    | grep -vE 'paroquant\]|registration failed'
done
docker rm -f vllm-exp >/dev/null 2>&1
echo "== restore production"
unset RADIANCE_GDN_EMPTY_OUT RADIANCE_FP8_STREAM
bash deploy-vllm-qwen38-paro.sh 2>&1 | tail -1 | grep -oE 'READY-VLLM|kv_tokens=[0-9]+' | tr '\n' ' '; echo
P=http://192.168.88.89:1246
GP="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $P --model qwen38-27b --runs 1"
$GP --label pw --depths 2000 --n-predict 64 --tag w1 >/dev/null 2>&1
echo "prod d=8000 :: $($GP --label prodr4 --depths 8000 --n-predict 600 --tag dec 2>&1 | grep -oE 'decode= *[0-9.]+ t/s|sha=[0-9a-f]+' | tr '\n' ' ')"
docker logs vllm-qwen38 2>&1 | grep -c 'Memory Fault'
echo P31_DONE
