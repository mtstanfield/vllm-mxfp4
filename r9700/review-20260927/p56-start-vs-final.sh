#!/bin/bash
# Start vs finalist, same harness, same host (production stays DOWN):
#   start = deploy-vllm-qwen38-paro.start.sh (production as of the review's start, 2026-09-27 morning)
#   final = deploy-vllm-qwen38-paro.sh (round 5b + stream 1 + fused draft head)
# Per config: greedy 8k/100k + prefill probes (p7 arm), prefill 8k/43k/100k x2, sampled 12x600, 72 held-out omp prompts,
# torch-profiler median decode step at 8k and 100k. Tag = power state (e.g. cap250 / cap300).
set -u
cd /mnt/user/appdata/llama-gemma31b
TAG=${1:-cap250}
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
P=/mnt/user/appdata/vllm-radiance/persist
HW=$(dirname $(ls /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap))
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
echo "== power cap $(( $(cat $HW/power1_cap) / 1000000 )) W"
docker stop vllm-qwen38 >/dev/null 2>&1
unset R4D_KEY R4D_ATTN_FP8 RADIANCE_GDN_EMPTY_OUT RADIANCE_FP8_STREAM CACHE FAST_DRAFT
for cfg in start final; do
  L=$cfg-$TAG
  HD=$P/cache-029-paro-tp1s/tprof/$L; rm -rf $HD; mkdir -p $HD
  PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
  if [ $cfg = start ]; then LA=deploy-vllm-qwen38-paro.start.sh; AE="-e RADIANCE_LOCAL_AOT_ENVKEY=1"; else LA=deploy-vllm-qwen38-paro.sh; AE=""; fi
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  # final: the launcher's own EXTRA_ENV (R4_ENV etc.); p7 would override it with BASE_ENV + ARM_ENV, so hand it over
  if [ $cfg = final ]; then AE="$(bash -c 'source <(grep -E "^(P=|export RADIANCE_FP8_STREAM|export RADIANCE_GDN_EMPTY_OUT|TUNABLE_ENV=|R4_ENV=)" deploy-vllm-qwen38-paro.sh); echo "-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 $TUNABLE_ENV $R4_ENV"')"; fi
  LAUNCHER=$LA EXTRA="$MTP $PROF" ARM_ENV="$AE" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill" | cut -c1-230
  docker logs vllm-exp 2>&1 | grep -oE "R4D_KEY=[^ ]+|libr4d [^ ]+ ->|draft vocab armed[^,]*|DRAFT_FUSED|kernel override [^ ]+|fp8 stream[^,]*" | sort -u | head -6
  for i in 1 2; do $G --label $L --depths 8000,43000,100000 --n-predict 16 --tag s$i$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed "s/^/prefill$i /" | cut -c1-120; done
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
    -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py $B $L 24 6144 256 2048,4096,6144" 2>&1 | grep TRACE
  for D in 8000 100000; do
    $G --label fill --depths $D --n-predict 1 --tag p$D >/dev/null 2>&1
    curl -s -X POST $B/start_profile >/dev/null
    $G --label $L --depths $D --n-predict 150 --tag p$D >/dev/null 2>&1
    curl -s -X POST $B/stop_profile >/dev/null
    sleep 25
    T=$(ls -tr $HD/rank0*.json* | tail -1)
    docker run --rm --entrypoint python3 -v $HD:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro $IMG /b/step_profile.py /t/$(basename $T) \
      --marker _rejection_kernel --top 0 2>&1 | grep -E "steps [0-9]+: wall" | head -1 | sed "s/^/step d=$D /"
  done
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P56_DONE
