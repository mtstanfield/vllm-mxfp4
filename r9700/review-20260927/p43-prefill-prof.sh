#!/bin/bash
# (p43 = p21 on the round-5 production env) Prefill kernel profile, production settings (+ torch profiler): one fresh 16k-token prompt (2 steps of 8192) and one
# fresh 100k-token prompt (13 steps, attention grows with depth), each alone in its own profiler window, max_tokens 1.
set -u
cd /mnt/user/appdata/llama-gemma31b
L=${1:-pp5}
HOSTDIR=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/tprof/$L; rm -rf $HOSTDIR; mkdir -p $HOSTDIR
PROD="-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1"
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
B=http://192.168.88.89:1252
docker stop vllm-qwen38 >/dev/null 2>&1
EXTRA="$MTP $PROF" ARM_ENV="$PROD ${X:-}" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile'
docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1 --label w --depths 2000 --n-predict 8 --tag w1 >/dev/null 2>&1
for N in 16384 100000; do
  curl -s -X POST $B/start_profile >/dev/null
  docker exec llama-hip-dev python3 /repo/bench/kvusage_probe.py --base $B --tokens $N --every 1000 2>&1 | grep RESULT
  curl -s -X POST $B/stop_profile >/dev/null
  sleep 25
done
ls -la $HOSTDIR | grep rank0
echo P43_DONE
