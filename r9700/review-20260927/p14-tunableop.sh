#!/bin/bash
# hipBLASLt's heuristic picks slow solutions for the MTP drafter's skinny fp8 GEMMs (profile p13: ~1.3 ms of GEMM per
# draft forward at 230-430 GB/s; fp8mm_bench.py: TunableOp finds 2-3.5x faster solutions for fc/qkv/o_proj/down).
# Serve with the tuned table READ-ONLY (TUNING=0: shapes not in the table keep the default) and record untuned keys to
# prove the table is actually hit. Greedy dec 8k/100k (compare p12 eA 114.7 t/s @8k), sampled 12x600 (compare p10 dA
# 80.4), then a torch-profiled 8k decode (compare p13).
set -u
cd /mnt/user/appdata/llama-gemma31b
L=${1:-tA}
T=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/tunableop
rm -f $T/untuned*.csv
HOSTDIR=/mnt/user/appdata/vllm-radiance/persist/cache-029-paro-tp1s/tprof/$L; rm -rf $HOSTDIR; mkdir -p $HOSTDIR
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
TUN="-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/${TABLE:-skinny}%d.csv"
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
PROF="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/tprof/$L\",\"torch_profiler_with_stack\":false}"
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
G="$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
EXTRA="$MTP $PROF" ARM_ENV="$VOC $TUN ${X:-}" RUNS=12 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|TOTAL|GRAND'
for D in 8000 100000; do
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  echo "greedy d=$D dec :: $($G --label $L --depths $D --n-predict 600 --tag dec 2>&1 | grep -E 'prompt=') :: $($PY /repo/bench/vllm_accept_delta.py delta)"
done
$G --label fill --depths 8000 --n-predict 1 --tag p8000 >/dev/null 2>&1
curl -s -X POST $B/start_profile >/dev/null
echo "profiled 8k :: $($G --label $L --depths 8000 --n-predict 150 --tag p8000 2>&1 | grep -E 'prompt=')"
curl -s -X POST $B/stop_profile >/dev/null
sleep 20
echo "== untuned GEMM keys the serve looked up (skinny fp8 = M<=10):"
cat $T/untuned*.csv 2>/dev/null | grep -E 'ScaledGemm' | grep -E '_(1|2|5|10)_[0-9]+_ld' | cut -c1-160 | head -20
echo "   (all untuned keys: $(cat $T/untuned*.csv 2>/dev/null | wc -l))"
F=$(ls -t $HOSTDIR/rank0.*.json.gz 2>/dev/null | head -1)
[ -n "$F" ] && docker run --rm --entrypoint python3 -v $HOSTDIR:/t:ro -v /mnt/user/appdata/llama-gemma31b/bench:/b:ro \
  ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake /b/step_profile.py /t/$(basename $F) --marker '^_rejection_kernel$' --top 14 2>&1 | grep -v paroquant | cut -c1-190
echo P14_DONE
