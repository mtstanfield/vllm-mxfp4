#!/bin/bash
# Rotation stream 1 actually in the graph (radiance_paroquant.py RADIANCE_PQ_STREAM_CLASSFWD=1: class-level dispatch;
# the stock instance-level forward override is ignored by torch.compile). Fresh compile-cache dir (the AOT key does not
# see the change): CACHE=persist/cache-029-paro-s1 (-> ...-s1-tp1s, seeded with triton + tunableop).
# Production settings otherwise. Checks: install log, stream-1 op in the compiled pieces, sampled 12x600, greedy 8k/100k
# (sha vs 55222d73 / 86cd9c47), prefill 8k/43k.
set -u
cd /mnt/user/appdata/llama-gemma31b
L=${1:-s1}
P=/mnt/user/appdata/vllm-radiance/persist
PROD='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
G="$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
CACHE=$P/cache-029-paro-s1 ARM_ENV="$PROD -e RADIANCE_PQ_STREAM_CLASSFWD=1 ${X:-}" RUNS=12 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E 'deployed|FAILED|cold compile|TOTAL|GRAND'
docker logs vllm-exp 2>&1 | grep -E 'class-level forward|rot stream installed' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | sort -u
H=$(docker logs vllm-exp 2>&1 | grep -oE 'torch_aot_compile/[0-9a-f]+' | sort -u | tr '\n' ' ')
for h in $H; do echo "$h: pieces with pqm_add_rms_rot = $(grep -rl pqm_add_rms_rot $P/cache-029-paro-s1-tp1s/vllm/torch_compile_cache/$h/inductor_cache 2>/dev/null | wc -l)"; done
for D in 8000 100000; do
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  echo "greedy d=$D dec :: $($G --label $L --depths $D --n-predict 600 --tag dec 2>&1 | grep -oE 'decode= *[0-9.]+ t/s|sha=[0-9a-f]+' | tr '\n' ' ') :: $($PY /repo/bench/vllm_accept_delta.py delta)"
done
$G --label $L --depths 8000,43000 --n-predict 16 --tag p$((RANDOM % 90 + 10)) 2>&1 | grep -oE 'depth=[0-9]+|prefill= *[0-9.]+ t/s' | tr '\n' ' '; echo
echo P22_DONE
