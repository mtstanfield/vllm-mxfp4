#!/bin/bash
# SPEC depth with the tuned fp8 table (drafting is now ~0.45 ms/pass cheaper, so the SPEC 4 optimum may have moved).
# Arms: SPEC 3 and SPEC 5 (SPEC 4 + table = p14 tB: sampled 84.5, greedy 8k 121.3 / 100k 62.0). Sampled 12 x 600,
# greedy dec 8k / 100k.
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
TUN='-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv'
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
G="$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
for S in 3 5; do
  MTP="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$S,\"attention_backend\":\"R4D\",\"disable_padded_drafter_batch\":true,\"draft_sample_method\":\"probabilistic\"}"
  echo "== SPEC $S"
  EXTRA="$MTP" ARM_ENV="$VOC $TUN" RUNS=12 bash jobs/p7-vllm-sampled.sh s$S 2>&1 | grep -E 'deployed|FAILED|cold compile|TOTAL|GRAND'
  for D in 8000 100000; do
    $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
    echo "greedy d=$D dec :: $($G --label s$S --depths $D --n-predict 600 --tag dec 2>&1 | grep -oE 'decode= *[0-9.]+ t/s|sha=[0-9a-f]+' | tr '\n' ' ') :: $($PY /repo/bench/vllm_accept_delta.py delta)"
  done
done
echo P19_DONE
