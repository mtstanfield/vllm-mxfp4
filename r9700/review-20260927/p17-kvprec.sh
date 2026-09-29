#!/bin/bash
# fp8 vs bf16 KV cache. Census (p15): K/V sit in e4m3's normal range, round trip 2.65% RMS per element at any scale
# (mantissa-bound; calibration buys nothing). What does that cost end to end?
#   served PPL (wiki 60 / code 30 x 2048): kP = prod (fp8 KV, R4D_ATTN_FP8=15), k0 = fp8 KV + f16 legs (mode 0),
#   kB = bf16 KV (--kv-cache-dtype auto; the fp8 legs do not apply)
#   long context: lastpos_kl collect with bf16 KV (MAXLEN 131072), compared with p8/p9d's fp8 references (0b, 15)
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
PY="docker exec llama-hip-dev python3"
R=bench/results/p8
docker stop vllm-qwen38 >/dev/null 2>&1
echo "== PPL"
KV_MEM=6000000000 MAXLEN=32768 R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=15 ARM_ENV="$VOC" bash jobs/p7-vllm-ppl.sh kP 2>&1 | grep -E 'deployed|FAILED|PPL|ppl'
KV_MEM=6000000000 MAXLEN=32768 R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 ARM_ENV="$VOC" bash jobs/p7-vllm-ppl.sh k0 2>&1 | grep -E 'deployed|FAILED|PPL|ppl'
KV_MEM=6000000000 MAXLEN=32768 R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 EXTRA="$MTP --kv-cache-dtype auto" ARM_ENV="$VOC" \
  bash jobs/p7-vllm-ppl.sh kB 2>&1 | grep -E 'deployed|FAILED|PPL|ppl'
echo "== long context (bf16 KV)"
R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 MAXLEN=131072 EXTRA="$MTP --kv-cache-dtype auto" ARM_ENV="$VOC" NOBENCH=1 \
  bash jobs/p7-vllm-arm.sh kBL 2>&1 | grep -E 'deployed|FAILED|cold compile'
docker logs vllm-exp 2>&1 | grep -oE "kv_cache_dtype[^,]*" | sort -u | head -2
$PY /repo/bench/lastpos_kl.py collect --base http://192.168.88.89:1252 --out /repo/$R/lastpos-bf16kv.json 2>&1 | tail -1
for M in 0b 15 3; do
  echo "-- bf16 KV vs fp8 KV mode $M"; $PY /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-bf16kv.json /repo/$R/lastpos-$M.json 2>&1 | tail -9
done
echo P17_DONE
