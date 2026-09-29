#!/bin/bash
# State-logic checks on the candidate production config (libr4d rx9y, R4D_ATTN_FP8=15, 48k draft vocab):
#  A = MTP SPEC 4, B = no speculation. Each: bench/spec_equiv.py collect (greedy, top-20 per generated token, short +
#  long-context prompts) and bench/prefix_equiv.py (prefix-cache resume vs cold, L1 around the cache block).
#  Then spec_equiv compare B (ref) vs A.
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p8; mkdir -p $R
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
B=http://192.168.88.89:1252
for ARM in A B; do
  echo "== $ARM"
  if [ $ARM = A ]; then EX=''; else EX=' '; fi
  EXTRA="${EX:-}" R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=15 ARM_ENV="$VOC" NOBENCH=1 bash jobs/p7-vllm-arm.sh eq$ARM 2>&1 | grep -E 'deployed|FAILED'
  [ $ARM = B ] && unset EXTRA
  BLK=$(docker logs vllm-exp 2>&1 | grep -oE 'attention block size to [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  docker logs vllm-exp 2>&1 | grep -oE "speculative_config=[^,]*" | head -1
  echo "block=$BLK"
  docker exec llama-hip-dev python3 /repo/bench/spec_equiv.py collect --long --base $B --out /repo/$R/spec-$ARM.json 2>&1 | tail -9
  docker exec llama-hip-dev python3 /repo/bench/prefix_equiv.py --base $B --block ${BLK:-848} 2>&1 | tail -20
done
docker exec llama-hip-dev python3 /repo/bench/spec_equiv.py compare /repo/$R/spec-B.json /repo/$R/spec-A.json 2>&1 | tail -12
echo P8_EQUIV_DONE
