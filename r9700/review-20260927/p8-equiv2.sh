#!/bin/bash
# Follow-up to p8-equiv.sh:
#  B  = no speculation (EXTRA '--speculative-config null'; the launcher's own spec config is overridden) -> spec_equiv vs A.
#  F  = fp32 GDN state with the rest of the single-GPU profile (EXTRA carries the MTP config + --mamba-ssm-cache-dtype
#       float32, which the launcher then leaves alone) -> prefix_equiv (hit-verified) and lastpos_kl vs the fp16-state
#       mode-15 run (bench/results/p8/lastpos-15.json).
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p8
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252

echo "== B (no speculation)"
EXTRA='--speculative-config null' R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=15 ARM_ENV="$VOC" NOBENCH=1 bash jobs/p7-vllm-arm.sh eqB2 2>&1 | grep -E 'deployed|FAILED'
docker logs vllm-exp 2>&1 | grep -oE "speculative_config=[^,]*" | head -1
docker exec llama-hip-dev python3 /repo/bench/spec_equiv.py collect --long --base $B --out /repo/$R/spec-B.json 2>&1 | tail -1
docker exec llama-hip-dev python3 /repo/bench/spec_equiv.py compare /repo/$R/spec-B.json /repo/$R/spec-A.json 2>&1 | tail -10

echo "== F (fp32 GDN state)"
EXTRA="$MTP --mamba-ssm-cache-dtype float32" MAXLEN=131072 R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=15 ARM_ENV="$VOC" NOBENCH=1 \
  bash jobs/p7-vllm-arm.sh eqF 2>&1 | grep -E 'deployed|FAILED'
docker logs vllm-exp 2>&1 | grep -E "mamba_ssm_dtype|mamba-ssm-cache-dtype|ssm_cache_dtype" | head -2 | cut -c1-200
BLK=$(docker logs vllm-exp 2>&1 | grep -oE 'attention block size to [0-9]+' | tail -1 | grep -oE '[0-9]+$')
echo "block=$BLK"
docker exec llama-hip-dev python3 /repo/bench/prefix_equiv.py --base $B --block ${BLK:-848} 2>&1 | tail -19
docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py collect --base $B --out /repo/$R/lastpos-15f32.json 2>&1 | tail -1
docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-15f32.json /repo/$R/lastpos-15.json 2>&1 | tail -9
echo P8_EQUIV2_DONE
