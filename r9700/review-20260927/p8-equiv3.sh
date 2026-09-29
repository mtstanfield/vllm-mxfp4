#!/bin/bash
# Controls with the f16 attention legs (R4D_ATTN_FP8=0), which should not depend on how a prompt is chunked:
#  F0 = fp32 GDN state: prefix_equiv + lastpos collect (-> vs lastpos-0.json = fp16 state, mode 0: the fp16 state's cost)
#  H0 = fp16 GDN state (production profile): prefix_equiv
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p8
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252
run() {  # <label> <extra>
  EXTRA="$2" MAXLEN=131072 R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 ARM_ENV="$VOC" NOBENCH=1 bash jobs/p7-vllm-arm.sh $1 2>&1 | grep -E 'deployed|FAILED'
  docker exec vllm-exp env | grep R4D_ATTN_FP8
  BLK=$(docker logs vllm-exp 2>&1 | grep -oE 'attention block size to [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  echo "block=$BLK"
  docker exec llama-hip-dev python3 /repo/bench/prefix_equiv.py --base $B --block ${BLK:-848} 2>&1 | tail -19
}
echo "== F0 (fp32 state, f16 attention)"
run eqF0 "$MTP --mamba-ssm-cache-dtype float32"
docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py collect --base $B --out /repo/$R/lastpos-0f32.json 2>&1 | tail -1
docker exec llama-hip-dev python3 /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-0f32.json /repo/$R/lastpos-0.json 2>&1 | tail -9
echo "== H0 (fp16 state, f16 attention)"
run eqH0 ""
echo P8_EQUIV3_DONE
