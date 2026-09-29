#!/bin/bash
# Localize the prefix-cache discrepancy (docs/vllm-radiance-review-20260927.md, round 2). Most exact config throughout:
# f16 attention legs (R4D_ATTN_FP8=0) and fp32 GDN state. Four boots:
#   R8  = Radiance all-R4D GDN, 8192-token prefill steps: cold_collect + prefix_equiv --delta 1 (resume one token past)
#   R16 = same, 16384-token steps: cold_collect                       -> chunk sensitivity of the R4D path (R8 vs R16)
#   S8  = stock vLLM GDN (RADIANCE_GDN_SCAN_OFF=1), 8192: cold_collect + prefix_equiv --delta 300
#   S16 = stock, 16384: cold_collect                                  -> chunk sensitivity of the stock path (S8 vs S16)
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p9; mkdir -p $R
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252
PY="docker exec llama-hip-dev python3"
boot() {  # <label> <chunk> <extra arm env>
  CHUNK=$2 KV_MEM=6000000000 MAXLEN=98304 EXTRA="$MTP --mamba-ssm-cache-dtype float32" R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 \
    ARM_ENV="$VOC $3" NOBENCH=1 bash jobs/p7-vllm-arm.sh $1 2>&1 | grep -E 'deployed|FAILED'
  docker logs vllm-exp 2>&1 | grep -oE "max_num_batched_tokens': [0-9]+|attention block size to [0-9]+" | sort -u | tr '\n' ' '; echo
}
blk() { docker logs vllm-exp 2>&1 | grep -oE 'attention block size to [0-9]+' | tail -1 | grep -oE '[0-9]+$'; }

echo "== R8"; boot pR8 8192 ""
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-R8.json | tail -1
$PY /repo/bench/prefix_equiv.py --base $B --block $(blk) --delta 1 | tail -19
echo "== R16"; boot pR16 16384 ""
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-R16.json | tail -1
echo "== S8"; boot pS8 8192 "-e RADIANCE_GDN_SCAN_OFF=1"
docker logs vllm-exp 2>&1 | grep -c 'RADIANCE_GDN_SCAN_OFF'
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-S8.json | tail -1
$PY /repo/bench/prefix_equiv.py --base $B --block $(blk) --delta 300 | tail -19
echo "== S16"; boot pS16 16384 "-e RADIANCE_GDN_SCAN_OFF=1"
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-S16.json | tail -1
echo "== chunk sensitivity, R4D path";  $PY /repo/bench/cold_collect.py compare /repo/$R/cold-R8.json /repo/$R/cold-R16.json
echo "== chunk sensitivity, stock path"; $PY /repo/bench/cold_collect.py compare /repo/$R/cold-S8.json /repo/$R/cold-S16.json
echo "== R4D vs stock (8192)";            $PY /repo/bench/cold_collect.py compare /repo/$R/cold-S8.json /repo/$R/cold-R8.json
echo P9_DONE
