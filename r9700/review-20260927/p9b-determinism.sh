#!/bin/bash
# p9 follow-up. Scheduler analysis (vllm/v1/core/sched/scheduler.py _mamba_block_aligned_split, use_eagle backs the
# cache hit off one block): with 1616-token blocks and 8192-token steps, resuming L2 = k*b+1 after filling L1 = k*b
# (or L2 = k*b+2 after k*b+1) runs EXACTLY the cold request's own last prefill step from a snapshot produced by the
# same steps -> must be bit-equal if the engine is deterministic and the snapshot/restore is exact. p9 R8 measured
# nonzero there (code 12928->12929 KL 0.147). Here: is the engine even run-to-run deterministic?
#   D8 = the R8 config: cold_collect twice (same boot) + vs p9's cold-R8 (other boot); schedule-identical prefix pairs
#        with a second cold run as the noise floor.
#   E8 = stock GDN (RADIANCE_GDN_SCAN_OFF=1): cold_collect twice -> is nondeterminism in the R4D GDN path or elsewhere
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p9; mkdir -p $R
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252
PY="docker exec llama-hip-dev python3"
PAIRS=12928:12929,12929:12930,32320:32321,32321:32322,72720:72721,72721:72722
boot() {  # <label> <chunk> <extra arm env>
  CHUNK=$2 KV_MEM=6000000000 MAXLEN=98304 EXTRA="$MTP --mamba-ssm-cache-dtype float32" R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 \
    ARM_ENV="$VOC $3" NOBENCH=1 bash jobs/p7-vllm-arm.sh $1 2>&1 | grep -E 'deployed|FAILED'
  docker logs vllm-exp 2>&1 | grep -oE "max_num_batched_tokens': [0-9]+|attention block size to [0-9]+" | sort -u | tr '\n' ' '; echo
}
echo "== D8"; boot pD8 8192 ""
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-D8a.json | tail -1
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-D8b.json | tail -1
echo "-- same boot";  $PY /repo/bench/cold_collect.py compare /repo/$R/cold-D8a.json /repo/$R/cold-D8b.json
echo "-- other boot"; $PY /repo/bench/cold_collect.py compare /repo/$R/cold-R8.json /repo/$R/cold-D8a.json | tail -1
echo "-- schedule-identical prefix pairs"
$PY /repo/bench/prefix_equiv.py --base $B --pairs $PAIRS --cold2
echo "== E8"; boot pE8 8192 "-e RADIANCE_GDN_SCAN_OFF=1"
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-E8a.json | tail -1
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-E8b.json | tail -1
echo "-- same boot";  $PY /repo/bench/cold_collect.py compare /repo/$R/cold-E8a.json /repo/$R/cold-E8b.json
echo "-- other boot"; $PY /repo/bench/cold_collect.py compare /repo/$R/cold-S8.json /repo/$R/cold-E8a.json | tail -1
echo P9B_DONE
