#!/bin/bash
# p9 confirmation. p9b: same boot is bit-deterministic; schedule-identical resumes were bit-equal except where the fill
# request itself resumed from a snapshot an EARLIER case left (a different step schedule). Here:
#   F8 = the D8 config again (R4D GDN, cached compile): cold vs p9b's D8a (two cached-compile boots) and the pairs with a
#        per-case salt (fill + cached share it, nothing carried over) -> expect every pair bit-equal
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p9; mkdir -p $R
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
B=http://192.168.88.89:1252
PY="docker exec llama-hip-dev python3"
PAIRS=12928:12929,12929:12930,32320:32321,32321:32322,72720:72721,72721:72722
echo "== F8"
CHUNK=8192 KV_MEM=6000000000 MAXLEN=98304 EXTRA="$MTP --mamba-ssm-cache-dtype float32" R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 \
  ARM_ENV="$VOC" NOBENCH=1 bash jobs/p7-vllm-arm.sh pF8 2>&1 | grep -E 'deployed|FAILED|cold compile'
grep -oE 'torch.compile took [0-9.]+ s' /tmp/p7-pF8-boot1.log | tr '\n' ' '; echo
$PY /repo/bench/cold_collect.py collect --base $B --out /repo/$R/cold-F8.json | tail -1
echo "-- two cached-compile boots (D8a vs F8)"; $PY /repo/bench/cold_collect.py compare /repo/$R/cold-D8a.json /repo/$R/cold-F8.json
echo "-- schedule-identical pairs, per-case salt"
$PY /repo/bench/prefix_equiv.py --base $B --pairs $PAIRS --case-salt
echo P9C_DONE
