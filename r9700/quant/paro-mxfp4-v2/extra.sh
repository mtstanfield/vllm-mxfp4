#!/bin/bash
# extra.sh - queued after pipeline.sh: GPTQ static act-order variant (same mixed calibration), its fp8-activation
# re-score, then winner re-selection; the MTP refit + package + finish run again only if the winner changed.
#   tmux new -d -s extra 'bash /workspace/pq2/extra.sh 2>&1 | tee -a /workspace/pq2/logs/extra.log'
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH WORK=${WORK:-/workspace/pq2}
cd "$WORK"; mkdir -p logs done
MIX=${MIX:-0.3}
echo "== extra: waiting for pipeline.sh to finish ($(date +%T))"
until grep -q "pipeline finished" logs/pipeline.log || ! tmux has-session -t pq2 2>/dev/null; do sleep 60; done
grep -q "pipeline finished" logs/pipeline.log || { echo "== extra: pipeline.sh ended WITHOUT finishing - not running"; exit 1; }
OLD=$(cat WINNER)

step() {
  local n=$1; shift
  [ -f "done/$n" ] && { echo "== $n: already done"; return 0; }
  echo "== $n: start $(date +%T)"
  if "$@" > "logs/$n.log" 2>&1; then touch "done/$n"; echo "== $n: ok $(date +%T)"; python3 summary.py || true; return 0
  else echo "== $n: FAILED (logs/$n.log)"; tail -20 "logs/$n.log"; return 1; fi
}

# act-order on the best configuration so far: trained rotations, base weights, GPTQ + scale search, mixed calibration
OPT=/workspace/opt/Qwen3.8-27B-bf16
step mxrot1_gptq_ss_mix_ao     python3 run.py --name mxrot1_gptq_ss_mix_ao --quant gptq_ss --rot-dir $OPT --no-tuned-weights \
                                 --calib-mix $MIX --actorder --save-codes || exit 1
step mxrot1_gptq_ss_mix_ao_a8  python3 run.py --from-codes mxrot1_gptq_ss_mix_ao --act-fp8 || exit 1
rm -f done/pick; step pick python3 pick.py || exit 1
V=$(cat WINNER)
if [ "$V" = "$OLD" ]; then
  echo "== extra finished $(date +%T): winner unchanged ($V); the pipeline's checkpoint stands"
  exit 0
fi
echo "== extra: new winner $V (was $OLD)"
step "mtp_$V"  python3 mtp_refit.py --codes "$V" --act-fp8 --train ${MTP_TRAIN:-1024} --epochs ${MTP_EPOCHS:-2} --mix $MIX
MTP_ARG=""; [ -f "mtp/$V/model-mtp.safetensors" ] && MTP_ARG="--mtp mtp/$V/model-mtp.safetensors"
step "package_$V"  python3 package.py --variant "$V" --out /workspace/models/pkg-$V
step "finish_$V"   python3 finish.py --pkg /workspace/models/pkg-$V --extras finalist-extras $MTP_ARG \
                     --out /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-$V
echo "== extra finished $(date +%T): /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-$V"
python3 summary.py; tail -1 mtp_results.jsonl 2>/dev/null
