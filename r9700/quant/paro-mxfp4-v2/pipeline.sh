#!/bin/bash
# pipeline.sh - the whole experiment, unattended, in order of cost. Every step appends to $WORK/results.jsonl and logs
# to $WORK/logs/; re-running skips finished steps (marker files), and the optimizer resumes per layer.
#   tmux new -s pq2 'bash /workspace/pq2/pipeline.sh 2>&1 | tee -a /workspace/pq2/logs/pipeline.log'
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH WORK=${WORK:-/workspace/pq2}
cd "$WORK"; mkdir -p logs done
TRAIN=${TRAIN:-512}          # optimizer windows of 2048 tokens (z-lab used 2048; we warm-start from their rotations)
EPOCHS=${EPOCHS:-"3 2"}      # stage 1 (angles + channel scales), stage 2 (weights)
OPT=/workspace/opt/Qwen3.8-27B-bf16

step() {   # step <name> <command...>
  local n=$1; shift
  [ -f "done/$n" ] && { echo "== $n: already done"; return 0; }
  echo "== $n: start $(date +%T)"
  if "$@" > "logs/$n.log" 2>&1; then touch "done/$n"; echo "== $n: ok $(date +%T)"; python3 summary.py || true; return 0
  else echo "== $n: FAILED (logs/$n.log)"; tail -20 "logs/$n.log"; return 1; fi
}

step selftest  python3 selftest.py                                             || exit 1
step ref       python3 run.py --make-ref                                       || exit 1
step redhat    python3 run.py --external /workspace/models/RedHat-Qwen3.8-27B-MXFP4 --name redhat_gptq_awq  # public control
step rtn       python3 run.py --name rtn --quant rtn --save-codes              # = the finalist, rebuilt (sanity)
step ss        python3 run.py --name ss --quant ss
step gptq_ss   python3 run.py --name gptq_ss --quant gptq_ss --save-codes      # traces only: wiki PPL regressed
# calibration from here on: 70% agent traces + 30% general text (wikitext-2 train), to keep side-chat quality
MIX=${MIX:-0.3}
step gptq_ss_mix  python3 run.py --name gptq_ss_mix --quant gptq_ss --calib-mix $MIX --save-codes
step optimize  env PQ2_MODE=search PQ2_MIX=$MIX python3 optimize_mxfp4.py --model /workspace/models/Qwen3.8-27B-bf16 \
                 --params "channel_scales:0.05,angles:0.05" "weight:1e-5" --epochs $EPOCHS \
                 --group-size 128 --n-bit 4 --num-rotations 8 \
                 --skipped-modules linear_attn.in_proj_a linear_attn.in_proj_b \
                 --datasets traces --val-dataset traces --train-size $TRAIN --validation-size 32 \
                 --batch-size 16 --seqlen 2048 --cache-shards 4 --output-dir /workspace/opt --resume --seed 0
if [ -f done/optimize ]; then            # rotation variants only on a finished optimizer run
  step mxrot_ss            python3 run.py --name mxrot_ss --quant ss --rot-dir $OPT
  step mxrot_gptq_ss_mix   python3 run.py --name mxrot_gptq_ss_mix --quant gptq_ss --rot-dir $OPT --calib-mix $MIX --save-codes
  step mxrot1_gptq_ss_mix  python3 run.py --name mxrot1_gptq_ss_mix --quant gptq_ss --rot-dir $OPT --no-tuned-weights \
                             --calib-mix $MIX --save-codes
else
  echo "== optimizer did not finish: skipping the mxrot_* variants"
fi
# the same variants with the server's per-token fp8 activations emulated (closest pod proxy for "as served")
for v in rtn gptq_ss gptq_ss_mix mxrot_gptq_ss_mix mxrot1_gptq_ss_mix; do
  [ -f "codes/$v/codes.safetensors" ] && step "${v}_a8" python3 run.py --from-codes $v --act-fp8
done
python3 summary.py

# ---- ship: best body -> MTP head re-fit against it -> package -> finalist-layout check
step pick  python3 pick.py || exit 1
V=$(cat WINNER)
step "mtp_$V"  python3 mtp_refit.py --codes "$V" --act-fp8 --train ${MTP_TRAIN:-1024} --epochs ${MTP_EPOCHS:-2} --mix $MIX
MTP_ARG=""; [ -f "mtp/$V/model-mtp.safetensors" ] && MTP_ARG="--mtp mtp/$V/model-mtp.safetensors"
step "package_$V"  python3 package.py --variant "$V" --out /workspace/models/pkg-$V
step "finish_$V"   python3 finish.py --pkg /workspace/models/pkg-$V --extras finalist-extras $MTP_ARG \
                     --out /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-$V
echo "== pipeline finished $(date +%T): /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2-$V"
python3 summary.py; tail -1 mtp_results.jsonl 2>/dev/null
