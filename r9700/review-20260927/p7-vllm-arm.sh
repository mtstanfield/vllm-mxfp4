#!/bin/bash
# One vLLM experiment arm on the rig (container vllm-exp, :1252, deploy-vllm-exp.sh via the finalist launcher
# deploy-vllm-qwen38-paro.sh): deploy with the caller's env, redeploy once if torch.compile ran (a cold-compiled engine
# measures slow), then bench: greedy MTP decode at 8k and 100k (fixed prompt, 600 tokens, tok/step + acceptance from
# /metrics) and cold prefill at 8k / 43k (random short tags).
# Usage: [env...] p7-vllm-arm.sh <label>    Env: anything the launcher reads (FAST_DRAFT, SPEC via EXTRA, MAXLEN, R4D_KEY,
#        SINGLE_GPU_PROFILE, ...), ARM_ENV = extra container -e flags appended to the finalist's EXTRA_ENV, NOBENCH=1
set -u
cd /mnt/user/appdata/llama-gemma31b
L=$1
export WRAPPER=deploy-vllm-exp.sh NAME=vllm-exp PORT=1252
BASE_ENV='-e PYTHONHASHSEED=0 -e RADIANCE_PAROQUANT_INSTALL=1 -e RADIANCE_PAROQUANT=1 -e RADIANCE_PQ_I8=1 -e RADIANCE_PQ_PG=1 -e RADIANCE_PQ_ZPE=1 -e RADIANCE_PQ_ROT_STREAM=1 -e RADIANCE_PQ_ROT_STREAM2=1 -e RADIANCE_GDN_LAZY=0'
export EXTRA_ENV="$BASE_ENV ${ARM_ENV:-}"
deploy() {
  local t0=$(date +%s)
  OUT=$(bash ${LAUNCHER:-deploy-vllm-qwen38-paro.sh} 2>&1 | tail -3)   # LAUNCHER: e.g. deploy-vllm-qwen38-paro.start.sh
  echo "$OUT" | grep -q READY-VLLM || { echo "DEPLOY FAILED ($L):"; echo "$OUT"; docker logs --tail 40 vllm-exp 2>&1; exit 1; }
  echo "deployed in $(( $(date +%s) - t0 )) s: $(echo "$OUT" | grep -oE 'kv_tokens=[0-9]+')"
}
deploy
docker logs vllm-exp > /tmp/p7-$L-boot1.log 2>&1
CT=$(grep -oE 'torch.compile took [0-9.]+' /tmp/p7-$L-boot1.log | awk '{s += $NF} END {print int(s)}')
# A fully cached compile totals ~4-5 s. Anything more is a (partial) fresh compile, and a fresh-compile boot is NOT
# bit-identical to the cached boots that follow it (p9: KL 3.4e-3 vs 0 between two cached boots), so never measure on it.
if [ "${CT:-0}" -gt 10 ]; then
  echo "cold compile ($(grep -oE 'torch.compile took [0-9.]+ s' /tmp/p7-$L-boot1.log | head -1)) -> redeploy once"
  deploy
fi
docker logs vllm-exp 2>&1 | grep -E '\[run\] attn=|single-GPU profile|INT2_DRAFT_HEAD|draft vocab|\[radiance.gdn\]|libr4d .* ->' | cut -c1-200 | sort -u
[ "${NOBENCH:-0}" = 1 ] && exit 0
B=http://192.168.88.89:1252
export VLLM_METRICS=$B/metrics
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L-warm --depths 2000 --runs 1 --n-predict 64 --tag w1 >/dev/null 2>&1
for D in 8000 100000; do
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  R=$($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L --depths $D --runs 1 --n-predict 600 --tag dec 2>&1 | grep -E 'prompt=')
  A=$($PY /repo/bench/vllm_accept_delta.py delta)
  echo "decode d=$D :: $R :: $A"
done
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label $L --depths 8000,43000 --runs 1 --n-predict 16 \
  --tag p$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill /'
echo P7_ARM_DONE $L
