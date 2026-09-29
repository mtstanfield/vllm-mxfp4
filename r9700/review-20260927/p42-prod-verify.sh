#!/bin/bash
# Round-5 production (:1246) verification on a cached-compile boot: restart once (the first boot compiled), then greedy
# 8k/100k (new sha baseline: the drafter changed), cold prefill 8k/43k/100k, sampled 12x600, held-out trace acceptance.
set -u
cd /mnt/user/appdata/llama-gemma31b
B=http://192.168.88.89:1246
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
echo "== restart (load: $(cut -d' ' -f1-3 /proc/loadavg))"
bash deploy-vllm-qwen38-paro.sh 2>&1 | grep -oE 'READY-VLLM|kv_tokens=[0-9]+' | tr '\n' ' '; echo
docker logs vllm-qwen38 2>&1 | grep -E 'torch.compile took|Memory Fault|fingerprint' | cut -c1-160
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label warm --depths 2000 --runs 1 --n-predict 64 --tag w1 >/dev/null 2>&1
for D in 8000 100000; do
  $PY /repo/bench/vllm_accept_delta.py snap >/dev/null
  R=$($PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label prodfinal --depths $D --runs 1 --n-predict 600 --tag dec 2>&1 | grep -E 'prompt=')
  echo "decode d=$D :: $R :: $($PY /repo/bench/vllm_accept_delta.py delta)"
done
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label prodfinal --depths 8000,43000,100000 --runs 1 --n-predict 16 \
  --tag r$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill /'
docker exec llama-hip-dev python3 /repo/bench/spec_sampled.py --base $B --prompts code --runs 1 --n 64 --label warm >/dev/null 2>&1
docker exec llama-hip-dev python3 /repo/bench/spec_sampled.py --base $B --runs 12 --n 600 --metrics $B/metrics --label prodfinal 2>&1 | grep -E 'TOTAL|GRAND'
docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
  -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py $B prodfinal 24 6144 256" 2>&1 | grep -v registration
echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
docker logs vllm-qwen38 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
echo P42_DONE
