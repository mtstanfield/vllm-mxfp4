#!/bin/bash
# Final candidate for production (all byte-exact levers from p23-p25), on its own AOT artifact:
#   rot3 prefill producers + SKIP_HS, split-token decode producers, GDN core_attn_out fill skip (EMPTY_OUT + patch_gdn_nocopy
#   with the z view OFF), AOT env key. Kernel .so = -split2 (mr3 + split + stream-1 split, gated byte-exact).
# fin: greedy 8k/100k (sha must be 55222d73 / 86cd9c47), cold prefill 8k/43k x2 + 100k, sampled 12x600 (acceptance must
# equal prod's .531/.430/.725). Then zv2: boot the z-view-only config that failed in p25 and keep the log tail.
set -u
cd /mnt/user/appdata/llama-gemma31b
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
FIN='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
docker stop vllm-qwen38 >/dev/null 2>&1
export RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=1
echo "== fin"
ARM_ENV="$FIN" bash jobs/p7-vllm-arm.sh fin 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
docker logs vllm-exp 2>&1 | grep -E 'aot_envkey|kernel override|gdn-nocopy|Directly load|torch.compile took' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-200 | sort | uniq -c
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label fin --depths 8000,43000,100000 --runs 1 \
  --n-predict 16 --tag q$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh fin 2>&1 | grep -E "TOTAL|GRAND"
echo "== zv2 (z view only; failed to boot in p25)"
export RADIANCE_GDN_EMPTY_OUT=0
ZV='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1'
ARM_ENV="$ZV" NOBENCH=1 bash jobs/p7-vllm-arm.sh zv2 > /tmp/p26-zv2.log 2>&1
grep -E 'deployed|FAILED' /tmp/p26-zv2.log
grep -E 'Error|error|Traceback|raise|RuntimeError|assert' /tmp/p26-zv2.log | grep -v 'registration failed' | cut -c1-300 | head -30
docker rm -f vllm-exp >/dev/null 2>&1
echo P26_DONE
