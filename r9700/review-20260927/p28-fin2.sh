#!/bin/bash
# Final candidate after the p26/p27 fault hunt: rot3 on the norm-fed rotate only (ew producers stock, RADIANCE_PQM_ROT3_EW
# unset), split-token decode producers, AOT env key (base-seeded), NO GDN no-copy / EMPTY_OUT. Every boot's full log is
# kept and scanned for GPU memory faults. Bench: greedy 8k/100k (sha 55222d73 / 86cd9c47), cold prefill, sampled 12x600;
# then two more plain reboots (fault scan only).
set -u
cd /mnt/user/appdata/llama-gemma31b
B=http://192.168.88.89:1252
PY="docker exec -e VLLM_METRICS=$B/metrics llama-hip-dev python3"
F2='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1'
export RADIANCE_GDN_EMPTY_OUT=0 RADIANCE_FP8_STREAM=1
docker stop vllm-qwen38 >/dev/null 2>&1
logwatch() {   # $1 = tag; follows the next vllm-exp container's log to /tmp/p28-$1.log
  (for i in $(seq 1 180); do docker ps --format '{{.Names}}' | grep -q vllm-exp && break; sleep 1; done
   timeout 900 docker logs -f vllm-exp > /tmp/p28-$1.log 2>&1) &
}
echo "== fin2"
logwatch b1
ARM_ENV="$F2" bash jobs/p7-vllm-arm.sh fin2 2>&1 | grep -E 'deployed|FAILED|cold compile|decode d=|prefill'
docker logs vllm-exp 2>&1 | grep -E 'aot_envkey|kernel override|MISMATCH|Directly load|Compiling model' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-160 | sort | uniq -c
$PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label fin2 --depths 8000,43000,100000 --runs 1 \
  --n-predict 16 --tag q$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh fin2 2>&1 | grep -E "TOTAL|GRAND"
docker logs vllm-exp > /tmp/p28-b2.log 2>&1
for r in 3 4; do
  logwatch b$r
  ARM_ENV="$F2" NOBENCH=1 bash jobs/p7-vllm-arm.sh fin2r$r 2>&1 | grep -E 'deployed|FAILED'
  sleep 2; docker logs vllm-exp > /tmp/p28-b$r.log 2>&1
done
for f in /tmp/p28-b*.log; do echo "$f: faults $(grep -c 'Memory Fault' $f), mismatch $(grep -c MISMATCH $f)"; done
docker rm -f vllm-exp >/dev/null 2>&1
echo P28_DONE
