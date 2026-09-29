#!/bin/bash
# rot4 (select-free read-routed rotation core, kernel .so -0.29-rot4 + RADIANCE_PQM_ROT4=1) vs round-5 production env on
# the rig: greedy 8k/100k shas must be IDENTICAL (byte-exact producers), cold prefill 8k / 43k / 100k twice each.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in "r5:-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so" "rot4:-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1"; do
  L=${arm%%:*}; X=${arm#*:}
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $X" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill|Error|error:" | cut -c1-240
  for i in 1 2; do $G --label $L --depths 8000,43000,100000 --n-predict 16 --tag q$i$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed "s/^/prefill$i /"; done
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P44_DONE
