#!/bin/bash
# (1) why does rotation stream 1 (host RADIANCE_FP8_STREAM=0) fail to boot? full log of one boot.
# (2) z integrity: eager boot with the in-place z read (NOCOPY_Z=1) + RADIANCE_DEBUG_ZCHECK=1 -> does the GDN core op
#     change the projection's z region? One prefill + decode request. Production restored at the end.
set -u
cd /mnt/user/appdata/llama-gemma31b
P5='-e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
echo "== (1) stream 1 boot"
RADIANCE_FP8_STREAM=0 bash jobs/dbg-boot.sh s1dbg /tmp/s1dbg.log $P5 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0
docker rm -f vllm-exp >/dev/null 2>&1
echo "== (2) z check (eager)"
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
KV_MEM=5000000000 MAXLEN=65536 EXTRA="$MTP --enforce-eager" bash jobs/dbg-boot.sh zchk /tmp/zchk.log $P5 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=1 -e RADIANCE_DEBUG_ZCHECK=1
docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base http://192.168.88.89:1252 --model qwen38-27b --runs 1 --label z --depths 3000 --n-predict 24 --tag z1 2>&1 | grep -oE "sha=[0-9a-f]+"
sleep 2
docker logs vllm-exp 2>&1 | grep "\[zcheck\]" | awk '{print $0}' | cut -c1-160 | sort | uniq -c | sort -rn | head -12
docker logs vllm-exp 2>&1 | grep -c "\[zcheck\]"
docker rm -f vllm-exp >/dev/null 2>&1
echo "== restore production"
bash deploy-vllm-qwen38-paro.sh 2>&1 | tail -1 | grep -oE 'READY-VLLM|kv_tokens=[0-9]+' | tr '\n' ' '; echo
echo P51_DONE
