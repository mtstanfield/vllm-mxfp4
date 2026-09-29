#!/bin/bash
# Decode fusions, round 3 (production stays DOWN; no restore here):
#   qknr2 = fused split+QK-RMSNorm+mRoPE+gate v2 (single rounding, RADIANCE_QKNR_ROUND=0)
#   s1    = rotation stream 1 live (host RADIANCE_FP8_STREAM=0) with CACHE pinned to the production dir (FP8_STREAM=0
#           drops serve-mxfp4.sh's -tp1s suffix -> other /cache: no TunableOp table, no GPTQ file = round 4's "5% slower")
#   each: greedy 8k/100k + prefill, sampled 12x600, held-out trace acceptance, served PPL.
#   zchk: eager boots, in-place z read (NOCOPY_Z=1) vs the copy (0), same prompt -> same text?
# Reference (p49/p50, base): sampled 89.08 (.518/.411/.704), trace .5590 77.82 t/s, PPL 5.9765 / 2.3040, 8k step 35.3 ms.
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
P=/mnt/user/appdata/vllm-radiance/persist
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
trace() { docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
  -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py http://192.168.88.89:1252 $1 24 6144 256" 2>&1 | grep TRACE; }
docker stop vllm-qwen38 >/dev/null 2>&1
full() {   # label, extra -e flags
  local L=$1; shift
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 $*" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill" | cut -c1-230
  docker logs vllm-exp 2>&1 | grep -E "fused-qknr\] applied|Traceback|Error:" | sort -u | head -3 | cut -c1-160
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  trace $L
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
  ARM_ENV="$F -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 $*" bash jobs/p7-vllm-ppl.sh ppl-$L 2>&1 | grep -E "FINAL|FAILED"
}
full qknr2 -e RADIANCE_LOCAL_FUSED_QKNR=1
RADIANCE_FP8_STREAM=0 CACHE=$P/cache-029-paro-tp1s full s1   # serve-mxfp4.sh appends -tp1s only when FP8S=1
echo "== zchk (eager)"
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
for z in 0 1; do
  KV_MEM=5000000000 MAXLEN=65536 EXTRA="$MTP --enforce-eager" ARM_ENV="$F -e RADIANCE_LOCAL_GDN_NOCOPY_Z=$z" NOBENCH=1 bash jobs/p7-vllm-arm.sh ez$z 2>&1 | grep -E "deployed|FAILED"
  for d in 3000 20000; do
    echo "   NOCOPY_Z=$z d=$d $(docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base http://192.168.88.89:1252 --model qwen38-27b --runs 1 --label ez$z --depths $d --n-predict 64 --tag z1 2>&1 | grep -oE 'sha=[0-9a-f]+')"
  done
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P52_DONE
