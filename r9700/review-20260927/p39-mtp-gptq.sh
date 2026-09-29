#!/bin/bash
# MTP drafter acceptance recovery: GPTQ requant of the drafter to MXFP4 on its own inputs (instead of load-time RTN).
#  1. calibration boot (eager, fp8 drafter, RADIANCE_MTP_HCOLLECT): 48 x 4096-token windows of the user's omp traces
#     (traces.npz calib, the body's v2 set) with sampled continuations -> H per drafter linear (draft / prefill rows)
#  2. mtp_gptq.py (custom-quant/mtp-gptq) -> mtp_gptq_a<alpha>.pt
#  3. round-5 candidate + RADIANCE_MTP_MXFP4_FILE: sampled 12x600 (acceptance vs RTN .521/.415/.682, 88.6-88.8 t/s)
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
P=/mnt/user/appdata/vllm-radiance/persist
HC=$P/cache-029-paro-tp1s/mtp_hc   # = /cache in the container (SINGLE_GPU_PROFILE adds -tp1s)
S=/mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq
V2=/mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1'
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":4,"attention_backend":"R4D","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
export R4D_KEY=b9e42ab-rx9z
docker stop vllm-qwen38 >/dev/null 2>&1
if [ ! -f $HC/SAVED ]; then
  mkdir -p $HC
  echo "== calibration boot (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  KV_MEM=5000000000 MAXLEN=65536 EXTRA="$MTP --enforce-eager" ARM_ENV="$F -e RADIANCE_MTP_HCOLLECT=/cache/mtp_hc" NOBENCH=1 \
    bash jobs/p7-vllm-arm.sh hc 2>&1 | grep -E 'deployed|FAILED'
  docker logs vllm-exp 2>&1 | grep -c 'mtp_hcollect\]' | sed 's/^/   collecting layers: /'
  docker run --rm --network host --entrypoint bash -v $V2:/pqv2 -v $HC:/hc -v $S:/s $IMG \
    -lc "python3 /s/mtp_hc_drive.py http://192.168.88.89:1252 /hc ${NWIN:-48} ${MAXTOK:-320}" 2>&1 | grep -v registration
  docker rm -f vllm-exp >/dev/null 2>&1
fi
ls -la $HC
for a in ${ALPHAS:-0.5}; do
  [ -f $HC/mtp_gptq_a$a.pt ] || docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash \
    -v $V2:/pqv2 -v $HC:/hc -v $S:/s $IMG -lc "python3 /s/mtp_gptq.py /hc /hc/mtp_gptq_a$a.pt $a" 2>&1 | grep -v registration
done
for a in ${ALPHAS:-0.5}; do
  echo "== gptq a=$a (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_hc/mtp_gptq_a$a.pt" NOBENCH=1 \
    bash jobs/p7-vllm-arm.sh gq$a 2>&1 | grep -E 'deployed|FAILED|cold compile'
  docker logs vllm-exp 2>&1 | grep 'mtp_mxfp4\]' | cut -c1-200
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh gq$a 2>&1 | grep -E "TOTAL|GRAND"
  echo "   load after: $(cut -d' ' -f1-3 /proc/loadavg)"
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P39_DONE
