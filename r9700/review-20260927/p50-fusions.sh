#!/bin/bash
# Decode fusions, round 2 of the A/B (same RADIANCE_AB_SESSION=p49 artifacts for base/qknr -> cached boots):
#   base, qknr (fused split+QK-RMSNorm+mRoPE+gate): held-out omp-window acceptance (trace_accept.py, 24 x 6144+256)
#   s1 = base + rotation stream 1 live (host RADIANCE_FP8_STREAM=0: radiance_arnq no longer overwrites the layer
#        forwards, so add+RMSNorm+rotate+quant run as ONE split kernel per norm site instead of inductor add+rms + rotate):
#        greedy 8k/100k, prefill, sampled 12x600, trace acceptance, served PPL.
set -u
cd /mnt/user/appdata/llama-gemma31b
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1 -e RADIANCE_AB_SESSION=p49'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
trace() { docker run --rm --network host --entrypoint bash -v /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-v2:/pqv2 \
  -v /mnt/user/appdata/vllm-radiance/custom-quant/mtp-gptq:/s $IMG -lc "python3 /s/trace_accept.py http://192.168.88.89:1252 $1 24 6144 256" 2>&1 | grep TRACE; }
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in "base:" "qknr:-e RADIANCE_LOCAL_FUSED_QKNR=1"; do
  L=${arm%%:*}; X=${arm#*:}
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $X" NOBENCH=1 bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile"
  trace $L
done
echo "== s1 (load: $(cut -d' ' -f1-3 /proc/loadavg))"
export RADIANCE_FP8_STREAM=0
ARM_ENV="$F" bash jobs/p7-vllm-arm.sh s1 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill" | cut -c1-230
docker logs vllm-exp 2>&1 | grep -iE "rot stream|fp8 stream|arnq" | sort -u | head -5 | cut -c1-200
RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh s1 2>&1 | grep -E "TOTAL|GRAND"
trace s1
docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
ARM_ENV="$F" bash jobs/p7-vllm-ppl.sh ppl-s1 2>&1 | grep -E "FINAL|FAILED"
unset RADIANCE_FP8_STREAM
docker rm -f vllm-exp >/dev/null 2>&1
echo "== restore production"
bash deploy-vllm-qwen38-paro.sh 2>&1 | tail -1 | grep -oE 'READY-VLLM|kv_tokens=[0-9]+' | tr '\n' ' '; echo
echo P50_DONE
