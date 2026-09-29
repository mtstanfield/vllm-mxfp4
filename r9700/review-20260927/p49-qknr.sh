#!/bin/bash
# Decode fusion 1: vLLM's fused split + QK-RMSNorm + (m)RoPE + gate kernel on ROCm (local/patch_fused_qknr.py,
# RADIANCE_LOCAL_FUSED_QKNR=1; radiance_qknr = the upstream kernel restructured for the AMD Triton backend). A/B on the
# rig vs the round-5b production env, both compiled in this session: greedy 8k/100k, prefill, sampled 12x600, served PPL
# (wiki 60 / code 30 chunks). Production restored at the end.
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so -e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in "base:-e RADIANCE_AB_SESSION=p49" "qknr:-e RADIANCE_AB_SESSION=p49 -e RADIANCE_LOCAL_FUSED_QKNR=1"; do
  L=${arm%%:*}; X=${arm#*:}
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $X" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|cold compile|decode d=|prefill|fused-qknr" | cut -c1-230
  docker logs vllm-exp 2>&1 | grep -cE "fused-qknr\] applied" | sed 's/^/   qknr patch applied: /'
  RUNS=12 NODEPLOY=1 bash jobs/p7-vllm-sampled.sh $L 2>&1 | grep -E "TOTAL|GRAND"
  docker logs vllm-exp 2>&1 | grep -c 'Memory Fault' | sed 's/^/   memory faults: /'
  ARM_ENV="$F $X" bash jobs/p7-vllm-ppl.sh ppl-$L 2>&1 | grep -E "PPL|ppl|FAILED" | tail -2
done
docker rm -f vllm-exp >/dev/null 2>&1
echo "== restore production"
bash deploy-vllm-qwen38-paro.sh 2>&1 | tail -1 | grep -oE 'READY-VLLM|kv_tokens=[0-9]+' | tr '\n' ' '; echo
echo P49_DONE
