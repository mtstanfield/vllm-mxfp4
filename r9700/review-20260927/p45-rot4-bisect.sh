#!/bin/bash
# rot4 changes the 100k greedy text (c2b0a3bf vs 98b2b349) although every kernel gate is exact. Bisect on the rig, all
# arms on the rot4 .so: ROT4 off (does the rebuilt .so itself differ?), then rot4 limited to one K at a time
# (5120 = norm-fed qkvz/gate_up/qkv, 6144 = ew1/ew2 o_proj/out_proj, 17408 = ew0 down).
set -u
cd /mnt/user/appdata/llama-gemma31b
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 -e RADIANCE_MTP_MXFP4=1 -e RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-rot4.so'
export R4D_KEY=b9e42ab-rx9z RADIANCE_GDN_EMPTY_OUT=1
docker stop vllm-qwen38 >/dev/null 2>&1
for arm in "so_off:" "k5120:-e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT4_K=5120" "k6144:-e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT4_K=6144" "k17408:-e RADIANCE_PQM_ROT4=1 -e RADIANCE_PQM_ROT4_K=17408"; do
  L=${arm%%:*}; X=${arm#*:}
  echo "== $L (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  ARM_ENV="$F $X" bash jobs/p7-vllm-arm.sh $L 2>&1 | grep -E "deployed|FAILED|decode d=" | cut -c1-240
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P45_DONE
