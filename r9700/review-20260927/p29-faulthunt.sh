#!/bin/bash
# Reproduce the rot3-ew memory fault (p25-p27) with every kernel serialized (AMD_SERIALIZE_KERNEL=3) and every rot3
# launch logged with pointers/sizes (RADIANCE_PQM_ROT3_DEBUG), so the fault lands on the last logged call.
# Config = the p25 eo / p26 fin set (GDN no-copy Z=0 + EMPTY_OUT + rot3 incl. ew + SKIP_HS + split). Up to 3 boots.
set -u
cd /mnt/user/appdata/llama-gemma31b
C='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1 -e RADIANCE_PQM_ROT3_DEBUG=3000 -e AMD_SERIALIZE_KERNEL=3'
export RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=1
docker stop vllm-qwen38 >/dev/null 2>&1
for r in 1 2 3; do
  (for i in $(seq 1 180); do docker ps --format '{{.Names}}' | grep -q vllm-exp && break; sleep 1; done
   timeout 1500 docker logs -f vllm-exp > /tmp/p29-r$r.log 2>&1) &
  ARM_ENV="$C" NOBENCH=1 timeout 1500 bash jobs/p7-vllm-arm.sh fh$r 2>&1 | grep -E 'deployed|FAILED' | head -2
  sleep 3
  F=$(grep -c 'Memory Fault' /tmp/p29-r$r.log)
  echo "boot $r: faults $F, rot3 launches logged $(grep -c 'rot3dbg' /tmp/p29-r$r.log), mismatches $(grep -c MISMATCH /tmp/p29-r$r.log)"
  if [ "$F" -gt 0 ]; then
    grep -E 'Memory Fault' /tmp/p29-r$r.log | cut -c1-300
    echo "-- last rot3 launches before the fault:"
    grep 'rot3dbg' /tmp/p29-r$r.log | tail -4 | cut -c1-900
    break
  fi
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P29_DONE
