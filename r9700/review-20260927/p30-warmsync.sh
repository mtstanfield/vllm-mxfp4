#!/bin/bash
# A/B the GDN warm-up synchronize (local/patch_gdn_warmup_sync.py) against the boot memory fault of p25-p27:
# the previously failing config (rot3 incl. ew + SKIP_HS + GDN no-copy Z=0 + EMPTY_OUT + split), no serialization.
# ctl: 3 boot attempts without the fix; fix: 5 with it. Every container boot runs the profile forward once.
set -u
cd /mnt/user/appdata/llama-gemma31b
C='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_ROT3_EW=1 -e RADIANCE_PQM_SKIP_HS=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_NOCOPY=1 -e RADIANCE_LOCAL_GDN_NOCOPY_Z=0 -e RADIANCE_GDN_EMPTY_OUT=1'
export RADIANCE_GDN_EMPTY_OUT=1 RADIANCE_FP8_STREAM=1
docker stop vllm-qwen38 >/dev/null 2>&1
attempt() {   # $1 tag, $2 extra env
  (for i in $(seq 1 180); do docker ps --format '{{.Names}}' | grep -q vllm-exp && break; sleep 1; done
   timeout 600 docker logs -f vllm-exp > /tmp/p30-$1.log 2>&1) &
  OUT=$(WRAPPER=deploy-vllm-exp.sh NAME=vllm-exp PORT=1252 EXTRA_ENV="-e PYTHONHASHSEED=0 -e RADIANCE_PAROQUANT_INSTALL=1 -e RADIANCE_PAROQUANT=1 -e RADIANCE_PQ_I8=1 -e RADIANCE_PQ_PG=1 -e RADIANCE_PQ_ZPE=1 -e RADIANCE_PQ_ROT_STREAM=1 -e RADIANCE_PQ_ROT_STREAM2=1 -e RADIANCE_GDN_LAZY=0 $C $2" timeout 600 bash deploy-vllm-qwen38-paro.sh 2>&1 | tail -2)
  sleep 3
  echo "$1: $(echo "$OUT" | grep -oE 'READY-VLLM|ENGINE_INIT_FAILED' | head -1) faults=$(grep -c 'Memory Fault' /tmp/p30-$1.log) sync=$(grep -c 'gdn-warmup-sync\] applied' /tmp/p30-$1.log) compile=$(grep -oE 'torch.compile took [0-9.]+' /tmp/p30-$1.log | head -1)"
  grep -E 'Memory Fault' /tmp/p30-$1.log | grep -oE 'faulting addr: [0-9a-fx]+, kernel: [a-z_0-9]+' | head -1
}
for r in 1 2 3; do attempt ctl$r ""; done
for r in 1 2 3 4 5; do attempt fix$r "-e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1"; done
docker rm -f vllm-exp >/dev/null 2>&1
echo P30_DONE
