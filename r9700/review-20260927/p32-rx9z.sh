#!/bin/bash
# libr4d rx9z (= rx9y + mode-15 prefill attention: PF8 + PVDEN summed by fp8 dot4 after the PV WMMAs; bit-identical in
# attn_bench) vs rx9y: harness check of the full builds, then end-to-end on the rig with the round-4 production env:
# greedy 8k/100k (sha must stay 55222d73 / 86cd9c47), cold prefill 8k / 43k / 100k twice each.
set -u
cd /mnt/user/appdata/llama-gemma31b
R=/mnt/user/appdata/vllm-radiance/persist/libr4d-029
W=$R/work-attn
docker stop vllm-qwen38 >/dev/null 2>&1
echo "== harness (full builds)"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash -v $W:/w -v $R:/r \
  ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake -lc 'export R4D_ATTN_FP8=15; rm -rf /tmp/attnb
    python3 /w/attn_bench.py /r/b9e42ab-rx9y rx9y 8192 8192 65536 131072; python3 /w/attn_bench.py /r/b9e42ab-rx9z rx9z 8192 8192 65536 131072
    python3 -c "
import torch
for d in (8192, 65536, 131072):
    print(d, \"BIT-EXACT\" if torch.equal(torch.load(f\"/tmp/attnb/rx9y_{d}.pt\"), torch.load(f\"/tmp/attnb/rx9z_{d}.pt\")) else \"DIFFERS\")"' 2>&1 | grep -v registration
F='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt -e RADIANCE_DRAFT_EXACTSET=1 -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv -e RADIANCE_LOCAL_AOT_ENVKEY=1 -e RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so -e RADIANCE_PQM_ROT3=1 -e RADIANCE_PQM_SPLIT=1 -e RADIANCE_LOCAL_GDN_WARMUP_SYNC=1'
B=http://192.168.88.89:1252
G="docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1"
for key in b9e42ab-rx9y b9e42ab-rx9z; do
  echo "== $key"
  R4D_KEY=$key ARM_ENV="$F" bash jobs/p7-vllm-arm.sh $key 2>&1 | grep -E 'deployed|FAILED|decode d=|prefill|libr4d'
  $G --label $key --depths 8000,43000,100000 --n-predict 16 --tag r$((RANDOM % 90 + 10)) 2>&1 | grep -E 'prompt=' | sed 's/^/prefill2 /'
done
docker rm -f vllm-exp >/dev/null 2>&1
echo P32_DONE
