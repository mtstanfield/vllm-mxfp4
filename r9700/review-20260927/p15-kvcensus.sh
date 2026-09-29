#!/bin/bash
# fp8 KV-cache census (radiance_r4d_attn.py RADIANCE_KV_STATS), loaded over the baked backend through the env-gated
# local overlay hook (local/patch_local_overlay.py; serve-mxfp4.sh does not copy radiance_r4d_attn.py). One 40k prefill
# of each document; per attention layer: K/V magnitudes vs e4m3's range and the cache round-trip error at scale 1.0 vs an
# amax-calibrated per-tensor scale.
set -u
cd /mnt/user/appdata/llama-gemma31b
VOC='-e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt'
B=http://192.168.88.89:1252
docker stop vllm-qwen38 >/dev/null 2>&1
TUN='-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/mtp%d.csv -e PYTORCH_TUNABLEOP_VERBOSE=1'
ARM_ENV="$VOC -e RADIANCE_LOCAL_OVERLAY=radiance_r4d_attn.py -e RADIANCE_KV_STATS=16384 $TUN" NOBENCH=1 bash jobs/p7-vllm-arm.sh kvc 2>&1 | grep -E 'deployed|FAILED'
docker logs vllm-exp 2>&1 | grep -E 'local overlay' | sort -u
docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1 --label kv --depths 40000 \
  --n-predict 1 --tag kv1 2>&1 | grep -oE 'prompt= *[0-9]+'
sleep 2
docker logs vllm-exp 2>&1 | grep 'radiance.kvstats' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | cut -c1-470
echo "== TunableOp in the serve (verbose)"
docker logs vllm-exp 2>&1 | grep -iE 'tuning results|validation|Loading results|tunableop|could not' | sed -E 's/^\([A-Za-z]+ pid=[0-9]+\) //' | sort | uniq -c | head -20
docker exec llama-hip-dev python3 /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --runs 1 --label t --depths 8000   --n-predict 600 --tag dec 2>&1 | grep -E 'prompt='
echo P15_DONE
