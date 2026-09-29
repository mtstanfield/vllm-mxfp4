#!/bin/bash
# Decode attention at long context (p31: 9.5 of 48.5 ms/step at 100k). The split law picks 32 splits x 4 kv heads = 128
# workgroups of 1-2 waves at TP=1 (it was tuned at TP=2: 64 x 2). Sweep the split count on the production library
# (rx9z) with dec_bench.py: q_len 5 (verify) and 1 (draft), DRAM-fed, max_ctx = the graph bound 262,144.
set -u
R=/mnt/user/appdata/vllm-radiance/persist/libr4d-029
W=$R/work-attn
docker stop vllm-qwen38 >/dev/null 2>&1
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash -v $W:/w -v $R:/r \
  ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake -lc "python3 /w/dec_bench.py /r/b9e42ab-rx9z ${SPL:-0,16,32,64,128,256} ${CTXS:-8192,32768,100000,200000}" 2>&1 | grep -v registration
echo P38_DONE
