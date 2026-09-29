#!/bin/bash
# After p38: decode split-law variant bench (work-dec builds), then install the H-collection/GPTQ-file drafter patch
# (inert unless RADIANCE_MTP_HCOLLECT / RADIANCE_MTP_MXFP4_FILE are set) and run p39 (MTP drafter GPTQ).
set -u
cd /mnt/user/appdata/llama-gemma31b
until grep -q P38_DONE /tmp/p38.out 2>/dev/null; do sleep 20; done
R=/mnt/user/appdata/vllm-radiance/persist/libr4d-029
echo "== decode split-law variants (splits=0 = the law, graph max_ctx)"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash -v $R/work-dec:/w \
  ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake -lc 'bash /w/dec_variants.sh "192:128:1 384:128:1 768:256:1 1536:256:1 768:256:16 768:256:32 1536:256:32"' 2>&1 | grep -v registration
L=/mnt/user/appdata/vllm-radiance-next/local
cp -p $L/patch_paroquant_install.py /mnt/user/appdata/vllm-radiance-next/patch_paroquant_install.py.bak-20260928day
mv $L/patch_paroquant_install.py.day2 $L/patch_paroquant_install.py
echo "== p39"
bash jobs/p39-mtp-gptq.sh
echo P40_DONE
