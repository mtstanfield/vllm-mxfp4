#!/bin/bash
# Is the prefill MXFP4 GEMM (~59% of fp8 peak) at its hard limit? (1) pure-WMMA peak with sclk/power sampled,
# (2) the A-tiled kernel vs hipBLASLt fp8 vs bf16 at the production shapes (M=8192). Production restored afterwards.
set -u
cd /mnt/user/appdata/llama-gemma31b
W=/mnt/user/appdata/vllm-radiance/persist/gemm-work
IMG=ggz14/vllm-radiance-mxfp4:0.13.0-1e82407-prebake
HW=$(dirname $(ls /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap))
docker stop vllm-qwen38 >/dev/null 2>&1
sleep 3
echo "== WMMA peak (idle: $(( $(cat $HW/freq1_input) / 1000000 )) MHz, $(( $(cat $HW/power1_average) / 1000000 )) W)"
for cfg in "8 2048 8192" "4 2048 8192" "16 1024 8192"; do
  ( while :; do echo "$(cat $HW/freq1_input) $(cat $HW/power1_average)"; sleep 0.05; done ) > /tmp/clk.$$ &
  SP=$!
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash -v $W:/w $IMG -lc "LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib /w/wmma_peak $cfg" 2>&1 | grep -v registration
  kill $SP
  sort -n /tmp/clk.$$ | awk '$1 > 1500000000 {f[n]=$1; p[n]=$2; n++} END {if (n) printf "   busy samples %d: median sclk %d MHz, median power %d W\n", n, f[int(n/2)]/1e6, p[int(n/2)]/1e6}'
done
rm -f /tmp/clk.$$
echo "== GEMM shapes"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint bash -v $W:/w $IMG -lc "python3 /w/gemm_bench.py /w" 2>&1 | grep -v registration
echo "== restoring production"
bash deploy-vllm-qwen38-paro.sh 2>&1 | grep -oE "READY-VLLM|kv_tokens=[0-9]+" | tr "\n" " "; echo
echo P47_DONE
