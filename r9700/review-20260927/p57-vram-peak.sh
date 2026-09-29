#!/bin/bash
# Peak VRAM of the prod config under long prefills (for sizing a KV_MEM push): sysfs vram_used sampled every 0.1 s.
set -u
cd /mnt/user/appdata/llama-gemma31b
B=http://192.168.88.89:1246
S=/sys/class/drm/card0/device/mem_info_vram_used
PY="docker exec llama-hip-dev python3"
peak() { local m=0 v; while [ ! -f /tmp/p57.stop ]; do v=$(cat $S); [ $v -gt $m ] && m=$v && echo $m > /tmp/p57.max; sleep 0.1; done; }
rm -f /tmp/p57.stop; echo 0 > /tmp/p57.max; peak & SP=$!
mark() { echo "$1: now $(( $(cat $S)/1048576 )) MiB, peak so far $(( $(cat /tmp/p57.max)/1048576 )) MiB"; }
mark idle
for D in ${DEPTHS:-8000 120000 225000}; do   # depth N is ~1.124 N real tokens: 225000 -> ~253k (MAXLEN 262,144)
  $PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label vram --depths $D --runs 1 --n-predict 16 --tag v$RANDOM 2>&1 | grep -E "prompt=" | cut -c1-150
  mark "after $D"
done
CP=""; for i in 1 2; do $PY /repo/bench/oai_bench_greedy.py depth --base $B --model qwen38-27b --label vram --depths ${CDEPTH:-100000} --runs 1 --n-predict 256 --tag c$i$RANDOM >/dev/null 2>&1 & CP="$CP $!"; done; wait $CP
mark "after 2x100k concurrent"
touch /tmp/p57.stop; wait $SP 2>/dev/null
curl -s $B/metrics | grep -E "^vllm:num_preemptions_total"
