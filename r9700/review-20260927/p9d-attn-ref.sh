#!/bin/bash
# p8's lastpos reference (mode 0, attn0) ran on a partial fresh-compile boot (torch.compile 21.5 s; modes 3/7/15/14 were
# cached, 4.6-4.7 s) and p9 showed a fresh-compile boot is not bit-identical to the cached boots after it. Re-collect the
# mode-0 reference on a cached boot and re-score every saved mode against it.
set -u
cd /mnt/user/appdata/llama-gemma31b
R=bench/results/p8
PY="docker exec llama-hip-dev python3"
R4D_KEY=b9e42ab-rx9y R4D_ATTN_FP8=0 MAXLEN=131072 NOBENCH=1 bash jobs/p7-vllm-arm.sh attn0b 2>&1 | grep -E 'deployed|FAILED|cold compile'
grep -oE 'torch.compile took [0-9.]+ s' /tmp/p7-attn0b-boot1.log | tr '\n' ' '; echo
$PY /repo/bench/lastpos_kl.py collect --base http://192.168.88.89:1252 --out /repo/$R/lastpos-0b.json 2>&1 | tail -1
echo "-- fresh-compile mode 0 vs cached mode 0"; $PY /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-0b.json /repo/$R/lastpos-0.json 2>&1 | tail -9
for M in 3 7 15 14; do
  echo "-- mode $M vs cached mode 0"; $PY /repo/bench/lastpos_kl.py compare /repo/$R/lastpos-0b.json /repo/$R/lastpos-$M.json 2>&1 | tail -9
done
echo P9D_DONE
