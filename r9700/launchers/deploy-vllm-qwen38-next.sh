#!/bin/bash
# Deploy the vLLM-Radiance MXFP4 engine (Qwen3.8-27B, one R9700) as production on :1246.
# Wraps the pinned upstream launcher (serve-mxfp4.sh @ 92eed82, github.com/GGZ14/vllm-mxfp4; was codeberg 1000e61 until 2026-09-15): its own container command
# becomes our long-lived container (-d --restart unless-stopped, no --rm). Refuses while
# llama-qwen38 runs.
#
# QUOTING RISK (found + resolved 2026-09-07, see task-1-report.md for the evidence):
# serve-mxfp4.sh's last argv element (before our --served-model-name override) is a MULTI-LINE
# bash script passed as one `-lc '<script>'` argument. `DRY_RUN=1` prints the argv by `echo`ing
# it, which flattens quoting entirely and leaves the script's embedded newlines as real
# newlines in the output. `grep -E '^docker run '` on that output therefore only ever matches
# the FIRST line -- everything from the `-lc` script body onward (the whole patch/build script,
# the model path, --served-model-name, --port, every vllm serve flag) is silently dropped.
# `eval`-ing that truncated string is unsafe: it does not fail loudly, it launches `docker run`
# with a bare, argument-less `-lc` and none of the real serve flags. VERIFIED broken by
# constructing the equivalent CMD and inspecting it -- see task-1-report.md.
#
# RESOLUTION: never touch the launcher's docker command as text. Put a `docker` shim earlier
# in PATH that intercepts the real (non-DRY_RUN) `docker run ...` invocations the launcher
# makes, rewrites the argv ARRAY in place for the server call only (prepend -d --restart
# unless-stopped and $EXTRA_ENV, drop any --rm), and re-execs the real docker binary with that
# array. The -lc script argument (newlines, quotes and all) never passes through a string or
# `eval` -- it stays one argv element the whole way, exactly as the shell built it.
#
# SCOPING (fix round 1, 2026-09-07): serve-mxfp4.sh issues TWO "docker run" calls, not one --
# see the shim's own header comment below for exactly which, and why only the server call may
# be rewritten. The shim identifies the server call positively (by --privileged) rather than
# rewriting every "docker run" it sees, so the libr4d auto-build's own `docker run --rm ...`
# (which must block in the foreground so its `mv .build $R4D_KEY` only publishes a finished
# build) passes through untouched.
#
# Exit codes: 3 llama running, 4 pin mismatch, 5 no container, 6 died, 7 timeout, 8 no kv_tokens,
#             9 engine init failed (prints ENGINE_INIT_FAILED estimate=N; container removed, no crash loop)
# Knobs (env): SPEC_METHOD=mtp|dflash SPEC=8 MAXLEN=194560 MAXSEQS=1 CHUNK=8192 GPU_UTIL=0.98 KV_MEM=8600000000 (bytes; auto = profile)
#              EXTRA='<extra vllm serve flags>' EXTRA_ENV='-e X=1 -e Y=2' (raw docker -e flags, word-split)
set -e
SELF_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SELF_DIR/bin/env.sh"
UP=/mnt/user/appdata/vllm-radiance-next   # NEXT: evaluation tree at the new fork commit
PIN_COMMIT=1e82407
export PATH=/mnt/user/appdata/llama-gemma31b/hostshim:$PATH   # host python3 shim (new serve script needs python3 on the host)
# baked image (bake-image.sh): the fork prologue + our local patches + the hipcc kernels pre-applied, tagged per pin. Preferred
# automatically when present for THIS pin; IMAGE=<tag> still overrides; absent -> the stock image + patches at boot (2026-09-17)
BAKED_IMAGE="vllm-radiance:0.9.3-baked-$PIN_COMMIT"
if [ -z "${IMAGE:-}" ] && docker image inspect "$BAKED_IMAGE" >/dev/null 2>&1; then IMAGE=$BAKED_IMAGE; fi
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
NAME=vllm-qwen38
PORT=1246
export MODELS=/mnt/user/Models/vllm-radiance HF_CACHE=/mnt/user/Models/hf-cache RUNTIME=docker TP=1
export CACHE=${CACHE:-$UP/persist/cache} R4D_CACHE=${R4D_CACHE:-$UP/persist/libr4d}   # CACHE overridable: a config that changes the compiled graph (e.g. RADIANCE_SKINNY_GEMM=all) gets its own dir
# Defaults = the MTP production config (2026-09-08): the in-checkpoint MTP head (needs vllm-patches/patch_quark_fp8_dynamic_token.py
# installed in vllm-radiance/local/), SPEC must be a POWER OF TWO in mtp mode (radiance n-gram matcher), single slot, KV pool PINNED
# (vLLM's profiler drifts run to run and its 'estimated maximum model length' is ~8k optimistic), MAXLEN from the strict block formula:
#   blocks/request = cdiv(MAXLEN, 1616) [17 attn layers incl. MTP, 1 group] + 3 x (2 + SPEC) [GDN groups]  <=  KV_MEM / 56.3 MB
# dflash fallback (151k ctx, ~9% faster decode below 100k): SPEC_METHOD=dflash SPEC=5 KV_MEM=6400000000 MAXLEN=151552
# 2026-09-09 measured (vllm-batch-bench.jsonl on .200): SPEC 8->4 = -12% decode @8k but +7% @100k and the KV pool grows
# 194,560 -> 212,756 tokens (GDN state blocks per seq 3x(2+SPEC): 30 -> 18); MAXLEN raised to 212,480 to spend it. MAXSEQS=2 =
# continuous batching ~1.7x aggregate at -10-12% per-seq; the pool is shared dynamically so a lone session still gets all of it.
# 2026-09-16: lm_head requantized to FP8 per-channel (custom-quant/fp8_lmhead.py, served via local/patch_quark_lmhead_fp8.py):
# -1.19 GiB of weights -> KV_MEM 8.6e9 -> 9.7e9, pool 215,313 -> 247,197 tokens, MAXLEN 212,480 -> 245,760. Measured: wikitext PPL
# 6.1000 -> 6.1044 (+0.07%), decode/acceptance unchanged (drafter reads the fp8 head directly), a 244,827-token request served,
# NIAH 8/8 at 215k depth. SNAP=<the old dir> KV_MEM=8600000000 MAXLEN=212480 restores the bf16-head checkpoint.
export SNAP=${SNAP:-/mnt/user/Models/vllm-radiance/Qwen3.8-27B-MXFP4-mtpfp8-lmfp8}
export SPEC_METHOD=${SPEC_METHOD:-mtp} SPEC=${SPEC:-4} MAXLEN=${MAXLEN:-245760} MAXSEQS=${MAXSEQS:-2}
# 2026-09-10: KV offloading tiers (measured: an evicted 100k prefix restores in 4.9 s instead of a 46 s re-prefill,
# and survives an engine restart from the NVMe tier). RAM tier = pinned /dev/shm mmap (8 GB; a 100k prompt is ~7 GB
# for this hybrid model, so RAM alone is only a staging buffer); NVMe tier at persist/cache/kv-fs has NO built-in
# cap -> the kv-fs-prune User Script keeps it under CAP_GB by LRU (needs local/patch_kvfs_touch_lru.py).
# DISABLED BY DEFAULT since 2026-09-10 (same day): in 45 min of agent work the tiers served 13% of GPU misses (~110 s of
# prefill) but wrote 73 GB to the shared 970 EVO Plus (~2.3 TB/day = the drive's remaining rating in months), and a
# RAM-only tier is useless below ~22 GB on this box. Opt in for a run with:  EXTRA="$KV_OFFLOAD_DEFAULT" ./deploy-vllm-qwen38.sh
# (then re-enable the kv-fs-prune User Script). PYTHONHASHSEED stays fixed so filenames are stable when it is used.
KV_OFFLOAD_DEFAULT='--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":8000000000,"block_size":1616,"secondary_tiers":[{"type":"fs","root_dir":"/cache/kv-fs"}]}}'
export CHUNK=${CHUNK:-8192} GPU_UTIL=${GPU_UTIL:-0.98} KV_MEM=${KV_MEM:-9700000000} EXTRA=${EXTRA:-}   # 9.7e9 since the fp8 lm_head (was 8.6e9)
# server-side sampling defaults (clients like omp send their own per request; this covers everything else).
# THINKING profile from the model card = temp 1.0 / top-p 0.95 / top-k 20 / min-p 0, same as deploy-qwen38-27b.sh.
GEN_CONFIG=${GEN_CONFIG:-'{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0}'}
export IMAGE NAME PORT
EXTRA_ENV=${EXTRA_ENV:--e PYTHONHASHSEED=0}   # fixed seed: KV-offload fs-tier block filenames must be stable across restarts

if docker ps --format '{{.Names}}' | grep -qx llama-qwen38; then
  echo "refusing: llama-qwen38 (llama.cpp production) is running; use switch-engine.sh vllm after a human OK"; exit 3
fi
cur=$(git -c safe.directory=$UP -C $UP rev-parse --short HEAD)
[ "$cur" = "$PIN_COMMIT" ] || { echo "refusing: $UP is at $cur, pinned $PIN_COMMIT (update the pin deliberately)"; exit 4; }
mkdir -p "$CACHE" "$R4D_CACHE"

# DRY_RUN=1 must never touch the live container (2026-09-07: an unguarded rm here killed production during a dry run)
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: printing the launcher argv only; $NAME is left untouched"
else
  docker rm -f $NAME >/dev/null 2>&1 || true
  for i in $(seq 1 20); do docker ps -a --format '{{.Names}}' | grep -qx $NAME || break; sleep 0.5; done
fi

# docker shim: rewrites ONLY the launcher's server-container `docker run ...` argv into a
# long-lived container, without ever reconstructing it as text (see QUOTING RISK above). Every
# other docker subcommand and docker-run invocation the launcher makes passes through to the
# real binary unchanged.
SHIM_DIR=$(mktemp -d)
trap 'rm -rf "$SHIM_DIR"' EXIT
REAL_DOCKER=$(command -v docker) || { echo "docker not on PATH"; exit 5; }
# Assumptions this shim makes about the pinned launcher (serve-mxfp4.sh @ $PIN_COMMIT) -- a
# pin bump must re-check these:
#   (re-verified 2026-09-15 at 92eed82: still exactly two calls, --privileged only on the server call, patch_quark_mxfp4.py still
#    the first patch line the shim hooks; launcher now defaults --tool-call-parser qwen3_coder (same parser as qwen3_xml) and
#    VLLM_NO_USAGE_STATS=1; new patch_nvfp4_mxfp4.py is inert unless RADIANCE_NVFP4_MXFP4=1)
#   1. It issues exactly two "docker run" invocations:
#        (a) the one-time libr4d auto-build step (~serve-mxfp4.sh:502-503):
#            "$RUNTIME run --rm --entrypoint bash -v ...:/work:z -w /work "$IMAGE" -c ./build.sh"
#            guarded by `if [ ! -f "$R4D_CACHE/$R4D_KEY/r4d.so" ]` (~serve-mxfp4.sh:493). This
#            MUST run to completion in the foreground (no -d) so the following
#            `mv "$R4D_CACHE/.build" "$R4D_CACHE/$R4D_KEY"` only ever publishes a finished
#            build -- forcing -d here would let the mv publish a half-built libr4d, orphan the
#            build container as --restart unless-stopped, and leave the server silently
#            falling back to the image's stock (NaN-producing) kernel.
#        (b) the server invocation (~serve-mxfp4.sh:748):
#            "$RUNTIME run "${RT_FLAGS[@]}" --name "$NAME" --privileged --ipc=host --network=host ..."
#            This is the one we want turned into a long-lived container.
#   2. Call (a) always carries --rm and never --privileged; call (b) never carries --rm and is
#      the ONLY call that carries --privileged. The shim uses --privileged as the positive
#      identifier for "this is the server call" and rewrites only that one; every other
#      "docker run" (currently just call (a)) is passed through byte-for-byte unchanged.
# NOTE: EXTRA_ENV below is spliced into this heredoc UNQUOTED at shim-*generation* time (here,
# now), so its value becomes literal shell source inside the generated shim script -- not
# merely a word-split argument list at shim run time. Shell metacharacters in EXTRA_ENV
# (;, $(...), backticks, quotes) would execute when the shim runs. Only ever pass trusted,
# operator-supplied values through EXTRA_ENV.
cat > "$SHIM_DIR/docker" <<SHIMEOF
#!/bin/bash
if [ "\$1" = run ]; then
  shift
  is_server=0
  for a in "\$@"; do [ "\$a" = --privileged ] && is_server=1; done
  if [ "\$is_server" = 1 ]; then
    ARGS=()
    for a in "\$@"; do [ "\$a" = --rm ] && continue; ARGS+=("\$a"); done
    # rewrite the launcher's --override-generation-config value with ours (GEN_CONFIG, baked in at generation time)
    OUT=(); rep=0; for a in "\${ARGS[@]}"; do if [ "\$rep" = 1 ]; then OUT+=('$GEN_CONFIG'); rep=0; continue; fi; OUT+=("\$a"); [ "\$a" = --override-generation-config ] && rep=1; done
    # our local patches (/patches/local/*.py = vllm-radiance/local/ on the host, untracked): run them inside the
    # container right after the launcher's first patch, by rewriting the -lc script argument. No-op when the dir is empty.
    # skip the ~1-3 min hipcc of the MXFP4 kernel when the image already carries it (baked image); no-op on the stock image
    HIPCC_GUARD='[ -e "\$SP"/radiance_mxfp4_fp8.so ] || hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201'
    LP_HOOK='python3 patch_quark_mxfp4.py; for lp in /patches/local/*.py; do [ -e "\$lp" ] && echo "[local patch] \$lp" && python3 "\$lp"; done'
    OUT2=(); prev=""; for a in "\${OUT[@]}"; do
      if [ "\$prev" = -lc ]; then a="\${a/python3 patch_quark_mxfp4.py/"\$LP_HOOK"}"; a="\${a/hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201/"\$HIPCC_GUARD"}"; fi
      OUT2+=("\$a"); prev="\$a"; done
    # persistent aiter JIT dir: aiter ignores AITER_ROOT_DIR and rebuilds module_aiter_core (~270 s) into the container
    # overlay on EVERY boot; AITER_JIT_DIR is honoured for both the build dir and the finished .so (2026-09-17)
    exec "$REAL_DOCKER" run -d --restart unless-stopped --ulimit nofile=65536:65536 -v ${AITER_DIR:-$UP/persist/aiter-jit}:/aiter-jit -e AITER_JIT_DIR=/aiter-jit $EXTRA_ENV "\${OUT2[@]}"
  fi
  exec "$REAL_DOCKER" run "\$@"
fi
exec "$REAL_DOCKER" "\$@"
SHIMEOF
chmod +x "$SHIM_DIR/docker"

(cd "$UP" && PATH="$SHIM_DIR:$PATH" bash serve-mxfp4.sh \
   --served-model-name qwen38-27b qwen38-27b-fast qwen38-27b-sub qwen38-27b-sub-fast) \
  || { echo "launcher did not start the container"; exit 5; }

[ "${DRY_RUN:-0}" = 1 ] && exit 0
# 2026-09-10: the KV offloading connector mmaps its RAM tier as /dev/shm/vllm_offload_<id>.mmap and a killed
# engine (which is how this wrapper replaces it) never unlinks it. Three orphans filled /dev/shm and the next
# engine died in madvise ("OSError: [Errno 14] Bad address"). Remove orphans that no process holds open.
for f in /dev/shm/vllm_offload_*.mmap; do
  [ -e "$f" ] || continue
  if ! lsof "$f" >/dev/null 2>&1; then rm -f "$f" && echo "[deploy] removed orphaned $f"; fi
done
for i in $(seq 1 240); do   # first start compiles Triton kernels: allow 20 min
  if curl -s -m 3 http://localhost:$PORT/health >/dev/null 2>&1 && curl -s -m 3 http://localhost:$PORT/v1/models | grep -q qwen38-27b; then break; fi
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "container died:"; docker logs --tail 30 $NAME; exit 6; }
  # engine-init failure (e.g. MAXLEN does not fit in KV memory): under --restart unless-stopped docker would
  # relaunch it forever and the loop above never sees it die. Detect, tear down, report the estimate (exit 9).
  if docker logs --tail 300 $NAME 2>&1 | grep -q "Engine core initialization failed" || [ "$(docker inspect $NAME --format '{{.RestartCount}}' 2>/dev/null || echo 0)" -ge 1 ]; then
    EST=$(docker logs $NAME 2>&1 | grep -oE 'estimated maximum model length is [0-9]+' | tail -1 | grep -oE '[0-9]+$' || true)
    FAILLOG=$JOBS_DIR/vllm-init-fail-$(date +%Y%m%d-%H%M%S).log
    docker logs $NAME > "$FAILLOG" 2>&1; echo "full log saved: $FAILLOG"
    grep -iE "error|usage:|invalid|traceback" "$FAILLOG" | grep -v ulimit | tail -8 | cut -c1-240
    docker rm -f $NAME >/dev/null 2>&1 || true
    echo "ENGINE_INIT_FAILED estimate=${EST:-?}"
    exit 9
  fi
  sleep 5
done
curl -s -m 3 http://localhost:$PORT/health >/dev/null || { echo "not healthy after 20 min"; docker logs --tail 30 $NAME; exit 7; }
KV=$(docker logs $NAME 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1 | tr -d , | grep -oE '[0-9]+' || true)
if [ -z "$KV" ]; then
  echo "kv_tokens not found in $NAME logs after health check passed (log tail follows):"
  docker logs --tail 30 $NAME
  exit 8
fi
echo "READY-VLLM [Qwen3.8-27B MXFP4] spec=$SPEC_METHOD/$SPEC maxlen=$MAXLEN maxseqs=$MAXSEQS chunk=$CHUNK util=$GPU_UTIL kv=$KV_MEM gen='$GEN_CONFIG' extra='$EXTRA' env='$EXTRA_ENV' kv_tokens=$KV img=$IMAGE repo=$PIN_COMMIT"
