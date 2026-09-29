# Shared paths for the R9700 llama.cpp dev/deploy scripts (sourced, not executed).
# Change SRC here only; every script reads it.
REPO_DIR=/mnt/user/appdata/llama-gemma31b          # this git repo (deploy scripts, patches, benches)
SRC=/mnt/cache/appdata/llama.cpp              # llama.cpp source tree (upstream master + local patches applied); under the NFS-exported appdata share since 2026-09-05
DEV_BUILD=/mnt/cache/llama-dev-build               # persistent cmake build dir for the dev container
DEV_CCACHE=/mnt/cache/llama-dev-ccache
DEV_IMAGE=llama.cpp:dev-vulkan
DEV_NAME=llama-dev
DEV_PORT=1247                                      # dev llama-server (never the production :1246)
PROD_IMAGE=llama.cpp:server-vulkan-mtp
EXP_IMAGE=llama.cpp:server-vulkan-mtp-exp
GPU_DEV=/dev/dri/renderD128
MODELS=/mnt/user/Models
JOBS_DIR=$REPO_DIR/jobs
GITC="git -c safe.directory=$SRC"
CMAKE_FLAGS="-DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
# Patch sets. PROD_PATCHES are what the canonical tree carries (applied, uncommitted) and what
# rebuild-llama-mtp.sh bakes into the production image. EXP_PATCHES are applied by rebuild-llama-exp.sh
# on top for the experimental images and must NOT be left applied in the tree between builds.
PROD_PATCHES="$REPO_DIR/mtp-draft-guard.patch $REPO_DIR/slot-pin-cache.patch"
EXP_PATCHES="$REPO_DIR/vk-fa-smallrows-knob.patch $REPO_DIR/vk-fa-int8k.patch"

# r9700-hip fork (HIP/ROCm parity project; spec docs/superpowers/specs/2026-09-26-hip-rdna4-parity-design.md)
HIP_SRC=/mnt/cache/appdata/llama-hip               # git worktree of $SRC's repo, branch r9700-hip
HIP_BUILD=/mnt/cache/llama-hip-dev-build
HIP_CCACHE=/mnt/cache/llama-hip-dev-ccache
HIP_IMAGE=llama.cpp:dev-hip
HIP_NAME=llama-hip-dev
HIP_PORT=1248                                      # dev HIP llama-server (1246 prod, 1247 Vulkan dev, 1249 vision sidecar)
HIP_CMAKE_FLAGS="-DGGML_HIP=ON -DGGML_HIP_R9700=ON -DGPU_TARGETS=gfx1201 -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache"
