#!/bin/bash
# Build the paroquant kernel module (radiance_paroquant_kernel.so) from paroquant/radiance_paroquant.hip
# with the serving image compiler, so radiance_gemm.py can use its pq_skinny_bf16 as the skinny-GEMM
# fallback (our libr4d b9e42ab-rx6 predates gemm_bf16_nt_m64). local/patch_paroquant_kernel.py installs
# it at container start. No GPU needed to compile. Re-run after a pin bump that touches paroquant/.
set -e
UP=/mnt/user/appdata/vllm-radiance
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
docker run --rm --entrypoint bash -v $UP:/patches "$IMAGE" -lc "cd /patches/paroquant && hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 \$(python3 -m pybind11 --includes) radiance_paroquant.hip -o /patches/persist/paroquant/radiance_paroquant_kernel.so && git -C /patches rev-parse --short HEAD > /patches/persist/paroquant/built-from.txt 2>/dev/null || true; ls -la /patches/persist/paroquant/"
