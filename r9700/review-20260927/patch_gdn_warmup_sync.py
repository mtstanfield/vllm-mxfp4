#!/usr/bin/env python3
"""Local patch (ours, 2026-09-28): synchronize before the empty_cache() at the end of vLLM 0.29's GDN prefill-kernel
warm-up (QwenGatedDeltaNetAttention._warmup_prefill_kernels).

The warm-up runs INSIDE the model forward -- once per GDN layer, on the boot profile run -- and ends with
torch.accelerator.empty_cache(). The host is far ahead of the GPU there: tensors of earlier layers whose last kernel is
still queued/running are already free in the caching allocator, and empty_cache hands their segments back to the driver.
On ROCm that unmaps memory an in-flight kernel is reading -> "Memory Fault ... page-aligned address" (seen 2026-09-28 in
5 of 7 boots of configs with a slower prefill producer; never under AMD_SERIALIZE_KERNEL=3). A device synchronize before
the empty_cache closes the window; it costs nothing outside the one-time warm-up.

Gated: RADIANCE_LOCAL_GDN_WARMUP_SYNC=1 applies it; unset leaves the file untouched."""
import os
import sys
import sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_LOCAL_GDN_WARMUP_SYNC", "0") != "1":
    sys.exit(0)

F = (Path(sysconfig.get_paths()["purelib"]) / "vllm" / "model_executor" / "layers" / "mamba" / "gdn"
     / "qwen_gdn_linear_attn.py")
src = F.read_text()
if "patch_gdn_warmup_sync" in src:
    print("[gdn-warmup-sync] already applied")
    sys.exit(0)
OLD = (
    "        torch.accelerator.empty_cache()\n"
    "\n"
    "    def _forward_core_rocm(\n"
)
NEW = (
    "        torch.cuda.synchronize()   # patch_gdn_warmup_sync.py: no in-flight kernel may read what empty_cache unmaps\n"
    "        torch.accelerator.empty_cache()\n"
    "\n"
    "    def _forward_core_rocm(\n"
)
n = src.count(OLD)
if n != 1:
    print(f"[gdn-warmup-sync] anchor found {n} times, NOT applied", file=sys.stderr)
    sys.exit(0)
F.write_text(src.replace(OLD, NEW))
print("[gdn-warmup-sync] applied: synchronize before the GDN warm-up empty_cache")
