#!/usr/bin/env python3
"""Local patch (ours, 2026-09-28): drop the two per-GDN-layer glue kernels vLLM 0.29 added to the ROCm core entry
(QwenGatedDeltaNetAttention._forward_core_rocm, the use_aiter=True custom-op body):

  1. `core_attn_out.zero_()` -- redundant when the all-R4D path serves the step: its fused decode kernel zeroes the
     cudagraph pad rows itself and its other paths zero the tail in Python when RADIANCE_GDN_EMPTY_OUT=1 (radiance_gdn.py
     _zero_tail), and the Triton fallback in _forward_core zeroes under EMPTY_OUT too (patch_gdn_glue.py). So the
     unconditional fill is skipped only when ALL and EMPTY_OUT are both on.
  2. `z_out[:] = z` -- the gate slice copied out of the qkvz projection into the caller's buffer. The rotation-stream GDN
     forward (radiance_paroquant._rot_gdn_forward_hip, RADIANCE_LOCAL_GDN_NOCOPY=1) hands a 0-element z_out and reads z
     as a strided view of the projection instead (the gated-norm producer takes a row stride), so the copy is skipped
     when z_out is empty.

Decode: 96 launches per forward (2 x 48 GDN layers), ~3.5 us HIP-graph gap each plus the kernels. Prefill: a 100 MB fill
plus a 200 MB copy per GDN layer per 8k chunk. Byte-identical by construction (pad rows are never read by real rows).

Gated: RADIANCE_LOCAL_GDN_NOCOPY=1 applies it; unset leaves the file untouched (A/B-able)."""
import os
import sys
import sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_LOCAL_GDN_NOCOPY", "0") != "1":
    sys.exit(0)

F = (Path(sysconfig.get_paths()["purelib"]) / "vllm" / "model_executor" / "layers" / "mamba" / "gdn"
     / "qwen_gdn_linear_attn.py")
src = F.read_text()
if "patch_gdn_nocopy" in src:
    print("[gdn-nocopy] already applied")
    sys.exit(0)

OLD = (
    "        core_attn_out.zero_()\n"
    "        num_tokens_all = qkvz.shape[0]\n"
    "        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(\n"
    "            qkvz, ba, num_tokens_all\n"
    "        )\n"
    "        z_out[:] = z\n"
)
NEW = (
    "        # patch_gdn_nocopy.py: the all-R4D path writes (or zeroes) every row itself under EMPTY_OUT\n"
    "        if not (_radiance_gdn is not None and _radiance_gdn.ALL and _radiance_gdn.EMPTY_OUT):\n"
    "            core_attn_out.zero_()\n"
    "        num_tokens_all = qkvz.shape[0]\n"
    "        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(\n"
    "            qkvz, ba, num_tokens_all\n"
    "        )\n"
    "        if z_out.numel():   # patch_gdn_nocopy.py: an empty z_out = the caller reads z from qkvz\n"
    "            z_out[:] = z\n"
)
n = src.count(OLD)
if n != 1:
    print(f"[gdn-nocopy] anchor found {n} times, NOT applied", file=sys.stderr)
    sys.exit(0)
if "_radiance_gdn" not in src:
    print("[gdn-nocopy] no _radiance_gdn in the module (patch_r4d.py not applied?), NOT applied", file=sys.stderr)
    sys.exit(0)
F.write_text(src.replace(OLD, NEW))
print("[gdn-nocopy] applied: core_attn_out fill gated on ALL+EMPTY_OUT, z copy skipped for an empty z_out")
