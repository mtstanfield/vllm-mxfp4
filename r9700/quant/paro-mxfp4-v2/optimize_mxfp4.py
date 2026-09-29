#!/usr/bin/env python3
"""optimize_mxfp4.py - re-train ParoQuant rotations (and optionally weights) against the MXFP4 grid, on our traces.

Runs z-lab's own layer-wise optimizer (paroquant.cli.optimize) with three patches:
  1. the fake-quantizer is OCP MXFP4 (block 32, power-of-two scales, STE) instead of int4 affine groups of 128;
     rotation groups stay 128 = what the checkpoint and the serving kernel use;
  2. every linear warm-starts from z-lab's TRAINED pairs/angles/channel scales (not random pairs at zero angle), so the
     optimizer's best-on-validation checkpointing can never end below z-lab's rotations;
  3. calibration/validation windows come from traces.npz (the omp agent sessions), not wikitext/c4/pile.
Output: one <layer>.<module>.pt per linear under --output-dir/<model name>/, consumed by run.py --rot-dir.

  python3 optimize_mxfp4.py --model /workspace/models/Qwen3.8-27B-bf16 \
    --params "channel_scales:0.05,angles:0.05" --epochs 3 --group-size 128 --n-bit 4 --num-rotations 8 \
    --skipped-modules linear_attn.in_proj_a linear_attn.in_proj_b \
    --datasets traces --val-dataset traces --train-size 512 --validation-size 32 --batch-size 16 --seqlen 2048 \
    --output-dir /workspace/opt --resume --seed 0
Env: PQ2_PARO (z-lab checkpoint dir), PQ2_TRACES (traces.npz), PQ2_MODE (ocp|search, default search).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx, rot                                                        # noqa: E402

import paroquant.cli.optimize as opt                                   # noqa: E402
from paroquant.optim import qlinear                                    # noqa: E402
from paroquant.kernels.cuda import scaled_pairwise_rotation            # noqa: E402

PARO = os.environ.get("PQ2_PARO", "/workspace/models/Qwen3.8-27B-PARO")
TRACES = os.environ.get("PQ2_TRACES", "/workspace/pq2/traces.npz")
MODE = os.environ.get("PQ2_MODE", "search")
_rp = rot.RotParams(PARO)
_seen = {"warm": 0, "cold": 0}


# ---- 1. MXFP4 fake quantizer inside the rotated basis ----------------------------------------------------------------
def _pseudo_quantize_mxfp4(self, weight):
    weight = weight * self.channel_scales
    weight = self.checkpointed(scaled_pairwise_rotation, weight, self.pairs_grouped, self.angles_grouped, None, self.group_size)
    weight = mx.fake_quant_ste(weight.float(), MODE).to(weight.dtype)
    weight = self.checkpointed(scaled_pairwise_rotation, weight, torch.flip(self.pairs_grouped, dims=[0]),
                               -torch.flip(self.angles_grouped, dims=[0]), None, self.group_size)
    return weight / self.channel_scales.view(1, -1)


qlinear.PseudoQuantizedLinear._pseudo_quantize = _pseudo_quantize_mxfp4


# ---- 2. warm start from z-lab's trained rotations --------------------------------------------------------------------
_orig_load_model = opt.load_model


def _load_model_tagged(*args, **kwargs):
    model = _orig_load_model(*args, **kwargs)
    for name, m in model.named_modules():
        g = rot.LAYER_RE.search(name)
        if g and isinstance(m, torch.nn.Linear):
            m._pq2_key = (int(g.group(1)), g.group(2))
    return model


opt.load_model = _load_model_tagged
_orig_init = qlinear.PseudoQuantizedLinear.__init__


def _init_warm(self, linear, rotation_pairs, channel_scales, *, group_size, n_bits, num_rotations):
    key = getattr(linear, "_pq2_key", None)
    if key in _rp.src:
        p = _rp.get(key, device="cuda")
        assert p["pairs"].shape[0] == num_rotations, (key, p["pairs"].shape, num_rotations)
        rotation_pairs = [p["pairs"].short(), p["theta"].float(), torch.zeros_like(p["theta"], dtype=torch.bool)]
        channel_scales = p["cs"].view(1, -1).to(torch.float16)        # PseudoQuantizedLinear keeps weight in fp16
        _seen["warm"] += 1
    else:
        _seen["cold"] += 1
        print(f"[pq2] no z-lab rotation for {key}: random init", flush=True)
    _orig_init(self, linear, rotation_pairs, channel_scales, group_size=group_size, n_bits=n_bits, num_rotations=num_rotations)


qlinear.PseudoQuantizedLinear.__init__ = _init_warm


# the random-pair search + kernel-format packing is replaced by the warm start; skip their slow python loops
def _dummy_pairs(w, group_size, num_rotations, num_pairs_factor, seed):
    k = w.shape[0] * group_size
    return [[(2 * i, 2 * i + 1) for i in range(k // 2)] for _ in range(num_rotations)]


def _dummy_kernel_data(pairs_group, angles_group, group_size=128):
    r, half = len(angles_group), angles_group[0].shape[0]
    pairs = (torch.arange(2 * half) % group_size).to(torch.short).repeat(r, 1)
    return pairs, torch.zeros(r, half), torch.zeros(r, half, dtype=torch.bool)


opt.get_random_rotation_pairs = _dummy_pairs
opt.transform_to_kernel_data = _dummy_kernel_data


# ---- 3. calibration windows from the agent traces --------------------------------------------------------------------
def _windows(n, block, offset):
    c = np.load(TRACES)["calib"]
    per = c.shape[1] // block
    rows = [torch.from_numpy(c[i // per, (i % per) * block:(i % per + 1) * block].astype(np.int64))
            for i in range(offset, offset + n)]
    assert len(rows) == n and all(len(r) == block for r in rows), "not enough calibration windows"
    return rows


MIX = float(os.environ.get("PQ2_MIX", "0"))


def _train_windows(datasets, *, tokenizer, n_samples, block_size, seed, split):
    import gen
    w = _windows(n_samples, block_size, 0)
    w = gen.mix(w, lambda k: gen.general_windows(tokenizer, k, block_size), MIX)
    print(f"[pq2] optimizer train windows: {len(w)} ({round(MIX * len(w))} from wikitext-2 train)", flush=True)
    return w


opt.get_mixed_calib_dataset = _train_windows
# validation = the LAST windows of the pool (never used for training at train_size <= pool - validation_size)
def _val_windows(data="traces", *, tokenizer, n_samples, block_size, seed, split):
    import gen
    w = _windows(n_samples, block_size, (np.load(TRACES)["calib"].size // block_size) - n_samples)
    # general part of validation from the wikitext-2 VALIDATION split (train feeds training, test feeds the eval)
    return gen.mix(w, lambda k: gen.general_windows(tokenizer, k, block_size, split="validation"), MIX)


opt.get_calib_dataset = _val_windows

if __name__ == "__main__":
    opt.main()
    print(f"[pq2] warm-started {_seen['warm']} linears, random-init {_seen['cold']}", flush=True)
