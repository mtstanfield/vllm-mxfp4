#!/usr/bin/env python3
"""package.py - write a servable paroquant_mxfp4 checkpoint from a run.py --save-codes variant.

Same layout as paro-mxfp4-pod/build_hybrid_pod.py (the finalist): base shards mirrored, each quantized module replaced by
packed e2m1 weight + e8m0 weight_scale + its pairs/theta/channel_scales, every other tensor cast to fp16, no mtp.*
(Tower grafts the fp8 MTP head, fp8 lm_head and fp8 embed exactly as for the finalist).

  python3 package.py --variant gptq_ss --out /workspace/models/Qwen3.8-27B-PARO-MXFP4-v2
"""
import argparse, json, os, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import rot

WORK = Path(os.environ.get("WORK", "/workspace/pq2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--base", default="/workspace/models/Qwen3.8-27B-bf16")
    ap.add_argument("--paro", default="/workspace/models/Qwen3.8-27B-PARO")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, paro, out = Path(a.base), Path(a.paro), Path(a.out)
    vdir = WORK / "codes" / a.variant
    meta = json.load(open(vdir / "variant.json"))
    out.mkdir(parents=True, exist_ok=True)
    codes = safe_open(str(vdir / "codes.safetensors"), framework="pt")
    have = {k.rsplit(".", 1)[0] for k in codes.keys() if k.endswith(".weight_scale")}

    index = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    wmap, n_q = {}, 0
    for shard in sorted(set(index.values())):
        t = {}
        with safe_open(str(base / shard), framework="pt") as f:
            for name in f.keys():
                if name.startswith("mtp."):
                    continue
                mod = name[:-len(".weight")] if name.endswith(".weight") else None
                g = rot.LAYER_RE.search(mod) if mod else None
                short = f"{g.group(1)}.{g.group(2)}" if g else None
                if short in have and "visual" not in mod:
                    for leaf in ("weight", "weight_scale", "pairs", "theta", "channel_scales"):
                        t[f"{mod}.{leaf}"] = codes.get_tensor(f"{short}.{leaf}")
                    n_q += 1
                else:
                    x = f.get_tensor(name)
                    t[name] = x.to(torch.float16) if x.is_floating_point() else x
        save_file(t, str(out / shard), metadata={"format": "pt"})
        wmap.update({k: shard for k in t})
        print(f"  {shard}: quantized so far {n_q}", flush=True)
    assert n_q == len(have), (n_q, len(have))
    json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
    for fn in paro.iterdir():
        if fn.suffix in (".json", ".jinja", ".txt") and fn.name != "model.safetensors.index.json":
            shutil.copy(fn, out / fn.name)
    cfg = json.load(open(paro / "config.json"))
    cfg["quantization_config"] = {"quant_method": "paroquant_mxfp4", "format": "mxfp4", "bits": 4, "group_size": rot.GS,
                                  "mxfp4_block": 32, "krot": int(cfg["quantization_config"]["krot"]),
                                  "scale_rule": meta["quant"], "rotations": meta["rot_dir"] or "z-lab/Qwen3.8-27B-PARO",
                                  "variant": meta["name"]}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"DONE {out}: {n_q} quantized modules", flush=True)


if __name__ == "__main__":
    main()
