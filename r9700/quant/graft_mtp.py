#!/usr/bin/env python3
"""graft_mtp.py - give the published ParoQuant int5 checkpoint (no MTP head) the MTP head from our
Quark checkpoint, so vLLM-Radiance can run it with in-checkpoint MTP speculative decoding instead of the
external DFlash drafter.

  python3 graft_mtp.py --int5 /models/Qwen3.8-27B-PARO-int5 --head-from /models/Qwen3.8-27B-MXFP4-mtpfp8-lmfp8 \
                       --out /models/Qwen3.8-27B-PARO-int5-mtp [--head-dtype bf16|fp8]

- The 18 int5 shards are HARD-LINKED into --out (same filesystem, no copy). Nothing in them is touched.
- mtp.* tensors are read from --head-from (our checkpoint stores the 8 MTP projections as fp8 e4m3 +
  per-channel scale; norms bf16). With --head-dtype bf16 (default, phase 1) the projections are dequantized
  to bf16 so the ParoQuant plugin can leave them unquantized via RADIANCE_PQ_SKIP=.*mtp.* . With fp8 they are
  written as-is (needs the plugin's fp8-head patch, phase 2).
- A new shard model-mtp.safetensors is added and the index merged. config.json = the int5 config (it already
  carries mtp_num_hidden_layers=1); tokenizer/template files copied from the int5 dir.
"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int5", required=True); ap.add_argument("--head-from", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--head-dtype", default="bf16", choices=["bf16", "fp8"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.int5, "model.safetensors.index.json")
    idx = json.load(open(idx_path)); wm = dict(idx["weight_map"])
    shards = sorted(set(wm.values()))
    for s in shards:
        dst = os.path.join(a.out, s)
        if not os.path.exists(dst):
            os.link(os.path.join(a.int5, s), dst)
    print(f"linked {len(shards)} int5 shards")

    src = os.path.join(a.head_from, "model.safetensors")
    f = safe_open(src, "pt", device="cpu")
    names = [n for n in f.keys() if n.startswith("mtp.")]
    assert names, "no mtp.* tensors in --head-from"
    out = {}
    for n in names:
        t = f.get_tensor(n)
        if n.endswith(".weight_scale"):
            if a.head_dtype == "fp8": out[n] = t
            continue
        if t.dtype == torch.float8_e4m3fn:
            if a.head_dtype == "fp8":
                out[n] = t
            else:
                s = f.get_tensor(n[:-len(".weight")] + ".weight_scale").float()
                out[n] = (t.float() * s.unsqueeze(1)).to(torch.bfloat16)
        else:
            out[n] = t.to(torch.bfloat16) if t.is_floating_point() else t
    shard = "model-mtp.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.out, shard), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"wrote {shard}: {len(out)} tensors, {total/2**30:.2f} GiB, head dtype {a.head_dtype}")
    for k in out: wm[k] = shard
    idx["weight_map"] = wm
    idx.setdefault("metadata", {})["total_size"] = idx.get("metadata", {}).get("total_size", 0) + total
    json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.int5, "config.json")))
    tc = cfg.get("text_config", cfg)
    assert tc.get("mtp_num_hidden_layers", 0) >= 1, "int5 config lacks mtp_num_hidden_layers"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "crc32.txt"):
        p = os.path.join(a.int5, n)
        if os.path.exists(p): shutil.copy(p, os.path.join(a.out, n))
    print("done:", a.out)

if __name__ == "__main__":
    main()
