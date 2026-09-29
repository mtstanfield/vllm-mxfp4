#!/usr/bin/env python3
"""splice_mix3.py - three-way MIXED checkpoint: a SHARDED body checkpoint (e.g. rotation-MXFP4 `paroquant_mxfp4` with fp8 heads
already grafted) with chosen decoder layers replaced by the ParoQuant int5 layers. No quantization is computed.

  python3 splice_mix3.py --body /models/Qwen3.8-27B-PARO-MXFP4-mtpfp8-lmfp8 --int5 /models/Qwen3.8-27B-PARO-int5 \
                         --layers 3,7,... --out /models/Qwen3.8-27B-MIX3-attn16

Output: every body shard rewritten WITHOUT the swapped layers' tensors (unchanged shards are hard-linked), one
model-int5-L<N>.safetensors per swapped layer, merged index, config = body config with quantization_config =
int5 quantization_config + {"fp8_heads": <body's>, "pq_layers": [...], "body_config": <body quantization_config>,
"body_layers": ["model.language_model.layers.*"]}. The patched plugin routes: fp8_heads -> Quark fp8, pq_layers -> int5,
body_layers -> embedded ParoQuantMXFP4Config(body_config), everything else -> the int5 class defaults (visual fp16 etc.)."""
import argparse, json, os, pathlib, shutil
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True); ap.add_argument("--int5", required=True)
    ap.add_argument("--layers", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers = sorted({int(x) for x in a.layers.split(",") if x.strip()})
    body, i5, out = pathlib.Path(a.body), pathlib.Path(a.int5), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pfx = [f"model.language_model.layers.{n}." for n in layers]
    def swapped(name): return any(name.startswith(p) for p in pfx)

    bidx = json.load(open(body / "model.safetensors.index.json"))["weight_map"]
    wm = {}; dropped = 0; rewritten = 0; linked = 0
    for shard in sorted(set(bidx.values())):
        names = [k for k, s in bidx.items() if s == shard]
        keep = [k for k in names if not swapped(k)]
        if len(keep) == len(names):
            dst = out / shard
            if not dst.exists(): os.link(body / shard, dst)
            linked += 1
        else:
            t = {}
            with safe_open(body / shard, "pt", device="cpu") as f:
                for k in keep: t[k] = f.get_tensor(k).contiguous()
            save_file(t, str(out / shard), metadata={"format": "pt"}); rewritten += 1
            dropped += len(names) - len(keep)
        for k in keep: wm[k] = shard
    print(f"body: {linked} shards linked, {rewritten} rewritten, {dropped} tensors dropped (layers {layers})")

    iidx = json.load(open(i5 / "model.safetensors.index.json"))["weight_map"]
    by_shard = {}
    for name, shard in iidx.items():
        if swapped(name): by_shard.setdefault(shard, []).append(name)
    per_layer = {n: {} for n in layers}
    for shard, names in sorted(by_shard.items()):
        with safe_open(i5 / shard, "pt", device="cpu") as f:
            for name in names:
                n = int(name.split(".layers.")[1].split(".")[0]); per_layer[n][name] = f.get_tensor(name).contiguous()
    total = 0
    for n in layers:
        fn = f"model-int5-L{n}.safetensors"; assert per_layer[n], f"no int5 tensors for layer {n}"
        save_file(per_layer[n], str(out / fn), metadata={"format": "pt"})
        for k, t in per_layer[n].items(): wm[k] = fn; total += t.numel() * t.element_size()
        per_layer[n] = None
    print(f"int5 layers: {len(layers)} files, {total/2**30:.2f} GiB")
    json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(out / "model.safetensors.index.json", "w"), indent=2)

    cfg = json.load(open(body / "config.json")); bqc = cfg["quantization_config"]
    iqc = json.load(open(i5 / "config.json"))["quantization_config"]
    mix = dict(iqc)
    mix["fp8_heads"] = list(bqc.get("fp8_heads") or [])
    mix["pq_layers"] = [f"model.language_model.layers.{n}.*" for n in layers]
    mix["body_config"] = {k: v for k, v in bqc.items() if k not in ("fp8_heads", "pq_layers", "quark_config", "body_config", "body_layers")}
    mix["body_layers"] = ["model.language_model.layers.*"]
    cfg["quantization_config"] = mix
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for nme in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if (body / nme).exists(): shutil.copy(body / nme, out / nme)
    print("done:", out, "| body quant_method:", mix["body_config"].get("quant_method"), "| pq_layers:", len(mix["pq_layers"]))

if __name__ == "__main__":
    main()
