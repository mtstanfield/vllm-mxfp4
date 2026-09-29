#!/usr/bin/env python3
"""splice_mix.py - per-layer MIXED checkpoint: AMD Quark MXFP4 body (+ its fp8 lm_head / fp8 MTP head) with chosen decoder
layers replaced by the ParoQuant int5 layers, from the two checkpoints we already have. No quantization is computed.

  python3 splice_mix.py --quark /models/Qwen3.8-27B-MXFP4-mtpfp8-lmfp8 --int5 /models/Qwen3.8-27B-PARO-int5 \
                        --layers 3,7,11 --out /models/Qwen3.8-27B-MIX-attn16

Output: model-quark.safetensors (the Quark file streamed WITHOUT the swapped layers' tensors), one model-int5-L<N>.safetensors
per swapped layer (all int5 tensors of that layer), merged index, config.json = Quark config with quantization_config =
int5 quantization_config + {"pq_layers": [fnmatch patterns], "quark_config": <the Quark quantization_config>}; the patched
radiance_paroquant plugin routes every prefix NOT matching pq_layers to an embedded QuarkConfig built from quark_config.
"""
import argparse, json, os, pathlib, shutil, struct
from safetensors import safe_open
from safetensors.torch import save_file

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quark", required=True); ap.add_argument("--int5", required=True)
    ap.add_argument("--layers", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers = sorted({int(x) for x in a.layers.split(",") if x.strip()})
    q, i5, out = pathlib.Path(a.quark), pathlib.Path(a.int5), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pfx = {n: f"model.language_model.layers.{n}." for n in layers}
    def swapped(name): return any(name.startswith(p) for p in pfx.values())

    # 1. Quark file minus the swapped layers (streamed)
    hdr, base = read_header(q / "model.safetensors")
    keep = [k for k in hdr if k != "__metadata__" and not swapped(k)]
    dropped = [k for k in hdr if k != "__metadata__" and swapped(k)]
    out_hdr, off, plan = {}, 0, []
    for k in keep:
        m = hdr[k]; nb = m["data_offsets"][1] - m["data_offsets"][0]
        out_hdr[k] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + nb]}; plan.append((k, nb)); off += nb
    out_hdr["__metadata__"] = {"format": "pt"}
    blob = json.dumps(out_hdr).encode(); blob += b" " * ((8 - (len(blob) % 8)) % 8)
    qf = "model-quark.safetensors"; wm = {}
    with open(q / "model.safetensors", "rb") as fin, open(out / (qf + ".tmp"), "wb") as fout:
        fout.write(struct.pack("<Q", len(blob))); fout.write(blob)
        for k, nb in plan:
            a0 = hdr[k]["data_offsets"][0]; fin.seek(base + a0); left = nb
            while left:
                c = fin.read(min(left, 64 << 20)); fout.write(c); left -= len(c)
            wm[k] = qf
    os.replace(out / (qf + ".tmp"), out / qf)
    print(f"quark file: kept {len(keep)} tensors, dropped {len(dropped)} (layers {layers}), {off/2**30:.2f} GiB")

    # 2. int5 tensors of the swapped layers, one file per layer
    idx = json.load(open(i5 / "model.safetensors.index.json"))["weight_map"]
    by_shard = {}
    for name, shard in idx.items():
        if swapped(name): by_shard.setdefault(shard, []).append(name)
    per_layer = {n: {} for n in layers}
    for shard, names in sorted(by_shard.items()):
        with safe_open(i5 / shard, "pt", device="cpu") as f:
            for name in names:
                n = int(name.split(".layers.")[1].split(".")[0]); per_layer[n][name] = f.get_tensor(name).contiguous()
    total = 0
    for n in layers:
        fn = f"model-int5-L{n}.safetensors"
        assert per_layer[n], f"no int5 tensors for layer {n}"
        save_file(per_layer[n], str(out / fn), metadata={"format": "pt"})
        for k, t in per_layer[n].items(): wm[k] = fn; total += t.numel() * t.element_size()
        per_layer[n] = None
    print(f"int5 layers: {len(layers)} files, {total/2**30:.2f} GiB")
    json.dump({"metadata": {"total_size": off + total}, "weight_map": wm}, open(out / "model.safetensors.index.json", "w"), indent=2)

    # 3. config: Quark config.json as base, quantization_config = int5's + routing info
    cfg = json.load(open(q / "config.json")); qqc = cfg["quantization_config"]
    iqc = json.load(open(i5 / "config.json"))["quantization_config"]
    mix = dict(iqc); mix["pq_layers"] = [f"model.language_model.layers.{n}.*" for n in layers]; mix["quark_config"] = qqc
    cfg["quantization_config"] = mix
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for nme in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if (q / nme).exists(): shutil.copy(q / nme, out / nme)
    print("done:", out, "pq_layers:", len(mix["pq_layers"]))

if __name__ == "__main__":
    main()
