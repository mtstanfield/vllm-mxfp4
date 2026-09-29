#!/usr/bin/env python3
"""finish.py - turn a package.py output into a drop-in replacement for the finalist checkpoint, on the pod.

The finalist's fp8 MTP head, fp8 lm_head and fp8 row embed do not depend on the body quant, so its three extra shards
are reused as-is (uploaded to --extras): the bf16 lm_head/embed are removed from the body shards, the extra shards are
added, the index merged, and config gets the same fp8_heads/fp8_embed keys. Then every tensor name, dtype and shape is
checked against the finalist's manifest: the server loads the result exactly like the finalist or this exits non-zero.

  python3 finish.py --pkg /workspace/models/v2-gptq_ss --extras /workspace/pq2/finalist-extras --out /workspace/models/FINAL
"""
import argparse, json, os, shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

DROP = ("lm_head.weight", "model.language_model.embed_tokens.weight")
EXTRA = ("model-mtp.safetensors", "model-lmhead-fp8.safetensors", "model-embed-fp8.safetensors")


def header(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True); ap.add_argument("--extras", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", default=None, help="refit model-mtp.safetensors (mtp_refit.py) instead of the finalist's")
    a = ap.parse_args()
    pkg, ext, out = Path(a.pkg), Path(a.extras), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wmap = {}
    for shard in sorted(set(json.load(open(pkg / "model.safetensors.index.json"))["weight_map"].values())):
        keys = list(header(pkg / shard))
        if any(k in DROP for k in keys):
            with safe_open(str(pkg / shard), framework="pt") as f:
                t = {k: f.get_tensor(k) for k in keys if k not in DROP}
            save_file(t, str(out / shard), metadata={"format": "pt"})
        else:
            if (out / shard).exists():
                (out / shard).unlink()
            os.link(pkg / shard, out / shard) if os.stat(pkg).st_dev == os.stat(out).st_dev else shutil.copy(pkg / shard, out / shard)
        wmap.update({k: shard for k in keys if k not in DROP})
    for e in EXTRA:
        src = Path(a.mtp) if (a.mtp and e == "model-mtp.safetensors") else ext / e
        shutil.copy(src, out / e)
        wmap.update({k: e for k in header(out / e)})
    print(f"MTP head: {'refit ' + a.mtp if a.mtp else 'finalist'}")
    json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
    for fn in pkg.iterdir():
        if fn.suffix in (".json", ".jinja", ".txt") and fn.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy(fn, out / fn.name)
    cfg = json.load(open(pkg / "config.json"))
    cfg["quantization_config"]["fp8_heads"] = ["*lm_head", "mtp.*"]
    cfg["quantization_config"]["fp8_embed"] = True
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    # ---- verify against the finalist
    fin = json.load(open(ext / "finalist-manifest.json"))
    mine = {}
    for shard in set(wmap.values()):
        for k, v in header(out / shard).items():
            mine[k] = [v["dtype"], v["shape"]]
    bad = sorted(set(fin) ^ set(mine))
    bad += [k for k in set(fin) & set(mine) if fin[k][:2] != mine[k]]
    print(f"{len(mine)} tensors, finalist {len(fin)}; mismatches: {len(bad)}")
    for k in bad[:20]:
        print("  ", k, fin.get(k, "MISSING in finalist")[:2] if k in fin else "MISSING in finalist", mine.get(k, "MISSING here"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
