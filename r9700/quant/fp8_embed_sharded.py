#!/usr/bin/env python3
"""fp8_embed_sharded.py - in-place FP8 e4m3 per-ROW embed_tokens for a SHARDED checkpoint (derived from fp8_lmhead_sharded.py) whose shards may be
hard links (the grafted ParoQuant int5 dir). Same recipe/layout as fp8_lmhead.py (lm_head.weight F8_E4M3 [N,K] +
lm_head.weight_scale F32 [N]).

  python3 fp8_embed_sharded.py <dir>

- writes model-lmhead-fp8.safetensors with the two tensors
- rewrites the shard that held the bf16 lm_head WITHOUT it, into a new file that replaces the directory entry
  (os.replace), so a hard-linked original shard in another checkpoint dir is never modified
- index: lm_head.weight -> new shard, + lm_head.weight_scale; config.json: quantization_config["fp8_heads"] = --heads
  (the patched radiance_paroquant plugin routes those prefixes to Quark's fp8 per-channel scheme)
"""
import argparse, json, os, pathlib, struct
import torch

FP8_MAX = 448.0
SDT = {torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3"}

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def write_safetensors(path, tensors_or_copies, src=None, src_base=0):
    """tensors_or_copies: list of (name, dtype_str, shape, ('tensor', t) | ('copy', (s0, e0)))"""
    hdr, off = {}, 0
    entries = []
    for name, dt, shape, payload in tensors_or_copies:
        nb = payload[1].numel() * payload[1].element_size() if payload[0] == "tensor" else payload[1][1] - payload[1][0]
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        entries.append((payload, nb)); off += nb
    hdr["__metadata__"] = {"format": "pt"}
    blob = json.dumps(hdr).encode(); blob += b" " * ((8 - (len(blob) % 8)) % 8)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob))); fout.write(blob)
        fin = open(src, "rb") if src else None
        for payload, nb in entries:
            if payload[0] == "tensor":
                fout.write(payload[1].contiguous().view(torch.uint8).numpy().tobytes())
            else:
                a0, b0 = payload[1]; fin.seek(src_base + a0); left = b0 - a0
                while left:
                    c = fin.read(min(left, 32 << 20)); fout.write(c); left -= len(c)
        if fin: fin.close()
    os.replace(tmp, path)
    return off

NAME = "model.language_model.embed_tokens.weight"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    idx_p = d / "model.safetensors.index.json"
    idx = json.load(open(idx_p)); wm = idx["weight_map"]
    shard = wm[NAME]; sp = d / shard
    hdr, base = read_header(sp)
    m = hdr[NAME]
    TDT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    assert m["dtype"] in TDT, m["dtype"]
    s0, e0 = m["data_offsets"]
    with open(sp, "rb") as f:
        f.seek(base + s0); buf = bytearray(f.read(e0 - s0))
    w = torch.frombuffer(buf, dtype=TDT[m["dtype"]]).reshape(m["shape"]).float()
    amax = w.abs().amax(dim=1).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    q = (w / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    rel = ((q.float() * scale.unsqueeze(1) - w).norm() / w.norm()).item()
    print(f"embed_tokens {tuple(w.shape)} from {shard} (nlink={os.stat(sp).st_nlink}): fp8 per-channel rel err {rel:.4f}; "
          f"{(e0-s0)/2**30:.2f} GiB -> {(q.numel()+scale.numel()*4)/2**30:.2f} GiB")
    del w, buf

    new_shard = "model-embed-fp8.safetensors"
    n1 = write_safetensors(d / new_shard, [(NAME, "F8_E4M3", list(q.shape), ("tensor", q)),
                                           (NAME + "_scale", "F32", list(scale.shape), ("tensor", scale))])
    # rewrite the old shard without lm_head.weight (new inode; the hard-linked original is untouched)
    keep = [(k, hdr[k]["dtype"], hdr[k]["shape"], ("copy", tuple(hdr[k]["data_offsets"])))
            for k in hdr if k not in ("__metadata__", NAME)]
    # write_safetensors replaces `sp` while reading from it: read from a temporary hard link to the old inode
    old = str(sp) + ".oldlink"; os.link(sp, old)
    try:
        n2 = write_safetensors(sp, keep, src=old, src_base=base)
    finally:
        os.unlink(old)
    print(f"rewrote {shard} without embed_tokens ({len(keep)} tensors, {n2/2**30:.2f} GiB); nlink now {os.stat(sp).st_nlink}")

    wm[NAME] = new_shard; wm[NAME + "_scale"] = new_shard
    md = idx.setdefault("metadata", {})
    md["total_size"] = md.get("total_size", 0) - (e0 - s0) + n1
    _tmp = str(idx_p) + ".tmp"; json.dump(idx, open(_tmp, "w"), indent=2); os.replace(_tmp, idx_p)   # hard-link safe
    cfg_p = d / "config.json"; cfg = json.load(open(cfg_p))
    qc = cfg["quantization_config"]
    qc["fp8_embed"] = True
    _tmp = str(cfg_p) + ".tmp"; json.dump(cfg, open(_tmp, "w"), indent=2); os.replace(_tmp, cfg_p)   # hard-link safe
    print("config fp8_embed = True")

if __name__ == "__main__":
    main()
