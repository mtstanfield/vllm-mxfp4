#!/usr/bin/env python3
"""fp8_lmhead.py - requantize a Quark-MXFP4 checkpoint's bf16 lm_head to FP8 e4m3 per-output-channel (same recipe
and tensor layout as fp8_mtp.py uses for the MTP head), streaming the 18 GB single-file checkpoint without loading it.

Why: lm_head.weight is 248320 x 5120 bf16 = 2.37 GiB, the second-largest tensor after the decoder. FP8 halves it
(-1.19 GiB ~= +30k tokens of KV at this engine's ~40 KB/token). The fork's NVFP4 route already serves an FP8
per-channel lm_head and its int2 draft/verify heads read it directly.

Tensor layout written:  lm_head.weight  F8_E4M3 [N, K]   +   lm_head.weight_scale  F32 [N]
Config written (default --config-mode quark-layer): remove "lm_head" from quantization_config.exclude and add
  layer_quant_config["lm_head"] = the FP8_CFG block (weight fp8_e4m3 per_channel static, input fp8_e4m3 dynamic per_tensor)
  -- identical to how the MTP projections are declared. Use --config-mode none to write only the tensors.

  python3 fp8_lmhead.py <src-dir> <dst-dir> [--config-mode quark-layer|none]
"""
import argparse, json, pathlib, shutil, struct
import torch

FP8_MAX = 448.0
TDT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
SDT = {torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3"}

def spec(dtype, dynamic, qscheme, ch_axis):
    return {"block_size": None, "ch_axis": ch_axis, "dtype": dtype, "enable_buffer_reuse": False,
            "group_size": None, "is_dynamic": dynamic, "is_scale_quant": False,
            "max_input_numel": 4194304, "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver" if qscheme == "per_channel" else "PerTensorMinMaxObserver",
            "qscheme": qscheme, "round_method": "half_even", "scale_calculation_mode": None,
            "scale_format": None, "scale_type": "float", "symmetric": True}
FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
           "weight": spec("fp8_e4m3", False, "per_channel", 0),
           "input_tensors": spec("fp8_e4m3", True, "per_tensor", -1)}

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--config-mode", default="quark-layer", choices=["quark-layer", "none"])
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    sf = src / "model.safetensors"
    hdr, base = read_header(sf)
    name = "lm_head.weight"
    m = hdr[name]; assert m["dtype"] == "BF16", m["dtype"]
    s0, e0 = m["data_offsets"]
    with open(sf, "rb") as f:
        f.seek(base + s0); buf = bytearray(f.read(e0 - s0))
    w = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(m["shape"]).float()
    amax = w.abs().amax(dim=1).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    q = (w / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    rel = ((q.float() * scale.unsqueeze(1) - w).norm() / w.norm()).item()
    print(f"lm_head {tuple(w.shape)}: fp8 per-channel rel err {rel:.4f} (the MTP head's was ~0.026); "
          f"{(e0-s0)/2**30:.2f} GiB -> {(q.numel()+scale.numel()*4)/2**30:.2f} GiB")
    new = {name: q, "lm_head.weight_scale": scale}

    names = [k for k in hdr if k != "__metadata__"]
    out_hdr, off, plan = {}, 0, []
    for k in names:
        if k in new:
            t = new[k]; nb = t.numel() * t.element_size()
            out_hdr[k] = {"dtype": SDT[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
            plan.append(("new", k, nb)); off += nb
        else:
            mm = hdr[k]; nb = mm["data_offsets"][1] - mm["data_offsets"][0]
            out_hdr[k] = {"dtype": mm["dtype"], "shape": mm["shape"], "data_offsets": [off, off + nb]}
            plan.append(("copy", k, nb)); off += nb
    for k, t in new.items():
        if k not in out_hdr:
            nb = t.numel() * t.element_size()
            out_hdr[k] = {"dtype": SDT[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
            plan.append(("new", k, nb)); off += nb
    out_hdr["__metadata__"] = {"format": "pt"}
    blob = json.dumps(out_hdr).encode(); blob += b" " * ((8 - (len(blob) % 8)) % 8)
    outf = dst / "model.safetensors"
    with open(sf, "rb") as fin, open(outf, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob))); fout.write(blob)
        for kind, k, nb in plan:
            if kind == "new":
                fout.write(new[k].contiguous().view(torch.uint8).numpy().tobytes())
            else:
                a0, b0 = hdr[k]["data_offsets"]; fin.seek(base + a0); left = b0 - a0
                while left:
                    c = fin.read(min(left, 32 << 20)); fout.write(c); left -= len(c)
    print(f"wrote {outf} ({outf.stat().st_size / 2**30:.2f} GiB)")

    cfg = json.loads((src / "config.json").read_text())
    if a.config_mode == "quark-layer":
        qc = cfg["quantization_config"]
        qc["exclude"] = [e for e in qc["exclude"] if e not in ("lm_head", "lm_head.weight")]
        # pattern, not a bare name: inside the multimodal wrapper the head's vLLM prefix is 'language_model.lm_head'
        # (Quark matches layer_quant_config keys with fnmatch), and patch_quark_lmhead_fp8.py gates on that match.
        qc.setdefault("layer_quant_config", {})["*lm_head"] = FP8_CFG
        print(f"config: lm_head removed from exclude ({len(qc['exclude'])} left), layer_quant_config now {len(qc['layer_quant_config'])} entries")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if (src / n).exists(): shutil.copy(src / n, dst / n)

if __name__ == "__main__":
    main()
