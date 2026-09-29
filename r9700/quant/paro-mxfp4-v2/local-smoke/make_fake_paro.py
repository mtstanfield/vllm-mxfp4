"""local smoke tests only: z-lab-shaped rotation params (random pairs/angles/scales) for a small Qwen3.5 model."""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
g = torch.Generator().manual_seed(0); t = {}
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
for name, shard in idx.items():
    if ".layers." not in name or not name.endswith(".weight") or "visual" in name or name.startswith("mtp") or "norm" in name:
        continue
    if "in_proj_a" in name or "in_proj_b" in name or "conv1d" in name:
        continue
    with safe_open(f"{src}/{shard}", framework="pt") as f:
        shp = f.get_slice(name).get_shape()
    if len(shp) != 2:
        continue
    k = shp[1]; mod = name[:-len(".weight")]
    pairs = torch.stack([torch.cat([torch.randperm(128, generator=g) for _ in range(k // 128)]) for _ in range(8)]).short()
    t[f"{mod}.pairs"] = pairs
    t[f"{mod}.theta"] = (torch.randn(8, k // 2, generator=g) * 0.8).half()
    t[f"{mod}.channel_scales"] = (1 / (0.7 + 0.6 * torch.rand(1, k, generator=g))).half()
save_file(t, f"{dst}/model.safetensors")
cfg = json.load(open(f"{src}/config.json")); cfg["quantization_config"] = {"quant_method": "paroquant", "krot": 8}
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=1)
for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    open(f"{dst}/{fn}", "wb").write(open(f"{src}/{fn}", "rb").read())
print(len(t) // 3, "modules")
