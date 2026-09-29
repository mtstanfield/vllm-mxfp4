"""local only: a compressed-tensors-style MXFP4 checkpoint (plain OCP RTN, no rotation) of the small model."""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
import mx
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
shard = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]; t = {}
with safe_open(f"{src}/{next(iter(set(shard.values())))}", framework="pt") as f:
    for k in f.keys():
        x = f.get_tensor(k).clone()
        if ".layers." in k and k.endswith("proj.weight") and "visual" not in k and not k.startswith("mtp") \
                and "in_proj_a" not in k and "in_proj_b" not in k and x.dim() == 2:
            p, e = mx.pack(*mx.quant_matrix(x.float().cuda(), "ocp"))
            t[k[:-7] + ".weight_packed"] = p.cpu(); t[k[:-7] + ".weight_scale"] = e.cpu()
        else:
            t[k] = x
save_file(t, f"{dst}/model.safetensors")
json.dump({"weight_map": {k: "model.safetensors" for k in t}}, open(f"{dst}/model.safetensors.index.json", "w"))
print(sum(k.endswith("_packed") for k in t), "packed")
