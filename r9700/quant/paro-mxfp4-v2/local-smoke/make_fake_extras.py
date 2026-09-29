"""local smoke tests only: a 'finalist' fp8 MTP shard + manifest for a small Qwen3.5 model (our own fp8 export of its head)."""
import json, os, sys, torch
from transformers import AutoConfig
from safetensors.torch import save_file
import mtp_refit as M
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
cfg = AutoConfig.from_pretrained(src).text_config; cfg._attn_implementation = "sdpa"
m = M.MTP(cfg, cfg.layer_types.index("full_attention")).float()
m.load_state_dict(M.to_module_sd(M.load_mtp_sd(src), m))
sd = m.state_dict(); t = {}
for k, v in sd.items():
    ck = M.ckpt_name(k)
    if any(k == p + ".weight" for p in M.PROJ):
        s = (v.abs().amax(1) / 448).clamp(min=1e-12); t[ck] = (v / s.view(-1, 1)).to(torch.float8_e4m3fn); t[ck[:-7] + ".weight_scale"] = s
    else:
        t[ck] = v.bfloat16()
save_file({k: v.contiguous() for k, v in t.items()}, f"{dst}/model-mtp.safetensors")
dt = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32", torch.bfloat16: "BF16"}
json.dump({k: [dt[v.dtype], list(v.shape), "model-mtp.safetensors"] for k, v in t.items()}, open(f"{dst}/finalist-manifest.json", "w"))
print(len(t), "tensors")
