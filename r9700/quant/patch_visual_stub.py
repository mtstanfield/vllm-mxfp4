#!/usr/bin/env python3
"""Local patch (ours): vision sidecar mode. With RADIANCE_VISUAL_STUB=1 the Qwen3.5 model does NOT build or load the
visual tower (0.86 GiB bf16 on the GPU); images must arrive as precomputed `image_embeds` (the tower's output incl.
DeepStack levels, [tokens, out_hidden_size * (1 + len(deepstack_visual_indexes))]) via the chat API with
--enable-mm-embeds. A tiny stub keeps the attributes the model reads (dtype, device, spatial_merge_size,
deepstack_visual_indexes) and answers pixel_values calls (only the profiler's dummy run) with zeros of the right shape.
Gated: without the env var the file is left untouched (prints "skipped")."""
import os, pathlib
if os.environ.get("RADIANCE_VISUAL_STUB", "0") != "1":
    print("[local patch] visual stub: skipped (RADIANCE_VISUAL_STUB != 1)"); raise SystemExit(0)
p = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py")
s = p.read_text()
if "local patch: visual stub" in s:
    print("[local patch] visual stub: already applied"); raise SystemExit(0)
old = ('        with self._mark_tower_model(vllm_config, {"image", "video"}):\n'
       '            self.visual = Qwen3_VisionTransformer(\n'
       '                config.vision_config,\n'
       '                norm_eps=getattr(config, "rms_norm_eps", 1e-6),\n'
       '                quant_config=quant_config,\n'
       '                prefix=maybe_prefix(prefix, "visual"),\n'
       '            )\n')
new = ('        with self._mark_tower_model(vllm_config, {"image", "video"}):\n'
       '            self.visual = _VisualStub(config.vision_config, self.visual_dim + self.multiscale_dim)  # local patch: visual stub\n')
assert s.count(old) >= 1, "visual construction anchor not found"
s = s.replace(old, new, 1)   # first occurrence = the dense Qwen3_5ForConditionalGeneration
stub = '''

class _VisualStub(torch.nn.Module):
    """local patch: visual stub -- stands in for Qwen3_VisionTransformer when images arrive as image_embeds."""
    def __init__(self, vision_config, out_dim: int):
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.deepstack_visual_indexes = getattr(vision_config, "deepstack_visual_indexes", [])
        self.out_dim = out_dim
        self.register_buffer("_anchor", torch.zeros(1, dtype=torch.bfloat16), persistent=False)
    @property
    def dtype(self): return self._anchor.dtype
    @property
    def device(self): return self._anchor.device
    def forward(self, pixel_values, grid_thw):
        grid = grid_thw if torch.is_tensor(grid_thw) else torch.tensor(grid_thw)
        n = int((grid.prod(-1) // self.spatial_merge_unit).sum().item())
        return torch.zeros(n, self.out_dim, dtype=self.dtype, device=self._anchor.device)

'''
# insert the stub ABOVE the decorator block of the dense class (@MULTIMODAL_REGISTRY.register_processor(... info=Qwen3_5ProcessingInfo)):
# between decorator and class the registration would land on the stub (2026-09-17 bug: model fell back to Qwen3VLProcessingInfo
# and --enable-mm-embeds failed its config type check).
lines = s.split("\n")
ci = next(i for i, l in enumerate(lines) if l.startswith("class Qwen3_5ForConditionalGeneration("))
di = ci
while di > 0 and (lines[di - 1].startswith("@") or lines[di - 1].startswith("    ") or lines[di - 1].startswith(")")):
    di -= 1
assert lines[di].startswith("@"), f"decorator block not found above the class (line {di}: {lines[di]!r})"
lines[di:di] = stub.strip("\n").split("\n") + [""]
s = "\n".join(lines)
old_skip = '        loader = AutoWeightsLoader(\n            self,\n            skip_prefixes=["mtp."],\n        )\n        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n'
assert old_skip in s, "loader anchor not found"
new_skip = '        loader = AutoWeightsLoader(\n            self,\n            skip_prefixes=["mtp.", "visual."],  # local patch: visual stub (weights never loaded)\n        )\n        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n'
s = s.replace(old_skip, new_skip)   # every ForConditionalGeneration loader (dense + moe); harmless where visual is real
p.write_text(s)
print("[local patch] visual stub applied (no visual tower on the GPU; images via --enable-mm-embeds image_embeds only)")
