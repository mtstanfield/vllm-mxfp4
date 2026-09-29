#!/usr/bin/env python3
"""vision_sidecar.py - run the Qwen3.5 vision tower on the CPU and proxy the OpenAI chat API to vLLM, rewriting every
image_url content part into precomputed `image_embeds` (vLLM --enable-mm-embeds), so the GPU model can run with the
visual tower stubbed out (local/patch_visual_stub.py, RADIANCE_VISUAL_STUB=1) and spend that ~0.9 GiB on KV cache.

This checkpoint has no DeepStack levels (vision_config.deepstack_visual_indexes == []), so vLLM's expected embeds are
exactly the HF Qwen3_5VisionModel merged output: [n_tokens, out_hidden_size] with n_tokens = prod(grid_thw) / merge^2.

  python3 vision_sidecar.py --ckpt /models/<dir> --listen 1247 --upstream http://127.0.0.1:1246
  python3 vision_sidecar.py --ckpt /models/<dir> --selftest            # CPU timing on a synthetic screenshot
"""
import argparse, asyncio, base64, io, json, os, struct, sys, time
import torch
from safetensors import safe_open
from PIL import Image, ImageDraw

def load_visual(ckpt: str, dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    cfg = AutoConfig.from_pretrained(ckpt)
    vc = cfg.vision_config
    assert not getattr(vc, "deepstack_visual_indexes", []), "DeepStack levels present: sidecar needs the deepstack mergers"
    vc._attn_implementation = "eager"   # CPU: flash-attention-for-cpu bf16 = 62 s/screenshot, MATH sdpa 204 s, eager 19 s
    model = Qwen3_5VisionModel(vc).to(dtype).eval()
    for m in model.modules():
        if hasattr(m, "config"): m.config._attn_implementation = "eager"
    idx_p = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx_p):
        wm = json.load(open(idx_p))["weight_map"]
        shards = sorted({s for k, s in wm.items() if k.startswith("model.visual.")})
    else:
        shards = ["model.safetensors"]
    sd = {}
    for s in shards:
        with safe_open(os.path.join(ckpt, s), "pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[sidecar] state dict: missing={missing[:5]} unexpected={unexpected[:5]}", file=sys.stderr)
    proc = AutoImageProcessor.from_pretrained(ckpt)
    print(f"[sidecar] visual tower loaded: {len(sd)} tensors, depth {vc.depth}, out {vc.out_hidden_size}, merge {vc.spatial_merge_size}", file=sys.stderr)
    return model, proc, vc

@torch.no_grad()
def embed_image(model, proc, img: Image.Image):
    t0 = time.time()
    out = proc(images=[img.convert("RGB")], return_tensors="pt")
    pv = out["pixel_values"].to(next(model.parameters()).dtype); grid = out["image_grid_thw"]
    res = model(pv, grid_thw=grid, return_dict=True)
    emb = res.pooler_output.to(torch.bfloat16).contiguous()
    dt = time.time() - t0
    return emb, grid, dt

def b64_tensor(t: torch.Tensor) -> str:
    buf = io.BytesIO(); torch.save(t, buf); return base64.b64encode(buf.getvalue()).decode()

def decode_image(url: str, session=None):
    if url.startswith("data:"):
        head, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data)))
    raise ValueError("only data: URLs are handled by the sidecar (omp sends base64)")

async def rewrite_body(body: dict, model, proc, log):
    n = 0
    for m in body.get("messages", []):
        c = m.get("content")
        if not isinstance(c, list): continue
        for i, part in enumerate(c):
            if not isinstance(part, dict) or part.get("type") != "image_url": continue
            url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else str(part.get("image_url"))
            img = decode_image(url)
            emb, grid, dt = await asyncio.get_event_loop().run_in_executor(None, embed_image, model, proc, img)
            c[i] = {"type": "image_embeds", "image_embeds": {"image_embeds": b64_tensor(emb), "image_grid_thw": b64_tensor(grid[0])}}   # per-image grid must be 1-D [3] (a [1,3] made vLLM see sizes [1,1])
            n += 1
            log(f"[sidecar] image {img.size} -> {tuple(emb.shape)} tokens={emb.shape[0]} in {dt:.2f}s")
    return n

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--listen", type=int, default=1247)
    ap.add_argument("--upstream", default="http://127.0.0.1:1246"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    model, proc, vc = load_visual(a.ckpt, torch.float32 if a.dtype == "fp32" else torch.bfloat16)
    if a.selftest:
        img = Image.new("RGB", (1280, 896), "white"); d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 600, 300], outline="red", width=6); d.text((60, 60), "SIDECAR TEST 4271", fill="black")
        for k in range(3):
            emb, grid, dt = embed_image(model, proc, img)
            print(f"selftest run {k}: grid {grid.tolist()} -> {tuple(emb.shape)} in {dt:.2f}s (threads {torch.get_num_threads()})")
        return
    from aiohttp import web, ClientSession
    log = lambda s: print(s, file=sys.stderr, flush=True)
    session = ClientSession()
    async def handle(req: web.Request):
        raw = await req.read()
        if req.method == "POST" and req.path.endswith("/chat/completions"):
            try:
                body = json.loads(raw)
                if await rewrite_body(body, model, proc, log): raw = json.dumps(body).encode()
            except Exception as e:
                log(f"[sidecar] rewrite failed, passing through: {e!r}")
        headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
        async with session.request(req.method, a.upstream + req.path_qs, data=raw, headers=headers, timeout=None) as up:
            resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
            await resp.prepare(req)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof(); return resp
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handle)
    log(f"[sidecar] listening on :{a.listen}, upstream {a.upstream}")
    runner = web.AppRunner(app); await runner.setup(); await web.TCPSite(runner, "0.0.0.0", a.listen).start()
    while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
