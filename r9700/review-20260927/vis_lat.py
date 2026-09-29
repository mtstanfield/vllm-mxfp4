#!/usr/bin/env python3
"""vis_lat.py <base> [n] -- screenshot latency: n synthetic 1280x896 screenshots, each with a unique nonce drawn in (defeats the
encoder/prefix caches, so the vision tower runs every time); reports time-to-first-token (max_tokens 1) per image, the median,
and one full answer for a correctness eyeball."""
import base64, io, json, random, statistics, sys, time, urllib.request
from PIL import Image, ImageDraw
base = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

def png_b64(nonce):
    img = Image.new("RGB", (1280, 896), "white"); d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 620, 320], outline="red", width=8)
    d.ellipse([760, 120, 1160, 520], fill="blue")
    d.text((80, 90), "ORDER 4271 CONFIRMED", fill="black")
    d.text((80, 130), "total: 63.20 EUR", fill="black")
    d.text((80, 700), f"ref {nonce}", fill="gray")
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

def ask(b64, q, mt):
    body = {"model": "qwen38-27b", "max_tokens": mt, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                                       {"type": "text", "text": q}]}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); return r, time.time() - t

q = "What text is written in the image, and what two shapes do you see (with colours)? Answer in one line."
ts = []
for i in range(n):
    r, dt = ask(png_b64(random.randrange(10**9)), q, 1); ts.append(dt)
r, dt = ask(png_b64(random.randrange(10**9)), q, 120)
print(f"VIS ttft(s) {[round(t, 3) for t in ts]} median {statistics.median(ts[1:] or ts):.3f} (first excluded) "
      f"prompt_tokens={r["usage"]["prompt_tokens"]} answer({dt:.1f}s): {r["choices"][0]["message"]["content"].strip()[:160]}")
