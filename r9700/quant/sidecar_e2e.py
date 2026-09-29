#!/usr/bin/env python3
"""sidecar_e2e.py - end-to-end check of the vision sidecar: draw a synthetic screenshot with known text/shapes, send it as a
normal image_url chat request THROUGH the sidecar (which rewrites it into image_embeds), and print the model's answer.
  python3 sidecar_e2e.py --url http://127.0.0.1:1247 [--direct http://127.0.0.1:1246]"""
import argparse, base64, io, json, time, urllib.request
from PIL import Image, ImageDraw

def png_b64():
    img = Image.new("RGB", (1280, 896), "white"); d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 620, 320], outline="red", width=8)
    d.ellipse([760, 120, 1160, 520], fill="blue")
    d.text((80, 90), "ORDER 4271 CONFIRMED", fill="black")
    d.text((80, 130), "total: 63.20 EUR", fill="black")
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

def ask(url, b64, q):
    body = {"model": "qwen38-27b", "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                                       {"type": "text", "text": q}]}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t
    return r["choices"][0]["message"]["content"].strip(), dt, r.get("usage", {})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:1247"); ap.add_argument("--direct", default=None)
    a = ap.parse_args(); b64 = png_b64()
    q = "What text is written in the image, and what two shapes do you see (with colours)? Answer in one line."
    for name, u in (("via sidecar", a.url),) + ((("direct", a.direct),) if a.direct else ()):
        try:
            ans, dt, usage = ask(u, b64, q); print(f"[{name}] {dt:.1f}s prompt_tokens={usage.get('prompt_tokens')} :: {ans}")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
