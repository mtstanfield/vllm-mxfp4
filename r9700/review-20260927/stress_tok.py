#!/usr/bin/env python3
"""stress_tok.py <base> <len1,len2,...> [max_tokens] -- KV-pool / VRAM stress with EXACT prompt lengths: one request per length,
all sent at once, each a fresh random token-id prompt (no prefix-cache hits), ignore_eos so every request decodes max_tokens.
Prints per request: asked/served prompt tokens, completion tokens, wall, ok/error."""
import json, random, sys, threading, time, urllib.request
base, lens = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 64
out = [None] * len(lens)

def run(i, n):
    rng = random.Random(time.time_ns() + i)
    body = {"model": "qwen38-27b", "prompt": [rng.randrange(1000, 150000) for _ in range(n)], "max_tokens": mt,
            "temperature": 0.7, "ignore_eos": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        u = json.load(urllib.request.urlopen(req, timeout=3600))["usage"]
        out[i] = f"asked {n} served {u["prompt_tokens"]} gen {u["completion_tokens"]} wall {time.time() - t:.1f}s ok"
    except Exception as e:
        out[i] = f"asked {n} ERROR after {time.time() - t:.1f}s: {e}"

th = [threading.Thread(target=run, args=(i, n)) for i, n in enumerate(lens)]
[t.start() for t in th]; [t.join() for t in th]
print("\n".join("STRESS " + o for o in out), flush=True)
