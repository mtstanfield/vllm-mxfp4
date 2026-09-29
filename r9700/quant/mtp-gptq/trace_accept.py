#!/usr/bin/env python3
"""trace_accept.py <base url> <label> [n_windows] [prompt_len] [max_tokens] -- drafter acceptance on the user's own workload:
the held-out omp-session windows of traces.npz (eval8k: whole session files never used for calibration), each cut to a
prompt of prompt_len tokens and continued with production-like sampling (temperature 1.0, top_p 0.95, fixed seed per
window), one request at a time. Reports accepted/drafted from the server's Prometheus counters and the decode rate
(generated tokens / (wall - a separately measured 1-token prefill of the same prompt))."""
import json, os, sys, time, urllib.request
import numpy as np

base, label = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
pl = int(sys.argv[4]) if len(sys.argv) > 4 else 6144
mt = int(sys.argv[5]) if len(sys.argv) > 5 else 256
ev = np.load(os.environ.get("TRACES", "/pqv2/traces.npz"))["eval8k"][:n]
cuts = [int(c) for c in sys.argv[6].split(",")] if len(sys.argv) > 6 else [pl]


def post(body):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r, time.time() - t0


def counters():
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        for k in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total"):
            if line.startswith(k):
                out[k] = out.get(k, 0.0) + float(line.split()[-1])
    return out


gen = dec_t = 0.0
c0 = counters()
for i, (w, pl) in enumerate((w, c) for w in ev for c in cuts):
    prompt = w[:pl].tolist()
    _, tp = post({"model": "qwen38-27b", "prompt": prompt, "max_tokens": 1, "temperature": 0})      # prefill (+cache)
    r, tw = post({"model": "qwen38-27b", "prompt": prompt, "max_tokens": mt, "temperature": 1.0, "top_p": 0.95,
                  "seed": 1000 + i})
    g = r["usage"]["completion_tokens"]
    gen += g
    dec_t += max(tw - 0.05, 1e-3)          # the prompt is prefix-cached by the probe: wall ~= decode
c1 = counters()
acc = c1["vllm:spec_decode_num_accepted_tokens_total"] - c0["vllm:spec_decode_num_accepted_tokens_total"]
dr = c1["vllm:spec_decode_num_draft_tokens_total"] - c0["vllm:spec_decode_num_draft_tokens_total"]
print(f"TRACE {label}: {n} windows x cuts {cuts} +{mt}: generated {int(gen)}, accept {acc / dr:.4f} ({int(acc)}/{int(dr)}), "
      f"decode {gen / dec_t:.2f} t/s", flush=True)
