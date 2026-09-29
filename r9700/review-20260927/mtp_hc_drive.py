#!/usr/bin/env python3
"""mtp_hc_drive.py <base url> <hc host dir as seen here> [n_windows] [max_tokens] -- drive a RADIANCE_MTP_HCOLLECT
calibration run: switch collection on (<dir>/ON), send n calibration windows from traces.npz (the user's omp sessions,
the body's v2 calibration set; 4096 tokens each, spread over the whole set) as raw token prompts with sampled
continuations, 2 in flight (the serve's MAXSEQS), then request the dump (<dir>/SAVE) and wait for <dir>/SAVED."""
import json, os, sys, threading, time, urllib.request
import numpy as np

base, hc = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 320
calib = np.load(os.environ.get("TRACES", "/pqv2/traces.npz"))["calib"]
idx = np.linspace(0, len(calib) - 1, n).round().astype(int)
for f in ("SAVED", "SAVE"):
    if os.path.exists(os.path.join(hc, f)):
        os.remove(os.path.join(hc, f))
open(os.path.join(hc, "ON"), "w").close()
lock, todo, stats = threading.Lock(), list(idx), {"req": 0, "gen": 0}


def worker():
    while True:
        with lock:
            if not todo:
                return
            i = int(todo.pop(0))
        body = {"model": "qwen38-27b", "prompt": calib[i].tolist(), "max_tokens": mt, "temperature": 1.0, "top_p": 0.95,
                "seed": i}
        req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        with lock:
            stats["req"] += 1
            stats["gen"] += r["usage"]["completion_tokens"]


t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(2)]
[t.start() for t in ts]
[t.join() for t in ts]
print(f"{stats['req']} requests, {stats['gen']} generated tokens, {time.time() - t0:.0f} s", flush=True)
os.remove(os.path.join(hc, "ON"))
open(os.path.join(hc, "SAVE"), "w").close()
# the poll runs inside the drafter's linears: keep a trickle of requests going until the dump is written
for k in range(60):
    if os.path.exists(os.path.join(hc, "SAVED")):
        print("SAVED:", open(os.path.join(hc, "SAVED")).read().strip())
        break
    body = {"model": "qwen38-27b", "prompt": calib[0][:64].tolist(), "max_tokens": 16, "temperature": 0}
    urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                                  headers={"Content-Type": "application/json"}), timeout=300).read()
    time.sleep(2)
else:
    print("NO DUMP")
