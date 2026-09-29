"""spec_sampled.py -- engine-agnostic sampled decode benchmark (vLLM or llama.cpp, OpenAI chat API, streaming): the
spec_prompt.py prompts (code / prose / json) with the SERVER's default sampling (production: temp 1.0, top-p 0.95,
top-k 20 -- nothing is sent), ignore_eos so every run decodes --n tokens. Decode t/s = (tokens - 1) / (last - first
token time); TOTAL = all tokens / all decode time over the runs (MTP acceptance is bimodal per sampled run, so judge
totals over many runs, not medians). --metrics adds vLLM's spec-decode acceptance per prompt from /metrics deltas.
Usage: spec_sampled.py --base URL [--prompts code,prose,json] [--runs 8] [--n 600] [--metrics URL/metrics]"""
import argparse, ast, json, re, sys, time, urllib.request, os

# the prompts of spec_prompt.py, read without running it (it is a script: importing it sends a request)
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_prompt.py")).read()
PROMPTS = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
               if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PROMPTS" for t in n.targets))

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", default="qwen38-27b")
ap.add_argument("--prompts", default="code,prose,json")
ap.add_argument("--runs", type=int, default=8)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--metrics", default=None)
ap.add_argument("--label", default="")
a = ap.parse_args()


def spec_counters():
    m = urllib.request.urlopen(a.metrics, timeout=10).read().decode()
    return {k: float(v) for k, v in re.findall(
        r"^vllm:(spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|spec_decode_num_drafts_total)"
        r"\{[^}]*\} (\S+)$", m, re.M)}


grand_tok = grand_s = 0.0
for p in a.prompts.split(","):
    before = spec_counters() if a.metrics else None
    tok = secs = 0.0
    for r in range(a.runs):
        body = {"model": a.model, "messages": [{"role": "user", "content": PROMPTS[p]}], "max_tokens": a.n,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        first = last = None
        n = 0
        with urllib.request.urlopen(req, timeout=900) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens", n)
                ch = d.get("choices") or []
                if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")
                           or (ch[0].get("delta") or {}).get("reasoning")):
                    now = time.time()
                    first = first or now
                    last = now
        dt = (last - first) if first and last and last > first else float("nan")
        print(f"{a.label} {p} run{r}: n={n} decode={(n - 1) / dt:.2f} t/s", flush=True)
        tok += n - 1
        secs += dt
    acc = ""
    if before is not None:
        after = spec_counters()
        d = {k: after[k] - before[k] for k in after}
        acc = (f" accept={d['spec_decode_num_accepted_tokens_total'] / max(1, d['spec_decode_num_draft_tokens_total']):.3f}"
               f" tok/step={(tok + a.runs) / max(1, d['spec_decode_num_drafts_total']):.2f}")
    print(f"TOTAL {a.label} {p}: {tok / secs:.2f} t/s over {a.runs} runs{acc}", flush=True)
    grand_tok += tok
    grand_s += secs
print(f"GRAND {a.label}: {grand_tok / grand_s:.2f} t/s over all prompts", flush=True)
