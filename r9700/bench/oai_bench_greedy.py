"""Engine-agnostic benchmark over the OpenAI chat API (works for llama.cpp and vLLM alike).
# GREEDY variant of oai_bench.py (2026-09-15): temperature 0 / top_k 1 / seed 1 so the continuation is deterministic per engine
# and MTP acceptance stops being a dice roll (sampled decode at 60k swung 41-63 t/s run to run). Prints a sha of the text
# so two engines can be told apart (different compile caches produce different numerics -> different greedy text) and
# dumps each completion next to this file. Pair with vllm_accept_delta.py for tok/step + acceptance per bench.

Measures, with streaming: TTFT (= prefill time), prefill tok/s (prompt_tokens / TTFT), decode tok/s
(completion_tokens / (total - TTFT)), using the server's `usage` counts (stream_options include_usage).

  depth   --depths 8000,100000,200000 [--runs 2]     prefill + decode at each depth; the second run at a depth
                                                     repeats the SAME prompt -> shows prefix-cache hit (TTFT collapse)
  swap    --orch 190000 --sub 60000                  orchestrator <-> subagent alternation: O1 cold, S1 cold, O2 resume,
                                                     S1 resume, S2 cold (2nd subagent), O3 resume, S1 return, ...
Usage: oai_bench.py --base http://host:port --model NAME depth|swap [...]
"""
import argparse, json, statistics, sys, time, urllib.request, hashlib, os

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["depth", "swap"])
ap.add_argument("--base", default="http://192.168.88.89:1246")
ap.add_argument("--model", default="qwen38")
ap.add_argument("--label", default="run")
ap.add_argument("--depths", default="8000,100000,200000")
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--n-predict", type=int, default=200)
ap.add_argument("--orch", type=int, default=190000)
ap.add_argument("--sub", type=int, default=60000)
ap.add_argument("--effort", default="medium")
ap.add_argument("--tag", default=None,
                help="COLD-prefill mode: unique per invocation. Each depth d gets system prompt 'Session <tag>d<d>. ...' "
                     "and filler tag '<tag>d<d>', so no prefix is shared with any earlier request (prefix caches miss).")
ap.add_argument("--timeout", type=int, default=3600)
args = ap.parse_args()
EP = args.base.rstrip("/") + "/v1/chat/completions"

SYSTEM = ("You are a coding agent. Answer tersely. " +
          " ".join(f"Rule {i}: tool {i} must be called with argument set {i * 3 % 11}." for i in range(300)))


def filler(tag, depth):
    n = int(max(1, (depth - 5900) / 25.0))
    return "\n".join(f"{tag} fact {i}: the code for item {i} is {(i * 7 + 3) % 97}" for i in range(n))


def stream(messages, max_tokens):
    body = {"model": args.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 1,
            "chat_template_kwargs": {"reasoning_effort": args.effort}}
    req = urllib.request.Request(EP, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None; n_chunks = 0; txt = ""
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}):
                d = ch[0]["delta"]
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and t_first is None:
                    t_first = time.time()
                n_chunks += 1
                txt += (d.get("content") or d.get("reasoning") or d.get("reasoning_content") or "")
    t_end = time.time()
    pt = (usage or {}).get("prompt_tokens"); ct = (usage or {}).get("completion_tokens")
    ttft = (t_first or t_end) - t0
    dec = (t_end - (t_first or t_end))
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": ttft,
            "prefill_tps": (pt / ttft) if pt and ttft > 0 else None,
            "decode_tps": (ct / dec) if ct and dec > 0 else None, "wall_s": t_end - t0, "text": txt}


def show(name, r):
    print(f"{name:28s} prompt={r['prompt_tokens']!s:>7} ttft={r['ttft_s']:8.2f}s prefill={r['prefill_tps'] or 0:8.1f} t/s "
          f"gen={r['completion_tokens']!s:>4} decode={r['decode_tps'] or 0:6.1f} t/s wall={r['wall_s']:7.1f}s sha={hashlib.sha1(r['text'].encode()).hexdigest()[:8]}", flush=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "txt-" + name.replace(" ", "_") + ".txt"), "w", encoding="utf-8").write(r["text"])


if args.mode == "depth":
    for d in [int(x) for x in args.depths.split(",")]:
        tag = f"{args.tag}d{d}" if args.tag else "dd"
        system = f"Session {tag}. " + SYSTEM if args.tag else SYSTEM
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Memorize this list.\n{filler(tag, d)}\nThen write a long, detailed essay about lighthouses."}]
        dec = []
        for i in range(args.runs):
            r = stream(msgs, args.n_predict)
            show(f"{args.label} depth={d} run{i}", r)
            if r["decode_tps"]:
                dec.append(r["decode_tps"])
        if dec:
            print(f"RESULT {args.label} depth={d}: decode median {statistics.median(dec):.1f} t/s", flush=True)
else:
    orch = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Memorize this list.\n{filler('orch', args.orch)}"}]
    subA = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Memorize this list.\n{filler('subA', args.sub)}"}]
    subB = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Memorize this list.\n{filler('subB', args.sub)}"}]

    def turn(name, conv, ask):
        msgs = conv + [{"role": "user", "content": ask}]
        r = stream(msgs, 64)
        show(name, r)
        conv.append({"role": "user", "content": ask}); conv.append({"role": "assistant", "content": "ok."})

    turn("O1 orch cold", orch, "What is orch item 42? One line.")
    turn("S1a subA cold", subA, "What is subA item 7? One line.")
    turn("O2 orch resume", orch, "And item 43?")
    turn("S1b subA resume", subA, "And item 8?")
    turn("S2a subB cold", subB, "What is subB item 9? One line.")
    turn("O3 orch resume", orch, "And item 44?")
    turn("S1c subA return", subA, "And item 9?")
    turn("O4 orch resume", orch, "And item 45?")
    turn("S2b subB return", subB, "And item 10?")
    turn("O5 orch resume", orch, "And item 46?")
