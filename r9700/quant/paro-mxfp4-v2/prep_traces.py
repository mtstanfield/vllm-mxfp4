#!/usr/bin/env python3
"""prep_traces.py - turn omp agent sessions into calibration + held-out eval token sets for the MXFP4 requant.

Runs on .200 (~/.venvs/quantprep: transformers without torch). Each session file is split into segments at compactions,
rendered with the chat template vLLM actually serves (qwen-fixed-v22.3.jinja, preserve_thinking on), tokenized, and cut
into windows. Held-out = whole session files, so no eval window shares a file with calibration.

  ~/.venvs/quantprep/bin/python prep_traces.py --out ~/quantprep/traces.npz
"""
import argparse, glob, json, os, random
import numpy as np
from transformers import AutoTokenizer

SESS = os.path.expanduser("~/.omp/agent/sessions")
TOK = os.path.expanduser("~/quantprep/tok")
# share of the calibration pool per project (blades = Fable-27B outputs, same workload, different model -> smallest share)
PROJECTS = {"-projects-stars-wasm": 0.40, "-projects-chips-wasm": 0.30, "-projects-skifree-wasm": 0.20, "-projects-blades_browser": 0.10}
EVAL_PROJECTS = ("-projects-stars-wasm", "-projects-chips-wasm", "-projects-skifree-wasm")   # Qwen3.8-written only


def blocks(content):
    return content if isinstance(content, list) else [{"type": "text", "text": content or ""}]


def to_messages(path):
    """Yield (segment_messages, models) split at compactions; a compaction's summary opens the next segment."""
    msgs, models = [], set()
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type")
        if t == "compaction":
            if msgs:
                yield msgs, models
            msgs, models = [], set()
            if r.get("summary"):
                msgs.append({"role": "user", "content": r["summary"]})
            continue
        if t == "custom_message" and r.get("display") and isinstance(r.get("content"), str):
            msgs.append({"role": "user", "content": r["content"]})
            continue
        if t != "message":
            continue
        m = r["message"]; role = m.get("role")
        if role in ("user", "developer"):
            text = "\n".join(b.get("text", "") for b in blocks(m.get("content")) if b.get("type") == "text")
            if text.strip():
                msgs.append({"role": "user" if role == "user" else "system", "content": text})
        elif role == "assistant":
            think, text, calls = [], [], []
            for b in blocks(m.get("content")):
                if b.get("type") == "thinking":
                    think.append(b.get("thinking", ""))
                elif b.get("type") == "text":
                    text.append(b.get("text", ""))
                elif b.get("type") == "toolCall":
                    args = b.get("arguments")
                    calls.append({"id": b.get("id", ""), "type": "function",
                                  "function": {"name": b.get("name", ""), "arguments": args if isinstance(args, dict) else {}}})
            if not (think or text or calls):
                continue
            a = {"role": "assistant", "content": "".join(text).strip()}
            if think:
                a["reasoning_content"] = "\n".join(think).strip()
            if calls:
                a["tool_calls"] = calls
            msgs.append(a); models.add(f"{m.get('provider')}/{m.get('model')}")
        elif role == "toolResult":
            parts = []
            for b in blocks(m.get("content")):
                parts.append(b.get("text", "") if b.get("type") == "text" else "[image]")
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId", ""), "content": "\n".join(parts)})
    if msgs:
        yield msgs, models


def render(tok, template, msgs):
    # the template wants a user turn before any assistant turn; compaction segments already start with one
    if msgs[0]["role"] not in ("user", "system"):
        msgs = [{"role": "user", "content": "(continued)"}] + msgs
    return tok.apply_chat_template(msgs, chat_template=template, tokenize=False, add_generation_prompt=False)


def assistant_mask(text, offsets):
    """1 for tokens inside an assistant turn body (what the model itself writes: reasoning, text, tool calls)."""
    spans, i = [], 0
    while True:
        s = text.find("<|im_start|>assistant\n", i)
        if s < 0:
            break
        s += len("<|im_start|>assistant\n")
        e = text.find("<|im_end|>", s)
        e = len(text) if e < 0 else e + len("<|im_end|>")
        spans.append((s, e)); i = e
    mask = np.zeros(len(offsets), np.uint8)
    if not spans:
        return mask
    starts = np.array([o[0] for o in offsets])
    for s, e in spans:
        mask[(starts >= s) & (starts < e)] = 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/quantprep/traces.npz"))
    ap.add_argument("--calib-windows", type=int, default=2048)
    ap.add_argument("--calib-len", type=int, default=4096)
    ap.add_argument("--eval-files-per-project", type=int, default=6)
    ap.add_argument("--eval8k", type=int, default=24)
    ap.add_argument("--eval32k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(TOK)
    template = open(os.path.join(TOK, "qwen-fixed-v22.3.jinja")).read()

    # ---- choose held-out files: subagent sessions (small, many, same task) from the Qwen3.8 projects
    split, meta = {}, {"projects": {}, "eval_files": [], "template": "qwen-fixed-v22.3.jinja"}
    for proj in PROJECTS:
        files = sorted(glob.glob(f"{SESS}/{proj}/**/*.jsonl", recursive=True))
        subs = [f for f in files if os.path.dirname(f) != f"{SESS}/{proj}" and os.path.getsize(f) > 200_000]
        held = set(rnd.sample(subs, min(a.eval_files_per_project, len(subs)))) if proj in EVAL_PROJECTS else set()
        for f in files:
            split[f] = "eval" if f in held else "calib"
        meta["eval_files"] += sorted(os.path.relpath(f, SESS) for f in held)

    # ---- render + tokenize every segment
    segs = {"calib": {p: [] for p in PROJECTS}, "eval": []}
    for f, which in sorted(split.items()):
        proj = os.path.relpath(f, SESS).split("/")[0]
        for msgs, models in to_messages(f):
            if len(msgs) < 3:
                continue
            text = render(tok, template, msgs)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = np.asarray(enc["input_ids"], np.int32)
            if len(ids) < 2048:
                continue
            m = assistant_mask(text, enc["offset_mapping"])
            rec = {"ids": ids, "mask": m, "file": os.path.relpath(f, SESS), "models": sorted(models)}
            (segs["eval"] if which == "eval" else segs["calib"][proj]).append(rec)
        print(f"{which:5s} {os.path.relpath(f, SESS)[:90]}", flush=True)

    # ---- calibration pool: windows sampled by project share, weighted by segment length
    L = a.calib_len; calib, calib_src = [], []
    for proj, share in PROJECTS.items():
        pool = [s for s in segs["calib"][proj] if len(s["ids"]) >= L]
        tot = sum(len(s["ids"]) for s in pool)
        n = round(a.calib_windows * share)
        meta["projects"][proj] = {"segments": len(pool), "tokens": int(tot), "calib_windows": n}
        weights = [len(s["ids"]) for s in pool]
        for s in rnd.choices(pool, weights=weights, k=n):
            o = rnd.randrange(0, len(s["ids"]) - L + 1)
            calib.append(s["ids"][o:o + L]); calib_src.append(proj)
    order = list(range(len(calib))); rnd.shuffle(order)
    calib = np.stack([calib[i] for i in order]); calib_src = [calib_src[i] for i in order]

    # ---- held-out eval: 32k windows from segment STARTS (a real task prompt + the run it produced), 8k windows anywhere
    ev = segs["eval"]
    long_ = [s for s in ev if len(s["ids"]) >= 32768]
    e32 = rnd.sample(long_, min(a.eval32k, len(long_)))
    e8, e8m = [], []
    for s in rnd.choices(ev, weights=[len(s["ids"]) for s in ev], k=a.eval8k):
        o = rnd.randrange(0, len(s["ids"]) - 8192 + 1) if len(s["ids"]) >= 8192 else None
        if o is None:
            continue
        e8.append(s["ids"][o:o + 8192]); e8m.append(s["mask"][o:o + 8192])
    meta.update({"calib_windows": int(len(calib)), "calib_len": L, "calib_src": calib_src,
                 "eval8k": len(e8), "eval32k": len(e32), "eval_segments": len(ev),
                 "eval_tokens": int(sum(len(s["ids"]) for s in ev)),
                 "eval8k_assistant_frac": float(np.mean([m.mean() for m in e8m])) if e8m else 0.0})
    np.savez_compressed(a.out, calib=calib,
                        eval8k=np.stack(e8), eval8k_mask=np.stack(e8m),
                        eval32k=np.stack([s["ids"][:32768] for s in e32]), eval32k_mask=np.stack([s["mask"][:32768] for s in e32]))
    json.dump(meta, open(a.out.replace(".npz", ".meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("calib_src",)}, indent=1))


if __name__ == "__main__":
    main()
