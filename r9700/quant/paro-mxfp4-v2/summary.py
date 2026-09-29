#!/usr/bin/env python3
"""summary.py - results.jsonl as a table. The trace columns that matter are the ASSISTANT-token ones (what the model
writes); all-token trace KL is shown last for reference only (dominated by tool-output noise, see pick.py)."""
import json, os, sys
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("WORK", "/workspace/pq2"), "results.jsonl")
if not os.path.exists(path):
    print("(no results yet)"); sys.exit(0)
rows = [json.loads(l) for l in open(path) if l.strip()]
base = next((r for r in rows if r["name"] == "bf16"), None)


def d(r, s, k="ppl"):
    if s not in r or k not in r[s]:
        return "      -"
    if base is None or r is base:
        return f"{r[s][k]:7.3f}"
    return f"{(r[s][k] / base[s][k] - 1) * 100:+6.2f}%"


def f(r, s, k, fmt):
    return format(r[s][k], fmt) if s in r and r[s].get(k) is not None else "-"


print(f"{'variant':20s} {'wiki':>7s} {'code':>7s} {'8k asst':>7s} | {'KL wiki':>7s} {'KL code':>7s} {'KL8k as':>7s} {'KL32k as':>8s} | "
      f"{'top1 8k as':>10s} {'top1 32k as':>11s} | {'KL think8k':>10s} {'KL ans8k':>8s} | {'KL8k all':>8s}")
for r in rows:
    print(f"{r['name'][:20]:20s} {d(r,'wiki')} {d(r,'code')} {d(r,'eval8k','ppl_asst')} | "
          f"{f(r,'wiki','kl','7.4f')} {f(r,'code','kl','7.4f')} {f(r,'eval8k','kl_asst','7.4f')} {f(r,'eval32k','kl_asst','8.4f')} | "
          f"{f(r,'eval8k','top1_asst','10.4f')} {f(r,'eval32k','top1_asst','11.4f')} | {f(r,'eval8k','kl_think','10.4f')} {f(r,'eval8k','kl_answer','8.4f')} | {f(r,'eval8k','kl','8.4f')}")
