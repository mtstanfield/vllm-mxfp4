#!/usr/bin/env python3
"""pick.py - choose the body variant to ship among variants with saved codes, using the fp8-activation (+a8) re-score
when it exists.
Score = mean of KL on: ASSISTANT tokens of the held-out traces at 8k and 32k, the code sample, and wiki.
Guard: a variant may not regress wiki PPL more than 0.5 points beyond the finalist rebuild (rtn) - trace-only GPTQ cut
trace KL by two thirds but took wiki PPL from +0.78% to +2.30%, and the model also serves general side chats.
All-token trace KL is NOT used: the non-assistant trace tokens are mostly tool output (hex dumps, addresses, listings)
where bf16 is confidently wrong (PPL ~47 at 32k), so their KL is dominated by a few extreme positions (p99 ~20 nats).
Writes $WORK/WINNER."""
import json, os
from pathlib import Path
WORK = Path(os.environ.get("WORK", "/workspace/pq2"))
rows = {r["name"]: r for r in (json.loads(l) for l in open(WORK / "results.jsonl") if l.strip())}
have = [p.name for p in (WORK / "codes").iterdir() if (p / "codes.safetensors").exists()]
base = rows["bf16"]


def score(r):
    parts = [r["eval8k"]["kl_asst"], r["code"]["kl"], r["wiki"]["kl"]]
    if "eval32k" in r:
        parts.append(r["eval32k"]["kl_asst"])
    return sum(parts) / len(parts)


def wiki_delta(r):
    return (r["wiki"]["ppl"] / base["wiki"]["ppl"] - 1) * 100


_ref = rows.get("rtn+a8") or rows.get("rtn")          # compare like with like (a8 rows carry the activation cost)
limit = wiki_delta(_ref) + 0.5 if _ref else float("inf")
cands, dropped = {}, {}
for v in have:
    r = rows.get(v + "+a8") or rows.get(v)
    if not (r and "eval8k" in r and "kl_asst" in r["eval8k"]):
        continue
    think_bad = _ref and r["eval8k"].get("kl_think") is not None and _ref["eval8k"].get("kl_think") is not None         and r["eval8k"]["kl_think"] > _ref["eval8k"]["kl_think"]          # reasoning must not drift further than the finalist's
    (cands if wiki_delta(r) <= limit and not think_bad else dropped)[v] = score(r)
for v, s in sorted(cands.items(), key=lambda x: x[1]):
    print(f"  {v:22s} {s:.5f}  wiki {wiki_delta(rows.get(v + '+a8') or rows[v]):+.2f}%{'  (a8)' if v + '+a8' in rows else ''}")
for v, s in dropped.items():
    print(f"  {v:22s} {s:.5f}  EXCLUDED (wiki {wiki_delta(rows.get(v + '+a8') or rows[v]):+.2f}% vs limit {limit:+.2f}%, or reasoning KL above the finalist's)")
if "redhat_gptq_awq" in rows:
    print(f"  (control redhat_gptq_awq {score(rows['redhat_gptq_awq']):.5f}, wiki {wiki_delta(rows['redhat_gptq_awq']):+.2f}%, weights-only)")
best = min(cands, key=cands.get)
if "rtn" in cands and best != "rtn":
    print(f"winner {best}: {cands[best]:.5f} vs the finalist rebuild {cands['rtn']:.5f} ({(1 - cands[best] / cands['rtn']) * 100:.0f}% lower)")
(WORK / "WINNER").write_text(best)
print(best)
