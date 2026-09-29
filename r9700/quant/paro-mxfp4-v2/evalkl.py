"""evalkl.py - quality harness: PPL on wiki/code (pod-comparable) + KL vs the bf16 model on held-out agent traces.

The reference pass stores the bf16 model's top-K next-token log-probs per position; a variant is scored with
KL(P_ref || Q) over those K ids plus one lumped tail bucket (top-32 carries almost all the mass), top-1 agreement,
and NLL. Trace metrics skip the first 256 positions and are also reported for assistant-written tokens only.
"""
import math, time
import numpy as np
import torch

TOPK = 32
SKIP = 256
THINK_OPEN, THINK_CLOSE = 248068, 248069          # Qwen3.5/3.8 vocab: <think>, </think> (single tokens)


def think_mask(ids):
    """True inside <think>...</think> (reasoning). A window that opens mid-reasoning (first tag seen is </think>)
    counts its prefix as reasoning."""
    ids = ids.tolist(); m = [False] * len(ids)
    first = next((t for t in ids if t in (THINK_OPEN, THINK_CLOSE)), None)
    inside = first == THINK_CLOSE
    for i, t in enumerate(ids):
        if t == THINK_OPEN:
            inside = True; continue
        if t == THINK_CLOSE:
            inside = False; continue
        m[i] = inside
    return torch.tensor(m)


def build_sets(tok, traces_npz, code_path, which=("wiki", "code", "eval8k", "eval32k"), wiki_tokens=40960, code_chunks=30,
               seq=2048, max_seqs=None):
    sets = {}
    z = np.load(traces_npz)
    for name in ("eval32k", "eval8k"):                  # longest first: an attention-memory problem shows up in minutes
        if name in which:
            sets[name] = [(torch.from_numpy(z[name][i].astype(np.int64)), torch.from_numpy(z[name + "_mask"][i].astype(bool)))
                          for i in range(z[name].shape[0])]
    if "wiki" in which:                                  # same text/tokens/chunking as paro-mxfp4-pod/ppl_dir.py
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text, return_tensors="pt").input_ids[0][:wiki_tokens]
        sets["wiki"] = [(ids[i:i + seq], None) for i in range(0, len(ids) - seq + 1, seq)]
    if "code" in which:
        ids = tok(open(code_path, encoding="utf-8", errors="replace").read(), return_tensors="pt").input_ids[0]
        sets["code"] = [(ids[i * seq:(i + 1) * seq], None) for i in range(min(code_chunks, len(ids) // seq))]
    if max_seqs:
        sets = {k: v[:max_seqs] for k, v in sets.items()}
    return sets


_HEAD = {}


@torch.no_grad()
def _logprob_chunks(model, ids, chunk=1024):
    """fp32 lm_head: bf16 logits tie often at the top, which makes argmax/top-1 agreement noisy"""
    w = _HEAD.get(id(model))
    if w is None:
        w = _HEAD[id(model)] = model.lm_head.weight.float()
    h = model.model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False).last_hidden_state[0]
    for s in range(0, h.shape[0] - 1, chunk):
        e = min(s + chunk, h.shape[0] - 1)                     # position t predicts token t+1
        yield s, e, torch.log_softmax(h[s:e].float() @ w.T, -1)


@torch.no_grad()
def make_ref(model, sets):
    ref = {}
    for name, seqs in sets.items():
        ref[name] = []
        for ids, _ in seqs:
            ti, tl = [], []
            for s, e, lp in _logprob_chunks(model, ids):
                v, i = lp.topk(TOPK, -1)
                ti.append(i.int().cpu()); tl.append(v.half().cpu())
            ref[name].append((torch.cat(ti), torch.cat(tl)))
    return ref


@torch.no_grad()
def score(model, sets, ref=None):
    out, t0 = {}, time.time()
    for name, seqs in sets.items():
        nll, kl, top1, asst, think = [], [], [], [], []
        for si, (ids, mask) in enumerate(seqs):
            if name.startswith("eval"):
                think.append(think_mask(ids)[1:])
            tgt = ids[1:].cuda()
            for s, e, lp in _logprob_chunks(model, ids):
                nll.append(-lp.gather(1, tgt[s:e, None])[:, 0].cpu())
                if ref is not None:
                    ri, rl = ref[name][si][0][s:e].cuda().long(), ref[name][si][1][s:e].cuda().float()
                    p = rl.exp(); q = lp.gather(1, ri)
                    k = (p * (rl - q)).sum(1)
                    pt = (1 - p.sum(1)).clamp(min=0); qt = (1 - q.exp().sum(1)).clamp(min=1e-12)
                    k = k + torch.where(pt > 1e-8, pt * (pt.clamp(min=1e-12).log() - qt.log()), torch.zeros_like(pt))
                    qa = lp.argmax(1, keepdim=True)          # agree = Q's argmax is one of P's (tied) argmaxes
                    top1.append(((ri == qa) & (rl == rl[:, :1])).any(1).cpu()); kl.append(k.cpu())
                asst.append(mask[1:][s:e] if mask is not None else torch.ones(e - s, dtype=torch.bool))
        nll = torch.cat(nll); a = torch.cat(asst)
        keep = torch.ones_like(a)
        if name.startswith("eval"):                          # drop each sequence's first SKIP predictions
            pos = torch.cat([torch.arange(len(ids) - 1) for ids, _ in seqs])
            keep = pos >= SKIP
        r = {"ppl": math.exp(nll[keep].mean().item()), "tokens": int(keep.sum())}
        if name.startswith("eval"):
            r["ppl_asst"] = math.exp(nll[keep & a].mean().item())
        if ref is not None:
            kl = torch.cat(kl); t1 = torch.cat(top1)
            r.update(kl=kl[keep].mean().item(), kl_p99=kl[keep].quantile(0.99).item() if keep.sum() < 16_000_000 else None,
                     top1=t1[keep].float().mean().item())
            if name.startswith("eval"):
                r.update(kl_asst=kl[keep & a].mean().item(), top1_asst=t1[keep & a].float().mean().item())
                th = torch.cat(think) & a & keep                  # reasoning inside assistant turns
                ans = a & keep & ~torch.cat(think)                # the rest of the assistant turn: answers + tool calls
                r.update(kl_think=kl[th].mean().item() if th.any() else None, top1_think=t1[th].float().mean().item() if th.any() else None,
                         kl_answer=kl[ans].mean().item() if ans.any() else None, think_tokens=int(th.sum()))
        out[name] = r
    out["eval_s"] = round(time.time() - t0)
    return out
