"""gen.py - general-text calibration windows (wikitext-2 TRAIN split; the eval uses the test split) mixed into the
agent-trace calibration so GPTQ / rotation training / MTP refit don't trade general-text quality for trace quality
(trace-only GPTQ: trace KL -66% but wiki PPL +2.30% vs the finalist's +0.78%)."""
import torch

_CACHE = {}


def general_windows(tok, n, length, split="train"):
    if n <= 0:
        return []
    key = (split, id(tok))
    if key not in _CACHE:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
        _CACHE[key] = tok(text, return_tensors="pt").input_ids[0]
    ids = _CACHE[key]
    stride = max(length, (len(ids) - length) // n)
    out = [ids[i * stride:i * stride + length].clone() for i in range(n) if i * stride + length <= len(ids)]
    assert len(out) == n, f"only {len(out)} general windows of {length} available (wanted {n})"
    return out


def mix(trace_windows, general, frac, seed=0):
    """replace round(frac * n) trace windows (spread evenly) with general windows; keeps the count."""
    n = len(trace_windows); k = round(frac * n)
    if k == 0:
        return list(trace_windows)
    g = general(k)
    out = list(trace_windows)
    for j, i in enumerate(torch.linspace(0, n - 1, k).long().tolist()):
        out[i] = g[j]
    return out
