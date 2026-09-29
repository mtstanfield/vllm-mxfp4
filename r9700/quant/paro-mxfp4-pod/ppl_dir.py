#!/usr/bin/env python3
"""ppl_dir.py - wikitext-2 perplexity of a plain HF checkpoint directory (bf16 base or the builder's fp16 PSEUDO
checkpoint), same scoring as custom-quant/verify_dequant_ppl.py so deltas are comparable with our earlier pod runs.
  python3 ppl_dir.py --model /workspace/models/Qwen3.8-27B-bf16 [--tokens 40960] [--seq-len 2048]"""
import argparse, math, time, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tokens", type=int, default=40960)
    ap.add_argument("--seq-len", type=int, default=2048); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sep = chr(10) + chr(10)
    text = sep.join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text, return_tensors="pt").input_ids[0][: a.tokens]
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device).eval()
    nll, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - a.seq_len + 1, a.seq_len):
            x = ids[i:i + a.seq_len].unsqueeze(0).to(a.device)
            out = model(input_ids=x, labels=x)
            nll += out.loss.item() * (a.seq_len - 1); n += a.seq_len - 1
    print(f"{a.model}: ppl {math.exp(nll / n):.4f} over {n} tokens ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
