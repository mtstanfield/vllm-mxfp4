#!/usr/bin/env python3
"""verify_dequant_ppl.py - perplexity of the quantized checkpoint (weights dequantized to bf16 in memory)
versus the bf16 original, on the same wikitext-2 test slice. Runs on the calibration pod, no vLLM needed.

  python3 verify_dequant_ppl.py --src /work/src --quant /work/out [--tokens 40960] [--seq-len 2048]

Expected: a small gap. AMD's MXFP4 of base Qwen3.8-27B measures WikiText-2 PPL 8.33 (the fork's README);
a finetune will have its own bf16 baseline, so judge the DELTA, not the absolute number.
Note: this dequantizes MXFP4 exactly as the kernel does (e2m1 * 2^e8m0); activations stay bf16 here,
whereas the server also quantizes activations (W4A4/W4A8), so this is a slightly optimistic bound.
"""
import argparse, json, math, os, time
import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def dequant_mxfp4(packed, e8m0, k):
    n = packed.shape[0]
    code = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    code[:, 0::2] = packed & 0xF; code[:, 1::2] = packed >> 4
    grid = E2M1.to(packed.device)
    mag = grid[(code & 0x7).long()]
    val = torch.where((code & 0x8) > 0, -mag, mag)
    scale = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    return (val.reshape(n, k // 32, 32) * scale).reshape(n, k)

@torch.no_grad()
def ppl(model, tok, text, tokens, seq_len, dev):
    ids = tok(text, return_tensors="pt").input_ids[0][:tokens]
    nll, cnt = 0.0, 0
    for i in range(0, ids.numel() - 1, seq_len):
        chunk = ids[i:i + seq_len + 1].unsqueeze(0).to(dev)
        if chunk.shape[1] < 2: break
        logits = model(input_ids=chunk[:, :-1], use_cache=False).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum")
        nll += loss.item(); cnt += chunk.shape[1] - 1
    return math.exp(nll / cnt), cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--quant", required=True)
    ap.add_argument("--tokens", type=int, default=40960); ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.src)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])

    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16, device_map=dev).eval()
    t0 = time.time(); p0, n = ppl(model, tok, text, args.tokens, args.seq_len, dev)
    print(f"bf16 original : ppl {p0:.4f} over {n} tokens ({time.time()-t0:.0f}s)")

    # overwrite quantized linears (and the folded norms) with the dequantized checkpoint
    idx = json.load(open(os.path.join(args.quant, "model.safetensors.index.json")))["weight_map"]
    handles = {}
    def get(name):
        f = idx[name]
        if f not in handles: handles[f] = safe_open(os.path.join(args.quant, f), "pt", device="cpu")
        return handles[f].get_tensor(name)
    sd = model.state_dict()
    replaced = 0
    for name in idx:
        if name.endswith(".weight_scale") or name not in sd: continue   # mtp.* may not be materialised by transformers
        mod = name[:-7] if name.endswith(".weight") else None
        if mod and (mod + ".weight_scale") in idx:
            w = get(name); s = get(mod + ".weight_scale")
            if w.dtype == torch.uint8:
                deq = dequant_mxfp4(w.to(dev), s.to(dev), w.shape[1] * 2)
            else:  # fp8 mtp head
                deq = w.to(dev).float() * s.to(dev).float().unsqueeze(1)
            sd[name].copy_(deq.to(sd[name].dtype)); replaced += 1
        elif name in sd and ("layernorm" in name):
            sd[name].copy_(get(name).to(sd[name].dtype))   # folded norms
    print(f"replaced {replaced} quantized linears + folded norms")
    t0 = time.time(); p1, _ = ppl(model, tok, text, args.tokens, args.seq_len, dev)
    print(f"mxfp4 dequant : ppl {p1:.4f} ({time.time()-t0:.0f}s)   delta {p1-p0:+.4f} ({(p1/p0-1)*100:+.2f}%)")

if __name__ == "__main__":
    main()
