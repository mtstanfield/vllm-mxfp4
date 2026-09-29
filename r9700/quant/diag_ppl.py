#!/usr/bin/env python3
"""diag_ppl.py - localize a bad quant: swap in ONE group of quantized tensors at a time and measure PPL.
  python3 diag_ppl.py --src SRC --quant OUT --tokens 8192
Groups: mlp (gate/up/down + post_attention_layernorm), attn (q/k/v/o + input_layernorm on full-attn layers),
gdn (in_proj_* / out_proj + input_layernorm on linear-attn layers). Also 'mlp-nonorm' = mlp linears WITHOUT
the folded norm (should be WORSE than mlp if the fold is what keeps it right), and a direct tensor check:
dequant(W_out)/s vs W_src where s is implied by the norm ratio.
"""
import argparse, json, math, os
import torch
from safetensors import safe_open
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def dequant(packed, e8m0, k):
    n = packed.shape[0]
    code = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    code[:, 0::2] = packed & 0xF; code[:, 1::2] = packed >> 4
    grid = E2M1.to(packed.device); mag = grid[(code & 0x7).long()]
    val = torch.where((code & 0x8) > 0, -mag, mag)
    return (val.reshape(n, k // 32, 32) * torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)).reshape(n, k)

class CK:
    def __init__(self, d):
        idx = os.path.join(d, "model.safetensors.index.json")
        self.d = d; self.map = json.load(open(idx))["weight_map"]; self.h = {}
    def get(self, n):
        f = self.map[n]
        if f not in self.h: self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(n)
    def has(self, n): return n in self.map
    def deq(self, mod, dev):
        w = self.get(mod + ".weight").to(dev); s = self.get(mod + ".weight_scale").to(dev)
        return dequant(w, s, w.shape[1] * 2) if w.dtype == torch.uint8 else w.float() * s.float().unsqueeze(1)

@torch.no_grad()
def ppl(model, ids, seq_len, dev):
    nll, cnt = 0.0, 0
    for i in range(0, ids.numel() - 1, seq_len):
        chunk = ids[i:i + seq_len + 1].unsqueeze(0).to(dev)
        if chunk.shape[1] < 2: break
        logits = model(input_ids=chunk[:, :-1], use_cache=False).logits.float()
        nll += torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum").item(); cnt += chunk.shape[1] - 1
    return math.exp(nll / cnt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--quant", required=True)
    ap.add_argument("--tokens", type=int, default=8192); ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(a.src)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][:a.tokens]
    src, out = CK(a.src), CK(a.quant)
    LP = "model.language_model.layers"
    cfg = json.load(open(os.path.join(a.src, "config.json")))["text_config"]
    nl, lt = cfg["num_hidden_layers"], cfg["layer_types"]

    # --- direct tensor check on two layers: recover s from the norm ratio, compare dequant(W)/s to W_src
    print("=== tensor check (rel err of dequant(W_out)/s vs W_src; ~0.12 expected)")
    for l in (3, 5):
        P = f"{LP}.{l}"
        s = src.get(f"{P}.post_attention_layernorm.weight").float() / out.get(f"{P}.post_attention_layernorm.weight").float()
        for m in ("gate_proj",):
            W = src.get(f"{P}.mlp.{m}.weight").float().to(dev); D = out.deq(f"{P}.mlp.{m}", dev) / s.to(dev)
            print(f"  L{l} mlp.{m}: rel {((D - W).norm() / W.norm()).item():.4f}   s range {s.min().item():.3f}..{s.max().item():.3f}")
        if lt[l] == "full_attention":
            s = src.get(f"{P}.input_layernorm.weight").float() / out.get(f"{P}.input_layernorm.weight").float()
            W = src.get(f"{P}.self_attn.q_proj.weight").float().to(dev); D = out.deq(f"{P}.self_attn.q_proj", dev) / s.to(dev)
            print(f"  L{l} attn.q_proj: rel {((D - W).norm() / W.norm()).item():.4f}")
            W = src.get(f"{P}.self_attn.o_proj.weight").float().to(dev); D = out.deq(f"{P}.self_attn.o_proj", dev)
            print(f"  L{l} attn.o_proj (RTN): rel {((D - W).norm() / W.norm()).item():.4f}")
        else:
            s = src.get(f"{P}.input_layernorm.weight").float() / out.get(f"{P}.input_layernorm.weight").float()
            for m in ("in_proj_qkv", "in_proj_a"):
                W = src.get(f"{P}.linear_attn.{m}.weight").float().to(dev); D = out.deq(f"{P}.linear_attn.{m}", dev) / s.to(dev)
                print(f"  L{l} gdn.{m}: rel {((D - W).norm() / W.norm()).item():.4f}")
        # down: implied s2 from up rows
        Wu = src.get(f"{P}.mlp.up_proj.weight").float().to(dev); Du = out.deq(f"{P}.mlp.up_proj", dev) / s.to(dev) if False else None

    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16, device_map=dev).eval()
    base = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(LP)}
    sd = model.state_dict()
    print(f"bf16 baseline ppl {ppl(model, ids, a.seq_len, dev):.4f}")

    def restore():
        for k, v in base.items(): sd[k].copy_(v)
    def apply(names):
        for n in names:
            if n.endswith(".weight") and out.has(n[:-7] + ".weight_scale"):
                sd[n].copy_(out.deq(n[:-7], dev).to(sd[n].dtype))
            elif n in sd:
                sd[n].copy_(out.get(n).to(sd[n].dtype))
    groups = {"mlp": [], "mlp-nonorm": [], "attn": [], "gdn": [], "norms-only": []}
    for l in range(nl):
        P = f"{LP}.{l}"
        mlp = [f"{P}.mlp.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        groups["mlp"] += mlp + [f"{P}.post_attention_layernorm.weight"]
        groups["mlp-nonorm"] += mlp
        groups["norms-only"] += [f"{P}.post_attention_layernorm.weight", f"{P}.input_layernorm.weight"]
        if lt[l] == "full_attention":
            groups["attn"] += [f"{P}.self_attn.{m}.weight" for m in ("q_proj", "k_proj", "v_proj", "o_proj")] + [f"{P}.input_layernorm.weight"]
        else:
            groups["gdn"] += [f"{P}.linear_attn.{m}.weight" for m in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")] + [f"{P}.input_layernorm.weight"]
    for g, names in groups.items():
        restore(); apply(names)
        print(f"group {g:11s} ({len(names):4d} tensors): ppl {ppl(model, ids, a.seq_len, dev):.4f}", flush=True)
    # single layer, mlp only
    restore(); apply([n for n in groups["mlp"] if f".layers.0." in n])
    print(f"layer0 mlp only: ppl {ppl(model, ids, a.seq_len, dev):.4f}")

if __name__ == "__main__":
    main()
