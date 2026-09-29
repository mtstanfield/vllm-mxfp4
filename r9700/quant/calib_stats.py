#!/usr/bin/env python3
"""calib_stats.py - per-input-channel activation statistics for AWQ scale search (Qwen3.5/3.8-27B).

Loads the bf16 checkpoint, hooks the input of one representative linear per "fold group" in every
decoder layer, runs calibration text through it, and writes mean |x| per input channel:

  layers.N.attn_in   input of self_attn.q_proj              (full-attention layers)   [5120]
  layers.N.gdn_in    input of linear_attn.in_proj_qkv       (linear-attention layers) [5120]
  layers.N.mlp_in    input of mlp.gate_proj                                            [5120]
  layers.N.down_in   input of mlp.down_proj (= act(gate)*up)                           [17408]

Those are exactly the four places quantize_mxfp4_qwen35.py folds an AWQ scale (into input_layernorm,
post_attention_layernorm, and up_proj's rows). o_proj / out_proj / MTP are round-to-nearest and need
no statistics.

Usage (single 80 GB GPU is enough for the bf16 27B):
  python3 calib_stats.py --model /work/src --out /work/stats.pt [--samples 128] [--seq-len 2048]
Calibration mix (any source that fails to download is skipped with a warning):
  wikitext-2 train text, UltraChat SFT conversations rendered through the model's chat template,
  and code from the-stack-smol. Change --mix to reweight.
"""
import argparse, random, sys, time, warnings
import torch

def build_texts(tok, n, mix, seed):
    rnd = random.Random(seed)
    texts = []
    want = {k: int(round(n * v)) for k, v in mix.items()}
    from datasets import load_dataset
    # 1. wikitext (datasets>=4 needs the namespaced id)
    if want.get("wiki", 0):
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            docs = [t for t in ds["text"] if len(t) > 400]
            rnd.shuffle(docs)
            # glue paragraphs into long samples
            buf, out = "", []
            for d in docs:
                buf += d + "\n"
                if len(buf) > 12000:
                    out.append(buf); buf = ""
                if len(out) >= want["wiki"]: break
            texts += [("wiki", t) for t in out]
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"wikitext unavailable: {e}")
    # 2. chat (rendered with the chat template so the sampled distribution includes role tokens)
    if want.get("chat", 0):
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
            got = 0
            for row in ds:
                msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"] if m["role"] in ("user", "assistant", "system")]
                if len(msgs) < 2: continue
                try:
                    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    t = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                texts.append(("chat", t)); got += 1
                if got >= want["chat"]: break
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"ultrachat unavailable: {e}")
    # 3. code - two ungated sources (verified loadable under datasets 5): half raw multi-language functions
    #    from CodeSearchNet glued into long files, half coding Q/A rendered through the chat template
    if want.get("code", 0):
        got = 0
        half = max(1, want["code"] // 2)
        try:
            ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
            buf = []
            for row in ds:
                buf.append(row.get("whole_func_string") or "")
                if sum(len(b) for b in buf) > 9000:
                    texts.append(("code", "\n\n".join(buf))); buf = []; got += 1
                    if got >= half: break
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"code_search_net unavailable: {e}")
        try:
            ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
            for row in ds:
                q, a = row.get("query") or "", row.get("answer") or ""
                if len(a) < 800: continue
                msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                try:
                    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    t = f"user: {q}\nassistant: {a}"
                texts.append(("code", t)); got += 1
                if got >= want["code"]: break
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"CodeFeedback unavailable: {e}")
        if got < want["code"]:
            warnings.warn(f"only {got}/{want['code']} code samples found")
    rnd.shuffle(texts)
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--mix", default="wiki=0.3,chat=0.4,code=0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    mix = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}

    from transformers import AutoTokenizer, AutoConfig, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    tc = cfg.text_config
    layer_types = list(tc.layer_types)
    print(f"layers={tc.num_hidden_layers} full_attn={layer_types.count('full_attention')} linear_attn={layer_types.count('linear_attention')}")

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s")

    lm = model.model.language_model
    stats = {}   # name -> [sum|x| (float64 on cpu), count]
    hooks = []
    def hook_for(name):
        def pre(mod, inputs):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).abs().float().sum(0).double().cpu()
            n = inputs[0].numel() // inputs[0].shape[-1]
            if name in stats:
                stats[name][0] += x; stats[name][1] += n
            else:
                stats[name] = [x, n]
        return pre
    for i, layer in enumerate(lm.layers):
        if layer_types[i] == "full_attention":
            hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(hook_for(f"layers.{i}.attn_in")))
        else:
            hooks.append(layer.linear_attn.in_proj_qkv.register_forward_pre_hook(hook_for(f"layers.{i}.gdn_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(hook_for(f"layers.{i}.mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook_for(f"layers.{i}.down_in")))

    texts = build_texts(tok, args.samples, mix, args.seed)
    if not texts:
        sys.exit("no calibration text could be loaded")
    kinds = {}
    for k, _ in texts: kinds[k] = kinds.get(k, 0) + 1
    print(f"calibration samples: {len(texts)} {kinds}")

    t0 = time.time()
    with torch.no_grad():
        for j, (kind, t) in enumerate(texts):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.to(args.device)
            if ids.shape[1] < 64: continue
            model(input_ids=ids, use_cache=False)
            if (j + 1) % 16 == 0:
                print(f"  {j+1}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    for h in hooks: h.remove()

    out = {name: (s / max(n, 1)).float() for name, (s, n) in stats.items()}
    tokens = next(iter(stats.values()))[1]
    torch.save({"absmean": out, "tokens": tokens, "samples": len(texts), "mix": kinds, "model": args.model}, args.out)
    print(f"wrote {args.out}: {len(out)} tensors over {tokens} tokens in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
