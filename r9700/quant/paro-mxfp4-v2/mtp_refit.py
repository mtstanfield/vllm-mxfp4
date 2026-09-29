#!/usr/bin/env python3
"""mtp_refit.py - re-fit the MTP draft head to the QUANTIZED body it will draft for, and export it in the finalist's
fp8 layout (model-mtp.safetensors: 8 projections F8_E4M3 + F32 per-channel scale, 7 norms BF16).

Why: Qwen trained the head on bf16 hidden states and bf16 targets. Served, it reads the MXFP4 body's hidden states and
its drafts are accepted against the MXFP4 body's distribution, so part of the draft rejections are the head being fitted
to a different model. Self-distillation on the user's traces fixes that without touching the body.

Matches vLLM's Qwen3_5MultiTokenPredictor: x = fc(cat(pre_fc_norm_embedding(embed(tok[t+1])), pre_fc_norm_hidden(h[t])))
-> one full-attention decoder layer -> norm -> the shared lm_head, with h = the body's final-normed hidden state.
Target at t = the body's distribution for token t+2 (its logits at position t+1).

Metrics (held-out 8k trace windows): acceptance = sum_v min(p, q) (what probabilistic draft sampling accepts per step at
temperature 1) and greedy agreement argmax q == argmax p, for the finalist's fp8 head, the bf16 base head, and the refit.
  python3 mtp_refit.py --codes gptq_ss --act-fp8
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

import run, evalkl

WORK = run.WORK
PROJ = ("fc", "layer.self_attn.q_proj", "layer.self_attn.k_proj", "layer.self_attn.v_proj", "layer.self_attn.o_proj",
        "layer.mlp.gate_proj", "layer.mlp.up_proj", "layer.mlp.down_proj")


class MTP(nn.Module):
    def __init__(self, cfg, full_idx):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        super().__init__()
        h = cfg.hidden_size
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layer = Qwen3_5DecoderLayer(cfg, full_idx)
        self.norm = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)

    def forward(self, emb, hid, pe):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hid)], dim=-1))
        return self.norm(self.layer(x, position_embeddings=pe))


def ckpt_name(k):          # module state-dict key -> checkpoint key
    return "mtp." + k.replace("layer.", "layers.0.", 1)


def load_mtp_sd(base_dir, fp8_shard=None):
    """bf16 base head, or the finalist's fp8 head dequantized (w = fp8 * per-channel scale)."""
    if fp8_shard:
        f = safe_open(str(fp8_shard), framework="pt")
        keys = f.keys()
        sd = {}
        for k in keys:
            if k.endswith(".weight_scale"):
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                t = t.float() * f.get_tensor(k[: -len(".weight")] + ".weight_scale").float().view(-1, 1)
            sd[k] = t
    else:
        idx = json.load(open(Path(base_dir) / "model.safetensors.index.json"))["weight_map"]
        sd = {}
        for k, shard in idx.items():
            if k.startswith("mtp."):
                with safe_open(str(Path(base_dir) / shard), framework="pt") as f:
                    sd[k] = f.get_tensor(k)
    return sd


def to_module_sd(sd, mtp):
    want = set(mtp.state_dict())
    out = {k[len("mtp."):].replace("layers.0.", "layer.", 1): v.float() for k, v in sd.items()}
    missing = want - set(out)
    assert not missing, f"MTP weights missing from the checkpoint: {sorted(missing)}"
    return {k: v for k, v in out.items() if k in want}


def fp8_channel(w):
    """per-output-channel fp8 e4m3, computed on CPU (GPU division rounds scales differently -> deterministic bytes)."""
    w = w.detach().float().cpu()
    scale = (w.abs().amax(1) / 448.0).clamp(min=1e-12)
    return (w / scale.view(-1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def fp8_export(mtp, manifest):
    """-> {ckpt key: tensor} in exactly the finalist's names/dtypes/shapes."""
    t = {}
    sd = mtp.state_dict()
    for k, v in sd.items():
        ck = ckpt_name(k)
        if any(k == p + ".weight" for p in PROJ):
            t[ck], t[ck[: -len(".weight")] + ".weight_scale"] = fp8_channel(v)
        else:
            t[ck] = v.detach().cpu().to(torch.bfloat16)
    want = {k: v[:2] for k, v in manifest.items() if k.startswith("mtp.")}
    have = {k: [{"torch.float8_e4m3fn": "F8_E4M3", "torch.float32": "F32", "torch.bfloat16": "BF16"}[str(v.dtype)], list(v.shape)]
            for k, v in t.items()}
    assert want == have, f"mtp export differs from the finalist layout: {set(want) ^ set(have)} " \
                         f"{[k for k in want if k in have and want[k] != have[k]][:5]}"
    return {k: v.contiguous().cpu() for k, v in t.items()}


def fp8_roundtrip(mtp):
    """replace the projections with exactly what the exported shard dequantizes to (norms -> bf16 as exported)."""
    with torch.no_grad():
        for k, p in mtp.named_parameters():
            if any(k == q + ".weight" for q in PROJ):
                q8, s = fp8_channel(p)
                p.copy_((q8.float() * s.view(-1, 1)).to(p.device, p.dtype))
            else:
                p.copy_(p.to(torch.bfloat16).float())


def assistant_mask(ids, im_start, im_end, asst):
    """True for tokens inside an assistant turn (what the model writes, i.e. where MTP drafts), from the chat-template
    tokens; a window that starts mid-turn counts as outside until its first <|im_start|>."""
    ids = ids.tolist(); m = [False] * len(ids); inside = False; i = 0
    while i < len(ids):
        if ids[i] == im_start:
            inside = ids[i + 1:i + 1 + len(asst)] == asst
            i += 1 + (len(asst) if inside else 0)
            continue
        m[i] = inside
        if ids[i] == im_end:
            inside = False
        i += 1
    return torch.tensor(m)


@torch.no_grad()
def collect(model, windows, w_lm, topk, pe_store, masks):
    """body hidden (final norm, bf16, CPU) + the body's top-k next-token log-probs per position + assistant mask."""
    lm = model.model.language_model
    out = []
    for ids, mask in zip(windows, masks):
        h = model.model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False).last_hidden_state[0]
        ti, tl = [], []
        for s in range(0, h.shape[0], 1024):
            lp = torch.log_softmax(h[s:s + 1024].float() @ w_lm.T, -1)
            v, i = lp.topk(topk, -1); ti.append(i.int()); tl.append(v.half())
        out.append((ids, h.to(torch.bfloat16).cpu(), torch.cat(ti).cpu(), torch.cat(tl).cpu(), mask))
        if len(ids) not in pe_store:
            pe_store[len(ids)] = pe_store["_last"]
    return out


def kl_and_accept(q_lp, ri, rl):
    """q_lp [n, V] student log-probs; ri/rl [n, K] teacher top-k ids/log-probs -> (kl, accept, greedy) per row."""
    p = rl.float().exp(); q = q_lp.gather(1, ri.long())
    kl = (p * (rl.float() - q)).sum(1)
    pt = (1 - p.sum(1)).clamp(min=0); qt = (1 - q.exp().sum(1)).clamp(min=1e-12)
    kl = kl + torch.where(pt > 1e-8, pt * (pt.clamp(min=1e-12).log() - qt.log()), torch.zeros_like(pt))
    acc = torch.minimum(p, q.exp()).sum(1) + torch.minimum(pt, qt)
    greedy = q_lp.argmax(1) == ri[:, 0].long()
    return kl, acc, greedy


def mtp_logits_loss(mtp, embed, w_lm, rec, pe, grad=True, chunk=1024):
    """MTP at position t sees h[t] + embed(tok[t+1]) and predicts tok[t+2]; the loss and the metrics count only
    positions whose predicted token is assistant-written (where drafting actually happens)."""
    ids, hid, ri, rl, mask = rec
    ids = ids.cuda(); hid = hid.cuda(); ri = ri.cuda(); rl = rl.cuda()
    n = len(ids) - 1
    w = torch.zeros(n, device="cuda"); w[: n - 1] = mask[2:].float().cuda()      # target token t+2 is assistant
    denom = w.sum().clamp(min=1.0)
    cos, sin = pe
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = mtp(embed(ids[1:]).unsqueeze(0), hid[:-1].unsqueeze(0), (cos[..., :n, :], sin[..., :n, :]))[0]
    tot_kl, stats = 0.0, []
    for s in range(0, n, chunk):
        wc = w[s:s + chunk]
        if wc.sum() == 0:
            continue

        def f(xc, s=s, wc=wc):
            q = torch.log_softmax(xc.float() @ w_lm.T, -1)
            k, a, g = kl_and_accept(q, ri[s + 1:s + 1 + xc.shape[0]], rl[s + 1:s + 1 + xc.shape[0]])
            return (k * wc).sum(), a.detach()[wc > 0], g.detach()[wc > 0]
        xc = x[s:s + chunk]
        k, a, g = checkpoint(f, xc, use_reentrant=False) if grad else f(xc)
        if grad:
            (k / denom).backward(retain_graph=True)
        tot_kl += k.item(); stats.append((a, g))
    if not stats:
        return 0.0, torch.zeros(0, device="cuda"), torch.zeros(0, dtype=torch.bool, device="cuda")
    return tot_kl / denom.item(), torch.cat([a for a, _ in stats]), torch.cat([g for _, g in stats])


@torch.no_grad()
def evaluate(mtp, embed, w_lm, recs, pes):
    mtp.eval(); kls, accs, grs = [], [], []
    for rec in recs:
        kl, a, g = mtp_logits_loss(mtp, embed, w_lm, rec, pes[len(rec[0])], grad=False)
        if len(a):
            kls.append(kl * len(a)); accs.append(a); grs.append(g)
    a = torch.cat(accs).float(); g = torch.cat(grs).float()
    return {"kl": float(np.sum(kls) / len(a)), "accept": a.mean().item(), "greedy": g.mean().item(), "positions": len(a),
            # expected accepted drafts per step for SPEC=4 if each position's rate were independent (a rough guide)
            "exp_len_spec4_iid": sum(a.mean().item() ** i for i in range(1, 5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="the body variant (run.py --save-codes name)")
    ap.add_argument("--act-fp8", action="store_true")
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-bf16")
    ap.add_argument("--paro", default="/workspace/models/Qwen3.8-27B-PARO")
    ap.add_argument("--extras", default=str(WORK / "finalist-extras"))
    ap.add_argument("--traces", default=str(WORK / "traces.npz"))
    ap.add_argument("--train", type=int, default=1024, help="2048-token training windows")
    ap.add_argument("--eval", type=int, default=12, help="held-out 8k windows")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--mix", type=float, default=0.3, help="fraction of training windows from general text")
    ap.add_argument("--eval-wiki", type=int, default=8, help="held-out wikitext-2 test windows (2048)")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = WORK / "mtp" / a.codes; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model-mtp.safetensors").unlink(missing_ok=True)      # only an adopted refit may leave a head here
    tok, model = run.load_model(a.model)
    rp = run.rot.RotParams(a.paro)
    mods = run.targets(model, rp)
    stored = run.load_codes(mods, a.codes)
    if a.act_fp8:
        run.add_act_fp8(mods, stored)
    cfg = model.config.text_config
    full_idx = cfg.layer_types.index("full_attention")
    pes = {}
    lm = model.model.language_model

    def grab(mod, args, kwargs):
        pe = kwargs["position_embeddings"]; pes["_last"] = (pe[0].detach().clone(), pe[1].detach().clone())
    hk = lm.layers[full_idx].register_forward_pre_hook(grab, with_kwargs=True)
    w_lm = model.lm_head.weight.detach().float()

    c = np.load(a.traces)
    calib = [torch.from_numpy(r[j:j + 2048].astype(np.int64)) for r in c["calib"] for j in (0, 2048)]
    # train on the END of the pool (the GPTQ/optimizer calibration used the start); eval = held-out sessions
    train_w = calib[-a.train:]
    ne = min(a.eval, c["eval8k"].shape[0])
    eval_w = [torch.from_numpy(c["eval8k"][i].astype(np.int64)) for i in range(ne)]
    eval_m = [torch.from_numpy(c["eval8k_mask"][i].astype(bool)) for i in range(ne)]
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    asst = tok.encode("assistant", add_special_tokens=False)
    train_m = [assistant_mask(w, im_start, im_end, asst) for w in train_w]
    # the token-level rule must agree with prep_traces' character-span mask on the eval windows
    def _agree(w, m):                          # compare from the first <|im_start|> on (before it the role is unknown)
        first = (w == im_start).nonzero()
        s = int(first[0]) if len(first) else len(w)
        return (assistant_mask(w, im_start, im_end, asst)[s:] == m[s:]).float().mean().item() if s < len(w) else 1.0
    agree = np.mean([_agree(w, m) for w, m in zip(eval_w, eval_m)])
    print(f"assistant-mask agreement with prep_traces on eval windows: {agree:.4f}; train assistant share "
          f"{np.mean([m.float().mean().item() for m in train_m]):.3f}", flush=True)
    assert agree > 0.99, "token-level assistant mask disagrees with the prepared eval masks"
    # general text in training (wikitext-2 train, all positions) and as a second held-out set (wikitext-2 test), so the
    # head doesn't get better at drafting agent turns by getting worse at drafting side chats
    import gen
    k = round(a.mix * len(train_w))
    if k:
        gw = gen.general_windows(tok, k, 2048)
        for j, i in enumerate(torch.linspace(0, len(train_w) - 1, k).long().tolist()):
            train_w[i] = gw[j]; train_m[i] = torch.ones(2048, dtype=torch.bool)
    wiki_w = gen.general_windows(tok, a.eval_wiki, 2048, split="test")
    ev = collect(model, eval_w, w_lm, a.topk, pes, eval_m)
    ev_wiki = collect(model, wiki_w, w_lm, a.topk, pes, [torch.ones(len(w), dtype=torch.bool) for w in wiki_w])
    tr = collect(model, train_w, w_lm, a.topk, pes, train_m)
    hk.remove()
    t_collect = time.time() - t0
    embed = model.model.language_model.embed_tokens
    for p in embed.parameters():
        p.requires_grad_(False)
    # drop the body; keep embed + lm_head
    for layer in lm.layers:
        layer.to("meta")
    torch.cuda.empty_cache()

    manifest = json.load(open(Path(a.extras) / "finalist-manifest.json"))
    results = {"codes": a.codes, "act_fp8": a.act_fp8, "train_windows": len(tr), "eval_windows": len(ev), "collect_s": round(t_collect)}
    heads = {"finalist_fp8": load_mtp_sd(None, Path(a.extras) / "model-mtp.safetensors"), "base_bf16": load_mtp_sd(a.model)}
    mtp = MTP(cfg, full_idx).cuda().float()

    def ev_both():
        r = evaluate(mtp, embed, w_lm, ev, pes)
        r["wiki"] = evaluate(mtp, embed, w_lm, ev_wiki, pes)
        r["select"] = 0.7 * r["accept"] + 0.3 * r["wiki"]["accept"]
        return r

    for name, sd in heads.items():
        mtp.load_state_dict(to_module_sd(sd, mtp), strict=True)
        results[name] = ev_both()
        print(name, results[name], flush=True)
    assert results["base_bf16"]["greedy"] > 0.3, "base MTP head barely agrees with the body - wiring is wrong"
    # informational: does our fp8 recipe reproduce the finalist's MTP bytes from the bf16 base head?
    mtp.load_state_dict(to_module_sd(heads["base_bf16"], mtp))
    ours = fp8_export(mtp, manifest)
    with safe_open(str(Path(a.extras) / "model-mtp.safetensors"), framework="pt") as f:
        same = [torch.equal(ours[k].view(torch.uint8), f.get_tensor(k).view(torch.uint8)) for k in ours if ours[k].dtype == torch.float8_e4m3fn]
    results["fp8_recipe_matches_finalist"] = f"{sum(same)}/{len(same)} projections byte-identical"
    print(results["fp8_recipe_matches_finalist"], flush=True)

    # ---- self-distillation from the base bf16 head
    opt = torch.optim.AdamW(mtp.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(tr) / a.batch); step = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    best = (results["base_bf16"]["select"], {k: v.detach().clone() for k, v in mtp.state_dict().items()}, "base_bf16")
    g = torch.Generator().manual_seed(0)
    for ep in range(a.epochs):
        mtp.train()
        order = torch.randperm(len(tr), generator=g).tolist()
        for b in range(0, len(order), a.batch):
            opt.zero_grad(set_to_none=True)
            losses = []
            for i in order[b:b + a.batch]:
                kl, _, _ = mtp_logits_loss(mtp, embed, w_lm, tr[i], pes[len(tr[i][0])])
                losses.append(kl)
            for p in mtp.parameters():
                if p.grad is not None:
                    p.grad /= len(losses)
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"  ep {ep} step {step}/{steps} train kl {np.mean(losses):.4f}", flush=True)
        r = ev_both()
        results[f"epoch{ep}"] = r
        print(f"epoch {ep}: {r}", flush=True)
        if r["select"] > best[0]:
            best = (r["select"], {k: v.detach().clone() for k, v in mtp.state_dict().items()}, f"epoch{ep}")

    mtp.load_state_dict(best[1])
    fp8_roundtrip(mtp)                                 # score exactly what will be served
    results["refit_fp8"] = ev_both()
    results["best"] = best[2]
    print("refit_fp8", results["refit_fp8"], flush=True)
    fin, new = results["finalist_fp8"], results["refit_fp8"]
    keep = new["accept"] > fin["accept"] and new["wiki"]["accept"] >= fin["wiki"]["accept"] - 0.005
    results["adopted"] = keep
    if keep:
        save_file(fp8_export(mtp, manifest), str(out_dir / "model-mtp.safetensors"), metadata={"format": "pt"})
        print(f"refit head -> {out_dir / 'model-mtp.safetensors'}", flush=True)
    else:
        print("refit did not beat the finalist head; keeping the finalist's model-mtp.safetensors", flush=True)
    results["total_s"] = round(time.time() - t0)
    open(WORK / "mtp_results.jsonl", "a").write(json.dumps(results) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
