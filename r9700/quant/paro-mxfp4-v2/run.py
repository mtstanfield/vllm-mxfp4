#!/usr/bin/env python3
"""run.py - build one MXFP4 variant IN MEMORY (bf16 model, weights replaced by their pseudo-quantized values), score it
against the bf16 reference, and optionally save the packed codes for package.py.

  python3 run.py --make-ref                                 # bf16 baseline + reference top-K (once)
  python3 run.py --name rtn      --quant rtn                # z-lab rotations + OCP rule = the finalist, rebuilt
  python3 run.py --name ss       --quant ss                 # + per-block scale search
  python3 run.py --name gptq_ss  --quant gptq_ss --save-codes
  python3 run.py --name mxrot_ss --quant ss --rot-dir /workspace/opt/Qwen3.8-27B-bf16   # optimizer output
Results append to $WORK/results.jsonl.
"""
import argparse, json, os, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

import mx, rot, gptq, evalkl

WORK = Path(os.environ.get("WORK", "/workspace/pq2"))
# linears that read the same input tensor share one Hessian (verified on the first batch)
SIBLINGS = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
            "in_proj_qkv": "gdn_in", "in_proj_z": "gdn_in", "gate_proj": "mlp_in", "up_proj": "mlp_in"}


def load_model(path, attn="sdpa"):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                            attn_implementation=attn).eval()
    return tok, model


def targets(model, rp):
    """(layer, rel) -> nn.Linear for every module z-lab quantized (the set the serving plugin expects)."""
    want, found = set(rp.keys()), {}
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or "visual" in name or "mtp" in name:
            continue
        g = rot.LAYER_RE.search(name)
        if g and (int(g.group(1)), g.group(2)) in want:
            found[(int(g.group(1)), g.group(2))] = m
    missing = want - set(found)
    assert not missing, f"{len(missing)} quantized modules not found in the model, e.g. {sorted(missing)[:3]}"
    return found


def group_of(key):
    layer, rel = key
    parent, leaf = rel.rsplit(".", 1) if "." in rel else ("", rel)
    return (layer, parent, SIBLINGS.get(leaf, leaf))


@torch.no_grad()
def quantize_module(lin, p, mode, h=None, actorder=False):
    w = (p["weight"].to(lin.weight.device) if p["weight"] is not None else lin.weight).float()
    wr = rot.rotate(w * p["cs"].view(1, -1), p["pairs"], p["theta"])
    if h is None:
        code, sign, e = mx.quant_matrix(wr, "search" if mode == "ss" else "ocp")
    else:
        hr = rot.rotate_hessian(h, p["pairs"], p["theta"], p["cs"])
        code, sign, e = gptq.gptq_mxfp4(wr, hr, "search" if mode == "gptq_ss" else "ocp", actorder=actorder)
    q = mx.dequant_matrix(code, sign, e)
    rel_err = ((q - wr).norm() / wr.norm()).item()
    lin.weight.copy_((rot.unrotate(q, p["pairs"], p["theta"]) / p["cs"].view(1, -1)).to(lin.weight.dtype))
    return mx.pack(code, sign, e), rel_err


@torch.no_grad()
def load_codes(mods, name):
    """Set pseudo weights from a saved variant; returns the per-module rotation params actually stored."""
    from safetensors import safe_open
    f = safe_open(str(WORK / "codes" / name / "codes.safetensors"), framework="pt", device="cuda")
    params = {}
    for (layer, rel), lin in mods.items():
        pre = f"{layer}.{rel}"
        q = mx.dequant_matrix(*mx.unpack(f.get_tensor(f"{pre}.weight"), f.get_tensor(f"{pre}.weight_scale")))
        p = {"pairs": f.get_tensor(f"{pre}.pairs"), "theta": f.get_tensor(f"{pre}.theta").float(),
             "cs": 1.0 / f.get_tensor(f"{pre}.channel_scales").float().view(-1)}
        lin.weight.copy_((rot.unrotate(q, p["pairs"], p["theta"]) / p["cs"].view(1, -1)).to(lin.weight.dtype))
        params[(layer, rel)] = p
    return params


@torch.no_grad()
def load_external_mxfp4(model, ckpt_dir):
    """Load a compressed-tensors 'mxfp4-pack-quantized' checkpoint (e.g. RedHatAI/Qwen3.8-27B-MXFP4: GPTQ + AWQ
    smoothing, no rotations) as pseudo-quantized bf16 weights. Every other tensor it ships (AWQ-rescaled norms etc.)
    replaces the base's too. Weights-only: its W4A4 activations are not emulated (our server would run it W4A8)."""
    from safetensors import safe_open
    params = dict(model.named_parameters())
    idx = json.load(open(Path(ckpt_dir) / "model.safetensors.index.json"))["weight_map"]
    n_q = n_other = 0
    for shard in sorted(set(idx.values())):
        with safe_open(str(Path(ckpt_dir) / shard), framework="pt", device="cuda") as f:
            for k in f.keys():
                if k.endswith(".weight_packed"):
                    name = k[: -len("_packed")]
                    packed = f.get_tensor(k)
                    code, sign, e = mx.unpack(packed, f.get_tensor(name[: -len(".weight")] + ".weight_scale"))
                    params[name].copy_(mx.dequant_matrix(code, sign, e).to(params[name].dtype)); n_q += 1
                elif k.endswith(".weight_scale") or k.startswith("mtp."):
                    continue
                elif k in params and tuple(params[k].shape) == tuple(f.get_slice(k).get_shape()):
                    params[k].copy_(f.get_tensor(k).to(params[k].dtype)); n_other += 1
    print(f"external: {n_q} MXFP4 linears, {n_other} other tensors loaded from {ckpt_dir}", flush=True)
    assert n_other > 0, "no non-quantized tensors matched - AWQ-rescaled norms would be missing"
    assert n_q >= 100, "too few MXFP4 linears found - format mismatch?"


def add_act_fp8(mods, params):
    """Emulate the server's activation path: x_rot = rot(x / cs) quantized to fp8 e4m3 per token (W4A8), then mapped
    back so the pseudo weights (unrot(Q) / cs) see exactly x_rot_q @ Q^T."""
    def make(p):
        def f(mod, inp):
            x = inp[0]; shp = x.shape
            xr = rot.rotate(x.reshape(-1, shp[-1]).float() / p["cs"], p["pairs"], p["theta"])
            s = xr.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            xq = (xr / s).to(torch.float8_e4m3fn).float() * s
            return (rot.unrotate(xq, p["pairs"], p["theta"]) * p["cs"]).to(x.dtype).reshape(shp), *inp[1:]
        return f
    return [lin.register_forward_pre_hook(make(params[k])) for k, lin in mods.items()]


class HessianCollector:
    def __init__(self, mods):
        self.mods, self.H, self.n, self.hooks, self.check = mods, {}, defaultdict(int), [], {}
        for key, lin in mods.items():
            self.hooks.append(lin.register_forward_pre_hook(self._hook(key)))

    def _hook(self, key):
        g = group_of(key)

        def f(mod, inp):
            x = inp[0].detach()
            x2 = x.reshape(-1, x.shape[-1])
            if g in self.check:                                   # a sibling already counted this batch's input
                if not torch.equal(self.check[g], x2[:4]):
                    raise RuntimeError(f"sibling group {g} saw two different inputs in one batch ({key})")
                return
            self.check[g] = x2[:4].clone()
            h = self.H.get(g)
            if h is None:
                h = self.H[g] = torch.zeros(x2.shape[1], x2.shape[1], device=x.device, dtype=torch.float32)
            xf = x2.float()
            h.addmm_(xf.T, xf)
            self.n[g] += x2.shape[0]
        return f

    def new_batch(self):
        self.check = {}

    def hessian(self, key):
        g = group_of(key)
        return self.H[g] / self.n[g]

    def close(self):
        for h in self.hooks:
            h.remove()


class _Stop(Exception):
    pass


def _raise_stop(*_):
    raise _Stop()


def calib_windows(path, n, length):
    c = np.load(path)["calib"]
    out = []
    for row in c:
        for j in range(0, c.shape[1] - length + 1, length):
            out.append(torch.from_numpy(row[j:j + length].astype(np.int64)))
            if len(out) == n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-bf16")
    ap.add_argument("--paro", default="/workspace/models/Qwen3.8-27B-PARO")
    ap.add_argument("--rot-dir", default=None, help="optimizer output dir (per-module .pt) overriding z-lab's rotations")
    ap.add_argument("--no-tuned-weights", action="store_true", help="with --rot-dir: keep base weights, take only rotations")
    ap.add_argument("--quant", choices=["rtn", "ss", "gptq", "gptq_ss"], default="rtn")
    ap.add_argument("--name", default=None)
    ap.add_argument("--traces", default=str(WORK / "traces.npz"))
    ap.add_argument("--code", default=str(WORK / "code-sample.txt"))
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-len", type=int, default=4096)
    ap.add_argument("--calib-mix", type=float, default=0.0, help="fraction of calibration windows from general text")
    ap.add_argument("--actorder", action="store_true", help="GPTQ static act-order (descending diag(H))")
    ap.add_argument("--window", type=int, default=8, help="layers per Hessian pass (earlier windows already quantized)")
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--save-codes", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--sets", default="wiki,code,eval8k,eval32k")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences per eval set (smoke tests)")
    ap.add_argument("--from-codes", default=None, help="evaluate a saved variant (no re-quantization)")
    ap.add_argument("--external", default=None, help="score a compressed-tensors MXFP4 checkpoint dir (control)")
    ap.add_argument("--act-fp8", action="store_true", help="with --from-codes: emulate the server's per-token fp8 activations")
    ap.add_argument("--attn", default="sdpa", help="sdpa | kernels-community/flash-attn2 (if sdpa OOMs on the 32k set)")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = load_model(a.model, a.attn)
    sets = evalkl.build_sets(tok, a.traces, a.code, a.sets.split(","), max_seqs=a.max_seqs)
    ref_path = WORK / "ref.pt"

    if a.make_ref:
        ref = evalkl.make_ref(model, sets)
        torch.save(ref, ref_path)
        res = evalkl.score(model, sets, ref)
        rec = {"name": "bf16", "time": time.strftime("%F %T"), **res}
        open(WORK / "results.jsonl", "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=1)); return

    if a.external:
        load_external_mxfp4(model, a.external)
        res = evalkl.score(model, sets, torch.load(ref_path) if ref_path.exists() else None)
        rec = {"name": a.name or Path(a.external).name, "external": a.external, "time": time.strftime("%F %T"), **res}
        open(WORK / "results.jsonl", "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=1)); return

    rp = rot.RotParams(a.paro, a.rot_dir)
    mods = targets(model, rp)
    print(f"{len(mods)} modules to quantize, mode {a.quant}, rotations {'override ' + a.rot_dir if a.rot_dir else 'z-lab'}", flush=True)
    codes, errs = {}, []

    def params(key):
        p = rp.get(key)
        if a.no_tuned_weights:
            p["weight"] = None
        return p

    if a.from_codes:
        stored = load_codes(mods, a.from_codes)
        if a.act_fp8:
            add_act_fp8(mods, stored)
        name = a.name or a.from_codes + ("+a8" if a.act_fp8 else "")
        res = evalkl.score(model, sets, torch.load(ref_path) if ref_path.exists() else None)
        rec = {"name": name, "from_codes": a.from_codes, "act_fp8": a.act_fp8, "time": time.strftime("%F %T"), **res}
        open(WORK / "results.jsonl", "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=1)); return

    if a.quant in ("rtn", "ss"):
        for key, lin in sorted(mods.items()):
            (packed, e8m0), err = quantize_module(lin, params(key), a.quant)
            errs.append(err)
            if a.save_codes:
                codes[key] = (packed.cpu(), e8m0.cpu())
    else:
        calib = calib_windows(a.traces, a.calib_n, a.calib_len)
        if a.calib_mix > 0:
            import gen
            calib = gen.mix(calib, lambda k: gen.general_windows(tok, k, a.calib_len), a.calib_mix)
            print(f"calibration: {len(calib)} windows, {round(a.calib_mix * len(calib))} from wikitext-2 train", flush=True)
        layers = sorted({k[0] for k in mods})
        blocks = model.model.language_model.layers
        for w0 in range(0, len(layers), a.window):
            wl = layers[w0:w0 + a.window]
            wmods = {k: m for k, m in mods.items() if k[0] in wl}
            hc = HessianCollector(wmods)
            stop = blocks[wl[-1]].register_forward_hook(_raise_stop)   # nothing past this window is needed
            tw = time.time()
            for ids in calib:
                hc.new_batch()
                try:
                    with torch.no_grad():
                        model.model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False)
                except _Stop:
                    pass
            stop.remove(); hc.close()
            th = time.time() - tw
            for key in sorted(wmods):
                (packed, e8m0), err = quantize_module(wmods[key], params(key), a.quant, hc.hessian(key), a.actorder)
                errs.append(err)
                if a.save_codes:
                    codes[key] = (packed.cpu(), e8m0.cpu())
            del hc; torch.cuda.empty_cache()
            print(f"  layers {wl[0]}-{wl[-1]}: hessians {th:.0f}s, gptq {time.time()-tw-th:.0f}s", flush=True)
    tq = time.time() - t0
    print(f"quantized in {tq:.0f}s, mean rel weight err {np.mean(errs):.4f}", flush=True)

    name = a.name or f"{a.quant}{'-rot' if a.rot_dir else ''}"
    if a.save_codes:
        out = WORK / "codes" / name
        out.mkdir(parents=True, exist_ok=True)
        t = {}
        for (layer, rel), (packed, e8m0) in codes.items():
            p = rp.get((layer, rel), device="cpu"); fmt = rp.stored_format((layer, rel))
            pre = f"{layer}.{rel}"
            t[f"{pre}.weight"] = packed; t[f"{pre}.weight_scale"] = e8m0
            for leaf, val in (("pairs", p["pairs"]), ("theta", p["theta"]), ("channel_scales", 1.0 / p["cs"])):  # cs stored inverted
                dt, shp = fmt[leaf]
                t[f"{pre}.{leaf}"] = val.reshape(shp).to(_dtype(dt)).contiguous()
        save_file(t, str(out / "codes.safetensors"))
        json.dump({"name": name, "quant": a.quant, "rot_dir": a.rot_dir, "tuned_weights": bool(a.rot_dir) and not a.no_tuned_weights},
                  open(out / "variant.json", "w"), indent=1)
        print(f"codes -> {out}", flush=True)

    if not a.skip_eval:
        ref = torch.load(ref_path) if ref_path.exists() else None
        res = evalkl.score(model, sets, ref)
        rec = {"name": name, "quant": a.quant, "rot_dir": a.rot_dir, "calib_n": a.calib_n, "calib_mix": a.calib_mix, "quant_s": round(tq),
               "rel_err": float(np.mean(errs)), "time": time.strftime("%F %T"), **res}
        open(WORK / "results.jsonl", "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=1))


def _dtype(s):
    return {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I16": torch.int16, "I32": torch.int32}[s]


if __name__ == "__main__":
    main()
