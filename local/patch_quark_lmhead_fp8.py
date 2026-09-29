#!/usr/bin/env python3
"""Local patch (ours, not radiance): let a Quark checkpoint quantize its LM head.

QuarkConfig.get_quant_method only hands out QuarkLinearMethod for LinearBase layers; ParallelLMHead subclasses
VocabParallelEmbedding, so a Quark checkpoint whose lm_head is stored as FP8 e4m3 + per-channel scale (what
custom-quant/fp8_lmhead.py writes, mirroring the fp8 MTP head) would otherwise get UnquantizedEmbeddingMethod
and load garbage. compressed-tensors already handles ParallelLMHead exactly this way; this mirrors it, gated on
an explicit layer_quant_config match so an ordinary checkpoint (lm_head in `exclude`, or no entry) is untouched.
"""
import pathlib
p = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py")
s = p.read_text()

imp_anchor = "from vllm.model_executor.layers.linear import (\n"
imp_new = ("from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # local patch: fp8 lm_head\n"
           + imp_anchor)
old = ("        if isinstance(layer, LinearBase):\n"
       "            scheme = self.get_scheme(layer=layer, layer_name=prefix)\n"
       "            layer.scheme = scheme\n"
       "            return QuarkLinearMethod(self)\n"
       "        if isinstance(layer, Attention):\n")
new = ("        # local patch: quantized LM head (mirrors compressed_tensors' ParallelLMHead branch). Only when an\n"
       "        # explicit layer_quant_config pattern matches the head; otherwise leave it unquantized.\n"
       "        # NOTE: the head's prefix is 'language_model.lm_head' inside the multimodal wrapper, so the checkpoint's\n"
       "        # layer_quant_config key must be the pattern '*lm_head' (fp8_lmhead.py writes it that way).\n"
       "        if isinstance(layer, ParallelLMHead):\n"
       "            _lqc = self.quant_config.get(\"layer_quant_config\") or {}\n"
       "            if any(fnmatch.fnmatchcase(prefix, _pat) for _pat in _lqc):\n"
       "                scheme = self.get_scheme(layer=layer, layer_name=prefix)\n"
       "                layer.scheme = scheme\n"
       "                return QuarkLinearMethod(self)\n"
       "            return None\n"
       + old)
if "local patch: fp8 lm_head" in s:
    print("[local patch] quark fp8 lm_head: already applied")
else:
    assert imp_anchor in s, "linear import anchor not found"
    assert old in s, "LinearBase branch not found"
    assert "import fnmatch" in s, "quark.py no longer imports fnmatch"
    s = s.replace(imp_anchor, imp_new, 1).replace(old, new, 1)
    p.write_text(s)
    print("[local patch] quark fp8 lm_head applied")
