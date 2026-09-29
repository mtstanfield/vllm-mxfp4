#!/usr/bin/env python3
"""Local patch (ours, not radiance): QuarkW8A8Fp8 maps a DYNAMIC per-tensor activation config to the STATIC
per-tensor quant key, so the scaled-mm kernel builds QuantFP8(static=True) while the scheme registers no
input_scale -> 'assert (scale is not None) == self.static' in input_quant_fp8.py. Hit by the fp8-requantized
MTP head (mtp.fc etc.: input_tensors per_tensor + is_dynamic=true).
Map any dynamic activation config to the dynamic per-TOKEN key: the kernel vLLM selects for per-channel weights
(ChannelWiseTorchFP8ScaledMMLinearKernel) unpads the activation scale per token (torch.narrow), so a (1,)
per-tensor dynamic scale breaks torch.compile there; per-token dynamic quant is the supported path and is at
least as accurate."""
import pathlib
p = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py")
s = p.read_text()
old = ("        self.activation_quant_key = (\n"
       "            kFp8DynamicTokenSym if per_token_activation else kFp8StaticTensorSym\n"
       "        )\n")
new = ("        # local patch: dynamic activations (per-token OR per-tensor config) -> dynamic per-token key\n"
       "        self.activation_quant_key = (\n"
       "            kFp8StaticTensorSym if self.is_static_input_scheme else kFp8DynamicTokenSym\n"
       "        )\n")
if new in s:
    print("[local patch] quark fp8 dynamic-token: already applied")
else:
    assert old in s, "activation_quant_key block not found"
    p.write_text(s.replace(old, new, 1))
    print("[local patch] quark fp8 dynamic-token key applied")
