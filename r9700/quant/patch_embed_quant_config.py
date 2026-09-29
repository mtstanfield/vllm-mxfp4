#!/usr/bin/env python3
"""Local patch (ours): let the token embedding ask the quant config for a method. The fork's Qwen3_5Model builds
VocabParallelEmbedding(vocab, hidden) with NO quant_config/prefix, so a checkpoint with an fp8 row-quantized
embed_tokens (custom-quant/fp8_embed_sharded.py, config `fp8_embed: true`, plugin _PQFp8RowEmbeddingMethod) could never
be routed. Passing quant_config is harmless for every other config: Quark returns None for embeddings -> unquantized."""
import pathlib
p = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py")
s = p.read_text()
old = ("        self.embed_tokens = VocabParallelEmbedding(\n"
       "            self.vocab_size,\n"
       "            config.hidden_size,\n"
       "        )\n")
new = ("        self.embed_tokens = VocabParallelEmbedding(\n"
       "            self.vocab_size,\n"
       "            config.hidden_size,\n"
       "            quant_config=self.quant_config,   # local patch: fp8 row embed_tokens (None -> unquantized as before)\n"
       "            prefix=f\"{prefix}.embed_tokens\",\n"
       "        )\n")
if "local patch: fp8 row embed_tokens" in s:
    print("[local patch] embed quant_config: already applied")
else:
    assert s.count(old) >= 1, "embed_tokens constructor anchor not found"
    s = s.replace(old, new)
    p.write_text(s)
    print(f"[local patch] embed quant_config applied ({new and s.count('local patch: fp8 row embed_tokens')} site(s))")

# --- the MTP drafter builds its own embed_tokens the same way and loads the checkpoint's embed tensors before the target's
# embedding is shared into it, so it needs the same routing (fp8 rows load into the fp8 method, then get replaced).
p2 = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py")
s2 = p2.read_text()
old2 = ("        self.embed_tokens = VocabParallelEmbedding(\n"
        "            config.vocab_size,\n"
        "            config.hidden_size,\n"
        "        )\n")
new2 = ("        self.embed_tokens = VocabParallelEmbedding(\n"
        "            config.vocab_size,\n"
        "            config.hidden_size,\n"
        "            quant_config=quant_config,   # local patch: fp8 row embed_tokens (drafter side)\n"
        "            prefix=f\"{prefix}.embed_tokens\",\n"
        "        )\n")
if "local patch: fp8 row embed_tokens (drafter side)" in s2:
    print("[local patch] embed quant_config (mtp drafter): already applied")
else:
    if old2 not in s2:
        import re
        m = re.search(r"        self\.embed_tokens = VocabParallelEmbedding\(\n(.*?)\n        \)\n", s2, re.S)
        assert m, "drafter embed_tokens constructor anchor not found"
        old2 = m.group(0)
        new2 = old2[:-len("        )\n")] + "            quant_config=quant_config,   # local patch: fp8 row embed_tokens (drafter side)\n" + "            prefix=f\"{prefix}.embed_tokens\",\n        )\n"
    s2 = s2.replace(old2, new2, 1)
    p2.write_text(s2)
    print("[local patch] embed quant_config (mtp drafter) applied")
