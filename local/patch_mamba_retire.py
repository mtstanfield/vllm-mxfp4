#!/usr/bin/env python3
"""Local patch (ours, 2026-09-27): backport vLLM PR #55450 "[Bugfix][Core] Retire Mamba states across null gaps" (merged
upstream 2026-09-11, after v0.29.0) into the 0.29 image.

In mamba_cache_mode=align, prefill leaves null gaps between the per-step state blocks. The base
_remove_blocks_in_range scans backwards and STOPS at the first null block, so every older state block stays pinned to
the running request until it finishes -- a long prefill holds one extra block per mamba group per prefill step.
The fix: an align-mode override that skips nulls instead of stopping, and remembers how far it has retired.

Gated: RADIANCE_LOCAL_MAMBA_RETIRE=1 applies it; unset leaves the file untouched (A/B-able)."""
import os
import sys
import sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_LOCAL_MAMBA_RETIRE", "0") != "1":
    sys.exit(0)

F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "v1" / "core" / "single_type_kv_cache_manager.py"
src = F.read_text()
if "_num_retired_blocks" in src:
    print("[mamba-retire] already applied")
    sys.exit(0)

EDITS = [
    (   # 1. per-request retired prefix, next to the align-mode bookkeeping
        "            self.last_state_block_idx: dict[str, int] = {}\n",
        "            self.last_state_block_idx: dict[str, int] = {}\n"
        "            self._num_retired_blocks: dict[str, int] = {}\n",
    ),
    (   # 2. the override, placed right before MambaManager.remove_skipped_blocks
        "    def remove_skipped_blocks(\n"
        "        self,\n"
        "        request_id: str,\n"
        "        processed_computed_tokens: int,\n"
        "        num_prompt_tokens: int | None = None,\n"
        "    ) -> None:\n"
        "        assert isinstance(self.kv_cache_spec, MambaSpec)\n",
        "    def _remove_blocks_in_range(\n"
        "        self, request_id: str, first_block: int, last_block: int\n"
        "    ) -> None:\n"
        "        # backport of vLLM #55450: retire align-mode states across null gaps\n"
        "        if self.mamba_cache_mode != \"align\":\n"
        "            return super()._remove_blocks_in_range(request_id, first_block, last_block)\n"
        "        blocks = self.req_to_blocks.get(request_id, [])\n"
        "        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))\n"
        "        last_block = min(last_block, len(blocks))\n"
        "        if first_block >= last_block:\n"
        "            return\n"
        "        freed: list[KVCacheBlock] = []\n"
        "        # Mamba prefill leaves null gaps between states awaiting retirement.\n"
        "        for i in range(last_block - 1, first_block - 1, -1):\n"
        "            if blocks[i].is_null:\n"
        "                continue\n"
        "            freed.append(blocks[i])\n"
        "            blocks[i] = self._null_block\n"
        "        if freed:\n"
        "            self.block_pool.free_blocks(freed)\n"
        "        self._num_retired_blocks[request_id] = last_block\n"
        "\n"
        "    def remove_skipped_blocks(\n"
        "        self,\n"
        "        request_id: str,\n"
        "        processed_computed_tokens: int,\n"
        "        num_prompt_tokens: int | None = None,\n"
        "    ) -> None:\n"
        "        assert isinstance(self.kv_cache_spec, MambaSpec)\n",
    ),
    (   # 3. forget the request's retired prefix when its blocks go back to the pool
        "            self.last_state_block_idx.pop(request_id, None)\n",
        "            self.last_state_block_idx.pop(request_id, None)\n"
        "            self._num_retired_blocks.pop(request_id, None)\n",
    ),
]
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"[mamba-retire] anchor found {n} times, NOT applied: {old.splitlines()[0]!r}")
        sys.exit(0)
for old, new in EDITS:
    src = src.replace(old, new)
F.write_text(src)
print("[mamba-retire] applied vLLM #55450 backport to", F)
