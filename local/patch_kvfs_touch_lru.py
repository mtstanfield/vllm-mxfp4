#!/usr/bin/env python3
"""Local patch (ours, not radiance): make the KV-offload filesystem tier touch a block file on every lookup,
so its mtime means "last used" instead of "created". The cache pool is mounted noatime and the tier never
rewrites an existing block, so without this the only timestamp is creation time, and pruning by it would be
FIFO -- which evicts a deep session's OPENING blocks first, exactly the ones a prefix hit chains from.
kv-fs-prune.sh (Unraid User Script) deletes least-recently-used files down to a size cap using that mtime.
Cost: one utime() per looked-up block file (~250 per 100k-token lookup), off the request path (lookup thread).
"""
import pathlib
p = pathlib.Path("/opt/vllm/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py")
s = p.read_text()
old = ("        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]\n"
       "        if _HAS_BATCH_LOOKUP_C:\n")
new = ("        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]\n"
       "        # local patch: touch on lookup so mtime == last use (LRU pruning by kv-fs-prune.sh; pool is noatime)\n"
       "        for _p in paths:\n"
       "            try:\n"
       "                os.utime(_p, None)\n"
       "            except OSError:\n"
       "                pass\n"
       "        if _HAS_BATCH_LOOKUP_C:\n")
if new in s:
    print("[local patch] kvfs touch-on-lookup: already applied")
else:
    assert old in s, "fs tier batch_lookup block not found"
    assert "import os" in s, "fs manager does not import os"
    p.write_text(s.replace(old, new, 1))
    print("[local patch] kvfs touch-on-lookup applied")
