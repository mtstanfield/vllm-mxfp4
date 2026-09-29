#!/usr/bin/env python3
"""Local patch (ours, 2026-09-27): RADIANCE_LOCAL_OVERLAY="radiance_r4d_attn.py ..." copies those modules from /patches
over the image's baked site-packages copies at container start. serve-mxfp4.sh copies only its own fixed list of
radiance_*.py, so an edit to any other module (e.g. the R4D attention backend) is silently ignored without this.
Unset (production): nothing happens."""
import os
import shutil
import sysconfig

names = os.environ.get("RADIANCE_LOCAL_OVERLAY", "").split()
sp = sysconfig.get_paths()["purelib"]
for n in names:
    src = os.path.join("/patches", n)
    if os.path.basename(n) != n or not n.endswith(".py") or not os.path.isfile(src):
        print(f"[local overlay] skipped {n!r}: not a /patches/*.py file")
        continue
    shutil.copy2(src, os.path.join(sp, n))
    print(f"[local overlay] {n} -> {sp}")
