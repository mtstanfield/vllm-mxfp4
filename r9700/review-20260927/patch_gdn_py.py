"""radiance_gdn.py (vllm-radiance-next tree): bind libr4d rx9x's 16-bit-state chunk scans and use them in fused_prefill
instead of declining a 16-bit SSM cache to FLA. A stock libr4d has no such entry point (the name check in
_bind_narrow_state rejects the fp32 row select() falls back to), so this changes nothing there.
RADIANCE_GDN_SCAN_NARROW=0 keeps the FLA path even when the kernel exists (A/B)."""
import pathlib, shutil
p = pathlib.Path("/mnt/user/appdata/vllm-radiance-next/radiance_gdn.py")
s = p.read_text()
if "_CHUNK_SCAN_NARROW" in s:
    print("already patched")
    raise SystemExit(0)
shutil.copy(p, str(p) + ".bak-20260927")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("""_FUSED_UPDATE_NARROW = {
    dt: _bind_narrow_state("gdn_fused_update", tag, frag, conv_width=CONV_WIDTH,
                           head_k=HEAD_K, head_v=HEAD_V)
    for dt, (tag, frag) in _STATE_TAGS.items()
}""", """_FUSED_UPDATE_NARROW = {
    dt: _bind_narrow_state("gdn_fused_update", tag, frag, conv_width=CONV_WIDTH,
                           head_k=HEAD_K, head_v=HEAD_V)
    for dt, (tag, frag) in _STATE_TAGS.items()
}
# local (2026-09-27): the chunked PREFILL scan on the 16-bit ssm cache -- libr4d rx9x only (exact decay, 16-bit
# initial/final state, fp32 inside). Without it an fp16 cache sends every prefill GDN layer to FLA. A build without
# the entry point fails the name check above and leaves this empty. RADIANCE_GDN_SCAN_NARROW=0 forces FLA (A/B).
_SCAN_NARROW = os.environ.get("RADIANCE_GDN_SCAN_NARROW", "1") == "1"
_CHUNK_SCAN_NARROW = {
    dt: _bind_narrow_state("gdn_chunk_scan", tag, frag, head_k=HEAD_K, head_v=HEAD_V, chunk=CHUNK)
    for dt, (tag, frag) in _STATE_TAGS.items()
} if _SCAN_NARROW else {}""")
rep("""    if initial_state.dtype != torch.float32:
        return _bail(f"state dtype {initial_state.dtype}")""",
"""    scan = _CHUNK_SCAN
    if initial_state.dtype != torch.float32:
        scan = _CHUNK_SCAN_NARROW.get(initial_state.dtype)      # local: rx9x 16-bit-state scan
        if scan is None:
            return _bail(f"state dtype {initial_state.dtype}")""")
rep("""    _CHUNK_SCAN(
""", """    scan(
""")
p.write_text(s)
print("radiance_gdn.py patched")
