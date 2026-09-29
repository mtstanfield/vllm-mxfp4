import pathlib
p = pathlib.Path("/mnt/user/appdata/vllm-radiance-next/radiance_gdn.py")
s = p.read_text()
if "RADIANCE_GDN_SCAN_OFF" in s:
    print("already"); raise SystemExit(0)
old = '_SCAN_NARROW = os.environ.get("RADIANCE_GDN_SCAN_NARROW", "1") == "1"\n'
new = old + ('# local (2026-09-27): RADIANCE_GDN_SCAN_OFF=1 makes fused_prefill decline every call, i.e. prefill GDN runs FLA\n'
             '# with whatever state dtype -- the reference arm for A/B-ing the libr4d chunk scan (default 0: no change).\n'
             '_SCAN_OFF = os.environ.get("RADIANCE_GDN_SCAN_OFF", "0") == "1"\n')
assert s.count(old) == 1
s = s.replace(old, new)
old2 = "    if initial_state is None or not output_final_state:\n"
assert s.count(old2) == 1
s = s.replace(old2, "    if _SCAN_OFF:\n        return _bail(\"RADIANCE_GDN_SCAN_OFF\")\n" + old2)
p.write_text(s)
print("patched")
