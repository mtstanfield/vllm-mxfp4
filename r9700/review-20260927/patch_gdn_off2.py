import pathlib
p = pathlib.Path("/mnt/user/appdata/vllm-radiance-next/radiance_gdn.py")
s = p.read_text()
if 'no("RADIANCE_GDN_SCAN_OFF")' in s:
    print("already"); raise SystemExit(0)
old = "        _fb(why)\n        return None\n"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, old + "\n    if _SCAN_OFF:                       # local: the whole layer takes the stock (FLA) path\n"
                          "        return no(\"RADIANCE_GDN_SCAN_OFF\")\n", 1)
p.write_text(s)
print("patched")
