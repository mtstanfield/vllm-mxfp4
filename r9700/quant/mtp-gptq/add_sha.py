"""add_sha.py <hc dir> <in.pt> <out.pt> -- add the src_sha1 fingerprint (mtp_gptq.src_sha1) to a GPTQ file made before
fingerprints existed, from the calibration dumps' fp8 weights + scales."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
hc, src, dst = sys.argv[1:4]
import hashlib
def src_sha1(w8, scale):
    h = hashlib.sha1(w8.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    h.update(scale.float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()
res = torch.load(src, map_location="cpu")
for p, ent in res.items():
    d = torch.load(os.path.join(hc, p + ".pt"), map_location="cpu")
    ent["src_sha1"] = src_sha1(d["w8"], d["scale"])
    print(p, ent["src_sha1"])
torch.save(res, dst)
print("wrote", dst)
