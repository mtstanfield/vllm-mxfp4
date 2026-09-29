"""rot.py - ParoQuant scaled pairwise rotations (the transform vLLM-Radiance applies online) + the per-module parameters.

Convention (paroquant/kernels/cuda/rotation.cuh): rotations r = 0..KROT-1 applied in order, each a set of disjoint Givens
pairs inside 128-channel groups: x_i' = c*x_i + s*x_j, x_j' = -s*x_i + c*x_j. Weights are stored as
W_rot = rot(W * cs) and the server feeds rot(x / cs), so x @ W.T is preserved.
"""
import re
from pathlib import Path
import torch
from safetensors import safe_open

GS = 128
try:
    from paroquant.kernels.cuda import scaled_pairwise_rotation as _kernel
except Exception:                                  # no nvcc (local smoke tests): exact torch reference below
    _kernel = None

LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)$")


def rotate_ref(x, pairs, theta, gs=GS):
    shp = x.shape; h = shp[-1]; y = x.reshape(-1, h).float().clone()
    ng = h // gs
    off = (torch.arange(ng, device=x.device) * gs).unsqueeze(1)
    for r in range(pairs.shape[0]):
        iv = pairs[r].view(ng, gs).long()
        di = (iv[:, 0::2] + off).reshape(-1); dj = (iv[:, 1::2] + off).reshape(-1)
        th = theta[r].float(); c, s = th.cos(), th.sin()
        a, b = y[:, di], y[:, dj]
        y[:, di] = a * c + b * s
        y[:, dj] = b * c - a * s
    return y.reshape(shp)


def rotate(x, pairs, theta, force_ref=False):
    """x [..., K] float32 -> rotated along the last dim."""
    if _kernel is not None and x.is_cuda and not force_ref:
        h = x.shape[-1]
        return _kernel(x.reshape(-1, h).contiguous(), pairs, theta.float(), None, GS).reshape(x.shape)
    return rotate_ref(x, pairs, theta)


def unrotate(x, pairs, theta, force_ref=False):
    return rotate(x, torch.flip(pairs, [0]), -torch.flip(theta, [0]), force_ref)


def rotate_hessian(h, pairs, theta, cs):
    """E[x_rot^T x_rot] for x_rot = rot(x / cs), given H = E[x^T x]:  R D H D R^T with D = diag(1/cs)."""
    d = (1.0 / cs.float()).view(-1)
    m = h * d.view(-1, 1) * d.view(1, -1)
    a = rotate(m, pairs, theta)                  # M R^T (rows rotated)
    return rotate(a.T.contiguous(), pairs, theta)   # R M R^T


class RotParams:
    """Per-module (pairs, theta, cs) where the weight is quantized as rot(W * cs). Source: z-lab's PARO checkpoint
    (channel_scales stored inverted), optionally overridden by our optimizer's per-module .pt files."""

    def __init__(self, paro_dir, override_dir=None):
        p = Path(paro_dir)
        files = sorted(p.glob("*.safetensors"))
        self.src, self.dtypes = {}, {}
        for f in files:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    if k.endswith(".theta"):
                        mod = k[: -len(".theta")]
                        m = LAYER_RE.search(mod)
                        if m:
                            self.src[(int(m.group(1)), m.group(2))] = (f, mod)
        self.override = Path(override_dir) if override_dir else None

    def keys(self):
        return sorted(self.src)

    def stored_format(self, key):
        """{leaf: (safetensors dtype str, shape)} of z-lab's tensors, so saved variants keep the exact layout."""
        f, mod = self.src[key]
        with safe_open(f, framework="pt") as h:
            return {leaf: (h.get_slice(f"{mod}.{leaf}").get_dtype(), tuple(h.get_slice(f"{mod}.{leaf}").get_shape()))
                    for leaf in ("theta", "pairs", "channel_scales")}

    def get(self, key, device="cuda"):
        """-> dict(pairs short [KROT,K], theta f32 [KROT,K/2], cs f32 [K], weight fp16 or None)."""
        if self.override is not None:
            f = self.override / f"{key[0]}.{key[1]}.pt"
            if not f.exists():          # never fall back silently: a variant named mxrot_* must use trained rotations
                raise FileNotFoundError(f"trained rotation missing for {key}: {f}")
            if f.exists():
                sd = torch.load(f, map_location=device)
                # the checkpoint stores 1/cs and theta as fp16: quantize against exactly what the server will apply
                cs = 1.0 / (1.0 / sd["channel_scales"].float().view(-1)).half().float()
                return {"pairs": sd["pairs_grouped"].to(device), "theta": sd["angles_grouped"].half().float().to(device),
                        "cs": cs.to(device), "weight": sd.get("weight"), "src": "override"}
        f, mod = self.src[key]
        with safe_open(f, framework="pt") as h:
            pairs = h.get_tensor(f"{mod}.pairs").to(device)
            theta = h.get_tensor(f"{mod}.theta").float().to(device)
            cs = 1.0 / h.get_tensor(f"{mod}.channel_scales").float().view(-1).to(device)
        return {"pairs": pairs, "theta": theta, "cs": cs, "weight": None, "src": "zlab"}
