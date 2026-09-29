"""local smoke test of optimize_mxfp4.py on Windows: z-lab's CUDA rotation kernel can't build here, so register a
differentiable torch implementation (rot.rotate_ref) as paroquant.kernels.cuda before anything imports it."""
import os, runpy, sys, types
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "pylib"), os.path.join(here, "..", "paroquant")]
import rot
stub = types.ModuleType("paroquant.kernels.cuda")
stub.scaled_pairwise_rotation = lambda x, idx, theta, scales=None, group_size=128: rot.rotate_ref(
    x if scales is None else x * scales, idx, theta, group_size).to(x.dtype)
sys.modules["paroquant.kernels.cuda"] = stub
rot._kernel = None
sys.argv = [os.path.join(here, "optimize_mxfp4.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
