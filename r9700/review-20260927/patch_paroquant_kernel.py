"""Local patch (2026-09-15): install the prebuilt paroquant kernel module so radiance_gemm.py can fall back to
its pq_skinny_bf16 kernel -- our libr4d build b9e42ab-rx6 has no gemm_bf16_nt_m64, so upstream skinny GEMM was
silently disabled. Built once by persist/paroquant/build.sh. Inert unless RADIANCE_SKINNY_GEMM=all: the only
skinny shape in this model (GDN in_proj_ba 96x5120) sits in radiance_gemm._CFG_TAXED.
serve-mxfp4.sh does not refresh radiance_gemm.py (only the paroquant launcher does) and the copy baked into the
0.9.3 image predates the fallback, so the repo copy is installed alongside the module."""
import shutil
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
SRC = Path("/patches/persist/paroquant/radiance_paroquant_kernel.so")
if SRC.exists():
    shutil.copy2(SRC, SP / "radiance_paroquant_kernel.so")
    shutil.copy2("/patches/radiance_gemm.py", SP / "radiance_gemm.py")
    print("[local patch] paroquant kernel module + repo radiance_gemm.py installed (skinny GEMM fallback available)")
else:
    print("[local patch] paroquant kernel module not built (persist/paroquant/build.sh); skinny GEMM stays off")
