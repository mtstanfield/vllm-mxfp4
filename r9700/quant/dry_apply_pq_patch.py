import ast, shutil, tempfile
from pathlib import Path
src = open("/patches/local/patch_paroquant_install.py").read()
start = src.index("def _patch_plugin("); end = src.index("_patch_plugin(SP / py.name")
ns = {"Path": Path}
exec(src[start:end], ns)
for name, anchor in (("radiance_paroquant.py", '@register_quantization_config("paroquant")' + chr(10)),
                     ("radiance_paroquant_mxfp4.py", '@register_quantization_config("paroquant_mxfp4")' + chr(10))):
    tmp = Path(tempfile.mkdtemp()) / name
    shutil.copy(Path("/patches/paroquant") / name, tmp)
    before = tmp.read_text()
    ns["_patch_plugin"](tmp, anchor)
    after = tmp.read_text(); ast.parse(after)
    print(f"{name}: patched, parses, +{len(after)-len(before)} chars, routes={'_quark_full' in after}, mapper={'apply_vllm_mapper' in after}")
