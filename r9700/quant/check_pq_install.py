# run inside the image with RADIANCE_PAROQUANT_INSTALL=1: apply the install patch, then prove a FRESH interpreter
# imports the plugin through sitecustomize (what every vLLM process will do at startup).
import ast, subprocess, sys
exec(open("/patches/local/patch_paroquant_install.py").read())
sc = open("/usr/lib/python3.12/sitecustomize.py").read()
ast.parse(sc)
print("sitecustomize parses; tail:", repr(sc[-160:]))
r = subprocess.run([sys.executable, "-c", "import sys; print('paroquant' in sys.modules and 'radiance_paroquant' in sys.modules, 'MARK12345')"],
                   capture_output=True, text=True)
print("FRESH-IMPORT", "OK" if "MARK12345" in r.stdout and "registration failed" not in r.stderr else "FAILED")
print("stdout:", r.stdout.strip()[-200:]); print("stderr tail:", r.stderr.strip()[-400:])
