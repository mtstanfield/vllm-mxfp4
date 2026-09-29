"""local smoke-test launcher: the embedded python ignores PYTHONPATH (._pth), so prepend pylib and run the target."""
import os, runpy, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "pylib")); sys.path.insert(0, here)
target = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
