import runpy, os, sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)
runpy.run_path(os.path.join(here, "autocull_gui.py"), run_name="__main__")
