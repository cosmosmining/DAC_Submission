"""hard_exit.py (11 Nov 2026, NEW): start a driver exactly as `python driver.py ...` does (same pattern as tools/groupgnn/gg_launch3.py:
sys.path[0] = the driver's folder, run as __main__), but END THE PROCESS when the driver raises.
Why: on the rented H100 two failed large-design runs printed their traceback and then stayed alive for good (the main thread had ended,
worker threads slept), holding 17-64 GB of memory and 34 GB of GPU memory each and blocking their lane. A normal end is left untouched.
Use: python -u hard_exit.py [gg_launch3.py] tools/day1/cold_d1.py <flags>"""
import os, runpy, sys, traceback

drv = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(drv)
try:
    runpy.run_path(drv, run_name="__main__")
except SystemExit as e:
    c = e.code
    if c not in (None, 0):
        print(f"[hard_exit] SystemExit {c!r}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(c if isinstance(c, int) else 1)
except BaseException:
    traceback.print_exc()
    print("[hard_exit] the driver raised; ending the process", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
