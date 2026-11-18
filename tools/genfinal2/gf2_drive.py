#!/usr/bin/env python3
"""gf2_drive.py (genfinal2, NEW 14 Nov 2026): gf2_drive.py <driver.py> [driver args...]: loads the patched DTLA stage (gf2_patch.py,
module GF2 = gf2_hook.py, mode from env GF2_MODE), then runs the driver as __main__ (as tools/distill1/di_drive.py). At the end
(every flow ends in os._exit) one line "gf2_drive: mode=<m> calls=<n> gen=<n> dumps=<n> fallbacks=<n>" goes to stderr."""
import os, runpy, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gf2_patch, gf2_hook


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__); sys.exit(2)
    m = gf2_hook.mode()
    if m == "dump" and not (os.environ.get("GF2_OUT") and os.environ.get("GF2_BIN0")):
        sys.stderr.write("gf2_drive: dump needs GF2_OUT and GF2_BIN0\n"); sys.exit(2)
    if m in ("gen", "gens") and not os.environ.get("GF2_MODELS"):
        sys.stderr.write("gf2_drive: gen/gens need GF2_MODELS\n"); sys.exit(2)
    gf2_patch.load_stage()
    real = os._exit

    def guard(code):
        S = gf2_hook.ST
        sys.stderr.write("gf2_drive: mode=%s calls=%d gen=%d dumps=%d fallbacks=%d\n" % (m or "off", S["calls"], S["gen"], S["dumps"], S["fallback"]))
        sys.stderr.flush()
        real(code)
    os._exit = guard
    drv = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(drv))
    runpy.run_path(drv, run_name="__main__")
    guard(0)


if __name__ == "__main__":
    main()
