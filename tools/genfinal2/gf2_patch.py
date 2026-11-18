#!/usr/bin/env python3
"""gf2_patch.py (genfinal2, NEW 14 Nov 2026): exact text insertions into COPIES of two sources (nothing in the repo is edited; pattern of
tools/distill1/di_patch.py). Each anchor must occur exactly once, else AssertionError.

tools/best8/dgr_stage_r.py (the DTLA stage the final flow imports as dgr_stage_r):
  G1  the plain descent `        res = plain_run(True, tl)` -> `        res = GF2.descend(C, S, a, X, plain_run, tl, W, FT)`
      (gf2_hook.descend: GF2_MODE unset = this same plain call; dump / gen / gens, see gf2_hook.py)
  G2  `    if a.dgr_gnn:\\n        GN.capture(...)` -> `    if a.dgr_gnn or GF2.on():` (the GNN inputs need the cells' background
      demand, capacity and layer, taken while the free nets are out of the demand map; capture only adds keys to the cells dict)
tools/sched/run_n1.sh (launcher of the final flow): the python line
  N1  `$PY -u $R/tools/h100big/hard_exit.py ${GG_GNN:+$R/tools/groupgnn/gg_launch3.py} $CMD`
      -> `$PY -u ${GF2_DIR:?}/gf2_drive.py $R/tools/h100big/hard_exit.py $CMD`   (no GNN-start launcher)
CLI: gf2_patch.py --check      patch + compile the stage, patch + bash -n the launcher (nothing runs)
     gf2_patch.py --runner IN OUT   write the patched launcher"""
import os, subprocess, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGE = os.path.join(ROOT, "tools", "best8", "dgr_stage_r.py")
RUNNER = os.path.join(ROOT, "tools", "sched", "run_n1.sh")

G1 = "        res = plain_run(True, tl)\n"
G1_NEW = "        res = GF2.descend(C, S, a, X, plain_run, tl, W, FT)\n"
G2 = "    if a.dgr_gnn:\n        GN.capture(A, la.d, la.c, Xg, Yg)\n"
G2_NEW = "    if a.dgr_gnn or GF2.on():\n        GN.capture(A, la.d, la.c, Xg, Yg)\n"
N1 = "$PY -u $R/tools/h100big/hard_exit.py ${GG_GNN:+$R/tools/groupgnn/gg_launch3.py} $CMD"
N1_NEW = "$PY -u ${GF2_DIR:?}/gf2_drive.py $R/tools/h100big/hard_exit.py $CMD"


def _rep(s, old, new, nm):
    assert s.count(old) == 1, "gf2: anchor %s x%d (need exactly 1)" % (nm, s.count(old))
    return s.replace(old, new)


def patch_stage(s):
    s = _rep(s, G1, G1_NEW, "G1 plain descent")
    return _rep(s, G2, G2_NEW, "G2 capture")


def patch_runner(s):
    return _rep(s, N1, N1_NEW, "N1 python line")


def load_stage():
    """the patched dgr_stage_r as sys.modules['dgr_stage_r'] (paths as the chain sets them: tools/best7, tools/best8, this folder)"""
    for p in (HERE, os.path.join(ROOT, "tools", "best8"), os.path.join(ROOT, "tools", "best7")):
        if p not in sys.path:
            sys.path.insert(0, p)
    src = patch_stage(open(STAGE).read())
    mod = types.ModuleType("dgr_stage_r"); mod.__file__ = STAGE
    import gf2_hook
    mod.GF2 = gf2_hook
    sys.modules["dgr_stage_r"] = mod
    exec(compile(src, STAGE, "exec"), mod.__dict__)
    return mod


def check():
    s0 = open(STAGE).read(); s = patch_stage(s0); compile(s, STAGE, "exec")
    r = patch_runner(open(RUNNER).read())
    rc = subprocess.run(["bash", "-n"], input=r, text=True, capture_output=True).returncode
    return ["stage: ok, +%d lines" % (s.count("\n") - s0.count("\n")), "runner: %s (bash -n rc %d)" % ("ok" if rc == 0 else "SYNTAX ERROR", rc)], rc


if __name__ == "__main__":
    if "--check" in sys.argv:
        out, rc = check()
        for l in out:
            print("gf2_patch", l)
        sys.exit(1 if rc else 0)
    elif sys.argv[1:2] == ["--runner"] and len(sys.argv) == 4:
        try:
            src = patch_runner(open(sys.argv[2]).read())
        except (AssertionError, OSError) as e:
            sys.stderr.write("gf2_patch --runner: %s\n" % e); sys.exit(3)
        with open(sys.argv[3], "w") as fh:
            fh.write(src)
        print("gf2_patch runner: %s -> %s" % (sys.argv[2], sys.argv[3]))
    else:
        sys.stderr.write(__doc__); sys.exit(2)
