#!/usr/bin/env python3
"""cold_y_pp2.py (powerpred2y, NEW 2026-11-13): driver of the FINAL flow with the SCORE-route power list (PP2_LIST=score): the power
part o2 of power_round_i = the same number of nets, taken from all priced nets not in o1, ranked by activity x reducible capacitance
(tools/powerpred2/pp2_sel.score_list, imported). It runs tools/sched/cold_n1.py (the final flow's driver, run_n1.sh arms n1y / n1)
with builtins.compile hooked, so the chain source cold_n1 compiles (tools/final/chain_y.py or tools/day1/chain_d1.py, after cold_n1's
own --sched_* insertions) is replaced by a patched copy (y_patch.patch_y). Nothing in the repo is edited.

Placement: run_n1.sh starts `$SD/cold_n1.py`; point SCHED_DIR at a folder where cold_n1.py is a symlink to this file and run_n1.sh a
symlink to the real one (run_h100_pp2y.sh makes it). This file resolves its own folder with realpath, so the symlink works.
Arguments: the cold_n1 arguments, unchanged (incl. --sched_src y|d1 --sched_root <tree> and the FINAL4 extras).
Env: PP2_LIST score (the arm) | plain (identity run: must reproduce the plain flow's route bytes); PP_MODE off | log (no TIME selection
here); PP_OUT=<prefix> (records kind=patch / list / round in <prefix>.pp2.jsonl, label files of pp_lib); PP2Y_COLD=<cold_n1.py> (default
<sched_root>/tools/sched/cold_n1.py; use the copy the baseline ran with); PP2Y_DRYRUN=<file>: write the patched source, compile, exit 0.
Exit codes: 2 bad env / arguments, 3 this machine or tree cannot run the final flow (pp2_hook.server_check, imported), 4 the chain patch
was never applied (pp2_patch.install_exit_guard: also on os._exit(0))."""
import builtins, json, os, py_compile, runpy, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "tools", "powerpred2"), os.path.join(ROOT, "tools", "powerpred"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import y_patch

TAG = "cold_y_pp2"
REFUSED = ("--sched_fill", "--sched_value")   # cold_n1 paths that pass power_round_i itself (no PPL.wrap round record)


def _opt(argv, key, default=None):
    """value of the LAST occurrence (cold_n1.py's own parse keeps the last one)"""
    i = len(argv) - 1 - argv[::-1].index(key) if key in argv else -1
    return argv[i + 1] if 0 <= i < len(argv) - 1 else default


def checks(env, argv):
    """y-specific start checks on top of pp2_sel.preflight -> list of error texts"""
    E = []
    if env.get("PP2_LIST", "plain") not in ("plain", "score"):
        E.append("PP2_LIST must be score (the arm) or plain (identity run)")
    if env.get("PP_MODE", "log") not in ("off", "log"):
        E.append("PP_MODE=%s: this driver runs the SCORE route only (off | log); TIME modes run through tools/powerpred2" % env["PP_MODE"])
    for k in ("--sched_src", "--sched_model", "--sched_root"):
        if argv.count(k) > 1:
            E.append("%s given %d times (pp2_hook.server_check reads the first, cold_n1 the last)" % (k, argv.count(k)))
    if _opt(argv, "--sched_src") not in ("y", "d1"):
        E.append("--sched_src y (chain_y) or d1 (chain_d1, MemPool-Group big mode) is required")
    if _opt(argv, "--sched_model", "off") != "off":
        E.append("--sched_model must be off (the final flow FINAL4)")
    for k in REFUSED:
        if _opt(argv, k, "0") not in ("0", None):
            E.append("%s is refused: it calls power_round_i outside the wrapped call site" % k)
    return E


def make_hook(orig, dryrun=None, state=None, log=None):
    """builtins.compile replacement: a source whose file name ends in chain_y.py / chain_d1.py and that defines power_round_i is
    replaced by y_patch.patch_y(source) (state['applied'] += 1); every other compile is unchanged"""
    state = state if state is not None else {"applied": 0, "files": []}

    def comp(src, fn, mode, *a, **k):
        if mode == "exec" and isinstance(src, str) and str(fn).endswith(y_patch.SUFFIXES) and "def power_round_i" in src:
            src, rep = y_patch.patch_y(src, os.path.basename(str(fn)))
            state["applied"] += 1; state["files"].append(str(fn))
            rec = dict(kind="patch", file=str(fn), list=os.environ.get("PP2_LIST", "plain"), **rep)
            (log or (lambda r: print("PP2Y " + json.dumps(r), flush=True)))(rec)
            if dryrun:
                with open(dryrun, "w") as fh:
                    fh.write(src)
                py_compile.compile(dryrun, cfile=dryrun + "c", doraise=True)
                print("%s dryrun: patched source written to %s and compiled" % (TAG, dryrun), flush=True)
                os._exit(0)
        return orig(src, fn, mode, *a, **k)
    return comp


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import pp_lib, pp2_sel, pp2_patch, pp2_hook
    err = pp2_sel.preflight(argv=[TAG] + argv) + checks(os.environ, argv)
    if err:
        sys.stderr.write("%s: not started (exit 2):\n  %s\n" % (TAG, "\n  ".join(err)))
        sys.exit(2)
    miss, root = pp2_hook.server_check(argv)
    drv = os.environ.get("PP2Y_COLD") or os.path.join(root, "tools", "sched", "cold_n1.py")
    for nm in ("tools/powerpred2y/y_patch.py", "tools/powerpred2y/cold_y_pp2.py"):
        if not os.path.isfile(os.path.join(root, nm)):
            miss.append("%s in the tree" % nm)
    if not os.path.isfile(drv):
        miss.append("PP2Y_COLD %s" % drv)
    if miss:
        sys.stderr.write("%s: not started (exit 3): the final flow cannot run in %s on %s. Missing:\n  - %s\n"
                         % (TAG, root, sys.platform, "\n  - ".join(miss)))
        sys.exit(3)
    pp2_sel.install(pp_lib)
    builtins.PPL = pp_lib; builtins.PPL2 = pp2_sel
    print("pp2y env: PP2_LIST=%s PP_MODE=%s PP_OUT=%s cold=%s" % (os.environ.get("PP2_LIST", "plain"), os.environ.get("PP_MODE", "log"),
                                                                 os.environ.get("PP_OUT", ""), drv), flush=True)
    pp2_patch.install_exit_guard(TAG)             # before cold_n1.py:677 wraps os._exit
    builtins.compile = make_hook(builtins.compile, os.environ.get("PP2Y_DRYRUN"), pp2_patch.STATE, pp2_sel.log)
    sys.argv = [drv] + argv
    sys.path[0] = os.path.dirname(drv)
    runpy.run_path(drv, run_name="__main__")
    if pp2_patch.STATE["applied"] == 0:
        sys.stderr.write("%s: ERROR the chain patch was never applied (no chain_y.py / chain_d1.py source was compiled)\n" % TAG)
        sys.exit(4)


if __name__ == "__main__":
    main()
