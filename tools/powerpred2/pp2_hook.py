#!/usr/bin/env python3
"""pp2_hook.py (powerpred2; round 2, 13 Nov 2026): the pp2 selection and SCORE list for the FINAL flow (tools/sched/cold_n1.py running
tools/final/chain_y.py, --sched_src y; also chain_d1 / chain_tf sources). It needs no shim of tools/powerpred: it installs pp2_sel into
tools/powerpred/pp_lib.py in memory, hooks builtins.compile with pp2_patch.patch for sources ending in chain_y.py, chain_d1.py or chain_tf.py,
and runs <sched_root>/tools/sched/cold_n1.py with the same arguments.

Documented command (from the root of a code tree that holds this repo's tools/, including tools/powerpred and tools/powerpred2):
  <python> -I -B -u tools/h100big/hard_exit.py tools/powerpred2/pp2_hook.py --sched_src y --sched_root <tree> <n1y arm flags incl. --design D>
Before anything runs it checks, and stops at once (exit 2: env / model / layer list; exit 3: this machine or tree cannot run the final flow)
with a message saying what is missing:
  * pp2_sel.preflight(): PP_MODE, PP_FRAC (required in potential mode), PP_TARGET, PP2_KEEP_O1, PP2_LIST, PP_MODEL (named
    model_wo_<--design>.npz, exists, 43 inputs), PP_LAYER_DIR (if given: must be 0,1,0,1,0,1,0,1,0,1); unknown PP_* / PP2_* names and names
    the chosen mode ignores are refused;
  * <tree>/tools/sched/cold_n1.py and the chain source exist;
  * the native libraries the n1y arm loads exist for this platform (list FINAL_LIBS; none of them is built in this repo on 13 Nov:
    findings_round1.json scout_mac.blockers[1]);
  * CUDA, when --dgr > 0 (DTLA).
In this repository on the Mac the third check fails, so the hook stops with exit 3 and prints the server launch it needs. Exit 4 if the
chain patch was never applied (a silent plain run under a pp2 label, review F14): checked after runpy and in an os._exit wrapper installed
before tools/sched/cold_n1.py:677 wraps os._exit (its completed flows and cold_n1.py:576 end with os._exit(0)).
Not covered (server path only): power rounds that tools/sched/cold_n1.py calls without the wrap (SCHED.fill, value_loop pass power_round_i
itself, cold_n1.py:412,425) write no round record; pp2_sel records each such round as kind=unwrapped and pp2_summary then leaves TIME
not decided."""
import builtins, os, runpy, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUFFIXES = ("chain_y.py", "chain_d1.py", "chain_tf.py")
# native libraries of the final arm n1y (cfg_d: --par7 --timer5 --n_*; cfg_x: --t_*; --load8; chain_y write_cpp), relative to tools/
FINAL_LIBS = ("best5/libn_norm", "best5/libn_ofc", "best5/libn_seg", "best5/libn_b2", "best5/libn_sel", "best5/libn_rs", "best5/libn_rr",
              "best10/libt_csta", "best10/libt_rr", "best10/libt_fpk", "best11/libl8_kern", "best11/libi8_kern", "timer5/libcsta5",
              "par7/libp7_pf", "par7/libp7_dp", "c1hfast/libnorm_write", "engine/libcsta_inc")


def _opt(argv, key, default=None):
    return argv[argv.index(key) + 1] if key in argv and argv.index(key) + 1 < len(argv) else default


def server_check(argv, root=None, plat=None, cuda=None):
    """-> list of missing items for the final flow (empty = can start)."""
    plat = sys.platform if plat is None else plat
    root = os.path.abspath(root or _opt(argv, "--sched_root") or os.environ.get("SCHED_ROOT") or REPO)
    ext = ".dylib" if plat == "darwin" else ".so"
    miss = []
    drv = os.path.join(root, "tools", "sched", "cold_n1.py")
    if not os.path.isfile(drv):
        miss.append("driver %s" % drv)
    src = {"y": "tools/final/chain_y.py", "d1": "tools/day1/chain_d1.py"}.get(_opt(argv, "--sched_src", "d1"))
    if src is None:
        miss.append("--sched_src must be y or d1")
    elif not os.path.isfile(os.path.join(root, src)):
        miss.append("chain source %s" % os.path.join(root, src))
    for nm in ("tools/powerpred/pp_lib.py", "tools/powerpred2/pp2_sel.py", "tools/powerpred2/pp2_lib.py", "tools/powerpred2/pp2_patch.py"):
        if not os.path.isfile(os.path.join(root, nm)):
            miss.append("%s in the tree" % nm)
    nolib = [b + ext for b in FINAL_LIBS if not os.path.isfile(os.path.join(root, "tools", b + ext))]
    if nolib:
        miss.append("native builds (%s) of %d libraries: %s" % (ext, len(nolib), ", ".join(nolib)))
    try:
        dg = float(_opt(argv, "--dgr", "0"))
    except ValueError:
        dg = 0.0
    if dg > 0 and not nolib:
        if cuda is None:
            try:
                import torch
                cuda = bool(torch.cuda.is_available())
            except Exception:
                cuda = False
        if not cuda:
            miss.append("CUDA (torch.cuda) for --dgr %g (DTLA)" % dg)
    return miss, root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, os.path.join(REPO, "tools", "powerpred"))
    import pp_lib, pp2_sel, pp2_patch
    err = pp2_sel.preflight(argv=["pp2_hook"] + argv)
    if err:
        sys.stderr.write("pp2_hook: not started (exit 2):\n  " + "\n  ".join(err) + "\n")
        sys.exit(2)
    miss, root = server_check(argv)
    if miss:
        sys.stderr.write(
            "pp2_hook: not started (exit 3): the final flow cannot run in %s on %s. Missing:\n  - %s\n"
            "The server launch needs: a code tree with these native builds and CUDA (the GPU server), this repo's tools/powerpred and "
            "tools/powerpred2 copied into it, and from that tree:\n"
            "  <python> -I -B -u tools/h100big/hard_exit.py tools/powerpred2/pp2_hook.py --sched_src y --sched_root <tree> <n1y arm flags, "
            "with --design>\n  with PP_MODE / PP_TARGET / PP2_LIST / PP_OUT set as in experiments/powerpred2_2026-11-13/RULES.md.\n"
            % (root, sys.platform, "\n  - ".join(miss)))
        sys.exit(3)
    pp2_sel.install(pp_lib)
    builtins.PPL = pp_lib; builtins.PPL2 = pp2_sel
    pp2_patch.install_exit_guard("pp2_hook")
    pp2_patch.install_compile_hook(SUFFIXES, os.environ.get("PP2_DRYRUN"))
    drv = os.path.join(root, "tools", "sched", "cold_n1.py")
    sys.argv = [drv] + argv
    sys.path[0] = os.path.dirname(drv)
    runpy.run_path(drv, run_name="__main__")
    if pp2_patch.STATE["applied"] == 0:
        sys.stderr.write("pp2_hook: ERROR the chain patch was never applied (no %s source with power_round_i was compiled)\n" % "/".join(SUFFIXES))
        sys.exit(4)


if __name__ == "__main__":
    main()
