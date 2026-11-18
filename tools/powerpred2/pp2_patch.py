#!/usr/bin/env python3
"""pp2_patch.py (powerpred2, NEW 2026-11-13 round 2; fix round 2 the same day): the one text patch of power_round_i used by both drivers
(cold_i_pp2.py for tools/best/chain_i.py on the Mac, pp2_hook.py for chain_y / chain_d1 / chain_tf on the server). Nothing in the repo is
edited: the patch is applied to the source string inside a builtins.compile hook.

Anchors (each must occur exactly once in the source; checked on chain_i, chain_y, chain_tf, chain_d1 on 13 Nov):
  SK  `            ub = abs(S.w3) * float(cc[allr].sum()) * (1.0 - RHO)`   (the --pw_skip bound, on the flow's own list)
      -> one line added after it: PPL2.note_bound(ub, a.pw_skip)   (records the bound; no change to the test)
  A2  `        sl0 = slack_now(cur) if (a.pw_repair and a.pw_batch <= 0) else None`   (first line AFTER the --pw_skip block)
      -> inserted before it:
           o2_, rank = PPL2.score_list(S, r_, kp, cc, o1, cp, rank, o2_, a)   # SCORE route (PP2_LIST=score); unchanged otherwise
           allr = np.concatenate([o1, o2_])
           PPL.pre(S, r_, o_, wts3, kp, cc, o1, allr, a)                     # features / labels (tools/powerpred/pp_lib.py)
           allr = PPL.select(allr)                                           # TIME route (pp2_sel)
           o1, o2_ = PPL2.split_parts(o1, o2_, allr)                         # o1 / o2_ = their kept nets; allr == [o1, o2_] asserted
  BT  the one top-level batch call `cb, _ = try_batch(cb, allr[...], 0)` (chain_i: allr[s_:s_ + a.pw_batch]; chain_y / tf / d1:
      allr[s_:e_]) -> `cb, _ = PPL2.timed_batch(try_batch, cb, allr[...], 0, o1, info)`: wall seconds of each top-level batch (prom_pw +
      gate + its bisection), split between the timing part and the power part by net count (RULES.md amendment A6, condition 2), and its
      gate count = the growth of info["batches"] (one entry per try_batch call that ran prom_pw + gate, nested bisection calls included;
      chain_i.py:1467, chain_y.py:2467, chain_tf.py:2671, chain_d1.py:2678). The bisection calls inside try_batch are not touched.
  every `= power_round_i(cur)` -> `= PPL.wrap(power_round_i, cur)`        (round record + labels)
Why after the skip block (review F15, skeptic 1 and 5): a round that --pw_skip skips never reaches PPL.pre, so it writes no label file and no
label row; the pp2 wrap records it as skipped. The skip decision is the flow's own (computed on the unchanged list) in every variant.
Why split_parts (review F24, task d): chain_y / chain_tf / chain_d1 use n1_ = len(o1) in the batch loop as the boundary between the timing-
weight part and the power part of allr; after a selection o1 must hold exactly the kept timing-weight nets so that boundary stays right.
install_exit_guard (skeptic round 1: the exit-4 check after runpy never ran, because chain_i.py:1731, chain_d1.py:3638 and cold_n1.py:576
end every completed flow with os._exit(0)): os._exit is wrapped BEFORE the driver (cold_i.py:184 / cold_n1.py:677) wraps it, so their
_exit_dump calls this guard last; an exit code 0 with the patch never applied becomes exit code 4."""
import re
import sys

A1 = "        allr = np.concatenate([o1, o2_])\n"
SK = "            ub = abs(S.w3) * float(cc[allr].sum()) * (1.0 - RHO)\n"
A2 = "        sl0 = slack_now(cur) if (a.pw_repair and a.pw_batch <= 0) else None\n"
SK_INS = "            PPL2.note_bound(ub, a.pw_skip)\n"
A2_INS = ("        o2_, rank = PPL2.score_list(S, r_, kp, cc, o1, cp, rank, o2_, a)\n"
          "        allr = np.concatenate([o1, o2_])\n"
          "        PPL.pre(S, r_, o_, wts3, kp, cc, o1, allr, a)\n"
          "        allr = PPL.select(allr)\n"
          "        o1, o2_ = PPL2.split_parts(o1, o2_, allr)\n")
BT_RE = r"cb, _ = try_batch\(cb, (allr\[[^\]]+\]), 0\)"
BT_SUB = r"cb, _ = PPL2.timed_batch(try_batch, cb, \1, 0, o1, info)"
STATE = {"applied": 0, "files": [], "guard": None}


def patch(s, tag="pp2"):
    for nm, an in (("allr", A1), ("pw_skip bound", SK), ("sl0 (after the skip block)", A2)):
        assert s.count(an) == 1, "%s: anchor %s x%d" % (tag, nm, s.count(an))
    assert s.index(A1) < s.index(SK) < s.index(A2), "%s: anchors out of order (list, skip bound, after-skip line)" % tag
    s = s.replace(SK, SK + SK_INS).replace(A2, A2_INS + A2)
    s, nb = re.subn(BT_RE, BT_SUB, s)
    assert nb == 1, "%s: top-level batch call `cb, _ = try_batch(cb, allr[...], 0)` x%d (need exactly 1)" % (tag, nb)
    assert s.index(A2_INS) < s.index("PPL2.timed_batch("), "%s: batch call before the selection" % tag
    s, n = re.subn(r"= power_round_i\(cur\)", "= PPL.wrap(power_round_i, cur)", s)
    assert n >= 1, "%s: no power_round_i call" % tag
    print("pp2 patch: ok, calls wrapped:", n, "batch loop timed:", nb, flush=True)
    return s


def install_exit_guard(tag, err=None):
    """wrap os._exit: code 0 while no chain source was patched -> exit code 4 with a message (a plain run under a pp2 label)."""
    import os
    real = os._exit

    def guard(code):
        if code == 0 and STATE["applied"] == 0:
            try:
                (err or sys.stderr).write("%s: ERROR the chain patch was never applied (os._exit(0) reached with no patched power_round_i): "
                                          "exit code 4\n" % tag)
                (err or sys.stderr).flush(); sys.stdout.flush()
            finally:
                real(4)
        real(code)

    os._exit = guard
    STATE["guard"] = guard
    return guard


def install_compile_hook(suffixes, dryrun=None):
    """builtins.compile hook: sources whose file name ends in one of suffixes and that define power_round_i get patch(). dryrun=<file>:
    write the patched source there, compile it and end the process."""
    import builtins, os
    _c = builtins.compile

    def comp(src, fn, mode, *a, **k):
        if mode == "exec" and isinstance(src, str) and str(fn).endswith(tuple(suffixes)) and "def power_round_i" in src:
            src = patch(src)
            STATE["applied"] += 1; STATE["files"].append(str(fn))
            if dryrun:
                with open(dryrun, "w") as fh:
                    fh.write(src)
                _c(src, fn, mode, *a, **k)
                print("pp2 dryrun: patched source compiled and written to", dryrun, flush=True)
                os._exit(0)
        return _c(src, fn, mode, *a, **k)

    builtins.compile = comp
    return comp
