#!/usr/bin/env python3
"""y_patch.py (powerpred2y, NEW 2026-11-13): the SCORE-route text patch of power_round_i for the FINAL flow (tools/final/chain_y.py,
smaller designs; tools/day1/chain_d1.py, MemPool-Group in big mode), applied to a COPY of the source (a string, or a file written by
`build`). Nothing in the repo is edited.

The insertions are tools/powerpred2/pp2_patch.patch, imported (not copied):
  SK  after `ub = abs(S.w3) * float(cc[allr].sum()) * (1.0 - RHO)` (the --pw_skip bound on the flow's OWN list, chain_y.py:2451):
      PPL2.note_bound(ub, a.pw_skip)
  A2  before `sl0 = slack_now(cur) if ...` (chain_y.py:2457, the first line AFTER the --pw_skip block):
      o2_, rank = PPL2.score_list(...); allr = np.concatenate([o1, o2_]) (the chain_y.py:2448 anchor line, repeated);
      PPL.pre(...); allr = PPL.select(allr); o1, o2_ = PPL2.split_parts(o1, o2_, allr)
  BT  the batch call -> PPL2.timed_batch(...); every `= power_round_i(cur)` -> `= PPL.wrap(power_round_i, cur)`.
pw_skip on the plain list: the skip test runs before the SCORE list, on the flow's own list (the fix cold_i_pp2 has); --sched_a4 raises
a.pw_skip around the call (tools/sched/cold_n1.py rules_patch), so the A4 decision is also the plain flow's.

chain_y / chain_d1 specifics checked here on the PATCHED text (each line must occur exactly once in power_round_i, in this order):
  the inserted block, then `s_ = 0; n1_ = len(o1); bs_ = a.pw_batch` (the o1 / o2 boundary of the batch loop) with no assignment to
  o1, o2_ or allr between them (n1_ is the length of o1 and allr = [o1, o2_]); `if not DLG("pw_batch"):` (deadline guard), the
  --t_pw_grow line, the timed batch call, `DLE("pw_batch")`, the --t_pw_stop / --t_pw_rej / --t_pw_stop2 lines, then the --pw2_nets
  line `o2b = rank[a.pw_nets:a.pw_nets + a.pw2_nets]` (reads the SCORE order); no unwrapped `power_round_i(cur)` call; a reference
  `"power": power_round_i` (cold_n1 --sched_fill / --sched_value) is refused (its rounds would bypass PPL.wrap).
Use:
  python -I -B y_patch.py build <chain source> <out.py>     # patch a copy, write it, py_compile it (cfile <out.py>c), print the checks"""
import hashlib, os, py_compile, re, sys

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "tools", "powerpred2") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools", "powerpred2"))
import pp2_patch as P2

SUFFIXES = ("chain_y.py", "chain_d1.py")
BOUND = "            s_ = 0; n1_ = len(o1); bs_ = a.pw_batch\n"
ORDERED = (
    ("skip return", "            if ub < a.pw_skip:\n"),
    ("inserted SCORE list", "        o2_, rank = PPL2.score_list(S, r_, kp, cc, o1, cp, rank, o2_, a)\n"),
    ("inserted split", "        o1, o2_ = PPL2.split_parts(o1, o2_, allr)\n"),
    ("n1_ boundary", BOUND),
    ("DLG pw_batch", '                if not DLG("pw_batch"):                    # merge6 (b): deadline -> no further power batch\n'),
    ("t_pw_grow", "                e_ = min(s_ + bs_, n1_) if (s_ < n1_ and a.t_pw_grow) else s_ + bs_   # speed10: --t_pw_grow: a list's batches double while they are kept\n"),
    ("timed batch", "                cb, _ = PPL2.timed_batch(try_batch, cb, allr[s_:e_], 0, o1, info)\n"),
    ("DLE pw_batch", '                DLE("pw_batch")\n'),
    ("t_pw_stop", "                if a.t_pw_stop > 0 and s_ < n1_ and g_ < a.t_pw_stop:"),
    ("t_pw_rej", '                elif a.t_pw_rej and s_ < n1_ and (not info["batches"] or c0_ is cb):'),
    ("t_pw_stop2", "                elif a.t_pw_stop2 > 0 and s_ >= n1_ + a.pw_batch and g_ < a.t_pw_stop2:"),
    ("pw2 from rank", "            o2b = rank[a.pw_nets:a.pw_nets + a.pw2_nets]\n"),
)
ASSIGN_RE = re.compile(r"^\s*(?:o1|o2_|allr)\s*(?:,[^=\n]*)?=(?!=)", re.M)


def body(s):
    """text of power_round_i (from its def to the next def at the same indent)"""
    i = s.index("    def power_round_i(cur):\n")
    j = s.find("\n    def ", i + 10)
    return s[i:] if j < 0 else s[i:j + 1]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def check_y(s, tag="y"):
    """chain_y / chain_d1 checks on a patched source; -> report dict (asserts on any failure)"""
    b = body(s)
    pos = []
    for nm, ln in ORDERED:
        assert b.count(ln) == 1, "%s: power_round_i: line '%s' x%d (need 1)" % (tag, nm, b.count(ln))
        pos.append(b.index(ln))
    assert pos == sorted(pos), "%s: power_round_i lines out of order: %s" % (tag, [n for _, n in sorted(zip(pos, [x[0] for x in ORDERED]))])
    i_split = b.index(ORDERED[2][1]) + len(ORDERED[2][1]); i_bound = b.index(BOUND)
    mid = ASSIGN_RE.findall(b[i_split:i_bound])
    assert not mid, "%s: o1 / o2_ / allr reassigned between the split and n1_ = len(o1): %s" % (tag, mid)
    i_ret = b.index(ORDERED[0][1]); i_ins = b.index(ORDERED[1][1])
    assert "PPL2.score_list" not in b[:i_ins] and b[i_ret:i_ins].count("return False, {}, r_,") == 1, \
        "%s: the --pw_skip test is not on the plain list before the SCORE list" % tag
    bare = re.findall(r"(?<!PPL\.wrap\()(?<!def )power_round_i\(cur\)", s)
    assert not bare, "%s: %d unwrapped power_round_i(cur) call(s)" % (tag, len(bare))
    assert '"power": power_round_i' not in s, "%s: a scheduler path passes power_round_i itself (--sched_fill / --sched_value)" % tag
    return {"n_wrapped": s.count("PPL.wrap(power_round_i, cur)"), "lines_checked": len(ORDERED), "order_ok": True}


def patch_y(src, tag="y"):
    """-> (patched source, report). pp2_patch.patch (the insertions) + check_y (chain_y specifics)."""
    assert src.count("    def power_round_i(cur):\n") == 1, "%s: def power_round_i x%d" % (tag, src.count("    def power_round_i(cur):\n"))
    assert body(src).count(BOUND) == 1, "%s: the n1_ boundary line is not in power_round_i" % tag
    assert body(src).count(P2.A1) == 1, "%s: anchor 'allr = np.concatenate([o1, o2_])' not in power_round_i" % tag
    s = P2.patch(src, tag)
    rep = check_y(s, tag)
    rep.update(md5_in=md5(src), md5_out=md5(s), anchor_line=src[:src.index(P2.A1)].count("\n") + 1)
    return s, rep


def build(src_path, out_path):
    """patch a copy of src_path into out_path and py_compile it (cfile = out_path + 'c'). -> report"""
    with open(src_path) as fh:
        src = fh.read()
    s, rep = patch_y(src, os.path.basename(src_path))
    with open(out_path, "w") as fh:
        fh.write(s)
    py_compile.compile(out_path, cfile=out_path + "c", doraise=True)
    rep.update(src=src_path, out=out_path, compiled=True)
    return rep


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        r = build(sys.argv[2], sys.argv[3])
        print("y_patch build ok", " ".join("%s=%s" % kv for kv in sorted(r.items())), flush=True)
    else:
        sys.exit(__doc__)
