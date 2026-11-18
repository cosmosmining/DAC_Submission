#!/usr/bin/env python3
"""chain_x.py (coordinator, 7 Nov 2026, NEW): tools/dgr2d/chain_s.py + --load8 (tools/best11/chain_u.py) by git merge-file, base tools/best6/chain_p.py; one conflict, this docstring. Off = chain_s.
chain_s: chain_r + --d2 (tools/dgr2d/mk_s.py). chain_p.py: tools/best5/chain_n.py + `git merge-file` of tools/big8/chain_b8.py (base chain_m, 2 conflicts, both additions, resolved by
chain_u.py: tools/best6/chain_p.py + --load8 1 (an internal note), off by default (off = chain_p's path):
  --load8 1          big mode, cold (through tools/best11/cold_u.py): tools/best11/load8_lib.py (pieces and tests there).
  --load8_stop 1     test: exit after the preload, the first timer and the activity, with the timers (also with --load8 0).
  env LOAD8_PROF=F   second-by-second profile (tools/best11/prof8.py); measurement only.
chain_p.py: tools/best5/chain_n.py + `git merge-file` of tools/big8/chain_b8.py (base chain_m, 2 conflicts, both additions, resolved by
hand: timer5's library line + big8's bound kernel; big8's DP pair, else par7's, else the serial DPs) + (an internal note; off by default = chain_n's route bytes):
  --dl_chunks K   only with --deadline_s (small mode): after the first pass and BEFORE the overflow stage, up to K checked chunks of --dl_chunk_nets nets (default 4000) in the
                  order timing + power (tools/gnn1x/infer1x.py arm Ht4k: score = tau / max tau + cc / max cc over the universe tau > 0 or the top --pw_nets by cc, computed once at
                  the first-pass state; ties as the harness: route-net order; weights refreshed after a kept chunk). A chunk = one promote (FLUTE allowed, power price, timing weight
                  scale --dl_chunk_tw, default 2 = the harness's, an internal note section 10) + one exact re-time; kept only if the estimate falls; a rejected chunk is undone, no bisection.
                  A chunk starts only if now + estimate <= X (estimate: the previous chunk, first one kind "dl_chunk" of --deadline_est, else 0). A chunk that does not fit does NOT
                  expire the deadline: the later stages keep their own guards, in today's order. If a chunk was kept and the overflow stage runs, the required-time pass runs first (it reads net_slack).
  --big8 1        tools/big8's cold big-mode pieces (an internal note).
  env P11_WAIT_ROUTE=F  big mode: every route read waits for file F (big5's early timer reads the route inside the preload, tools/best6/cold_p.py --wait_route covers only main's read).
"""
"""chain_n.py: tools/best4/chain_m.py + `git merge-file` of tools/par7/chain_p7.py (base chain_k) and tools/timer5/chain_t5.py
(base chain_j) + the switches of an internal note, all off by default (off = chain_m's route bytes).
"""
"""chain_m.py: `git merge-file tools/best3/chain_k.py tools/best2/chain_j.py tools/big5/chain_big.py`
(0 conflicts; = the hosts' tools/big5/chain_kb.py, md5 e4467395) + two switches, both off by default (off = chain_kb's route):
  --route_bin 1   --route is the router's binary route R5B1 (InstantGR_r5b -bin), read by tools/router5/r5bin.py (small mode only).
  --deadline_s X  [--deadline_est J]: before the overflow stage, each crit round, the power round, each power batch, the reshape
                  stage, each reshape pass, each post-crit round and the ovf round: if (wall since process start, or env
                  DL_T0_EPOCH) + (seconds of the previous unit of that kind; first one: J's value, else 0) > X, every later unit is
                  skipped and the chain writes its kept (verified) state. json walls.merge6 records the units that ran.
  timer5 (tools/timer5/chain_t5.py) is NOT merged: an internal note reports no identity-test result (R1/R2).
an internal note. chain_k's header follows.
"""
"""chain_p7.py: tools/best3/chain_k.py + --par7 (an internal note), off by default (= chain_k):
  --par7 1  (a) the promote rounds' phase 2 (crit, power, post-crit, overflow-ranked rounds) on PF_THREADS threads in a level
            schedule (tools/par7/p7_pf.cc), (b) the two first-pass DPs at the same time on --par7_pt threads each (tools/par7/p7_dp.py);
            both give the sequential loop's results (same route bytes).
  Every log entry carries p7_t = [wall, process CPU, promote calls, re-times] (measurement); json "par7" has the per-call tables.
chain_k's header follows.
"""
"""chain_k.py: tools/best2/chain_j.py + the three speed regions of 6 Nov, merged with git merge-file
against chain_j.py (an internal note), each behind its own flag, all OFF by default (= chain_j's route bytes):
  --pre5 1   tools/pre5 (an internal note): cold parallel preload, in-process switching activity, prior without torch.
             merge5: switched OFF automatically in big mode (--big on / auto on MemPool group / cluster): pre5 was not made for it.
  --mid5 1   tools/mid5 (an internal note): route read to the end of the overflow stage.
  --late5 1  tools/late5 (an internal note): critical rounds to the written route.
Speed flags with their settings: tools/best3/jobs/SPEED.flags. chain_j's own header follows.
"""
"""chain_j.py: copy of tools/best/chain_i.py (recipe R6c) with the pieces that passed on 6 Nov,
each behind its own flag, all OFF by default (every new flag off = chain_i.py's decisions and route bytes). an internal note.
  a. --engine micro --stages ...: the verified small-batch engine of tools/engine2/micro2.py (merged with git merge-file, base
     chain_h.py) in place of chain_i's crit rounds and power round; afterwards chain_i's after-stages (--post_crit, --ovfround,
     and --rs) run on its state and the b<label> route is written. --micro_big rounds|micro (default rounds): in big mode
     the engine falls back to chain_i's rounds (a design-size switch set by the chain).
  b. --post_crit_big K: big mode only, --post_crit K (tools/wave3/big/chain_gb2.py arm p2yc: 3 gated crit rounds after the
     big power stage).
  c. --rs on ...: the re-route stage of tools/reshape/chain_r.py (reshape_stage verbatim), run after the final power round
     (--rs_at power: before --post_crit / --ovfround; --rs_at end: after them).
  d. speed switches with identical routes (tools/speed4/chain_s4.py items): --s4_ofc 1 (cached official overflow cost),
     --s4_esc 1 (C++ escape / re-route cost).
chain_i.py's docstring follows.
chain_i.py: copy of tools/fast3x/chain_h.py with the pieces that passed on 5 Nov, each behind
its own flag, all OFF by default (every new flag off = chain_h.py's decisions and route bytes). an internal note.
  b. overflow stage of tools/wave3/ovf2/stage_w3.py (chain_w3.py): --ovf_impl, --ovf_timer, --ovf_rr_passes, --ovf_order,
     --ovf_esc_pre, --ovf_big_nets, --ovf_hmax_big, --ovf_big_runs, --ovf_big_passes, --ovf_esc_check (REQUIRED with
     --ovf_order re), --ovf_skip. The incremental timer of chain_h and of the stage are ONE library: tools/best/libi_kern
     (= h_csta_kern.cc + w3_inc_backward), loaded whenever --h_inc/--h_crit or --ovf_timer inc is on.
  c. --post_crit K: up to K score-checked crit rounds after the final power round (chain_tw.py / chain_t3.py), with
     --tw_scale (exists) for the timing weight.
  e. power round: --pw_batch N (+ --pw_bisect_depth D, --pw_bisect_min M): the round's net list in batches of N, each
     kept only if the estimate falls, a rejected batch split in halves (tools/engine/micro_chain.py arm C6kP500);
     --pw2_nets N: a second power round on the next N power nets (tools/wave3/big/chain_gb.py p2r/p2x/p2y);
     --pw_skip T: skip the round when |w3| * sum kpn*ctot * (1 - Cmin/Cmax) < T; --pw_repair: on a rejected (unbatched)
     round undo the power-list nets whose slack fell below min(before, 0) and re-test once (tools/srvw/power/chain_p.py).
  d. --ovfround N (+ --ovfround_slack S, --ovfround_open 1, --ovfround_skip T): the overflow-ranked kernel round after
     the power round (tools/wave3/teacher/chain_t3.py setting B); --ovfround_open 1 ranks on OPEN cells only (capacity
     > 1e-3; the bsg fault), --ovfround_skip T skips it when w4 x (own overflow cost of the ranked nets) < T.
  --auto_cfg: none of these is a per-case flag; any design-dependent switch is set by the chain itself (logged in json).
Original docstring follows.
chain_h.py: copy of tools/fast3x/chain_g.py with the third speed round as switches, all OFF
by default (no switch = chain_g.py bytes). an internal note has the identity tests and the seconds.
  --h_read 1      item 1: route reader tools/fast3x/h_route_read.py (hand-written parser, threads; falls back to route_cpp)
  --h_write N     item 2: normalise + format + write in N threads (tools/fast3x/h_write.py = tools/engine/io_write.cc)
  --h_dem 1       items 3a + 5: the router route's demand map of the escape test is kept; each candidate's map is that map
                  minus the router rows plus the candidate rows of the nets the single pass changed (integer-valued sums, exact)
  --h_asm 1       item 3b: assemble() with the per-net loop in C++ (tools/fast3x/h_asm.py)
  --h_inc 1       items 3c + 4a: incremental exact timer (tools/engine/csta_inc_kern.cc, built as tools/fast3x/h_libcsta):
                  second candidate of the check and every crit / power re-time (tools/engine/chain_inc.py's retime_inc)
  --h_crit 1      item 4b: round weights from csta_hcrit (threaded, without the unused roundings; same net/ok/poff)
  --h_graph 1     item 6: GNN graph without the np.unique(axis=1) calls (tools/fast3x/h_graph.py)
  --h_all 1       all of the above (h_write = 8)
  --h_pre 1       item 7 (not in --h_all): cyclic GC paused during the preload, its objects frozen afterwards (gc.freeze);
                  the overflow stage's pin table from the C++ .net parse the preload does anyway (tools/fast3x/h_netpins.py)
  --h_check       test mode: each switched item also runs the original path and compares (arrays / bytes); counts in json
Original docstring follows.
chain_g.py: tools/fast3x/chain_f.py + the overflow stage of tools/ovf3/chain_ovf.py
(overflow_stage() and its flags moved unchanged; insertion points of an internal note section 5). Built by
tools/fast3x/chain_g_merge.py. Every new behaviour is behind a flag that is OFF by default (--ovf off = chain_f.py bytes).
Recipe g1 = c4 flags + --of_rule official --promote_impl full --ovf auto. --of_rule keeps chain_f's meaning (floor|official).
Original docstring follows.
chain_f.py: copy of tools/nearmiss/chain_k.py, same decisions, same route bytes, less waste:
  1. the power round is not reverted after the final write (only when another snapshot follows);
  2. linear-time via rule in timing_promote.apply / LayerAssigner.commit_net (tools/fast3x/fast_apply.py, patched at run time);
  3. check on the GNN prior: identical layer choices -> one evaluation, pick noprior; the pick is restored from a kept copy
     (demand + attached state) instead of a third evaluation;
  4. kernel calls with pointers computed once per round (tools/fast3x/promote_fast.py);
  5. pin-escape pins and its demand model are built only when the blocked-share rule turns the escape on (the share test
     uses the state's own demand array, which the check zeroes anyway);
  6. timer array: changed nets normalised by the C++ normaliser (tools/fast3x/fast_norm.py), only nets changed since the last call.
  Hooks (off by default): --promote_impl py|full (full: tools/fast3x/promote_full.py), --min_gain_frac F.
  Test mode --check: demand array mirrored with the original apply/commit_net (np.array_equal after every round) and the
  timer array compared with fastmode4's Python one at every call.
  Switches to turn one item off (for A/B): --no_item N (repeatable).
Original docstring follows.
chain_k.py: copy of chain_two.py with K extra gated C++ crit rounds, snapshots (a = rounds only, b = + exact power round, reverted afterwards),
and --gate L: pass-level estimator gate between lam 0 and lam L (held-out GNN prior, gnnfix rule). Original docstring follows.
chain_two.py: chain_one + two switches, ONE process, ONE fixed recipe for all cases.
  --escape auto|on|off (default auto): in-memory pin escape (top --n_hot blocked cells, --esc_passes) BEFORE the single pass.
        auto = ON when the blocked share of the router output's overflow cost exceeds --esc_thresh (default 0.5).
  --power_round: after the gated crit round, a second in-memory round with the exact power cost (tools/tilefix/promote_pw: crit nets +
        top-power nets), gated by the same exact 4-term C++ re-time score check (reverted if the score does not improve).
Original docstrings follow.
chain_one_bsg.py: copy of tools/c1hfast/chain_one.py with the pin escape inserted in memory:
route read once -> escape (escape3d.escape on the in-memory route: search only, la/pins preloaded in the route-window thread, changes spliced
into the column arrays, no file re-read or write) -> C++ timer -> build2 -> C++ DP -> C++ round -> one C++ write. Extra flags: --n_hot (1290), --esc_passes (2), --no_escape.
Original docstring follows.
chain_one.py: ONE process, route -> C++ timer (csta) -> build2 -> C++ dp_split DP -> in-memory hand-off ->
C++ promote round (exact C++ re-time gate) -> write. No Python timer, no report file, no route re-parse, no design reload.
Thread at t=0 (during the route window): imports, tech, csta cache + kernel, StaDesign + cap + TimingLA + pin tables + activity (route independent).
  chain_one.py --design D --suite S --route R --lam L --out O [--json J] [--route_wall S] [--write py|cpp] [--sp_out F]
"""
import argparse, json, os, sys, threading, time, collections, types
T0 = time.perf_counter()
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "tools" / "c1hfast"
if os.environ.get("LOAD8_PROF"):                           # load8: profile thread (measurement only)
    sys.path.append(str(ROOT / "tools" / "best11")); import prof8; prof8.start(T0, os.environ["LOAD8_PROF"])
os.environ["FAST_STA_V3"] = "1"; os.environ.pop("FAST_STA_V2", None)
for p in (ROOT / "tools" / "c1hfast", ROOT / "tools" / "bsgfast", ROOT / "tools" / "tilefix", ROOT / "tools" / "speedcut", ROOT / "tools" / "gapla", ROOT / "tools" / "fast_io" / "route_cpp",
          ROOT / "tools" / "fast_la", ROOT, ROOT / "tools" / "fast_path", ROOT / "tools" / "fast_path"):
    sys.path.insert(0, str(p))
F3X = ROOT / "tools" / "fast3x"
BEST = ROOT / "tools" / "best"
sys.path.append(str(BEST))                                 # chain_i: the chain_h helper copies + their libraries (before F3X)
sys.path.append(str(F3X))                                  # appended: never shadows the modules above
sys.path.append(str(ROOT / "tools" / "wave3" / "ovf2"))    # reshape: ovf2_lib (appended)
sys.path.append(str(ROOT / "tools" / "ovf3"))              # chain_g: ovf_lib (appended, never shadows)
sys.path.append(str(ROOT / "tools" / "wave3" / "ovf2"))    # chain_w3: ovf2_lib (appended)
E2 = ROOT / "tools" / "engine2"
sys.path.append(str(E2))                                  # chain_j a: engine2 (ofc.py; appended)
RSH = ROOT / "tools" / "reshape"
sys.path.append(str(RSH))                                 # chain_j c: reshape (rs_lib; appended)
import numpy as np
PRE = {}; G = {}
H = {"read": 0, "write": 0, "dem": 0, "asm": 0, "inc": 0, "crit": 0, "graph": 0, "pre": 0, "check": 0}   # chain_h switches (set in main)
HCHK = {}                                                  # --h_check results: item -> [n_compared, n_bad]
HT = {}                                                    # chain_h timers


TMK = []                                                   # speed10: stage marks of the first pass (name, perf_counter)


def TM(name):
    TMK.append((name, time.perf_counter()))


def hchk(item, ok, info=None):
    c = HCHK.setdefault(item, [0, 0]); c[0] += 1; c[1] += (not ok)
    print(f"[h_check] {item}: equal={bool(ok)}" + (f" {info}" if info else ""), flush=True)
def _argv(k, d=None):
    for _i, _v in enumerate(sys.argv):
        if _v == k and _i + 1 < len(sys.argv):
            return sys.argv[_i + 1]
    return d


# chain_i big mode (--big on|off|auto, default off; auto = on when the design's .net file exceeds 1 GB, i.e. the MemPool
# group / cluster designs): the large-case items of tools/wave3/big/chain_gb.py (B1-B6 of chain_b + L1-L3; BIG below, each
# off with --no_big N), the chain_h switches and the W3 stage flags OFF (chain_gb's validated base), the power round unbatched.
_BM = _argv("--big", "off")
if _BM == "auto":
    _nf0 = ROOT / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks" / str(_argv("--suite", "visible")) / str(_argv("--design", "")) / (str(_argv("--design", "")) + ".net")
    try:
        _BM = "on" if os.path.getsize(_nf0) > 1e9 else "off"
    except OSError:
        _BM = "off"
BIGM = _BM == "on"
PRE5 = any(v == "--pre5" and i + 1 < len(sys.argv) and sys.argv[i + 1] == "1" for i, v in enumerate(sys.argv)) and not BIGM   # pre5 (merge5: not in big mode)
L8S = _argv("--load8", "0") == "1" and PRE5                # load8 on pre5's path (bsg and the small cases): parser and StaDesign (tools/best11/load8_lib.py)
if L8S:
    sys.path.append(str(ROOT / "tools" / "best11")); import load8_lib as L8L
    L8L.early_small(T0)
if PRE5:                                                   # pre5: lazy torch/dtla, C++ DEF/.net parse + Liberty threads from here
    sys.path.append(str(ROOT / "tools" / "pre5"))
    if _argv("--timer5", "0") == "1":                       # speed9: pre5's kernel_bind hard-codes tools/best/libi_kern: the one line changed at load (--timer5 1 was a no-op with --pre5 1)
        import types as _ty
        _p5f = ROOT / "tools" / "pre5" / "pre5_load.py"; _p5s = _p5f.read_text()
        _p5o = '_hl = str(BEST / ("libi_kern." + ("dylib" if sys.platform == "darwin" else "so")))'
        assert _p5s.count(_p5o) == 1
        _p5s = _p5s.replace(_p5o, ('_hl = str(ROOT / "tools" / "best10" / ("libt_csta." + ("dylib" if sys.platform == "darwin" else "so")))' if _argv("--t_csta", "0") == "1" else   # speed10
                                   '_hl = str(ROOT / "tools" / "timer5" / ("libcsta5." + ("dylib" if sys.platform == "darwin" else "so")))'))
        os.environ.setdefault("CSTA5_SPIN_US", "0")         # tools/timer5/csta5.py's default (rule R5 there)
        P5L = _ty.ModuleType("pre5_load"); P5L.__file__ = str(_p5f); sys.modules["pre5_load"] = P5L
        exec(compile(_p5s, str(_p5f), "exec"), P5L.__dict__)
    else:
        import pre5_load as P5L
    P5L.early(ROOT, sys.argv, T0)
    if L8S: L8L.install_small(P5L)
BIG = {1, 2, 3, 4, 5, 6, 7} if BIGM else set()
for _i, _v in enumerate(sys.argv):
    if _v == "--no_big" and _i + 1 < len(sys.argv):
        BIG.discard(int(sys.argv[_i + 1]))
FLD = None
NPK_DICT = [True]
L8 = {"on": _argv("--load8", "0") == "1" and BIGM}         # load8: tools/best11/load8_lib.py (an internal note)
if L8["on"]:
    sys.path.append(str(ROOT / "tools" / "best11")); import load8_lib as L8L
    L8L.early(ROOT, sys.argv, T0, bool((globals().get("COLD") or {}).get("bind_mem")))
B8 = {"on": _argv("--big8", "0") == "1"}                  # big8: tools/big8/big8_lib.py (an internal note)
if B8["on"]:
    sys.path.append(str(ROOT / "tools" / "big8")); import big8_lib as B8L
    B8L.early(sys.argv, BIGM and bool((globals().get("COLD") or {}).get("bind_mem")), T0)
B5 = {"on": _argv("--big5", "0") == "1"}                  # big5: tools/big5/big5_lib.py (an internal note)
if B5["on"]:
    sys.path.append(str(ROOT / "tools" / "big5")); import big5_lib as B5L; B5L.early(ROOT, sys.argv, T0, BIGM)
# merge6 (b) --deadline_s X: before each stage, round and batch after the single pass, the chain compares the wall since process
# start (or since env DL_T0_EPOCH, the harness t0) with X; if now + estimate > X it goes straight to the write (every later unit
# skipped). Estimate = seconds of the previous unit of the same kind in this run, else --deadline_est (json {kind: s}), else 0.
# Off (no --deadline_s): DLG() is True and DLE() does nothing, so no decision changes.
DL = {"on": False, "expired": False, "X": None, "last": {}, "est": {}, "cur": {}, "ran": [], "skipped": None, "ref": None}
DL_ALIAS = {"post_crit": "crit_round"}


def _dl_ref():
    e = os.environ.get("DL_T0_EPOCH")
    if e:
        return "harness_t0", float(e)
    try:                                                   # Linux: process start from /proc (clock ticks since boot)
        st = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        start_boot = float(st[19]) / os.sysconf("SC_CLK_TCK")
        return "proc_start", time.time() - (time.clock_gettime(time.CLOCK_BOOTTIME) - start_boot)
    except Exception:
        return "module_T0", time.time() - (time.perf_counter() - T0)


def _dl_now():
    return time.time() - DL["ref"][1]


def DLG(kind):
    if not DL["on"]:
        return True
    if DL["expired"]:
        return False
    now = _dl_now()
    est = DL["last"].get(kind, DL["last"].get(DL_ALIAS.get(kind, ""), DL["est"].get(kind, 0.0)))
    if now + est > DL["X"]:
        DL["expired"] = True; DL["skipped"] = [kind, round(now, 3), round(float(est), 3)]
        return False
    DL["cur"][kind] = now
    return True


def DLE(kind):
    if not DL["on"]:
        return
    t_ = DL["cur"].pop(kind, None)
    if t_ is not None:
        d_ = _dl_now() - t_; DL["last"][kind] = d_; DL["ran"].append([kind, round(t_, 3), round(d_, 3)])


class _NpinsKey:
    """chain_gb: == {_key(net): number of pins with >= 1 access point} of chain_g's preload (dict built once)."""
    def __init__(self, lp, cnt, names=None, canon=None):
        self.lp, self.cnt, self.d = lp, cnt, None
        if names is not None:
            c = cnt.tolist()
            self.d = {nm: c[i] for i, nm in enumerate(names) if canon is None or canon[i]}
    def get(self, k, d=None):
        if self.d is not None:
            return self.d.get(k, d)
        i = self.lp._idx(k)
        return int(self.cnt[i]) if i >= 0 else d
    def __len__(self): return len(self.cnt)


def _pin_maps_arrays(lp, X, Y):
    """chain_gb L1: ovf_lib.pin_count_map(pins, X, Y) and {key: len(pins[key])} from the loader's arrays."""
    t = time.perf_counter()
    nd = lp.nd
    nn = int(nd.nn)
    pin_off = np.asarray(nd.pin_off, np.int64); ap_off = np.asarray(nd.ap_off, np.int64); aps = np.asarray(nd.aps)
    has = (ap_off[1:] - ap_off[:-1]) > 0
    cs = np.zeros(len(has) + 1, np.int64); cs[1:] = np.cumsum(has)
    cnt = cs[pin_off[1:nn + 1]] - cs[pin_off[:nn]]
    names = nd.names_raw.replace("\\", "").split("\n")[:-1]
    canon = np.ones(nn, bool)
    ndup = 0
    if len(set(names)) != len(names):
        dset = {nm for nm, c in collections.Counter(names).items() if c > 1}
        for i, nm in enumerate(names):
            if nm in dset:
                ndup += 1
                canon[i] = (i == lp._idx(nm))
    npk = _NpinsKey(lp, cnt, names, canon if ndup else None) if NPK_DICT[0] else _NpinsKey(lp, cnt)
    del names
    assert int(pin_off[nn]) == len(has) and int(ap_off[-1]) == len(aps), (int(pin_off[nn]), len(has), int(ap_off[-1]), len(aps))
    pin_net = np.repeat(np.arange(nn, dtype=np.int64), pin_off[1:nn + 1] - pin_off[:nn])
    ap_pin = np.repeat(np.arange(len(has), dtype=np.int64), ap_off[1:] - ap_off[:-1])
    ap_net = pin_net[ap_pin]
    x = aps[:, 1].astype(np.int64); y = aps[:, 2].astype(np.int64)
    ok = (x >= 0) & (x < X) & (y >= 0) & (y < Y) & canon[ap_net]
    code = np.unique(ap_net[ok] * (int(X) * int(Y)) + x[ok] * int(Y) + y[ok])
    pc = np.bincount(code % (int(X) * int(Y)), minlength=int(X) * int(Y)).astype(np.float64).reshape(int(X), int(Y))
    info = {"nets": nn, "dup_name_nets": ndup, "aps": int(len(aps)), "s": time.perf_counter() - t, "npk_dict": bool(NPK_DICT[0])}
    return pc, npk, info


ITEMS = set(range(1, 7))                                   # items in force (--no_item removes)
CHECK = {"on": False, "demand_rounds": 0, "demand_bad": 0, "ta_calls": 0, "ta_bad": 0}


def load_all(a):
    """route independent work; runs in a thread while the router runs"""
    t = time.perf_counter()
    try:
        import fast_sta_v3
        sys.modules["fast_sta"] = fast_sta_v3
        import io_run as IR
        G["IR"] = IR; G["B"], G["V1"], G["V2"] = IR.B, IR.V1, IR.V2
        if os.environ.get("ONEPASS_DATA_ROOT"):
            _r = Path(os.environ["ONEPASS_DATA_ROOT"])
            IR.V1.DEFAULT_DATA_ROOT = _r; IR.V2.DEFAULT_DATA_ROOT = _r; IR.V2.DEFAULT_GNN_ROOT = _r / "experiments" / "fastgnn_lofo" / "models"
        import dp_split_api as DS
        G["DS"] = DS
        import fastmode4 as F4, fastmode as FM
        G["F4"], G["FM"] = F4, FM
        import csta as CS, fast_timer as FT
        G["CS"], G["FT"] = CS, FT
        import promote_cpp as PC
        G["PC"] = PC
        from route_cpp import read_route
        G["read_route"] = read_route
        if H["read"]:                                          # item 1
            import h_route_read as HRR
            G["read_route"] = lambda p, _f=read_route: HRR.read_route(p, 8, fallback=_f)
            G["read_route_orig"] = read_route
        if L8["on"] and a.route_bin:                           # load8: --route_bin 1 in big mode: the early read too (same arrays as the text route, ROUTER5 R-BIN)
            sys.path.append(str(ROOT / "tools" / "router5")); import r5bin as R5B
            G["read_route"] = lambda p, _n=int(os.environ.get("OMP_NUM_THREADS", "8")): R5B.read_route_bin(p, _n)
        if os.environ.get("P11_WAIT_ROUTE"):                   # dlchunk: router beside the chain in big mode (docstring)
            def _rr_wait(p, *aa, _f=G["read_route"], _w=os.environ["P11_WAIT_ROUTE"], **kk):
                while not os.path.exists(_w):
                    time.sleep(0.005)
                return _f(p, *aa, **kk)
            G["read_route"] = _rr_wait
        if L8["on"]: G["read_route"] = L8L.early_route(G["read_route"], a.route)   # load8: read beside the load
        PRE["t_imports"] = time.perf_counter() - t
        bench = IR.V1.DEFAULT_DATA_ROOT / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
        if L8["on"] and L8L.use_np_prior():                    # load8: pre5's prior, no torch
            if a.lam > 0 or a.gate > 0: L8L.load_prior(a, IR, PRE)
        elif a.lam > 0 or a.gate > 0:
            IR.V2._fallback_torch_scatter()
            import torch
            torch.set_num_threads(8)
            import dtla_lagnn as DD
            PRE["DD"] = DD
            for _kv in (a.gnn_family or []):                       # chain_gb L3: held-out fold for designs without their own
                _d_, _f_ = _kv.split("=")
                IR.V2.FAMILY_FOR_DESIGN.setdefault(_d_, _f_)
            PRE["model"] = DD.load(str(IR.V2.DEFAULT_GNN_ROOT / f"fold_{IR.V2.family_for(a.design)}.pt"), device="cpu")
        PRE["tech"] = FT.read_tech(str(bench))
        if a.build == "b2":
            sys.path.insert(0, str(ROOT / "tools" / "onepass"))
            import build2_fast as B2
            G["B2"] = B2
            if L8["on"]: L8L.patch_b2(B2)                       # load8: build_fast2's two name dicts at C speed
            _cp = os.path.join(B2.BENCH_ROOT, a.suite, a.design, a.design + ".cap"); _np = os.path.join(B2.BENCH_ROOT, a.suite, a.design, a.design + ".net")
            def _b2p():                                        # load8 b2: on a thread (off: here, as chain_p)
                t2 = time.perf_counter(); _cap = B2.parse_cap_fast(_cp); _net = B2.parse_net_fast(_np)
                if H["pre"]:
                    PRE["h_net"] = (_np, _net)
                _la = B2.TimingLA(_cap, 0.0, B2.GRID / 2000.0, 5.0, 2.0)
                PRE["t_preparse"] = time.perf_counter() - t2
                _OTL = B2.TimingLA
                def _TL(cap, *aa, _c=_cap, _l=[_la], _o=_OTL):
                    return _l.pop() if (cap is _c and _l) else _o(cap, *aa)
                B2.TimingLA = _TL
                _oc, _on = B2.parse_cap_fast, B2.parse_net_fast
                B2.parse_cap_fast = lambda p, _c=_cap, _p=_cp, _o=_oc: _c if p == _p else _o(p)
                B2.parse_net_fast = lambda p, _c=_net, _p=_np, _o=_on: _c if p == _p else _o(p)
            if L8["on"] and L8L.EARLY.get("cold") and L8L.pc("b2"):
                L8L.EARLY["B2P"] = L8L.Fut("B2P", _b2p); L8L.R["pieces"]["b2"] = True
            else:
                _b2p()
        cache = a.cache or str(ROOT / f"experiments/gapla/csta/cache/{a.design}.{'v' if a.suite == 'visible' else 'b'}.npz")
        if a.design == "mempool_tile_wrap" and a.suite == "visible":
            cache = str(ROOT / "experiments/gapla/csta/cache/tile.v.npz")
        cd = CS.load_cache(cache)
        def _kernel_block():                                   # load8: after the pin tables (the bind runs on its own thread); off: here, as chain_p
          if H["inc"] or H["crit"] or a.ovf_timer == "inc":     # items 3c/4a/4b + the W3 stage timer: ONE library (tools/best/i_kern.cc)
            _sp = list(sys.path)
            sys.path.append(str(ROOT / "tools" / "engine"))
            try:
                import csta_inc as CI                          # (csta_inc inserts tools/gapla at sys.path[0]; restored below)
            finally:
                sys.path[:] = _sp
            G["CI"] = CI
            _hl = str(BEST / ("libi_kern." + ("dylib" if sys.platform == "darwin" else "so")))
            if getattr(a, "timer5", 0) == 1:                   # timer5: parallel exact timer, same interface (tools/timer5/csta5.py)
                sys.path.append(str(ROOT / "tools" / "timer5")); import csta5 as CI; G["CI"] = CI; _hl = CI.DEFAULT_LIB
            if L8["on"]: L8L.join_bind(B8L)                     # load8: the bind ran beside the state
            if B8["on"] and "K" in B8L.R:                      # big8 sta: bound the moment the StaDesign was complete
                K = B8L.R["K"]; PRE["inc_info"] = B8L.R["inc_info"]
            else:
                K = CI.IncKernel(_hl); K.bind_design(cd); PRE["inc_info"] = K.bind_rc_design(cd)
            K.L.w3_inc_backward.argtypes = [__import__("ctypes").c_void_p]
            if (H["inc"] or a.ovf_timer == "inc") and not K.inc_ok:
                raise RuntimeError(f"incremental timer refused this design: {PRE['inc_info']}")
            if H["check"] and H["inc"]:                        # reference: a byte copy of the library, own static state
                import shutil
                _rl = str(Path(os.environ.get("H_TMP") or os.environ.get("TMPDIR", "/tmp")) / f"h_libcsta_ref.{os.getpid()}.dylib")
                shutil.copyfile(_hl, _rl)
                Kr = CI.IncKernel(_rl); Kr.bind_design(cd); Kr.bind_rc_design(cd)
                PRE["Kref"] = Kr
          else:
            K = CS.Kernel(); K.bind_design(cd); K.bind_rc_design(cd)
          PRE["cd"], PRE["K"] = cd, K
          if B8["on"]: B8L.after_bind(G, PRE, BIGM)           # big8: cold big mode -> early timer + dfull now
          PRE["t_csta"] = time.perf_counter() - t
        _l8k = L8["on"] and L8L.EARLY.get("cold") and L8L.pc("bind")
        if not _l8k: _kernel_block()
        PRE["S"] = ChainState(a.design, a.suite)           # StaDesign, cap, TimingLA, pin tables, activity
        from rsmt_route import Flute
        PRE["fl"] = Flute()
        PRE["t_state"] = time.perf_counter() - t
        if a.escape != "off" or a.ovf != "off":                 # chain_ovf.py line 98
            t3 = time.perf_counter()
            import escape3d_fast                                   # patches E.dijkstra/marginal/apply
            import escape3d as E
            from timing_promote import TimingLA as _TLA, GRID as _GR
            G["E"] = E
            _lazy = 7 in BIG and FLD is not None and a.ovf != "off"
            if 5 not in ITEMS or a.ovf != "off":                   # the stage needs the pins (chain_ovf.py line 104)
                _nf = os.path.join(E.BENCH_ROOT, a.suite, a.design, a.design + ".net")
                if _lazy:                                          # chain_gb L1: LazyPins over the loader's one .net parse
                    _rp = os.path.realpath(_nf)
                    _hit = [p for p in FLD._NET if os.path.realpath(p) == _rp]
                    PRE["esc_nd"] = FLD.get_net(_hit[0] if _hit else _nf)
                    PRE["esc_pins"] = FLD.LazyPins(PRE["esc_nd"])
                    PRE["esc_pins_reused_parse"] = bool(_hit)
                elif H["pre"] and "h_net" in PRE and os.path.samefile(_nf, PRE["h_net"][0]):   # item 7: from the C++ parse
                    import h_netpins
                    PRE["esc_pins"] = h_netpins.esc_pins_from_fast(E, PRE["h_net"][1])
                    if H["check"]:
                        hchk("netpins", PRE["esc_pins"] == {E._key(n["name"]): n["pins"] for n in E.parse_net_ispd25(_nf)})
                else:
                    PRE["esc_pins"] = {E._key(n["name"]): n["pins"] for n in E.parse_net_ispd25(_nf)}
            if 5 not in ITEMS and a.escape != "off":               # original: demand model in every run (chain_ovf.py lines 105-106)
                PRE["esc_la"] = _TLA(PRE["S"].cap, 0.0, _GR / 2000.0, 5.0, 2.0)
            PRE["t_esc_pre"] = time.perf_counter() - t3
            if a.ovf != "off":                                    # OVF3 STAGE insertion point 1 of 2: route independent pieces (preload thread)
                t4 = time.perf_counter()
                import ovf_lib as OL
                G["OL"] = OL
                if a.ovf_impl == "cpp":                           # chain_w3: the C++ reroute loop
                    import ovf2_lib as OVF2
                    G["OVF2"] = OVF2
                la_ = PRE["S"].la
                if _lazy:                                          # chain_gb L1
                    if L8["on"] and L8L.EARLY.get("cold") and L8L.pc("pins"):   # load8: C++ tables
                        PRE["ovf_pcmap"], PRE["ovf_npins_key"], PRE["ovf_pins_info"] = L8L.pin_maps(PRE["esc_pins"], la_.X, la_.Y, _NpinsKey, NPK_DICT[0])
                    else:
                        PRE["ovf_pcmap"], PRE["ovf_npins_key"], PRE["ovf_pins_info"] = _pin_maps_arrays(PRE["esc_pins"], la_.X, la_.Y)
                else:
                    PRE["ovf_pcmap"] = OL.pin_count_map(PRE["esc_pins"], la_.X, la_.Y)
                    PRE["ovf_npins_key"] = {k: len(v) for k, v in PRE["esc_pins"].items()}
                PRE["t_ovf_pre"] = time.perf_counter() - t4
        if _l8k: _kernel_block()
        if L8["on"]: L8L.join_all()
    except BaseException as e:
        import traceback; traceback.print_exc()
        PRE["err"] = repr(e)
    PRE["preload_s"] = time.perf_counter() - t
    PRE["preload_end"] = time.perf_counter() - T0


class _Segs:
    def __init__(self, arr, off): self.arr, self.off = arr, off
    def __getitem__(self, i): return [tuple(r) for r in self.arr[self.off[i]:self.off[i + 1]].tolist()]
    def __len__(self): return len(self.off) - 1


class _Nets:
    def __init__(self, names, segs): self.names, self.segs = names, segs
    def __getitem__(self, i): return (self.names[i], self.segs[i], None)
    def __iter__(self):
        for i in range(len(self.names)): yield (self.names[i], self.segs[i], None)
    def __len__(self): return len(self.names)


def make_state_class():
    import fastmode4 as F4, fastmode as FM, ispd25_score as SC
    import fast_sta as FS
    from timing_promote import TimingLA, GRID, _key
    from ispd25_adapter import parse_cap_auto

    class ChainState(F4.State4):
        def __init__(self, design, suite):                   # route independent part of State3.__init__
            self.design, self.suite = design, suite
            self.case = SC.case_of(design, suite)
            _, self.r1, self.r2, self.r3, self.nend, (self.w1, self.w2, self.w3, self.w4) = SC.CASES[self.case]
            self.SD = FS.StaDesign(design, suite)
            self.T = None
            SD = self.SD
            self.cap = parse_cap_auto(os.path.join(FS.KIT, suite, design, design + ".cap"))
            self.la = TimingLA(self.cap, self.w4, GRID / 2000.0, 5.0, 2.0)
            self.D = F4.F3._NetView(SD)
            _l8 = L8["on"] and L8L.EARLY.get("cold")
            if L8S:
                self.tidx_key = L8L.tidx_key(SD)
            elif not _l8:
                self.tidx_key = {_key(nm): i for i, nm in enumerate(SD.nf_names)}
            nf = len(SD.nf_names)
            if _l8:
                L8L.start_dicts(self)
            if _l8 or L8S:
                self.pin_start, self.pin_end = L8L.pin_ranges(SD)
            else:
                self.pin_start = np.searchsorted(SD.nf_pin_net, np.arange(nf))
                self.pin_end = np.searchsorted(SD.nf_pin_net, np.arange(nf), side="right")
            self.ap_start = np.cumsum(SD.ap_cnt) - SD.ap_cnt
            self._rc, self._dirty = None, set()
            self.new = {}
            self.norm_timing = True
            import glob
            fa = sorted(glob.glob(str(ROOT / f"experiments/net_slack{'_blind' if suite == 'blind' else ''}/{design}.*.act")))
            if G.get("act"):                                       # chain_gb L2: --act FILE
                fa = [G["act"]]
            PRE["act_file"] = fa[0] if fa else None
            if fa:
                self.activity_file(fa[0])

        def attach(self, names, off, arr6):
            """route dependent part: names, per-net offsets into arr6 [S,6] GCell segments (== read_route25 + demand)"""
            SD = self.SD
            self.segs = _Segs(arr6, off)
            self.nets = _Nets(names, self.segs)
            _c = self.__dict__.get("_b5")
            if 5 in BIG and _c is not None and _c[0] == names:    # chain_gb B5: same net list as the last attach -> same dict/array
                self.ridx = dict(_c[1]); self.tid = _c[2].copy()
            else:
                if (L8S or L8["on"]) and L8L.pc("names") and hasattr(SD.net_names, "table"):   # load8: C speed, C++ table
                    _c8 = L8L.names_tables(names, SD); self.ridx, self.tid = _c8[2], _c8[3]
                else:
                    self.ridx = {_key(nm): i for i, nm in enumerate(names)}
                    self.tid = np.array([SD.net_index.get(_key(nm), -1) for nm in names], np.int64)
                if 5 in BIG:
                    self._b5 = (list(names), dict(self.ridx), self.tid.copy())
            self.new = {}
            cnt = np.diff(off)
            self.base_arr = np.concatenate([np.repeat(np.arange(len(names)), cnt)[:, None], arr6], 1)
            FM.build_demand_fast(self.la, self.base_arr)
            self.__dict__.pop("_f3x_nc", None); self.__dict__.pop("_f3x_pr", None)

        def attach_h(self, names, off, arr6, d):
            """chain_h item 3a: == attach() with the demand array given (d == build_demand_fast of arr6 on a zero map)"""
            SD = self.SD
            self.segs = _Segs(arr6, off)
            self.nets = _Nets(names, self.segs)
            c = self.__dict__.get("_h_nm")                  # ridx / tid depend on the names only: computed once, copies handed out
            if c is None or c[0] != len(names) or c[1] != names:
                if (L8S or (L8["on"] and L8L.EARLY.get("cold"))) and L8L.pc("names") and hasattr(SD.net_names, "table"):   # load8: C speed, C++ table
                    c = self._h_nm = L8L.names_tables(names, SD)
                else:
                    ks = [_key(nm) for nm in names]
                    c = self._h_nm = (len(names), list(names), {k: i for i, k in enumerate(ks)},
                                      np.array([SD.net_index.get(k, -1) for k in ks], np.int64), ks)
            self.ridx = dict(c[2])
            self.new = {}
            self.tid = c[3].copy()
            cnt = np.diff(off)
            self.base_arr = (G["TFP"].base_of(len(names), off, arr6) if "TFP" in G else   # speed10: on threads (same array)
                             np.concatenate([np.repeat(np.arange(len(names)), cnt)[:, None], arr6], 1))
            self.la.d[...] = d
            self.__dict__.pop("_f3x_nc", None); self.__dict__.pop("_f3x_pr", None)

        def timer_array(self):
            if 6 not in ITEMS:
                return super().timer_array()
            import fast_norm as FN
            out = FN.timer_array(self)
            if CHECK["on"]:
                ref = super().timer_array()
                CHECK["ta_calls"] += 1
                ok = ref.dtype == out.dtype and np.array_equal(ref, out)
                if not ok:
                    CHECK["ta_bad"] += 1
                print(f"[check] timer_array call {CHECK['ta_calls']}: {len(out)} rows, {len(self.new)} changed nets, equal={ok}", flush=True)
            return out
    return ChainState


def assemble(P, choice, rt, IR, V1, RE):
    """io_run.write_fast up to the column arrays (same arithmetic), returns (noff, cols)"""
    C = RE.GCELL_DBU; h = C // 2
    rn = V1._np(P["run_net"], np.int64); cand = V1._np(P["cand"], np.int64)
    dirs = list(P["dirs"])
    by_net = collections.defaultdict(list)
    for r, n in enumerate(rn.tolist()):
        by_net[int(n)].append(r)
    new = {}
    for n, runs in by_net.items():
        wires = collections.defaultdict(list)
        node_layers = collections.defaultdict(set)
        for r in runs:
            z = int(cand[r, int(choice[r])])
            if z < 0:
                continue
            d = int(P["run_dir"][r]); f = int(P["run_fixed"][r])
            lo = int(P["run_lo"][r]); hi = int(P["run_hi"][r])
            wires[(z, f)].append((lo, hi))
            a = (lo, f) if d == 0 else (f, lo)
            b = (hi, f) if d == 0 else (f, hi)
            node_layers[a].add(z); node_layers[b].add(z)
        for xy, span in P["net_nodes"][n].items():
            node_layers[xy].update((int(span[0]), int(span[1])))
        buf = []
        for (z, f), ivs in wires.items():
            for lo, hi in ivs:
                buf.append((lo, f, z, hi, f, z) if dirs[z] == 0 else (f, lo, z, f, hi, z))
        for (x, y), zs in node_layers.items():
            if zs:
                for z in range(min(zs), max(zs)):
                    buf.append((x, y, z, x, y, z + 1))
        new[int(P["net_route_idx"][n])] = buf
    off = rt.offsets.astype(np.int64)
    nn = len(rt.net_names)
    cnt = np.diff(off).copy()
    for i, b in new.items():
        cnt[i] = len(b)
    noff = np.zeros(nn + 1, dtype=np.uint64); noff[1:] = np.cumsum(cnt)
    tot = int(noff[-1])
    cols = [np.empty(tot, np.int32) for _ in range(6)]
    src = [rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1]
    keep = np.ones(nn, bool); keep[list(new)] = False
    kn = np.flatnonzero(keep & (cnt > 0))
    if len(kn):
        c = cnt[kn]
        s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn].astype(np.int64), c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        for k in range(6):
            cols[k][d0 + ar] = src[k][s0 + ar]
    for i, b in new.items():
        if b:
            arr = np.array(b, dtype=np.int64); o = int(noff[i]); m = len(b)
            cols[0][o:o+m] = arr[:, 0] * C + h; cols[1][o:o+m] = arr[:, 1] * C + h; cols[2][o:o+m] = arr[:, 2]
            cols[3][o:o+m] = arr[:, 3] * C + h; cols[4][o:o+m] = arr[:, 4] * C + h; cols[5][o:o+m] = arr[:, 5]
    return noff, cols, len(new)


def write_cols(IR, path, names, noff, cols):
    import ctypes
    blob = ("\n".join(names) + "\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    I32 = ctypes.POINTER(ctypes.c_int32)
    WL = IR._W
    _lb = ROOT / "tools" / "srvw" / "bigcase" / "libio_write_b.so"
    if 4 in BIG and _lb.exists():                              # chain_gb B4: parallel formatting, same bytes (io_write_b.cc)
        WL = ctypes.CDLL(str(_lb))
        WL.blocks_write.argtypes = IR._W.blocks_write.argtypes; WL.blocks_write.restype = ctypes.c_int
    rc = WL.blocks_write(os.fsencode(str(path)), blob, len(names), noff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                         *(c.ctypes.data_as(I32) for c in cols))
    assert rc == 0


def write_cpp(S, rt, noff, cols, arr6, IR, out, W):
    """normalize_net of every net + the C++ block writer (== S.write(normalize=True) bytes)"""
    import ctypes
    t = time.perf_counter()
    lib = ctypes.CDLL(str(HERE / ("libnorm_write.dylib" if sys.platform == "darwin" else "libnorm_write.so")))
    P_ = ctypes.c_void_p
    lib.nw_run.argtypes = [ctypes.c_long] + [P_] * 6 + [ctypes.c_int]; lib.nw_run.restype = ctypes.c_long
    lib.nw_fetch.argtypes = [P_] * 7
    nn = len(rt.net_names)
    off = noff.astype(np.int64)
    start = off[:-1].copy(); cnt = np.diff(off).copy()
    segs = arr6.astype(np.int32)
    if S.new:
        ex = []; pos = len(segs)
        for i, sg in S.new.items():
            start[i] = pos; cnt[i] = len(sg); pos += len(sg); ex.extend(sg)
        segs = np.concatenate([segs, np.array(ex, np.int32).reshape(-1, 6)], 0)
    segs = np.ascontiguousarray(segs)
    D = S.D
    pa0 = np.zeros(nn, np.int64); pa1 = np.zeros(nn, np.int64)
    from timing_promote import _key
    tk = S.tidx_key; ps, pe, aps, ac = S.pin_start, S.pin_end, S.ap_start, D.ap_cnt
    for i, nm in enumerate(rt.net_names):
        j = tk.get(_key(nm))
        if j is not None and pe[j] > ps[j]:
            pa0[i] = aps[ps[j]]; pa1[i] = aps[pe[j] - 1] + ac[pe[j] - 1]
    ap = np.ascontiguousarray(np.asarray(D.ap, np.int32).reshape(-1, 3))
    W["w_prep_s"] = time.perf_counter() - t; t = time.perf_counter()
    tot = lib.nw_run(nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data, pa0.ctypes.data, pa1.ctypes.data, ap.ctypes.data, 4200)
    oc = [np.empty(tot, np.int32) for _ in range(6)]; oo = np.zeros(nn + 1, np.uint64)
    lib.nw_fetch(*(c.ctypes.data for c in oc), oo.ctypes.data)
    W["w_norm_s"] = time.perf_counter() - t; t = time.perf_counter()
    write_cols(IR, out, rt.net_names, oo, oc)
    W["w_io_s"] = time.perf_counter() - t


def escape_inmem(a, rt, W):
    E, FM_ = G["E"], G["FM"]
    C = 4200; h = C // 2; GR = 4200
    t = time.perf_counter()
    off = rt.offsets.astype(np.int64)
    arr6 = np.empty((len(rt.x0), 6), np.int64)
    arr6[:, 0] = rt.x0.astype(np.int64) // GR; arr6[:, 1] = rt.y0.astype(np.int64) // GR; arr6[:, 2] = rt.z0
    arr6[:, 3] = rt.x1.astype(np.int64) // GR; arr6[:, 4] = rt.y1.astype(np.int64) // GR; arr6[:, 5] = rt.z1
    nets = _Nets(list(rt.net_names), _Segs(arr6, off))
    lazy = "esc_la" not in PRE
    if lazy:                     # item 5: the share test runs on the state's demand array (same cap -> same c, u, L; d is
        la = PRE["S"].la         # still all zero here, and eval_cols zeroes it before every attach)
        la.d[:] = 0
    else:
        la = PRE["esc_la"]
    base = np.concatenate([np.repeat(np.arange(len(nets)), np.diff(off))[:, None], arr6], 1)
    if H["dem"]:                                       # item 5: the router route's map in C++ (same values), kept for item 3a
        import h_dem as HD
        HD.build(FM_, la, base)
        if H["check"]:
            ref_ = la.d.copy(); la.d[:] = 0; FM_.build_demand_fast(la, base)
            hchk("dem_router", np.array_equal(ref_, la.d), HD.STATS); del ref_
    else:
        FM_.build_demand_fast(la, base)
    del base
    # blocked share of the overflow cost (zero-capacity cells, excess over the floor), measured on the router output
    tot = blk = 0.0
    M5_ = G.get("M5") if getattr(a, "mid5", 0) else None
    sh5 = M5_.blocked_share(la, a.of_rule == "official", a.esc_thresh, int(os.environ.get("MID5_THREADS", "8")), E) if M5_ else None
    for z in (range(1, la.L) if sh5 is None else ()):     # mid5 C: the C++ share when its on/off decision is certain
        d, c = la.d[z], la.c[z]
        with np.errstate(over="ignore"):
            allc = np.where(c > E.CAP_EPS, np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0)), np.exp(np.minimum(0.75 * d, 700.0)))
            if a.of_rule == "official":                    # official evaluator: an idle blocked cell costs nothing
                allc = np.where((c <= E.CAP_EPS) & (d <= 0), 0.0, allc)
            bl = np.where((c <= E.CAP_EPS) & (d > 0), np.exp(np.minimum(0.75 * d, 700.0)) - 1.0, 0.0)
        tot += float(la.u[z]) * float(allc.sum()); blk += float(la.u[z]) * float(bl.sum())
    share = (blk / tot if tot > 0 else 0.0) if sh5 is None else sh5
    W["esc_blocked_share"] = share
    on = a.escape == "on" or (a.escape == "auto" and share > a.esc_thresh)
    W["esc_on"] = bool(on)
    if "T_EVT" in G: G["T_EVT"].set()                      # speed10: --t_esc_wait
    W["esc_prep_s"] = time.perf_counter() - t; t = time.perf_counter()
    if not on and H["dem"]:                  # chain_h items 3a + 5: keep the router route's map and rows for the candidates
        PRE["h_Dr"] = la.d.copy(); PRE["h_arr6r"] = arr6; PRE["h_offr"] = off
        W["h_keep_s"] = time.perf_counter() - t; t = time.perf_counter()
    if not on:
        if lazy:
            la.d[:] = 0
        PRE.pop("esc_la", None)
        if a.ovf == "off": PRE.pop("esc_pins", None)
        return rt
    if lazy:                     # escape on (bsg only; NOT tested on the Mac): build what the original preloaded, same values
        from timing_promote import TimingLA as _TLA, GRID as _GR
        t5 = time.perf_counter()
        la_s = la
        la = _TLA(PRE["S"].cap, 0.0, _GR / 2000.0, 5.0, 2.0)
        la.d[...] = la_s.d                                     # == build_demand_fast on a fresh zero array (same input)
        la_s.d[:] = 0
        if "esc_pins" not in PRE:                              # chain_g: already loaded when --ovf is on (same dict)
            _netp = os.path.join(E.BENCH_ROOT, a.suite, a.design, a.design + ".net")
            if 2 in BIG and FLD is not None:                   # chain_gb B2: lazy per-net pins over the loader's one .net parse
                _rp = os.path.realpath(_netp)
                _hit = [p for p in FLD._NET if os.path.realpath(p) == _rp]
                PRE["esc_pins"] = FLD.LazyPins(FLD.get_net(_hit[0] if _hit else _netp))
                W["esc_pins_reused_parse"] = bool(_hit)
            else:
                PRE["esc_pins"] = {E._key(n["name"]): n["pins"] for n in E.parse_net_ispd25(_netp)}
        W["esc_lazy_pre_s"] = time.perf_counter() - t5
        t = time.perf_counter()
    E.emit = lambda la_, wires, vias: _emit_tuples(la_, wires, vias)         # GCell tuples instead of text lines
    new_lines, stats, _ = E.escape(None, PRE["esc_pins"], nets, a.n_hot, 12, 1.0, 0.1, a.esc_passes, la=la)
    W["esc_search_s"] = time.perf_counter() - t; t = time.perf_counter()
    W["esc_nets_changed"] = len(new_lines); W["esc_stats"] = dict(stats)
    nn = len(nets)
    cnt = np.diff(off).copy()
    for i, b in new_lines.items():
        cnt[i] = len(b)
    noff = np.zeros(nn + 1, np.int64); noff[1:] = np.cumsum(cnt)
    tot = int(noff[-1])
    cols = [np.empty(tot, np.int32) for _ in range(6)]
    src = [rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1]
    keep = np.ones(nn, bool); keep[list(new_lines)] = False
    kn = np.flatnonzero(keep & (cnt > 0))
    c = cnt[kn]
    s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn], c)
    ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
    for k in range(6):
        cols[k][d0 + ar] = src[k][s0 + ar]
    for i, b in new_lines.items():
        if b:
            arr = np.array(b, dtype=np.int64); o = int(noff[i]); m = len(b)
            cols[0][o:o+m] = arr[:, 0] * C + h; cols[1][o:o+m] = arr[:, 1] * C + h; cols[2][o:o+m] = arr[:, 2]
            cols[3][o:o+m] = arr[:, 3] * C + h; cols[4][o:o+m] = arr[:, 4] * C + h; cols[5][o:o+m] = arr[:, 5]
    rt2 = types.SimpleNamespace(net_names=rt.net_names, offsets=noff, x0=cols[0], y0=cols[1], z0=cols[2], x1=cols[3], y1=cols[4], z1=cols[5],
                                net_id=np.repeat(np.arange(nn), cnt).astype(rt.net_id.dtype))
    W["esc_splice_s"] = time.perf_counter() - t
    del arr6, nets, la
    PRE.pop("esc_la", None)
    if a.ovf == "off": PRE.pop("esc_pins", None)
    import gc; gc.collect()
    return rt2


def _emit_tuples(la, wires, vias):
    buf = []
    for (z, f), ivs in wires.items():
        for a_, b_ in ivs:
            buf.append((a_, f, z, b_, f, z) if la.dirs[z] == 0 else (f, a_, z, f, b_, z))
    for (x, y, z) in sorted(vias):
        buf.append((x, y, z, x, y, z + 1))
    return buf


# ---- moved verbatim from tools/ovf3/chain_ovf.py lines 357-475 ----


def B5_ON(a):
    return getattr(a, "big5", 0) == 1 and not BIGM


def overflow_stage(S, cur, a, W, log, K, mk):
    """OVF3 stage (chain_g) + W3 options. Each sub-stage is a candidate kept only if the exact in-run 4-term estimate
    (official overflow rule) improves; else S.revert(undo) (and K.reject() with the incremental timer)."""
    OL = G["OL"]
    pins = PRE["esc_pins"]
    rec = {"stage": "ovf", "w3": {"impl": a.ovf_impl, "timer": a.ovf_timer, "skip": a.ovf_skip, "rr_passes": a.ovf_rr_passes,
                                  "order": a.ovf_order, "esc_pre": a.ovf_esc_pre, "big_nets": a.ovf_big_nets, "hmax_big": a.ovf_hmax_big, "big_runs": a.ovf_big_runs, "big_passes": a.ovf_big_passes, "max_runs": a.ovf_max_runs, "hmax": a.ovf_hmax, "hot": a.ovf_hot}}
    t0 = time.perf_counter()
    chunked = bool(DL["on"] and getattr(a, "dl_ovf_chunk", 0))
    rec["dl_ovf_chunk"] = int(chunked)
    def dl_ovf_batch(tag):
        if not chunked:
            return True
        ok = DLG("ovf_batch")
        if not ok:
            rec["deadline_stop"] = tag
        return ok
    def dl_ovf_done():
        if chunked:
            DLE("ovf_batch")
    cost0 = None
    if getattr(a, "n_pre", 0) and a.ovf_impl == "cpp" and B5_ON(a):   # speed9: on the overflowed cells only (the stage's selection is tools/late5's, which needs no map)
        sys.path.append(str(ROOT / "tools" / "best5")); import n_sel as NSEL_
        sys.path.append(str(ROOT / "tools" / "late5"))
        pt, cost0 = NSEL_.pretest_fast(S, int(os.environ.get("MID5_THREADS", "8")))
    elif a.ovf_impl == "cpp" and a.mid5 and G.get("M5") is not None:   # mid5 G: the cost map on threads (same array)
        pt, cost0 = G["M5"].pretest_cost(S, OL, int(os.environ.get("MID5_THREADS", "8")))
        if os.environ.get("MID5_CHECK") == "1":
            pr_, cr_ = G["OVF2"].pretest_cost(S)
            print(f"[mid5 check] pretest equal={pr_ == pt and cr_.dtype == cost0.dtype and np.array_equal(cr_.view(np.uint64), cost0.view(np.uint64))} strides {cr_.strides} {cost0.strides} {pr_} {pt}", flush=True)
    elif a.ovf_impl == "cpp":
        pt, cost0 = G["OVF2"].pretest_cost(S)           # same numbers as OL.pretest; cost array kept for the first selection
    else:
        pt = OL.pretest(S)
    rec["pretest"] = pt
    if os.environ.get("OVF3_DEBUG"):
        import hashlib
        rec["dbg_d_md5"] = hashlib.md5(np.ascontiguousarray(S.la.d).tobytes()).hexdigest()[:8]
        rec["dbg_slack_md5"] = hashlib.md5(np.asarray(cur[0]["net_slack"], np.float64).tobytes()).hexdigest()[:8]
        rec["dbg_nguard"] = int((np.asarray(cur[0]["net_slack"]) < 0.05).sum())
    rec["pretest_s"] = time.perf_counter() - t0
    if pt["bound_score"] < a.ovf_eps:
        rec["skipped"] = "bound %.3g < eps %.3g" % (pt["bound_score"], a.ovf_eps)
        log.append(rec)
        return cur
    tid = S.tid
    o0 = cur[0]
    sl0 = np.asarray(o0["net_slack"], np.float64)
    slack0 = np.where(tid >= 0, sl0[np.maximum(tid, 0)], 9.0)
    # W3 skip test (cheap: one pass over the nets' slack; the pretest's bound is already computed)
    t_ = time.perf_counter()
    ug = float((slack0 >= OL.SLACK_GUARD).mean()) if len(slack0) else 0.0
    sq = {"bound": pt["bound_score"], "unguarded": ug, "bound_ug": pt["bound_score"] * ug,
          "bound_ug2": pt["bound_score"] * ug * ug, "blk_ug": pt["bound_blk_score"] * ug}
    rec["skip_q"] = sq; rec["skip_s"] = time.perf_counter() - t_
    if a.ovf_skip > 0 and sq[a.ovf_skip_key] < a.ovf_skip:
        rec["skipped"] = "%s %.4g < %.4g" % (a.ovf_skip_key, sq[a.ovf_skip_key], a.ovf_skip)
        rec["s"] = time.perf_counter() - t0
        log.append(rec)
        print("[ovf3] " + json.dumps(rec), flush=True)
        return cur

    OFO = G["OFO"]

    fast_of = a.ovf_timer == "inc" and getattr(G["FM"], "_of_official_patched", False)

    def off_score(r_):                                   # the 4-term estimate with the OFFICIAL blocked-cell rule
        # with --of_rule official mk's r_["of"] IS of_official.overflow_cost of the same state (the same function on the same
        # array, so the same double): the second map scan is skipped, the arithmetic below is unchanged (same bits)
        ofv = r_["of"] if fast_of else OFO.overflow_cost(S.la)
        return float(r_["score"] - S.w4 * r_["of"] + S.w4 * ofv)

    cur_off = [off_score(cur[2])]
    rec["est_in_off"] = cur_off[0]
    inc = [a.ovf_timer == "inc"]
    if inc[0]:
        v_ = S.tid[S.tid >= 0]
        if len(np.unique(v_)) != len(v_):                # one route net per timer net is what the subset update assumes
            inc[0] = False
            rec["inc_refused"] = "several route nets map to one timer net"
    rec["retimes"] = []

    def retime_inc(tag):
        import fast_norm as FN
        t_0 = time.perf_counter()
        new, snap = S.new, K.snap
        ch = [i for i, sg in new.items() if snap.get(i) is not sg] + [i for i in snap if i not in new]
        cache = S.__dict__.setdefault("_f3x_nc", {})
        stale = [i for i in ch if i in new and (cache.get(i) or (None,))[0] is not new[i]]
        if stale:
            for i, b in G.get("NORM", FN._normalise)(S, stale).items():
                cache[i] = (new[i], b)
        off = S.segs.off
        parts = [cache[i][1] if i in new else S.base_arr[off[i]:off[i + 1]] for i in ch]
        arr = np.concatenate(parts, 0) if parts else np.zeros((0, 7), np.int64)
        sub = arr.astype(np.int64, copy=True)
        sub[:, 0] = S.tid[arr[:, 0]] if len(arr) else sub[:, 0]
        sub = sub[sub[:, 0] >= 0]
        nets = np.unique(S.tid[np.asarray(ch, np.int64)]) if ch else np.zeros(0, np.int64)
        nets = nets[nets >= 0]
        t_1 = time.perf_counter()
        K.begin(); K.build_rc_subset(sub, nets); t_2 = time.perf_counter()
        o_ = K.update(nets); t_3 = time.perf_counter()
        K.pending_snap = dict(new)
        rc_ = K.rc
        r_ = mk(o_, rc_)
        rec["retimes"].append({"tag": tag, "mode": "inc", "nets": int(len(nets)), "select_s": t_1 - t_0, "rc_s": t_2 - t_1,
                               "update_s": t_3 - t_2, "estimate_s": time.perf_counter() - t_3})
        return o_, rc_, r_

    def esc_will_run():
        return bool(a.ovf_esc and pt["bound_blk_score"] >= a.ovf_eps and not (a.ovf_esc_pre == "skip" and W.get("esc_on")))

    def retime_gate(undo, tag, cur_):
        t_ = time.perf_counter()
        if inc[0]:
            o_, rc_, r_ = retime_inc(tag)
        else:
            arr = S.timer_array(); rc_ = K.build_rc(arr.astype(np.int64))
            o_ = K.analyze(rc_["c1"], rc_["c2"], rc_["rpi"], rc_["ctot"], rc_["elm"], rc_["par"])
            r_ = mk(o_, rc_)
        so = off_score(r_)
        keep = bool(undo) and so < cur_off[0] - 1e-9
        rec[tag + "_est"] = {k: float(r_[k]) for k in ("wns", "tns", "of", "pw", "score")}
        rec[tag + "_est"]["score_off"] = so
        if keep:
            cur_off[0] = so
        rec[tag + "_kept"] = keep
        if not keep:
            S.revert(undo)
            if inc[0]:
                K.reject()
        elif inc[0]:
            K.commit()
            stb = np.zeros(2)
            # the slack of the kept state is read only by a part that follows (the escape after the reroute, or the reroute after
            # the escape); otherwise the required-time pass is skipped and net_slack is not carried (nothing after the stage reads it)
            need_sl = (tag == "rr" and a.ovf_order == "re" and esc_will_run()) or (tag == "esc" and a.ovf_order == "er" and a.ovf_rr)
            if need_sl:
                K.L.w3_inc_backward(stb.ctypes.data_as(__import__("ctypes").c_void_p))   # RAT / slack of the kept state
            else:
                o_ = {k: v for k, v in o_.items() if k != "net_slack"}
            # copies of what later code reads (K.crit: AT, BEST, rmin; slack guard: net_slack; estimates: scalars, net_rd_kohm);
            # the other kernel views (RAT, SL, gate_d, wire_d, pin_slack, ep_sl_all) are not read after the stage and are dropped
            o_ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o_.items()
                  if k not in ("RAT", "SL", "gate_d", "wire_d", "pin_slack", "ep_sl_all")}
            rc_ = {k: v.copy() for k, v in rc_.items()}
            rec[tag + "_backward_s"] = float(stb[0])
        rec[tag + "_retime_s"] = time.perf_counter() - t_
        return (o_, rc_, r_) if keep else cur_

    from timing_promote import _key as _k
    state = {"cur": cur, "esc_undo": None}

    def part_escape():
        cur_ = state["cur"]
        if not (a.ovf_esc and pt["bound_blk_score"] >= a.ovf_eps):
            return
        if a.ovf_esc_pre == "skip" and W.get("esc_on"):  # W3: the pre-pass escape already ran on this route (c4 rule)
            rec["esc_skipped"] = "pre-pass escape on"
            return
        o_ = cur_[0]
        sl_def = np.asarray(o_["net_slack"], np.float64)
        slack_r = np.where(tid >= 0, sl_def[np.maximum(tid, 0)], 9.0)
        t_ = time.perf_counter()
        M5_ = G.get("M5") if a.mid5 else None
        if M5_ is not None:                              # mid5 B: the same set, vectorised (cached keys / pin counts)
            guarded = M5_.guarded_keys(S, PRE, OL, _k, slack_r)
        else:
            npk = PRE["ovf_npins_key"]
            nseg = np.diff(S.segs.off)
            guarded = frozenset(_k(nm) for i, nm in enumerate(S.nets.names)
                                if slack_r[i] < OL.SLACK_GUARD or npk.get(_k(nm), 0) > OL.MAX_PINS or nseg[i] > OL.RB.BIG_NET)
        rec["esc_guarded"] = len(guarded)
        undo, st = {}, {}
        for p in range(a.ovf_passes):
            if not dl_ovf_batch("escape_%d" % (p + 1)):
                break
            if M5_ is not None:                          # mid5 B: the escape pass in C++ (tools/mid5/mid5_kern.cc)
                f_ = M5_.escape_pass_checked if os.environ.get("MID5_CHECK") == "1" else M5_.escape_pass
                u_ = f_(S, pins, a.ovf_hot, guarded, OL, G["E"], PRE, _k, stats=st, astar=a.ovf_astar)
            else:
                u_ = OL.escape_pass(S, pins, a.ovf_hot, guarded, stats=st, astar=a.ovf_astar)
            for i, sg in u_.items():
                undo.setdefault(i, sg)
            dl_ovf_done()
            if not u_:
                break
            if a.t_esc_p2 > 0 and p == 0 and len(u_) < a.t_esc_p2:   # speed10
                rec["t_esc_p2_skip"] = len(u_)
                break
        if a.ovf_esc_check and undo:                     # W3: drop every escaped net the evaluator would call open
            from timing_promote import to_wires_vias as _twv
            bad = {}
            for i in list(undo):
                w_, v_ = _twv(S.new[i], S.la.dirs)
                if not OL.net_valid_fast(S.la, w_, v_, pins.get(_k(S.nets.names[i]), [])):
                    bad[i] = undo.pop(i)
            if bad:
                S.revert(bad)
            rec["esc_invalid_reverted"] = len(bad)
        rec["esc_s"] = time.perf_counter() - t_; rec["esc_nets"] = len(undo); rec["esc_stats"] = st
        if a.ovf_gate == "one" and a.ovf_rr and a.ovf_order == "er":
            state["esc_undo"] = undo
        else:
            state["cur"] = retime_gate(undo, "esc", cur_)

    def part_reroute():
        cur_ = state["cur"]
        if not a.ovf_rr:
            return
        t_ = time.perf_counter()
        if a.mid5 and G.get("M5") is not None:          # mid5 B: cached (same values)
            npins_r = G["M5"].keys_npins(S, PRE, _k)[1].copy()
        else:
            npk = PRE["ovf_npins_key"]
            npins_r = np.array([npk.get(_k(nm), 0) for nm in S.nets.names], np.int64)
        o_ = cur_[0]
        sl_def = np.asarray(o_["net_slack"], np.float64)
        slack_r = np.where(tid >= 0, sl_def[np.maximum(tid, 0)], 9.0)
        undo = {}
        did = False
        ps = -1
        npass = max(1, a.ovf_rr_passes); big = False
        while ps + 1 < npass:
            if not dl_ovf_batch("reroute_%d" % (ps + 2)):
                break
            ps += 1
            if OL.ALOG is not None: OL.ALOG.clear()
            ts = time.perf_counter()
            if ps == 0 or a.ovf_rr_resel:
                if a.ovf_impl == "cpp":                  # same result as OL.select_c (ovf2_lib docstring)
                    c0 = cost0 if (ps == 0 and a.ovf_order == "re") else None      # demand unchanged since the pretest
                    r9_ = None
                    if getattr(a, "n_sel", 0) and B5_ON(a):   # speed9: the same chosen nets, mask and pairs on threads
                        sys.path.append(str(ROOT / "tools" / "best5")); sys.path.append(str(ROOT / "tools" / "late5")); import n_sel as NSEL_
                        r9_ = NSEL_.select_c(S, S.la, slack_r, npins_r, a.ovf_cap, a.ovf_maxnets, int(os.environ.get("MID5_THREADS", "8")), share=True)
                    chosen, info = r9_ if r9_ is not None else G["OVF2"].select_c_fast(S, S.la, slack_r, npins_r, a.ovf_cap, a.ovf_maxnets, cost=c0)
                else:
                    chosen, info = OL.select_c(S, S.la, slack_r, npins_r, a.ovf_cap, a.ovf_maxnets)
            else:                                        # W3 --ovf_rr_resel 0: later passes reuse the first pass' nets
                info = dict(rec["rr_select"]); info["reused"] = True
            if ps == 0:
                rec["rr_select"] = info; rec["rr_select_s"] = time.perf_counter() - t_
            else:
                rec.setdefault("rr_more", []).append({"pass": ps + 1, "select": info, "select_s": time.perf_counter() - ts})
            if not (chosen and float(S.w4) * info.get("cost_uncovered", 0.0) >= 0.0):
                break
            pin_on = (info.get("blk_share_ov", 0.0) > 0.5) if a.ovf_pin == "auto" else bool(int(a.ovf_pin))
            if ps == 0:
                rec["rr_pin_on"] = pin_on
            t2 = time.perf_counter(); st = {}; tm = {}
            if ps == 0 and a.ovf_big_nets and len(chosen) > a.ovf_big_nets:   # W3: large first selection (bsg): narrow windows,
                big = True                                                     # own run budget and pass count
                if a.ovf_big_passes:
                    npass = a.ovf_big_passes
                rec["rr_big"] = {"hmax": a.ovf_hmax_big, "runs": a.ovf_big_runs or a.ovf_max_runs, "passes": npass}
            hmax_ = a.ovf_hmax_big if big else a.ovf_hmax
            runs_ = (a.ovf_big_runs or a.ovf_max_runs) if big else (a.t_runs or a.ovf_max_runs)   # speed10: --t_runs
            if a.ovf_impl == "cpp" and a.ovf_rank and getattr(a, "n_rr", -1) >= 0 and getattr(a, "t_rr", 0):   # speed10: tools/best10/t_rr
                sys.path.append(str(ROOT / "tools" / "best10")); import t_rr as TRR_
                u_ = TRR_.reroute_rank(S, pins, chosen, W=a.ovf_W, pin_on=pin_on, pcmap=PRE.get("ovf_pcmap"), stats=st, astar=a.ovf_astar, max_runs=runs_, hmax=hmax_, timing=tm,
                                       nthr=int(os.environ.get("MID5_THREADS", "8")), batch=a.n_rr_batch, policy=a.n_rr, shift=a.n_blk,
                                       cfg={"rH": a.t_rect_h, "rW": a.t_rect_w, "dH": a.t_long_h, "dN": a.t_long_n, "popcap": a.t_popcap, "areaB": a.t_rr_area})
                rec["n_rr"] = TRR_.RR[-1]; rec["t_rr"] = {"area": TRR_.CFG.get("area"), "cut": TRR_.CFG.get("cut")}
            elif a.ovf_impl == "cpp" and a.ovf_rank and getattr(a, "n_rr", -1) >= 0:   # speed9: tools/best5/n_rr
                sys.path.append(str(ROOT / "tools" / "best5")); import n_rr as NRR_
                u_ = NRR_.reroute_rank(S, pins, chosen, W=a.ovf_W, pin_on=pin_on, pcmap=PRE.get("ovf_pcmap"), stats=st, astar=a.ovf_astar, max_runs=runs_, hmax=hmax_, timing=tm,
                                       nthr=int(os.environ.get("MID5_THREADS", "8")), batch=a.n_rr_batch, policy=a.n_rr, shift=a.n_blk)
                rec["n_rr"] = NRR_.RR[-1]
            elif a.ovf_impl == "cpp" and a.ovf_rank and a.mid5 and a.mid5_rr and G.get("M5") is not None:   # mid5 D: loop on T threads
                f_ = G["M5"].reroute_rank_checked if os.environ.get("MID5_CHECK") == "1" else G["M5"].reroute_rank
                u_ = f_(S, pins, chosen, W=a.ovf_W, pin_on=pin_on, pcmap=PRE.get("ovf_pcmap"), stats=st, astar=a.ovf_astar,
                        max_runs=runs_, hmax=hmax_, timing=tm, nthr=int(os.environ.get("MID5_THREADS", "8")), batch=a.mid5_batch)
            elif a.ovf_impl == "cpp" and a.ovf_rank:
                u_ = G["OVF2"].reroute_rank_cpp(S, pins, chosen, a.ovf_W, pin_on, PRE.get("ovf_pcmap"), st, a.ovf_astar,
                                                runs_, hmax_, timing=tm)
            elif a.ovf_rank:
                u_ = OL.reroute_rank(S, pins, chosen, a.ovf_W, pin_on, PRE.get("ovf_pcmap"), st, a.ovf_astar, runs_, hmax_)
            else:
                u_ = OL.reroute_c(S, pins, chosen, a.ovf_W, pin_on, PRE.get("ovf_pcmap"), bool(a.ovf_uf), st, a.ovf_astar, a.ovf_rr_hot)
            for i, sg in u_.items():
                undo.setdefault(i, sg)
            did = True
            if ps == 0:
                rec["rr_search_s"] = time.perf_counter() - t2; rec["rr_stats"] = st; rec["rr_cpp_t"] = tm
                if OL.ALOG is not None:
                    al = np.array(OL.ALOG, np.float64).reshape(-1, 3)
                    rec["rr_alog"] = {"calls": len(al), "s": float(al[:, 1].sum())}
            else:
                rec["rr_more"][-1].update(search_s=time.perf_counter() - t2, stats=st)
            if not u_:
                break
            dl_ovf_done()
        if did:
            if state["esc_undo"] is not None:            # --ovf_gate one: escape + reroute checked once
                cur1 = retime_gate(undo, "rr", cur_)
                if cur1 is cur_:
                    cur_ = retime_gate(state["esc_undo"], "esc", cur_)
                else:
                    cur_ = cur1; rec["esc_kept"] = "with_rr"
                state["esc_undo"] = None
            else:
                cur_ = retime_gate(undo, "rr", cur_)
            state["cur"] = cur_
        rec["rr_s"] = time.perf_counter() - t_

    def n_invalid():
        from timing_promote import to_wires_vias as _twv
        n_ = 0
        for i, sg in S.new.items():
            w_, v_ = _twv(sg, S.la.dirs)
            n_ += not OL.net_valid_fast(S.la, w_, v_, pins.get(_k(S.nets.names[i]), []))
        return n_

    for part in ((part_escape, part_reroute) if a.ovf_order == "er" else (part_reroute, part_escape)):
        part()
        if os.environ.get("W3_VALID", "0") == "1":       # diagnostics: invalid nets among the changed ones after each part
            rec.setdefault("invalid_after", []).append((part.__name__, len(S.new), n_invalid()))
    if state["esc_undo"] is not None:
        state["cur"] = retime_gate(state["esc_undo"], "esc", state["cur"])
    cur = state["cur"]
    rec["s"] = time.perf_counter() - t0
    rec["est"] = {k: float(cur[2][k]) for k in ("wns", "tns", "of", "pw", "score")}
    rec["est"]["score_off"] = cur_off[0]
    rec["kept"] = bool(rec.get("esc_kept") or rec.get("rr_kept"))
    if os.environ.get("W3_MD5", "0") == "1":             # identity diagnostics: changed nets + demand map after the stage
        import hashlib
        h = hashlib.md5()
        for i in sorted(S.new):
            h.update(repr((int(i), [tuple(int(v) for v in r) for r in S.new[i]])).encode())
        rec["md5_new"] = h.hexdigest()[:12]; rec["n_new"] = len(S.new)
        rec["md5_d"] = hashlib.md5(np.ascontiguousarray(S.la.d).tobytes()).hexdigest()[:12]
        if "net_slack" in cur[0]:
            rec["md5_slack"] = hashlib.md5(np.ascontiguousarray(np.asarray(cur[0]["net_slack"], np.float64)).tobytes()).hexdigest()[:12]
    log.append(rec)
    print("[ovf3] " + json.dumps(rec), flush=True)
    return cur

# ---- end of moved block ----


def reshape_stage(S, cur, a, log, K, mk, weights, retime, H, PRE):
    import rs_lib
    import ovf2_lib as O2
    from timing_promote import _key as _k
    t0 = time.perf_counter()
    rec = {"stage": "reshape", "order": a.rs_order, "classes": []}
    L5_ = None; RR_ = rs_lib.reroute_rs
    if getattr(a, "late5", 0):                          # late5: C++ selection (tools/late5/l5.py), parallel re-route loop (l5_rs)
        import l5 as L5_
        import l5_rs
        if os.environ.get("L5_RS", "1") == "1":
            RR_ = l5_rs.reroute_rs
    if getattr(a, "n_rs", -1) >= 0:                     # speed9: tools/best5/n_rs
        sys.path.append(str(ROOT / "tools" / "best5")); import n_rs as NRS_
        NRS_.CFG[:] = [a.n_rs, a.n_rs_batch, a.n_blk]
        RR_ = NRS_.reroute_rs
    o_, rcd_, r_ = cur
    rec["est_in"] = float(r_["score"])
    nn_t = len(S.SD.net_names)
    tid = np.asarray(S.tid, np.int64)
    nr = len(tid)
    ok = tid >= 0
    tsafe = np.maximum(tid, 0)
    wts = np.asarray(weights(o_, rcd_), np.float64)
    kp = np.asarray(S.kpn, np.float64) if S.kpn is not None else np.zeros(nn_t)
    ctot = np.asarray(r_["net_ctot"], np.float64)
    rd = np.asarray(r_["net_rd_kohm"], np.float64)
    cl = np.clip(0.5 * ctot, 1.0, 1e5)
    tw_r = np.where(ok, a.rs_tw * wts[tsafe], 0.0)
    pw_r = np.where(ok, a.rs_pw * float(S.w3) * kp[tsafe], 0.0)
    cl_r = np.where(ok, cl[tsafe], 1.0)
    rd_r = np.where(ok, rd[tsafe], 0.05)
    cc_r = np.where(ok & (kp[tsafe] > 0), kp[tsafe] * ctot[tsafe], 0.0)
    pins = PRE["esc_pins"]
    sb_r = None; ss_r = None; soft = 0.0
    if a.rs_sb >= 0:                                    # slack budget: a net off the violating paths may get at most
        sl = None                                       # (slack - margin) x rs_sb ns slower (lumped proxy); priced nets free
        if H["inc"] and hasattr(K.L, "w3_inc_backward"):
            import ctypes
            stb = np.zeros(2); K.L.w3_inc_backward(stb.ctypes.data_as(ctypes.c_void_p))
            sl = np.array(K.V["net_slack"], np.float64); rec["backward_s"] = float(stb[0])
        elif "net_slack" in o_:
            sl = np.asarray(o_["net_slack"], np.float64)
        if sl is not None:
            slr = np.where(ok, sl[tsafe], 9.0)
            if a.rs_sbmode == "wns":                     # hard: never below the current WNS (- margin); soft: TNS price
                wns0 = float(r_["wns"])
                sb_r = np.where(tw_r > 0, 1e30, np.maximum(slr - wns0 - a.rs_margin, 0.0) * a.rs_sb)
                ss_r = np.maximum(slr, 0.0)
                soft = a.rs_soft * abs(float(S.w2)) / float(S.nend)
                rec["soft"] = soft
            else:
                sb_r = np.where(tw_r > 0, 1e30, np.maximum(slr - a.rs_margin, 0.0) * a.rs_sb)
            rec["sb_nets_zero"] = int((sb_r == 0).sum())
        else:
            rec["sb"] = "no slack available: off"
    rec["prep_s"] = time.perf_counter() - t0
    state = {"cur": cur}
    touched = set()

    def check(undo, tag, crec):
        """exact re-time; keep, else halve (first half by rank), else undo. returns number of nets kept."""
        cur_ = state["cur"]
        s0 = float(cur_[2]["score"])
        o1, rc1, r1 = retime(tag)
        crec["retimes"] += 1
        T_ = ("wns", "tns", "of", "pw", "score")
        crec.setdefault("tries", []).append({"tag": tag, "nets": len(undo), "d": {k: float(r1[k] - cur_[2][k]) for k in T_}})
        if r1["score"] < s0 - 1e-9:
            if H["inc"]:
                K.commit()
            state["cur"] = (o1, rc1, r1)
            touched.update(undo)
            return len(undo), "kept"
        S.revert(undo) if not a.rs_halve or len(undo) < 2 else None
        if H["inc"]:
            K.reject()
        if not a.rs_halve or len(undo) < 2:
            return 0, "undone"
        keys = sorted(undo, key=lambda i: crec["rank"].get(i, 1 << 60))
        for lev in range(int(a.rs_halve)):
            h = len(keys) // 2
            if h < 1:
                break
            S.revert({i: undo[i] for i in keys[h:]})
            keys = keys[:h]
            o2, rc2, r2 = retime(tag + "_half%d" % (lev + 1))
            crec["retimes"] += 1
            crec["tries"].append({"tag": tag + "_half%d" % (lev + 1), "nets": h, "d": {k: float(r2[k] - cur_[2][k]) for k in T_}})
            if r2["score"] < s0 - 1e-9:
                if H["inc"]:
                    K.commit()
                state["cur"] = (o2, rc2, r2)
                touched.update(keys)
                return h, "halved%d" % (lev + 1)
            if H["inc"]:
                K.reject()
        S.revert({i: undo[i] for i in keys})
        return 0, "undone"

    c_dead = False
    for cls in [c.strip() for c in a.rs_order.split(",") if c.strip()]:
        if cls == "c" and c_dead and a.rs_cstop:           # an earlier c pass kept nothing: the next would pick the same runs
            rec["classes"].append({"cls": "c", "skipped": "previous c pass kept nothing"})
            continue
        if not DLG("rs_pass"):                             # merge6 (b): deadline -> no further class pass
            rec["classes"].append({"cls": cls, "skipped": "deadline"})
            break
        tc = time.perf_counter()
        crec = {"cls": cls, "batches": [], "retimes": 0, "est_in": float(state["cur"][2]["score"]), "stats": {}}
        if cls in ("a", "b"):
            if cls == "a":
                cand = np.nonzero(tw_r > 0)[0]
                nets = cand[np.argsort(-tw_r[cand], kind="stable")][:a.rs_a_nets]
            else:
                cand = np.nonzero((cc_r > 0) & (tw_r <= 0))[0]
                nets = cand[np.argsort(-cc_r[cand], kind="stable")][:a.rs_b_nets]
            crec["nets"] = int(len(nets))
            crec["rank"] = {int(i): k for k, i in enumerate(nets.tolist())}
            miss = 0
            for b in range(a.rs_batches):
                tb = time.perf_counter(); st = {}
                undo = RR_(S, pins, nets.tolist(), 1, tw_r, cl_r, rd_r, pw_r, a.rs_W, a.rs_hmax,
                                         a.rs_bruns, a.rs_minlen, st, skip=b * a.rs_bruns, sb_r=sb_r, ss_r=ss_r, soft=soft)
                ts = time.perf_counter() - tb
                if not undo:
                    crec["batches"].append({"b": b, "changed": 0, "search_s": ts, "stats": st})
                    if st.get("cand_runs", 0) <= b * a.rs_bruns:
                        break
                    miss += 1
                    if miss >= a.rs_stop:
                        break
                    continue
                kept, how = check(undo, "rs_" + cls, crec)
                crec["batches"].append({"b": b, "changed": len(undo), "kept": kept, "how": how, "search_s": ts,
                                        "s": time.perf_counter() - tb, "est": float(state["cur"][2]["score"]), "stats": st})
                miss = 0 if kept else miss + 1
                if miss >= a.rs_stop:
                    break
        elif cls == "c":
            if "npins_r" not in state:
                npk = PRE["ovf_npins_key"]
                state["npins_r"] = np.array([npk.get(_k(nm), 0) for nm in S.nets.names], np.int64)
            npins_r = state["npins_r"]
            o0 = state["cur"][0]
            if a.rs_c_guard > -1e8 and "net_slack" in o0:
                sl = np.asarray(o0["net_slack"], np.float64)
                slack_r = np.where(ok, sl[tsafe], 9.0)
            else:
                slack_r = np.full(nr, 9.0)
                crec["guard"] = "none" if a.rs_c_guard <= -1e8 else "no net_slack in the state: none"
            tb = time.perf_counter()
            if a.rs_c_twguard:                              # nets on violating paths are not chosen
                slack_r = np.where(tw_r > 0, -9.0, slack_r)
            r5_ = None
            if getattr(a, "n_sel", 0):                      # speed9: same chosen nets, mask and pairs on threads
                sys.path.append(str(ROOT / "tools" / "best5")); import n_sel as NSEL_
                r5_ = NSEL_.select_c(S, S.la, slack_r, npins_r, a.rs_c_cap, 0, int(os.environ.get("MID5_THREADS", "8")))
            if r5_ is None:
                r5_ = L5_.select_c(S, S.la, slack_r, npins_r, a.rs_c_cap, 0) if (L5_ is not None and os.environ.get("L5_SEL", "1") == "1") else None   # late5: C++
            if r5_ is not None and os.environ.get("L5_CHECK"):    # late5 diagnostics: compare with the original selection
                ref_ = O2.select_c_fast(S, S.la, slack_r, npins_r, a.rs_c_cap, 0)
                print("[l5check] select_c equal", ref_[0] == r5_[0], len(ref_[0]), len(r5_[0]), ref_[1], r5_[1], flush=True)
            chosen, info = r5_ if r5_ is not None else O2.select_c_fast(S, S.la, slack_r, npins_r, a.rs_c_cap, 0)
            crec["select"] = {k: v for k, v in info.items() if not isinstance(v, (list, dict))}
            if "n_sel_t" in info: crec["n_sel_t"] = info["n_sel_t"]
            crec["select_s"] = time.perf_counter() - tb
            crec["rank"] = {int(i): k for k, i in enumerate(chosen)}
            if chosen:
                st = {}
                nrun = max(a.rs_c_runs, int(a.rs_c_runs_frac * len(chosen)))
                crec["runs"] = nrun
                if getattr(a, "n_rs", -1) == 5 and getattr(a, "n_rs_big_exact", 0):   # speed9, fallback after the official bsg v result: a selection larger than the base budget runs on the task graph (serial loop's result)
                    NRS_.CFG[0] = 4 if nrun > a.rs_c_runs else 5
                undo = RR_(S, pins, chosen, 0, tw_r, cl_r, rd_r, pw_r, a.rs_W, a.rs_hmax, nrun, 1, st, sb_r=sb_r, ss_r=ss_r, soft=soft)
                ts = time.perf_counter() - tb
                if undo:
                    kept, how = check(undo, "rs_c", crec)
                else:
                    kept, how = 0, "nothing"
                crec["batches"].append({"b": 0, "changed": len(undo), "kept": kept, "how": how, "search_s": ts,
                                        "est": float(state["cur"][2]["score"]), "stats": st})
                c_dead = not kept
            else:
                c_dead = True
        elif cls == "p":
            nets = sorted(S.new) if a.rs_p_all else sorted(touched)
            crec["nets"] = len(nets)
            crec["rank"] = {int(i): k for k, i in enumerate(nets)}
            tb = time.perf_counter(); st = {}
            undo = RR_(S, pins, nets, 2, tw_r, cl_r, rd_r, pw_r, a.rs_W, 0, 0, 1, st)
            ts = time.perf_counter() - tb
            if undo:
                kept, how = check(undo, "rs_p", crec)
            else:
                kept, how = 0, "nothing"
            crec["batches"].append({"b": 0, "changed": len(undo), "kept": kept, "how": how, "search_s": ts,
                                    "est": float(state["cur"][2]["score"]), "stats": st})
        crec.pop("rank", None)
        crec["est_out"] = float(state["cur"][2]["score"])
        crec["gain"] = crec["est_in"] - crec["est_out"]
        crec["s"] = time.perf_counter() - tc
        crec["gain_per_s"] = crec["gain"] / max(crec["s"], 1e-9)
        rec["classes"].append(crec)
        DLE("rs_pass")
        if cls == "c" and getattr(a, "n_rs_stop", 0.0) > 0 and crec["gain"] < a.n_rs_stop:   # speed9: this pass paid too little for another one
            c_dead = True
    if getattr(a, "n_rs", -1) >= 0:
        NRS_.free()
    cur = state["cur"]
    rec["s"] = time.perf_counter() - t0
    rec["est"] = {k: float(cur[2][k]) for k in ("wns", "tns", "of", "pw", "score")}
    rec["gain"] = rec["est_in"] - rec["est"]["score"]
    log.append(rec)
    print("[reshape] " + json.dumps(rec), flush=True)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hot", type=int, default=1290); ap.add_argument("--esc_passes", type=int, default=2); ap.add_argument("--escape", choices=("auto", "on", "off"), default="auto"); ap.add_argument("--esc_thresh", type=float, default=0.5); ap.add_argument("--pw_nets", type=int, default=6000)
    ap.add_argument("--design", required=True); ap.add_argument("--suite", default="visible")
    ap.add_argument("--route", required=True); ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--gate", type=float, default=0.0, help="L>0: pass-level gate between lam 0 and lam L (arm c)")
    ap.add_argument("--out", required=True, help="prefix; writes <out>.<tag>.route")
    ap.add_argument("--snaps", default="3:2,5:4", help="total_crit_rounds:label,... ; 1 chain round + k extra => 1+k")
    ap.add_argument("--power", default="1", help="1: for every snapshot also write the b variant (+ power round, reverted afterwards)")
    ap.add_argument("--tag", default="x"); ap.add_argument("--only_b", action="store_true")
    ap.add_argument("--json"); ap.add_argument("--route_wall", type=float, default=0.0); ap.add_argument("--cache")
    ap.add_argument("--max_nets", type=int, default=6000); ap.add_argument("--npaths", type=int, default=5000)
    ap.add_argument("--round_nets", type=int, default=6000); ap.add_argument("--tw_scale", type=float, default=2.0)
    ap.add_argument("--crit_n", type=int, default=60000); ap.add_argument("--dw", type=float, default=1.0); ap.add_argument("--pw", type=float, default=None)
    ap.add_argument("--overlap", type=int, default=1); ap.add_argument("--build", choices=("b1", "b2"), default="b2")
    ap.add_argument("--promote_impl", choices=("py", "full"), default="py", help="full: tools/fast3x/promote_full.py round loops")
    ap.add_argument("--min_gain_frac", type=float, default=0.0, help="stop the crit rounds when a kept round gains < F x total gain so far")
    ap.add_argument("--check", action="store_true", help="test mode: demand mirrored with the original apply/commit_net, timer array vs Python")
    ap.add_argument("--no_item", type=int, action="append", default=[], help="turn item N (1-6) off")
    ap.add_argument("--of_rule", choices=("floor", "official"), default="floor",
                    help="official: overflow total with the evaluator's blocked-cell rule (tools/offix/of_official.py) in the accept test and the blocked-share test")
    # OVF3 stage flags (off = chain_k.py). Defaults = the delivered setting A (an internal note): `--ovf auto` alone
    ap.add_argument("--ovf", choices=("off", "auto"), default="off")
    ap.add_argument("--ovf_eps", type=float, default=1e-3, help="pre-test: skip when w4 x overflowed-cell cost < eps")
    ap.add_argument("--ovf_hot", type=int, default=1290); ap.add_argument("--ovf_passes", type=int, default=2)
    ap.add_argument("--ovf_esc", type=int, default=1); ap.add_argument("--ovf_rr", type=int, default=1)
    ap.add_argument("--ovf_W", type=int, default=14); ap.add_argument("--ovf_cap", type=float, default=0.25)
    ap.add_argument("--ovf_maxnets", type=int, default=0); ap.add_argument("--ovf_pin", choices=("auto", "0", "1"), default="0")
    ap.add_argument("--ovf_uf", type=int, default=1, help="1: interval union-find (same answers), 0: repair3d's")
    ap.add_argument("--ovf_rr_hot", type=int, default=0, help="reroute only runs crossing the top-N overflowed cells by cost (0 = all)")
    ap.add_argument("--ovf_rank", type=int, default=1, help="1: ovf_lib.reroute_rank (runs by own overflow cost, budget); 0: repair3d_big.repair_fast order")
    ap.add_argument("--ovf_max_runs", type=int, default=1000); ap.add_argument("--ovf_hmax", type=int, default=40)
    ap.add_argument("--ovf_gate", choices=("per", "one"), default="per", help="one: escape + reroute share one re-time (escape alone re-checked only on rejection)")
    ap.add_argument("--ovf_astar", choices=("c", "win"), default="win", help="win: tools/ovf3/astar3d_win.cpp (same paths)")
    for k_ in ("read", "write", "dem", "asm", "inc", "crit", "graph", "all", "pre"):
        ap.add_argument("--h_" + k_, type=int, default=0)
    ap.add_argument("--h_check", action="store_true")
    # b. chain_w3 stage flags (defaults = chain_g's stage)
    ap.add_argument("--ovf_impl", choices=("py", "cpp"), default="py", help="cpp: reroute per-run loop in C++ (tools/wave3/ovf2/rr_loop.cc)")
    ap.add_argument("--ovf_timer", choices=("full", "inc"), default="full", help="inc: incremental exact timer in the stage accept tests")
    ap.add_argument("--ovf_skip", type=float, default=0.0, help="skip the stage when skip_q[--ovf_skip_key] < X (0 = off)")
    ap.add_argument("--dl_ovf_chunk", type=int, default=0, help="deadline mode: check between overflow escape/reroute batches (0 = unchanged)")
    ap.add_argument("--dl_order", choices=("off", "term"), default="off", help="term: when WNS and TNS are met after the first pass, skip crit and power")
    ap.add_argument("--ovf_skip_key", default="bound_ug")
    ap.add_argument("--ovf_rr_passes", type=int, default=1)
    ap.add_argument("--ovf_order", choices=("er", "re"), default="er")
    ap.add_argument("--ovf_esc_pre", choices=("run", "skip"), default="run", help="skip: no stage escape when the pre-pass escape ran")
    ap.add_argument("--ovf_rr_resel", type=int, default=1, help="0: reroute passes after the first reuse its chosen nets (no new selection)")
    ap.add_argument("--ovf_big_nets", type=int, default=0, help="selections of more than N nets use --ovf_hmax_big (0 = off)")
    ap.add_argument("--ovf_hmax_big", type=int, default=40)
    ap.add_argument("--ovf_esc_check", type=int, default=0, help="1: escaped nets that fail net_valid_fast are reverted")
    ap.add_argument("--ovf_big_runs", type=int, default=0, help="run budget for a large selection (0 = --ovf_max_runs)")
    ap.add_argument("--ovf_big_passes", type=int, default=0, help="passes for a large selection (0 = --ovf_rr_passes)")
    # c./d./e. chain_i flags (defaults = chain_h)
    ap.add_argument("--post_crit", type=int, default=0, help="N>0: up to N gated crit rounds after the final power round")
    ap.add_argument("--pw_batch", type=int, default=0, help="N>0: power round in batches of N nets, each gated, bisection on rejection")
    ap.add_argument("--pw_bisect_depth", type=int, default=3); ap.add_argument("--pw_bisect_min", type=int, default=64)
    ap.add_argument("--pw2_nets", type=int, default=0, help="N>0: second power round on the next N power nets")
    ap.add_argument("--pw_skip", type=float, default=None, help="skip the power round when its power-gain bound < T")
    ap.add_argument("--pw_repair", action="store_true", help="unbatched power round: repair step on rejection (chain_p.py)")
    ap.add_argument("--ovfround", type=int, default=0, help="N>0: overflow-ranked kernel round on the top N nets after the power round")
    ap.add_argument("--ovfround_slack", type=float, default=0.05, help="only nets with timer slack >= this (ns); -1e9 = no guard")
    ap.add_argument("--ovfround_open", type=int, default=0, help="1: rank on open cells only (capacity > 1e-3)")
    ap.add_argument("--ovfround_skip", type=float, default=0.0, help="skip the kernel round when w4 x own overflow cost of its nets < T")
    # big mode (chain_gb items; read from sys.argv at import as well)
    ap.add_argument("--big", choices=("off", "on", "auto"), default="off")
    ap.add_argument("--no_big", type=int, action="append", default=[], help="turn chain_gb item N off (read from sys.argv at import)")
    ap.add_argument("--ovf_pins_check", action="store_true"); ap.add_argument("--npk_table", action="store_true")
    ap.add_argument("--act", default=None, help="chain_gb L2: activity file (default: the glob of chain_g)")
    ap.add_argument("--gnn_family", action="append", default=None, help="chain_gb L3: DESIGN=FAMILY of the held-out GNN fold")
    ap.add_argument("--pw2_big", type=int, default=0, help="big mode only: --pw2_nets N (second power round) when --pw2_nets is 0")
    # chain_j a. micro engine (tools/engine2/micro2.py flags, same defaults)
    ap.add_argument("--engine", choices=("rounds", "micro"), default="rounds")
    ap.add_argument("--stages", default="C5,P"); ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--pB", type=int, default=-1); ap.add_argument("--R", type=int, default=1)
    ap.add_argument("--Rkept", type=int, default=0); ap.add_argument("--lazy", type=int, default=0)
    ap.add_argument("--bisect_min", type=int, default=64); ap.add_argument("--bisect_depth", type=int, default=3)
    ap.add_argument("--bisect_reuse", type=int, default=0); ap.add_argument("--skip_nochange", type=int, default=0)
    ap.add_argument("--fresh_crit", type=int, default=1); ap.add_argument("--e_inc", type=int, default=1)
    ap.add_argument("--qB", type=int, default=2000); ap.add_argument("--qk", type=int, default=4)
    ap.add_argument("--q_margin", type=float, default=0.05); ap.add_argument("--q_nets", type=int, default=0)
    ap.add_argument("--q_fl", type=int, default=1)
    ap.add_argument("--e_dp", choices=("py", "full"), default="py")
    ap.add_argument("--e_ofc", type=int, default=0); ap.add_argument("--e_ofc_check", type=int, default=0)
    ap.add_argument("--e_snap", type=int, default=0); ap.add_argument("--e_capply", type=int, default=0)
    # reshape stage (off = chain_h)
    ap.add_argument("--rs", choices=("off", "on"), default="off")
    ap.add_argument("--rs_order", default="c,a,b,p", help="classes in order: a timing nets, b power nets, c over-full runs, p prune")
    ap.add_argument("--rs_tw", type=float, default=1.0, help="multiplier on the round weights in the price")
    ap.add_argument("--rs_pw", type=float, default=1.0, help="multiplier on w3 x switching coefficient in the price")
    ap.add_argument("--rs_a_nets", type=int, default=6000); ap.add_argument("--rs_b_nets", type=int, default=6000)
    ap.add_argument("--rs_bruns", type=int, default=400, help="candidate runs per batch (classes a, b)")
    ap.add_argument("--rs_batches", type=int, default=4, help="max batches per class a / b")
    ap.add_argument("--rs_c_runs", type=int, default=1000); ap.add_argument("--rs_c_guard", type=float, default=-1e9,
                    help="class c: nets with slack below this are not chosen (needs net_slack; -1e9 = no guard)")
    ap.add_argument("--rs_c_cap", type=float, default=0.25)
    ap.add_argument("--rs_c_runs_frac", type=float, default=0.0, help="class c runs = max(rs_c_runs, frac x chosen nets)")
    ap.add_argument("--rs_cstop", type=int, default=1, help="1: skip later c passes once one kept nothing")
    ap.add_argument("--rs_minlen", type=int, default=2); ap.add_argument("--rs_W", type=int, default=14)
    ap.add_argument("--rs_hmax", type=int, default=40)
    ap.add_argument("--rs_halve", type=int, default=1, help="N: a rejected batch is retried with its first half, quarter, ... (N levels)")
    ap.add_argument("--rs_stop", type=int, default=2, help="stop a class after N batches in a row with nothing kept")
    ap.add_argument("--rs_c_twguard", type=int, default=0, help="1: class c skips nets with timing weight > 0")
    ap.add_argument("--rs_sb", type=float, default=-1.0, help=">= 0: slack budget scale (needs the backward pass); < 0 off")
    ap.add_argument("--rs_margin", type=float, default=0.0, help="slack budget margin (ns)")
    ap.add_argument("--rs_sbmode", choices=("slack", "wns"), default="slack", help="wns: budget = slack - WNS - margin, + soft TNS price")
    ap.add_argument("--rs_soft", type=float, default=1.0, help="soft TNS price = this x |w2| / nend per ns beyond positive slack")
    ap.add_argument("--rs_p_all", type=int, default=0, help="1: prune every changed net (default: the nets this stage changed)")
    # chain_j switches (all off by default)
    ap.add_argument("--micro_big", choices=("rounds", "micro"), default="rounds", help="big mode: engine used when --engine micro")
    ap.add_argument("--post_crit_big", type=int, default=0, help="big mode only: --post_crit N (chain_gb2 p2yc)")
    ap.add_argument("--big8", type=int, default=0, help="big8: tools/big8/big8_lib.py pieces (an internal note); 0 = chain_m")
    ap.add_argument("--big8_pw_batch", type=int, default=0, help="big8, big mode: power-round batch size (0 = --pw_batch)")
    ap.add_argument("--big5", type=int, default=0, help="big5: large-case pieces (tools/big5/big5_lib.py); 0 = chain_j")
    ap.add_argument("--big5_act", choices=("file", "child", "thread"), default="file", help="big5, big mode: prepared activity file | pre5 child process | pre5 kernel on a thread")
    ap.add_argument("--big5_act_threads", type=int, default=0, help="big5: threads of the activity child (0 = OMP_NUM_THREADS / 2)")
    ap.add_argument("--big5_mem", type=int, default=0, help="big5: 1 = memory / CPU marks by stage (json big5); 2 = + largest holders")
    ap.add_argument("--route_bin", type=int, default=0, help="merge6 (a): 1 = --route is the router's binary route R5B1 (tools/router5/r5bin.py)")
    ap.add_argument("--deadline_s", type=float, default=None, help="merge6 (b): stop starting stages / rounds / batches that would end after X s; write")
    ap.add_argument("--deadline_est", default=None, help="merge6 (b): json {kind: seconds} for the first unit of each kind")
    ap.add_argument("--rs_at", choices=("power", "end"), default="power", help="reshape stage after the power round / after the after-stages")
    for k_ in ("ofc", "esc"):
        ap.add_argument("--s4_" + k_, type=int, default=0)
    ap.add_argument("--s4_check", action="store_true")
    ap.add_argument("--load8", type=int, default=0, help="load8: big-design load on threads, without torch (tools/best11/load8_lib.py)")
    ap.add_argument("--load8_stop", type=int, default=0, help="load8 test: exit after the preload, the first timer and the activity")
    ap.add_argument("--pre5", type=int, default=0, help="pre5: cold parallel preload (tools/pre5/pre5_load.py)")
    ap.add_argument("--pre5_threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--pre5_act", choices=("child", "thread", "file"), default="child", help="activity: child process | thread | the prepared .act (identity tests only)")
    ap.add_argument("--pre5_act_threads", type=int, default=0, help="threads of the activity child (0 = max(2, pre5_threads / 2))")
    ap.add_argument("--pre5_gnn", choices=("np", "torch", "check"), default="np", help="prior forward: C++ (np), torch, or both compared (check)")
    ap.add_argument("--pre5_preimp", type=int, default=0, help="1: post-route imports done in the preload")
    ap.add_argument("--pre5_stop", type=int, default=0, help="test: exit after the preload (and the activity) with the timers")
    ap.add_argument("--pre5_pcmap", type=int, default=1, help="overflow pin-count map from the .net arrays (numpy, exact)")
    ap.add_argument("--pre5_check", type=int, default=0, help="test: replaced preload pieces compared with the originals")
    ap.add_argument("--mid5", type=int, default=0, help="1: tools/mid5 drivers from the route read to the end of the overflow stage (same route)")
    ap.add_argument("--mid5_cpu", type=int, default=0, help="1: process CPU seconds per region stage in the json (mid5_cpu)")
    ap.add_argument("--mid5_rr", type=int, default=0, help="with --mid5 1: the overflow stage's reroute loop on threads (tools/mid5/mid5_rr.cc)")
    ap.add_argument("--mid5_batch", type=int, default=32, help="--mid5_rr: candidate runs per speculative batch")
    ap.add_argument("--late5", type=int, default=0, help="late5: C++ drivers of the late region (tools/late5; off = chain_j)")
    ap.add_argument("--par7", type=int, default=0, help="par7: 1 = layer DPs on threads, level schedule (same route); 0 = chain_k")
    ap.add_argument("--par7_pt", type=int, default=4, help="par7: threads per first-pass candidate DP (both run at once)")
    ap.add_argument("--timer5", type=int, default=0, help="1: the timer of tools/timer5 (csta5.py / libcsta5), bit-identical")
    ap.add_argument("--n_rr", type=int, default=-1, help="speed9: overflow stage's reroute in tools/best5/n_rr.cc: 4 task graph (same route), 1-3 snapshot batches (route changes), 0 = mid5's speculative loop; -1 off")
    ap.add_argument("--n_rr_batch", type=int, default=32, help="speed9: runs per snapshot batch (--n_rr 1-3)")
    ap.add_argument("--n_rs", type=int, default=-1, help="speed9: reshape's loop in tools/best5/n_rs.cc: 4 task graph (same route), 1-3 snapshot batches, 0 = late5's pipeline; -1 off")
    ap.add_argument("--n_rs_batch", type=int, default=32, help="speed9: runs per snapshot batch (--n_rs 1-3)")
    ap.add_argument("--n_blk", type=int, default=2, help="speed9: task graph block side = 2**n_blk cells")
    ap.add_argument("--n_ofc", type=int, default=0, help="speed9: cached overflow cost with C++ compare and sums on threads (same bits)")
    ap.add_argument("--n_sel", type=int, default=0, help="speed9: reshape's selection, mask and pairs on threads (same nets)")
    ap.add_argument("--n_norm", type=int, default=0, help="speed9: normaliser of the re-times on threads (same rows)")
    ap.add_argument("--n_rs_stop", type=float, default=0.0, help="speed9: no further reshape c pass after a pass with an estimate gain below this")
    ap.add_argument("--n_pre", type=int, default=0, help="speed9: the overflow stage's pretest on the overflowed cells only")
    ap.add_argument("--n_b2", type=int, default=0, help="speed9: builder's per-net loop in C++ on threads (same P)")
    ap.add_argument("--n_seg", type=int, default=0, help="speed9: seg_from_route on threads (same array)")
    ap.add_argument("--n_rs_big_exact", type=int, default=0, help="speed9: --n_rs 5, but task graph on a pass whose run budget exceeds --rs_c_runs (bsg)")
    ap.add_argument("--dl_chunks", type=int, default=0, help="dlchunk: with --deadline_s, up to K checked timing + power chunks before the overflow stage; 0 = chain_n")
    ap.add_argument("--dl_chunk_nets", type=int, default=4000, help="dlchunk: nets per chunk")
    ap.add_argument("--dl_chunk_tw", type=float, default=2.0, help="dlchunk: timing weight scale of the chunks (the harness's value; CHOSEN's --tw_scale is for the rounds)")
    ap.add_argument("--dgr", type=int, default=0, help="dgr1x: K steps of the timing descent as a stage (tools/best7/dgr_stage.py); 0 = chain_p's route")
    ap.add_argument("--d2", type=int, default=-1, help="dgr2d: K of the step over 2-D candidates (tools/dgr2d/d2_stage.py) in the place of the descent; -1: chain_r's route"); ap.add_argument("--d2_tm", type=float, default=1.0); ap.add_argument("--d2_pw", type=float, default=1.0); ap.add_argument("--d2_uvia", type=float, default=1.0); ap.add_argument("--d2_rounds", type=int, default=2); ap.add_argument("--d2_flags", default="")
    ap.add_argument("--dgr_at", choices=("fp", "ov", "end"), default="fp", help="dgr1x: after the first pass | after the overflow stage | before the final write")
    ap.add_argument("--dgr_tau", type=float, default=0.002); ap.add_argument("--dgr_nets", type=int, default=0, help="dgr1x: top N nets by the cut's score (0 = all)")
    ap.add_argument("--dgr_hold", type=int, default=1000, help="dgr1x: nets with more route rows than this are held (their RC tree changes shape when layers move: E2ET)")
    ap.add_argument("--dgr_lr", type=float, default=0.1); ap.add_argument("--dgr_lrk", type=float, default=10.0, help="dgr1x: lr = max(dgr_lr, dgr_lrk / K, dgr_lrc / K^2)"); ap.add_argument("--dgr_lrc", type=float, default=0.0)
    ap.add_argument("--dgr_t0", type=float, default=1.0); ap.add_argument("--dgr_t1", type=float, default=0.05); ap.add_argument("--dgr_init", type=float, default=4.0)
    ap.add_argument("--dgr_stop", type=float, default=0.0); ap.add_argument("--dgr_pat", type=int, default=5)
    ap.add_argument("--dgr_dl", type=float, default=0.0, help="dgr1x: relative input move an arc ignores (0: exact values)"); ap.add_argument("--dgr_dat", type=float, default=0.0, help="dgr1x: arrival move in ns not passed on")
    ap.add_argument("--dgr_wt", type=float, default=1.0); ap.add_argument("--dgr_wo", type=float, default=1.0); ap.add_argument("--dgr_wp", type=float, default=1.0)
    ap.add_argument("--dgr_split", type=int, default=3, help="dgr1x: halvings of a rejected change")
    ap.add_argument("--dgr_rounds", type=int, default=1, help="dgr1x: repeat the stage while it fits the deadline and gains"); ap.add_argument("--dgr_round_min", type=float, default=0.01)
    ap.add_argument("--dgr_defer", type=int, default=0, help="dgr1x: 1 = at a state without timing violation the overflow stage goes before the descent")
    ap.add_argument("--dgr_skip_fp", type=int, default=0, help="dgr1x: 1 = no first-pass layer DP: the descent starts from the router's route; 2 = the same unless a net with more than --dgr_hold rows lies on a violating path; 3 = also the first pass whenever it and one round fit the deadline (needs --lam 0 --gate 0)")
    ap.add_argument("--dgr_gnn", default="", help="gnnflow: weights file (tools/best8/export_w.py): the network gives the descent's start logits; empty: chain_q's route")
    ap.add_argument("--dgr_gnn_k", type=int, default=5, help="gnnflow: steps after the network (0: the network's argmax); on rejection the plain --dgr K arm from the same state, as far as the deadline allows")
    ap.add_argument("--dgr_gnn_lr", type=float, default=0.4); ap.add_argument("--dgr_gnn_t1", type=float, default=1.0); ap.add_argument("--dgr_gnn_tau", type=float, default=0.01, help="gnnflow: tau of the gradient input (the models were trained at 0.01)")
    ap.add_argument("--dgr_gnn_rounds", type=int, default=1, help="gnnflow: rounds of the stage that start from the network; later rounds are plain")
    ap.add_argument("--dgr_gnn_sm", type=float, default=1.0, help="gnnflow: multiplier of the gradient input's scale (the median over a cut of the nets is larger than over all nets, which the models saw)"); ap.add_argument("--dgr_gnn_x0", type=int, default=0, help="gnnflow: 1 = the background demand column is 0 (in training every net was free: background near 0 everywhere)")
    ap.add_argument("--dgr_stop_after", type=int, default=0, help="dgr1x test: 1 = no later stage, straight to the write")
    ap.add_argument("--t_rr", type=int, default=0, help="speed10: 1 = overflow re-route through tools/best10/t_rr (same route with the next four at 0)")
    ap.add_argument("--t_rect_h", type=int, default=0, help="speed10: a run with half window above this searches a rectangle"); ap.add_argument("--t_rect_w", type=int, default=14, help="speed10: half width across the run")
    ap.add_argument("--t_long_h", type=int, default=0, help="speed10: runs with half window above this go last"); ap.add_argument("--t_long_n", type=int, default=0, help="speed10: only the first N of them (0 = all)")
    ap.add_argument("--t_popcap", type=int, default=0, help="speed10: an A* with more pops gives up")
    ap.add_argument("--t_rr_area", type=int, default=0, help="speed10: work budget of the overflow re-route: cells of the runs' windows, rank order")
    ap.add_argument("--t_runs", type=int, default=0, help="speed10: run budget of the overflow re-route in place of --ovf_max_runs (not for a large selection)")
    ap.add_argument("--t_csta", type=int, default=0, help="speed10: 1 = tools/best10/libt_csta: csta_crit's entries on threads (same values)")
    ap.add_argument("--t_first", type=int, default=0, help="speed10: 1 = the first analysis without the required-time pass and without copies (nothing reads them)")
    ap.add_argument("--t_np_nb", type=int, default=0, help="speed10: 1 = the first candidate's analysis without the required-time pass; done once the pick is known")
    ap.add_argument("--t_ofc0", type=int, default=0, help="speed10: 1 = the overflow cache's first call on threads per layer (same bits)")
    ap.add_argument("--t_esc_wait", type=int, default=0, help="speed10: N > 0: above N route rows the first timer waits for the escape decision and runs once, on the final route (same route)")
    ap.add_argument("--t_warm", type=int, default=0, help="speed10: 1 = one forward pass on the pin-only RC after the preload, before the route read (same values afterwards: the first real pass starts from scratch)")
    ap.add_argument("--t_no_post", type=int, default=0, help="speed10: 1 = no post_crit round after the power round")
    ap.add_argument("--t_pw_stop", type=float, default=0.0, help="speed10: power round: rest of the critical-net list skipped after a batch with an estimate gain below this")
    ap.add_argument("--t_pw_rej", type=int, default=0, help="speed10: power round: rest of the critical-net list skipped after a batch that kept nothing (bisection included)")
    ap.add_argument("--t_pw_grow", type=int, default=0, help="speed10: 1 = power batches double while they are kept (fewer re-times), start again at --pw_batch after a rejected one and at the power list")
    ap.add_argument("--t_pw_stop2", type=float, default=0.0, help="speed10: power round: rest of the power-net list skipped after a batch with an estimate gain below this")
    ap.add_argument("--t_gnn_bg", type=int, default=0, help="speed10: 1 = GNN forward on a thread next to the first candidate's DP, assembly and timer (same logits, same choices)")
    ap.add_argument("--t_same", type=int, default=0, help="speed10: 1 = nets whose new rows equal their current rows are not reported by a promote round")
    ap.add_argument("--t_pre", type=int, default=0, help="speed10: 1 = name tables between the preload and the route read")
    ap.add_argument("--t_fp2", type=int, default=0, help="speed10: 1 = the candidates' segment and base arrays on threads (same arrays)")
    ap.add_argument("--t_esc_p2", type=int, default=0, help="speed10: N > 0: the overflow stage's second escape pass only if the first changed at least N nets")
    ap.add_argument("--t_small", type=int, default=0, help="speed10: N > 0: above N route rows the route-changing switches are off")
    ap.add_argument("--t_crit_stop", type=float, default=0.0, help="speed10: X > 0: no further crit round after two kept rounds in a row, each with an estimate gain below X")
    ap.add_argument("--t_gc", type=int, default=0, help="speed10: 1 = no cyclic GC after the preload (nothing a run makes needs it; the process ends with os._exit)")
    ap.add_argument("--t_fp3", type=int, default=0, help="speed10: 1 = the assembly's kept rows on threads, the net-name lookup of the segment arrays once (same arrays)")
    ap.add_argument("--t_prof", type=int, default=0, help="speed10: 1 = first-pass marks in the json (walls.t_marks)")
    ap.add_argument("--n_same", type=int, default=0, help="speed9: nets with unchanged rows left out of the re-times (same timer input, same route)")
    a = ap.parse_args()
    if BIGM:                                               # speed9: the switches were tuned and scored on the small cases + bsg only: chain_m's path on MemPool group / cluster
        a.n_rr = a.n_rs = -1
        a.n_ofc = a.n_sel = a.n_norm = a.n_pre = a.n_b2 = a.n_seg = a.n_same = a.par7 = a.timer5 = 0
        a.n_rs_stop = 0.0
        for k_ in [k2_ for k2_ in vars(a) if k2_.startswith("t_") and k2_ != "t_prof"]:   # speed10
            setattr(a, k_, 0)
        for k_, f_ in (("ovf_max_runs", int), ("ovfround_skip", float)):       # the first value on the command line: CHOSEN.flags', not the setting's
            v_ = _argv("--" + k_)
            setattr(a, k_, f_(v_) if v_ is not None else ap.get_default(k_))
    if a.n_norm:                                           # speed9: tools/best5/n_norm (same rows)
        sys.path.append(str(ROOT / "tools" / "best5")); import n_norm as NNORM_
        G["NORM"] = NNORM_.normalise
    PRE5_BIG_OFF = bool(BIGM and a.pre5)                   # merge5: pre5 is switched off in big mode (recorded in walls.merge5)
    if PRE5_BIG_OFF:
        a.pre5 = 0
    if a.route_bin and BIGM and not L8["on"]:             # load8: wired (read in load_all)
        sys.exit("chain_m.py: --route_bin is not wired for big mode (big5's early route read)")
    if a.deadline_s is not None:                           # merge6 (b)
        DL.update(on=True, X=float(a.deadline_s), ref=_dl_ref())
        if a.deadline_est:
            DL["est"] = {k: float(v) for k, v in json.loads(Path(a.deadline_est).read_text()).items()}
    G["act"] = a.act
    NPK_DICT[0] = not a.npk_table
    BIGREC = {"mode": a.big, "on": BIGM, "items": sorted(BIG)}
    if BIGM:                                               # chain_gb's validated base: chain_h switches and W3 stage flags off
        if a.gnn_family is None:
            a.gnn_family = ["mempool_group=MemPool", "mempool_cluster=MemPool"]
        ch_ = {}
        for k_ in ("h_read", "h_write", "h_dem", "h_asm", "h_inc", "h_crit", "h_graph", "h_all", "h_pre", "ovf_impl", "ovf_timer",
                   "ovf_rr_passes", "ovf_order", "ovf_esc_pre", "ovf_big_nets", "ovf_hmax_big", "ovf_big_runs", "ovf_big_passes",
                   "ovf_esc_check", "ovf_max_runs", "ovf_hmax", "ovf_hot", "pw_batch"):
            if B5["on"] and k_ in B5L.KEEP: continue        # big5: exact switches kept in big mode
            if B8["on"] and k_ == "pw_batch" and B8L.pc("pwb", "0"):   # big8: power round in checked batches in big mode
                if a.big8_pw_batch > 0: a.pw_batch = a.big8_pw_batch
                B8L.R["pieces"]["pwb"] = a.pw_batch; continue
            d_ = ap.get_default(k_)
            if getattr(a, k_) != d_:
                ch_[k_] = [getattr(a, k_), d_]; setattr(a, k_, d_)
        if a.pw2_nets == 0 and a.pw2_big > 0:
            ch_["pw2_nets"] = [0, a.pw2_big]; a.pw2_nets = a.pw2_big
        if a.post_crit_big > 0:                             # chain_j b: big mode post_crit (chain_gb2 p2yc)
            ch_["post_crit"] = [a.post_crit, a.post_crit_big]; a.post_crit = a.post_crit_big
        for k_ in ("s4_ofc", "s4_esc"):                     # chain_j d: speed4 items are checked on the small cases only
            if getattr(a, k_):
                ch_[k_] = [getattr(a, k_), 0]; setattr(a, k_, 0)
        if a.rs != "off":                                   # chain_j c: reshape stage is a small-case stage (lazy pins in big mode)
            ch_["rs"] = [a.rs, "off"]; a.rs = "off"
        if a.engine == "micro" and a.micro_big == "rounds":  # chain_j a: big mode keeps chain_i's rounds
            ch_["engine"] = ["micro", "rounds"]; a.engine = "rounds"
        BIGREC["reset"] = ch_
    if a.ovf_order == "re" and not a.ovf_esc_check:
        sys.exit("chain_i.py: --ovf_order re needs --ovf_esc_check 1 (escape after the reroute leaves open nets otherwise)")
    if a.engine == "micro" and not (a.h_inc or a.h_all) and a.e_inc:
        sys.exit("chain_j.py: --engine micro needs the incremental timer (--h_inc 1 or --h_all 1)")
    if a.post_crit or a.pw_batch or a.pw2_nets or a.ovfround or a.rs != "off" or a.engine == "micro":
        assert len(a.snaps.split(",")) == 1 and a.only_b and a.power == "1", "chain_i after-stages need one snapshot, --only_b, --power 1"
    for k_ in ("read", "write", "dem", "asm", "inc", "crit", "graph"):
        H[k_] = getattr(a, "h_" + k_) or (a.h_all and (8 if k_ == "write" else 1))
    H["pre"] = int(a.h_pre)                                # item 7 (not part of --h_all)
    H["check"] = bool(a.h_check)
    a.write = "cpp"; a.power_round = True
    ITEMS.difference_update(a.no_item)
    CHECK["on"] = bool(a.check)
    PROM = None
    if a.promote_impl == "full":
        if not (F3X / "promote_full.py").exists():
            sys.exit(f"micro2.py: --promote_impl full needs {F3X / 'promote_full.py'} (not found)")
        import promote_full as PROM
    snaps = {int(x.split(":")[0]): x.split(":")[1] for x in a.snaps.split(",")}
    W = {"import_s": time.perf_counter() - T0}
    DGS = None
    if a.dgr:                                              # dgr1x: device and tables on a thread, no torch
        sys.path.insert(0, str(ROOT / "tools" / "best7")); sys.path.insert(0, str(ROOT / "tools" / "best8")); import dgr_stage_r as DGS
        DGS.begin(a.dgr_gnn)
    if a.pre5:                                             # pre5
        th = threading.Thread(target=P5L.load, args=(a, globals()), daemon=True, name="preload"); th.start()
    else:
        th = threading.Thread(target=load_all, args=(a,), daemon=True); th.start()
    time.sleep(a.route_wall)
    t_rd = time.perf_counter(); th.join()
    W["preload_wait_s"] = time.perf_counter() - t_rd
    assert "err" not in PRE, PRE.get("err")
    if a.load8_stop:                                       # load8 test: the load's timers, then exit (first timer and activity joined)
        W["l8_preload_joined"] = time.perf_counter() - T0
        for k in ("t_imports", "t_csta", "t_state", "t_preparse", "preload_s", "preload_end", "t_esc_pre", "t_fastload_install", "t_ovf_pre"): W[k] = PRE.get(k)
        if B5["on"]:
            _tm = B5L.EARLY.get("tm")
            if _tm is not None: _tm.join()
            W["l8_first_timer_end"] = time.perf_counter() - T0
            import pre5_load as _P5L
            if "ACT" in _P5L.EARLY: _P5L.EARLY["ACT"].get()
            W["l8_act_end"] = time.perf_counter() - T0
        if FLD is not None: W["fastload_stats"] = {k: round(v, 3) for k, v in FLD.STATS.items()}
        rec = {"design": a.design, "suite": a.suite, "walls": W, "loadavg": os.getloadavg(), "cpu_s": time.process_time(), "stop": True}
        if B5["on"]: rec["big5"] = B5L.record()
        if B8["on"]: rec["big8"] = B8L.record()
        if L8["on"]: rec["load8"] = L8L.record()
        if a.json: Path(a.json).write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps({"walls": {k: v for k, v in W.items() if isinstance(v, float)}}), flush=True)
        os._exit(0)
    if a.pre5 or a.pre5_stop:                              # pre5: records; test mode stops here (also --pre5 0: chain_j's preload)
        W["pre5"] = PRE.get("pre5")
        if a.pre5_stop:
            if a.pre5 and "ACT" in P5L.EARLY:
                P5L.EARLY["ACT"].get()
            W["t_imports"], W["preload_s"], W["preload_end"] = PRE.get("t_imports"), PRE.get("preload_s"), PRE.get("preload_end")
            W["stop_at_s"] = time.perf_counter() - T0
            rec = {"design": a.design, "suite": a.suite, "walls": W, "loadavg": os.getloadavg(), "cpu_s": time.process_time()}
            if a.json: Path(a.json).write_text(json.dumps(rec, indent=1, default=str))
            os._exit(0)
    for k in ("t_imports", "t_csta", "t_state", "t_preparse", "preload_s", "preload_end"): W[k] = PRE.get(k)
    W["big"] = BIGREC
    W["merge5"] = {"pre5": int(a.pre5), "mid5": int(a.mid5), "late5": int(a.late5), "pre5_off_in_big_mode": PRE5_BIG_OFF}
    if BIGM:
        for k in ("t_esc_pre", "t_fastload_install", "act_file", "gnn_fold", "ovf_pins_info", "esc_pins_reused_parse", "t_ovf_pre", "csta_lib", "route_lib"):
            W[k] = PRE.get(k)
    _S0 = PRE["S"]
    W["consts"] = {k: float(getattr(_S0, k)) for k in ("w1", "w2", "w3", "w4", "r1", "r2", "nend") if getattr(_S0, k, None) is not None}
    if a.t_warm:                                           # speed10
        t_ = time.perf_counter(); K_ = PRE["K"]; rc_ = K_.build_rc(np.zeros((0, 7), np.int64))
        K_.full(rc_["c1"], rc_["c2"], rc_["rpi"], rc_["ctot"], rc_["elm"], rc_["par"], backward=False, copy=False); W["t_warm_s"] = time.perf_counter() - t_; del rc_
    if a.t_pre and a.pre5 and not BIGM:                    # speed10: route-independent tables while the router runs
        sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP_
        from timing_promote import _key as _key_t
        TFP_.pre_tables(a, G, PRE, PRE["S"], _key_t); W["t_pre_s"] = TFP_.PIDX["t_s"]
    if a.t_gc:                                             # speed10
        import gc as _gc; _gc.disable()
    tpost = time.perf_counter(); TM("start")
    W["p7_cpu_post"] = sum(os.times()[:2])                  # par7: process CPU at the route read
    IR, B, V1, V2, CS, FT, FM, F4, DS, PC = (G[k] for k in ("IR", "B", "V1", "V2", "CS", "FT", "FM", "F4", "DS", "PC"))
    import route_export as RE
    import timing_promote as TP_, fastmode3 as F3_
    import fast_apply as FA
    if a.of_rule == "official":                            # before the first estimate (mk calls FM.overflow_cost at run time)
        sys.path.append(str(ROOT / "tools" / "offix"))
        import of_official
        of_official.patch()
    W["of_rule"] = a.of_rule
    if a.ovf != "off":                                     # chain_ovf.py lines 521-522: the stage's own test always uses it
        sys.path.append(str(ROOT / "tools" / "offix"))
        import of_official
        G["OFO"] = of_official
    if a.s4_ofc and a.of_rule == "official" and a.ovf != "off":     # chain_j d: speed4 item 4b (both users of the official cost)
        sys.path.append(str(ROOT / "tools" / "speed4"))
        import s4_ovf
        G["S4OFC"] = s4_ovf.install(FM, of_official, a.s4_check)
        if a.n_ofc:                                         # speed9: the same cache with the map compare and the sums on threads (same bits)
            sys.path.append(str(ROOT / "tools" / "best5")); import n_ofc as NOFC_
            if a.t_ofc0:                                    # speed10: the first call's per-layer numpy work on threads (same calls, same bits)
                sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP_
                TFP_.ofc0_install(NOFC_, int(os.environ.get("MID5_THREADS", "8")))
            G["S4OFC"] = NOFC_.install(FM, of_official, a.s4_check, int(os.environ.get("MID5_THREADS", "8")))
    if a.s4_esc:                                           # chain_j d: speed4 item 4c (before the escape / overflow stage use them)
        sys.path.append(str(ROOT / "tools" / "speed4"))
        import s4_escape
        s4_escape.install(a.s4_check)
        G["S4ESC"] = s4_escape
    TM("imports_installs")
    SHADOW = FA._Shadow(PRE["S"].la) if a.check and 2 in ITEMS else None
    if 2 in ITEMS:
        FA.install(TP_, RE, (FM, F3_, F4), shadow=SHADOW)
    if a.n_seg and not BIGM:                               # speed9: the timer's segment array on threads (same array); before big5 wraps it
        sys.path.append(str(ROOT / "tools" / "best5")); import n_seg as NSEG_
        NSEG_.install(G["CS"])
        if L8S and L8L.pc("seg"): L8L.install_seg(G["CS"])   # load8: names through the C++ table, once per names tuple
    if L8S and "B2" in G: L8L.patch_b2(G["B2"])            # load8: build_fast2's two name dicts at C speed
    if (L8S or L8["on"]) and "E" in G: L8L.patch_escape(G["E"])   # load8: the escape's candidate scan without a Python tuple per net
    if a.n_seg and not BIGM and a.t_fp3:                   # speed10: the net-name lookup once
        sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP3_
        TFP3_.seg_install(G["CS"])
    if B5["on"]: B5L.install(a, G, PRE, W, BIGM, PROM)  # big5: big-mode speed pieces + stage marks
    if PROM is not None:
        prom_crit, prom_pw = PROM.promote_cpp, PROM.promote_pw
    elif 4 in ITEMS:
        import promote_fast as PF
        prom_crit, prom_pw = PF.promote_crit, PF.promote_pw
    else:
        import promote_pw as PW_
        prom_crit, prom_pw = PC.promote_cpp, PW_.promote_cpp
    W["items"] = sorted(ITEMS); W["promote_impl"] = a.promote_impl; W["ovf"] = a.ovf
    TM("fa_b5_install")
    cd, K, S = PRE["cd"], PRE["K"], PRE["S"]
    if DGS is not None:
        DGS.tables(cd, (S.w1, S.w2, S.nend))
    SKIPFP = False
    dr = V1.DEFAULT_DATA_ROOT
    if H["inc"] or a.ovf_timer == "inc":
        K.S = S
    TFP_ = None
    if a.t_fp2:                                            # speed10
        sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP_
        G["TFP"] = TFP_
    M5 = CPU5 = None
    if a.mid5 or a.mid5_cpu:                               # mid5: drivers + counters (tools/mid5/mid5.py)
        sys.path.append(str(ROOT / "tools" / "mid5"))
        import mid5 as M5
        CPU5 = M5.CPU(a.mid5_cpu)
        if a.mid5:
            G["M5"] = M5
    TM("mid5_import")
    T5_0 = time.perf_counter()                             # mid5: wall of the region (route read .. end of the overflow stage)
    if CPU5: CPU5.start("route_read")
    if a.route_bin and not (L8["on"] and BIGM):            # merge6 (a): read the router's binary route (same arrays, ROUTER5 R-BIN); load8 big mode: set in load_all
        sys.path.append(str(ROOT / "tools" / "router5")); import r5bin as R5B
        G["read_route"] = lambda p, _n=int(os.environ.get("OMP_NUM_THREADS", "8")): R5B.read_route_bin(p, _n)
    t = time.perf_counter(); rt = G["read_route"](str(a.route)); W["route_read_s"] = time.perf_counter() - t
    if DGS is not None:
        DGS.names(S, list(rt.net_names))
    TM("route_read")
    if a.t_small > 0 and len(rt.x0) > a.t_small:           # speed10
        a.t_rect_h = a.t_no_post = a.t_pw_rej = a.t_esc_p2 = a.t_rr_area = 0; a.t_runs = 0; a.t_crit_stop = 0.0; a.pw_bisect_depth = max(a.pw_bisect_depth, 2); W["t_small_off"] = True
    if CPU5: CPU5.stop("route_read"); CPU5.start("escape_test")
    if H["check"] and H["read"]:
        r0_ = G["read_route_orig"](str(a.route))
        hchk("read", all(getattr(r0_, k).dtype == getattr(rt, k).dtype and np.array_equal(getattr(r0_, k), getattr(rt, k))
                         for k in ("net_id", "x0", "y0", "z0", "x1", "y1", "z1", "offsets")) and tuple(r0_.net_names) == tuple(rt.net_names))
        del r0_
    R5 = None
    if a.mid5 and a.escape != "off":                      # mid5 E: the escape test on a thread while the first timer runs
        R5 = {}
        def _esc5(rt_=rt):
            try:
                R5["rt"] = escape_inmem(a, rt_, W)
            except BaseException as e_:
                R5["err"] = e_
        if a.t_esc_wait > 0 and len(rt.x0) > a.t_esc_wait: G["T_EVT"] = threading.Event()   # speed10
        th5 = threading.Thread(target=_esc5, daemon=True); th5.start()
        if "T_EVT" in G:                                   # speed10: the decision first; with the escape on, no first timer on the route it replaces
            G["T_EVT"].wait(); W["t_esc_waited"] = bool(W.get("esc_on"))
            if W.get("esc_on"):
                th5.join()
                if "err" in R5:
                    raise R5["err"]
                rt = R5["rt"]
    elif a.escape != "off":
        rt = escape_inmem(a, rt, W)
    if CPU5: CPU5.stop("escape_test"); CPU5.start("timer_crit")
    TM("esc_thread_start")
    t = time.perf_counter()
    seg = CS.seg_from_route(cd, rt, FT.GRID); TM("t1_seg"); rcd = K.build_rc(seg); TM("t1_build_rc")
    o = (K.full(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"], backward=False, copy=False) if a.t_first
         else K.analyze(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"])); TM("t1_analyze")
    if a.t_prof: W["t_first_stats"] = np.asarray(o["stats"]).tolist()
    if os.environ.get("T_FWD_TEST"):                       # speed10 test: the same forward pass again, N times, then exit
        tt_ = [float(o["stats"][0])]
        for _ in range(int(os.environ["T_FWD_TEST"])):
            o_ = K.full(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"], backward=False, copy=False); tt_.append(float(o_["stats"][0]))
        print("T_FWD_TEST", a.design, a.suite, "warm", round(W.get("t_warm_s", 0.0), 3), [round(x, 3) for x in tt_], "load", os.getloadavg()[0], flush=True); os._exit(0)
    SKIPFP = False
    if a.dgr and a.dgr_skip_fp:
        SKIPFP = True if a.dgr_skip_fp == 1 else DGS.skip_rule(a, S, rt, o, W, _dl_now() if DL["on"] else None, DL["X"], DL["est"])
    if not SKIPFP:
        cr = K.crit(o, rcd, a.npaths); TM("t1_crit"); crit_data = CS.to_crit_data(cd, cr); TM("t1_crit_data")
    if R5 is not None:
        th5.join(); TM("t1_esc_join")
        if "err" in R5:
            raise R5["err"]
        W["mid5_esc_join_s"] = time.perf_counter() - t
        if R5["rt"] is not rt:                            # escape on: the route changed -> the timer on the escaped route
            rt = R5["rt"]
            seg = CS.seg_from_route(cd, rt, FT.GRID); rcd = K.build_rc(seg)
            o = K.analyze(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"])
            if not (a.dgr and a.dgr_skip_fp):
                cr = K.crit(o, rcd, a.npaths); crit_data = CS.to_crit_data(cd, cr)
            W["mid5_timer_redo"] = True
    timer_rec = None if SKIPFP else {"design": cd, "result": {"net_ctot": rcd["ctot"].max(1), "net_rd_kohm": o["net_rd_kohm"]}}
    W["timer_crit_s"] = time.perf_counter() - t
    if CPU5: CPU5.stop("timer_crit"); CPU5.start("build")
    TM("t1_rest")
    t = time.perf_counter(); tm = {}
    if a.n_b2 and a.build == "b2":                         # speed9: the builder's per-net loop in C++ on threads (same P)
        sys.path.append(str(ROOT / "tools" / "best5")); import n_b2 as NB2_
        if a.t_pre:                                        # speed10: + the name dicts from the wait (same dicts)
            sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP_
            TFP_.b2_install(G["B2"])
        else:
            NB2_.install(G["B2"])
        W["n_b2"] = NB2_.STAT
    if not SKIPFP:
        P = (G["B2"].build_fast2 if a.build == "b2" else B.build_fast)(a.design, a.suite, str(a.route), None, a.max_nets, rt=rt, crit_data=crit_data, timing=tm)
        P.update(route_path=str(a.route), design=a.design, suite=a.suite); TM("b_build2")
        orig = IR.TP.parse_net_rc
        IR.TP.parse_net_rc = lambda _p, _nr=crit_data[1]: _nr
        try:
            V2.add_real_tech_rc(P, timer_rec, dr, Path("<mem>"))
        finally:
            IR.TP.parse_net_rc = orig
        TM("b_tech_rc")
    W["build_s"] = time.perf_counter() - t
    W["b2_tm"] = {k: round(v, 4) for k, v in tm.items()}
    if CPU5: CPU5.stop("build")
    t = time.perf_counter()
    lam_hi = a.gate if a.gate > 0 else a.lam
    GB = None
    if lam_hi > 0 and a.pre5 and a.pre5_gnn == "np" and a.t_gnn_bg and a.par7 and 3 in ITEMS and a.mid5 and not H["check"] and not B8["on"]:   # speed10: on a thread; joined before the prior's DP
        GB = {}
        def _gnn_bg(P_=P):
            t_b = time.perf_counter()
            try:
                out_b, W["pre5_gnn"], _ = PRE["prior"].logits(P_, a.pre5_threads)
                GB["out"] = out_b
            except BaseException as e_:
                GB["err"] = e_
            W["t_gnn_bg_s"] = time.perf_counter() - t_b
        GB["th"] = threading.Thread(target=_gnn_bg, daemon=True); GB["th"].start()
    elif lam_hi > 0 and ((a.pre5 and a.pre5_gnn == "np") or (L8["on"] and "prior" in PRE)):   # pre5, load8: prior forward in C++ (tools/pre5/gnn_np.py), no torch
        out_, W["pre5_gnn"], _ = PRE["prior"].logits(P, a.pre5_threads)
        P["gnn_logits"] = out_
    elif lam_hi > 0:
        import torch
        DD = PRE["DD"]
        if H["graph"]:                                     # item 6
            import h_graph
            t6 = time.perf_counter(); x, ei, sid = h_graph.graph(P); W["h_graph_s"] = time.perf_counter() - t6
            if H["check"]:
                x0_, e0_, s0_ = DD.graph(P)
                ok_ = all(torch.equal(x0_[k], x[k]) and x0_[k].dtype == x[k].dtype for k in x0_) and set(x0_) == set(x)
                ok_ = ok_ and set(e0_) == set(ei) and all(torch.equal(e0_[k], ei[k]) and e0_[k].dtype == ei[k].dtype for k in e0_)
                hchk("graph", ok_ and torch.equal(s0_, sid))
        else:
            x, ei, sid = DD.graph(P)
        with torch.no_grad():
            lg = DD.to_logits(PRE["model"](x, ei), sid).clamp(-60.0, 60.0)
        out_ = lg.detach().cpu().numpy().astype(np.float64)
        out_[~(np.asarray(P["cand"]) >= 0)] = -np.inf
        P["gnn_logits"] = out_
        if a.pre5 and a.pre5_gnn == "check":               # pre5 test: C++ prior vs torch (torch's logits are used)
            o2_, W["pre5_gnn"], (x2_, e2_, s2_) = PRE["prior"].logits(P, a.pre5_threads)
            fin_ = np.isfinite(out_)
            W["pre5_gnn_check"] = {"same_inf": bool(np.array_equal(fin_, np.isfinite(o2_))),
                                   "max_abs": float(np.abs(out_[fin_] - o2_[fin_]).max()) if fin_.any() else 0.0,
                                   "n_diff": int((out_[fin_] != o2_[fin_]).sum()), "n": int(fin_.sum()),
                                   "argmax_diff": int((np.argmax(np.where(fin_, out_, -1e300), 1) != np.argmax(np.where(fin_, o2_, -1e300), 1)).sum()),
                                   "graph_equal": bool(all(np.array_equal(x[k].numpy(), x2_[k]) for k in x) and all(np.array_equal(ei[k].numpy(), e2_[k]) for k in ei)
                                                       and np.array_equal(sid.numpy(), s2_))}
    W["gnn_s"] = time.perf_counter() - t; TM("gnn")
    G_ = FT.GRID

    def mk(o_, rcd_):
        ctot = np.asarray(rcd_["ctot"]).max(1)
        r = {"wns": o_["wns"], "tns": o_["tns"], "net_ctot": ctot, "net_rd_kohm": o_["net_rd_kohm"], "missing": 0, "nviol": 0, "n_ep": len(o_["ep_slack"])}
        r["of"] = FM.overflow_cost(S.la)
        r["pw"] = float((S.kpn * ctot).sum()) if S.kpn is not None else 0.0
        r["score"] = S.w1 * (r["wns"] - S.r1) + S.w2 * (r["tns"] - S.r2) / S.nend + S.w4 * r["of"] + S.w3 * r["pw"]
        return r

    if H["dem"]:
        import h_dem as HD
    ASM = assemble
    if H["asm"]:                                           # item 3b
        import h_asm
        if a.t_fp3:                                        # speed10
            sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP3_
            TFP3_.asm_install(h_asm)
        def ASM(P_, ch_, rt_, IR_, V1_, RE_):
            out_ = h_asm.assemble(P_, ch_, rt_, RE_)
            if H["check"]:
                ref_ = assemble(P_, ch_, rt_, IR_, V1_, RE_)
                hchk("asm", ref_[0].dtype == out_[0].dtype and np.array_equal(ref_[0], out_[0]) and ref_[2] == out_[2]
                     and all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(ref_[1], out_[1])))
            return out_

    def eval_cols(noff_, cols_, keep_attached=True, dem=None, inc_nets=None):
        """attach demand of a candidate single-pass route and time it (estimator). returns (rt2, arr6, o, rcd, r)"""
        rt2_ = types.SimpleNamespace(net_names=rt.net_names, offsets=noff_.astype(np.int64), x0=cols_[0], y0=cols_[1], z0=cols_[2], x1=cols_[3], y1=cols_[4], z1=cols_[5])
        a6 = np.empty((len(cols_[0]), 6), np.int64)
        a6[:, 0] = cols_[0].astype(np.int64) // G_; a6[:, 1] = cols_[1].astype(np.int64) // G_; a6[:, 2] = cols_[2]
        a6[:, 3] = cols_[3].astype(np.int64) // G_; a6[:, 4] = cols_[4].astype(np.int64) // G_; a6[:, 5] = cols_[5]
        if dem is not None:                            # item 3a: map from a known map + the changed nets
            t_ = time.perf_counter()
            D_old, a6_old, off_old, idx_ = dem
            d_ = HD.derive(FM, S.la, D_old, a6_old, off_old, a6, noff_.astype(np.int64), idx_)
            S.attach_h(list(rt.net_names), noff_.astype(np.int64), a6, d_)
            HT["dem_s"] = HT.get("dem_s", 0.0) + time.perf_counter() - t_
            if H["check"]:
                ref_ = S.la.d.copy(); S.la.d[:] = 0; FM.build_demand_fast(S.la, S.base_arr)
                hchk("dem", np.array_equal(ref_, S.la.d), f"{len(idx_)} nets"); S.la.d[...] = ref_; del ref_
        else:
            S.la.d[:] = 0
            S.attach(list(rt.net_names), noff_.astype(np.int64), a6)
        seg_ = CS.seg_from_route(cd, rt2_, G_)
        if inc_nets is not None:                       # item 3c: incremental re-time from the kernel's current state
            t_ = time.perf_counter()
            if not hasattr(cd, "_nf_index"):
                cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
            nid = np.array([cd._nf_index.get(rt.net_names[i], -1) for i in inc_nets], np.int64)
            nets = np.unique(np.asarray(cd.nf_net)[nid[nid >= 0]].astype(np.int64)) if len(nid) else np.zeros(0, np.int64)
            nets = nets[nets >= 0]
            sub = seg_[np.isin(seg_[:, 0], nets)]
            K.begin(); K.build_rc_subset(sub, nets)
            o_ = K.update(nets)
            K.pending_snap = dict(S.new)
            rc_ = K.rc
            HT["inc_cand_s"] = time.perf_counter() - t_; HT["inc_cand_nets"] = int(len(nets))
            if H["check"]:
                rcf = K.build_rc(seg_)
                bv = lambda x: np.ascontiguousarray(x).view(np.uint64 if np.asarray(x).dtype.itemsize == 8 else np.uint8).ravel()
                nm_ = lambda x, y: int((bv(x) != bv(y)).sum())
                rcm = sum(nm_(rc_[k], rcf[k]) for k in ("c1", "c2", "rpi", "ctot", "par", "elm"))
                Kr = PRE["Kref"]; orf = Kr.analyze(rcf["c1"], rcf["c2"], rcf["rpi"], rcf["ctot"], rcf["elm"], rcf["par"])
                om = sum(nm_(o_[k], orf[k]) for k in ("AT", "SL", "gate_d", "wire_d", "BEST", "net_rd", "rmin", "ep_sl_all"))
                hchk("inc_cand", rcm == 0 and om == 0 and o_["wns"] == orf["wns"] and o_["tns"] == orf["tns"], f"rc_mis {rcm} arr_mis {om} nets {len(nets)}")
        else:
            rc_ = K.build_rc(seg_)
            o_ = K.analyze(rc_["c1"], rc_["c2"], rc_["rpi"], rc_["ctot"], rc_["elm"], rc_["par"])
        return rt2_, a6, o_, rc_, mk(o_, rc_)

    def mk5(o_, rcd_, of_):                                # == mk with the overflow cost of the same map computed before
        ctot = np.asarray(rcd_["ctot"]).max(1)
        r = {"wns": o_["wns"], "tns": o_["tns"], "net_ctot": ctot, "net_rd_kohm": o_["net_rd_kohm"], "missing": 0, "nviol": 0, "n_ep": len(o_["ep_slack"])}
        r["of"] = of_
        r["pw"] = float((S.kpn * ctot).sum()) if S.kpn is not None else 0.0
        r["score"] = S.w1 * (r["wns"] - S.r1) + S.w2 * (r["tns"] - S.r2) / S.nend + S.w4 * r["of"] + S.w3 * r["pw"]
        return r

    def eval_cols5(noff_, cols_, dem=None, inc_nets=None, side=None, seg_pre=None):
        """mid5 F: eval_cols (H["check"] off) with the timer kernel on a thread while this thread attaches the demand, computes
        the overflow cost of the attached map and runs side() (the other candidate's assembly + segments). Kernel state and
        demand state are disjoint and every piece is a pure function of its inputs: same values. Returns (eval_cols tuple, side())."""
        rt2_ = types.SimpleNamespace(net_names=rt.net_names, offsets=noff_.astype(np.int64), x0=cols_[0], y0=cols_[1], z0=cols_[2], x1=cols_[3], y1=cols_[4], z1=cols_[5])
        a6 = TFP_.a6_of(cols_, G_) if a.t_fp2 else None    # speed10: on threads (same array)
        if a6 is None:
            a6 = np.empty((len(cols_[0]), 6), np.int64)
            a6[:, 0] = cols_[0].astype(np.int64) // G_; a6[:, 1] = cols_[1].astype(np.int64) // G_; a6[:, 2] = cols_[2]
            a6[:, 3] = cols_[3].astype(np.int64) // G_; a6[:, 4] = cols_[4].astype(np.int64) // G_; a6[:, 5] = cols_[5]
        TM("e_a6")
        seg_ = seg_pre if seg_pre is not None else CS.seg_from_route(cd, rt2_, G_)
        TM("e_seg")
        out = {}
        t_ = time.perf_counter()
        if inc_nets is not None:
            if not hasattr(cd, "_nf_index"):
                cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
            nid = np.array([cd._nf_index.get(rt.net_names[i], -1) for i in inc_nets], np.int64)
            nets = np.unique(np.asarray(cd.nf_net)[nid[nid >= 0]].astype(np.int64)) if len(nid) else np.zeros(0, np.int64)
            nets = nets[nets >= 0]
            sub = seg_[np.isin(seg_[:, 0], nets)]
            def _kern():
                try:
                    K.begin(); K.build_rc_subset(sub, nets); TM("k_rc_sub"); out["o"] = K.update(nets); TM("k_update")
                except BaseException as e_:
                    out["err"] = e_
        else:
            def _kern():
                try:
                    rc0 = K.build_rc(seg_); out["rc"] = rc0; TM("k_build_rc")
                    out["o"] = (K.full(rc0["c1"], rc0["c2"], rc0["rpi"], rc0["ctot"], rc0["elm"], rc0["par"], backward=False, copy=False) if a.t_np_nb
                                else K.analyze(rc0["c1"], rc0["c2"], rc0["rpi"], rc0["ctot"], rc0["elm"], rc0["par"])); TM("k_analyze")
                except BaseException as e_:
                    out["err"] = e_
        thk = threading.Thread(target=_kern, daemon=True); thk.start()
        if dem is not None:
            t2_ = time.perf_counter()
            D_old, a6_old, off_old, idx_ = dem
            TM("e_kstart"); d_ = HD.derive(FM, S.la, D_old, a6_old, off_old, a6, noff_.astype(np.int64), idx_); TM("e_derive")
            S.attach_h(list(rt.net_names), noff_.astype(np.int64), a6, d_)
            HT["dem_s"] = HT.get("dem_s", 0.0) + time.perf_counter() - t2_
        else:
            S.la.d[:] = 0
            S.attach(list(rt.net_names), noff_.astype(np.int64), a6)
        TM("e_dem")
        of_ = FM.overflow_cost(S.la); TM("e_of")
        so_ = side() if side is not None else None
        TM("e_side")
        thk.join(); TM("e_join")
        if "err" in out:
            raise out["err"]
        o_ = out["o"]
        if inc_nets is not None:
            K.pending_snap = dict(S.new)
            rc_ = K.rc
            HT["inc_cand_s"] = time.perf_counter() - t_; HT["inc_cand_nets"] = int(len(nets))
        else:
            rc_ = out["rc"]
        return (rt2_, a6, o_, rc_, mk5(o_, rc_, of_)), so_

    if B8["on"] and B8L.pc("mem"):                         # big8 mem: the first analysis' objects are not read from here on
        B8L.drop_early_refs(G, K, CS); o = rcd = seg = cr = crit_data = timer_rec = None
        x = ei = sid = lg = out_ = None; B8L.trim("gate_start")   # the prior's graph tensors: used only to make the logits in P
    t = time.perf_counter()
    gate_rec = None
    if CPU5: CPU5.start("gate")
    if a.gate > 0:
        if 3 not in ITEMS:
            cands = {}
            for nm, lam_ in (("noprior", 0.0), ("prior", a.gate)):
                ch_, dp_ = DS.assign_v2_cpp(P, "numpy", a.dw, a.pw, V1.DEFAULT_VIA_COST, lam_)
                noff_, cols_, nchg_ = assemble(P, ch_, rt, IR, V1, RE)
                cands[nm] = (noff_, cols_)
            est = {}; res = {}
            for nm in ("noprior", "prior"):
                res[nm] = eval_cols(*cands[nm])
                est[nm] = {k: float(res[nm][4][k]) for k in ("wns", "tns", "of", "pw", "score")}
            pick = "prior" if est["prior"]["score"] < est["noprior"]["score"] else "noprior"
            noff, cols = cands[pick]
            rt2, arr6, o2, rcd2, r0 = res[pick] if pick == "prior" else eval_cols(noff, cols)     # re-attach when the last attached one is not the pick
            gate_rec = {"pick": pick, "est": est}
        else:
            t_p7 = time.perf_counter()
            chs = B8L.dp_pair(P, a, V1) if B8["on"] else None   # big8: both first-pass DPs at once (par7 (b))
            if chs is not None:
                pass
            elif a.par7 and GB is not None:              # speed10: the first candidate's DP now (its cost does not read the logits at lam 0), the prior's after the GNN (side5)
                sys.path.append(str(ROOT / "tools" / "par7")); import p7_dp as P7DP
                ch7_, W["p7_dp"] = P7DP.assign_pair(P, a.dw, a.pw, V1.DEFAULT_VIA_COST, (0.0,), 2 * a.par7_pt)
                chs = {"noprior": ch7_[0], "prior": None}
            elif a.par7:                                 # par7 (b): both candidates' DPs at once, --par7_pt threads each (same choices)
                sys.path.append(str(ROOT / "tools" / "par7")); import p7_dp as P7DP
                ch7_, W["p7_dp"] = P7DP.assign_pair(P, a.dw, a.pw, V1.DEFAULT_VIA_COST, (0.0, a.gate), a.par7_pt)
                chs = {"noprior": ch7_[0], "prior": ch7_[1]}
            else:
                chs = {nm: DS.assign_v2_cpp(P, "numpy", a.dw, a.pw, V1.DEFAULT_VIA_COST, lam_)[0] for nm, lam_ in (("noprior", 0.0), ("prior", a.gate))}
            W["p7_dp_s"] = time.perf_counter() - t_p7; TM("g_dp_pair")
            same = GB is None and np.array_equal(np.asarray(chs["noprior"]), np.asarray(chs["prior"]))
            est = {}; res = {}
            t3 = time.perf_counter()
            noff, cols, _ = ASM(P, chs["noprior"], rt, IR, V1, RE)
            HT["asm_s"] = time.perf_counter() - t3; TM("g_asm_np")
            use_dem = H["dem"] and "h_Dr" in PRE
            rn_ = np.asarray(P["run_net"], np.int64); nri_ = np.asarray(P["net_route_idx"], np.int64)
            if use_dem:                                # nets the single pass rebuilds (assemble's `new`): all DP nets
                idx_dp = np.unique(nri_[np.unique(rn_)]) if len(rn_) else np.zeros(0, np.int64)
                dem_np = (PRE["h_Dr"], PRE["h_arr6r"], PRE["h_offr"], idx_dp)
            SD5 = None
            if a.mid5 and not H["check"]:                  # mid5 F: noprior timer || its demand + the prior's assembly / segments
                def side5():
                    nonlocal same
                    if GB is not None:                     # speed10: GNN joined, the prior's DP, then as below
                        GB["th"].join()
                        if "err" in GB:
                            raise GB["err"]
                        P["gnn_logits"] = GB["out"]; TM("g_gnn_join")
                        ch7_, _ = P7DP.assign_pair(P, a.dw, a.pw, V1.DEFAULT_VIA_COST, (a.gate,), 2 * a.par7_pt)
                        chs["prior"] = ch7_[0]; TM("g_dp_prior")
                        same = np.array_equal(np.asarray(chs["noprior"]), np.asarray(chs["prior"]))
                    if same:
                        return None
                    t3_ = time.perf_counter()
                    np_, cp_, _ = ASM(P, chs["prior"], rt, IR, V1, RE)
                    HT["asm_s"] += time.perf_counter() - t3_
                    r2p_ = types.SimpleNamespace(net_names=rt.net_names, offsets=np_.astype(np.int64), x0=cp_[0], y0=cp_[1], z0=cp_[2], x1=cp_[3], y1=cp_[4], z1=cp_[5])
                    return np_, cp_, CS.seg_from_route(cd, r2p_, G_)
                res["noprior"], SD5 = eval_cols5(noff, cols, dem=dem_np if use_dem else None, side=side5)
            else:
                res["noprior"] = eval_cols(noff, cols, dem=dem_np if use_dem else None)
            TM("g_eval_np")
            if os.environ.get("T_FWD_NP"):                 # speed10 test: forward pass on the first candidate's RC, N times, then exit
                rq_ = res["noprior"][3]; tt_ = [float(res["noprior"][2]["stats"][0])]; o0_ = dict(rq_)
                for _ in range(int(os.environ["T_FWD_NP"])):
                    o_ = K.full(rq_["c1"], rq_["c2"], rq_["rpi"], rq_["ctot"], rq_["elm"], rq_["par"], backward=False, copy=False); tt_.append(float(o_["stats"][0]))
                print("T_FWD_NP", a.design, a.suite, [round(x, 3) for x in tt_], "first", W.get("t_first_stats"), "load", os.getloadavg()[0], flush=True); os._exit(0)
            est["noprior"] = {k: float(res["noprior"][4][k]) for k in ("wns", "tns", "of", "pw", "score")}
            PRE.pop("h_Dr", None); PRE.pop("h_arr6r", None); PRE.pop("h_offr", None)
            if B8["on"] and B8L.pc("mem"):                          # big8 mem: dem_np read only by the noprior evaluation; P only by
                dem_np = None                                       # the two assemblies and rn_ / nri_ (arrays taken above): with mid5
                if SD5 is not None:                                 # F the prior's assembly is finished here
                    P = None; B8L.trim("noprior_done")
            inc_used = False
            if same:                                   # identical layer choices -> identical candidate; the original picks noprior on the tie
                est["prior"] = dict(est["noprior"]); pick = "noprior"
                rt2, arr6, o2, rcd2, r0 = res["noprior"]
            else:
                kept_np = (S.la.d.copy(), S.segs, S.nets, S.ridx, S.new, S.tid, S.base_arr)   # what attach() set
                if SD5 is not None:                        # mid5 F: assembled while the noprior timer ran
                    noff_p, cols_p, seg_p5 = SD5
                else:
                    t3 = time.perf_counter()
                    noff_p, cols_p, _ = ASM(P, chs["prior"], rt, IR, V1, RE)
                    HT["asm_s"] += time.perf_counter() - t3
                # route nets whose rows can differ between the two candidates: nets with a run whose choice differs
                dif = np.nonzero(np.asarray(chs["noprior"]) != np.asarray(chs["prior"]))[0]
                idx_ch = np.unique(nri_[np.unique(rn_[dif])]) if len(dif) else np.zeros(0, np.int64)
                HT["cand_changed_nets"] = int(len(idx_ch))
                dem_p = (kept_np[0], res["noprior"][1], noff.astype(np.int64), idx_ch) if H["dem"] else None
                inc_used = bool(H["inc"])
                if SD5 is not None:                        # mid5 F: prior update || its demand
                    res["prior"], _ = eval_cols5(noff_p, cols_p, dem=dem_p, inc_nets=idx_ch if inc_used else None, seg_pre=seg_p5)
                else:
                    res["prior"] = eval_cols(noff_p, cols_p, dem=dem_p, inc_nets=idx_ch if inc_used else None)
                TM("g_eval_prior")
                est["prior"] = {k: float(res["prior"][4][k]) for k in ("wns", "tns", "of", "pw", "score")}
                pick = "prior" if est["prior"]["score"] < est["noprior"]["score"] else "noprior"
                if pick == "prior":
                    noff, cols = noff_p, cols_p
                    if inc_used:
                        K.commit()
                else:                                  # restore the noprior attach instead of a third evaluation
                    S.la.d[...] = kept_np[0]
                    S.segs, S.nets, S.ridx, S.new, S.tid, S.base_arr = kept_np[1:]
                    S.__dict__.pop("_f3x_nc", None); S.__dict__.pop("_f3x_pr", None)
                    if inc_used:                       # back to the noprior state, its required times included (== chain_g's
                        K.reject(); K.flush_pending()  # re-analysis of noprior below)
                del kept_np
                rt2, arr6, o2, rcd2, r0 = res[pick]
                if pick == "prior" and inc_used:
                    rcd2 = {k: np.array(v, copy=True) for k, v in K.rc.items()}   # cur must not follow the kernel's live RC
                    if a.ovf != "off" and a.mid5:      # mid5 A: the required-time pass alone on the committed state (== full)
                        o2 = M5.gate_backward(K)
                    elif a.ovf != "off":               # the overflow stage reads net_slack (not kept by the update): full analysis
                        o2 = K.analyze(K.rc["c1"], K.rc["c2"], K.rc["rpi"], K.rc["ctot"], K.rc["elm"], K.rc["par"])
                    else:
                        o2 = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in o2.items()}
                    # r0 stays res["prior"][4]: mk of the same values
                if pick == "noprior" and inc_used:
                    pass                               # kernel state == noprior full analysis already (reject + flush)
                elif pick == "noprior":
                    # K.crit (the round weights) reads the required times the kernel kept from its LAST analyze (static
                    # S_reqk/S_valid in csta_kern.cc), so the kernel must analyze the pick last, as the original's third
                    # evaluation does. build_rc is pure (outputs only), so analyze on the kept RC arrays == that evaluation.
                    o2 = K.analyze(rcd2["c1"], rcd2["c2"], rcd2["rpi"], rcd2["ctot"], rcd2["elm"], rcd2["par"])
                    r0 = mk(o2, rcd2)
            if a.t_np_nb and pick == "noprior":        # speed10: the pass skipped in the first candidate's analysis, on the kept state (same values as K.analyze)
                o2 = M5.gate_backward(K)
            gate_rec = {"pick": pick, "est": est, "identical_choices": bool(same)}
        W["gate_s"] = time.perf_counter() - t; TM("g_settle")
        if CPU5: CPU5.stop("gate")
    else:
        if SKIPFP:                                         # the router's route is the first state: the first timer's result stands, demand attached
            noff = np.asarray(rt.offsets, np.int64); cols = [np.ascontiguousarray(getattr(rt, k_)) for k_ in ("x0", "y0", "z0", "x1", "y1", "z1")]
            arr6 = np.empty((len(cols[0]), 6), np.int64)
            arr6[:, 0] = cols[0].astype(np.int64) // G_; arr6[:, 1] = cols[1].astype(np.int64) // G_; arr6[:, 2] = cols[2]
            arr6[:, 3] = cols[3].astype(np.int64) // G_; arr6[:, 4] = cols[4].astype(np.int64) // G_; arr6[:, 5] = cols[5]
            if H["dem"] and "h_Dr" in PRE and np.array_equal(PRE["h_arr6r"], arr6):
                S.attach_h(list(rt.net_names), noff, arr6, PRE["h_Dr"])
            else:
                S.la.d[:] = 0
                S.attach(list(rt.net_names), noff, arr6)
            PRE.pop("h_Dr", None); PRE.pop("h_arr6r", None); PRE.pop("h_offr", None)
            rt2 = types.SimpleNamespace(net_names=rt.net_names, offsets=noff, x0=cols[0], y0=cols[1], z0=cols[2], x1=cols[3], y1=cols[4], z1=cols[5])
            o2, rcd2 = o, rcd; r0 = mk(o2, rcd2)
            W["skip_fp"] = True
        else:
            choice, dp = DS.assign_v2_cpp(P, "numpy", a.dw, a.pw, V1.DEFAULT_VIA_COST, a.lam)
            W["dp_s"] = time.perf_counter() - t
            t = time.perf_counter()
            noff, cols, nchg = assemble(P, choice, rt, IR, V1, RE)
            W["assemble_s"] = time.perf_counter() - t
            t = time.perf_counter()
            rt2, arr6, o2, rcd2, r0 = eval_cols(noff, cols)
        W["attach_timer_s"] = time.perf_counter() - t
    W["singlepass_s"] = time.perf_counter() - tpost; TM("single_end")
    if a.t_prof: W["t_marks"] = [(n_, round(t2_ - TMK[0][1], 4)) for n_, t2_ in TMK]; W["t_ht"] = {k: v for k, v in HT.items() if isinstance(v, (int, float))}
    if B8["on"]: B8L.after_check(S)                        # big8: attributes of the StaDesign not read after the check
    nn_ = len(S.SD.net_names)

    def crit_h(o_, rcd_, npaths, slack_max):
        """item 4b: csta_hcrit -> the three fields weights() reads (npaths, poff, net, ok)"""
        import ctypes
        L_ = K.L
        if not getattr(L_, "_h_set", False):
            L_.csta_hcrit.restype = ctypes.c_int64; L_.csta_hcrit.argtypes = [ctypes.c_int64, ctypes.c_double] + [ctypes.c_void_p] * 6
            L_.csta_hcrit_n.restype = ctypes.c_int64; L_.csta_hcrit_get.argtypes = [ctypes.c_void_p] * 3; L_._h_set = True
        ctot = np.ascontiguousarray(rcd_["ctot"].max(1)); rd = np.ascontiguousarray(o_["net_rd_kohm"])
        st = np.zeros(4); p = lambda x: x.ctypes.data_as(ctypes.c_void_p)
        AT = np.ascontiguousarray(o_["AT"]); BE = np.ascontiguousarray(o_["BEST"]); RM = np.ascontiguousarray(o_["rmin"])
        npth = L_.csta_hcrit(npaths, slack_max, p(AT), p(BE), p(RM), p(ctot), p(rd), p(st))
        ne = L_.csta_hcrit_n()
        poff = np.empty(npth + 1, np.int64); net = np.empty(ne, np.int64); ok = np.empty(ne, np.uint8)
        L_.csta_hcrit_get(p(poff), p(net), p(ok))
        return dict(npaths=npth, poff=poff, net=net, ok=ok)

    def weights(o_, rcd_):
        t_ = time.perf_counter()
        cr_ = crit_h(o_, rcd_, a.crit_n, 0.0) if H["crit"] else K.crit(o_, rcd_, a.crit_n, 0.0)
        okm = cr_["ok"].astype(bool) & (cr_["net"] >= 0) & (cr_["net"] < nn_)
        cnt = np.bincount(cr_["net"][okm], minlength=nn_).astype(float)
        on = np.zeros(nn_, bool)
        if cr_["npaths"] > 0:
            w0 = cr_["net"][cr_["poff"][0]:cr_["poff"][1]]; w0 = w0[(w0 >= 0) & (w0 < nn_)]; on[w0] = True
        HT["weights_s"] = HT.get("weights_s", 0.0) + time.perf_counter() - t_
        if H["check"] and H["crit"]:
            c0_ = K.crit(o_, rcd_, a.crit_n, 0.0)
            hchk("crit", c0_["npaths"] == cr_["npaths"] and np.array_equal(c0_["poff"], cr_["poff"]) and np.array_equal(c0_["net"], cr_["net"])
                 and np.array_equal(c0_["ok"], cr_["ok"]), f"{cr_['npaths']} paths")
        return abs(S.w2) * cnt / S.nend + abs(S.w1) * on

    import fast_norm as FN
    RETIME = []
    L5H = {}                                       # late5: drivers set at the region start (tools/late5/l5.py)

    def retime_full(tag):
        t0 = time.perf_counter()
        arr = S.timer_array(); rc_ = K.build_rc(arr.astype(np.int64))
        o_ = K.analyze(rc_["c1"], rc_["c2"], rc_["rpi"], rc_["ctot"], rc_["elm"], rc_["par"])
        r_ = mk(o_, rc_)
        RETIME.append({"tag": tag, "mode": "full", "s": time.perf_counter() - t0})
        return o_, rc_, r_

    def retime_inc(tag):
        """== tools/engine/chain_inc.py retime_inc: rows of the nets changed since the kernel's state -> subset RC -> update"""
        t0 = time.perf_counter()
        if getattr(K, "_tid_checked", None) is not S.tid:
            v = S.tid[S.tid >= 0]
            if len(np.unique(v)) != len(v):
                raise RuntimeError("chain_h: several route nets map to one timer net; incremental re-time not valid")
            K._tid_checked = S.tid
        new, snap = S.new, K.snap
        ch = [i for i, sg in new.items() if snap.get(i) is not sg] + [i for i in snap if i not in new]
        n_same_ = 0
        if a.n_same:                                       # speed9: a net whose rows equal the kernel's has nothing to re-time (same timer input)
            ch0_ = ch; ch = []; sg_ = S.segs
            for i in ch0_:
                s1_ = new.get(i); s0_ = snap.get(i)
                if s1_ is not None and s0_ is not None:
                    if s1_ != s0_:
                        ch.append(i)
                elif s1_ is not None:
                    if s1_ != sg_[i]:
                        ch.append(i)
                elif s0_ != sg_[i]:
                    ch.append(i)
            n_same_ = len(ch0_) - len(ch)
        cache = S.__dict__.setdefault("_f3x_nc", {})
        stale = [i for i in ch if i in new and (cache.get(i) or (None,))[0] is not new[i]]
        if stale:
            for i, b in L5H.get("norm", FN._normalise)(S, stale).items():
                cache[i] = (new[i], b)
        off = S.segs.off
        parts = [cache[i][1] if i in new else S.base_arr[off[i]:off[i + 1]] for i in ch]
        arr = np.concatenate(parts, 0) if parts else np.zeros((0, 7), np.int64)
        sub = arr.astype(np.int64, copy=True)
        sub[:, 0] = S.tid[arr[:, 0]] if len(arr) else sub[:, 0]
        sub = sub[sub[:, 0] >= 0]
        nets = np.unique(S.tid[np.asarray(ch, np.int64)]) if ch else np.zeros(0, np.int64)
        nets = nets[nets >= 0]
        t1 = time.perf_counter()
        K.begin(); K.build_rc_subset(sub, nets)
        o_ = K.update(nets); t2 = time.perf_counter()
        K.pending_snap = dict(new)
        rc_ = K.rc
        r_ = mk(o_, rc_)
        RETIME.append({"tag": tag, "mode": "inc", "s": time.perf_counter() - t0, "select_s": t1 - t0, "update_s": t2 - t1,
                       "nets": int(len(nets)), "same": n_same_})
        if H["check"]:
            arrf = S.timer_array(); rcf = K.build_rc(arrf.astype(np.int64))
            orf = PRE["Kref"].analyze(rcf["c1"], rcf["c2"], rcf["rpi"], rcf["ctot"], rcf["elm"], rcf["par"])
            rrf = mk(orf, rcf)
            bv = lambda x: np.ascontiguousarray(x).view(np.uint64 if np.asarray(x).dtype.itemsize == 8 else np.uint8).ravel()
            nm_ = lambda x, y: int((bv(x) != bv(y)).sum())
            rcm = sum(nm_(rc_[k], rcf[k]) for k in ("c1", "c2", "rpi", "ctot", "par", "elm"))
            om = sum(nm_(o_[k], orf[k]) for k in ("AT", "SL", "gate_d", "wire_d", "BEST", "net_rd", "rmin", "ep_sl_all"))
            hchk("inc_" + tag, rcm == 0 and om == 0 and r_["score"] == rrf["score"], f"rc_mis {rcm} arr_mis {om} nets {len(nets)}")
        return o_, rc_, r_

    def retime(tag):
        return retime_inc(tag) if H["inc"] else retime_full(tag)

    def crit_round(cur):
        o_, rcd_, r_ = cur
        t_ = time.perf_counter()
        wts = weights(o_, rcd_)
        undo, st = prom_crit(S, r_, a.round_nets, a.tw_scale, PRE["fl"], wts)
        orr, rcdr, r1 = retime("crit")
        keep = r1["score"] < r_["score"] - 1e-9
        if not keep:
            S.revert(undo)
            if H["inc"]:
                K.reject()
            dcheck("crit round (reverted)")
            return False, cur, len(undo), time.perf_counter() - t_
        if H["inc"]:
            K.commit()
        dcheck("crit round (kept)")
        return True, (orr, rcdr, r1), len(undo), time.perf_counter() - t_

    def power_round(cur):
        o_, rcd_, r_ = cur
        t_ = time.perf_counter()
        wts3 = weights(o_, rcd_)
        kp = np.asarray(S.kpn, np.float64)
        pwn = S.w3 * kp
        cc = np.where(kp > 0, kp * np.asarray(r_["net_ctot"], np.float64), 0.0)
        cand3 = np.nonzero(wts3 > 0)[0]
        o1 = cand3[np.argsort(-wts3[cand3], kind="stable")][:a.round_nets]
        seen = np.zeros(nn_, bool); seen[o1] = True
        cp = np.nonzero((cc > 0) & ~seen)[0]
        o2_ = cp[np.argsort(-cc[cp], kind="stable")][:a.pw_nets]
        undo2, st2 = prom_pw(S, r_, a.round_nets, a.tw_scale, PRE["fl"], wts3, pwn=pwn, order=np.concatenate([o1, o2_]))
        o3, rcd3, r3 = retime("power")
        keep2 = r3["score"] < r_["score"] - 1e-9
        if not keep2:
            S.revert(undo2)
            if H["inc"]:
                K.reject()
        dcheck("power round (%s)" % ("kept" if keep2 else "reverted"))
        return keep2, undo2, r3, time.perf_counter() - t_

    # ───────── chain_i after-stages (pieces c, d, e) ─────────
    import ctypes as _ct
    PWS = {}                                       # state after power_round_i: "cur" (o, rcd, r), "wts", "settled", "info"
    Cz_ = np.asarray(S.la.C, np.float64)[1:] if hasattr(S.la, "C") else np.zeros(0)
    Cz_ = Cz_[Cz_ > 0]
    RHO = float(Cz_.min() / Cz_.max()) if len(Cz_) else 1.0

    def est5(r_):
        return {k: float(r_[k]) for k in ("wns", "tns", "of", "pw", "score")}

    def gate(undo, r_base, tag):
        """one exact re-time of the current state; kept iff the estimate falls, else S.revert(undo).
        With the incremental timer the kernel is settled afterwards (commit / reject). -> (keep, (o, rc, r) of the candidate)"""
        o_c, rc_c, r_c = retime(tag)
        keep = r_c["score"] < r_base["score"] - 1e-9
        if keep:
            if H["inc"]:
                K.commit()
        else:
            S.revert(undo)
            if H["inc"]:
                K.reject()
        return keep, (o_c, rc_c, r_c)

    def backward():
        """inc timer: RAT / net slack of the kernel's committed state (w3_inc_backward); the net_slack view is then current"""
        stb = np.zeros(2); K.L.w3_inc_backward(stb.ctypes.data_as(_ct.c_void_p))
        return np.array(K.V["net_slack"], np.float64, copy=True)

    def slack_now(cur_):
        """net slack of the state cur_ (== the kernel's committed state with --h_inc)"""
        if H["inc"]:
            return backward()
        if "net_slack" not in cur_[0]:             # stage output with its inc timer and no later full analysis
            rc_c = cur_[1]
            o_n = K.analyze(rc_c["c1"], rc_c["c2"], rc_c["rpi"], rc_c["ctot"], rc_c["elm"], rc_c["par"])
            return np.asarray(o_n["net_slack"], np.float64)
        return np.asarray(cur_[0]["net_slack"], np.float64)

    def power_round_i(cur):
        """chain_h's power round + --pw_skip, --pw_batch (bisection), --pw_repair (unbatched), --pw2_nets. Leaves the kernel
        settled (PWS["settled"]) and the final state in PWS["cur"]."""
        o_, rcd_, r_ = cur
        t_ = time.perf_counter()
        info = {}; PWS.clear(); PWS["info"] = info
        if S.kpn is None:                          # chain_gb L2: no activity file -> no power price
            info["skipped"] = "no activity file (S.kpn is None)"
            PWS["cur"] = cur; PWS["settled"] = True; PWS["wts"] = weights(o_, rcd_)
            return False, {}, r_, time.perf_counter() - t_
        wts3 = weights(o_, rcd_)
        PWS["wts"] = wts3
        kp = np.asarray(S.kpn, np.float64)
        pwn = S.w3 * kp
        cc = np.where(kp > 0, kp * np.asarray(r_["net_ctot"], np.float64), 0.0)
        cand3 = np.nonzero(wts3 > 0)[0]
        o1 = cand3[np.argsort(-wts3[cand3], kind="stable")][:a.round_nets]
        seen = np.zeros(nn_, bool); seen[o1] = True
        cp = np.nonzero((cc > 0) & ~seen)[0]
        rank = cp[np.argsort(-cc[cp], kind="stable")]
        o2_ = rank[:a.pw_nets]
        allr = np.concatenate([o1, o2_])
        info.update(n_crit=int(len(o1)), n_pw=int(len(o2_)), n_pw_cand=int(len(cp)))
        if a.pw_skip is not None:
            ub = abs(S.w3) * float(cc[allr].sum()) * (1.0 - RHO)
            info["ub_power"] = ub; info["rho"] = RHO
            if ub < a.pw_skip:
                info["skipped"] = True
                PWS["cur"] = cur; PWS["settled"] = True
                return False, {}, r_, time.perf_counter() - t_
        sl0 = slack_now(cur) if (a.pw_repair and a.pw_batch <= 0) else None
        undo_all = {}; cb = cur; anykeep = False
        if a.pw_batch > 0:                         # e(i): batches of N with bisection (micro_chain.try_batch)
            info["batches"] = []

            def try_batch(cb_, nets_, depth):
                if len(nets_) == 0:
                    return cb_, 0
                u_, _ = prom_pw(S, cb_[2], len(nets_), a.tw_scale, PRE["fl"], wts3, pwn=pwn, order=np.asarray(nets_))
                k_, c_ = gate(u_, cb_[2], "power_b")
                info["batches"].append([int(len(nets_)), int(depth), bool(k_), float(c_[2]["score"])])
                if k_:
                    for i_, sg_ in u_.items():
                        undo_all.setdefault(i_, sg_)
                    return c_, len(u_)
                kept_ = 0
                if len(u_) > 0 and depth < a.pw_bisect_depth and len(nets_) >= 2 * a.pw_bisect_min:
                    h_ = len(nets_) // 2
                    cb_, k1 = try_batch(cb_, nets_[:h_], depth + 1)
                    cb_, k2 = try_batch(cb_, nets_[h_:], depth + 1)
                    kept_ = k1 + k2
                return cb_, kept_

            s_ = 0; n1_ = len(o1); bs_ = a.pw_batch
            while s_ < len(allr):                          # speed10: while (was: for, same batches when --t_pw_stop / --t_pw_stop2 are 0)
                if not DLG("pw_batch"):                    # merge6 (b): deadline -> no further power batch
                    info["deadline_stop_at"] = int(s_)
                    break
                c0_ = cb
                e_ = min(s_ + bs_, n1_) if (s_ < n1_ and a.t_pw_grow) else s_ + bs_   # speed10: --t_pw_grow: a list's batches double while they are kept
                cb, _ = try_batch(cb, allr[s_:e_], 0)
                DLE("pw_batch")
                g_ = c0_[2]["score"] - cb[2]["score"]; s_ = e_
                if a.t_pw_grow:
                    bs_ = 2 * bs_ if (cb is not c0_ and s_ != n1_) else a.pw_batch
                if a.t_pw_stop > 0 and s_ < n1_ and g_ < a.t_pw_stop:   # speed10: rest of the critical-net list skipped; the power nets keep their batches
                    info["t_pw_stop_at"] = int(s_); s_ = n1_
                elif a.t_pw_rej and s_ < n1_ and (not info["batches"] or c0_ is cb):   # speed10: the same at the first critical-net batch that kept nothing
                    info["t_pw_stop_at"] = int(s_); s_ = n1_
                elif a.t_pw_stop2 > 0 and s_ >= n1_ + a.pw_batch and g_ < a.t_pw_stop2:
                    info["t_pw_stop2_at"] = int(s_); break
            anykeep = cb is not cur
        else:
            undo2, _ = prom_pw(S, r_, a.round_nets, a.tw_scale, PRE["fl"], wts3, pwn=pwn, order=allr)
            o3, rcd3, r3 = retime("power")
            keep2 = r3["score"] < r_["score"] - 1e-9
            info["round1"] = dict(est5(r3), kept=bool(keep2))
            if keep2:
                if H["inc"]:
                    K.commit()
                cb = (o3, rcd3, r3); undo_all = dict(undo2); anykeep = True
            elif a.pw_repair:                       # e(iii): undo the power-list nets whose slack fell below min(before, 0)
                if H["inc"]:
                    K.commit(); sl3 = backward()    # kernel = candidate
                else:
                    sl3 = np.asarray(o3["net_slack"], np.float64)
                bad = {}
                for n in o2_.tolist():
                    i = S.ridx.get(S.SD.net_names[n])
                    if i is not None and i in undo2 and sl3[n] < min(sl0[n], 0.0) - 1e-12:
                        bad[i] = undo2[i]
                info["repair_reverted"] = len(bad)
                rest = {k_: v_ for k_, v_ in undo2.items() if k_ not in bad}
                keep4 = False
                if bad:
                    S.revert(bad)
                    o4, rcd4, r4 = retime("power_repair")
                    keep4 = r4["score"] < r_["score"] - 1e-9
                    info["repair"] = dict(est5(r4), kept=bool(keep4))
                    if keep4:
                        if H["inc"]:
                            K.commit()
                        cb = (o4, rcd4, r4); undo_all = rest; anykeep = True
                    elif H["inc"]:
                        K.reject()
                if not keep4:
                    S.revert(rest)
                    if H["inc"]:                     # kernel back to the pre-round state (same values, settled)
                        retime("power_restore"); K.commit()
            else:
                S.revert(undo2)
                if H["inc"]:
                    K.reject()
        if a.pw2_nets > 0:                         # e(ii): second power round on the next N power nets (chain_gb L4)
            o2b = rank[a.pw_nets:a.pw_nets + a.pw2_nets]
            if not anykeep and not H["inc"] and cb is cur:   # kernel's last analyze was a rejected state
                rc_c = cur[1]
                ob = K.analyze(rc_c["c1"], rc_c["c2"], rc_c["rpi"], rc_c["ctot"], rc_c["elm"], rc_c["par"])
                cb = (ob, rc_c, r_)
            wtsb = weights(cb[0], cb[1])
            ub_, _ = prom_pw(S, cb[2], a.round_nets, a.tw_scale, PRE["fl"], wtsb, pwn=pwn, order=o2b)
            k5, c5 = gate(ub_, cb[2], "power2")
            info["round2"] = dict(est5(c5[2]), kept=bool(k5), n=int(len(o2b)), promoted=len(ub_))
            if k5:
                for i_, sg_ in ub_.items():
                    undo_all.setdefault(i_, sg_)
                cb = c5; anykeep = True
        PWS["cur"] = cb; PWS["settled"] = True
        dcheck("power round i (%s)" % ("kept" if anykeep else "reverted"))
        return anykeep, undo_all, cb[2], time.perf_counter() - t_

    def ovf_order(n_max, slack_t, slack_min):
        """chain_t3.ovf_order + --ovfround_open: route nets ranked by the overflow cost their own wire demand adds on cells
        with d > c (official-rule cost, marginal of 2 demand units per wire GCell, vias ignored)"""
        if a.late5:                                  # late5: marginal cost on the cells where it is non-zero + C++ score
            r5_ = L5.ovf_order(S, n_max, slack_t, slack_min, a.ovfround_open)
            if r5_ is not None:
                return r5_
        import repair3d_astar as RA
        OL = G["OL"]
        la = S.la; L = la.L
        u = la.u[:, None, None]
        c0 = RA._cost(u, la.d, la.c); c1 = RA._cost(u, np.maximum(la.d - 2.0, 0.0), la.c)
        msk = la.d > la.c
        if a.ovfround_open:                          # the bsg fault: blocked cells (capacity 0) are not re-decidable
            msk = msk & (la.c > 1e-3)
        Hm = np.where(msk, c0 - c1, 0.0); Hm[0] = 0.0
        arr = OL.cur_arr(S, wires_only=True)
        sc = np.zeros(len(S.nets.names), np.float64)
        for z in range(1, L):
            m = arr[:, 3] == z
            if not m.any() or not Hm[z].any():
                continue
            w = arr[m]
            if la.dirs[z] == 0:
                P_ = np.zeros((la.X + 1, la.Y)); P_[1:] = np.cumsum(Hm[z], 0)
                a_ = np.minimum(w[:, 1], w[:, 4]); b_ = np.maximum(w[:, 1], w[:, 4]); f_ = w[:, 2]
                v = P_[b_ + 1, f_] - P_[a_, f_]
            else:
                P_ = np.zeros((la.X, la.Y + 1)); P_[:, 1:] = np.cumsum(Hm[z], 1)
                a_ = np.minimum(w[:, 2], w[:, 5]); b_ = np.maximum(w[:, 2], w[:, 5]); f_ = w[:, 1]
                v = P_[f_, b_ + 1] - P_[f_, a_]
            np.add.at(sc, w[:, 0], v)
        ri = np.nonzero(sc > 0)[0]
        ri = ri[np.argsort(-sc[ri], kind="stable")]
        tid = S.tid[ri]
        ok = tid >= 0
        if slack_min is not None:
            ok &= np.asarray(slack_t, np.float64)[np.maximum(tid, 0)] >= slack_min
        ri, tid = ri[ok][:n_max], tid[ok][:n_max]
        return tid.astype(np.int64), ri, sc

    def ovf_round(cur_, wts, tag):
        """chain_t3.ovf_round (d) with the chain_i gate + --ovfround_skip"""
        rec = {"stage": tag}
        t0 = time.perf_counter()
        sl_ = slack_now(cur_)
        order, ri, sc = ovf_order(a.ovfround, sl_, a.ovfround_slack if a.ovfround_slack >= -1e8 else None)
        rec["cand_nets"] = int(len(order)); rec["order_s"] = time.perf_counter() - t0
        rec["own_cost_top"] = float(sc[ri].sum()) if len(ri) else 0.0
        rec["bound"] = float(S.w4) * rec["own_cost_top"]
        if len(order) == 0 or (a.ovfround_skip > 0 and rec["bound"] < a.ovfround_skip):
            rec["kept"] = False; rec["skipped"] = len(order) > 0; rec["s"] = time.perf_counter() - t0
            log.append(rec); return cur_
        kp = np.asarray(S.kpn, np.float64) if S.kpn is not None else np.zeros(len(wts))
        t1 = time.perf_counter()
        undo, st = prom_pw(S, cur_[2], len(order), a.tw_scale, PRE["fl"], wts, pwn=S.w3 * kp, order=order)
        rec["round_s"] = time.perf_counter() - t1; rec["changed"] = len(undo)
        keep = False
        if undo:
            t2 = time.perf_counter()
            keep, c2 = gate(undo, cur_[2], "ovfround")
            rec["cand_est"] = est5(c2[2]); rec["retime_s"] = time.perf_counter() - t2
            if keep:
                cur_ = c2
        rec["kept"] = bool(keep); rec["s"] = time.perf_counter() - t0; rec["est"] = est5(cur_[2])
        log.append(rec)
        print("[ovfround] " + json.dumps(rec), flush=True)
        return cur_

    def after_stages(kept2, r3):
        """after the final power round, before the write: --post_crit rounds, then --ovfround (chain_t3's order)"""
        cur_ = PWS["cur"]
        if a.rs != "off" and a.rs_at == "power" and DLG("reshape"):   # chain_j c: reshape stage right after the power round
            cur_ = rs_run(cur_); DLE("reshape")
        if a.post_crit > 0 and not a.t_no_post:
            t_ = time.perf_counter()
            if not H["inc"] and not kept2:          # K.crit reads the kernel's LAST analyze: make it the current state
                rc_c = cur_[1]
                o_r = K.analyze(rc_c["c1"], rc_c["c2"], rc_c["rpi"], rc_c["ctot"], rc_c["elm"], rc_c["par"])
                cur_ = (o_r, rc_c, cur_[2])
            for j in range(a.post_crit):
                if not DLG("post_crit"):            # merge6 (b)
                    break
                kp_, cur_, nprom, dt = crit_round(cur_)
                DLE("post_crit")
                log.append({"stage": f"post_crit_{j + 1}", "kept": bool(kp_), "promoted": nprom, "s": dt, "est": est5(cur_[2])})
                if not kp_:
                    break
            W["post_crit_s"] = time.perf_counter() - t_
        if a.ovfround > 0 and a.ovf != "off" and DLG("ovf_round"):
            t_ = time.perf_counter()
            cur_ = ovf_round(cur_, PWS["wts"], "ovf_round_1")
            W["ovf_round_s"] = time.perf_counter() - t_
            DLE("ovf_round")
        if a.rs != "off" and a.rs_at == "end" and DLG("reshape"):     # chain_j c: reshape stage after the other after-stages
            cur_ = rs_run(cur_); DLE("reshape")
        return cur_

    def rs_run(cur_):
        """chain_j c: tools/reshape/chain_r.py reshape_stage on the settled state cur_ (kernel committed == cur_ with --h_inc)"""
        t_rs = time.perf_counter()
        if not H["inc"]:
            sys.exit("chain_j.py: --rs on needs the incremental timer (--h_inc 1 or --h_all 1)")
        cur_ = reshape_stage(S, cur_, a, log, K, mk, weights, retime, H, PRE)
        W["rs_stage_s"] = W.get("rs_stage_s", 0.0) + time.perf_counter() - t_rs
        return cur_

    PWX = bool(a.pw_batch > 0 or a.pw2_nets > 0 or a.pw_skip is not None or a.pw_repair or a.post_crit > 0 or a.ovfround > 0
               or a.rs != "off")

    def wr(tag):
        t_ = time.perf_counter()
        p = f"{a.out}.{tag}.route"
        if H["write"]:                                     # item 2
            import h_write
            h_write.write_par(S, rt, noff, cols, arr6, p, HT.setdefault("write", {}), H["write"])
            if H["check"]:
                import filecmp
                write_cpp(S, rt, noff, cols, arr6, IR, p + ".ref", {})
                hchk("write", filecmp.cmp(p, p + ".ref", shallow=False)); os.remove(p + ".ref")
        else:
            write_cpp(S, rt, noff, cols, arr6, IR, p, {})
        return p, time.perf_counter() - t_

    def dcheck(tag):
        if SHADOW is None:
            return
        ok = np.array_equal(S.la.d, SHADOW.d)
        CHECK["demand_rounds"] += 1; CHECK["demand_bad"] += not ok
        print(f"[check] after {tag}: la.d equal to the original apply/commit_net = {ok} (sum {S.la.d.sum():.0f})", flush=True)

    if SHADOW is not None:                         # mirror from here on: every apply/commit_net also runs the original code
        SHADOW.d[...] = S.la.d; SHADOW.la = S.la
    cur = (o2, rcd2, r0)
    class _P7Log(list):                            # par7: every entry stamped [wall, process CPU, promote calls, re-times]
        def append(self, e):
            if isinstance(e, dict):
                pf_ = L5H.get("pf"); ot_ = os.times()
                e["p7_t"] = [time.perf_counter(), ot_.user + ot_.system, len(pf_.TIMES) if pf_ is not None else 0, len(RETIME)]
            super().append(e)
    log = _P7Log(); log.append({"stage": "single_pass", "est": {k: float(r0[k]) for k in ("wns", "tns", "of", "pw", "score")}})
    ot_ = os.times(); W["p7_t_singlepass"] = [tpost, W.get("p7_cpu_post", 0.0), time.perf_counter(), ot_.user + ot_.system]
    if B5["on"]: log = B5L.at_stages(log, S, _Segs, W, BIGM)   # big5: stage marks; big mode: C++ revert
    # ───────── dlchunk: deadline mode, checked chunks before the overflow stage (docstring; --dl_chunks 0 does not enter) ─────────
    DLC = {"kept": 0}

    def dl_chunks_run(cur_):
        rec = {"stage": "dl_chunks", "K": a.dl_chunks, "nets": a.dl_chunk_nets, "tw": a.dl_chunk_tw, "est_in": est5(cur_[2]), "chunks": []}
        t0_ = time.perf_counter()
        if S.kpn is None:                                  # no power price: the arm's score is undefined
            rec["skipped"] = "no activity"; log.append(rec)
            return cur_
        pw_x = prom_pw; rv0_ = S.revert
        if a.late5:                                        # the late region's drivers (same results; chain_n binds them after the overflow stage)
            sys.path.append(str(ROOT / "tools" / "late5")); import l5 as L5x
            S.revert = L5x.make_revert(S, _Segs, rv0_)
            L5H.setdefault("norm", L5x.normalise)
            if a.n_norm:
                sys.path.append(str(ROOT / "tools" / "best5")); import n_norm as NNx
                L5H["norm"] = NNx.normalise
            if PROM is not None and os.environ.get("L5_PF", "1") == "1":
                if a.par7:
                    sys.path.append(str(ROOT / "tools" / "par7")); import p7_pf as PFx
                    PFx.PAR[0] = int(a.par7)
                else:
                    import l5_pf as PFx
                pw_x = PFx.promote_pw
        wts_ = weights(cur_[0], cur_[1])
        kp_ = np.asarray(S.kpn, np.float64); pwn_ = S.w3 * kp_
        cc_ = np.where(kp_ > 0, kp_ * np.asarray(cur_[2]["net_ctot"], np.float64), 0.0)
        tid_ = np.asarray(S.tid, np.int64)                 # the order: tools/gnn1x/micro1x.py save_labels3 + tools/teach3/ds3.py + infer1x.py arm Ht4k, same types (float32 rows)
        sd2r_ = -np.ones(nn_, np.int64); ok_ = (tid_ >= 0) & (tid_ < nn_); sd2r_[tid_[ok_]] = np.nonzero(ok_)[0]
        cp_ = np.nonzero(cc_ > 0)[0]; cp_ = cp_[np.argsort(-cc_[cp_], kind="stable")][:a.pw_nets]
        u_r = sd2r_[np.union1d(np.nonzero(wts_ > 0)[0], cp_)]; u_r = np.unique(u_r[u_r >= 0])
        sd_ = tid_[u_r]; P_ = abs(S.w2) / max(S.nend, 1) if S.w2 else 1.0
        t32 = wts_[sd_].astype(np.float32); c32 = (cc_[sd_].astype(np.float32).astype(np.float64) * S.w3 / P_).astype(np.float32)
        sc_ = t32 / max(t32.max(), 1e-30) + c32 / max(c32.max(), 1e-30) if len(sd_) else np.zeros(0)
        order_ = sd_[np.argsort(-sc_, kind="stable")]
        rec["U"] = int(len(order_)); rec["order_s"] = time.perf_counter() - t0_
        fresh_ = True
        for c_i in range(a.dl_chunks):
            nets_ = np.ascontiguousarray(order_[c_i * a.dl_chunk_nets:(c_i + 1) * a.dl_chunk_nets], np.int64)
            if len(nets_) == 0:
                break
            now_ = _dl_now(); e_ = DL["last"].get("dl_chunk", DL["est"].get("dl_chunk", 0.0))
            if now_ + e_ > DL["X"]:                        # this unit does not fit; the deadline is not expired by it
                rec["stop"] = [c_i, round(now_, 3), round(float(e_), 3)]
                break
            DL["cur"]["dl_chunk"] = now_
            t_c = time.perf_counter()
            if not fresh_:
                wts_ = weights(cur_[0], cur_[1])
            u_, st_ = pw_x(S, cur_[2], len(nets_), a.dl_chunk_tw, PRE["fl"], wts_, pwn=pwn_, order=nets_)
            t_d = time.perf_counter()
            k_, c_ = gate(u_, cur_[2], "dl_chunk")
            if k_:
                cur_ = c_; DLC["kept"] += 1
            elif H["inc"]:                                 # rejected: the kept state's required-time rows (micro engine's fresh_crit)
                K.flush_pending()
            fresh_ = not k_
            DLE("dl_chunk"); dcheck("dl chunk")
            rec["chunks"].append({"nets": int(len(nets_)), "changed": len(u_), "kept": bool(k_), "est": est5(c_[2]), "promote_s": t_d - t_c, "s": time.perf_counter() - t_c})
        S.revert = rv0_
        rec["s"] = time.perf_counter() - t0_; rec["est"] = est5(cur_[2]); log.append(rec)
        print("[dlchunk] " + json.dumps({k: rec[k] for k in ("U", "s", "chunks", "est")}), flush=True)
        return cur_

    def dl_settle(cur_):
        """a chunk was kept: state for the overflow stage = what the check leaves when it picks the prior (RC copies, required-time pass)"""
        rc_ = {k: np.array(v, copy=True) for k, v in K.rc.items()}
        if a.mid5 and M5 is not None and H["inc"]:
            o_ = M5.gate_backward(K)
        else:
            o_ = K.analyze(K.rc["c1"], K.rc["c2"], K.rc["rpi"], K.rc["ctot"], K.rc["elm"], K.rc["par"])
        return (o_, rc_, cur_[2])

    if a.dl_chunks > 0 and DL["on"] and not BIGM:
        cur = dl_chunks_run(cur)
    # ───────── dgr1x: the descent as a stage ─────────
    def dgr_run(cur_, at):
        """--dgr K at one of three places: one exact test of the whole rounded change (the chain's gate)"""
        import h_dem as HD_
        t_ = time.perf_counter()
        rv0_ = S.revert
        if a.late5 and at != "end":
            sys.path.append(str(ROOT / "tools" / "late5")); import l5 as L5x
            L5H.setdefault("norm", L5x.normalise)
            if a.n_norm:
                sys.path.append(str(ROOT / "tools" / "best5")); import n_norm as NNx
                L5H["norm"] = NNx.normalise
        DGS.wrap_revert(S)
        a._dgr_Cz = np.asarray(S.la.C, np.float64)
        now_ = None; dl_ = None
        tw_ = float(DL["est"].get("dgr_write", 0.0)); te_ = float(DL["est"].get("dgr_test", 0.0)); tb_ = float(DL["est"].get("dgr_build", 0.0))
        if DL["on"]:
            now_ = _dl_now; dl_ = DL["X"]
            if _dl_now() + tb_ + te_ + tw_ > dl_:
                rec_ = {"stage": "dgr", "at": at, "skipped": "deadline", "now": round(_dl_now(), 3)}
                log.append(rec_); print("[dgr1x] " + json.dumps(rec_), flush=True)
                S.revert = rv0_
                return cur_
            DL["cur"]["dgr"] = _dl_now()
        c_ = cur_
        for rn_ in range(max(1, a.dgr_rounds)):            # further rounds only while the last round's seconds still fit (dl) and the last round gained
            if rn_ and DL["on"] and _dl_now() + float(rec_.get("stage_s", 0.0)) + tw_ > dl_:
                break
            t_ = time.perf_counter()
            if a.d2 >= 0:
                sys.path.insert(0, str(ROOT / "tools" / "dgr2d")); import d2_stage as D2S
                k_, c_, rec_ = D2S.run(S, c_, a, cd, gate, FT, HD_, DLnow=now_, deadline=dl_, write_est=tw_, test_est=te_)
            else:
                k_, c_, rec_ = DGS.run(S, c_, a, cd, gate, FT, HD_, DLnow=now_, deadline=dl_, write_est=tw_, test_est=te_)
            if not k_ and H["inc"] and rec_.get("tries"):  # rejected: the kept state's required-time rows
                K.flush_pending()
            rec_["at"] = at; rec_["round"] = rn_; rec_["stage_s"] = time.perf_counter() - t_; rec_["est"] = est5(c_[2])
            log.append(rec_); W["dgr_s"] = W.get("dgr_s", 0.0) + rec_["stage_s"]
            print("[dgr1x] " + json.dumps(rec_, default=float), flush=True)
            if k_: DLC["kept"] += 1
            if not k_ or rec_.get("gain", 0.0) < a.dgr_round_min:
                break
            cur_ = c_
        DGS.end()
        S.revert = rv0_
        DLE("dgr")
        return c_

    if a.dgr and os.environ.get("DGR_SWEEP") and a.dgr_at == "fp":      # development: settings from one state (tools/best7/dgr_stage.py sweep), no route written
        import h_dem as HD_
        rv0_ = S.revert
        if a.late5:
            sys.path.append(str(ROOT / "tools" / "late5")); import l5 as L5x
            S.revert = L5x.make_revert(S, _Segs, rv0_)
            L5H.setdefault("norm", L5x.normalise)
            if a.n_norm:
                sys.path.append(str(ROOT / "tools" / "best5")); import n_norm as NNx
                L5H["norm"] = NNx.normalise
        a._dgr_Cz = np.asarray(S.la.C, np.float64)
        DGS.sweep(S, cur, a, cd, retime, K, FT, HD_, os.environ["DGR_SWEEP_OUT"])
        sys.stdout.flush(); os._exit(0)
    DGR_DEFER = bool(a.dgr and a.dgr_at == "fp" and a.dgr_defer and float(cur[2]["wns"]) >= 0.0 and float(cur[2]["tns"]) >= 0.0)   # no violation at this state: the descent's main lever is off; the overflow stage goes first
    if a.dgr and a.dgr_at == "fp" and not DGR_DEFER and (not DL["on"] or not DL["expired"]):
        cur = dgr_run(cur, "fp")
    # ───────── OVF3 STAGE (insertion point 2 of 2: after the single pass, before the crit rounds) ─────────
    if a.ovf != "off":
        if os.environ.get("OVF3_SWEEP"):                  # experiment only: several stage settings on one loaded state
            import copy
            new0, d0 = dict(S.new), np.array(S.la.d, copy=True)
            for ov_ in json.loads(os.environ["OVF3_SWEEP"]):
                a2 = copy.copy(a); tag_ = ov_.pop("tag")
                for k_, v_ in ov_.items(): setattr(a2, k_, v_)
                l2 = []; t_ = time.perf_counter()
                overflow_stage(S, cur, a2, W, l2, K, mk)
                l2[-1]["sweep_tag"] = tag_; l2[-1]["sweep_s"] = time.perf_counter() - t_; l2[-1]["stage"] = "ovf_sweep"
                log.append(l2[-1])
                S.new = dict(new0); S.la.d[:] = d0
        if DLG("ovf_stage"):                               # merge6 (b)
            t_ = time.perf_counter()
            if CPU5: CPU5.start("ovf")
            if DLC["kept"]: cur = dl_settle(cur)           # dlchunk
            cur = overflow_stage(S, cur, a, W, log, K, mk)
            W["ovf_stage_s"] = time.perf_counter() - t_
            if CPU5: CPU5.stop("ovf")
            DLE("ovf_stage")
    if a.dgr and (a.dgr_at == "ov" or DGR_DEFER) and (not DL["on"] or not DL["expired"]):
        cur = dgr_run(cur, "ov")
    if M5 is not None: W["mid5_region_s"] = time.perf_counter() - T5_0 - float(W.get("gnn_s") or 0.0)
    if CPU5: W["mid5_cpu"] = CPU5.out()
    if a.mid5: W["mid5_esc"] = dict(M5.ESC); W["mid5_rr"] = dict(M5.RR)
    if os.environ.get("MID5_STOP") == "1" and M5 is not None:   # dev runs: state after the region, then exit
        print("MID5_STATE " + M5.state_md5(S, cur), flush=True)
        print("MID5_W " + json.dumps({k: v for k, v in W.items() if isinstance(v, (int, float, str, dict))}, default=str), flush=True)
        os._exit(0)
    # ───────── end OVF3 STAGE ─────────
    # ───────── ENGINE2 micro engine (tools/engine2/micro2.py; from tools/engine/micro_chain.py) ─────────
    if a.engine == "micro":
        import promote_fast as PF_m
        MC = {"tests": 0, "extra_analyze": 0, "flush": 0, "batches_kept": 0, "batches_rejected": 0, "batches_recovered": 0,
              "sub_kept": 0, "sub_rejected": 0, "sub_skipped_nochange": 0, "nets_tried": 0, "nets_kept": 0, "nets_reapplied": 0,
              "weight_refresh": 0, "refresh_skipped_lazy": 0, "t_tests": 0.0, "t_promote": 0.0, "t_revert": 0.0,
              "t_weights": 0.0, "t_reapply": 0.0, "q_batches": 0, "q_kept": 0, "q_confirm": 0, "q_confirm_fail": 0,
              "q_undone_batches": 0, "t_q_est": 0.0}
        LAST = {"cur": True}                     # kernel's last analyze == current kept state
        INC = bool(H["inc"]) and bool(a.e_inc)
        OF_orig = FM.overflow_cost
        if a.e_ofc:                              # per-cell cached official overflow cost (tools/engine2/ofc.py; same bits)
            if a.of_rule != "official":
                sys.exit("micro2.py: --e_ofc 1 needs --of_rule official")
            import ofc as OFCm
            OFC_ = (OFCm.OFC2 if a.e_ofc == 2 else OFCm.OFC)(S.la, ref=OF_orig, check=bool(a.e_ofc_check))
            FM.overflow_cost = OFC_

        def est_of(r_):
            e = {k: float(r_[k]) for k in ("wns", "tns", "of", "pw", "score")}
            e.update(t_wns=float(S.w1 * (r_["wns"] - S.r1)), t_tns=float(S.w2 * (r_["tns"] - S.r2) / S.nend),
                     t_of=float(S.w4 * r_["of"]), t_pw=float(S.w3 * r_["pw"]))
            return e

        def retime_m():   # chain_j: renamed (chain_i main has retime(tag))
            t_ = time.perf_counter()
            o_, rc_, r_ = retime_inc("m") if INC else retime_full("m")
            MC["tests"] += 1; MC["t_tests"] += time.perf_counter() - t_
            return (o_, rc_, r_)

        def accept():
            if INC:
                K.commit()

        def reject():
            if INC:
                K.reject()

        def wts_of(cur_):
            t_ = time.perf_counter()
            if a.fresh_crit and not LAST["cur"]:
                if INC:                            # rejected candidate's required-time rows -> the kept state's (== a re-analysis)
                    K.flush_pending(); MC["flush"] += 1
                else:
                    rc_ = cur_[1]
                    K.analyze(rc_["c1"], rc_["c2"], rc_["rpi"], rc_["ctot"], rc_["elm"], rc_["par"])
                    MC["extra_analyze"] += 1
                LAST["cur"] = True
            w_ = weights(cur_[0], cur_[1])
            MC["t_weights"] += time.perf_counter() - t_
            return w_

        if a.e_dp == "full":
            import promote_full as PFULL

        def promote(r_, nets_, tw_, fl_, w_, pwn_, pw_):
            """DP of one batch in the given order: Python (promote_fast._promote) or C++ (promote_full._run, == recipe g1's)"""
            if a.e_dp == "full":
                out_ = PFULL._run(S, r_, np.ascontiguousarray(nets_, np.int64), tw_, fl_, w_, pwn_ if pw_ else None, bool(pw_))
                S.la.tw = 0.0
                if pw_:
                    S.la.pn = 0.0
                return out_
            return PF_m._promote(S, r_, len(nets_), tw_, fl_, w_, pwn_, np.asarray(nets_), pw_)

        DBUF = np.empty_like(S.la.d) if a.e_snap == 1 else None

        def seg_rows(ids_, lists):
            ln = [len(x) for x in lists]
            flat = [t for x in lists for t in x]
            ar = np.zeros((len(flat), 7), np.int64)
            if flat:
                ar[:, 1:] = np.asarray(flat, np.int64).reshape(-1, 6)
                ar[:, 0] = np.repeat(np.asarray(ids_, np.int64), ln)
            return ar

        def rows_ok(ar):
            """every index inside the map (then tools/engine2/h_dem.cc hd_add never refuses half-way)"""
            la = S.la
            if not len(ar):
                return True
            x1, y1, z1, x2, y2, z2 = (ar[:, k] for k in range(1, 7))
            return bool(((np.minimum(z1, z2) >= 0) & (np.maximum(z1, z2) <= la.L - 1) & (np.minimum(x1, x2) >= 0) & (np.maximum(x1, x2) < la.X)
                         & (np.minimum(y1, y2) >= 0) & (np.maximum(y1, y2) < la.Y)).all())

        def cswap(ids_, cur_, tgt_):
            """--e_snap 2: demand of nets ids_ from segments cur_ to tgt_ with the C++ adder (exact: integer-valued sums).
            False (nothing changed) when some index is outside the map."""
            import h_dem as HD_
            ro, rn = seg_rows(ids_, [cur_[i] for i in ids_]), seg_rows(ids_, [tgt_[i] for i in ids_])
            if not (rows_ok(ro) and rows_ok(rn)):
                MC["cswap_py"] = MC.get("cswap_py", 0) + 1
                return False
            ok1 = HD_._add(S.la, S.la.d, ro, -1.0); ok2 = ok1 and HD_._add(S.la, S.la.d, rn, 1.0)
            if not (ok1 and ok2):
                raise RuntimeError("micro2.py: hd_add refused rows that passed rows_ok (state no longer exact)")
            return True

        def snap_d():
            if DBUF is not None:
                np.copyto(DBUF, S.la.d)

        def undo_to(undo_):
            """== S.revert(undo_) (fastmode.revert + fastmode3's dirty marks); --e_snap 1: the demand map is restored from the
            copy taken before the batch (demand values are integer-valued, so the copy equals the original's add/subtract)"""
            touch([S.new.get(i, S.segs[i]) for i in undo_], undo_.values())
            if a.e_snap == 2:
                ids_ = list(undo_)
                if not cswap(ids_, {i: S.new.get(i, S.segs[i]) for i in ids_}, undo_):
                    S.revert(undo_); return
            elif DBUF is None:
                S.revert(undo_); return
            else:
                np.copyto(S.la.d, DBUF)
            for i, old in undo_.items():
                if old is S.segs[i]:
                    S.new.pop(i, None)
                else:
                    S.new[i] = old
            S._dirty.update(int(S.tid[i]) for i in undo_)

        def capply(ids_, old_, new_):
            """--e_capply 1: apply the stored assignments of a half with the C++ demand adder (tools/engine2/h_dem.py _add:
            build_demand_fast's per-net contributions, integer-valued, so == timing_promote.apply of the same segments);
            S.new / dirty marks as fastmode.revert sets them. False (state restored from the snapshot) if the adder refuses."""
            import h_dem as HD_
            def rows(lists):
                ln = [len(x) for x in lists]
                flat = [t for x in lists for t in x]
                ar = np.zeros((len(flat), 7), np.int64)
                if flat:
                    ar[:, 1:] = np.asarray(flat, np.int64).reshape(-1, 6)
                    ar[:, 0] = np.repeat(np.asarray(ids_, np.int64), ln)
                return ar
            ro, rn = rows([old_[i] for i in ids_]), rows([new_[i] for i in ids_])
            if not (HD_._add(S.la, S.la.d, ro, -1.0) and HD_._add(S.la, S.la.d, rn, 1.0)):
                np.copyto(S.la.d, DBUF); MC["capply_refused"] = MC.get("capply_refused", 0) + 1
                return False
            for i in ids_:
                S.new[i] = new_[i]
            S._dirty.update(int(S.tid[i]) for i in ids_)
            return True

        def touch(*lists):
            """--e_ofc 2: report segments whose demand was added / removed (lists of per-net segment lists)"""
            if a.e_ofc != 2:
                return
            flat = [t for L_ in lists for sg in L_ for t in sg]
            if flat:
                OFC_.touch(np.asarray(flat, np.int64).reshape(-1, 6))

        def rids(nets_):
            nm_ = S.SD.net_names
            return [S.ridx.get(nm_[int(n)]) for n in nets_]

        def try_batch(cur_, nets_, ctx, depth=0, reuse=None):
            """promote nets_ (SD net ids, in order), exact re-time, keep or undo + bisect. -> (cur, kept nets)
            reuse = (old, new) per route net from the first DP of the parent batch (--bisect_reuse 1): a half is applied
            from the stored assignments instead of a new DP."""
            if len(nets_) == 0:
                return cur_, 0
            wts_, pwn_, pw_, fl_ = ctx
            t_ = time.perf_counter()
            snap_d()
            if reuse is None:
                undo_, st_ = promote(cur_[2], nets_, a.tw_scale, fl_, wts_, pwn_, pw_)
                MC["t_promote"] += time.perf_counter() - t_
                MC["nets_tried"] += len(nets_)
                touch(undo_.values(), [S.new[i] for i in undo_])
                chg_ = [i for i in undo_ if S.new[i] != undo_[i]] if a.skip_nochange else list(undo_)
                if a.skip_nochange and not chg_:   # nothing moved: the state is bit-identical -> the test would reject
                    undo_to(undo_)
                    MC["batches_rejected" if depth == 0 else "sub_rejected"] += 1; MC["sub_skipped_nochange"] += 1
                    return cur_, 0
                newm_ = {i: S.new[i] for i in undo_} if a.bisect_reuse else None
            else:
                old_all, newm_ = reuse
                ids_ = [i for i in rids(nets_) if i is not None and i in newm_ and newm_[i] != old_all[i]]
                if not ids_:
                    MC["sub_skipped_nochange"] += 1
                    return cur_, 0
                undo_ = {i: S.new.get(i, S.segs[i]) for i in ids_}
                touch(undo_.values(), [newm_[i] for i in ids_])
                if a.e_snap == 2 and cswap(ids_, undo_, newm_):
                    for i in ids_:
                        S.new[i] = newm_[i]
                    S._dirty.update(int(S.tid[i]) for i in ids_)
                elif not (a.e_capply and DBUF is not None and capply(ids_, undo_, newm_)):
                    S.revert({i: newm_[i] for i in ids_})
                MC["nets_reapplied"] += len(ids_); MC["t_reapply"] += time.perf_counter() - t_
                chg_ = ids_
            new_ = retime_m(); LAST["cur"] = False
            if new_[2]["score"] < cur_[2]["score"] - 1e-9:
                accept(); LAST["cur"] = True
                MC["batches_kept" if depth == 0 else "sub_kept"] += 1; MC["nets_kept"] += len(undo_)
                return new_, len(undo_)
            t_ = time.perf_counter(); undo_to(undo_); reject(); MC["t_revert"] += time.perf_counter() - t_
            MC["batches_rejected" if depth == 0 else "sub_rejected"] += 1
            kept_ = 0
            if len(undo_) > 0 and depth < a.bisect_depth and len(nets_) >= 2 * a.bisect_min:
                h_ = len(nets_) // 2
                rz = reuse if reuse is not None else ((dict(undo_), newm_) if newm_ is not None else None)
                cur_, k1 = try_batch(cur_, nets_[:h_], ctx, depth + 1, rz)
                cur_, k2 = try_batch(cur_, nets_[h_:], ctx, depth + 1, rz)
                kept_ = k1 + k2
                if depth == 0 and kept_ > 0:
                    MC["batches_recovered"] += 1
            return cur_, kept_

        def bsz(v, n_):
            return a.B if v < 0 else (max(n_, 1) if v == 0 else v)

        def crit_micro(cur_, nmax):
            wts_ = wts_of(cur_)
            cand_ = np.nonzero(wts_ > 0)[0]
            order_ = cand_[np.argsort(-wts_[cand_], kind="stable")][:nmax]
            done_ = np.zeros(nn_, bool); pos_ = 0; nb_ = 0; proc_ = 0; s0_ = cur_[2]["score"]; ks_ = 0
            while proc_ < nmax:
                due = (a.R > 0 and nb_ > 0 and nb_ % a.R == 0) or (a.Rkept > 0 and ks_ >= a.Rkept)
                if due and a.lazy and ks_ == 0:    # nothing kept since the last refresh: same state -> same weights
                    MC["refresh_skipped_lazy"] += 1
                elif due:
                    wts_ = wts_of(cur_); MC["weight_refresh"] += 1; ks_ = 0
                    cand_ = np.nonzero((wts_ > 0) & ~done_)[0]
                    order_ = cand_[np.argsort(-wts_[cand_], kind="stable")][:nmax - proc_]; pos_ = 0
                bt_ = order_[pos_:pos_ + a.B]
                if len(bt_) == 0:
                    break
                pos_ += len(bt_); proc_ += len(bt_); done_[bt_] = True
                cur_, kn_ = try_batch(cur_, bt_, (wts_, None, False, PRE["fl"]))
                ks_ += kn_ > 0
                nb_ += 1
            return cur_, cur_[2]["score"] < s0_ - 1e-9, proc_

        def power_micro(cur_):
            wts3_ = wts_of(cur_)
            kp_ = np.asarray(S.kpn, np.float64)
            pwn_ = S.w3 * kp_
            cc_ = np.where(kp_ > 0, kp_ * np.asarray(cur_[2]["net_ctot"], np.float64), 0.0)
            c3_ = np.nonzero(wts3_ > 0)[0]
            o1_ = c3_[np.argsort(-wts3_[c3_], kind="stable")][:a.round_nets]
            seen_ = np.zeros(nn_, bool); seen_[o1_] = True
            cp_ = np.nonzero((cc_ > 0) & ~seen_)[0]
            o2b_ = cp_[np.argsort(-cc_[cp_], kind="stable")][:a.pw_nets]
            order_ = np.concatenate([o1_, o2b_]); b_ = bsz(a.pB, len(order_))
            for s_ in range(0, len(order_), b_):
                cur_, _ = try_batch(cur_, order_[s_:s_ + b_], (wts3_, pwn_, True, PRE["fl"]))
            return cur_, len(order_)

        def q_power(cur_):
            """item 5: power batches guarded by slack, no timer inside. Nets with worst net slack >= --q_margin (from a full
            analysis at the stage start), timing weight 0, ordered by switching-power cost; each batch is decided by the exact
            change of the overflow and power terms (overflow: full demand-map cost; power: subset RC of the changed nets);
            every --qk batches (and at the end) one exact re-time confirms: if the 4-term estimate did not fall below the
            last confirmed one, every batch since then is undone."""
            t_ = time.perf_counter()
            rc0 = cur_[1]
            o_full = K.analyze(rc0["c1"], rc0["c2"], rc0["rpi"], rc0["ctot"], rc0["elm"], rc0["par"])   # net_slack, state sync
            MC["extra_analyze"] += 1; LAST["cur"] = True
            r_full = mk(o_full, rc0)
            cur_ = (o_full, rc0, r_full) if not INC else (o_full, K.rc, r_full)
            sl_ = np.asarray(o_full["net_slack"], np.float64)
            kp_ = np.asarray(S.kpn, np.float64) if S.kpn is not None else np.zeros(nn_)
            pwn_ = S.w3 * kp_
            ctot_ = np.asarray(cur_[2]["net_ctot"], np.float64).copy()
            cc_ = np.where(kp_ > 0, kp_ * ctot_, 0.0)
            inr = np.zeros(nn_, bool); tt = S.tid[S.tid >= 0]; inr[tt[tt < nn_]] = True
            elig = inr & (cc_ > 0) & (sl_ >= a.q_margin)
            ce_ = np.nonzero(elig)[0]
            order_ = ce_[np.argsort(-cc_[ce_], kind="stable")]
            if a.q_nets > 0:
                order_ = order_[:a.q_nets]
            zero_ = np.zeros(nn_)
            rinfo = {"eligible": int(elig.sum()), "guarded": int((inr & (cc_ > 0) & (sl_ < a.q_margin)).sum()), "visited": int(len(order_)),
                     "rank_s": time.perf_counter() - t_}
            conf_ = cur_                               # last confirmed state
            pend_ = []                                 # undo dicts of batches kept since the last confirmation
            of_c = float(cur_[2]["of"]); pw_c = float(cur_[2]["pw"]); est_c = float(cur_[2]["score"])
            b_ = a.qB if a.qB > 0 else max(len(order_), 1)
            fl_ = PRE["fl"] if a.q_fl else None
            CS_tid = S.tid

            def ctot_of(rnets):
                """ctot (max over rf) of the changed SD nets from their new segments, without touching the kernel state"""
                ts = time.perf_counter()
                cache = S.__dict__.setdefault("_f3x_nc", {})
                stale = [i for i in rnets if i in S.new and (cache.get(i) or (None,))[0] is not S.new[i]]
                if stale:
                    for i, b in FN._normalise(S, stale).items():
                        cache[i] = (S.new[i], b)
                off = S.segs.off
                parts = [cache[i][1] if i in S.new else S.base_arr[off[i]:off[i + 1]] for i in rnets]
                arr = np.concatenate(parts, 0) if parts else np.zeros((0, 7), np.int64)
                sub = arr.astype(np.int64, copy=True)
                sub[:, 0] = CS_tid[arr[:, 0]] if len(arr) else sub[:, 0]
                sub = sub[sub[:, 0] >= 0]
                nets = np.unique(CS_tid[np.asarray(rnets, np.int64)]); nets = nets[nets >= 0]
                if INC:
                    K.begin(); K.build_rc_subset(sub, nets); v = K.rc["ctot"][nets].max(1).copy(); K.reject()
                else:
                    rcx = K.build_rc(S.timer_array().astype(np.int64)); v = rcx["ctot"][nets].max(1)
                MC["t_q_est"] += time.perf_counter() - ts
                return nets, v

            def confirm(conf_, pend_, ctot_):
                ex = retime_m(); LAST["cur"] = False; MC["q_confirm"] += 1
                if ex[2]["score"] < conf_[2]["score"] - 1e-9:
                    accept(); LAST["cur"] = True; conf_ = ex
                    ctot_ = np.asarray(ex[2]["net_ctot"], np.float64).copy()
                else:
                    reject(); MC["q_confirm_fail"] += 1; MC["q_undone_batches"] += len(pend_)
                    for u_ in reversed(pend_):
                        touch([S.new.get(i, S.segs[i]) for i in u_], u_.values()); S.revert(u_)
                    ctot_ = np.asarray(conf_[2]["net_ctot"], np.float64).copy()
                return conf_, ctot_, float(conf_[2]["of"])

            nb = 0
            for s_ in range(0, len(order_), b_):
                bt_ = order_[s_:s_ + b_]
                t1 = time.perf_counter()
                undo_, _ = promote(cur_[2], bt_, a.tw_scale, fl_, zero_, pwn_, True)
                MC["t_promote"] += time.perf_counter() - t1; MC["nets_tried"] += len(bt_); MC["q_batches"] += 1
                touch(undo_.values(), [S.new[i] for i in undo_])
                chg = [i for i in undo_ if S.new[i] != undo_[i]]
                if not chg:
                    touch(undo_.values()); S.revert(undo_); continue
                t1 = time.perf_counter()
                of_n = FM.overflow_cost(S.la)
                MC["t_q_est"] += time.perf_counter() - t1
                nets, v = ctot_of(chg)
                dpw = float((kp_[nets] * (v - ctot_[nets])).sum())
                d_ = S.w4 * (of_n - of_c) + S.w3 * dpw
                if d_ < -1e-9:
                    of_c = of_n; pw_c += dpw; ctot_[nets] = v
                    pend_.append(undo_); MC["q_kept"] += 1; MC["nets_kept"] += len(undo_)
                else:
                    touch([S.new[i] for i in undo_], undo_.values()); S.revert(undo_)
                nb += 1
                if pend_ and a.qk > 0 and len(pend_) >= a.qk:
                    conf_, ctot_, of_c = confirm(conf_, pend_, ctot_)
                    pend_ = []; cur_ = conf_
            if pend_:
                conf_, ctot_, of_c = confirm(conf_, pend_, ctot_)
                pend_ = []
            rinfo["s"] = time.perf_counter() - t_
            MC.setdefault("q", []).append(rinfo)
            return conf_, len(order_)

        for stg in a.stages.split(","):
            stg = stg.strip(); t_s = time.perf_counter(); m0 = {k: v for k, v in MC.items() if not isinstance(v, list)}
            if stg.startswith("C"):
                kmax = int(stg[1:] or 1); nr = 0
                for _k in range(kmax):
                    cur, kept_r, _np_ = crit_micro(cur, a.round_nets); nr += 1
                    if not kept_r:
                        break
                ent = {"rounds": nr}
            elif stg == "P":
                cur, nv = power_micro(cur); ent = {"visited": nv}
            elif stg == "Q":
                cur, nv = q_power(cur); ent = {"visited": nv}
            elif stg == "W":
                MC["writes"] = MC.get("writes", 0) + 1
                p_w, dtw = wr("w%d" % MC["writes"]); ent = {"path": p_w, "nets_changed_vs_single_pass": len(S.new)}
                m0["writes"] = MC["writes"]
            else:
                sys.exit(f"micro2.py: unknown stage {stg}")
            ent.update(stage=stg, s=time.perf_counter() - t_s, est=est_of(cur[2]),
                       counters={k: (MC[k] - m0[k]) for k in m0})
            log.append(ent)
            print(f"[micro2] {stg}: est {cur[2]['score']:.4f} {ent['counters']}", flush=True)
        FM.overflow_cost = OF_orig
        if a.e_ofc:
            MC["ofc"] = {"calls": OFC_.calls, "cells": OFC_.cells, "check_bad": OFC_.bad if a.e_ofc_check else None,
                         "maxrel": getattr(OFC_, "maxrel", None)}
        # chain_j a: settle the kernel on the kept state, then chain_i's after-stages and the b<label> write
        if INC and not LAST["cur"]:
            K.flush_pending(); MC["flush"] += 1; LAST["cur"] = True
        PWS.clear(); PWS["cur"] = cur; PWS["settled"] = True; PWS["wts"] = weights(cur[0], cur[1])
        if a.post_crit > 0 or a.ovfround > 0 or a.rs != "off":
            t_a = time.perf_counter()
            cur = after_stages(True, cur[2])
            W["after_stages_s"] = time.perf_counter() - t_a
            log.append({"stage": "final", "est": est5(cur[2])})
        lab_m = snaps[max(snaps)]
        p_m, dtw = wr("b" + lab_m)
        log.append({"stage": "write_b" + lab_m, "path": p_m, "s": dtw, "nets_changed_vs_single_pass": len(S.new)})
        W["micro"] = {k: v for k, v in MC.items()}
        W["micro_args"] = {k: getattr(a, k) for k in ("stages", "B", "pB", "R", "Rkept", "lazy", "bisect_min", "bisect_depth",
                                                       "bisect_reuse", "skip_nochange", "fresh_crit", "e_inc", "e_dp", "e_ofc", "e_snap", "e_capply", "qB", "qk", "q_margin", "q_nets", "q_fl")}
        W["final_est"] = est_of(cur[2])
    # ───────── end ENGINE2 micro engine ─────────
    if a.late5:                                            # late5: region drivers (tools/late5/l5.py, l5_kern.cc)
        sys.path.append(str(ROOT / "tools" / "late5"))
        import l5 as L5
        S.revert = L5.make_revert(S, _Segs, S.revert)
        L5H["norm"] = L5.normalise
        if a.n_norm:                                       # speed9: the same rows, the normaliser on threads
            sys.path.append(str(ROOT / "tools" / "best5")); import n_norm as NNORM_
            L5H["norm"] = NNORM_.normalise
        if PROM is not None and os.environ.get("L5_PF", "1") == "1":   # promote rounds: FLUTE preparation in parallel (l5_pf)
            import l5_pf as L5PF
            P8_ = B8L.pf_module() if B8["on"] else None   # big8: par7 phase 2 on threads (same outputs)
            if P8_ is not None: L5PF = P8_
            prom_crit, prom_pw = L5PF.promote_cpp, L5PF.promote_pw
            L5H["pf"] = L5PF
        if a.par7 and PROM is not None and os.environ.get("L5_PF", "1") == "1":   # par7 (a): promote phase 2 on threads (same outputs)
            sys.path.append(str(ROOT / "tools" / "par7")); import p7_pf as P7PF
            P7PF.PAR[0] = int(a.par7); prom_crit, prom_pw = P7PF.promote_cpp, P7PF.promote_pw; L5H["pf"] = P7PF
            if a.t_same:                                   # speed10
                sys.path.append(str(ROOT / "tools" / "best10")); import t_fp as TFP_
                TFP_.same_install(P7PF)
        if a.of_rule == "official" and "S4OFC" not in G:  # exact cached official overflow cost (tools/speed4/s4_ovf.py)
            sys.path.append(str(ROOT / "tools" / "speed4"))
            import s4_ovf, of_official as OFO_
            G["S4OFC"] = s4_ovf.install(FM, OFO_, False)
    nround = 0; stuck = False; maxr = max(snaps) if a.engine != "micro" else 0; gain_tot = 0.0
    while nround < maxr:
        if not stuck and DLG("crit_round"):        # merge6 (b): deadline -> no further round (as if stuck)
            s_prev = cur[2]["score"]
            kept, cur, nprom, dt = crit_round(cur)
            DLE("crit_round")
            nround += 1
            stuck = not kept
            if a.t_crit_stop > 0 and kept:         # speed10
                g_ = s_prev - cur[2]["score"]
                stuck = g_ < a.t_crit_stop and W.get("t_crit_small", False)
                W["t_crit_small"] = g_ < a.t_crit_stop
                if stuck: W["t_crit_stop_at"] = nround
            if kept and a.min_gain_frac > 0:       # hook (off by default): stop when this round's gain is a small share
                g = s_prev - cur[2]["score"]; gain_tot += g
                if g < a.min_gain_frac * gain_tot:
                    stuck = True
            log.append({"stage": f"crit_round_{nround}", "kept": bool(kept), "promoted": nprom, "s": dt, "est": {k: float(cur[2][k]) for k in ("wns", "tns", "of", "pw", "score")}})
        else:
            nround += 1
        if nround in snaps:
            lab = snaps[nround]
            if not (a.only_b and a.power == "1"):
                p, dtw = wr("a" + lab)
                log.append({"stage": "write_a" + lab, "path": p, "s": dtw, "rounds": nround})
            if a.power == "1" and PWX:             # chain_i: one snapshot (asserted); kernel settled by power_round_i
                if DLG("power_round"):             # merge6 (b): skipped -> straight to the write (S holds the kept state)
                    kept2, undo2, r3, dtp = power_round_i(cur)
                    DLE("power_round")
                    log.append({"stage": "power_round_after_%d" % nround, "kept": bool(kept2), "promoted": len(undo2), "s": dtp,
                                "est": {k: float(r3[k]) for k in ("wns", "tns", "of", "pw", "score")}, "pw_info": PWS.get("info")})
                    if a.post_crit > 0 or a.ovfround > 0 or a.rs != "off":
                        t_a = time.perf_counter()
                        cur = after_stages(kept2, r3)
                        W["after_stages_s"] = time.perf_counter() - t_a
                        log.append({"stage": "final", "est": est5(cur[2])})
                elif DL["on"]:
                    log.append({"stage": "final", "est": est5(cur[2]), "deadline": True})
                if a.dgr and a.dgr_at == "end":     # dgr1x: quality setting: after everything
                    cur = dgr_run(cur, "end")
                if DL["on"]: DL["write_at"] = round(_dl_now(), 3)
                if B8["on"] and BIGM and B8L.pc("mem"): B8L.before_write(K)   # big8 mem: last timer use was above
                p, dtw = wr("b" + lab)
                log.append({"stage": "write_b" + lab, "path": p, "s": dtw})
            elif a.power == "1":
                kept2, undo2, r3, dtp = power_round(cur)
                log.append({"stage": "power_round_after_%d" % nround, "kept": bool(kept2), "promoted": len(undo2), "s": dtp,
                            "est": {k: float(r3[k]) for k in ("wns", "tns", "of", "pw", "score")}})
                p, dtw = wr("b" + lab)
                log.append({"stage": "write_b" + lab, "path": p, "s": dtw})
                if kept2 and (1 not in ITEMS or nround < maxr):    # item 1: no revert after the final write
                    S.revert(undo2)
                    if H["inc"]:
                        K.reject()                     # cur is the pre-power state (chain_inc.py)
                    dcheck("power round revert")
                elif kept2 and H["inc"]:
                    K.commit()
    W["post_router_s"] = time.perf_counter() - tpost + W["preload_wait_s"]
    W["merge6"] = {"route_bin": int(a.route_bin), "deadline_s": a.deadline_s}
    if DL["on"]:
        W["merge6"].update(deadline_ref=DL["ref"][0], expired=DL["expired"], skipped=DL["skipped"], ran=DL["ran"],
                           write_at=DL.get("write_at"), est_used=DL["est"], end_at=round(_dl_now(), 3))
    W["process_total_s"] = time.perf_counter() - T0
    rec = {"design": a.design, "suite": a.suite, "lam": a.lam, "gate": gate_rec, "escape_on": W.get("esc_on"), "walls": W, "log": log, "loadavg": os.getloadavg()}
    rec["h"] = dict(H); rec["h_times"] = HT; rec["retime"] = RETIME
    if a.t_same: rec["t_same"] = dict(TFP_.SAME)
    rec["chain_i"] = {k: getattr(a, k) for k in ("ovf_impl", "ovf_timer", "ovf_order", "ovf_rr_passes", "ovf_max_runs", "ovf_hmax", "ovf_hot",
                                                 "ovf_esc_pre", "ovf_esc_check", "ovf_big_nets", "ovf_hmax_big", "ovf_big_runs", "ovf_big_passes",
                                                 "tw_scale", "post_crit", "pw_batch", "pw_bisect_depth", "pw_bisect_min", "pw2_nets",
                                                 "pw_skip", "pw_repair", "ovfround", "ovfround_slack", "ovfround_open", "ovfround_skip")}
    rec["chain_j"] = {k: getattr(a, k) for k in ("engine", "stages", "B", "R", "lazy", "skip_nochange", "bisect_reuse", "bisect_depth",
                                                 "e_dp", "e_snap", "e_ofc", "micro_big", "post_crit_big", "rs", "rs_at", "rs_order",
                                                 "s4_ofc", "s4_esc")}
    if "S4OFC" in G:
        rec["s4_ofc"] = dict(G["S4OFC"].stats)
    if "S4ESC" in G:
        rec["s4_esc"] = dict(G["S4ESC"].STATS)
    rec["inc_info"] = PRE.get("inc_info")
    if a.late5:
        rec["late5"] = dict(L5.STATS)
        PM_ = L5H.get("pf", PROM)
        if PM_ is not None and getattr(PM_, "TIMES", None):    # promote per-call (prep, C++ call, unpack, phase1, dp, rest, thr, phase1a)
            TT_ = np.asarray(PM_.TIMES, np.float64)
            rec["late5"]["pf_times_sum"] = [round(float(x), 4) for x in TT_.sum(0)[:8]]; rec["late5"]["pf_calls"] = int(len(TT_))
            rec["par7"] = {"par7": int(a.par7), "pf_times": [[round(float(x), 5) for x in r_[:8]] for r_ in TT_],
                           "levels": getattr(PM_, "P7", None), "retime": [[r_.get("tag"), round(float(r_.get("s", 0.0)), 5)] for r_ in RETIME]}
    if H["check"]:
        rec["h_check"] = HCHK
        rec["h_check_bad"] = sum(v[1] for v in HCHK.values())
    if a.check:
        rec["check"] = dict(CHECK)
    if B5["on"]: rec["big5"] = B5L.record()
    if B8["on"]: rec["big8"] = B8L.record()
    if L8["on"] or L8S: rec["load8"] = L8L.record()
    if a.json: Path(a.json).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    os._exit(0)


if __name__ == "__main__":
    if BIGM:                                                      # chain_gb S3 fix: ctypes.string_at for blobs over 2^31-1 bytes
        import ctypes as _ct
        _sa0 = _ct.string_at
        def _string_at64(ptr, size=-1, _o=_sa0):
            if size is None or size < 2**31 - 1:
                return _o(ptr, size)
            a_ = ptr if isinstance(ptr, int) else _ct.cast(ptr, _ct.c_void_p).value
            return (_ct.c_char * int(size)).from_address(a_).raw
        _ct.string_at = _string_at64
    _orig = load_all
    def load_all(a, _o=_orig):                                    # noqa
        import gc
        if H["pre"]:                                              # item 7: no cyclic-GC passes during the preload; afterwards
            gc.disable()                                          # its objects go to the permanent generation (gc.freeze)
        try:
            import fast_sta_v3; sys.modules["fast_sta"] = fast_sta_v3
            global ChainState, FLD
            if 1 in BIG or 2 in BIG:                              # chain_gb B1: before make_state_class binds parse_cap_auto
                _t = time.perf_counter()
                sys.path.append(str(ROOT / "tools" / "comp2" / "loader"))
                sys.path.append(str(ROOT / "tools" / "onepass"))
                import fast_load
                FLD = fast_load
                if 1 in BIG:
                    fast_load.install()
                if L8["on"] and L8L.install(fast_sta_v3, fast_load, a, G, PRE, H, BEST, BIGM, B8L):
                    pass                                          # load8: cap, StaDesign, big8's install_load, the bind on its own thread
                elif B8["on"] and BIGM:
                    B8L.install_load(fast_sta_v3); B8L.wrap_sta(fast_sta_v3, G, PRE, H, BEST, BIGM)
                PRE["t_fastload_install"] = time.perf_counter() - _t
            if 3 in BIG:                                          # chain_gb B3: memo timer kernel (same API, same outputs)
                import csta as _CS
                _lb = str(ROOT / "tools" / "srvw" / "bigcase" / "libcsta_b.so")
                if os.path.exists(_lb):
                    _CS.Kernel.__init__.__defaults__ = (_lb,)
                    PRE["csta_lib"] = _lb
            if 6 in BIG:                                          # chain_gb B6: fast route reader (same document)
                import route_cpp as _RC
                _lr = ROOT / "tools" / "srvw" / "bigcase" / ("libroute_cpp_b.dylib" if sys.platform == "darwin" else "libroute_cpp_b.so")
                if _lr.exists():
                    _RC._LIB = _lr
                    PRE["route_lib"] = str(_lr)
            ChainState = make_state_class()
            if L8["on"] and L8L.EARLY.get("cold"): ChainState = L8L.state_class(ChainState)   # load8: tidx_key on a thread
            if B5["on"]: ChainState = B5L.state_class(ChainState, G, PRE)   # big5: activity child, early timer
            return _o(a)
        finally:
            if H["pre"]:
                PRE.pop("h_net", None)
                gc.freeze(); gc.enable()
                PRE["preload_end"] = time.perf_counter() - T0
    main()
