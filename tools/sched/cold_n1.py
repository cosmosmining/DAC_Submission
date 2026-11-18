#!/usr/bin/env python3
"""cold_n1.py (sched 2026-11-12, NEW; plan N1 / N2 of NEXT_ROUND_LESSONS_2026-11-12.md): tools/sched/cold_sched.py (below) plus
  - fixcut's fixed-work cuts exactly as tools/fixcut/cold_fc.py applies them: tools/fixcut/fc_lib.py loaded by file, its import hooks,
    its two chain insertions (pre5_load source patch, single-pass dem / rcsub); every cut OFF unless named in env FIXCUT (unset or
    empty = cold_sched.py's executed source and modules); the cold json gets "fixcut";
  - N2: --dgr_skip_fp in big mode. big8 mem dropped the first timer's result (o = rcd = None) before the single pass, and the skip
    branch then timed None (TypeError in mk). The drop now keeps o / rcd when the pass is skipped, and big mode takes copies of them
    at the skip point; nothing changes when the pass runs;
  - env N1_DRYRUN=<file>: write the patched chain source there and exit before running it (patch check).
cold_sched.py (sched 2026-11-12, NEW): the cold driver of the flow (a byte-for-byte copy of the logic of tools/day1/cold_d1.py,
which is tools/final/cold_y.py with another SRC) with the learned deadline rule of tools/sched/sched_rt.py patched into the chain's
scheduler. No existing file is edited: the chain source is read, patched in memory and executed, as cold_d1.py already does.

Own switches (removed from argv before the chain parses it):
  --sched_src d1|y        chain: d1 = tools/day1/chain_d1.py (big-design arm d1), y = tools/final/chain_y.py (frozen arm cur)
  --sched_root DIR        repo root (default: env SCHED_ROOT, else the repo this file sits in)
  --sched_model FILE|off  learned model (tools/sched/model_v1.json); 'off' (default) = exactly the old driver, nothing patched
  --sched_q Q             quantile of the unit predictions (default 0.8: zero late H100 runs in the held-out replay)
  --sched_tail_q Q        quantile of the tail prediction (default = --sched_q)
  --sched_margin S        extra seconds kept free (default 0)
  --sched_fit 0|1         1 = a DTLA round 0 that does not fit frees fewer nets so that it does (default 0)
  --sched_fit_min F       smallest share of the planned nets for --sched_fit 1 (default 0.25)
  --sched_log FILE        decisions as one json line (also in the run json, walls.sched)
  --sched_fill 0|1        anytime fill: when the fixed stage list ends before the deadline, more units (power round + its post-crit
                          rounds, crit round, DTLA stage on fresh nets) while now + predicted unit + predicted tail <= X, each time
                          the type with the highest expected gain per predicted second (expected gain = its last measured gain in
                          this run, else --sched_fill_prior); needs --sched_model
  --sched_fill_fresh 0|1  the fill's DTLA stage frees only nets no earlier DTLA round freed (default 1)
  --sched_fill_prior J    json {"power": g, "crit": g, "dgr": g} of est-score gains for types not yet measured in the run
  --sched_fill_min_rate R stop the fill when the best expected gain per second is below R (default 1e-4)
  --sched_extra_dgr N     (plan R2) N more DTLA rounds after the last post-critical round (before the write), same settings;
                          works with the old scheduler (--sched_model off)
  --sched_k F             (plan R1) DTLA net budget on designs with more than 300,000 eligible nets: the top max(--dgr_nets,
                          F x eligible) nets of the flow's own ranking (current rule: F = 0.07); smaller designs unchanged
  --sched_k0 1            (plan R3) zero descent steps: the start (e.g. the GNN start of tools/groupgnn) is rounded and checked
  --sched_exit_dgr 1      (plan R1, R3) after the DTLA stage at the first pass: write the run json (single pass + DTLA records,
                          walls) and end the process; no route
  --sched_stop_frac F     (A1, off by default) the next DTLA round of a stage runs only if the previous round gained at least F x
                          the gain of the stage's first round; a first round without gain ends the stage
  --sched_a4 1            (A4, off by default) when WNS = TNS = 0 after the critical-path rounds, the power round is skipped if the
                          first-order bound of its power gain (the chain's --pw_skip bound) is below the smallest gain of a kept round
                          so far (implemented by raising --pw_skip to that gain for the call)
  --skip_ovf_rule 1       skip the overflow stage when its first-order gain is below --skip_ovf_thr (default 0.01 score units):
                          bound = |w4| x sum over overflowed cells (capacity > 0) of cost(d) - cost(c) (the cost drop if every
                          overflow were removed; >= what the stage can gain); no overflowed cell -> bound 0 -> skipped
  --crit_stop_frac F      stop the critical-path rounds when a round gains less than F x the first round's gain; the same for the
                          post-critical rounds on designs below 300,000 nets
  --power_first_rule 1    run the power round before the critical-path rounds when WNS and TNS are already met (the state's own
                          timing terms); the snapshot's power round is then skipped and its after-stages run as usual. The
                          first-order comparison |w3| dP vs |w1| dWNS + |w2| dTNS / N is not implemented (needs the power round's
                          candidate weights and a timing-gain estimate before any round)
  --sched_value 1|2       value rule for the refinement units after the first pass (needs --sched_model): 1 = start the unit with
                          the highest predicted gain per predicted second that fits (gain model --sched_gain), 2 = DTLA or not by
                          the predicted gain of the two in-order plans that fit, then the fixed order skipping what does not fit;
                          the chain's own DTLA-at-first-pass, overflow stage and later rounds are then skipped
  --sched_gain FILE       gain model (default tools/sched/gain_v1.json next to this file)
  --sched_prof FILE       sampling profile of every Python thread (sys._current_frames every 5 ms; Python 3.12's cProfile is
                          one global monitor and mixes threads): FILE.txt = per thread inclusive / self seconds per function,
                          and the main thread's timeline (which function under the chain's main() runs when); FILE.json raw
Chain insertions (each must match exactly once, asserted): bind after DL.update(on=True,...); DLG's comparison; the DTLA stage's
write reserve and its two round guards; the dl_chunk guard; the record next to walls.merge6; with --sched_fill 1 the fill call
just before the final write. With --sched_fit 1 or --sched_fill 1 the module dgr_stage_r (tools/best8) is loaded through an import
hook with three insertions: the freed-net count goes through SCHED.nets_hook, the ranked list through SCHED.order_hook (fresh nets)
and the chosen nets through SCHED.freed_hook (record)."""
import os, sys, time, json, threading
T0 = time.perf_counter()
from pathlib import Path

ME = Path(__file__).resolve().parent
_SOWN = {"--sched_src": "d1", "--sched_root": None, "--sched_model": "off", "--sched_q": "0.8", "--sched_tail_q": None,
         "--sched_margin": "0", "--sched_fit": "0", "--sched_fit_min": "0.25", "--sched_log": None, "--sched_prof": None,
         "--sched_fill": "0", "--sched_fill_fresh": "1", "--sched_fill_prior": None, "--sched_fill_min_rate": "1e-4",
         "--sched_extra_dgr": "0", "--sched_k": None, "--sched_k0": "0", "--sched_exit_dgr": "0", "--sched_value": "0", "--sched_gain": None,
         "--sched_stop_frac": "0", "--sched_a4": "0", "--skip_ovf_rule": "0", "--skip_ovf_thr": "0.01", "--crit_stop_frac": "0",
         "--power_first_rule": "0"}
SOPT = dict(_SOWN); _a = [sys.argv[0]]; _i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _SOWN and _i + 1 < len(sys.argv):
        SOPT[sys.argv[_i]] = sys.argv[_i + 1]; _i += 2
    else:
        _a.append(sys.argv[_i]); _i += 1
sys.argv = _a
ROOT = Path(SOPT["--sched_root"] or os.environ.get("SCHED_ROOT") or ME.parents[1]).resolve()
SRC = ROOT / {"d1": "tools/day1/chain_d1.py", "y": "tools/final/chain_y.py"}[SOPT["--sched_src"]]
if not SRC.exists():
    raise SystemExit("cold_sched: chain not found: %s (pass --sched_root <repo>)" % SRC)
HERE = SRC.parent                                            # as the old driver: its own folder (tools/day1 or tools/final)
sys.path[0] = str(HERE)                                      # hard_exit.py put this file's folder there; the old driver had HERE
if str(ME) not in sys.path:
    sys.path.append(str(ME))                                 # sched_rt / sched_model, after every chain folder (no shadowing)
import importlib.util as _ilu                                # N1: fixcut's cuts, loaded by file as tools/fixcut/cold_fc.py does
_fs = _ilu.spec_from_file_location("fc_lib", str(ROOT / "tools" / "fixcut" / "fc_lib.py")); FC_LIB = _ilu.module_from_spec(_fs)
sys.modules["fc_lib"] = FC_LIB; _fs.loader.exec_module(FC_LIB)
FC_LIB.install_hooks()
_FCS = SOPT["--sched_src"]
SCHED = None
if SOPT["--sched_model"] not in (None, "", "off"):
    import sched_rt as _SRT
    SCHED = _SRT.Sched(SOPT["--sched_model"], float(SOPT["--sched_q"]),
                       float(SOPT["--sched_tail_q"]) if SOPT["--sched_tail_q"] else None, float(SOPT["--sched_margin"]),
                       SOPT["--sched_log"], int(SOPT["--sched_fit"]), float(SOPT["--sched_fit_min"]))
    SCHED.set_fill(int(SOPT["--sched_fill"]), int(SOPT["--sched_fill_fresh"]), SOPT["--sched_fill_prior"], float(SOPT["--sched_fill_min_rate"]))
    SCHED.set_value(int(SOPT["--sched_value"]), SOPT["--sched_gain"] or str(ME / "gain_v1.json"))
elif int(SOPT["--sched_fill"]) or int(SOPT["--sched_value"]):
    raise SystemExit("cold_sched: --sched_fill 1 needs --sched_model")
EXTRA = {"extra_dgr": int(SOPT["--sched_extra_dgr"]), "k": float(SOPT["--sched_k"]) if SOPT["--sched_k"] else None,
         "k0": int(SOPT["--sched_k0"]), "exit_dgr": int(SOPT["--sched_exit_dgr"]), "nets_log": [],
         "stop_frac": float(SOPT["--sched_stop_frac"]), "a4": int(SOPT["--sched_a4"]), "a1_log": [], "a4_log": [],
         "skip_ovf": int(SOPT["--skip_ovf_rule"]), "skip_ovf_thr": float(SOPT["--skip_ovf_thr"]), "ovf_log": [],
         "crit_frac": float(SOPT["--crit_stop_frac"]), "crit_first": {}, "crit_log": [], "power_first": int(SOPT["--power_first_rule"]),
         "pf_log": []}
PROF = {"on": bool(SOPT["--sched_prof"]), "done": False, "smp": None}


class _Sampler(threading.Thread):
    """stack sampler of all other threads: seconds per (thread, stack) and the main thread's timeline under main()"""
    def __init__(self, iv=0.005):
        super().__init__(name="sched_sampler", daemon=True)
        self.iv = iv; self.halt = False; self.acc = {}; self.tl = []; self.n = 0; self.t0 = time.perf_counter()

    def run(self):
        me = threading.get_ident(); last = time.perf_counter()
        while not self.halt:
            time.sleep(self.iv)
            now = time.perf_counter(); dt = now - last; last = now
            names = {t.ident: t.name for t in threading.enumerate()}
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                st = []; f = fr; dep = 0
                while f is not None and dep < 80:
                    co = f.f_code; st.append("%s:%s:%d" % (co.co_filename.rsplit("/", 1)[-1], co.co_name, co.co_firstlineno)); f = f.f_back; dep += 1
                st = tuple(reversed(st)); tn = names.get(tid, str(tid))
                self.acc[(tn, st)] = self.acc.get((tn, st), 0.0) + dt
                if tn == "MainThread":
                    j = next((i for i, x in enumerate(st) if ":main:" in x and x.startswith(("chain_d1.py", "chain_y.py"))), None)
                    key = st[j + 1] if (j is not None and j + 1 < len(st)) else (st[-1] if st else "?")
                    if self.tl and self.tl[-1][0] == key:
                        self.tl[-1][2] = now - self.t0
                    else:
                        self.tl.append([key, round(now - self.t0, 3), round(now - self.t0, 3)])
            self.n += 1

    def report(self, path):
        self.halt = True
        inc, slf, tot = {}, {}, {}
        for (tn, st), sec in self.acc.items():
            tot[tn] = tot.get(tn, 0.0) + sec
            for fn in set(st):
                inc[(tn, fn)] = inc.get((tn, fn), 0.0) + sec
            if st:
                slf[(tn, st[-1])] = slf.get((tn, st[-1]), 0.0) + sec
        out = ["samples %d, interval %.3f s, threads (s): %s" % (self.n, self.iv, json.dumps({k: round(v, 2) for k, v in sorted(tot.items(), key=lambda x: -x[1])}))]
        for tn in sorted(tot, key=lambda x: -tot[x]):
            k_ = 45 if tn == "MainThread" else 15
            out.append("\n== %s: inclusive s (top %d)" % (tn, k_))
            for (t_, fn), sec in sorted(((k, v) for k, v in inc.items() if k[0] == tn), key=lambda x: -x[1])[:k_]:
                out.append("%9.3f  %s" % (sec, fn))
            out.append("-- %s: self s (top %d)" % (tn, k_))
            for (t_, fn), sec in sorted(((k, v) for k, v in slf.items() if k[0] == tn), key=lambda x: -x[1])[:k_]:
                out.append("%9.3f  %s" % (sec, fn))
        out.append("\n== MainThread timeline (function under main(), start s, end s, from sampler start)")
        for key, a_, b_ in self.tl:
            if b_ - a_ >= 0.05:
                out.append("%8.3f %8.3f %7.3f  %s" % (a_, b_, b_ - a_, key))
        Path(path + ".txt").write_text("\n".join(out) + "\n")
        Path(path + ".json").write_text(json.dumps({"tl": self.tl, "acc": [[tn, list(st), sec] for (tn, st), sec in self.acc.items() if sec >= 0.02]}))


if PROF["on"]:
    PROF["smp"] = _Sampler(); PROF["smp"].start()


# ---------------------------------------------------------------- from here: tools/day1/cold_d1.py (unchanged logic)
if sys.platform == "darwin":
    sys.path.insert(0, str(HERE.parent / "best2" / "mac_rs"))   # as cold_j: Mac-only byte copy of rs_lib.py + librs_loop.dylib
OWN = {"--io_cold_lib": 0, "--io_bind_mem": 0, "--wait_route": 1, "--act": 1, "--act_done": 1, "--act_mode": 1,
       "--audit_open": 1, "--cold_json": 1, "--allow_warm": 0}
IO = {}
_argv = [str(SRC)]; _i = 1
while _i < len(sys.argv):
    k = sys.argv[_i]
    if k in OWN:
        if OWN[k]:
            IO[k[2:]] = sys.argv[_i + 1]; _i += 2
        else:
            IO[k[2:]] = True; _i += 1
    else:
        _argv.append(k); _i += 1
sys.argv = _argv


def _av(k, d=None):
    for i_, v_ in enumerate(sys.argv):
        if v_ == k and i_ + 1 < len(sys.argv):
            return sys.argv[i_ + 1]
    return d


PRE5_ASKED = _av("--pre5", "0") == "1"
ST = {"switches": dict(IO), "pre5_asked": PRE5_ASKED, "pre5_act": _av("--pre5_act", "child") if PRE5_ASKED else None,
      "sched": {k[2:]: v for k, v in SOPT.items()}}
if PRE5_ASKED and ST["pre5_act"] == "file" and not IO.get("allow_warm"):
    raise SystemExit("cold_k: --pre5_act file reads the prepared per-case activity file (not cold); pass --allow_warm for identity tests")
MODE = IO.get("act_mode", "lazy") if IO.get("act") else ("pre5" if PRE5_ASKED else "prepared")
assert MODE in ("prepared", "lazy", "pre5"), "cold_k.py: only --act_mode lazy"
_OPENED = {}
if IO.get("audit_open"):
    def _hook(ev, args):
        if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
        elif ev == "ctypes.dlopen" and args and args[0]:
            p = "dlopen:" + os.fsdecode(args[0])
        elif ev == "subprocess.Popen" and len(args) > 1:
            p = "exec:" + " ".join(os.fsdecode(x) for x in (args[1] or [])[:4])
        else:
            return
        if p not in _OPENED:
            _OPENED[p] = round(time.perf_counter() - T0, 3)
    sys.addaudithook(_hook)
    import subprocess as _sp
    _OP = _sp.Popen

    class _AuditPopen(_OP):                                   # the activity child runs under tools/best3/audit_child.py
        def __init__(self, args, *aa, **kk):
            if isinstance(args, (list, tuple)) and any(str(x).endswith("act_child.py") for x in args):
                i_ = [j_ for j_, x in enumerate(args) if str(x).endswith("act_child.py")][0]
                args = list(args[:i_]) + [str(HERE.parent / "best3" / "audit_child.py"), IO["audit_open"] + ".child"] + list(args[i_:])
                ST["act_child_audited"] = True
            super().__init__(args, *aa, **kk)
    _sp.Popen = _AuditPopen
import numpy as np
NS = {}                                                      # chain_k's globals once executed
ACT = {"dens": None, "err": None}


def _watch_act():
    try:
        while not os.path.exists(IO["act_done"]):
            time.sleep(0.01)
        ST["act_file_ready_at_s"] = time.perf_counter() - T0
        t1 = time.perf_counter()
        S = NS["PRE"]["S"]
        from timing_promote import _key
        dens = np.zeros(len(S.SD.net_names))
        idx = S.SD.net_index
        for line in open(IO["act"]):
            f = line.split()
            if len(f) >= 2 and not (len(f) >= 3 and f[2] in ("input", "clock")):
                i = idx.get(_key(f[0]))
                if i is not None:
                    try:
                        dens[i] = float(f[1])
                    except ValueError:
                        pass
        ACT["dens"] = dens
        ST["act_parse_s"] = time.perf_counter() - t1
    except BaseException as e:
        ACT["err"] = repr(e)
    ST["act_thread_end_s"] = time.perf_counter() - T0


def _real_kpn(S):
    th = NS["PRE"].get("act_thread")
    t = time.perf_counter()
    if th is not None:
        th.join()
    if ACT["err"]:
        raise SystemExit("cold_k: activity failed: " + ACT["err"])
    ST.setdefault("act_wait_s", time.perf_counter() - t)
    ST.setdefault("act_needed_at_s", t - T0)
    S.__dict__["_kpn_real"] = None
    S.set_activity(ACT["dens"])


def c_state(Base):
    if MODE in ("prepared", "pre5"):
        return Base
    class ColdState(Base):
        def __init__(self, *aa, **kk):
            import glob as _g
            _og = _g.glob
            _g.glob = lambda pat, *a_, _o=_og, **k_: [] if "net_slack" in str(pat) else _o(pat, *a_, **k_)
            try:
                Base.__init__(self, *aa, **kk)
            finally:
                _g.glob = _og

        def _get_kpn(self):
            d = self.__dict__
            if "_kpn_real" in d and d["_kpn_real"] is not None:
                return d["_kpn_real"]
            _real_kpn(self)
            return d["_kpn_real"]

        def _set_kpn(self, v):
            self.__dict__["_kpn_real"] = v
        kpn = property(_get_kpn, _set_kpn)
    return ColdState


def c_lib(fast_sta_v3):
    if IO.get("io_cold_lib") and not ST.get("_lib"):
        ST["_lib"] = True
        _ol = fast_sta_v3.StaLib.load
        fast_sta_v3.StaLib.load = staticmethod(lambda lib_dir=None, cache=None, _o=_ol:
                                               _o(lib_dir, cache="/dev/null/no_sta_lib.pkl") if lib_dir else _o(cache="/dev/null/no_sta_lib.pkl"))


def c_reader(rr0):
    def rr(path, _r=rr0):
        t = time.perf_counter()
        ST["route_read_called_at_s"] = t - T0
        if IO.get("wait_route"):
            while not os.path.exists(IO["wait_route"]):
                time.sleep(0.005)
            ST["route_ready_at_s"] = time.perf_counter() - T0
        ST["route_wait_s"] = time.perf_counter() - t
        t = time.perf_counter()
        r = _r(path)
        ST["route_read_s"] = time.perf_counter() - t
        return r
    return rr


def c_after_state():
    ST["t_state_end_s"] = time.perf_counter() - T0
    if MODE == "lazy":
        th = threading.Thread(target=_watch_act, daemon=True); th.start()
        NS["PRE"]["act_thread"] = th


COLD = {"state": c_state, "lib": c_lib, "reader": c_reader, "after_state": c_after_state, "bind_mem": bool(IO.get("io_bind_mem"))}


def rep(s, old, new):
    n = s.count(old)
    assert n == 1, (n, old[:70])
    return s.replace(old, new)


s = SRC.read_text()
s = rep(s, '        sys.modules["fast_sta"] = fast_sta_v3\n        import io_run as IR\n',
        '        sys.modules["fast_sta"] = fast_sta_v3\n        COLD["lib"](fast_sta_v3)\n        import io_run as IR\n')
s = rep(s, '    t = time.perf_counter(); rt = G["read_route"](str(a.route)); W["route_read_s"] = time.perf_counter() - t\n',
        '    t = time.perf_counter(); rt = COLD["reader"](G["read_route"])(str(a.route)); W["route_read_s"] = time.perf_counter() - t\n')
s = rep(s, '        cd = CS.load_cache(cache)\n',
        '        if COLD["bind_mem"]:                                  # cold_k: state first, the timer binds its StaDesign\n'
        '            t5_ = time.perf_counter(); PRE["S"] = ChainState(a.design, a.suite); COLD["after_state"]()\n'
        '            COLD["t_state_s"] = time.perf_counter() - t5_\n'
        '            cd = PRE["S"].SD\n'
        '        else:\n'
        '            cd = CS.load_cache(cache)\n')
s = rep(s, '        PRE["S"] = ChainState(a.design, a.suite)           # StaDesign, cap, TimingLA, pin tables, activity\n',
        '        if not COLD["bind_mem"]:\n'
        '            PRE["S"] = ChainState(a.design, a.suite)           # StaDesign, cap, TimingLA, pin tables, activity\n'
        '            COLD["after_state"]()\n')
s = rep(s, '            ChainState = make_state_class()\n', '            ChainState = COLD["state"](make_state_class())\n')
# ---------------------------------------------------------------- N1: tools/fixcut/cold_fc.py's insertions (verbatim)
_p5x = '        exec(compile(_p5s, str(_p5f), "exec"), P5L.__dict__)\n'   # the chain execs pre5_load's source itself (--timer5 1)
if FC_LIB.PATCHES.get("pre5_load") and s.count(_p5x) == 1:
    s = rep(s, _p5x, '        P5L.__dict__["FC_LIB"] = FC_LIB; _p5s = FC_LIB.patch_src("pre5_load", _p5s)   # fixcut\n' + _p5x)
if FC_LIB.on("dem") or FC_LIB.on("rcsub"):                  # fixcut: the single pass (gate 0) evaluation
    _sp_old = {"d1": '            rt2, arr6, o2, rcd2, r0 = eval_cols(noff, cols, fp_label="singlepass_eval")\n',
               "y": '            rt2, arr6, o2, rcd2, r0 = eval_cols(noff, cols)\n'}[_FCS]
    _sp_call = _sp_old.strip()[:-1] + ", dem=_fcd, inc_nets=_fci)"
    s = rep(s, _sp_old,
            '            _fcd, _fci = FC_LIB.sp_args(P, PRE, H, K, len(rt.net_names))   # fixcut dem / rcsub ((None, None): as before)\n'
            '            ' + _sp_call + '\n'
            '            _fco = FC_LIB.sp_after(_fcd, _fci, PRE, S, FM, K, lambda: CS.seg_from_route(cd, rt2, G_))\n'
            '            if _fco is not None:\n'
            '                o2, rcd2 = _fco\n')
# ---------------------------------------------------------------- N2: --dgr_skip_fp in big mode (big8 mem dropped o / rcd)
_n2a = '        B8L.drop_early_refs(G, K, CS); o = rcd = seg = cr = crit_data = timer_rec = None\n'
_n2b = '            o2, rcd2 = o, rcd; r0 = mk(o2, rcd2)\n'
if s.count(_n2a) == 1 and s.count(_n2b) == 1:
    s = rep(s, _n2a, '        B8L.drop_early_refs(G, K, CS); seg = cr = crit_data = timer_rec = None\n'
                     '        if not SKIPFP: o = rcd = None                       # sched N2: the skip branch reads the first timer\'s result\n')
    s = rep(s, _n2b, '            o2, rcd2 = _N2_COPY(o, rcd) if B8["on"] else (o, rcd)   # sched N2: big mode keeps copies (the kernel moves on)\n'
                     '            r0 = mk(o2, rcd2)\n')
    ST["n2_fix"] = True


def _n2_copy(o_, rcd_):
    cp = lambda d: {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in d.items()} if hasattr(d, "items") else d
    return cp(o_), cp(rcd_)
# ---------------------------------------------------------------- end of the cold_d1.py insertions; the scheduler (tools/sched)


def sched_patch(s):
    """the learned rule into the chain's scheduler (sched_rt.Sched bound as global SCHED)"""
    s = rep(s, '        DL.update(on=True, X=float(a.deadline_s), ref=_dl_ref())\n',
            '        DL.update(on=True, X=float(a.deadline_s), ref=_dl_ref())\n'
            '        SCHED.bind(DL, a, BIGM, _dl_now, lambda: len(PRE["S"].SD.net_names))   # tools/sched\n')
    s = rep(s, '    if now + est > DL["X"]:\n',
            '    est = SCHED.est(kind, now, est)                         # tools/sched: unit + tail + margin\n'
            '    if now + est > DL["X"]:\n')
    s = rep(s, '        tw_ = float(DL["est"].get("dgr_write", 0.0)); te_ = float(DL["est"].get("dgr_test", 0.0)); tb_ = float(DL["est"].get("dgr_build", 0.0))\n',
            '        tw_ = float(DL["est"].get("dgr_write", 0.0)); te_ = float(DL["est"].get("dgr_test", 0.0)); tb_ = float(DL["est"].get("dgr_build", 0.0))\n'
            '        if DL["on"]: tw_ = SCHED.dgr_write(tw_)                 # tools/sched: the stage keeps the tail free\n')
    s = rep(s, '            if _dl_now() + tb_ + te_ + tw_ > dl_:\n',
            '            if SCHED.dgr_refuse(0, None, tb_ + te_ + tw_):       # tools/sched\n')
    s = rep(s, '            if rn_ and DL["on"] and _dl_now() + float(rec_.get("stage_s", 0.0)) + tw_ > dl_:\n',
            '            if rn_ and DL["on"] and SCHED.dgr_refuse(rn_, float(rec_.get("stage_s", 0.0)), float(rec_.get("stage_s", 0.0)) + tw_):   # tools/sched\n')
    s = rep(s, '            now_ = _dl_now(); e_ = DL["last"].get("dl_chunk", DL["est"].get("dl_chunk", 0.0))\n',
            '            now_ = _dl_now(); e_ = DL["last"].get("dl_chunk", DL["est"].get("dl_chunk", 0.0)); e_ = SCHED.est("dl_chunk", now_, e_)   # tools/sched\n')
    s = rep(s, '    W["merge6"] = {"route_bin": int(a.route_bin), "deadline_s": a.deadline_s}\n',
            '    W["merge6"] = {"route_bin": int(a.route_bin), "deadline_s": a.deadline_s}\n'
            '    W["sched"] = SCHED.record()                                # tools/sched\n')
    if SCHED.fill_on:
        s = rep(s, '                if DL["on"]: DL["write_at"] = round(_dl_now(), 3)\n',
                '                if DL["on"] and not locals().get("ZT", False): cur = SCHED.fill(cur, {"crit": crit_round, "power": power_round_i, "after": after_stages, '
                '"dgr": dgr_run, "PWS": PWS, "log": log, "est5": est5, "DLG": DLG, "DLE": DLE, "a": a})   # tools/sched: anytime fill\n'
                '                if DL["on"]: DL["write_at"] = round(_dl_now(), 3)\n')
    return s


def value_patch(s):
    """--sched_value: the chain's DTLA at the first pass (and at 'ov'), its overflow stage are off; the value loop runs right before the
    critical-path loop and leaves the schedule expired, so the chain goes on to its write"""
    s = rep(s, '    if a.dgr and a.dgr_at == "fp" and not DGR_DEFER and ', '    if a.dgr and a.dgr_at == "fp" and not SCHEDV_ON and not DGR_DEFER and ')
    s = rep(s, '    if a.dgr and (a.dgr_at == "ov" or DGR_DEFER) and ', '    if a.dgr and not SCHEDV_ON and (a.dgr_at == "ov" or DGR_DEFER) and ')
    s = rep(s, '        if DLG("ovf_stage"):                               # merge6 (b)\n', '        if not SCHEDV_ON and DLG("ovf_stage"):                               # merge6 (b)\n')
    s = rep(s, '    nround = 0; stuck = False; maxr = max(snaps) if a.engine != "micro" else 0; gain_tot = 0.0\n',
            '    if SCHEDV_ON: cur = SCHED.value_loop(cur, {"crit": crit_round, "power": power_round_i, "PWS": PWS, "dgr": dgr_run, "settle": dl_settle, "DLC": DLC, '
            '"ovf": (lambda c_: overflow_stage(S, c_, a, W, log, K, mk)), "rs": rs_run, "log": log, "est5": est5, "DLG": DLG, "DLE": DLE, "a": a})   # tools/sched value rule\n'
            '    nround = 0; stuck = False; maxr = max(snaps) if a.engine != "micro" else 0; gain_tot = 0.0\n')
    return s


def extra_patch(s):
    """plan R2 (extra DTLA rounds before the write) and R1/R3 (exit after the first-pass DTLA stage); works without the model"""
    s = rep(s, '        cur = dgr_run(cur, "fp")\n',
            '        cur = dgr_run(cur, "fp")\n'
            '        if SCHEDX["exit_dgr"]: SCHEDX_EXIT(a, log, W)          # tools/sched (plan R1/R3): no route\n')
    s = rep(s, '                    cur = dgr_run(cur, "end")\n',
            '                    cur = dgr_run(cur, "end")\n'
            '                if SCHEDX["extra_dgr"] > 0 and a.dgr and not locals().get("ZT", False):   # tools/sched (plan R2): DTLA round(s) after the post-critical rounds\n'
            '                    _xr = a.dgr_rounds; a.dgr_rounds = int(SCHEDX["extra_dgr"])\n'
            '                    try:\n'
            '                        cur = dgr_run(cur, "end")\n'
            '                    finally:\n'
            '                        a.dgr_rounds = _xr\n')
    return s


def rules_patch(s):
    """the step rules of the scheduling plan, each only when its switch is on: A1 (DTLA round stop), A4 / power-first (around the
    snapshot's power round), skip_ovf (overflow stage), crit_frac (critical-path and small-design post-critical rounds); record"""
    X = EXTRA
    if X["stop_frac"] > 0:
        s = rep(s, '            if not k_ or rec_.get("gain", 0.0) < a.dgr_round_min:\n',
                '            if SCHEDX["stop_frac"] > 0 and SCHEDX_A1(rn_, k_, rec_):   # tools/sched A1\n'
                '                break\n'
                '            if not k_ or rec_.get("gain", 0.0) < a.dgr_round_min:\n')
    if X["a4"] or X["power_first"]:
        s = rep(s, '                    kept2, undo2, r3, dtp = power_round_i(cur)\n',
                '                    _pws = a.pw_skip                                   # tools/sched A4 / power-first rule\n'
                '                    if SCHEDX["a4"] and not SCHEDX.get("pf_fired"): a.pw_skip = SCHEDX_A4(cur, log, a.pw_skip)\n'
                '                    try:\n'
                '                        if SCHEDX.get("pf_fired"):\n'
                '                            kept2, undo2, r3, dtp = SCHEDX_PFNOOP(cur, PWS)\n'
                '                        else:\n'
                '                            kept2, undo2, r3, dtp = power_round_i(cur)\n'
                '                    finally:\n'
                '                        a.pw_skip = _pws\n')
    if X["power_first"]:
        s = rep(s, '    nround = 0; stuck = False; maxr = max(snaps) if a.engine != "micro" else 0; gain_tot = 0.0\n',
                '    if SCHEDX["power_first"] and a.power == "1" and not DL.get("expired") and not DL.get("power_first_done") and SCHEDX_PFRULE(cur):   # tools/sched power-first rule\n'
                '        if (not DL["on"]) or DLG("power_round"):\n'
                '            kept_pf_, undo_pf_, r_pf_, dt_pf_ = power_round_i(cur); cur = PWS.get("cur", cur)\n'
                '            if DL["on"]: DLE("power_round")\n'
                '            SCHEDX["pf_fired"] = True; SCHEDX["pf_wts"] = PWS.get("wts")\n'
                '            log.append({"stage": "power_round_first_rule", "kept": bool(kept_pf_), "promoted": len(undo_pf_), "s": dt_pf_, "est": est5(cur[2]), "pw_info": PWS.get("info")})\n'
                '    nround = 0; stuck = False; maxr = max(snaps) if a.engine != "micro" else 0; gain_tot = 0.0\n')
    if X["skip_ovf"]:
        s = rep(s, 'DLG("ovf_stage"):                               # merge6 (b)',
                '(not (SCHEDX["skip_ovf"] and SCHEDX_OVF(S, cur, a))) and DLG("ovf_stage"):   # merge6 (b) + tools/sched skip_ovf')
    if X["crit_frac"] > 0:
        s = rep(s, '            DLE("crit_round")\n            nround += 1\n            stuck = not kept\n',
                '            DLE("crit_round")\n            nround += 1\n            stuck = not kept\n'
                '            if SCHEDX["crit_frac"] > 0 and kept and SCHEDX_CRIT("crit", s_prev - cur[2]["score"]): stuck = True   # tools/sched crit_frac\n')
        s = rep(s, '                kp_, cur_, nprom, dt = crit_round(cur_)\n', '                _s0c = cur_[2]["score"]; kp_, cur_, nprom, dt = crit_round(cur_)\n')
        s = rep(s, '                if not kp_:\n                    break\n',
                '                if not kp_ or (SCHEDX["crit_frac"] > 0 and SCHEDX_SMALL() and SCHEDX_CRIT("post", _s0c - cur_[2]["score"])):   # tools/sched crit_frac\n'
                '                    break\n')
    s = rep(s, '    W["merge6"] = {"route_bin": int(a.route_bin), "deadline_s": a.deadline_s}\n',
            '    W["merge6"] = {"route_bin": int(a.route_bin), "deadline_s": a.deadline_s}\n'
            '    W["sched_extra"] = {k_: v_ for k_, v_ in SCHEDX.items() if k_ not in ("nets_log", "pf_wts")}   # tools/sched\n')
    return s


def _ovf_skip(S, cur, a):
    try:
        import numpy as np
        la = S.la; d = np.asarray(la.d); c = np.asarray(la.c)
        over = (d > c) & (c > 1e-3)
        n = int(over.sum()); bound = 0.0
        if n:
            import repair3d_astar as RA
            u = la.u[:, None, None]
            bound = float(abs(S.w4)) * float(np.where(over, RA._cost(u, d, c) - RA._cost(u, np.minimum(d, c), c), 0.0).sum())
        skip = bound < EXTRA["skip_ovf_thr"]
        EXTRA["ovf_log"].append({"n_over": n, "bound": round(bound, 6), "skip": bool(skip), "wns": float(cur[2]["wns"]), "tns": float(cur[2]["tns"])})
        return skip
    except Exception as e:                                  # never skip on an error
        EXTRA["ovf_log"].append({"error": repr(e)[:200]})
        return False


def _crit(kind, g):
    f = EXTRA["crit_first"]
    if kind not in f:
        f[kind] = float(g)
        EXTRA["crit_log"].append([kind, round(float(g), 6), False]); return False
    stop = f[kind] <= 0.0 or float(g) < EXTRA["crit_frac"] * f[kind]
    EXTRA["crit_log"].append([kind, round(float(g), 6), bool(stop)])
    return stop


def _small():
    try:
        return len(NS["PRE"]["S"].SD.net_names) < 300000
    except Exception:
        return False


def _pf_rule(cur):
    r = cur[2]; met = float(r["wns"]) >= 0.0 and float(r["tns"]) >= 0.0
    EXTRA["pf_log"].append({"wns": float(r["wns"]), "tns": float(r["tns"]), "fire": bool(met)})
    return met


def _pf_noop(cur, PWS):
    """the snapshot's power round when it already ran before the critical-path rounds: no change; its after-stages see the current state"""
    PWS["cur"] = cur; PWS["settled"] = True; PWS["wts"] = EXTRA.get("pf_wts", PWS.get("wts")); PWS["info"] = {"skipped": "power round ran first"}
    return False, {}, cur[2], 0.0


def _a1(rn, kept, rec):
    g = float(rec.get("gain", 0.0)) if kept else 0.0
    if rn == 0:
        EXTRA["g1"] = g
    g1 = float(EXTRA.get("g1", 0.0))
    stop = g1 <= 0.0 or g < EXTRA["stop_frac"] * g1
    EXTRA["a1_log"].append([int(rn), round(g, 6), round(g1, 6), bool(stop)])
    return stop


def _a4(cur, log, pw_skip):
    r = cur[2]; wns, tns = float(r["wns"]), float(r["tns"])
    if wns < 0.0 or tns < 0.0:
        EXTRA["a4_log"].append({"wns": wns, "tns": tns, "applied": False}); return pw_skip
    gains = []; prev = None
    for e in log:
        est = e.get("est"); sc = est.get("score") if isinstance(est, dict) else est
        if not isinstance(sc, (int, float)):
            continue
        if prev is not None and e.get("kept") and prev - float(sc) > 0:
            gains.append(prev - float(sc))
        prev = float(sc)
    thr = min(gains) if gains else 0.0
    new = max(float(pw_skip or 0.0), thr)
    EXTRA["a4_log"].append({"wns": wns, "tns": tns, "smallest_kept_gain": round(thr, 6), "pw_skip_used": round(new, 6), "applied": True})
    return new


def _exit_dgr(a, log, W):
    rec = {"design": a.design, "suite": a.suite, "exit_after_dgr": True, "extra": {k: v for k, v in EXTRA.items()},
           "log": [e for e in log if e.get("stage") in ("single_pass", "dgr")],
           "walls": {k: W[k] for k in ("singlepass_s", "dgr_s", "attach_timer_s", "build_s", "gnn_s") if k in W},
           "t_exit_s": time.perf_counter() - T0}
    if getattr(a, "json", None):
        Path(a.json).write_text(json.dumps(rec, indent=1, default=float))
    print("[sched] exit after the DTLA stage: " + json.dumps({"nets": EXTRA["nets_log"]}), flush=True)
    sys.stdout.flush(); os._exit(0)


def _h_n(n, elig_n=None, nets_n=None):
    if EXTRA["k"] is not None and elig_n is not None and elig_n > 300000:
        n = max(int(nets_n or 0), int(elig_n * EXTRA["k"]))
    EXTRA["nets_log"].append([int(elig_n if elig_n is not None else -1), int(n)])
    return SCHED.nets_hook(n) if SCHED is not None else n


def _h_o(o):
    return SCHED.order_hook(o) if SCHED is not None else o


def _h_freed(idx):
    if SCHED is not None:
        SCHED.freed_hook(idx)


def _h_run(plain_run, X, a, tau, reuse, tl):
    """the descent; --sched_k0 1: none (the start is what gets rounded and checked)"""
    if not EXTRA["k0"]:
        return plain_run(reuse, tl)
    e = X.eval(T=1.0, mode=1, tau=tau, grad=False)
    v = [float(e.get(k, 0.0)) for k in ("of", "pw", "wns", "tns", "total")] + [0.0, 0.0, 0.0]
    return {"steps": 0, "vals": np.array([v, v]), "prof": {}}


def dgr_patch(src):
    """the DTLA stage's net choice through SCHED: count (nets_hook, --sched_fit), ranked list (order_hook: fresh nets in the fill),
    chosen nets (freed_hook: recorded); three insertions in prepare"""
    src = rep(src, '    o = elig[np.argsort(-score[elig], kind="stable")]\n',
              '    o = _SCHED_O(elig[np.argsort(-score[elig], kind="stable")])   # tools/sched\n')
    src = rep(src, '        o = o[:(nets_n if len(elig) <= 300000 else max(nets_n, int(len(elig) * 0.07)))]\n',
              '        o = o[:_SCHED_N(nets_n if len(elig) <= 300000 else max(nets_n, int(len(elig) * 0.07)), len(elig), nets_n)]   # tools/sched\n')
    src = rep(src, '        res = plain_run(True, tl)\n', '        res = _SCHED_RUN(plain_run, X, a, tau, True, tl)          # tools/sched (plan R3: --sched_k0)\n')
    return rep(src, '    idx = np.sort(o)\n', '    idx = np.sort(o); _SCHED_FREED(idx)                           # tools/sched\n')


if SCHED is not None:
    s = sched_patch(s)
    if SCHED.value_mode:
        s = value_patch(s)
if EXTRA["extra_dgr"] > 0 or EXTRA["exit_dgr"]:
    s = extra_patch(s)
if EXTRA["stop_frac"] > 0 or EXTRA["a4"] or EXTRA["skip_ovf"] or EXTRA["crit_frac"] > 0 or EXTRA["power_first"]:
    s = rules_patch(s)
if (SCHED is not None and (SCHED.fit or SCHED.fill_on)) or EXTRA["k"] is not None or EXTRA["k0"]:
    if True:
        import importlib.abc, importlib.util

        class _DgrHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path=None, target=None):
                if name != "dgr_stage_r":
                    return None
                f = ROOT / "tools" / "best8" / "dgr_stage_r.py"
                return importlib.util.spec_from_file_location(name, str(f), loader=self)

            def create_module(self, spec):
                return None

            def exec_module(self, mod):
                src_ = dgr_patch(Path(mod.__spec__.origin).read_text())
                mod.__dict__["_SCHED_N"] = _h_n; mod.__dict__["_SCHED_O"] = _h_o
                mod.__dict__["_SCHED_FREED"] = _h_freed; mod.__dict__["_SCHED_RUN"] = _h_run
                exec(compile(src_, mod.__spec__.origin, "exec"), mod.__dict__)
        sys.meta_path.insert(0, _DgrHook())


def _prof_dump():
    if not PROF["on"] or PROF["done"]:
        return
    PROF["done"] = True
    try:
        PROF["smp"].report(SOPT["--sched_prof"])
    except Exception as e:
        ST["prof_err"] = repr(e)


def _dump():
    _prof_dump()
    ST["process_s_at_dump"] = time.perf_counter() - T0
    ST["loadavg"] = os.getloadavg()
    ST["act_mode"] = MODE
    ST["pre5_active"] = bool(NS.get("PRE5"))
    try:
        p5 = (NS.get("PRE") or {}).get("pre5") or {}
        ST["preload_end_s"] = (NS.get("PRE") or {}).get("preload_end")
        ST["pre5_act_needed_at_s"], ST["pre5_act_wait_s"] = p5.get("act_needed_at_s"), p5.get("act_wait_s")
        ST["pre5_act_file_skipped"] = p5.get("act_file_skipped")
    except Exception:
        pass
    if "t_state_s" in COLD:
        ST["t_state_s"] = COLD["t_state_s"]
    ST["fixcut"] = dict(FC_LIB.ST, src=_FCS)
    if IO.get("audit_open"):
        Path(IO["audit_open"]).write_text("\n".join(f"{v}\t{k}" for k, v in _OPENED.items()) + "\n")
    if IO.get("cold_json"):
        Path(IO["cold_json"]).write_text(json.dumps(ST, indent=1, default=float))


_exit = os._exit
def _exit_dump(code, _e=_exit):
    try:
        _dump()
    finally:
        _e(code)
os._exit = _exit_dump
if os.environ.get("N1_DRYRUN"):                             # patch check: the executed source, then stop
    Path(os.environ["N1_DRYRUN"]).write_text(s); print("N1_DRYRUN ok", len(s), "n2_fix", ST.get("n2_fix", False), "cuts", sorted(FC_LIB.CUTS)); _exit(0)
NS.update({"__name__": "__main__", "__file__": str(SRC), "COLD": COLD, "FC_LIB": FC_LIB, "_N2_COPY": _n2_copy, "SCHED": SCHED, "SCHEDX": EXTRA, "SCHEDX_EXIT": _exit_dgr,
           "SCHEDV_ON": bool(SCHED is not None and SCHED.value_mode), "SCHEDX_A1": _a1, "SCHEDX_A4": _a4,
           "SCHEDX_OVF": _ovf_skip, "SCHEDX_CRIT": _crit, "SCHEDX_SMALL": _small, "SCHEDX_PFRULE": _pf_rule, "SCHEDX_PFNOOP": _pf_noop})
try:
    exec(compile(s, str(SRC), "exec"), NS)
finally:
    _dump()
