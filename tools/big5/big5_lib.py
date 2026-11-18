#!/usr/bin/env python3
"""big5_lib.py: the --big5 1 pieces of tools/big5/chain_big.py (copy of tools/best2/chain_j.py).
an internal note. Everything here is called from five short hunks of chain_big.py; the speed pieces are installed
by patching the modules / objects chain_j already calls (so chain_j's own lines stay as they are and `git merge-file` with
the other copies of chain_j (tools/pre5, tools/mid5, tools/late5 -> tools/best3/chain_k.py) does not conflict):

  early()      at import (after BIG is known): memory/CPU sampler thread (--big5_mem 1|2); in big mode with --big5_act child the
               switching activity of tools/pre5/act_child.py in a CHILD process from t0 (pre5 did not enable it in big mode:
               its loader replaces chain_j's big-mode load path; only the activity is taken here, on chain_j's own loader).
  KEEP         big mode: the reset of chain_j (h_* switches, overflow-stage impl / timer) is not applied to these keys.
  state_class  ChainState with the lazy activity (tools/pre5/pre5_load._cold_state) when the activity child runs.
  install()    after the preload, before the route is read (big mode, each piece with its own --big5_x switch, default on):
                 sel : overflow-stage selection ovf2_lib.select_c_fast -> tools/late5/l5.select_c (C++ pairs + greedy; same chosen)
                 esc : overflow-stage escape pass -> tools/mid5/mid5.escape_pass (C++) with the LazyPins seen as a dict
                 norm: fast_norm._normalise -> tools/late5/l5.normalise (same blocks)
                 pf  : promote_full.promote_cpp / promote_pw -> tools/late5/l5_pf (FLUTE preparation in parallel; same result)
                 s4  : tools/speed4 s4_ovf (cached official overflow cost) + s4_escape (C++ escape / re-route cost)
               plus stage marks on the timer / build calls (no value changes).
  at_stages()  right after chain_j builds its stage log (before the overflow stage): S.revert -> tools/late5/l5.make_revert
               (rev); the log becomes a MarkList (every appended stage record is a time / memory / CPU mark).
  record()     json "big5": marks, peak memory by segment, activity child, piece counters; --big5_mem 2: the largest holders.
"""
import gc, os, sys, threading, time, types, json, subprocess
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEEP = ("h_read", "h_write", "h_dem", "h_asm", "h_inc", "h_crit", "h_graph", "h_all", "h_pre", "ovf_impl", "ovf_timer")
R = {"marks": [], "pieces": {}, "t0": None}
EARLY = {}
_PG = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _arg(argv, k, d=None):
    for i, v in enumerate(argv):
        if v == k and i + 1 < len(argv):
            return argv[i + 1]
    return d


def _rss_pid(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PG
    except Exception:
        return 0


def _cpu_pid(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        p = s[s.rindex(")") + 2:].split()
        return (int(p[11]) + int(p[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return 0.0


class _Sampler:
    """RSS of this process and of the activity child every dt s (CLOCK perf_counter relative to T0); peak kept exactly"""
    def __init__(self, dt):
        self.dt, self.s, self.peak, self.peak_sum, self.stop = dt, [], 0, 0, False
        self.child = None
        self.th = threading.Thread(target=self._run, name="big5_mem", daemon=True); self.th.start()

    def _run(self):
        me = os.getpid(); k = 0
        while not self.stop:
            r = _rss_pid(me); c = _rss_pid(self.child) if self.child else 0
            self.peak = max(self.peak, r); self.peak_sum = max(self.peak_sum, r + c)
            if k % max(1, int(1.0 / self.dt)) == 0:          # one stored sample per second
                self.s.append((round(time.perf_counter() - R["t0"], 2), round(r / 1e9, 3), round(c / 1e9, 3)))
            k += 1
            time.sleep(self.dt)


def early(root, argv, t0, bigm):
    R["t0"] = t0
    R["mem_level"] = int(_arg(argv, "--big5_mem", "0") or 0)
    R["bigm"] = bool(bigm)
    if R["mem_level"]:
        EARLY["mem"] = _Sampler(0.2)
    if bigm and _pc("graph"):                                  # pre5's C++ levelize / SCC in StaDesign._graph (exact; pre5_check)
        sys.path.append(str(ROOT / "tools" / "pre5"))
        import fast_sta_v3 as FS
        import pre5_kern as PK
        FS.StaDesign._graph = PK.graph
        R["pieces"]["graph"] = True
    cpath = _arg(argv, "--cache")
    if bigm and cpath and _pc("cache"):                         # timer cache read from t0 on a thread (zlib releases the GIL);
        import csta as CS_                                       # load_all's CS.load_cache(path) gets this object (same function)
        o_lc = CS_.load_cache
        box = {}
        def _lc():
            t = time.perf_counter()
            try:
                box["v"] = o_lc(cpath)
            except BaseException as e:
                box["e"] = e
            R["pieces"]["cache"] = {"read_s": round(time.perf_counter() - t, 2), "end_s": round(time.perf_counter() - R["t0"], 2)}
        th = threading.Thread(target=_lc, name="big5_cache", daemon=True); th.start()
        def load_cache(path, _o=o_lc):
            if os.path.realpath(str(path)) == os.path.realpath(cpath) and th is not None:
                t = time.perf_counter(); th.join(); R["pieces"]["cache"]["wait_s"] = round(time.perf_counter() - t, 2)
                CS_.load_cache = _o
                if "e" in box:
                    raise box["e"]
                return box.pop("v")
            return _o(path)
        CS_.load_cache = load_cache
    act = _arg(argv, "--big5_act", "file")
    R["act_mode"] = act
    if bigm and act == "child":                              # pre5's activity child (tools/pre5/act_child.py), raw files only
        sys.path.append(str(ROOT / "tools" / "pre5"))
        import pre5_load as P5L
        if P5L.T0 is None:
            P5L.T0 = t0
        nt = int(_arg(argv, "--big5_act_threads", "0") or 0) or max(2, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)
        d, s = _arg(argv, "--design"), _arg(argv, "--suite", "visible")
        t = time.perf_counter()
        pr = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools" / "pre5" / "act_child.py"), d, s, str(nt)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        R["act"] = {"threads": nt, "spawn_s": time.perf_counter() - t, "pid": pr.pid}
        if "mem" in EARLY:
            EARLY["mem"].child = pr.pid
        EARLY["act_pr"] = pr

        def child():
            out, err = pr.communicate()
            R["act"]["done_at_s"] = time.perf_counter() - t0
            R["act"]["cpu_s"] = None
            if pr.returncode != 0 or not out.startswith(b"P5ACT"):
                raise RuntimeError("act_child failed rc=%s: %s" % (pr.returncode, err.decode(errors="replace")[-2000:]))
            n = int(np.frombuffer(out[5:13], np.int64)[0])
            dens = np.frombuffer(out[13:13 + 8 * n], np.float64).copy()
            for ln in err.decode(errors="replace").splitlines():
                if ln.startswith("P5ACTINFO "):
                    R["act"]["info"] = json.loads(ln[10:])
            return dens
        P5L.EARLY["ACT"] = P5L._Fut("ACT", child)


def state_class(CS, G=None, PRE=None):
    """ChainState for --big5 1: the activity child's lazy kpn (big mode, --big5_act child); big mode: its __init__ (called right
    after the timer kernel is bound, chain_j load_all line 274) first starts the EARLY TIMER thread: route read + timer segments
    + RC + the cold full analysis of the router route, next to the rest of the preload (bigspeed --bs_early, here without hunks:
    main's own calls get these objects back from identity-keyed wrappers, see install)."""
    thr_act = R.get("bigm") and R.get("act_mode") == "thread"
    if "act_pr" in EARLY or thr_act:
        sys.path.append(str(ROOT / "tools" / "pre5")); sys.path.append(str(ROOT / "tools" / "speed4"))   # act_inproc (speed4)
        import pre5_load as P5L
        if P5L.T0 is None:
            P5L.T0 = R["t0"]
        CS = P5L._cold_state(CS, "child")                       # lazy kpn from P5L.EARLY["ACT"] (child or thread)
    if not (R.get("bigm") and (_pc("early") or thr_act) and G is not None):
        return CS

    class EarlyState(CS):
        def __init__(self, design, suite):
            self._big5_init(design, suite)
            if thr_act and "ACT" not in P5L.EARLY:            # --big5_act thread: pre5's _act on THIS StaDesign (no second copy)
                nt = int(_arg(sys.argv, "--big5_act_threads", "0") or 0) or max(2, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)
                R["act"] = {"mode": "thread", "threads": nt, "start_s": time.perf_counter() - R["t0"]}
                def _a(SD=self.SD, nt=nt):
                    v = P5L._act(SD, nt); R["act"]["done_at_s"] = time.perf_counter() - R["t0"]; return v
                P5L.EARLY["ACT"] = P5L._Fut("ACT", _a)

        def _big5_init(self, design, suite):
            if not _pc("early"):
                return super().__init__(design, suite)
            if _pc("dfull") and "dfull" not in R["pieces"] and hasattr(PRE.get("K"), "set_rc"):
                install_dfull(PRE["K"])                         # before the first full analysis (it must see it)
            if "tm" not in EARLY:
                EARLY["tm"] = threading.Thread(target=_early_timer, args=(G, PRE), name="big5_early", daemon=True)
                EARLY["tm"].start()
            super().__init__(design, suite)
    return EarlyState


def install_dfull(K, frac=0.5):
    """tools/bigspeed/chain_bs.py bs_install_dfull (d6 --bs_dfull 1; its check mode left out): every K.analyze on the incremental
    kernel when few nets changed = begin / set_rc / update / commit + w3_inc_backward on the nets whose RC differs bit for bit
    (an Elmore pin row is mapped to its net; unmapped or shared rows, an open transaction, the first call or > frac of the nets
    -> the full analysis). bigspeed: bit-equal to the full analysis in its check mode; same md5 on bsg / group."""
    import ctypes
    D = K.D; nn = len(D.net_names); P = int(D.P)
    ap_off = np.asarray(K.keep["ap_off"], np.int64); ap_idx = np.asarray(K.keep["ap_idx"], np.int64); ap_pid = np.asarray(K.keep["ap_pid"], np.int64)
    pins_t = ap_pid[ap_idx]; net_t = np.repeat(np.arange(nn, dtype=np.int64), np.diff(ap_off))
    okp = pins_t >= 0
    pnet = np.full(P, -1, np.int64); pnet[pins_t[okp]] = net_t[okp]
    multi = np.bincount(pins_t[okp], minlength=P) > 1
    del pins_t, net_t, okp, ap_off, ap_idx, ap_pid
    st = {"calls": 0, "diff": 0, "full": 0, "fallback": {}, "nets": [], "s_diff": 0.0, "s_full": 0.0}
    R["pieces"]["dfull"] = st
    o_full, o_begin, o_commit, o_reject = K.full, K.begin, K.commit, K.reject
    K._bs_state = False; K._bs_open = False
    def full(*aa, **kw):
        r = o_full(*aa, **kw); K._bs_state = True; K._bs_open = False; return r
    def begin():
        o_begin(); K._bs_open = True
    def commit():
        o_commit(); K._bs_open = False
    def reject():
        n = o_reject(); K._bs_open = False; return n
    K.full, K.begin, K.commit, K.reject = full, begin, commit, reject
    u64 = lambda x: np.ascontiguousarray(x).view(np.uint64)

    def analyze(rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par):
        t0 = time.perf_counter(); st["calls"] += 1
        Rr = K.rc; A = dict(c1=rc_c1, c2=rc_c2, rpi=rc_rpi, ctot=rc_ctot, elm=rc_elm, par=rc_par)
        why = None
        if not K._bs_state: why = "first"
        elif K._bs_open: why = "open_txn"
        else:
            for k in A:
                x = np.asarray(A[k])
                if x.dtype != Rr[k].dtype or x.size != Rr[k].size:
                    why = "dtype"; break
                A[k] = np.ascontiguousarray(x.reshape(Rr[k].shape))
        if why is None:
            ch = np.asarray(A["par"]) != Rr["par"]
            for k in ("c1", "c2", "rpi", "ctot"):
                ch |= (u64(A[k]) != u64(Rr[k])).any(1)
            pe = np.flatnonzero((u64(A["elm"]) != u64(Rr["elm"])).any(1))
            if len(pe):
                if (pnet[pe] < 0).any() or multi[pe].any():
                    why = "elm_unmapped"
                else:
                    ch[pnet[pe]] = True
            nets = np.flatnonzero(ch).astype(np.int64)
            if why is None and len(nets) > frac * nn: why = "many"
        if why is not None:
            st["full"] += 1; st["fallback"][why] = st["fallback"].get(why, 0) + 1
            o = full(rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par, backward=True, copy=True)
            st["s_full"] += time.perf_counter() - t0
            return o
        o_begin(); K.set_rc(nets, A); K.update(nets); o_commit()
        stb = np.zeros(2); K.L.w3_inc_backward(stb.ctypes.data_as(ctypes.c_void_p))
        if K.S is not None:
            K.snap = dict(K.S.new); K.pending_snap = None
        o = K._out(np.zeros(4), True)
        st["diff"] += 1; st["nets"].append(int(len(nets))); st["s_diff"] += time.perf_counter() - t0
        return o
    K.analyze = analyze


def _early_timer(G, PRE):
    t = time.perf_counter(); E = R.setdefault("early", {"start_s": t - R["t0"]})
    try:
        route = _arg(sys.argv, "--route")
        CS, FT = G["CS"], G["FT"]
        cd, K = PRE["cd"], PRE["K"]
        rt = G["read_route"](str(route)); t1 = time.perf_counter()
        seg = CS.seg_from_route(cd, rt, FT.GRID); t2 = time.perf_counter()
        rcd = K.build_rc(seg); t3 = time.perf_counter()
        o = K.analyze(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"]); t4 = time.perf_counter()
        EARLY["tm_out"] = {"route": str(route), "rt": rt, "seg": seg, "rcd": rcd, "o": o}
        E.update(read_s=t1 - t, seg_s=t2 - t1, rc_s=t3 - t2, analyze_s=t4 - t3, end_s=t4 - R["t0"])
    except BaseException as e:
        import traceback; traceback.print_exc()
        E["err"] = repr(e)


def _use_early(G, K, CS):
    """main's first route read / segments / RC / analysis return the early thread's objects (first call each, identity-checked
    on the arguments: the same inputs, so the same values); any other call goes to the original"""
    th = EARLY.get("tm")
    if th is None:
        return
    t = time.perf_counter(); th.join(); R["early"]["join_wait_s"] = time.perf_counter() - t
    out = EARLY.pop("tm_out", None)
    if out is None:
        return
    st = R["early"]; st["hits"] = []
    rr, sg, br, an = G["read_route"], CS.seg_from_route, K.build_rc, K.analyze

    def read_route(p, *aa, **kk):
        if out.get("rt") is not None and str(p) == out["route"]:
            v = out.pop("rt"); st["hits"].append("read"); return v
        return rr(p, *aa, **kk)

    def seg_from_route(cd, rt, *aa, **kk):
        if out.get("seg") is not None and rt is out.get("rt_ref"):
            v = out.pop("seg"); st["hits"].append("seg"); return v
        return sg(cd, rt, *aa, **kk)

    def build_rc(seg, *aa, **kk):
        if out.get("rcd") is not None and seg is out.get("seg_ref") and not aa and not kk:
            v = out.pop("rcd"); st["hits"].append("rc"); return v
        return br(seg, *aa, **kk)

    def analyze(c1, *aa, **kk):
        if out.get("o") is not None and out.get("rcd_ref") is not None and c1 is out["rcd_ref"]["c1"]:
            v = out.pop("o"); out.pop("rcd_ref", None); st["hits"].append("analyze"); return v
        return an(c1, *aa, **kk)
    out["rt_ref"], out["seg_ref"], out["rcd_ref"] = out["rt"], out["seg"], out["rcd"]
    G["read_route"] = read_route; CS.seg_from_route = seg_from_route; K.build_rc = build_rc; K.analyze = analyze


def _pc(name, default="1"):
    v = os.environ.get("BIG5_" + name.upper())
    return (v if v is not None else default) == "1"


class _DictPins(dict):
    """an empty dict whose lookups go to the big-mode LazyPins (mid5.escape_pass checks isinstance(pins, dict) and reads pins.get only)"""
    def __init__(self, lp): super().__init__(); self._lp = lp
    def get(self, k, d=None):
        try:
            return self._lp[k]
        except KeyError:
            return d
    def __getitem__(self, k): return self._lp[k]
    def __contains__(self, k): return k in self._lp
    def __len__(self): return len(self._lp)
    def __iter__(self): return iter(self._lp)


def mark(name):
    t = time.perf_counter() - R["t0"]
    m = EARLY.get("mem")
    R["marks"].append((name, round(t, 3), round(time.process_time(), 2),
                       round(_rss_pid(os.getpid()) / 1e9, 3) if m else None, round(m.peak / 1e9, 3) if m else None))


def _wrap_mark(fn, name, once=True):
    st = {"n": 0}
    def w(*aa, **kk):
        try:
            return fn(*aa, **kk)
        finally:
            st["n"] += 1
            if not once or st["n"] == 1:
                mark(name)
    w._big5 = True
    return w


def _timed(fn, name):
    c = R.setdefault("calls", {})
    def w(*aa, **kk):
        t = time.perf_counter()
        try:
            return fn(*aa, **kk)
        finally:
            dt = time.perf_counter() - t
            e = c.setdefault(name, [0, 0.0, 0.0, []]); e[0] += 1; e[1] += dt; e[2] = max(e[2], dt)
            if len(e[3]) < 12:
                e[3].append((round(t - R["t0"], 1), round(dt, 2)))
    return w


def install(a, G, PRE, W, bigm, PROM):
    """after the preload; big-mode pieces. Returns the PROM module to use (same object, functions patched)."""
    mark("preload_joined")
    R["PRE"] = PRE
    P = R["pieces"]
    G["read_route"] = _wrap_mark(G["read_route"], "route_read")
    K = PRE["K"]
    if not bigm and _pc("dfull") and "dfull" not in R["pieces"] and hasattr(K, "set_rc"):
        install_dfull(K)                                       # small mode (bsg): before the first analysis, which follows
    K.crit = _wrap_mark(_timed(K.crit, "K.crit"), "first_timer")   # first call = the first exact timer + crit
    for nm in ("build_rc", "analyze", "update", "build_rc_subset"):
        if hasattr(K, nm):
            setattr(K, nm, _timed(getattr(K, nm), "K." + nm))
    CS = G.get("CS")
    if CS is not None:
        CS.seg_from_route = _timed(CS.seg_from_route, "CS.seg_from_route"); CS.to_crit_data = _timed(CS.to_crit_data, "CS.to_crit_data")
    if bigm and CS is not None:
        _use_early(G, K, CS)                                   # big mode: the early timer's objects (identity-keyed)
    if "B2" in G:
        G["B2"].build_fast2 = _wrap_mark(G["B2"].build_fast2, "build")
    if R["mem_level"] >= 2:
        R["holders_preload"] = holders({"PRE": PRE})
    if _pc("sel") and "OVF2" in G:                              # late5 l5.select_c (C++) for the overflow-stage selection
        sys.path.append(str(ROOT / "tools" / "late5"))
        import l5 as L5
        O2 = G["OVF2"]; orig = O2.select_c_fast
        P["sel"] = {"cpp": 0, "fallback": 0}
        def sel(S, la, slack_r, npins_r, cap_frac=None, max_nets=0, cost=None, _o=orig):
            r = L5.select_c(S, la, slack_r, npins_r, cap_frac, max_nets)
            if r is None:
                P["sel"]["fallback"] += 1
                return _o(S, la, slack_r, npins_r, cap_frac, max_nets, cost=cost)
            P["sel"]["cpp"] += 1
            return r
        O2.select_c_fast = sel
    if _pc("esc") and "OL" in G:                                # mid5 C++ escape pass (LazyPins seen as a dict)
        sys.path.append(str(ROOT / "tools" / "mid5"))
        import mid5 as M5
        import escape3d as E
        from timing_promote import _key as _k
        OL = G["OL"]; o_esc = OL.escape_pass; m_esc = M5.escape_pass
        P["esc"] = M5.ESC

        class _OLx:                                            # OL as mid5 sees it: its fallback is the original pass
            def __getattr__(self, k): return getattr(OL, k)
        OLx = _OLx(); OLx.escape_pass = o_esc

        def m5(S, pins, n_hot, guarded, OL_, E_, PRE_, k_, W=12, stats=None, astar="win", _m=m_esc):
            if not isinstance(pins, dict):
                pins = _DictPins(pins)
            return _m(S, pins, n_hot, guarded, OLx, E_, PRE_, k_, W=W, stats=stats, astar=astar)
        M5.escape_pass = m5                                    # also what chain_k --mid5 1 calls

        def esc(S, pins, n_hot, guarded, W=12, stats=None, astar="win"):
            return m5(S, pins, n_hot, guarded, OLx, E, PRE, _k, W=W, stats=stats, astar=astar)
        OL.escape_pass = esc
    if not bigm:                                               # small mode: dfull, sel, esc only (late5's own flag does the rest)
        return
    if _pc("norm"):
        sys.path.append(str(ROOT / "tools" / "late5"))
        import l5 as L5
        import fast_norm as FN
        FN._normalise = L5.normalise
        P["norm"] = True
    if _pc("pf") and PROM is not None:
        sys.path.append(str(ROOT / "tools" / "late5"))
        import l5_pf as L5PF
        PROM.promote_cpp, PROM.promote_pw = L5PF.promote_cpp, L5PF.promote_pw
        P["pf"] = True
    if _pc("s4", "0"):                                         # speed4 4b + 4c (chain_j d; reset in big mode by chain_j)
        sys.path.append(str(ROOT / "tools" / "speed4"))
        if a.of_rule == "official" and a.ovf != "off" and "S4OFC" not in G:
            import s4_ovf, of_official
            G["S4OFC"] = s4_ovf.install(G["FM"], of_official, False)
        import s4_escape
        s4_escape.install(False); G["S4ESC"] = s4_escape
        P["s4"] = True


def _wcache(S, rt):
    """== the per-state cache of h_write.write_par (same function calls), computed on a thread during the overflow stage"""
    import h_write as HW
    t = time.perf_counter()
    pa0, pa1 = HW._pin_ranges(S, rt.net_names)
    ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
    blob = ("\n".join(rt.net_names) + "\n").encode()
    S._h_wcache = (rt.net_names, pa0, pa1, ap, blob)
    R["pieces"]["wcache_s"] = round(time.perf_counter() - t, 2)


def write_par5(S, rt, noff, cols, arr6, out, W, nthreads=8):
    """== h_write.write_par (same arrays to the same C++ writer, same bytes): the rows of the changed nets flattened with
    np.fromiter (no list of tuples), the per-state cache from the thread started before the overflow stage"""
    import h_write as HW
    from itertools import chain
    t = time.perf_counter()
    L = HW._lib()
    nn = len(rt.net_names)
    off = noff.astype(np.int64)
    start = off[:-1].copy(); cnt = np.diff(off).copy()
    segs = arr6.astype(np.int32)
    if S.new:
        ids = np.fromiter(S.new.keys(), np.int64, len(S.new))
        lists = list(S.new.values())
        ln = np.fromiter((len(x) for x in lists), np.int64, len(lists))
        st_ = np.zeros(len(ln), np.int64)
        if len(ln) > 1:
            st_[1:] = np.cumsum(ln)[:-1]
        start[ids] = len(segs) + st_; cnt[ids] = ln              # dict order = the original loop's order
        m = int(ln.sum())
        ex = np.fromiter(chain.from_iterable(chain.from_iterable(lists)), np.int32, 6 * m).reshape(m, 6) if m else np.zeros((0, 6), np.int32)
        segs = np.concatenate([segs, ex], 0)
    segs = np.ascontiguousarray(segs)
    th = EARLY.pop("wc", None)
    if th is not None:
        th.join()
    c = S.__dict__.get("_h_wcache")
    if c is None or c[0] is not rt.net_names:
        pa0, pa1 = HW._pin_ranges(S, rt.net_names)
        ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        blob = ("\n".join(rt.net_names) + "\n").encode()
        c = S._h_wcache = (rt.net_names, pa0, pa1, ap, blob)
    _, pa0, pa1, ap, blob = c
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W["w_prep_s"] = time.perf_counter() - t; t = time.perf_counter()
    rc = L.iow_write(os.fsencode(str(out)), blob, nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data,
                     pa0.ctypes.data, pa1.ctypes.data, ap.ctypes.data, 4200, int(nthreads))
    assert rc >= 0, rc
    W["w_norm_io_s"] = time.perf_counter() - t
    import ctypes
    tt = (ctypes.c_double * 2)(); L.iow_times(tt)
    W["w_norm_fmt_s"], W["w_file_s"] = tt[0], tt[1]


class MarkList(list):
    def append(self, rec):
        super().append(rec)
        try:
            mark("stage:" + str(rec.get("stage")))
            if rec.get("stage") == "ovf" and R.get("bigm") and _pc("free"):     # pins / .net parse: read only up to here
                PRE = R["PRE"]; fr = []
                for k in ("esc_pins", "esc_nd", "ovf_pcmap", "ovf_npins_key", "mid5_keys"):
                    if PRE.pop(k, None) is not None:
                        fr.append(k)
                FLD = sys.modules.get("fast_load")
                if FLD is not None and getattr(FLD, "_NET", None):
                    FLD._NET.clear(); fr.append("fast_load._NET")
                gc.collect(); _trim()
                R["freed_after_ovf"] = fr
                mark("freed_after_ovf")
            if R["mem_level"] >= 2 and rec.get("stage") in ("ovf", "power_round_after_5"):
                R.setdefault("holders_stage", {})[str(rec.get("stage"))] = holders({"main": _main_locals(), "PRE": R["PRE"]})
        except Exception as e:                                 # marks never change the run
            R.setdefault("mark_err", repr(e))


def _main_locals():
    f = sys._getframe(1)
    while f is not None and f.f_code.co_name != "main":
        f = f.f_back
    if f is None:
        return {}
    d = dict(f.f_locals); f.f_locals.clear()                   # no stale references left in the frame's cached dict
    return d


def at_stages(log, S, SegsCls, W, bigm):
    """before the overflow stage: the stage log as a MarkList; big mode: late5's C++ revert"""
    mark("check_done")
    f = sys._getframe(1)
    if R["mem_level"] >= 2:
        _d = dict(f.f_locals); f.f_locals.clear()
        R["holders_check"] = holders({"main": _d, "PRE": R["PRE"]}); del _d
    if _pc("free"):
        try:
            R["freed_main"] = free_main(f)                     # CPython 3.12 (PyFrame_LocalsToFast); elsewhere skipped
        except Exception as e:
            R["freed_main"] = ["skipped: " + repr(e)]
        mark("freed_main")
    if _pc("write"):                                   # write: cache on a thread now, faster row flatten at the write
        import h_write as HW
        rt = f.f_locals.get("rt"); f.f_locals.clear()
        if rt is not None and getattr(HW, "write_par", None) is not None:
            HW.write_par = write_par5
            EARLY["wc"] = threading.Thread(target=_wcache, args=(S, rt), name="big5_wcache", daemon=True); EARLY["wc"].start()
            R["pieces"]["write"] = True
        del rt
    if _pc("rev"):
        sys.path.append(str(ROOT / "tools" / "late5"))
        import l5 as L5
        S.revert = L5.make_revert(S, SegsCls, S.revert)
        R["pieces"]["rev"] = L5.STATS
    return MarkList(log)


def _size(o, seen, depth=0, smp=False):
    """(owned bytes, native-view bytes) held by o: numpy buffers once per base (a base that is not an ndarray = memory owned by
    a C++ library or bytes object: counted as view), containers recursively (large lists / dicts / sets sampled)"""
    i = id(o)
    if i in seen:
        return 0, 0
    seen.add(i)
    if isinstance(o, np.ndarray) and smp:                     # inside a sampled container: the element's own bytes only
        return int(o.nbytes) + 112, 0
    if isinstance(o, np.ndarray):
        b = o
        while isinstance(b.base, np.ndarray):
            b = b.base
        if b is not o:
            if id(b) in seen:
                return 0, 0
            seen.add(id(b))
        if b.base is None:
            return int(b.nbytes) + 112, 0
        return 112, int(b.nbytes)
    sz = sys.getsizeof(o, 64)
    if depth > 6 or isinstance(o, (str, bytes, bytearray, int, float, bool, types.ModuleType, type, types.FunctionType)):
        return sz, 0
    ow = vw = 0
    if isinstance(o, (list, tuple, set, frozenset, dict)):
        it = list(o.items()) if isinstance(o, dict) and len(o) <= 2000 else None
        n = len(o)
        if n == 0:
            return sz, 0
        if n > 2000:                                           # sampled (fresh seen: shared small objects counted per container)
            smp_ = [x for _, x in zip(range(1000), (o.items() if isinstance(o, dict) else o))]
            for x in smp_:
                a_, b_ = _size(x, set(), depth + 1, True); ow += a_; vw += b_
            return sz + int(ow * n / len(smp_)), int(vw * n / len(smp_))
        for x in (it if it is not None else o):
            a_, b_ = _size(x, seen, depth + 1, smp); ow += a_; vw += b_
        return sz + ow, vw
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict):
        a_, b_ = _size(d, seen, depth + 1, smp)
        return sz + a_, b_
    return sz, 0


def _kids(o):
    if isinstance(o, dict):
        return [(str(k), v) for k, v in o.items()]
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict) and not isinstance(o, (type, types.ModuleType, types.FunctionType)):
        return list(d.items())
    return []


def holders(roots, top=25, depth=3):
    """largest holders by name path (root.attr.attr), deepest name first so a buffer is charged to its most specific name;
    owned = numpy-owned + Python objects, view = arrays over memory a C++ library or a bytes object owns"""
    t = time.perf_counter()
    items = []
    def walk(nm, o, d):
        if isinstance(o, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)) or callable(o) and not isinstance(o, (dict, list)) and not hasattr(o, "__dict__"):
            return
        items.append((nm, o, d))
        if d < depth:
            for k, v in _kids(o):
                if isinstance(v, (int, float, str, bool, type(None))):
                    continue
                walk(nm + "." + k, v, d + 1)
    for rn, r in roots.items():
        for k, v in _kids(r):
            walk(rn + "." + k, v, 1)
    items.sort(key=lambda x: -x[2])
    seen = set(); out = []
    for nm, v, _ in items:
        try:
            ow, vw = _size(v, seen)
        except Exception:
            ow = vw = 0
        if ow + vw > 100e6:
            out.append((nm, round(ow / 1e9, 3), round(vw / 1e9, 3), type(v).__name__))
    out.sort(key=lambda x: -(x[1] + x[2]))
    rss = _rss_pid(os.getpid()) / 1e9
    return {"rss_gb": round(rss, 3), "owned_gb": round(sum(x[1] for x in out), 3), "view_gb": round(sum(x[2] for x in out), 3),
            "top": out[:top], "s": round(time.perf_counter() - t, 2)}


FREE_MAIN = ("res", "o", "rcd", "seg", "cr", "crit_data", "timer_rec", "P", "x", "ei", "sid", "lg", "out_", "chs", "dem_np",
             "dem_p", "cands", "noff_p", "cols_p", "est", "SD5", "seg_p5")   # SD5 / seg_p5: mid5's prior-candidate arrays (chain_k)


def free_main(frame, names=FREE_MAIN):
    """drop main's references to locals that are never read after the check (chain_j lines 1529-2416 read none of them):
    exact (only references go; an object that is still used elsewhere stays alive). CPython 3.12 frame write-back."""
    import ctypes
    fl = frame.f_locals
    done = []
    for nm in names:
        if nm in fl and fl[nm] is not None:
            fl[nm] = None; done.append(nm)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    fl.clear()                                                 # the frame's cached locals dict must not keep stale objects alive
    gc.collect()
    _trim()
    return done


def _trim():
    """BIG5_TRIM=1: glibc malloc_trim(0) after a free (returns freed heap pages to the OS; no value changes)"""
    if os.environ.get("BIG5_TRIM", "0") != "1" or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        t = time.perf_counter(); ctypes.CDLL("libc.so.6").malloc_trim(0)
        R.setdefault("trim_s", []).append(round(time.perf_counter() - t, 2))
    except Exception as e:
        R["trim_err"] = repr(e)


def record():
    mark("end")
    m = EARLY.get("mem")
    out = {"marks": R["marks"], "pieces": {k: (dict(v) if isinstance(v, dict) else v) for k, v in R["pieces"].items()},
           "act_mode": R.get("act_mode")}
    if "act" in R:
        out["act"] = R["act"]
        try:
            import pre5_load as P5L
            out["act_wait"] = {k: v for k, v in P5L.R.items() if k.startswith("act_")}
        except Exception:
            pass
    if m is not None:
        m.stop = True
        out["rss_peak_gb"] = round(m.peak / 1e9, 3); out["rss_peak_with_child_gb"] = round(m.peak_sum / 1e9, 3)
        out["samples"] = m.s
        seg = []; prev = ("start", 0.0, 0.0)                   # peak RSS and CPU threads per segment between marks
        for nm, t, cpu, rss, pk in R["marks"]:
            ss = [x for x in m.s if prev[1] <= x[0] <= t]
            seg.append({"seg": nm, "t0": prev[1], "t1": t, "wall": round(t - prev[1], 2),
                        "thr": round((cpu - prev[2]) / max(t - prev[1], 1e-9), 2),
                        "rss_max_gb": max([x[1] for x in ss] + [rss or 0.0]), "child_max_gb": max([x[2] for x in ss] + [0.0])})
            prev = (nm, t, cpu)
        out["segments"] = seg
    out["calls"] = R.get("calls", {})
    for k in ("holders_preload", "holders_check", "holders_stage", "mark_err", "freed_main", "freed_after_ovf", "early", "trim_s", "trim_err"):
        if k in R:
            out[k] = R[k]
    return out
