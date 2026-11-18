#!/usr/bin/env python3
"""big8_lib.py: the --big8 1 pieces of tools/big8/chain_b8.py (copy of tools/best4/chain_m.py).
an internal note. Each piece can be switched off with env BIG8_<NAME>=0.

  cold   (big mode, cold start through tools/big8/cold_b8.py --io_bind_mem): the timer is bound to the in-memory StaDesign AFTER the
         state is built, so big5's early timer (started in ChainState.__init__, before any timer exists) cannot run there; big5's
         early start is switched off (BIG5_EARLY=0) and after_bind() starts the same early-timer thread (big5_lib._early_timer:
         route read, segments, RC, cold full analysis) and installs big5's dfull right after the bind. main's first read / seg / RC /
         analyze get those objects back through big5's identity-keyed wrappers (big5_lib._use_early, unchanged).
  load   (cold big mode) .def / .net C++ parses + Liberty parse on threads from t0.
  sta    (cold big mode) timer bind + early timer as soon as the StaDesign is complete (inside ChainState.__init__).
  mem    the first analysis' objects and the prior's graph tensors dropped at the start of the check; big5's identity keys dropped;
         dem_np and P after their last use inside the check; before the final write the kernel's Python-side arrays.
         BIG8_SDFREE=<attrs> (tools/big8/sdfree.txt, from a BIG8_TRACE=1 run): StaDesign attributes not read after the check.
  seg    (off by default, BIG8_SEG=1) csta.seg_from_route's column work on threads: same integers (test_seg.py), no time gained on group
         (early seg 33.2 s against 32.4 s: the time is the names dict and the GIL shared with the state build, not the columns).
  pf     promote rounds (crit, power, post-crit, overflow-ranked) on tools/par7/p7_pf (phase 2 on PF_THREADS threads in a level
         schedule; md5-equal to l5_pf on the six small cases in $TREE/runs/par7) in place of tools/late5/l5_pf.
  dp     single pass: both candidates' first-pass DPs at once (tools/par7/p7_dp.assign_pair, 4 threads each; same choices).
  pwb    (off by default, BIG8_PWB=1) big mode: the power round in checked batches (chain_m resets --pw_batch in big mode; here it
         is kept, batch size --big8_pw_batch, default --pw_batch). Measured on group v: all 12 batches kept, same estimate, slower.
"""
import os, sys, time, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = {"pieces": {}}


def pc(name, default="1"):
    return os.environ.get("BIG8_" + name.upper(), default) != "0"


def _arg(argv, k, d=None):
    for i, v in enumerate(argv):
        if v == k and i + 1 < len(argv):
            return argv[i + 1]
    return d


EARLY = {}


class _Fut:
    def __init__(self, name, fn):
        self.v, self.e = None, None
        def run():
            t = time.perf_counter()
            try:
                self.v = fn()
            except BaseException as e_:
                self.e = e_
            R["load"][name] = {"s": round(time.perf_counter() - t, 2), "end_s": round(time.perf_counter() - EARLY["t0"], 2)}
        self.th = threading.Thread(target=run, name="big8_" + name, daemon=True); self.th.start()

    def get(self):
        self.th.join()
        if self.e is not None:
            raise self.e
        return self.v


def early(argv, cold, t0=None):
    """at import of chain_b8 (before big5's early()): cold big mode -> big5's in-init early timer off (started after the bind);
    load: the .def / .net C++ parses and the Liberty parse on threads from t0 (the same functions; the results are handed to
    FastStaDesign / build2 by the wrappers of install_load)"""
    R["cold"] = bool(cold)
    EARLY["t0"] = t0 if t0 is not None else time.perf_counter()
    if not cold:
        return
    os.environ.setdefault("BIG5_TRIM", "1")                 # big5: malloc_trim after its frees
    if pc("cold"):
        os.environ["BIG5_EARLY"] = "0"
        R["pieces"]["cold_early_after_bind"] = True
    if pc("load"):
        R["load"] = {}
        sys.path.append(str(ROOT / "tools" / "comp2" / "loader"))
        d, s = _arg(argv, "--design"), _arg(argv, "--suite", "visible")
        base = Path(os.environ.get("ONEPASS_DATA_ROOT") or ROOT) / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks" / s / d
        EARLY["def"] = os.path.realpath(str(base / (d + ".def"))); EARLY["net"] = os.path.realpath(str(base / (d + ".net")))
        nt = int(os.environ.get("FASTLOAD_THREADS", "8"))

        import fast_load as FLD, fastload_lib as FL             # light imports; the originals captured before install_load wraps them
        import fast_sta_v3 as FS
        oD, og, ol = FL.DefData, FLD.get_net, FS.StaLib.load

        def parse():
            df = oD(EARLY["def"], nt)
            og(EARLY["net"])
            return df
        EARLY["P"] = _Fut("P", parse)

        def lib():
            return ol(os.path.join(FS.KIT, "NanGate45/lib"), cache="/dev/null/big8_no_sta_lib.pkl")
        EARLY["LIB"] = _Fut("LIB", lib)


def install_load(FS):
    """__main__ wrapper of load_all, after fast_load.install(): DefData / get_net / StaLib.load return the early threads' objects"""
    if "P" not in EARLY:
        return
    import fast_load as FLD, fastload_lib as FL
    oD, og, ol = FL.DefData, FLD.get_net, FS.StaLib.load
    libdir = os.path.realpath(os.path.join(FS.KIT, "NanGate45/lib"))

    def DefData(path, nt=FL.NT, _o=oD):
        if os.path.realpath(path) == EARLY["def"] and "P" in EARLY:
            return EARLY.pop("P").get()
        return _o(path, nt)

    def get_net(path, _o=og):
        f = EARLY.get("P")
        if f is not None and f.th is not threading.current_thread():
            f.th.join()
        if os.path.realpath(path) == EARLY["net"]:          # one parse: the early thread's, whatever spelling of the path
            for p_, v_ in list(FLD._NET.items()):
                if os.path.realpath(p_) == EARLY["net"]:
                    FLD._NET.setdefault(os.path.abspath(path), v_)
                    break
        return _o(path)

    def load(lib_dir=os.path.join(FS.KIT, "NanGate45/lib"), cache=None, _o=ol):
        if os.path.realpath(lib_dir) == libdir and "LIB" in EARLY:
            return EARLY["LIB"].get()
        return _o(lib_dir, cache=cache) if cache else _o(lib_dir)
    FL.DefData, FLD.get_net = DefData, get_net
    FS.StaLib.load = staticmethod(load)
    R["pieces"]["load"] = True


def wrap_sta(FS, G, PRE, H, BEST, bigm):
    """cold big mode: the timer kernel is bound and the early timer started the moment the StaDesign is complete (inside
    ChainState.__init__), not after the rest of the state; load_all then finds R['K'] (no second bind)"""
    if not (bigm and R.get("cold") and pc("cold") and pc("sta")):
        return
    Cls = FS.StaDesign

    class B8Sta(Cls):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk)
            if "K" in R:
                return
            t = time.perf_counter()
            _sp = list(sys.path)
            sys.path.append(str(ROOT / "tools" / "engine"))
            try:
                import csta_inc as CI
            finally:
                sys.path[:] = _sp
            G["CI"] = CI
            hl = str(BEST / ("libi_kern." + ("dylib" if sys.platform == "darwin" else "so")))
            K = CI.IncKernel(hl); K.bind_design(self); R["inc_info"] = K.bind_rc_design(self)
            K.L.w3_inc_backward.argtypes = [__import__("ctypes").c_void_p]
            R["K"] = K
            PRE["cd"], PRE["K"] = self, K
            R["bind_s"] = round(time.perf_counter() - t, 2)
            after_bind(G, PRE, bigm)
    FS.StaDesign = B8Sta
    R["pieces"]["sta"] = True


def install_seg(CS):
    """seg: csta.seg_from_route with the column work (integer division by the grid, np.repeat) on threads; numpy releases the GIL in
    these loops; same integers. Only for a big route (below that the original)."""
    import numpy as np
    if not pc("seg", "0") or getattr(CS, "_b8_seg", False):
        return
    from concurrent.futures import ThreadPoolExecutor
    orig = CS.seg_from_route
    nt = int(os.environ.get("OMP_NUM_THREADS", "8"))

    def seg_from_route(cd, rt, grid, _o=orig):
        n = len(rt.x0)
        if n < 2000000:
            return _o(cd, rt, grid)
        if not hasattr(cd, "_nf_index"):
            cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
        idx = cd._nf_index
        nid = np.fromiter((idx.get(x, -1) for x in rt.net_names), np.int64, len(rt.net_names))
        de = np.where(nid >= 0, cd.nf_net[np.maximum(nid, 0)], -1)
        cnt = np.diff(np.asarray(rt.offsets, np.int64))
        seg = np.empty((n, 7), np.int64)
        off = np.asarray(rt.offsets, np.int64)
        cols = ((rt.x0, 1, True), (rt.y0, 2, True), (rt.z0, 3, False), (rt.x1, 4, True), (rt.y1, 5, True), (rt.z1, 6, False))
        chunks = [(i * n // nt, (i + 1) * n // nt) for i in range(nt)]

        def one(c):
            s, e = c
            ns = np.searchsorted(off, s, side="right") - 1; ne = np.searchsorted(off, e, side="left")
            r = np.repeat(de[ns:ne], np.minimum(off[ns + 1:ne + 1], e) - np.maximum(off[ns:ne], s))
            seg[s:e, 0] = r
            for arr, k, dv in cols:
                v = np.asarray(arr[s:e], np.int64)
                seg[s:e, k] = v // grid if dv else v
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(one, chunks))
        return seg[seg[:, 0] >= 0]
    CS.seg_from_route = seg_from_route
    CS._b8_seg = True
    R["pieces"]["seg"] = True


def after_bind(G, PRE, bigm):
    """load_all, right after PRE['cd'], PRE['K'] are set (cold: after the state + bind)"""
    if bigm and "CS" in G:
        install_seg(G["CS"])
    if not (bigm and R.get("cold") and pc("cold")):
        return
    sys.path.append(str(ROOT / "tools" / "big5"))
    import big5_lib as B5L
    t = time.perf_counter()
    if pc("dfull") and "dfull" not in B5L.R["pieces"] and hasattr(PRE.get("K"), "set_rc"):
        B5L.install_dfull(PRE["K"])
    if "tm" not in B5L.EARLY:
        B5L.EARLY["tm"] = threading.Thread(target=B5L._early_timer, args=(G, PRE), name="big8_early", daemon=True)
        B5L.EARLY["tm"].start()
    R["after_bind_s"] = round(time.perf_counter() - t, 3)
    R["after_bind_at_s"] = round(time.perf_counter() - B5L.R["t0"], 2) if B5L.R.get("t0") else None


def pf_module():
    """par7's promote module (same API as l5_pf) with phase 2 on threads, or None"""
    if not pc("pf"):
        return None
    sys.path.append(str(ROOT / "tools" / "par7"))
    import p7_pf as P7PF
    P7PF.PAR[0] = 1
    R["pieces"]["pf"] = True
    return P7PF


def dp_pair(P, a, V1):
    """both first-pass DPs at once (par7 (b)); returns {"noprior": ch, "prior": ch} or None"""
    if not pc("dp"):
        return None
    sys.path.append(str(ROOT / "tools" / "par7"))
    import p7_dp as P7DP
    t = time.perf_counter()
    ch, info = P7DP.assign_pair(P, a.dw, a.pw, V1.DEFAULT_VIA_COST, (0.0, a.gate), int(os.environ.get("BIG8_DP_THREADS", "4")))
    R["pieces"]["dp"] = {"s": round(time.perf_counter() - t, 3), "info": info}
    return {"noprior": ch[0], "prior": ch[1]}


def _closure_dicts(fn, key, seen=None, depth=0):
    seen = seen if seen is not None else set()
    if fn is None or id(fn) in seen or depth > 6:
        return []
    seen.add(id(fn)); out = []
    for c in (getattr(fn, "__closure__", None) or ()):
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if isinstance(v, dict) and key in v:
            out.append(v)
        elif callable(v):
            out += _closure_dicts(v, key, seen, depth + 1)
    return out


def drop_early_refs(G, K, CS):
    """big5's early-timer wrappers keep rt_ref / seg_ref / rcd_ref (the router route, its timer segments and RC: the identity keys)
    for the whole run; they are only compared with main's first calls, which are over at the gate -> drop them (references only)"""
    n = 0
    for fn in (G.get("read_route"), getattr(CS, "seg_from_route", None), getattr(K, "build_rc", None), getattr(K, "analyze", None)):
        for d in _closure_dicts(fn, "rt_ref"):
            for k in ("rt_ref", "seg_ref", "rcd_ref", "rt", "seg", "rcd", "o"):
                if d.pop(k, None) is not None:
                    n += 1
    R["pieces"]["early_refs_dropped"] = n


def trim(tag):
    import gc
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    R.setdefault("trim", []).append(tag)


def _nbytes(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        return int(v.nbytes)
    if isinstance(v, (list, tuple)):
        n = sys.getsizeof(v)
        if len(v) and isinstance(v[0], str):
            n += len(v) * (sys.getsizeof(v[0]) + 8)
        return int(n)
    if isinstance(v, dict):
        n = sys.getsizeof(v)
        if len(v):
            k0 = next(iter(v))
            if isinstance(k0, str):
                n += len(v) * sys.getsizeof(k0)
            if isinstance(v[k0], __import__("numpy").ndarray):
                n += sum(int(x.nbytes) for x in v.values() if hasattr(x, "nbytes"))
        return int(n)
    return 0


def after_check(S):
    """right after the single pass. BIG8_TRACE=1: every attribute of the StaDesign read from here on is recorded (diagnostic, one
    run: which attributes could be dropped). Otherwise BIG8_SDFREE: the attributes listed are dropped (a later read raises)."""
    SD = S.SD
    if os.environ.get("BIG8_TRACE") == "1":
        seen = {}
        Cls = SD.__class__

        class Traced(Cls):
            def __getattribute__(self, k):
                if k not in seen:
                    seen[k] = time.perf_counter() - EARLY["t0"]
                return object.__getattribute__(self, k)
        SD.__class__ = Traced
        R["sd_read_after"] = seen
        R["sd_sizes_gb"] = {k: round(_nbytes(v) / 1e9, 3) for k, v in vars(SD).items()}
        return
    names = [x for x in os.environ.get("BIG8_SDFREE", "").split(",") if x]
    if not names and R.get("cold") and pc("mem") and pc("sdfree"):          # cold big mode: the list of tools/big8/sdfree.txt
        names = [x for x in (HERE / "sdfree.txt").read_text().strip().split(",") if x]
    if names:
        got = []
        for k in names:
            if k in vars(SD):
                delattr(SD, k); got.append(k)
        trim("sdfree"); R["sdfree"] = got


def before_write(K):
    """big mode: no timer call follows the final estimate: the kernel's Python-side copies go before the write's text buffers"""
    for nm in ("keep", "rc", "V"):
        v = getattr(K, nm, None)
        if isinstance(v, dict):
            v.clear()
    K.snap = None; K.pending_snap = None
    R.pop("K", None)
    trim("before_write")


def record():
    out = {k: v for k, v in R.items() if k not in ("K", "inc_info")}
    try:
        import p7_pf as P7PF
        out["pf_times"] = [[round(float(x), 3) for x in t[:7]] for t in P7PF.TIMES]
        out["p7"] = P7PF.P7
    except Exception:
        pass
    try:
        import p7_pf as P7PF
        out["pf_levels_calls"] = len(P7PF.P7)
    except Exception:
        pass
    return out
