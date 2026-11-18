#!/usr/bin/env python3
"""fc_lib.py (fixcut 2026-11-12, NEW): the cuts of the fixed work (load + first pass + write + exit) used by
tools/fixcut/cold_fc.py. Every cut is OFF unless named in env FIXCUT (comma list, or "all"); with FIXCUT unset the driver
executes the frozen chain exactly as tools/day1/cold_d1.py / tools/final/cold_y.py do (no text inserted, no module patched).

  dem    single pass (gate 0): the demand map of the DP route = the router route's map (kept by the escape thread,
         PRE["h_Dr"], C++ h_dem.build) - the old rows + the new rows of the nets the DP rebuilt (h_dem.derive), set with
         attach_h, instead of zeroing the map and rebuilding it from every row (fastmode build_demand_fast). Every demand
         contribution is a small integer, so the float64 map does not depend on the order of the additions: the same map
         bit for bit (tools/best/h_dem.py; item 3a of chain_h, used by the gate path of the same chains).
  rcsub  single pass: the RC rows of the DP nets only (IncKernel.build_rc_subset) instead of a full build_rc of every
         net; the nets whose rows changed bit for bit (net rows, or the Elmore row of one of its pins) are found among the
         DP nets (every other net keeps its segments, hence its rows) and only they are updated, then the required-time
         pass and the result copies: the same net set and the same kernel calls as big5's "dfull" diff analysis, which
         finds them by comparing a full RC build with the kernel's. Used only when dfull is installed and valid, the
         kernel reports no multi-writer pins and the DP nets are at most half of the nets; otherwise the original call.
  b2par  small mode (--pre5 1): the B2 pre-parse (C++ .cap tokenizer/strtod, the shared .net parse in B2 form, TimingLA)
         on its own thread beside the state, the kernel bind and FLUTE; its import and sys.path change stay where they
         were. Joined by the escape-pin thread (the only reader during the load) and before the load ends.
  names  big mode (--fp2 1): load8_lib_f2.names_tables (the attach's name tables) takes the normalised name blob that
         table_for_names already made for the same name-list object (build_fast2's name index) instead of joining and
         encoding the 3.4M names again (pure function of the same object: same bytes).
  ecopy  big mode: the early timer's first full analysis (forward + backward) returns views of the kernel state instead
         of copies (about 3.2 GB on MemPool-Group). Its readers (K.crit, to_crit_data, the DTLA skip rule, timer_rec) all
         run before the next kernel change, and big8 drops the references before the single pass.
Env FIXCUT_CHECK=1: after a cut, the reference computation is also done and compared bit for bit (printed, and in
the cold json under "fixcut"): dem -> build_demand_fast on a zero map; rcsub -> full build_rc vs the kernel's RC rows.
"""
import os, sys, time, threading
import importlib.abc, importlib.machinery, importlib.util
import numpy as np

CUTS_ALL = ("dem", "rcsub", "b2par", "ecopy", "names")
_env = os.environ.get("FIXCUT", "").replace(" ", "")
CUTS = set(CUTS_ALL) if _env == "all" else {c for c in _env.split(",") if c}
_bad = CUTS - set(CUTS_ALL)
if _bad:
    raise SystemExit("fixcut: unknown cut(s) %s (known: %s)" % (sorted(_bad), ",".join(CUTS_ALL)))
CHECK = os.environ.get("FIXCUT_CHECK", "0") == "1"
ST = {"cuts": sorted(CUTS), "check": CHECK, "events": {}}


def on(name):
    return name in CUTS


def _ev(k, v):
    ST["events"][k] = v


# ------------------------------------------------------------------------------------------- single pass (dem, rcsub)
def sp_args(P, PRE, H, K, nn_route):
    """(dem, inc_nets) for the single pass's eval_cols; (None, None) = the original call"""
    t = time.perf_counter()
    dem = inc = None
    if not (on("dem") or on("rcsub")):
        return None, None
    rn = np.asarray(P["run_net"], np.int64); nri = np.asarray(P["net_route_idx"], np.int64)
    idx = np.unique(nri[np.unique(rn)]) if len(rn) else np.zeros(0, np.int64)     # == assemble's `new` keys (all DP nets)
    ST["dp_nets"] = int(len(idx))
    if on("dem"):
        if H.get("dem") and "h_Dr" in PRE and "h_arr6r" in PRE and "h_offr" in PRE:
            dem = (PRE["h_Dr"], PRE["h_arr6r"], PRE["h_offr"], idx)
            _ev("dem", "derive")
        else:
            _ev("dem", "off: no router map (escape on or --h_dem 0)")
    if on("rcsub"):
        why = None
        if not H.get("inc"):
            why = "no incremental kernel (--h_inc 0)"
        elif not hasattr(K, "build_rc_subset") or not getattr(K, "inc_ok", False):
            why = "kernel without build_rc_subset / inc refused"
        elif not getattr(K, "_bs_state", False):
            why = "dfull not installed or no full analysis yet"
        elif getattr(K, "_bs_open", False):
            why = "open transaction"
        elif len(idx) > 0.5 * len(K.D.net_names):
            why = "many nets (dfull would run the full analysis)"
        elif (getattr(K, "info", None) or {}).get("multi_writer", 1) != 0:
            why = "multi-writer pins"
        if why is None:
            inc = idx
            _rcsub_wrap(K)
            _ev("rcsub", "subset")
        else:
            _ev("rcsub", "off: " + why)
    ST["sp_args_s"] = round(time.perf_counter() - t, 4)
    return dem, inc


def _pins_of(K, nets):
    """mapped pins of the given nets (the kernel's access-point lists, as dfull's pin -> net map) and their net"""
    ap_off = np.asarray(K.keep["ap_off"], np.int64); ap_idx = np.asarray(K.keep["ap_idx"], np.int64)
    ap_pid = np.asarray(K.keep["ap_pid"], np.int64)
    c = ap_off[nets + 1] - ap_off[nets]
    tot = int(c.sum())
    sel = np.repeat(ap_off[nets], c) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c))
    pins = ap_pid[ap_idx[sel]]; pin_net = np.repeat(nets, c)
    ok = pins >= 0
    return pins[ok], pin_net[ok]


def _rows(K, nets, pins):
    R = K.rc
    return [np.array(R[k][nets], copy=True) for k in ("c1", "c2", "rpi", "ctot", "par")] + [np.array(R["elm"][pins], copy=True)]


def _u64(x):
    x = np.ascontiguousarray(x)
    return x.view(np.uint64) if x.dtype.itemsize == 8 else x


def _rcsub_wrap(K):
    """one-shot wrappers for eval_cols' incremental path: build_rc_subset records which DP nets changed bit for bit,
    update gets those nets (sorted, unique: dfull's np.flatnonzero order); removed again by sp_after"""
    st = {}
    o_brs, o_upd = K.build_rc_subset, K.update
    full_old = {k: np.array(v, copy=True) for k, v in K.rc.items()} if CHECK else None

    def brs(seg, nets, *aa, **kk):
        n_ = np.ascontiguousarray(nets, np.int64)
        pins, pin_net = _pins_of(K, n_)
        old = _rows(K, n_, pins)
        r = o_brs(seg, nets, *aa, **kk)
        new = _rows(K, n_, pins)
        ch = old[4] != new[4]
        for a_, b_ in zip(old[:4], new[:4]):
            ch |= (_u64(a_) != _u64(b_)).reshape(len(n_), -1).any(1)
        pe = (_u64(old[5]) != _u64(new[5])).reshape(len(pins), -1).any(1)
        st["nets"] = np.union1d(n_[ch], pin_net[pe]).astype(np.int64)
        if CHECK:
            ST["_rcsub_last"] = st["nets"].copy()
        ST["rcsub_dp_nets"] = int(len(n_)); ST["rcsub_changed_nets"] = int(len(st["nets"]))
        return r

    def upd(nets, *aa, **kk):
        n_ = st.pop("nets", None)
        return o_upd(n_ if n_ is not None else nets, *aa, **kk)
    K.build_rc_subset, K.update = brs, upd
    ST["_rcsub"] = (K, o_brs, o_upd, st, full_old)


def _dfull_set(K, old, new):
    """CHECK: big5 dfull's changed-net set from two full RC states (its rules: net rows, pin Elmore rows via the pin map)"""
    D = K.D; nn = len(D.net_names); P = int(D.P)
    ap_off = np.asarray(K.keep["ap_off"], np.int64); ap_idx = np.asarray(K.keep["ap_idx"], np.int64); ap_pid = np.asarray(K.keep["ap_pid"], np.int64)
    pins_t = ap_pid[ap_idx]; net_t = np.repeat(np.arange(nn, dtype=np.int64), np.diff(ap_off)); okp = pins_t >= 0
    pnet = np.full(P, -1, np.int64); pnet[pins_t[okp]] = net_t[okp]
    multi = np.bincount(pins_t[okp], minlength=P) > 1
    ch = np.asarray(new["par"]).reshape(-1) != np.asarray(old["par"]).reshape(-1)
    for k in ("c1", "c2", "rpi", "ctot"):
        ch |= (_u64(np.asarray(new[k]).reshape(old[k].shape)) != _u64(old[k])).any(1)
    pe = np.flatnonzero((_u64(np.asarray(new["elm"]).reshape(old["elm"].shape)) != _u64(old["elm"])).any(1))
    bad = bool(len(pe) and ((pnet[pe] < 0).any() or multi[pe].any()))
    if len(pe):
        ch[pnet[pe][pnet[pe] >= 0]] = True
    return np.flatnonzero(ch).astype(np.int64), bad


def sp_after(dem, inc, PRE, S, FM, K, seg_fn=None):
    """after eval_cols: (o2, rcd2) replacements for rcsub (commit + required times + copies, == dfull's diff path), the
    router map released once used (dem), and the FIXCUT_CHECK comparisons"""
    out = None
    t = time.perf_counter()
    if inc is not None:
        import ctypes
        _k, o_brs, o_upd, st_, full_old = ST.pop("_rcsub")
        K.build_rc_subset, K.update = o_brs, o_upd           # wrappers off before anything else runs
        K.commit()                                           # dfull-wrapped: o_commit + open flag off; snap = pending (S.new)
        stb = np.zeros(2); K.L.w3_inc_backward(stb.ctypes.data_as(ctypes.c_void_p))
        if K.S is not None:
            K.snap = dict(K.S.new); K.pending_snap = None
        o = K._out(np.zeros(4), True)                         # == dfull: copies, stats zeros
        rc = {k: np.array(v, copy=True) for k, v in K.rc.items()}
        rc["stats"] = np.zeros(4)
        out = (o, rc)
        st = sys.modules.get("big5_lib")
        if st is not None:                                    # big5's dfull counters: one diff call, as the original
            try:
                d = st.R["pieces"].get("dfull")
                if d is not None:
                    d["calls"] += 1; d["diff"] += 1; d.setdefault("fc_subset_nets", []).append(int(len(inc)))
            except Exception:
                pass
        ST["rcsub_commit_s"] = round(time.perf_counter() - t, 4)
        if CHECK and full_old is not None and seg_fn is not None:
            ref = K.build_rc(seg_fn())
            dset, dbad = _dfull_set(K, full_old, ref)
            mine = np.asarray(ST.pop("_rcsub_last", np.zeros(0, np.int64)))
            same = bool(np.array_equal(mine, dset))
            ST["check_rcsub_set"] = {"dfull_nets": int(len(dset)), "dfull_fallback": dbad, "subset_changed": int(len(mine)), "same_set": same}
            print("[fixcut check] rcsub changed nets %d vs dfull set %d: same %s (dfull fallback %s)" % (len(mine), len(dset), same, dbad), flush=True)
            bv = lambda x: np.ascontiguousarray(x).view(np.uint64 if np.asarray(x).dtype.itemsize == 8 else np.uint8).ravel()
            mis = {k: int((bv(rc[k]) != bv(np.asarray(ref[k]).reshape(rc[k].shape))).sum()) for k in ("c1", "c2", "rpi", "ctot", "par", "elm")}
            ST["check_rcsub"] = mis
            print("[fixcut check] rcsub RC rows vs full build_rc: mismatches %s" % mis, flush=True)
    if dem is not None:
        if CHECK:
            import h_dem as HD
            z = HD._Z(S.la); FM.build_demand_fast(z, S.base_arr)
            eq = bool(np.array_equal(z.d, S.la.d))
            ST["check_dem"] = {"equal": eq, "nets": int(len(dem[3]))}
            print("[fixcut check] dem derived map == full build: %s (%d nets)" % (eq, len(dem[3])), flush=True)
        PRE.pop("h_Dr", None); PRE.pop("h_arr6r", None); PRE.pop("h_offr", None)
    return out


def early_views(K, rcd):
    """ecopy: the early timer's first analysis without the result copies (same kernel call and state as dfull's first
    K.analyze: full(backward=True) through dfull's wrapper)"""
    t = time.perf_counter()
    o = K.full(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"], backward=True, copy=False)
    st = sys.modules.get("big5_lib")
    if st is not None:
        try:
            d = st.R["pieces"].get("dfull")
            if d is not None:
                d["calls"] += 1; d["full"] += 1; d["fallback"]["first"] = d["fallback"].get("first", 0) + 1
                d["s_full"] += time.perf_counter() - t
        except Exception:
            pass
    ST["ecopy_full_s"] = round(time.perf_counter() - t, 4)
    _ev("ecopy", "views")
    return o


# ------------------------------------------------------------------------------------------- source patches of modules
def _rep(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise SystemExit("fixcut: %s: patch site found %d times (expected 1): %r" % (what, n, old[:80]))
    return s.replace(old, new)


PATCHES = {}          # module name -> function(source) -> patched source


def _p_pre5(s):
    s = _rep(s, '            def b2():\n'
                '                sys.path.insert(0, str(ROOT / "tools" / "onepass"))\n'
                '                import build2_fast as B2\n',
                '            def b2(_fc_b2=None):\n'
                '                if _fc_b2 is None:\n'
                '                    sys.path.insert(0, str(ROOT / "tools" / "onepass"))\n'
                '                    import build2_fast as B2\n'
                '                else:                                          # fixcut b2par: imported by the load thread\n'
                '                    B2 = _fc_b2\n', "pre5_load b2 head")
    s = _rep(s, '            _timed("b2_preparse", b2)\n',
                '            if FC_LIB.on("b2par"):                             # fixcut b2par: same import, same sys.path order;\n'
                '                sys.path.insert(0, str(ROOT / "tools" / "onepass"))   # the parse on a thread\n'
                '                import build2_fast as _FC_B2\n'
                '                EARLY["FC_B2"] = _Fut("FC_B2", _timed, "b2_preparse", b2, _FC_B2)\n'
                '            else:\n'
                '                _timed("b2_preparse", b2)\n', "pre5_load b2 call")
    s = _rep(s, '            if not (a.escape != "off" or a.ovf != "off"):\n'
                '                return\n'
                '            t3 = time.perf_counter()\n',
                '            if not (a.escape != "off" or a.ovf != "off"):\n'
                '                return\n'
                '            if "FC_B2" in EARLY:                               # fixcut b2par: h_net / XY come from it\n'
                '                EARLY["FC_B2"].get()\n'
                '            t3 = time.perf_counter()\n', "pre5_load pins wait")
    s = _rep(s, '        PRE["t_state"] = time.perf_counter() - t\n        fp.get()\n',
                '        PRE["t_state"] = time.perf_counter() - t\n        fp.get()\n'
                '        if "FC_B2" in EARLY:                                   # fixcut b2par: done before the load ends\n'
                '            EARLY.pop("FC_B2").get()\n', "pre5_load end join")
    return s


def _p_l8f2(s):
    s = _rep(s, '    n = len(names)\n    b, off = _normalised_blob(names)\n    tid = SD.net_names.table().lookup_blob(b, off, strip=False)\n',
                '    n = len(names)\n'
                '    _fce = _TABLE_CACHE.get(id(names)) if FC_LIB.on("names") else None    # fixcut names: table_for_names\' blob\n'
                '    b, off = (_fce[2], _fce[3]) if (_fce is not None and _fce[0] is names) else _normalised_blob(names)\n'
                '    tid = SD.net_names.table().lookup_blob(b, off, strip=False)\n', "load8_lib_f2 names_tables")
    return s


def _p_big5(s):
    s = _rep(s, '        o = K.analyze(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"]); t4 = time.perf_counter()\n',
                '        o = (FC_LIB.early_views(K, rcd) if FC_LIB.on("ecopy") else\n'
                '             K.analyze(rcd["c1"], rcd["c2"], rcd["rpi"], rcd["ctot"], rcd["elm"], rcd["par"])); t4 = time.perf_counter()\n',
             "big5_lib early timer analyze")
    return s


if on("b2par"):
    PATCHES["pre5_load"] = _p_pre5
if on("ecopy"):
    PATCHES["big5_lib"] = _p_big5
if on("names"):
    PATCHES["load8_lib_f2"] = _p_l8f2


def patch_src(name, src):
    """the same patch for a chain that reads and execs the module's source itself (chain_y / chain_d1 --pre5 1 --timer5 1)"""
    fn = PATCHES.get(name)
    if fn is None:
        return src
    ST.setdefault("patched", []).append(name + "(chain exec)")
    return fn(src)


class _Loader(importlib.abc.SourceLoader):
    def __init__(self, name, path, fn):
        self.name, self.path, self.fn = name, path, fn

    def get_filename(self, fullname):
        return self.path

    def get_data(self, path):
        with open(path, "rb") as fh:
            src = fh.read().decode("utf-8")
        if path == self.path:
            src = self.fn(src)
            ST.setdefault("patched", []).append(self.name)
        return src.encode("utf-8")

    def path_stats(self, path):                              # no bytecode cache for the patched source
        raise OSError

    def exec_module(self, module):
        module.__dict__["FC_LIB"] = sys.modules[__name__]
        super().exec_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    """the patched modules: found exactly where the normal path search finds them (same file), source patched in memory"""
    def find_spec(self, fullname, path=None, target=None):
        fn = PATCHES.get(fullname)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        return importlib.util.spec_from_file_location(fullname, spec.origin, loader=_Loader(fullname, spec.origin, fn))


def install_hooks():
    if PATCHES and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        ST["hooks"] = sorted(PATCHES)
