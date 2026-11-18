"""t_fp.py: exact first-pass pieces of tools/best10/chain_t.py.
  ofc0_install(NOFC, T)  tools/best5/n_ofc.OfCache._init with the per-layer numpy work on Python threads: the same calls with the same arrays (numpy releases the GIL in these ufuncs), so the same bits."""
import ctypes
import threading
import numpy as np


def _pmap(fn, items, T):
    """fn(i, item) for all items on at most T Python threads; results in order"""
    n = len(items); out = [None] * n; nx = [0]; mu = threading.Lock(); err = []

    def w():
        while True:
            with mu:
                i = nx[0]; nx[0] += 1
            if i >= n:
                return
            try:
                out[i] = fn(items[i])
            except BaseException as e:
                err.append(e); return
    th = [threading.Thread(target=w, daemon=True) for _ in range(min(T, n) - 1)]
    for t in th:
        t.start()
    w()
    for t in th:
        t.join()
    if err:
        raise err[0]
    return out


def ofc0_install(NOFC, T):
    def _init(self, la):
        self.key = (id(la), id(la.c), la.d.shape)
        self.la = la
        L = la.L; self.L = L; self.Y = la.d.shape[2]
        zs = list(range(1, L))
        self.dprev = [None] + _pmap(lambda z: np.array(la.d[z], copy=True), zs, T)
        self.loss = [None] + _pmap(lambda z: NOFC._loss(la.d[z], la.c[z]), zs, T)
        self.stats["full"] += 1
        ford = la.d[1].flags.f_contiguous and not la.d[1].flags.c_contiguous
        def _ct(a):
            return a.flags.f_contiguous if ford else a.flags.c_contiguous
        same = all(_ct(self.loss[z]) and _ct(la.d[z]) and _ct(self.dprev[z]) for z in range(1, L))
        self.ord = "F" if ford else "C"; self.shape2 = la.d.shape[1:]
        self.native = False
        if same:
            self.n = int(self.loss[1].size)
            self.dptr = (ctypes.c_void_p * L)(*([0] + [la.d[z].ctypes.data for z in range(1, L)]))
            self.pptr = (ctypes.c_void_p * L)(*([0] + [self.dprev[z].ctypes.data for z in range(1, L)]))
            self.lptr = (ctypes.c_void_p * L)(*([0] + [self.loss[z].ctypes.data for z in range(1, L)]))
            self.idx = [None] + [np.empty(self.n, np.int64) for _ in range(1, L)]
            self.iptr = (ctypes.c_void_p * L)(*([0] + [self.idx[z].ctypes.data for z in range(1, L)]))
            self.cnt = np.zeros(L, np.int64); self.sums = np.zeros(L)
            self.so.n_ofc_sum(L, self.n, self.lptr, self.sums.ctypes.data, self.T)
            ref = _pmap(lambda z: float(self.loss[z].sum()), zs, T)
            self.native = all(r == float(self.sums[z]) for r, z in zip(ref, zs))
        self.stats["native"] = int(self.native)
        self.stats["layout"] = [list(la.d.strides), list(la.c.strides), list(self.loss[1].strides), list(la.d.shape), bool(same)]
    NOFC.OfCache._init = _init


# ---------------------------------------------------------------- kernels
import os, sys, time
_K = [None]
_P = ctypes.c_void_p
_T = int(os.environ.get("MID5_THREADS", "8"))


def _k():
    if _K[0] is None:
        L = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libt_fpk." + ("dylib" if sys.platform == "darwin" else "so")))
        L.tf_same.argtypes = [ctypes.c_int64] + [_P] * 8 + [ctypes.c_int]
        L.tf_a6.argtypes = [ctypes.c_int64] + [_P] * 6 + [ctypes.c_int64, _P, ctypes.c_int]
        L.tf_base.argtypes = [ctypes.c_int64, _P, _P, _P, ctypes.c_int]
        L.tf_keep.argtypes = [ctypes.c_int64, _P, _P, _P, _P, _P, _P, ctypes.c_int]
        _K[0] = L
    return _K[0]


# ---------------------------------------------------------------- --t_same: nets whose new rows equal their current rows are not reported as changed
SAME = {"calls": 0, "nets": 0, "dropped": 0, "t": 0.0, "dups": 0}
_LASTP = {}


def same_install(PF):
    """tools/par7/p7_pf (or tools/late5/l5_pf): _pack_new remembers the current rows' addresses, _unpack drops the nets the round gave back unchanged.
    Their demand is already the same (removed and added again): nothing else in S changes. undo, S.new, S._dirty, the re-time's net list and the packed cache all get shorter."""
    _pn0, _un0 = PF._pack_new, PF._unpack

    def _pack_new(S, st, nroute):
        ptr, cnt = _pn0(S, st, nroute)
        _LASTP["ptr"], _LASTP["cnt"] = ptr, cnt
        return ptr, cnt

    def _unpack(S, st, nc, nsg, changed, off, segs, undo, new):
        if not nc:
            return _un0(S, st, nc, nsg, changed, off, segs, undo, new)
        t = time.perf_counter()
        barr, boff = PF._base(S, st)
        ch = changed[:nc]
        SAME["calls"] += 1; SAME["nets"] += int(nc)
        if len(np.unique(ch)) != nc:                                 # a net twice in one round: the second compare would be against the wrong rows
            SAME["dups"] += 1
            return _un0(S, st, nc, nsg, changed, off, segs, undo, new)
        keep = np.zeros(nc, np.uint8)
        _k().tf_same(nc, ch.ctypes.data, off.ctypes.data, segs.ctypes.data, _LASTP["ptr"].ctypes.data, _LASTP["cnt"].ctypes.data, boff.ctypes.data, barr.ctypes.data, keep.ctypes.data, _T)
        k = np.flatnonzero(keep)
        SAME["dropped"] += int(nc) - len(k)
        if len(k) == nc:
            SAME["t"] += time.perf_counter() - t
            return _un0(S, st, nc, nsg, changed, off, segs, undo, new)
        c = off[k + 1] - off[k]
        o2 = np.zeros(len(k) + 1, np.int64); o2[1:] = np.cumsum(c)
        tot = int(o2[-1])
        s2 = np.zeros((max(tot, 1), 6), np.int64)
        if tot:
            src = np.repeat(off[k], c) + (np.arange(tot) - np.repeat(np.cumsum(c) - c, c))
            s2[:tot] = segs[src]
        SAME["t"] += time.perf_counter() - t
        return _un0(S, st, len(k), tot, ch[k], o2, s2, undo, new)
    PF._pack_new, PF._unpack = _pack_new, _unpack


# ---------------------------------------------------------------- --t_pre: tables that depend on the net names only, before the route arrives
PIDX = {}


def pin_idx(names, _key, like=None):
    """{_key(n): i} (later duplicate wins); the dict made in the wait when the names are the .net file's, in its order"""
    c = PIDX.get("d")
    if c is not None and (c[0] is names or c[0] == list(names)):
        return c[1]
    return {_key(n): i for i, n in enumerate(names)}


def pre_tables(a, G, PRE, S, _key):
    """--t_pre 1: the names' tables of attach_h, the builder's and the promote's static tables, from the preload's .net parse. If the route's names differ (they do not, in the runs seen) the consumers rebuild: their compare is on the list."""
    t = time.perf_counter()
    B2 = G["B2"]
    pn = B2.parse_net_fast(os.path.join(B2.BENCH_ROOT, a.suite, a.design, a.design + ".net"))[0]
    names = list(pn)
    ks = [_key(nm) for nm in names]
    d = {k: i for i, k in enumerate(ks)}
    PIDX["d"] = (names, d)
    S._h_nm = (len(names), names, d, np.array([S.SD.net_index.get(k, -1) for k in ks], np.int64), ks)
    SD = S.SD
    cd = PRE["cd"]
    if not hasattr(cd, "_nf_index"):
        cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
    PIDX["t_s"] = time.perf_counter() - t
    return names


_B2_A = "    pin_idx = {_key(n): i for i, n in enumerate(pn_names)}      # later duplicate wins, as the dict comp\n"
_B2_B = "    idx = {_key(n): i for i, n in enumerate(net_names_rt)}\n"


def b2_install(B2):
    """tools/best5/n_b2.install + the two name dicts through pin_idx() above (text replaced in the same source, as n_b2 does)"""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best5"))
    import n_b2 as NB
    from pathlib import Path
    src = Path(B2.__file__).read_text()
    a = src.index("def build_fast2(")
    b = src.index("build_fast = build_fast2", a)
    f = src[a:b]
    assert f.count(_B2_A) == 1 and f.count(_B2_B) == 1
    f = f.replace(_B2_A, "    pin_idx = _T_FP.pin_idx(pn_names, _key)\n").replace(_B2_B, "    idx = _T_FP.pin_idx(net_names_rt, _key)\n")
    la = f.index(NB._LOOP_A); lb = f.index(NB._LOOP_B, la) + len(NB._LOOP_B)
    loop = "".join("    " + l + "\n" for l in f[la:lb].rstrip("\n").split("\n"))
    f = f[:la] + (NB._NEW % loop.rstrip("\n")) + f[lb:]
    B2._N_B2 = NB; B2._N_B2_CHECK = os.environ.get("N_B2_CHECK") == "1"; B2._T_FP = sys.modules[__name__]
    exec(compile(f, B2.__file__, "exec"), B2.__dict__)


# ---------------------------------------------------------------- --t_fp2: columns on threads
def a6_of(cols, G_):
    """== the six column lines of eval_cols5 (int32 columns; else None)"""
    if not all(np.asarray(c).dtype == np.int32 and np.asarray(c).flags.c_contiguous for c in cols):
        return None
    a6 = np.empty((len(cols[0]), 6), np.int64)
    _k().tf_a6(len(cols[0]), *(np.asarray(c).ctypes.data for c in cols), int(G_), a6.ctypes.data, _T)
    return a6


def base_of(n, off, a6):
    """== np.concatenate([np.repeat(np.arange(n), np.diff(off))[:, None], a6], 1)"""
    off = np.ascontiguousarray(off, np.int64); a6 = np.ascontiguousarray(a6, np.int64)
    b = np.empty((len(a6), 7), np.int64)
    _k().tf_base(n, off.ctypes.data, a6.ctypes.data, b.ctypes.data, _T)
    return b


# ---------------------------------------------------------------- --t_fp3: small exact copies
_ASM_OLD = """    if len(kn):
        c = cnt[kn]
        s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn].astype(np.int64), c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        for k in range(6):
            cols[k][d0 + ar] = src[k][s0 + ar]
"""
_ASM_NEW = """    if len(kn) and not _T_FP.keep_rows(kn, cnt, off, noff, src, cols):
        c = cnt[kn]
        s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn].astype(np.int64), c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        for k in range(6):
            cols[k][d0 + ar] = src[k][s0 + ar]
"""


def keep_rows(kn, cnt, off, noff, src, cols):
    """the kept nets' rows of the assembly on threads (== the fancy-index copies); False: not int32 / not contiguous, the caller copies"""
    s6 = [np.asarray(x) for x in src]
    if not all(x.dtype == np.int32 and x.flags.c_contiguous for x in s6):
        return False
    kn = np.ascontiguousarray(kn, np.int64); cnt = np.ascontiguousarray(cnt, np.int64); off = np.ascontiguousarray(off, np.int64); no = np.ascontiguousarray(noff, np.int64)
    sp = (ctypes.c_void_p * 6)(*[x.ctypes.data for x in s6]); dp = (ctypes.c_void_p * 6)(*[x.ctypes.data for x in cols])
    _k().tf_keep(len(kn), kn.ctypes.data, cnt.ctypes.data, off.ctypes.data, no.ctypes.data, ctypes.addressof(sp), ctypes.addressof(dp), _T)
    return True


def asm_install(h_asm):
    """h_asm.assemble with keep_rows (text replaced in the same source, as n_b2 does)"""
    from pathlib import Path
    src = Path(h_asm.__file__).read_text()
    assert src.count(_ASM_OLD) == 1
    a = src.index("def assemble(P, choice, rt, RE):")
    f = src[a:].replace(_ASM_OLD, _ASM_NEW)
    h_asm._T_FP = sys.modules[__name__]
    exec(compile(f, h_asm.__file__, "exec"), h_asm.__dict__)


def seg_install(CS):
    """tools/best5/n_seg's seg_from_route with the net-name lookup kept per names list (same array; the three candidates share one list)"""
    orig = CS.seg_from_route
    c = {}

    def seg_from_route(cd, rt, grid, _o=orig):
        names = rt.net_names
        if c.get("n") is not names:
            idx = getattr(cd, "_nf_index", None)
            if idx is None:
                cd._nf_index = idx = {nm: i for i, nm in enumerate(cd.nf_names)}
            nid = np.fromiter((idx.get(n, -1) for n in names), np.int64, len(names))
            c["n"] = names; c["d"] = np.where(nid >= 0, cd.nf_net[np.maximum(nid, 0)], -1)
        cols = [np.asarray(getattr(rt, k)) for k in ("x0", "y0", "z0", "x1", "y1", "z1")]
        if not all(x.dtype == np.int32 and x.flags.c_contiguous for x in cols):
            return _o(cd, rt, grid)
        de = np.ascontiguousarray(c["d"], np.int64); off = np.ascontiguousarray(rt.offsets, np.int64)
        L = sys.modules["n_seg"]._L; a = [x.ctypes.data for x in cols]; T = sys.modules["n_seg"].T
        n = L.n_seg_fill(len(de), off.ctypes.data, de.ctypes.data, *a, int(grid), None, T, 1)
        seg = np.empty((n, 7), np.int64)
        L.n_seg_fill(len(de), off.ctypes.data, de.ctypes.data, *a, int(grid), seg.ctypes.data, T, 0)
        return seg
    CS.seg_from_route = seg_from_route
