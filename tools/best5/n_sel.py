"""n_sel.py: tools/late5/l5.py select_c for the reshape stage: the same chosen nets; the overflow mask and the (net, cell) pairs on threads
(n_sel.cc), the base wire rows kept between calls instead of copied. The info field blk_share_ov (not read in reshape) is not computed. None -> the caller runs the original."""
import ctypes, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_L = None
_C = {}
T_SEL = {}
_EMPTY = np.zeros((1, 6), np.int64)


def lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(os.path.join(HERE, "libn_sel." + ("dylib" if sys.platform == "darwin" else "so")))
        P, I, D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
        L.n_ov_mask.argtypes = [I, I, I, P, P, P, P, P, ctypes.c_int]
        L.n_ov_cells.argtypes = [I, I, I, P, P, P, P, P, ctypes.c_int]
        L.n_ov_get.argtypes = [P, P]
        L.n_selc_pairs.argtypes = [I, I, I, P, P, P, I, P, P, P, P, I, P, P, I, D, I, P, ctypes.c_int]; L.n_selc_pairs.restype = ctypes.c_int
        L.l5_selc_cells.argtypes = [P]
        L.l5_selc_greedy.argtypes = [P, P, I, P, P]; L.l5_selc_greedy.restype = ctypes.c_int64
        _L = L
    return _L


def pretest_fast(S, threads=8):
    """the numbers of ovf2_lib.pretest_cost computed on the overflowed cells only (the sums run over fewer zeros, so the last bits of the logged sums can differ);
    no full cost map: the selection of tools/late5 evaluates its own cells. -> (pt, None)"""
    import ovf_lib as OL
    import l5 as L5
    la = S.la; L_ = lib(); L, X, Y = la.L, la.X, la.Y
    sd = np.array(la.d.strides, np.int64) // 8; sc = np.array(la.c.strides, np.int64) // 8
    cnt = np.zeros(2, np.int64)
    L_.n_ov_cells(L, X, Y, la.d.ctypes.data, sd.ctypes.data, la.c.ctypes.data, sc.ctypes.data, cnt.ctypes.data, threads)
    b = np.empty(int(cnt[0]), np.int64); o = np.empty(int(cnt[1]), np.int64)
    L_.n_ov_get(b.ctypes.data, o.ctypes.data)
    u = np.asarray(la.u, np.float64)
    def cost(q):
        z, r = np.divmod(q, X * Y); x, y = np.divmod(r, Y)
        return L5._cost_at(u[z], la.d[z, x, y], la.c[z, x, y])
    cb, co = float(cost(b).sum()), float(cost(o).sum())
    return {"ov_cells": int(len(b) + len(o)), "blocked_cells": int(len(b)), "blk_cost": cb, "open_cost": co,
            "bound_score": float(S.w4) * (cb + co), "bound_blk_score": float(S.w4) * cb,
            "blk_share_ov": cb / max(cb + co, 1e-30)}, None


def select_c(S, la, slack_r, npins_r, cap_frac=None, max_nets=0, threads=8, share=False):
    import ovf_lib as OL
    import l5 as L5
    if not (la.d.dtype == np.float64 and la.c.dtype == np.float64):
        return None
    t0 = time.perf_counter()
    L_ = lib()
    if cap_frac is None:
        cap_frac = OL.CAP_B
    L, X, Y = la.L, la.X, la.Y
    ov8 = _C.get("ov8")
    if ov8 is None or ov8.size != L * X * Y:
        ov8 = _C["ov8"] = np.empty(L * X * Y, np.uint8)
    sd = np.array(la.d.strides, np.int64) // 8; sc = np.array(la.c.strides, np.int64) // 8
    L_.n_ov_mask(L, X, Y, la.d.ctypes.data, sd.ctypes.data, la.c.ctypes.data, sc.ctypes.data, ov8.ctypes.data, threads)
    arr = S.base_arr
    if _C.get("arr") is not arr:
        _C["arr"] = arr
        _C["base"] = np.ascontiguousarray(arr[arr[:, 3] == arr[:, 6]], np.int64)
    base = _C["base"]
    nnet = len(slack_r)
    chm = np.zeros(max(nnet, int(arr[:, 0].max()) + 1 if len(arr) else 0), np.uint8)
    if S.new:
        chm[np.fromiter(S.new.keys(), np.int64)] = 1
    # the changed nets' rows: int64 [k, 6] blocks already packed for the promote kernel (S._pf_static["packed"]), else packed here once per list object
    pk = getattr(S, "_pf_static", {}).get("packed", {}) if hasattr(S, "_pf_static") else {}
    own = _C.setdefault("own", {})
    new = S.new
    ids_ = np.fromiter(new.keys(), np.int64, len(new))
    ps, cs = [], []
    for i, lst in new.items():
        c = pk.get(i)
        if c is None or c[0] is not lst:
            c = own.get(i)
            if c is None or c[0] is not lst:
                a_ = np.ascontiguousarray(np.array(lst, np.int64).reshape(-1, 6)) if lst else _EMPTY
                own[i] = c = (lst, a_, a_.ctypes.data, len(lst))
        ps.append(c[2]); cs.append(c[3])
    ptr = np.array(ps, np.int64); cnt = np.array(cs, np.int64)
    t1 = time.perf_counter()
    sl = np.ascontiguousarray(slack_r, np.float64); npn = np.ascontiguousarray(npins_r, np.int64)
    dirs = np.ascontiguousarray(np.asarray(la.dirs, np.int64))
    out = np.zeros(4, np.int64)
    rc = L_.n_selc_pairs(L, X, Y, ov8.ctypes.data, dirs.ctypes.data, base.ctypes.data, len(base), chm.ctypes.data, ids_.ctypes.data, ptr.ctypes.data, cnt.ctypes.data, len(ids_), sl.ctypes.data, npn.ctypes.data,
                         len(sl), float(OL.SLACK_GUARD), int(OL.MAX_PINS), out.ctypes.data, threads)
    if rc != 0:
        return None
    t2 = time.perf_counter()
    ncross, npool, npairs, ncells = (int(v) for v in out)
    if ncross == 0:
        return [], {"pool": 0}
    info = {"nets_crossing": ncross, "pool": npool, "ov_cells": int(ov8.sum())}
    if npool == 0:
        return [], info
    cells = np.zeros(ncells, np.int64); L_.l5_selc_cells(cells.ctypes.data)
    zc_, r_ = np.divmod(cells, X * Y); xc_, yc_ = np.divmod(r_, Y)
    ccost = np.ascontiguousarray(L5._cost_at(np.asarray(la.u, np.float64)[zc_], la.d[zc_, xc_, yc_], la.c[zc_, xc_, yc_]), np.float64)
    cap_k = max(1, int(round(cap_frac * npool)))
    if max_nets:
        cap_k = min(cap_k, max_nets)
    ch = np.zeros(npool, np.int64); inf2 = np.zeros(2)
    nch = int(L_.l5_selc_greedy(ccost.ctypes.data, sl.ctypes.data, cap_k, ch.ctypes.data, inf2.ctypes.data))
    t3 = time.perf_counter()
    if share:                                       # blk_share_ov for the overflow stage (the last bits of the sums can differ; it is read against 0.5)
        info["blk_share_ov"] = pretest_fast(S, threads)[0]["blk_share_ov"]
    info.update({"cells_total": ncells, "cells_uncovered": int(inf2[0]), "chosen": nch, "cost_uncovered": float(inf2[1]),
                 "n_sel_t": [round(t1 - t0, 4), round(t2 - t1, 4), round(t3 - t2, 4)]})
    return [int(v) for v in ch[:nch]], info
