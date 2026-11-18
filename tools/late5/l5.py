"""l5.py: Python front end of tools/late5/l5_kern.cc for tools/late5/chain_late.py --late5 1.

revert_fast(S, undo): == fastmode3.State3.revert (fastmode.State.revert + dirty marks) with the demand change made by one C++
call (l5_swap) instead of to_wires_vias + apply per net. S.new is updated by the same rule as fastmode.State.revert:
`old is S.segs[i]` -> pop, else S.new[i] = old; with the chain's _Segs class S.segs[i] builds a new list on every call, so the
test is always False and every undone net gets S.new[i] = old (the same objects as the original). Rows outside the map ->
the original revert (the C++ adder would refuse where numpy wraps).
"""
import ctypes, os, sys
from itertools import chain
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_L = None
STATS = {"revert_fast": 0, "revert_py": 0, "nets": 0}


def lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(os.path.join(HERE, "libl5_kern." + ("dylib" if sys.platform == "darwin" else "so")))
        P, I = ctypes.c_void_p, ctypes.c_int64
        L.l5_swap.argtypes = [P, I, I, I, I, I, I, P, P, I, P, I]
        L.l5_swap.restype = ctypes.c_int
        _L = L
    return _L


def _rows(ids, lists):
    """[m,7] int64 (route net, x1..z2) of the segment lists (tuples of 6 ints), net-grouped in the order of ids"""
    ln = [len(x) for x in lists]
    m = sum(ln)
    ar = np.empty((m, 7), np.int64)
    if m:
        ar[:, 1:] = np.fromiter(chain.from_iterable(chain.from_iterable(lists)), np.int64, 6 * m).reshape(m, 6)
        ar[:, 0] = np.repeat(np.asarray(ids, np.int64), ln)
    return ar


def make_revert(S, SegsCls, orig):
    """bound replacement for S.revert; orig = the original bound method (fallback)"""
    la = S.la
    dirs = np.ascontiguousarray(np.asarray(la.dirs, np.int64))
    segs_new_each_call = isinstance(S.segs, SegsCls)
    L_ = lib()

    def revert(undo):
        if not undo:
            return
        if not segs_new_each_call or S.segs is None or not isinstance(S.segs, SegsCls) or la is not S.la:
            STATS["revert_py"] += 1
            return orig(undo)
        ids = list(undo)
        new = S.new
        arr, off = S.segs.arr, S.segs.off
        cur = []
        for i in ids:
            sg = new.get(i)
            cur.append(sg if sg is not None else None)
        # current rows: S.new lists, or the base rows (same values as S.segs[i])
        parts = []
        for i, sg in zip(ids, cur):
            if sg is None:
                b = arr[off[i]:off[i + 1]]
                p = np.empty((len(b), 7), np.int64); p[:, 0] = i; p[:, 1:] = b
                parts.append(p)
            else:
                parts.append(_rows([i], [sg]))
        rc_ = np.ascontiguousarray(np.concatenate(parts, 0)) if parts else np.zeros((0, 7), np.int64)
        olds = [undo[i] for i in ids]
        ro_ = _rows(ids, olds)
        d = la.d
        st = [x // 8 for x in d.strides]
        rc = L_.l5_swap(d.ctypes.data, la.L, la.X, la.Y, st[0], st[1], st[2], dirs.ctypes.data,
                        rc_.ctypes.data, len(rc_), ro_.ctypes.data, len(ro_))
        if rc == 1:
            STATS["revert_py"] += 1
            return orig(undo)
        if rc != 0:
            raise RuntimeError(f"l5_swap failed rc={rc} after rows_ok (state no longer exact)")
        for i, old in zip(ids, olds):
            new[i] = old
        S._dirty.update(int(S.tid[i]) for i in ids)
        STATS["revert_fast"] += 1; STATS["nets"] += len(ids)

    return revert


def _setup2(L):
    if getattr(L, "_l5_s2", False):
        return
    P, I, D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
    L.l5_ovf_score.argtypes = [I, I, I, P, P, P, I, I, P]; L.l5_ovf_score.restype = ctypes.c_int
    L.l5_selc_pairs.argtypes = [I, I, I, P, P, P, I, P, P, I, D, I, P]; L.l5_selc_pairs.restype = ctypes.c_int
    L.l5_selc_cells.argtypes = [P]; L.l5_selc_cells.restype = None
    L.l5_selc_greedy.argtypes = [P, P, I, P, P]; L.l5_selc_greedy.restype = ctypes.c_int64
    L._l5_s2 = True


def _ext_rows(S, wires_only):
    """rows (i, x1..z2) of S.new in dict order (== the list comprehension of ovf_lib.cur_arr / select_c_fast), wires only"""
    new = S.new
    if not new:
        return np.zeros((0, 7), np.int64)
    ids = list(new.keys())
    ext = _rows(ids, [new[i] for i in ids])
    if wires_only:
        ext = ext[ext[:, 3] == ext[:, 6]]
    return ext


def cur_wires(S):
    """== ovf_lib.cur_arr(S, wires_only=True) (same rows, same order)"""
    arr = S.base_arr
    sel = arr[:, 3] == arr[:, 6]
    if S.new:
        chm = np.zeros(max(len(S.nets.names), int(arr[:, 0].max()) + 1 if len(arr) else 0), bool)
        chm[np.fromiter(S.new.keys(), np.int64)] = True
        sel &= ~chm[arr[:, 0]]
        return np.concatenate([arr[sel], _ext_rows(S, True)], 0)
    return arr[sel]


def _cost_at(u, d, c):
    """repair3d_astar._cost on 1-D gathered cells (the same elementwise expression)"""
    import escape3d as E
    open_ = c > E.CAP_EPS
    e = np.where(open_, np.minimum(0.25 * d - 0.5 * c, 700.0), np.minimum(0.75 * d, 700.0))
    return np.where(open_ | (d > 0), u * np.exp(e), 0.0)


def ovf_order(S, n_max, slack_t, slack_min, open_only):
    """== chain_j.ovf_order: (tid, ri, sc). Marginal cost evaluated only on the cells it is non-zero on; score in C++."""
    L_ = lib(); _setup2(L_)
    la = S.la; L, X, Y = la.L, la.X, la.Y
    msk = la.d > la.c
    if open_only:
        msk = msk & (la.c > 1e-3)
    msk[0] = False
    Hm = np.zeros((L, X, Y), np.float64)
    iz, ix, iy = np.nonzero(msk)
    if len(iz):
        u = np.asarray(la.u, np.float64)[iz]
        dv, cv = la.d[iz, ix, iy], la.c[iz, ix, iy]
        Hm[iz, ix, iy] = _cost_at(u, dv, cv) - _cost_at(u, np.maximum(dv - 2.0, 0.0), cv)
    arr = np.ascontiguousarray(cur_wires(S), np.int64)
    nnet = len(S.nets.names)
    sc = np.zeros(nnet, np.float64)
    dirs = np.ascontiguousarray(np.asarray(la.dirs, np.int64))
    rc = L_.l5_ovf_score(L, X, Y, Hm.ctypes.data, dirs.ctypes.data, arr.ctypes.data, len(arr), nnet, sc.ctypes.data)
    if rc != 0:
        return None
    ri = np.nonzero(sc > 0)[0]
    ri = ri[np.argsort(-sc[ri], kind="stable")]
    tid = S.tid[ri]
    ok = tid >= 0
    if slack_min is not None:
        ok &= np.asarray(slack_t, np.float64)[np.maximum(tid, 0)] >= slack_min
    ri, tid = ri[ok][:n_max], tid[ok][:n_max]
    return tid.astype(np.int64), ri, sc


def select_c(S, la, slack_r, npins_r, cap_frac=None, max_nets=0):
    """== ovf2_lib.select_c_fast(S, la, slack_r, npins_r, cap_frac, max_nets) (chosen identical; info cost_uncovered may differ
    in the last bits: Python sums it over a set). None -> the caller runs the original."""
    import ovf_lib as OL
    L_ = lib(); _setup2(L_)
    if cap_frac is None:
        cap_frac = OL.CAP_B
    blk, ovr = OL.ov_masks(la)
    ov = blk | ovr
    L, X, Y = la.L, la.X, la.Y
    arr = S.base_arr
    sel = arr[:, 3] == arr[:, 6]
    if S.new:
        chm = np.zeros(len(S.nets.names), bool)
        chm[np.fromiter(S.new.keys(), np.int64)] = True
        sel &= ~chm[arr[:, 0]]
        wa = np.concatenate([arr[sel], _ext_rows(S, True)], 0)
    else:
        wa = arr[sel]
    wa = np.ascontiguousarray(wa, np.int64)
    ov8 = np.ascontiguousarray(ov, np.uint8)
    sl = np.ascontiguousarray(slack_r, np.float64); npn = np.ascontiguousarray(npins_r, np.int64)
    dirs = np.ascontiguousarray(np.asarray(la.dirs, np.int64))
    out = np.zeros(4, np.int64)
    rc = L_.l5_selc_pairs(L, X, Y, ov8.ctypes.data, dirs.ctypes.data, wa.ctypes.data, len(wa), sl.ctypes.data, npn.ctypes.data,
                          len(sl), float(OL.SLACK_GUARD), int(OL.MAX_PINS), out.ctypes.data)
    if rc != 0:
        return None
    ncross, npool, npairs, ncells = (int(v) for v in out)
    if ncross == 0:
        return [], {"pool": 0}
    info = {"nets_crossing": ncross, "pool": npool, "ov_cells": int(ov.sum())}
    if npool == 0:
        return [], info
    cells = np.zeros(ncells, np.int64); L_.l5_selc_cells(cells.ctypes.data)
    zc_, r_ = np.divmod(cells, X * Y); xc_, yc_ = np.divmod(r_, Y)
    ccost = np.ascontiguousarray(_cost_at(np.asarray(la.u, np.float64)[zc_], la.d[zc_, xc_, yc_], la.c[zc_, xc_, yc_]), np.float64)
    cap_k = max(1, int(round(cap_frac * npool)))
    if max_nets:
        cap_k = min(cap_k, max_nets)
    ch = np.zeros(npool, np.int64); inf2 = np.zeros(2)
    nch = int(L_.l5_selc_greedy(ccost.ctypes.data, sl.ctypes.data, cap_k, ch.ctypes.data, inf2.ctypes.data))
    # blk_share_ov on the overflowed cells only (info; the original sums over the full map)
    oz, ox, oy = np.nonzero(ov)
    cov_ = _cost_at(np.asarray(la.u, np.float64)[oz], la.d[oz, ox, oy], la.c[oz, ox, oy])
    info.update({"cells_total": ncells, "cells_uncovered": int(inf2[0]), "chosen": nch, "cost_uncovered": float(inf2[1]),
                 "blk_share_ov": float((cov_ * blk[oz, ox, oy]).sum() / max(float(cov_.sum()), 1e-30))})
    return [int(v) for v in ch[:nch]], info


def normalise(S, idx):
    """== tools/fast3x/fast_norm._normalise(S, idx) (same caches S._f3x_ap / S._f3x_pr, same C++ normaliser call, same output
    dict of [m,7] int64 blocks); the segment lists are flattened with np.fromiter instead of list.extend + np.array."""
    import fast_norm as FN
    if not hasattr(S, "_f3x_ap"):
        S._f3x_ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        S._f3x_pr = {}
    pr = S._f3x_pr
    nn = len(idx)
    new = S.new
    lists = [new[i] for i in idx]
    cnt = np.fromiter((len(sg) for sg in lists), np.int64, nn)
    pa0 = np.zeros(nn, np.int64); pa1 = np.zeros(nn, np.int64)
    for k, i in enumerate(idx):
        r = pr.get(i)
        if r is None:
            r = pr[i] = FN._pin_range(S, i)
        pa0[k], pa1[k] = r
    start = np.zeros(nn, np.int64)
    if nn > 1:
        start[1:] = np.cumsum(cnt)[:-1]
    m = int(cnt.sum())
    if m:
        segs = np.fromiter(chain.from_iterable(chain.from_iterable(lists)), np.int32, 6 * m).reshape(m, 6)
    else:
        segs = np.zeros((1, 6), np.int32)
    segs = np.ascontiguousarray(segs)
    L = FN._lib
    tot = L.nw_run(nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data, pa0.ctypes.data, pa1.ctypes.data, S._f3x_ap.ctypes.data, 1)
    oc = [np.empty(max(tot, 1), np.int32) for _ in range(6)]; oo = np.zeros(nn + 1, np.uint64)
    L.nw_fetch(*(c.ctypes.data for c in oc), oo.ctypes.data)
    blk = np.empty((tot, 7), np.int64)
    for k in range(6):
        blk[:, k + 1] = oc[k][:tot]
    oo = oo.astype(np.int64)
    if tot:
        blk[:, 0] = np.repeat(np.asarray(idx, np.int64), np.diff(oo))
    ol = oo.tolist()
    return {i: blk[ol[k]:ol[k + 1]] for k, i in enumerate(idx)}
