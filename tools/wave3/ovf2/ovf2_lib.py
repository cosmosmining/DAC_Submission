#!/usr/bin/env python3
"""ovf2_lib.py: C++ front end of the overflow stage's ranked reroute.

  reroute_rank_cpp(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax)
      same contract and same result as tools/ovf3/ovf_lib.reroute_rank (undo dict, S.new / S.la / S._dirty changed the same way),
      the whole per-run loop in tools/wave3/ovf2/rr_loop.cc (one call), the A* = the same compiled astar3d_win function.
      Falls back to ovf_lib.reroute_rank when a library or the window A* is missing or the arrays are not float64.
  removable_bound(S, slack_r, npins_r)  cheap skip-test quantity (see chain_w3.py --ovf_skip).
"""
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(HERE, f"libw3_rr_{sys.platform}.so")
_SO = None
_PINC = {}                                   # route-independent pin tables per net key: (pin counts, ap array)


def _load():
    global _SO
    if _SO is None and os.path.exists(_LIB):
        so = ctypes.CDLL(_LIB)
        V = ctypes.c_void_p
        so.w3_rr_rank.restype = ctypes.c_int
        so.w3_rr_rank.argtypes = [ctypes.c_int] * 3 + [V] * 6 + [V] + [ctypes.c_int] + [V] * 6 + [ctypes.c_int] * 3 + [V]
        so.w3_rr_nseg.restype = ctypes.c_int64
        so.w3_rr_get.argtypes = [V, V, V]
        _SO = so
    return _SO


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _pins_of(pins, key):
    t = _PINC.get(key)
    if t is None:
        pl = pins.get(key, [])
        cnt = np.array([len(aps) for aps in pl], np.int64)
        flat = [int(v) for aps in pl for ap in aps for v in ap[:3]]
        t = _PINC[key] = (cnt, np.array(flat, np.int32).reshape(-1, 3))
    return t


def reroute_rank_cpp(S, pins, chosen, W=14, pin_on=False, pcmap=None, stats=None, astar="win", max_runs=0, hmax=0, timing=None):
    import gc
    ge = gc.isenabled()
    gc.disable()                      # no collector passes over the chain's large heap while tuples are built (same result)
    try:
        return _rr_cpp(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax, timing)
    finally:
        if ge:
            gc.enable()


def _rr_cpp(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax, timing):
    import ovf_lib as OL
    from timing_promote import _key
    so = _load()
    la = S.la
    if so is None or astar != "win" or OL._CW is None or la.d.dtype != np.float64 or la.c.dtype != np.float64:
        return OL.reroute_rank(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax)
    t0 = time.perf_counter()
    if pin_on and pcmap is not None:
        c_eff = np.array(la.c, dtype=np.float64, copy=True)
        for z in (1, 2):
            if z < la.L:
                c_eff[z] = np.where(c_eff[z] > OL.CAP_EPS, np.maximum(c_eff[z] - OL.PIN_BETA * pcmap, 0.5 * c_eff[z]), c_eff[z])
        cs = c_eff
    else:
        cs = la.c
    if getattr(la, "_c_dirs", None) is None:
        la._c_dirs = np.asarray(la.dirs, np.int32)
        la._c_u = np.ascontiguousarray(la.u, np.float64)
    d, c = la.d, la.c
    strides = np.array([v // 8 for v in d.strides] + [v // 8 for v in c.strides] + [v // 8 for v in cs.strides], np.int64)
    chosen = list(chosen)
    n = len(chosen)
    ridx = np.asarray(chosen, np.int64)
    names = S.nets.names
    arr, off = S.segs.arr, S.segs.off
    parts, cnts = [], np.zeros(n, np.int64)
    for s, i in enumerate(chosen):
        sg = S.new.get(i)
        if sg is None:
            a_ = arr[off[i]:off[i + 1]]
        else:
            a_ = np.asarray(sg, np.int64).reshape(-1, 6) if len(sg) else np.zeros((0, 6), np.int64)
        parts.append(a_); cnts[s] = len(a_)
    segs = np.ascontiguousarray(np.concatenate(parts, 0) if parts else np.zeros((0, 6)), np.int32)
    seg_off = np.zeros(n + 1, np.int64); seg_off[1:] = np.cumsum(cnts)
    pc, ap_parts = [], []
    for i in chosen:
        cnt, aps = _pins_of(pins, _key(names[i]))
        pc.append(cnt); ap_parts.append(aps)
    npin = np.array([len(x) for x in pc], np.int64)
    pin_off = np.zeros(n + 1, np.int64); pin_off[1:] = np.cumsum(npin)
    allc = np.concatenate(pc) if pc else np.zeros(0, np.int64)
    ap_off = np.zeros(len(allc) + 1, np.int64); ap_off[1:] = np.cumsum(allc)
    aps = np.ascontiguousarray(np.concatenate(ap_parts, 0) if ap_parts else np.zeros((0, 3)), np.int32)
    if len(aps) == 0:
        aps = np.zeros((1, 3), np.int32)
    fptr = ctypes.cast(OL._CW.astar3d_win, ctypes.c_void_p).value
    st = np.zeros(16, np.int64)
    t1 = time.perf_counter()
    nch = so.w3_rr_rank(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, cs.ctypes.data, _p(la._c_u), _p(la._c_dirs), _p(strides),
                        fptr, n, _p(ridx), _p(seg_off), _p(segs), _p(pin_off), _p(ap_off), _p(aps), int(W), int(hmax), int(max_runs), _p(st))
    t2 = time.perf_counter()
    ns = int(so.w3_rr_nseg())
    slots = np.zeros(max(nch, 1), np.int32); roff = np.zeros(nch + 1, np.int64); rseg = np.zeros((max(ns, 1), 6), np.int32)
    so.w3_rr_get(_p(slots), _p(roff), _p(rseg))
    view = OL.CurNets(S)
    undo = {}
    rl = rseg[:ns].tolist()
    for k in range(nch):
        i = chosen[int(slots[k])]
        undo[i] = view.segs(i)
        S.new[i] = [tuple(r) for r in rl[roff[k]:roff[k + 1]]]
        S._dirty.add(int(S.tid[i]))
    keys = ["big_skip", "hmax_skip", "cand_runs", "stale", "pin_end", "spur", "spur_end", "no_path", "reject_invalid",
            "rerouted", "kept", "nets_changed", "astar_calls"]
    if stats is not None:
        for k_, v_ in zip(keys, st.tolist()):
            if v_ or k_ in ("cand_runs", "nets_changed"):
                stats[k_] = int(v_)
    if timing is not None:
        timing.update(prep_s=t1 - t0, loop_s=t2 - t1, out_s=time.perf_counter() - t2)
    return undo


def pretest_cost(S):
    """ovf_lib.pretest (same numbers) + the cost array it computed, for reuse by the first selection on the same demand."""
    import ovf_lib as OL
    la = S.la
    blk, ovr = OL.ov_masks(la)
    cost = OL.RA._cost(la.u[:, None, None], la.d, la.c)
    cb, co = float((cost * blk).sum()), float((cost * ovr).sum())
    return {"ov_cells": int(blk.sum() + ovr.sum()), "blocked_cells": int(blk.sum()), "blk_cost": cb, "open_cost": co,
            "bound_score": float(S.w4) * (cb + co), "bound_blk_score": float(S.w4) * cb,
            "blk_share_ov": cb / max(cb + co, 1e-30)}, cost


def select_c_fast(S, la, slack_r, npins_r, cap_frac=None, max_nets=0, cost=None):
    """ovf_lib.select_c with the same result, faster on large routes: (1) the changed-net filter of cur_arr by a boolean lookup
    instead of np.isin; (2) only wire runs whose track holds an overflowed cell inside the run (prefix counts of the overflow mask)
    are expanded into cells. The (net, cell) pairs go through np.unique in select_c, so their order never mattered; everything
    after the pair list is select_c's code unchanged."""
    import heapq
    import ovf_lib as OL
    RA = OL.RA
    if cap_frac is None:
        cap_frac = OL.CAP_B
    blk, ovr = OL.ov_masks(la)
    ov = blk | ovr
    if cost is None:                                   # else: the caller's array of the same expression on the same demand
        cost = RA._cost(la.u[:, None, None], la.d, la.c)
    L, X, Y = la.L, la.X, la.Y
    arr = S.base_arr
    sel = arr[:, 3] == arr[:, 6]
    if S.new:
        chm = np.zeros(len(S.nets.names), bool)
        chm[np.fromiter(S.new.keys(), np.int64)] = True
        sel &= ~chm[arr[:, 0]]
        ext = [(i,) + tuple(sg) for i, segs in S.new.items() for sg in segs]
        ext = np.array(ext, np.int64).reshape(-1, 7) if ext else np.zeros((0, 7), np.int64)
        ext = ext[ext[:, 3] == ext[:, 6]]
        wa = np.concatenate([arr[sel], ext], 0)
    else:
        wa = arr[sel]
    n, x1, y1, z1, x2, y2 = (wa[:, k] for k in range(6))
    dirs = np.asarray(la.dirs)
    zc = np.clip(z1, 0, L - 1)
    hor = (dirs[zc] == 0) & (y1 == y2) & (x1 != x2) & (z1 > 0)
    ver = (dirs[zc] == 1) & (x1 == x2) & (y1 != y2) & (z1 > 0)
    ovi = ov.astype(np.int32)
    PX = np.zeros((L, X + 1, Y), np.int32); np.cumsum(ovi, axis=1, out=PX[:, 1:, :])
    PY = np.zeros((L, X, Y + 1), np.int32); np.cumsum(ovi, axis=2, out=PY[:, :, 1:])
    if hor.any():
        h = np.flatnonzero(hor)
        lo, hi = np.minimum(x1[h], x2[h]), np.maximum(x1[h], x2[h])
        k = PX[z1[h], hi, y1[h]] - PX[z1[h], lo, y1[h]]
        hor[h[k == 0]] = False
    if ver.any():
        v = np.flatnonzero(ver)
        lo, hi = np.minimum(y1[v], y2[v]), np.maximum(y1[v], y2[v])
        k = PY[z1[v], x1[v], hi] - PY[z1[v], x1[v], lo]
        ver[v[k == 0]] = False
    del PX, PY, ovi
    pairs_n, pairs_c = [], []
    for msk, isx in ((hor, True), (ver, False)):
        if not msk.any():
            continue
        nn_, z_ = n[msk], z1[msk]
        lo = np.minimum(x1[msk], x2[msk]) if isx else np.minimum(y1[msk], y2[msk])
        hi = np.maximum(x1[msk], x2[msk]) if isx else np.maximum(y1[msk], y2[msk])
        fx = y1[msk] if isx else x1[msk]
        ln = hi - lo
        tot = int(ln.sum())
        rep = np.repeat(np.arange(len(ln)), ln)
        t = lo[rep] + (np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln))
        zz = z_[rep]
        xx = t if isx else fx[rep]
        yy = fx[rep] if isx else t
        lin = (zz * X + xx) * Y + yy
        hit = ov.reshape(-1)[lin] if ov.flags.c_contiguous else ov[zz, xx, yy]
        pairs_n.append(nn_[rep][hit]); pairs_c.append(lin[hit])
    if not pairs_n:
        return [], {"pool": 0}
    pn = np.concatenate(pairs_n); pc = np.concatenate(pairs_c)
    pr = np.unique(pn * (L * X * Y) + pc)
    pn, pc = pr // (L * X * Y), pr % (L * X * Y)
    ovn = np.unique(pn)
    pool_m = (slack_r[pn] >= OL.SLACK_GUARD) & (npins_r[pn] <= OL.MAX_PINS)
    pn, pc = pn[pool_m], pc[pool_m]
    pool = np.unique(pn)
    info = {"nets_crossing": int(len(ovn)), "pool": int(len(pool)), "ov_cells": int(ov.sum())}
    if not len(pool):
        return [], info
    zc_, r_ = np.divmod(pc, X * Y); xc_, yc_ = np.divmod(r_, Y)
    cc = cost[zc_, xc_, yc_]
    bounds = np.searchsorted(pn, pool, side="left"), np.searchsorted(pn, pool, side="right")
    cells_of = {}
    cell_cost = {}
    pcl, ccl = pc.tolist(), cc.tolist()
    for j, i in enumerate(pool.tolist()):
        s0, s1 = int(bounds[0][j]), int(bounds[1][j])
        cl = pcl[s0:s1]
        cells_of[i] = cl
        for k in range(s0, s1):
            cell_cost[pcl[k]] = ccl[k]
    sl = slack_r
    wt = {i: 1.0 + max(0.0, 0.5 - float(sl[i])) / 0.5 for i in cells_of}
    cap_k = max(1, int(round(cap_frac * len(pool))))
    if max_nets:
        cap_k = min(cap_k, max_nets)
    uncovered = set(cell_cost)
    hq = [(-sum(cell_cost[c] for c in cells_of[i]) / wt[i], i) for i in cells_of]
    heapq.heapify(hq)
    chosen = []
    while hq and uncovered and len(chosen) < cap_k:
        negv, i = heapq.heappop(hq)
        val = sum(cell_cost[c] for c in cells_of[i] if c in uncovered) / wt[i]
        if val <= 0:
            continue
        if hq and val < -hq[0][0] - 1e-12:
            heapq.heappush(hq, (-val, i))
            continue
        chosen.append(i)
        uncovered.difference_update(cells_of[i])
    info.update({"cells_total": len(cell_cost), "cells_uncovered": len(uncovered), "chosen": len(chosen),
                 "cost_uncovered": float(sum(cell_cost[c] for c in uncovered)),
                 "blk_share_ov": float((cost * blk).sum() / max(float((cost * ov).sum()), 1e-30))})
    return sorted(chosen), info
