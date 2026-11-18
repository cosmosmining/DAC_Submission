#!/usr/bin/env python3
"""ovf_lib.py: pieces of the in-memory OVERFLOW stage of tools/ovf3/chain_ovf.py.

Everything works on the chain's already-loaded state S (fastmode4.State4 subclass of chain_k.py: S.la demand,
S.base_arr / S.segs = single-pass route, S.new = changed nets) and changes S.new and S.la in place, returning an
undo dict {route net index: old segments} that S.revert() understands. Nothing is re-read or written.

  pretest(S)            overflow cost on overflowed cells (blocked with demand, or open with d > 2c), split
                        blocked / open, and the upper bound w4 x that cost on what the stage can gain.
  escape_pass(...)      escape3d.escape (pin escape out of blocked hotspot cells, compiled A*) on S, one pass per call,
                        restricted to the nets with a via next to a hotspot (vectorised scan; escape3d's own candidate
                        rule then decides exactly as before).
  select_c(...)         rule c of tools/comp2/ripup_select (pool, greedy weighted hitting set), vectorised.
  reroute_c(...)        repair3d_big.repair_fast (one pass, overflowed cells, window A*) on the chosen nets only,
                        optional pin-density term, union-find replaced by an interval-level one (same answers).
READ-ONLY reuse of escape3d, escape3d_fast, repair3d, repair3d_astar, repair3d_big, timing_promote.
"""
import heapq
import time
from collections import defaultdict

import numpy as np

import escape3d as E                                         # READ-ONLY reuse
import escape3d_fast                                         # noqa: F401  (patches E.dijkstra / marginal / apply)
import repair3d as Rp
import repair3d_astar as RA
import repair3d_big as RB
from timing_promote import _key

CAP_EPS = E.CAP_EPS
SLACK_GUARD, MAX_PINS, CAP_B, PIN_BETA = 0.05, 1000, 0.25, 0.5


def emit_tuples(la, wires, vias):
    """escape3d.emit with GCell 6-tuples instead of text lines (same order)."""
    buf = []
    for (z, f), ivs in wires.items():
        for a_, b_ in ivs:
            buf.append((a_, f, z, b_, f, z) if la.dirs[z] == 0 else (f, a_, z, f, b_, z))
    for (x, y, z) in sorted(vias):
        buf.append((x, y, z, x, y, z + 1))
    return buf


class CurNets:
    """read_route25-like view (name, current segments, None) of S, optionally restricted to route indices idx."""
    def __init__(self, S, idx=None):
        self.S, self.idx = S, idx
        self.names = S.nets.names
        self._c = {}                                     # base segments decoded once per view (escape3d re-reads nets[i]
        #                                                  for every stack: 39 s of 45 on ariane blind without this)

    def segs(self, i):
        sg = self.S.new.get(i)
        if sg is not None:
            return sg
        sg = self._c.get(i)
        if sg is None:
            sg = self._c[i] = self.S.segs[i]
        return sg

    def __getitem__(self, k):
        i = k if self.idx is None else self.idx[k]
        return (self.names[i], self.segs(i), None)

    def __len__(self):
        return len(self.names) if self.idx is None else len(self.idx)

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]


def cur_arr(S, vias_only=False, wires_only=False):
    """[n, 7] (route net, x1, y1, z1, x2, y2, z2) of the current route (base rows of unchanged nets + S.new)."""
    arr = S.base_arr
    sel = None
    if vias_only:
        sel = arr[:, 3] != arr[:, 6]
    elif wires_only:
        sel = arr[:, 3] == arr[:, 6]
    if S.new:
        ch = np.fromiter(S.new.keys(), np.int64)
        keep = ~np.isin(arr[:, 0], ch)
        sel = keep if sel is None else (sel & keep)
        ext = [(i,) + tuple(sg) for i, segs in S.new.items() for sg in segs]
        ext = np.array(ext, np.int64).reshape(-1, 7) if ext else np.zeros((0, 7), np.int64)
        if vias_only:
            ext = ext[ext[:, 3] != ext[:, 6]]
        elif wires_only:
            ext = ext[ext[:, 3] == ext[:, 6]]
        return np.concatenate([arr[sel], ext], 0)
    return arr if sel is None else arr[sel]


def ov_masks(la):
    blk = (la.c <= CAP_EPS) & (la.d > 0)
    blk[0] = False
    ovr = (la.c > CAP_EPS) & (la.d > 2.0 * la.c)
    ovr[0] = False
    return blk, ovr


def pretest(S):
    la = S.la
    blk, ovr = ov_masks(la)
    cost = RA._cost(la.u[:, None, None], la.d, la.c)
    cb, co = float((cost * blk).sum()), float((cost * ovr).sum())
    return {"ov_cells": int(blk.sum() + ovr.sum()), "blocked_cells": int(blk.sum()), "blk_cost": cb, "open_cost": co,
            "bound_score": float(S.w4) * (cb + co), "bound_blk_score": float(S.w4) * cb,
            "blk_share_ov": cb / max(cb + co, 1e-30)}


# ───────────────────────────── union-find on intervals (same answers as repair3d.components / net_valid)

def _iv_components(la, wires, vias):
    """interval-level union-find. returns (nodes, find, locate): nodes = list of ('w', z, fx, a, b) / ('p', z, x, y)."""
    dirs = la.dirs
    nodes, par = [], []
    track = {}
    for (z, fx), ivs in wires.items():
        lst = sorted(ivs)
        ks = []
        prev_b, prev_k = None, None
        for a, b in lst:
            k = len(nodes)
            nodes.append(("w", z, fx, a, b)); par.append(k); ks.append(k)
            if prev_b is not None and a <= prev_b:          # touching / overlapping on one track (merged inputs never)
                par[k] = prev_k
            if prev_b is None or b > prev_b:
                prev_b, prev_k = b, k
        track[(z, fx)] = (lst, ks)
    pnode = {}

    def find(k):
        while par[k] != k:
            par[k] = par[par[k]]
            k = par[k]
        return k

    def locate(z, x, y):
        """node of point (z, x, y): covering interval, else a point node (created)."""
        if dirs[z] == 0:
            t = track.get((z, y)); pos = x
        else:
            t = track.get((z, x)); pos = y
        if t is not None:
            lst, ks = t
            lo, hi = 0, len(lst)
            while lo < hi:                                      # last interval with a <= pos
                m = (lo + hi) >> 1
                if lst[m][0] <= pos:
                    lo = m + 1
                else:
                    hi = m
            j = lo - 1
            while j >= 0:
                if lst[j][1] >= pos:
                    return ks[j]
                if lst[j][1] < pos:
                    break
                j -= 1
        q = (z, x, y)
        k = pnode.get(q)
        if k is None:
            k = len(nodes)
            nodes.append(("p", z, x, y)); par.append(k); pnode[q] = k
        return k

    for (x, y, z) in vias:
        k1, k2 = find(locate(z, x, y)), find(locate(z + 1, x, y))
        if k1 != k2:
            par[k1] = k2
    return nodes, find, track, pnode


def components_fast(la, wires, vias):
    """repair3d.components: dict point -> root (roots are representative points of the component)."""
    nodes, find, _, _ = _iv_components(la, wires, vias)
    dirs = la.dirs
    rep = {}
    out = {}
    for k, nd in enumerate(nodes):
        r = find(k)
        if r not in rep:
            n0 = nodes[r]
            rep[r] = ((n0[1], n0[3], n0[2]) if dirs[n0[1]] == 0 else (n0[1], n0[2], n0[3])) if n0[0] == "w" else n0[1:]
        rp = rep[r]
        if nd[0] == "w":
            _, z, fx, a, b = nd
            if dirs[z] == 0:
                for t in range(a, b + 1):
                    out[(z, t, fx)] = rp
            else:
                for t in range(a, b + 1):
                    out[(z, fx, t)] = rp
        else:
            out[nd[1:]] = rp
    return out


def net_valid_fast(la, wires, vias, pins):
    """repair3d.net_valid without expanding points."""
    nodes, find, track, pnode = _iv_components(la, wires, vias)
    if not nodes:
        return False
    r0 = find(0)
    for k in range(1, len(nodes)):
        if find(k) != r0:
            return False
    dirs = la.dirs
    for aps in pins:
        ok = False
        for (l, x, y) in aps:
            if (l, x, y) in pnode:
                ok = True
                break
            t = track.get((l, y) if dirs[l] == 0 else (l, x))
            if t is not None:
                pos = x if dirs[l] == 0 else y
                for a, b in t[0]:
                    if a <= pos <= b:
                        ok = True
                        break
                if ok:
                    break
        if not ok:
            return False
    return True


# ───────────────────────────── pin escape (escape3d) on S

def escape_pass(S, pins, n_hot, guarded, W=12, stats=None, astar="c"):
    """one escape3d.escape pass on S.la / S.new. returns {route idx: old segs} of the nets it changed."""
    la = S.la
    E.emit = emit_tuples
    E.dijkstra = astar_win if astar == "win" else RA.astar_c
    blk = (la.c <= CAP_EPS) & (la.d > 0)
    blk[0] = False
    zz, xx, yy = np.nonzero(blk)
    if len(zz) == 0:
        return {}
    costs = la.u[zz] * np.exp(np.minimum(0.75 * la.d[zz, xx, yy], 700.0))
    order = np.argsort(-costs)[:n_hot]
    hx, hy = xx[order].astype(np.int64), yy[order].astype(np.int64)
    Y = la.Y
    keys = np.unique(np.concatenate([hx * Y + hy, (hx + 1) * Y + hy, hx * Y + hy + 1]))   # superset of escape3d's hot_xy
    va = cur_arr(S, vias_only=True)
    m = np.isin(va[:, 1] * Y + va[:, 2], keys)
    idx = np.unique(va[m, 0]).tolist()
    sub = CurNets(S, idx)
    new_lines, st, _ = E.escape(None, pins, sub, n_hot, W, 1.0, 0.1, 1, la=la, guarded=guarded)
    undo = {}
    for k, segs in new_lines.items():
        i = idx[k]
        undo[i] = sub.segs(i)
        S.new[i] = segs
        S._dirty.add(int(S.tid[i]))
    if stats is not None:
        for k_, v_ in st.items():
            stats[k_] = stats.get(k_, 0) + v_
        stats["scan_nets"] = stats.get("scan_nets", 0) + len(idx)
    return undo


# ───────────────────────────── rule c selection + reroute on S

def select_c(S, la, slack_r, npins_r, cap_frac=CAP_B, max_nets=0):
    """ripup_select rule c: pool = nets crossing an overflowed cell with slack >= guard and <= MAX_PINS pins; greedy
    weighted hitting set over the pool's overflowed cells (weight 1 + max(0, 0.5 - slack) / 0.5); stop when all are
    covered or cap_frac of the pool is chosen. returns (chosen route indices sorted, info)."""
    blk, ovr = ov_masks(la)
    ov = blk | ovr
    cost = RA._cost(la.u[:, None, None], la.d, la.c)
    L, X, Y = la.L, la.X, la.Y
    wa = cur_arr(S, wires_only=True)
    n, x1, y1, z1, x2, y2 = (wa[:, k] for k in range(6))
    dirs = np.asarray(la.dirs)
    zc = np.clip(z1, 0, L - 1)
    hor = (dirs[zc] == 0) & (y1 == y2) & (x1 != x2) & (z1 > 0)
    ver = (dirs[zc] == 1) & (x1 == x2) & (y1 != y2) & (z1 > 0)
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
    pool_m = (slack_r[pn] >= SLACK_GUARD) & (npins_r[pn] <= MAX_PINS)
    pn, pc = pn[pool_m], pc[pool_m]
    pool = np.unique(pn)
    info = {"nets_crossing": int(len(ovn)), "pool": int(len(pool)), "ov_cells": int(ov.sum())}
    if not len(pool):
        return [], info
    zc_, r_ = np.divmod(pc, X * Y); xc_, yc_ = np.divmod(r_, Y)
    cc = cost[zc_, xc_, yc_]
    # per-net cell lists (pn sorted by net because pr is sorted)
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


class SearchLA:
    """view of la for the A* search only: same demand array, capacity with the pin-density term."""
    def __init__(self, la, c):
        self.L, self.X, self.Y, self.dirs, self.u, self.d, self.c = la.L, la.X, la.Y, la.dirs, la.u, la.d, c
        self._c_dirs = None


def pin_count_map(pins, X, Y):
    """number of nets with an access point in each (x, y) GCell (ripup_select's pin-density term), route independent."""
    xs, ys = [], []
    for nm, aps_all in pins.items():
        seen = set()
        for aps in aps_all:
            for ap_ in aps:
                seen.add((int(ap_[1]), int(ap_[2])))
        for x, y in seen:
            xs.append(x); ys.append(y)
    xs, ys = np.asarray(xs, np.int64), np.asarray(ys, np.int64)
    ok = (xs >= 0) & (xs < X) & (ys >= 0) & (ys < Y)
    pc = np.zeros((X, Y))
    np.add.at(pc, (xs[ok], ys[ok]), 1.0)
    return pc


def reroute_c(S, pins, chosen, W=14, pin_on=False, pcmap=None, fast_uf=True, stats=None, astar="c", n_hot=0):
    """repair3d_big.repair_fast, one pass over the overflowed cells, on the chosen nets only. returns undo."""
    la = S.la
    blk, ovr = ov_masks(la)
    n_ov = int(blk.sum() + ovr.sum())
    Rp.apply, Rp.marginal, Rp.emit = RA.apply_fast, RA.marginal_fast, emit_tuples
    if fast_uf:
        Rp.components, Rp.net_valid = components_fast, net_valid_fast
    RB.LOCAL_OK, RB.RRR_TOP, RB.ALL_RUNS = True, 0, 0
    base = astar_win if astar == "win" else RA.astar_c
    if pin_on and pcmap is not None:
        c_eff = np.array(la.c, dtype=np.float64, copy=True)
        for z in (1, 2):
            if z < la.L:
                c_eff[z] = np.where(c_eff[z] > CAP_EPS, np.maximum(c_eff[z] - PIN_BETA * pcmap, 0.5 * c_eff[z]), c_eff[z])
        SL = SearchLA(la, c_eff)
        Rp.dijkstra = lambda la_, *args: base(SL, *args)
    else:
        Rp.dijkstra = base
    sub = CurNets(S, list(chosen))
    new_lines, st = RB.repair_fast(la, pins, sub, n_hot=(n_hot if n_hot else n_ov + 1), W=W, guarded=frozenset(), passes=1, overcap=True,
                                   all_runs=0)
    undo = {}
    for k, segs in new_lines.items():
        i = sub.idx[k]
        undo[i] = sub.segs(i)
        S.new[i] = segs
        S._dirty.add(int(S.tid[i]))
    if stats is not None:
        stats.update(st)
        stats["nets_changed"] = len(new_lines)
    return undo


# ───────────────────────────── window A* (tools/ovf3/astar3d_win.cpp: same path as repair3d_astar.astar_c, faster)

def _load_win():
    import ctypes
    import os
    import sys
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libastar3d_win_{sys.platform}.so")
    if not os.path.exists(lib):
        return None
    so = ctypes.CDLL(lib)
    P = np.ctypeslib.ndpointer
    so.astar3d_win.restype = ctypes.c_int
    so.astar3d_win.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2 + [P(np.float64, flags="C"), P(np.int32, flags="C"),
                               P(np.int64, flags="C"), P(np.int32, flags="C"), ctypes.c_int, P(np.int32, flags="C"),
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                               P(np.int32, flags="C"), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    return so


_CW = _load_win()
_PATHW = [np.zeros((1 << 16, 3), np.int32)]
ASTAR_CALLS = [0]
ALOG = [] if __import__("os").environ.get("OVF3_ALOG") else None     # (window half, seconds, path nodes) per call


def astar_win(la, sources, targets, x0, y0, W, step_pen, via_pen):
    """repair3d_astar.astar_c with the window-buffer library; same contract and result."""
    import ctypes
    import math
    if _CW is None:
        return RA.astar_c(la, sources, targets, x0, y0, W, step_pen, via_pen)
    if not targets:
        return math.inf, None
    if getattr(la, "_c_dirs", None) is None:
        la._c_dirs = np.asarray(la.dirs, np.int32)
        la._c_u = np.ascontiguousarray(la.u, np.float64)
    d, c = la.d, la.c
    if d.dtype != np.float64 or c.dtype != np.float64:
        return RA.astar_c(la, sources, targets, x0, y0, W, step_pen, via_pen)
    strides = np.array([v // 8 for v in d.strides] + [v // 8 for v in c.strides], np.int64)
    src = np.asarray(list(sources), np.int32).reshape(-1, 3)
    tgt = np.asarray(list(targets), np.int32).reshape(-1, 3)
    cost = ctypes.c_double()
    ASTAR_CALLS[0] += 1
    t0_ = time.perf_counter() if ALOG is not None else 0.0
    while True:
        buf = _PATHW[0]
        n = _CW.astar3d_win(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, la._c_u, la._c_dirs, strides, src, len(src),
                            tgt, len(tgt), int(x0), int(y0), int(W), float(step_pen), float(via_pen), buf, len(buf),
                            ctypes.byref(cost))
        if n >= 0:
            break
        _PATHW[0] = np.zeros((-n * 2, 3), np.int32)
    if ALOG is not None:
        ALOG.append((int(W), time.perf_counter() - t0_, n))
    if n == 0:
        return math.inf, None
    return cost.value, [tuple(int(v) for v in r) for r in buf[:n]]


# ───────────────────────────── ranked, budgeted reroute (repair3d_big.repair_fast's small-net decision, new order)

def reroute_rank(S, pins, chosen, W=14, pin_on=False, pcmap=None, stats=None, astar="win", max_runs=0, hmax=0):
    """the same per-run decision as repair3d_big.repair_fast (drop the run, reconnect the pieces by window A*, keep if
    the net's model cost falls and the net stays valid), but the candidate runs of the chosen nets are ordered by the
    cost their own demand adds on overflowed cells (largest first) instead of by length, optionally only the first
    max_runs of them and only runs whose search window half-width is <= hmax. Nets above repair3d_big.BIG_NET are
    skipped (repair_fast's local big-net path is not reproduced here). returns undo."""
    from timing_promote import to_wires_vias
    la = S.la
    blk, ovr = ov_masks(la)
    ov = blk | ovr
    u3 = la.u[:, None, None]
    m = (RA._cost(u3, la.d, la.c) - RA._cost(u3, np.maximum(la.d - 2, 0), la.c)) * ov
    pref, prefc = {}, {}
    for z in range(1, la.L):
        ax = 0 if la.dirs[z] == 0 else 1
        pz = np.cumsum(m[z], axis=ax); pc = np.cumsum(ov[z].astype(np.int32), axis=ax)
        z0 = np.zeros_like(pz[:1]) if ax == 0 else np.zeros_like(pz[:, :1])
        c0 = np.zeros_like(pc[:1]) if ax == 0 else np.zeros_like(pc[:, :1])
        pref[z] = np.concatenate([z0, pz], axis=ax); prefc[z] = np.concatenate([c0, pc], axis=ax)
    apply, marginal = RA.apply_fast, RA.marginal_fast
    Rp.emit = emit_tuples
    base = astar_win if astar == "win" else RA.astar_c
    if pin_on and pcmap is not None:
        c_eff = np.array(la.c, dtype=np.float64, copy=True)
        for z in (1, 2):
            if z < la.L:
                c_eff[z] = np.where(c_eff[z] > CAP_EPS, np.maximum(c_eff[z] - PIN_BETA * pcmap, 0.5 * c_eff[z]), c_eff[z])
        SL = SearchLA(la, c_eff)
        search = lambda la_, *args: base(SL, *args)
    else:
        search = base
    st = defaultdict(int)
    cur = {}
    cand = []
    names = S.nets.names
    view = CurNets(S)
    for i in chosen:
        w, v = to_wires_vias(view.segs(i), la.dirs)
        if len(v) + sum(map(len, w.values())) > RB.BIG_NET:
            st["big_skip"] += 1
            continue
        cur[i] = (w, v)
        for (z, fx), ivs in w.items():
            if z not in pref:
                continue
            P, C = pref[z], prefc[z]
            h = la.dirs[z] == 0
            for a, b in ivs:
                k = (C[b, fx] - C[a, fx]) if h else (C[fx, b] - C[fx, a])
                if not k:
                    continue
                if hmax and (b - a) // 2 + W > hmax:
                    st["hmax_skip"] += 1
                    continue
                sc = float((P[b, fx] - P[a, fx]) if h else (P[fx, b] - P[fx, a]))
                cand.append((-sc, i, (z, fx), a, b))
    cand.sort()
    st["cand_runs"] = len(cand)
    if max_runs:
        cand = cand[:max_runs]
    changed = set()
    for _, i, key, a, b in cand:
        w, v = cur[i]
        if key not in w or (a, b) not in w[key]:
            st["stale"] += 1
            continue
        pin_list = pins.get(_key(names[i]), [])
        apply(la, w, v, -1)
        old = marginal(la, w, v)
        w_rest = Rp.drop_run(w, key, a, b)
        z, fx = key
        A = (z, a, fx) if la.dirs[z] == 0 else (z, fx, a)
        B = (z, b, fx) if la.dirs[z] == 0 else (z, fx, b)
        comp = components_fast(la, w_rest, v)
        pin_pts = {tuple(ap) for aps in pin_list for ap in aps}
        for q in (A, B):
            if q not in comp and q in pin_pts:
                comp[q] = q
                st["pin_end"] += 1
        ends = [q for q in (A, B) if q in comp]
        if not ends:
            w2, v2 = w_rest, v
            st["spur"] += 1
        elif len(ends) == 2 and comp[A] == comp[B]:
            w2, v2 = w_rest, v
        else:
            if len(ends) == 1:
                st["spur_end"] += 1
            A0, B0 = A, B
            A = ends[0]
            x0 = (A0[1] + B0[1]) // 2
            y0 = (A0[2] + B0[2]) // 2
            half = max(abs(A0[1] - B0[1]), abs(A0[2] - B0[2])) // 2 + W
            inwin = lambda q: abs(q[1] - x0) <= half and abs(q[2] - y0) <= half
            groups = defaultdict(set)
            for q, r in comp.items():
                groups[r].add(q)
            tree = {q for q in groups[comp[A]] if inwin(q)}
            others = [g for r, g in groups.items() if r != comp[A] and any(inwin(q) for q in g)]
            others.sort(key=lambda g: min(abs(q[1] - A[1]) + abs(q[2] - A[2]) for q in g))
            w2, v2 = w_rest, set(v)
            ok_all = True
            for g in others:
                tg = {q for q in g if inwin(q)}
                if not tree or not tg:
                    ok_all = False
                    break
                _, path = search(la, list(tree), tg, x0, y0, half, 1.0, 0.1)
                if path is None:
                    ok_all = False
                    break
                pw, pv = Rp.path_to_wires_vias(la, path)
                w2, v2 = Rp.merge_wires(w2, pw), v2 | pv
                tree |= set(path) | tg
            if not ok_all:
                apply(la, w, v, +1)
                st["no_path"] += 1
                continue
        if not net_valid_fast(la, w2, v2, pin_list):
            apply(la, w, v, +1)
            st["reject_invalid"] += 1
            continue
        new = marginal(la, w2, v2)
        if new < old - 1e-9:
            apply(la, w2, v2, +1)
            cur[i] = (w2, v2)
            changed.add(i)
            st["rerouted"] += 1
        else:
            apply(la, w, v, +1)
            st["kept"] += 1
    undo = {}
    for i in sorted(changed):
        undo[i] = view.segs(i)
        S.new[i] = emit_tuples(la, *cur[i])
        S._dirty.add(int(S.tid[i]))
    if stats is not None:
        stats.update(st)
        stats["nets_changed"] = len(changed)
    return undo
