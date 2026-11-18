#!/usr/bin/env python3
"""h_asm.py: chain_g.assemble with its per-net Python loop in C++ (tools/fast3x/h_asm.cc) and
the copy of the new nets' rows vectorised. Returns (noff uint64, cols 6 x int32, number of rebuilt nets) equal to
chain_g.assemble (np.array_equal + dtype in chain_h --h_check).
"""
import ctypes, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_L = None
_NODES = {}


def _lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(str(HERE / ("libh_asm.dylib" if sys.platform == "darwin" else "libh_asm.so")))
        L.ha_run.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 10; L.ha_run.restype = ctypes.c_int64
        L.ha_fetch.argtypes = [ctypes.c_void_p] * 2
        _L = L
    return _L


def _np64(x):
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.ascontiguousarray(x, np.int64)


def assemble(P, choice, rt, RE):
    C = RE.GCELL_DBU; h = C // 2
    rn = _np64(P["run_net"]); cand = _np64(P["cand"])
    R = len(rn)
    ch = np.asarray(choice).astype(np.int64)
    z = np.ascontiguousarray(cand[np.arange(R), ch]) if R else np.zeros(0, np.int64)
    key = (id(P), R)
    pre = _NODES.get(key)
    if pre is None:                                  # route independent of the choice: net order, runs per net, nodes
        if R:
            u, fi = np.unique(rn, return_index=True)
            order = u[np.argsort(fi, kind="stable")]
        else:
            order = np.zeros(0, np.int64)
        rank = np.empty(int(rn.max()) + 1 if R else 0, np.int64); rank[order] = np.arange(len(order))
        runs = np.argsort(rank[rn], kind="stable").astype(np.int64) if R else np.zeros(0, np.int64)
        roff = np.zeros(len(order) + 1, np.int64); roff[1:] = np.cumsum(np.bincount(rank[rn], minlength=len(order))) if R else 0
        flat = []; nofs = np.zeros(len(order) + 1, np.int64)
        NN = P["net_nodes"]
        for q, n in enumerate(order.tolist()):
            items = NN[n].items()
            for (x, y), span in items:
                flat.append((int(x), int(y), int(span[0]), int(span[1])))
            nofs[q + 1] = len(flat)
        nodes = np.ascontiguousarray(np.array(flat, np.int64).reshape(-1, 4)) if flat else np.zeros((1, 4), np.int64)
        dirs = np.ascontiguousarray(np.asarray(list(P["dirs"]), np.int64))
        nri = _np64(P["net_route_idx"])
        cols_r = [_np64(P[k]) for k in ("run_dir", "run_fixed", "run_lo", "run_hi")]
        pre = _NODES[key] = (order, runs, roff, nofs, nodes, dirs, nri, cols_r)
    order, runs, roff, nofs, nodes, dirs, nri, cols_r = pre
    L = _lib()
    nn_dp = len(order)
    tot_new = L.ha_run(nn_dp, roff.ctypes.data, runs.ctypes.data, z.ctypes.data, *(c.ctypes.data for c in cols_r),
                       dirs.ctypes.data, nofs.ctypes.data, nodes.ctypes.data)
    flatb = np.empty((max(tot_new, 1), 6), np.int64); cntb = np.empty(max(nn_dp, 1), np.int64)
    L.ha_fetch(flatb.ctypes.data, cntb.ctypes.data)
    startb = np.zeros(nn_dp, np.int64)
    if nn_dp > 1:
        startb[1:] = np.cumsum(cntb[:nn_dp])[:-1]
    # new: {route index: block} in processing order; a repeated route index keeps the last block (dict overwrite)
    new = {}
    for q, i in enumerate(nri[order].tolist()):
        new[int(i)] = q
    off = rt.offsets.astype(np.int64)
    nn = len(rt.net_names)
    cnt = np.diff(off).copy()
    ids = np.fromiter(new.keys(), np.int64, len(new)); qs = np.fromiter(new.values(), np.int64, len(new))
    cnt[ids] = cntb[qs]
    noff = np.zeros(nn + 1, dtype=np.uint64); noff[1:] = np.cumsum(cnt)
    tot = int(noff[-1])
    cols = [np.empty(tot, np.int32) for _ in range(6)]
    src = [rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1]
    keep = np.ones(nn, bool); keep[ids] = False
    kn = np.flatnonzero(keep & (cnt > 0))
    if len(kn):
        c = cnt[kn]
        s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn].astype(np.int64), c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        for k in range(6):
            cols[k][d0 + ar] = src[k][s0 + ar]
    m = cntb[qs]
    sel = m > 0
    if sel.any():
        c = m[sel]
        d0 = np.repeat(noff[ids[sel]].astype(np.int64), c); s0 = np.repeat(startb[qs[sel]], c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        rows = flatb[s0 + ar]
        dst = d0 + ar
        cols[0][dst] = rows[:, 0] * C + h; cols[1][dst] = rows[:, 1] * C + h; cols[2][dst] = rows[:, 2]
        cols[3][dst] = rows[:, 3] * C + h; cols[4][dst] = rows[:, 4] * C + h; cols[5][dst] = rows[:, 5]
    return noff, cols, len(new)
