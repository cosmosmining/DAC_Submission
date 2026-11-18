#!/usr/bin/env python3
"""h_dem.py: demand maps for chain_h items 3a and 5.
fastmode.build_demand_fast(la, seg) adds per-net contributions (wire runs merged per net/layer/track, unit vias unique per
net, coverage tested against the same net's wires) to la.d; every contribution is a small integer (+-2, +1, +2), so the map
is a sum of integers and the float64 result does not depend on the order of the additions. Hence
  * build(la, seg): the same map from tools/fast3x/h_dem.cc (same rules, per net, C++), and
  * derive(...): map(new route) == map(old route) - map(old rows of the changed nets) + map(new rows of the changed nets)
are exact (np.array_equal against build_demand_fast in chain_h --h_check). If the C++ code meets an index outside the map
(numpy would wrap it) the Python function is used instead.
"""
import ctypes, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_L = None
STATS = {"cpp": 0, "py": 0}


def _lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(str(HERE / ("libh_dem.dylib" if sys.platform == "darwin" else "libh_dem.so")))
        L.hd_add.argtypes = [ctypes.c_void_p] + [ctypes.c_int64] * 6 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_double]
        L.hd_add.restype = ctypes.c_int
        _L = L
    return _L


class _Z:
    """the attributes build_demand_fast reads, with a zero map"""
    def __init__(self, la):
        self.L, self.X, self.Y, self.dirs = la.L, la.X, la.Y, la.dirs
        self.d = np.zeros_like(la.d)


def _add(la, d, seg, sign):
    """d += sign * build_demand_fast contribution of seg (C++); returns False if the C++ path refused (d unchanged? no: caller retries)"""
    assert d.dtype == np.float64 and d.shape == (la.L, la.X, la.Y) and all(x % 8 == 0 and x >= 0 for x in d.strides)
    seg = np.ascontiguousarray(seg, np.int64)
    dirs = np.ascontiguousarray(np.asarray(la.dirs, np.int64))
    st = [x // 8 for x in d.strides]
    return _lib().hd_add(d.ctypes.data, la.L, la.X, la.Y, st[0], st[1], st[2], dirs.ctypes.data, seg.ctypes.data, len(seg), float(sign)) == 0


def build(FM, la, seg):
    """== FM.build_demand_fast(la, seg) (adds to la.d)"""
    d0 = la.d.copy() if la.d.any() else None
    if _add(la, la.d, seg, 1.0):
        STATS["cpp"] += 1
        return
    STATS["py"] += 1
    if d0 is None:
        la.d[:] = 0
    else:
        la.d[...] = d0
    FM.build_demand_fast(la, seg)


def rows_of(a6, off, idx):
    """[m,7] (net, x1, y1, z1, x2, y2, z2) rows of nets idx (route order kept), net column = route index"""
    idx = np.asarray(idx, np.int64)
    off = np.asarray(off, np.int64)
    c = off[idx + 1] - off[idx]
    tot = int(c.sum())
    out = np.empty((tot, 7), np.int64)
    if tot:
        src = np.repeat(off[idx], c) + (np.arange(tot) - np.repeat(np.cumsum(c) - c, c))
        out[:, 0] = np.repeat(idx, c)
        out[:, 1:] = a6[src]
    return out


def derive(FM, la, D_old, a6_old, off_old, a6_new, off_new, idx):
    """map of the new route from the old route's map D_old (idx: nets whose rows may differ)"""
    r_old, r_new = rows_of(a6_old, off_old, idx), rows_of(a6_new, off_new, idx)
    d = D_old.copy()
    if _add(la, d, r_old, -1.0) and _add(la, d, r_new, 1.0):
        STATS["cpp"] += 1
        return d
    STATS["py"] += 1
    z1 = _Z(la); FM.build_demand_fast(z1, r_old)
    z2 = _Z(la); FM.build_demand_fast(z2, r_new)
    d = D_old - z1.d
    d += z2.d
    return d
