#!/usr/bin/env python3
"""kern8.py: ctypes front of libl8_kern.so (tools/best11/l8_kern.cc). Every function == the numpy line named beside it (tools/best11/test_kern8.py)."""
import ctypes as c, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_L = c.CDLL(os.path.join(HERE, "libl8_kern." + ("dylib" if sys.platform == "darwin" else "so")))
V, I, N = c.c_void_p, c.c_int64, c.c_int
_L.l8_argsort.argtypes = [I, V, V, N]
_L.l8_take8.argtypes = [I, V, V, V, N]
_L.l8_take1.argtypes = [I, V, V, V, N]
_L.l8_bounds.argtypes = [I, V, I, V]
_L.l8_levelize.argtypes = [I, I, V, V, V, V, N]; _L.l8_levelize.restype = N
_L.l8_ce_count.argtypes = [I, V, V, V, V, V, N]; _L.l8_ce_count.restype = I
_L.l8_ce_fill.argtypes = [I] + [V] * 10 + [V, I, I, I] + [V] * 11 + [N]; _L.l8_ce_fill.restype = I
_L.l8_ce_groups.argtypes = [I, V, V, V, V, I, V, V, V, N]
_L.l8_pin_maps.argtypes = [I, V, V, V, V, I, I, V, V, N]
_L.l8_seg32.argtypes = [I, V, V] + [V] * 6 + [I, V, N]; _L.l8_seg32.restype = I
_L.l8_seg64.argtypes = [I, V, V] + [V] * 6 + [I, V, N]; _L.l8_seg64.restype = I
NT = [int(os.environ.get("OMP_NUM_THREADS", "8"))]
BIG = 100000                     # below this the numpy call


def _p(a):
    return a.ctypes.data


def _i8(a):
    return np.ascontiguousarray(a, np.int64)


def argsort(key):
    """== np.argsort(key, kind="stable")"""
    if len(key) < BIG or key.dtype.kind not in "iu" or key.ndim != 1:
        return np.argsort(key, kind="stable")
    k = _i8(key)
    out = np.empty(len(k), np.int64)
    _L.l8_argsort(len(k), _p(k), _p(out), NT[0])
    return out


def take(a, idx):
    """== a[idx] (idx: int64, in bounds, non-negative or checked by the caller's data)"""
    if len(idx) < BIG or a.ndim != 1 or idx.dtype != np.int64 or a.itemsize not in (1, 8) or not a.flags.c_contiguous or not idx.flags.c_contiguous:
        return a[idx]
    out = np.empty(len(idx), a.dtype)
    (_L.l8_take8 if a.itemsize == 8 else _L.l8_take1)(len(idx), _p(idx), _p(a), _p(out), NT[0])
    return out


def bounds(a, m):
    """== np.searchsorted(a, np.arange(m)) for sorted int64 a"""
    a = _i8(a)
    out = np.empty(m, np.int64)
    _L.l8_bounds(len(a), _p(a), m, _p(out))
    return out


def unique_small(a, m):
    """== np.unique(a) for int64 values in [0, m)"""
    mark = np.zeros(m, bool)
    mark[a] = True
    return np.flatnonzero(mark)


def levelize(P, efr, eto):
    """== tools/pre5/pre5_kern.levelize"""
    efr = _i8(efr); eto = _i8(eto)
    lev = np.zeros(P, np.int64); left = np.zeros(P, np.uint8)
    _L.l8_levelize(P, len(efr), _p(efr), _p(eto), _p(lev), _p(left), NT[0])
    return lev, left.astype(bool)


def ce_edges(ai, fr, to, kind, sense, tcr, tcf, trt, tft, lib, lev, nlev, P):
    """-> dict of the _graph arrays ce_arc ce_irf ce_orf ce_td ce_ts ce_from ce_to ce_lib ce_lo ce_grp g_start g_lo g_to g_orf, or None (composite key too wide: caller's numpy path)"""
    ai = _i8(ai)
    A = [_i8(x) for x in (fr, to, kind, sense, tcr, tcf, trt, tft, lib)]
    fr, to, kind, sense, tcr, tcf, trt, tft, lib = A
    lev = _i8(lev)
    nt = NT[0]
    m = _L.l8_ce_count(len(ai), _p(ai), _p(kind), _p(sense), _p(tcr), _p(tcf), nt)
    o = {k: np.empty(m, np.int64) for k in ("ce_arc", "ce_irf", "ce_orf", "ce_td", "ce_ts", "ce_from", "ce_to", "ce_lib", "ce_grp")}
    o["ce_lo"] = np.empty(nlev + 1, np.int64)
    gs = np.empty(m, np.int64)
    ng = _L.l8_ce_fill(len(ai), _p(ai), _p(fr), _p(to), _p(kind), _p(sense), _p(tcr), _p(tcf), _p(trt), _p(tft), _p(lib), _p(lev), nlev, P, m,
                       _p(o["ce_arc"]), _p(o["ce_irf"]), _p(o["ce_orf"]), _p(o["ce_td"]), _p(o["ce_ts"]), _p(o["ce_from"]), _p(o["ce_to"]), _p(o["ce_lib"]), _p(o["ce_lo"]), _p(o["ce_grp"]), _p(gs), nt)
    if ng < 0:
        return None
    o["g_start"] = gs[:ng].copy(); del gs
    o["g_to"] = np.empty(ng, np.int64); o["g_orf"] = np.empty(ng, np.int64); o["g_lo"] = np.empty(nlev + 1, np.int64)
    _L.l8_ce_groups(ng, _p(o["g_start"]), _p(o["ce_to"]), _p(o["ce_orf"]), _p(lev), nlev, _p(o["g_to"]), _p(o["g_orf"]), _p(o["g_lo"]), nt)
    return o


def pin_maps(nn, pin_off, ap_off, aps, canon, X, Y):
    """-> (pc float64 [X, Y], cnt int64 [nn])"""
    pin_off = _i8(pin_off); ap_off = _i8(ap_off); aps = np.ascontiguousarray(aps, np.int32)
    pc = np.empty(int(X) * int(Y), np.float64); cnt = np.empty(nn, np.int64)
    cn = None if canon is None else np.ascontiguousarray(canon, np.uint8)
    _L.l8_pin_maps(nn, _p(pin_off), _p(ap_off), _p(aps), None if cn is None else _p(cn), int(X), int(Y), _p(pc), _p(cnt), NT[0])
    return pc.reshape(int(X), int(Y)), cnt


def seg(de, offsets, cols, grid):
    """the column work of csta.seg_from_route: int64 [kept, 7]; cols = (x0, y0, z0, x1, y1, z1), all int32 or all int64; None = other dtypes (caller's numpy path)"""
    dt = cols[0].dtype
    if any(x.dtype != dt for x in cols) or dt not in (np.int32, np.int64) or offsets.dtype != np.uint64:
        return None
    n = len(cols[0]); nn = len(de)
    de = _i8(de)
    out = np.empty((n, 7), np.int64)
    cols = [np.ascontiguousarray(x) for x in cols]
    offsets = np.ascontiguousarray(offsets)
    kept = (_L.l8_seg32 if dt == np.int32 else _L.l8_seg64)(nn, _p(de), _p(offsets), *[_p(x) for x in cols], int(grid), _p(out), NT[0])
    return out[:kept].copy() if kept != n else out
