#!/usr/bin/env python3
"""s4_escape.py: item 4c. repair3d_astar.apply_fast / marginal_fast (used by the pin escape
escape3d.escape through escape3d_fast, and by the overflow stage's reroute) replaced by the exact C++ twins of
tools/speed4/s4_esc.cc. The Python side only flattens the net's wires dict and vias set in their iteration order.
install(check=False): patches escape3d.apply / .marginal and repair3d_astar.apply_fast / .marginal_fast; with check=True
every marginal is also computed by the original and compared (exact); counts in STATS."""
import ctypes, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = {"marginal": 0, "apply": 0, "check": 0, "check_bad": 0, "fallback": 0}
_L = None
_I, _V = ctypes.c_int64, ctypes.c_void_p


def _lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(os.path.join(HERE, "libs4_esc." + ("dylib" if sys.platform == "darwin" else "so")))
        L.s4_apply.argtypes = [_V, _I, _I, _I, _I, _I, _I, _V, _I, _V, _V, _V, _V, _I, _V, _V, _V, _I]
        L.s4_marginal.argtypes = [_V, _V, _V, _I, _I, _I, _I, _I, _I, _V, _I, _I, _V, _V, _V, _V, _I, _V, _V, _V]
        L.s4_marginal.restype = ctypes.c_double
        _L = L
    return _L


_LA = {}


def _la(la):
    k = id(la)
    v = _LA.get(k)
    if v is None or v[0] is not la.d:
        d, c, u = la.d, la.c, la.u
        ok = d.dtype == np.float64 and c.dtype == np.float64 and d.strides == c.strides and u.flags.c_contiguous and u.dtype == np.float64
        st = tuple(s // 8 for s in d.strides)
        v = (d, ok, st, np.ascontiguousarray(np.asarray(la.dirs, np.int8)))
        _LA[k] = v
    return v


def _flat(wires, vias):
    iv = [(z, f, a, b) for (z, f), ivs in wires.items() for (a, b) in ivs]
    A = np.array(iv, np.int64).reshape(-1, 4) if iv else np.zeros((0, 4), np.int64)
    vl = list(vias)
    B = np.array(vl, np.int64).reshape(-1, 3) if vl else np.zeros((0, 3), np.int64)
    A = [np.ascontiguousarray(A[:, k]) for k in range(4)]
    B = [np.ascontiguousarray(B[:, k]) for k in range(3)]
    return A, B


def install(check=False):
    import escape3d as E
    import repair3d_astar as RA
    if getattr(RA.apply_fast, "_s4", False):
        return
    oa, om = RA.apply_fast, RA.marginal_fast
    L = _lib()

    def apply_s4(la, wires, vias, sign, _oa=oa):
        d, ok, st, dirs = _la(la)
        if not ok:
            STATS["fallback"] += 1
            return _oa(la, wires, vias, sign)
        A, B = _flat(wires, vias)
        STATS["apply"] += 1
        L.s4_apply(d.ctypes.data, st[0], st[1], st[2], la.X, la.Y, la.L, dirs.ctypes.data, len(A[0]),
                   A[0].ctypes.data, A[1].ctypes.data, A[2].ctypes.data, A[3].ctypes.data, len(B[0]),
                   B[0].ctypes.data, B[1].ctypes.data, B[2].ctypes.data, int(sign))

    def marginal_s4(la, wires, vias, _om=om):
        d, ok, st, dirs = _la(la)
        if not ok:
            STATS["fallback"] += 1
            return _om(la, wires, vias)
        A, B = _flat(wires, vias)
        STATS["marginal"] += 1
        r = L.s4_marginal(d.ctypes.data, la.c.ctypes.data, la.u.ctypes.data, st[0], st[1], st[2], la.X, la.Y, la.L, dirs.ctypes.data,
                          len(wires), len(A[0]), A[0].ctypes.data, A[1].ctypes.data, A[2].ctypes.data, A[3].ctypes.data, len(B[0]),
                          B[0].ctypes.data, B[1].ctypes.data, B[2].ctypes.data)
        if check:
            ref = _om(la, wires, vias)
            STATS["check"] += 1
            if ref != r:
                STATS["check_bad"] += 1
                print(f"[s4_esc] marginal MISMATCH {ref!r} vs {r!r}", flush=True)
        return r
    apply_s4._s4 = marginal_s4._s4 = True
    RA.apply_fast, RA.marginal_fast = apply_s4, marginal_s4
    if E.apply is oa:
        E.apply = apply_s4
    if E.marginal is om:
        E.marginal = marginal_s4
