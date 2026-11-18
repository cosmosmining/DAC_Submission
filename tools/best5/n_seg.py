"""n_seg.py: CS.seg_from_route with the row fill on threads (n_seg.cc); same array. Arrays that are not int32 -> the original."""
import ctypes, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_L = ctypes.CDLL(os.path.join(HERE, "libn_seg." + ("dylib" if sys.platform == "darwin" else "so")))
_P = ctypes.c_void_p
_L.n_seg_fill.argtypes = [ctypes.c_int64] + [_P] * 8 + [ctypes.c_int64, _P, ctypes.c_int, ctypes.c_int]; _L.n_seg_fill.restype = ctypes.c_int64
T = int(os.environ.get("MID5_THREADS", "8"))


def install(CS):
    orig = CS.seg_from_route
    if getattr(orig, "_n9", False):
        return

    def seg_from_route(cd, rt, grid, _o=orig):
        cols = [np.asarray(getattr(rt, k)) for k in ("x0", "y0", "z0", "x1", "y1", "z1")]
        if not all(c.dtype == np.int32 and c.flags.c_contiguous for c in cols):
            return _o(cd, rt, grid)
        if not hasattr(cd, "_nf_index"):
            cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
        idx = cd._nf_index
        nid = np.fromiter((idx.get(n, -1) for n in rt.net_names), np.int64, len(rt.net_names))
        de = np.ascontiguousarray(np.where(nid >= 0, cd.nf_net[np.maximum(nid, 0)], -1), np.int64)
        off = np.ascontiguousarray(rt.offsets, np.int64)
        a = [c.ctypes.data for c in cols]
        n = _L.n_seg_fill(len(de), off.ctypes.data, de.ctypes.data, *a, int(grid), None, T, 1)
        seg = np.empty((n, 7), np.int64)
        _L.n_seg_fill(len(de), off.ctypes.data, de.ctypes.data, *a, int(grid), seg.ctypes.data, T, 0)
        if os.environ.get("N_SEG_CHECK") == "1":
            ref = _o(cd, rt, grid)
            print("[n_seg check] equal=%s rows %d" % (np.array_equal(ref, seg), len(seg)), flush=True)
        return seg
    seg_from_route._n9 = True
    CS.seg_from_route = seg_from_route
