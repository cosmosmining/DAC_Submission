"""fast_norm.py: fastmode4.State4.timer_array (norm_timing=True) with the changed nets
normalised by the C++ normaliser of tools/c1hfast/norm_write.cc (nw_run / nw_fetch, the same seg_normalize.normalize_net
rules and output order; called with grid=1 so the output stays in GCells). Only nets whose segment list changed since the
previous call (identity of the list in S.new, as State4._norm's cache) are re-normalised, in one C++ call.
timer_array(S) returns the same [S,7] int64 array as State4.timer_array (checked with --check in chain_f.py).
"""
import ctypes, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lib = ctypes.CDLL(os.path.join(_ROOT, "tools", "c1hfast", "libnorm_write." + ("dylib" if sys.platform == "darwin" else "so")))
_P = ctypes.c_void_p
_lib.nw_run.argtypes = [ctypes.c_long] + [_P] * 6 + [ctypes.c_int]; _lib.nw_run.restype = ctypes.c_long
_lib.nw_fetch.argtypes = [_P] * 7; _lib.nw_fetch.restype = None


def _pin_range(S, i):
    from timing_promote import _key
    nm = S.nets.names[i] if hasattr(S.nets, "names") else S.nets[i][0]
    j = S.tidx_key.get(_key(nm))
    if j is not None and S.pin_end[j] > S.pin_start[j]:
        return int(S.ap_start[S.pin_start[j]]), int(S.ap_start[S.pin_end[j] - 1] + S.D.ap_cnt[S.pin_end[j] - 1])
    return 0, 0


def _normalise(S, idx):
    """C++ normalize_net of the nets idx (current S.new segments) -> {i: [m,7] int64 (i, x1, y1, z1, x2, y2, z2)}"""
    if not hasattr(S, "_f3x_ap"):
        S._f3x_ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        S._f3x_pr = {}
    pr = S._f3x_pr
    nn = len(idx)
    cnt = np.zeros(nn, np.int64); pa0 = np.zeros(nn, np.int64); pa1 = np.zeros(nn, np.int64)
    flat = []
    for k, i in enumerate(idx):
        sg = S.new[i]
        cnt[k] = len(sg); flat.extend(sg)
        r = pr.get(i)
        if r is None:
            r = pr[i] = _pin_range(S, i)
        pa0[k], pa1[k] = r
    start = np.zeros(nn, np.int64)
    if nn > 1:
        start[1:] = np.cumsum(cnt)[:-1]
    segs = np.ascontiguousarray(np.array(flat, np.int32).reshape(-1, 6)) if flat else np.zeros((1, 6), np.int32)
    tot = _lib.nw_run(nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data, pa0.ctypes.data, pa1.ctypes.data,
                      S._f3x_ap.ctypes.data, 1)
    oc = [np.empty(max(tot, 1), np.int32) for _ in range(6)]; oo = np.zeros(nn + 1, np.uint64)
    _lib.nw_fetch(*(c.ctypes.data for c in oc), oo.ctypes.data)
    blk = np.empty((tot, 7), np.int64)
    for k in range(6):
        blk[:, k + 1] = oc[k][:tot]
    oo = oo.astype(np.int64)
    out = {}
    for k, i in enumerate(idx):
        b = blk[oo[k]:oo[k + 1]]
        b[:, 0] = i
        out[i] = b
    return out


def timer_array(S):
    """== fastmode4.State4.timer_array with norm_timing (changed nets normalised)."""
    arr = S.base_arr
    if S.new:
        cache = S.__dict__.setdefault("_f3x_nc", {})
        stale = [i for i, sg in S.new.items() if (cache.get(i) or (None,))[0] is not sg]
        if stale:
            for i, b in _normalise(S, stale).items():
                cache[i] = (S.new[i], b)
        ch = np.fromiter(S.new.keys(), np.int64)
        keep = ~np.isin(arr[:, 0], ch)
        parts = [cache[i][1] for i in S.new]
        ext = np.concatenate(parts, 0) if parts else np.zeros((0, 7), np.int64)
        arr = np.concatenate([arr[keep], ext], 0)
    out = arr.copy()
    out[:, 0] = S.tid[arr[:, 0]]
    return out[out[:, 0] >= 0]
