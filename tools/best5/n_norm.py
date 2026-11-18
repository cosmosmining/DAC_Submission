"""n_norm.py: tools/late5/l5.py normalise with the C++ normaliser on threads (n_norm.cc): the same output dict of [m,7] int64 blocks, same rows."""
import ctypes, os, sys
from itertools import chain
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_L = ctypes.CDLL(os.path.join(HERE, "libn_norm." + ("dylib" if sys.platform == "darwin" else "so")))
_P = ctypes.c_void_p
_L.n_nw_run.argtypes = [ctypes.c_long] + [_P] * 6 + [ctypes.c_int] * 2; _L.n_nw_run.restype = ctypes.c_long
_L.n_nw_fetch.argtypes = [_P, _P, _P, ctypes.c_int]
THREADS = int(os.environ.get("MID5_THREADS", "8"))


def normalise(S, idx):
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
    tot = _L.n_nw_run(nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data, pa0.ctypes.data, pa1.ctypes.data, S._f3x_ap.ctypes.data, 1, THREADS)
    blk = np.empty((tot, 7), np.int64); oo = np.zeros(nn + 1, np.int64)
    ids = np.asarray(idx, np.int64)
    _L.n_nw_fetch(ids.ctypes.data, blk.ctypes.data, oo.ctypes.data, THREADS)
    ol = oo.tolist()
    return {i: blk[ol[k]:ol[k + 1]] for k, i in enumerate(idx)}
