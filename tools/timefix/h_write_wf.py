#!/usr/bin/env python3
"""h_write_wf.py: tools/fast3x/h_write.py with the library tools/timefix/libh_iowrite_wf.so
(tools/timefix/io_write_wf.cc, same bytes); nothing else changed. Original docstring:
h_write.py: chain_g.write_cpp (normalize_net of every net + the text writer) done by
tools/engine/io_write.cc (built here as libh_iowrite.dylib with clang++ -O2 -std=c++17 -pthread, no fast-math): normalise
and format with a hand-written integer formatter over nets in N threads (contiguous net ranges), one write of the
concatenated buffers. The per-net pin ranges (pa0/pa1) are computed vectorised once per state and kept.
Same bytes as chain_g.write_cpp (cmp in the chain's --h_check mode).
"""
import ctypes, os, sys, time
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_L = None


def _lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(str(HERE / ("libh_iowrite_wf.dylib" if sys.platform == "darwin" else "libh_iowrite_wf.so")))
        P = ctypes.c_void_p
        L.iow_write.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_long] + [P] * 6 + [ctypes.c_int, ctypes.c_int]
        L.iow_write.restype = ctypes.c_long
        L.iow_times.argtypes = [P]
        _L = L
    return _L


def _pin_ranges(S, names):
    """== the per-net loop of chain_g.write_cpp (same j per net, same condition, same values), vectorised"""
    from timing_promote import _key
    nn = len(names)
    tk = S.tidx_key
    c = S.__dict__.get("_h_nm")                     # chain_h attach_h keeps the _key of every name
    if c is not None and c[0] == nn and c[1] == list(names):
        jj = np.fromiter((tk.get(k, -1) for k in c[4]), np.int64, nn)
    else:
        jj = np.fromiter((tk.get(_key(nm), -1) for nm in names), np.int64, nn)
    ok = jj >= 0; jv = np.where(ok, jj, 0)
    ps_ = np.asarray(S.pin_start, np.int64)[jv]; pe_ = np.asarray(S.pin_end, np.int64)[jv]
    ok &= pe_ > ps_
    aps_ = np.asarray(S.ap_start, np.int64); ac_ = np.asarray(S.D.ap_cnt, np.int64)
    pa0 = np.zeros(nn, np.int64); pa1 = np.zeros(nn, np.int64)
    pa0[ok] = aps_[ps_[ok]]; pa1[ok] = aps_[pe_[ok] - 1] + ac_[pe_[ok] - 1]
    return pa0, pa1


def write_par(S, rt, noff, cols, arr6, out, W, nthreads=8):
    t = time.perf_counter()
    L = _lib()
    nn = len(rt.net_names)
    off = noff.astype(np.int64)
    start = off[:-1].copy(); cnt = np.diff(off).copy()
    segs = arr6.astype(np.int32)
    if S.new:
        ex = []; pos = len(segs)
        for i, sg in S.new.items():
            start[i] = pos; cnt[i] = len(sg); pos += len(sg); ex.extend(sg)
        segs = np.concatenate([segs, np.array(ex, np.int32).reshape(-1, 6)], 0)
    segs = np.ascontiguousarray(segs)
    c = S.__dict__.get("_h_wcache")
    if c is None or c[0] is not rt.net_names:
        pa0, pa1 = _pin_ranges(S, rt.net_names)
        ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        blob = ("\n".join(rt.net_names) + "\n").encode()
        c = S._h_wcache = (rt.net_names, pa0, pa1, ap, blob)
    _, pa0, pa1, ap, blob = c
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W["w_prep_s"] = time.perf_counter() - t; t = time.perf_counter()
    rc = L.iow_write(os.fsencode(str(out)), blob, nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data,
                     pa0.ctypes.data, pa1.ctypes.data, ap.ctypes.data, 4200, int(nthreads))
    assert rc >= 0, rc
    W["w_norm_io_s"] = time.perf_counter() - t
    tt = (ctypes.c_double * 2)(); L.iow_times(tt)
    W["w_norm_fmt_s"], W["w_file_s"] = tt[0], tt[1]
