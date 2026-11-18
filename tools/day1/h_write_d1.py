#!/usr/bin/env python3
"""h_write_d1.py: tools/timefix/h_write_wf.py (= tools/fast3x/h_write.py + the writefast
library) with the write prep done from ONE join of the route's names. Same bytes.

Cause of the +2-5 s write prep under --fp2 1 (an internal note): chain_tf's attach_h gets rt.net_names itself
under fp2 (a list copy otherwise), and load8_lib_f2.names_tables keeps that object in the cache (c[1] = c[4] = names). The
old _pin_ranges then (a) built list(names) only to compare it with c[1] (never equal: c[1] is not a list), and (b) fell
back to `tk.get(_key(nm), -1) for nm in names` in a Python generator; write_par then joined the names a second time for
the blob. Here: s = "\\n".join(names) once; keys = s.replace("\\\\", "").split("\\n") (== [_key(nm) for nm in names]
whenever no name holds a newline; otherwise the per-name loop is used); jj from map(dict.get) at C speed; blob = (s +
"\\n").encode(). jj, pa0, pa1, ap, blob are the old values exactly, so the C++ writer writes the same file.
WF (set by the chain): True = tools/timefix/libh_iowrite_wf.so (--writefast 1), False = tools/fast3x/libh_iowrite.so.
Extra json fields (write stage): w_prep_segs_s, w_prep_names_s, w_prep_pr_s, w_prep_ap_s (sub-timers of w_prep_s)."""
import ctypes, itertools, os, sys, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WF = True
MODE = 1        # chain: --wprep N. 1 = the name-join fix only. 2 = also big5's write_par5 prep (vectorised flatten of the
                # changed nets, join of big5's cache thread): --writefast imports this module instead of h_write, so big5's
                # replacement of h_write.write_par (tools/big5/big5_lib.py write_par5) never reached the writefast path.
_L = {}


def _lib():
    k = bool(WF)
    if k not in _L:
        so = "dylib" if sys.platform == "darwin" else "so"
        p = (ROOT / "tools" / "timefix" / f"libh_iowrite_wf.{so}") if k else (ROOT / "tools" / "fast3x" / f"libh_iowrite.{so}")
        L = ctypes.CDLL(str(p))
        P = ctypes.c_void_p
        L.iow_write.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_long] + [P] * 6 + [ctypes.c_int, ctypes.c_int]
        L.iow_write.restype = ctypes.c_long
        L.iow_times.argtypes = [P]
        _L[k] = L
    return _L[k]


def _keys(names, nn):
    """(joined names, [_key(nm) for nm in names]) from one join"""
    from timing_promote import _key
    s = "\n".join(names)
    ks = s.replace("\\", "").split("\n") if nn else []
    if len(ks) != nn:                                   # a name with a newline: the per-name definition
        ks = [_key(nm) for nm in names]
    return s, ks


def _pin_ranges(S, ks, nn):
    """== h_write._pin_ranges (same j per net, same condition, same values) with the keys given"""
    tk = S.tidx_key
    jj = np.fromiter(map(tk.get, ks, itertools.repeat(-1, nn)), np.int64, nn)
    ok = jj >= 0; jv = np.where(ok, jj, 0)
    ps_ = np.asarray(S.pin_start, np.int64)[jv]; pe_ = np.asarray(S.pin_end, np.int64)[jv]
    ok &= pe_ > ps_
    aps_ = np.asarray(S.ap_start, np.int64); ac_ = np.asarray(S.D.ap_cnt, np.int64)
    pa0 = np.zeros(nn, np.int64); pa1 = np.zeros(nn, np.int64)
    pa0[ok] = aps_[ps_[ok]]; pa1[ok] = aps_[pe_[ok] - 1] + ac_[pe_[ok] - 1]
    return pa0, pa1


def write_par(S, rt, noff, cols, arr6, out, W, nthreads=8):
    t = time.perf_counter(); t0 = t
    L = _lib()
    nn = len(rt.net_names)
    off = noff.astype(np.int64)
    start = off[:-1].copy(); cnt = np.diff(off).copy()
    segs = arr6.astype(np.int32)
    if S.new and MODE >= 2:                             # == big5 write_par5: same start / cnt / rows, dict order kept
        from itertools import chain
        ids = np.fromiter(S.new.keys(), np.int64, len(S.new))
        lists = list(S.new.values())
        ln = np.fromiter((len(x) for x in lists), np.int64, len(lists))
        st_ = np.zeros(len(ln), np.int64)
        if len(ln) > 1:
            st_[1:] = np.cumsum(ln)[:-1]
        start[ids] = len(segs) + st_; cnt[ids] = ln
        m = int(ln.sum())
        ex = np.fromiter(chain.from_iterable(chain.from_iterable(lists)), np.int32, 6 * m).reshape(m, 6) if m else np.zeros((0, 6), np.int32)
        segs = np.concatenate([segs, ex], 0)
    elif S.new:
        ex = []; pos = len(segs)
        for i, sg in S.new.items():
            start[i] = pos; cnt[i] = len(sg); pos += len(sg); ex.extend(sg)
        segs = np.concatenate([segs, np.array(ex, np.int32).reshape(-1, 6)], 0)
    segs = np.ascontiguousarray(segs)
    W["w_prep_segs_s"] = time.perf_counter() - t
    if MODE >= 2 and "big5_lib" in sys.modules:         # the cache big5 computes on a thread during the overflow stage
        th = sys.modules["big5_lib"].EARLY.pop("wc", None)
        if th is not None:
            t1 = time.perf_counter(); th.join(); W["w_prep_join_s"] = time.perf_counter() - t1
    c = S.__dict__.get("_h_wcache")
    if c is None or c[0] is not rt.net_names:
        t = time.perf_counter()
        s_, ks = _keys(rt.net_names, nn)
        W["w_prep_names_s"] = time.perf_counter() - t; t = time.perf_counter()
        pa0, pa1 = _pin_ranges(S, ks, nn)
        W["w_prep_pr_s"] = time.perf_counter() - t; t = time.perf_counter()
        ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        blob = (s_ + "\n").encode()
        W["w_prep_ap_s"] = time.perf_counter() - t
        c = S._h_wcache = (rt.net_names, pa0, pa1, ap, blob)
        del s_, ks
    _, pa0, pa1, ap, blob = c
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W["w_prep_s"] = time.perf_counter() - t0; t = time.perf_counter()
    rc = L.iow_write(os.fsencode(str(out)), blob, nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data,
                     pa0.ctypes.data, pa1.ctypes.data, ap.ctypes.data, 4200, int(nthreads))
    assert rc >= 0, rc
    W["w_norm_io_s"] = time.perf_counter() - t
    tt = (ctypes.c_double * 2)(); L.iow_times(tt)
    W["w_norm_fmt_s"], W["w_file_s"] = tt[0], tt[1]
    W["w_lib"] = "wf" if WF else "fast3x"
