#!/usr/bin/env python3
"""r5bin.py (router5, 2026-11-06, NEW): Python wrapper of libr5bin (r5bin.cc), the reader of the compact binary route
R5B1 that InstantGR_r5 writes with -bin <file> next to its text route.

    read_route_bin(path, threads=8) -> route_cpp.Route with the same net_id, x0, y0, z0, x1, y1, z1 (int32), offsets
    (uint64) and net_names (tuple of str) as route_cpp.read_route(<contest route converted from the text route of the
    same run by route_cpp.convert_instantgr>). The Route has no C handle (_handle None): Route.write is not available on
    it (the chain writes with its own writer); everything the chain reads from the Route is there.

    python r5bin.py check <text .PR_output> <.r5b> <tmp dir> [reps]
        compares the arrays with route_cpp.read_route (and tools/fast3x/h_route_read when it loads) on that pair and
        times: convert (text -> contest route), read_route, h_route_read, read_route_bin (min of reps); prints one line.
Build the library: g++ -O3 -std=c++17 -fPIC -shared -pthread r5bin.cc -o libr5bin.so
"""
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_LIB = HERE / ("libr5bin.dylib" if os.uname().sysname == "Darwin" else "libr5bin.so")
_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(_LIB))
        lib.r5b_read.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.r5b_read.restype = ctypes.c_void_p
        lib.r5b_free.argtypes = [ctypes.c_void_p]
        lib.r5b_error.restype = ctypes.c_char_p
        for n in ("r5b_num_nets", "r5b_num_segments", "r5b_names_len"):
            getattr(lib, n).argtypes = [ctypes.c_void_p]
            getattr(lib, n).restype = ctypes.c_uint64
        lib.r5b_offsets.argtypes = [ctypes.c_void_p]
        lib.r5b_offsets.restype = ctypes.POINTER(ctypes.c_uint64)
        lib.r5b_col.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.r5b_col.restype = ctypes.POINTER(ctypes.c_int32)
        lib.r5b_names.argtypes = [ctypes.c_void_p]
        lib.r5b_names.restype = ctypes.c_void_p
        _lib = lib
    return _lib


def _route_cls():
    sys.path.insert(0, str(ROOT / "tools" / "fast_io" / "route_cpp"))
    try:
        from route_cpp import Route
        return Route
    except Exception:
        return None
    finally:
        sys.path.pop(0)


def read_route_bin(path, threads=8):
    lib = _load()
    h = lib.r5b_read(os.fsencode(str(path)), int(threads))
    if not h:
        raise RuntimeError(lib.r5b_error().decode())
    try:
        nn = int(lib.r5b_num_nets(h))
        ns = int(lib.r5b_num_segments(h))
        cols = []
        for c in range(7):
            cols.append(np.ctypeslib.as_array(lib.r5b_col(h, c), shape=(ns,)).copy() if ns else np.empty(0, np.int32))
        offsets = np.ctypeslib.as_array(lib.r5b_offsets(h), shape=(nn + 1,)).copy()
        blen = int(lib.r5b_names_len(h))
        blob = ctypes.string_at(lib.r5b_names(h), blen)
        names = tuple(blob.decode("utf-8").split("\n")) if nn else ()
    finally:
        lib.r5b_free(h)
    arrays = dict(zip(("net_id", "x0", "y0", "z0", "x1", "y1", "z1"), cols))
    Route = _route_cls()
    if Route is None:
        from types import SimpleNamespace
        return SimpleNamespace(**arrays, offsets=offsets, net_names=names)
    return Route(**arrays, offsets=offsets, net_names=names, _handle=None, _lib=None)


KEYS = ("net_id", "x0", "y0", "z0", "x1", "y1", "z1", "offsets")


def same(a, b):
    return all(getattr(a, k).dtype == getattr(b, k).dtype and np.array_equal(getattr(a, k), getattr(b, k)) for k in KEYS) \
        and tuple(a.net_names) == tuple(b.net_names)


def _check(text, binf, tmpdir, reps):
    sys.path.insert(0, str(ROOT / "tools" / "fast_io" / "route_cpp"))
    import route_cpp as RC
    hrr = None
    try:
        sys.path.insert(0, str(ROOT / "tools" / "fast3x"))
        import h_route_read as hrr
    except Exception as e:
        print("h_route_read not loaded:", e)
    contest = Path(tmpdir) / (Path(text).name + ".contest.route")
    tc = []
    for _ in range(reps):
        t = time.perf_counter(); RC.convert_instantgr(text, contest); tc.append(time.perf_counter() - t)
    tr, th, tb = [], [], []
    for _ in range(reps):
        t = time.perf_counter(); r_txt = RC.read_route(contest); tr.append(time.perf_counter() - t)
        if hrr is not None:
            t = time.perf_counter(); r_h = hrr.read_route(str(contest), 8, fallback=RC.read_route); th.append(time.perf_counter() - t)
        t = time.perf_counter(); r_bin = read_route_bin(binf, 8); tb.append(time.perf_counter() - t)
    eq = same(r_bin, r_txt)
    eqh = same(r_h, r_txt) if hrr is not None else None
    print(f"CHECK {Path(text).name} equal_bin_vs_read_route {eq} equal_h_vs_read_route {eqh} nets {len(r_txt.net_names)} "
          f"segs {r_txt.x0.size} convert_s {min(tc):.3f} read_route_s {min(tr):.3f} "
          f"h_route_read_s {min(th) if th else float('nan'):.3f} bin_s {min(tb):.3f} bin_bytes {os.path.getsize(binf)} "
          f"text_bytes {os.path.getsize(text)} contest_bytes {os.path.getsize(contest)}")
    os.unlink(contest)
    return eq


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "check":
        ok = _check(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
        sys.exit(0 if ok else 1)
    print(__doc__)
