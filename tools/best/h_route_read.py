#!/usr/bin/env python3
"""h_route_read.py: fast contest-route reader (tools/fast3x/h_route_read.cc): whole file read
once, hand-written number parser, threads over net-block chunks. Returns an object with the fields chains read from
route_cpp.read_route (net_id, x0, y0, z0, x1, y1, z1 int32; offsets uint64; net_names tuple of str) with identical values
and dtypes. A file that is not in the canonical layout is handed to route_cpp.read_route unchanged (fallback).
  python3 h_route_read.py --test R1.route [R2.route ...]   compares every field with route_cpp.read_route (np.array_equal)
"""
import ctypes, os, sys, time
import numpy as np
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_LIB = None
STATS = {}


def _lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(str(HERE / ("libh_route_read.dylib" if sys.platform == "darwin" else "libh_route_read.so")))
        P = ctypes.c_void_p
        L.hrr_read.argtypes = [ctypes.c_char_p, ctypes.c_int, P, P]; L.hrr_read.restype = ctypes.c_int
        L.hrr_names_size.restype = ctypes.c_int64
        L.hrr_fetch.argtypes = [P] * 9
        _LIB = L
    return _LIB


class HRoute(SimpleNamespace):
    @property
    def n_segments(self): return int(self.x0.size)
    @property
    def n_nets(self): return len(self.net_names)


def read_route(path, nthreads=8, fallback=None):
    t = time.perf_counter()
    L = _lib()
    nn = ctypes.c_int64(0); ns = ctypes.c_int64(0)
    rc = L.hrr_read(os.fsencode(str(path)), int(nthreads), ctypes.addressof(nn), ctypes.addressof(ns))
    if rc != 0:
        STATS["fallback"] = rc
        if fallback is None:
            sys.path.insert(0, str(ROOT / "tools" / "fast_io" / "route_cpp"))
            from route_cpp import read_route as fallback
        return fallback(path)
    nn, ns = nn.value, ns.value
    cols = [np.empty(ns, np.int32) for _ in range(7)]
    off = np.empty(nn + 1, np.uint64)
    nb = int(L.hrr_names_size())
    blob = ctypes.create_string_buffer(max(nb, 1))
    L.hrr_fetch(*(c.ctypes.data for c in cols), off.ctypes.data, blob)
    names = tuple(blob.raw[:nb].decode("utf-8").split("\n")[:-1]) if nb else ()
    STATS["fallback"] = 0; STATS["s"] = time.perf_counter() - t
    return HRoute(net_id=cols[0], x0=cols[1], y0=cols[2], z0=cols[3], x1=cols[4], y1=cols[5], z1=cols[6], offsets=off, net_names=names)


def _test(paths):
    sys.path.insert(0, str(ROOT / "tools" / "fast_io" / "route_cpp"))
    from route_cpp import read_route as ref_read
    bad = 0
    for p in paths:
        t0 = time.perf_counter(); a = ref_read(p); t1 = time.perf_counter(); b = read_route(p); t2 = time.perf_counter()
        res = []
        for k in ("net_id", "x0", "y0", "z0", "x1", "y1", "z1", "offsets"):
            x, y = getattr(a, k), getattr(b, k)
            res.append(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y))
        res.append(tuple(a.net_names) == tuple(b.net_names) and all(type(s) is str for s in b.net_names))
        ok = all(res); bad += not ok
        print(f"{p}: nets {len(a.net_names)} segs {len(a.x0)} equal={ok} {res} fallback={STATS.get('fallback')} "
              f"route_cpp {t1 - t0:.3f}s h {t2 - t1:.3f}s", flush=True)
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] == ["--test"]:
        sys.exit(1 if _test(sys.argv[2:]) else 0)
    r = read_route(sys.argv[1]); print(len(r.net_names), len(r.x0))
