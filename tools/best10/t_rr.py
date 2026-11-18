"""t_rr.py: tools/best5/n_rr.py with the kernel tools/best10/t_rr.cc (rectangle windows, long runs last, A* pop cap; tools/best10/mk_t_rr.py). Off: n_rr's task graph."""
import ctypes, gc, os, sys, time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best5"))
import n_rr as NR

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "dylib" if sys.platform == "darwin" else "so"
_L = [None]; _V = ctypes.c_void_p
RR = NR.RR
CFG = {"rH": 0, "rW": 0, "dH": 0, "dN": 0, "popcap": 0}


class _Lib:
    """n_rr._rr calls L.n_rr_rank(... policy, shift, dinfo), L.n_rr_nseg, L.n_rr_get: forwarded to the t_rr kernel + CFG"""
    def __init__(self, L):
        self.L = L

    def n_rr_rank(self, *aa):
        rv = self.L.t_rr_rank(*aa, int(CFG["rH"]), int(CFG["rW"]), int(CFG["dH"]), int(CFG["dN"]), ctypes.c_int64(int(CFG["popcap"])), ctypes.c_int64(int(CFG.get("areaB", 0))))
        di = ctypes.cast(aa[-1], ctypes.POINTER(ctypes.c_double))
        CFG["area"] = float(di[6]); CFG["cut"] = float(di[7])
        return rv

    def n_rr_nseg(self):
        return self.L.t_rr_nseg()

    def n_rr_get(self, *aa):
        return self.L.t_rr_get(*aa)


def lib():
    if _L[0] is None:
        L = ctypes.CDLL(os.path.join(HERE, "libt_rr." + EXT))
        V = _V
        L.t_rr_rank.restype = ctypes.c_int
        L.t_rr_rank.argtypes = [ctypes.c_int] * 3 + [V] * 6 + [V] + [ctypes.c_int] + [V] * 6 + [ctypes.c_int] * 3 + [V] + [ctypes.c_int] * 2 + [V] + [ctypes.c_int] * 2 + [V] + [ctypes.c_int] * 4 + [ctypes.c_int64] * 2
        L.t_rr_nseg.restype = ctypes.c_int64
        L.t_rr_get.argtypes = [V, V, V]
        _L[0] = _Lib(L)
    return _L[0]


def reroute_rank(S, pins, chosen, W=14, pin_on=False, pcmap=None, stats=None, astar="win", max_runs=0, hmax=0, timing=None, nthr=8, batch=32, policy=4, shift=2, cfg=None):
    if cfg:
        CFG.update(cfg)
    NR.lib = lib                                                   # n_rr._rr binds its kernel through this module's lib()
    return NR.reroute_rank(S, pins, chosen, W=W, pin_on=pin_on, pcmap=pcmap, stats=stats, astar=astar, max_runs=max_runs, hmax=hmax, timing=timing, nthr=nthr, batch=batch, policy=policy, shift=shift)
