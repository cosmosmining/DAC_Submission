"""n_ofc.py: tools/speed4/s4_ovf.OfCache with the map compare and the layer sums in C++ on threads (n_ofc.cc); the exp of
changed cells stays in numpy. Same value bit for bit: at the first call each layer's C++ sum is compared with numpy's; any difference (or an array that is not
contiguous) -> the whole cache stays on numpy's path."""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_EPS = 1e-3


def _loss(d, c):
    with np.errstate(over="ignore"):
        open_ = np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0))
        blocked = np.where(d > 0, np.exp(np.minimum(0.75 * d, 700.0)), 0.0)
        return np.where(c > CAP_EPS, open_, blocked)


class OfCache:
    def __init__(self, orig, check=False, threads=8):
        self.orig, self.check, self.T = orig, check, int(threads)
        self.key = None
        self.stats = {"calls": 0, "full": 0, "cells": 0, "check_bad": 0, "native": 0, "kern": 1}
        self.so = ctypes.CDLL(os.path.join(HERE, "libn_ofc." + ("dylib" if sys.platform == "darwin" else "so")))
        V = ctypes.c_void_p
        self.so.n_ofc_diff.argtypes = [ctypes.c_int, ctypes.c_int64, V, V, V, V, ctypes.c_int]
        self.so.n_ofc_sum.argtypes = [ctypes.c_int, ctypes.c_int64, V, V, ctypes.c_int]

    def _init(self, la):
        self.key = (id(la), id(la.c), la.d.shape)
        self.la = la
        L = la.L; self.L = L; self.Y = la.d.shape[2]
        self.dprev = [None] + [np.array(la.d[z], copy=True) for z in range(1, L)]
        self.loss = [None] + [_loss(la.d[z], la.c[z]) for z in range(1, L)]
        self.stats["full"] += 1
        ford = la.d[1].flags.f_contiguous and not la.d[1].flags.c_contiguous      # memory order of the demand layers
        def _ct(a):
            return a.flags.f_contiguous if ford else a.flags.c_contiguous
        same = all(_ct(self.loss[z]) and _ct(la.d[z]) and _ct(self.dprev[z]) for z in range(1, L))
        self.ord = "F" if ford else "C"; self.shape2 = la.d.shape[1:]
        self.native = False
        if same:
            self.n = int(self.loss[1].size)
            self.dptr = (ctypes.c_void_p * L)(*([0] + [la.d[z].ctypes.data for z in range(1, L)]))
            self.pptr = (ctypes.c_void_p * L)(*([0] + [self.dprev[z].ctypes.data for z in range(1, L)]))
            self.lptr = (ctypes.c_void_p * L)(*([0] + [self.loss[z].ctypes.data for z in range(1, L)]))
            self.idx = [None] + [np.empty(self.n, np.int64) for _ in range(1, L)]
            self.iptr = (ctypes.c_void_p * L)(*([0] + [self.idx[z].ctypes.data for z in range(1, L)]))
            self.cnt = np.zeros(L, np.int64); self.sums = np.zeros(L)
            self.so.n_ofc_sum(L, self.n, self.lptr, self.sums.ctypes.data, self.T)
            self.native = all(float(self.loss[z].sum()) == float(self.sums[z]) for z in range(1, L))
        self.stats["native"] = int(self.native)
        self.stats["layout"] = [list(la.d.strides), list(la.c.strides), list(self.loss[1].strides), list(la.d.shape), bool(same)]

    def __call__(self, la):
        self.stats["calls"] += 1
        if self.key != (id(la), id(la.c), la.d.shape) or self.la is not la:
            self._init(la)
        elif self.native:
            for z in range(1, self.L):                # la.d may have been replaced by an array of the same shape
                if la.d[z].ctypes.data != self.dptr[z]:
                    self.dptr[z] = la.d[z].ctypes.data
            self.so.n_ofc_diff(self.L, self.n, self.dptr, self.pptr, self.iptr, self.cnt.ctypes.data, self.T)
            for z in range(1, self.L):
                m = int(self.cnt[z])
                if m:
                    ix = np.unravel_index(self.idx[z][:m], self.shape2, order=self.ord)      # (x, y) of the memory positions
                    self.loss[z][ix] = _loss(la.d[z][ix], la.c[z][ix])
                    self.stats["cells"] += m
        else:
            for z in range(1, la.L):
                d, dp = la.d[z], self.dprev[z]
                ch = d != dp
                if ch.any():
                    ix = np.nonzero(ch)
                    dv = d[ix]
                    self.loss[z][ix] = _loss(dv, la.c[z][ix])
                    dp[ix] = dv
                    self.stats["cells"] += len(dv)
        tot = 0.0
        if self.native:
            self.so.n_ofc_sum(self.L, self.n, self.lptr, self.sums.ctypes.data, self.T)
            for z in range(1, self.L):
                tot += float(la.u[z]) * float(self.sums[z])
        else:
            for z in range(1, la.L):
                tot += float(la.u[z]) * float(self.loss[z].sum())
        if self.check:
            ref = self.orig(la)
            if ref != tot:
                self.stats["check_bad"] += 1
                print(f"[n_ofc] MISMATCH {ref!r} vs {tot!r}", flush=True)
        return tot


def install(FM, OFO, check=False, threads=8):
    """over tools/speed4/s4_ovf.install: FM.overflow_cost and OFO.overflow_cost -> one cache (the original is the C's .orig when s4 is installed)"""
    orig = getattr(FM.overflow_cost, "orig", FM.overflow_cost)
    C = OfCache(orig, check, threads)
    C._s4 = True
    FM.overflow_cost = C
    OFO.overflow_cost = C
    return C
