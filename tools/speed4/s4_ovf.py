#!/usr/bin/env python3
"""s4_ovf.py: item 4b, the official overflow cost (tools/offix/of_official.overflow_cost)
without the full-map exp in every estimate. Per demand model `la` the per-cell loss arrays of layers 1..L-1 are kept; a call
recomputes only the cells whose demand changed since the previous call (same elementwise expressions on those cells) and
sums the kept arrays with the SAME numpy reductions (loss.sum() per layer, then u_z x sum accumulated in layer order), so
the result is the original's value bit for bit as long as the elementwise values are (checked: check=True compares every
call with the original). Capacities are taken as fixed per `la` (they never change in the chain); a new `la` object or
a capacity array of another identity rebuilds the cache."""
import numpy as np

CAP_EPS = 1e-3


def _loss(d, c):
    with np.errstate(over="ignore"):
        open_ = np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0))
        blocked = np.where(d > 0, np.exp(np.minimum(0.75 * d, 700.0)), 0.0)
        return np.where(c > CAP_EPS, open_, blocked)


class OfCache:
    def __init__(self, orig, check=False):
        self.orig, self.check = orig, check
        self.key = None
        self.stats = {"calls": 0, "full": 0, "cells": 0, "check_bad": 0}

    def _init(self, la):
        self.key = (id(la), id(la.c), la.d.shape)
        self.la = la
        self.dprev = [None] + [np.array(la.d[z], copy=True) for z in range(1, la.L)]
        self.loss = [None] + [_loss(la.d[z], la.c[z]) for z in range(1, la.L)]
        self.stats["full"] += 1

    def __call__(self, la):
        self.stats["calls"] += 1
        if self.key != (id(la), id(la.c), la.d.shape) or self.la is not la:
            self._init(la)
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
        for z in range(1, la.L):
            tot += float(la.u[z]) * float(self.loss[z].sum())
        if self.check:
            ref = self.orig(la)
            if ref != tot:
                self.stats["check_bad"] += 1
                print(f"[s4_ofc] MISMATCH {ref!r} vs {tot!r}", flush=True)
        return tot


def install(FM, OFO, check=False):
    """FM.overflow_cost (patched to the official rule) and OFO.overflow_cost -> one cache (same function underneath)"""
    if getattr(FM.overflow_cost, "_s4", False):
        return FM.overflow_cost
    assert FM.overflow_cost is OFO.overflow_cost, "s4_ovf.install: of_official.patch() first"
    C = OfCache(OFO.overflow_cost, check)
    C._s4 = True
    FM.overflow_cost = C
    OFO.overflow_cost = C
    return C
