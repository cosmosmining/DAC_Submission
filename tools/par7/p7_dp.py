#!/usr/bin/env python3
"""p7_dp.py: the first-pass layer DP (tools/speedcut/dp_split_api.assign_v2_cpp) on threads.

  chs, info = assign_pair(P, dw, pw, via_cost, lams, threads)   # == [assign_v2_cpp(P, "numpy", dw, pw, via_cost, l)[0] for l in lams]

Each candidate's DP runs on `threads` threads with the level schedule of p7_dp_kern.cc (dp_run_par: same choices as the
sequential dp_run), and the candidates run at the same time (one Python thread each; ctypes releases the GIL).
DPState is dp_split_api.DPState with its context made by libp7_dp (same Ctx struct, same constructor code).
"""
import ctypes, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "tools" / "speedcut", ROOT / "tools" / "fast_la", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import dp_split_api as DS  # noqa: E402
V1 = DS.V1
_LIB = None
STATS = []


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    ext = "dylib" if sys.platform == "darwin" else "so"
    so = HERE / f"libp7_dp.{ext}"
    src = HERE / "p7_dp_kern.cc"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.check_call(["clang++" if sys.platform == "darwin" else "g++", "-O3", "-std=c++17", "-ffp-contract=off",
                               "-fPIC", "-shared", "-pthread", "-o", str(so), str(src)])
    L = ctypes.CDLL(str(so))
    i64, dbl, vp = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
    L.dp_ctx_new.restype = vp
    L.dp_ctx_new.argtypes = [i64] * 5 + [vp] * 4 + [i64, vp, vp, i64, vp, i64, vp] + [vp] * 6 + [dbl] * 8
    L.dp_ctx_free.argtypes = [vp]
    L.dp_run.restype = i64
    L.dp_run.argtypes = [vp, vp, i64, vp, vp, vp, vp]
    L.dp_run_par.restype = i64
    L.dp_run_par.argtypes = [vp, vp, i64, vp, vp, vp, vp, i64, i64, vp]
    _LIB = L
    return L


class _Swap:
    """dp_split_api.DPState.__init__ looks its library up through DS._lib(): build the state under our library (lock held)."""
    mu = threading.Lock()


class State(DS.DPState):
    def __init__(self, P, delay_weight, power_weight, via_cost, gnn_lambda):
        with _Swap.mu:
            old = DS._lib
            DS._lib = _lib
            try:
                super().__init__(P, delay_weight, power_weight, via_cost, gnn_lambda)
            finally:
                DS._lib = old

    def run_par(self, order, crit_eff, T):
        order = np.ascontiguousarray(order, dtype=np.int64)
        crit_eff = np.ascontiguousarray(crit_eff, dtype=np.float64)
        out = np.zeros(1); st = np.zeros(12)
        fb = _lib().dp_run_par(self.ctx, DS._p(order), len(order), DS._p(crit_eff), DS._p(self.demand),
                               DS._p(self.choice), DS._p(out), len(self.demand), int(T), DS._p(st))
        self.fallback += int(fb); self.local_cost += float(out[0])
        return st

    def close(self):
        if self.ctx:
            _lib().dp_ctx_free(self.ctx); self.ctx = None


def assign_pair(P, dw, pw, via_cost, lams, threads):
    t0 = time.perf_counter()
    crit = DS.criticality(P)
    out = [None] * len(lams); info = [None] * len(lams); err = []

    def one(j, lam):
        try:
            t_ = time.perf_counter()
            st = State(P, dw, pw, via_cost, lam)
            t1 = time.perf_counter()
            order = st.order(crit)
            s_ = st.run_par(order, crit, threads)
            out[j] = st.choice.copy()
            info[j] = {"lam": float(lam), "prep_s": t1 - t_, "kernel_s": time.perf_counter() - t1, "nets": int(s_[0]),
                       "levels": int(s_[1]), "max_level": int(s_[2]), "level_calc_s": float(s_[3]), "par_s": float(s_[8]),
                       "nets_in_level_size": {"1": int(s_[4]), "2-7": int(s_[5]), "8-63": int(s_[6]), ">=64": int(s_[7])},
                       "fallback": int(st.fallback)}
            st.close()
        except BaseException as e:      # re-raised in the caller
            err.append(e)

    th = [threading.Thread(target=one, args=(j, l), daemon=True) for j, l in enumerate(lams)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if err:
        raise err[0]
    STATS.append({"wall_s": time.perf_counter() - t0, "threads_each": int(threads), "cands": info})
    return out, info
