#!/usr/bin/env python3
"""dp_split_api.py: C++ DP v2 and its congestion / critical-tail split.

Drop-in for sp_onepass2.py:

    import dp_split_api as DS          # tools/speedcut on sys.path
    choice, stats = DS.assign_v2_cpp(P, "numpy", dw, pw, via_cost, lam)      # == gapla_dp_v2.assign_v2

Same arithmetic order as gapla_dp_v2.assign_v2 (built with -ffp-contract=off), route md5-identical to v2
(checked by experiments/speedcut/dp_split/run_parity.py).

Split API (the DP is per net, nets in criticality order, demand committed after each net):

    st = DS.prepare(P, dw, pw, via_cost, lam)        # criticality-independent: arrays + CSR (can run before the report)
    st.run(order, crit_eff)                           # process nets `order`, demand/choice persist in st (call twice = split)
    crit = DS.criticality(P)                          # needs the report (p_net, p_path, slack0)
    choice, stats = DS.assign_split(P, cong_nets, ...)  # congestion part (cong_nets, first) + critical tail (rest)

``run`` releases the GIL (ctypes), so the congestion part can run in a thread next to the timer.
"""
from __future__ import annotations
import ctypes, subprocess, sys, time, threading
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "tools" / "fast_la", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gapla_dp as V1  # noqa: E402

_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    ext = "dylib" if sys.platform == "darwin" else "so"
    so = HERE / f"libdp_split_kern.{ext}"
    src = HERE / "dp_split_kern.cc"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.check_call(["clang++" if sys.platform == "darwin" else "g++", "-O3", "-std=c++17",
                               "-ffp-contract=off", "-fPIC", "-shared", "-o", str(so), str(src)])
    L = ctypes.CDLL(str(so))
    i64, dbl, vp = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
    L.dp_ctx_new.restype = vp
    L.dp_ctx_new.argtypes = [i64] * 5 + [vp] * 4 + [i64, vp, vp, i64, vp, i64, vp] + [vp] * 6 + [dbl] * 8
    L.dp_ctx_free.argtypes = [vp]
    L.dp_run.restype = i64
    L.dp_run.argtypes = [vp, vp, i64, vp, vp, vp, vp]
    _LIB = L
    return L


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def criticality(P) -> np.ndarray:
    """Same function as the Python DP (needs the STA report: p_net, p_path, slack0)."""
    return V1._criticality(P)


class DPState:
    def __init__(self, P, delay_weight=1.0, power_weight=None, via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0):
        t0 = time.perf_counter()
        self.P = P
        rn = np.ascontiguousarray(V1._np(P["run_net"], np.int64))
        self.R = R = len(rn)
        self.K = K = int(P["K"]); self.L = L = int(P["L"])
        self.nnet = nnet = int(rn.max()) + 1 if R else 0
        self.rn = rn
        self.nets = np.unique(rn)
        c64 = lambda k: np.ascontiguousarray(V1._np(P[k], np.int64))
        f64 = lambda a: np.ascontiguousarray(V1._np(a, np.float64))
        self._keep = keep = [rn]
        cand = c64("cand"); p0 = c64("run_p0"); p1 = c64("run_p1")
        ent_rk = c64("ent_rk"); ent_cell = c64("ent_cell")
        pt_lo = np.ascontiguousarray(V1._np(P.get("pt_lo", np.zeros(0)), np.int64))
        pt_hi = np.ascontiguousarray(V1._np(P.get("pt_hi", np.zeros(0)), np.int64))
        dcoef = f64(P.get("dcoef", np.zeros((R, K)))); cap = f64(P["cap"]); u = f64(P["u"])
        Cz = f64(P.get("Cz", np.ones(L))); run_len = f64(P["run_len"])
        self.demand0 = f64(P["d0"]).copy()
        self.n_cells = len(self.demand0)
        keep += [cand, p0, p1, ent_rk, ent_cell, pt_lo, pt_hi, dcoef, cap, u, Cz, run_len]
        w1, w2, w3, w4 = V1._score_weights(P)
        self.pw = float(w3 * V1.DEFAULT_POWER_SCALE if power_weight is None else power_weight)
        self.dw = float(delay_weight); self.via_cost = float(via_cost); self.lam = float(gnn_lambda)
        gl = None
        if "gnn_logits" in P:      # same arithmetic as gapla_dp_v2._V2CostModel.__init__
            g = np.asarray(P["gnn_logits"], dtype=np.float64)
            masked = np.where(np.asarray(P["cand"]) >= 0, g, -np.inf)
            mx = np.max(masked, axis=1, keepdims=True)
            den = np.log(np.exp(masked - mx).sum(axis=1, keepdims=True)) + mx
            gl = np.ascontiguousarray(masked - den)
            keep.append(gl)
        npts = int(max(p0.max(), p1.max())) + 1 if R else 1
        self.ctx = _lib().dp_ctx_new(R, K, L, nnet, npts, _p(rn), _p(cand), _p(p0), _p(p1),
                                     len(ent_rk), _p(ent_rk), _p(ent_cell),
                                     len(pt_lo), _p(pt_lo), len(pt_hi), _p(pt_hi),
                                     _p(dcoef), _p(cap), _p(u), _p(Cz), _p(run_len),
                                     _p(gl) if gl is not None else None,
                                     float(P.get("um", 2.1)), float(w4), self.dw, self.pw, self.via_cost,
                                     self.lam, float(V1.EXP_HALF), float(V1.CAP_EPS))
        self.demand = self.demand0.copy()
        self.choice = V1._np(P.get("run_cur", np.zeros(R, np.int64)), np.int64).copy()
        self.fallback = 0
        self.local_cost = 0.0
        self.prep_s = time.perf_counter() - t0

    def order(self, crit, nets=None):
        nets = self.nets if nets is None else np.asarray(nets, dtype=np.int64)
        # same total order as sorted(key=(-crit, n))
        return np.ascontiguousarray(nets[np.lexsort((nets, -crit[nets]))], dtype=np.int64)

    def run(self, order, crit_eff):
        """Process nets in `order` with delay-term criticality crit_eff[n]; demand/choice persist."""
        order = np.ascontiguousarray(order, dtype=np.int64)
        crit_eff = np.ascontiguousarray(crit_eff, dtype=np.float64)
        out = np.zeros(1)
        fb = _lib().dp_run(self.ctx, _p(order), len(order), _p(crit_eff), _p(self.demand),
                           _p(self.choice), _p(out))
        self.fallback += int(fb); self.local_cost += float(out[0])
        return int(fb)

    def close(self):
        if self.ctx:
            _lib().dp_ctx_free(self.ctx); self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def prepare(P, delay_weight=1.0, power_weight=None, via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0) -> DPState:
    return DPState(P, delay_weight, power_weight, via_cost, gnn_lambda)


def assign_v2_cpp(P, backend="numpy", delay_weight=1.0, power_weight=None,
                  via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0):
    """Drop-in for gapla_dp_v2.assign_v2 (backend ignored: always CPU)."""
    t0 = time.perf_counter()
    st = DPState(P, delay_weight, power_weight, via_cost, gnn_lambda)
    t1 = time.perf_counter()
    crit = criticality(P)
    order = st.order(crit)
    t2 = time.perf_counter()
    st.run(order, crit)
    t3 = time.perf_counter()
    choice = st.choice.copy()
    stats = {"backend": "cpp", "nets": int(len(order)), "runs": int(st.R), "fallback_nets": st.fallback,
             "wall_s": t3 - t0, "prep_s": t1 - t0, "crit_order_s": t2 - t1, "kernel_s": t3 - t2,
             "criticality": "one_sta_path_count_logistic_slack", "delay_model": "technology_RC_pi_elmore",
             "gnn_lambda": float(gnn_lambda), "local_cost": st.local_cost}
    st.close()
    return choice, stats


def assign_split(P, cong_nets, cong_delay=False, delay_weight=1.0, power_weight=None,
                 via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0, crit=None, st=None):
    """Two-part DP: `cong_nets` first (v2 order among themselves, delay term dropped when cong_delay is False),
    then every other net in v2 criticality order on the demand left by the first part.

    The congestion part needs no report (crit_eff = 0); the tail needs the criticality.  Returns (choice, stats)
    with the wall of each part.  With cong_nets = [] this equals assign_v2_cpp (== v2)."""
    t0 = time.perf_counter()
    st = st or DPState(P, delay_weight, power_weight, via_cost, gnn_lambda)
    t1 = time.perf_counter()
    crit = criticality(P) if crit is None else crit
    cong = np.asarray(sorted(set(int(n) for n in cong_nets)), dtype=np.int64)
    in_cong = np.zeros(st.nnet, bool); in_cong[cong] = True
    tail = st.nets[~in_cong[st.nets]]
    o_c = st.order(crit, cong) if len(cong) else np.zeros(0, np.int64)
    o_t = st.order(crit, tail)
    t2 = time.perf_counter()
    ce_c = crit if cong_delay else np.zeros_like(crit)
    st.run(o_c, ce_c)
    t3 = time.perf_counter()
    st.run(o_t, crit)
    t4 = time.perf_counter()
    choice = st.choice.copy()
    stats = {"backend": "cpp_split", "nets": int(len(st.nets)), "cong_nets": int(len(cong)), "tail_nets": int(len(tail)),
             "cong_delay": bool(cong_delay), "fallback_nets": st.fallback, "prep_s": t1 - t0,
             "crit_order_s": t2 - t1, "cong_kernel_s": t3 - t2, "tail_kernel_s": t4 - t3, "wall_s": t4 - t0,
             "local_cost": st.local_cost}
    st.close()
    return choice, stats


def run_cong_in_thread(st: DPState, order, crit_eff):
    """Start the congestion part in a thread (GIL released inside ctypes); returns (thread, result dict)."""
    res = {}
    def _w():
        t = time.perf_counter(); res["t_start"] = t
        st.run(order, crit_eff)
        res["t_end"] = time.perf_counter(); res["wall_s"] = res["t_end"] - t
    th = threading.Thread(target=_w, daemon=True)
    th.start()
    return th, res
