#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_weights.py — RC-weighted, timing-driven routing cost for DGR.

Implements the shared idea of the 2026 timing-driven global-routing papers, in
DGR's differentiable framework (NEW file; model.py / main_stochastic.py are NOT
modified):

  * Xiao et al., "Gradient Guided RC Weighting for Timing-Driven Global
    Routing," ISPD 2026.
  * Wang et al., "RC-Scaled Timing Driven Routing," ISCAS 2026.
  * Zhou et al., "GPU-Accelerated Global Routing with Balanced Timing and
    Congestion Optimization," ASP-DAC 2026.
  * Wu et al., "An Adaptive Cost-Based Via and Congestion Co-Optimization
    Framework," DATE 2026.

Shared approach they all use: weight each net's wirelength / via cost by a
timing criticality so that near-critical nets are pushed onto shorter, lower-RC
(fewer-via, higher-layer) routes, trading a little congestion headroom for
slack. DGR's objective is already differentiable in the per-subnet route
distribution `p`, so a per-net criticality weight enters as a simple
multiplicative factor on the WL and via terms — no change to the optimizer.

This module provides:
  criticality_from_slack(slack)          -> per-net weight w in [1, w_max]
  rc_delay_proxy(length, n_via, ...)     -> Elmore-style RC delay per net
  timing_weighted_costs(wl, via, w, ...) -> criticality-scaled per-net costs
  gradient_guided_update(w, dTNS_dnet)   -> iterative RC-weight refinement
                                            (Xiao ISPD'26 style)
It is engine-agnostic: `w` can multiply FastDGR's per-subnet WL/via terms
(dgr_fast) or be passed to CUGR2 as per-net weights. Slack comes from the
ISPD'25 benchmark (.net per-pin pre-routing slack) or from an STA pass.

Honest note: this is a faithful implementation of the *approach class* those
papers share (RC/criticality-weighted routing cost), synthesized from their
titles + the ISPD'25 timing-driven formulation — not a bit-reproduction of any
single paper's exact weighting function. The weighting mode is selectable so it
can be matched to a specific paper's formula when its details are available.
"""
import numpy as np


def criticality_from_slack(slack, mode="exp", w_max=8.0, gamma=1.0,
                           eps=1e-12):
    """Per-net timing weight from slack (more negative slack => higher weight).

    slack : array [N] of per-net slack (ns); negative = timing violation.
    mode  : 'exp'   w = 1 + (w_max-1)*(1 - exp(gamma*s_norm))     (smooth)
            'rank'  w by slack rank percentile (robust to slack scale)
            'linear'w = 1 + (w_max-1)*clip((-s)/(-s_min), 0, 1)
            'step'  w = w_max where slack<0 else 1 (hard critical set)
    Returns w in [1, w_max]; non-critical (large positive slack) -> ~1.
    """
    s = np.asarray(slack, dtype=np.float64).ravel()
    if s.size == 0:
        return np.ones(0)
    if mode == "step":
        return np.where(s < 0, w_max, 1.0)
    if mode == "rank":
        order = s.argsort()                      # most negative first
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(s.size)
        pct = 1.0 - rank / max(s.size - 1, 1)    # 1 for worst slack
        return 1.0 + (w_max - 1.0) * pct
    smin = s.min()
    if mode == "linear":
        denom = -min(smin, -eps)
        crit = np.clip((-s) / (denom + eps), 0.0, 1.0)
        return 1.0 + (w_max - 1.0) * crit
    # 'exp' (default): smooth, emphasizes the negative-slack tail
    s_norm = (s - smin) / (abs(smin) + eps)      # 0 at worst, grows for slack>smin
    crit = np.exp(-gamma * s_norm)               # 1 at worst slack, decays
    crit = (crit - crit.min()) / (crit.max() - crit.min() + eps)
    return 1.0 + (w_max - 1.0) * crit


def rc_delay_proxy(length, n_via, unit_r=1.0, unit_c=1.0, via_r=5.0,
                   pin_c=0.0):
    """Elmore-style interconnect RC delay proxy per net (relative units).
    delay ~ R*C ; R = unit_r*length + via_r*n_via ; C = unit_c*length + pin_c.
    Returns array [N]. Used to rank nets and to define RC-scaling targets
    (Wang ISCAS'26) and to measure the timing effect of a routing choice.
    """
    length = np.asarray(length, dtype=np.float64)
    n_via = np.asarray(n_via, dtype=np.float64)
    R = unit_r * length + via_r * n_via
    C = unit_c * length + pin_c
    return R * C


def timing_weighted_costs(wl_per_net, via_per_net, weights,
                          via_weight_frac=1.0):
    """Criticality-scale the per-net WL and via costs.
    Critical nets get their WL and via costs amplified so the optimizer
    prefers shorter / lower-via routes for them (balanced timing/congestion,
    Zhou ASP-DAC'26; via/congestion co-opt, Wu DATE'26).

    via_weight_frac in [0,1]: how strongly criticality also scales the via
    term (vias add series R on critical nets, so >0 is timing-motivated).
    Returns (wl_weighted, via_weighted).
    """
    w = np.asarray(weights, dtype=np.float64)
    wl = np.asarray(wl_per_net, dtype=np.float64) * w
    vw = 1.0 + via_weight_frac * (w - 1.0)
    via = np.asarray(via_per_net, dtype=np.float64) * vw
    return wl, via


def gradient_guided_update(weights, dTNS_dnet, step=0.5, w_max=8.0):
    """One gradient-guided RC-weight refinement step (Xiao ISPD'26 style).
    Increase weights on nets whose shortening most improves TNS (large
    negative dTNS/d(net cost) => this net is timing-limiting). Keeps w in
    [1, w_max]. dTNS_dnet: sensitivity of total negative slack to each net's
    routed cost (from a fast incremental STA or the RC proxy gradient).
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    g = np.asarray(dTNS_dnet, dtype=np.float64)
    # nets that most help TNS when shortened have the most negative gradient;
    # boost -> 1 for the most-negative g, 0 for the least (Xiao ISPD'26 style)
    if np.ptp(g) > 0:
        boost = (g.max() - g) / (g.max() - g.min() + 1e-12)   # 1 for most-critical
        w = w + step * (w_max - 1.0) * boost
    return np.clip(w, 1.0, w_max)


def summarize(weights):
    w = np.asarray(weights, dtype=np.float64)
    return {"n": int(w.size), "n_critical(w>1.5)": int((w > 1.5).sum()),
            "w_mean": float(w.mean()) if w.size else 0.0,
            "w_max": float(w.max()) if w.size else 0.0}


if __name__ == "__main__":
    # self-test on synthetic slack (no benchmarks / GPU needed)
    rng = np.random.default_rng(0)
    slack = rng.normal(0.2, 0.3, 2000)          # ns; some nets negative
    slack[:50] = rng.uniform(-0.8, -0.1, 50)    # a critical tail
    length = rng.uniform(5, 500, 2000)
    nvia = rng.integers(2, 40, 2000)
    ok = 0
    for mode in ("exp", "rank", "linear", "step"):
        w = criticality_from_slack(slack, mode=mode, w_max=8.0)
        assert w.min() >= 1.0 - 1e-9 and w.max() <= 8.0 + 1e-9, mode
        # worst-slack net must be among the most weighted
        assert w[slack.argmin()] >= np.percentile(w, 90), mode
        print(f"  [PASS] mode={mode:6} {summarize(w)}")
        ok += 1
    wl, via = timing_weighted_costs(length, nvia,
                                    criticality_from_slack(slack))
    assert (wl >= length - 1e-6).all() and (via >= nvia - 1e-6).all()
    print("  [PASS] timing_weighted_costs monotone")
    d = rc_delay_proxy(length, nvia)
    w2 = gradient_guided_update(np.ones(2000), -d, step=0.5)
    assert w2.max() <= 8.0 + 1e-9 and w2[d.argmax()] >= np.percentile(w2, 90)
    print(f"  [PASS] gradient_guided_update {summarize(w2)}")
    print(f"{ok+2} checks passed")
