#!/usr/bin/env python3
"""GAP-LA v2: one-shot STA ordering, technology RC, and an optional GNN prior.

This is intentionally a new front end around :mod:`gapla_dp`.  The v1 DP and
route writer are reused, while this file supplies the pieces needed for a
closer GAP-LA experiment:

* one repository ``fast_sta`` analysis produces the critical report and the
  per-net load/driver values used by the look-ahead term;
* the RC table is read from ``NanGate45/setRC.tcl`` and the tech LEF through
  ``fast_timer.read_tech``;
* nets are assigned in one criticality order and demand is committed after
  each net, so later DP costs see remaining capacity;
* the optional LOFO GNN model contributes ``-lambda * log_softmax`` to
  the per-run candidate cost.

The evaluator and official score are deliberately outside this module.  The
caller can pass the output route to ``the official evaluator script``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import gapla_dp as V1  # noqa: E402  (the merged v1 implementation)

DEFAULT_DATA_ROOT = Path("/path/to/repo")
DEFAULT_GNN_ROOT = DEFAULT_DATA_ROOT / "experiments" / "fastgnn_lofo" / "models"
FAMILY_FOR_DESIGN = {"mempool_tile_wrap": "MemPool"}


def _fallback_torch_scatter() -> None:
    """Provide the two tiny torch_scatter helpers used by the model import.

    The CPU venv has torch-geometric but not the optional torch-scatter wheel.
    The forward path used here only needs the model class; installing a wheel
    is outside the task, so make the import self-contained and deterministic.
    """
    try:
        import torch_scatter  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    import torch

    mod = types.ModuleType("torch_scatter")

    def scatter_sum(src, index, dim=0, out=None, dim_size=None):
        if dim != 0:
            raise NotImplementedError("fallback scatter_sum supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() else 0
        size = max(size, int(dim_size or 0))
        shape = (size,) + tuple(src.shape[1:])
        ans = src.new_zeros(shape) if out is None else out.zero_()
        return ans.index_add_(0, index, src)

    def scatter_softmax(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("fallback scatter_softmax supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() else 0
        size = max(size, int(dim_size or 0))
        shape = (size,) + tuple(src.shape[1:])
        mx = src.new_full(shape, -torch.inf)
        mx.scatter_reduce_(0, index.reshape((-1,) + (1,) * (src.ndim - 1)).expand_as(src),
                           src, reduce="amax", include_self=True)
        ex = torch.exp(src - mx[index])
        den = src.new_zeros(shape).index_add_(0, index, ex)
        return ex / den[index].clamp_min(torch.finfo(src.dtype).tiny)

    mod.scatter_sum = scatter_sum
    mod.scatter_softmax = scatter_softmax
    sys.modules["torch_scatter"] = mod


def run_sta_once(design: str, suite: str, route: Path, data_root: Path,
                 crit_out: Path, npaths: int) -> tuple[dict, dict]:
    """Run the repository timer exactly once and write its critical report."""
    t0 = time.perf_counter()
    bench = data_root / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
    sys.path.insert(0, str(ROOT))
    FS = importlib.import_module("fast_sta")
    # StaDesign's explicit kit argument is important: benchmark inputs are in
    # the read-only data checkout, not in this code worktree.
    D = FS.StaDesign(design, suite, kit=str(bench))
    t_design = time.perf_counter()
    seg = FS.read_route(D, str(route))
    t_route = time.perf_counter()
    r = FS.StaTimer(D).analyze(seg, rat=True, keep=True)
    t_sta = time.perf_counter()
    _fallback_torch_scatter()
    FM4 = importlib.import_module("fastmode4")
    crit_out.parent.mkdir(parents=True, exist_ok=True)
    paths = FM4.write_crit(D, r, str(crit_out), npaths=npaths, slack_max=0.0)
    t_crit = time.perf_counter()
    stages = {
        "timer_design_s": t_design - t0,
        "timer_route_read_s": t_route - t_design,
        "timer_sta_s": t_sta - t_route,
        "crit_write_s": t_crit - t_sta,
        "timer_total_s": t_crit - t0,
        "crit_paths": int(paths),
        "timer_wns": float(r["wns"]),
        "timer_tns": float(r["tns"]),
        "timer_missing": int(r["missing"]),
    }
    # Keep the timer's net arrays for per-net Elmore load and driver RC.
    return {"design": D, "result": r}, stages


def add_real_tech_rc(P: dict, timer: dict | None, data_root: Path,
                     crit_path: Path | None = None) -> dict:
    """Replace builder fallback RC with the design's setRC/tech-LEF values."""
    sys.path.insert(0, str(ROOT))
    import fast_timer as FT

    bench = data_root / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
    layer_rc, via_r = FT.read_tech(str(bench))
    if len(layer_rc) != int(P["L"]):
        raise ValueError(f"tech has {len(layer_rc)} layers but problem has L={P['L']}")
    P["layer_rc"] = layer_rc
    P["via_r"] = via_r
    P["Cz"] = layer_rc[:, 1].astype(np.float32)

    D = timer["design"] if timer else None
    r = timer["result"] if timer else None
    if D is not None:
        net_index = {_n.replace("\\", ""): i for i, _n in enumerate(D.net_names)}
    else:
        net_index = {}
    pnet_names = [str(n).replace("\\", "") for n in P["net_names"]]
    did = np.asarray([net_index.get(n, -1) for n in pnet_names], dtype=np.int64)
    if r is not None:
        ctot = np.asarray(r["net_ctot"], dtype=np.float64)
        rdrv = np.asarray(r["net_rd_kohm"], dtype=np.float64)
        c_load = np.where(did >= 0, 0.5 * ctot[np.maximum(did, 0)], 5.0)
        r_drv = np.where(did >= 0, rdrv[np.maximum(did, 0)], 2.0)
    else:
        if crit_path is None:
            raise ValueError("technology RC fallback requires the critical report")
        TP = importlib.import_module("timing_promote")
        report_rc = TP.parse_net_rc(str(crit_path))
        c_load, r_drv = [], []
        for name in pnet_names:
            cl, rd = TP.net_rc_params(report_rc.get(name), 5.0, 2.0)
            c_load.append(cl); r_drv.append(rd)
        c_load, r_drv = np.asarray(c_load, dtype=np.float64), np.asarray(r_drv, dtype=np.float64)
    c_load = np.clip(c_load, 1.0, 1.0e5)
    r_drv = np.clip(r_drv, 0.05, 20.0)
    rn = np.asarray(P["run_net"], dtype=np.int64)
    lum = np.asarray(P["run_len"], dtype=np.float64) * float(P["um"])
    cand = np.asarray(P["cand"], dtype=np.int64)
    dcoef = np.zeros_like(cand, dtype=np.float32)
    for k in range(cand.shape[1]):
        z = np.clip(cand[:, k], 0, len(layer_rc) - 1)
        rz = layer_rc[z, 0]
        cz = layer_rc[z, 1]
        # Elmore of a distributed wire represented by its pi model: driver R
        # and downstream load see the full wire C; the wire's own R/C term is
        # the standard half-C / half-R contribution.
        val = (lum * (rz * c_load[rn] + r_drv[rn] * cz * lum)
               + 0.5 * rz * cz * lum * lum) / 1000.0
        dcoef[:, k] = np.where(cand[:, k] >= 0, val, 0.0)
    P["dcoef"] = dcoef
    return {"rc_source": str(bench / "NanGate45"),
            "layers": int(len(layer_rc)),
            "c_load_fF_min": float(c_load.min()) if len(c_load) else 0.0,
            "c_load_fF_max": float(c_load.max()) if len(c_load) else 0.0,
            "r_drv_kohm_min": float(r_drv.min()) if len(r_drv) else 0.0,
            "r_drv_kohm_max": float(r_drv.max()) if len(r_drv) else 0.0}


def gnn_logits(P: dict, model_path: Path) -> tuple[np.ndarray, dict]:
    """Run the same graph/load/to_logits inference path as fastmode4.py."""
    _fallback_torch_scatter()
    import torch
    torch.set_num_threads(min(8, max(1, int(os.environ.get("OMP_NUM_THREADS", "8")))))
    sys.path.insert(0, str(ROOT))
    import dtla_lagnn as DD

    t0 = time.perf_counter()
    model = DD.load(str(model_path), device="cpu")
    x, ei, sid = DD.graph(P)
    with torch.no_grad():
        raw = model(x, ei)
        logits = DD.to_logits(raw, sid).clamp(-60.0, 60.0)
    out = logits.detach().cpu().numpy().astype(np.float64)
    valid = np.asarray(P["cand"]) >= 0
    out[~valid] = -np.inf
    return out, {"gnn_model": str(model_path), "gnn_runs": int(out.shape[0]),
                 "gnn_wall_s": time.perf_counter() - t0}


class _V2CostModel(V1._CostModel):
    def __init__(self, P, backend="auto", delay_weight=1.0, power_weight=None,
                 via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0):
        super().__init__(P, backend, delay_weight, power_weight, via_cost)
        self.gnn_lambda = float(gnn_lambda)
        self.gnn = np.asarray(P.get("gnn_logits"), dtype=np.float64) if "gnn_logits" in P else None
        if self.gnn is not None:
            masked = np.where(np.asarray(P["cand"]) >= 0, self.gnn, -np.inf)
            mx = np.max(masked, axis=1, keepdims=True)
            den = np.log(np.exp(masked - mx).sum(axis=1, keepdims=True)) + mx
            self.gnn_logp = masked - den
        else:
            self.gnn_logp = None

    def unary(self, runs, cells):
        out = super().unary(runs, cells)
        if self.gnn_logp is not None and self.gnn_lambda:
            out = out - self.gnn_lambda * self.gnn_logp[runs]
        return out


def assign_v2(P: dict, backend="numpy", delay_weight=1.0, power_weight=None,
              via_cost=V1.DEFAULT_VIA_COST, gnn_lambda=0.0) -> tuple[np.ndarray, dict]:
    """One criticality-ordered DP pass with demand commit and optional prior."""
    t0 = time.perf_counter()
    rn = V1._np(P["run_net"], np.int64)
    R, K = len(rn), int(P["K"])
    choice = V1._np(P.get("run_cur", np.zeros(R, np.int64)), np.int64).copy()
    model = _V2CostModel(P, backend, delay_weight, power_weight, via_cost, gnn_lambda)
    all_cells = V1._candidate_cells(P, np.arange(R, dtype=np.int64), K)
    order = sorted((int(n) for n in np.unique(rn)), key=lambda n: (-float(model.crit[n]), n))
    fallback = 0
    local_cost = 0.0
    for n in order:
        runs = np.flatnonzero(rn == n).astype(np.int64)
        cells = [all_cells[int(r)] for r in runs.tolist()]
        unary = model.unary(runs, cells)
        c, cost, ok = V1._tree_dp(P, n, unary, runs, model)
        fallback += int(not ok)
        choice[runs] = c
        model.commit(runs, c, cells)
        local_cost += cost
    return choice, {
        "backend": model.backend, "nets": int(len(order)), "runs": int(R),
        "fallback_nets": int(fallback), "wall_s": time.perf_counter() - t0,
        "criticality": "one_sta_path_count_logistic_slack",
        "delay_model": "technology_RC_pi_elmore",
        "gnn_lambda": float(gnn_lambda), "local_cost": float(local_cost),
    }


def family_for(design: str) -> str:
    return FAMILY_FOR_DESIGN.get(design, design)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--suite", choices=("visible", "blind"), default="visible")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--crit-out")
    ap.add_argument("--json")
    ap.add_argument("--max-nets", type=int, default=6000)
    ap.add_argument("--npaths", type=int, default=5000)
    ap.add_argument("--backend", choices=("numpy", "auto", "cuda"), default="numpy")
    ap.add_argument("--delay-weight", type=float, default=1.0)
    ap.add_argument("--power-weight", type=float, default=None)
    ap.add_argument("--via-cost", type=float, default=V1.DEFAULT_VIA_COST)
    ap.add_argument("--gnn-model")
    ap.add_argument("--gnn-lambda", type=float, default=0.0)
    ap.add_argument("--no-timer", action="store_true", help="use --crit instead of running fast_sta once")
    ap.add_argument("--crit", help="existing report, only with --no-timer")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    t0 = time.perf_counter()
    data_root = Path(a.data_root).resolve()
    route = V1._path(a.route, data_root)
    out = Path(a.out).resolve()
    crit = Path(a.crit_out).resolve() if a.crit_out else out.with_suffix(".crit.rpt")
    timer_rec, timer_stages = None, {"timer_total_s": 0.0}
    if a.no_timer:
        if not a.crit:
            raise SystemExit("--no-timer requires --crit")
        crit = V1._path(a.crit, data_root)
    else:
        timer_rec, timer_stages = run_sta_once(a.design, a.suite, route, data_root, crit, a.npaths)
    t_timer = time.perf_counter()
    P = V1.build_problem_from_route(str(route), str(crit), a.design, a.suite, a.max_nets, data_root)
    t_build = time.perf_counter()
    rc_stats = add_real_tech_rc(P, timer_rec, data_root, crit)
    gnn_stats = {}
    if a.gnn_model:
        P["gnn_logits"], gnn_stats = gnn_logits(P, V1._path(a.gnn_model, data_root))
    choice, dp_stats = assign_v2(P, a.backend, a.delay_weight, a.power_weight,
                                 a.via_cost, a.gnn_lambda)
    t_dp = time.perf_counter()
    write_stats = V1.write_route(P, choice, out)
    t_write = time.perf_counter()
    rec = {
        "input": str(route), "crit": str(crit), "out": str(out),
        "design": a.design, "suite": a.suite, "gnn_family": family_for(a.design),
        "lambda": float(a.gnn_lambda), "timer": timer_stages,
        "build_s": t_build - t_timer, "rc_s": 0.0, "gnn": gnn_stats,
        "dp": dp_stats, "write": write_stats,
        "walls": {"total_s": t_write - t0, "timer_s": t_timer - t0,
                   "build_s": t_build - t_timer, "dp_s": t_dp - t_build,
                   "write_s": t_write - t_dp},
        "rc": rc_stats,
    }
    if a.json:
        j = Path(a.json).resolve(); j.parent.mkdir(parents=True, exist_ok=True)
        j.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
