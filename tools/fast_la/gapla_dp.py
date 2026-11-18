#!/usr/bin/env python3
"""One-pass GAP-LA-style layer assignment for an InstantGR route.

This module deliberately has no STA loop.  A route and one already-generated
critical-path report can be converted to the frozen DTLA representation used
by this repository, or that representation can be supplied directly with
``--problem``.  Nets are visited once, in a fixed criticality order.  For
each net, a small dynamic program assigns candidate layers to the route tree;
the chosen wire demand is then committed before the next net is considered.

The local cost follows the published GAP-LA ingredients and the repository's
contest evaluator:

* exact open/blocked-cell overflow marginal from the current demand;
* via-stack cost at tree junctions and pins;
* ``f' = f + w_d * r_est * c`` represented by the per-candidate Elmore
  coefficient ``dcoef`` in the frozen problem, multiplied by a fixed
  logistic/path criticality weight;
* wire capacitance power cost.

The DP is intentionally conservative at a branch: a node's child messages are
combined through a virtual parent layer.  This keeps the tree recurrence
vectorised over the five candidates and is legal for arbitrary-degree trees.
The route topology and XY geometry never change; only metal layers and the
required via stacks are rewritten.

Examples (all outputs should be written in the caller's worktree):

  python tools/fast_la/gapla_dp.py --problem experiments/dtla_frozen/test/ariane_blind.instantgr.pt \
      --out out/gapla_dp/ariane.gapladp.norm.route

  python tools/fast_la/gapla_dp.py --route experiments/ispd25_routes/ariane/ariane.igrnf_e0r1.route \
      --crit experiments/ispd25_routes/ariane/ariane.igrnf_e0r1.crit.rpt \
      --design ariane --suite visible --out ariane.gapladp.norm.route
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/path/to/repo")
EXP_HALF = np.exp(0.5) - 1.0
CAP_EPS = 1.0e-3
LOGISTIC_TAU_NS = 0.05
PIN_WEIGHT_MAX = 4.0
# The frozen arrays do not carry the evaluator's unit-via scalar.  GAP-LA's
# via term is therefore kept as a small score-local regulariser; the evaluator
# still receives the exact via stacks emitted by the writer.
DEFAULT_VIA_COST = 0.05
DEFAULT_POWER_SCALE = 1.0e-3


def _np(value, dtype=None):
    """Convert torch/numpy scalar containers without retaining torch graphs."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _path(path: str | os.PathLike[str], data_root: Path | None = None) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    if data_root is not None and (data_root / p).exists():
        return (data_root / p).resolve()
    if (DEFAULT_DATA_ROOT / p).exists():
        return (DEFAULT_DATA_ROOT / p).resolve()
    return p.resolve()


def load_problem(path: str | os.PathLike[str]) -> dict:
    """Load an ``*.instantgr.pt`` frozen problem on CPU."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the normal venv has torch
        raise RuntimeError("--problem input requires torch") from exc
    p = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(p, dict):
        raise ValueError(f"{path}: expected a dictionary frozen problem")
    return p


def build_problem_from_route(route: str, crit: str, design: str, suite: str,
                             max_nets: int, data_root: Path) -> dict:
    """Build the frozen arrays from a route and one existing STA report.

    ``dtla.build`` is a read-only repository builder.  Its benchmark root is
    redirected to the external read-only data checkout because this worktree
    intentionally contains code but not the large contest benchmark files.
    No timer or OpenROAD process is started here.
    """
    sys.path.insert(0, str(ROOT))
    dtla = importlib.import_module("dtla")
    timing_promote = importlib.import_module("timing_promote")
    bench = data_root / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
    dtla.BENCH_ROOT = str(bench)
    timing_promote.BENCH_ROOT = str(bench)
    route_abs = _path(route, data_root)
    crit_abs = _path(crit, data_root)
    if not route_abs.is_file() or not crit_abs.is_file():
        raise FileNotFoundError(f"route/crit not found: {route_abs}, {crit_abs}")
    p = dtla.build(design, suite, str(route_abs), str(crit_abs), max_nets=max_nets)
    p["route_path"] = str(route_abs)
    p["design"] = design
    p["suite"] = suite
    return p


def _criticality(P: dict) -> np.ndarray:
    """Fixed GAP-LA-like pin weight: path count times a logistic slack weight."""
    rn = _np(P["run_net"], np.int64)
    nn = int(rn.max()) + 1 if rn.size else 0
    counts = np.zeros(nn, dtype=np.float64)
    worst = np.full(nn, np.inf, dtype=np.float64)
    pnet = _np(P.get("p_net", []), np.int64)
    ppath = _np(P.get("p_path", []), np.int64)
    slack = _np(P.get("slack0", []), np.float64)
    if pnet.size:
        np.add.at(counts, pnet, 1.0)
        np.minimum.at(worst, pnet, slack[ppath])
    # Positive value means more critical.  The logistic is fixed, not fitted
    # in the DP, and remains well behaved when a problem has no violating path.
    x = np.clip(-np.nan_to_num(worst, nan=0.0, posinf=0.0) / LOGISTIC_TAU_NS, -60.0, 60.0)
    pin = 1.0 + PIN_WEIGHT_MAX / (1.0 + np.exp(-x))
    return pin * (1.0 + np.log1p(counts))


def _score_weights(P: dict) -> tuple[float, float, float, float]:
    """Return contest weights when the local checkout exposes ispd25_score."""
    try:
        sys.path.insert(0, str(ROOT))
        score = importlib.import_module("ispd25_score")
        return tuple(float(x) for x in score.CASES[int(P["case"])][5])
    except (ImportError, KeyError, IndexError, TypeError):
        return (-1.0, -10.0, 300.0, 1.0e-6)


def _candidate_cells(P: dict, runs: np.ndarray, K: int) -> list[list[np.ndarray]]:
    """Build compact cell lists for the requested run IDs."""
    ent_rk = _np(P.get("ent_rk", []), np.int64)
    ent_cell = _np(P.get("ent_cell", []), np.int64)
    by = {}
    for rk, cell in zip(ent_rk.tolist(), ent_cell.tolist()):
        by.setdefault(int(rk), []).append(int(cell))
    out = []
    for r in runs.tolist():
        out.append([np.asarray(by.get(int(r * K + k), []), dtype=np.int64)
                    for k in range(K)])
    return out


class _CostModel:
    """Vectorised local cost model over one net's candidate runs."""

    def __init__(self, P: dict, backend: str = "auto", delay_weight: float = 1.0,
                 power_weight: float | None = None, via_cost: float = DEFAULT_VIA_COST):
        self.P = P
        self.R = len(P["run_net"])
        self.K = int(P["K"])
        self.L = int(P["L"])
        self.d0 = _np(P["d0"], np.float64)
        self.cap = _np(P["cap"], np.float64)
        self.u = _np(P["u"], np.float64)
        self.dcoef = _np(P.get("dcoef", np.zeros((self.R, self.K))), np.float64)
        self.Cz = _np(P.get("Cz", np.ones(self.L)), np.float64)
        self.run_len = _np(P["run_len"], np.float64)
        self.um = float(P.get("um", 2.1))
        self.crit = _criticality(P)
        self.w1, self.w2, self.w3, self.w4 = _score_weights(P)
        self.delay_weight = float(delay_weight)
        self.power_weight = float(self.w3 * DEFAULT_POWER_SCALE if power_weight is None else power_weight)
        self.via_cost = float(via_cost)
        self.demand = self.d0.copy()
        self.backend = "numpy"
        self._torch = None
        self._ent_cell = _np(P.get("ent_cell", []), np.int64)
        self._ent_rk = _np(P.get("ent_rk", []), np.int64)
        self._entries_by_run = [[] for _ in range(self.R)]
        for i, rk in enumerate(self._ent_rk.tolist()):
            self._entries_by_run[int(rk) // self.K].append(i)
        if backend in ("auto", "cuda", "torch"):
            try:
                import torch
                if backend == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable")
                if torch.cuda.is_available() and backend in ("auto", "cuda"):
                    self.backend = "cuda"
                    self._torch = torch
                    self._tdemand = torch.as_tensor(self.demand, dtype=torch.float32, device="cuda")
                    self._tcell = torch.as_tensor(self._ent_cell, dtype=torch.long, device="cuda")
                    self._trk = torch.as_tensor(self._ent_rk, dtype=torch.long, device="cuda")
            except ImportError:
                if backend in ("cuda", "torch"):
                    raise

    def _marginal_numpy(self, cell: np.ndarray) -> np.ndarray:
        d = self.demand[cell]
        c = self.cap[cell]
        u = self.u[cell]
        with np.errstate(over="ignore", invalid="ignore"):
            open_cost = u * np.exp(np.minimum(0.25 * (d + 2.0) - 0.5 * c, 60.0)) * EXP_HALF
            blocked = u * (np.exp(np.minimum(0.75 * (d + 2.0), 60.0))
                           - np.exp(np.minimum(0.75 * d, 60.0)))
        return np.where(c > CAP_EPS, open_cost, blocked)

    def unary(self, runs: np.ndarray, cells: list[list[np.ndarray]]) -> np.ndarray:
        """Return [len(runs), K] costs, including current-demand overflow."""
        entry_ids = [i for r in runs.tolist() for i in self._entries_by_run[int(r)]]
        entry_ids = np.asarray(entry_ids, dtype=np.int64)
        local_rk = self._ent_rk[entry_ids].copy() if entry_ids.size else np.zeros(0, np.int64)
        if local_rk.size:
            # Convert global r*K+k indexing to this net's compact row indexing.
            row_of = {int(r): j for j, r in enumerate(runs.tolist())}
            local_rk = np.asarray([row_of[int(x) // self.K] * self.K + int(x) % self.K
                                   for x in local_rk.tolist()], dtype=np.int64)
        if self.backend == "cuda":
            torch = self._torch
            if entry_ids.size:
                eid = torch.as_tensor(entry_ids, dtype=torch.long, device="cuda")
                cell = self._tcell[eid]
                d = self._tdemand[cell]
                cap = torch.as_tensor(self.cap, dtype=torch.float32, device="cuda")
                uu = torch.as_tensor(self.u, dtype=torch.float32, device="cuda")
                c = cap[cell]
                u = uu[cell]
                cost = torch.where(
                    c > CAP_EPS,
                    u * torch.exp(torch.clamp(0.25 * (d + 2.0) - 0.5 * c, max=60.0)) * float(EXP_HALF),
                    u * (torch.exp(torch.clamp(0.75 * (d + 2.0), max=60.0))
                         - torch.exp(torch.clamp(0.75 * d, max=60.0))),
                )
                all_cost = torch.zeros(len(runs) * self.K, dtype=torch.float32, device="cuda")
                all_cost.index_add_(0, torch.as_tensor(local_rk, dtype=torch.long, device="cuda"), cost)
                ov = all_cost.reshape(len(runs), self.K)
                ov = ov.cpu().numpy().astype(np.float64)
            else:
                ov = np.zeros((len(runs), self.K), dtype=np.float64)
        else:
            # One bincount replaces the inner cell loop and is the hot path on
            # CPU.  Only this net's entries are reduced, so the whole pass is
            # linear in the frozen route rather than in nets-times-route.
            vals = self._marginal_numpy(self._ent_cell[entry_ids]) if entry_ids.size else np.zeros(0)
            all_cost = np.bincount(local_rk, weights=vals, minlength=len(runs) * self.K)
            ov = all_cost.reshape(len(runs), self.K)
        dly = self.delay_weight * self.crit[_np(self.P["run_net"], np.int64)[runs]][:, None] * self.dcoef[runs]
        cap_power = self.power_weight * self.Cz[_np(self.P["cand"], np.int64)[runs].clip(min=0)] \
            * self.run_len[runs, None] * self.um
        valid = _np(self.P["cand"], np.int64)[runs] >= 0
        out = self.w4 * ov + dly + cap_power
        return np.where(valid, out, np.inf)

    def commit(self, runs: np.ndarray, choice: np.ndarray, cells: list[list[np.ndarray]]) -> None:
        for j, r in enumerate(runs.tolist()):
            idx = cells[j][int(choice[j])]
            if idx.size:
                self.demand[idx] += 2.0
                if self.backend == "cuda":
                    self._tdemand[ self._torch.as_tensor(idx, device="cuda", dtype=self._torch.long) ] += 2.0

    def via(self, a: int, b: int) -> float:
        return self.via_cost * abs(int(a) - int(b))

    def pin_stack(self, P: dict, point: int, layer: int) -> float:
        lo = int(_np(P["pt_lo"], np.int64)[point]) if len(P.get("pt_lo", [])) else self.L
        hi = int(_np(P["pt_hi"], np.int64)[point]) if len(P.get("pt_hi", [])) else -1
        if hi < 0 or lo >= self.L:
            return 0.0
        if layer < lo:
            return self.via(layer, lo)
        if layer > hi:
            return self.via(hi, layer)
        return 0.0


def _tree_dp(P: dict, net: int, unary: np.ndarray, runs: np.ndarray,
             model: _CostModel) -> tuple[np.ndarray, float, bool]:
    """Solve one net's run tree; return candidate indices and local cost."""
    p0 = _np(P["run_p0"], np.int64)
    p1 = _np(P["run_p1"], np.int64)
    cand = _np(P["cand"], np.int64)
    K = model.K
    adj: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for r in runs.tolist():
        adj[int(p0[r])].append((int(r), int(p1[r])))
        adj[int(p1[r])].append((int(r), int(p0[r])))
    if not adj:
        return np.zeros(0, dtype=np.int64), 0.0, True
    root = next(iter(adj))
    parent: dict[int, tuple[int, int] | None] = {root: None}
    order = [root]
    edge_to_child: dict[int, tuple[int, int, int]] = {}
    for u in order:
        for r, v in adj[u]:
            if v in parent:
                continue
            parent[v] = (u, r)
            edge_to_child[r] = (u, v, r)
            order.append(v)
    # Loops or disconnected geometry are not expected from a tree route.  A
    # deterministic unary fallback is safer than emitting a disconnected route.
    if len(parent) != len(adj) or len(edge_to_child) != len(runs):
        choice = np.argmin(unary, axis=1).astype(np.int64)
        return choice, float(np.take_along_axis(unary, choice[:, None], 1).sum()), False

    msg: dict[int, np.ndarray] = {}
    best_child: dict[tuple[int, int], int] = {}
    # Reverse node order gives post-order messages.  Each edge's message is
    # indexed by its own candidate slot, not by physical layer number.
    for v in reversed(order[1:]):
        u, r_parent = parent[v]
        edge_cost = unary[np.where(runs == r_parent)[0][0]].copy()
        for zslot in range(K):
            z = int(cand[r_parent, zslot])
            if z < 0:
                edge_cost[zslot] = np.inf
                continue
            total = model.pin_stack(P, v, z)
            for rc, child in adj[v]:
                if rc == r_parent or child not in parent:
                    continue
                child_msg = msg[rc]
                child_layers = cand[rc]
                vals = child_msg + np.array([model.via(z, zz) if zz >= 0 else np.inf
                                             for zz in child_layers])
                ck = int(np.argmin(vals))
                total += float(vals[ck])
                best_child[(r_parent, zslot, rc)] = ck
            edge_cost[zslot] += total
        msg[r_parent] = edge_cost

    # Root has a virtual layer.  Min over all legal layers gives a compact way
    # to account for branch-to-branch via stacks without changing the tree.
    root_children = [(r, v) for r, v in adj[root] if v in parent]
    root_cost = np.full(model.L, 0.0, dtype=np.float64)
    for anchor in range(model.L):
        total = 0.0
        for r, _ in root_children:
            vals = msg[r] + np.array([model.via(anchor, int(z)) if z >= 0 else np.inf
                                      for z in cand[r]])
            total += float(np.min(vals))
        root_cost[anchor] = total
    anchor = int(np.argmin(root_cost))
    choice = np.full(len(runs), 0, dtype=np.int64)
    run_pos = {int(r): i for i, r in enumerate(runs.tolist())}

    def descend(v: int, incoming_layer: int | None) -> None:
        for r, child in adj[v]:
            if child not in parent:
                continue
            vals = msg[r] + np.array([
                model.via(incoming_layer, int(z)) if incoming_layer is not None and z >= 0 else 0.0
                for z in cand[r]
            ])
            k = int(np.argmin(vals))
            choice[run_pos[r]] = k
            descend_edge(child, r, int(cand[r, k]))

    def descend_edge(v: int, incoming_run: int, incoming_layer: int) -> None:
        for r, child in adj[v]:
            if r == incoming_run or child not in parent:
                continue
            vals = msg[r] + np.array([model.via(incoming_layer, int(z)) if z >= 0 else np.inf
                                      for z in cand[r]])
            k = int(np.argmin(vals))
            choice[run_pos[r]] = k
            descend_edge(child, r, int(cand[r, k]))

    descend(root, anchor)
    local = float(np.take_along_axis(unary, choice[:, None], 1).sum())
    return choice, local + float(root_cost[anchor]), True


def assign(P: dict, backend: str = "auto", delay_weight: float = 1.0,
           power_weight: float | None = None, via_cost: float = DEFAULT_VIA_COST) -> tuple[np.ndarray, dict]:
    """Assign all frozen runs once and return candidate-slot choices."""
    t0 = time.perf_counter()
    rn = _np(P["run_net"], np.int64)
    R, K = len(rn), int(P["K"])
    choice = _np(P.get("run_cur", np.zeros(R, np.int64)), np.int64).copy()
    model = _CostModel(P, backend, delay_weight, power_weight, via_cost)
    all_cells = _candidate_cells(P, np.arange(R, dtype=np.int64), K)
    nets = np.unique(rn)
    crit = model.crit
    order = sorted((int(n) for n in nets), key=lambda n: (-float(crit[n]), n))
    used_fallback = 0
    local_cost = 0.0
    for n in order:
        runs = np.flatnonzero(rn == n).astype(np.int64)
        cells = [all_cells[int(r)] for r in runs.tolist()]
        unary = model.unary(runs, cells)
        c, cost, tree_ok = _tree_dp(P, n, unary, runs, model)
        if not tree_ok:
            used_fallback += 1
        choice[runs] = c
        model.commit(runs, c, cells)
        local_cost += cost
    elapsed = time.perf_counter() - t0
    stats = {
        "backend": model.backend,
        "nets": int(len(order)),
        "runs": int(R),
        "fallback_nets": int(used_fallback),
        "wall_s": elapsed,
        "criticality": "path_count_logistic_slack",
        "logistic_tau_ns": LOGISTIC_TAU_NS,
        "pin_weight_max": PIN_WEIGHT_MAX,
        "delay_weight": float(delay_weight),
        "power_weight": float(model.power_weight),
        "via_cost": float(via_cost),
        "local_cost": float(local_cost),
    }
    return choice, stats


def write_route(P: dict, choice: np.ndarray, out_path: str | os.PathLike[str]) -> dict:
    """Write a contest-format route, preserving all unassigned net blocks."""
    t0 = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import route_export as RE
    timing = importlib.import_module("timing_promote")
    route_path = _path(P["route_path"])
    nets = timing.read_route25(str(route_path))
    t_read = time.perf_counter()
    rn = _np(P["run_net"], np.int64)
    cand = _np(P["cand"], np.int64)
    dirs = list(P["dirs"])
    by_net: dict[int, list[int]] = collections.defaultdict(list)
    for r, n in enumerate(rn.tolist()):
        by_net[int(n)].append(r)
    new: dict[int, list[str]] = {}
    for n, runs in by_net.items():
        wires: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
        node_layers: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
        point_xy: dict[int, tuple[int, int]] = {}
        for r in runs:
            k = int(choice[r])
            z = int(cand[r, k])
            if z < 0:
                continue
            d = int(P["run_dir"][r]); f = int(P["run_fixed"][r])
            lo = int(P["run_lo"][r]); hi = int(P["run_hi"][r])
            wires[(z, f)].append((lo, hi))
            a = (lo, f) if d == 0 else (f, lo)
            b = (hi, f) if d == 0 else (f, hi)
            point_xy[int(P["run_p0"][r])] = a
            point_xy[int(P["run_p1"][r])] = b
            node_layers[a].add(z); node_layers[b].add(z)
        net_nodes = P.get("net_nodes")
        if net_nodes is not None and n < len(net_nodes):
            for xy, span in net_nodes[n].items():
                node_layers[xy].update((int(span[0]), int(span[1])))
        else:
            # Frozen files intentionally omit the Python bookkeeping object.
            # Reconstruct pin stack endpoints from the run point IDs and the
            # compact pt_lo/pt_hi arrays.
            plo = _np(P.get("pt_lo", []), np.int64)
            phi = _np(P.get("pt_hi", []), np.int64)
            for pid, xy in point_xy.items():
                if pid < len(plo) and int(phi[pid]) >= 0:
                    node_layers[xy].update((int(plo[pid]), int(phi[pid])))
        buf = []
        for (z, f), ivs in wires.items():
            for lo, hi in ivs:
                if dirs[z] == 0:
                    buf.append(RE.seg_line(lo, f, z, hi, f, z, "ispd25").strip())
                else:
                    buf.append(RE.seg_line(f, lo, z, f, hi, z, "ispd25").strip())
        for (x, y), zs in node_layers.items():
            if zs:
                for z in range(min(zs), max(zs)):
                    buf.append(RE.seg_line(x, y, z, x, y, z + 1, "ispd25").strip())
        route_i = int(P["net_route_idx"][n])
        new[route_i] = buf
    t_apply = time.perf_counter()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for line in new.get(i, lines):
                fh.write(line + "\n")
            fh.write(")\n")
    t_write = time.perf_counter()
    return {"route": str(out), "nets_changed": len(new), "route_nets": len(nets),
            "route_read_s": t_read - t0, "apply_s": t_apply - t_read,
            "write_s": t_write - t_apply, "write_total_s": t_write - t0}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--problem", help="frozen *.instantgr.pt input")
    inp.add_argument("--route", help="InstantGR route input")
    ap.add_argument("--base-route", help="override a missing route_path stored in --problem")
    ap.add_argument("--crit", help="one precomputed STA report; required with --route")
    ap.add_argument("--design")
    ap.add_argument("--suite", choices=("visible", "blind"), default="visible")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--max-nets", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("auto", "numpy", "cuda"), default="auto")
    ap.add_argument("--delay-weight", type=float, default=1.0)
    ap.add_argument("--power-weight", type=float, default=None)
    ap.add_argument("--via-cost", type=float, default=DEFAULT_VIA_COST)
    ap.add_argument("--json", help="optional timing/statistics JSON")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    t0 = time.perf_counter()
    if a.problem:
        P = load_problem(_path(a.problem))
        if a.base_route:
            P["route_path"] = str(_path(a.base_route))
    else:
        if not a.crit or not a.design:
            raise SystemExit("--route requires --crit and --design")
        P = build_problem_from_route(a.route, a.crit, a.design, a.suite, a.max_nets,
                                     _path(a.data_root))
    t_load = time.perf_counter() - t0
    choice, stats = assign(P, a.backend, a.delay_weight, a.power_weight, a.via_cost)
    t_dp = time.perf_counter() - t0 - t_load
    write = write_route(P, choice, a.out)
    rec = {"input": a.problem or a.route, "crit": a.crit, "load_s": t_load,
           "dp_s": t_dp, "stats": stats, **write}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

