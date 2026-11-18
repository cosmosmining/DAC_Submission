#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
route_export.py — turn a 2-D routing solution into a legal 3-D ISPD'24/'25
global-routing solution, using a layer assignment that minimizes the TRUE
contest congestion cost. Shared back-end for all methods so they are compared
on their 2-D decisions only:
  native-OpenROAD (segments from its own GR), standard DGR, GNN-DGR, iMF, and
  timing-weighted DGR (2-D guides written by main_stochastic.py / dgr_fast.py).

Why (audit, 2026-09): macros block m1-m4 in 23-46% of ISPD'25 GCells while m5+
stay free, and m7 / m10 carry power straps (15% / 31% zero capacity). A 2-D
solution cannot see this — the damage happens in layer assignment. Each 2-D
run is placed by a DP whose costs are the evaluator's exact marginals:
  one more wire on GCell (x,y,z)   (demand d in half-units, evaluator.cpp)
    open    c > 1e-3 : u_z * exp(0.25 d - 0.5 c) * (e^0.5 - 1)
    blocked c <= 1e-3: u_z * (exp(0.75 (d+2)) - exp(0.75 d))   [u_z e^1.5 if d = 0]
  one via z -> z+1 at (x,y): unit_via + stacked-via demand marginal on layer z
                             (none on layer 0, which the evaluator never scores)
  timing-weighted DGR: + alpha * (crit_net - 1) * R_z / R_m2 * run_length
Nets are committed sequentially (critical, then long nets first), so later
nets price the congestion earlier nets created. Same-net overlapping wires
are merged (the evaluator counts every segment occurrence).

Connectivity follows evaluator.cpp::read_gr_solution / check_connectivity:
  * a wire on layer 0 is a violation (metal1 is not a routing layer) -> open;
    ISPD'25 vias must be single-layer (zh == zl + 1);
  * the trace starts only from pin-0 access points the route itself marks, so
    each pin GCell gets a via stack spanning ALL pin layers there (two pins in
    one GCell on metal1 and metal3 need a 0->2 stack), and a multi-pin net
    whose pins share one GCell and layer still gets one via to mark them —
    an empty route is an open.

Input 2-D guide (util.write_CUGR_input):   "<net> <net_ID> <child_pin_idx>"
followed by "x y" turning points, parent -> child. The first path per
(net, child) is kept (the highest-probability candidate is written first).
A net whose guide paths do not join all of its pin GCells falls back to
pattern-L (counted in the log).

  python3 route_export.py --cap X.cap --net X.net --guide G.txt --out X.route
  python3 route_export.py --cap X.cap --net X.net --synthetic_l --out X.route
  python3 route_export.py --self_test
NEW file; reuses ispd25_adapter / congestion_breakdown / timing_weights read-only.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ispd25_adapter import parse_cap_auto, parse_net_ispd25, net_slacks  # noqa: E402

GCELL_DBU = 4200
CAP_EPS = 1e-3
EXP_CLIP = 700.0
E05M1 = math.exp(0.5) - 1.0
E025M1 = math.exp(0.25) - 1.0
# ISPD'25 NanGate45 setRC.tcl resistance per layer (used only by the timing term)
RC_RES = {"metal1": 5.4286e-03, "metal2": 3.5714e-03, "metal3": 3.5714e-03,
          "metal4": 1.5e-03, "metal5": 1.5e-03, "metal6": 1.5e-03,
          "metal7": 1.875e-04, "metal8": 1.875e-04, "metal9": 3.75e-05,
          "metal10": 3.75e-05}
EVAL25 = os.path.join(ROOT, "ispd_bench", "ISPD2025", "ISPD2025_benchmarks",
                      "evaluation", "evaluator")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  inputs
# ════════════════════════════════════════════════════════════════════════
def read_guide(path):
    """util.write_CUGR_input guide -> {net_name: {child_idx: [bend points]}}.
    Per selected candidate it writes "<net> <net_ID> <child_idx>" followed ONLY by
    the candidate's bend points (via_map columns, reversed to parent -> child); a
    straight candidate has no point lines. The end points are the child pin and
    its parent on convert()'s nearest-neighbour chain (guide_polys). The first
    candidate per (net, child) is kept (the highest-probability one is written first)."""
    out = defaultdict(dict)
    cur = None
    with open(path) as fh:
        for raw in fh:
            t = raw.split()
            if len(t) == 3:
                child, k = out[t[0]], int(t[2])
                cur = None if k in child else child.setdefault(k, [])
            elif len(t) == 2 and cur is not None:
                cur.append((int(t[0]), int(t[1])))
    return out


def pin_nodes(net):
    """convert()-identical pin locations: each pin -> its lowest access point
    (l, x, y). Pins are grouped by GCell; a node keeps the (lowest, highest)
    layer of its pins so the via stack there reaches every pin."""
    nodes = {}
    for aps in net["pins"]:
        l0, x0, y0 = min(aps)
        prev = nodes.get((x0, y0))
        nodes[(x0, y0)] = (l0, l0) if prev is None else (min(prev[0], l0),
                                                         max(prev[1], l0))
    return nodes


def nn_chain_order(points):
    """convert()'s nearest-neighbour chain order (same tie-breaking)."""
    order, rest = [0], set(range(1, len(points)))
    while rest:
        lx, ly = points[order[-1]]
        nxt = min(rest, key=lambda j: abs(points[j][0] - lx) + abs(points[j][1] - ly))
        order.append(nxt)
        rest.remove(nxt)
    return [points[j] for j in order]


def nn_chain(points):
    """-> [(child, parent), ...] along convert()'s chain."""
    pts = nn_chain_order(points)
    return [(pts[i], pts[i - 1]) for i in range(1, len(pts))]


def _rectilinear(poly):
    return all(a[0] == b[0] or a[1] == b[1] for a, b in zip(poly, poly[1:]))


def guide_polys(nodes, children, tree="chain"):
    """{child_idx: bends} -> polylines parent -> bends -> child on the net's tree
    (tree_topology.tree_order, which must match the pack's pin order); None when the
    guide does not fit this net."""
    if not children:
        return None
    from tree_topology import tree_order
    order, parent = tree_order(list(nodes.keys()), tree)
    polys = []
    for k, bends in children.items():
        if not 1 <= k < len(order):
            return None
        par = order[parent[k]]
        poly = [par] + list(bends) + [order[k]]
        if not _rectilinear(poly):
            poly = [par] + list(reversed(bends)) + [order[k]]
            if not _rectilinear(poly):
                return None
        polys.append(poly)
    return polys


def tree_connects(nodes, polys):
    """True if the polylines (joined at their endpoints) connect every pin GCell."""
    if len(nodes) <= 1:
        return True
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for pts in polys:
        if pts:
            parent[find(pts[0])] = find(pts[-1])
    return len({find(p) for p in nodes}) == 1


def to_runs(pts):
    """polyline -> straight runs (dir, fixed, lo, hi, sx, sy, ex, ey); dir 0 = along x."""
    runs = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 != x1 and y0 != y1:                       # non-rectilinear: x first
            runs.append((0, y0, min(x0, x1), max(x0, x1), x0, y0, x1, y0))
            runs.append((1, x1, min(y0, y1), max(y0, y1), x1, y0, x1, y1))
        elif x0 != x1:
            runs.append((0, y0, min(x0, x1), max(x0, x1), x0, y0, x1, y1))
        elif y0 != y1:
            runs.append((1, x0, min(y0, y1), max(y0, y1), x0, y0, x1, y1))
    return runs


def segments_to_polys(segs, nodes):
    """3-D route segments of one net (any layer; vias ignored) -> 2-D single-run
    polylines split at every run end, same-net crossing and pin GCell, so the
    tree stays connected when each run gets its own layer. Imports the 2-D
    topology of native OpenROAD segments (or any other router's route)."""
    runs = defaultdict(list)                     # (dir, fixed) -> [(lo, hi)]
    for (xl, yl, zl, xh, yh, zh) in segs:
        if zl != zh:
            continue
        if yl == yh and xh > xl:
            runs[(0, yl)].append((xl, xh))
        elif xl == xh and yh > yl:
            runs[(1, xl)].append((yl, yh))
    merged = []
    for (d, fixed), ivs in runs.items():
        ivs.sort()
        lo, hi = ivs[0]
        for a, b in ivs[1:]:
            if a <= hi:
                hi = max(hi, b)
            else:
                merged.append((d, fixed, lo, hi))
                lo, hi = a, b
        merged.append((d, fixed, lo, hi))
    keys = set(nodes)
    for d, fixed, a, b in merged:
        keys.add((a, fixed) if d == 0 else (fixed, a))
        keys.add((b, fixed) if d == 0 else (fixed, b))
    hs = [m for m in merged if m[0] == 0]
    vs = [m for m in merged if m[0] == 1]
    for _, y, xa, xb in hs:
        for _, x, ya, yb in vs:
            if xa <= x <= xb and ya <= y <= yb:
                keys.add((x, y))
    by_row, by_col = defaultdict(list), defaultdict(list)
    for (x, y) in keys:
        by_row[y].append(x)
        by_col[x].append(y)
    polys = []
    for d, fixed, a, b in merged:
        on = by_row[fixed] if d == 0 else by_col[fixed]
        cuts = sorted({a, b} | {c for c in on if a < c < b})
        for u, v in zip(cuts, cuts[1:]):
            polys.append([(u, fixed), (v, fixed)] if d == 0 else [(fixed, u), (fixed, v)])
    return polys


# ════════════════════════════════════════════════════════════════════════
#  contest-exact layer assignment
# ════════════════════════════════════════════════════════════════════════
class LayerAssigner:
    def __init__(self, cap, alpha_timing=0.0, naive=False):
        self.L, self.X, self.Y = cap["L"], cap["X"], cap["Y"]
        self.c = np.stack([l["cap"] for l in cap["layers"]]).astype(np.float64)
        self.d = np.zeros_like(self.c)                       # demand, half-units
        self.dirs = [l["dir"] for l in cap["layers"]]
        self.u = np.asarray(cap["short_costs"], dtype=np.float64)
        self.uvia = float(cap["unit_via"])
        self.H = [z for z in range(1, self.L) if self.dirs[z] == 0]
        self.V = [z for z in range(1, self.L) if self.dirs[z] == 1]
        names = [l["name"] for l in cap["layers"]]
        r = np.array([RC_RES.get(n, np.nan) for n in names], dtype=np.float64)
        ref = np.nanmax(r[1:]) if np.isfinite(r[1:]).any() else 1.0
        self.rrel = np.where(np.isfinite(r), r / ref, 1.0)
        self.alpha = float(alpha_timing)
        self.naive = naive

    def wire_marginal(self, z, sl):
        dz, cz, u = self.d[z][sl], self.c[z][sl], self.u[z]
        with np.errstate(over="ignore", invalid="ignore"):
            open_ = u * np.exp(np.minimum(0.25 * dz - 0.5 * cz, EXP_CLIP)) * E05M1
            blk = np.where(dz > 0,
                           u * (np.exp(np.minimum(0.75 * (dz + 2), EXP_CLIP))
                                - np.exp(np.minimum(0.75 * dz, EXP_CLIP))),
                           u * math.exp(1.5))
        return np.where(cz > CAP_EPS, open_, blk)

    def run_cost(self, run, z, crit):
        dr, fixed, a, b = run[:4]
        sl = (slice(a, b), fixed) if dr == 0 else (fixed, slice(a, b))
        cost = float(self.wire_marginal(z, sl).sum())
        if self.alpha and crit > 1.0:
            cost += self.alpha * (crit - 1.0) * self.rrel[z] * (b - a)
        return cost

    def via_cost(self, x, y, za, zb):
        lo, hi = (za, zb) if za <= zb else (zb, za)
        tot = 0.0
        for z in range(lo, hi):
            tot += self.uvia
            if z != 0 and self.dirs[z] in (0, 1):        # layer 0 is never scored
                dz, cz, u = self.d[z, x, y], self.c[z, x, y], self.u[z]
                if cz > CAP_EPS:
                    tot += 2.0 * u * math.exp(min(0.25 * dz - 0.5 * cz, EXP_CLIP)) * E025M1
                else:
                    base = math.exp(min(0.75 * dz, EXP_CLIP)) if dz > 0 else 0.0
                    tot += 2.0 * u * (math.exp(min(0.75 * (dz + 1), EXP_CLIP)) - base)
        return tot

    def stack_cost(self, x, y, span, z):
        """via cost of extending the pin stack span=(lo, hi) at (x, y) to layer z
        (span None: no pin there, the stack is free to start at z)."""
        if span is None:
            return 0.0
        lo, hi = span
        if z > hi:
            return self.via_cost(x, y, hi, z)
        if z < lo:
            return self.via_cost(x, y, z, lo)
        return 0.0

    def assign_path(self, runs, span_start, span_end, crit=1.0):
        """DP over runs -> (layers, cost)."""
        R = len(runs)
        if R == 0:
            return [], 0.0
        cands = [self.H if r[0] == 0 else self.V for r in runs]
        if self.naive:                                   # via-minimizing baseline:
            return [c[0] for c in cands], 0.0            # always the lowest layer
        rc = [[self.run_cost(runs[i], z, crit) for z in cands[i]] for i in range(R)]
        sx, sy = runs[0][4], runs[0][5]
        best = [rc[0][k] + self.stack_cost(sx, sy, span_start, z)
                for k, z in enumerate(cands[0])]
        back = []
        for i in range(1, R):
            tx, ty = runs[i][4], runs[i][5]
            nb, bk = [], []
            for k, z in enumerate(cands[i]):
                vals = [best[j] + self.via_cost(tx, ty, zp, z)
                        for j, zp in enumerate(cands[i - 1])]
                j = int(np.argmin(vals))
                nb.append(vals[j] + rc[i][k])
                bk.append(j)
            best = nb
            back.append(bk)
        ex, ey = runs[-1][6], runs[-1][7]
        final = [best[k] + self.stack_cost(ex, ey, span_end, z)
                 for k, z in enumerate(cands[-1])]
        k = int(np.argmin(final))
        layers = [0] * R
        layers[-1] = cands[-1][k]
        for i in range(R - 1, 0, -1):
            k = back[i - 1][k]
            layers[i - 1] = cands[i - 1][k]
        return layers, float(final[int(np.argmin(final))])

    def commit_net(self, wires, vias):
        """wires: {(z, fixed): [(a, b), ...]} merged; vias: {(x, y, z)} single-layer."""
        X, Y, d = self.X, self.Y, self.d
        covered = defaultdict(list)
        for (z, fixed), ivs in wires.items():
            for a, b in ivs:
                if self.dirs[z] == 0:
                    d[z, a:b, fixed] += 2
                else:
                    d[z, fixed, a:b] += 2
                covered[z].append((fixed, a, b))
        for (x, y, z) in vias:
            if z == 0:                                   # layer 0 demand is never scored
                continue
            dz = self.dirs[z]
            hit = False
            for fixed, lo, hi in covered.get(z, ()):
                if (dz == 0 and fixed == y and lo <= x <= hi) or \
                        (dz == 1 and fixed == x and lo <= y <= hi):
                    hit = True
                    break
            if hit:
                continue
            if dz == 0:
                if 0 < x < X - 1:
                    d[z, x - 1, y] += 1
                    d[z, x, y] += 1
                elif x > 0:
                    d[z, x - 1, y] += 2
                elif x < X - 1:
                    d[z, x, y] += 2
            elif dz == 1:
                if 0 < y < Y - 1:
                    d[z, x, y - 1] += 1
                    d[z, x, y] += 1
                elif y > 0:
                    d[z, x, y - 1] += 2
                elif y < Y - 1:
                    d[z, x, y] += 2


def route_net(la, nodes, polys, crit, mark=True):
    """3-D wires + single-layer vias for one net. mark=True (multi-pin net):
    guarantee the route marks the pins even when all share one GCell/layer."""
    wires_raw = defaultdict(list)
    node_layers = defaultdict(set)
    for pts in polys:
        runs = to_runs(pts)
        if not runs:
            continue
        layers, _ = la.assign_path(runs, nodes.get(pts[0]), nodes.get(pts[-1]), crit)
        for r, z in zip(runs, layers):
            wires_raw[(z, r[1])].append((r[2], r[3]))
            node_layers[(r[4], r[5])].add(z)
            node_layers[(r[6], r[7])].add(z)
    for xy, (lo, hi) in nodes.items():
        node_layers[xy].update((lo, hi))
    vias = set()
    for (x, y), zs in node_layers.items():
        for z in range(min(zs), max(zs)):
            vias.add((x, y, z))
    if mark and not wires_raw and not vias and nodes:
        (x, y), (lo, _) = next(iter(nodes.items()))
        vias.add((x, y, lo if lo < la.L - 1 else lo - 1))
    wires = {}
    for key, ivs in wires_raw.items():
        ivs.sort()
        merged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        wires[key] = [tuple(m) for m in merged]
    la.commit_net(wires, vias)
    return wires, vias


def l_polys(la, nodes, crit, tree="chain"):
    """pattern-L baseline on the net's tree (chain or MST), per edge the cheaper L."""
    from tree_topology import tree_pairs
    pts = list(nodes.keys())
    polys = []
    for (cx, cy), (px, py) in tree_pairs(pts, tree) if len(pts) > 1 else []:
        best = None
        for mid in ((cx, py), (px, cy)):
            poly = [(px, py), mid, (cx, cy)]
            runs = to_runs(poly)
            _, cost = la.assign_path(runs, nodes[(px, py)], nodes[(cx, cy)], crit)
            if best is None or cost < best[0]:
                best = (cost, poly)
        polys.append(best[1])
    return polys


def seg_line(xl, yl, zl, xh, yh, zh, fmt):
    if fmt == "ispd24":
        return f"{xl} {yl} {zl} {xh} {yh} {zh}\n"
    c, h = GCELL_DBU, GCELL_DBU // 2
    return f"{xl*c+h} {yl*c+h} metal{zl+1} {xh*c+h} {yh*c+h} metal{zh+1}\n"


def write_net(fh, name, wires, vias, dirs, fmt):
    fh.write(f"{name}\n(\n")
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            if dirs[z] == 0:
                fh.write(seg_line(a, fixed, z, b, fixed, z, fmt))
            else:
                fh.write(seg_line(fixed, a, z, fixed, b, z, fmt))
    for (x, y, z) in sorted(vias):
        fh.write(seg_line(x, y, z, x, y, z + 1, fmt))
    fh.write(")\n")


def export(cap_path, net_path, out_path, guide=None, fmt="ispd25",
           alpha=0.0, naive=False, limit=None, segments=None, tree="chain"):
    """-> (cap, nets, stats). guide: 2-D DGR guide; segments: 3-D route whose
    2-D topology is imported; neither: pattern-L."""
    t0 = time.time()
    cap = parse_cap_auto(cap_path)
    nets = parse_net_ispd25(net_path, max_nets=limit)
    log(f"loaded {cap['format']} cap {cap['X']}x{cap['Y']}x{cap['L']}, "
        f"{len(nets):,} nets ({time.time()-t0:.1f}s)")
    crit = np.ones(len(nets))
    if alpha > 0:
        import timing_weights as TW
        sl = np.array(net_slacks(nets), dtype=np.float64)
        fin = sl[np.isfinite(sl)]
        if fin.size:
            sl = np.where(np.isfinite(sl), sl, fin.max())
            crit = TW.criticality_from_slack(sl, mode="exp", w_max=8.0)
    paths = read_guide(guide) if guide else None
    seg_by_net = None
    if segments:
        from congestion_breakdown import iter_route
        seg_by_net = {name: segs for name, segs in iter_route(segments)}
        log(f"imported 2-D topology of {len(seg_by_net):,} nets from {segments}")
    la = LayerAssigner(cap, alpha_timing=alpha, naive=naive)
    jobs = []
    stats = {"nets": len(nets), "fallback_L": 0, "guide_nets": 0, "local_nets": 0}
    for i, n in enumerate(nets):
        nodes = pin_nodes(n)
        xs = [p[0] for p in nodes] or [0]
        ys = [p[1] for p in nodes] or [0]
        jobs.append((-float(crit[i]), -((max(xs) - min(xs)) + (max(ys) - min(ys))), i, nodes))
    jobs.sort()
    t0 = time.time()
    with open(out_path, "w") as fh:
        for k, (_, _, i, nodes) in enumerate(jobs):
            n = nets[i]
            if len(nodes) <= 1:
                polys = []
                stats["local_nets"] += 1
            elif seg_by_net is not None:
                polys = segments_to_polys(seg_by_net.get(n["name"], []), nodes)
                if tree_connects(nodes, polys):
                    stats["guide_nets"] += 1
                else:
                    polys = l_polys(la, nodes, float(crit[i]), tree)
                    stats["fallback_L"] += 1
            elif paths is not None:
                polys = guide_polys(nodes, paths.get(n["name"]), tree)
                if polys is not None and tree_connects(nodes, polys):
                    stats["guide_nets"] += 1
                else:
                    polys = l_polys(la, nodes, float(crit[i]), tree)
                    stats["fallback_L"] += 1
            else:
                polys = l_polys(la, nodes, float(crit[i]), tree)
            wires, vias = route_net(la, nodes, polys, float(crit[i]),
                                    mark=len(n["pins"]) > 1)
            write_net(fh, n["name"], wires, vias, la.dirs, fmt)
            if (k + 1) % 50000 == 0:
                log(f"  routed {k+1:,}/{len(jobs):,} nets ({time.time()-t0:.0f}s)")
    log(f"wrote {out_path}: {stats} ({time.time()-t0:.1f}s)")
    return cap, nets, stats


# ════════════════════════════════════════════════════════════════════════
#  connectivity check (evaluator.cpp::check_connectivity, Python)
# ════════════════════════════════════════════════════════════════════════
def check_connectivity(cap, nets, route_path):
    from congestion_breakdown import iter_route
    dirs = [l["dir"] for l in cap["layers"]]
    by_name = {n["name"]: n for n in nets}
    opens, seen = 0, set()
    for name, segs in iter_route(route_path):
        seen.add(name)
        flag, violation = set(), False
        for (xl, yl, zl, xh, yh, zh) in segs:
            if zl != zh:
                for z in range(zl, zh):
                    flag.add((xl, yl, z))
                    flag.add((xl, yl, z + 1))
            elif zl == 0:                      # evaluator: layer 0 is not routable
                violation = True
            elif dirs[zl] == 0:
                for x in range(xl, xh + 1):
                    flag.add((x, yl, zl))
            else:
                for y in range(yl, yh + 1):
                    flag.add((xl, y, zl))
        if violation:
            opens += 1
            continue
        pins = by_name[name]["pins"]
        stack = [(x, y, z) for (z, x, y) in pins[0] if (x, y, z) in flag]
        traced = set(stack)
        while stack:
            x, y, z = stack.pop()
            nbrs = [(x, y, z - 1), (x, y, z + 1)]
            if z != 0 and dirs[z] == 0:        # no in-plane moves on layer 0
                nbrs += [(x - 1, y, z), (x + 1, y, z)]
            elif z != 0 and dirs[z] == 1:
                nbrs += [(x, y - 1, z), (x, y + 1, z)]
            for p in nbrs:
                if p in flag and p not in traced:
                    traced.add(p)
                    stack.append(p)
        for aps in pins[1:]:
            if not any((x, y, z) in traced for (z, x, y) in aps):
                opens += 1
                break
    missing = sum(1 for n in nets if n["name"] not in seen)
    return opens, missing


# ════════════════════════════════════════════════════════════════════════
#  self-test
# ════════════════════════════════════════════════════════════════════════
def _tiny_case(dirpath):
    X, Y = 8, 6
    names = ["metal1", "metal2", "metal3", "metal4", "metal5", "metal6"]
    dirs = [0, 1, 0, 1, 0, 1]
    lines = [f"6 {X} {Y}", "0.5 4 5 5 5 5 5 5 -1 -10 300 7e-07",
             " ".join(["4200"] * (X - 1)), " ".join(["4200"] * (Y - 1))]
    for z, (nm, dr) in enumerate(zip(names, dirs)):
        lines.append(f"{nm} {dr} 1")
        for y in range(Y):
            row = []
            for x in range(X):
                macro = (3 <= x <= 5) and (1 <= y <= 4) and z <= 3
                row.append("0" if macro else ("4" if z <= 3 else "3"))
            lines.append(" ".join(row))
    cap = os.path.join(dirpath, "t.cap")
    open(cap, "w").write("\n".join(lines) + "\n")
    net = os.path.join(dirpath, "t.net")
    # net names need > 1 character: evaluator read_nets ignores shorter header lines
    open(net, "w").write(
        "netA\n(\nA/p0, -2e-10, [(0, 0, 2)]\nA/p1, 1e-10, [(0, 7, 2)]\n)\n"    # crosses macro
        "netB\n(\nB/p0, 5e-10, [(0, 1, 0)]\nB/p1, 5e-10, [(1, 1, 5)]\n"
        "B/p2, 5e-10, [(0, 6, 5), (0, 7, 5)]\n)\n"
        "netC\n(\nC/p0, 1e-9, [(0, 6, 0)]\nC/p1, 1e-9, [(2, 6, 0)]\n)\n"        # one GCell, m1+m3
        "netD\n(\nD/p0, 1e-9, [(0, 2, 5)]\nD/p1, 1e-9, [(0, 2, 5)]\n)\n"        # one GCell, m1 only
        "netE\n(\nE/p0, 1e-9, [(0, 0, 0)]\n)\n"                                 # single pin
        "netF\n(\nF/p0, 1e-9, [(0, 0, 0)]\nF/p1, 1e-9, [(0, 2, 3)]\n)\n")       # bent guide
    guide = os.path.join(dirpath, "t.guide")
    # util.write_CUGR_input format: header, then bend points only (parent -> child)
    open(guide, "w").write("netA 0 1\n"                 # straight candidate: no point lines
                           "netB 1 1\n"                 # netB misses child 2 -> fallback
                           "netF 5 1\n0 3\n"            # (0,0) -> bend (0,3) -> (2,3)
                           "netF 5 1\n2 0\n")           # lower-probability duplicate: ignored
    return cap, net, guide


def self_test():
    from congestion_breakdown import analyze
    with tempfile.TemporaryDirectory() as td:
        cap_p, net_p, guide_p = _tiny_case(td)
        results = {}
        for tag, kw in (("synthetic_l", {}), ("guide", {"guide": guide_p}),
                        ("naive_lowest_layer", {"naive": True}),
                        ("ispd24_format", {"fmt": "ispd24"})):
            out = os.path.join(td, f"{tag}.route")
            cap, nets, stats = export(cap_p, net_p, out, **kw)
            opens, missing = check_connectivity(cap, nets, out)
            rep, _ = analyze(cap_p, out, top=0)
            blk = sum(l["blocked_gcells_used"] for l in rep["layers"])
            results[tag] = (opens, missing, blk, rep["summary"]["overflow_cost"],
                            rep["summary"]["malformed_segments"], out, stats,
                            rep["summary"]["overflow_parts"]["blocked"])
        for tag in results:
            opens, missing, blk, of, bad = results[tag][:5]
            assert opens == 0 and missing == 0 and bad == 0, (tag, results[tag])
        print("  [PASS] all nets connected (evaluator rule) incl. same-GCell m1+m3 pins, "
              "same-GCell m1 pins, single-pin net; 0 malformed")
        for tag in ("synthetic_l", "guide", "ispd24_format"):
            assert results[tag][2] == 0, (tag, "macro GCells must be avoided", results[tag])
        print("  [PASS] contest-aware DP routes net A over the macro on metal5, "
              "0 blocked GCells used")
        st = results["guide"][6]
        assert st["guide_nets"] == 2 and st["fallback_L"] == 1 and st["local_nets"] == 3, st
        import re
        txt = open(results["guide"][5]).read()
        f_block = txt[txt.index("netF\n"):].split(")\n", 1)[0]
        assert re.search(r"^2100 2100 metal\d+ 2100 14700 metal\d+$", f_block, re.M), f_block
        assert re.search(r"^2100 14700 metal\d+ 10500 14700 metal\d+$", f_block, re.M), f_block
        print("  [PASS] guides used for A (straight) and F (bend point, first candidate); "
              "B (guide misses a pin) falls back to pattern-L")
        n_blk, n_of, n_bc = (results["naive_lowest_layer"][i] for i in (2, 3, 7))
        s_of, s_bc = results["synthetic_l"][3], results["synthetic_l"][7]
        assert n_blk > 0 and n_bc > 0 and s_bc == 0 and n_of > s_of, results
        print(f"  [PASS] lowest-layer baseline uses {n_blk} blocked GCells (blocked cost "
              f"{n_bc:.1f} vs 0); overflow cost {n_of:.1f} vs {s_of:.1f}")
        assert results["synthetic_l"][3] == results["ispd24_format"][3]
        print("  [PASS] ISPD'25 DBU and ISPD'24 GCell outputs score identically")
        seg_out = os.path.join(td, "reimport.route")
        cap, nets, st = export(cap_p, net_p, seg_out, segments=results["synthetic_l"][5])
        o, m = check_connectivity(cap, nets, seg_out)
        assert o == 0 and m == 0 and st["fallback_L"] == 0 and st["guide_nets"] == 3, (o, m, st)
        print("  [PASS] segments mode: a 3-D route's 2-D topology re-imported, connected, "
              "no pattern-L fallback")
        bad_route = os.path.join(td, "metal1_wire.route")
        open(bad_route, "w").write("netA\n(\n0 2 0 7 2 0\n0 2 0 0 2 1\n)\n")
        o, m = check_connectivity(cap, nets, bad_route)
        assert o == 1 and m == len(nets) - 1, (o, m)
        print("  [PASS] a wire on metal1 (layer 0) is an open, as in the evaluator")
        if os.path.exists(EVAL25):
            for tag in ("synthetic_l", "guide"):
                txt = subprocess.run([EVAL25, cap_p, net_p, results[tag][5]],
                                     capture_output=True, text=True, timeout=120).stdout
                lines = [l for l in txt.splitlines() if "open nets" in l or "incompleted" in l]
                assert len(lines) == 2 and all(l.strip().endswith(": 0") for l in lines), txt
                ref = [l for l in txt.splitlines() if l.startswith("overflow cost")]
                ours = results[tag][3]
                off = float(ref[0].split()[-1]) if ref else float("nan")
                assert abs(off - ours) <= 1e-3 * max(1.0, abs(ours)), (tag, off, ours)
                print(f"  [PASS] OFFICIAL ISPD'25 evaluator ({tag}): "
                      f"{' | '.join(l.strip() for l in lines)}; overflow cost "
                      f"{off:.4f} == port {ours:.4f}")
        print("checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap")
    ap.add_argument("--net")
    ap.add_argument("--guide", help="2-D guide from main_stochastic.py / dgr_fast.py")
    ap.add_argument("--segments", help="3-D route (e.g. OpenROAD write_global_route_segments) "
                                       "whose 2-D topology is re-layered")
    ap.add_argument("--synthetic_l", action="store_true",
                    help="no guide: pattern-L baseline + contest-aware layers")
    ap.add_argument("--naive", action="store_true", help="lowest-layer baseline")
    ap.add_argument("--alpha_timing", type=float, default=0.0,
                    help=">0 enables slack-weighted low-R layer preference")
    ap.add_argument("--format", default="ispd25", choices=["ispd25", "ispd24"])
    ap.add_argument("--tree", default="chain", choices=["chain", "mst"],
                    help="per-net topology: convert()'s NN chain or a minimum spanning tree")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="python connectivity check")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not (a.cap and a.net and a.out):
        return self_test()
    if not (a.guide or a.segments or a.synthetic_l or a.naive):
        ap.error("give --guide, --segments, --synthetic_l or --naive")
    cap, nets, _ = export(a.cap, a.net, a.out, guide=a.guide, fmt=a.format,
                          alpha=a.alpha_timing, naive=a.naive, limit=a.limit,
                          segments=a.segments, tree=a.tree)
    if a.check:
        opens, missing = check_connectivity(cap, nets, a.out)
        log(f"connectivity: opens={opens} missing_nets={missing}")


if __name__ == "__main__":
    main()
