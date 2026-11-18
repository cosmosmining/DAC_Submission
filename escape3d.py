#!/usr/bin/env python3
"""escape3d.py — spread pin via stacks out of blocked-GCell hotspots (exact contest cost).

Finding (bsg_chip, ISPD'25, InstantGR route): ~80% of the 96.5M overflow is BLOCKED-cell
cost, ~80% of that sits in cells that hold only via stacks, and the top 1,000 cells carry
99% of it. They are dense macro pins (e.g. SRAM data pins on metal3) inside regions where
metal1-4 have zero capacity: every pin climbs through the blocked layers in the SAME GCell,
and the evaluator charges exp(0.75*d) per blocked cell — 10 stacks in one cell cost ~1e5x
more than the same stacks spread one per cell.

For every net whose pin tail (vias from the pin layer up to the net's lowest wire at that
GCell) feeds a hotspot cell:
  1. remove the net from the exact evaluator demand model (route_export.LayerAssigner rules:
     wire -> +2 on cells [lo, hi); via (x,y,z) -> +1/+1 (or +2 at the border) on the two
     cells of layer z it touches unless the net has a wire on z covering (x,y))
  2. drop the tail; 3-D Dijkstra (window +-W GCells, metal1 never used for wires) from ANY
     access point of that pin to ANY point of the net's remaining tree, edge costs = exact
     marginal contest cost at the current demand (+ a tiny per-step length penalty)
  3. keep the new connection only if the net's exact marginal cost drops (the old tail is
     always feasible, so the model cost never increases)
Output: ISPD'25 route (run seg_normalize.py before OpenROAD PPA).

  python escape3d.py --design bsg_chip [--suite blind] --route X.norm.route --out Y.route
  python escape3d.py --self_test
"""
import argparse
import heapq
import math
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import route_export as RE                                   # READ-ONLY reuse
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from la_swap import used_access_points
from timing_promote import TimingLA, apply, to_wires_vias, read_route25, _key, BENCH_ROOT, GRID

CAP_EPS = 1e-3


def log(m):
    print(f"[escape3d {time.strftime('%H:%M:%S')}] {m}", flush=True)


def cell_cost(u, d, c):
    if c > CAP_EPS:
        return u * math.exp(min(0.25 * d - 0.5 * c, 700.0))
    return u * math.exp(min(0.75 * d, 700.0)) if d > 0 else 0.0


def demand_delta(la, wires, vias):
    """{(z, x, y): +demand} that apply(+1) would add (same rules)."""
    X, Y = la.X, la.Y
    out = defaultdict(float)
    covered = defaultdict(list)
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            for t in range(a, b):
                out[(z, t, fixed) if la.dirs[z] == 0 else (z, fixed, t)] += 2
            covered[z].append((fixed, a, b))
    for (x, y, z) in vias:
        if z == 0:
            continue
        dz = la.dirs[z]
        if any((dz == 0 and f == y and lo <= x <= hi) or (dz == 1 and f == x and lo <= y <= hi)
               for f, lo, hi in covered.get(z, ())):
            continue
        if dz == 0:
            if 0 < x < X - 1:
                out[(z, x - 1, y)] += 1
                out[(z, x, y)] += 1
            elif x > 0:
                out[(z, x - 1, y)] += 2
            elif x < X - 1:
                out[(z, x, y)] += 2
        elif dz == 1:
            if 0 < y < Y - 1:
                out[(z, x, y - 1)] += 1
                out[(z, x, y)] += 1
            elif y > 0:
                out[(z, x, y - 1)] += 2
            elif y < Y - 1:
                out[(z, x, y)] += 2
    return out


def marginal(la, wires, vias):
    """exact model cost of adding this net to the current demand."""
    tot = 0.0
    for (z, x, y), dd in demand_delta(la, wires, vias).items():
        if z == 0:
            continue
        d, c, u = float(la.d[z, x, y]), float(la.c[z, x, y]), float(la.u[z])
        tot += cell_cost(u, d + dd, c) - cell_cost(u, d, c)
    return tot


def edge_wire(la, z, x, y):
    """marginal cost of +2 demand on (z, x, y)."""
    d, c, u = float(la.d[z, x, y]), float(la.c[z, x, y]), float(la.u[z])
    return cell_cost(u, d + 2, c) - cell_cost(u, d, c)


def edge_via(la, x, y, z):
    """marginal cost of a via between z and z+1 at (x, y) (demand on layer z)."""
    if z == 0:
        return 0.0
    X, Y = la.X, la.Y
    cells = []
    if la.dirs[z] == 0:
        cells = [((x - 1, y), 1), ((x, y), 1)] if 0 < x < X - 1 else ([((x - 1, y), 2)] if x > 0 else [((x, y), 2)])
    else:
        cells = [((x, y - 1), 1), ((x, y), 1)] if 0 < y < Y - 1 else ([((x, y - 1), 2)] if y > 0 else [((x, y), 2)])
    tot = 0.0
    for (cx, cy), dd in cells:
        d, c, u = float(la.d[z, cx, cy]), float(la.c[z, cx, cy]), float(la.u[z])
        tot += cell_cost(u, d + dd, c) - cell_cost(u, d, c)
    return tot


def tree_points(la, wires, vias):
    pts = set()
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            for t in range(a, b + 1):
                pts.add((z, t, fixed) if la.dirs[z] == 0 else (z, fixed, t))
    for (x, y, z) in vias:
        pts.add((z, x, y))
        pts.add((z + 1, x, y))
    return pts


def dijkstra(la, sources, targets, x0, y0, W, step_pen, via_pen):
    """3-D shortest path from any source (z,x,y) to any target inside the window."""
    L = la.L
    xlo, xhi = max(0, x0 - W), min(la.X - 1, x0 + W)
    ylo, yhi = max(0, y0 - W), min(la.Y - 1, y0 + W)
    dist, prev = {}, {}
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        dcur, node = heapq.heappop(pq)
        if dcur > dist.get(node, math.inf):
            continue
        if node in targets:
            path = [node]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return dcur, path[::-1]
        z, x, y = node
        nbrs = []
        if z + 1 < L:
            nbrs.append(((z + 1, x, y), edge_via(la, x, y, z) + via_pen))
        if z - 1 >= 0:
            nbrs.append(((z - 1, x, y), edge_via(la, x, y, z - 1) + via_pen))
        if z >= 1:                                      # metal1 never carries wires
            if la.dirs[z] == 0:
                if x + 1 <= xhi:
                    nbrs.append(((z, x + 1, y), edge_wire(la, z, x, y) + step_pen))
                if x - 1 >= xlo:
                    nbrs.append(((z, x - 1, y), edge_wire(la, z, x - 1, y) + step_pen))
            else:
                if y + 1 <= yhi:
                    nbrs.append(((z, x, y + 1), edge_wire(la, z, x, y) + step_pen))
                if y - 1 >= ylo:
                    nbrs.append(((z, x, y - 1), edge_wire(la, z, x, y - 1) + step_pen))
        for nb, w in nbrs:
            nd = dcur + w
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(pq, (nd, nb))
    return math.inf, None


def path_to_wires_vias(la, path):
    wires, vias = defaultdict(list), set()
    for (z1, x1, y1), (z2, x2, y2) in zip(path, path[1:]):
        if z1 != z2:
            vias.add((x1, y1, min(z1, z2)))
        elif la.dirs[z1] == 0:
            wires[(z1, y1)].append((min(x1, x2), max(x1, x2)))
        else:
            wires[(z1, x1)].append((min(y1, y2), max(y1, y2)))
    return wires, vias


def merge_wires(*ws):
    raw = defaultdict(list)
    for w in ws:
        for k, ivs in w.items():
            raw[k] += list(ivs)
    out = {}
    for k, ivs in raw.items():
        ivs.sort()
        m = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        out[k] = [tuple(v) for v in m]
    return out


def emit(la, wires, vias):
    buf = []
    for (z, f), ivs in wires.items():
        for a, b in ivs:
            if la.dirs[z] == 0:
                buf.append(RE.seg_line(a, f, z, b, f, z, "ispd25").strip())
            else:
                buf.append(RE.seg_line(f, a, z, f, b, z, "ispd25").strip())
    for (x, y, z) in sorted(vias):
        buf.append(RE.seg_line(x, y, z, x, y, z + 1, "ispd25").strip())
    return buf


def escape(cap, pins, nets, n_hot=3000, W=12, step_pen=1.0, via_pen=0.1, passes=2, la=None, guarded=frozenset()):
    """pins: {key: [[(l,x,y)...] per pin]}; nets: read_route25 list. Returns (new_lines, stats)."""
    t0 = time.time()
    if la is None:
        la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
        for _, segs, _ in nets:
            w, v = to_wires_vias(segs, la.dirs)
            apply(la, w, v, +1)
    new_lines, stats = {}, defaultdict(int)
    cur = {}                                          # nets changed so far -> (wires, vias)

    def get(i):
        return cur[i] if i in cur else to_wires_vias(nets[i][1], la.dirs)
    for p in range(passes):
        # hotspot cells: blocked, currently costly
        blk = (la.c <= CAP_EPS) & (la.d > 0)
        blk[0] = False
        zz, xx, yy = np.nonzero(blk)
        costs = la.u[zz] * np.exp(np.minimum(0.75 * la.d[zz, xx, yy], 700.0))
        order = np.argsort(-costs)[:n_hot]
        hot = {(int(zz[k]), int(xx[k]), int(yy[k])) for k in order}
        if not hot:
            break
        hot_xy = {(x, y) for (_, x, y) in hot} | {(x + 1, y) for (_, x, y) in hot} | {(x, y + 1) for (_, x, y) in hot}
        # candidate (net, gcell) pairs: a via stack at a GCell next to a hotspot cell
        cand = []
        for i, (name, segs, _) in enumerate(nets):
            if i in cur:
                gs = {(x, y) for (x, y, z) in cur[i][1] if (x, y) in hot_xy}
            else:                                     # cheap scan of the raw via segments
                gs = {(x1, y1) for (x1, y1, z1, x2, y2, z2) in segs if z1 != z2 and (x1, y1) in hot_xy}
            for g in gs:
                cand.append((i, g))
        log(f"pass {p + 1}: {len(hot):,} hotspot cells (top cost {float(costs[order[0]]):.3g}), "
            f"{len(cand):,} (net, GCell) stacks next to them")
        gain = 0.0
        for i, (gx, gy) in cand:
            name, segs, _ = nets[i]
            k = _key(name)
            if k not in pins:
                continue
            if k in guarded:                            # near-critical: timing owns it
                stats["guarded"] += 1
                continue
            w, v = get(i)
            stack = sorted(z for (x, y, z) in v if (x, y) == (gx, gy))
            if not stack:
                continue
            # pins of this net whose used access point is at this GCell
            used = used_access_points(segs, pins[k])
            here = [(ap, aps) for ap, aps in zip(used, pins[k]) if ap is not None and (ap[1], ap[2]) == (gx, gy)]
            if len(here) != 1:
                stats["skip_pins_at_gcell_" + ("0" if not here else "many")] += 1
                continue
            (lp, _, _), aps = here[0]
            wl = [z for (z, f), ivs in w.items() for a, b in ivs
                  if (la.dirs[z] == 0 and f == gy and a <= gx <= b) or (la.dirs[z] == 1 and f == gx and a <= gy <= b)]
            zw = min(wl) if wl else max(stack) + 1
            tail = {(gx, gy, z) for z in range(lp, zw) if (gx, gy, z) in v}
            if not tail:
                continue
            apply(la, w, v, -1)
            old = marginal(la, w, v)
            v_rest = v - tail
            targets = tree_points(la, w, v_rest)
            if not targets:                              # the pin tail was the whole net
                apply(la, w, v, +1)
                stats["skip_tail_is_net"] += 1
                continue
            sources = [(l, x, y) for (l, x, y) in aps if abs(x - gx) <= W and abs(y - gy) <= W]
            _, path = dijkstra(la, sources, targets, gx, gy, W, step_pen, via_pen)
            if path is None:
                apply(la, w, v, +1)
                stats["no_path"] += 1
                continue
            pw, pv = path_to_wires_vias(la, path)
            w2, v2 = merge_wires(w, pw), v_rest | pv
            new = marginal(la, w2, v2)
            if new < old - 1e-9:
                apply(la, w2, v2, +1)
                cur[i] = (w2, v2)
                new_lines[i] = emit(la, w2, v2)
                gain += old - new
                stats["moved"] += 1
            else:
                apply(la, w, v, +1)
                stats["kept"] += 1
        log(f"pass {p + 1}: model overflow gain {gain:,.0f}  {dict(stats)} ({time.time() - t0:.0f}s)")
    return new_lines, dict(stats), la


def run(design, suite, route_in, route_out, n_hot, W, step_pen, passes, crit=None, guard_slack=None):
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    pins = {_key(n["name"]): n["pins"]
            for n in parse_net_ispd25(os.path.join(BENCH_ROOT, suite, design, design + ".net"))}
    nets = read_route25(route_in)
    guarded = set()
    if crit and guard_slack is not None:
        from timing_promote import parse_crit
        for sl, ns in parse_crit(crit):
            if sl < guard_slack:
                guarded.update(_key(n) for n in ns)
        log(f"slack guard {guard_slack} ns: {len(guarded):,} nets protected")
    new_lines, stats, _ = escape(cap, pins, nets, n_hot, W, step_pen, 0.1, passes, guarded=frozenset(guarded))
    with open(route_out, "w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for s in new_lines.get(i, lines):
                fh.write(s + "\n")
            fh.write(")\n")
    log(f"wrote {route_out}: {len(new_lines):,} nets changed")


def self_test():
    """6 nets, each with one pin on metal3 (z=2) in the SAME GCell of a region where
    metal1-4 are blocked; each net climbs to open metal5 (z=4, horizontal) and runs to a
    far pin. Escaping must spread the stacks and cut the exact model cost."""
    X, Y, L = 30, 12, 6
    dirs = [0, 1, 0, 1, 0, 1]
    layers = []
    for z in range(L):
        c = np.full((X, Y), 6.0)
        if 1 <= z <= 3:
            c[3:12, :] = 0.0                               # blocked region x in [3, 12)
        layers.append({"name": f"metal{z+1}", "dir": dirs[z], "cap": c})
    cap = {"L": L, "X": X, "Y": Y, "layers": layers, "short_costs": [5.0] * L, "unit_via": 4.0}
    pins, nets = {}, []
    for n in range(6):
        name = f"n{n}"
        pins[name] = [[(2, 7, 5)], [(0, 25, 5 + (n % 3))]]
        segs = [(7, 5, 2, 7, 5, 3), (7, 5, 3, 7, 5, 4), (7, 5, 4, 25, 5, 4)]
        yy = 5 + (n % 3)
        if yy != 5:
            segs += [(25, 5, 4, 25, 5, 5), (25, 5, 5, 25, yy, 5), (25, yy, 0, 25, yy, 1),
                     (25, yy, 1, 25, yy, 2), (25, yy, 2, 25, yy, 3), (25, yy, 3, 25, yy, 4), (25, yy, 4, 25, yy, 5)]
        else:
            segs += [(25, 5, 0, 25, 5, 1), (25, 5, 1, 25, 5, 2), (25, 5, 2, 25, 5, 3), (25, 5, 3, 25, 5, 4)]
        nets.append((name, segs, []))
    la = TimingLA(cap, 0.0, 2.1, 5.0, 2.0)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)

    def total(la):
        t = 0.0
        for z in range(1, L):
            for x in range(X):
                for y in range(Y):
                    if la.d[z, x, y] > 0 or la.c[z, x, y] > CAP_EPS:
                        t += cell_cost(la.u[z], la.d[z, x, y], la.c[z, x, y])
        return t
    before = total(la)
    new_lines, stats, la = escape(cap, pins, nets, n_hot=50, W=6, step_pen=1.0, via_pen=0.1, passes=2, la=la)
    after = total(la)
    ok = after < 0.5 * before and stats.get("moved", 0) >= 1
    # connectivity: every changed net still connects its two pins (union-find over 3-D points)
    for i, lines in new_lines.items():
        w, v = to_wires_vias([(int(s.split()[0]) // GRID, int(s.split()[1]) // GRID, int(s.split()[2][5:]) - 1,
                               int(s.split()[3]) // GRID, int(s.split()[4]) // GRID, int(s.split()[5][5:]) - 1)
                              for s in lines], la.dirs)
        pts = tree_points(la, w, v)
        par = {p: p for p in pts}

        def f(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for (z, fx), ivs in w.items():
            for a, b in ivs:
                for t in range(a, b):
                    p1 = (z, t, fx) if la.dirs[z] == 0 else (z, fx, t)
                    p2 = (z, t + 1, fx) if la.dirs[z] == 0 else (z, fx, t + 1)
                    par[f(p1)] = f(p2)
        for (x, y, z) in v:
            par[f((z, x, y))] = f((z + 1, x, y))
        a, b = pins[nets[i][0]][0][0], pins[nets[i][0]][1][0]
        conn = a in pts and b in pts and f(a) == f(b)
        ok &= conn
        if not conn:
            print(f"  net {nets[i][0]} disconnected after escape")
    print(f"  exact model cost {before:,.0f} -> {after:,.0f}; {stats}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route")
    ap.add_argument("--out")
    ap.add_argument("--n_hot", type=int, default=3000)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--step_pen", type=float, default=1.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--crit", default=None, help="STA report (crit_nets.tcl) for the slack guard")
    ap.add_argument("--guard_slack", type=float, default=None)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    run(a.design, a.suite, a.route, a.out, a.n_hot, a.window, a.step_pen, a.passes, a.crit, a.guard_slack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
