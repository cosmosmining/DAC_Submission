#!/usr/bin/env python3
"""repair3d.py — reroute WIRE runs out of blocked-GCell hotspots (exact contest cost).

After escape3d.py removed bsg's pin-stack hotspots, what is left is different: on blind
bsg 24.2M of the 25.2M blocked cost sits in metal2 cells that hold only WIRES (runs laid
straight through zero-capacity metal2), and the top 100 cells carry 87%. Layer changes
alone cannot always fix a run whose whole column/row is blocked on every layer of that
direction, so this tool reroutes the run in 3-D:

for each net with a wire run covering one of the top-N blocked cells (nets on reported STA
paths are skipped with --crit/--guard_slack):
  1. remove the net from the exact evaluator demand model; old = its exact marginal cost
  2. drop the run; union-find splits the remaining 3-D tree into components
  3. if the run's two ends are still connected, the run was redundant; otherwise 3-D
     Dijkstra (escape3d.dijkstra: exact marginal costs, metal1 never carries wires) from
     every point of one side to any point of the other inside the run's box +- W
  4. keep the result only if its exact marginal cost is lower (the dropped run itself is
     always a feasible path, so the model cost never increases)

  python repair3d.py --design bsg_chip --suite blind --route X.norm.route --out Y.route \
                     [--crit X.crit.rpt --guard_slack 1e9]
  python repair3d.py --self_test
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from timing_promote import TimingLA, apply, to_wires_vias, read_route25, _key, BENCH_ROOT, GRID
from escape3d import (CAP_EPS, cell_cost, marginal, dijkstra, path_to_wires_vias, merge_wires,
                      emit, tree_points)


def log(m):
    print(f"[repair3d {time.strftime('%H:%M:%S')}] {m}", flush=True)


def components(la, wires, vias):
    """union-find over the 3-D points of a (partial) route -> dict point -> root."""
    pts = tree_points(la, wires, vias)
    par = {p: p for p in pts}

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for (z, fx), ivs in wires.items():
        for a, b in ivs:
            for t in range(a, b):
                p1 = (z, t, fx) if la.dirs[z] == 0 else (z, fx, t)
                p2 = (z, t + 1, fx) if la.dirs[z] == 0 else (z, fx, t + 1)
                par[f(p1)] = f(p2)
    for (x, y, z) in vias:
        par[f((z, x, y))] = f((z + 1, x, y))
    return {p: f(p) for p in pts}


def net_valid(la, wires, vias, pins):
    """the evaluator's validity: one connected component that touches at least one
    access point (l, x, y) of every pin."""
    comp = components(la, wires, vias)
    if not comp:
        return False
    roots = set(comp.values())
    if len(roots) != 1:
        return False
    for aps in pins:
        if not any((l, x, y) in comp for (l, x, y) in aps):
            return False
    return True


def drop_run(wires, key, a, b):
    """remove interval [a, b] from wires[key] (intervals are merged, so exact match)."""
    out = {k: list(v) for k, v in wires.items()}
    out[key] = [iv for iv in out[key] if iv != (a, b)]
    if not out[key]:
        del out[key]
    return out


def repair(la, pins, nets, n_hot=2000, W=10, step_pen=1.0, via_pen=0.1, guarded=frozenset(), passes=2,
           overcap=False):
    new_lines, stats = {}, defaultdict(int)
    cur = {}

    def get(i):
        return cur[i] if i in cur else to_wires_vias(nets[i][1], la.dirs)
    for p in range(passes):
        blk = (la.c <= CAP_EPS) & (la.d > 0)
        blk[0] = False
        zz, xx, yy = np.nonzero(blk)
        costs = la.u[zz] * np.exp(np.minimum(0.75 * la.d[zz, xx, yy], 700.0))
        if overcap:   # also OPEN cells over capacity (d > 2c), ranked by their exact evaluator cost
            ov = (la.c > CAP_EPS) & (la.d > 2.0 * la.c)
            ov[0] = False
            z2, x2, y2 = np.nonzero(ov)
            c2 = la.u[z2] * np.exp(np.minimum(0.25 * la.d[z2, x2, y2] - 0.5 * la.c[z2, x2, y2], 700.0))
            zz, xx, yy = np.concatenate([zz, z2]), np.concatenate([xx, x2]), np.concatenate([yy, y2])
            costs = np.concatenate([costs, c2])
        if len(zz) == 0:
            break
        order = np.argsort(-costs)[:n_hot]
        hot = {(int(zz[k]), int(xx[k]), int(yy[k])) for k in order}
        # candidate runs: (net index, (z, fixed), a, b) whose demand cells hit a hotspot
        cand = set()
        for i, (name, segs, _) in enumerate(nets):
            if _key(name) in guarded:
                continue
            w = get(i)[0] if i in cur else None
            if w is None:
                # cheap pre-check on raw segments before decoding the net
                hit = False
                for x1, y1, z1, x2, y2, z2 in segs:
                    if z1 != z2:
                        continue
                    if y1 == y2:
                        lo, hi = min(x1, x2), max(x1, x2)
                        if any((z1, t, y1) in hot for t in range(lo, hi)):
                            hit = True
                            break
                    else:
                        lo, hi = min(y1, y2), max(y1, y2)
                        if any((z1, x1, t) in hot for t in range(lo, hi)):
                            hit = True
                            break
                if not hit:
                    continue
                w = get(i)[0]
            for (z, fx), ivs in w.items():
                for a, b in ivs:
                    cells = ((z, t, fx) if la.dirs[z] == 0 else (z, fx, t) for t in range(a, b))
                    if any(c in hot for c in cells):
                        cand.add((i, (z, fx), a, b))
        log(f"pass {p + 1}: {len(hot):,} hotspot cells (top cost {float(costs[order[0]]):.3g}), "
            f"{len(cand):,} candidate runs")
        gain = 0.0
        for (i, key, a, b) in sorted(cand, key=lambda c: -(c[3] - c[2])):
            w, v = get(i)
            if key not in w or (a, b) not in w[key]:
                stats["stale"] += 1
                continue
            apply(la, w, v, -1)
            old = marginal(la, w, v)
            w_rest = drop_run(w, key, a, b)
            z, fx = key
            A = (z, a, fx) if la.dirs[z] == 0 else (z, fx, a)
            B = (z, b, fx) if la.dirs[z] == 0 else (z, fx, b)
            comp = components(la, w_rest, v)
            # an end touching nothing once the run is dropped is either a pin access point
            # (a one-point piece that must be reconnected) or the tip of a spur (dropped with
            # the run); net_valid() below still checks every pin and one component
            pin_pts = {tuple(ap) for aps in pins.get(_key(nets[i][0]), []) for ap in aps}
            for q in (A, B):
                if q not in comp and q in pin_pts:
                    comp[q] = q
                    stats["pin_end"] += 1
            ends = [q for q in (A, B) if q in comp]
            if not ends:
                w2, v2 = w_rest, v                      # an isolated spur run
                stats["spur"] += 1
            elif len(ends) == 2 and comp[A] == comp[B]:
                w2, v2 = w_rest, v                      # redundant run
            else:
                if len(ends) == 1:
                    stats["spur_end"] += 1
                A0, B0 = A, B                           # the window spans the dropped run
                A = ends[0]
                # reconnect EVERY piece the run held together (its two ends and any pin
                # stack or branch that landed on its interior): grow from A's piece
                x0 = (A0[1] + B0[1]) // 2
                y0 = (A0[2] + B0[2]) // 2
                half = max(abs(A0[1] - B0[1]), abs(A0[2] - B0[2])) // 2 + W
                inwin = lambda q: abs(q[1] - x0) <= half and abs(q[2] - y0) <= half
                groups = defaultdict(set)
                for q, r in comp.items():
                    groups[r].add(q)
                tree = {q for q in groups[comp[A]] if inwin(q)}
                others = [g for r, g in groups.items() if r != comp[A] and any(inwin(q) for q in g)]
                others.sort(key=lambda g: min(abs(q[1] - A[1]) + abs(q[2] - A[2]) for q in g))
                w2, v2 = w_rest, set(v)
                ok_all = True
                for g in others:
                    tg = {q for q in g if inwin(q)}
                    if not tree or not tg:
                        ok_all = False
                        break
                    _, path = dijkstra(la, list(tree), tg, x0, y0, half, step_pen, via_pen)
                    if path is None:
                        ok_all = False
                        break
                    pw, pv = path_to_wires_vias(la, path)
                    w2, v2 = merge_wires(w2, pw), v2 | pv
                    tree |= set(path) | tg
                if not ok_all:
                    apply(la, w, v, +1)
                    stats["no_path"] += 1
                    continue
            if not net_valid(la, w2, v2, pins.get(_key(nets[i][0]), [])):
                apply(la, w, v, +1)                 # would disconnect a pin / split the net
                stats["reject_invalid"] += 1
                continue
            new = marginal(la, w2, v2)
            if new < old - 1e-9:
                apply(la, w2, v2, +1)
                cur[i] = (w2, v2)
                new_lines[i] = emit(la, w2, v2)
                gain += old - new
                stats["rerouted"] += 1
            else:
                apply(la, w, v, +1)
                stats["kept"] += 1
        log(f"pass {p + 1}: model overflow gain {gain:,.0f}  {dict(stats)}")
    return new_lines, dict(stats)


def run(design, suite, route_in, route_out, n_hot, W, passes, crit=None, guard_slack=None, overcap=False,
        net_slack=None, slack_margin=0.05):
    t0 = time.time()
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    pins = {_key(n["name"]): n["pins"]
            for n in parse_net_ispd25(os.path.join(BENCH_ROOT, suite, design, design + ".net"))}
    nets = read_route25(route_in)
    la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)
    guarded = set()
    if crit and guard_slack is not None:
        from timing_promote import parse_crit
        for sl, ns in parse_crit(crit):
            if sl < guard_slack:
                guarded.update(_key(n) for n in ns)
        log(f"slack guard {guard_slack} ns: {len(guarded):,} nets protected")
    if net_slack:                     # per-net worst slack (net_slack.tcl): freeze every net with
        n0 = len(guarded)             # slack < margin -> every path with slack < margin is untouched
        with open(net_slack) as fh:
            for line in fh:
                f = line.rsplit(None, 1)
                if len(f) == 2:
                    try:
                        if float(f[1]) < slack_margin:
                            guarded.add(_key(f[0]))
                    except ValueError:
                        pass
        log(f"net slack < {slack_margin} ns: {len(guarded) - n0:,} more nets protected")
    log(f"{design}[{suite}]: {len(nets):,} nets loaded ({time.time() - t0:.0f}s)")
    new_lines, stats = repair(la, pins, nets, n_hot, W, guarded=frozenset(guarded), passes=passes,
                              overcap=overcap)
    with open(route_out, "w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for s in new_lines.get(i, lines):
                fh.write(s + "\n")
            fh.write(")\n")
    log(f"wrote {route_out}: {len(new_lines):,} nets changed ({time.time() - t0:.0f}s)")


def self_test():
    """A net whose V run on metal2 crosses a blocked band; metal4 (V) is open above it.
    repair3d must move the crossing off the blocked cells and keep the net connected."""
    X, Y, L = 12, 20, 6
    dirs = [0, 1, 0, 1, 0, 1]
    layers = []
    for z in range(L):
        c = np.full((X, Y), 6.0)
        if z == 1:
            c[4, 8:12] = 0.0                      # metal2 blocked in column x=4, y 8..11
        layers.append({"name": f"metal{z+1}", "dir": dirs[z], "cap": c})
    cap = {"L": L, "X": X, "Y": Y, "layers": layers, "short_costs": [5.0] * L, "unit_via": 4.0}
    segs = [(4, 2, 0, 4, 2, 1), (4, 2, 1, 4, 18, 1), (4, 18, 0, 4, 18, 1),     # pin-to-pin V run on metal2
            (4, 15, 0, 4, 15, 1)]                                                # 3rd pin stack mid-run
    nets = [("n0", segs, [])]
    la = TimingLA(cap, 0.0, 2.1, 5.0, 2.0)
    w, v = to_wires_vias(segs, la.dirs)
    apply(la, w, v, +1)

    def total():
        t = 0.0
        for z in range(1, L):
            for x in range(X):
                for y in range(Y):
                    if la.d[z, x, y] > 0 or la.c[z, x, y] > CAP_EPS:
                        t += cell_cost(la.u[z], la.d[z, x, y], la.c[z, x, y])
        return t
    before = total()
    pins = {"n0": [[(0, 4, 2)], [(0, 4, 18)], [(0, 4, 15)]]}
    new_lines, stats = repair(la, pins, nets, n_hot=10, W=4, passes=1)
    after = total()
    ok = after <= before
    # connectivity of the rewritten net
    lines = new_lines.get(0, [])
    w2, v2 = to_wires_vias([(int(s.split()[0]) // GRID, int(s.split()[1]) // GRID, int(s.split()[2][5:]) - 1,
                             int(s.split()[3]) // GRID, int(s.split()[4]) // GRID, int(s.split()[5][5:]) - 1)
                            for s in lines], la.dirs)
    comp = components(la, w2, v2)
    conn = net_valid(la, w2, v2, pins["n0"]) if lines else True
    blocked_left = sum(1 for (z, fx), ivs in w2.items() for lo, hi in ivs for t in range(lo, hi)
                       if z == 1 and fx == 4 and 8 <= t < 12)
    ok &= conn and blocked_left == 0
    print(f"  exact model cost {before:,.1f} -> {after:,.1f}; connected {conn}; blocked metal2 cells still used "
          f"{blocked_left}; {stats}")
    # case 2: the run ends ON the pins' metal2 access points (no via stacks) — both ends touch
    # nothing once the run is dropped; they must be reconnected as one-point pieces, not skipped
    la2 = TimingLA(cap, 0.0, 2.1, 5.0, 2.0)
    segs2 = [(4, 2, 1, 4, 18, 1)]
    w, v = to_wires_vias(segs2, la2.dirs)
    apply(la2, w, v, +1)
    pins2 = {"n1": [[(1, 4, 2)], [(1, 4, 18)]]}
    nl2, st2 = repair(la2, pins2, [("n1", segs2, [])], n_hot=10, W=4, passes=1)
    lines2 = nl2.get(0, [])
    w3, v3 = to_wires_vias([(int(s.split()[0]) // GRID, int(s.split()[1]) // GRID, int(s.split()[2][5:]) - 1,
                             int(s.split()[3]) // GRID, int(s.split()[4]) // GRID, int(s.split()[5][5:]) - 1)
                            for s in lines2], la2.dirs)
    conn2 = bool(lines2) and net_valid(la2, w3, v3, pins2["n1"])
    left2 = sum(1 for (z, fx), ivs in w3.items() for lo, hi in ivs for t in range(lo, hi)
                if z == 1 and fx == 4 and 8 <= t < 12)
    ok &= conn2 and left2 == 0 and st2.get("skip_dangling", 0) == 0
    print(f"  pin-ended run: rerouted {bool(lines2)}, connected {conn2}, blocked cells still used {left2}; {dict(st2)}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route")
    ap.add_argument("--out")
    ap.add_argument("--n_hot", type=int, default=2000)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--crit", default=None)
    ap.add_argument("--guard_slack", type=float, default=None)
    ap.add_argument("--overcap", action="store_true", help="hotspots include open cells with demand > 2*capacity")
    ap.add_argument("--net_slack", default=None, help="per-net worst slack file (net_slack.tcl)")
    ap.add_argument("--slack_margin", type=float, default=0.05)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    run(a.design, a.suite, a.route, a.out, a.n_hot, a.window, a.passes, a.crit, a.guard_slack, a.overcap,
        a.net_slack, a.slack_margin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
