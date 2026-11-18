#!/usr/bin/env python3
"""repair3d_astar.py — repair3d.py, faster (new file; escape3d.py, repair3d.py and timing_promote.py are unchanged).

A repair pass on NVDLA_b took 85 s to 1,400 s with the same 2,000 hotspots. Profile of the slow pass (third repair
pass on the f10r route): every candidate run lifts its whole net, prices it twice and puts it back, and on the giant
nets (84,730 wire tracks) each of those calls took ~1 s, because the evaluator's via rule (a via adds demand only
where no wire of the same net covers it) was tested against every wire of the via's layer -- quadratic in net size.
Search was 10 s of the pass. Here, with identical results:

* apply / marginal test the via rule on the via's own track only (wires[(z, track)]): the same demand, linear time;
  marginal also sums the cost of big nets with numpy (float summation order may differ in the last bits);
* the window search is A*: every wire step costs >= step_pen and every via >= via_pen (cell_cost grows with demand),
  so h = step_pen x (x / y distance to the targets' bounding box) + via_pen x (layer distance to their range) is
  admissible and consistent -- a path of the same, optimal cost (ties may break differently), mostly the corridor;
* nets with more than repair3d_big.BIG_NET wire intervals + vias are decided with local updates (repair3d_big.py; --no_big
  runs repair3d's loop for every net);
* --max_segs N (optional, changes results): nets with more than N route segments are frozen;
* --rrr K (optional): rip-up and re-route -- every wire run of >= K GCells is a candidate, not only hotspot runs.

  python repair3d_astar.py <repair3d.py arguments> [--max_segs N]   # fastmode8.py --repair_astar uses it
  python repair3d_astar.py --check <design> <suite> <route>         # fast vs original apply / marginal / search
"""
import heapq
import math
import sys

import numpy as np

import escape3d as E                                         # READ-ONLY reuse
import repair3d as Rp                                        # READ-ONLY reuse
from timing_promote import apply as apply_orig, _key

BIG = 64                                                     # nets with more tracks or cells: numpy marginal


def _load_c():
    """tools/libastar3d_<platform>.so (tools/astar3d.cpp), or None."""
    import ctypes
    import os
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", f"libastar3d_{sys.platform}.so")
    if not os.path.exists(lib):
        return None
    try:
        so = ctypes.CDLL(lib)
    except OSError:
        return None
    P = np.ctypeslib.ndpointer
    so.astar3d.restype = ctypes.c_int
    so.astar3d.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2 + [P(np.float64, flags="C"), P(np.int32, flags="C"),
                           P(np.int64, flags="C"), P(np.int32, flags="C"), ctypes.c_int, P(np.int32, flags="C"),
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                           P(np.int32, flags="C"), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    return so


_C = _load_c()
_PATH = [np.zeros((1 << 16, 3), np.int32)]


def astar_c(la, sources, targets, x0, y0, W, step_pen, via_pen):
    """astar() in C (same costs and tie-breaking, so the same path); Python astar() when the library is missing."""
    if _C is None:
        return astar(la, sources, targets, x0, y0, W, step_pen, via_pen)
    import ctypes
    if not targets:
        return math.inf, None
    if getattr(la, "_c_dirs", None) is None:
        la._c_dirs = np.asarray(la.dirs, np.int32)
        la._c_u = np.ascontiguousarray(la.u, np.float64)
    d, c = la.d, la.c                                        # passed with their strides: no copy of the grid
    if d.dtype != np.float64 or c.dtype != np.float64:
        return astar(la, sources, targets, x0, y0, W, step_pen, via_pen)
    strides = np.array([v // 8 for v in d.strides] + [v // 8 for v in c.strides], np.int64)
    src = np.asarray(list(sources), np.int32).reshape(-1, 3)
    tgt = np.asarray(list(targets), np.int32).reshape(-1, 3)
    cost = ctypes.c_double()
    while True:
        buf = _PATH[0]
        n = _C.astar3d(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, la._c_u, la._c_dirs, strides, src, len(src),
                       tgt, len(tgt), int(x0), int(y0), int(W), float(step_pen), float(via_pen), buf, len(buf),
                       ctypes.byref(cost))
        if n >= 0:
            break
        _PATH[0] = np.zeros((-n * 2, 3), np.int32)
    if n == 0:
        return math.inf, None
    return cost.value, [tuple(int(v) for v in r) for r in buf[:n]]


def _via_cells(la, x, y, z):
    """(x, y, demand) cells of a via between z and z + 1 at (x, y), on layer z (timing_promote.apply's rule)."""
    if la.dirs[z] == 0:
        if 0 < x < la.X - 1:
            return ((x - 1, y, 1), (x, y, 1))
        if x > 0:
            return ((x - 1, y, 2),)
        return ((x, y, 2),) if x < la.X - 1 else ()
    if 0 < y < la.Y - 1:
        return ((x, y - 1, 1), (x, y, 1))
    if y > 0:
        return ((x, y - 1, 2),)
    return ((x, y, 2),) if y < la.Y - 1 else ()


def _covered(la, wires, x, y, z):
    """a wire of the same net on layer z covers (x, y) (closed interval, as in timing_promote.apply)."""
    if la.dirs[z] == 0:
        return any(lo <= x <= hi for lo, hi in wires.get((z, y), ()))
    return any(lo <= y <= hi for lo, hi in wires.get((z, x), ()))


def apply_fast(la, wires, vias, sign):
    """timing_promote.apply (commit_net with a sign) with the via rule tested on the via's own track."""
    d, dirs = la.d, la.dirs
    for (z, fixed), ivs in wires.items():
        if dirs[z] == 0:
            for a, b in ivs:
                d[z, a:b, fixed] += 2 * sign
        else:
            for a, b in ivs:
                d[z, fixed, a:b] += 2 * sign
    for (x, y, z) in vias:
        if z == 0 or _covered(la, wires, x, y, z):
            continue
        for cx, cy, dd in _via_cells(la, x, y, z):
            d[z, cx, cy] += dd * sign


def marginal_fast(la, wires, vias):
    """escape3d.marginal: exact model cost of adding this net to the current demand."""
    if len(wires) <= BIG and sum(b - a for ivs in wires.values() for a, b in ivs) + 2 * len(vias) <= 2 * BIG:
        return _marginal_small(la, wires, vias)
    X, Y = la.X, la.Y
    zs, fs, a_s, b_s, hs = [], [], [], [], []
    for (z, fixed), ivs in wires.items():
        h = la.dirs[z] == 0
        for a, b in ivs:
            if b > a and z > 0:
                zs.append(z); fs.append(fixed); a_s.append(a); b_s.append(b); hs.append(h)
    parts_lin, parts_dd = [], []
    if zs:
        zs, fs, a_s, b_s, hs = map(np.asarray, (zs, fs, a_s, b_s, hs))
        n = b_s - a_s
        rep = np.repeat(np.arange(len(n)), n)
        t = a_s[rep] + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
        x = np.where(hs[rep], t, fs[rep])
        y = np.where(hs[rep], fs[rep], t)
        parts_lin.append((zs[rep] * X + x) * Y + y)
        parts_dd.append(np.full(len(t), 2.0))
    vl, vd = [], []
    for (x, y, z) in vias:
        if z == 0 or _covered(la, wires, x, y, z):
            continue
        for cx, cy, dd in _via_cells(la, x, y, z):
            vl.append((z * X + cx) * Y + cy)
            vd.append(float(dd))
    if vl:
        parts_lin.append(np.asarray(vl, dtype=np.int64))
        parts_dd.append(np.asarray(vd))
    if not parts_lin:
        return 0.0
    lin = np.concatenate(parts_lin)
    u_lin, inv = np.unique(lin, return_inverse=True)
    dd = np.bincount(inv, weights=np.concatenate(parts_dd))
    z, r = np.divmod(u_lin, X * Y)
    x, y = np.divmod(r, Y)
    d, c, u = la.d[z, x, y], la.c[z, x, y], la.u[z]          # la.d is not C-contiguous: reshape(-1) would copy it
    return float(np.sum(_cost(u, d + dd, c) - _cost(u, d, c)))


def _marginal_small(la, wires, vias):
    """escape3d.marginal for small nets (Python loop; the via rule on the via's own track)."""
    out = {}
    for (z, fixed), ivs in wires.items():
        h = la.dirs[z] == 0
        for a, b in ivs:
            for t in range(a, b):
                k = (z, t, fixed) if h else (z, fixed, t)
                out[k] = out.get(k, 0) + 2
    for (x, y, z) in vias:
        if z == 0 or _covered(la, wires, x, y, z):
            continue
        for cx, cy, dd in _via_cells(la, x, y, z):
            out[(z, cx, cy)] = out.get((z, cx, cy), 0) + dd
    tot = 0.0
    cell_cost = E.cell_cost
    for (z, x, y), dd in out.items():
        if z == 0:
            continue
        d, c, u = float(la.d[z, x, y]), float(la.c[z, x, y]), float(la.u[z])
        tot += cell_cost(u, d + dd, c) - cell_cost(u, d, c)
    return tot


def _cost(u, d, c):
    """escape3d.cell_cost on arrays."""
    open_ = c > E.CAP_EPS
    e = np.where(open_, np.minimum(0.25 * d - 0.5 * c, 700.0), np.minimum(0.75 * d, 700.0))
    return np.where(open_ | (d > 0), u * np.exp(e), 0.0)


def astar(la, sources, targets, x0, y0, W, step_pen, via_pen):
    """escape3d.dijkstra's contract: 3-D shortest path from any source (z,x,y) to any target inside the window."""
    L, dirs = la.L, la.dirs
    xlo, xhi = max(0, x0 - W), min(la.X - 1, x0 + W)
    ylo, yhi = max(0, y0 - W), min(la.Y - 1, y0 + W)
    if not targets:
        return math.inf, None
    zmin = min(t[0] for t in targets); zmax = max(t[0] for t in targets)
    xmin = min(t[1] for t in targets); xmax = max(t[1] for t in targets)
    ymin = min(t[2] for t in targets); ymax = max(t[2] for t in targets)

    def h(z, x, y):
        return (step_pen * (max(0, xmin - x, x - xmax) + max(0, ymin - y, y - ymax))
                + via_pen * max(0, zmin - z, z - zmax))
    dist, prev = {}, {}
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (h(*s), 0.0, s))
    edge_wire, edge_via = E.edge_wire, E.edge_via
    while pq:
        _, dcur, node = heapq.heappop(pq)
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
        if z >= 1:                                          # metal1 never carries wires
            if dirs[z] == 0:
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
                heapq.heappush(pq, (nd + h(*nb), nd, nb))
    return math.inf, None


def check(design, suite, route, n=60, seed=0):
    """fast vs original: apply (demand arrays), marginal (big + random nets) and search costs (random windows)."""
    import os
    import time
    from ispd25_adapter import parse_cap_auto
    from timing_promote import TimingLA, to_wires_vias, read_route25, BENCH_ROOT, GRID
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    nets = read_route25(route)
    la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
    t0 = time.time()
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply_fast(la, w, v, +1)
    print(f"load {len(nets):,} nets with apply_fast: {time.time() - t0:.1f}s")
    rng = np.random.default_rng(seed)
    big = sorted(range(len(nets)), key=lambda i: -len(nets[i][1]))[:5]
    pick = big + [int(i) for i in rng.integers(len(nets), size=300)]
    t_o = t_f = 0.0
    worst_m = 0.0
    same_d = True
    for i in pick:
        w, v = to_wires_vias(nets[i][1], la.dirs)
        d0 = la.d.copy()
        apply_orig(la, w, v, -1)
        d1 = la.d.copy()
        la.d[:] = d0
        apply_fast(la, w, v, -1)
        same_d &= bool(np.array_equal(la.d, d1))
        t = time.time(); mo = E.marginal(la, w, v); t_o += time.time() - t
        t = time.time(); mf = marginal_fast(la, w, v); t_f += time.time() - t
        worst_m = max(worst_m, abs(mo - mf) / max(1.0, abs(mo)))
        apply_fast(la, w, v, +1)
    print(f"apply: identical demand {same_d}; marginal on {len(pick)} nets (5 largest: "
          f"{[len(nets[i][1]) for i in big]} segments): original {t_o:.1f}s, fast {t_f:.1f}s, "
          f"max relative difference {worst_m:.1e}")
    td = ta = tc = 0.0
    worst = 0.0
    same_path = True
    for k in range(n):
        segs = nets[int(rng.integers(len(nets)))][1]
        if len(segs) < 2:
            continue
        s, t = segs[0], segs[-1]
        A, B = (s[2], s[0], s[1]), (t[5], t[3], t[4])
        x0, y0 = (A[1] + B[1]) // 2, (A[2] + B[2]) // 2
        half = max(abs(A[1] - B[1]), abs(A[2] - B[2])) // 2 + 10
        t0 = time.time()
        cd, _ = E.dijkstra(la, [A], {B}, x0, y0, half, 1.0, 0.1)
        td += time.time() - t0
        t0 = time.time()
        ca, pa = astar(la, [A], {B}, x0, y0, half, 1.0, 0.1)
        ta += time.time() - t0
        t0 = time.time()
        cc, pc = astar_c(la, [A], {B}, x0, y0, half, 1.0, 0.1)
        tc += time.time() - t0
        same_path &= (pa == pc)
        if math.isfinite(cd) or math.isfinite(ca):
            worst = max(worst, abs(cd - ca) / max(1.0, abs(cd)), abs(cc - ca) / max(1.0, abs(ca)))
    print(f"{n} searches: Dijkstra {td:.1f}s, A* {ta:.1f}s, C A* {tc:.2f}s ({ta / max(tc, 1e-9):.0f}x faster than "
          f"Python A*, library {'loaded' if _C is not None else 'MISSING'}); same paths C vs Python: {same_path}; "
          f"max relative cost difference {worst:.2e}")
    return 0 if same_d and worst_m < 1e-9 and worst < 1e-9 and same_path else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check(*sys.argv[2:5])
    max_segs = 0
    if "--max_segs" in sys.argv:
        k = sys.argv.index("--max_segs")
        max_segs = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    no_big = "--no_big" in sys.argv
    if no_big:
        sys.argv.remove("--no_big")
    rrr = rrr_top = 0
    if "--rrr" in sys.argv:                                 # every wire run >= K GCells is a candidate
        k = sys.argv.index("--rrr")
        rrr = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    if "--rrr_top" in sys.argv:                             # ... only the N runs whose own demand costs most
        k = sys.argv.index("--rrr_top")
        rrr_top = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    Rp.dijkstra, Rp.apply, Rp.marginal = astar_c, apply_fast, marginal_fast   # names repair3d imported
    if not no_big:                                          # big nets decided locally (repair3d_big.py)
        import repair3d_big as RB
        RB.LOCAL_OK, RB.ALL_RUNS, RB.RRR_TOP = True, rrr, rrr_top
        Rp.repair = RB.repair_fast
    if max_segs:
        orig = Rp.repair

        def repair(la, pins, nets, *args, guarded=frozenset(), **kw):
            big = {_key(name) for name, segs, _ in nets if len(segs) > max_segs}
            Rp.log(f"{len(big):,} nets with more than {max_segs:,} route segments frozen")
            return orig(la, pins, nets, *args, guarded=frozenset(guarded) | big, **kw)
        Rp.repair = repair
    return Rp.main()


if __name__ == "__main__":
    sys.exit(main())
