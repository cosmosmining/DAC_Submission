#!/usr/bin/env python3
"""repair3d_big.py — repair3d.repair with local updates for big nets (new file; repair3d.py is unchanged).

repair3d re-routes one dropped wire run at a time, but prices, lifts, re-adds and re-checks the WHOLE net for every
candidate run: on NVDLA's 96,996-segment net each candidate run cost ~1.5 s (two union-finds over ~400k points, two
whole-net marginals, two whole-net demand updates), so the third repair pass spent 470 of its 500 s on that one net.
Here nets with more than BIG_NET wire intervals + vias (connected, covering every pin; checked once) are
handled with the same decisions computed locally; smaller nets run repair3d's code unchanged:

* demand: the net is lifted only inside the search window + 1 GCell (all the cells the search and the decision read),
  from a track index of its wires and a bucket index of its vias, and put back there afterwards;
* pieces: dropping a run splits the net only at the vias that land on it. A breadth-first search over wire intervals
  and via points, one front per such via, stops once at most one front is still growing: the stopped fronts are the
  small pieces, everything else belongs to the one big piece (the net was connected before the drop);
* validity: then automatic for connectivity (every piece touches the window and is reconnected), and pins are
  checked only for the pins with an access point on the dropped run;
* cost: the old / new model costs differ only on the dropped run's cells, the new path's cells and the vias whose
  coverage changes, so the comparison sums those cells only.

Ties in the order the pieces are reconnected (equal distance to the start end) can break differently from repair3d,
which orders them by set iteration; everything else is the same decision. Used by repair3d_astar.py.
"""
import bisect
from collections import defaultdict, deque

import numpy as np

import escape3d as E                                         # READ-ONLY reuse
import repair3d as Rp                                        # READ-ONLY reuse
import repair3d_astar as RA
from timing_promote import to_wires_vias, _key

Rp_via_cells = RA._via_cells

BIG_NET = 4000                                               # wire intervals + vias above which a net is local
ALL_RUNS = 0            # > 0: every wire run at least this long is a candidate (rip-up and re-route), not hotspots
LOCAL_OK = False        # big-net path on: set by repair3d_astar.py, whose marginal is the overflow model local_cost uses
RRR_TOP = 0             # with ALL_RUNS: keep only the K runs whose own demand costs most (0 = all)
RRR_MAX_PINS = 1000     # with ALL_RUNS: nets with more pins are not re-routed (high-fanout nets are timing-sensitive:
#                         re-routing NVDLA's 10,110-sink reset net for overflow alone cost 47 ns of WNS)
BUCKET = 16


class BigNet:
    """one big net: wires {(z, track): sorted disjoint intervals}, vias, via adjacency / buckets, pin access points."""

    def __init__(self, la, w, v, pins):
        self.la = la
        self.w = {k: sorted(ivs) for k, ivs in w.items()}
        self.v = set()
        self.vadj = defaultdict(set)
        self.vb = defaultdict(set)
        for via in v:
            self.add_via(via)
        self.pins = pins
        self.ap = defaultdict(list)
        for k, aps in enumerate(pins):
            for ap in aps:
                self.ap[tuple(ap)].append(k)

    # ----------------------------------------------------------------- structure
    def add_via(self, via):
        if via in self.v:
            return
        x, y, z = via
        self.v.add(via)
        self.vadj[(z, x, y)].add((z + 1, x, y))
        self.vadj[(z + 1, x, y)].add((z, x, y))
        self.vb[(x // BUCKET, y // BUCKET)].add(via)

    def track(self, q):
        z, x, y = q
        return ((z, y), x) if self.la.dirs[z] == 0 else ((z, x), y)

    def find_iv(self, key, t):
        """the interval (a, b) of track key with a <= t <= b, or None."""
        ivs = self.w.get(key)
        if not ivs:
            return None
        k = bisect.bisect_right(ivs, (t, float("inf"))) - 1
        if k >= 0 and ivs[k][0] <= t <= ivs[k][1]:
            return ivs[k]
        return None

    def elem(self, q, drop):
        """element holding point q once run `drop` = (key, a, b) is removed: (0, key, a, b) interval, (1, q) via
        point, or None (not a point of the net)."""
        key, t = self.track(q)
        iv = self.find_iv(key, t)
        if iv is not None and not (key == drop[0] and iv[0] == drop[1]):
            return (0, key, iv[0], iv[1])
        return (1, q) if q in self.vadj else None

    def points(self, e):
        if e[0] == 1:
            return [e[1]]
        (z, f), a, b = e[1], e[2], e[3]
        return [(z, t, f) for t in range(a, b + 1)] if self.la.dirs[z] == 0 else [(z, f, t) for t in range(a, b + 1)]

    def neighbours(self, e, drop):
        for q in self.points(e):
            for r in self.vadj.get(q, ()):
                yield self.elem(r, drop)

    def pieces(self, drop):
        """pieces of the net without run drop: (attachment points, owner {elem: label}, find, big root or None)."""
        (z, f), a, b = drop
        run_pts = [(z, t, f) for t in range(a, b + 1)] if self.la.dirs[z] == 0 else [(z, f, t) for t in range(a, b + 1)]
        att = [q for q in run_pts if q in self.vadj]
        parent = list(range(len(att)))

        def find(l):
            while parent[l] != l:
                parent[l] = parent[parent[l]]
                l = parent[l]
            return l
        owner, front = {}, {}
        for l, q in enumerate(att):
            e = (1, q)
            owner[e] = l
            front[l] = deque([e])
        active = set(front)
        while len(active) > 1:
            for l in list(active):
                if l not in active:
                    continue
                dq = front[l]
                if not dq:
                    active.discard(l)
                    continue
                e = dq.popleft()
                for nb in self.neighbours(e, drop):
                    o = owner.get(nb)
                    if o is None:
                        owner[nb] = l
                        dq.append(nb)
                        continue
                    ro = find(o)
                    if ro != l:                              # two fronts met: one piece
                        parent[ro] = l
                        dq.extend(front.pop(ro, ()))
                        active.discard(ro)
                if not dq:
                    active.discard(l)
        big = next(iter(active)) if active else None
        return att, owner, find, big

    # ----------------------------------------------------------------- demand
    def contrib(self, R):
        """the net's demand on the cells of R = (x0, x1, y0, y1) (inclusive): [(z, x, y | slice, amount)]."""
        la = self.la
        x0, x1, y0, y1 = R
        out = []
        for z in range(1, la.L):
            if la.dirs[z] == 0:
                for y in range(y0, y1 + 1):
                    for a, b in self.w.get((z, y), ()):
                        lo, hi = max(a, x0), min(b, x1 + 1)
                        if lo < hi:
                            out.append((z, slice(lo, hi), y, 2))
            else:
                for x in range(x0, x1 + 1):
                    for a, b in self.w.get((z, x), ()):
                        lo, hi = max(a, y0), min(b, y1 + 1)
                        if lo < hi:
                            out.append((z, x, slice(lo, hi), 2))
        for via in self.vias_in(x0, x1 + 1, y0, y1 + 1):
            x, y, z = via
            if z == 0 or self.covered(x, y, z):
                continue
            for cx, cy, dd in Rp_via_cells(la, x, y, z):
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    out.append((z, cx, cy, dd))
        return out

    def vias_in(self, x0, x1, y0, y1):
        for bx in range(max(x0, 0) // BUCKET, x1 // BUCKET + 1):
            for by in range(max(y0, 0) // BUCKET, y1 // BUCKET + 1):
                for via in self.vb.get((bx, by), ()):
                    if x0 <= via[0] <= x1 and y0 <= via[1] <= y1:
                        yield via

    def covered(self, x, y, z, extra=None, drop=None):
        """a wire of the net on layer z covers point (x, y) (closed), optionally without run drop / with extra wires."""
        key, t = ((z, y), x) if self.la.dirs[z] == 0 else ((z, x), y)
        iv = self.find_iv(key, t)
        if iv is not None and not (drop is not None and key == drop[0] and iv[0] == drop[1]):
            return True
        if extra:
            return any(lo <= t <= hi for lo, hi in extra.get(key, ()))
        return False

    def unit_wired(self, z, x, y, extra=None, drop=None):
        """a wire of the net covers the demand unit (z, x, y) (half-open intervals)."""
        key, t = ((z, y), x) if self.la.dirs[z] == 0 else ((z, x), y)
        ivs = self.w.get(key, ())
        k = bisect.bisect_right(ivs, (t, float("inf"))) - 1
        if k >= 0 and ivs[k][0] <= t < ivs[k][1] and not (drop is not None and key == drop[0] and ivs[k][0] == drop[1]):
            return True
        if extra:
            return any(lo <= t < hi for lo, hi in extra.get(key, ()))
        return False

    def merge_in(self, pw):
        for key, ivs in pw.items():
            self.w[key] = [tuple(iv) for iv in _merge(list(self.w.get(key, ())) + list(ivs))]

    def drop_iv(self, key, a, b):
        self.w[key] = [iv for iv in self.w[key] if iv != (a, b)]
        if not self.w[key]:
            del self.w[key]


def _merge(ivs):
    ivs.sort()
    m = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= m[-1][1]:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return m


def _apply_contrib(la, items, sign):
    d = la.d
    for z, xs, ys, amt in items:
        d[z, xs, ys] += amt * sign


def local_cost(la, bn, cells, drop, pw, pv):
    """(old, new) model cost of the net on `cells` (z, x, y) with the net lifted there: old = current route,
    new = current - run drop + wires pw + vias pv."""
    X, Y = la.X, la.Y
    old_t = new_t = 0.0
    for (z, x, y) in cells:
        if z == 0:
            continue
        d_o = 2.0 if bn.unit_wired(z, x, y) else 0.0
        d_n = 2.0 if bn.unit_wired(z, x, y, pw, drop) else 0.0
        if la.dirs[z] == 0:
            cands = ((x, y, z), (x + 1, y, z))
        else:
            cands = ((x, y, z), (x, y + 1, z))
        for via in cands:
            vx, vy, vz = via
            if not (0 <= vx < X and 0 <= vy < Y):
                continue
            in_old, in_new = via in bn.v, via in bn.v or via in pv
            if not (in_old or in_new):
                continue
            dd = sum(amt for cx, cy, amt in Rp_via_cells(la, vx, vy, vz) if (cx, cy) == (x, y))
            if not dd:
                continue
            if in_old and not bn.covered(vx, vy, vz):
                d_o += dd
            if in_new and not bn.covered(vx, vy, vz, pw, drop):
                d_n += dd
        if d_o == d_n:
            continue
        d, c, u = float(la.d[z, x, y]), float(la.c[z, x, y]), float(la.u[z])
        old_t += E.cell_cost(u, d + d_o, c) - E.cell_cost(u, d, c)
        new_t += E.cell_cost(u, d + d_n, c) - E.cell_cost(u, d, c)
    return old_t, new_t


def _cells_of(la, wires, vias):
    out = set()
    for (z, f), ivs in wires.items():
        for a, b in ivs:
            for t in range(a, b):
                out.add((z, t, f) if la.dirs[z] == 0 else (z, f, t))
    for (x, y, z) in vias:
        if z:
            for cx, cy, _ in Rp_via_cells(la, x, y, z):
                out.add((z, cx, cy))
    return out


def big_step(la, bn, key, a, b, pin_list, W, step_pen, via_pen, stats):
    """one candidate run of a big net, decided like repair3d.repair -> (accepted, model gain)."""
    dirs = la.dirs
    z, fx = key
    drop = (key, a, b)
    A0 = (z, a, fx) if dirs[z] == 0 else (z, fx, a)
    B0 = (z, b, fx) if dirs[z] == 0 else (z, fx, b)
    x0 = (A0[1] + B0[1]) // 2
    y0 = (A0[2] + B0[2]) // 2
    half = max(abs(A0[1] - B0[1]), abs(A0[2] - B0[2])) // 2 + W
    R = (max(0, x0 - half - 1), min(la.X - 1, x0 + half + 1), max(0, y0 - half - 1), min(la.Y - 1, y0 + half + 1))
    lifted = bn.contrib(R)
    _apply_contrib(la, lifted, -1)
    try:
        return _big_decide(la, bn, drop, A0, B0, x0, y0, half, R, lifted, pin_list, step_pen, via_pen, stats)
    except Exception:
        _apply_contrib(la, lifted, +1)
        raise


def _big_decide(la, bn, drop, A0, B0, x0, y0, half, R, lifted, pin_list, step_pen, via_pen, stats):
    key, a, b = drop
    att, owner, find, bigroot = bn.pieces(drop)
    roots = sorted({find(l) for l in range(len(att))})

    def piece(q):
        e = bn.elem(q, drop)
        if e is None:
            return None
        o = owner.get(e)
        return ("piece", find(o) if o is not None else bigroot)
    single = [q for q in (A0, B0) if piece(q) is None and q in bn.ap]
    stats["pin_end"] += len(single)

    def gid(q):
        return ("pin", q) if q in single else piece(q)
    ends = [q for q in (A0, B0) if gid(q) is not None]
    run_pts = set(bn.points((0, key, a, b)))
    pw, pv = {}, set()
    if not ends or (len(ends) == 2 and gid(A0) == gid(B0)):
        if not ends:
            stats["spur"] += 1
        valid = len(roots) == 1 and not single
    else:
        if len(ends) == 1:
            stats["spur_end"] += 1
        A = ends[0]
        gA = gid(A)
        win = defaultdict(set)
        for q in window_points(bn, x0, y0, half, drop):
            win[gid(q)].add(q)
        for q in single:
            win[("pin", q)].add(q)
        groups = [("pin", q) for q in single] + [("piece", r) for r in roots]
        others = [g for g in groups if g != gA and win.get(g)]

        def dist(g):
            if g[0] == "pin":
                return abs(g[1][1] - A[1]) + abs(g[1][2] - A[2])
            if g[1] != bigroot:
                pts = [q for e, l in owner.items() if find(l) == g[1] for q in bn.points(e)]
                return min(abs(q[1] - A[1]) + abs(q[2] - A[2]) for q in pts)
            dm = min(abs(q[1] - A[1]) + abs(q[2] - A[2]) for q in win[g])
            room = min(A[1] - (x0 - half), (x0 + half) - A[1], A[2] - (y0 - half), (y0 + half) - A[2])
            if dm > room:                                    # a point outside the window could be nearer
                dm = min(dm, _min_dist_all(bn, drop, A, owner, find, bigroot))
            return dm
        others.sort(key=dist)
        tree = set(win.get(gA, ()))
        ok_all = True
        for g in others:
            tg = set(win[g])
            if not tree or not tg:
                ok_all = False
                break
            _, path = Rp.dijkstra(la, list(tree), tg, x0, y0, half, step_pen, via_pen)
            if path is None:
                ok_all = False
                break
            ppw, ppv = Rp.path_to_wires_vias(la, path)
            for k, ivs in ppw.items():
                pw.setdefault(k, []).extend(ivs)
            pv |= ppv
            tree |= set(path) | tg
        if not ok_all:
            _apply_contrib(la, lifted, +1)
            stats["no_path"] += 1
            return False, 0.0
        pw = {k: [tuple(iv) for iv in _merge(list(ivs))] for k, ivs in pw.items()}
        valid = True
    if valid:                                                # pins with an access point on the dropped run
        for k in {k for q in run_pts for k in bn.ap.get(q, ())}:
            if not any(_has_point(bn, tuple(ap), drop, pw, pv) for ap in pin_list[k]):
                valid = False
                break
    if not valid:
        _apply_contrib(la, lifted, +1)
        stats["reject_invalid"] += 1
        return False, 0.0
    cells = _cells_of(la, {key: [(a, b)]}, ()) | _cells_of(la, pw, pv)
    for q in run_pts:                                        # vias on the run lose its cover
        cells |= {(q[0], cx, cy) for cx, cy, _ in Rp_via_cells(la, q[1], q[2], q[0])}
    for (z, f), ivs in pw.items():                           # vias newly covered by path wires
        for lo, hi in ivs:
            for t in range(lo, hi + 1):
                x, y = (t, f) if la.dirs[z] == 0 else (f, t)
                if (x, y, z) in bn.v:
                    cells |= {(z, cx, cy) for cx, cy, _ in Rp_via_cells(la, x, y, z)}
    old, new = local_cost(la, bn, cells, drop, pw, pv)
    if new < old - 1e-9:
        bn.drop_iv(key, a, b)
        bn.merge_in(pw)
        for via in pv:
            bn.add_via(via)
        _apply_contrib(la, bn.contrib(R), +1)
        stats["rerouted"] += 1
        return True, old - new
    _apply_contrib(la, lifted, +1)
    stats["kept"] += 1
    return False, 0.0


def _has_point(bn, q, drop, pw, pv):
    z, x, y = q
    if bn.covered(x, y, z, pw, drop):
        return True
    if q in bn.vadj:                                         # endpoint of an existing via (vias are never removed)
        return True
    return (x, y, z) in pv or (x, y, z - 1) in pv


def window_points(bn, x0, y0, half, drop):
    la = bn.la
    xl, xh, yl, yh = x0 - half, x0 + half, y0 - half, y0 + half
    out = set()
    for z in range(1, la.L):
        if la.dirs[z] == 0:
            for y in range(max(yl, 0), min(yh, la.Y - 1) + 1):
                for a, b in bn.w.get((z, y), ()):
                    if (z, y) == drop[0] and a == drop[1]:
                        continue
                    for t in range(max(a, xl), min(b, xh) + 1):
                        out.add((z, t, y))
        else:
            for x in range(max(xl, 0), min(xh, la.X - 1) + 1):
                for a, b in bn.w.get((z, x), ()):
                    if (z, x) == drop[0] and a == drop[1]:
                        continue
                    for t in range(max(a, yl), min(b, yh) + 1):
                        out.add((z, x, t))
    for (x, y, z) in bn.vias_in(xl, xh, yl, yh):
        out.add((z, x, y))
        out.add((z + 1, x, y))
    return out


def _min_dist_all(bn, drop, A, owner, find, bigroot):
    """nearest point of the big piece to A over the whole net (rare fallback)."""
    best = float("inf")
    small = {e for e, l in owner.items() if find(l) != bigroot}
    for key, ivs in bn.w.items():
        for a, b in ivs:
            if key == drop[0] and a == drop[1]:
                continue
            e = (0, key, a, b)
            if e in small:
                continue
            for q in bn.points(e):
                best = min(best, abs(q[1] - A[1]) + abs(q[2] - A[2]))
    for q in bn.vadj:
        e = bn.elem(q, drop)
        if e is not None and e not in small:
            best = min(best, abs(q[1] - A[1]) + abs(q[2] - A[2]))
    return best


def repair_fast(la, pins, nets, n_hot=2000, W=10, step_pen=1.0, via_pen=0.1, guarded=frozenset(), passes=2,
                overcap=False, all_runs=None):
    """repair3d.repair; nets with more than BIG_NET wire intervals + vias (connected, every pin covered) are decided
    locally."""
    CAP_EPS, apply, marginal = Rp.CAP_EPS, Rp.apply, Rp.marginal
    all_runs = ALL_RUNS if all_runs is None else all_runs
    use_big = LOCAL_OK                                       # local_cost models the overflow objective only
    new_lines, stats = {}, defaultdict(int)
    cur, bigs, small_ok = {}, {}, set()

    def get(i):
        if i in bigs:
            return bigs[i].w, bigs[i].v
        return cur[i] if i in cur else to_wires_vias(nets[i][1], la.dirs)
    for p in range(passes):
        blk = (la.c <= CAP_EPS) & (la.d > 0)
        blk[0] = False
        zz, xx, yy = np.nonzero(blk)
        costs = la.u[zz] * np.exp(np.minimum(0.75 * la.d[zz, xx, yy], 700.0))
        if overcap:
            ov = (la.c > CAP_EPS) & (la.d > 2.0 * la.c)
            ov[0] = False
            z2, x2, y2 = np.nonzero(ov)
            c2 = la.u[z2] * np.exp(np.minimum(0.25 * la.d[z2, x2, y2] - 0.5 * la.c[z2, x2, y2], 700.0))
            zz, xx, yy = np.concatenate([zz, z2]), np.concatenate([xx, x2]), np.concatenate([yy, y2])
            costs = np.concatenate([costs, c2])
        if len(zz) == 0 and not all_runs:
            break
        order = np.argsort(-costs)[:n_hot]
        hot = {(int(zz[k]), int(xx[k]), int(yy[k])) for k in order}
        cand = set()
        if all_runs and RRR_TOP:                             # cost of each run's own demand, by prefix sums per track
            m = RA._cost(la.u[:, None, None], la.d, la.c) - RA._cost(la.u[:, None, None], np.maximum(la.d - 2, 0), la.c)
            pref = {}
            for z in range(1, la.L):
                if la.dirs[z] == 0:
                    pz = np.zeros((la.X + 1, la.Y))
                    pz[1:] = np.cumsum(m[z], axis=0)
                else:
                    pz = np.zeros((la.X, la.Y + 1))
                    pz[:, 1:] = np.cumsum(m[z], axis=1)
                pref[z] = pz
            scored = []
        for i, (name, segs, _) in enumerate(nets):
            if _key(name) in guarded:
                continue
            if all_runs:
                if RRR_MAX_PINS and len(pins.get(_key(name), ())) > RRR_MAX_PINS:
                    continue
                for (z, fx), ivs in get(i)[0].items():
                    for a, b in ivs:
                        if b - a >= all_runs:
                            if RRR_TOP and z in pref:
                                pz = pref[z]
                                c = pz[b, fx] - pz[a, fx] if la.dirs[z] == 0 else pz[fx, b] - pz[fx, a]
                                scored.append((float(c), (i, (z, fx), a, b)))
                            else:
                                cand.add((i, (z, fx), a, b))
                continue
            w = get(i)[0] if (i in cur or i in bigs) else None
            if w is None:
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
        if all_runs and RRR_TOP:
            scored.sort(key=lambda t: -t[0])
            cand |= {c for _, c in scored[:RRR_TOP]}
        Rp.log(f"pass {p + 1}: " + (f"all runs >= {all_runs} GCells" + (f" (top {RRR_TOP:,} by own cost)" if RRR_TOP else "")
               if all_runs else
               f"{len(hot):,} hotspot cells (top cost {float(costs[order[0]]):.3g})") + f", {len(cand):,} candidate runs")
        gain = 0.0
        for (i, key, a, b) in sorted(cand, key=lambda c: -(c[3] - c[2])):
            w, v = get(i)
            if key not in w or (a, b) not in w[key]:
                stats["stale"] += 1
                continue
            pin_list = pins.get(_key(nets[i][0]), [])
            if use_big and i not in bigs and i not in small_ok and len(v) + sum(map(len, w.values())) > BIG_NET:
                if Rp.net_valid(la, w, v, pin_list):
                    bigs[i] = BigNet(la, w, v, pin_list)
                    cur.pop(i, None)
                    stats["big_nets"] += 1
                else:
                    small_ok.add(i)                          # not connected / a pin uncovered: repair3d's code
            if i in bigs:
                ok, g = big_step(la, bigs[i], key, a, b, pin_list, W, step_pen, via_pen, stats)
                if ok:
                    gain += g
                    new_lines[i] = None                      # emitted at the end
                continue
            apply(la, w, v, -1)
            old = marginal(la, w, v)
            w_rest = Rp.drop_run(w, key, a, b)
            z, fx = key
            A = (z, a, fx) if la.dirs[z] == 0 else (z, fx, a)
            B = (z, b, fx) if la.dirs[z] == 0 else (z, fx, b)
            comp = Rp.components(la, w_rest, v)
            pin_pts = {tuple(ap) for aps in pin_list for ap in aps}
            for q in (A, B):
                if q not in comp and q in pin_pts:
                    comp[q] = q
                    stats["pin_end"] += 1
            ends = [q for q in (A, B) if q in comp]
            if not ends:
                w2, v2 = w_rest, v
                stats["spur"] += 1
            elif len(ends) == 2 and comp[A] == comp[B]:
                w2, v2 = w_rest, v
            else:
                if len(ends) == 1:
                    stats["spur_end"] += 1
                A0, B0 = A, B
                A = ends[0]
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
                    _, path = Rp.dijkstra(la, list(tree), tg, x0, y0, half, step_pen, via_pen)
                    if path is None:
                        ok_all = False
                        break
                    pw, pv = Rp.path_to_wires_vias(la, path)
                    w2, v2 = Rp.merge_wires(w2, pw), v2 | pv
                    tree |= set(path) | tg
                if not ok_all:
                    apply(la, w, v, +1)
                    stats["no_path"] += 1
                    continue
            if not Rp.net_valid(la, w2, v2, pin_list):
                apply(la, w, v, +1)
                stats["reject_invalid"] += 1
                continue
            new = marginal(la, w2, v2)
            if new < old - 1e-9:
                apply(la, w2, v2, +1)
                cur[i] = (w2, v2)
                new_lines[i] = Rp.emit(la, w2, v2)
                gain += old - new
                stats["rerouted"] += 1
            else:
                apply(la, w, v, +1)
                stats["kept"] += 1
        Rp.log(f"pass {p + 1}: model overflow gain {gain:,.0f}  {dict(stats)}")
    for i, bn in bigs.items():
        if i in new_lines:
            new_lines[i] = Rp.emit(la, bn.w, bn.v)
    return new_lines, dict(stats)
