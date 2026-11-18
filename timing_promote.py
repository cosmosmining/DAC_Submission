#!/usr/bin/env python3
"""timing_promote.py — STA-driven critical-net layer promotion on top of any ISPD'25 route.

Why: on ISPD'25 the quality score is timing-dominated
    score = w1*(WNS-r1) + w2*(TNS-r2)/N_end + w3*(power-r3) + w4*overflow
and only ~0.1-0.6% of nets are timing-critical (GAP-LA Table VIII). InstantGR's route has
the best congestion we have but is timing-blind (ariane TNS -1454 vs GAP-LA -998). So:
keep every net of the base route, and re-assign ONLY the nets on violating paths.

Per critical net n (worst slack first):
  1. remove n's demand from the exact evaluator demand model (route_export.LayerAssigner)
  2. candidate 2-D topologies: the base route's own (with the access points it used) and
     a FLUTE Steiner tree (rsmt_route.py)
  3. layer assignment by the LayerAssigner DP with a SCORE-UNIT cost per run of length l:
        w4 * overflow marginal                                   (exact contest overflow)
      + tw_n * l_um * (R_z * C_load + C_z * R_drv) / 1000        (Elmore-style delay, ns)
     R_z, C_z per layer from GAP-LA Table I (the contest's modified NanGate45); via R too.
     tw_n = |w2| * k_n / N_end + |w1| * [n on the worst path]   (score per ns of delay)
     k_n  = # violating endpoints whose worst path uses n (STA report).
     Per the RC table metal9/10 are fastest; metal4-8 have MORE capacitance than metal2.
  4. keep the cheaper topology, commit, write.
Non-critical nets are written back byte-for-byte (their segments are re-emitted as read).

Critical paths come from ppa_run/official/crit_nets.tcl (report_checks -fields net).
Run seg_normalize.py on the output before OpenROAD PPA.

  python timing_promote.py --design ariane --route base.route --crit crit.rpt --out new.route
  python timing_promote.py --self_test
"""
import argparse
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import route_export as RE                                   # READ-ONLY reuse
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from la_swap import used_access_points, nodes_from_aps
import ispd25_score as SC

BENCH_ROOT = os.path.join(ROOT, "ispd_bench/ISPD2025/ISPD2025_benchmarks")
BENCH = os.path.join(BENCH_ROOT, "visible")
GRID = 4200
C_VIA_LOAD = 2.0          # fF: load seen by a via stack at a sink pin
# GAP-LA Table I (ISPD'25 modified NanGate45): R kOhm/um, C fF/um, R of the via ABOVE (kOhm)
RC = {"metal1": (5.4286e-3, 7.41819e-2, 5e-3), "metal2": (3.5714e-3, 6.74606e-2, 5e-3),
      "metal3": (3.5714e-3, 8.88758e-2, 5e-3), "metal4": (1.5e-3, 1.07121e-1, 3e-3),
      "metal5": (1.5e-3, 1.08964e-1, 3e-3), "metal6": (1.5e-3, 1.02044e-1, 3e-3),
      "metal7": (1.875e-4, 1.10436e-1, 1e-3), "metal8": (1.875e-4, 9.69714e-2, 1e-3),
      "metal9": (3.75e-5, 3.6864e-2, 0.5e-3), "metal10": (3.75e-5, 2.8042e-2, 0.0)}


def log(m):
    print(f"[timing_promote {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _key(name):
    return name.replace("\\", "")


# ─────────────────────────── STA report parsing ───────────────────────────
_NET = re.compile(r"\s(\S+) \(net\)\s*$")
_SLACK = re.compile(r"^\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s+slack \((VIOLATED|MET)\)")


def parse_crit(path):
    """-> list of (slack, [net names in path order])."""
    paths, nets = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("Startpoint:"):
                nets = []
                continue
            m = _NET.search(line)
            if m:
                nets.append(m.group(1))
                continue
            m = _SLACK.match(line)
            if m:
                paths.append((float(m.group(1)), nets))
                nets = []
    return paths


_DRV = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\^v]\s+\S+\s+\(\S+\)\s*$")


def parse_net_rc(path):
    """-> {net_key: (fanout, load_fF, driver_delay_ns)} from report_checks -fields
    {net fanout capacitance}: the driver-pin line carries fanout / load / delay, the next
    line names the net."""
    out, last = {}, None
    with open(path) as fh:
        for line in fh:
            m = _DRV.match(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue
            m = _NET.search(line)
            if m and last is not None:
                k = _key(m.group(1))
                if k not in out or last[1] > out[k][1]:
                    out[k] = last
                last = None
    return out


R_DRV_MIN = float(os.environ.get("TP_R_DRV_MIN", "0.05"))   # kOhm floor (slew effects)


def net_rc_params(info, c_load, r_drv):
    """per-net (c_load fF, r_drv kOhm) for the delay proxy: half the measured load as the
    mean downstream capacitance, driver resistance = driver delay / load, floored at
    TP_R_DRV_MIN — a net's capacitance also slows the NEXT gate through its input slew,
    which driver delay / load does not see (STA pairs gave ~0.4-0.8 ps per fF)."""
    if info is None:
        return c_load, max(r_drv, R_DRV_MIN)
    _, cap, dly = info
    if cap <= 0:
        return c_load, max(r_drv, R_DRV_MIN)
    return min(max(0.5 * cap, 1.0), 1e5), min(max(dly / cap * 1000.0, R_DRV_MIN), 20.0)


def net_weights(paths, case):
    """-> {net_key: (tw score/ns, worst slack, k)} for nets on violating paths."""
    _, _, _, _, nend, (w1, w2, _, _) = SC.CASES[case]
    viol = [(s, ns) for s, ns in paths if s < 0]
    if not viol:
        return {}
    wns = min(s for s, _ in viol)
    k, worst, on_wns = defaultdict(int), {}, set()
    for s, ns in viol:
        for n in set(ns):
            kn = _key(n)
            k[kn] += 1
            worst[kn] = min(worst.get(kn, 0.0), s)
            if s <= wns + 1e-9:
                on_wns.add(kn)
    return {n: (abs(w2) * k[n] / nend + abs(w1) * (n in on_wns), worst[n], k[n]) for n in k}


# ─────────────────────────── timing-aware assigner ───────────────────────────
class TimingLA(RE.LayerAssigner):
    """LayerAssigner whose costs are in ISPD'25 SCORE units for the net being routed."""

    def __init__(self, cap, w4, um_per_gcell, c_load, r_drv):
        super().__init__(cap)
        self.w4 = float(w4)
        self.um = float(um_per_gcell)
        names = [l["name"] for l in cap["layers"]]
        self.R = np.array([RC[n][0] for n in names])
        self.C = np.array([RC[n][1] for n in names])
        self.Rv = np.array([RC[n][2] for n in names])
        self.c_load, self.r_drv = float(c_load), float(r_drv)
        self.tw = 0.0                                   # set per net

    def delay_ns(self, z, gcells):
        l = gcells * self.um
        return l * (self.R[z] * self.c_load + self.C[z] * self.r_drv) / 1000.0 \
            + self.R[z] * self.C[z] * l * l / 2.0 / 1000.0

    def run_cost(self, run, z, crit):
        dr, fixed, a, b = run[:4]
        sl = (slice(a, b), fixed) if dr == 0 else (fixed, slice(a, b))
        ov = float(self.wire_marginal(z, sl).sum())
        return self.w4 * ov + self.tw * self.delay_ns(z, b - a)

    def via_cost(self, x, y, za, zb):
        lo, hi = (za, zb) if za <= zb else (zb, za)
        tot = 0.0
        for z in range(lo, hi):
            # a via stack drives what hangs below it — at a sink pin that is one pin
            # (~C_VIA_LOAD fF), not the whole net load (10k-sink reset net: 9.7 pF)
            tot += self.tw * self.Rv[z] * C_VIA_LOAD / 1000.0
            if z != 0 and self.dirs[z] in (0, 1):
                dz, cz, u = self.d[z, x, y], self.c[z, x, y], self.u[z]
                if cz > RE.CAP_EPS:
                    m = 2.0 * u * math.exp(min(0.25 * dz - 0.5 * cz, RE.EXP_CLIP)) * RE.E025M1
                else:
                    base = math.exp(min(0.75 * dz, RE.EXP_CLIP)) if dz > 0 else 0.0
                    m = 2.0 * u * (math.exp(min(0.75 * (dz + 1), RE.EXP_CLIP)) - base)
                tot += self.w4 * m
        return tot


def apply(la, wires, vias, sign):
    """commit_net() with a sign (the evaluator's via-demand rule included)."""
    X, Y, d = la.X, la.Y, la.d
    covered = defaultdict(list)
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            if la.dirs[z] == 0:
                d[z, a:b, fixed] += 2 * sign
            else:
                d[z, fixed, a:b] += 2 * sign
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
                d[z, x - 1, y] += sign
                d[z, x, y] += sign
            elif x > 0:
                d[z, x - 1, y] += 2 * sign
            elif x < X - 1:
                d[z, x, y] += 2 * sign
        elif dz == 1:
            if 0 < y < Y - 1:
                d[z, x, y - 1] += sign
                d[z, x, y] += sign
            elif y > 0:
                d[z, x, y - 1] += 2 * sign
            elif y < Y - 1:
                d[z, x, y] += 2 * sign


def to_wires_vias(segs, dirs):
    """GCell segments -> ({(z, fixed): [(a, b)] merged}, {(x, y, z)} unit vias)."""
    raw = defaultdict(list)
    vias = set()
    for x1, y1, z1, x2, y2, z2 in segs:
        if z1 != z2:
            for z in range(min(z1, z2), max(z1, z2)):
                vias.add((x1, y1, z))
        elif dirs[z1] == 0 and y1 == y2 and x1 != x2:
            raw[(z1, y1)].append((min(x1, x2), max(x1, x2)))
        elif dirs[z1] == 1 and x1 == x2 and y1 != y2:
            raw[(z1, x1)].append((min(y1, y2), max(y1, y2)))
    wires = {}
    for key, ivs in raw.items():
        ivs.sort()
        m = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        wires[key] = [tuple(v) for v in m]
    return wires, vias


def read_route25(path):
    """-> ordered list of (name, [GCell segments], [raw lines])."""
    out, name, segs, lines = [], None, [], []
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if name is None:
                name, segs, lines = s, [], []
                continue
            if s == "(":
                continue
            if s == ")":
                out.append((name, segs, lines))
                name = None
                continue
            t = s.split()
            if len(t) == 6:
                lines.append(s)
                segs.append((int(t[0]) // GRID, int(t[1]) // GRID, int(t[2][5:]) - 1,
                             int(t[3]) // GRID, int(t[4]) // GRID, int(t[5][5:]) - 1))
    return out


def promote(design, route_in, crit_path, route_out, max_nets=4000, c_load=5.0, r_drv=2.0,
            topo="both", cap_path=None, net_path=None, min_k=1, suite="visible", per_net_rc=True,
            tw_scale=1.0):
    t0 = time.time()
    case = SC.case_of(design, suite)
    w4 = SC.CASES[case][5][3]
    cap_path = cap_path or os.path.join(BENCH_ROOT, suite, design, design + ".cap")
    net_path = net_path or os.path.join(BENCH_ROOT, suite, design, design + ".net")
    cap = parse_cap_auto(cap_path)
    la = TimingLA(cap, w4, GRID / 2000.0, c_load, r_drv)
    pins = {_key(n["name"]): n["pins"] for n in parse_net_ispd25(net_path)}
    nets = read_route25(route_in)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)
    log(f"{design}: {len(nets):,} nets committed ({time.time() - t0:.0f}s)")
    tw = net_weights(parse_crit(crit_path), case)
    rc = parse_net_rc(crit_path) if per_net_rc else {}
    order = sorted((n for n in tw if tw[n][2] >= min_k), key=lambda n: -tw[n][0])[:max_nets]
    log(f"{len(tw):,} nets on violating paths; promoting {len(order):,} (w4={w4})")
    idx = {_key(n): i for i, (n, _, _) in enumerate(nets)}
    fl = None
    if topo in ("both", "rsmt"):
        from rsmt_route import Flute, rsmt_polys
        fl = Flute()
    new_lines, stats = {}, defaultdict(int)
    for kname in order:
        i = idx.get(kname)
        if i is None or kname not in pins:
            stats["not_in_route"] += 1
            continue
        name, segs, _ = nets[i]
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, -1)
        chosen = used_access_points(segs, pins[kname])
        nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins[kname], chosen)])
        la.tw = tw[kname][0] * tw_scale     # tw_scale > 1: buy timing with overflow (score margin to spend)
        la.c_load, la.r_drv = net_rc_params(rc.get(kname), c_load, r_drv)
        cands = []
        if topo in ("both", "base"):
            p = RE.segments_to_polys(segs, nodes)
            if RE.tree_connects(nodes, p):
                cands.append(("base", p))
        if fl is not None and len(nodes) > 1:
            cands.append(("rsmt", rsmt_polys(la, fl, nodes, 1.0)))
        if not cands:                                   # keep the original route
            apply(la, w, v, +1)
            stats["kept"] += 1
            continue
        best = None
        for tag, polys in cands:
            cost = 0.0
            for pts in polys:
                runs = RE.to_runs(pts)
                if runs:
                    cost += la.assign_path(runs, nodes.get(pts[0]), nodes.get(pts[-1]), 1.0)[1]
            if best is None or cost < best[0]:
                best = (cost, tag, polys)
        wires, vias = RE.route_net(la, nodes, best[2], 1.0, mark=True)
        stats["topo_" + best[1]] += 1
        buf = []
        for (z, fixed), ivs in wires.items():
            for a, b in ivs:
                if la.dirs[z] == 0:
                    buf.append(RE.seg_line(a, fixed, z, b, fixed, z, "ispd25").strip())
                else:
                    buf.append(RE.seg_line(fixed, a, z, fixed, b, z, "ispd25").strip())
        for (x, y, z) in sorted(vias):
            buf.append(RE.seg_line(x, y, z, x, y, z + 1, "ispd25").strip())
        new_lines[i] = buf
        stats["promoted"] += 1
    with open(route_out, "w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for s in new_lines.get(i, lines):
                fh.write(s + "\n")
            fh.write(")\n")
    log(f"wrote {route_out}: {dict(stats)} ({time.time() - t0:.0f}s)")
    return dict(stats)


def self_test():
    ok = True
    # 1) report parsing: two violating paths share net n2
    d = tempfile.mkdtemp(prefix="tprom_")
    rpt = os.path.join(d, "c.rpt")
    open(rpt, "w").write(
        "Startpoint: a\nEndpoint: b\n  0.10 ^ u1/Z (BUF_X1)\n     2    1.2    n1 (net)\n"
        "  0.20 ^ u2/Z (BUF_X1)\n     1    0.8    i\\[0\\]/n2 (net)\n   -0.3000   slack (VIOLATED)\n\n"
        "Startpoint: c\nEndpoint: d\n     1    0.8    i\\[0\\]/n2 (net)\n   -0.1000   slack (VIOLATED)\n\n"
        "Startpoint: e\nEndpoint: f\n     1    0.8    n3 (net)\n    0.0500   slack (MET)\n")
    p = parse_crit(rpt)
    w = net_weights(p, 1)
    good = len(p) == 3 and set(w) == {"n1", "i[0]/n2"} and w["i[0]/n2"][2] == 2 \
        and w["n1"][0] > 0 and w["i[0]/n2"][1] == -0.3
    ok &= good
    print(f"  STA report parse / weights: {good}  {w}")
    rpt2 = os.path.join(d, "c2.rpt")
    open(rpt2, "w").write("Startpoint: a\n 10109 19345.38086   24.14700   30.1 ^ u/g18/Z (MUX2_X1)\n"
                          "                                  rst_n (net)\n   -76.1   slack (VIOLATED)\n")
    rc = parse_net_rc(rpt2)
    cl, rd = net_rc_params(rc.get("rst_n"), 5.0, 2.0)
    good = rc.get("rst_n") == (10109, 19345.38086, 24.147) and abs(cl - 9672.69) < 0.1 and abs(rd - 1.248) < 0.01
    ok &= good
    print(f"  per-net RC from STA: {rc.get('rst_n')} -> c_load {cl:.1f} fF, r_drv {rd:.3f} kOhm: {good}")
    # 2) RC model: metal10 fastest, metal4-8 slower than metal2 for driver-limited nets
    cap = {"L": 10, "X": 4, "Y": 4, "layers": [{"name": f"metal{z+1}", "dir": z % 2,
           "cap": np.full((4, 4), 5.0)} for z in range(10)], "short_costs": [1.0] * 10,
           "unit_via": 4.0}
    la = TimingLA(cap, 3e-7, 2.1, 5.0, 2.0)
    dly = [la.delay_ns(z, 50) for z in range(10)]
    good = int(np.argmin(dly)) == 9 and dly[3] > dly[1] and dly[8] < dly[1]
    ok &= good
    print(f"  RC ordering (ns per 50 GCells): " + " ".join(f"m{z+1}:{v:.4f}" for z, v in enumerate(dly))
          + f"  {'ok' if good else 'FAIL'}")
    # 3) apply(+1) then apply(-1) restores the demand exactly
    w_, v_ = to_wires_vias([(0, 1, 2, 3, 1, 2), (3, 1, 2, 3, 1, 5), (1, 0, 1, 1, 3, 1)], la.dirs)
    d0 = la.d.copy()
    apply(la, w_, v_, +1)
    moved = float(np.abs(la.d - d0).sum())
    apply(la, w_, v_, -1)
    good = moved > 0 and np.array_equal(la.d, d0)
    ok &= good
    print(f"  demand add/remove round trip: {good}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--route")
    ap.add_argument("--crit")
    ap.add_argument("--out")
    ap.add_argument("--max_nets", type=int, default=4000)
    ap.add_argument("--c_load", type=float, default=5.0, help="fF, downstream load in the delay proxy")
    ap.add_argument("--r_drv", type=float, default=2.0, help="kOhm, driver resistance in the delay proxy")
    ap.add_argument("--topo", default="both", choices=["both", "base", "rsmt"])
    ap.add_argument("--min_k", type=int, default=1)
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--fixed_rc", action="store_true", help="global c_load/r_drv (old model)")
    ap.add_argument("--tw_scale", type=float, default=1.0, help="multiplier on the STA timing weights")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    promote(a.design, a.route, a.crit, a.out, a.max_nets, a.c_load, a.r_drv, a.topo, min_k=a.min_k,
            suite=a.suite, per_net_rc=not a.fixed_rc, tw_scale=a.tw_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
