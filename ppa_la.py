#!/usr/bin/env python3
"""ppa_la.py — score-unit layer re-assignment of ALL nets (overflow + power + timing).

ISPD'25 quality score = w1*dWNS + w2*dTNS/N_end + w3*dPower + w4*overflow. Wirelength and
via count are NOT scored directly — they matter only through overflow, delay and power.
Switching power is linear in net capacitance, and per GAP-LA Table I the layer capacitance
differs by ~4x (metal10 0.028 fF/um, metal2 0.067, metal4-8 ~0.10-0.11). On mempool_tile
(w3 = 300) 1 mW of switching power is 0.3 score; InstantGR puts much of its wire on metal4-8.

Coordinate descent from a complete base route (InstantGR's, or a timing_promote output):
for every net (critical nets first, then longest first): remove it from the exact demand
model, re-run the LayerAssigner DP on its own 2-D topology with the per-run cost
      w4 * overflow_marginal
    + w3 * kP * C_z * l_um                 (kP = dPower/dC in W per fF, calibrated below)
    + tw_n * delay_ns(z, l)                (only for nets on violating STA paths)
and keep the new layers only if they are cheaper than re-adding the old ones under the same
cost — so the modelled objective never increases. 2-D topology and access points of every
net are kept (non-critical nets); critical nets may also switch to a FLUTE Steiner tree.

kP: switching power = sum_n a_n * C_n * V^2 * f; with no per-net activities we use one
global slope, kP = P_switch / (C_wire + C_pin_est), P_switch from a PPA report of the base.

  python ppa_la.py --design mempool_tile_wrap --route base.route --ppa base.ppa.txt \
                   [--crit base.crit.rpt] --out new.route
  python ppa_la.py --self_test
"""
import argparse
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
import ispd25_score as SC
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from la_swap import used_access_points, nodes_from_aps
from timing_promote import (TimingLA, apply, to_wires_vias, read_route25, parse_crit,
                            net_weights, _key, BENCH_ROOT, GRID, RC, parse_net_rc, net_rc_params)


def log(m):
    print(f"[ppa_la {time.strftime('%H:%M:%S')}] {m}", flush=True)


class PPALA(TimingLA):
    def __init__(self, cap, w4, um, c_load, r_drv, pw):
        super().__init__(cap, w4, um, c_load, r_drv)
        self.pw = float(pw)                       # score per fF of wire capacitance

    def run_cost(self, run, z, crit):
        dr, fixed, a, b = run[:4]
        return super().run_cost(run, z, crit) + self.pw * self.C[z] * (b - a) * self.um


def route_wire_cap(nets, la):
    """total wire capacitance (fF) of a parsed route."""
    tot = 0.0
    for _, segs, _ in nets:
        for x1, y1, z1, x2, y2, z2 in segs:
            if z1 == z2:
                tot += (abs(x2 - x1) + abs(y2 - y1)) * la.um * la.C[z1]
    return tot


def switching_power(ppa_path):
    t = open(ppa_path).read()
    m = re.search(r"^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", t, re.M)
    return float(m.group(2))


def old_cost(la, w, v):
    """cost of re-adding the current wires/vias of a net (same cost function as the DP)."""
    c = 0.0
    for (z, fixed), ivs in w.items():
        for a, b in ivs:
            run = (la.dirs[z], fixed, a, b)
            c += la.run_cost(run, z, 1.0)
    for (x, y, z) in v:
        c += la.via_cost(x, y, z, z + 1)
    return c


def exact_cost(la, wires, vias):
    """w4 * exact overflow marginal (escape3d.marginal mirrors the evaluator demand rules)
    + power on the wire capacitance + the net's timing weight on its per-run delay."""
    from escape3d import marginal
    c = la.w4 * marginal(la, wires, vias)
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            c += la.pw * la.C[z] * (b - a) * la.um
            if la.tw:
                c += la.tw * la.delay_ns(z, b - a)
    for (x, y, z) in vias:
        if la.tw:
            c += la.tw * la.Rv[z] * 2.0 / 1000.0
    return c


def net_delay(la, wires):
    return sum(la.delay_ns(z, b - a) for (z, fixed), ivs in wires.items() for a, b in ivs)


def emit(la, wires, vias):
    buf = []
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            if la.dirs[z] == 0:
                buf.append(RE.seg_line(a, fixed, z, b, fixed, z, "ispd25").strip())
            else:
                buf.append(RE.seg_line(fixed, a, z, fixed, b, z, "ispd25").strip())
    for (x, y, z) in sorted(vias):
        buf.append(RE.seg_line(x, y, z, x, y, z + 1, "ispd25").strip())
    return buf


def run(design, route_in, ppa_path, route_out, crit_path=None, passes=1, c_load=5.0, r_drv=2.0,
        pin_frac=0.5, kp=None, power_scale=1.0, max_crit=4000, suite="visible", protect_crit=False,
        exact=True, guard_slack=None, no_slower=False, net_slack=None, slack_margin=0.05,
        slack_frac=0.25, net_act=None, vdd=1.1, guard_soft=False, tw_scale=1.0):
    t0 = time.time()
    case = SC.case_of(design, suite)
    _, _, _, _, _, (w1, w2, w3, w4) = SC.CASES[case]
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    pins = {_key(n["name"]): n["pins"]
            for n in parse_net_ispd25(os.path.join(BENCH_ROOT, suite, design, design + ".net"))}
    nets = read_route25(route_in)
    la = PPALA(cap, w4, GRID / 2000.0, c_load, r_drv, 0.0)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)
    cw = route_wire_cap(nets, la)
    if kp is None:
        psw = switching_power(ppa_path)
        kp = psw / (cw * (1.0 + pin_frac))           # W per fF
    la.pw = w3 * kp * power_scale
    log(f"{design}: {len(nets):,} nets, wire C {cw / 1e6:.3f} nF, kP {kp:.3e} W/fF -> "
        f"power weight {la.pw:.3e} score/fF, w4 {w4} ({time.time() - t0:.0f}s)")
    # per-net switching activity (net_activity.tcl = OpenSTA's propagated density, the one
    # report_power uses): P_n = 0.5 * Vdd^2 * density_n * C_n. On ariane >90% of nets have
    # density 0 and 1% of nets carry 99% of the density-weighted wire C, so the global slope
    # above pays overflow for moves that save no power. With --net_act every net gets its own
    # power weight (0 for inactive nets -> pure congestion/timing moves for them).
    pw_net = None
    if net_act:
        dens = {}
        with open(net_act) as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2:
                    # report_power sums over INSTANCES: a net driven by a top-level port
                    # (origin input/clock, e.g. rst_ni, the ideal clock) adds no reported power
                    if len(f) >= 3 and f[2] in ("input", "clock"):
                        continue
                    try:
                        dens[_key(f[0])] = float(f[1])
                    except ValueError:
                        pass
        kpn = 0.5 * vdd * vdd * 1e-15                  # W per fF per (transition/s)
        pw_net = {k: w3 * kpn * d * power_scale for k, d in dens.items() if d > 0}
        dcw = sum(dens.get(_key(n), 0.0) * sum((abs(s[3] - s[0]) + abs(s[4] - s[1])) * la.um * la.C[s[2]]
                                                for s in segs if s[2] == s[5]) for n, segs, _ in nets)
        log(f"net activity: {len(dens):,} nets, {len(pw_net):,} active; modelled wire switching "
            f"power {kpn * dcw:.4e} W of reported switching {switching_power(ppa_path):.4e} W")
    paths_all = parse_crit(crit_path) if crit_path else []
    rc = parse_net_rc(crit_path) if crit_path else {}
    tw = net_weights(paths_all, case) if crit_path else {}
    crit = set(sorted(tw, key=lambda n: -tw[n][0])[:max_crit])
    guarded = set()
    if guard_slack is not None:          # near-critical nets (incl. MET paths) are never touched
        for sl, ns in paths_all:
            if sl < guard_slack:
                guarded.update(_key(n) for n in ns)
        log(f"slack guard {guard_slack} ns: {len(guarded):,} nets protected")
    # per-net worst slack of EVERY net (net_slack.tcl): report_checks lists only the NPATHS worst
    # endpoints, so guarding reported nets alone lets the pass slow unreported violating paths.
    # Every net on a path has slack <= the path's slack, so protecting all nets with slack <
    # margin freezes every path with slack < margin; the other nets may slow down by at most
    # slack_frac * (slack - margin) each (delay proxy) — paths accumulate several such nets.
    slack = {}
    if net_slack:
        with open(net_slack) as fh:
            for line in fh:
                t = line.rsplit(None, 1)
                if len(t) == 2:
                    try:
                        slack[_key(t[0])] = float(t[1])
                    except ValueError:
                        pass
        tight = {k for k, s in slack.items() if s < slack_margin}
        guarded |= tight
        log(f"net slack: {len(slack):,} nets with slack, {len(tight):,} with slack < {slack_margin} ns "
            f"protected; the rest may slow by <= {slack_frac} x (slack - margin)")
    fl = None
    if crit:
        from rsmt_route import Flute, rsmt_polys
        fl = Flute()

    def span(segs):
        xs = [s[0] for s in segs] + [s[3] for s in segs]
        ys = [s[1] for s in segs] + [s[4] for s in segs]
        return (max(xs) - min(xs) + max(ys) - min(ys)) if segs else 0
    if pw_net is None:
        order = sorted(range(len(nets)), key=lambda i: (_key(nets[i][0]) not in crit, -span(nets[i][1])))
    else:
        order = sorted(range(len(nets)), key=lambda i: (_key(nets[i][0]) not in crit,
                                                         -pw_net.get(_key(nets[i][0]), 0.0) * span(nets[i][1]),
                                                         -span(nets[i][1])))
    new_lines = {}
    stats = defaultdict(int)
    for p in range(passes):
        dcost = 0.0
        for k, i in enumerate(order):
            name, segs, lines = nets[i]
            kname = _key(name)
            if kname not in pins or not segs:
                continue
            soft = guard_soft and kname in guarded and not (protect_crit and kname in crit)
            if ((protect_crit and kname in crit) or kname in guarded) and not soft:
                stats["protected"] += 1
                continue
            w, v = to_wires_vias(segs, la.dirs)
            apply(la, w, v, -1)
            la.tw = tw[kname][0] * tw_scale if kname in crit else 0.0   # tw_scale > 1: timing over score
            if pw_net is not None:
                la.pw = pw_net.get(kname, 0.0)
            c_old = old_cost(la, w, v)
            chosen = used_access_points(segs, pins[kname])
            nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins[kname], chosen)])
            cands = []
            pb = RE.segments_to_polys(segs, nodes)
            if RE.tree_connects(nodes, pb):
                cands.append(("base", pb))
            if kname in crit and fl is not None and len(nodes) > 1:
                cands.append(("rsmt", rsmt_polys(la, fl, nodes, 1.0)))
            best = None
            for tag, polys in cands:
                c = 0.0
                for pts in polys:
                    runs = RE.to_runs(pts)
                    if runs:
                        c += la.assign_path(runs, nodes.get(pts[0]), nodes.get(pts[-1]), 1.0)[1]
                if best is None or c < best[0]:
                    best = (c, tag, polys)
            if best is None or best[0] >= c_old - 1e-12:
                apply(la, w, v, +1)                    # keep the current layers
                stats["kept"] += 1
                continue
            wires, vias = RE.route_net(la, nodes, best[2], 1.0, mark=True)
            if exact:
                # route_net committed the candidate; score old vs new with the EXACT
                # evaluator-model marginal (net's own overlapping runs included)
                apply(la, wires, vias, -1)
                e_new = exact_cost(la, wires, vias)
                e_old = exact_cost(la, w, v)
                slower = False
                if no_slower or soft or kname in slack:
                    la.c_load, la.r_drv = net_rc_params(rc.get(kname), 5.0, 2.0)
                    d_new, d_old = net_delay(la, wires), net_delay(la, w)
                    if no_slower or soft:           # never make a (guarded) net slower (delay proxy)
                        slower = d_new > d_old + 1e-9
                        if soft and slower:
                            stats["soft_rejected"] += 1
                    if not soft and kname in slack and d_new - d_old > slack_frac * (slack[kname] - slack_margin):
                        slower = True
                        stats["slack_rejected"] += 1
                if slower or e_new >= e_old - 1e-12:
                    apply(la, w, v, +1)
                    stats["kept_exact"] += 1
                    continue
                apply(la, wires, vias, +1)
                dcost += e_new - e_old
            else:
                dcost += best[0] - c_old
            buf = emit(la, wires, vias)
            new_lines[i] = buf
            # refresh the stored segments so a second pass starts from the new route
            nets[i] = (name, [(int(s.split()[0]) // GRID, int(s.split()[1]) // GRID,
                               int(s.split()[2][5:]) - 1, int(s.split()[3]) // GRID,
                               int(s.split()[4]) // GRID, int(s.split()[5][5:]) - 1)
                              for s in buf], buf)
            stats["changed_" + best[1]] += 1
            if (k + 1) % 100000 == 0:
                log(f"  pass {p + 1}: {k + 1:,}/{len(order):,} nets, modelled d-score {dcost:+.4f}")
        log(f"pass {p + 1} done: modelled d-score {dcost:+.4f}  {dict(stats)}")
    with open(route_out, "w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for s in new_lines.get(i, lines):
                fh.write(s + "\n")
            fh.write(")\n")
    log(f"wrote {route_out}: wire C {route_wire_cap(nets, la) / 1e6:.3f} nF ({time.time() - t0:.0f}s)")
    return dict(stats)


def self_test():
    ok = True
    cap = {"L": 10, "X": 8, "Y": 8, "layers": [{"name": f"metal{z+1}", "dir": z % 2,
           "cap": np.full((8, 8), 6.0)} for z in range(10)], "short_costs": [1.0] * 10,
           "unit_via": 4.0}
    la = PPALA(cap, 1e-6, 2.1, 5.0, 2.0, pw=0.0)
    # a long x-run: with no power/timing weight the DP is congestion-only; with a power
    # weight it must pick the lowest-capacitance legal x-layer (metal9 dir 0 -> z=8)
    runs = [(0, 3, 0, 7, 0, 3, 7, 3)]
    z0 = la.assign_path(runs, None, None)[0][0]
    la.pw = 1.0
    z1 = la.assign_path(runs, None, None)[0][0]
    xs = [z for z in range(1, 10) if z % 2 == 0]
    want = min(xs, key=lambda z: la.C[z])
    good = z1 == want
    ok &= good
    print(f"  power-weighted run -> metal{z1 + 1} (lowest-C x-layer metal{want + 1}); "
          f"congestion-only picked metal{z0 + 1}: {'ok' if good else 'FAIL'}")
    # old_cost equals the DP cost of the same assignment for a single run
    w, v = {(z1, 3): [(0, 7)]}, set()
    oc = old_cost(la, w, v)
    dp = la.assign_path(runs, None, None)[1]
    good = abs(oc - dp) < 1e-9
    ok &= good
    print(f"  old_cost == DP cost for the same layers: {good} ({oc:.4g} vs {dp:.4g})")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--route")
    ap.add_argument("--ppa", help="PPA report of the base route (switching power for kP)")
    ap.add_argument("--crit", default=None)
    ap.add_argument("--out")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--pin_frac", type=float, default=0.5, help="C_pin / C_wire estimate for kP")
    ap.add_argument("--power_scale", type=float, default=1.0)
    ap.add_argument("--c_load", type=float, default=5.0)
    ap.add_argument("--r_drv", type=float, default=2.0)
    ap.add_argument("--max_crit", type=int, default=4000)
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--protect_crit", action="store_true", help="leave every net on a violating path untouched")
    ap.add_argument("--approx_accept", action="store_true", help="old DP-cost acceptance (inexact overflow)")
    ap.add_argument("--guard_slack", type=float, default=None, help="protect nets on paths with slack < this (ns)")
    ap.add_argument("--no_slower", action="store_true", help="reject moves that increase the net's delay proxy")
    ap.add_argument("--net_slack", default=None, help="per-net worst slack file from net_slack.tcl")
    ap.add_argument("--slack_margin", type=float, default=0.05, help="protect every net with slack < this (ns)")
    ap.add_argument("--slack_frac", type=float, default=0.25,
                    help="other nets may slow by <= this x (slack - margin) (delay proxy, ns)")
    ap.add_argument("--tw_scale", type=float, default=1.0, help="multiplier on the STA timing weights")
    ap.add_argument("--guard_soft", action="store_true",
                    help="guarded nets may still move if their delay proxy does not increase")
    ap.add_argument("--net_act", default=None, help="per-net switching activity from net_activity.tcl")
    ap.add_argument("--vdd", type=float, default=1.1, help="supply voltage (NanGate45 nom_voltage 1.10)")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    run(a.design, a.route, a.ppa, a.out, a.crit, a.passes, a.c_load, a.r_drv, a.pin_frac,
        None, a.power_scale, a.max_crit, a.suite, a.protect_crit, not a.approx_accept, a.guard_slack,
        a.no_slower, a.net_slack, a.slack_margin, a.slack_frac, a.net_act, a.vdd, a.guard_soft, a.tw_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
