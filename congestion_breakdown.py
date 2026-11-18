#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
congestion_breakdown.py — exact Python port of the ISPD'24/'25 contest
congestion score, DECOMPOSED so it is visible where "overflow cost" comes from.

Official evaluator (evaluator.cpp, identical in ISPD'24 and ISPD'25):
  per GCell on routing layer z, demand d = 2*wires + stacked_via_demand
    capacity c >  0.001          : cost += exp(0.5 * ((d - 2c)/2)) = exp(0.25d - 0.5c)
    capacity c <= 0.001 and d > 0: cost += exp(0.5 * 1.5 d)         = exp(0.75d)
  overflow_cost = sum_z unit_overflow_cost[z] * sum_gcells cost
  wl_cost  = unit_length_cost * sum wires * gcell_length
  via_cost = unit_via_cost * (#via segments)

One edge 40 tracks over capacity adds ~e^20 = 5e8; twenty wires across one
capacity-0 (macro-blocked) edge add ~e^30 = 1e13. The sum also includes an
exp(negative) "baseline" term from EVERY uncongested edge, which is why even
contest winners score millions. This tool splits the score into
  blocked  : capacity-0 edges carrying demand     (macro / power-strap blockage)
  overflow : edges with d > 2c                    (wires over capacity)
  baseline : edges within capacity                (always present)
per layer, and lists the worst hotspots.

Layer 0 (metal1) is never a routing layer in either evaluator (read_graph calls
set_direction only for z != 0): it has no capacity, adds no WL or congestion,
and a wire on it is a connectivity violation (the net becomes an open). Vias
from it still cost unit_via. Validated: this port reproduces the official
ISPD'24 evaluator on ariane133_51 exactly (WL 9296578.1002, via 3358316, and
per-layer overflow for metal2..metal10).

Route formats (auto-detected per segment line, as the evaluators do):
  ISPD'24: "xl yl zl xh yh zh"               GCell coordinates, layer indices
  ISPD'25: "xl yl metalA xh yh metalB"       DBU coordinates (GCell = 4200 DBU),
                                             layer = int(name[5:]) - 1
Via demand follows update_stacked_via_counter exactly: a via point counts once
per net, is skipped where the same net already has a wire on the via's lower
layer, and splits 1+1 onto the two adjacent GCells along the layer direction
(2 onto the single neighbour at the grid boundary).

NEW file; reuses ispd25_adapter.parse_cap_auto (ISPD'24 + ISPD'25 headers).

  python3 congestion_breakdown.py --cap X.cap --route X.route [--top 15] [--json out.json]
  python3 congestion_breakdown.py --self_test
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ispd25_adapter import parse_cap_auto          # READ-ONLY reuse

GRID_DBU = 4200          # ISPD'25 evaluator hard-codes this GCell size
SLOPE = 0.5              # evaluator overflow_slope
CAP_EPS = 1e-3


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def iter_route(path):
    """Yield (net_name, [(xl, yl, zl, xh, yh, zh), ...]) with min/max normalized."""
    name, segs = None, []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if name is None:
                name, segs = line, []
                continue
            if line[0] == "(":
                continue
            if line[0] == ")":
                yield name, segs
                name = None
                continue
            t = line.split()
            if len(t) < 6:
                continue
            if _is_int(t[2]):                             # ISPD'24
                xl, yl, zl, xh, yh, zh = (int(v) for v in t[:6])
            else:                                          # ISPD'25
                xl, yl = int(t[0]) // GRID_DBU, int(t[1]) // GRID_DBU
                xh, yh = int(t[3]) // GRID_DBU, int(t[4]) // GRID_DBU
                zl, zh = int(t[2][5:]) - 1, int(t[5][5:]) - 1
            if xl > xh:
                xl, xh = xh, xl
            if yl > yh:
                yl, yh = yh, yl
            if zl > zh:
                zl, zh = zh, zl
            segs.append((xl, yl, zl, xh, yh, zh))
    if name is not None:
        yield name, segs


class Accumulator:
    """Wire / stacked-via / via-count accumulation mirroring evaluator.cpp."""

    def __init__(self, L, X, Y, dirs):
        self.L, self.X, self.Y, self.dirs = L, X, Y, dirs
        # read_graph: `if (z != 0) set_direction(...)` -> layer 0 is never routable.
        # update_stacked_via_counter still uses the file direction for layer 0.
        self.route_dir = [None if z == 0 else d for z, d in enumerate(dirs)]
        self.wires = np.zeros((L, X, Y), dtype=np.int32)
        self.svia = np.zeros((L, X, Y), dtype=np.int32)
        self.total_vias = np.zeros(L, dtype=np.int64)
        self.n_nets = 0
        self.n_bad_segments = 0
        self._h = []          # pending horizontal wires (z, y, xl, xh)
        self._v = []          # pending vertical wires   (z, x, yl, yh)
        self._dh_z, self._dv_z = [], []

    def _flush(self):
        if self._h:
            self._dh_z.append(np.asarray(self._h, dtype=np.int64))
            self._h = []
        if self._v:
            self._dv_z.append(np.asarray(self._v, dtype=np.int64))
            self._v = []

    def begin(self):
        self._dh_z, self._dv_z = [], []

    def add_net(self, segs):
        self.n_nets += 1
        L, X, Y, dirs = self.L, self.X, self.Y, self.dirs
        wires_on = {}                      # z -> list of (fixed, lo, hi) inclusive
        vias = []
        for (xl, yl, zl, xh, yh, zh) in segs:
            if zl != zh:                                   # via
                if xl != xh or yl != yh or zh >= L:
                    self.n_bad_segments += 1
                    continue
                for z in range(zl, zh):
                    self.total_vias[z] += 1
                    vias.append((xl, yl, z))
            else:                                          # wire
                z = zl
                if z >= L:
                    self.n_bad_segments += 1
                    continue
                rd = self.route_dir[z]
                if rd == 0 and yl == yh and xh > xl:
                    self._h.append((z, yl, xl, xh))
                    wires_on.setdefault(z, []).append((yl, xl, xh))
                elif rd == 1 and xl == xh and yh > yl:
                    self._v.append((z, xl, yl, yh))
                    wires_on.setdefault(z, []).append((xl, yl, yh))
                else:                      # wrong direction or non-routing layer 0
                    self.n_bad_segments += 1
        seen = set()
        for (x, y, z) in vias:
            key = (x, y, z)
            if key in seen:
                continue
            covered = False
            for (fixed, lo, hi) in wires_on.get(z, ()):
                if dirs[z] == 0 and fixed == y and lo <= x <= hi:
                    covered = True
                    break
                if dirs[z] == 1 and fixed == x and lo <= y <= hi:
                    covered = True
                    break
            seen.add(key)
            if covered:
                continue
            if dirs[z] == 0:
                if 0 < x < X - 1:
                    self.svia[z, x - 1, y] += 1
                    self.svia[z, x, y] += 1
                elif x > 0:
                    self.svia[z, x - 1, y] += 2
                elif x < X - 1:
                    self.svia[z, x, y] += 2
            elif dirs[z] == 1:
                if 0 < y < Y - 1:
                    self.svia[z, x, y - 1] += 1
                    self.svia[z, x, y] += 1
                elif y > 0:
                    self.svia[z, x, y - 1] += 2
                elif y < Y - 1:
                    self.svia[z, x, y] += 2
        if len(self._h) + len(self._v) > 200_000:
            self._flush()

    def finish(self):
        self._flush()
        L, X, Y = self.L, self.X, self.Y
        dh = np.zeros((L, X + 1, Y), dtype=np.int32)
        dv = np.zeros((L, X, Y + 1), dtype=np.int32)
        for a in self._dh_z:          # columns: z, y, xl, xh  -> covers x in [xl, xh)
            np.add.at(dh, (a[:, 0], a[:, 2], a[:, 1]), 1)
            np.add.at(dh, (a[:, 0], a[:, 3], a[:, 1]), -1)
        for a in self._dv_z:          # columns: z, x, yl, yh  -> covers y in [yl, yh)
            np.add.at(dv, (a[:, 0], a[:, 1], a[:, 2]), 1)
            np.add.at(dv, (a[:, 0], a[:, 1], a[:, 3]), -1)
        self.wires += np.cumsum(dh, axis=1)[:, :X, :].astype(np.int32)
        self.wires += np.cumsum(dv, axis=2)[:, :, :Y].astype(np.int32)


def score(cap, acc, top=15):
    L, X, Y = cap["L"], cap["X"], cap["Y"]
    unit_of = cap["short_costs"]
    xe = np.append(cap["x_edge_len"], 0.0)       # cell_width(x) = x_edge_len[x]
    ye = np.append(cap["y_edge_len"], 0.0)
    rep = {"layers": [], "hotspots": []}
    wl_cost = via_cost = of_cost = 0.0
    parts = {"blocked": 0.0, "overflow": 0.0, "baseline": 0.0}
    n_inf = 0
    cand = []
    for z in range(L):
        lay = cap["layers"][z]
        via_cost += float(acc.total_vias[z]) * cap["unit_via"]
        if z == 0 or lay["dir"] not in (0, 1):   # layer 0 is not a routing layer
            continue
        c = lay["cap"]
        w = acc.wires[z].astype(np.float64)
        d = 2.0 * w + acc.svia[z]
        open_ = c > CAP_EPS
        with np.errstate(over="ignore"):
            cost = np.where(open_, np.exp(SLOPE * (d - 2.0 * c) / 2.0),
                            np.where(d > 0, np.exp(SLOPE * 1.5 * d), 0.0))
        n_inf += int(np.isinf(cost).sum())
        blk = (~open_) & (d > 0)
        ovf = open_ & (d > 2.0 * c)
        base = open_ & ~ovf
        u = unit_of[z]
        b_c, o_c, s_c = (float(cost[m].sum()) * u for m in (blk, ovf, base))
        parts["blocked"] += b_c
        parts["overflow"] += o_c
        parts["baseline"] += s_c
        of_cost += b_c + o_c + s_c
        if lay["dir"] == 0:
            wl = float((w * xe[:, None]).sum())
        else:
            wl = float((w * ye[None, :]).sum())
        wl_cost += wl * cap["unit_wire"]
        over_tracks = np.where(open_, d / 2.0 - c, d / 2.0)
        rep["layers"].append({
            "layer": lay["name"], "dir": lay["dir"],
            "blocked_cost": b_c, "overflow_cost": o_c, "baseline_cost": s_c,
            "blocked_gcells_used": int(blk.sum()),
            "overflowed_gcells": int(ovf.sum()),
            "max_over_tracks": float(over_tracks.max()) if over_tracks.size else 0.0,
            "wires": int(acc.wires[z].sum()), "vias_from_layer": int(acc.total_vias[z]),
        })
        if top:
            flat = (cost * u).ravel()
            k = min(top, flat.size)
            idx = np.argpartition(flat, -k)[-k:]
            for i in idx:
                x, y = divmod(int(i), Y)
                cand.append((float(flat[i]), lay["name"], x, y, float(c[x, y]),
                             float(d[x, y]) / 2.0,
                             "blocked" if not open_[x, y] else
                             ("overflow" if d[x, y] > 2 * c[x, y] else "baseline")))
    cand.sort(key=lambda t: -t[0] if math.isfinite(t[0]) else -math.inf)
    rep["hotspots"] = [{"cost": v, "layer": n, "x": x, "y": y, "cap": c_,
                        "demand_tracks": dt, "class": k}
                       for (v, n, x, y, c_, dt, k) in cand[:top]]
    rep["summary"] = {
        "wirelength_cost": wl_cost, "via_cost": via_cost,
        "overflow_cost": of_cost, "total_cost": wl_cost + via_cost + of_cost,
        "overflow_share": {k: (v / of_cost if of_cost else 0.0)
                           for k, v in parts.items()},
        "overflow_parts": parts, "inf_gcells": n_inf,
        "nets_in_route": acc.n_nets, "malformed_segments": acc.n_bad_segments,
    }
    return rep


def analyze(cap_path, route_path, top=15):
    cap = parse_cap_auto(cap_path)
    dirs = [l["dir"] for l in cap["layers"]]
    acc = Accumulator(cap["L"], cap["X"], cap["Y"], dirs)
    acc.begin()
    for _, segs in iter_route(route_path):
        acc.add_net(segs)
    acc.finish()
    rep = score(cap, acc, top)
    rep["summary"]["cap_format"] = cap["format"]
    return rep, acc


def print_report(rep):
    s = rep["summary"]
    print(f"nets={s['nets_in_route']}  malformed_segments={s['malformed_segments']}"
          f"  inf_gcells={s['inf_gcells']}  ({s.get('cap_format', '?')} .cap)")
    print(f"wirelength cost {s['wirelength_cost']:.4f}   via cost {s['via_cost']:.4f}   "
          f"overflow cost {s['overflow_cost']:.4f}   total cost {s['total_cost']:.4f}")
    sh = s["overflow_share"]
    print(f"overflow share: blocked {100*sh['blocked']:.2f}%  "
          f"over-capacity {100*sh['overflow']:.2f}%  baseline {100*sh['baseline']:.2f}%")
    print(f"{'layer':8}{'dir':>4}{'blocked':>14}{'over-cap':>14}{'baseline':>14}"
          f"{'#blkUsed':>10}{'#ovf':>8}{'maxOver':>9}")
    for l in rep["layers"]:
        print(f"{l['layer']:8}{l['dir']:>4}{l['blocked_cost']:>14.4g}"
              f"{l['overflow_cost']:>14.4g}{l['baseline_cost']:>14.4g}"
              f"{l['blocked_gcells_used']:>10}{l['overflowed_gcells']:>8}"
              f"{l['max_over_tracks']:>9.1f}")
    print("worst hotspots:")
    for h in rep["hotspots"]:
        print(f"  {h['cost']:>12.4g}  {h['layer']:8} ({h['x']},{h['y']}) "
              f"cap={h['cap']:.0f} demand={h['demand_tracks']:.1f} [{h['class']}]")


def self_test():
    """Hand-built 3-layer 3x2 case: layer 0 unroutable (evaluator rule), a wire
    crossing a blocked GCell, a covered via, an uncovered via, a malformed
    metal1 wire, and both route formats."""
    import tempfile
    cap_txt = ("3 3 2\n"
               "0.5 4 5 5 5\n"                 # unit_wire, unit_via, unit_of[z]
               "10 10\n7\n"                    # x lengths (X-1), y lengths (Y-1)
               "metal1 0 1\n1 1 1\n1 1 1\n"    # layer 0: never a routing layer
               "metal2 1 1\n1 1 1\n1 1 1\n"
               "metal3 0 1\n1 0 1\n1 1 1\n")   # rows y=0, y=1 ; cap(1,0)=0 blocked
    route24 = ("A\n(\n0 0 2 2 0 2\n2 0 1 2 0 2\n2 0 1 2 1 1\n)\n"
               "B\n(\n1 1 0 1 1 1\n)\n"
               "C\n(\n0 1 0 1 1 0\n)\n")
    route25 = ("A\n(\n0 0 metal3 8400 0 metal3\n8400 0 metal2 8400 0 metal3\n"
               "8400 0 metal2 8400 4200 metal2\n)\n"
               "B\n(\n4200 4200 metal1 4200 4200 metal2\n)\n"
               "C\n(\n0 4200 metal1 4200 4200 metal1\n)\n")
    fc = tempfile.NamedTemporaryFile("w", suffix=".cap", delete=False)
    fc.write(cap_txt)
    fc.close()
    reps = []
    for rt in (route24, route25):
        fr = tempfile.NamedTemporaryFile("w", suffix=".route", delete=False)
        fr.write(rt)
        fr.close()
        reps.append(analyze(fc.name, fr.name, top=3))
        os.unlink(fr.name)
    os.unlink(fc.name)
    (r24, a24), (r25, a25) = reps
    exp_w = np.zeros((3, 3, 2), np.int32)
    exp_w[2, 0, 0] = exp_w[2, 1, 0] = 1   # A: metal3 x in [0,2) on y=0, crosses blocked (1,0)
    exp_w[1, 2, 0] = 1                    # A: metal2 y in [0,1) on x=2
    exp_s = np.zeros((3, 3, 2), np.int32)
    exp_s[0, 0, 1] = exp_s[0, 1, 1] = 1   # B: via on layer 0 splits 1+1 (never scored)
    # A's via at (2,0,z1) is covered by A's own metal2 wire -> no stacked demand;
    # C's metal1 wire is on the non-routing layer 0 -> malformed, not counted
    for acc in (a24, a25):
        assert (acc.wires == exp_w).all(), acc.wires
        assert (acc.svia == exp_s).all(), acc.svia
        assert list(acc.total_vias) == [1, 1, 0], acc.total_vias
        assert acc.n_bad_segments == 1
    capm = np.array([[[1, 1], [1, 1], [1, 1]],
                     [[1, 1], [1, 1], [1, 1]],
                     [[1, 1], [0, 1], [1, 1]]], float)      # [z][x][y]
    d = 2.0 * exp_w + exp_s
    expc = 0.0
    for z in (1, 2):                                         # layer 0 not scored
        for x in range(3):
            for y in range(2):
                c = capm[z, x, y]
                if c > CAP_EPS:
                    expc += 5 * math.exp(0.25 * d[z, x, y] - 0.5 * c)
                elif d[z, x, y] > 0:
                    expc += 5 * math.exp(0.75 * d[z, x, y])
    for r in (r24, r25):
        s = r["summary"]
        assert abs(s["overflow_cost"] - expc) < 1e-9 * max(1, expc), (s, expc)
        assert abs(s["via_cost"] - 8.0) < 1e-12              # 2 via segments * 4
        assert abs(s["wirelength_cost"] - 0.5 * (10 + 10 + 7)) < 1e-12
        assert [l["layer"] for l in r["layers"]] == ["metal2", "metal3"]
        assert r["layers"][1]["blocked_gcells_used"] == 1
        assert s["malformed_segments"] == 1
    assert r24["summary"]["overflow_cost"] == r25["summary"]["overflow_cost"]
    print("  [PASS] wire / via-dedup / stacked-via split match evaluator rules")
    print("  [PASS] layer 0 not scored; metal1 wire flagged as violation")
    print("  [PASS] overflow, via, wirelength cost match the evaluator formula")
    print("  [PASS] ISPD'24 GCell and ISPD'25 DBU route formats agree")
    print("  [PASS] blocked-edge usage detected")
    print("5 checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap")
    ap.add_argument("--route")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not (a.cap and a.route):
        return self_test()
    rep, _ = analyze(a.cap, a.route, a.top)
    print_report(rep)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
