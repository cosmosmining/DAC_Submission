#!/usr/bin/env python3
"""rsmt_route.py — pattern-L + contest-exact LA on FLUTE rectilinear Steiner trees.

Same router as `route_export.py --synthetic_l` (per tree edge the cheaper L under
route_export.LayerAssigner, same net order, same writer) — only the per-net tree changes:
    chain / mst : spanning trees over the pin GCells (tree_topology.py)
    rsmt        : FLUTE rectilinear Steiner minimal tree (InstantGR's flute.hpp through
                  tools/libflute_bridge.so), Steiner points become extra tree nodes
so `L_mst` vs `L_rsmt` isolates the tree topology, as tree_ab.sh did for chain vs mst.

Combine with ap_select.py (a filtered .net) to test access points and trees together;
always score against the ORIGINAL .net.

  python rsmt_route.py --cap X.cap --net X.net --out X.L_rsmt.PR_output [--format ispd24]
  python rsmt_route.py --self_test
"""
import argparse
import ctypes
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import route_export as RE                                   # READ-ONLY reuse
from ispd25_adapter import parse_cap_auto, parse_net_ispd25

LIB = os.path.join(ROOT, "tools", "libflute_bridge.so")
LUT_DIR = os.path.join(ROOT, "external", "InstantGR", "run")   # POWV9.dat / POST9.dat


def log(m):
    print(f"[rsmt_route {time.strftime('%H:%M:%S')}] {m}", flush=True)


class Flute:
    def __init__(self, lib=LIB, lut_dir=LUT_DIR):
        self.lib = ctypes.CDLL(lib)
        self.lib.flute_init(os.path.join(lut_dir, "POWV9.dat").encode(),
                            os.path.join(lut_dir, "POST9.dat").encode())
        self.cap = 0
        self._grow(64)

    def _grow(self, d):
        if d <= self.cap:
            return
        self.cap = max(d, 2 * self.cap)
        n, m = self.cap, 2 * self.cap
        self.x, self.y = (ctypes.c_int * n)(), (ctypes.c_int * n)()
        self.ox, self.oy, self.on = (ctypes.c_int * m)(), (ctypes.c_int * m)(), (ctypes.c_int * m)()

    def edges(self, pts):
        """[(x, y)] distinct points -> [((x1, y1), (x2, y2))] Steiner-tree edges, no
        zero-length or duplicate edges."""
        d = len(pts)
        if d < 2:
            return []
        if d == 2:
            return [(pts[0], pts[1])]
        self._grow(d)
        for i, (x, y) in enumerate(pts):
            self.x[i], self.y[i] = x, y
        m = self.lib.flute_tree(d, self.x, self.y, self.ox, self.oy, self.on)
        out, seen = [], set()
        for i in range(m):
            j = self.on[i]
            a, b = (self.ox[i], self.oy[i]), (self.ox[j], self.oy[j])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append((a, b))
        return out


def rsmt_polys(la, fl, nodes, crit=1.0):
    polys = []
    for (px, py), (cx, cy) in fl.edges(list(nodes.keys())):
        best = None
        for mid in ((cx, py), (px, cy)):
            poly = [(px, py), mid, (cx, cy)]
            _, cost = la.assign_path(RE.to_runs(poly), nodes.get((px, py)), nodes.get((cx, cy)), crit)
            if best is None or cost < best[0]:
                best = (cost, poly)
        polys.append(best[1])
    return polys


def export_rsmt(cap_path, net_path, out_path, fmt="ispd24"):
    t0 = time.time()
    cap = parse_cap_auto(cap_path)
    nets = parse_net_ispd25(net_path)
    log(f"loaded {cap['format']} cap {cap['X']}x{cap['Y']}x{cap['L']}, {len(nets):,} nets")
    la = RE.LayerAssigner(cap)
    fl = Flute()
    jobs = []
    for i, n in enumerate(nets):                       # export()'s order with crit = 1
        nodes = RE.pin_nodes(n)
        xs = [p[0] for p in nodes] or [0]
        ys = [p[1] for p in nodes] or [0]
        jobs.append((-1.0, -((max(xs) - min(xs)) + (max(ys) - min(ys))), i, nodes))
    jobs.sort()
    stats = {"nets": len(nets), "local_nets": 0, "steiner_nodes": 0, "disconnected": 0}
    with open(out_path, "w") as fh:
        for k, (_, _, i, nodes) in enumerate(jobs):
            n = nets[i]
            if len(nodes) <= 1:
                polys = []
                stats["local_nets"] += 1
            else:
                polys = rsmt_polys(la, fl, nodes)
                ends = {p for poly in polys for p in (poly[0], poly[-1])}
                stats["steiner_nodes"] += len(ends - set(nodes))
                if not RE.tree_connects(nodes, polys):
                    stats["disconnected"] += 1
            wires, vias = RE.route_net(la, nodes, polys, 1.0, mark=len(n["pins"]) > 1)
            RE.write_net(fh, n["name"], wires, vias, la.dirs, fmt)
            if (k + 1) % 100000 == 0:
                log(f"  routed {k + 1:,}/{len(jobs):,} nets ({time.time() - t0:.0f}s)")
    log(f"wrote {out_path}: {stats} ({time.time() - t0:.0f}s)")
    return stats


def self_test():
    ok = True
    fl = Flute()
    # 4 pins at the corners of a plus: the Steiner tree (length 4) beats any spanning tree (6)
    pts = [(0, 1), (2, 1), (1, 0), (1, 2)]
    e = fl.edges(pts)
    length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in e)
    good = length == 4 and any((1, 1) in (a, b) for a, b in e)
    ok &= good
    print(f"  plus-shaped net: RSMT length {length} (want 4, MST 6), Steiner (1,1) used: {good}")
    # end-to-end on a tiny design: connected, official-format route, no opens
    d = tempfile.mkdtemp(prefix="rsmt_")
    X, Y, L = 4, 4, 4
    cap = [f"{L} {X} {Y}", "1 4 5 5 5 5", " ".join(["10"] * (X - 1)), " ".join(["10"] * (Y - 1))]
    for z, dr in enumerate([0, 0, 1, 0]):
        cap.append(f"M{z + 1} {dr} 0")
        cap += [" ".join(["8"] * X) for _ in range(Y)]
    open(os.path.join(d, "t.cap"), "w").write("\n".join(cap) + "\n")
    open(os.path.join(d, "t.net"), "w").write(
        "p\n(\n[(0, 0, 1)]\n[(0, 2, 1)]\n[(0, 1, 0)]\n[(0, 1, 2)]\n)\n"
        "q\n(\n[(0, 3, 0)]\n[(0, 3, 3)]\n)\n")
    st = export_rsmt(os.path.join(d, "t.cap"), os.path.join(d, "t.net"), os.path.join(d, "t.route"))
    capd = parse_cap_auto(os.path.join(d, "t.cap"))
    opens = RE.check_connectivity(capd, parse_net_ispd25(os.path.join(d, "t.net")),
                                  os.path.join(d, "t.route"))
    good = st["disconnected"] == 0 and st["steiner_nodes"] == 1 and opens[0] == 0
    ok &= good
    print(f"  tiny design: {st}, connectivity {opens}: {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap")
    ap.add_argument("--net")
    ap.add_argument("--out")
    ap.add_argument("--format", default="ispd24", choices=["ispd24", "ispd25"])
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    export_rsmt(a.cap, a.net, a.out, a.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
