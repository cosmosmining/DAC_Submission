#!/usr/bin/env python3
"""seg_normalize.py — make an ISPD'25 route's segment graph explicit for OpenROAD STA.

The contest congestion evaluator marks every GCell a wire covers, so a pin (or another
wire / via of the same net) touching the INTERIOR of a straight wire counts as connected.
OpenROAD's `read_global_route_segments` + `estimate_parasitics -global_routing` build the
net's RC tree from segment END POINTS only:
  * a pin in the middle of a wire -> "[WARNING GRT-xxxx] Missing route to pin ..." and that
    pin gets no wire parasitics (optimistic WNS/TNS, lower switching power);
  * a branch landing mid-wire (T-junction) is not joined to the trunk in the RC tree.
Found on our own routes: 24-30 missing pins on mempool_tile_wrap, 115-490 on NVDLA (all our
NVDLA routes reported the same WNS -48.10); InstantGR's writer splits at pins and had 0.

This tool splits every wire at every interior point where the SAME net has a segment end,
a via, or a pin access point on that layer. The GCell coverage — hence the official
congestion score — is unchanged (checked in the self-test with congestion_breakdown).

  python seg_normalize.py --net D.net --route D.X.route --out D.X.norm.route
  python seg_normalize.py --self_test
"""
import argparse
import os
import sys
import tempfile
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ispd25_adapter import parse_net_ispd25                 # READ-ONLY reuse

GRID = 4200
HALF = GRID // 2


def log(m):
    print(f"[seg_normalize {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _gc(v):
    return int(v) // GRID


def _dbu(g):
    return g * GRID + HALF


def normalize_net(segs, pin_pts):
    """segs: [(x1, y1, z1, x2, y2, z2)] in GCells (z 0-based); pin_pts: {(x, y, z)}.
    -> (list of segments, number of extra splits)."""
    wires, vias = [], []
    for x1, y1, z1, x2, y2, z2 in segs:
        if z1 != z2:
            vias.append((x1, y1, min(z1, z2), max(z1, z2)))
        elif (x1, y1) != (x2, y2):
            wires.append((x1, y1, x2, y2, z1))
    pts = set(pin_pts)
    for x1, y1, x2, y2, z in wires:
        pts.add((x1, y1, z))
        pts.add((x2, y2, z))
    for x, y, zl, zh in vias:
        for z in range(zl, zh + 1):
            pts.add((x, y, z))
    by_row, by_col = defaultdict(list), defaultdict(list)       # (z, y) -> xs ; (z, x) -> ys
    for (x, y, z) in pts:
        by_row[(z, y)].append(x)
        by_col[(z, x)].append(y)
    out, extra = [], 0
    for x1, y1, x2, y2, z in wires:
        if y1 == y2:
            lo, hi = min(x1, x2), max(x1, x2)
            cuts = sorted({lo, hi} | {c for c in by_row[(z, y1)] if lo < c < hi})
            extra += len(cuts) - 2
            out += [(a, y1, z, b, y1, z) for a, b in zip(cuts, cuts[1:])]
        elif x1 == x2:
            lo, hi = min(y1, y2), max(y1, y2)
            cuts = sorted({lo, hi} | {c for c in by_col[(z, x1)] if lo < c < hi})
            extra += len(cuts) - 2
            out += [(x1, a, z, x1, b, z) for a, b in zip(cuts, cuts[1:])]
        else:                                       # diagonal: not legal, keep as is
            out.append((x1, y1, z, x2, y2, z))
    for x, y, zl, zh in vias:
        out.append((x, y, zl, x, y, zh))
    return out, extra


def normalize(net_path, route_in, route_out):
    t0 = time.time()
    pins = {n["name"]: {(x, y, l) for aps in n["pins"] for (l, x, y) in aps}
            for n in parse_net_ispd25(net_path)}
    log(f"{len(pins):,} nets parsed ({time.time() - t0:.0f}s)")
    stats = defaultdict(int)
    with open(route_in) as fi, open(route_out, "w") as fo:
        name, segs = None, []
        for raw in fi:
            s = raw.strip()
            if not s:
                continue
            if name is None:
                name, segs = s, []
                continue
            if s == "(":
                continue
            if s == ")":
                out, extra = normalize_net(segs, pins.get(name, set()))
                stats["nets"] += 1
                stats["segments_in"] += len(segs)
                stats["segments_out"] += len(out)
                stats["splits"] += extra
                stats["nets_split"] += extra > 0
                fo.write(f"{name}\n(\n")
                for x1, y1, z1, x2, y2, z2 in out:
                    fo.write(f"{_dbu(x1)} {_dbu(y1)} metal{z1 + 1} {_dbu(x2)} {_dbu(y2)} metal{z2 + 1}\n")
                fo.write(")\n")
                name = None
                continue
            t = s.split()
            if len(t) != 6:
                continue
            segs.append((_gc(t[0]), _gc(t[1]), int(t[2][5:]) - 1,
                         _gc(t[3]), _gc(t[4]), int(t[5][5:]) - 1))
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"wrote {route_out}: {dict(stats)}")
    return dict(stats)


def self_test():
    from congestion_breakdown import iter_route
    ok = True
    # pin mid-wire + T-junction + via mid-wire
    segs = [(0, 5, 2, 10, 5, 2),          # metal3 trunk along x
            (4, 5, 1, 4, 5, 2),           # via up to the trunk at x=4 (mid-wire)
            (7, 0, 3, 7, 5, 3),           # metal4 branch landing at x=7 ...
            (7, 5, 2, 7, 5, 3)]           # ... via into the trunk at x=7 (mid-wire)
    out, extra = normalize_net(segs, {(2, 5, 2)})         # pin on metal3 at x=2 (mid-wire)
    trunk = sorted((s[0], s[3]) for s in out if s[2] == s[5] == 2)
    good = trunk == [(0, 2), (2, 4), (4, 7), (7, 10)] and extra == 3
    ok &= good
    print(f"  trunk split at pin/via/T points -> {trunk}, extra={extra}: {'ok' if good else 'FAIL'}")
    # coverage identical -> official congestion accounting unchanged
    d = tempfile.mkdtemp(prefix="segnorm_")
    open(os.path.join(d, "t.net"), "w").write("n\n(\np1, 0.0, [(2, 2, 5)]\np2, 0.0, [(1, 4, 5)]\n)\n")
    with open(os.path.join(d, "a.route"), "w") as fh:
        fh.write("n\n(\n")
        for x1, y1, z1, x2, y2, z2 in segs:
            fh.write(f"{_dbu(x1)} {_dbu(y1)} metal{z1+1} {_dbu(x2)} {_dbu(y2)} metal{z2+1}\n")
        fh.write(")\n")
    normalize(os.path.join(d, "t.net"), os.path.join(d, "a.route"), os.path.join(d, "b.route"))

    def cover(p):
        c = set()
        for _, ss in iter_route(p):
            for xl, yl, zl, xh, yh, zh in ss:
                if zl == zh:
                    c |= {(x, y, zl) for x in range(xl, xh + 1) for y in range(yl, yh + 1)}
                else:
                    c |= {(xl, yl, z) for z in range(zl, zh + 1)}
        return c
    same = cover(os.path.join(d, "a.route")) == cover(os.path.join(d, "b.route"))
    ok &= same
    print(f"  GCell coverage unchanged: {same}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net")
    ap.add_argument("--route")
    ap.add_argument("--out")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    normalize(a.net, a.route, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
