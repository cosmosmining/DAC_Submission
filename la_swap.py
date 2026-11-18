#!/usr/bin/env python3
"""la_swap.py — keep a reference route's 2-D topology, redo ONLY layer assignment.

Question it answers: at IDENTICAL 2-D topology, how much of a reference router's
score comes from its layer assignment (LA) vs ours (route_export.LayerAssigner,
contest-exact DP)?

    score(reference) - score(la_swap(reference))  = value of the reference LA over ours
    score(la_swap(reference)) - score(our route)  = value of the reference 2-D topology

Why a new tool: route_export.export(segments=...) imports another route's 2-D wires but
attaches them to OUR pin nodes — every pin at its LOWEST access point (pin_nodes). A router
that connects a pin through a different access point (InstantGR does this for most
multi-pin nets on ariane133_68) then fails tree_connects and silently falls back to
pattern-L, so the "swap" measures our own router. Here each pin is placed at the access
point the reference route actually touches (same test as the official evaluator's
check_connectivity), so the imported topology stays connected.

route_export.pin_nodes is replaced ONLY inside this process (no file is modified).

  python la_swap.py --cap X.cap --net X.net --ref X.instantgr.PR_output --out X.swap.PR_output
  python la_swap.py --self_test
"""
import argparse
import os
import sys
import tempfile
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import route_export as RE                                   # READ-ONLY reuse
from congestion_breakdown import iter_route
from ispd25_adapter import parse_net_ispd25


def log(m):
    print(f"[la_swap {time.strftime('%H:%M:%S')}] {m}", flush=True)


def used_access_points(segs, pins):
    """-> [ (l, x, y) or None ] per pin: the first access point the route touches."""
    wire_at = defaultdict(set)      # (x, y) -> layers of wires covering the GCell
    via_at = defaultdict(list)      # (x, y) -> [(zl, zh)]
    ap_gc = {(x, y) for aps in pins for (_, x, y) in aps}
    rows, cols = defaultdict(set), defaultdict(set)
    for (x, y) in ap_gc:
        rows[y].add(x)
        cols[x].add(y)
    for xl, yl, zl, xh, yh, zh in segs:
        if zl != zh:
            if xl == xh and yl == yh and (xl, yl) in ap_gc:
                via_at[(xl, yl)].append((zl, zh))
        elif yl == yh:
            for x in rows.get(yl, ()):
                if xl <= x <= xh:
                    wire_at[(x, yl)].add(zl)
        else:
            for y in cols.get(xl, ()):
                if yl <= y <= yh:
                    wire_at[(xl, y)].add(zl)
    out = []
    for aps in pins:
        hit = None
        for (z, x, y) in aps:
            g = (x, y)
            if z in wire_at.get(g, ()) or any(a <= z <= b for a, b in via_at.get(g, ())):
                hit = (z, x, y)
                break
        out.append(hit)
    return out


def nodes_from_aps(chosen):
    """pin_nodes()-compatible dict {(x, y): (lowest layer, highest layer)}."""
    nodes = {}
    for (l0, x0, y0) in chosen:
        prev = nodes.get((x0, y0))
        nodes[(x0, y0)] = (l0, l0) if prev is None else (min(prev[0], l0), max(prev[1], l0))
    return nodes


def build_node_map(net_path, ref_path):
    nets = {n["name"]: n["pins"] for n in parse_net_ispd25(net_path)}
    node_map, stats = {}, defaultdict(int)
    for name, segs in iter_route(ref_path):
        pins = nets.get(name)
        if pins is None:
            continue
        chosen = used_access_points(segs, pins)
        for aps, c in zip(pins, chosen):
            if c is None:
                stats["pins_untouched"] += 1          # keep our lowest-AP choice
            elif c != min(aps):
                stats["pins_moved_off_lowest_ap"] += 1
            else:
                stats["pins_at_lowest_ap"] += 1
        node_map[name] = nodes_from_aps([c if c is not None else min(aps)
                                         for aps, c in zip(pins, chosen)])
    return node_map, dict(stats)


def swap(cap_path, net_path, ref_path, out_path, fmt="ispd24", tree="mst", ap_mode="used"):
    orig = RE.pin_nodes
    stats = {}
    if ap_mode == "used":
        t0 = time.time()
        node_map, stats = build_node_map(net_path, ref_path)
        log(f"access points from reference: {stats} ({time.time() - t0:.0f}s)")
        RE.pin_nodes = lambda n: node_map.get(n["name"]) or orig(n)
    try:
        _, _, st = RE.export(cap_path, net_path, out_path, fmt=fmt, segments=ref_path, tree=tree)
    finally:
        RE.pin_nodes = orig
    st.update(stats)
    routed = st["nets"] - st["local_nets"]
    log(f"imported topology kept for {st['guide_nets']:,}/{routed:,} multi-GCell nets "
        f"({100.0 * st['guide_nets'] / max(1, routed):.1f}%), fallback_L {st['fallback_L']:,}")
    return st


# ─────────────────────────────── self test ───────────────────────────────

def self_test():
    """A 2-pin net whose second pin has two access points; the reference route uses the
    NON-lowest one. lowest-AP mode must fall back to L, used-AP mode must keep the route."""
    d = tempfile.mkdtemp(prefix="la_swap_")
    X, Y, L = 6, 5, 4
    cap = [f"{L} {X} {Y}", "1 4 5 5 5 5", " ".join(["10"] * (X - 1)), " ".join(["10"] * (Y - 1))]
    for z, dr in enumerate([0, 0, 1, 0]):
        cap.append(f"M{z + 1} {dr} 0")
        cap += [" ".join(["8"] * X) for _ in range(Y)]
    open(os.path.join(d, "t.cap"), "w").write("\n".join(cap) + "\n")
    # pin 2 access points: (0,4,0) [lowest] and (0,4,3); reference uses (4,3)
    open(os.path.join(d, "t.net"), "w").write("a\n(\n[(0, 0, 3)]\n[(0, 4, 0), (0, 4, 3)]\n)\n")
    open(os.path.join(d, "ref.route"), "w").write(
        "a\n(\n0 3 0 0 3 1\n0 3 1 4 3 1\n4 3 0 4 3 1\n)\n")
    ok = True
    for mode, want_kept in (("lowest", 0), ("used", 1)):
        out = os.path.join(d, f"{mode}.route")
        st = swap(os.path.join(d, "t.cap"), os.path.join(d, "t.net"),
                  os.path.join(d, "ref.route"), out, ap_mode=mode)
        good = st["guide_nets"] == want_kept
        ok &= good
        print(f"  ap_mode={mode:6s} kept={st['guide_nets']} want={want_kept} "
              f"{'ok' if good else 'FAIL'}")
        cap_d = RE.parse_cap_auto(os.path.join(d, "t.cap"))
        nets = parse_net_ispd25(os.path.join(d, "t.net"))
        conn = RE.check_connectivity(cap_d, nets, out)
        print(f"    connectivity: {conn}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap")
    ap.add_argument("--net")
    ap.add_argument("--ref", help="reference 3-D route whose 2-D topology is kept")
    ap.add_argument("--out")
    ap.add_argument("--format", default="ispd24", choices=["ispd24", "ispd25"])
    ap.add_argument("--tree", default="mst", choices=["chain", "mst"], help="fallback-L tree only")
    ap.add_argument("--ap_mode", default="used", choices=["used", "lowest"])
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    swap(a.cap, a.net, a.ref, a.out, a.format, a.tree, a.ap_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
