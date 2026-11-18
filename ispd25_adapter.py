#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ispd25_adapter.py — parse ISPD'25 (Performance-Driven Large-Scale Global
Routing) simplified inputs. The .cap format is identical to ISPD'24 (reused
from ispd24_adapter); the .net format is EXTENDED with a per-pin name and a
per-pin PRE-ROUTING SLACK estimate, which is the timing signal that feeds
timing_weights.py.

ISPD'25 .net line (inside a net's parentheses):
    <pin_name>, <slack_seconds>, [(layer, gcell_x, gcell_y), ...]

This module adds `parse_net_ispd25()` (captures name + slack + access points)
and `net_slacks()` (per-net worst slack, ns), and reuses ispd24_adapter for
the .cap grid. NEW file; ispd24_adapter and all core files are untouched.

  python3 ispd25_adapter.py <design>.net            # summary + slack stats
  python3 ispd25_adapter.py --self_test
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import ispd24_adapter as A24                        # reuse parse_cap READ-ONLY

# access points use comma OR space separators: (0, 955, 350) or (0 955 350)
_AP = re.compile(r"\(\s*(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)\s*\)")


def parse_net_ispd25(path, max_nets=None):
    """Parse ISPD'25 .net -> list of dicts:
       {name, pins:[[(l,x,y),...]], pin_names:[...], pin_slack:[float seconds]}.
    Robust to the two-line net header ('Name' then '(') and to escaped
    bus-index brackets in pin names (gen_tiles\\[8\\]...)."""
    nets = []
    name, pins, pnames, pslack, depth = None, None, None, None, 0
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "(":
                depth += 1
                continue
            if line == ")":
                depth -= 1
                if depth == 0 and name is not None:
                    nets.append({"name": name, "pins": pins,
                                 "pin_names": pnames, "pin_slack": pslack})
                    name, pins, pnames, pslack = None, None, None, None
                    if max_nets and len(nets) >= max_nets:
                        break
                continue
            if depth == 0:
                name = line.split()[0]
                pins, pnames, pslack = [], [], []
            else:
                # pin line: "<pin_name>, <slack>, [(l,x,y), ...]"
                # split on '[' so the AP commas don't collide with the slack
                head, _, tail = line.partition("[")
                aps = [(int(a), int(b), int(c))
                       for a, b, c in _AP.findall(tail if tail else head)]
                if not aps:
                    continue
                pins.append(aps)
                hp = head.split(",")
                pnames.append(hp[0].strip())
                sv = float("nan")
                if len(hp) > 1:
                    try:
                        sv = float(hp[1])
                    except ValueError:
                        sv = float("nan")
                pslack.append(sv)
    return nets


def net_slacks(nets, ns=True):
    """Per-net worst (min) pin slack. Returns list aligned with nets.
    ns=True -> nanoseconds (ISPD'25 slack is in seconds)."""
    import math
    out = []
    scale = 1e9 if ns else 1.0
    for n in nets:
        vals = [s for s in n["pin_slack"] if not math.isnan(s)]
        out.append(min(vals) * scale if vals else float("nan"))
    return out


def summarize(net_path):
    nets = parse_net_ispd25(net_path)
    sl = net_slacks(nets)
    import math
    fin = [s for s in sl if not math.isnan(s)]
    neg = [s for s in fin if s < 0]
    return {
        "nets": len(nets),
        "nets_with_slack": len(fin),
        "nets_negative_slack": len(neg),
        "worst_slack_ns": round(min(fin), 4) if fin else None,
        "mean_slack_ns": round(sum(fin) / len(fin), 4) if fin else None,
        "TNS_ns(pre-routing)": round(sum(neg), 4) if neg else 0.0,
    }


def self_test():
    """parse a tiny synthetic ISPD'25 .net (no benchmark needed)."""
    import tempfile
    txt = ("Net0\n(\nA/Z, -8.14435e-10, [(0, 955, 350)]\n"
           "B/D, 1.18725e-09, [(0, 955, 348)]\n)\n"
           "Net1\n(\nC/Q, -1.16145e-09, [(0, 1314, 672)]\n"
           "D/A, 7.58446e-10, [(2, 1387, 911)]\n)\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".net", delete=False)
    f.write(txt); f.close()
    nets = parse_net_ispd25(f.name)
    assert len(nets) == 2, "net count"
    assert nets[0]["pins"][0] == [(0, 955, 350)], "AP parse"
    sl = net_slacks(nets)
    assert abs(sl[0] - (-0.814435)) < 1e-4, f"slack ns {sl[0]}"
    assert abs(sl[1] - (-1.16145)) < 1e-4, f"slack ns {sl[1]}"
    # slack -> timing weights
    import timing_weights as TW
    w = TW.criticality_from_slack(sl, mode="exp", w_max=8.0)
    assert w.min() >= 1.0 and w.max() <= 8.0 and w[1] >= w[0]  # net1 worse
    os.unlink(f.name)
    print("  [PASS] parse_net_ispd25 (name+slack+AP)")
    print("  [PASS] net_slacks in ns")
    print("  [PASS] slack -> timing_weights integration")
    print("3 checks passed")


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_cap_auto(path):
    """Parse an ISPD'24 OR ISPD'25 .cap into the ispd24_adapter.parse_cap dict.

    BUG FIX: ISPD'25 inserts 4 PPA weights (w1..w4) on header line 2, right
    after the per-layer overflow costs ("remember to update your parser" —
    contest page). ispd24_adapter.parse_cap does not skip them, so on ISPD'25
    files every edge length and capacity is shifted by 4 tokens (it crashes on
    the first layer name; a parser without that check silently misreads the
    whole capacity map). Format is detected from where the first layer NAME
    token sits. Vectorized (numpy) so 200 MB caps parse in seconds.
    """
    import numpy as np
    toks = open(path).read().split()
    L, X, Y = int(toks[0]), int(toks[1]), int(toks[2])
    first_24 = 5 + L + (X - 1) + (Y - 1)
    if first_24 + 4 < len(toks) and _is_num(toks[first_24]) \
            and not _is_num(toks[first_24 + 4]):
        fmt, nw = "ispd25", 4
    elif not _is_num(toks[first_24]):
        fmt, nw = "ispd24", 0
    else:
        raise ValueError(f"{path}: unrecognized .cap header layout")
    short = [float(t) for t in toks[5:5 + L]]
    weights = [float(t) for t in toks[5 + L:5 + L + nw]] if nw else None
    i = 5 + L + nw
    xe = np.array(toks[i:i + X - 1], dtype=np.float64); i += X - 1
    ye = np.array(toks[i:i + Y - 1], dtype=np.float64); i += Y - 1
    layers = []
    for _ in range(L):
        name, d, ml = toks[i], int(toks[i + 1]), float(toks[i + 2]); i += 3
        cap = np.array(toks[i:i + X * Y], dtype=np.float64).reshape(Y, X).T
        i += X * Y
        layers.append({"name": name, "dir": d, "min_len": ml, "cap": cap})
    if i != len(toks):
        raise ValueError(f"{path}: .cap parse incomplete ({fmt}), "
                         f"{len(toks) - i} tokens left")
    return {"L": L, "X": X, "Y": Y, "unit_wire": float(toks[3]),
            "unit_via": float(toks[4]), "short_costs": short,
            "x_edge_len": xe, "y_edge_len": ye, "layers": layers,
            "format": fmt, "ppa_weights": weights}


def nets_for_convert(nets):
    """ISPD'25 net dicts -> (name, pin_groups) tuples for ispd24_adapter.convert."""
    return [(n["name"], n["pins"]) for n in nets]


def self_test_cap():
    """synthetic ISPD'24 and ISPD'25 caps (2 layers, 3x2 grid) parse identically."""
    import tempfile
    body = ("3 4 5\n"                         # x_edge_len (X-1=2), y_edge_len (Y-1=1)
            "metal1 0 1\n1 2 3\n4 5 6\n"
            "metal2 1 1\n7 8 9\n10 11 12\n")
    c24 = "2 3 2\n0.5 4 5 5\n" + body
    c25 = "2 3 2\n0.5 4 5 5 -1 -10 300 7e-07\n" + body
    out = []
    for txt in (c24, c25):
        f = tempfile.NamedTemporaryFile("w", suffix=".cap", delete=False)
        f.write(txt); f.close()
        out.append(parse_cap_auto(f.name)); os.unlink(f.name)
    a, b = out
    assert a["format"] == "ispd24" and b["format"] == "ispd25"
    assert b["ppa_weights"] == [-1.0, -10.0, 300.0, 7e-07]
    for la, lb in zip(a["layers"], b["layers"]):
        assert la["name"] == lb["name"] and (la["cap"] == lb["cap"]).all()
    assert a["layers"][0]["cap"][2, 1] == 6.0   # row y=1, col x=2
    print("  [PASS] parse_cap_auto detects ispd24 vs ispd25 header")
    print("  [PASS] ISPD'25 PPA weights skipped; capacities identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net", nargs="?", help="ISPD'25 .net file")
    ap.add_argument("--cap", help="summarize an ISPD'24/'25 .cap instead")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not (a.net or a.cap):
        self_test()
        return self_test_cap()
    import json
    if a.cap:
        c = parse_cap_auto(a.cap)
        print(json.dumps({"format": c["format"], "L": c["L"],
                          "grid": [c["X"], c["Y"]],
                          "ppa_weights": c["ppa_weights"],
                          "layers": [(l["name"], l["dir"],
                                      round(float((l["cap"] <= 0).mean()), 4))
                                     for l in c["layers"]]}, indent=2))
        return
    print(json.dumps(summarize(a.net), indent=2))


if __name__ == "__main__":
    main()
