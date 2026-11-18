#!/usr/bin/env python3
"""h_netpins.py: item 7 (preload). The overflow stage's pin table
    PRE["esc_pins"] = {escape3d._key(n["name"]): n["pins"] for n in escape3d.parse_net_ispd25(<design>.net)}
(pins = list of lists of (l, x, y) int tuples, nets without any access point keep pins == [] ... as the parser gives) built
from the C++ .net parse that the preload already does for build2 (build2_fast.parse_net_fast: names, pin_off, ap_off, aps).
  esc_pins_from_fast(E, net_fast) -> dict equal (==, same key order, tuple/int types) to the original
  python3 h_netpins.py --test        compares both on the six Mac cases
"""
import sys, time


def esc_pins_from_fast(E, net):
    import gc
    en = gc.isenabled(); gc.disable()                         # millions of small tuples: no cyclic-GC passes meanwhile
    try:
        return _conv(E, net)
    finally:
        if en:
            gc.enable()


def _conv(E, net):
    names, pin_off, ap_off, aps = net
    flat = list(map(tuple, aps.tolist()))                     # (l, x, y) Python ints, as int(a), int(b), int(c)
    po = pin_off.tolist(); ao = ap_off.tolist()
    key = E._key
    out = {}
    for i, nm in enumerate(names):
        out[key(nm)] = [flat[ao[p]:ao[p + 1]] for p in range(po[i], po[i + 1])]
    return out


def _test():
    import os
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    for p in ("tools/c1hfast", "tools/bsgfast", "tools/tilefix", "tools/speedcut", "tools/gapla", "tools/fast_io/route_cpp",
              "tools/fast_la", "", "tools/fast_path", "tools/onepass"):
        sys.path.insert(0, str(ROOT / p))
    import escape3d as E
    import build2_fast as B2
    bad = 0
    for s in ("visible", "blind"):
        for d in ("ariane", "mempool_tile_wrap", "NV_NVDLA_partition_c"):
            f = os.path.join(E.BENCH_ROOT, s, d, d + ".net")
            t0 = time.perf_counter(); ref = {E._key(n["name"]): n["pins"] for n in E.parse_net_ispd25(f)}; t1 = time.perf_counter()
            net = B2.parse_net_fast(os.path.join(B2.BENCH_ROOT, s, d, d + ".net")); t2 = time.perf_counter()
            new = esc_pins_from_fast(E, net); t3 = time.perf_counter()
            ok = ref == new and list(ref) == list(new)
            ok = ok and all(type(t) is tuple and all(type(v) is int for v in t) for pins in list(new.values())[:2000] for pin in pins for t in pin)
            bad += not ok
            print(f"{s} {d}: nets {len(ref)} equal={ok} parse_net_ispd25 {t1 - t0:.3f}s parse_net_fast {t2 - t1:.3f}s convert {t3 - t2:.3f}s", flush=True)
    return bad


if __name__ == "__main__":
    sys.exit(1 if _test() else 0)
