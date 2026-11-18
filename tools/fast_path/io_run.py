#!/usr/bin/env python3
"""Fast DP-v2 driver: route_cpp read once, vectorised build, in-memory crit, C++ block writer.
  io_run.py --design D --suite S --route R --crit C --out O --json J [--assert-build] [--timer]
--crit: existing STA report (read once, single pass).  --timer: run fast_sta in-process and hand the
report to the builder through memory (no crit file)."""
import argparse, collections, ctypes, json, os, sys, time
T0 = time.perf_counter()
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "fast_la"))
sys.path.insert(0, str(ROOT / "tools" / "fast_io" / "route_cpp"))
import io_build as B
import gapla_dp as V1
import gapla_dp_v2 as V2
import route_export as RE
import timing_promote as TP
from route_cpp import read_route

_W = ctypes.CDLL(str(HERE / "libio_write.dylib" if sys.platform == "darwin" else HERE / "libio_write.so"))
_I32 = ctypes.POINTER(ctypes.c_int32)
_W.blocks_write.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)] + [_I32] * 6
_W.blocks_write.restype = ctypes.c_int


def write_fast(P, choice, rt, out_path):
    """Same bytes as gapla_dp.write_route, from the already-read Route (no second read)."""
    t0 = time.perf_counter()
    C = RE.GCELL_DBU; h = C // 2
    rn = V1._np(P["run_net"], np.int64); cand = V1._np(P["cand"], np.int64)
    dirs = list(P["dirs"])
    by_net = collections.defaultdict(list)
    for r, n in enumerate(rn.tolist()):
        by_net[int(n)].append(r)
    new = {}
    for n, runs in by_net.items():
        wires = collections.defaultdict(list)
        node_layers = collections.defaultdict(set)
        for r in runs:
            z = int(cand[r, int(choice[r])])
            if z < 0:
                continue
            d = int(P["run_dir"][r]); f = int(P["run_fixed"][r])
            lo = int(P["run_lo"][r]); hi = int(P["run_hi"][r])
            wires[(z, f)].append((lo, hi))
            a = (lo, f) if d == 0 else (f, lo)
            b = (hi, f) if d == 0 else (f, hi)
            node_layers[a].add(z); node_layers[b].add(z)
        for xy, span in P["net_nodes"][n].items():
            node_layers[xy].update((int(span[0]), int(span[1])))
        buf = []
        for (z, f), ivs in wires.items():
            for lo, hi in ivs:
                buf.append((lo, f, z, hi, f, z) if dirs[z] == 0 else (f, lo, z, f, hi, z))
        for (x, y), zs in node_layers.items():
            if zs:
                for z in range(min(zs), max(zs)):
                    buf.append((x, y, z, x, y, z + 1))
        new[int(P["net_route_idx"][n])] = buf
    # assemble: columns x0 y0 z0 x1 y1 z1 (DBU, zero-based metal)
    off = rt.offsets.astype(np.int64)
    nn = len(rt.net_names)
    cnt = np.diff(off).copy()
    for i, b in new.items():
        cnt[i] = len(b)
    noff = np.zeros(nn + 1, dtype=np.uint64); noff[1:] = np.cumsum(cnt)
    tot = int(noff[-1])
    cols = [np.empty(tot, np.int32) for _ in range(6)]
    src = [rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1]
    keep = np.ones(nn, bool); keep[list(new)] = False
    # unchanged nets: vectorised copy by destination/source index maps
    kn = np.flatnonzero(keep & (cnt > 0))
    if len(kn):
        c = cnt[kn]
        s0 = np.repeat(off[kn], c); d0 = np.repeat(noff[kn].astype(np.int64), c)
        ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        for k in range(6):
            cols[k][d0 + ar] = src[k][s0 + ar]
    for i, b in new.items():
        if b:
            a = np.array(b, dtype=np.int64); o = int(noff[i]); m = len(b)
            cols[0][o:o+m] = a[:, 0] * C + h; cols[1][o:o+m] = a[:, 1] * C + h; cols[2][o:o+m] = a[:, 2]
            cols[3][o:o+m] = a[:, 3] * C + h; cols[4][o:o+m] = a[:, 4] * C + h; cols[5][o:o+m] = a[:, 5]
    blob = ("\n".join(rt.net_names) + "\n").encode()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rc = _W.blocks_write(os.fsencode(str(out_path)), blob, nn, noff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                         *(c.ctypes.data_as(_I32) for c in cols))
    assert rc == 0
    return {"write_total_s": time.perf_counter() - t0, "nets_changed": len(new)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True); ap.add_argument("--suite", default="visible")
    ap.add_argument("--route", required=True); ap.add_argument("--crit"); ap.add_argument("--out", required=True)
    ap.add_argument("--json"); ap.add_argument("--max-nets", type=int, default=6000)
    ap.add_argument("--npaths", type=int, default=5000)
    ap.add_argument("--assert-build", action="store_true")
    ap.add_argument("--timer", action="store_true")
    a = ap.parse_args(argv)
    dr = V1.DEFAULT_DATA_ROOT
    route = V1._path(a.route, dr)
    walls = {"import_s": time.perf_counter() - T0}
    t = time.perf_counter()
    rt = read_route(str(route))
    walls["route_read_s"] = time.perf_counter() - t
    t = time.perf_counter()
    timer_rec = None
    if a.timer:
        import importlib
        bench = dr / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
        FS = importlib.import_module("fast_sta")
        D = FS.StaDesign(a.design, a.suite, kit=str(bench))
        seg = FS.read_route(D, str(route))
        r = FS.StaTimer(D).analyze(seg, rat=True, keep=True)
        V2._fallback_torch_scatter()
        walls["timer_s"] = time.perf_counter() - t
        t = time.perf_counter()
        lines = B.crit_inmem(D, r, a.npaths)
        crit_data = B.read_crit_lines(lines)
        timer_rec = {"design": D, "result": r}
        walls["crit_handoff_s"] = time.perf_counter() - t
    else:
        crit_data = B.read_crit(str(V1._path(a.crit, dr)))
        walls["crit_read_s"] = time.perf_counter() - t
    t = time.perf_counter(); tm = {}
    P = B.build_fast(a.design, a.suite, str(route), None, a.max_nets, rt=rt, crit_data=crit_data, timing=tm)
    walls["build_s"] = time.perf_counter() - t; walls["build_parts"] = {k: round(v, 3) for k, v in tm.items()}
    if a.assert_build:
        Po = V1.build_problem_from_route(str(route), a.crit, a.design, a.suite, a.max_nets, dr)
        for k in ("route_path", "design", "suite"):
            Po.pop(k, None); P.pop(k, None)
        B.assert_equal(Po, P); walls["assert_build"] = "EQUAL"
        P.update(design=a.design, suite=a.suite, route_path=str(route))
    else:
        P.update(route_path=str(route))
    t = time.perf_counter()
    P["route_path"] = str(route); P["design"] = a.design; P["suite"] = a.suite
    orig = TP.parse_net_rc
    TP.parse_net_rc = lambda _p, _nr=crit_data[1]: _nr
    try:
        V2.add_real_tech_rc(P, timer_rec, dr, Path("<mem>"))
    finally:
        TP.parse_net_rc = orig
    choice, dp = V2.assign_v2(P, "numpy", 1.0, None, V1.DEFAULT_VIA_COST, 0.0)
    walls["dp_s"] = time.perf_counter() - t
    t = time.perf_counter()
    ws = write_fast(P, choice, rt, a.out)
    walls["write_s"] = time.perf_counter() - t
    walls["total_s"] = time.perf_counter() - T0
    rec = {"walls": walls, "dp_runs": dp["runs"], "write": ws, "out": str(a.out), "load": os.getloadavg()[0]}
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))

if __name__ == "__main__":
    main()
