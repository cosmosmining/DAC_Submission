#!/usr/bin/env python3
"""Fast DP-v2 problem builder.  Same output dict as dtla.build, vectorised.

API
  read_crit_lines(lines) / read_crit(path) -> (paths, netrc)    single pass, replaces parse_crit + parse_net_rc
  crit_inmem(D, r, npaths) -> list[str]   STA result -> crit report lines without touching disk
  build_fast(design, suite, route, crit, max_nets=6000, rt=None, crit_data=None) -> dict (== dtla.build)
      route : path or a route_cpp.Route (read once per process); crit_data=(paths, netrc) skips the file.
  assert_equal(P_old, P_new)   exact equality of every entry (arrays: dtype+values, lists: ==)
"""
import io, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fast_io", "route_cpp"))
import route_export as RE
import ispd25_score as SC
import dtla
from dtla import build as build_old  # noqa: F401  (reference)
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from la_swap import used_access_points, nodes_from_aps
import timing_promote as TP
from timing_promote import (TimingLA, apply, to_wires_vias, net_weights, _key, GRID, RC,
                            net_rc_params, _NET, _SLACK, _DRV)

BENCH_ROOT = TP.BENCH_ROOT


def read_crit_lines(lines):
    """One pass over the report; identical to (parse_crit(path), parse_net_rc(path))."""
    paths, nets = [], []
    rc, last = {}, None
    for line in lines:
        if line.startswith("Startpoint:"):
            nets = []
            continue
        if "(net)" in line:
            m = _NET.search(line)
            if m:
                nets.append(m.group(1))
                if last is not None:                       # parse_net_rc's net-line branch
                    k = _key(m.group(1))
                    if k not in rc or last[1] > rc[k][1]:
                        rc[k] = last
                    last = None
                continue
        elif "slack (" in line:
            m = _SLACK.match(line)
            if m:
                paths.append((float(m.group(1)), nets))
                nets = []
        # driver-line detector (parse_net_rc tries it first on every line)
        if "^" in line or "v" in line:
            m = _DRV.match(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    return paths, rc


def read_crit(path):
    with open(path) as fh:
        return read_crit_lines(fh)


def crit_inmem(D, r, npaths, slack_max=0.0):
    """fastmode4.write_crit into memory (its open() is redirected to a StringIO)."""
    import fastmode4 as FM4
    sio = io.StringIO()
    sio.close = lambda: None
    FM4.open = lambda p, m="r": sio
    try:
        FM4.write_crit(D, r, "<mem>", npaths=npaths, slack_max=slack_max)
    finally:
        del FM4.open
    return sio.getvalue().splitlines(True)


def route_to_nets(rt):
    """route_cpp.Route -> [(name, [(x,y,z,x,y,z) GCell ints], None)] == read_route25 (lines unused)."""
    g = GRID
    x0 = (rt.x0 // g).tolist(); y0 = (rt.y0 // g).tolist(); z0 = rt.z0.tolist()
    x1 = (rt.x1 // g).tolist(); y1 = (rt.y1 // g).tolist(); z1 = rt.z1.tolist()
    seg = list(zip(x0, y0, z0, x1, y1, z1))
    off = rt.offsets.tolist()
    return [(rt.net_names[i], seg[off[i]:off[i + 1]], None) for i in range(len(rt.net_names))]


def build_fast(design, suite, route_path, crit_path, max_nets=6000, extra_power_nets=0,
               rt=None, crit_data=None, nets=None, timing=None):
    t0 = time.time()
    tm = timing if timing is not None else {}
    def lap(k, t):
        tm[k] = tm.get(k, 0.0) + time.time() - t
        return time.time()
    t = time.time()
    case = SC.case_of(design, suite)
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    t = lap("cap", t)
    pins = {_key(n["name"]): n["pins"]
            for n in parse_net_ispd25(os.path.join(BENCH_ROOT, suite, design, design + ".net"))}
    t = lap("pins", t)
    if nets is None:
        if rt is None:
            from route_cpp import read_route
            rt = read_route(route_path)
        nets = route_to_nets(rt)
    t = lap("route", t)
    la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)
    t = lap("commit", t)
    paths_all, netrc = crit_data if crit_data is not None else read_crit(crit_path)
    paths = [(s, [_key(n) for n in ns]) for s, ns in paths_all if s < 0]
    tw = net_weights(paths_all, case)
    t = lap("crit", t)
    idx = {_key(n): i for i, (n, _, _) in enumerate(nets)}
    chosen = [n for n in sorted(tw, key=lambda n: -tw[n][0]) if n in idx and n in pins][:max_nets]
    if extra_power_nets:
        raise NotImplementedError("extra_power_nets not used by DP v2")
    H = [z for z in range(1, la.L) if la.dirs[z] == 0]
    V = [z for z in range(1, la.L) if la.dirs[z] == 1]
    K = max(len(H), len(V))
    run_net, run_dir, run_fixed, run_lo, run_hi, run_len = [], [], [], [], [], []
    run_cur, run_p0, run_p1 = [], [], []
    net_cl, net_rd = [], []
    pt_id, pt_xy, pt_lo, pt_hi = {}, [], [], []
    net_names, net_route_idx, net_nodes = [], [], []
    for kname in chosen:
        i = idx[kname]
        name, segs, _ = nets[i]
        ch = used_access_points(segs, pins[kname])
        nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins[kname], ch)])
        polys = RE.segments_to_polys(segs, nodes)
        if not polys or not RE.tree_connects(nodes, polys):
            continue
        w_cur, v_cur = to_wires_vias(segs, la.dirs)
        n_id = len(net_names)
        cl, rd = net_rc_params(netrc.get(kname), 5.0, 2.0)
        net_cl.append(cl); net_rd.append(rd)
        net_names.append(name); net_route_idx.append(i); net_nodes.append(nodes)
        apply(la, w_cur, v_cur, -1)
        for pts in polys:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            d = 0 if y0 == y1 else 1
            fixed = y0 if d == 0 else x0
            lo, hi = (min(x0, x1), max(x0, x1)) if d == 0 else (min(y0, y1), max(y0, y1))
            cands = H if d == 0 else V
            cur = 0
            for k, z in enumerate(cands):
                if any(a <= lo and hi <= b for a, b in w_cur.get((z, fixed), ())):
                    cur = k
                    break
            ids = []
            for xy in ((x0, y0), (x1, y1)):
                if xy not in pt_id:
                    pt_id[xy] = len(pt_xy)
                    pt_xy.append(xy)
                    sp = nodes.get(xy)
                    pt_lo.append(sp[0] if sp else la.L)
                    pt_hi.append(sp[1] if sp else -1)
                ids.append(pt_id[xy])
            run_net.append(n_id); run_dir.append(d); run_fixed.append(fixed)
            run_lo.append(lo); run_hi.append(hi); run_len.append(hi - lo)
            run_cur.append(cur); run_p0.append(ids[0]); run_p1.append(ids[1])
        pt_id = {}
    t = lap("runs", t)
    R = len(run_net)
    rdir = np.array(run_dir, dtype=np.int64)
    # candidate table
    cand = np.full((R, K), -1, dtype=np.int64)
    Ha, Va = np.array(H, dtype=np.int64), np.array(V, dtype=np.int64)
    if R:
        mh = rdir == 0
        cand[mh, :len(H)] = Ha
        cand[~mh, :len(V)] = Va
    # touched cells, vectorised: entry order = (run, k, t) exactly as the python loops
    rlo, rhi = np.array(run_lo, dtype=np.int64), np.array(run_hi, dtype=np.int64)
    rfx = np.array(run_fixed, dtype=np.int64)
    rr, kk = np.nonzero(cand >= 0)                       # row-major == (r, k) loop order
    n_t = np.maximum(rhi[rr] - rlo[rr], 0)
    rep = np.repeat(np.arange(len(rr)), n_t)
    starts = np.cumsum(n_t) - n_t
    tt = (np.arange(len(rep)) - np.repeat(starts, n_t)) + rlo[rr][rep]
    rE, kE = rr[rep], kk[rep]
    zE = cand[rE, kE]
    horiz = rdir[rE] == 0
    xE = np.where(horiz, tt, rfx[rE]); yE = np.where(horiz, rfx[rE], tt)
    Xn, Yn = int(la.X), int(la.Y)
    flat = (zE * Xn + xE) * Yn + yE
    uq, first, inv = np.unique(flat, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")             # cell ids by first appearance
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    ent_cell = rank[inv.reshape(-1)].astype(np.int64)
    ent_rk = (rE * K + kE).astype(np.int64)
    sel = first[order]
    cz, cx, cy = zE[sel].astype(np.int64), xE[sel].astype(np.int64), yE[sel].astype(np.int64)
    d0 = la.d[cz, cx, cy].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    capv = la.c[cz, cx, cy].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    u = la.u[cz].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    t = lap("cells", t)
    nidx = {_key(n): j for j, n in enumerate(net_names)}
    p_net, p_path, slack0 = [], [], []
    for pi, (s, ns) in enumerate(paths):
        slack0.append(s)
        for n in set(ns):
            j = nidx.get(n)
            if j is not None:
                p_net.append(j); p_path.append(pi)
    names = [l["name"] for l in cap["layers"]]
    Cz = np.array([RC[n][1] for n in names], dtype=np.float32)
    Rz = np.array([RC[n][0] for n in names], dtype=np.float64)
    um = GRID / 2000.0
    dcoef = np.zeros(cand.shape, np.float32)
    rn = np.array(run_net, dtype=np.int64)
    lum = np.array(run_len, np.float64) * um
    cl_r, rd_r = np.array(net_cl)[rn], np.array(net_rd)[rn]
    for k in range(K):
        z = np.clip(cand[:, k], 0, la.L - 1)
        dk = lum * (Rz[z] * cl_r + Cz[z] * rd_r) / 1000.0 + Rz[z] * Cz[z] * lum * lum / 2.0 / 1000.0
        dcoef[:, k] = np.where(cand[:, k] >= 0, dk, 0.0)
    lap("tail", t)
    tm["build_total"] = time.time() - t0
    return {
        "case": case, "design": design, "suite": suite, "L": la.L, "K": K,
        "run_net": np.array(run_net), "run_len": np.array(run_len, np.float32),
        "run_cur": np.array(run_cur), "run_p0": np.array(run_p0), "run_p1": np.array(run_p1),
        "run_dir": np.array(run_dir), "run_fixed": np.array(run_fixed),
        "run_lo": np.array(run_lo), "run_hi": np.array(run_hi),
        "cand": cand, "ent_rk": ent_rk, "ent_cell": ent_cell,
        "d0": d0, "cap": capv, "u": u, "cell_z": cz,
        "pt_lo": np.array(pt_lo), "pt_hi": np.array(pt_hi),
        "p_net": np.array(p_net), "p_path": np.array(p_path),
        "slack0": np.array(slack0, np.float32), "Cz": Cz, "um": GRID / 2000.0, "dcoef": dcoef,
        "net_names": net_names, "net_route_idx": net_route_idx, "net_nodes": net_nodes,
        "pt_xy": pt_xy, "dirs": la.dirs, "route_path": route_path,
    }


def assert_equal(A, B):
    assert A.keys() == B.keys(), (set(A) ^ set(B))
    for k in A:
        a, b = A[k], B[k]
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape \
                and np.array_equal(a, b), f"mismatch {k}: {a.dtype} {a.shape} vs {b.dtype} {b.shape}"
        else:
            assert a == b if not (hasattr(a, "__len__") and hasattr(b, "__len__") and
                                  isinstance(a, np.ndarray)) else True, f"mismatch {k}"
    return True
