#!/usr/bin/env python3
"""build_fast2: C++-kernel problem builder. Same signature and output as
tools/fast_path/io_build.build_fast.  Kernels (tools/onepass/build2_kern.cc, ctypes):
cap tokenizer+strtod, .net parser, demand commit.  Build: clang++ -std=c++17 -O3 -fPIC -shared
tools/onepass/build2_kern.cc -o tools/onepass/libbuild2_kern.dylib   (.so on Linux)."""
import ctypes, os, sys, time
from pathlib import Path
import numpy as np
_HERE_PATH = Path(__file__).resolve().parent
_ROOT_PATH = next((p for p in _HERE_PATH.parents if (p / "tools" / "final" / "chain_y4.py").exists()), _HERE_PATH)
HERE = str(_HERE_PATH)
ROOT = str(_ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT, "tools", "fast_path"))
import io_build as B
from io_build import (read_crit_lines, read_crit, crit_inmem, assert_equal, build_fast as build_fast_io,  # noqa: F401
                      RE, SC, TimingLA, GRID, RC, net_weights, net_rc_params, _key, to_wires_vias,
                      used_access_points, nodes_from_aps, BENCH_ROOT)
from ispd25_adapter import parse_cap_auto  # noqa: F401  (reference)

_KERN_DIR = os.path.join(ROOT, "tools", "onepass")
_L = ctypes.CDLL(os.path.join(_KERN_DIR, "libbuild2_kern.dylib" if sys.platform == "darwin" else "libbuild2_kern.so"))
_c = ctypes
_L.tok_count.argtypes = [_c.c_char_p, _c.c_int64]; _L.tok_count.restype = _c.c_int64
_L.tok_fill.argtypes = [_c.c_char_p, _c.c_int64, _c.c_void_p, _c.c_void_p]
_L.parse_dbl.argtypes = [_c.c_char_p, _c.c_void_p, _c.c_int64, _c.c_void_p]
_L.net_parse.argtypes = [_c.c_char_p]; _L.net_parse.restype = _c.c_void_p
for _n in ("net_nnets", "net_npins", "net_naps", "net_names_len"):
    getattr(_L, _n).argtypes = [_c.c_void_p]; getattr(_L, _n).restype = _c.c_int64
_L.net_names.argtypes = [_c.c_void_p]; _L.net_names.restype = _c.c_void_p
for _n in ("net_pin_off", "net_ap_off", "net_aps"):
    getattr(_L, _n).argtypes = [_c.c_void_p]; getattr(_L, _n).restype = _c.c_void_p
_L.net_free.argtypes = [_c.c_void_p]
_L.commit_nets.argtypes = [_c.c_void_p] * 6 + [_c.c_void_p, _c.c_int64, _c.c_void_p, _c.c_int32,
                                               _c.c_void_p, _c.c_int, _c.c_int, _c.c_int, _c.c_void_p, _c.c_int]
_L.commit_nets.restype = _c.c_int


def _name_table_index(names):
    """Return a C++ table-backed name index; never materialize a name dict."""
    import load8_lib_f2 as _L8F2
    return _L8F2.table_index(names)


def _is_num(b):
    try:
        float(b); return True
    except ValueError:
        return False


def parse_cap_fast(path):
    """== ispd25_adapter.parse_cap_auto (same dict), tokenizer + strtod in C++."""
    with open(path, "rb") as fh:
        buf = fh.read()
    buf = buf + b"\0"
    n = len(buf) - 1
    nt = _L.tok_count(buf, n)
    st = np.empty(nt, np.int64); en = np.empty(nt, np.int64)
    _L.tok_fill(buf, n, st.ctypes.data, en.ctypes.data)
    tk = lambda i: buf[st[i]:en[i]]
    L, X, Y = int(tk(0)), int(tk(1)), int(tk(2))
    first_24 = 5 + L + (X - 1) + (Y - 1)
    if first_24 + 4 < nt and _is_num(tk(first_24)) and not _is_num(tk(first_24 + 4)):
        fmt, nw = "ispd25", 4
    elif not _is_num(tk(first_24)):
        fmt, nw = "ispd24", 0
    else:
        raise ValueError(f"{path}: unrecognized .cap header layout")

    def dbl(a, m):
        out = np.empty(m, np.float64)
        _L.parse_dbl(buf, st[a:a + m].ctypes.data, m, out.ctypes.data)
        return out
    short = dbl(5, L).tolist()
    weights = dbl(5 + L, nw).tolist() if nw else None
    i = 5 + L + nw
    xe = dbl(i, X - 1); i += X - 1
    ye = dbl(i, Y - 1); i += Y - 1
    layers = []
    for _ in range(L):
        name, d, ml = tk(i).decode(), int(tk(i + 1)), float(tk(i + 2)); i += 3
        layers.append({"name": name, "dir": d, "min_len": ml, "cap": dbl(i, X * Y).reshape(Y, X).T})
        i += X * Y
    if i != nt:
        raise ValueError(f"{path}: .cap parse incomplete ({fmt}), {nt - i} tokens left")
    return {"L": L, "X": X, "Y": Y, "unit_wire": float(tk(3)), "unit_via": float(tk(4)),
            "short_costs": short, "x_edge_len": xe, "y_edge_len": ye, "layers": layers,
            "format": fmt, "ppa_weights": weights}


def _np_from(ptr, n, ctype, dtype):
    if n == 0:
        return np.empty(0, dtype)
    return np.ctypeslib.as_array(_c.cast(ptr, _c.POINTER(ctype)), shape=(n,)).copy()


def parse_net_fast(path):
    """-> (names list, pin_off, ap_off, aps (naps,3)) of the ISPD'25 .net (== parse_net_ispd25)."""
    h = _L.net_parse(os.fsencode(path))
    if not h:
        raise IOError(path)
    try:
        nn, npn, nap = _L.net_nnets(h), _L.net_npins(h), _L.net_naps(h)
        nl = _L.net_names_len(h)
        blob = _c.string_at(_L.net_names(h), nl).decode()
        names = blob.split("\n")[:-1] if nn else []
        pin_off = _np_from(_L.net_pin_off(h), nn + 1, _c.c_int64, np.int64)
        ap_off = _np_from(_L.net_ap_off(h), npn + 1, _c.c_int64, np.int64)
        aps = _np_from(_L.net_aps(h), nap * 3, _c.c_int32, np.int32).reshape(nap, 3)
    finally:
        _L.net_free(h)
    return names, pin_off, ap_off, aps


def build_fast2(design, suite, route_path, crit_path, max_nets=6000, extra_power_nets=0,
                rt=None, crit_data=None, nets=None, timing=None):
    if nets is not None:
        return build_fast_io(design, suite, route_path, crit_path, max_nets, extra_power_nets,
                             rt=rt, crit_data=crit_data, nets=nets, timing=timing)
    t0 = time.time()
    tm = timing if timing is not None else {}
    def lap(k, t):
        tm[k] = tm.get(k, 0.0) + time.time() - t
        return time.time()
    t = time.time()
    case = SC.case_of(design, suite)
    cap = parse_cap_fast(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    t = lap("cap", t)
    pn_names, pin_off, ap_off, aps = parse_net_fast(os.path.join(BENCH_ROOT, suite, design, design + ".net"))
    pin_idx = _name_table_index(pn_names)                       # later duplicate wins, as the C++ table
    t = lap("pins", t)
    if rt is None:
        from route_cpp import read_route
        rt = read_route(route_path)
    off = np.ascontiguousarray(rt.offsets, dtype=np.int64)
    net_names_rt = rt.net_names
    sx0, sy0, sz0, sx1, sy1, sz1 = (np.ascontiguousarray(a, dtype=np.int32)
                                    for a in (rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1))
    gx0 = (sx0 // GRID).astype(np.int32); gy0 = (sy0 // GRID).astype(np.int32)
    gx1 = (sx1 // GRID).astype(np.int32); gy1 = (sy1 // GRID).astype(np.int32)
    G6 = np.stack([gx0, gy0, sz0, gx1, gy1, sz1], axis=1)
    t = lap("route", t)
    la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
    t = lap("commit", t)                      # commit happens after the run loop (order-free, integer sums)
    paths_all, netrc = crit_data if crit_data is not None else read_crit(crit_path)
    paths = [(s, [_key(n) for n in ns]) for s, ns in paths_all if s < 0]
    tw = net_weights(paths_all, case)
    t = lap("crit", t)
    idx = _name_table_index(net_names_rt)
    chosen = [n for n in sorted(tw, key=lambda n: -tw[n][0]) if n in idx and n in pin_idx][:max_nets]
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
    mask = np.ones(len(net_names_rt), np.uint8)
    for kname in chosen:
        i = idx[kname]
        name = net_names_rt[i]
        segs = list(map(tuple, G6[off[i]:off[i + 1]].tolist()))
        j = pin_idx[kname]
        pa = aps[ap_off[pin_off[j]]:ap_off[pin_off[j + 1]]].tolist()
        po = (ap_off[pin_off[j]:pin_off[j + 1] + 1] - ap_off[pin_off[j]]).tolist()
        pins_j = [[tuple(a) for a in pa[po[q]:po[q + 1]]] for q in range(len(po) - 1)]
        ch = used_access_points(segs, pins_j)
        nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins_j, ch)])
        polys = RE.segments_to_polys(segs, nodes)
        if not polys or not RE.tree_connects(nodes, polys):
            continue
        w_cur, v_cur = to_wires_vias(segs, la.dirs)
        n_id = len(net_names)
        cl, rd = net_rc_params(netrc.get(kname), 5.0, 2.0)
        net_cl.append(cl); net_rd.append(rd)
        net_names.append(name); net_route_idx.append(i); net_nodes.append(nodes)
        mask[i] = 0                                           # apply(-1) == leave out of the commit
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
    dirs32 = np.array(la.dirs, dtype=np.int32)
    dd = np.zeros(la.d.shape, np.float64)       # C-contiguous scratch; la.d starts at zero
    rc = _L.commit_nets(*(a.ctypes.data for a in (gx0, gy0, sz0, gx1, gy1, sz1)),
                        off.ctypes.data, len(net_names_rt), mask.ctypes.data, 1,
                        dirs32.ctypes.data, la.L, int(la.X), int(la.Y), dd.ctypes.data, 1)
    assert rc == 0, f"commit_nets rc={rc}"
    la.d = dd
    t = lap("commit", t)
    R = len(run_net)
    rdir = np.array(run_dir, dtype=np.int64)
    cand = np.full((R, K), -1, dtype=np.int64)
    Ha, Va = np.array(H, dtype=np.int64), np.array(V, dtype=np.int64)
    if R:
        mh = rdir == 0
        cand[mh, :len(H)] = Ha
        cand[~mh, :len(V)] = Va
    rlo, rhi = np.array(run_lo, dtype=np.int64), np.array(run_hi, dtype=np.int64)
    rfx = np.array(run_fixed, dtype=np.int64)
    rr, kk = np.nonzero(cand >= 0)
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
    order = np.argsort(first, kind="stable")
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


build_fast = build_fast2
