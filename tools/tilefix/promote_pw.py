"""promote_cpp.py (NEW, 2026-11-05): State3.promote with the layer-DP hot loop (LayerAssigner.assign_path + TimingLA costs) in C++.
Same control flow and arithmetic order as fastmode3.State3.promote; route must be md5-identical to the Python promote."""
import ctypes, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "libpromote_pw_kern." + ("dylib" if sys.platform == "darwin" else "so")))
_P = ctypes.c_void_p
_lib.assign_batch.argtypes = [_P] * 10 + [ctypes.c_int] * 3 + [ctypes.c_double] * 5 + [_P] * 3 + [ctypes.c_int, _P, _P, ctypes.c_double]
_lib.assign_batch.restype = None
_lib.np_sum.argtypes = [_P, ctypes.c_long]; _lib.np_sum.restype = ctypes.c_double


class Batch:
    """per-net: demand is constant during all cost evaluations, so one C call per candidate set."""
    def __init__(self, la):
        self.la = la
        assert la.d.dtype == np.float64 and la.c.dtype == np.float64
        self.cc = la.c
        import math
        self.cst = np.array([math.exp(0.5) - 1.0, math.exp(0.25) - 1.0, math.exp(1.5)])
        self.sd = np.array(la.d.strides, np.int64) // 8; self.sc = np.array(la.c.strides, np.int64) // 8
        self.dirs = np.ascontiguousarray(la.dirs, dtype=np.int32)
        self.u = np.ascontiguousarray(la.u, dtype=np.float64)
        self.R = np.ascontiguousarray(la.R, dtype=np.float64); self.C = np.ascontiguousarray(la.C, dtype=np.float64)
        self.Rv = np.ascontiguousarray(la.Rv, dtype=np.float64)

    def run(self, polys_runs, spans):
        """polys_runs: list (per poly) of run tuples; spans: list of (span_start, span_end) each None or (lo,hi). -> (layers per poly, cost per poly)"""
        la = self.la
        P = len(polys_runs)
        poff = np.zeros(P + 1, np.int32)
        for i, r in enumerate(polys_runs):
            poff[i + 1] = poff[i] + len(r)
        flat = [t for r in polys_runs for t in r]
        runs = np.array(flat, dtype=np.int32).reshape(-1, 8) if flat else np.zeros((0, 8), np.int32)
        sp = np.full((P, 4), -1, np.int32)
        for i, (s, e) in enumerate(spans):
            if s is not None: sp[i, 0], sp[i, 1] = s
            if e is not None: sp[i, 2], sp[i, 3] = e
        layers = np.zeros(max(len(flat), 1), np.int32); cost = np.zeros(P, np.float64)
        _lib.assign_batch(la.d.ctypes.data, self.cc.ctypes.data, self.u.ctypes.data, self.R.ctypes.data, self.C.ctypes.data,
                          self.Rv.ctypes.data, self.cst.ctypes.data, self.dirs.ctypes.data, self.sd.ctypes.data, self.sc.ctypes.data, la.L, la.X, la.Y, la.w4, la.um, la.tw, la.c_load, la.r_drv,
                          runs.ctypes.data, poff.ctypes.data, sp.ctypes.data, P, layers.ctypes.data, cost.ctypes.data, float(getattr(la, 'pn', 0.0)))
        out = [layers[poff[i]:poff[i + 1]].tolist() for i in range(P)], cost.tolist()
        if os.environ.get("PROMOTE_CHECK"):
            for i in range(P):
                if not polys_runs[i]: continue
                ly, cs = la.assign_path(polys_runs[i], spans[i][0], spans[i][1], 1.0)
                if ly != out[0][i] or cs != out[1][i]:
                    self.bad = getattr(self, "bad", 0) + 1
                    if self.bad < 6: print("MISMATCH", polys_runs[i], spans[i], ly, out[0][i], repr(cs), repr(out[1][i]), flush=True)
        return out


def promote_cpp(S, r, max_nets, tw_scale, fl, w, pwn=None, order=None):
    import route_export as RE
    import fastmode as FM
    from timing_promote import apply, to_wires_vias
    from la_swap import used_access_points, nodes_from_aps
    SD, la = S.SD, S.la
    B = Batch(la)
    if order is None:
        cand = np.nonzero(w > 0)[0]
        order = cand[np.argsort(-w[cand], kind="stable")][:max_nets]
    la.pn = 0.0
    cl = np.clip(0.5 * r["net_ctot"], 1.0, 1e5)
    rd = r["net_rd_kohm"]
    undo = {}
    stats = defaultdict(int)
    for n in order:
        kname = SD.net_names[n]
        i = S.ridx.get(kname)
        pins = S.pins_of(kname)
        if i is None or not pins:
            stats["not_in_route"] += 1
            continue
        segs = S.new.get(i, S.segs[i])
        wv, vv = to_wires_vias(segs, la.dirs)
        apply(la, wv, vv, -1)
        chosen = used_access_points(segs, pins)
        nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins, chosen)])
        la.tw = float(w[n]) * tw_scale
        la.pn = float(pwn[n]) if pwn is not None else 0.0
        la.c_load, la.r_drv = float(cl[n]), float(rd[n])
        cands = []
        p = RE.segments_to_polys(segs, nodes)
        if RE.tree_connects(nodes, p):
            cands.append(("base", p))
        if fl is not None and len(nodes) > 1:
            # rsmt_polys: per FLUTE edge the cheaper of two L shapes (strict <, first wins ties)
            edges = fl.edges(list(nodes.keys()))
            pl, sp = [], []
            for (px, py), (cx, cy) in edges:
                for mid in ((cx, py), (px, cy)):
                    pl.append([(px, py), mid, (cx, cy)]); sp.append((nodes.get((px, py)), nodes.get((cx, cy))))
            rp = []
            if pl:
                _, costs = B.run([RE.to_runs(q) for q in pl], sp)
                for e in range(len(edges)):
                    bst = None
                    for j in (0, 1):
                        if bst is None or costs[2 * e + j] < bst[0]:
                            bst = (costs[2 * e + j], pl[2 * e + j])
                    rp.append(bst[1])
            cands.append(("rsmt", rp))
        if not cands:
            apply(la, wv, vv, +1)
            stats["kept"] += 1
            continue
        best = None
        for tag, polys in cands:
            runs_l, ids = [], []
            for pts in polys:
                runs = RE.to_runs(pts)
                if runs:
                    runs_l.append(runs); ids.append(pts)
            lay, costs = B.run(runs_l, [(nodes.get(q[0]), nodes.get(q[-1])) for q in ids]) if runs_l else ([], [])
            cost = 0.0
            for c_ in costs:
                cost += c_
            if best is None or cost < best[0]:
                best = (cost, tag, polys, runs_l, lay, ids)
        # route_net with the layers already computed (demand unchanged since the evaluation)
        wires_raw = defaultdict(list)
        node_layers = defaultdict(set)
        for runs, layers in zip(best[3], best[4]):
            for rr, z in zip(runs, layers):
                wires_raw[(z, rr[1])].append((rr[2], rr[3]))
                node_layers[(rr[4], rr[5])].add(z)
                node_layers[(rr[6], rr[7])].add(z)
        for xy, (lo, hi) in nodes.items():
            node_layers[xy].update((lo, hi))
        vias = set()
        for (x, y), zs in node_layers.items():
            for z in range(min(zs), max(zs)):
                vias.add((x, y, z))
        if not wires_raw and not vias and nodes:
            (x, y), (lo, _) = next(iter(nodes.items()))
            vias.add((x, y, lo if lo < la.L - 1 else lo - 1))
        wires = {}
        for key, ivs in wires_raw.items():
            ivs.sort()
            merged = [list(ivs[0])]
            for a, b in ivs[1:]:
                if a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            wires[key] = [tuple(m) for m in merged]
        la.commit_net(wires, vias)
        undo.setdefault(i, segs)
        S.new[i] = FM.segs_of(wires, vias, la.dirs)
        S._dirty.add(int(S.tid[i]))
        stats["topo_" + best[1]] += 1
    la.tw = 0.0; la.pn = 0.0
    return undo, dict(stats)
