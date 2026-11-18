"""promote_fast.py: tools/c1hfast/promote_cpp.promote_cpp (critical rounds) and
tools/tilefix/promote_pw.promote_cpp (power round) with the kernel-call overhead removed: the 10 constant pointers
(demand, capacity, costs, strides, ...) are computed once per round, and the 5 per-call arrays live in reusable
buffers whose pointers are also computed once (re-taken only when a buffer has to grow). The per-net control flow,
the kernels and every argument value are unchanged (same libraries, same inputs), so the route is identical.
promote_crit(S, r, max_nets, tw_scale, fl, w)                   == promote_cpp.promote_cpp
promote_pw(S, r, max_nets, tw_scale, fl, w, pwn=None, order=None) == promote_pw.promote_cpp
"""
import ctypes, math, os, sys
from collections import defaultdict
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXT = "dylib" if sys.platform == "darwin" else "so"
_P = ctypes.c_void_p
_ARGS = [_P] * 10 + [ctypes.c_int] * 3 + [ctypes.c_double] * 5 + [_P] * 3 + [ctypes.c_int, _P, _P]
_libc = ctypes.CDLL(os.path.join(_ROOT, "tools", "c1hfast", "libpromote_kern.dylib"))
_libc.assign_batch.argtypes = _ARGS; _libc.assign_batch.restype = None
_libp = ctypes.CDLL(os.path.join(_ROOT, "tools", "tilefix", "libpromote_pw_kern." + _EXT))
_libp.assign_batch.argtypes = _ARGS + [ctypes.c_double]; _libp.assign_batch.restype = None
_CHECK = bool(os.environ.get("PROMOTE_CHECK"))
_PTRCHECK = bool(os.environ.get("FAST3X_PTRCHECK"))


def _ptr(a):
    return a.ctypes.data


class Batch:
    """per-net: demand is constant during all cost evaluations, so one C call per candidate set."""
    def __init__(self, la, pw):
        self.la, self.pw = la, pw
        assert la.d.dtype == np.float64 and la.c.dtype == np.float64
        self.fn = (_libp if pw else _libc).assign_batch
        self.cst = np.array([math.exp(0.5) - 1.0, math.exp(0.25) - 1.0, math.exp(1.5)])
        self.sd = np.array(la.d.strides, np.int64) // 8; self.sc = np.array(la.c.strides, np.int64) // 8
        self.dirs = np.ascontiguousarray(la.dirs, dtype=np.int32)
        self.u = np.ascontiguousarray(la.u, dtype=np.float64)
        self.R = np.ascontiguousarray(la.R, dtype=np.float64); self.C = np.ascontiguousarray(la.C, dtype=np.float64)
        self.Rv = np.ascontiguousarray(la.Rv, dtype=np.float64)
        self.dptr = _ptr(la.d)
        self.const = (self.dptr, _ptr(la.c), _ptr(self.u), _ptr(self.R), _ptr(self.C), _ptr(self.Rv), _ptr(self.cst),
                      _ptr(self.dirs), _ptr(self.sd), _ptr(self.sc), la.L, la.X, la.Y, la.w4, la.um)
        self._grow_r(4096); self._grow_p(1024)

    def _grow_r(self, n):
        self.cap_r = n
        self.runs = np.zeros((n, 8), np.int32); self.layers = np.zeros(n, np.int32)
        self.p_runs, self.p_layers = _ptr(self.runs), _ptr(self.layers)

    def _grow_p(self, n):
        self.cap_p = n
        self.poff = np.zeros(n + 1, np.int32); self.sp = np.full((n, 4), -1, np.int32); self.cost = np.zeros(n, np.float64)
        self.p_poff, self.p_sp, self.p_cost = _ptr(self.poff), _ptr(self.sp), _ptr(self.cost)

    def run(self, polys_runs, spans):
        """== promote_cpp.Batch.run: -> (layers per poly, cost per poly)"""
        la = self.la
        P = len(polys_runs)
        poff = [0] * (P + 1)
        flat = []
        for i, r in enumerate(polys_runs):
            poff[i + 1] = poff[i] + len(r)
            flat.extend(r)
        n = len(flat)
        if n > self.cap_r:
            self._grow_r(max(n, 2 * self.cap_r))
        if P > self.cap_p:
            self._grow_p(max(P, 2 * self.cap_p))
        if n:
            self.runs[:n] = flat
        self.poff[:P + 1] = poff
        spl = []
        for s, e in spans:                                   # == sp[i] = -1, then (s, e) where given
            if s is None: s = (-1, -1)
            if e is None: e = (-1, -1)
            spl.append((s[0], s[1], e[0], e[1]))
        if P:
            self.sp[:P] = spl
        if _PTRCHECK:
            assert _ptr(la.d) == self.dptr
        if self.pw:
            self.fn(*self.const, la.tw, la.c_load, la.r_drv, self.p_runs, self.p_poff, self.p_sp, P, self.p_layers, self.p_cost,
                    float(getattr(la, 'pn', 0.0)))
        else:
            self.fn(*self.const, la.tw, la.c_load, la.r_drv, self.p_runs, self.p_poff, self.p_sp, P, self.p_layers, self.p_cost)
        lay = self.layers[:n].tolist()
        out = [lay[poff[i]:poff[i + 1]] for i in range(P)], self.cost[:P].tolist()
        if _CHECK:
            for i in range(P):
                if not polys_runs[i]: continue
                ly, cs = la.assign_path(polys_runs[i], spans[i][0], spans[i][1], 1.0)
                if ly != out[0][i] or cs != out[1][i]:
                    self.bad = getattr(self, "bad", 0) + 1
                    if self.bad < 6: print("MISMATCH", polys_runs[i], spans[i], ly, out[0][i], repr(cs), repr(out[1][i]), flush=True)
        return out


def _promote(S, r, max_nets, tw_scale, fl, w, pwn, order, pw):
    import route_export as RE
    import fastmode as FM
    from timing_promote import apply, to_wires_vias
    from la_swap import used_access_points, nodes_from_aps
    SD, la = S.SD, S.la
    B = Batch(la, pw)
    if order is None:
        cand = np.nonzero(w > 0)[0]
        order = cand[np.argsort(-w[cand], kind="stable")][:max_nets]
    if pw:
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
        if pw:
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
    la.tw = 0.0
    if pw:
        la.pn = 0.0
    return undo, dict(stats)


def promote_crit(S, r, max_nets, tw_scale, fl, w):
    return _promote(S, r, max_nets, tw_scale, fl, w, None, None, False)


def promote_pw(S, r, max_nets, tw_scale, fl, w, pwn=None, order=None):
    return _promote(S, r, max_nets, tw_scale, fl, w, pwn, order, True)
