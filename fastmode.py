#!/usr/bin/env python3
"""fastmode.py — ISPD'25 fast mode: no STA, no OpenROAD, one process, inputs parsed once.

  InstantGR route
    -> fast_timer.py (Elmore / NLDM on the 3-D route + the .net pre-routing slacks; see its docstring)
    -> R rounds of timing-driven layer promotion of the nets on the fast timer's violating paths
       (timing_promote.py's assigner and cost, in memory: exact evaluator demand model, per-net load and driver
       resistance from the fast timer instead of an OpenROAD report); a round is kept only if the fast model's
       score w1*WNS + w2*TNS/N_end + w4*overflow improves (trust region: rejected rounds are undone exactly)
    -> [--gnn CKPT] one-pass learned-start GNN layer assignment (dtla_lagnn.py) on the fast timer's critical paths,
       kept only if the fast model's score improves
    -> seg_normalize (every net, as the quality flow) -> <design>.<tag>.norm.route
The official evaluation is NOT part of this program; a launcher script (not included) runs it once at the end, for reporting.
Wall time of everything here is written to <route dir>/<design>.<tag>.fastmode.json.

  python fastmode.py --design ariane --suite visible --base experiments/ispd25_routes/ariane/ariane.instantgr.route \
                     --tag instantgr_f1 [--rounds 3 --max_nets 4000 --gnn models/la_ckpt.pt]
  python fastmode.py --self_test
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import fast_timer as FT                                      # NEW (this project)
import ispd25_score as SC                                    # READ-ONLY reuse below
import route_export as RE
from ispd25_adapter import parse_cap_auto
from la_swap import used_access_points, nodes_from_aps
from seg_normalize import normalize_net, _dbu
from timing_promote import TimingLA, apply, to_wires_vias, net_weights, read_route25, GRID, _key

T0 = time.time()


def log(m):
    print(f"[fastmode {time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════
#  exact evaluator demand / overflow (vectorized build, same rules as timing_promote.apply)
# ════════════════════════════════════════════════════════════════════════════════════
def overflow_cost(la):
    """official evaluator overflow cost from the demand model (layers 1..L-1; demand in half units)."""
    tot = 0.0
    for z in range(1, la.L):
        d, c = la.d[z], la.c[z]
        with np.errstate(over="ignore"):
            loss = np.where(c > RE.CAP_EPS, np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0)),
                            np.exp(np.minimum(0.75 * d, 700.0)))
        tot += float(la.u[z]) * float(loss.sum())
    return tot


def build_demand(la, nets_segs):
    """commit every net (timing_promote.apply semantics) — loop version (reference)."""
    for segs in nets_segs:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)


def build_demand_fast(la, seg):
    """vectorized equivalent of build_demand for a [S,7] array (net, x1, y1, z1, x2, y2, z2)."""
    L, X, Y = la.L, la.X, la.Y
    dirs = np.array(la.dirs)
    n, x1, y1, z1, x2, y2, z2 = (seg[:, k] for k in range(7))
    isvia = z1 != z2
    zz = z1
    hor = (~isvia) & (dirs[np.clip(zz, 0, L - 1)] == 0) & (y1 == y2) & (x1 != x2)
    ver = (~isvia) & (dirs[np.clip(zz, 0, L - 1)] == 1) & (x1 == x2) & (y1 != y2)
    merged = {}
    for d_, m in ((0, hor), (1, ver)):
        if not m.any():
            merged[d_] = (np.zeros(0, np.int64),) * 5
            continue
        nn_, z_ = n[m], zz[m]
        fx = y1[m] if d_ == 0 else x1[m]
        a = np.minimum(x1[m], x2[m]) if d_ == 0 else np.minimum(y1[m], y2[m])
        b = np.maximum(x1[m], x2[m]) if d_ == 0 else np.maximum(y1[m], y2[m])
        o = np.lexsort((a, fx, z_, nn_))
        nn_, z_, fx, a, b = nn_[o], z_[o], fx[o], a[o], b[o]
        grp = np.r_[True, (nn_[1:] != nn_[:-1]) | (z_[1:] != z_[:-1]) | (fx[1:] != fx[:-1])]
        gid = np.cumsum(grp) - 1
        big = (max(X, Y) + 2)
        cm = np.maximum.accumulate(gid * big + b) - gid * big            # running max of b within group
        prev = np.r_[-1, cm[:-1]]
        start = grp | (a > prev)                                          # a <= running max -> overlap/touch
        sid = np.cumsum(start) - 1
        ma = a[start]
        mb = np.zeros(int(sid[-1]) + 1, np.int64)
        np.maximum.at(mb, sid, b)
        merged[d_] = (nn_[start], z_[start], fx[start], ma, mb)
        # wire demand: cells a..b-1 along the run, +2 each (difference array)
        mz, mf = z_[start], fx[start]
        if d_ == 0:
            diff = np.zeros((L, X + 1, Y))
            np.add.at(diff, (mz, ma, mf), 2.0)
            np.add.at(diff, (mz, mb, mf), -2.0)
            la.d += np.cumsum(diff, axis=1)[:, :X, :]
        else:
            diff = np.zeros((L, X, Y + 1))
            np.add.at(diff, (mz, mf, ma), 2.0)
            np.add.at(diff, (mz, mf, mb), -2.0)
            la.d += np.cumsum(diff, axis=2)[:, :, :Y]
    # unit vias (unique per net), skipped on layer 0 and where a same-net wire on that layer covers the cell
    vi = np.nonzero(isvia)[0]
    if not len(vi):
        return
    zl, zh = np.minimum(z1[vi], z2[vi]), np.maximum(z1[vi], z2[vi])
    span = zh - zl
    rep = np.repeat(vi, span)
    vz = np.minimum(z1[rep], z2[rep]) + (np.arange(len(rep)) - np.repeat(np.cumsum(span) - span, span))
    vn, vx, vy = n[rep], x1[rep], y1[rep]
    key = ((vn * L + vz) * X + vx) * Y + vy
    key, first = np.unique(key, return_index=True)
    vn, vz, vx, vy = vn[first], vz[first], vx[first], vy[first]
    keep = vz != 0
    vn, vz, vx, vy = vn[keep], vz[keep], vx[keep], vy[keep]
    vdir = dirs[vz]
    covered = np.zeros(len(vn), bool)
    for d_ in (0, 1):
        mn, mz, mf, ma, mb = merged[d_]
        sel = np.nonzero(vdir == d_)[0]
        if not len(sel) or not len(mn):
            continue
        fq = vy[sel] if d_ == 0 else vx[sel]
        tq = vx[sel] if d_ == 0 else vy[sel]
        big = max(X, Y) + 2
        gk = (mn * L + mz) * big + mf
        gq = (vn[sel] * L + vz[sel]) * big + fq
        o = np.lexsort((ma, gk))
        gk_s, ma_s, mb_s = gk[o], ma[o], mb[o]
        comp = gk_s * big + ma_s
        pos = np.searchsorted(comp, gq * big + tq, side="right") - 1
        ok = pos >= 0
        p2 = np.maximum(pos, 0)
        hit = ok & (gk_s[p2] == gq) & (ma_s[p2] <= tq) & (tq <= mb_s[p2])
        covered[sel[hit]] = True
    vn, vz, vx, vy, vdir = vn[~covered], vz[~covered], vx[~covered], vy[~covered], vdir[~covered]
    for d_ in (0, 1):
        m = vdir == d_
        z_, x_, y_ = vz[m], vx[m], vy[m]
        t = x_ if d_ == 0 else y_
        T = X if d_ == 0 else Y
        mid = (t > 0) & (t < T - 1)
        hi_ = (~mid) & (t > 0)
        lo_ = (~mid) & (~hi_) & (t < T - 1)
        if d_ == 0:
            np.add.at(la.d, (z_[mid], x_[mid] - 1, y_[mid]), 1.0)
            np.add.at(la.d, (z_[mid], x_[mid], y_[mid]), 1.0)
            np.add.at(la.d, (z_[hi_], x_[hi_] - 1, y_[hi_]), 2.0)
            np.add.at(la.d, (z_[lo_], x_[lo_], y_[lo_]), 2.0)
        else:
            np.add.at(la.d, (z_[mid], x_[mid], y_[mid] - 1), 1.0)
            np.add.at(la.d, (z_[mid], x_[mid], y_[mid]), 1.0)
            np.add.at(la.d, (z_[hi_], x_[hi_], y_[hi_] - 1), 2.0)
            np.add.at(la.d, (z_[lo_], x_[lo_], y_[lo_]), 2.0)


# ════════════════════════════════════════════════════════════════════════════════════
#  state
# ════════════════════════════════════════════════════════════════════════════════════
def segs_of(wires, vias, dirs):
    out = []
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            out.append((a, fixed, z, b, fixed, z) if dirs[z] == 0 else (fixed, a, z, fixed, b, z))
    for (x, y, z) in sorted(vias):
        out.append((x, y, z, x, y, z + 1))
    return out


class State:
    def __init__(self, design, suite, base, fast_demand=True):
        self.design, self.suite = design, suite
        self.case = SC.case_of(design, suite)
        _, self.r1, self.r2, self.r3, self.nend, (self.w1, self.w2, self.w3, self.w4) = SC.CASES[self.case]
        t = time.time()
        self.D = FT.Design(design, suite)
        self.T = FT.Timer(self.D)
        self.t_design = time.time() - t
        t = time.time()
        bench = os.path.join(FT.KIT, suite, design)
        self.cap = parse_cap_auto(os.path.join(bench, design + ".cap"))
        self.la = TimingLA(self.cap, self.w4, GRID / 2000.0, 5.0, 2.0)
        self.nets = read_route25(base)                          # [(name, segs, lines)]
        self.ridx = {_key(nm): i for i, (nm, _, _) in enumerate(self.nets)}
        self.segs = [s for _, s, _ in self.nets]
        self.new = {}                                           # route index -> new segments (changed nets)
        self.tid = np.array([self.D.net_index.get(nm, -1) for nm, _, _ in self.nets], np.int64)
        self.tidx_key = {_key(nm): i for i, nm in enumerate(self.D.net_names)}
        counts = np.array([len(s) for s in self.segs], np.int64)
        flat = [x for s in self.segs for x in s]
        arr = np.array(flat, np.int64).reshape(-1, 6) if flat else np.zeros((0, 6), np.int64)
        self.base_arr = np.concatenate([np.repeat(np.arange(len(self.segs)), counts)[:, None], arr], 1)
        self.t_route = time.time() - t
        t = time.time()
        if fast_demand:
            build_demand_fast(self.la, self.base_arr)
        else:
            build_demand(self.la, self.segs)
        self.t_demand = time.time() - t
        # pin ranges per timer net
        D = self.D
        self.pin_start = np.searchsorted(D.pin_net, np.arange(len(D.net_names)))
        self.pin_end = np.searchsorted(D.pin_net, np.arange(len(D.net_names)), side="right")
        self.ap_start = np.cumsum(D.ap_cnt) - D.ap_cnt
        log(f"{design}[{suite}] case {self.case}: design {self.t_design:.1f}s, cap+route {self.t_route:.1f}s, "
            f"demand {self.t_demand:.1f}s; {len(self.segs):,} route nets, {len(self.base_arr):,} segments")

    def pins_of(self, kname):
        """parse_net_ispd25-style pin list [[(l, x, y), ...], ...] of a net (by backslash-free name)."""
        j = self.tidx_key.get(kname)
        if j is None:
            return None
        D = self.D
        out = []
        for p in range(self.pin_start[j], self.pin_end[j]):
            s = self.ap_start[p]
            out.append([tuple(int(v) for v in D.ap[k]) for k in range(s, s + D.ap_cnt[p])])
        return out

    def timer_array(self):
        """[S,7] array (timer net id, x1, y1, z1, x2, y2, z2) of the current route."""
        arr = self.base_arr
        if self.new:
            ch = np.fromiter(self.new.keys(), np.int64)
            keep = ~np.isin(arr[:, 0], ch)
            extra = [(i,) + s for i, segs in self.new.items() for s in segs]
            ext = np.array(extra, np.int64).reshape(-1, 7) if extra else np.zeros((0, 7), np.int64)
            arr = np.concatenate([arr[keep], ext], 0)
        out = arr.copy()
        out[:, 0] = self.tid[arr[:, 0]]
        return out[out[:, 0] >= 0]

    def time(self):
        r = self.T.analyze(self.timer_array())
        r["of"] = overflow_cost(self.la)
        r["score"] = (self.w1 * (r["wns"] - self.r1) + self.w2 * (r["tns"] - self.r2) / self.nend
                      + self.w4 * r["of"])
        return r

    # ------------------------------------------------------------------ critical paths from the timer
    def crit_paths(self, r, npaths=5000, margin=0.0):
        """traced worst paths of the npaths worst endpoints with est slack < margin; slacks shifted by -margin
        (pessimism for the timer's tail error: near-critical paths get TNS/WNS weight too)."""
        D = self.D
        est = r["pin_est"] - margin
        cand = np.nonzero(est < 0)[0]
        cand = cand[np.argsort(est[cand], kind="stable")][:npaths]
        paths = []
        for e in cand:
            nets = []
            for q in self.T.trace(int(e)):
                if D.pin_is_drv[q] and not D.net_clock[D.pin_net[q]]:
                    nets.append(D.net_names[D.pin_net[q]])
            paths.append((float(est[e]), nets))
        return paths

    def net_rc(self, r):
        """per-net (c_load fF, r_drv kOhm) for TimingLA (net_rc_params conventions)."""
        rd = self.T._rdrv(r)
        cl = np.clip(0.5 * r["net_ctot"], 1.0, 1e5)
        return cl, rd

    # ------------------------------------------------------------------ one promotion round
    def promote(self, r, max_nets=4000, tw_scale=1.0, fl=None, npaths=5000, margin=0.0):
        D, la = self.D, self.la
        paths = self.crit_paths(r, npaths, margin)
        tw = net_weights(paths, self.case)
        cl, rd = self.net_rc(r)
        order = sorted(tw, key=lambda n: -tw[n][0])[:max_nets]
        undo = {}
        stats = defaultdict(int)
        for kname in order:
            i = self.ridx.get(kname)
            j = self.tidx_key.get(kname)
            pins = self.pins_of(kname)
            if i is None or j is None or not pins:
                stats["not_in_route"] += 1
                continue
            segs = self.new.get(i, self.segs[i])
            w, v = to_wires_vias(segs, la.dirs)
            apply(la, w, v, -1)
            chosen = used_access_points(segs, pins)
            nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins, chosen)])
            la.tw = tw[kname][0] * tw_scale
            la.c_load, la.r_drv = float(cl[j]), float(rd[j])
            cands = []
            p = RE.segments_to_polys(segs, nodes)
            if RE.tree_connects(nodes, p):
                cands.append(("base", p))
            if fl is not None and len(nodes) > 1:
                from rsmt_route import rsmt_polys
                cands.append(("rsmt", rsmt_polys(la, fl, nodes, 1.0)))
            if not cands:
                apply(la, w, v, +1)
                stats["kept"] += 1
                continue
            best = None
            for tag, polys in cands:
                cost = 0.0
                for pts in polys:
                    runs = RE.to_runs(pts)
                    if runs:
                        cost += la.assign_path(runs, nodes.get(pts[0]), nodes.get(pts[-1]), 1.0)[1]
                if best is None or cost < best[0]:
                    best = (cost, tag, polys)
            wires, vias = RE.route_net(la, nodes, best[2], 1.0, mark=True)     # commits the new demand
            undo.setdefault(i, segs)
            self.new[i] = segs_of(wires, vias, la.dirs)
            stats["topo_" + best[1]] += 1
        la.tw = 0.0
        return undo, dict(stats)

    def revert(self, undo):
        la = self.la
        for i, old in undo.items():
            cur = self.new.get(i, self.segs[i])
            w, v = to_wires_vias(cur, la.dirs)
            apply(la, w, v, -1)
            w, v = to_wires_vias(old, la.dirs)
            apply(la, w, v, +1)
            if old is self.segs[i]:
                self.new.pop(i, None)
            else:
                self.new[i] = old

    def replace_from_route(self, path):
        """adopt the nets that differ in another route file of the same design (GNN pass output)."""
        la = self.la
        undo = {}
        for name, segs, _ in read_route25(path):
            i = self.ridx.get(_key(name))
            if i is None:
                continue
            cur = self.new.get(i, self.segs[i])
            if sorted(segs) == sorted(cur):
                continue
            w, v = to_wires_vias(cur, la.dirs)
            apply(la, w, v, -1)
            w, v = to_wires_vias(segs, la.dirs)
            apply(la, w, v, +1)
            undo.setdefault(i, cur)
            self.new[i] = segs
        return undo

    # ------------------------------------------------------------------ output
    def write(self, path, normalize=True):
        """every net: (normalized) segments; unchanged nets keep their segments."""
        D = self.D
        stats = defaultdict(int)
        with open(path, "w") as fh:
            for i, (name, segs, lines) in enumerate(self.nets):
                cur = self.new.get(i, segs)
                if normalize:
                    j = self.tidx_key.get(_key(name))
                    pts = set()
                    if j is not None:
                        a0 = self.ap_start[self.pin_start[j]] if self.pin_end[j] > self.pin_start[j] else 0
                        a1 = (self.ap_start[self.pin_end[j] - 1] + D.ap_cnt[self.pin_end[j] - 1]
                              if self.pin_end[j] > self.pin_start[j] else 0)
                        for (l, x, y) in D.ap[a0:a1]:
                            pts.add((int(x), int(y), int(l)))
                    out, extra = normalize_net(cur, pts)
                    stats["splits"] += extra
                else:
                    out = cur
                fh.write(f"{name}\n(\n")
                for x1, y1, z1, x2, y2, z2 in out:
                    fh.write(f"{_dbu(x1)} {_dbu(y1)} metal{z1 + 1} {_dbu(x2)} {_dbu(y2)} metal{z2 + 1}\n")
                fh.write(")\n")
        return dict(stats)


# ════════════════════════════════════════════════════════════════════════════════════
#  pipeline
# ════════════════════════════════════════════════════════════════════════════════════
def run(a):
    t_start = time.time()
    S = State(a.design, a.suite, a.base, fast_demand=not a.loop_demand)
    timeline = {"design": S.t_design, "route": S.t_route, "demand": S.t_demand}
    t = time.time()
    r = S.time()
    timeline["timer0"] = time.time() - t
    hist = [{"step": "base", "wns": r["wns"], "tns": r["tns"], "of": r["of"], "score": r["score"],
             "nviol": r["nviol"]}]
    log(f"base: est WNS {r['wns']:.4f} TNS {r['tns']:.1f} ({r['nviol']} viol) OF {r['of']:.4g} "
        f"-> est score {r['score']:+.4f} (timer {timeline['timer0']:.1f}s)")
    fl = None
    if a.topo == "both":
        try:
            from rsmt_route import Flute
            fl = Flute()
        except Exception as e:                                   # FLUTE bridge missing -> base topology only
            log(f"FLUTE unavailable ({e}); base topology only")
    best = r
    for k in range(1, a.rounds + 1):
        t = time.time()
        mx = max(int(a.max_nets * (a.shrink ** (k - 1))), 50)
        undo, st = S.promote(best, mx, a.tw_scale, fl, a.npaths, a.slack_margin)
        tp = time.time() - t
        t = time.time()
        r = S.time()
        tt = time.time() - t
        keep = r["score"] < best["score"] - 1e-9
        log(f"round {k}: promoted {len(undo):,} nets (max {mx}) {st} in {tp:.1f}s; est WNS {r['wns']:.4f} "
            f"TNS {r['tns']:.1f} OF {r['of']:.4g} score {r['score']:+.4f} ({tt:.1f}s) -> "
            f"{'KEEP' if keep else 'reject'}")
        hist.append({"step": f"promote{k}", "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"],
                     "score": r["score"], "kept": bool(keep), "t_opt": tp, "t_timer": tt})
        timeline[f"promote{k}"] = tp + tt
        if keep:
            best = r
        else:
            S.revert(undo)
            if a.stop_on_reject:
                break
    if a.gnn:
        t = time.time()
        tmp_route = os.path.join(a.out_dir, f"{a.design}.{a.tag}.pregnn.route")
        tmp_crit = os.path.join(a.out_dir, f"{a.design}.{a.tag}.est.crit.rpt")
        tmp_out = os.path.join(a.out_dir, f"{a.design}.{a.tag}.gnn.route")
        S.write(tmp_route, normalize=False)
        S.T.write_crit(best, tmp_crit, a.npaths)
        import dtla_lagnn as DD
        ns = argparse.Namespace(design=a.design, suite=a.suite, route=tmp_route, crit=tmp_crit, ppa=None,
                                net_act=None, ckpt=a.gnn, max_nets=a.gnn_max_nets, extra_power_nets=0,
                                steps=a.gnn_steps, out=tmp_out)
        DD.route(ns)
        undo = S.replace_from_route(tmp_out)
        tg = time.time() - t
        t = time.time()
        r = S.time()
        tt = time.time() - t
        keep = r["score"] < best["score"] - 1e-9
        log(f"GNN pass: {len(undo):,} nets changed in {tg:.1f}s; est WNS {r['wns']:.4f} TNS {r['tns']:.1f} "
            f"OF {r['of']:.4g} score {r['score']:+.4f} -> {'KEEP' if keep else 'reject'}")
        hist.append({"step": "gnn", "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"],
                     "score": r["score"], "kept": bool(keep), "t_opt": tg, "t_timer": tt})
        timeline["gnn"] = tg + tt
        if keep:
            best = r
        else:
            S.revert(undo)
        for f in (tmp_route, tmp_out):
            if os.path.exists(f) and not a.keep_tmp:
                os.remove(f)
    t = time.time()
    out = os.path.join(a.out_dir, f"{a.design}.{a.tag}.norm.route")
    st = S.write(out + f".tmp.{os.getpid()}", normalize=True)
    os.replace(out + f".tmp.{os.getpid()}", out)
    timeline["write_norm"] = time.time() - t
    wall = time.time() - t_start
    log(f"wrote {out} ({len(S.new):,} nets changed, {st}); fast-mode wall {wall:.1f}s; est score "
        f"{best['score']:+.4f} (WNS {best['wns']:.4f} TNS {best['tns']:.1f} OF {best['of']:.4g})")
    rec = {"design": a.design, "suite": a.suite, "case": S.case, "tag": a.tag, "base": a.base, "wall_s": wall,
           "timeline": timeline, "history": hist, "nets_changed": len(S.new),
           "args": {k: v for k, v in vars(a).items() if k != "func"}}
    with open(os.path.join(a.out_dir, f"{a.design}.{a.tag}.fastmode.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return 0


# ════════════════════════════════════════════════════════════════════════════════════
#  self test: vectorized demand == loop demand, overflow formula
# ════════════════════════════════════════════════════════════════════════════════════
def self_test():
    ok = True
    rng = np.random.default_rng(1)
    L, X, Y = 6, 12, 10
    cap = {"L": L, "X": X, "Y": Y, "layers": [{"name": f"metal{z+1}", "dir": z % 2,
           "cap": rng.integers(0, 4, (X, Y)).astype(float)} for z in range(L)], "short_costs": [1.0] * L,
           "unit_via": 4.0}
    nets = []
    for n in range(40):
        segs = []
        for _ in range(rng.integers(1, 6)):
            z = int(rng.integers(1, L))
            if rng.random() < 0.4:
                x, y = int(rng.integers(0, X)), int(rng.integers(0, Y))
                za = int(rng.integers(0, L - 1))
                segs.append((x, y, za, x, y, int(rng.integers(za + 1, L))))
            elif z % 2 == 0:
                y = int(rng.integers(0, Y)); a, b = sorted(rng.choice(X, 2, replace=False))
                segs.append((int(a), y, z, int(b), y, z))
            else:
                x = int(rng.integers(0, X)); a, b = sorted(rng.choice(Y, 2, replace=False))
                segs.append((x, int(a), z, x, int(b), z))
        nets.append(segs)
    la1 = TimingLA(cap, 1.0, 2.1, 5.0, 2.0)
    la2 = TimingLA(cap, 1.0, 2.1, 5.0, 2.0)
    build_demand(la1, nets)
    arr = np.array([(i,) + s for i, segs in enumerate(nets) for s in segs], np.int64)
    build_demand_fast(la2, arr)
    good = np.array_equal(la1.d, la2.d)
    ok &= good
    print(f"  vectorized demand == timing_promote.apply loop: {good} (sum {la1.d.sum():.0f} vs {la2.d.sum():.0f})")
    of = overflow_cost(la1)
    good = math.isfinite(of) and of > 0
    ok &= good
    print(f"  overflow cost {of:.4g}: {good}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--base", help="base route (InstantGR output)")
    ap.add_argument("--tag", default="instantgr_f1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_nets", type=int, default=4000)
    ap.add_argument("--shrink", type=float, default=0.5, help="max_nets factor per round (trust region)")
    ap.add_argument("--tw_scale", type=float, default=1.0)
    ap.add_argument("--npaths", type=int, default=5000)
    ap.add_argument("--slack_margin", type=float, default=0.0, help="ns: promote nets on paths with est slack < margin")
    ap.add_argument("--topo", default="both", choices=["both", "base"])
    ap.add_argument("--stop_on_reject", action="store_true")
    ap.add_argument("--gnn", default=None, help="LA checkpoint for one GNN pass (e.g. models/la_ckpt.pt)")
    ap.add_argument("--gnn_max_nets", type=int, default=6000)
    ap.add_argument("--gnn_steps", type=int, default=0)
    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--loop_demand", action="store_true", help="reference (slow) demand build")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.out_dir is None:
        a.out_dir = os.path.join(ROOT, f"experiments/ispd25_routes{'_blind' if a.suite == 'blind' else ''}/{a.design}")
    os.makedirs(a.out_dir, exist_ok=True)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
