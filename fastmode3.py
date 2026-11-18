#!/usr/bin/env python3
"""fastmode3.py — ISPD'25 fast mode f3: fastmode.py's pipeline with the full-graph timer fast_sta.py (INSTA-style).

  InstantGR route
    -> fast_sta.py: full-graph STA of the 3-D route (OpenSTA dmp_ceff_elmore port, OpenROAD grt RC, SDC clocks)
    -> R rounds of timing-driven layer promotion (timing_promote's assigner and cost, in memory, as f1), where the nets
       and their weights come from the timer's INSTA-style TNS gradient: weight(net) = |w2| * (# violating endpoints
       whose worst path runs through the net) / N_end + |w1| * (net on the WNS path); every round re-times the WHOLE
       graph and is kept only if the fast model's score w1*WNS + w2*TNS/N + w4*overflow improves
    -> seg_normalize (every net) -> <design>.<tag>.norm.route
The official evaluation is NOT part of this program (a launcher script (not included) runs it once, for reporting). fastmode.py
(f1/f2) is reused read-only; this file only replaces its timer and its net ranking.

  python fastmode3.py --design ariane --suite visible --base experiments/ispd25_routes/ariane/ariane.instantgr.route \
                      --tag instantgr_f3 [--rounds 3 --max_nets 4000 --slack_margin 0.0]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import fast_sta as FS                                        # NEW (this agent)
import fastmode as FM                                        # READ-ONLY reuse (f1/f2 driver)
import ispd25_score as SC
import route_export as RE
from ispd25_adapter import parse_cap_auto
from la_swap import used_access_points, nodes_from_aps
from timing_promote import TimingLA, apply, to_wires_vias, read_route25, GRID, _key

T0 = time.time()


def log(m):
    print(f"[fastmode3 {time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s] {m}", flush=True)


class _NetView:
    """the .net-file arrays fastmode.State.pins_of() / write() read."""

    def __init__(self, SD):
        self.ap, self.ap_cnt = SD.ap, SD.ap_cnt


class State3(FM.State):
    def __init__(self, design, suite, base):                 # noqa: super().__init__ would build fast_timer's model
        self.design, self.suite = design, suite
        self.case = SC.case_of(design, suite)
        _, self.r1, self.r2, self.r3, self.nend, (self.w1, self.w2, self.w3, self.w4) = SC.CASES[self.case]
        t = time.time()
        self.SD = FS.StaDesign(design, suite)
        self.T = FS.StaTimer(self.SD)
        self.t_design = time.time() - t
        t = time.time()
        bench = os.path.join(FS.KIT, suite, design)
        self.cap = parse_cap_auto(os.path.join(bench, design + ".cap"))
        self.la = TimingLA(self.cap, self.w4, GRID / 2000.0, 5.0, 2.0)
        self.nets = read_route25(base)
        self.ridx = {_key(nm): i for i, (nm, _, _) in enumerate(self.nets)}
        self.segs = [s for _, s, _ in self.nets]
        self.new = {}
        SD = self.SD
        self.tid = np.array([SD.net_index.get(_key(nm), -1) for nm, _, _ in self.nets], np.int64)   # DEF net ids
        self.D = _NetView(SD)
        self.tidx_key = {_key(nm): i for i, nm in enumerate(SD.nf_names)}
        counts = np.array([len(s) for s in self.segs], np.int64)
        flat = [x for s in self.segs for x in s]
        arr = np.array(flat, np.int64).reshape(-1, 6) if flat else np.zeros((0, 6), np.int64)
        self.base_arr = np.concatenate([np.repeat(np.arange(len(self.segs)), counts)[:, None], arr], 1)
        self.t_route = time.time() - t
        t = time.time()
        FM.build_demand_fast(self.la, self.base_arr)
        self.t_demand = time.time() - t
        nf = len(SD.nf_names)
        self.pin_start = np.searchsorted(SD.nf_pin_net, np.arange(nf))
        self.pin_end = np.searchsorted(SD.nf_pin_net, np.arange(nf), side="right")
        self.ap_start = np.cumsum(SD.ap_cnt) - SD.ap_cnt
        self._rc, self._dirty = None, set()                    # RC cache + DEF nets changed since its build
        log(f"{design}[{suite}] case {self.case}: design+timing graph {self.t_design:.1f}s, cap+route {self.t_route:.1f}s, "
            f"demand {self.t_demand:.1f}s; {len(self.segs):,} route nets, {len(self.base_arr):,} segments")

    def time(self, margin=0.0):
        arr = self.timer_array()
        t = time.time()
        if self._rc is None:
            self._rc = FS.build_rc(self.SD, arr)
        elif self._dirty:
            FS.update_rc(self.SD, self._rc, arr, np.fromiter(self._dirty, np.int64))
        self._dirty.clear()
        t_rc = time.time() - t
        r = self.T.analyze(arr, rat=False, margin=margin, rc=self._rc)
        r["t_rc"] = t_rc
        r["of"] = FM.overflow_cost(self.la)
        r["score"] = (self.w1 * (r["wns"] - self.r1) + self.w2 * (r["tns"] - self.r2) / self.nend
                      + self.w4 * r["of"])
        return r

    def weights(self, r):
        """INSTA-style score gradient per DEF net: |w2| * k / N_end + |w1| * on_wns."""
        return abs(self.w2) * r["net_cnt"] / self.nend + abs(self.w1) * r["net_on_wns"]

    def promote(self, r, max_nets=4000, tw_scale=1.0, fl=None, npaths=None, margin=0.0):
        SD, la = self.SD, self.la
        w = self.weights(r)
        cand = np.nonzero(w > 0)[0]
        order = cand[np.argsort(-w[cand], kind="stable")][:max_nets]
        cl = np.clip(0.5 * r["net_ctot"], 1.0, 1e5)
        rd = r["net_rd_kohm"]
        undo = {}
        stats = defaultdict(int)
        for n in order:
            kname = SD.net_names[n]
            i = self.ridx.get(kname)
            pins = self.pins_of(kname)
            if i is None or not pins:
                stats["not_in_route"] += 1
                continue
            segs = self.new.get(i, self.segs[i])
            wv, vv = to_wires_vias(segs, la.dirs)
            apply(la, wv, vv, -1)
            chosen = used_access_points(segs, pins)
            nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins, chosen)])
            la.tw = float(w[n]) * tw_scale
            la.c_load, la.r_drv = float(cl[n]), float(rd[n])
            cands = []
            p = RE.segments_to_polys(segs, nodes)
            if RE.tree_connects(nodes, p):
                cands.append(("base", p))
            if fl is not None and len(nodes) > 1:
                from rsmt_route import rsmt_polys
                cands.append(("rsmt", rsmt_polys(la, fl, nodes, 1.0)))
            if not cands:
                apply(la, wv, vv, +1)
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
            wires, vias = RE.route_net(la, nodes, best[2], 1.0, mark=True)
            undo.setdefault(i, segs)
            self.new[i] = FM.segs_of(wires, vias, la.dirs)
            self._dirty.add(int(self.tid[i]))
            stats["topo_" + best[1]] += 1
        la.tw = 0.0
        return undo, dict(stats)

    def revert(self, undo):
        super().revert(undo)
        self._dirty.update(int(self.tid[i]) for i in undo)


def run(a):
    t_start = time.time()
    S = State3(a.design, a.suite, a.base)
    timeline = {"design": S.t_design, "route": S.t_route, "demand": S.t_demand}
    t = time.time()
    r = S.time(a.slack_margin)
    timeline["timer0"] = time.time() - t
    hist = [{"step": "base", "wns": r["wns"], "tns": r["tns"], "of": r["of"], "score": r["score"], "nviol": r["nviol"],
             "t_rc": r["t_rc"], "t_prop": r["t_prop"]}]
    log(f"base: est WNS {r['wns']:.4f} TNS {r['tns']:.1f} ({r['nviol']}/{r['n_ep']} viol, {r['missing']} missing pins) "
        f"OF {r['of']:.4g} -> est score {r['score']:+.4f} (timer {timeline['timer0']:.1f}s: rc {r['t_rc']:.1f} prop "
        f"{r['t_prop']:.1f})")
    fl = None
    if a.topo == "both":
        try:
            from rsmt_route import Flute
            fl = Flute()
        except Exception as e:
            log(f"FLUTE unavailable ({e}); base topology only")
    best = r
    for k in range(1, a.rounds + 1):
        t = time.time()
        mx = max(int(a.max_nets * (a.shrink ** (k - 1))), 50)
        undo, st = S.promote(best, mx, a.tw_scale, fl, None, a.slack_margin)
        tp = time.time() - t
        t = time.time()
        r = S.time(a.slack_margin)
        tt = time.time() - t
        keep = r["score"] < best["score"] - 1e-9
        log(f"round {k}: promoted {len(undo):,} nets (max {mx}) {st} in {tp:.1f}s; est WNS {r['wns']:.4f} TNS "
            f"{r['tns']:.1f} OF {r['of']:.4g} score {r['score']:+.4f} ({tt:.1f}s) -> {'KEEP' if keep else 'reject'}")
        hist.append({"step": f"promote{k}", "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"],
                     "score": r["score"], "kept": bool(keep), "t_opt": tp, "t_timer": tt})
        timeline[f"promote{k}"] = tp + tt
        if keep:
            best = r
        else:
            S.revert(undo)
            if a.stop_on_reject:
                break
    t = time.time()
    out = os.path.join(a.out_dir, f"{a.design}.{a.tag}.norm.route")
    st = S.write(out + f".tmp.{os.getpid()}", normalize=True)
    os.replace(out + f".tmp.{os.getpid()}", out)
    timeline["write_norm"] = time.time() - t
    wall = time.time() - t_start
    log(f"wrote {out} ({len(S.new):,} nets changed, {st}); fast-mode wall {wall:.1f}s; est score {best['score']:+.4f} "
        f"(WNS {best['wns']:.4f} TNS {best['tns']:.1f} OF {best['of']:.4g})")
    rec = {"design": a.design, "suite": a.suite, "case": S.case, "tag": a.tag, "base": a.base, "wall_s": wall,
           "timer": "fast_sta", "timeline": timeline, "history": hist, "nets_changed": len(S.new),
           "est": {"wns": best["wns"], "tns": best["tns"], "of": best["of"], "score": best["score"]},
           "args": {k: v for k, v in vars(a).items() if k != "func"}}
    with open(os.path.join(a.out_dir, f"{a.design}.{a.tag}.fastmode.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--base", required=True, help="base route (InstantGR output)")
    ap.add_argument("--tag", default="instantgr_f3")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_nets", type=int, default=4000)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--tw_scale", type=float, default=1.0)
    ap.add_argument("--slack_margin", type=float, default=0.0, help="ns: endpoints with slack < margin count")
    ap.add_argument("--topo", default="both", choices=["both", "base"])
    ap.add_argument("--stop_on_reject", action="store_true")
    a = ap.parse_args()
    if not a.tag.startswith("instantgr_f3"):
        ap.error("tag must start with instantgr_f3")
    if a.out_dir is None:
        a.out_dir = os.path.join(ROOT, f"experiments/ispd25_routes{'_blind' if a.suite == 'blind' else ''}/{a.design}")
    os.makedirs(a.out_dir, exist_ok=True)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
