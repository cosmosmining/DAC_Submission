#!/usr/bin/env python3
"""fastmode4.py — ISPD'25 fast mode f4: fastmode3.py (fast_sta timer, INSTA-weighted promotion) plus the passes the
f3 results lack (an internal note: bsg needs pin escape; tile / NVDLA need power).

  InstantGR route
    -> [--escape on|auto] escape3d.py (subprocess): spread pin via stacks out of blocked macro-pin hotspots (exact
       contest overflow; never worse by construction); auto = only where blocked cells hold most of the overflow
       (bsg: escape3d takes ~65 s there and ~840 s on ariane blind, where it does not help)
    -> [--dtla] one DTLA pass (dtla.py's differentiable layer assignment; learned start with --gnn) on the
       timer's critical nets plus the --power_nets longest other nets, kept only if the fast score improves. Its
       critical-path report is written from fast_sta's own arrival times (write_crit below), in the crit_nets.tcl
       format dtla.build reads. It runs BEFORE the promotion rounds by default: on the InstantGR route the fast timer
       agrees with DTLA's model (f4_diag.py, ariane v: model -7.15, fast_sta -6.63), after promotion it does not.
    -> fast_sta.py STA; R rounds of timing promotion (fastmode3.State3.promote), each kept only if the fast score
       improves
  With per-net switching activity (--act FILE: an OpenSTA net_activity dump, e.g. ppa_run/official/
  net_activity_noroute.tcl; --activity fast: fast_power.py, which is NOT accurate: ariane v switching power 0.0716 W
  against OpenROAD's 0.00094 W, while the OpenSTA activities give 0.00093 W) every accept/reject also prices power:
  fast score = w1*WNS + w2*TNS/N + w4*overflow + w3*P_sw, where P_sw = sum over nets of 0.5 V^2 density_n C_n (the
  route-dependent part of the power; nets driven by ports and clock nets are left out, as in dtla_lagnn.route).
    -> seg_normalize -> <design>.<tag>.norm.route
The official evaluation is not part of this program (a launcher script (not included) runs it once, for reporting).
fastmode.py, fastmode3.py, fast_sta.py, fast_power.py, dtla.py, dtla_lagnn.py and escape3d.py are reused read-only.

  python fastmode4.py --design bsg_chip --suite visible --base experiments/ispd25_routes/bsg_chip/bsg_chip.instantgr.route \
                      --tag instantgr_f4 --escape auto --dtla --act X.act [--gnn models/la_ckpt.pt]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
if os.environ.get("FAST_STA_V2"):                            # the OpenSTA-consistency fixes (fast_sta_v2.py header)
    import fast_sta_v2
    sys.modules["fast_sta"] = fast_sta_v2                    # fastmode3 / fast_power then import the fixed timer too
import dtla as DT                                            # READ-ONLY reuse
import fast_sta as FS                                        # READ-ONLY reuse
import fastmode as FM                                        # READ-ONLY reuse
import fastmode3 as F3                                       # READ-ONLY reuse
import route_export as RE                                    # READ-ONLY reuse
from timing_promote import _key, apply, read_route25, to_wires_vias

T0 = time.time()
VDD = 1.1


def log(m):
    print(f"[fastmode4 {time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s] {m}", flush=True)


# ─────────────────────────────── critical paths from fast_sta ───────────────────────────────

def endpoint_rows(D, r):
    """(check-row pins, slack [rows, K, 2]) from analyze(keep=True): the endpoint section of
    fast_sta.StaTimer.analyze, recomputed from the kept arrival times and slews."""
    T = D.lib.tab
    AT, SL = r["AT"], r["SL"]
    R = D.ck_rows
    rp = R["pin"]
    setup = np.zeros((len(rp), 2))
    for rf, tk in ((0, "tr"), (1, "tf")):
        tid = R[tk]
        m = tid >= 0
        setup[m, rf] = T(tid[m], R["cks"][m], SL[rp[m], rf])
        setup[(~m) & (R["type"] <= 1), rf] = np.nan
    req = D.REL[:, R["evt"]].T
    reqk = req[:, :, None] - setup[:, None, :] - R["off"][:, None, None]
    at = AT[rp]
    valid = (at > FS.NEG / 2) & (req[:, :, None] < FS.BIG / 2) & np.isfinite(reqk)
    return rp, np.where(valid, reqk - at, FS.BIG)


def trace(D, r, e, k, rf):
    """worst path into endpoint pin e for clock class k / transition rf (analyze's WNS-path tracing)."""
    AT, BEST = r["AT"], r["BEST"]
    nE = len(D.ce_from)
    path, p = [e], e
    for _ in range(100000):
        if D.pin_is_load[p] and not D.is_port[p] or (D.is_port[p] and not D.pin_isdrvdir[p]):
            q = D.net_drv[D.pin_net[p]]
            if q < 0 or AT[q, k, rf] <= FS.NEG / 2:
                break
            p = q
        else:
            eb = BEST[p, k, rf]
            if eb < 0 or eb >= nE:
                break
            p, rf = int(D.ce_from[eb]), int(D.ce_irf[eb])
        path.append(p)
    return path[::-1]


def worst_rows(rp, sl, slack_max=0.0):
    """the worst check row of every endpoint pin with slack < slack_max, worst first."""
    rmin = sl.reshape(len(rp), -1).min(1)
    rows = np.nonzero(rmin < slack_max)[0]
    rows = rows[np.lexsort((rmin[rows], rp[rows]))]
    rows = rows[np.r_[True, rp[rows][1:] != rp[rows][:-1]]] if len(rows) else rows
    return rows[np.argsort(rmin[rows], kind="stable")], rmin


def check_rows(D, r):
    """the recomputed endpoint slacks and WNS path must equal analyze's own (else the report would lie)."""
    rp, sl = endpoint_rows(D, r)
    rows, rmin = worst_rows(rp, sl, FS.BIG / 2)
    ep = np.full(D.P, FS.BIG)
    np.minimum.at(ep, rp, rmin)
    same_ep = np.array_equal(ep[r["ep_pin"]], r["ep_slack"])
    same_wns = True
    if len(rows) and len(r["wns_path"]):
        j = int(np.argmin(sl[rows[0]].reshape(-1)))
        same_wns = trace(D, r, int(rp[rows[0]]), j // 2, j % 2) == list(r["wns_path"])
    return bool(same_ep), bool(same_wns)


def write_crit(D, r, path, npaths=5000, slack_max=0.0):
    """crit_nets.tcl-format report (what timing_promote.parse_crit / parse_net_rc and dtla.build read) of the worst
    path of each of the npaths worst endpoints with slack < slack_max, from fast_sta's analysis (keep=True). Driver
    delay = driver resistance x net load (kOhm x fF), so parse_net_rc recovers the timer's driver resistance."""
    rp, sl = endpoint_rows(D, r)
    rows, rmin = worst_rows(rp, sl, slack_max)
    rows = rows[:npaths]
    nn = len(D.net_names)
    fan = np.bincount(D.pin_net[D.pin_is_load & (D.pin_net >= 0)], minlength=nn)
    ctot, rd = r["net_ctot"], r["net_rd_kohm"]
    with open(path, "w") as fh:
        for row in rows:
            e = int(rp[row])
            j = int(np.argmin(sl[row].reshape(-1)))
            pins = trace(D, r, e, j // 2, j % 2)
            fh.write(f"Startpoint: {D.pin_name[pins[0]]}\nEndpoint: {D.pin_name[e]}\nPath Group: est\n"
                     f"Path Type: max\n\nFanout        Cap      Delay       Time   Description\n")
            t = 0.0
            for q in pins:
                n = D.pin_net[q]
                if not D.pin_is_drv[q] or n < 0:
                    continue
                cap = max(float(ctot[n]), 1e-3)
                dl = float(rd[n]) * cap / 1000.0
                t += dl
                inst = D.pin_inst[q]
                cell = D.cell_names[D.inst_ct[inst]] if inst >= 0 else "port"
                fh.write(f"  {int(fan[n]):5d} {cap:10.5f} {dl:10.5f} {t:10.5f} ^ {D.pin_name[q]} ({cell})\n"
                         f"                                          {D.net_names[n]} (net)\n")
            fh.write(f"                              {float(rmin[row]):.5f}   slack (VIOLATED)\n\n\n")
    return len(rows)


# ─────────────────────────────── state with power ───────────────────────────────

class State4(F3.State3):
    kpn = None                                               # per DEF net switching-power slope (W per fF), or None

    full_rc = True
    norm_timing = False                                      # time the changed nets on their seg_normalize'd segments

    def _norm(self, i, segs):
        """seg_normalize of one changed net (as write() does), cached per segment list."""
        c = self._normc.get(i) if hasattr(self, "_normc") else None
        if c is not None and c[0] is segs:
            return c[1]
        if not hasattr(self, "_normc"):
            self._normc = {}
        D = self.D
        j = self.tidx_key.get(_key(self.nets[i][0]))
        pts = set()
        if j is not None and self.pin_end[j] > self.pin_start[j]:
            a0 = self.ap_start[self.pin_start[j]]
            a1 = self.ap_start[self.pin_end[j] - 1] + D.ap_cnt[self.pin_end[j] - 1]
            for (l, x, y) in D.ap[a0:a1]:
                pts.add((int(x), int(y), int(l)))
        out, _ = FM.normalize_net(segs, pts)
        self._normc[i] = (segs, out)
        return out

    def timer_array(self):
        """fastmode.State.timer_array; with norm_timing the changed nets enter the timer normalized, like the route
        OpenROAD reads: on the raw in-memory segments of re-routed nets fast_sta is optimistic (ariane v f4b: in-run
        WNS -0.243 / TNS -403 after promotion, OpenROAD -0.406 / -836, while a fresh fast_sta run on the normalized
        route matches OpenROAD within 0.003 ns)."""
        if not self.norm_timing:
            return super().timer_array()
        arr = self.base_arr
        if self.new:
            ch = np.fromiter(self.new.keys(), np.int64)
            keep = ~np.isin(arr[:, 0], ch)
            extra = [(i,) + tuple(sg) for i, segs in self.new.items() for sg in self._norm(i, segs)]
            ext = np.array(extra, np.int64).reshape(-1, 7) if extra else np.zeros((0, 7), np.int64)
            arr = np.concatenate([arr[keep], ext], 0)
        out = arr.copy()
        out[:, 0] = self.tid[arr[:, 0]]
        return out[out[:, 0] >= 0]

    def time(self, margin=0.0, keep=False, rat=False):
        """fastmode3.State3.time + keep + the switching-power term, with the RC trees REBUILT from scratch every time:
        fastmode3's incremental FS.update_rc is optimistic after promotion (tile v instantgr_f3: in-run estimate WNS
        -0.370 / TNS -1256, a fresh build_rc of the same route -0.525 / -2456, OpenROAD -0.531 / -2470;
        a launcher script (not included)), and a full build costs ~2 s on ariane / tile."""
        arr = self.timer_array()
        t = time.time()
        if self._rc is None or self.full_rc:
            self._rc = FS.build_rc(self.SD, arr)
        elif self._dirty:
            FS.update_rc(self.SD, self._rc, arr, np.fromiter(self._dirty, np.int64))
        self._dirty.clear()
        t_rc = time.time() - t
        r = self.T.analyze(arr, rat=rat, margin=margin, rc=self._rc, keep=keep)
        r["t_rc"] = t_rc
        r["of"] = FM.overflow_cost(self.la)
        r["score"] = (self.w1 * (r["wns"] - self.r1) + self.w2 * (r["tns"] - self.r2) / self.nend
                      + self.w4 * r["of"])
        r["pw"] = float((self.kpn * r["net_ctot"]).sum()) if self.kpn is not None else 0.0
        r["score"] += self.w3 * r["pw"]
        return r

    def replace_from_route(self, path, only=None):
        """fastmode.State.replace_from_route (adopt the nets that differ in another route of the design), optionally
        only for the route-net indices in `only`; marks the adopted nets for the RC cache."""
        la = self.la
        undo = {}
        for name, segs, _ in read_route25(path):
            i = self.ridx.get(_key(name))
            if i is None or (only is not None and i not in only):
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
        self._dirty.update(int(self.tid[i]) for i in undo)
        return undo

    def _priced(self):
        """nets whose switching power report_power counts and dtla_lagnn.route prices: not port-driven, no clock."""
        SD = self.SD
        nn = len(SD.net_names)
        ok = np.ones(nn, bool)
        drv = SD.net_drv
        ok[(drv < 0)] = False
        ok[(drv >= 0) & SD.is_port[np.maximum(drv, 0)]] = False
        cp = SD.pin_net[SD.cnet & (SD.pin_net >= 0)]
        ok[cp] = False
        return ok

    def set_activity(self, dens_per_s):
        """per DEF net density (transitions / s) -> kpn = 0.5 V^2 density (W per fF)."""
        self.kpn = np.where(self._priced(), 0.5 * VDD * VDD * 1e-15 * np.asarray(dens_per_s, float), 0.0)

    def activity_fast(self, r_keep):
        import fast_power as FP                              # READ-ONLY reuse
        A = FP.Activity(self.SD, verbose=False)
        nd = A.net_density(A.densities(r_keep["AT"]))        # transitions per ns
        self.set_activity(np.asarray(nd, float) * 1e9)

    def activity_file(self, path):
        dens = np.zeros(len(self.SD.net_names))
        for line in open(path):
            f = line.split()
            if len(f) >= 2 and not (len(f) >= 3 and f[2] in ("input", "clock")):
                i = self.SD.net_index.get(_key(f[0]))
                if i is not None:
                    try:
                        dens[i] = float(f[1])
                    except ValueError:
                        pass
        self.set_activity(dens)


# ─────────────────────────────── DTLA / GNN pass ───────────────────────────────

def hard(obj, logits, C0):
    with torch.no_grad():
        h = torch.nn.functional.one_hot(obj.probs(logits).argmax(1), obj.K).float() * 50.0
        return float(obj.score(h, C0)[0])


def blocked_share(la):
    """share of the overflow cost above its zero-demand floor that sits in zero-capacity (blocked) cells: ~0.8 on
    bsg_chip, where escape3d pays off; escape is skipped (--escape auto) where it is small."""
    blk = tot = 0.0
    for z in range(1, la.L):
        d, c = la.d[z], la.c[z]
        b = c <= 1e-3                                         # route_export.CAP_EPS
        with np.errstate(over="ignore"):
            eb = np.exp(np.minimum(0.75 * d[b], 700.0)) - 1.0
            eo = np.exp(np.minimum(0.25 * d[~b] - 0.5 * c[~b], 700.0)) - np.exp(-0.5 * c[~b])
        blk += float(la.u[z]) * float(eb.sum())
        tot += float(la.u[z]) * float(eb.sum() + eo.sum())
    return blk / max(tot, 1e-30)


def identified_nets(S, P):
    """-> bool per DTLA net: dtla.build found, for every run of the net, ONE wire covering the whole run (its current
    layer). A straight run whose wires change layer midway (escape3d paths, some re-routed nets) gets slot 0, the
    lowest layer, as its 'current' layer, so DTLA's timing baseline (slack0 - delay change from the current layer) is
    wrong for that net: such nets are never adopted from DTLA's output."""
    safe = np.ones(len(P["net_names"]), bool)
    wires = {}
    for r in range(len(P["run_net"])):
        n = int(P["run_net"][r])
        if not safe[n]:
            continue
        if n not in wires:
            i = int(P["net_route_idx"][n])
            wires[n] = to_wires_vias(S.new.get(i, S.segs[i]), S.la.dirs)[0]
        z = int(P["cand"][r, P["run_cur"][r]])
        lo, hi, f = int(P["run_lo"][r]), int(P["run_hi"][r]), int(P["run_fixed"][r])
        if not any(a <= lo and hi <= b for a, b in wires[n].get((z, f), ())):
            safe[n] = False
    return safe


def dtla_pass(S, a, r_keep, work):
    """-> (undo, stats). DTLA on the timer's critical nets (+ the longest other nets for power), kept by the caller only
    if the fast score improves; never applied when DTLA's own model scores it worse than the current layers."""
    pre, crit, out = work + ".pre.route", work + ".crit.rpt", work + ".dtla.route"
    st = {}
    S.write(pre, normalize=False)
    st["paths"] = write_crit(S.SD, r_keep, crit, a.npaths)
    P = DT.build(S.design, S.suite, pre, crit, a.dtla_nets, a.power_nets)
    st["runs"] = len(P["run_net"])
    undo = {}
    if st["runs"]:
        if S.kpn is not None:
            idx = np.array([S.SD.net_index.get(_key(n), -1) for n in P["net_names"]])
            P["kPn"] = np.where(idx >= 0, S.kpn[np.maximum(idx, 0)], 0.0).astype(np.float32)
        P.update({"kP": 0.0, "a": 0.0, "tau": 0.02})
        obj = DT.Objective(P, 0.0, 0.0, 0.02)
        with torch.no_grad():
            cl = DT.current_logits(P)
            C0 = obj.terms(cl)["Cn"].detach()
            s_cur = hard(obj, cl, C0)
        if a.gnn:
            import dtla_lagnn as DD                        # READ-ONLY reuse
            t = time.time()
            x, ei, sid = DD.graph(P)
            with torch.no_grad():
                init = DD.to_logits(DD.load(a.gnn)(x, ei), sid).clamp(-20, 20) / a.gnn_scale
            st["gnn_s"] = round(time.time() - t, 2)
            st["d_gnn"] = hard(obj, init, C0) - s_cur
        else:
            init = torch.zeros_like(cl)
        th, sh, _ = DT.optimize(obj, init, steps=a.dtla_steps, log_every=10 ** 9)
        st["d_model"] = hard(obj, th, C0) - s_cur
        safe = identified_nets(S, P)
        st["nets"], st["nets_unidentified"] = len(safe), int((~safe).sum())
        if st["d_model"] < 0:
            DT.write(P, th, out)
            undo = S.replace_from_route(out, only={int(P["net_route_idx"][n]) for n in np.nonzero(safe)[0]})
    if not a.keep_tmp:
        for f in (pre, out, crit):
            if os.path.exists(f):
                os.remove(f)
    return undo, st


def dtla_step(S, a, best, hist, timeline):
    """one DTLA pass on the current state; kept only if the fast score improves. -> the new best timing record."""
    t = time.time()
    rk = S.time(a.slack_margin, keep=True)               # the current (= best) state, with arrival arrays
    same_ep, same_wns = check_rows(S.SD, rk)
    log(f"critical-path report check: endpoint slacks identical {same_ep}, WNS path identical {same_wns}")
    if same_ep and same_wns:
        undo, st = dtla_pass(S, a, rk, os.path.join(a.out_dir, f"{a.design}.{a.tag}.tmp{os.getpid()}"))
    else:
        undo, st = {}, {"skipped": "report check failed"}
    del rk
    tg = time.time() - t
    t = time.time()
    r = S.time(a.slack_margin) if undo else best
    tt = time.time() - t
    keep = bool(undo) and r["score"] < best["score"] - 1e-9
    log(f"DTLA pass{' (GNN start)' if a.gnn else ''}: {st}; {len(undo):,} nets changed in {tg:.1f}s; est WNS "
        f"{r['wns']:.4f} TNS {r['tns']:.1f} OF {r['of']:.4g} P_sw {r['pw']:.5f} score {r['score']:+.4f} -> "
        f"{'KEEP' if keep else 'reject'}")
    hist.append({"step": "dtla", "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"], "pw": r["pw"],
                 "score": r["score"], "kept": keep, "t_opt": tg, "t_timer": tt, **st})
    timeline["dtla"] = tg + tt
    if keep:
        best = r
    elif undo:
        S.revert(undo)
    return best


# ─────────────────────────────── fast PPA layer assignment ───────────────────────────────

def ppa_pass(S, a, r_rat):
    """ppa_la.py's coordinate descent (PPALA DP: w4 * overflow + w3 * per-net switching power + timing; exact
    acceptance; slack guard) restricted to the nets whose layers can buy power -- the ACTIVE nets (OpenSTA density > 0),
    largest power weight x wire capacitance first, plus the --ppa_long longest inactive nets (overflow only) -- with
    every net's slack, load and driver resistance from fast_sta instead of OpenROAD dumps. Nets with slack <
    --slack_guard are never touched; the others may slow down by at most --slack_frac x (slack - guard) (delay proxy).
    -> (undo, stats). ppa_la.py itself visits all nets (400-2,100 s per run on the small designs)."""
    import ppa_la as PL                                      # READ-ONLY reuse
    from la_swap import used_access_points, nodes_from_aps
    from timing_promote import TimingLA
    la, SD = S.la, S.SD
    slack, ctot, rd = r_rat["net_slack"], r_rat["net_ctot"], r_rat["net_rd_kohm"]
    kpn = S.kpn if S.kpn is not None else np.zeros(len(SD.net_names))
    st = defaultdict(int)
    cand, longs = [], []
    for i, segs in enumerate(S.segs):
        n = int(S.tid[i])
        if n < 0:
            continue
        if slack[n] < a.slack_guard:
            st["guarded"] += 1
            continue
        cur = S.new.get(i, segs)
        cw = sum((abs(g[3] - g[0]) + abs(g[4] - g[1])) * la.C[g[2]] for g in cur if g[2] == g[5])
        if kpn[n] > 0:
            cand.append((-kpn[n] * cw, i))
        elif a.ppa_long:
            longs.append((-cw, i))
    order = [i for _, i in sorted(cand)[:a.ppa_nets]] + [i for _, i in sorted(longs)[:a.ppa_long]]
    st["active_candidates"], st["visited"] = len(cand), len(order)
    if S.kpn is None:
        log("fast PPA pass: no per-net activity (--act / --activity) -> only the --ppa_long longest nets, overflow only")
    t_pass = time.time()
    w3 = S.w3
    undo = {}
    la.__class__ = PL.PPALA                                  # TimingLA + the power term in run_cost
    try:
        for i in order:
            n = int(S.tid[i])
            kname = _key(S.nets[i][0])
            pins = S.pins_of(kname)
            segs = S.new.get(i, S.segs[i])
            if not pins or not segs:
                continue
            w, v = to_wires_vias(segs, la.dirs)
            apply(la, w, v, -1)
            la.pw, la.tw = w3 * float(kpn[n]), 0.0
            c_old = PL.old_cost(la, w, v)
            chosen = used_access_points(segs, pins)
            nodes = nodes_from_aps([c if c is not None else min(p) for p, c in zip(pins, chosen)])
            polys = RE.segments_to_polys(segs, nodes)
            if not polys or not RE.tree_connects(nodes, polys):
                apply(la, w, v, +1)
                st["no_tree"] += 1
                continue
            c = 0.0
            for pts in polys:
                runs = RE.to_runs(pts)
                if runs:
                    c += la.assign_path(runs, nodes.get(pts[0]), nodes.get(pts[-1]), 1.0)[1]
            if c >= c_old - 1e-12:
                apply(la, w, v, +1)
                st["kept"] += 1
                continue
            wires, vias = RE.route_net(la, nodes, polys, 1.0, mark=True)
            apply(la, wires, vias, -1)
            e_new, e_old = PL.exact_cost(la, wires, vias), PL.exact_cost(la, w, v)
            la.c_load, la.r_drv = min(max(0.5 * float(ctot[n]), 1.0), 1e5), float(rd[n])
            slower = PL.net_delay(la, wires) - PL.net_delay(la, w) > a.slack_frac * (float(slack[n]) - a.slack_guard)
            if slower or e_new >= e_old - 1e-12:
                apply(la, w, v, +1)
                st["slack_rejected" if slower else "kept_exact"] += 1
                continue
            apply(la, wires, vias, +1)
            undo.setdefault(i, segs)
            S.new[i] = FM.segs_of(wires, vias, la.dirs)
            S._dirty.add(n)
            st["changed"] += 1
            if st["changed"] % 2000 == 0:
                log(f"  fast PPA pass: {st['changed']:,} nets changed so far ({time.time() - t_pass:.0f}s)")
    finally:
        la.__class__ = TimingLA
        la.pw, la.tw = 0.0, 0.0
    return undo, dict(st)


def ppa_step(S, a, best, hist, timeline, tag="ppa"):
    t = time.time()
    rr = S.time(a.slack_margin, rat=True)                    # every net's slack, load and driver resistance
    undo, st = ppa_pass(S, a, rr)
    del rr
    tg = time.time() - t
    t = time.time()
    r = S.time(a.slack_margin) if undo else best
    tt = time.time() - t
    keep = bool(undo) and r["score"] < best["score"] - 1e-9
    log(f"fast PPA pass: {st}; {len(undo):,} nets changed in {tg:.1f}s; est WNS {r['wns']:.4f} TNS {r['tns']:.1f} "
        f"OF {r['of']:.4g} P_sw {r['pw']:.5f} score {r['score']:+.4f} ({tt:.1f}s) -> {'KEEP' if keep else 'reject'}")
    hist.append({"step": tag, "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"], "pw": r["pw"],
                 "score": r["score"], "kept": keep, "t_opt": tg, "t_timer": tt, **st})
    timeline[tag] = tg + tt
    if keep:
        return r
    if undo:
        S.revert(undo)
    return best


def repair_step(S, a, best, hist, timeline, tag="repair"):
    """repair3d.py (3-D re-routing of wire runs out of blocked hotspots, exact contest cost) as a subprocess on the
    current in-memory route; its changed nets are adopted and kept only if the fast score improves."""
    t = time.time()
    work = os.path.join(a.out_dir, f"{a.design}.{a.tag}.{tag}.tmp{os.getpid()}")
    pre, out = work + ".pre.route", work + ".rep.route"
    S.write(pre, normalize=True)
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "repair3d.py"), "--design", a.design, "--suite", a.suite,
                         "--route", pre, "--out", out, "--n_hot", str(a.repair_hot), "--passes", "1",
                         "--window", str(a.repair_window)] + (["--overcap"] if a.repair_overcap else [])).returncode
    undo = S.replace_from_route(out) if rc == 0 and os.path.exists(out) else {}
    for f in (pre, out):
        if os.path.exists(f) and not a.keep_tmp:
            os.remove(f)
    tg = time.time() - t
    t = time.time()
    r = S.time(a.slack_margin) if undo else best
    tt = time.time() - t
    keep = bool(undo) and r["score"] < best["score"] - 1e-9
    log(f"3-D repair pass (exit {rc}): {len(undo):,} nets changed in {tg:.1f}s; est WNS {r['wns']:.4f} TNS {r['tns']:.1f} "
        f"OF {r['of']:.4g} P_sw {r['pw']:.5f} score {r['score']:+.4f} ({tt:.1f}s) -> {'KEEP' if keep else 'reject'}")
    hist.append({"step": tag, "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"], "pw": r["pw"],
                 "score": r["score"], "kept": keep, "t_opt": tg, "t_timer": tt, "exit": rc})
    timeline[tag] = tg + tt
    if keep:
        return r
    if undo:
        S.revert(undo)
    return best


# ─────────────────────────────── pipeline ───────────────────────────────

def run(a):
    t_start = time.time()
    timeline = {}
    State4.norm_timing = a.norm_timing
    S = State4(a.design, a.suite, a.base)
    timeline.update({"design": S.t_design, "route": S.t_route, "demand": S.t_demand})
    esc, n_esc = None, 0
    if a.escape != "off":
        t = time.time()
        share = blocked_share(S.la)
        timeline["blocked_share"] = time.time() - t
        if a.escape == "on" or share >= a.esc_share:
            esc = os.path.join(a.out_dir, f"{a.design}.{a.tag}.esc.route.tmp.{os.getpid()}")
            rc = subprocess.run([sys.executable, os.path.join(ROOT, "escape3d.py"), "--design", a.design, "--suite",
                                 a.suite, "--route", a.base, "--out", esc, "--passes", str(a.esc_passes)]).returncode
            if rc == 0 and os.path.exists(esc):
                n_esc = len(S.replace_from_route(esc))
            else:
                log(f"escape3d failed (exit {rc}); continuing from the InstantGR route")
            timeline["escape"] = time.time() - t
        log(f"blocked cells hold {share:.1%} of the overflow excess -> escape3d "
            f"{'%d nets changed in %.1fs' % (n_esc, timeline['escape']) if esc else 'skipped'}")
    t = time.time()
    r = S.time(a.slack_margin, keep=a.activity == "fast")
    if a.activity == "fast" or a.act:
        ta = time.time()
        S.activity_fast(r) if a.activity == "fast" else S.activity_file(a.act)
        timeline["activity"] = time.time() - ta
        r["pw"] = float((S.kpn * r["net_ctot"]).sum())
        r["score"] += S.w3 * r["pw"]
        log(f"switching-power slopes for {int((S.kpn > 0).sum()):,} nets ({timeline['activity']:.1f}s); "
            f"P_sw {r['pw']:.5f} W")
    for k in ("AT", "SL", "gate_d", "wire_d", "BEST", "rc"):
        r.pop(k, None)
    timeline["timer0"] = time.time() - t
    hist = [{"step": "base", "wns": r["wns"], "tns": r["tns"], "of": r["of"], "pw": r["pw"], "score": r["score"],
             "nviol": r["nviol"]}]
    log(f"base: est WNS {r['wns']:.4f} TNS {r['tns']:.1f} ({r['nviol']}/{r['n_ep']} viol, {r['missing']} missing pins) "
        f"OF {r['of']:.4g} P_sw {r['pw']:.5f} -> est score {r['score']:+.4f}")
    fl = None
    if a.topo == "both":
        try:
            from rsmt_route import Flute
            fl = Flute()
        except Exception as e:
            log(f"FLUTE unavailable ({e}); base topology only")
    best = r
    if a.dtla and a.dtla_when in ("before", "both"):
        best = dtla_step(S, a, best, hist, timeline)
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
            f"{r['tns']:.1f} ({r['missing']} missing pins) OF {r['of']:.4g} P_sw {r['pw']:.5f} score {r['score']:+.4f} "
            f"({tt:.1f}s) -> {'KEEP' if keep else 'reject'}")
        hist.append({"step": f"promote{k}", "nets": len(undo), "wns": r["wns"], "tns": r["tns"], "of": r["of"],
                     "pw": r["pw"], "score": r["score"], "kept": bool(keep), "t_opt": tp, "t_timer": tt})
        timeline[f"promote{k}"] = tp + tt
        if keep:
            best = r
        else:
            S.revert(undo)
            if a.stop_on_reject:
                break
    if a.dtla and a.dtla_when in ("after", "both"):
        best = dtla_step(S, a, best, hist, timeline)
    for k in range(1, a.ppa_rounds + 1):
        best = ppa_step(S, a, best, hist, timeline, tag=f"ppa{k}")
    for k in range(1, a.repair + 1):
        best = repair_step(S, a, best, hist, timeline, tag=f"repair{k}")
    t = time.time()
    out = os.path.join(a.out_dir, f"{a.design}.{a.tag}.norm.route")
    stw = S.write(out + f".tmp.{os.getpid()}", normalize=True)
    os.replace(out + f".tmp.{os.getpid()}", out)
    timeline["write_norm"] = time.time() - t
    if esc and os.path.exists(esc):
        os.remove(esc)
    wall = time.time() - t_start
    log(f"wrote {out} ({len(S.new):,} nets changed vs the InstantGR route, {stw}); "
        f"fast-mode wall {wall:.1f}s; est score {best['score']:+.4f} (WNS {best['wns']:.4f} TNS {best['tns']:.1f} "
        f"OF {best['of']:.4g} P_sw {best['pw']:.5f})")
    rec = {"design": a.design, "suite": a.suite, "case": S.case, "tag": a.tag, "base": a.base, "wall_s": wall,
           "timer": "fast_sta", "timeline": timeline, "history": hist, "nets_changed": len(S.new),
           "est": {"wns": best["wns"], "tns": best["tns"], "of": best["of"], "pw": best["pw"], "score": best["score"]},
           "args": {k: v for k, v in vars(a).items() if k != "func"}}
    with open(os.path.join(a.out_dir, f"{a.design}.{a.tag}.fastmode.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--base", required=True, help="base route (InstantGR output)")
    ap.add_argument("--tag", default="instantgr_f4")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_nets", type=int, default=4000)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--tw_scale", type=float, default=1.0)
    ap.add_argument("--slack_margin", type=float, default=0.0, help="ns: endpoints with slack < margin count")
    ap.add_argument("--topo", default="both", choices=["both", "base"])
    ap.add_argument("--stop_on_reject", action="store_true")
    ap.add_argument("--escape", default="off", choices=["off", "on", "auto"],
                    help="escape3d.py on the base route first; auto: only if blocked cells hold >= --esc_share of the "
                         "overflow excess")
    ap.add_argument("--esc_share", type=float, default=0.5)
    ap.add_argument("--esc_passes", type=int, default=2)
    ap.add_argument("--activity", default="none", choices=["none", "fast"], help="fast: fast_power.py activities")
    ap.add_argument("--act", default=None, help="OpenSTA net_activity.tcl dump instead of --activity fast")
    ap.add_argument("--dtla", action="store_true", help="one DTLA pass (before or after the promotion rounds)")
    ap.add_argument("--dtla_when", default="before", choices=["before", "after", "both"],
                    help="before: on the (escaped) InstantGR route, where it helps (f4_diag.py); after: f4a-f4g's order")
    ap.add_argument("--dtla_nets", type=int, default=6000, help="critical nets in the DTLA problem")
    ap.add_argument("--power_nets", type=int, default=0, help="longest non-critical nets added to the DTLA problem")
    ap.add_argument("--dtla_steps", type=int, default=300)
    ap.add_argument("--npaths", type=int, default=5000, help="paths in the critical-path report")
    ap.add_argument("--gnn", default=None, help="layer-assignment checkpoint: warm start for the DTLA pass")
    ap.add_argument("--gnn_scale", type=float, default=1.0, help="divide the GNN logits by this before DTLA")
    ap.add_argument("--norm_timing", action="store_true", help="time changed nets on normalized segments")
    ap.add_argument("--repair", type=int, default=0, help="repair3d.py passes after the PPA pass")
    ap.add_argument("--repair_hot", type=int, default=2000); ap.add_argument("--repair_window", type=int, default=10)
    ap.add_argument("--repair_overcap", action="store_true", help="hotspots include open cells with demand > 2x cap")
    ap.add_argument("--ppa_rounds", type=int, default=0, help="fast PPA layer-assignment passes after promotion")
    ap.add_argument("--ppa_nets", type=int, default=60000, help="active nets visited per PPA pass")
    ap.add_argument("--ppa_long", type=int, default=0, help="longest inactive nets visited per PPA pass (overflow)")
    ap.add_argument("--slack_guard", type=float, default=0.05, help="ns: nets with slack below are never touched")
    ap.add_argument("--slack_frac", type=float, default=0.25, help="allowed slow-down x (slack - guard)")
    ap.add_argument("--keep_tmp", action="store_true")
    a = ap.parse_args()
    if not a.tag.startswith("instantgr_f4"):
        ap.error("tag must start with instantgr_f4")
    if a.out_dir is None:
        a.out_dir = os.path.join(ROOT, f"experiments/ispd25_routes{'_blind' if a.suite == 'blind' else ''}/{a.design}")
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
