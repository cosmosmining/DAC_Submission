#!/usr/bin/env python3
"""dtla.py — Differentiable Timing-driven Layer Assignment on the ISPD'25 quality score.

DGR's idea (softmax over discrete routing candidates, optimized by gradient descent) applied
to the decision that dominates ISPD'25: the LAYERS of the timing-critical nets.

Variables: for every straight run r of every critical net, logits over its legal layers
(x-runs: horizontal layers, y-runs: vertical layers; metal1 excluded), p_r = softmax.
Differentiable objective, in score units (lower is better):

  w4 * tof(p)       contest overflow cost on every GCell the runs can touch, with expected
                    wire demand 2*p on top of the fixed demand of all other nets
                    (exp(0.25d - 0.5c) open, exp(0.75d) - 1 blocked; evaluator constants)
+ w3 * kP * C(p)    switching power ~ expected wire capacitance (GAP-LA Table I C per layer)
+ w1 * WNS_s(p)     smooth WNS  = -tau * logsumexp(-slack_p / tau)
+ w2/N * TNS_s(p)   smooth TNS  = -tau * sum softplus(-slack_p / tau)
  slack_p(p) = slack_p^STA - sum_{n in p} a * (C_n(p) - C_n^0)        (a: ns per fF,
               calibrated from STA pairs; see surrogate check)
  C_n includes E[#vias] * C_via, E[#vias] at a junction = sum_z P(min<=z<max), exact under
  independent softmaxes via per-run layer CDFs.

Then: argmax rounding -> the net's runs keep their 2-D geometry, get the chosen layers, and
via stacks join them (as route_export.route_net does) -> ISPD'25 route. STA verifies; the
driver script keeps the best iteration.

The initial logits come from the current layers ("current"), uniform, or the end-to-end
GNN (dtla_gnn.py, trained through this same objective — no teacher).

  python dtla.py --design ariane --route base.norm.route --crit base.crit.rpt \
                 --ppa base.norm.ppa.txt --out new.route [--init gnn:ckpt.pt]
  python dtla.py --self_test
"""
import argparse
import math
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import route_export as RE                                   # READ-ONLY reuse
import ispd25_score as SC
from ispd25_adapter import parse_cap_auto, parse_net_ispd25
from la_swap import used_access_points, nodes_from_aps
from timing_promote import (TimingLA, apply, to_wires_vias, read_route25, parse_crit,
                            net_weights, _key, BENCH_ROOT, GRID, RC, parse_net_rc, net_rc_params)

C_VIA = 0.05          # fF per via (small; keeps stacks from being free)
NEG = -1e4


def log(m):
    print(f"[dtla {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ─────────────────────────────── problem build ───────────────────────────────

def build(design, suite, route_path, crit_path, max_nets=6000, extra_power_nets=0):
    """-> dict of numpy arrays describing the relaxed problem + bookkeeping for writing."""
    t0 = time.time()
    case = SC.case_of(design, suite)
    cap = parse_cap_auto(os.path.join(BENCH_ROOT, suite, design, design + ".cap"))
    pins = {_key(n["name"]): n["pins"]
            for n in parse_net_ispd25(os.path.join(BENCH_ROOT, suite, design, design + ".net"))}
    nets = read_route25(route_path)
    la = TimingLA(cap, 0.0, GRID / 2000.0, 5.0, 2.0)
    for _, segs, _ in nets:
        w, v = to_wires_vias(segs, la.dirs)
        apply(la, w, v, +1)
    paths = [(s, [_key(n) for n in ns]) for s, ns in parse_crit(crit_path) if s < 0]
    netrc = parse_net_rc(crit_path)
    tw = net_weights(parse_crit(crit_path), case)
    idx = {_key(n): i for i, (n, _, _) in enumerate(nets)}
    chosen = [n for n in sorted(tw, key=lambda n: -tw[n][0]) if n in idx and n in pins][:max_nets]
    if extra_power_nets:                     # long non-critical nets (power / congestion)
        def span(segs):
            return sum(abs(s[3] - s[0]) + abs(s[4] - s[1]) for s in segs if s[2] == s[5])
        cs = set(chosen)
        rest = sorted((i for i in range(len(nets)) if _key(nets[i][0]) not in cs),
                      key=lambda i: -span(nets[i][1]))[:extra_power_nets]
        chosen += [_key(nets[i][0]) for i in rest if _key(nets[i][0]) in pins]
    H = [z for z in range(1, la.L) if la.dirs[z] == 0]
    V = [z for z in range(1, la.L) if la.dirs[z] == 1]
    K = max(len(H), len(V))
    run_net, run_dir, run_fixed, run_lo, run_hi, run_len = [], [], [], [], [], []
    run_cur, run_p0, run_p1 = [], [], []
    net_cl, net_rd = [], []
    pt_id, pt_xy, pt_lo, pt_hi = {}, [], [], []
    net_names, net_route_idx, net_nodes = [], [], []
    kept = 0
    for kname in chosen:
        i = idx[kname]
        name, segs, _ = nets[i]
        ch = used_access_points(segs, pins[kname])
        nodes = nodes_from_aps([c if c is not None else min(a) for a, c in zip(pins[kname], ch)])
        polys = RE.segments_to_polys(segs, nodes)
        if not polys or not RE.tree_connects(nodes, polys):
            continue
        # current layer of each run = a wire of this net covering the run on a legal layer
        w_cur, v_cur = to_wires_vias(segs, la.dirs)
        n_id = len(net_names)
        cl, rd = net_rc_params(netrc.get(kname), 5.0, 2.0)
        net_cl.append(cl)
        net_rd.append(rd)
        net_names.append(name)
        net_route_idx.append(i)
        net_nodes.append(nodes)
        apply(la, w_cur, v_cur, -1)          # background = everything but the chosen nets
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
                    pt_lo.append(sp[0] if sp else la.L)      # la.L = "no pin"
                    pt_hi.append(sp[1] if sp else -1)
                ids.append(pt_id[xy])
            run_net.append(n_id); run_dir.append(d); run_fixed.append(fixed)
            run_lo.append(lo); run_hi.append(hi); run_len.append(hi - lo)
            run_cur.append(cur); run_p0.append(ids[0]); run_p1.append(ids[1])
        kept += 1
        pt_id = {}          # points are per net (vias of different nets are separate)
    # candidate layer table per run
    R = len(run_net)
    cand = np.full((R, K), -1, dtype=np.int64)
    for r in range(R):
        cs = H if run_dir[r] == 0 else V
        cand[r, :len(cs)] = cs
    # touched cells: (run, k) -> list of flat cell ids in a compact index
    cell_key, cell_z, cell_x, cell_y = {}, [], [], []
    ent_rk, ent_cell = [], []
    for r in range(R):
        d, f, lo, hi = run_dir[r], run_fixed[r], run_lo[r], run_hi[r]
        for k in range(K):
            z = cand[r, k]
            if z < 0:
                continue
            for t in range(lo, hi):
                key = (z, t, f) if d == 0 else (z, f, t)
                c = cell_key.get(key)
                if c is None:
                    c = cell_key[key] = len(cell_z)
                    cell_z.append(key[0]); cell_x.append(key[1]); cell_y.append(key[2])
                ent_rk.append(r * K + k)
                ent_cell.append(c)
    cz, cx, cy = np.array(cell_z), np.array(cell_x), np.array(cell_y)
    d0 = la.d[cz, cx, cy].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    capv = la.c[cz, cx, cy].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    u = la.u[cz].astype(np.float32) if len(cz) else np.zeros(0, np.float32)
    # paths -> nets incidence (only chosen nets vary)
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
    # expected-delay coefficients (ns) per (run, slot): the greedy's validated Elmore proxy
    #   l*(R_z*c_load + C_z*r_drv)/1000 + R_z*C_z*l^2/2/1000, per-net c_load / r_drv from STA
    um = GRID / 2000.0
    dcoef = np.zeros(cand.shape, np.float32)
    rn = np.array(run_net, dtype=np.int64)
    lum = np.array(run_len, np.float64) * um
    cl_r, rd_r = np.array(net_cl)[rn], np.array(net_rd)[rn]
    for k in range(K):
        z = np.clip(cand[:, k], 0, la.L - 1)
        dk = lum * (Rz[z] * cl_r + Cz[z] * rd_r) / 1000.0 + Rz[z] * Cz[z] * lum * lum / 2.0 / 1000.0
        dcoef[:, k] = np.where(cand[:, k] >= 0, dk, 0.0)
    npts = sum(len(nn) for nn in net_nodes) * 4 + 4 * R + 1
    log(f"{design}[{suite}] case{case}: {kept:,} nets, {R:,} runs, {len(cz):,} touched cells, "
        f"{len(paths):,} violating paths ({time.time() - t0:.0f}s)")
    return {
        "case": case, "design": design, "suite": suite, "L": la.L, "K": K,
        "run_net": np.array(run_net), "run_len": np.array(run_len, np.float32),
        "run_cur": np.array(run_cur), "run_p0": np.array(run_p0), "run_p1": np.array(run_p1),
        "run_dir": np.array(run_dir), "run_fixed": np.array(run_fixed),
        "run_lo": np.array(run_lo), "run_hi": np.array(run_hi),
        "cand": cand, "ent_rk": np.array(ent_rk), "ent_cell": np.array(ent_cell),
        "d0": d0, "cap": capv, "u": u, "cell_z": cz,
        "pt_lo": np.array(pt_lo), "pt_hi": np.array(pt_hi),
        "p_net": np.array(p_net), "p_path": np.array(p_path),
        "slack0": np.array(slack0, np.float32), "Cz": Cz, "um": GRID / 2000.0, "dcoef": dcoef,
        "net_names": net_names, "net_route_idx": net_route_idx, "net_nodes": net_nodes,
        "pt_xy": pt_xy, "dirs": la.dirs, "route_path": route_path,
    }


# ─────────────────────────────── relaxed objective ───────────────────────────────

class Objective(torch.nn.Module):
    """score(logits) for one built problem."""

    def __init__(self, P, a_ns_per_ff, kP, tau=0.02, device="cpu"):
        super().__init__()
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        self.P = P
        _, r1, r2, r3, nend, (w1, w2, w3, w4) = SC.CASES[P["case"]]
        self.w1, self.w2, self.w3, self.w4, self.nend = w1, w2, w3, w4, nend
        self.a, self.kP, self.tau = a_ns_per_ff, kP, tau
        self.K, self.L = P["K"], P["L"]
        self.cand = t(P["cand"], torch.long)
        self.mask = self.cand >= 0
        self.len_um = t(P["run_len"]) * P["um"]
        self.run_net = t(P["run_net"], torch.long)
        self.nn = int(P["run_net"].max()) + 1 if len(P["run_net"]) else 0
        self.ent_rk = t(P["ent_rk"], torch.long)
        self.ent_cell = t(P["ent_cell"], torch.long)
        self.d0, self.capv, self.u = t(P["d0"]), t(P["cap"]), t(P["u"])
        self.open = self.capv > 1e-3
        self.Cz = t(P["Cz"])
        self.dcoef = t(P["dcoef"]) if "dcoef" in P else None
        self.run_cur = t(P["run_cur"], torch.long)
        self.Ccand = torch.where(self.mask, self.Cz[self.cand.clamp(min=0)], torch.zeros(()))
        self.p0, self.p1 = t(P["run_p0"], torch.long), t(P["run_p1"], torch.long)
        self.npt = int(max(P["run_p0"].max(), P["run_p1"].max())) + 1 if len(P["run_p0"]) else 0
        L = self.L
        zr = torch.arange(L, device=device)
        # pin indicator per point: P(max <= z) needs z >= pin_hi; P(min > z) needs z < pin_lo
        self.pin_hi_ok = (zr[None, :] >= t(P["pt_hi"], torch.long)[:, None]).float()
        self.pin_lo_ok = (zr[None, :] < t(P["pt_lo"], torch.long)[:, None]).float()
        self.p_net, self.p_path = t(P["p_net"], torch.long), t(P["p_path"], torch.long)
        self.slack0 = t(P["slack0"])
        self.npath = len(P["slack0"])
        # per-net switching-power slope (W/fF) from OpenSTA activities (net_activity.tcl);
        # without it one global kP prices every net alike (>90% of ariane nets never toggle)
        self.kPn = t(P["kPn"]) if "kPn" in P else None
        self.theta0 = None

    def probs(self, logits):
        return torch.softmax(logits.masked_fill(~self.mask, NEG), dim=1)

    def terms(self, logits, C0=None):
        return self.terms_p(self.probs(logits), C0)

    def terms_p(self, p, C0=None):
        """same as terms() but on per-run layer probabilities p [runs x K] (rows sum to 1,
        zero on invalid slots) — lets straight-through samples pass gradients."""
        # --- overflow on touched cells
        dem = self.d0 + torch.zeros_like(self.d0).index_add_(0, self.ent_cell, 2.0 * p.reshape(-1)[self.ent_rk])
        cost = torch.where(self.open, torch.exp(torch.clamp(0.25 * dem - 0.5 * self.capv, max=60.0)),
                           torch.expm1(torch.clamp(0.75 * dem, max=60.0)))
        tof = (self.u * cost).sum()
        # --- expected vias per junction (layer CDF trick)
        pl = torch.zeros(p.shape[0], self.L, device=p.device).scatter_add_(
            1, self.cand.clamp(min=0), p * self.mask)
        F = torch.cumsum(pl, 1).clamp(1e-6, 1.0)                     # P(z_r <= z)
        G = (1.0 - F).clamp(1e-6, 1.0)                               # P(z_r > z)
        lF = torch.zeros(self.npt, self.L, device=p.device)
        lG = torch.zeros(self.npt, self.L, device=p.device)
        for pid in (self.p0, self.p1):
            lF = lF.index_add(0, pid, torch.log(F))
            lG = lG.index_add(0, pid, torch.log(G))
        pmax_le = torch.exp(lF) * self.pin_hi_ok                     # P(max <= z)
        pmin_gt = torch.exp(lG) * self.pin_lo_ok                     # P(min > z)
        ev = ((1.0 - pmin_gt) - pmax_le).clamp(min=0.0)[:, :-1].sum(1)   # E[#vias] per point
        # --- capacitance per net
        crun = (p * self.Ccand).sum(1) * self.len_um
        Cn = torch.zeros(self.nn, device=p.device).index_add(0, self.run_net, crun)
        # attribute junction vias to the net of an incident run
        vnet = torch.zeros(self.npt, dtype=torch.long, device=p.device)
        vnet[self.p0] = self.run_net
        vnet[self.p1] = self.run_net                                 # points are per net
        Cn = Cn.index_add(0, vnet, ev * C_VIA)
        # --- timing surrogate
        if C0 is None:
            C0 = Cn.detach()
        dly = self.a * (Cn - C0)
        if self.dcoef is not None:
            d_run = (p * self.dcoef).sum(1) - self.dcoef[torch.arange(p.shape[0], device=p.device), self.run_cur]
            dly = dly + torch.zeros(self.nn, device=p.device).index_add(0, self.run_net, d_run)
        dpath = torch.zeros(self.npath, device=p.device).index_add(0, self.p_path, dly[self.p_net])
        slack = self.slack0 - dpath
        tau = self.tau
        wns = -tau * torch.logsumexp(-slack / tau, 0) if self.npath else torch.zeros(())
        tns = -(tau * torch.nn.functional.softplus(-slack / tau)).sum() if self.npath else torch.zeros(())
        return {"tof": tof, "C": Cn.sum(), "wns": wns, "tns": tns, "vias": ev.sum(), "Cn": Cn}

    def score(self, logits, C0):
        return self.score_p(self.probs(logits), C0)

    def score_p(self, p, C0):
        t = self.terms_p(p, C0)
        pw = (self.kPn * t["Cn"]).sum() if self.kPn is not None else self.kP * t["C"]
        s = (self.w4 * t["tof"] + self.w3 * pw + self.w1 * t["wns"]
             + self.w2 * t["tns"] / self.nend)
        return s, t


def current_logits(P, device="cpu", strength=4.0):
    th = torch.zeros(len(P["run_net"]), P["K"], device=device)
    th[torch.arange(th.shape[0]), torch.as_tensor(P["run_cur"], device=device)] = strength
    return th


def optimize(obj, logits0, steps=300, lr=0.1, log_every=100, t_end=0.02, check_every=25):
    """Adam on the relaxed score with the softmax temperature annealed 1 -> t_end, so the
    final distribution is ~one-hot; every `check_every` steps the ROUNDED assignment is
    scored and the best rounded solution is kept (starts from the rounded start, so the
    result is never worse than the start under this model — the relaxation gap of
    expected-demand overflow cannot leak into the output)."""
    th = logits0.clone().requires_grad_(True)

    def hard_of(t):
        return torch.nn.functional.one_hot(obj.probs(t).argmax(1), obj.K).float() * 50.0
    with torch.no_grad():
        C0 = obj.terms(logits0)["Cn"].detach()
        s0 = float(obj.score(hard_of(logits0), C0)[0])
    opt = torch.optim.Adam([th], lr=lr)
    best = (s0, hard_of(logits0).clone())
    for it in range(steps):
        T = t_end ** (it / max(1, steps - 1))
        s, t = obj.score(th / T, C0)
        opt.zero_grad()
        s.backward()
        opt.step()
        if it % check_every == 0 or it == steps - 1:
            with torch.no_grad():
                h = hard_of(th)
                sh = float(obj.score(h, C0)[0])
            if sh < best[0]:
                best = (sh, h.clone())
        if it % log_every == 0 or it == steps - 1:
            log(f"  step {it:4d} T {T:.3f} relaxed {float(s):+.4f} (wns {float(t['wns']):+.4f} "
                f"tns {float(t['tns']):+.1f} tof {float(t['tof']):.4g}) best rounded {best[0]:+.4f}")
    log(f"  kept rounded {best[0]:+.4f} vs start {s0:+.4f}")
    return best[1], best[0], s0


# ─────────────────────────────── write ───────────────────────────────

def write(P, logits, out_path):
    """round -> wires/vias for the chosen nets, everything else as read."""
    nets = read_route25(P["route_path"])
    choice = logits.argmax(1).cpu().numpy()
    z_of = P["cand"][np.arange(len(choice)), choice]
    by_net = defaultdict(list)
    for r, n in enumerate(P["run_net"]):
        by_net[int(n)].append(r)
    new = {}
    for n, runs in by_net.items():
        wires = defaultdict(list)
        node_layers = defaultdict(set)
        for r in runs:
            z = int(z_of[r])
            d, f, lo, hi = P["run_dir"][r], P["run_fixed"][r], P["run_lo"][r], P["run_hi"][r]
            wires[(z, int(f))].append((int(lo), int(hi)))
            a = (int(lo), int(f)) if d == 0 else (int(f), int(lo))
            b = (int(hi), int(f)) if d == 0 else (int(f), int(hi))
            node_layers[a].add(z)
            node_layers[b].add(z)
        for xy, (lo, hi) in P["net_nodes"][n].items():
            node_layers[xy].update((lo, hi))
        buf = []
        for (z, f), ivs in wires.items():
            for lo, hi in ivs:
                if P["dirs"][z] == 0:
                    buf.append(RE.seg_line(lo, f, z, hi, f, z, "ispd25").strip())
                else:
                    buf.append(RE.seg_line(f, lo, z, f, hi, z, "ispd25").strip())
        for (x, y), zs in node_layers.items():
            for z in range(min(zs), max(zs)):
                buf.append(RE.seg_line(x, y, z, x, y, z + 1, "ispd25").strip())
        new[P["net_route_idx"][n]] = buf
    with open(out_path, "w") as fh:
        for i, (name, _, lines) in enumerate(nets):
            fh.write(f"{name}\n(\n")
            for s in new.get(i, lines):
                fh.write(s + "\n")
            fh.write(")\n")
    log(f"wrote {out_path}: {len(new):,} nets re-assigned")


def calibrate_kp(P, ppa_path, pin_frac=0.5):
    """kP = P_switch / (C_wire_total * (1 + pin_frac)) from the base route + its PPA report."""
    from ppa_la import switching_power, route_wire_cap
    nets = read_route25(P["route_path"])

    class _LA:                                     # route_wire_cap needs .um and .C
        um = P["um"]
        C = P["Cz"]
    cw = route_wire_cap(nets, _LA)
    return switching_power(ppa_path) / (cw * (1.0 + pin_frac))


# ─────────────────────────────── self test ───────────────────────────────

def self_test():
    ok = True
    torch.manual_seed(0)
    # tiny synthetic problem: 2 runs of one net meeting at a junction, a pin at each end
    P = {"case": 1, "L": 10, "K": 5, "run_net": np.array([0, 0]),
         "run_len": np.array([20, 20], np.float32), "run_cur": np.array([0, 0]),
         "run_p0": np.array([0, 1]), "run_p1": np.array([1, 2]),
         "cand": np.array([[2, 4, 6, 8, -1], [1, 3, 5, 7, 9]]),
         "ent_rk": np.array([], np.int64), "ent_cell": np.array([], np.int64),
         "d0": np.zeros(0, np.float32), "cap": np.zeros(0, np.float32),
         "u": np.zeros(0, np.float32), "pt_lo": np.array([0, 10, 0]), "pt_hi": np.array([0, -1, 0]),
         "p_net": np.array([0]), "p_path": np.array([0]), "slack0": np.array([-0.2], np.float32),
         "Cz": np.array([RC[f"metal{z+1}"][1] for z in range(10)], np.float32), "um": 2.1}
    Rz = np.array([RC[f"metal{z+1}"][0] for z in range(10)])
    Cz = P["Cz"].astype(np.float64)
    l = 20 * 2.1
    P["dcoef"] = np.array([[l * (Rz[z] * 5.0 + Cz[z] * 2.0) / 1000 + Rz[z] * Cz[z] * l * l / 2000 if z >= 0 else 0.0
                            for z in row] for row in P["cand"]], np.float32)
    obj = Objective(P, a_ns_per_ff=0.0, kP=0.0)
    # 1) expected vias: runs on z=2 (x) and z=1 (y), pins on z=0 at both ends
    hard = torch.full((2, 5), NEG)
    hard[0, 0] = 50.0
    hard[1, 0] = 50.0
    t = obj.terms(hard)
    good = abs(float(t["vias"]) - (2 + 1 + 1)) < 1e-3       # pin0: 0->2, junction: 1..2, pin2: 0->1
    ok &= good
    print(f"  expected vias for a hard assignment = {float(t['vias']):.3f} (want 4): {'ok' if good else 'FAIL'}")
    # 2) optimizing the timing term moves both runs to the lowest-C legal layers
    th, sh, s0 = optimize(obj, current_logits(P), steps=200, lr=0.2, log_every=1000)
    zc = P["cand"][[0, 1], th.argmax(1).numpy()]
    good = zc[0] == 8 and zc[1] == 9 and sh < s0
    ok &= good
    print(f"  timing-only optimum -> metal{zc[0]+1}/metal{zc[1]+1} (want metal9/metal10), "
          f"score {s0:+.4f} -> {sh:+.4f}: {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route")
    ap.add_argument("--crit")
    ap.add_argument("--ppa")
    ap.add_argument("--out")
    ap.add_argument("--max_nets", type=int, default=6000)
    ap.add_argument("--extra_power_nets", type=int, default=0)
    ap.add_argument("--a", type=float, default=0.0, help="extra ns per fF on net C (default 0: the per-run Elmore proxy carries timing)")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--init", default="current", help="current | uniform | gnn:<ckpt>")
    ap.add_argument("--save_problem", default=None, help="write the built problem (.pt) for GNN training")
    ap.add_argument("--net_act", default=None, help="per-net switching activity (net_activity.tcl)")
    ap.add_argument("--vdd", type=float, default=1.1)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    P = build(a.design, a.suite, a.route, a.crit, a.max_nets, a.extra_power_nets)
    if len(P["run_net"]) == 0:                     # e.g. timing already met: nothing to optimise
        import shutil
        shutil.copyfile(a.route, a.out)
        log(f"no critical nets to re-assign; wrote {a.out} unchanged")
        return 0
    kP = calibrate_kp(P, a.ppa) if a.ppa else 0.0
    if a.net_act:                                  # per-net power slope 0.5 Vdd^2 density
        dens = {}
        with open(a.net_act) as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2:
                    if len(f) >= 3 and f[2] in ("input", "clock"):   # port-driven: no reported power
                        continue
                    try:
                        dens[f[0].replace("\\", "")] = float(f[1])
                    except ValueError:
                        pass
        P["kPn"] = np.array([0.5 * a.vdd ** 2 * 1e-15 * dens.get(n.replace("\\", ""), 0.0)
                             for n in P["net_names"]], np.float32)
        log(f"per-net activity: {int((P['kPn'] > 0).sum()):,} of {len(P['kPn']):,} critical nets toggle")
    obj = Objective(P, a.a, kP, a.tau)
    if a.save_problem:
        keep = {k: v for k, v in P.items() if k not in ("net_nodes", "pt_xy")}
        keep.update({"kP": kP, "a": a.a, "tau": a.tau})
        tmp = f"{a.save_problem}.tmp.{os.getpid()}"   # atomic: concurrent runs on the same base
        torch.save(keep, tmp)                           # once interleaved writes into one file
        os.replace(tmp, a.save_problem)
    if a.init == "current":
        th0 = current_logits(P)
    elif a.init == "uniform":
        th0 = torch.zeros(len(P["run_net"]), P["K"])
    else:
        from dtla_gnn import predict_logits
        th0 = predict_logits(a.init.split(":", 1)[1], P, obj)
    log(f"kP {kP:.3e} W/fF, a {a.a} ns/fF, tau {a.tau}, init {a.init}")
    th, sh, s0 = optimize(obj, th0, a.steps, a.lr)
    write(P, th, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
