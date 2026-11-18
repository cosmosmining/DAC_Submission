#!/usr/bin/env python3
"""fast_power.py — switching activity and route-dependent switching power for the fast mode (no OpenSTA).

A vectorized port of OpenSTA's activity propagation (power/Power.cc), on fast_sta.py's netlist:
  * seeds: input ports (not clocks) 0.1 transitions per clock cycle, duty 0.5 (Power::input_activity_); clock-network
    pins 2.0 per cycle, duty = (fall - rise) / period; constants (set_case_analysis) 0
  * combinational outputs: duty = P(f) (BDD duty with independent inputs), activity = sum_i act_i * P(df/dx_i)
    (Boolean difference), from the Liberty `function` (evaluated through truth tables, all instances of a cell type at
    once); outputs without a function (macros) stay unknown = 0
  * registers: Q activity = activity of the ff/latch data function, capped at 1.0 per cycle for flip-flops (not
    latches), QN duty = 1 - duty; register passes repeat until nothing changes (Power::ensureActivities)
  * density = activity / period of the pin's clock (the clock of its arrivals; unclocked logic: / 1 s, i.e. ~0,
    Power::findClkedActivity)
Switching power = sum over instance output pins of 0.5 * C_load * V^2 * density (Power::findSwitchingPower); only
C_load depends on the route, so a route change costs dP = 0.5 V^2 sum_n density_n dC_n.

  python fast_power.py --design ariane --suite visible --route R.norm.route [--act OpenSTA.act]  (check vs OpenSTA)
"""
import argparse
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import fast_sta as FS                      # NEW (this agent)
import fast_timer as FT                    # READ-ONLY reuse (Liberty group parser)

VDD = 1.1


def log(m):
    print(f"[fast_power {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════
#  Liberty boolean functions
# ════════════════════════════════════════════════════════════════════════════════════
_TOK = re.compile(r"\s*([A-Za-z_][\w\[\]\.]*|[01]|[!'&*|+^()])")


def parse_func(s):
    """Liberty function -> nested tuples ('var', name) | ('const', v) | ('not', e) | ('and'|'or'|'xor', a, b).
    Precedence (high to low): ' !, ^, & * (and juxtaposition), | +."""
    toks = _TOK.findall(s.replace('"', ""))
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = toks[pos[0]]
        pos[0] += 1
        return t

    def atom():
        t = take()
        if t == "!":
            e = ("not", atom())
        elif t == "(":
            e = orx()
            if peek() == ")":
                take()
        elif t in ("0", "1"):
            e = ("const", int(t))
        else:
            e = ("var", t)
        while peek() == "'":
            take()
            e = ("not", e)
        return e

    def xorx():
        e = atom()
        while peek() == "^":
            take()
            e = ("xor", e, atom())
        return e

    def andx():
        e = xorx()
        while True:
            t = peek()
            if t in ("&", "*"):
                take()
                e = ("and", e, xorx())
            elif t is not None and t not in ("|", "+", ")", "^", "'"):
                e = ("and", e, xorx())          # juxtaposition
            else:
                return e

    def orx():
        e = andx()
        while peek() in ("|", "+"):
            take()
            e = ("or", e, andx())
        return e
    return orx() if toks else ("const", 0)


def func_vars(e, out=None):
    out = [] if out is None else out
    if e[0] == "var":
        if e[1] not in out:
            out.append(e[1])
    elif e[0] != "const":
        for k in e[1:]:
            func_vars(k, out)
    return out


def eval_func(e, env):
    k = e[0]
    if k == "var":
        return env[e[1]]
    if k == "const":
        return np.full(len(next(iter(env.values()))) if env else 1, bool(e[1]))
    if k == "not":
        return ~eval_func(e[1], env)
    a, b = eval_func(e[1], env), eval_func(e[2], env)
    return (a & b) if k == "and" else (a | b) if k == "or" else (a ^ b)


def truth_table(e, names):
    n = len(names)
    idx = np.arange(2 ** n)
    env = {nm: ((idx >> i) & 1).astype(bool) for i, nm in enumerate(names)}
    if not names:
        return np.array([bool(e[1])]) if e[0] == "const" else np.array([False])
    return eval_func(e, env)


class FuncLib:
    """per cell: output pin functions and sequential groups (design-independent; parsed once, cached)."""
    VERSION = 1

    def __init__(self):
        self.out = {}          # cell -> {pin: (expr, var names, truth table)}
        self.seq = {}          # cell -> [(kind 'ff'|'latch', IQ, IQN, data expr, data vars, truth table)]

    @staticmethod
    def load(lib_dir=os.path.join(FS.KIT, "NanGate45/lib"), cache=None):
        import pickle
        files = sorted(os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".lib"))
        stamp = (FuncLib.VERSION, [(f, os.path.getmtime(f)) for f in files])
        cache = cache or os.path.join(ROOT, "experiments/fastmode/func_lib.pkl")
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as fh:
                    st, fl = pickle.load(fh)
                if st == stamp:
                    return fl
            except Exception:
                pass
        fl = FuncLib()
        for f in files:
            fl._add(FT._parse_groups(open(f).read()))
        try:
            tmp = f"{cache}.tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump((stamp, fl), fh)
            os.replace(tmp, cache)
        except OSError:
            pass
        return fl

    def _add(self, root):
        for libg in root.kids:
            if libg.name != "library":
                continue
            for cg in libg.kids:
                if cg.name != "cell" or not cg.args:
                    continue
                name = cg.args[0]
                outs, seqs = {}, []
                for g in cg.kids:
                    if g.name == "pin" and g.args and g.attrs.get("direction") in ("output", "inout") \
                            and g.attrs.get("function"):
                        e = parse_func(g.attrs["function"])
                        vs = func_vars(e)
                        for nm in g.args:
                            outs[nm] = (e, vs, truth_table(e, vs) if len(vs) <= 10 else None)
                    elif g.name in ("ff", "latch") and len(g.args) >= 1:
                        data = g.attrs.get("next_state" if g.name == "ff" else "data_in")
                        if data:
                            e = parse_func(data)
                            vs = func_vars(e)
                            seqs.append((g.name, g.args[0], g.args[1] if len(g.args) > 1 else None, e, vs,
                                         truth_table(e, vs) if len(vs) <= 10 else None))
                if outs:
                    self.out[name] = outs
                if seqs:
                    self.seq[name] = seqs


def _duty_act(T, names, getd, geta):
    """vectorized BDD duty / Boolean-difference activity of a truth table over (N instances x n vars)."""
    n = len(names)
    d = np.stack([getd(v) for v in names], 1) if n else np.zeros((0, 0))
    a = np.stack([geta(v) for v in names], 1) if n else np.zeros((0, 0))
    N = d.shape[0]
    idx = np.arange(2 ** n)
    bits = ((idx[None, :] >> np.arange(n)[:, None]) & 1).astype(bool)          # [n, 2^n]
    Tf = T.astype(float)
    P = np.ones((N, 2 ** n))
    for i in range(n):
        P *= np.where(bits[i][None, :], d[:, i:i + 1], 1.0 - d[:, i:i + 1])
    duty = P @ Tf
    act = np.zeros(N)
    for i in range(n):
        lo = idx[~bits[i]]
        diff = (T[lo] ^ T[lo | (1 << i)]).astype(float)
        Pm = np.ones((N, len(lo)))
        for j in range(n):
            if j != i:
                Pm *= np.where(bits[j][lo][None, :], d[:, j:j + 1], 1.0 - d[:, j:j + 1])
        act += a[:, i] * (Pm @ diff)
    return duty, act


# ════════════════════════════════════════════════════════════════════════════════════
#  activity propagation on fast_sta's netlist
# ════════════════════════════════════════════════════════════════════════════════════
class Activity:
    def __init__(self, D, fl=None, max_passes=50, tol=1e-4, verbose=True):
        t0 = time.time()
        self.D = D
        fl = fl or FuncLib.load()
        P = D.P
        act = np.zeros(P)
        duty = np.zeros(P)
        known = np.zeros(P, bool)
        cn, const = D.cnet, D.const
        # seeds
        clk = D.clk_of
        names = D.clk_names
        cks = [D.sdc["clocks"][n] for n in names]
        cduty = np.array([(c["wave"][1] - c["wave"][0]) / c["period"] for c in cks] + [0.5])
        inp = D.is_port & D.pin_isdrvdir & ~cn & (const < 0)
        act[inp], duty[inp], known[inp] = 0.1, 0.5, True
        act[cn], duty[cn], known[cn] = 2.0, cduty[clk[cn]], True
        cm = const >= 0
        act[cm], duty[cm], known[cm] = 0.0, const[cm].astype(float), True
        # per (cell type, output slot): instances, input slots of the function
        groups = []                      # (level, ct, out slot, [var slots or seq], kind)
        lev = D.lev
        for c, (m, sl) in enumerate(zip(D.cmods, D.slots)):
            if m is None or not len(D.mats[c]):
                continue
            outs = fl.out.get(m.name, {})
            seqs = fl.seq.get(m.name, [])
            M = D.mats[c]
            for pn, s in sl.items():
                f = outs.get(pn) or (outs.get(pn.split("[")[0]) if "[" in pn else None)
                if f is None or f[2] is None:
                    continue
                e, vs, T = f
                okp = M[:, s] >= 0
                if not okp.any():
                    continue
                groups.append((c, s, vs, T, np.nonzero(okp)[0]))
        # levelize: per level, the output-pin rows of every group and the load pins (copied from their drivers)
        maxl = int(lev.max(initial=0))
        self.lv_groups = [[] for _ in range(maxl + 1)]
        for gi, (c, s, vs, T, rows) in enumerate(groups):
            p = D.mats[c][rows, s]
            keep = ~cn[p] & (const[p] < 0)
            rows, p = rows[keep], p[keep]
            lv = lev[p]
            o = np.argsort(lv, kind="stable")
            rows, lv = rows[o], lv[o]
            b = np.r_[0, np.nonzero(np.diff(lv))[0] + 1, len(lv)]
            for k in range(len(b) - 1):
                if b[k + 1] > b[k]:
                    self.lv_groups[lv[b[k]]].append((gi, rows[b[k]:b[k + 1]]))
        ld = np.nonzero(D.pin_is_load & ~cn & (const < 0))[0]
        dr = D.net_drv[D.pin_net[ld]]
        o = np.argsort(lev[ld], kind="stable")
        self.ld, self.dr = ld[o], dr[o]
        self.ld_lo = np.searchsorted(lev[self.ld], np.arange(maxl + 2))
        self.maxl = maxl
        self.groups = groups
        self.fl = fl
        self.seq_groups = []
        for c, m in enumerate(D.cmods):
            if m is None or not len(D.mats[c]):
                continue
            for sq in fl.seq.get(m.name, []):
                self.seq_groups.append((c, sq))
        self.act, self.duty, self.known = act, duty, known
        # internal register state (IQ / IQN) per (ct, name) -> arrays over instances of that type
        self.iq = {}
        self._run(max_passes, tol)
        self.t = time.time() - t0
        if verbose:
            log(f"activities: {int(self.known.sum()):,} of {P:,} pins known, {self.passes} passes, "
                f"{self.t:.1f}s; mean activity of driver pins {self.act[D.pin_is_drv].mean():.4f}/cycle")

    def _pinvals(self, c, rows, name):
        """duty / activity of variable `name` of instances `rows` of cell type c (input pin or internal IQ)."""
        D = self.D
        sl = D.slots[c]
        s = sl.get(name)
        if s is not None:
            p = D.mats[c][rows, s]
            ok = p >= 0
            dv = np.where(ok, self.duty[np.maximum(p, 0)], 0.0)
            av = np.where(ok, self.act[np.maximum(p, 0)], 0.0)
            # unknown pins: activity 0, duty 0 (Power::findActivity)
            kn = np.where(ok, self.known[np.maximum(p, 0)], False)
            return np.where(kn, dv, 0.0), np.where(kn, av, 0.0)
        st = self.iq.get((c, name))
        if st is not None:
            return st[0][rows], st[1][rows]
        return np.zeros(len(rows)), np.zeros(len(rows))

    def _eval_outputs(self, todo):
        D = self.D
        changed = 0.0
        for gi, rows in todo:
            c, s, vs, T, _ = self.groups[gi]
            p = D.mats[c][rows, s]
            cache = {}

            def g(v, k):
                if v not in cache:
                    cache[v] = self._pinvals(c, rows, v)
                return cache[v][k]
            du, ac = _duty_act(T, vs, lambda v: g(v, 0), lambda v: g(v, 1))
            ch = np.max(np.abs(ac - self.act[p]), initial=0.0)
            changed = max(changed, ch, np.max(np.abs(du - self.duty[p]), initial=0.0))
            self.act[p], self.duty[p], self.known[p] = ac, du, True
        return changed

    def _loads(self, lv):
        a, b = self.ld_lo[lv], self.ld_lo[lv + 1]
        if b > a:
            ld, dr = self.ld[a:b], self.dr[a:b]
            self.act[ld], self.duty[ld], self.known[ld] = self.act[dr], self.duty[dr], self.known[dr]

    def _seed_regs(self):
        D = self.D
        ch = 0.0
        for c, (kind, iq, iqn, e, vs, T) in self.seq_groups:
            if T is None:
                continue
            rows = np.arange(len(D.mats[c]))
            cache = {}

            def g(v, k):
                if v not in cache:
                    cache[v] = self._pinvals(c, rows, v)
                return cache[v][k]
            du, ac = _duty_act(T, vs, lambda v: g(v, 0), lambda v: g(v, 1))
            if kind == "ff":
                ac = np.minimum(ac, 1.0)
            old = self.iq.get((c, iq))
            if old is not None:
                ch = max(ch, np.max(np.abs(old[1] - ac), initial=0.0), np.max(np.abs(old[0] - du), initial=0.0))
            else:
                ch = max(ch, np.max(ac, initial=0.0), 1.0)
            self.iq[(c, iq)] = (du, ac)
            if iqn:
                self.iq[(c, iqn)] = (1.0 - du, ac)
        return ch

    def _run(self, max_passes, tol):
        """levelized combinational propagation (loads copy their driver, outputs evaluate their function), then
        register D->Q seeding; repeated until the register activities stop changing (Power::ensureActivities)."""
        self.passes = 0
        for p in range(max_passes):
            self.passes = p + 1
            for lv in range(self.maxl + 1):
                self._loads(lv)
                if self.lv_groups[lv]:
                    self._eval_outputs(self.lv_groups[lv])
            if self._seed_regs() < tol and p > 0:
                break

    # ------------------------------------------------------------------ densities and power
    def densities(self, r_at=None):
        """per pin transitions/ns = activity / period of the pin's clock (min-period clock among its arrival classes;
        clock-network pins: their clock; unclocked: activity per second ~ 0)."""
        D = self.D
        names = D.clk_names
        per = np.array([D.sdc["clocks"][n]["period"] for n in names] + [1e9])
        pp = np.full(D.P, 1e9)                                      # 1 s in ns: unclocked
        if r_at is not None:
            for k in range(D.K):
                c = k // 2
                if c >= len(names):
                    continue
                has = (r_at[:, k, :] > FS.NEG / 2).any(1)
                pp = np.where(has, np.minimum(pp, per[c]), pp)
        cn = D.cnet
        pp[cn] = per[np.where(D.clk_of[cn] >= 0, D.clk_of[cn], len(names))]
        return self.act / pp

    def net_density(self, dens):
        """density of each net's driver pin (0 for nets driven by top-level ports: report_power sums instances)."""
        D = self.D
        nd = np.zeros(len(D.net_names))
        ok = (D.net_drv >= 0)
        drv = D.net_drv[ok]
        nd[ok] = np.where(D.is_port[drv], 0.0, dens[drv])
        return nd


def switching_power(net_dens, ctot):
    """W: 0.5 * C * V^2 * density (C fF, density 1/ns)."""
    return float(0.5 * VDD * VDD * np.sum(net_dens * ctot) * 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route", required=True)
    ap.add_argument("--act", default=None, help="OpenSTA net_activity.tcl dump to compare with")
    ap.add_argument("--ppa", default=None, help="OpenROAD PPA report (report_power) to compare with")
    a = ap.parse_args()
    D = FS.StaDesign(a.design, a.suite)
    r = FS.StaTimer(D).analyze(FS.read_route(D, a.route), rat=False, keep=True)
    A = Activity(D)
    dens = A.densities(r["AT"])
    nd = A.net_density(dens)
    ps = switching_power(nd, r["rc"].ctot.max(1))
    log(f"switching power (ours) {ps:.6e} W; nets with density > 0: {int((nd > 0).sum()):,}")
    if a.ppa and os.path.exists(a.ppa):
        t = open(a.ppa).read()
        m = re.search(r"^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", t, re.M)
        if m:
            log(f"OpenROAD report_power Total: internal {m.group(1)} switching {m.group(2)} leakage {m.group(3)} "
                f"total {m.group(4)} W")
    if a.act and os.path.exists(a.act):
        ref = {}
        for line in open(a.act):
            f = line.split()
            if len(f) >= 2:
                try:
                    ref[FS._key(f[0])] = (float(f[1]), f[2] if len(f) > 2 else "")
                except ValueError:
                    pass
        ours, theirs = [], []
        for nm, (v, org) in ref.items():
            i = D.net_index.get(nm)
            if i is None or org in ("input", "clock"):
                continue
            ours.append(nd[i] * 1e9)
            theirs.append(v)
        o, t_ = np.array(ours), np.array(theirs)
        rel = np.abs(o - t_) / np.maximum(np.abs(t_), 1e-9 * max(t_.max(initial=1.0), 1.0))
        log(f"vs OpenSTA activities ({len(o):,} nets): r {np.corrcoef(o, t_)[0, 1]:.4f}, sum ours {o.sum():.4e} "
            f"theirs {t_.sum():.4e} /s, exact (1%) {np.mean(rel < 0.01) * 100:.1f}% of nets, zero in both "
            f"{np.mean((o == 0) & (t_ == 0)) * 100:.1f}%")
        pw_ref = 0.5 * VDD * VDD * 1e-15 * sum(v * r['rc'].ctot.max(1)[D.net_index[nm]]
                                              for nm, (v, org) in ref.items()
                                              if nm in D.net_index and org not in ("input", "clock"))
        log(f"switching power with OpenSTA's activities and our C: {pw_ref:.6e} W")
    return 0


if __name__ == "__main__":
    sys.exit(main())
