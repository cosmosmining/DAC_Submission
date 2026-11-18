# fast_sta_v2.py = fast_sta.py with two OpenSTA-consistency fixes (2026-10-27), validated with `fast_sta.py calib`:
#   1. _root: a converged Newton iterate (fx == 0 or a step below tolerance) was flagged out-of-bracket and replaced by
#      the bracket midpoint, so dmp_load fell back to wire = Elmore, load slew = driver slew on huge nets;
#   2. extrapolated table slews are clipped at 0 (OpenSTA GateTableModel::gateDelay), which also triggers DmpPi's
#      Ceff = c1 + c2 fallback exactly as OpenSTA does.
#   bsg_chip blind instantgr_esc: est WNS 0.0000 -> -6.7900 (OpenROAD -6.8106), endpoint r 0.7842 -> 1.0000;
#   ariane / tile unchanged (r 1.0000, |bias| <= 0.003 ns). Used by fastmode4.py when FAST_STA_V2=1.
#!/usr/bin/env python3
"""fast_sta.py — INSTA-style full-graph static timing analysis of ISPD'25 global routes (CPU, vectorized numpy).

Why: fast_timer.py shifts the .net pre-routing slacks along ONE fixed worst-path forest, so after a route change it cannot
see another path becoming critical (ariane f1: estimated WNS -0.351 ns, OpenROAD -0.493 ns). This timer re-propagates
arrival and required times through the WHOLE gate-level timing graph for every route, like OpenSTA and NVIDIA INSTA
(levelized full-graph propagation). INSTA itself cannot be used here: it propagates per-arc delays dumped from a signoff
timer (PrimeTime attributes delay_max_rise / variation_delay_*, endpoint required times) and has no delay calculator, so
the effect of a route change on the arc delays is exactly what it cannot compute (see an internal note, section f3).
This file computes the arc delays itself, from the route, as OpenROAD's evaluation does:

  netlist  DEF COMPONENTS + NETS (every net: the .net omits 10-20 % of them, mostly one-GCell nets), Liberty arcs with
           rise/fall and unateness (positive / negative / non-unate, rising/falling-edge launch arcs, latch D->Q),
           setup / recovery checks, preset/clear arcs off (OpenSTA default), inferred clock-gating checks
           (set_clock_gating_check -setup 0 on AND/NAND = rising, OR/NOR = falling), set_case_analysis constants through
           buffers / inverters / simple gates (+ the `when` of timing checks), Liberty units (fakeram libs are in pF)
  clocks   ideal (the SDCs have no set_propagated_clock): create_clock period / waveform, set_clock_latency,
           set_clock_uncertainty (setup), set_clock_transition, set_input_delay / set_output_delay (-clock_fall, -max),
           set_false_path -from ports and -from clocks -to clocks; clock sense through the clock network (inverters);
           one arrival class per (launch clock, edge) -> exact multi-clock setup relations (bsg_chip has 8 clocks)
  wires    the RC network of every routed net exactly as OpenROAD grt MakeWireParasitics builds it from the route:
           segment ends are the nodes, a stacked via is ONE resistor (cut resistance of its lowest cut), a pin hangs on
           the GCell node of the layer above its access layer (+ that cut) or else of its own layer; setRC.tcl layer
           R/C, tech LEF cut R; driving-point moments y1..y3 with rise / fall pin caps -> pi model, Elmore per load
           (OpenSTA ReduceToPiElmore); nets without a route: lumped pin caps (OpenSTA LumpedCap)
  delays   OpenSTA dmp_ceff_elmore, ported and vectorized: gateModelRd, DmpCap / DmpZeroC2 / DmpPi selection,
           Newton-Raphson for (t0, dt, Ceff), driver slew from the Vo waveform, load delay / slew from the driver waveform
           through the Elmore pole (Vl); input ports: dspf wire delay; latch D->Q (time borrowing by max with G->Q)
  outputs  WNS / TNS (report_wns / report_tns: worst slack per endpoint), per-endpoint slack, per-pin slack (RAT - AT),
           the worst path, and an INSTA-style TNS gradient per net: the number of violating endpoints whose worst path
           runs through the net (hard-max backward pass), which the fast mode uses to rank nets for promotion

  python fast_sta.py --design ariane --suite visible --route R.norm.route
  python fast_sta.py calib --design ariane --suite visible --tags instantgr instantgr_tr1 ... --csv out.csv [--detail 2]
  python fast_sta.py --self_test
"""
import argparse
import math
import os
import pickle
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import fast_timer as FT                 # READ-ONLY reuse: Liberty group parser, .net parser, route reader, tech R/C

KIT = FT.KIT
GRID = FT.GRID
UM_PER_GC = FT.UM_PER_GC
NEG = -1e30
BIG = 1e30
STUB_UM = float(os.environ.get("FS_STUB_UM", "1.05"))   # pin -> GCell-centre wire (grt makeParasiticsToPin), um
_key = FT._key


def log(m):
    print(f"[fast_sta {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════
#  Liberty (rise/fall, unateness, units)
# ════════════════════════════════════════════════════════════════════════════════════
class Tables:
    """every NLDM table in one padded store; lookups are vectorized over (table id, x, y).
    axis a = input / related-pin transition, axis b = load / constrained-pin transition (normalized at load time)."""

    def __init__(self):
        self.a, self.b, self.v = [], [], []

    def add(self, a, b, v):
        self.a.append(np.asarray(a, float))
        self.b.append(np.asarray(b, float))
        self.v.append(np.asarray(v, float).reshape(len(a), len(b)))
        return len(self.v) - 1

    def freeze(self):
        T = len(self.v)
        M = max([2] + [max(len(a), len(b)) for a, b in zip(self.a, self.b)])
        A = np.full((T, M), np.inf)
        B = np.full((T, M), np.inf)
        V = np.zeros((T, M, M))
        NA = np.zeros(T, np.int64)
        NB = np.zeros(T, np.int64)
        for t in range(T):
            a, b, v = self.a[t], self.b[t], self.v[t]
            A[t, :len(a)] = a
            B[t, :len(b)] = b
            V[t, :len(a), :len(b)] = v
            NA[t], NB[t] = len(a), len(b)
        self.A, self.B, self.NA, self.NB, self.M, self.Vf = A, B, NA, NB, M, V.reshape(-1)
        self.a = self.b = self.v = None

    def __call__(self, tid, x, y):
        """bilinear interpolation, linear extrapolation outside (OpenSTA Table2::findValue / findAxisIndex)."""
        tid = np.asarray(tid, np.int64).ravel()
        n = len(tid)
        x = np.broadcast_to(np.asarray(x, float), (n,))
        y = np.broadcast_to(np.asarray(y, float), (n,))
        if n == 0:
            return np.zeros(0)
        A, B = self.A[tid], self.B[tid]
        ka = np.minimum((x[:, None] > A[:, 1:]).sum(1), self.NA[tid] - 2)
        kb = np.minimum((y[:, None] > B[:, 1:]).sum(1), self.NB[tid] - 2)
        r = np.arange(n)
        a0, a1, b0, b1 = A[r, ka], A[r, ka + 1], B[r, kb], B[r, kb + 1]
        fa = (x - a0) / np.where(a1 != a0, a1 - a0, 1.0)
        fb = (y - b0) / np.where(b1 != b0, b1 - b0, 1.0)
        M = self.M
        base = tid * (M * M) + ka * M + kb
        V = self.Vf
        return ((1 - fa) * (1 - fb) * V[base] + (1 - fa) * fb * V[base + 1]
                + fa * (1 - fb) * V[base + M] + fa * fb * V[base + M + 1])


CHECK_KIND = {"setup_rising": 0, "setup_falling": 1, "recovery_rising": 2, "recovery_falling": 3}
SKIP_TT = {"preset", "clear", "three_state_disable", "three_state_disable_rise", "three_state_disable_fall"}
SKIP_TT_PREFIX = ("hold", "removal", "min_pulse", "minimum_period", "skew", "non_seq", "nochange", "max_clock")
SEQ_GROUPS = {"ff", "latch", "ff_bank", "latch_bank", "statetable", "memory"}
DELAY_A = ("input_net_transition", "input_transition_time")
DELAY_B = ("total_output_net_capacitance",)
CHECK_A = ("related_pin_transition",)
CHECK_B = ("constrained_pin_transition",)


def _time_scale(s):
    m = re.match(r'\s*"?\s*([\d.eE+-]+)\s*([a-zA-Z]+)', str(s))
    if not m:
        return 1.0
    return float(m.group(1)) * {"ns": 1.0, "ps": 1e-3, "us": 1e3, "fs": 1e-6}.get(m.group(2).lower(), 1.0)


def _cap_scale(args):
    if not isinstance(args, list) or len(args) < 2:
        return 1.0
    return float(args[0]) * {"ff": 1.0, "pf": 1e3, "nf": 1e6}.get(args[1].strip('"').lower(), 1.0)


def _func_type(f):
    """'buf' | 'inv' | 'and' | 'nand' | 'or' | 'nor' | None for a Liberty function string."""
    s = f.replace(" ", "").replace('"', "").replace("*", "&").replace("+", "|")

    def strip(x):
        while x.startswith("(") and x.endswith(")"):
            d, ok = 0, True
            for i, ch in enumerate(x[1:-1]):
                d += (ch == "(") - (ch == ")")
                if d < 0:
                    ok = False
                    break
            if not ok or d != 0:
                break
            x = x[1:-1]
        return x
    s = strip(s)
    neg = False
    if s.startswith("!"):
        t = strip(s[1:])
        if t != s[1:] or re.fullmatch(r"[\w\[\]]+", t):
            neg, s = True, t
    var = r"[A-Za-z_][\w\[\]]*"
    if re.fullmatch(var, s):
        return "inv" if neg else "buf"
    if "&" in s and "|" not in s and all(re.fullmatch(var, t) for t in s.split("&")):
        return "nand" if neg else "and"
    if "|" in s and "&" not in s and all(re.fullmatch(var, t) for t in s.split("|")):
        return "nor" if neg else "or"
    return None


class CellM:
    __slots__ = ("name", "lib", "pins", "arcs", "checks", "seq", "latch", "icg", "ftype", "inputs")

    def pin(self, pn):
        p = self.pins.get(pn)
        if p is None and "[" in pn:
            p = self.pins.get(pn.split("[")[0])
        return p

    def arcs_to(self, pn):
        a = self.arcs.get(pn)
        if a is None and "[" in pn:
            a = self.arcs.get(pn.split("[")[0])
        return a or []

    def checks_on(self, pn):
        a = self.checks.get(pn)
        if a is None and "[" in pn:
            a = self.checks.get(pn.split("[")[0])
        return a or []


class StaLib:
    """cells[name] -> CellM; libs[i] -> thresholds; tab = Tables (times in ns, caps in fF)."""
    VERSION = 3

    def __init__(self):
        self.cells, self.libs, self.tab = {}, [], Tables()

    @staticmethod
    def load(lib_dir=os.path.join(KIT, "NanGate45/lib"), cache=None):
        files = sorted(os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".lib"))
        stamp = (StaLib.VERSION, [(f, os.path.getmtime(f)) for f in files])
        cache = cache or os.path.join(ROOT, "experiments/fastmode/sta_lib.pkl")
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as fh:
                    st, lib = pickle.load(fh)
                if st == stamp:
                    return lib
            except Exception:
                pass
        lib = StaLib()
        for f in files:
            lib._add(FT._parse_groups(open(f).read()))
        lib.tab.freeze()
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = f"{cache}.tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump((stamp, lib), fh)
            os.replace(tmp, cache)
        except OSError:
            pass
        return lib

    def _add(self, root):
        for libg in root.kids:
            if libg.name != "library":
                continue
            a = libg.attrs
            ts = _time_scale(a.get("time_unit", "1ns"))
            cs = _cap_scale(a.get("capacitive_load_unit", ["1", "ff"]))

            def pct(k, d):
                try:
                    return float(a.get(k, d)) / 100.0
                except ValueError:
                    return d / 100.0
            th = {"vth": (pct("output_threshold_pct_rise", 50), pct("output_threshold_pct_fall", 50)),
                  "vin": (pct("input_threshold_pct_rise", 50), pct("input_threshold_pct_fall", 50)),
                  "vl": (pct("slew_lower_threshold_pct_rise", 20), pct("slew_lower_threshold_pct_fall", 20)),
                  "vh": (pct("slew_upper_threshold_pct_rise", 80), pct("slew_upper_threshold_pct_fall", 80)),
                  "der": float(a.get("slew_derate_from_library", 1.0)), "ts": ts, "cs": cs,
                  "name": libg.args[0] if libg.args else "?"}
            li = len(self.libs)
            self.libs.append(th)
            templates = {}
            for g in libg.kids:
                if g.name in ("lu_table_template", "power_lut_template") and g.args:
                    t = {k: g.attrs[k] for k in ("variable_1", "variable_2") if k in g.attrs}
                    for k in ("index_1", "index_2"):
                        if k in g.attrs:
                            t[k] = FT._floats(g.attrs[k])
                    templates[g.args[0]] = t
            for cg in libg.kids:
                if cg.name == "cell" and cg.args:
                    self._cell(cg, templates, li, ts, cs)

    def _table(self, g, templates, check, ts, cs):
        if g is None:
            return -1
        tm = templates.get(g.args[0] if g.args else "", {})
        i1 = FT._floats(g.attrs["index_1"]) if "index_1" in g.attrs else list(tm.get("index_1", []))
        i2 = FT._floats(g.attrs["index_2"]) if "index_2" in g.attrs else list(tm.get("index_2", []))
        vals = FT._floats(g.attrs.get("values", ["0"]))
        v1, v2 = tm.get("variable_1"), tm.get("variable_2")
        n1, n2 = max(len(i1), 1), max(len(i2), 1)
        if len(vals) == n1 * n2 and (i1 or n1 == 1):
            V = np.array(vals, float).reshape(n1, n2)
            if not i2:
                v2 = None
        elif len(vals) == 1:
            V, i1, i2, v1, v2 = np.array(vals, float).reshape(1, 1), [], [], None, None
        elif len(vals) == len(i1):
            V, i2, v2 = np.array(vals, float).reshape(-1, 1), [], None
        else:
            return -1
        want_a, want_b = (CHECK_A, CHECK_B) if check else (DELAY_A, DELAY_B)
        sc = lambda var, idx: [x * (cs if var == "total_output_net_capacitance" else ts) for x in idx]
        i1s, i2s = sc(v1, i1), sc(v2, i2)
        if (v1 in want_b) or (v2 in want_a and v1 not in want_a):
            A, B, V = i2s, i1s, V.T
        else:
            A, B = i1s, i2s
        V = V * ts
        if len(A) < 2:
            x0 = A[0] if A else 0.0
            A, V = [x0, x0 + 1.0], np.repeat(V[:1, :], 2, axis=0)
        if len(B) < 2:
            x0 = B[0] if B else 0.0
            B, V = [x0, x0 + 1.0], np.repeat(V[:, :1], 2, axis=1)
        return self.tab.add(A, B, V)

    def _cell(self, cg, templates, li, ts, cs):
        m = CellM()
        m.name, m.lib, m.pins, m.arcs, m.checks = cg.args[0], li, {}, {}, {}
        m.seq = any(k.name in SEQ_GROUPS for k in cg.kids)
        m.latch = any(k.name in ("latch", "latch_bank") for k in cg.kids)
        m.icg = "clock_gating_integrated_cell" in cg.attrs

        def fcap(v):
            try:
                return float(v) * cs
            except (TypeError, ValueError):
                return 0.0

        def one(pg, pn, inherit=None):
            src = dict(inherit.attrs) if inherit is not None else {}
            src.update(pg.attrs)
            c = src.get("capacitance")
            m.pins[pn] = (src.get("direction", "input"), fcap(src.get("rise_capacitance", c)),
                          fcap(src.get("fall_capacitance", c)), src.get("clock", "false") == "true",
                          src.get("function", ""))
            for tg in (k for k in pg.kids if k.name == "timing"):
                tt = tg.attrs.get("timing_type", "combinational")
                when = tg.attrs.get("when", "")
                tabs = {k.name: k for k in tg.kids}
                for rp in tg.attrs.get("related_pin", "").split():
                    if tt in CHECK_KIND:
                        trc = self._table(tabs.get("rise_constraint"), templates, True, ts, cs)
                        tfc = self._table(tabs.get("fall_constraint"), templates, True, ts, cs)
                        if trc >= 0 or tfc >= 0:
                            m.checks.setdefault(pn, []).append((rp, CHECK_KIND[tt], trc, tfc, when))
                    elif tt in SKIP_TT or tt.startswith(SKIP_TT_PREFIX):
                        continue
                    else:
                        kind = 1 if tt == "rising_edge" else 2 if tt == "falling_edge" else 0
                        tcr = self._table(tabs.get("cell_rise"), templates, False, ts, cs)
                        tcf = self._table(tabs.get("cell_fall"), templates, False, ts, cs)
                        trt = self._table(tabs.get("rise_transition"), templates, False, ts, cs)
                        tft = self._table(tabs.get("fall_transition"), templates, False, ts, cs)
                        if tt.endswith("_rise"):
                            tcf = -1
                        elif tt.endswith("_fall"):
                            tcr = -1
                        if tcr < 0 and tcf < 0:
                            continue
                        sense = tg.attrs.get("timing_sense", "non_unate")
                        sn = 0 if sense == "positive_unate" else 1 if sense == "negative_unate" else 2
                        m.arcs.setdefault(pn, []).append((rp, kind, sn, tcr, tcf, trt, tft, when))

        for pg in cg.kids:
            if pg.name == "pin" and pg.args:
                for nm in pg.args:
                    one(pg, nm)
            elif pg.name in ("bus", "bundle") and pg.args:
                one(pg, pg.args[0])                         # bus-level model + arcs ('addr_in', 'rd_out')
                for sp in (k for k in pg.kids if k.name == "pin"):
                    for nm in sp.args:
                        one(sp, nm, inherit=pg)             # explicit bit pins: own timing only
        if any(a[1] for arcs in m.arcs.values() for a in arcs):
            m.seq = True
        outs = [p for p, v in m.pins.items() if v[0] in ("output", "inout") and v[4]]
        m.inputs = [p for p, v in m.pins.items() if v[0] == "input"]
        m.ftype = _func_type(m.pins[outs[0]][4]) if (len(outs) == 1 and not m.seq) else None
        self.cells[m.name] = m


# ════════════════════════════════════════════════════════════════════════════════════
#  SDC (the subset the ISPD'25 SDCs use)
# ════════════════════════════════════════════════════════════════════════════════════
def _brace_list(s):
    return [_key(t.strip("{}")) for t in s.replace("{", " ").replace("}", " ").split()]


def _targets(line, kind):
    out = []
    for body in re.findall(rf"\[get_{kind}\s+(\{{[^}}]*\}}|[^\]]+)\]", line):
        out += _brace_list(body)
    return out


def parse_sdc(path):
    c = {"clocks": {}, "order": [], "in": [], "out": [], "fp_ports": set(), "fp_clk": [], "case": {}, "ideal": set()}
    lat, unc, tr = {}, {}, {}
    for raw in open(path):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        cmd = line.split()[0]
        if cmd == "create_clock":
            nm = re.search(r"-name\s+(\S+)", line)
            per = float(re.search(r"-period\s+(\S+)", line).group(1))
            wf = re.search(r"-waveform\s+\{([^}]*)\}", line)
            w = [float(v) for v in wf.group(1).split()] if wf else [0.0, per / 2]
            src = _targets(line, "ports")
            name = nm.group(1) if nm else (src[0] if src else f"clk{len(c['order'])}")
            c["clocks"][name] = {"period": per, "wave": (w[0], w[1]), "src": src}
            c["order"].append(name)
        elif cmd == "set_clock_latency":
            v = float(line.split()[1]) if re.match(r"^-?[\d.]", line.split()[1]) else float(
                re.search(r"\s(-?[\d.]+)\s", line).group(1))
            for k in _targets(line, "clocks"):
                lat[k] = lat.get(k, 0.0) + v if "-source" in line else v
        elif cmd == "set_clock_uncertainty":
            if "-hold" in line and "-setup" not in line:
                continue
            v = float(re.search(r"\s(-?[\d.]+)\s", line + " ").group(1))
            for k in _targets(line, "clocks"):
                unc[k] = v
        elif cmd == "set_clock_transition":
            if "-min" in line and "-max" not in line:
                continue
            v = float(re.search(r"\s(-?[\d.]+)\s", line + " ").group(1))
            rfs = [0] if "-rise" in line else [1] if "-fall" in line else [0, 1]
            for k in _targets(line, "clocks"):
                t = tr.setdefault(k, [0.0, 0.0])
                for rf in rfs:
                    t[rf] = v
        elif cmd in ("set_input_delay", "set_output_delay"):
            if "-min" in line and "-max" not in line:
                continue
            v = float(re.search(r"(?:^|\s)(-?[\d.]+(?:[eE][-+]?\d+)?)(?=\s)", line.split(None, 1)[1] + " ").group(1))
            clk = _targets(line, "clocks")
            for p in _targets(line, "ports"):
                c["in" if cmd == "set_input_delay" else "out"].append(
                    (p, clk[0] if clk else None, 1 if "-clock_fall" in line else 0, v))
        elif cmd == "set_false_path":
            fr = re.search(r"-from\s+(\[[^\]]*\])", line)
            to = re.search(r"-to\s+(\[[^\]]*\])", line)
            if fr and "get_ports" in fr.group(1) and not to:
                c["fp_ports"] |= set(_targets(fr.group(1), "ports"))
            elif fr and to and "get_clocks" in fr.group(1) and "get_clocks" in to.group(1):
                c["fp_clk"].append((set(_targets(fr.group(1), "clocks")), set(_targets(to.group(1), "clocks"))))
        elif cmd == "set_case_analysis":
            v = line.split()[1]
            for p in _targets(line, "ports"):
                c["case"][p] = 1 if v in ("1", "one", "rising") else 0
        elif cmd == "set_ideal_network":
            c["ideal"] |= set(_targets(line, "ports"))
    for k, v in c["clocks"].items():
        v["lat"], v["unc"], v["tr"] = lat.get(k, 0.0), unc.get(k, 0.0), tuple(tr.get(k, [0.0, 0.0]))
    return c


def setup_relation(per_l, t_l, per_c, t_c, max_edges=4000):
    """most restrictive setup relation: min over launch edges (one common period) of the first capture edge strictly
    after the launch edge minus the launch edge (OpenSTA default for unrelated periods)."""
    q = 1e-4
    a, b = int(round(per_l / q)), int(round(per_c / q))
    lcm = a * b // math.gcd(a, b)
    n = min(lcm // a, max_edges)
    tl = t_l + np.arange(n) * per_l
    k = np.floor((tl - t_c) / per_c + 1e-9) + 1.0
    tc = t_c + k * per_c
    tc = np.where(tc <= tl + 1e-9, tc + per_c, tc)
    return float(np.min(tc - tl))


# ════════════════════════════════════════════════════════════════════════════════════
#  design: DEF netlist, .net access points, timing graph (route-independent, built once)
# ════════════════════════════════════════════════════════════════════════════════════
def parse_def_netlist(path):
    text = "\n" + open(path).read()

    def section(name):
        a = text.find(f"\n{name} ")
        if a < 0:
            return ""
        b = text.find(f"\nEND {name}", a)
        return text[a:b if b > 0 else len(text)]
    comps = {}
    for m in re.finditer(r"^-\s+(\S+)\s+(\S+)", section("COMPONENTS"), re.M):
        comps[_key(m.group(1))] = m.group(2)
    ports = {}
    for m in re.finditer(r"^-\s+(\S+)(.*?);", section("PINS"), re.M | re.S):
        d = re.search(r"DIRECTION\s+(\w+)", m.group(2))
        ports[_key(m.group(1))] = d.group(1) if d else "INPUT"
    names, cn_net, cn_inst, cn_pin, power = [], [], [], [], set()
    for m in re.finditer(r"^-\s+(\S+)|\(\s*(\S+)\s+(\S+)\s*\)|\+\s*USE\s+(\w+)", section("NETS"), re.M):
        if m.group(1) is not None:
            names.append(_key(m.group(1)))
        elif m.group(4) is not None:
            if m.group(4) in ("POWER", "GROUND"):
                power.add(len(names) - 1)
        elif m.group(2) != "*" and names:
            cn_net.append(len(names) - 1)
            cn_inst.append(m.group(2))
            cn_pin.append(m.group(3))
    del text
    return comps, ports, names, cn_net, cn_inst, cn_pin, power


class StaDesign:
    """netlist + timing graph + clocks of one design (route-independent; built once per process)."""

    def __init__(self, design, suite="visible", lib=None, kit=KIT, verbose=True):
        t0 = time.time()
        self.design, self.suite = design, suite
        d = os.path.join(kit, suite, design)
        self.lib = lib or StaLib.load(os.path.join(kit, "NanGate45/lib"))
        self.layer_rc, self.via_r = FT.read_tech(kit)                 # kOhm/um, fF/um ; kOhm per cut
        comps, ports, net_names, cn_net, cn_inst, cn_pin, power = parse_def_netlist(os.path.join(d, design + ".def"))
        t1 = time.time()
        self.net_names = net_names
        self.net_index = {nm: i for i, nm in enumerate(net_names)}
        nn = len(net_names)
        inst_names = list(comps)
        inst_id = {nm: i for i, nm in enumerate(inst_names)}
        cell_names = sorted(set(comps.values()))
        ct_id = {c: i for i, c in enumerate(cell_names)}
        inst_ct = np.array([ct_id[comps[nm]] for nm in inst_names], np.int64)
        self.inst_names, self.cell_names, self.inst_ct = inst_names, cell_names, inst_ct
        slots = [dict() for _ in cell_names]
        pin_name, pin_net, pin_inst, pin_slot = [], [], [], []
        port_dir = []
        for n, inst, pn in zip(cn_net, cn_inst, cn_pin):
            if n in power:
                continue
            if inst == "PIN":
                k = _key(pn)
                pin_name.append(k)
                pin_inst.append(-1)
                pin_slot.append(-1)
                port_dir.append(ports.get(k, "INPUT"))
            else:
                ik = _key(inst)
                ii = inst_id.get(ik)
                if ii is None:
                    continue
                pk = _key(pn)
                sl = slots[inst_ct[ii]]
                s = sl.get(pk)
                if s is None:
                    s = sl[pk] = len(sl)
                pin_name.append(ik + "/" + pk)
                pin_inst.append(ii)
                pin_slot.append(s)
                port_dir.append(None)
            pin_net.append(n)
        del cn_net, cn_inst, cn_pin
        P = len(pin_name)
        self.P = P
        self.pin_name = pin_name
        self.pin_net = np.array(pin_net, np.int64)
        self.pin_inst = np.array(pin_inst, np.int64)
        self.pin_slot = np.array(pin_slot, np.int64)
        is_port = self.pin_inst < 0
        self.pin_ct = np.where(is_port, -1, inst_ct[np.maximum(self.pin_inst, 0)])
        # per (cell type, slot) Liberty pin data
        lib = self.lib
        cmods = [lib.cells.get(c) for c in cell_names]
        self.cmods, self.slots = cmods, slots
        unknown = [c for c, m in zip(cell_names, cmods) if m is None]
        sl_off = np.zeros(len(cell_names) + 1, np.int64)
        sl_off[1:] = np.cumsum([len(s) for s in slots])
        ns = int(sl_off[-1])
        sdir = np.zeros(ns + 1, np.int64)          # 0 input, 1 output, 2 inout
        scap = np.zeros((ns + 1, 2))
        sclk = np.zeros(ns + 1, bool)
        for c, (m, sl) in enumerate(zip(cmods, slots)):
            for pn, s in sl.items():
                pm = m.pin(pn) if m is not None else None
                if pm is None:
                    continue
                g = sl_off[c] + s
                sdir[g] = 0 if pm[0] == "input" else (1 if pm[0] == "output" else 2)
                scap[g] = (pm[1], pm[2])
                sclk[g] = pm[3]
        gs = np.where(is_port, ns, sl_off[np.maximum(self.pin_ct, 0)] + np.maximum(self.pin_slot, 0))
        self.pin_cap = scap[gs]                                        # [P, 2] rise / fall
        pdir = sdir[gs]
        pd = np.array([0 if v is None else (1 if v in ("INPUT", "INOUT") else 0) for v in port_dir], np.int64)
        self.pin_isdrvdir = np.where(is_port, pd, (pdir >= 1).astype(np.int64)).astype(bool)
        self.pin_clockpin = sclk[gs] & ~is_port
        self.is_port = is_port
        # net driver: first cell output, else first input port
        idx = np.arange(P)
        drv = np.full(nn, P, np.int64)
        m_out = self.pin_isdrvdir & ~is_port
        np.minimum.at(drv, self.pin_net[m_out], idx[m_out])
        drv2 = np.full(nn, P, np.int64)
        m_in = self.pin_isdrvdir & is_port
        np.minimum.at(drv2, self.pin_net[m_in], idx[m_in])
        drv = np.where(drv < P, drv, drv2)
        drv[drv == P] = -1
        self.net_drv = drv
        is_drv = np.zeros(P, bool)
        is_drv[drv[drv >= 0]] = True
        self.pin_is_drv = is_drv
        self.pin_is_load = ~self.pin_isdrvdir & (drv[self.pin_net] >= 0)
        # inst -> pin matrix per cell type (vectorized arc construction)
        loc = np.zeros(len(inst_names), np.int64)
        counts = np.bincount(inst_ct, minlength=len(cell_names))
        order = np.argsort(inst_ct, kind="stable")
        starts = np.cumsum(counts) - counts
        loc[order] = np.arange(len(inst_names)) - np.repeat(starts, counts)
        mats = []
        ip = np.nonzero(~is_port)[0]
        ctp, locp, slp = self.pin_ct[ip], loc[self.pin_inst[ip]], self.pin_slot[ip]
        for c in range(len(cell_names)):
            mats.append(np.full((counts[c], max(len(slots[c]), 1)), -1, np.int64))
        o = np.argsort(ctp, kind="stable")
        b = np.r_[0, np.nonzero(np.diff(ctp[o]))[0] + 1, len(o)]
        for k in range(len(b) - 1):
            sel = o[b[k]:b[k + 1]]
            c = ctp[sel[0]]
            mats[c][locp[sel], slp[sel]] = ip[sel]
        # arcs and checks
        whens = {"": 0}
        A = {k: [] for k in ("fr", "to", "kind", "sense", "tcr", "tcf", "trt", "tft", "when", "lib")}
        C = {k: [] for k in ("pin", "cpin", "kind", "trc", "tfc", "when")}
        for c, (m, sl) in enumerate(zip(cmods, slots)):
            if m is None or counts[c] == 0:
                continue
            M = mats[c]
            for pn, s in sl.items():
                pm = m.pin(pn)
                if pm is None:
                    continue
                tp = M[:, s]
                if pm[0] != "input":
                    for (rp, kind, sn, tcr, tcf, trt, tft, when) in m.arcs_to(pn):
                        sf = sl.get(rp)
                        if sf is None:
                            continue
                        fp = M[:, sf]
                        ok = (fp >= 0) & (tp >= 0)
                        if not ok.any():
                            continue
                        n = int(ok.sum())
                        A["fr"].append(fp[ok]); A["to"].append(tp[ok])
                        for k, v in (("kind", kind), ("sense", sn), ("tcr", tcr), ("tcf", tcf), ("trt", trt),
                                     ("tft", tft), ("when", whens.setdefault(when, len(whens))), ("lib", m.lib)):
                            A[k].append(np.full(n, v, np.int64))
                for (rp, ck, trc, tfc, when) in m.checks_on(pn):
                    sr = sl.get(rp)
                    if sr is None:
                        continue
                    cp = M[:, sr]
                    ok = (cp >= 0) & (tp >= 0)
                    if not ok.any():
                        continue
                    n = int(ok.sum())
                    C["pin"].append(tp[ok]); C["cpin"].append(cp[ok])
                    for k, v in (("kind", ck), ("trc", trc), ("tfc", tfc), ("when", whens.setdefault(when, len(whens)))):
                        C[k].append(np.full(n, v, np.int64))
        cat = lambda l: np.concatenate(l) if l else np.zeros(0, np.int64)
        self.arc = {k: cat(v) for k, v in A.items()}
        self.chk = {k: cat(v) for k, v in C.items()}
        self.whens = [w for w, _ in sorted(whens.items(), key=lambda kv: kv[1])]
        self.mats, self.inst_loc = mats, loc
        t2 = time.time()
        # .net access points (only routed nets have them)
        N = FT.parse_net_fast(os.path.join(d, design + ".net"))
        pid = {nm: i for i, nm in enumerate(pin_name)}
        nf_pid = np.array([pid.get(nm, -1) for nm in N["pin_name"]], np.int64)
        self.nf_names = N["names"]
        self.nf_net = np.array([self.net_index.get(_key(nm), -1) for nm in N["names"]], np.int64)
        self.nf_pin_net = N["pin_net"]
        self.nf_pid = nf_pid
        self.ap_cnt, self.ap = N["ap_cnt"], N["ap"]
        self.ap_pin = np.repeat(np.arange(len(nf_pid)), self.ap_cnt)     # .net pin index per AP row
        self.ap_pid = nf_pid[self.ap_pin]
        self.pin_slack_net = np.full(P, np.nan)
        ok = nf_pid >= 0
        self.pin_slack_net[nf_pid[ok]] = N["pin_slack"][ok]
        self._pid = pid
        self.X = int(self.ap[:, 1].max(initial=0)) + 2
        self.Y = int(self.ap[:, 2].max(initial=0)) + 2
        t3 = time.time()
        self._clocks(os.path.join(d, design + ".sdc"))
        self._graph()
        if verbose:
            unm = int((nf_pid < 0).sum())
            innf = np.zeros(nn, bool)
            innf[self.nf_net[self.nf_net >= 0]] = True
            sig = np.bincount(self.pin_net, minlength=nn)
            log(f"  .net pins not in the DEF netlist: {unm:,}; DEF nets not in .net: {int((~innf).sum()):,} "
                f"(by #pins: {np.bincount(np.minimum(sig[~innf], 5), minlength=6).tolist()}); .net nets not in DEF: "
                f"{int((self.nf_net < 0).sum()):,}")
            log(f"{design}[{suite}]: {nn:,} nets ({len(N['names']):,} in .net), {P:,} pins, {len(inst_names):,} "
                f"instances, {len(self.arc['fr']):,} cell arcs, {len(self.ep_pin):,} endpoints, {self.K} arrival "
                f"classes, {self.nlev} levels, {len(unknown)} unknown cells {unknown[:5]}; parse DEF {t1 - t0:.1f}s, "
                f"arcs {t2 - t1:.1f}s, .net {t3 - t2:.1f}s, clocks+graph {time.time() - t3:.1f}s")

    # ------------------------------------------------------------------ clocks, constants, checks, levels
    def _clocks(self, sdc_path):
        S = parse_sdc(sdc_path)
        self.sdc = S
        P = self.P
        names = S["order"]
        self.clk_names = names
        nclk = len(names)
        ck = [S["clocks"][n] for n in names]
        cidx = {n: i for i, n in enumerate(names)}
        port_pid = {self.pin_name[i]: i for i in np.nonzero(self.is_port)[0]}
        A = self.arc
        # adjacency for propagation through the netlist (net edges + combinational cell arcs)
        load = np.nonzero(self.pin_is_load)[0]
        ne_drv = self.net_drv[self.pin_net[load]]
        comb = A["kind"] == 0
        src = np.r_[ne_drv, A["fr"][comb]]
        dst = np.r_[load, A["to"][comb]]
        inv = np.r_[np.zeros(len(load), np.int64), (A["sense"][comb] == 1).astype(np.int64)]
        o = np.argsort(src, kind="stable")
        src, dst, inv = src[o], dst[o], inv[o]
        st = np.searchsorted(src, np.arange(P + 1))
        # ---- constants (set_case_analysis on ports) through buf / inv / simple gates
        const = np.full(P, -1, np.int64)
        q = []
        for p, v in S["case"].items():
            i = port_pid.get(p)
            if i is not None:
                const[i] = v
                q.append(i)
        cm = self.cmods
        while q:
            nq = []
            for i in q:
                for e in range(st[i], st[i + 1]):
                    j = dst[e]
                    if const[j] >= 0:
                        continue
                    if not self.pin_isdrvdir[j]:                 # a load pin: takes the net value
                        const[j] = const[i]
                        nq.append(j)
                        continue
                    m = cm[self.pin_ct[j]] if self.pin_ct[j] >= 0 else None
                    ft = m.ftype if m is not None else None
                    if ft is None:
                        continue
                    ins = [self.mats[self.pin_ct[j]][self.inst_loc[self.pin_inst[j]], self.slots[self.pin_ct[j]][pn]]
                           for pn in m.inputs if pn in self.slots[self.pin_ct[j]]]
                    vals = [const[k] if k >= 0 else -1 for k in ins]
                    v = -1
                    if ft in ("buf", "inv") and vals and vals[0] >= 0:
                        v = vals[0] if ft == "buf" else 1 - vals[0]
                    elif ft in ("and", "nand"):
                        v = 0 if 0 in vals else (1 if vals and all(x == 1 for x in vals) else -1)
                        v = (1 - v) if (v >= 0 and ft == "nand") else v
                    elif ft in ("or", "nor"):
                        v = 1 if 1 in vals else (0 if vals and all(x == 0 for x in vals) else -1)
                        v = (1 - v) if (v >= 0 and ft == "nor") else v
                    if v >= 0:
                        const[j] = v
                        nq.append(j)
            q = nq
        self.const = const
        # ---- clock network: BFS from clock source ports through nets and combinational arcs (sense = inversion)
        clk = np.full(P, -1, np.int64)
        cinv = np.zeros(P, np.int64)
        seqpin = np.zeros(P, bool)
        ip = ~self.is_port
        seq_ct = np.array([bool(m is not None and (m.seq or m.icg)) for m in cm] + [False])
        seqpin[ip] = seq_ct[self.pin_ct[ip]]
        front = []
        for c, n in enumerate(names):
            for p in ck[c]["src"]:
                i = port_pid.get(p)
                if i is not None and clk[i] < 0:
                    clk[i] = c
                    front.append(i)
        front = np.array(front, np.int64)
        while len(front):
            e0, e1 = st[front], st[front + 1]
            cnt = e1 - e0
            if cnt.sum() == 0:
                break
            rep = np.repeat(np.arange(len(front)), cnt)
            ee = np.repeat(e0, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            tgt = dst[ee]
            fr = front[rep]
            new = clk[tgt] < 0
            # a sequential cell: only its clock pins join the clock network (and stop there); data pins never
            okp = new & (~seqpin[tgt] | self.pin_clockpin[tgt])
            tgt, fr, iv = tgt[okp], fr[okp], inv[ee][okp]
            tgt, first = np.unique(tgt, return_index=True)
            fr, iv = fr[first], iv[first]
            clk[tgt] = clk[fr]
            cinv[tgt] = cinv[fr] ^ iv
            front = tgt[~(seqpin[tgt] & self.pin_clockpin[tgt])]          # don't expand through register clocks
        self.clk_of, self.clk_inv = clk, cinv
        reg_ck = (clk >= 0) & self.pin_clockpin & seqpin
        self.reg_ck = reg_ck
        cn = clk >= 0                                                      # clock-network pins
        self.cnet = cn
        # ---- arrival classes = (clock, source edge); capture events likewise
        K = 2 * max(nclk, 1)
        per = np.array([c["period"] for c in ck] + [1.0])
        tsrc = np.array([[c["wave"][0] + c["lat"], c["wave"][1] + c["lat"]] for c in ck] + [[0.0, 0.0]])
        unc = np.array([c["unc"] for c in ck] + [0.0])
        REL = np.full((K, K), BIG)
        fp = S["fp_clk"]
        for lc in range(nclk):
            for le in (0, 1):
                for cc in range(nclk):
                    if any(names[lc] in a and names[cc] in b for a, b in fp):
                        continue
                    for ce in (0, 1):
                        REL[2 * lc + le, 2 * cc + ce] = setup_relation(per[lc], tsrc[lc, le], per[cc], tsrc[cc, ce])
        self.REL, self.K, self.clk_unc = REL, K, unc
        # ---- startpoints: register clock pins (pin rise = source edge inv) and input ports with input delays
        sp_pin, sp_cls, sp_rf, sp_at = [], [], [], []
        rck = np.nonzero(reg_ck)[0]
        for rf in (0, 1):
            sp_pin.append(rck)
            sp_cls.append(2 * clk[rck] + (rf ^ cinv[rck]))
            sp_rf.append(np.full(len(rck), rf))
            sp_at.append(np.zeros(len(rck)))
        trv = np.array([c["tr"] for c in ck] + [(0.0, 0.0)])
        ck_slew = np.zeros((P, 2))
        for rf in (0, 1):
            ck_slew[rck, rf] = trv[clk[rck], rf ^ cinv[rck]]
        self.ck_slew = ck_slew
        for p, cname, fall, v in S["in"]:
            i = port_pid.get(p)
            if i is None or cname not in cidx or p in S["fp_ports"] or const[i] >= 0 or cn[i]:
                continue
            for rf in (0, 1):
                sp_pin.append(np.array([i])); sp_cls.append(np.array([2 * cidx[cname] + fall]))
                sp_rf.append(np.array([rf])); sp_at.append(np.array([v]))
        self.sp_pin = np.concatenate(sp_pin).astype(np.int64)
        self.sp_cls = np.concatenate(sp_cls).astype(np.int64)
        self.sp_rf = np.concatenate(sp_rf).astype(np.int64)
        self.sp_at = np.concatenate(sp_at).astype(float)
        # ---- endpoint checks: rows (pin, capture event, rise table, fall table, clock slew rise/fall, offset)
        Cc = self.chk
        keep = np.ones(len(Cc["pin"]), bool)
        keep &= reg_ck[Cc["cpin"]] & (const[Cc["pin"]] < 0)
        # `when` of a check false under the case-analysis constants -> disabled (OpenSTA sim)
        if (const >= 0).any():
            wl = self._when_false(Cc["pin"], Cc["when"])
            keep &= ~wl
        cp = Cc["cpin"][keep]
        kind = Cc["kind"][keep]
        edge_at_pin = np.where((kind == 0) | (kind == 2), 0, 1)                # rising / falling check
        evt = 2 * clk[cp] + (edge_at_pin ^ cinv[cp])
        rows = {"pin": [Cc["pin"][keep]], "evt": [evt], "tr": [Cc["trc"][keep]], "tf": [Cc["tfc"][keep]],
                "cks": [ck_slew[cp, edge_at_pin]], "off": [unc[clk[cp]]], "type": [np.where(kind >= 2, 1, 0)]}
        # inferred clock-gating checks (setup 0): data inputs of AND/NAND (rising) / OR/NOR (falling) clock gates
        gate_rows = 0
        for c, m in enumerate(cm):
            if m is None or m.ftype not in ("and", "nand", "or", "nor") or not len(self.mats[c]):
                continue
            M = self.mats[c]
            sl = self.slots[c]
            ins = [sl[p] for p in m.inputs if p in sl]
            outs = [s for p, s in sl.items() if m.pin(p) is not None and m.pin(p)[0] != "input"]
            if len(ins) < 2 or not outs:
                continue
            Mi = M[:, ins]
            Mo = M[:, outs[0]]
            okm = Mo >= 0
            okm &= cn[np.maximum(Mo, 0)]
            isc = (Mi >= 0) & cn[np.maximum(Mi, 0)]
            g = okm & isc.any(1) & (~isc & (Mi >= 0)).any(1)
            if not g.any():
                continue
            for r in np.nonzero(g)[0]:
                cpin = Mi[r][isc[r]][0]
                e_pin = 0 if m.ftype in ("and", "nand") else 1
                for dp in Mi[r][(~isc[r]) & (Mi[r] >= 0)]:
                    if const[dp] >= 0:
                        continue
                    rows["pin"].append(np.array([dp])); rows["evt"].append(np.array([2 * clk[cpin] + (e_pin ^ cinv[cpin])]))
                    rows["tr"].append(np.array([-1])); rows["tf"].append(np.array([-1]))
                    rows["cks"].append(np.array([0.0])); rows["off"].append(np.array([unc[clk[cpin]]]))
                    rows["type"].append(np.array([2]))
                    gate_rows += 1
        # output ports with output delays
        for p, cname, fall, v in S["out"]:
            i = port_pid.get(p)
            if i is None or cname not in cidx or const[i] >= 0:
                continue
            c = cidx[cname]
            rows["pin"].append(np.array([i])); rows["evt"].append(np.array([2 * c + fall]))
            rows["tr"].append(np.array([-1])); rows["tf"].append(np.array([-1]))
            rows["cks"].append(np.array([0.0])); rows["off"].append(np.array([unc[c] + v]))
            rows["type"].append(np.array([3]))
        self.ck_rows = {k: np.concatenate(v) for k, v in rows.items()}
        self.ck_rows["evt"] = self.ck_rows["evt"].astype(np.int64)
        self.ep_pin = np.unique(self.ck_rows["pin"])
        self.n_gate_checks = gate_rows

    def _when_false(self, pins, when_ids):
        """True where a check's `when` (a conjunction of literals) is false under the constants of its instance."""
        out = np.zeros(len(pins), bool)
        inst = self.pin_inst[pins]
        hasc = np.zeros(len(self.inst_names), bool)
        cpin = np.nonzero(self.const >= 0)[0]
        hasc[self.pin_inst[cpin][self.pin_inst[cpin] >= 0]] = True
        for i in np.nonzero(hasc[np.maximum(inst, 0)] & (inst >= 0) & (when_ids > 0))[0]:
            w = self.whens[when_ids[i]]
            if "|" in w or "(" in w:
                continue
            c = self.pin_ct[pins[i]]
            M, sl, li = self.mats[c], self.slots[c], self.inst_loc[inst[i]]
            for lit in w.replace(" ", "").split("&"):
                neg = lit.startswith("!")
                nm = lit.lstrip("!")
                s = sl.get(nm)
                if s is None:
                    continue
                pj = M[li, s]
                if pj >= 0 and self.const[pj] >= 0 and self.const[pj] == (0 if neg else 1) ^ 1:
                    out[i] = True
                    break
        return out

    def _graph(self):
        """data timing graph: net edges + cell rf-edges, levelized (Kahn); clock network and constants removed."""
        P = self.P
        A = self.arc
        cn, const = self.cnet, self.const
        load = np.nonzero(self.pin_is_load & ~cn & (const < 0))[0]
        ne_drv = self.net_drv[self.pin_net[load]]
        ok = (const[ne_drv] < 0) & (~cn[ne_drv])
        ne_drv, ne_snk = ne_drv[ok], load[ok]
        fr, to, kind = A["fr"], A["to"], A["kind"]
        okc = (const[fr] < 0) & (const[to] < 0) & ~cn[to]
        okc &= np.where(kind == 0, ~cn[fr], self.reg_ck[fr])          # launch arcs only from register clock pins
        if (const >= 0).any() and okc.any():
            wf = self._when_false(to, A["when"])
            okc &= ~wf
        ai = np.nonzero(okc)[0]
        # levelize on pins
        efr = np.r_[ne_drv, fr[ai]]
        eto = np.r_[ne_snk, to[ai]]
        latch_dq = np.zeros(len(ai), bool)
        lat_ct = np.array([bool(m is not None and m.latch) for m in self.cmods] + [False])
        latch_dq = (kind[ai] == 0) & lat_ct[np.maximum(self.pin_ct[to[ai]], 0)] & (self.pin_ct[to[ai]] >= 0)
        for attempt in range(4):
            lev, left = _levelize(P, efr, eto)
            if not left.any():
                break
            # cycles: cell arcs inside one strongly connected component; drop latch D->Q ones first, then all
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            G = coo_matrix((np.ones(len(efr), np.int32), (efr, eto)), shape=(P, P)).tocsr()
            _, lab = connected_components(G, directed=True, connection="strong")
            sz = np.bincount(lab)
            inside = (lab[fr[ai]] == lab[to[ai]]) & (sz[lab[to[ai]]] > 1)
            bad = inside & latch_dq if (attempt == 0 and (inside & latch_dq).any()) else inside
            log(f"  {int(left.sum())} pins after cycles, {int((sz > 1).sum())} cyclic components: dropping "
                f"{int(bad.sum())} {'latch D->Q ' if bad is not inside else ''}arcs")
            ai, latch_dq = ai[~bad], latch_dq[~bad]
            efr = np.r_[ne_drv, fr[ai]]
            eto = np.r_[ne_snk, to[ai]]
        self.lev = lev
        self.nlev = int(lev.max(initial=0)) + 1
        # net edges sorted by sink level
        o = np.argsort(lev[ne_snk], kind="stable")
        self.ne_drv, self.ne_snk = ne_drv[o], ne_snk[o]
        lv = lev[self.ne_snk]
        self.ne_lo = np.searchsorted(lv, np.arange(self.nlev + 1))
        # cell rf-edges: expand arcs by unateness
        E = {k: [] for k in ("arc", "irf", "orf", "td", "ts")}
        sense, tcr, tcf, trt, tft = A["sense"][ai], A["tcr"][ai], A["tcf"][ai], A["trt"][ai], A["tft"][ai]
        kd = kind[ai]
        for orf, td, ts in ((0, tcr, trt), (1, tcf, tft)):
            for irf in (0, 1):
                same = irf == orf
                ok_sense = np.where(sense == 0, same, np.where(sense == 1, not same, True)).astype(bool)
                ok_kind = np.where(kd == 1, irf == 0, np.where(kd == 2, irf == 1, ok_sense)).astype(bool)
                m = (td >= 0) & ok_kind
                E["arc"].append(ai[m]); E["irf"].append(np.full(int(m.sum()), irf))
                E["orf"].append(np.full(int(m.sum()), orf)); E["td"].append(td[m]); E["ts"].append(ts[m])
        E = {k: np.concatenate(v).astype(np.int64) for k, v in E.items()}
        efr, eto = fr[E["arc"]], to[E["arc"]]
        o = np.lexsort((E["orf"], eto, lev[eto]))
        self.ce_arc, self.ce_irf, self.ce_orf = E["arc"][o], E["irf"][o], E["orf"][o]
        self.ce_td, self.ce_ts = E["td"][o], E["ts"][o]
        self.ce_from, self.ce_to = efr[o], eto[o]
        self.ce_lib = A["lib"][self.ce_arc]
        lv = lev[self.ce_to]
        self.ce_lo = np.searchsorted(lv, np.arange(self.nlev + 1))
        key = self.ce_to * 2 + self.ce_orf
        newg = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)
        self.g_start = np.nonzero(newg)[0]
        self.g_lo = np.searchsorted(lv[self.g_start], np.arange(self.nlev + 1))
        self.g_to = self.ce_to[self.g_start]
        self.g_orf = self.ce_orf[self.g_start]
        self.ce_grp = np.cumsum(newg) - 1
        # library thresholds per lib index
        L = self.lib.libs
        self.L_vth = np.array([l["vth"] for l in L])
        self.L_vl = np.array([l["vl"] for l in L])
        self.L_vh = np.array([l["vh"] for l in L])
        self.L_der = np.array([l["der"] for l in L])
        self.L_vin = np.array([l["vin"] for l in L])
        # per load pin: library (for input-port dspf thresholds); NanGate default
        self.def_lib = 0
        for i, l in enumerate(L):
            if "Nangate" in l["name"] or "NangateOpenCellLibrary" in l["name"]:
                self.def_lib = i


def _levelize(P, efr, eto):
    """longest-path levels by Kahn's algorithm (vectorized frontiers); -> (level, pins left on cycles)."""
    indeg = np.bincount(eto, minlength=P)
    o = np.argsort(efr, kind="stable")
    efr_s, eto_s = efr[o], eto[o]
    st = np.searchsorted(efr_s, np.arange(P + 1))
    lev = np.zeros(P, np.int64)
    front = np.nonzero(indeg == 0)[0]
    done = np.zeros(P, bool)
    done[front] = True
    k = 0
    indeg = indeg.copy()
    while len(front):
        e0, e1 = st[front], st[front + 1]
        cnt = e1 - e0
        if cnt.sum() == 0:
            break
        ee = np.repeat(e0, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        tg = eto_s[ee]
        np.subtract.at(indeg, tg, 1)
        k += 1
        cand = np.unique(tg)
        nf = cand[(indeg[cand] == 0) & ~done[cand]]
        lev[nf] = k
        done[nf] = True
        front = nf
    return lev, ~done


# ════════════════════════════════════════════════════════════════════════════════════
#  route -> RC networks (OpenROAD grt MakeWireParasitics + OpenSTA ReduceToPiElmore)
# ════════════════════════════════════════════════════════════════════════════════════
class RC:
    pass


def build_rc(D, seg, stub_um=STUB_UM):
    """seg: int64 [S, 7] (DEF net id, x1, y1, z1, x2, y2, z2), GCells, 0-based layers.
    -> RC with per net: par (has parasitics), c1/c2/rpi [nets, 2] (fF, kOhm), ctot [nets, 2]; per pin: elm [P, 2] ns
    (nan = no Elmore: not routed, unreachable or missing pin -> OpenSTA wire delay 0, load slew = driver slew)."""
    nn = len(D.net_names)
    P = D.P
    seg = seg[seg[:, 0] >= 0]
    n, x1, y1, z1, x2, y2, z2 = (seg[:, k] for k in range(7))
    isvia = z1 != z2
    keep = isvia | (x1 != x2) | (y1 != y2)
    n, x1, y1, z1, x2, y2, z2, isvia = n[keep], x1[keep], y1[keep], z1[keep], x2[keep], y2[keep], z2[keep], isvia[keep]
    L = 16
    X = int(max(D.X, x1.max(initial=0) + 2, x2.max(initial=0) + 2))
    Y = int(max(D.Y, y1.max(initial=0) + 2, y2.max(initial=0) + 2))
    key = lambda q, x, y, z: ((q * L + z) * X + x) * Y + y
    ka, kb = key(n, x1, y1, z1), key(n, x2, y2, z2)
    uk = np.unique(np.r_[ka, kb])
    NN = len(uk)
    ia, ib = np.searchsorted(uk, ka), np.searchsorted(uk, kb)
    lr, lc = D.layer_rc[:, 0], D.layer_rc[:, 1]
    vr = D.via_r
    glen = (np.abs(x2 - x1) + np.abs(y2 - y1)) * UM_PER_GC
    zl = np.minimum(z1, z2)
    Rs = np.where(isvia, vr[np.clip(zl, 0, len(vr) - 1)], lr[np.clip(z1, 0, len(lr) - 1)] * glen)
    Cs = np.where(isvia, 0.0, lc[np.clip(z1, 0, len(lc) - 1)] * glen)
    ncap = np.zeros(NN)
    np.add.at(ncap, ia, Cs / 2)
    np.add.at(ncap, ib, Cs / 2)
    routed = np.zeros(nn, bool)
    routed[n] = True
    # ---- pins -> GCell nodes (layer above the access layer + its cut, else the access layer)
    app = D.ap_pid
    okap = app >= 0
    okap &= routed[D.pin_net[np.maximum(app, 0)]]
    ai = np.nonzero(okap)[0]
    apl, apx, apy = D.ap[ai, 0], D.ap[ai, 1], D.ap[ai, 2]
    apn = D.pin_net[app[ai]]
    kup = key(apn, apx, apy, np.minimum(apl + 1, L - 1))
    ksm = key(apn, apx, apy, apl)
    pu = np.minimum(np.searchsorted(uk, kup), max(NN - 1, 0))
    ps = np.minimum(np.searchsorted(uk, ksm), max(NN - 1, 0))
    oku = (uk[pu] == kup) if NN else np.zeros(len(ai), bool)
    oks = (uk[ps] == ksm) if NN else np.zeros(len(ai), bool)
    gnode = np.where(oku, pu, np.where(oks, ps, -1))
    grv = np.where(oku, vr[np.clip(apl, 0, len(vr) - 1)], 0.0)
    stub_r = np.where(oku, lr[np.clip(apl + 1, 0, len(lr) - 1)], lr[np.clip(apl, 0, len(lr) - 1)]) * stub_um
    stub_c = np.where(oku, lc[np.clip(apl + 1, 0, len(lc) - 1)], lc[np.clip(apl, 0, len(lc) - 1)]) * stub_um
    good = gnode >= 0
    pp = app[ai][good]
    pp, first = np.unique(pp, return_index=True)                       # first matching AP per pin
    gnode, grv = gnode[good][first], grv[good][first]
    stub_r, stub_c = stub_r[good][first], stub_c[good][first]
    npn = len(pp)
    pin_node = np.full(P, -1, np.int64)
    pin_node[pp] = NN + np.arange(npn)
    NT = NN + npn
    cap = np.zeros((NT, 2))
    cap[:NN, 0] = ncap
    cap[:NN, 1] = ncap
    cap[NN:, :] = D.pin_cap[pp] + (stub_c / 2)[:, None]
    np.add.at(cap[:, 0], gnode, stub_c / 2)
    np.add.at(cap[:, 1], gnode, stub_c / 2)
    eu = np.r_[ia, NN + np.arange(npn)]
    ev = np.r_[ib, gnode]
    eR = np.r_[Rs, grv + stub_r]
    # ---- BFS from every driver pin node (super source NT)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order
    drv = D.net_drv
    hasd = (drv >= 0) & routed
    dn = np.full(nn, -1, np.int64)
    dn[hasd] = pin_node[drv[hasd]]
    roots = np.unique(dn[dn >= 0])
    su = np.r_[eu, np.full(len(roots), NT)]
    sv = np.r_[ev, roots]
    G = coo_matrix((np.ones(len(su), np.int32), (su, sv)), shape=(NT + 1, NT + 1)).tocsr()
    order, pred = breadth_first_order(G, NT, directed=False, return_predecessors=True)
    order = order[1:]
    par = np.full(NT, -1, np.int64)
    pr = pred[order]
    inner = pr != NT
    par[order[inner]] = pr[inner]
    reach = np.zeros(NT, bool)
    reach[order] = True
    # resistance of the tree edge (min over parallel resistors)
    a_, b_ = np.minimum(eu, ev), np.maximum(eu, ev)
    ek = a_ * (NT + 1) + b_
    o = np.lexsort((eR, ek))
    ek, eRs = ek[o], eR[o]
    kk = np.r_[True, ek[1:] != ek[:-1]] if len(ek) else np.zeros(0, bool)
    ek, eRs = ek[kk], eRs[kk]
    ch = order[inner]
    pa = par[ch]
    q = np.minimum(ch, pa) * (NT + 1) + np.maximum(ch, pa)
    pos = np.clip(np.searchsorted(ek, q), 0, max(len(ek) - 1, 0))
    Rp = np.zeros(NT)
    if len(ek):
        Rp[ch] = eRs[pos]
    depth = _depth(par)
    lvo = np.argsort(np.where(reach, depth, -1), kind="stable")
    lvo = lvo[reach[lvo]]
    ds = depth[lvo]
    bnd = np.r_[0, np.nonzero(np.diff(ds))[0] + 1, len(ds)]
    y1 = cap * reach[:, None]
    y2 = np.zeros((NT, 2))
    y3 = np.zeros((NT, 2))
    for b in range(len(bnd) - 2, 0, -1):                       # deepest level first (level 0 = roots)
        lvl = lvo[bnd[b]:bnd[b + 1]]
        pp_ = par[lvl]
        R = Rp[lvl][:, None]
        a1, a2, a3 = y1[lvl], y2[lvl], y3[lvl]
        np.add.at(y1, pp_, a1)
        np.add.at(y2, pp_, a2 - R * a1 * a1)
        np.add.at(y3, pp_, a3 - 2 * R * a1 * a2 + R * R * a1 ** 3)
    elm = np.zeros((NT, 2))
    for b in range(1, len(bnd) - 1):
        lvl = lvo[bnd[b]:bnd[b + 1]]
        elm[lvl] = elm[par[lvl]] + Rp[lvl][:, None] * y1[lvl] * 1e-3          # kOhm*fF = ps -> ns
    r = RC()
    r.routed = routed
    r.par = np.zeros(nn, bool)
    r.par[hasd] = True                                       # routed nets with a driver: a parasitic network exists
    Y1 = np.zeros((nn, 2)); Y2 = np.zeros((nn, 2)); Y3 = np.zeros((nn, 2))
    hv = dn >= 0
    Y1[hv], Y2[hv], Y3[hv] = y1[dn[hv]], y2[dn[hv]], y3[dn[hv]]
    with np.errstate(divide="ignore", invalid="ignore"):
        capld = (Y2 == 0) & (Y3 == 0)
        c1 = np.where(capld, Y1, Y2 * Y2 / Y3)
        c2 = np.where(capld, 0.0, Y1 - Y2 * Y2 / Y3)
        rpi = np.where(capld, 0.0, -Y3 * Y3 / (Y2 * Y2 * Y2))
    r.c1, r.c2, r.rpi = np.nan_to_num(c1), np.nan_to_num(c2), np.nan_to_num(rpi)
    # lumped load of nets without parasitics: pin caps of the loads (OpenSTA LumpedCap, no wire-load model)
    lc_ = np.zeros((nn, 2))
    ld = np.nonzero(D.pin_is_load)[0]
    for rf in (0, 1):
        lc_[:, rf] = np.bincount(D.pin_net[ld], weights=D.pin_cap[ld, rf], minlength=nn)
    r.ctot = np.where(r.par[:, None], Y1, lc_)
    r.ctot_wire = np.bincount(n, weights=Cs, minlength=nn)
    pe = np.full((P, 2), np.nan)
    okp = (pin_node >= 0)
    okp &= reach[np.maximum(pin_node, 0)]
    # the node must hang under its own net's driver (a reached node of another component is impossible: nets are
    # keyed separately) -> reachable means connected to the driver
    pe[okp] = elm[pin_node[okp]]
    pe[D.pin_is_drv] = np.nan
    r.elm = pe
    r.pin_node = pin_node
    r.missing = np.zeros(P, bool)
    r.missing[(pin_node < 0) & routed[D.pin_net] & (D.pin_is_load | D.pin_is_drv)] = True
    r.n_nodes = NT
    return r


def update_rc(D, rc, seg_all, nets):
    """re-extract the RC of `nets` (DEF net ids) from their rows of the full route array and patch rc in place
    (incremental timing after a promotion round: a few thousand nets instead of all)."""
    nets = np.unique(np.asarray(nets, np.int64))
    nets = nets[nets >= 0]
    if not len(nets):
        return rc
    sub = build_rc(D, seg_all[np.isin(seg_all[:, 0], nets)])
    for k in ("c1", "c2", "rpi", "ctot"):
        getattr(rc, k)[nets] = getattr(sub, k)[nets]
    rc.par[nets] = sub.par[nets]
    rc.routed[nets] = sub.routed[nets]
    rc.ctot_wire[nets] = sub.ctot_wire[nets]
    pins = np.isin(D.pin_net, nets)
    rc.elm[pins] = sub.elm[pins]
    rc.missing[pins] = sub.missing[pins]
    return rc


def _depth(par):
    """tree depth of every node from parent pointers (-1 = root) by pointer jumping."""
    n = len(par)
    d = (par >= 0).astype(np.int64)
    a = par.copy()
    for _ in range(64):
        live = a >= 0
        if not live.any():
            break
        am = np.maximum(a, 0)
        d = d + np.where(live, d[am], 0)
        a = np.where(live, a[am], -1)
    return d


# ════════════════════════════════════════════════════════════════════════════════════
#  OpenSTA dmp_ceff_elmore, vectorized (dcalc/DmpCeff.cc, DmpDelayCalc.cc); units ns, fF, ns/fF
# ════════════════════════════════════════════════════════════════════════════════════
def _ex(x):
    return np.exp(np.maximum(x, -700.0))


def _y0(u, rc):
    return u - rc * (1.0 - _ex(-u / rc))


def _yramp(t, t0, dt, rc):
    t1 = t - t0
    a = _y0(np.maximum(t1, 0.0), rc)
    b = _y0(np.maximum(t1 - dt, 0.0), rc)
    return np.where(t1 <= 0, 0.0, np.where(t1 <= dt, a / dt, (a - b) / dt))


def _dyramp(t, t0, dt, cl, rd):
    rc = rd * cl
    t1 = t - t0
    u1 = np.maximum(t1, 0.0)
    u2 = np.maximum(t1 - dt, 0.0)
    y0a, y0b = _y0(u1, rc), _y0(u2, rc)
    dta, dtb = 1.0 - _ex(-u1 / rc), 1.0 - _ex(-u2 / rc)
    dca = rd * ((1.0 + u1 / rc) * _ex(-u1 / rc) - 1.0)
    dcb = rd * ((1.0 + u2 / rc) * _ex(-u2 / rc) - 1.0)
    in_ramp = t1 <= dt
    d_t0 = np.where(in_ramp, -dta / dt, -(dta - dtb) / dt)
    d_dt = np.where(in_ramp, -y0a / (dt * dt), -(y0a + y0b) / (dt * dt) + dtb / dt)   # (sic, as DmpCeff.cc)
    d_cl = np.where(in_ramp, dca / dt, (dca - dcb) / dt)
    z = t1 <= 0
    return np.where(z, 0.0, d_t0), np.where(z, 0.0, d_dt), np.where(z, 0.0, d_cl)


def _root(fun, args, target, lo, hi, x0=None, iters=40, tol=1e-12):
    """vectorized safeguarded Newton (rtsafe) for fun(t, *args)[0] = target, fun increasing on [lo, hi]; converged
    elements leave the active set. -> (x, ok); ok = bracketed (OpenSTA findRoot fails otherwise)."""
    target = np.broadcast_to(np.asarray(target, float), lo.shape)
    with np.errstate(all="ignore"):
        flo = fun(lo, *args)[0] - target
        fhi = fun(hi, *args)[0] - target
    ok = (flo <= 0) & (fhi >= 0) & np.isfinite(flo) & np.isfinite(fhi)
    lo, hi = lo.copy(), hi.copy()
    x = 0.5 * (lo + hi) if x0 is None else np.where(np.isfinite(x0), np.clip(x0, lo, hi), 0.5 * (lo + hi))
    act = np.nonzero(ok)[0]
    for _ in range(iters):
        if not len(act):
            break
        a = tuple(v[act] for v in args)
        xa, la, ha = x[act], lo[act], hi[act]
        with np.errstate(all="ignore"):
            fx, dfx = fun(xa, *a)
            fx = fx - target[act]
            pos = fx > 0
            ha = np.where(pos, xa, ha)
            la = np.where(pos, la, xa)
            xn = xa - fx / dfx
        # converged (root hit exactly / Newton step below tol): keep the iterate -- never bisect it away
        done = (np.abs(xn - xa) <= tol * (1.0 + np.abs(xa))) | (fx == 0)
        bad = (~np.isfinite(xn) | (xn <= la) | (xn >= ha)) & ~done
        xn = np.where(bad, 0.5 * (la + ha), xn)
        x[act], lo[act], hi[act] = xn, la, ha
        act = act[~done]
    return x, ok & np.isfinite(x)


class Dmp:
    """per-element DMP state (driver of a net for one output transition)."""
    FIELDS = ("alg", "ceff", "t0", "dt", "vod", "valid", "slew", "k0", "k1", "k2", "k3", "k4", "p1", "p2", "vub",
              "rd", "vth", "vl", "vh", "der")


def _vo(t, t0, dt, k0, k1, k2, k3, k4, p1, p2):
    def V0(u):
        e1, e2 = _ex(-p1 * u), _ex(-p2 * u)
        return k0 * (k1 + k2 * u + k3 * e1 + k4 * e2), k0 * (k2 - k3 * p1 * e1 - k4 * p2 * e2)
    t1 = t - t0
    va, da = V0(np.maximum(t1, 0.0))
    vb, db = V0(np.maximum(t1 - dt, 0.0))
    v = np.where(t1 <= 0, 0.0, np.where(t1 <= dt, va / dt, (va - vb) / dt))
    d = np.where(t1 <= 0, 0.0, np.where(t1 <= dt, da / dt, (da - db) / dt))
    return v, d


def _vl(t, t0, dt, k0, k1, k2, k3, k4, p1, p2, p3):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        D1 = k0 * (k1 - k2 / p3)
        D3 = -p3 * k0 * k3 / (p1 - p3)
        D4 = -p3 * k0 * k4 / (p2 - p3)
        D5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3) + p3 * k4 / (p2 - p3))

    def V0(u):
        e1, e2, e3 = _ex(-p1 * u), _ex(-p2 * u), _ex(-p3 * u)
        return D1 + u + D3 * e1 + D4 * e2 + D5 * e3, 1.0 - D3 * p1 * e1 - D4 * p2 * e2 - D5 * p3 * e3
    t1 = t - t0
    va, da = V0(np.maximum(t1, 0.0))
    vb, db = V0(np.maximum(t1 - dt, 0.0))
    v = np.where(t1 <= 0, 0.0, np.where(t1 <= dt, va / dt, (va - vb) / dt))
    d = np.where(t1 <= 0, 0.0, np.where(t1 <= dt, da / dt, (da - db) / dt))
    return v, d


def dmp_gate(T, td, ts, sin, c2, rpi, c1, par, ctot, vth, vl, vh, der):
    """gate delay / driver slew of timing arcs (one row each) -> dict of arrays (Dmp.FIELDS + 'delay').
    rpi in kOhm, caps in fF, times in ns. par=False -> no parasitics (LumpedCap at ctot)."""
    n = len(td)
    out = {k: np.zeros(n) for k in Dmp.FIELDS}
    out["alg"] = np.zeros(n, np.int64)
    out["valid"] = np.zeros(n, bool)
    out["delay"] = np.zeros(n)
    if n == 0:
        return out
    rpi = rpi * 1e-3                                                  # kOhm -> ns/fF
    cpi = np.where(par, c1 + c2, ctot)
    d1 = T(td, sin, cpi)
    d2 = T(td, sin, cpi + 1.0)
    rd = -np.log(vth) * np.abs(d1 - d2) / 1.0                          # gateModelRd, ns/fF
    capm = (~par) | (rd < 1e-8) | (rpi < rd * 1e-3) | (c1 == 0) | (c1 < c2 * 1e-3) | (rpi == 0)
    zc2 = ~capm & (c2 < c1 * 1e-3)
    pim = ~capm & ~zc2
    out["rd"], out["vth"], out["vl"], out["vh"], out["der"] = rd, vth, vl, vh, der
    out["p2"][:] = 1.0
    # ---- DmpCap (also the fall-back of every failure): table at c1 + c2
    out["ceff"] = cpi.copy()
    out["delay"] = d1.copy()
    out["slew"] = np.where(ts >= 0, np.maximum(T(np.maximum(ts, 0), sin, cpi), 0.0), 0.0)
    out["alg"][:] = 0
    # ---- DmpPi
    ii = np.nonzero(pim)[0]
    if len(ii):
        _dmp_pi(T, ii, td, ts, sin, c2, rpi, c1, rd, vth, vl, vh, der, out)
    ii = np.nonzero(zc2)[0]
    if len(ii):
        _dmp_zc2(T, ii, td, ts, sin, rpi, c1, rd, vth, vl, vh, der, out)
    return out


def _gate_delays(T, td, ts, s, ceff, vth, vl, vh, der):
    tv = T(td, s, ceff)
    sl = np.maximum(T(ts, s, ceff), 0.0) * der                    # GateTableModel::gateDelay clips slew < 0 to 0
    return tv, tv - sl * (vth - vl) / (vh - vl), sl


def _newton(evalf, X, maxit=100, tol=0.01):
    """vectorized newtonRaphson (DmpCeff.cc): x += J^-1 (-F) until every |dx| <= |x| * tol; -> (X, ok)."""
    n, m = X.shape
    alive = np.ones(n, bool)
    ok = np.zeros(n, bool)
    for _ in range(maxit):
        act = np.nonzero(alive)[0]
        if not len(act):
            break
        F, J, bad, Xa = evalf(X[act], act)
        if Xa is not None:
            X[act] = Xa
        det = np.linalg.det(J)
        sing = ~np.isfinite(det) | (np.abs(det) < 1e-300) | bad | ~np.isfinite(F).all(1)
        Js = np.where(sing[:, None, None], np.eye(m)[None], J)
        p = np.linalg.solve(Js, -F[..., None])[..., 0]
        p[sing] = 0.0
        conv = np.all(np.abs(p) <= np.abs(X[act]) * tol, axis=1) & ~sing
        X[act] += p
        ok[act[conv]] = True
        alive[act[conv | sing]] = False
    return X, ok


def _dmp_pi(T, ii, td, ts, sin, c2a, rpia, c1a, rda, vtha, vla, vha, dera, out):
    td, ts, s = td[ii], ts[ii], sin[ii]
    c2, rpi, c1, rd = c2a[ii], rpia[ii], c1a[ii], rda[ii]
    vth, vl, vh, der = vtha[ii], vla[ii], vha[ii], dera[ii]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z1 = 1.0 / (rpi * c1)
        k0 = 1.0 / (rd * c2)
        a = rpi * rd * c1 * c2
        b = rd * (c1 + c2) + rpi * c1
        sq = np.sqrt(np.maximum(b * b - 4 * a, 0.0))
        p1 = (b + sq) / (2 * a)
        p2 = (b - sq) / (2 * a)
        p1p2 = p1 * p2
        k2 = z1 / p1p2
        k1 = (1.0 - k2 * (p1 + p2)) / p1p2
        k4 = (k1 * p1 + k2) / (p2 - p1)
        k3 = -k1 - k4
        zz = (c1 + c2) / (rpi * c1 * c2)
        Aa = zz / p1p2
        Bb = (zz - p1) / (p1 * (p1 - p2))
        Dd = (zz - p2) / (p2 * (p2 - p1))

    def evalf(Xa, act):
        t0, dt, ce = Xa[:, 0], Xa[:, 1], Xa[:, 2]
        g = lambda v: v[act]
        rdg, c1g, c2g = g(rd), g(c1), g(c2)
        bad = (ce < 0) | (ce > c1g + c2g) | (dt <= 0)
        cs = np.clip(ce, 1e-12, None)
        dts = np.where(dt > 0, dt, 1e-12)
        tv, tl, sl = _gate_delays(T, g(td), g(ts), g(s), cs, g(vth), g(vl), g(vh), g(der))
        bad |= sl == 0
        ct = np.minimum(sl / (g(vh) - g(vl)), 1.4 * dts)
        p1g, p2g, A_, B_, D_ = g(p1), g(p2), g(Aa), g(Bb), g(Dd)
        rce = rdg * cs
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e1, e2, erc = _ex(-p1g * dts), _ex(-p2g * dts), _ex(-dts / rce)
            y50 = _yramp(tv, t0, dts, rce)
            y20 = _yramp(tl, t0, dts, rce)
            ce1, ce2, cerc = _ex(-p1g * ct), _ex(-p2g * ct), _ex(-ct / rce)
            ipi = (A_ * ct + (B_ / p1g) * (1 - ce1) + (D_ / p2g) * (1 - ce2)) / (rdg * ct * dts)
            iceff = (rce * ct - rce * rce * (1 - cerc)) / (rdg * ct * dts)
            F = np.stack([y20 - g(vl), y50 - g(vth), ipi - iceff], 1)
            J = np.zeros((len(act), 3, 3))
            J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = _dyramp(tl, t0, dts, cs, rdg)
            J[:, 1, 0], J[:, 1, 1], J[:, 1, 2] = _dyramp(tv, t0, dts, cs, rdg)
            J[:, 2, 1] = (-A_ * dts + B_ * dts * e1 - (2 * B_ / p1g) * (1 - e1) + D_ * dts * e2
                          - (2 * D_ / p2g) * (1 - e2)
                          + rce * (dts + dts * erc - 2 * rce * (1 - erc))) / (rdg * dts ** 3)
            J[:, 2, 2] = (2 * rce - dts - (2 * rce + dts) * _ex(-dts / rce)) / (dts * dts)
        return F, J, bad, None

    def start(ce0, idx):
        tv, tl, sl = _gate_delays(T, td[idx], ts[idx], s[idx], ce0, vth[idx], vl[idx], vh[idx], der[idx])
        dt = sl / (vh[idx] - vl[idx])
        t0 = tv + np.log(1.0 - vth[idx]) * rd[idx] * ce0 - vth[idx] * dt
        return np.stack([t0, dt, ce0], 1)
    n = len(ii)
    allidx = np.arange(n)
    X = start(c1 + c2, allidx)
    X, ok = _newton(evalf, X)
    rt = np.nonzero(~ok)[0]
    if len(rt):                                                         # retry from Ceff = c2 (findDriverParamsPi)
        X2 = start(c2[rt], rt)
        sub_ok = np.zeros(len(rt), bool)

        def evalf2(Xa, act):
            return evalf(Xa, rt[act])
        X2, sub_ok = _newton(evalf2, X2)
        X[rt] = X2
        ok[rt] = sub_ok
    ceff = np.where(ok, X[:, 2], c1 + c2)
    t0, dt = X[:, 0], X[:, 1]
    delay = T(td, s, ceff)
    tslew = np.where(ts >= 0, np.maximum(T(np.maximum(ts, 0), s, ceff), 0.0), 0.0)
    # driver waveform Vo -> vth / vl / vh crossings (findDriverDelaySlew)
    vub = t0 + dt + (c1 + c2) * (rd + rpi) * 2.0
    good = ok & np.isfinite(k0 + k1 + k2 + k3 + k4 + p1 + p2) & (dt > 0)
    gi = np.nonzero(good)[0]
    valid = np.zeros(n, bool)
    vod = np.zeros(n)
    slew = np.where(ok, tslew, tslew)
    if len(gi):
        args = (t0[gi], dt[gi], k0[gi], k1[gi], k2[gi], k3[gi], k4[gi], p1[gi], p2[gi])
        span = tslew[gi] * der[gi] / (vh[gi] - vl[gi])
        d50, o1 = _root(_vo, args, vth[gi], t0[gi].copy(), vub[gi].copy(), delay[gi])
        tl, o2 = _root(_vo, args, vl[gi], t0[gi].copy(), d50.copy(), d50 - span * (vth[gi] - vl[gi]))
        th, o3 = _root(_vo, args, vh[gi], d50.copy(), vub[gi].copy(), d50 + span * (vh[gi] - vth[gi]))
        v = o1 & o2 & o3
        valid[gi] = v
        vod[gi] = d50
        slew[gi] = np.where(v, (th - tl) / der[gi], tslew[gi])
    o = out
    o["alg"][ii] = np.where(ok, 2, 0)
    o["ceff"][ii], o["delay"][ii], o["slew"][ii] = ceff, delay, slew
    o["t0"][ii], o["dt"][ii], o["vod"][ii], o["valid"][ii] = t0, dt, vod, valid & ok
    for k, v in (("k0", k0), ("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4), ("p1", p1), ("p2", p2), ("vub", vub)):
        o[k][ii] = np.nan_to_num(v)


def _dmp_zc2(T, ii, td, ts, sin, rpia, c1a, rda, vtha, vla, vha, dera, out):
    td, ts, s = td[ii], ts[ii], sin[ii]
    rpi, c1, rd = rpia[ii], c1a[ii], rda[ii]
    vth, vl, vh, der = vtha[ii], vla[ii], vha[ii], dera[ii]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z1 = 1.0 / (rpi * c1)
        p1 = 1.0 / (c1 * (rd + rpi))
        k0 = p1 / z1
        k2 = 1.0 / k0
        k1 = (p1 - z1) / (p1 * p1)
        k3 = -k1
    tv0, tl0, sl0 = _gate_delays(T, td, ts, s, c1, vth, vl, vh, der)
    rce = rd * c1

    def evalf(Xa, act):
        t0, dt = Xa[:, 0], Xa[:, 1]
        tv, tl = tv0[act], tl0[act]
        dt = np.where(dt <= 0, (tl - tv) / 100.0, dt)
        Xa[:, 1] = dt
        dts = np.where(dt > 0, dt, 1e-12)
        F = np.stack([_yramp(tl, t0, dts, rce[act]) - vl[act], _yramp(tv, t0, dts, rce[act]) - vth[act]], 1)
        J = np.zeros((len(act), 2, 2))
        a0, a1, _ = _dyramp(tl, t0, dts, c1[act], rd[act])
        b0, b1, _ = _dyramp(tv, t0, dts, c1[act], rd[act])
        J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = a0, a1, b0, b1
        return F, J, ~(dt > 0), Xa
    dt0 = sl0 / (vh - vl)
    X = np.stack([tv0 + np.log(1.0 - vth) * rce - vth * dt0, dt0], 1)
    X, ok = _newton(evalf, X)
    t0, dt = X[:, 0], X[:, 1]
    vub = t0 + dt + c1 * (rd + rpi) * 2.0
    n = len(ii)
    valid = np.zeros(n, bool)
    delay = T(td, s, c1)
    tslew = np.where(ts >= 0, np.maximum(T(np.maximum(ts, 0), s, c1), 0.0), 0.0)
    slew = tslew.copy()
    vod = np.zeros(n)
    k4 = np.zeros(n)
    p2 = np.ones(n)
    gi = np.nonzero(ok & np.isfinite(k0 + k1 + k2 + k3 + p1) & (dt > 0))[0]
    if len(gi):
        args = (t0[gi], dt[gi], k0[gi], k1[gi], k2[gi], k3[gi], k4[gi], p1[gi], p2[gi])
        span = tslew[gi] * der[gi] / (vh[gi] - vl[gi])
        d50, o1 = _root(_vo, args, vth[gi], t0[gi].copy(), vub[gi].copy(), tv0[gi])
        tl, o2 = _root(_vo, args, vl[gi], t0[gi].copy(), d50.copy(), d50 - span * (vth[gi] - vl[gi]))
        th, o3 = _root(_vo, args, vh[gi], d50.copy(), vub[gi].copy(), d50 + span * (vh[gi] - vth[gi]))
        v = o1 & o2 & o3
        valid[gi] = v
        vod[gi] = d50
        delay[gi] = np.where(v, d50, delay[gi])                          # DmpZeroC2: delay = Vo vth crossing
        slew[gi] = np.where(v, (th - tl) / der[gi], tslew[gi])
    o = out
    o["alg"][ii] = np.where(valid, 1, 0)
    o["ceff"][ii], o["delay"][ii], o["slew"][ii] = c1, delay, slew
    o["t0"][ii], o["dt"][ii], o["vod"][ii], o["valid"][ii] = t0, dt, vod, valid
    for k, v in (("k0", k0), ("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4), ("p1", p1), ("p2", p2), ("vub", vub)):
        o[k][ii] = np.nan_to_num(v)


def dmp_load(dp, elm):
    """DmpAlg::loadDelaySlew: wire delay / load slew of loads (rows) from their driver's DMP state (dict of arrays)."""
    slew = dp["slew"]
    n = len(elm)
    wd = np.where(np.isfinite(elm), elm, 0.0)
    ls = slew.copy()
    if n == 0:
        return wd, ls
    use = (dp["alg"] >= 1) & dp["valid"] & np.isfinite(elm) & (elm > 0) & ~(elm < slew * 1e-3)
    gi = np.nonzero(use)[0]
    if len(gi):
        g = lambda k: dp[k][gi]
        p3 = 1.0 / elm[gi]
        args = (g("t0"), g("dt"), g("k0"), g("k1"), g("k2"), g("k3"), g("k4"), g("p1"), g("p2"), p3)
        lo = g("t0").copy()
        hi = g("vub") + 2.0 * elm[gi]
        e = elm[gi]
        sp = (g("slew") * g("der") + 2.2 * e) / (g("vh") - g("vl"))
        ld, o1 = _root(_vl, args, g("vth"), lo.copy(), hi.copy(), g("vod") + 0.7 * e)
        tl, o2 = _root(_vl, args, g("vl"), lo.copy(), ld.copy(), ld - sp * (g("vth") - g("vl")))
        th, o3 = _root(_vl, args, g("vh"), ld.copy(), hi.copy(), ld + sp * (g("vh") - g("vth")))
        ok = o1 & o2 & o3
        d1 = ld - g("vod")
        s1 = (th - tl) / g("der")
        d1 = np.where(d1 < 0, elm[gi], d1)
        s1 = np.where(s1 < g("slew"), g("slew"), s1)
        wd[gi] = np.where(ok, d1, elm[gi])
        ls[gi] = np.where(ok, s1, g("slew"))
    return wd, ls


# ════════════════════════════════════════════════════════════════════════════════════
#  the timer
# ════════════════════════════════════════════════════════════════════════════════════
class StaTimer:
    def __init__(self, D):
        self.D = D

    def analyze(self, seg, rat=True, margin=0.0, rc=None, keep=False):
        D = self.D
        T = D.lib.tab
        P, K = D.P, D.K
        t0 = time.time()
        rc = rc or build_rc(D, seg)
        t1 = time.time()
        AT = np.full((P, K, 2), NEG)
        SL = np.zeros((P, 2))
        AT[D.sp_pin, D.sp_cls, D.sp_rf] = D.sp_at
        rck = np.nonzero(D.reg_ck)[0]
        SL[rck] = D.ck_slew[rck]
        nE = len(D.ce_from)
        gate_d = np.zeros(nE)
        BEST = np.full((P, K, 2), -1, np.int64)
        wire_d = np.zeros((P, 2))
        # per driver pin & rf: DMP state of the max-slew arc (loads are computed from it)
        dps = {k: np.zeros((P, 2)) for k in Dmp.FIELDS}
        dps["alg"] = np.full((P, 2), -1, np.int64)
        dps["valid"] = np.zeros((P, 2), bool)
        net_rd = np.zeros(len(D.net_names))
        for lv in range(1, D.nlev):
            a, b = D.ce_lo[lv], D.ce_lo[lv + 1]
            if b > a:
                frm, to, irf, orf = D.ce_from[a:b], D.ce_to[a:b], D.ce_irf[a:b], D.ce_orf[a:b]
                net = D.pin_net[to]
                li = D.ce_lib[a:b]
                res = dmp_gate(T, D.ce_td[a:b], D.ce_ts[a:b], SL[frm, irf], rc.c2[net, orf], rc.rpi[net, orf],
                               rc.c1[net, orf], rc.par[net], rc.ctot[net, orf], D.L_vth[li, orf], D.L_vl[li, orf],
                               D.L_vh[li, orf], D.L_der[li])
                gate_d[a:b] = res["delay"]
                ga, gb = D.g_lo[lv], D.g_lo[lv + 1]
                st = D.g_start[ga:gb] - a
                gto, gorf = D.g_to[ga:gb], D.g_orf[ga:gb]
                smax = np.maximum.reduceat(res["slew"], st)
                SL[gto, gorf] = smax
                ar = np.arange(b - a)
                gid = D.ce_grp[a:b] - D.ce_grp[a]
                pick = np.minimum.reduceat(np.where(res["slew"] >= smax[gid], ar, b - a + 1), st)
                for k in Dmp.FIELDS:
                    dps[k][gto, gorf] = res[k][pick]
                np.maximum.at(net_rd, net[pick], res["rd"][pick])
                cand = AT[frm, :, irf] + res["delay"][:, None]
                amax = np.maximum.reduceat(cand, st, axis=0)
                AT[gto, :, gorf] = amax
                tight = cand >= amax[gid] - 1e-12
                BEST[gto, :, gorf] = np.minimum.reduceat(np.where(tight, (a + ar)[:, None], nE), st, axis=0)
            a, b = D.ne_lo[lv], D.ne_lo[lv + 1]
            if b > a:
                drv, snk = D.ne_drv[a:b], D.ne_snk[a:b]
                nb = b - a
                elm = rc.elm[snk].T.reshape(-1)                         # [rise loads..., fall loads...]
                dp = {k: dps[k][drv].T.reshape(-1) for k in Dmp.FIELDS}
                wd, ls = dmp_load(dp, elm)
                prt = np.tile(D.is_port[drv], 2)
                if prt.any():                                          # input port: dspf wire delay / slew
                    rfv = np.repeat(np.arange(2), nb)[prt]
                    e = elm[prt]
                    okp = np.isfinite(e)
                    vin, vlo, vhi = D.L_vin[D.def_lib, rfv], D.L_vl[D.def_lib, rfv], D.L_vh[D.def_lib, rfv]
                    wd[prt] = np.where(okp, -e * np.log(1.0 - vin), 0.0)
                    ls[prt] = np.tile(SL[drv], (1, 1)).T.reshape(-1)[prt] + np.where(
                        okp, e * np.log((1 - vlo) / (1 - vhi)) / D.L_der[D.def_lib], 0.0)
                wd, ls = wd.reshape(2, nb).T, ls.reshape(2, nb).T
                wire_d[snk] = wd
                SL[snk] = ls
                AT[snk] = AT[drv] + wd[:, None, :]
        t2 = time.time()
        # ---- endpoints
        R = D.ck_rows
        rp = R["pin"]
        setup = np.zeros((len(rp), 2))
        for rf, tk in ((0, "tr"), (1, "tf")):
            tid = R[tk]
            m = tid >= 0
            setup[m, rf] = T(tid[m], R["cks"][m], SL[rp[m], rf])
            # a check row without a table for this data transition: no check for that transition (type 0/1)
            setup[(~m) & (R["type"] <= 1), rf] = np.nan
        req = D.REL[:, R["evt"]].T                                      # [rows, K]
        reqk = req[:, :, None] - setup[:, None, :] - R["off"][:, None, None]   # [rows, K, 2]
        at = AT[rp]
        valid = (at > NEG / 2) & (req[:, :, None] < BIG / 2) & np.isfinite(reqk)
        sl = np.where(valid, reqk - at, BIG)
        rmin = sl.reshape(len(rp), -1).min(1)
        ep_sl = np.full(P, BIG)
        np.minimum.at(ep_sl, rp, rmin)
        ep = D.ep_pin
        e = ep_sl[ep]
        has = e < BIG / 2
        e = e[has]
        epv = ep[has]
        wns = float(min(0.0, e.min())) if len(e) else 0.0
        tns = float(np.minimum(e, 0.0).sum()) if len(e) else 0.0
        r = {"wns": wns, "tns": tns, "wns_raw": float(e.min()) if len(e) else 0.0, "nviol": int((e < 0).sum()),
             "n_ep": int(len(e)), "ep_pin": epv, "ep_slack": e, "t_rc": t1 - t0, "t_prop": t2 - t1,
             "net_ctot": rc.ctot.max(1), "net_rd_kohm": np.clip(net_rd * 1e3, 0.05, 20.0), "missing": int(rc.missing.sum())}
        # ---- INSTA-style TNS gradient: violating endpoints' worst paths, counted per pin (hard-max backward pass)
        thr = margin
        rows_v = np.nonzero(rmin < thr)[0]
        cnt = np.zeros((P, K, 2))
        if len(rows_v):
            # worst row per endpoint pin, then its worst (k, rf)
            vv = rmin[rows_v]
            o = np.lexsort((vv, rp[rows_v]))
            rows_v = rows_v[o]
            fr = np.r_[True, rp[rows_v][1:] != rp[rows_v][:-1]]
            rows_v = rows_v[fr]
            flat = sl[rows_v].reshape(len(rows_v), -1).argmin(1)
            np.add.at(cnt, (rp[rows_v], flat // 2, flat % 2), 1.0)
            for lv in range(D.nlev - 1, 0, -1):
                a, b = D.ne_lo[lv], D.ne_lo[lv + 1]
                if b > a:
                    snk, drv = D.ne_snk[a:b], D.ne_drv[a:b]
                    c = cnt[snk]
                    if c.any():
                        np.add.at(cnt, drv, c)
                ga, gb = D.g_lo[lv], D.g_lo[lv + 1]
                if gb > ga:
                    gto, gorf = D.g_to[ga:gb], D.g_orf[ga:gb]
                    c = cnt[gto, :, gorf]                                # [G, K]
                    gi, kk = np.nonzero(c > 0)
                    if len(gi):
                        eb = BEST[gto[gi], kk, gorf[gi]]
                        okb = eb < nE
                        gi, kk, eb = gi[okb], kk[okb], eb[okb]
                        np.add.at(cnt, (D.ce_from[eb], kk, D.ce_irf[eb]), c[gi, kk])
        net_cnt = np.zeros(len(D.net_names))
        drvp = D.net_drv
        okd = drvp >= 0
        net_cnt[okd] = cnt[drvp[okd]].sum((1, 2))
        r["net_cnt"] = net_cnt
        # worst path (pins) by argmax tracing
        if len(e):
            wp = int(epv[np.argmin(e)])
            rows_w = np.nonzero(rp == wp)[0]
            fl = sl[rows_w].reshape(len(rows_w), -1)
            j = np.unravel_index(np.argmin(fl), fl.shape)
            k, rf = j[1] // 2, j[1] % 2
            path, p = [wp], wp
            for _ in range(100000):
                if D.pin_is_load[p] and not D.is_port[p] or (D.is_port[p] and not D.pin_isdrvdir[p]):
                    q = D.net_drv[D.pin_net[p]]
                    if q < 0 or AT[q, k, rf] <= NEG / 2:
                        break
                    p = q
                else:
                    eb = BEST[p, k, rf]
                    if eb < 0 or eb >= nE:
                        break
                    p, rf = int(D.ce_from[eb]), int(D.ce_irf[eb])
                path.append(p)
            r["wns_path"] = path[::-1]
            on = np.zeros(len(D.net_names), bool)
            on[D.pin_net[[q for q in path if D.pin_is_drv[q]]]] = True
            r["net_on_wns"] = on
        else:
            r["wns_path"] = []
            r["net_on_wns"] = np.zeros(len(D.net_names), bool)
        if rat:
            RAT = np.full((P, K, 2), BIG)
            np.minimum.at(RAT, rp, np.where(valid, reqk, BIG))
            for lv in range(D.nlev - 1, 0, -1):
                a, b = D.ne_lo[lv], D.ne_lo[lv + 1]
                if b > a:
                    snk, drv = D.ne_snk[a:b], D.ne_drv[a:b]
                    np.minimum.at(RAT, drv, RAT[snk] - wire_d[snk][:, None, :])
                a, b = D.ce_lo[lv], D.ce_lo[lv + 1]
                if b > a:
                    v = RAT[D.ce_to[a:b], :, D.ce_orf[a:b]] - gate_d[a:b][:, None]
                    np.minimum.at(RAT, (D.ce_from[a:b], slice(None), D.ce_irf[a:b]), v)
            ok = (AT > NEG / 2) & (RAT < BIG / 2)
            ps = np.where(ok, RAT - AT, BIG).reshape(P, -1).min(1)
            r["pin_slack"] = ps
            ns = np.full(len(D.net_names), BIG)
            np.minimum.at(ns, D.pin_net, ps)
            r["net_slack"] = ns
        r["t_total"] = time.time() - t0
        if keep:
            r.update({"AT": AT, "SL": SL, "gate_d": gate_d, "wire_d": wire_d, "BEST": BEST, "rc": rc})
        return r


# ════════════════════════════════════════════════════════════════════════════════════
#  route reading (DEF net ids) and calibration against OpenROAD reports
# ════════════════════════════════════════════════════════════════════════════════════
def read_route(D, path):
    """route file -> int64 [S, 7] with DEF net ids (fast_timer.read_route_arrays on the .net index, remapped)."""
    nf_index = {nm: i for i, nm in enumerate(D.nf_names)}
    seg = FT.read_route_arrays(path, nf_index)
    seg[:, 0] = D.nf_net[seg[:, 0]]
    return seg[seg[:, 0] >= 0]


_RLINE = re.compile(r"^\s*(?:(\d+)\s+([-\d.]+)\s+)?([-\d.]+)\s+([-\d.]+)\s+([\^v])\s+(\S+)\s+\((\S+)\)\s*$")


def parse_crit_paths(path, max_detail=0):
    """crit_nets.tcl report -> list of dicts: endpoint, slack, group, and (first max_detail paths) the data-path pins
    with their arrival time / incremental delay / fanout / cap."""
    out = []
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("Startpoint:"):
                cur = {"start": line.split()[1], "pins": [], "frozen": False, "group": ""}
                continue
            if cur is None:
                continue
            if line.startswith("Path Group:"):
                cur["group"] = line.split(":", 1)[1].strip()
                continue
            if "data arrival time" in line:
                cur["frozen"] = True
                continue
            if not cur["frozen"]:
                m = _RLINE.match(line)
                if m:
                    cur["pins"].append((_key(m.group(6)), float(m.group(4)), float(m.group(3)),
                                        int(m.group(1)) if m.group(1) else -1,
                                        float(m.group(2)) if m.group(2) else float("nan"), m.group(5)))
                continue
            if "slack (" in line:
                try:
                    cur["slack"] = float(line.split()[0])
                except ValueError:
                    cur = None
                    continue
                cur["end"] = cur["pins"][-1][0] if cur["pins"] else None
                if len(out) >= max_detail:
                    cur["pins"] = cur["pins"][-1:]
                out.append(cur)
                cur = None
    return out


def calib(a):
    D = StaDesign(a.design, a.suite)
    Tm = StaTimer(D)
    sfx = "_blind" if a.suite == "blind" else ""
    rows = []
    for ti, tag in enumerate(a.tags):
        rt = os.path.join(ROOT, f"experiments/ispd25_routes{sfx}/{a.design}/{a.design}.{tag}.norm.route")
        ppa = os.path.join(ROOT, f"experiments/ispd25_ppa{sfx}/{a.design}/{a.design}.{tag}.norm.ppa.txt")
        crit = os.path.join(ROOT, f"experiments/ispd25_ppa{sfx}/{a.design}/{a.design}.{tag}.crit.rpt")
        if not (os.path.exists(rt) and os.path.exists(ppa)):
            log(f"skip {tag}: missing files")
            continue
        try:
            w_or, t_or = FT.parse_ppa(ppa)
        except Exception:
            log(f"skip {tag}: no wns/tns")
            continue
        t0 = time.time()
        seg = read_route(D, rt)
        tp = time.time() - t0
        r = Tm.analyze(seg, rat=False, keep=(ti < a.detail_routes))
        row = {"design": a.design, "suite": a.suite, "tag": tag, "wns_or": w_or, "tns_or": t_or, "wns_est": r["wns"],
               "tns_est": r["tns"], "nviol_est": r["nviol"], "missing": r["missing"], "t_parse": round(tp, 2),
               "t_rc": round(r["t_rc"], 2), "t_prop": round(r["t_prop"], 2)}
        if os.path.exists(crit):
            cps = parse_crit_paths(crit, max_detail=a.detail if ti < a.detail_routes else 0)
            est = np.full(D.P, np.nan)
            est[r["ep_pin"]] = r["ep_slack"]
            pairs = []
            seen = set()
            for cp in cps:
                pid = D._pid.get(cp["end"])
                if pid is None or cp["end"] in seen or not np.isfinite(est[pid]):
                    continue
                seen.add(cp["end"])
                pairs.append((cp["slack"], est[pid]))
            if pairs:
                so, se = np.array(pairs).T
                row.update({"ep_n": len(pairs), "ep_mae": float(np.mean(np.abs(se - so))),
                            "ep_bias": float(np.mean(se - so)),
                            "ep_r": float(np.corrcoef(so, se)[0, 1]) if len(pairs) > 2 and so.std() > 0 else float("nan")})
            # worst path in detail: our arrival at each pin vs OpenROAD's; net load vs OpenROAD's Cap column
            if ti < a.detail_routes and "AT" in r:
                for cp in cps[:a.detail]:
                    _detail(D, r, cp)
        rows.append(row)
        log(f"{tag}: OR wns {w_or:.4f} tns {t_or:.1f} | est wns {r['wns']:.4f} tns {r['tns']:.1f} ({r['nviol']} viol, "
            f"{r['missing']} missing pins) | endpoints {row.get('ep_n', 0)} mae {row.get('ep_mae', float('nan')):.4f} "
            f"bias {row.get('ep_bias', float('nan')):+.4f} r {row.get('ep_r', float('nan')):.4f} | parse {tp:.1f}s "
            f"rc {r['t_rc']:.1f}s prop {r['t_prop']:.1f}s")
    if a.csv and rows:
        import csv
        keys = list(rows[0].keys()) + [k for k in ("ep_n", "ep_mae", "ep_bias", "ep_r") if k not in rows[0]]
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            if new:
                w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in keys})
    if len(rows) >= 3:
        wo = np.array([x["wns_or"] for x in rows]); we = np.array([x["wns_est"] for x in rows])
        to = np.array([x["tns_or"] for x in rows]); te = np.array([x["tns_est"] for x in rows])
        cr = lambda u, v: np.corrcoef(u, v)[0, 1] if u.std() > 0 and v.std() > 0 else float("nan")
        log(f"SUMMARY {a.design}[{a.suite}] n={len(rows)} WNS r={cr(wo, we):.3f} MAE={np.mean(np.abs(we - wo)):.4f} "
            f"bias={np.mean(we - wo):+.4f} | TNS r={cr(to, te):.3f} MAE={np.mean(np.abs(te - to)):.1f} "
            f"rel={np.mean(np.abs(te - to)) / max(np.mean(np.abs(to)), 1e-9):.3f} | dWNS r={cr(wo - wo[0], we - we[0]):.3f} "
            f"dTNS r={cr(to - to[0], te - te[0]):.3f}")


def _detail(D, r, cp):
    """print our arrival vs OpenROAD's along one reported path (driver pins carry fanout / cap)."""
    AT, SL, rc = r["AT"], r["SL"], r["rc"]
    lines = []
    t_or0 = None
    for (nm, t_or, dl, fan, capv, edge) in cp["pins"]:
        pid = D._pid.get(nm)
        if pid is None:
            lines.append(f"    {nm}: not in netlist")
            continue
        rf = 0 if edge == "^" else 1
        a = AT[pid, :, rf]
        k = int(np.argmax(a))
        c = int(D.clk_of[pid]) if D.clk_of[pid] >= 0 else -1
        ours = a[k]
        cls = k
        tl = 0.0
        ck = D.sdc["clocks"]
        if cls // 2 < len(D.clk_names):
            cinfo = ck[D.clk_names[cls // 2]]
            tl = cinfo["wave"][cls % 2] + cinfo["lat"]
        ours_abs = ours + tl if ours > NEG / 2 else float("nan")
        net = D.pin_net[pid]
        extra = ""
        if fan >= 0:
            extra = (f" fan {fan} cap OR {capv:.3f} ours {rc.ctot[net, rf]:.3f} (wire {rc.ctot_wire[net]:.3f}, "
                     f"par {bool(rc.par[net])}, pi c2 {rc.c2[net, rf]:.3f} rpi {rc.rpi[net, rf]:.4f} c1 {rc.c1[net, rf]:.3f})")
        lines.append(f"    {edge} {nm}: OR t {t_or:.5f} (+{dl:.5f}) ours {ours_abs:.5f} slew {SL[pid, rf]:.4f}{extra}")
    log(f"  path {cp['start']} -> {cp.get('end')} [{cp['group']}] OR slack {cp.get('slack')}:\n" + "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════════════
#  self test (tiny synthetic design, no contest files)
# ════════════════════════════════════════════════════════════════════════════════════
_TINY_LIB = r"""
library (tiny) {
  time_unit : "1ns"; capacitive_load_unit (1,ff);
  slew_lower_threshold_pct_rise : 30.00; slew_upper_threshold_pct_rise : 70.00;
  slew_lower_threshold_pct_fall : 30.00; slew_upper_threshold_pct_fall : 70.00; slew_derate_from_library : 1.0;
  input_threshold_pct_rise : 50; input_threshold_pct_fall : 50; output_threshold_pct_rise : 50; output_threshold_pct_fall : 50;
  lu_table_template (T2) { variable_1 : input_net_transition; variable_2 : total_output_net_capacitance;
     index_1 ("0.01, 0.1"); index_2 ("1, 11"); }
  lu_table_template (C2) { variable_1 : related_pin_transition; variable_2 : constrained_pin_transition;
     index_1 ("0.01, 0.1"); index_2 ("0.01, 0.11"); }
  cell (INV) {
    pin (A) { direction : input; capacitance : 1.0; rise_capacitance : 1.0; fall_capacitance : 0.9; }
    pin (ZN) { direction : output; function : "!A";
      timing () { related_pin : "A"; timing_sense : negative_unate;
        cell_rise (T2) { values ("0.010, 0.050", "0.020, 0.060"); }
        cell_fall (T2) { values ("0.008, 0.040", "0.018, 0.050"); }
        rise_transition (T2) { values ("0.01, 0.09", "0.02, 0.10"); }
        fall_transition (T2) { values ("0.01, 0.07", "0.02, 0.08"); } } } }
  cell (AND2) {
    pin (A1) { direction : input; capacitance : 1.0; }
    pin (A2) { direction : input; capacitance : 1.0; }
    pin (ZN) { direction : output; function : "(A1 & A2)";
      timing () { related_pin : "A1"; timing_sense : positive_unate;
        cell_rise (T2) { values ("0.02, 0.06", "0.03, 0.07"); } cell_fall (T2) { values ("0.02, 0.06", "0.03, 0.07"); }
        rise_transition (T2) { values ("0.01, 0.09", "0.02, 0.10"); } fall_transition (T2) { values ("0.01, 0.09", "0.02, 0.10"); } }
      timing () { related_pin : "A2"; timing_sense : positive_unate;
        cell_rise (T2) { values ("0.02, 0.06", "0.03, 0.07"); } cell_fall (T2) { values ("0.02, 0.06", "0.03, 0.07"); }
        rise_transition (T2) { values ("0.01, 0.09", "0.02, 0.10"); } fall_transition (T2) { values ("0.01, 0.09", "0.02, 0.10"); } } } }
  cell (DFF) { ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; }
    pin (D) { direction : input; capacitance : 1.0;
      timing () { related_pin : "CK"; timing_type : setup_rising;
        rise_constraint (C2) { values ("0.02, 0.03", "0.02, 0.03"); }
        fall_constraint (C2) { values ("0.02, 0.03", "0.02, 0.03"); } } }
    pin (CK) { direction : input; capacitance : 1.0; clock : true; }
    pin (Q) { direction : output;
      timing () { related_pin : "CK"; timing_type : rising_edge;
        cell_rise (T2) { values ("0.05, 0.07", "0.06, 0.08"); }
        cell_fall (T2) { values ("0.05, 0.07", "0.06, 0.08"); }
        rise_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); }
        fall_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } } } }
  cell (DLL) { latch (IQ, IQN) { data_in : "D"; enable : "!GN"; }
    pin (D) { direction : input; capacitance : 1.0;
      timing () { related_pin : "GN"; timing_type : setup_rising;
        rise_constraint (C2) { values ("0.02, 0.03", "0.02, 0.03"); } fall_constraint (C2) { values ("0.02, 0.03", "0.02, 0.03"); } } }
    pin (GN) { direction : input; capacitance : 1.0; clock : true; }
    pin (Q) { direction : output;
      timing () { related_pin : "D"; timing_sense : positive_unate;
        cell_rise (T2) { values ("0.05, 0.07", "0.06, 0.08"); } cell_fall (T2) { values ("0.05, 0.07", "0.06, 0.08"); }
        rise_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } fall_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } }
      timing () { related_pin : "GN"; timing_type : falling_edge;
        cell_rise (T2) { values ("0.05, 0.07", "0.06, 0.08"); } cell_fall (T2) { values ("0.05, 0.07", "0.06, 0.08"); }
        rise_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } fall_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } } } }
}
library (tinypf) {
  time_unit : "1ns"; capacitive_load_unit (1,pf);
  lu_table_template (L1) { variable_1 : total_output_net_capacitance; index_1 ("0.001, 0.011"); }
  cell (PFBUF) { pin (A) { direction : input; capacitance : 0.002; }
    pin (Z) { direction : output; function : "A";
      timing () { related_pin : "A"; timing_sense : positive_unate;
        cell_rise (L1) { values ("0.01, 0.05"); } cell_fall (L1) { values ("0.01, 0.05"); } } } }
}
"""


def self_test():
    import tempfile
    ok = True
    lib = StaLib()
    lib._add(FT._parse_groups(_TINY_LIB))
    lib.tab.freeze()
    T = lib.tab
    inv = lib.cells["INV"]
    tcr = inv.arcs["ZN"][0][3]
    v1 = float(T([tcr], 0.01, 6.0)[0])                 # mid-load interpolation: 0.03
    v2 = float(T([tcr], 0.01, 21.0)[0])                # extrapolation: 0.09
    pf = lib.cells["PFBUF"]
    tpf = pf.arcs["Z"][0][3]
    v3 = float(T([tpf], 0.05, 6.0)[0])                 # 1-D load table in pF: 6 fF -> 0.03
    good = (abs(v1 - 0.03) < 1e-9 and abs(v2 - 0.09) < 1e-9 and abs(v3 - 0.03) < 1e-9 and abs(pf.pins["A"][1] - 2.0) < 1e-9
            and inv.pins["A"][2] == 0.9 and inv.arcs["ZN"][0][2] == 1 and lib.cells["DFF"].seq
            and lib.cells["AND2"].ftype == "and" and inv.ftype == "inv" and lib.cells["DLL"].latch)
    ok &= good
    print(f"  liberty: interp {v1:.4f} extrap {v2:.4f} pF 1-D {v3:.4f} pin cap {pf.pins['A'][1]} unate/seq/ftype: {good}")
    # ---- DMP on a pure capacitive load == table; on a resistive pi: Ceff < Ctot, wire delay > 0
    td = np.array([tcr]); ts = np.array([inv.arcs["ZN"][0][5]])
    one = lambda v: np.array([v], float)
    rcap = dmp_gate(T, td, ts, one(0.02), one(0.0), one(0.0), one(6.0), np.array([True]), one(6.0), one(0.5),
                    one(0.3), one(0.7), one(1.0))
    tab = float(T(td, 0.02, 6.0)[0])
    rpi = dmp_gate(T, td, ts, one(0.02), one(2.0), one(2.0), one(4.0), np.array([True]), one(6.0), one(0.5),
                   one(0.3), one(0.7), one(1.0))
    wd, ls = dmp_load({k: v for k, v in rpi.items()}, one(0.008))
    good = (abs(rcap["delay"][0] - tab) < 1e-12 and rcap["alg"][0] == 0 and rpi["alg"][0] == 2
            and 2.0 < rpi["ceff"][0] < 6.0 and rpi["delay"][0] < tab and wd[0] > 0 and ls[0] >= rpi["slew"][0])
    ok &= good
    print(f"  dmp: cap delay {rcap['delay'][0]:.5f} == table {tab:.5f}; pi Ceff {rpi['ceff'][0]:.3f} delay "
          f"{rpi['delay'][0]:.5f} slew {rpi['slew'][0]:.4f} valid {bool(rpi['valid'][0])}; load wd {wd[0]:.5f} ls {ls[0]:.4f}: {good}")
    rel = (setup_relation(1.4, 0.0, 2.8, 0.0), setup_relation(2.8, 1.4, 1.4, 0.0), setup_relation(6.0, 0.0, 6.0, 3.4),
           setup_relation(1.3, 0.0, 1.3, 0.0))
    good = np.allclose(rel, (1.4, 1.4, 3.4, 1.3))
    ok &= good
    print(f"  setup relations {rel}: {good}")
    # ---- tiny design: ff1/Q -> u1 (INV) -> ff2/D ; clock clk -> ff1/CK, ff2/CK ; in1 (input delay) -> g1/A2 ;
    #      clk -> g1/A1 (AND clock gate) -> lat/GN ; lat/D <- ff1/Q ; out1 <- u1/ZN (output delay)
    d = tempfile.mkdtemp(prefix="fs_")
    kit = os.path.join(d, "kit")
    os.makedirs(os.path.join(kit, "visible/tiny"))
    os.makedirs(os.path.join(kit, "NanGate45/lib"))
    open(os.path.join(kit, "NanGate45/lib/tiny.lib"), "w").write(_TINY_LIB)
    comps = [("ff1", "DFF"), ("ff2", "DFF"), ("u1", "INV"), ("g1", "AND2"), ("lat", "DLL")]
    nets = {"clk": [("PIN", "clk"), ("ff1", "CK"), ("ff2", "CK"), ("g1", "A1")],
            "n1": [("ff1", "Q"), ("u1", "A"), ("lat", "D")],
            "n2": [("u1", "ZN"), ("ff2", "D"), ("PIN", "out1")],
            "n3": [("PIN", "in1"), ("g1", "A2")],
            "gck": [("g1", "ZN"), ("lat", "GN")]}
    with open(os.path.join(kit, "visible/tiny/tiny.def"), "w") as fh:
        fh.write(f"COMPONENTS {len(comps)} ;\n" + "".join(f"- {a} {b} + PLACED ( 0 0 ) N ;\n" for a, b in comps)
                 + "END COMPONENTS\nPINS 3 ;\n- clk + NET clk + DIRECTION INPUT ;\n- in1 + NET n3 + DIRECTION INPUT ;\n"
                 "- out1 + NET n2 + DIRECTION OUTPUT ;\nEND PINS\n"
                 f"NETS {len(nets)} ;\n" + "".join(f"- {n}\n  " + " ".join(f"( {a} {b} )" for a, b in cs) + " ;\n"
                                                    for n, cs in nets.items()) + "END NETS\n")
    aps = {"ff1/Q": (0, 1, 1), "u1/A": (0, 2, 1), "lat/D": (0, 1, 2), "u1/ZN": (0, 2, 1), "ff2/D": (0, 52, 1),
           "out1": (0, 52, 2)}
    with open(os.path.join(kit, "visible/tiny/tiny.net"), "w") as fh:
        for n in ("n1", "n2"):
            fh.write(f"{n}\n(\n" + "".join(f"{(a + '/' + b) if a != 'PIN' else b}, -1e-11, [{aps[(a + '/' + b) if a != 'PIN' else b]}]\n"
                                           for a, b in nets[n]) + ")\n")
    open(os.path.join(kit, "visible/tiny/tiny.sdc"), "w").write(
        "create_clock [get_ports {clk}] -name core -period 1.0 -waveform {0 0.5}\n"
        "set_input_delay 0.1 -clock [get_clocks {core}] [get_ports {in1}]\n"
        "set_output_delay 0.2 -clock [get_clocks {core}] [get_ports {out1}]\n"
        "set_clock_uncertainty 0.05 [get_clocks {core}]\n")
    D = StaDesign("tiny", "visible", lib=lib, kit=kit, verbose=False)
    pid = D._pid
    good = (D.reg_ck[pid["ff1/CK"]] and D.reg_ck[pid["lat/GN"]] and D.cnet[pid["g1/A1"]] and not D.cnet[pid["g1/A2"]]
            and D.n_gate_checks == 1 and set(D.pin_name[i] for i in D.ep_pin) == {"ff2/D", "lat/D", "g1/A2", "out1"})
    ok &= good
    print(f"  clocks / gating / endpoints {sorted(D.pin_name[i] for i in D.ep_pin)}: {good}")
    g = lambda v: v * GRID + GRID // 2
    uv = lambda x, y, z0, z1: "".join(f"{g(x)} {g(y)} metal{z} {g(x)} {g(y)} metal{z + 1}\n" for z in range(z0, z1))

    def route(z):
        r = (f"n1\n(\n" + uv(1, 1, 1, 3) + f"{g(1)} {g(1)} metal3 {g(2)} {g(1)} metal3\n" + uv(2, 1, 1, 3)
             + f"{g(1)} {g(1)} metal2 {g(1)} {g(2)} metal2\n" + uv(1, 2, 1, 2) + ")\n")
        r += (f"n2\n(\n" + uv(2, 1, 1, z) + f"{g(2)} {g(1)} metal{z} {g(52)} {g(1)} metal{z}\n" + uv(52, 1, 1, z)
              + f"{g(52)} {g(1)} metal2 {g(52)} {g(2)} metal2\n" + uv(52, 2, 1, 2) + ")\n")
        p = os.path.join(d, f"r{z}.route")
        open(p, "w").write(r)
        return p
    Tm = StaTimer(D)
    res = {z: Tm.analyze(read_route(D, route(z)), keep=True) for z in (3, 9)}
    r3, r9 = res[3], res[9]
    # hand check: ff2/D arrival = CK->Q + wire n1 + INV + wire n2 ; slack = 1.0 - setup - 0.05 - arrival
    sl3 = dict(zip([D.pin_name[i] for i in r3["ep_pin"]], r3["ep_slack"]))
    good = (r3["missing"] == 0 and sl3["ff2/D"] < sl3["lat/D"] + 1.0 and r9["wns_raw"] > r3["wns_raw"] - 1e-12
            and abs(sl3["g1/A2"] - (1.0 - 0.05 - 0.1)) < 1e-9 and r3["pin_slack"][pid["u1/A"]] < BIG / 2
            and r3["net_cnt"].sum() >= 0 and len(r3["wns_path"]) >= 3)
    ok &= good
    print(f"  tiny STA: slacks {{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(sl3.items()))}}}, metal9 WNS "
          f"{r9['wns_raw']:.4f} vs metal3 {r3['wns_raw']:.4f}, worst path {[D.pin_name[q] for q in r3['wns_path']]}: {good}")
    # RC of n2 on metal3 by hand: drv pin -v1- (2,1,m2) -v2- (2,1,m3) -wire- (52,1,m3) -v2- (52,1,m2) -v1- ff2/D ;
    # (52,1,m2) -m2 wire- (52,2,m2) -v1- out1 (port, 0 fF)
    rc = r3["rc"]
    n2 = D.net_index["n2"]
    rr, cc = D.layer_rc[2]
    Rw, Cw = rr * 50 * UM_PER_GC, cc * 50 * UM_PER_GC
    R2, C2 = D.layer_rc[1, 0] * UM_PER_GC, D.layer_rc[1, 1] * UM_PER_GC
    v1, v2 = D.via_r[0], D.via_r[1]
    ctot = Cw + C2 + 1.0
    hand = (v1 * ctot + v2 * ctot + Rw * (Cw / 2 + C2 + 1.0) + v2 * (C2 + 1.0) + v1 * 1.0) * 1e-3
    e = rc.elm[pid["ff2/D"], 0]
    good = abs(e - hand) < 1e-12 and abs(rc.ctot[n2, 0] - ctot) < 1e-9
    ok &= good
    print(f"  Elmore ff2/D {e * 1e3:.4f} ps (hand {hand * 1e3:.4f} ps), Ctot n2 {rc.ctot[n2, 0]:.4f} fF (hand {ctot:.4f}): {good}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("time", "calib"), default="time")
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--csv")
    ap.add_argument("--detail", type=int, default=0, help="calib: print our vs OpenROAD arrival along N worst paths")
    ap.add_argument("--detail_routes", type=int, default=1, help="calib: detail only for the first N routes")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "calib":
        calib(a)
        return 0
    t0 = time.time()
    D = StaDesign(a.design, a.suite)
    r = StaTimer(D).analyze(read_route(D, a.route))
    log(f"WNS {r['wns']:.4f} ns TNS {r['tns']:.1f} ns ({r['nviol']} / {r['n_ep']} violating endpoints, {r['missing']} "
        f"missing pins) rc {r['t_rc']:.1f}s prop {r['t_prop']:.1f}s total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
