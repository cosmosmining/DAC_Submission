#!/usr/bin/env python3
"""fast_timer.py — STA-free fast timing model for ISPD'25 global routes (no OpenROAD, no HeteroSTA).

Inputs are the contest inputs only: the .net (per-pin PRE-ROUTING slack estimates, seconds), the DEF
COMPONENTS section (cell of every instance), the Liberty files (pin capacitances, NLDM delay / transition /
constraint tables, sequential cells, clock pins), setRC.tcl + the tech LEF (wire R/C per layer, via R per
cut layer), the SDC (only set_ideal_network), and the 3-D route being timed.

Model (all vectorized with numpy; one pass per route):
  1. RC tree of every net exactly as OpenROAD's `estimate_parasitics -global_routing` builds it from the route:
     one resistor per wire segment (split where the same net has a segment end, via or pin access point on it,
     like seg_normalize.py), C/2 at both ends, per-layer R/C of setRC.tcl, cut resistance of every via from the
     tech LEF, Liberty input-pin capacitance at the sink nodes.
  2. Driving-point admittance moments (y1, y2, y3) of every net -> O'Brien-Savarino pi model -> effective
     capacitance of the driver (Qian-Pullela-Pillage style, ramp driver), gate delay and output slew from the
     driver cell's NLDM tables at C_eff. Elmore delay driver -> sink; the sink waveform is the driver ramp
     through a single pole tau = Elmore (the dmp_ceff_elmore construction), giving wire delay and sink slew.
  3. The same on the REFERENCE parasitics the .net slacks were estimated with: the same tree on metal3 R/C,
     no via resistance (placement-stage `set_wire_rc -signal -layer metal3`); ideal nets (SDC
     set_ideal_network) have zero reference delay.
  4. delta(sink) = [gate delay(C_eff) - ref] + [wire delay - ref] + [next-stage delay change from the sink
     slew] (or, at a timing endpoint, the change of the setup/recovery time with the data slew).
  5. Path structure from the .net slacks alone: a sink's predecessor is its net's driver; a combinational
     output's predecessor is the input pin of the same instance with the smallest slack (RAT(in) = RAT(out) -
     d(in->out), so that input is the one on the worst path through the output); sequential outputs and input
     ports are roots. delta_AT = sum of deltas along that worst-path forest (pointer jumping).
  6. Estimated endpoint slack = .net slack - delta_AT; WNS / TNS over the timing endpoints (data / async pins
     of sequential cells and constrained output ports).

  python fast_timer.py --design ariane --suite visible --route R.norm.route [--crit_out est.crit.rpt]
  python fast_timer.py calib --design ariane --suite visible --tags instantgr instantgr_tr1 ... --csv out.csv
  python fast_timer.py --self_test
"""
import argparse
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "ispd_bench/ISPD2025/ISPD2025_benchmarks")
GRID = 4200                       # DBU per GCell
DBU_PER_UM = 2000.0
UM_PER_GC = GRID / DBU_PER_UM     # 2.1 um
INF = 1e30
# fallbacks (NanGate45/setRC.tcl, kOhm/um and fF/um; tech LEF cut RESISTANCE in Ohm -> kOhm)
LAYER_RC = [(5.4286e-3, 7.41819e-2), (3.5714e-3, 6.74606e-2), (3.5714e-3, 8.88758e-2), (1.5e-3, 1.07121e-1),
            (1.5e-3, 1.08964e-1), (1.5e-3, 1.02044e-1), (1.875e-4, 1.10436e-1), (1.875e-4, 9.69714e-2),
            (3.75e-5, 3.6864e-2), (3.75e-5, 2.8042e-2)]
VIA_R = [5e-3, 5e-3, 5e-3, 3e-3, 3e-3, 3e-3, 1e-3, 1e-3, 5e-4]
REF_LAYER = 2                     # metal3 (set_wire_rc -signal -layer metal3)
S_IN = float(os.environ.get("FT_S_IN", "0.03"))       # ns, assumed input slew of every driver
CLK_SLEW = 0.0                                        # ideal clocks


def log(m):
    print(f"[fast_timer {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _key(name):
    return name.replace("\\", "")


# ════════════════════════════════════════════════════════════════════════════════════
#  technology: setRC.tcl + tech LEF
# ════════════════════════════════════════════════════════════════════════════════════
def read_tech(kit=KIT):
    rc = list(LAYER_RC)
    vr = list(VIA_R)
    try:
        for line in open(os.path.join(kit, "NanGate45/setRC.tcl")):
            m = re.match(r"\s*set_layer_rc\s+-layer\s+metal(\d+)\s+-resistance\s+(\S+)\s+-capacitance\s+(\S+)", line)
            if m:
                z = int(m.group(1)) - 1
                if 0 <= z < len(rc):
                    rc[z] = (float(m.group(2)), float(m.group(3)))
    except OSError:
        pass
    try:
        cur = None
        for line in open(os.path.join(kit, "NanGate45/lef/NangateOpenCellLibrary.tech.lef")):
            m = re.match(r"\s*LAYER\s+via(\d+)\s*$", line)
            if m:
                cur = int(m.group(1)) - 1
                continue
            if re.match(r"\s*END\s+via\d+", line):
                cur = None
                continue
            m = re.match(r"\s*RESISTANCE\s+([\d.eE+-]+)\s*;", line)
            if m and cur is not None and 0 <= cur < len(vr):
                vr[cur] = float(m.group(1)) / 1000.0          # Ohm -> kOhm
    except OSError:
        pass
    return np.array(rc, np.float64), np.array(vr, np.float64)


# ════════════════════════════════════════════════════════════════════════════════════
#  Liberty
# ════════════════════════════════════════════════════════════════════════════════════
_TOK = re.compile(r'"[^"]*"|[^\s{}():;,"]+|[{}():;,]')


class _G:
    __slots__ = ("name", "args", "attrs", "kids")

    def __init__(self, name, args):
        self.name, self.args, self.attrs, self.kids = name, args, {}, []


def _parse_groups(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S).replace("\\\r\n", " ").replace("\\\n", " ")
    toks = _TOK.findall(text)
    root = _G("root", [])
    st = [root]
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t == "}":
            if len(st) > 1:
                st.pop()
            i += 1
            continue
        if t in ("{", ")", ":", ";", ",", "("):
            i += 1
            continue
        nx = toks[i + 1] if i + 1 < n else ""
        if nx == ":":
            j = i + 2
            vals = []
            while j < n and toks[j] not in (";", "}", "{"):
                vals.append(toks[j].strip('"'))
                j += 1
            st[-1].attrs[t] = " ".join(vals)
            i = j + 1 if (j < n and toks[j] == ";") else j
        elif nx == "(":
            j = i + 2
            args = []
            while j < n and toks[j] != ")":
                if toks[j] != ",":
                    args.append(toks[j].strip('"'))
                j += 1
            j += 1
            if j < n and toks[j] == "{":
                g = _G(t, args)
                st[-1].kids.append(g)
                st.append(g)
                i = j + 1
            else:
                st[-1].attrs[t] = args
                i = j + 1 if (j < n and toks[j] == ";") else j
        else:
            i += 1
    return root


def _floats(args):
    out = []
    for a in args if isinstance(args, list) else [args]:
        out += [float(v) for v in a.replace(",", " ").split()]
    return out


DELAY_TT_SKIP = {"three_state_enable", "three_state_disable", "three_state_enable_rise",
                 "three_state_enable_fall", "three_state_disable_rise", "three_state_disable_fall"}
CHECK_TT = {"setup_rising", "setup_falling", "recovery_rising", "recovery_falling"}
EDGE_TT = {"rising_edge", "falling_edge"}


class Table:
    """NLDM table; evaluated with bilinear interpolation and linear extrapolation (like OpenSTA)."""
    __slots__ = ("i1", "i2", "v", "swap")

    def __init__(self, g, templates, want=("input_net_transition", "total_output_net_capacitance")):
        tm = templates.get(g.args[0] if g.args else "", {})
        i1 = _floats(g.attrs["index_1"]) if "index_1" in g.attrs else tm.get("index_1", [])
        i2 = _floats(g.attrs["index_2"]) if "index_2" in g.attrs else tm.get("index_2", [])
        v = _floats(g.attrs.get("values", ["0"]))
        v1, v2 = tm.get("variable_1", want[0]), tm.get("variable_2", want[1])
        self.i1 = np.array(i1 if i1 else [0.0], np.float64)
        self.i2 = np.array(i2 if i2 else [0.0], np.float64)
        n1, n2 = len(self.i1), len(self.i2)
        if len(v) == n1 * n2:
            self.v = np.array(v, np.float64).reshape(n1, n2)
        elif len(v) == 1:
            self.i1, self.i2, self.v = np.array([0.0]), np.array([0.0]), np.array(v, np.float64).reshape(1, 1)
        else:                                    # 1-D table written as one row
            self.i2 = np.array([0.0])
            self.v = np.array(v[:n1], np.float64).reshape(-1, 1)
        # self.swap: index_1 is the SECOND wanted variable (e.g. load first)
        self.swap = (v1 == want[1]) or (v2 == want[0] and v1 != want[0])

    def __call__(self, a, b):
        """a = first wanted variable (slew / related slew), b = second (load / constrained slew)."""
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        if self.swap:
            a, b = b, a
        return _bilinear(self.i1, self.i2, self.v, a, b)


def _interp_idx(ix, x):
    n = len(ix)
    if n == 1:
        return np.zeros(x.shape, np.int64), np.zeros(x.shape)
    k = np.clip(np.searchsorted(ix, x) - 1, 0, n - 2)
    den = ix[k + 1] - ix[k]
    return k, (x - ix[k]) / np.where(den == 0, 1.0, den)


def _bilinear(i1, i2, v, a, b):
    a, b = np.broadcast_arrays(a, b)
    ka, fa = _interp_idx(i1, a)
    kb, fb = _interp_idx(i2, b)
    n1, n2 = v.shape
    ka1 = np.minimum(ka + 1, n1 - 1)
    kb1 = np.minimum(kb + 1, n2 - 1)
    return ((1 - fa) * (1 - fb) * v[ka, kb] + (1 - fa) * fb * v[ka, kb1]
            + fa * (1 - fb) * v[ka1, kb] + fa * fb * v[ka1, kb1])


class PinModel:
    __slots__ = ("direction", "cap", "clock", "arcs", "checks", "edge")

    def __init__(self):
        self.direction, self.cap, self.clock = "input", 0.0, False
        self.arcs = []        # output pins: (related_pin, [delay tables], [transition tables])
        self.checks = []      # input pins: setup/recovery constraint tables (related slew, data slew)
        self.edge = False     # output launched by a clock edge (sequential startpoint)


class Lib:
    """cells[cell][pin] -> PinModel; seq[cell] -> bool; thresholds."""

    def __init__(self):
        self.cells, self.seq = {}, {}
        self.vl, self.vh, self.derate = 0.3, 0.7, 1.0

    @staticmethod
    def load(lib_dir=os.path.join(KIT, "NanGate45/lib"), cache=None):
        files = sorted(os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".lib"))
        stamp = [(f, os.path.getmtime(f)) for f in files]
        cache = cache or os.path.join(ROOT, "experiments/fastmode/lib_model.pkl")
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as fh:
                    st, lib = pickle.load(fh)
                if st == stamp:
                    return lib
            except Exception:
                pass
        lib = Lib()
        for f in files:
            lib._add(_parse_groups(open(f).read()), primary="Nangate" in os.path.basename(f))
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = f"{cache}.tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump((stamp, lib), fh)
            os.replace(tmp, cache)
        except OSError:
            pass
        return lib

    def _add(self, root, primary=True):
        for libg in root.kids:
            if libg.name != "library":
                continue
            a = libg.attrs
            try:
                if primary:
                    self.vl = float(a.get("slew_lower_threshold_pct_rise", self.vl * 100)) / 100
                    self.vh = float(a.get("slew_upper_threshold_pct_rise", self.vh * 100)) / 100
                    self.derate = float(a.get("slew_derate_from_library", self.derate))
            except ValueError:
                pass
            templates = {}
            for g in libg.kids:
                if g.name in ("lu_table_template", "power_lut_template") and g.args:
                    t = {k: g.attrs[k] for k in ("variable_1", "variable_2") if k in g.attrs}
                    for k in ("index_1", "index_2"):
                        if k in g.attrs:
                            t[k] = _floats(g.attrs[k])
                    templates[g.args[0]] = t
            for cg in libg.kids:
                if cg.name == "cell" and cg.args:
                    self._cell(cg, templates)

    def _cell(self, cg, templates):
        cname = cg.args[0]
        pins = {}
        seq = any(k.name in ("ff", "latch", "statetable", "ff_bank", "latch_bank", "memory") for k in cg.kids)

        def one_pin(pg, name, inherit=None):
            pm = PinModel()
            src = dict(inherit.attrs) if inherit is not None else {}
            src.update(pg.attrs)
            pm.direction = src.get("direction", "input")
            c = src.get("capacitance")
            if c is None:
                cs = [float(src[k]) for k in ("rise_capacitance", "fall_capacitance") if k in src]
                c = max(cs) if cs else 0.0
            pm.cap = float(c)
            pm.clock = src.get("clock", "false") == "true"
            tgs = [k for k in pg.kids if k.name == "timing"]
            if inherit is not None:
                tgs += [k for k in inherit.kids if k.name == "timing"]
            for tg in tgs:
                tt = tg.attrs.get("timing_type", "combinational")
                rel = tg.attrs.get("related_pin", "")
                tabs = {k.name: k for k in tg.kids}
                if tt in CHECK_TT:
                    cons = [Table(tabs[k], templates, ("related_pin_transition", "constrained_pin_transition"))
                            for k in ("rise_constraint", "fall_constraint") if k in tabs]
                    if cons:
                        pm.checks.append(cons)
                elif tt in DELAY_TT_SKIP or tt.startswith(("hold", "removal", "min_pulse", "skew", "non_seq",
                                                           "nochange", "max_clock", "minimum_period")):
                    continue
                else:
                    dl = [Table(tabs[k], templates) for k in ("cell_rise", "cell_fall") if k in tabs]
                    tr = [Table(tabs[k], templates) for k in ("rise_transition", "fall_transition") if k in tabs]
                    if dl:
                        pm.arcs.append((rel, dl, tr))
                        if tt in EDGE_TT:
                            pm.edge = True
            return pm

        for pg in cg.kids:
            if pg.name == "pin" and pg.args:
                for nm in pg.args:
                    pins[nm] = one_pin(pg, nm)
            elif pg.name in ("bus", "bundle") and pg.args:
                sub = [k for k in pg.kids if k.name == "pin"]
                pins[pg.args[0]] = one_pin(pg, pg.args[0])          # bus-level model ('addr_in')
                for sp in sub:
                    for nm in sp.args:
                        pins[nm] = one_pin(sp, nm, inherit=pg)
        if any(p.edge for p in pins.values()):
            seq = True
        self.cells[cname] = pins
        self.seq[cname] = seq

    def pin(self, cell, pin):
        pins = self.cells.get(cell)
        if pins is None:
            return None
        pm = pins.get(pin)
        if pm is None and "[" in pin:
            pm = pins.get(pin.split("[")[0])
        return pm


# ════════════════════════════════════════════════════════════════════════════════════
#  design inputs
# ════════════════════════════════════════════════════════════════════════════════════
def parse_def_components(path):
    """-> {instance (backslashes removed): cell}; reads only the COMPONENTS section."""
    comps = {}
    rx = re.compile(r"^-\s+(\S+)\s+(\S+)", re.M)
    buf = []
    inside = False
    with open(path) as fh:
        for line in fh:
            if not inside:
                if line.startswith("COMPONENTS"):
                    inside = True
                continue
            if line.startswith("END COMPONENTS"):
                break
            if line.startswith("-"):
                buf.append(line)
                if len(buf) >= 200000:
                    for m in rx.finditer("".join(buf)):
                        comps[_key(m.group(1))] = m.group(2)
                    buf = []
    for m in rx.finditer("".join(buf)):
        comps[_key(m.group(1))] = m.group(2)
    return comps


_PIN = re.compile(r"^\s*([^,\s]+)\s*,\s*([^,\s]+)\s*,\s*\[([^\]]*)\]", re.M)
_AP = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_net_fast(path):
    """ISPD'25 .net -> dict of flat arrays: net names, per-pin net id / name / slack (ns) / access points."""
    text = open(path).read()
    names, pin_net, pin_name, pin_slack, ap_cnt, ap_str = [], [], [], [], [], []
    pos = 0
    n = len(text)
    while True:
        # net header: a line that is not '(' / ')' and not a pin line
        j = text.find("\n", pos)
        if j < 0:
            break
        head = text[pos:j].strip()
        pos = j + 1
        if not head or head in ("(", ")"):
            continue
        k = text.find("\n)", pos)
        if k < 0:
            k = n
        body = text[pos:k]
        pos = k + 2
        nid = len(names)
        names.append(head.split()[0])
        for m in _PIN.finditer(body):
            pin_net.append(nid)
            pin_name.append(_key(m.group(1)))
            try:
                s = float(m.group(2))
            except ValueError:
                s = INF
            pin_slack.append(s * 1e9 if abs(s) < 1e20 else INF)
            a = m.group(3)
            ap_str.append(a)
            ap_cnt.append(a.count("("))
    aps = np.array(_AP.findall(" ".join(ap_str)), dtype=np.int64).reshape(-1, 3)
    return {"names": names, "pin_net": np.array(pin_net, np.int64), "pin_name": pin_name,
            "pin_slack": np.array(pin_slack, np.float64), "ap_cnt": np.array(ap_cnt, np.int64),
            "ap": aps}                                   # ap rows: (layer, x, y)


def parse_ideal(sdc_path):
    """set_ideal_network targets -> (ports, pins) sets (backslashes removed)."""
    ports, pins = set(), set()
    try:
        for line in open(sdc_path):
            if "set_ideal_network" not in line:
                continue
            for kind, body in re.findall(r"get_(ports|pins)\s+\{([^}]*)\}", line):
                for t in body.split():
                    (ports if kind == "ports" else pins).add(_key(t.strip("{}")))
    except OSError:
        pass
    return ports, pins


class Design:
    """Route-independent part of the timer (built once per design)."""

    def __init__(self, design, suite="visible", lib=None, kit=KIT, verbose=True):
        t0 = time.time()
        self.design, self.suite = design, suite
        d = os.path.join(kit, suite, design)
        self.lib = lib or Lib.load(os.path.join(kit, "NanGate45/lib"))
        self.layer_rc, self.via_r = read_tech(kit)
        comps = parse_def_components(os.path.join(d, design + ".def"))
        N = parse_net_fast(os.path.join(d, design + ".net"))
        t1 = time.time()
        self.net_names = N["names"]
        self.net_index = {nm: i for i, nm in enumerate(self.net_names)}
        nn = len(self.net_names)
        pin_net, names, slack = N["pin_net"], N["pin_name"], N["pin_slack"]
        P = len(names)
        # per-pin library model
        cellpin_id = {}
        cp_models = []
        pin_cp = np.full(P, -1, np.int64)
        pin_inst = np.full(P, -1, np.int64)
        inst_id = {}
        inst_cell = []
        for i, nm in enumerate(names):
            k = nm.rfind("/")
            if k < 0:
                continue                                       # top-level port
            inst, pn = nm[:k], nm[k + 1:]
            cell = comps.get(inst)
            ii = inst_id.get(inst)
            if ii is None:
                ii = inst_id[inst] = len(inst_cell)
                inst_cell.append(cell)
            pin_inst[i] = ii
            if cell is None:
                continue
            key = (cell, pn)
            c = cellpin_id.get(key)
            if c is None:
                pm = self.lib.pin(cell, pn)
                c = cellpin_id[key] = len(cp_models)
                cp_models.append((cell, pn, pm))
            pin_cp[i] = c
        ncp = len(cp_models)
        cp_dir = np.array([0 if (m[2] is None or m[2].direction == "input") else
                           (1 if m[2].direction == "output" else 2) for m in cp_models] + [0], np.int64)
        cp_cap = np.array([m[2].cap if m[2] is not None else 0.0 for m in cp_models] + [0.0])
        cp_clock = np.array([bool(m[2] is not None and m[2].clock) for m in cp_models] + [False])
        cp_check = np.array([bool(m[2] is not None and m[2].checks) for m in cp_models] + [False])
        cp_edge = np.array([bool(m[2] is not None and m[2].edge) for m in cp_models] + [False])
        cp_seq = np.array([bool(self.lib.seq.get(m[0], False)) for m in cp_models] + [False])
        cpi = np.where(pin_cp >= 0, pin_cp, ncp)
        is_port = pin_inst < 0
        self.pin_dir = cp_dir[cpi]
        self.pin_cap = cp_cap[cpi]
        self.pin_clock = cp_clock[cpi]
        # driver per net: the first output pin; else the first port (input port); else the first pin
        idx = np.arange(P)
        outp = (self.pin_dir >= 1) & ~is_port
        drv = np.full(nn, P, np.int64)
        np.minimum.at(drv, pin_net[outp], idx[outp])
        nodrv = drv == P
        prt = np.full(nn, P, np.int64)
        np.minimum.at(prt, pin_net[is_port], idx[is_port])
        firstpin = np.full(nn, P, np.int64)
        np.minimum.at(firstpin, pin_net, idx)
        drv = np.where(nodrv, np.where(prt < P, prt, firstpin), drv)
        drv[drv == P] = -1
        self.net_drv = drv
        is_drv = np.zeros(P, bool)
        is_drv[drv[drv >= 0]] = True
        self.pin_is_drv = is_drv
        self.pin_net, self.pin_name, self.pin_slack = pin_net, names, slack
        self.pin_inst, self.pin_cp, self.cp_models = pin_inst, pin_cp, cp_models
        self.inst_cell = inst_cell
        self.ap_cnt, self.ap = N["ap_cnt"], N["ap"]
        if int(self.ap_cnt.sum()) != len(self.ap):
            raise ValueError(f"access point count mismatch {int(self.ap_cnt.sum())} vs {len(self.ap)}")
        self.ap_pin = np.repeat(np.arange(P), self.ap_cnt)
        # endpoints: constrained data/async pins of sequential cells (sinks), constrained output ports
        sink = ~is_drv
        self.pin_endpoint = sink & ((cp_check[cpi] & ~self.pin_clock) | (is_port & (slack < INF / 2)))
        # clock nets (a sink is a clock pin): ideal clocks -> no data timing
        net_clock = np.zeros(nn, bool)
        net_clock[pin_net[sink & self.pin_clock]] = True
        self.net_clock = net_clock
        # ideal nets from the SDC (zero reference delay)
        iports, ipins = parse_ideal(os.path.join(d, design + ".sdc"))
        net_ideal = np.zeros(nn, bool)
        if iports or ipins:
            for i in np.nonzero(is_drv)[0]:
                nm = names[i]
                if nm in ipins or (is_port[i] and nm in iports):
                    net_ideal[pin_net[i]] = True
        self.net_ideal = net_ideal
        # worst-path forest: sinks -> driver; combinational outputs -> min-slack input of the instance
        pred = np.full(P, -1, np.int64)
        pred[sink] = drv[pin_net[sink]]
        start_out = is_drv & (is_port | cp_edge[cpi] | cp_seq[cpi] | (pin_cp < 0))
        cand = sink & ~is_port & ~self.pin_clock & (pin_inst >= 0)
        ci = np.nonzero(cand)[0]
        if len(ci):
            o = np.lexsort((slack[ci], pin_inst[ci]))
            ci = ci[o]
            firsti = np.r_[True, pin_inst[ci][1:] != pin_inst[ci][:-1]]
            best_in = np.full(len(inst_cell), -1, np.int64)
            best_in[pin_inst[ci][firsti]] = ci[firsti]
            comb_out = is_drv & ~start_out & (pin_inst >= 0)
            pred[comb_out] = best_in[pin_inst[comb_out]]
        pred[pred == idx] = -1
        self.pred = pred
        self.start_out = start_out
        # driver models per net
        self.net_drv_cp = np.where(drv >= 0, pin_cp[np.maximum(drv, 0)], -1)
        self.net_drv_cp[is_port[np.maximum(drv, 0)] | (drv < 0)] = -1
        # TNS endpoint count (for the score's /N_end, use the organizers' N_end instead)
        if verbose:
            log(f"{design}[{suite}]: {nn:,} nets, {P:,} pins, {len(inst_cell):,} instances, {ncp:,} cell pins "
                f"({sum(m[2] is None for m in cp_models)} unknown), {int(self.pin_endpoint.sum()):,} endpoints, "
                f"{int(net_clock.sum()):,} clock nets, {int(net_ideal.sum())} ideal nets; .net min slack "
                f"{np.min(slack[slack < INF / 2]):.4f} ns (parse {t1 - t0:.1f}s, build {time.time() - t1:.1f}s)")
        self._prep_tables()

    # ------------------------------------------------------------------ per cell-pin table groups
    def _prep_tables(self):
        """for every output cell-pin: list of delay / transition tables (averaged over arcs and rise/fall);
        for every input cell-pin: next-stage arcs (tables whose related pin is this input) and checks."""
        self.cp_out = {}
        self.cp_next = {}
        self.cp_checks = {}
        lib = self.lib
        for c, (cell, pn, pm) in enumerate(self.cp_models):
            if pm is None:
                continue
            if pm.direction != "input" and pm.arcs:
                self.cp_out[c] = ([t for _, dl, _ in pm.arcs for t in dl], [t for _, _, tr in pm.arcs for t in tr])
            if pm.direction == "input":
                if pm.checks:
                    self.cp_checks[c] = [t for cons in pm.checks for t in cons]
                outs = []
                for opn, opm in lib.cells.get(cell, {}).items():
                    if opm.direction == "input":
                        continue
                    for rel, dl, _ in opm.arcs:
                        if pn in rel.split():
                            outs += dl
                if outs:
                    self.cp_next[c] = outs


def _groups(ids):
    """yield (id, indices) for every distinct id (one sort instead of one mask per id)."""
    ids = np.asarray(ids)
    if not len(ids):
        return
    o = np.argsort(ids, kind="stable")
    s = ids[o]
    b = np.r_[0, np.nonzero(s[1:] != s[:-1])[0] + 1, len(s)]
    for k in range(len(b) - 1):
        yield s[b[k]], o[b[k]:b[k + 1]]


def _avg_tables(tabs, a, b):
    if not tabs:
        return np.zeros(np.broadcast(a, b).shape)
    acc = tabs[0](a, b)
    for t in tabs[1:]:
        acc = acc + t(a, b)
    return acc / len(tabs)


# ════════════════════════════════════════════════════════════════════════════════════
#  route -> RC trees (vectorized over all nets)
# ════════════════════════════════════════════════════════════════════════════════════
_BLOCK = re.compile(r"^(\S+)[ \t]*\n\([ \t]*\n([^()]*?)^\)", re.M)


def read_route_arrays(path, net_index):
    """route file -> int64 array [S, 7]: net, x1, y1, z1, x2, y2, z2 (GCells, layer 0-based)."""
    text = open(path).read()
    ids, bodies, counts = [], [], []
    miss = 0
    for m in _BLOCK.finditer(text):
        i = net_index.get(m.group(1))
        if i is None:
            miss += 1
            continue
        b = m.group(2)
        c = b.count("metal") // 2
        if c:
            ids.append(i)
            bodies.append(b)
            counts.append(c)
    big = " ".join(bodies).replace("metal", " ").replace("\n", " ")
    vals = np.fromstring(big, dtype=np.int64, sep=" ") if big.strip() else np.zeros(0, np.int64)
    seg = vals.reshape(-1, 6)
    if len(seg) != sum(counts):
        raise ValueError(f"{path}: parsed {len(seg)} segments, expected {sum(counts)}")
    out = np.empty((len(seg), 7), np.int64)
    out[:, 0] = np.repeat(np.array(ids, np.int64), counts)
    out[:, 1] = seg[:, 0] // GRID
    out[:, 2] = seg[:, 1] // GRID
    out[:, 3] = seg[:, 2] - 1
    out[:, 4] = seg[:, 3] // GRID
    out[:, 5] = seg[:, 4] // GRID
    out[:, 6] = seg[:, 5] - 1
    if miss:
        log(f"warning: {miss} route nets not in the .net file")
    return out


class RCResult:
    pass


DEBUG = os.environ.get("FT_DEBUG", "0") == "1"


def _tick(msg, t):
    if DEBUG:
        log(f"  {msg}: {time.time() - t:.2f}s")
    return time.time()


def build_rc(D, seg, X=None, Y=None):
    """RC trees for all nets of a route; returns per-node / per-pin arrays (see RCResult fields)."""
    tk = time.time()
    nets = seg[:, 0]
    x1, y1, z1, x2, y2, z2 = (seg[:, k] for k in range(1, 7))
    xl, xh = np.minimum(x1, x2), np.maximum(x1, x2)
    yl, yh = np.minimum(y1, y2), np.maximum(y1, y2)
    zl, zh = np.minimum(z1, z2), np.maximum(z1, z2)
    isvia = zl != zh
    iswire = (~isvia) & ((xl != xh) | (yl != yh))
    L = 16
    X = int(max(xh.max(initial=0), D.ap[:, 1].max(initial=0)) + 2) if X is None else X
    Y = int(max(yh.max(initial=0), D.ap[:, 2].max(initial=0)) + 2) if Y is None else Y

    def key(n, x, y, z):
        return ((n * L + z) * X + x) * Y + y

    # ---- points that split wires: wire ends, every layer of every via, pin access points
    vi = np.nonzero(isvia)[0]
    vspan = zh[vi] - zl[vi] + 1
    vrep = np.repeat(vi, vspan)
    vz = zl[vrep] + (np.arange(len(vrep)) - np.repeat(np.cumsum(vspan) - vspan, vspan))
    wi = np.nonzero(iswire)[0]
    pn = np.concatenate([nets[wi], nets[wi], nets[vrep], D.pin_net[D.ap_pin]])
    px = np.concatenate([xl[wi], xh[wi], x1[vrep], D.ap[:, 1]])
    py = np.concatenate([yl[wi], yh[wi], y1[vrep], D.ap[:, 2]])
    pz = np.concatenate([z1[wi], z1[wi], vz, D.ap[:, 0]])
    # ---- split wires (horizontal: line key (n, z, y), coordinate x; vertical: (n, z, x), coordinate y)
    sub_n, sub_z, sub_ax, sub_ay, sub_bx, sub_by = [], [], [], [], [], []
    for horiz in (True, False):
        w = wi[(yl[wi] == yh[wi])] if horiz else wi[(xl[wi] == xh[wi]) & (yl[wi] != yh[wi])]
        if not len(w):
            continue
        if horiz:
            lk_p, t_p = (pn * L + pz) * (Y + 1) + py, px
            lk_w, ta, tb, fx = (nets[w] * L + z1[w]) * (Y + 1) + y1[w], xl[w], xh[w], y1[w]
        else:
            lk_p, t_p = (pn * L + pz) * (X + 1) + px, py
            lk_w, ta, tb, fx = (nets[w] * L + z1[w]) * (X + 1) + x1[w], yl[w], yh[w], x1[w]
        T = max(X, Y) + 2
        ck = np.unique(lk_p * T + t_p)
        lo = np.searchsorted(ck, lk_w * T + ta, side="right")
        hi = np.searchsorted(ck, lk_w * T + tb, side="left")
        m = np.maximum(hi - lo, 0)
        nsub = m + 1
        rep = np.repeat(np.arange(len(w)), nsub)
        kk = np.arange(len(rep)) - np.repeat(np.cumsum(nsub) - nsub, nsub)
        inner = ck[np.minimum(lo[rep] + kk - 1, len(ck) - 1)] % T
        start = np.where(kk == 0, ta[rep], inner)
        inner2 = ck[np.minimum(lo[rep] + kk, len(ck) - 1)] % T
        end = np.where(kk == m[rep], tb[rep], inner2)
        sub_n.append(nets[w][rep])
        sub_z.append(z1[w][rep])
        if horiz:
            sub_ax.append(start); sub_bx.append(end)
            sub_ay.append(fx[rep]); sub_by.append(fx[rep])
        else:
            sub_ay.append(start); sub_by.append(end)
            sub_ax.append(fx[rep]); sub_bx.append(fx[rep])
    cat = lambda l: np.concatenate(l) if l else np.zeros(0, np.int64)
    wn, wz, wax, way, wbx, wby = map(cat, (sub_n, sub_z, sub_ax, sub_ay, sub_bx, sub_by))
    wlen = np.abs(wbx - wax) + np.abs(wby - way)
    tk = _tick(f"split {len(wn):,} sub-wires", tk)
    # unit vias
    vn, vx, vy = nets[vrep], x1[vrep], y1[vrep]
    first = np.r_[True, vrep[1:] != vrep[:-1]] if len(vrep) else np.zeros(0, bool)
    vbot = ~np.r_[vrep[1:] != vrep[:-1], True] if len(vrep) else np.zeros(0, bool)  # not the top layer
    # edges of unit vias: (z) -> (z+1) for every layer but the top of each via
    ue = np.nonzero(vbot)[0]
    e_un = vn[ue]
    e_ka = key(vn[ue], vx[ue], vy[ue], vz[ue])
    e_kb = key(vn[ue], vx[ue], vy[ue], vz[ue] + 1)
    e_vz = vz[ue]
    # ---- nodes
    k_wa = key(wn, wax, way, wz)
    k_wb = key(wn, wbx, wby, wz)
    allk = np.concatenate([k_wa, k_wb, e_ka, e_kb])
    uk = np.unique(allk)
    NN = len(uk)
    ia_w, ib_w = np.searchsorted(uk, k_wa), np.searchsorted(uk, k_wb)
    ia_v, ib_v = np.searchsorted(uk, e_ka), np.searchsorted(uk, e_kb)
    rc, vr = D.layer_rc, D.via_r
    R_w = rc[wz, 0] * wlen * UM_PER_GC
    C_w = rc[wz, 1] * wlen * UM_PER_GC
    R_wref = rc[REF_LAYER, 0] * wlen * UM_PER_GC
    C_wref = rc[REF_LAYER, 1] * wlen * UM_PER_GC
    R_v = vr[np.clip(e_vz, 0, len(vr) - 1)]
    eu = np.concatenate([ia_w, ia_v])
    ev = np.concatenate([ib_w, ib_v])
    eR = np.concatenate([R_w, R_v])
    eRr = np.concatenate([R_wref, np.zeros(len(R_v))])
    node_net = uk // (L * X * Y)
    # node caps (pi model: C/2 at both ends)
    cn = np.zeros(NN)
    cnr = np.zeros(NN)
    np.add.at(cn, ia_w, C_w / 2); np.add.at(cn, ib_w, C_w / 2)
    np.add.at(cnr, ia_w, C_wref / 2); np.add.at(cnr, ib_w, C_wref / 2)
    wire_c_net = np.bincount(wn, weights=C_w, minlength=len(D.net_names))
    # ---- pin -> node: first access point that is a route node; else the layer above; else any layer there
    P = len(D.pin_name)
    pin_node = np.full(P, -1, np.int64)
    apn = D.pin_net[D.ap_pin]
    for dz in (0, 1):
        kq = key(apn, D.ap[:, 1], D.ap[:, 2], np.minimum(D.ap[:, 0] + dz, L - 1))
        pos = np.searchsorted(uk, kq)
        ok = (pos < NN) & (uk[np.minimum(pos, NN - 1)] == kq)
        ok &= pin_node[D.ap_pin] < 0
        # first matching AP per pin
        cand = np.nonzero(ok)[0]
        if len(cand):
            pins_c = D.ap_pin[cand]
            firstc = np.r_[True, pins_c[1:] != pins_c[:-1]]
            pin_node[pins_c[firstc]] = pos[cand[firstc]]
    miss = np.nonzero(pin_node < 0)[0]
    if len(miss):                         # any layer at the AP's GCell (lowest node there)
        xy = (node_net * X + (uk // Y) % X) * Y + uk % Y
        zq = (uk // (X * Y)) % L
        o = np.lexsort((zq, xy))
        xys = xy[o]
        mm = np.isin(D.ap_pin, miss)
        apm = np.nonzero(mm)[0]
        q = (apn[apm] * X + D.ap[apm, 1]) * Y + D.ap[apm, 2]
        pos = np.searchsorted(xys, q)
        ok = (pos < NN) & (xys[np.minimum(pos, NN - 1)] == q)
        cand = apm[ok]
        if len(cand):
            pins_c = D.ap_pin[cand]
            firstc = np.r_[True, pins_c[1:] != pins_c[:-1]]
            pin_node[pins_c[firstc]] = o[pos[ok][firstc]]
    # sink pin caps
    sinkm = (~D.pin_is_drv) & (pin_node >= 0)
    np.add.at(cn, pin_node[sinkm], D.pin_cap[sinkm])
    np.add.at(cnr, pin_node[sinkm], D.pin_cap[sinkm])
    tk = _tick(f"nodes {NN:,} + pins", tk)
    # ---- BFS from every driver node (super source = node NN)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order
    drv_pin = D.net_drv
    has_drv = (drv_pin >= 0)
    dnode = np.full(len(D.net_names), -1, np.int64)
    dnode[has_drv] = pin_node[drv_pin[has_drv]]
    roots = np.unique(dnode[dnode >= 0])
    su = np.concatenate([eu, np.full(len(roots), NN)])
    sv = np.concatenate([ev, roots])
    G = coo_matrix((np.ones(len(su), np.float32), (su, sv)), shape=(NN + 1, NN + 1)).tocsr()
    order, predn = breadth_first_order(G, NN, directed=False, return_predecessors=True)
    order = order[1:]                                     # drop the super source
    par = predn[:NN].astype(np.int64)                     # -9999 unreachable, NN = root
    # edge resistance parent -> node (min over parallel edges)
    a_, b_ = np.minimum(eu, ev), np.maximum(eu, ev)
    ek = a_ * (NN + 1) + b_
    o = np.lexsort((eR, ek))
    ek, eRs, eRrs = ek[o], eR[o], eRr[o]
    keep = np.r_[True, ek[1:] != ek[:-1]]
    ek, eRs, eRrs = ek[keep], eRs[keep], eRrs[keep]
    nodes = order
    p = par[nodes]
    inner = p < NN
    q = np.minimum(nodes, np.where(inner, p, nodes)) * (NN + 1) + np.maximum(nodes, np.where(inner, p, nodes))
    pos = np.clip(np.searchsorted(ek, q), 0, max(len(ek) - 1, 0))
    Rp = np.zeros(NN)
    Rpr = np.zeros(NN)
    if len(ek):
        Rp[nodes[inner]] = eRs[pos[inner]]
        Rpr[nodes[inner]] = eRrs[pos[inner]]
    # depth via BFS order (parents come before children)
    depth = np.zeros(NN, np.int64)
    reach = np.zeros(NN, bool)
    reach[nodes] = True
    # pointer jumping for depth
    anc = np.where(inner, p, -1)
    anc_full = np.full(NN, -1, np.int64)
    anc_full[nodes] = anc
    dist = (anc_full >= 0).astype(np.int64)
    a2 = anc_full.copy()
    for _ in range(64):
        live = a2 >= 0
        if not live.any():
            break
        dist = dist + np.where(live, dist[np.maximum(a2, 0)], 0)
        a2 = np.where(live, a2[np.maximum(a2, 0)], -1)
    depth = dist
    tk = _tick(f"bfs + depth (max {int(depth.max(initial=0))})", tk)
    res = RCResult()
    res.NN, res.node_net, res.pin_node, res.reach = NN, node_net, pin_node, reach
    res.wire_c_net = wire_c_net
    res.dnode = dnode
    # ---- moments + Elmore, ours and reference
    lv_order = np.argsort(np.where(reach, depth, -1), kind="stable")
    lv_order = lv_order[reach[lv_order]]
    dsort = depth[lv_order]
    bounds = np.r_[0, np.nonzero(np.diff(dsort))[0] + 1, len(dsort)]
    parent = np.full(NN, -1, np.int64)
    parent[nodes[inner]] = p[inner]
    for tag, cnode, Rpar in (("", cn, Rp), ("_ref", cnr, Rpr)):
        y1 = cnode.copy()
        y2 = np.zeros(NN)
        y3 = np.zeros(NN)
        for b in range(len(bounds) - 2, -1, -1):        # deepest level first
            lvl = lv_order[bounds[b]:bounds[b + 1]]
            pp = parent[lvl]
            m = pp >= 0
            lvl, pp = lvl[m], pp[m]
            if not len(lvl):
                continue
            R = Rpar[lvl]
            a1, a2_, a3 = y1[lvl], y2[lvl], y3[lvl]
            np.add.at(y1, pp, a1)
            np.add.at(y2, pp, a2_ - R * a1 * a1)
            np.add.at(y3, pp, a3 - 2 * R * a1 * a2_ + R * R * a1 ** 3)
        elm = np.zeros(NN)
        for b in range(len(bounds) - 1):                # root level first
            lvl = lv_order[bounds[b]:bounds[b + 1]]
            pp = parent[lvl]
            m = pp >= 0
            elm[lvl[m]] = elm[pp[m]] + Rpar[lvl[m]] * y1[lvl[m]] / 1000.0   # kOhm*fF = ps -> ns
        setattr(res, "y1" + tag, y1)
        setattr(res, "y2" + tag, y2)
        setattr(res, "y3" + tag, y3)
        setattr(res, "elm" + tag, elm)
    tk = _tick(f"moments/elmore over {len(bounds) - 1} levels", tk)
    return res


# ════════════════════════════════════════════════════════════════════════════════════
#  delay model
# ════════════════════════════════════════════════════════════════════════════════════
def _ceff(y1, y2, y3, Tr):
    """effective capacitance of a pi model (C1 - R - C2 from moments) for a ramp of duration Tr (ns)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        C2 = np.where((y3 > 0) & (y2 < 0), y2 * y2 / y3, 0.0)
        C2 = np.clip(np.nan_to_num(C2), 0.0, y1)
        C1 = y1 - C2
        Rpi = np.where((y3 > 0) & (y2 < 0), -y3 * y3 / (y2 ** 3), 0.0)
        tau = np.nan_to_num(Rpi * C2 / 1000.0)            # ns
        x = np.where(tau > 0, Tr / np.maximum(tau, 1e-12), np.inf)
        k = np.where(tau > 0, 1.0 - (1.0 - np.exp(-np.minimum(x, 700))) / np.maximum(x, 1e-12), 1.0)
    return C1 + np.clip(k, 0.0, 1.0) * C2


def _ramp_pole(Tr, tau, v):
    """time at which a 0->1 ramp of duration Tr, through a single pole tau, reaches v (vectorized bisection)."""
    Tr = np.maximum(Tr, 1e-6)
    lo = np.zeros_like(Tr)
    hi = Tr + 30.0 * tau + 1e-6

    def resp(t):
        with np.errstate(over="ignore", invalid="ignore"):
            r1 = (t - tau * (1 - np.exp(-t / np.maximum(tau, 1e-12)))) / Tr
            r2 = 1 - tau / Tr * np.exp(-np.minimum(t, 1e6) / np.maximum(tau, 1e-12)) \
                * (np.exp(np.minimum(Tr / np.maximum(tau, 1e-12), 700)) - 1)
        return np.where(t <= Tr, r1, r2)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        up = resp(mid) >= v
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def sink_response(Tr, tau, vl, vh):
    """(wire delay, sink 10/90-style slew span) for driver ramp Tr through tau; small tau -> (tau, 0 extra)."""
    delay = tau.copy()
    span = (vh - vl) * Tr
    big = tau > 0.02 * Tr
    if big.any():
        t5 = _ramp_pole(Tr[big], tau[big], 0.5)
        tl = _ramp_pole(Tr[big], tau[big], vl)
        th = _ramp_pole(Tr[big], tau[big], vh)
        delay[big] = t5 - 0.5 * Tr[big]
        span = span.copy()
        span[big] = th - tl
    return delay, span


class Timer:
    """timing of routes of one design (Design built once)."""

    def __init__(self, D, s_in=S_IN, k_next=1.0, k_check=1.0):
        self.D = D
        self.s_in, self.k_next, self.k_check = s_in, k_next, k_check

    def _gate(self, cp_ids, Cload, slew_in):
        """mean gate delay / output slew of driver cell-pins at given loads (ns)."""
        d = np.zeros(len(cp_ids))
        s = np.full(len(cp_ids), slew_in if np.isscalar(slew_in) else 0.0)
        for c, m in _groups(cp_ids):
            if c < 0 or c not in self.D.cp_out:
                continue
            dl, tr = self.D.cp_out[c]
            si = slew_in if np.isscalar(slew_in) else slew_in[m]
            d[m] = _avg_tables(dl, si, Cload[m])
            if tr:
                s[m] = _avg_tables(tr, si, Cload[m])
        return d, s

    def analyze(self, seg, keep=False):
        D = self.D
        lib = D.lib
        t0 = time.time()
        rc = build_rc(D, seg)
        t1 = time.time()
        nn = len(D.net_names)
        ok = (rc.dnode >= 0)
        out = {}
        for tag in ("", "_ref"):
            y1 = np.zeros(nn); y2 = np.zeros(nn); y3 = np.zeros(nn)
            y1[ok] = getattr(rc, "y1" + tag)[rc.dnode[ok]]
            y2[ok] = getattr(rc, "y2" + tag)[rc.dnode[ok]]
            y3[ok] = getattr(rc, "y3" + tag)[rc.dnode[ok]]
            # unrouted nets: load = sink pin caps only
            nr = ~ok
            if nr.any():
                pc = np.bincount(D.pin_net[~D.pin_is_drv], weights=D.pin_cap[~D.pin_is_drv], minlength=nn)
                y1[nr] = pc[nr]
            if tag == "_ref":
                y1 = np.where(D.net_ideal, 0.0, y1)
                y2 = np.where(D.net_ideal, 0.0, y2)
                y3 = np.where(D.net_ideal, 0.0, y3)
            Ce = y1.copy()
            cps = D.net_drv_cp
            for _ in range(3):
                dg, sg = self._gate(cps, Ce, self.s_in)
                Tr = np.maximum(sg * lib.derate / max(lib.vh - lib.vl, 1e-3), 1e-4)
                Ce = _ceff(y1, y2, y3, Tr)
            dg, sg = self._gate(cps, Ce, self.s_in)
            if tag == "_ref":
                dg = np.where(D.net_ideal, 0.0, dg)
            Tr = np.maximum(sg * lib.derate / max(lib.vh - lib.vl, 1e-3), 1e-4)
            out["Ctot" + tag], out["Ceff" + tag], out["dg" + tag], out["sg" + tag], out["Tr" + tag] = y1, Ce, dg, sg, Tr
        # per sink: wire delay + slew at the sink
        P = len(D.pin_name)
        sink = (~D.pin_is_drv) & (rc.pin_node >= 0) & ok[D.pin_net]
        si = np.nonzero(sink)[0]
        net_si = D.pin_net[si]
        res = {}
        for tag in ("", "_ref"):
            tau = getattr(rc, "elm" + tag)[rc.pin_node[si]]
            if tag == "_ref":
                tau = np.where(D.net_ideal[net_si], 0.0, tau)
            Tr = out["Tr" + tag][net_si]
            wd, span = sink_response(Tr, tau, lib.vl, lib.vh)
            res["wd" + tag] = wd
            res["slew" + tag] = span / max(lib.derate, 1e-6)
        # delta per sink pin
        delta = np.zeros(P)
        dgate = out["dg"] - out["dg_ref"]
        dlt = dgate[net_si] + (res["wd"] - res["wd_ref"])
        # next stage / check changes from the sink slew
        cp = D.pin_cp[si]
        extra = np.zeros(len(si))
        for c, m in _groups(cp):
            if c < 0:
                continue
            if c in D.cp_checks and D.pin_endpoint[si[m]].any():
                tabs = D.cp_checks[c]
                extra[m] = self.k_check * (_avg_tables(tabs, CLK_SLEW, res["slew"][m])
                                           - _avg_tables(tabs, CLK_SLEW, res["slew_ref"][m]))
            elif c in D.cp_next:
                tabs = D.cp_next[c]
                cl = 5.0
                extra[m] = self.k_next * (_avg_tables(tabs, res["slew"][m], cl)
                                          - _avg_tables(tabs, res["slew_ref"][m], cl))
        dlt = dlt + extra
        clk = D.net_clock[net_si]
        dlt = np.where(clk, 0.0, dlt)
        delta[si] = dlt
        # accumulate along the worst-path forest
        dat = delta.copy()
        anc = D.pred.copy()
        for _ in range(64):
            live = anc >= 0
            if not live.any():
                break
            dat = dat + np.where(live, dat[np.maximum(anc, 0)], 0.0)
            anc = np.where(live, anc[np.maximum(anc, 0)], -1)
        ep = D.pin_endpoint & (D.pin_slack < INF / 2)
        est = np.full(P, INF)
        est[ep] = D.pin_slack[ep] - dat[ep]
        e = est[ep]
        wns = float(min(0.0, e.min())) if len(e) else 0.0
        tns = float(np.minimum(e, 0.0).sum()) if len(e) else 0.0
        t2 = time.time()
        r = {"wns": wns, "tns": tns, "wns_raw": float(e.min()) if len(e) else 0.0,
             "nviol": int((e < 0).sum()), "t_rc": t1 - t0, "t_delay": t2 - t1,
             "net_wire_c": rc.wire_c_net, "net_ctot": out["Ctot"], "net_ceff": out["Ceff"],
             "net_dg": out["dg"], "net_dgate_delta": dgate, "pin_delta": delta, "pin_dat": dat,
             "pin_est": est, "pin_node": rc.pin_node}
        if keep:
            r["rc"] = rc
            r["sink_wd"], r["sink_idx"] = res["wd"], si
        return r

    # ------------------------------------------------------------------ outputs for the optimizers
    def trace(self, pin):
        D = self.D
        path = []
        seen = 0
        while pin >= 0 and seen < 100000:
            path.append(pin)
            pin = D.pred[pin]
            seen += 1
        return path[::-1]

    def write_crit(self, r, path, npaths=5000, slack_max=0.0):
        """OpenSTA-like report (crit_nets.tcl format) of the worst traced path of each of the npaths worst
        estimated endpoints with slack < slack_max: timing_promote.parse_crit / parse_net_rc / dtla.build read it."""
        D = self.D
        est = r["pin_est"]
        cand = np.nonzero(est < slack_max)[0]
        cand = cand[np.argsort(est[cand], kind="stable")][:npaths]
        fan = np.bincount(D.pin_net[~D.pin_is_drv], minlength=len(D.net_names))
        ctot = r["net_ctot"]
        # effective driver resistance (kOhm): slope of the Liberty delay over load, per driver
        rdrv = self._rdrv(r)
        with open(path, "w") as fh:
            for e in cand:
                pins = self.trace(int(e))
                fh.write(f"Startpoint: {D.pin_name[pins[0]]}\nEndpoint: {D.pin_name[e]}\nPath Group: est\n"
                         f"Path Type: max\n\nFanout        Cap      Delay       Time   Description\n")
                t = 0.0
                for q in pins:
                    if not D.pin_is_drv[q]:
                        continue
                    n = D.pin_net[q]
                    if D.net_clock[n]:
                        continue
                    cap = max(float(ctot[n]), 1e-3)
                    dl = rdrv[n] * cap / 1000.0
                    t += dl
                    cell = D.inst_cell[D.pin_inst[q]] if D.pin_inst[q] >= 0 else "port"
                    nm = D.pin_name[q]
                    fh.write(f"  {int(fan[n]):5d} {cap:10.5f} {dl:10.5f} {t:10.5f} ^ {nm} ({cell})\n"
                             f"                                          {D.net_names[n]} (net)\n")
                fh.write(f"                              {float(est[e]):.5f}   slack (VIOLATED)\n\n\n")
        return len(cand)

    def _rdrv(self, r):
        D = self.D
        C = r["net_ceff"]
        d1, _ = self._gate(D.net_drv_cp, C, self.s_in)
        d2, _ = self._gate(D.net_drv_cp, C + 10.0, self.s_in)
        return np.clip((d2 - d1) / 10.0 * 1000.0, 0.05, 20.0)      # ns/fF -> kOhm


# ════════════════════════════════════════════════════════════════════════════════════
#  calibration against OpenROAD STA reports
# ════════════════════════════════════════════════════════════════════════════════════
def parse_crit_endpoints(path):
    """crit_nets.tcl report -> {endpoint pin (backslashes removed): slack ns}."""
    out = {}
    last_pin, frozen = None, False
    rx = re.compile(r"[\^v] (\S+) \(\S+\)\s*$")
    with open(path) as fh:
        for line in fh:
            if line.startswith("Startpoint:"):
                last_pin, frozen = None, False
                continue
            if "data arrival time" in line:
                frozen = True                    # the endpoint is the last pin of the arrival section
                continue
            if not frozen:
                m = rx.search(line)
                if m:
                    last_pin = m.group(1)
                continue
            if "slack (" in line and last_pin is not None:
                try:
                    out.setdefault(_key(last_pin), float(line.split()[0]))
                except ValueError:
                    pass
                last_pin, frozen = None, False
    return out


def parse_ppa(path):
    t = open(path).read()
    g = lambda k: float(re.search(rf"^{k}\s+(-?[\d.eE+-]+)", t, re.M).group(1))
    return g("wns"), g("tns")


def calib(a):
    import ispd25_score as SC
    D = Design(a.design, a.suite)
    T = Timer(D)
    sfx = "_blind" if a.suite == "blind" else ""
    rows = []
    for tag in a.tags:
        rt = os.path.join(ROOT, f"experiments/ispd25_routes{sfx}/{a.design}/{a.design}.{tag}.norm.route")
        ppa = os.path.join(ROOT, f"experiments/ispd25_ppa{sfx}/{a.design}/{a.design}.{tag}.norm.ppa.txt")
        crit = os.path.join(ROOT, f"experiments/ispd25_ppa{sfx}/{a.design}/{a.design}.{tag}.crit.rpt")
        if not (os.path.exists(rt) and os.path.exists(ppa)):
            log(f"skip {tag}: missing files")
            continue
        try:
            w_or, t_or = parse_ppa(ppa)
        except Exception:
            log(f"skip {tag}: no wns/tns in {ppa}")
            continue
        t0 = time.time()
        seg = read_route_arrays(rt, D.net_index)
        tp = time.time() - t0
        r = T.analyze(seg)
        row = {"design": a.design, "suite": a.suite, "tag": tag, "wns_or": w_or, "tns_or": t_or,
               "wns_est": r["wns"], "tns_est": r["tns"], "nviol_est": r["nviol"],
               "t_parse": round(tp, 2), "t_rc": round(r["t_rc"], 2), "t_delay": round(r["t_delay"], 2)}
        if os.path.exists(crit):
            ce = parse_crit_endpoints(crit)
            idx = {nm: i for i, nm in enumerate(D.pin_name)} if not hasattr(D, "_pidx") else D._pidx
            D._pidx = idx
            pairs = [(s, r["pin_est"][idx[nm]]) for nm, s in ce.items() if nm in idx and r["pin_est"][idx[nm]] < INF / 2]
            if pairs:
                so, se = np.array(pairs).T
                row.update({"ep_n": len(pairs), "ep_mae": float(np.mean(np.abs(se - so))),
                            "ep_bias": float(np.mean(se - so)),
                            "ep_r": float(np.corrcoef(so, se)[0, 1]) if len(pairs) > 2 else float("nan")})
        rows.append(row)
        log(f"{tag}: OR wns {w_or:.4f} tns {t_or:.1f} | est wns {r['wns']:.4f} tns {r['tns']:.1f} "
            f"({r['nviol']} viol) | endpoints {row.get('ep_n', 0)} mae {row.get('ep_mae', float('nan')):.4f} "
            f"bias {row.get('ep_bias', float('nan')):+.4f} r {row.get('ep_r', float('nan')):.3f} "
            f"| parse {tp:.1f}s rc {r['t_rc']:.1f}s delay {r['t_delay']:.1f}s")
    if a.csv and rows:
        import csv
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + [k for k in ("ep_n", "ep_mae", "ep_bias", "ep_r")
                                                                        if k not in rows[0]])
            if new:
                w.writeheader()
            for row in rows:
                w.writerow(row)
    if len(rows) >= 3:
        wo = np.array([x["wns_or"] for x in rows]); we = np.array([x["wns_est"] for x in rows])
        to = np.array([x["tns_or"] for x in rows]); te = np.array([x["tns_est"] for x in rows])
        cw = np.corrcoef(wo, we)[0, 1] if wo.std() > 0 and we.std() > 0 else float("nan")
        ct = np.corrcoef(to, te)[0, 1] if to.std() > 0 and te.std() > 0 else float("nan")
        log(f"SUMMARY {a.design}[{a.suite}] n={len(rows)} WNS r={cw:.3f} MAE={np.mean(np.abs(we - wo)):.4f} "
            f"bias={np.mean(we - wo):+.4f} | TNS r={ct:.3f} MAE={np.mean(np.abs(te - to)):.1f} "
            f"rel={np.mean(np.abs(te - to)) / max(np.mean(np.abs(to)), 1e-9):.3f}")


# ════════════════════════════════════════════════════════════════════════════════════
#  self test (tiny synthetic design, no contest files)
# ════════════════════════════════════════════════════════════════════════════════════
_TINY_LIB = r"""
library (tiny) {
  time_unit : "1ns"; capacitive_load_unit (1,ff);
  slew_lower_threshold_pct_rise : 30.00; slew_upper_threshold_pct_rise : 70.00; slew_derate_from_library : 1.0;
  lu_table_template (T2) { variable_1 : input_net_transition; variable_2 : total_output_net_capacitance;
     index_1 ("0.01, 0.1"); index_2 ("1, 11"); }
  lu_table_template (C2) { variable_1 : related_pin_transition; variable_2 : constrained_pin_transition;
     index_1 ("0.01, 0.1"); index_2 ("0.01, 0.11"); }
  cell (BUF) {
    pin (A) { direction : input; capacitance : 1.0; }
    pin (Z) { direction : output; function : "A";
      timing () { related_pin : "A";
        cell_rise (T2) { values ("0.01, 0.03", "0.02, 0.04"); }
        cell_fall (T2) { values ("0.01, 0.03", "0.02, 0.04"); }
        rise_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); }
        fall_transition (T2) { values ("0.01, 0.05", "0.02, 0.06"); } } } }
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
}
"""


def self_test():
    import tempfile
    ok = True
    d = tempfile.mkdtemp(prefix="ft_")
    lib = Lib()
    lib._add(_parse_groups(_TINY_LIB))
    t = lib.pin("BUF", "Z").arcs[0][1][0]
    v = float(t(0.01, 6.0))
    good = abs(v - 0.02) < 1e-9 and abs(float(t(0.01, 21.0)) - 0.05) < 1e-9 and lib.seq["DFF"] and not lib.seq["BUF"]
    ok &= good
    print(f"  liberty parse / bilinear / extrapolation / seq: {good} ({v:.4f})")
    # design: ff1/Q -> n1 -> buf/A ; buf/Z -> n2 -> ff2/D ; the route of n2 is long on metal2 or on metal9
    kit = os.path.join(d, "kit")
    os.makedirs(os.path.join(kit, "visible/tiny"))
    os.makedirs(os.path.join(kit, "NanGate45/lib"))
    open(os.path.join(kit, "NanGate45/lib/tiny.lib"), "w").write(_TINY_LIB)
    open(os.path.join(kit, "visible/tiny/tiny.def"), "w").write(
        "COMPONENTS 3 ;\n- ff1 DFF + PLACED ( 0 0 ) N ;\n- buf BUF + PLACED ( 0 0 ) N ;\n- ff2 DFF + PLACED ( 0 0 ) N ;\nEND COMPONENTS\n")
    open(os.path.join(kit, "visible/tiny/tiny.net"), "w").write(
        "n1\n(\nff1/Q, -1e-11, [(0, 1, 1)]\nbuf/A, -1e-11, [(0, 2, 1)]\n)\n"
        "n2\n(\nbuf/Z, -1e-11, [(0, 2, 1)]\nff2/D, -1e-11, [(0, 102, 1)]\n)\n")
    open(os.path.join(kit, "visible/tiny/tiny.sdc"), "w").write("create_clock -period 1 [get_ports clk]\n")
    D = Design("tiny", "visible", lib=lib, kit=kit, verbose=False)
    good = list(D.pred) == [-1, 0, 1, 2] and bool(D.pin_endpoint[3]) and not D.pin_endpoint[1]
    ok &= good
    print(f"  worst-path forest {list(D.pred)} endpoints {list(np.nonzero(D.pin_endpoint)[0])}: {good}")

    def route(z):
        g = lambda v: v * GRID + GRID // 2
        r = (f"n1\n(\n{g(1)} {g(1)} metal1 {g(1)} {g(1)} metal3\n{g(1)} {g(1)} metal3 {g(2)} {g(1)} metal3\n"
             f"{g(2)} {g(1)} metal1 {g(2)} {g(1)} metal3\n)\n")
        r += f"n2\n(\n{g(2)} {g(1)} metal1 {g(2)} {g(1)} metal{z}\n{g(2)} {g(1)} metal{z} {g(102)} {g(1)} metal{z}\n" \
             f"{g(102)} {g(1)} metal1 {g(102)} {g(1)} metal{z}\n)\n"
        p = os.path.join(d, f"r{z}.route")
        open(p, "w").write(r)
        return p
    T = Timer(D)
    res = {}
    for z in (3, 9):
        seg = read_route_arrays(route(z), D.net_index)
        res[z] = T.analyze(seg, keep=True)
    # metal3 route == reference (except the via R) -> ~0 delta; metal9: less C and R -> positive slack gain
    d3 = res[3]["pin_dat"][3]
    d9 = res[9]["pin_dat"][3]
    # hand check of the metal3 Elmore of n2: R = r3 * 210um, C = c3 * 210um, pin cap 1 fF
    r3, c3 = LAYER_RC[2]
    Rw, Cw = r3 * 100 * UM_PER_GC, c3 * 100 * UM_PER_GC
    elm_hand = Rw * (Cw / 2 + 1.0) / 1000.0
    rc = res[3]["rc"]
    elm_model = rc.elm_ref[rc.pin_node[3]]
    good = abs(elm_model - elm_hand) < 1e-9 and abs(d3) < 0.01 and d9 < d3 and res[9]["wns"] >= res[3]["wns"]
    ok &= good
    print(f"  Elmore ref {elm_model * 1e3:.3f} ps (hand {elm_hand * 1e3:.3f}); delta_AT metal3 {d3 * 1e3:+.3f} ps, "
          f"metal9 {d9 * 1e3:+.3f} ps; WNS {res[3]['wns']:.4f} / {res[9]['wns']:.4f}: {good}")
    crit = os.path.join(d, "est.crit.rpt")
    n = T.write_crit(res[3], crit, slack_max=1.0)
    from timing_promote import parse_crit, parse_net_rc
    pc = parse_crit(crit)
    nr = parse_net_rc(crit)
    good = n == 1 and len(pc) == 1 and pc[0][1] == ["n1", "n2"] and set(nr) == {"n1", "n2"}
    ok &= good
    print(f"  crit report round trip (timing_promote parsers): {pc} {nr}: {good}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("time", "calib"), default="time")
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--route")
    ap.add_argument("--crit_out")
    ap.add_argument("--npaths", type=int, default=5000)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--csv")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "calib":
        calib(a)
        return 0
    t0 = time.time()
    D = Design(a.design, a.suite)
    T = Timer(D)
    seg = read_route_arrays(a.route, D.net_index)
    r = T.analyze(seg)
    log(f"est WNS {r['wns']:.4f} ns TNS {r['tns']:.1f} ns ({r['nviol']} violating endpoints) "
        f"total {time.time() - t0:.1f}s")
    if a.crit_out:
        n = T.write_crit(r, a.crit_out, a.npaths)
        log(f"wrote {a.crit_out}: {n} paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
