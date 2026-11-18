#!/usr/bin/env python3
"""act_inproc.py: plan step R2, the switching activity INSIDE our process, on our own timing
graph (fast_sta_v3.StaDesign) instead of an OpenSTA run. The propagation is tools/speed4/act_kern.cc (OpenSTA's
Power::ensureActivities semantics, see there); this file builds its arrays:
  * Liberty functions per (cell, output pin) and the ff/latch data functions, parsed once per library (cached in memory),
    compiled to complement-edge ROBDDs with OpenSTA's variable order (first appearance in the function), their Boolean
    differences, and the sum order of evalBddActivity (LibertyPort pointer order ~ pin declaration order);
  * every instance pin OpenSTA has a vertex for: the DEF-connected pins of StaDesign plus pseudo pins for function inputs
    that are unconnected (levelization roots: seeded 0.1 / 0.5 like input ports) or tied to a POWER/GROUND net (constants);
  * constants (Sim): power/ground pins, set_case_analysis ports, 0/1 functions, propagated through nets and 3-valued
    function evaluation; arcs whose output does not depend on the input under the constants are disabled (sim sense none);
  * activity adjacency (ActivitySrchPred): wire edges + combinational arcs (not latch D->Q) + latch enable arcs, no
    register clk->Q, no preset/clear, no timing checks, nothing from / to a constant;
  * clock of each pin (Power::findClk / findInstClk): clock-network pins and input delays, propagated over the arrival
    graph (wires + all delay arcs); density = activity / period (unscaled without a clock).
API:  install_def_capture()   (before StaDesign is built: keeps the POWER/GROUND net pins the DEF parse skips)
      dens, info = run(SD)    (per DEF net density, == the .act file value the chain reads; info = counters / seconds)
"""
import ctypes, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None
CAP = {"power_pins": None}
O_UNKNOWN, O_INPUT, O_PROP, O_CLOCK, O_CONST = 0, 1, 2, 3, 4
_FUNC_CACHE = {}


def install_def_capture():
    """wrap fast_sta_v3.parse_def_netlist: also keep [(inst key, pin key, value 1|0)] of POWER/GROUND net connections."""
    import fast_sta_v3 as FS
    if getattr(FS.parse_def_netlist, "_s4", False):
        return
    _o = FS.parse_def_netlist

    def wrap(path, _o=_o):
        out = _o(path)
        comps, ports, names, cn_net, cn_inst, cn_pin, power = out
        pw = []
        if power:
            pset = power
            for n, inst, pn in zip(cn_net, cn_inst, cn_pin):
                if n in pset:
                    nm = names[n].upper()
                    v = 0 if ("VSS" in nm or "GND" in nm or "GROUND" in nm) else 1
                    pw.append((inst, pn, v))
        CAP["power_pins"] = pw
        return out
    wrap._s4 = True
    FS.parse_def_netlist = wrap


def power_pins_from_def(path):
    """[(inst, pin, 1|0)] of the signal connections of NETS entries with + USE POWER / GROUND (the DEF loaders skip them)"""
    txt = open(path).read()
    a = txt.find("\nNETS ")
    b = txt.find("\nEND NETS", a)
    out = []
    for kw, v in (("+ USE POWER", 1), ("+ USE GROUND", 0)):
        pos = txt.find(kw, a, b)
        while pos >= 0:
            st = txt.rfind("\n- ", a, pos)
            for m in re.finditer(r"\(\s*(\S+)\s+(\S+)\s*\)", txt[st:pos]):
                if m.group(1) not in ("*", "PIN"):
                    out.append((m.group(1), m.group(2), v))
            pos = txt.find(kw, pos + 1, b)
    return out


def _lib():
    global _LIB
    if _LIB is None:
        so = os.path.join(HERE, "libact_kern." + ("dylib" if sys.platform == "darwin" else "so"))
        L = ctypes.CDLL(so)
        V = ctypes.c_void_p
        L.act_run.restype = ctypes.c_int
        L.act_run.argtypes = [ctypes.c_int64] + [V] * 10 + [ctypes.c_int64, V] + [V] * 10 + [ctypes.c_int64] + [V] * 6 + [ctypes.c_int] + [V] * 4
        L.act_clkmask.restype = ctypes.c_int
        L.act_clkmask.argtypes = [ctypes.c_int64, V, V, V]
        L.act_density.argtypes = [ctypes.c_int64, V, V, V]
        _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


# ───────────────────────── Liberty functions -> truth tables -> ROBDD (complement edges, CUDD's canonical form)
_TOK = re.compile(r"\s*([A-Za-z_][\w\[\]\.]*|[01]|[!'&*|+^()])")


def parse_func(s):
    """('var', name) | ('const', v) | ('not', e) | ('and'|'or'|'xor', a, b); precedence ' ! > ^ > & * juxtaposition > | +"""
    toks = _TOK.findall(s.replace('"', ""))
    pos = [0]
    peek = lambda: toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = toks[pos[0]]; pos[0] += 1; return t

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
            take(); e = ("not", e)
        return e

    def xorx():
        e = atom()
        while peek() == "^":
            take(); e = ("xor", e, atom())
        return e

    def andx():
        e = xorx()
        while True:
            t = peek()
            if t in ("&", "*"):
                take(); e = ("and", e, xorx())
            elif t is not None and t not in ("|", "+", ")", "^", "'"):
                e = ("and", e, xorx())
            else:
                return e

    def orx():
        e = andx()
        while peek() in ("|", "+"):
            take(); e = ("or", e, andx())
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


def eval3(e, env):
    """3-valued (0, 1, -1 = X) evaluation; env name -> value"""
    k = e[0]
    if k == "var":
        return env.get(e[1], -1)
    if k == "const":
        return e[1]
    if k == "not":
        v = eval3(e[1], env)
        return -1 if v < 0 else 1 - v
    a, b = eval3(e[1], env), eval3(e[2], env)
    if k == "and":
        return 0 if (a == 0 or b == 0) else (1 if (a == 1 and b == 1) else -1)
    if k == "or":
        return 1 if (a == 1 or b == 1) else (0 if (a == 0 and b == 0) else -1)
    return -1 if (a < 0 or b < 0) else a ^ b


def truth(e, names):
    """truth table over names (bit i of the row index = names[i])"""
    n = len(names)
    rows = np.arange(1 << n)
    env = {nm: ((rows >> i) & 1).astype(bool) for i, nm in enumerate(names)}

    def ev(x):
        k = x[0]
        if k == "var":
            return env[x[1]]
        if k == "const":
            return np.full(len(rows), bool(x[1]))
        if k == "not":
            return ~ev(x[1])
        a, b = ev(x[1]), ev(x[2])
        return (a & b) if k == "and" else (a | b) if k == "or" else (a ^ b)
    return ev(e)


class BddPool:
    """node 0 = constant ONE; ptr = 2 * id + complement; then-children regular (CUDD)"""
    def __init__(self):
        self.var, self.T, self.E = [-1], [0], [0]
        self.u = {}

    def mk(self, v, T, E):
        if T == E:
            return T
        if T & 1:
            return self.mk(v, T ^ 1, E ^ 1) ^ 1
        k = (v, T, E)
        i = self.u.get(k)
        if i is None:
            i = len(self.var); self.var.append(v); self.T.append(T); self.E.append(E); self.u[k] = i
        return 2 * i

    def from_tt(self, tt, n, level=0):
        """tt: bool array over vars level..n-1 (bit 0 of its index = var `level`)"""
        if tt.all():
            return 0
        if not tt.any():
            return 1
        hi = self.from_tt(tt[1::2], n, level + 1)
        lo = self.from_tt(tt[0::2], n, level + 1)
        return self.mk(level, hi, lo)


def _seq_groups(lib_dir):
    """{cell: (kind 'ff'|'latch', IQ, IQN or None, data function string)} from the .lib text"""
    out = {}
    for f in sorted(os.listdir(lib_dir)):
        if not f.endswith(".lib"):
            continue
        txt = open(os.path.join(lib_dir, f)).read()
        cells = [(m.start(), m.group(1)) for m in re.finditer(r"\bcell\s*\(\s*\"?([\w]+)\"?\s*\)", txt)]
        for k, (a, nm) in enumerate(cells):
            b = cells[k + 1][0] if k + 1 < len(cells) else len(txt)
            m = re.search(r"\b(ff|latch)\s*\(\s*\"?(\w+)\"?\s*(?:,\s*\"?(\w+)\"?\s*)?\)\s*\{([^}]*)\}", txt[a:b])
            if not m:
                continue
            body = m.group(4)
            key = "next_state" if m.group(1) == "ff" else "data_in"
            d = re.search(key + r"\s*:\s*\"([^\"]*)\"", body)
            out[nm] = (m.group(1), m.group(2), m.group(3), d.group(1) if d else None)
    return out


def _compile(lib, cell_names, lib_dir):
    """per cell type: outputs {pin: ('direct', 0|1) | ('f', fid, vars in sum order, expr)}, seq info, pin order, arcs"""
    key = (id(lib), tuple(cell_names))
    if key in _FUNC_CACHE:
        return _FUNC_CACHE[key]
    seqg = _seq_groups(lib_dir)
    pool = BddPool()
    F = {"root": [], "nv": [], "diff": [], "pos": [], "direct": [], "DIFF": [], "POS": []}
    fid_of = {}

    def fn(expr_s, order_of, internal):
        e = parse_func(expr_s)
        names = func_vars(e)                                  # BDD var index = first appearance
        sk = (expr_s, tuple(sorted(names, key=lambda v: order_of.get(v, -1))), tuple(sorted(internal)))
        if sk in fid_of:
            return fid_of[sk][0], e, list(fid_of[sk][1])
        n = len(names)
        tt = truth(e, names)
        root = pool.from_tt(tt, n)
        sumo = sorted(range(n), key=lambda i: (order_of.get(names[i], -1), i))     # LibertyPort pointer order proxy
        if os.environ.get("ACT_S4_SUMORDER") == "rev":                              # experiment only
            sumo = sumo[::-1]
        elif os.environ.get("ACT_S4_SUMORDER") == "bdd":
            sumo = list(range(n))
        pos = [0] * n
        for s_, i in enumerate(sumo):
            pos[i] = s_
        diffs = []
        idx = np.arange(1 << n)
        for i in sumo:
            hi = tt[idx | (1 << i)]; lo = tt[idx & ~(1 << i)]
            diffs.append(pool.from_tt(hi ^ lo, n))
        fid = len(F["root"])
        F["root"].append(root); F["nv"].append(n); F["diff"].append(len(F["DIFF"])); F["pos"].append(len(F["POS"]))
        F["direct"].append(-1); F["DIFF"].extend(diffs); F["POS"].extend(pos)
        fid_of[sk] = (fid, [names[i] for i in sumo])
        return fid, e, [names[i] for i in sumo]
    direct = {}
    for w in (0, 1):
        direct[w] = len(F["root"])
        F["root"].append(0); F["nv"].append(0); F["diff"].append(len(F["DIFF"])); F["pos"].append(len(F["POS"])); F["direct"].append(w)
    cells = []
    for c, nm in enumerate(cell_names):
        m = lib.cells.get(nm)
        if m is None:
            cells.append(None); continue
        order_of = {p: i for i, p in enumerate(m.pins)}
        sg = seqg.get(nm)
        internal = {}
        if sg:
            internal[sg[1]] = 0
            if sg[2]:
                internal[sg[2]] = 1
        outs = {}
        for pn, pv in m.pins.items():
            if pv[0] in ("output", "inout") and pv[4]:
                fs = pv[4].strip().strip('"').strip()
                if fs in internal:
                    outs[pn] = ("direct", internal[fs], None, None)
                else:
                    fid, e, vs = fn(fs, order_of, internal)
                    outs[pn] = ("f", fid, vs, e)
        seq = None
        if sg and sg[3]:
            fid, e, vs = fn(sg[3], order_of, internal)
            seq = (sg[0] == "ff", sg[2] is not None, fid, vs, e)
        arcs_from = {}                                       # from pin -> [(to pin, kind)]
        for to, lst in m.arcs.items():
            for a in lst:
                arcs_from.setdefault(a[0], []).append((to, a[1]))
        cells.append({"outs": outs, "seq": seq, "latch": bool(m.latch), "hasseq": sg is not None, "order": order_of,
                      "internal": internal, "arcs_from": arcs_from})
    res = (pool, F, cells)
    _FUNC_CACHE[key] = res
    return res


def run(SD, lib_dir=None, max_passes=100, check_cycles=True):
    t0 = time.perf_counter()
    import fast_sta_v3 as FS
    from timing_promote import _key
    lib_dir = lib_dir or os.path.join(FS.KIT, "NanGate45", "lib")
    info = {}
    pool, F, cells = _compile(SD.lib, SD.cell_names, lib_dir)
    info["t_compile"] = time.perf_counter() - t0
    P = SD.P
    nn = len(SD.net_names)
    is_port = SD.is_port
    # ---- pseudo pins: function / data inputs without a DEF signal connection
    pw_pins = {}
    if CAP["power_pins"] is None:
        t_d = time.perf_counter()
        CAP["power_pins"] = power_pins_from_def(os.path.join(FS.KIT, SD.suite, SD.design, SD.design + ".def"))
        info["t_def_power_scan"] = time.perf_counter() - t_d
    if CAP["power_pins"]:
        iid = None
        for inst, pn, v in CAP["power_pins"]:
            if inst == "PIN":
                continue
            if iid is None:
                iid = {nm: i for i, nm in enumerate(SD.inst_names)}
            i = iid.get(_key(inst))
            if i is not None:
                pw_pins[(i, _key(pn))] = v
    info["power_pins"] = len(pw_pins)
    counts = [len(M) for M in SD.mats]
    inst_of_ct = [None] * len(SD.cell_names)
    order = np.argsort(SD.inst_ct, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    for c in range(len(SD.cell_names)):
        inst_of_ct[c] = order[starts[c]:starts[c + 1]]           # row r of mats[c] = instance inst_of_ct[c][r]
    extra_inst, extra_name, extra_val = [], [], []                # pseudo pins (P + k)
    colcache = {}

    def col(c, v):
        """pin ids of var v for every instance row of cell type c (pseudo pins created where missing)"""
        k = (c, v)
        if k in colcache:
            return colcache[k]
        M = SD.mats[c]
        s = SD.slots[c].get(v)
        a = M[:, s].copy() if s is not None else np.full(len(M), -1, np.int64)
        miss = np.nonzero(a < 0)[0]
        if len(miss):
            base = P + len(extra_inst)
            a[miss] = base + np.arange(len(miss))
            insts = inst_of_ct[c][miss]
            extra_inst.extend(insts.tolist()); extra_name.extend([v] * len(miss))
            vk = _key(v)
            extra_val.extend([pw_pins.get((int(i), vk), -1) for i in insts.tolist()])
        colcache[k] = a
        return a
    # ---- functions per output pin, seq data per instance
    func_pin, func_id, vp_rows = [], [], []                       # per (cell, output pin): pins, fid, [rows x nv] pin ids
    seq_rows = []
    for c, cm in enumerate(cells):
        if cm is None or not counts[c]:
            continue
        M = SD.mats[c]
        for pn, o in cm["outs"].items():
            s = SD.slots[c].get(pn)
            if s is None:
                continue
            pins = M[:, s]
            ok = pins >= 0
            if not ok.any():
                continue
            if o[0] == "direct":
                func_pin.append(pins[ok]); func_id.append(np.full(int(ok.sum()), o[1] if False else -100 - o[1], np.int64)); vp_rows.append(None)
                continue
            cols = [col(c, v) if v not in cm["internal"] else np.full(len(M), -10 - cm["internal"][v], np.int64) for v in o[2]]
            V = np.stack(cols, 1) if cols else np.zeros((len(M), 0), np.int64)
            func_pin.append(pins[ok]); func_id.append(np.full(int(ok.sum()), o[1], np.int64)); vp_rows.append(V[ok])
        if cm["seq"] is not None:
            isreg, hasinv, fid, vs, e = cm["seq"]
            cols = [col(c, v) if v not in cm["internal"] else np.full(len(M), -10 - cm["internal"][v], np.int64) for v in vs]
            V = np.stack(cols, 1) if cols else np.zeros((len(M), 0), np.int64)
            seq_rows.append((c, isreg, hasinv, fid, V))
    NPX = P + len(extra_inst)
    info["pseudo_pins"] = len(extra_inst)
    extra_inst = np.array(extra_inst, np.int64); extra_val = np.array(extra_val, np.int64)
    # ---- per pin arrays
    pin_inst = np.r_[SD.pin_inst, extra_inst]
    isload = np.r_[~SD.pin_isdrvdir, np.ones(len(extra_inst), bool)].astype(np.uint8)
    wfrom = np.full(NPX, -1, np.int32)
    ld = np.nonzero(~SD.pin_isdrvdir)[0]
    dr = SD.net_drv[SD.pin_net[ld]]
    okd = (dr >= 0) & (dr != ld)
    wfrom[ld[okd]] = dr[okd]
    FUNC = np.full(NPX, -1, np.int32)
    VPOFF = np.zeros(NPX, np.int64)
    pools, cur = [], 0
    direct_fid = {0: int(np.nonzero(np.array(F["direct"]) == 0)[0][0]), 1: int(np.nonzero(np.array(F["direct"]) == 1)[0][0])}
    for pins, fids, V in zip(func_pin, func_id, vp_rows):
        if V is None:
            w = -100 - int(fids[0])
            FUNC[pins] = direct_fid[w]
            continue
        FUNC[pins] = fids
        nv = V.shape[1]
        VPOFF[pins] = cur + np.arange(len(pins)) * nv
        pools.append(V.reshape(-1)); cur += len(pins) * nv
    # seq instances
    ns = sum(len(r[4]) for r in seq_rows)
    SEQI = np.full(NPX, -1, np.int32)
    s_func = np.zeros(ns, np.int32); s_vpoff = np.zeros(ns, np.int64); s_isreg = np.zeros(ns, np.uint8); s_hasinv = np.zeros(ns, np.uint8)
    seq_of_inst = np.full(len(SD.inst_names), -1, np.int64)
    k0 = 0
    for c, isreg, hasinv, fid, V in seq_rows:
        n = len(V); rng = np.arange(k0, k0 + n)
        insts = inst_of_ct[c]
        seq_of_inst[insts] = rng
        s_func[rng] = fid; s_isreg[rng] = isreg; s_hasinv[rng] = hasinv
        nv = V.shape[1]
        s_vpoff[rng] = cur + np.arange(n) * nv
        pools.append(V.reshape(-1)); cur += n * nv
        k0 += n
    VP = (np.concatenate(pools) if pools else np.zeros(0, np.int64)).astype(np.int32)
    hasi = pin_inst >= 0
    SEQI[hasi] = seq_of_inst[pin_inst[hasi]]
    # Q pins (direct IQ / IQN functions) per seq instance
    qp = np.nonzero((FUNC >= 0) & np.isin(FUNC, list(direct_fid.values())))[0]
    qs = SEQI[qp]
    qp, qs = qp[qs >= 0], qs[qs >= 0]
    o = np.argsort(qs, kind="stable")
    s_q = qp[o].astype(np.int32)
    s_qoff = np.searchsorted(qs[o], np.arange(ns + 1)).astype(np.int64)
    # ---- constants (Sim)
    cv = np.full(NPX, -1, np.int8)
    cv[P:] = np.where(extra_val >= 0, extra_val, -1)
    port_pid = {SD.pin_name[i]: i for i in np.nonzero(is_port)[0]}
    for p, v in SD.sdc["case"].items():
        i = port_pid.get(p)
        if i is not None:
            cv[i] = v
    # constant functions (tie cells)
    for c, cm in enumerate(cells):
        if cm is None or not counts[c]:
            continue
        for pn, o_ in cm["outs"].items():
            if o_[0] == "f" and not o_[2]:
                s = SD.slots[c].get(pn)
                if s is not None:
                    pins = SD.mats[c][:, s]; pins = pins[pins >= 0]
                    cv[pins] = eval3(o_[3], {})
    # net loads
    net_pins_o = np.argsort(SD.pin_net, kind="stable")
    net_off = np.searchsorted(SD.pin_net[net_pins_o], np.arange(nn + 1))
    inst_pins = {}                                               # lazily: instance -> its output pins / function rows
    t_s = time.perf_counter()
    q = list(np.nonzero(cv >= 0)[0])
    seen_inst = set()
    nsim = 0
    while q:
        nq = []
        for i in q:
            if i < P and SD.pin_isdrvdir[i] and SD.net_drv[SD.pin_net[i]] == i:   # driver: its net's loads
                n = SD.pin_net[i]
                for j in net_pins_o[net_off[n]:net_off[n + 1]]:
                    if j != i and cv[j] < 0 and not SD.pin_isdrvdir[j]:
                        cv[j] = cv[i]; nq.append(j)
            elif isload[i] and pin_inst[i] >= 0:                      # instance input: evaluate the instance outputs
                ii = int(pin_inst[i]); c = int(SD.inst_ct[ii]); cm = cells[c]
                if cm is None:
                    continue
                loc = SD.inst_loc[ii]
                for pn, o_ in cm["outs"].items():
                    if o_[0] != "f":
                        continue
                    s = SD.slots[c].get(pn)
                    if s is None:
                        continue
                    op = SD.mats[c][loc, s]
                    if op < 0 or cv[op] >= 0:
                        continue
                    env = {v: int(cv[col(c, v)[loc]]) for v in o_[2] if v not in cm["internal"]}
                    nsim += 1
                    r = eval3(o_[3], env)
                    if r >= 0:
                        cv[op] = r; nq.append(op)
        q = nq
    info["const_pins"] = int((cv >= 0).sum()); info["sim_evals"] = nsim
    KIND = np.zeros(NPX, np.int8)
    cn = np.r_[SD.cnet, np.zeros(len(extra_inst), bool)]
    KIND[cn] = 2
    KIND[cv >= 0] = 1
    # ---- sim-disabled arcs: instances with a constant input, arc x -> out independent of x under the constants
    A = SD.arc
    fr, to, kd = A["fr"], A["to"], A["kind"]
    lat_ct = np.array([bool(cm is not None and cm["latch"]) for cm in cells] + [False])
    ct_to = SD.pin_ct[to]
    is_lat = lat_ct[np.where(ct_to >= 0, ct_to, len(cells))]
    keep_act = np.where(kd == 0, ~is_lat, is_lat)
    okc = (KIND[fr] != 1) & (KIND[to] != 1)
    cinst = np.zeros(len(SD.inst_names), bool)
    cpi = np.nonzero((cv >= 0) & (pin_inst >= 0) & (isload == 1))[0]
    cinst[pin_inst[cpi]] = True
    dis = np.zeros(len(fr), bool)
    cand = np.nonzero(okc & cinst[np.maximum(SD.pin_inst[to], 0)] & (SD.pin_inst[to] >= 0) & (kd == 0))[0]
    for e in cand:
        ii = int(SD.pin_inst[to[e]]); c = int(SD.inst_ct[ii]); cm = cells[c]
        if cm is None:
            continue
        loc = SD.inst_loc[ii]
        opn = SD.pin_name[to[e]].rsplit("/", 1)[1]
        ipn = SD.pin_name[fr[e]].rsplit("/", 1)[1]
        o_ = cm["outs"].get(opn)
        if o_ is None or o_[0] != "f" or ipn not in o_[2]:
            continue
        env = {v: int(cv[col(c, v)[loc]]) for v in o_[2] if v not in cm["internal"]}
        free = [v for v in o_[2] if env.get(v, -1) < 0 and v != ipn]
        dep = False
        for bits in range(1 << len(free)):
            e2 = dict(env); e2.update({v: (bits >> j) & 1 for j, v in enumerate(free)})
            e2[ipn] = 0; a0 = eval3(o_[3], e2)
            e2[ipn] = 1; a1 = eval3(o_[3], e2)
            if a0 != a1:
                dep = True; break
        dis[e] = not dep
    info["sim_disabled_arcs"] = int(dis.sum()); info["t_sim"] = time.perf_counter() - t_s
    # ---- activity adjacency
    ld_all = np.nonzero(~SD.pin_isdrvdir)[0]
    wd = SD.net_drv[SD.pin_net[ld_all]]
    okw = (wd >= 0) & (wd != ld_all)
    wsrc, wdst = wd[okw], ld_all[okw]
    okw2 = (KIND[wsrc] != 1) & (KIND[wdst] != 1)
    sel = okc & keep_act & ~dis
    src = [wsrc[okw2], fr[sel]]; dst = [wdst[okw2], to[sel]]
    # pseudo unconnected inputs -> instance outputs (library arcs)
    roots_x = (P + np.nonzero(extra_val < 0)[0]).tolist()
    ps, pd, pall = [], [], []                                    # pseudo pins -> instance outputs (library arcs)
    for k in range(len(extra_inst)):
        ii = int(extra_inst[k]); c = int(SD.inst_ct[ii]); cm = cells[c]
        if cm is None:
            continue
        for tpn, kind_ in cm["arcs_from"].get(extra_name[k], []):
            s = SD.slots[c].get(tpn)
            if s is None:
                continue
            op = SD.mats[c][SD.inst_loc[ii], s]
            if op < 0:
                continue
            pall.append(op)
            if ((kind_ == 0) != cm["latch"]) and extra_val[k] < 0 and KIND[op] != 1:
                ps.append(P + k); pd.append(op)
    src.append(np.array(ps, np.int64)); dst.append(np.array(pd, np.int64))
    info["pseudo_arcs"] = len(ps)
    src = np.concatenate(src).astype(np.int64); dst = np.concatenate(dst).astype(np.int64)
    o = np.argsort(src, kind="stable")
    adj = dst[o].astype(np.int32)
    adj_off = np.searchsorted(src[o], np.arange(NPX + 1)).astype(np.int64)
    # ---- roots: no fanin (levelization) and not a clock source
    indeg = np.zeros(NPX, np.int64)
    np.add.at(indeg, wdst, 1)
    np.add.at(indeg, to, 1)
    np.add.at(indeg, np.array(pall, np.int64), 1)
    clk_src = set()
    for cn_ in SD.clk_names:
        for p in SD.sdc["clocks"][cn_]["src"]:
            i = port_pid.get(p)
            if i is not None:
                clk_src.add(i)
    rts = np.nonzero(indeg[:P] == 0)[0]
    rts = [int(r) for r in rts if r not in clk_src]
    roots = np.array(rts + roots_x + (P + np.nonzero(extra_val >= 0)[0]).tolist(), np.int32)
    info["roots"] = len(roots)
    # ---- clocks: duty of clock pins, clock reach masks, period per pin
    ck = [SD.sdc["clocks"][n] for n in SD.clk_names]
    per_f = np.array([np.float32(c_["period"] * 1e-9) for c_ in ck] + [np.float32(0)], np.float32)
    if os.environ.get("ACT_S4_DUTY") == "double":                                    # experiment only
        duty_f = np.array([np.float32((c_["wave"][1] - c_["wave"][0]) / c_["period"]) for c_ in ck] + [np.float32(0.5)], np.float32)
    else:
      duty_f = np.array([np.float32((np.float32(c_["wave"][1] * 1e-9) - np.float32(c_["wave"][0] * 1e-9)) / per_f[i])
                         for i, c_ in enumerate(ck)] + [np.float32(0.5)], np.float32)
    clk_of = np.r_[SD.clk_of, np.full(len(extra_inst), -1, np.int64)]
    CDUTY = np.where(clk_of >= 0, duty_f[np.where(clk_of >= 0, clk_of, len(ck))], np.float32(0.5)).astype(np.float32)
    mask = np.zeros(NPX, np.uint64)
    m0 = cn & (clk_of >= 0) & (KIND != 1)
    mask[m0] = (np.uint64(1) << clk_of[m0].astype(np.uint64))
    cidx = {n: i for i, n in enumerate(SD.clk_names)}
    for p, cname, fall, v in SD.sdc["in"]:
        i = port_pid.get(p)
        if i is not None and cname in cidx and KIND[i] != 1:
            mask[i] |= np.uint64(1) << np.uint64(cidx[cname])
    sela = okc & ~dis
    asrc = np.r_[wsrc[okw2], fr[sela]]; adst = np.r_[wdst[okw2], to[sela]]
    o = np.argsort(asrc, kind="stable")
    aadj = adst[o].astype(np.int32); aoff = np.searchsorted(asrc[o], np.arange(NPX + 1)).astype(np.int64)
    L = _lib()
    L.act_clkmask(NPX, _p(aoff), _p(aadj), _p(mask))
    # ---- propagate
    o_act = np.zeros(NPX, np.float32); o_duty = np.zeros(NPX, np.float32); o_org = np.zeros(NPX, np.int8)
    st = np.zeros(8)
    arr = lambda x, dt: np.ascontiguousarray(np.asarray(x, dt))
    nv_ = arr(pool.var, np.int32); nt_ = arr(pool.T, np.int32); ne_ = arr(pool.E, np.int32)
    fr_ = arr(F["root"], np.int32); fnv = arr(F["nv"], np.int32); fdf = arr(F["diff"], np.int32); fps = arr(F["pos"], np.int32)
    fdi = arr(F["direct"], np.int32); dfp = arr(F["DIFF"] or [0], np.int32); psp = arr(F["POS"] or [0], np.int32)
    keep = (KIND, wfrom, isload, FUNC, VPOFF, VP, SEQI, CDUTY, adj_off, adj, roots)
    t_r = time.perf_counter()
    rc = L.act_run(NPX, _p(KIND), _p(wfrom), _p(isload), _p(FUNC), _p(VPOFF), _p(VP if len(VP) else np.zeros(1, np.int32)), _p(SEQI), _p(CDUTY),
                   _p(adj_off), _p(adj), len(roots), _p(roots), _p(nv_), _p(nt_), _p(ne_), _p(fr_), _p(fnv), _p(fdf), _p(fps), _p(fdi), _p(dfp), _p(psp),
                   ns, _p(s_func), _p(s_vpoff), _p(s_isreg), _p(s_hasinv), _p(s_qoff), _p(s_q if len(s_q) else np.zeros(1, np.int32)),
                   max_passes, _p(o_act), _p(o_duty), _p(o_org), _p(st))
    del keep
    info.update(rc=rc, passes=int(st[0]), visits=int(st[1]), levels=int(st[2]), t_kernel=float(st[3]), cyc=int(st[5]), loop_edges_cut=int(st[6]),
                t_run_call=time.perf_counter() - t_r)
    if rc != 0:
        raise RuntimeError(f"act_inproc: cycle in the activity adjacency ({int(st[5])} pins)")
    # ---- densities per DEF net (findClkedActivity of the net's driver pin; the chain skips input / clock lines)
    drv = SD.net_drv
    okn = (drv >= 0)
    okn[okn] = ~is_port[drv[okn]]
    d = drv[okn]
    # effective activity (findActivity) of the driver
    ea = o_act[d].copy(); eo = o_org[d].copy()
    k1 = KIND[d] == 1; ea[k1] = 0; eo[k1] = O_CONST
    k2 = (KIND[d] == 2) & (o_org[d] == O_UNKNOWN); eo[k2] = O_CLOCK
    eu = (KIND[d] == 0) & (o_org[d] == O_UNKNOWN); ea[eu] = 0
    # clock: own mask, else the instance's pins (findInstClk), fastest period
    pm = mask[d]
    if (pm == 0).any():
        imask = np.zeros(len(SD.inst_names), np.uint64)
        ip = np.nonzero(pin_inst >= 0)[0]
        np.bitwise_or.at(imask, pin_inst[ip], mask[ip])
        z = pm == 0
        pm = pm.copy(); pm[z] = imask[pin_inst[d[z]]]
    per = np.zeros(len(d), np.float32)
    best = np.full(len(d), np.inf)
    for c_ in range(len(ck)):
        has = (pm >> np.uint64(c_)) & np.uint64(1)
        h = (has == 1) & (per_f[c_] < best)
        per[h] = per_f[c_]; best[h] = per_f[c_]
    dens_d = np.zeros(len(d))
    L.act_density(len(d), _p(np.ascontiguousarray(ea, np.float32)), _p(per), _p(dens_d))
    dens_d[(eo == O_INPUT) | (eo == O_CLOCK)] = 0.0
    dens = np.zeros(nn)
    dens[np.nonzero(okn)[0]] = dens_d
    info["t_total"] = time.perf_counter() - t0
    info["nonzero"] = int((dens != 0).sum())
    if os.environ.get("ACT_S4_DEBUG"):
        info["_dbg"] = dict(act=o_act, duty=o_duty, org=o_org, kind=KIND, pin_inst=pin_inst, extra_name=extra_name, cv=cv, mask=mask)
    return dens, info
