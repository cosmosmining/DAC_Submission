#!/usr/bin/env python3
"""load8_lib.py: the --load8 1 pieces of tools/best11/chain_u.py, cold big mode only (MemPool group / cluster through tools/best11/cold_u.py). an internal note.
Same values as chain_p's load (tests: test_fl8.py, test_kern8.py, test_sta8.py; final proof: route md5). Each piece off with env LOAD8_<NAME>=0.
  notorch  no torch / torch_geometric / jinja2 / requests / sympy import: pre5's stand-ins and pre5's C++ prior (NOT bit-equal to torch: judged by the route md5). 375 s of the 410 s of the cluster's cold run.
  fl8      tools/best11/fl8.cc + fl8_lib.py: DEF / .net parse with the serial phases on threads, the name lists as blobs (no Python string per pin / instance / DEF net).
  net2     the .net parse on its own thread from t0, beside the DEF parse (on a cold file system both reads wait at the same time).
  sdc      the .sdc parse on a thread from t0.
  cap      one .cap parse on a thread (the state gets a copy; chain_p parsed the file twice).
  rt       the route read on a thread as soon as the route is ready, not after the timer bind.
  sta      StaDesign: arcs, clocks, graph with threaded sorts / takes (tools/best11/l8_kern.cc).
  bind     timer bind + first timer + activity start the moment the StaDesign is complete, each on its own thread; the state goes on beside them.
  actx     tools/speed4/act_inproc with the power pins from the one DEF parse, the instance table in C++, threaded sorts (source patched at load).
  b2       the build2 pre-parse on a thread.
  pins     overflow pin tables in C++ (tools/best11/l8_kern.cc l8_pin_maps), dicts made at C speed.
  seg      csta.seg_from_route: names through the C++ table, columns on threads.
  names    attach_h's name tables at C speed, DEF ids through the C++ table.
"""
import itertools, os, re, sys, threading, time, types, textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "tools" / "final" / "chain_y4.py").exists()), HERE)
sys.path.insert(0, str(HERE))
try:
    import kern8 as K8
except (ImportError, OSError):                         # table-only tests do not need l8_kern
    K8 = None

R = {"pieces": {}, "t": {}, "fut": {}, "stats": {}}
EARLY = {}


_TABLE_CLASS = None
_TABLE_CACHE = {}
_NAMES_TABLES_CACHE = {}


class _TableIndex:
    """Mapping interface backed by the load8 C++ exact-name table."""

    __slots__ = ("table", "owner")

    def __init__(self, table, owner):
        self.table, self.owner = table, owner

    @staticmethod
    def _key(name):
        return name.replace("\\", "") if isinstance(name, str) else name

    def get(self, name, default=None):
        i = self.table.find(self._key(name), strip=False)
        return default if i < 0 else int(i)

    def __getitem__(self, name):
        i = self.table.find(self._key(name), strip=False)
        if i < 0:
            raise KeyError(name)
        return int(i)

    def __contains__(self, name):
        return self.get(name, -1) >= 0


def _table_class():
    """Use whichever existing load8 module owns the C++ name-table ABI."""
    global _TABLE_CLASS
    if _TABLE_CLASS is not None:
        return _TABLE_CLASS
    for module_name in ("fastload_lib", "fl8_lib"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "Table"):
            _TABLE_CLASS = module.Table
            return _TABLE_CLASS
    for module_name in ("fastload_lib", "fl8_lib"):
        try:
            module = __import__(module_name)
        except (ImportError, OSError):
            continue
        if hasattr(module, "Table"):
            _TABLE_CLASS = module.Table
            return _TABLE_CLASS
    raise RuntimeError("fp2 requires the existing load8 C++ name-table module")


def _normalised_blob(names):
    """Return the exact-name blob/offsets used by the old Python mapping semantics."""
    if hasattr(names, "raw") and hasattr(names, "off"):
        raw = bytes(names.raw)
        off = np.ascontiguousarray(np.asarray(names.off, dtype=np.int64))
    else:
        raw = ("\n".join(names) + "\n").encode("utf-8") if len(names) else b""
        ends = np.flatnonzero(np.frombuffer(raw, np.uint8) == 10)
        off = np.zeros(len(ends) + 1, np.int64)
        off[1:] = ends + 1
    if b"\\" in raw:
        raw = raw.replace(b"\\", b"")
        ends = np.flatnonzero(np.frombuffer(raw, np.uint8) == 10)
        off = np.zeros(len(ends) + 1, np.int64)
        off[1:] = ends + 1
    return raw, off


def table_for_names(names):
    """Build/cache the same C++ table for one stable name-list object."""
    entry = _TABLE_CACHE.get(id(names))
    if entry is not None and entry[0] is names:
        return entry[1]
    raw, off = _normalised_blob(names)
    table = _table_class().from_bytes(raw, off)
    _TABLE_CACHE[id(names)] = (names, table, raw, off)
    return table


def table_index(names):
    return _TableIndex(table_for_names(names), names)


def pc(name, default="1"):
    return os.environ.get("LOAD8_" + name.upper(), default) != "0"


def _arg(argv, k, d=None):
    for i, v in enumerate(argv):
        if v == k and i + 1 < len(argv):
            return argv[i + 1]
    return d


def now():
    return time.perf_counter() - EARLY["t0"]


class Fut:
    def __init__(self, name, fn, *aa):
        self.name, self.v, self.e = name, None, None

        def run():
            t = now()
            try:
                self.v = fn(*aa)
            except BaseException as e_:
                self.e = e_
            R["fut"][name] = [round(t, 2), round(now(), 2)]
        self.th = threading.Thread(target=run, name="l8_" + name, daemon=True); self.th.start()

    def get(self):
        self.th.join()
        if self.e is not None:
            raise self.e
        return self.v


def early(root, argv, t0, cold):
    """top of chain_u, before big8's and big5's early(): cold big mode only (the only path this task changes)"""
    EARLY["t0"] = t0
    R["cold"] = bool(cold)
    EARLY["cold"] = bool(cold)
    if not cold:
        return
    K8.NT[0] = int(os.environ.get("FASTLOAD_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
    d, s = _arg(argv, "--design"), _arg(argv, "--suite", "visible")
    base = Path(os.environ.get("ONEPASS_DATA_ROOT") or root) / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks" / s / d
    EARLY["base"], EARLY["design"] = base, d
    if pc("notorch"):
        sys.path.append(str(ROOT / "tools" / "pre5"))
        import pre5_load as P5L
        P5L.T0 = t0
        for nm in ("torch", "dtla"):
            if nm not in sys.modules:
                sys.modules[nm] = P5L._Lazy(nm)
        R["pieces"]["notorch"] = True
    if pc("fl8"):
        sys.path.append(str(ROOT / "tools" / "comp2" / "loader"))
        import fl8_lib as FL8
        sys.modules["fastload_lib"] = FL8                  # big8's early threads, fast_load and LazyPins get these classes
        R["pieces"]["fl8"] = True
        if pc("net2"):
            import fast_load as FLD
            og = FLD.get_net
            np_ = os.path.realpath(str(base / (d + ".net")))
            EARLY["N"] = Fut("NET", og, np_)

            def get_net(path, _o=og):
                f = EARLY.get("N")
                if f is not None and f.th is not threading.current_thread():
                    f.get()
                return _o(path)
            FLD.get_net = get_net
            R["pieces"]["net2"] = True
    if pc("sdc"):
        import fast_sta_v3 as FS
        sp = os.path.realpath(str(base / (d + ".sdc")))
        EARLY["SDC"] = (sp, Fut("SDC", FS.parse_sdc, sp))
        R["pieces"]["sdc"] = True


def early_small(t0):
    """--load8 1 on pre5's path (no big mode): the parser library and, through install_small, the StaDesign. Before pre5's early()."""
    EARLY["t0"] = t0
    K8.NT[0] = int(os.environ.get("FASTLOAD_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
    sys.path.append(str(ROOT / "tools" / "speed4"))
    import fl8_lib as FL8
    FL8.set_power(0)                                       # the activity runs in pre5's child process, which reads the .def itself
    sys.modules["fastload_lib"] = FL8
    R["pieces"]["small"] = True


def install_small(P5L):
    """pre5's _install_fast + the load8 StaDesign on top of fast_load's"""
    if not pc("sta"):
        return
    o = P5L._install_fast

    def w(NS, a):
        FLD = o(NS, a)
        import fast_sta_v3 as FS
        EARLY["no_power"] = True
        FS.StaDesign = make_sta(FS, FLD)
        return FLD
    P5L._install_fast = w
    R["pieces"]["sta"] = True


def install(FS, FLD, a, G, PRE, H, BEST, bigm, B8L):
    """__main__ wrapper of load_all, after fast_load.install(): cap, StaDesign, activity source; replaces big8's wrap_sta"""
    if not EARLY.get("cold"):
        return False
    if pc("cap"):
        import ispd25_adapter as AD
        import build2_fast as B2
        capp = os.path.realpath(str(EARLY["base"] / (EARLY["design"] + ".cap")))
        o_b2 = B2.parse_cap_fast                             # the C++ tokenizer; fast_load's AD.parse_cap_auto calls this same function
        EARLY["CAP"] = Fut("CAP", o_b2, capp)
        o_ad = AD.parse_cap_auto

        def b2_cap(path, _o=o_b2):
            if os.path.realpath(path) == capp:
                return EARLY["CAP"].get()
            return _o(path)

        def ad_cap(path, _o=o_ad):                           # the state's parse: same file, its own arrays
            if os.path.realpath(path) == capp:
                return _copy_cap(EARLY["CAP"].get())
            return _o(path)
        B2.parse_cap_fast = b2_cap; AD.parse_cap_auto = ad_cap
        R["pieces"]["cap"] = True
    if pc("sta"):
        FS.StaDesign = make_sta(FS, FLD)
        R["pieces"]["sta"] = True
    if pc("actx"):
        _patch_act()
    B8L.install_load(FS)
    if pc("bind"):
        wrap_sta(FS, G, PRE, BEST, bigm, B8L)
        R["pieces"]["bind"] = True
    else:
        B8L.wrap_sta(FS, G, PRE, H, BEST, bigm)
    return True


def _copy_cap(c):
    o = dict(c)
    o["x_edge_len"], o["y_edge_len"] = c["x_edge_len"].copy(), c["y_edge_len"].copy()
    o["short_costs"] = list(c["short_costs"])
    o["ppa_weights"] = list(c["ppa_weights"]) if c["ppa_weights"] is not None else None
    o["layers"] = [dict(l, cap=l["cap"].copy()) for l in c["layers"]]
    return o


def early_route(rr, route):
    """the route read on a thread from now on (rr waits for the route itself: P11_WAIT_ROUTE); the first read of this path gets its object"""
    if not (EARLY.get("cold") and pc("rt")):
        return rr
    route = str(route)
    box = {"f": Fut("RT", lambda: rr(route))}

    def read_route(p, *aa, **kk):
        f = box.get("f")
        if f is not None and str(p) == route and not aa and not kk:
            box.pop("f")
            return f.get()
        return rr(p, *aa, **kk)
    R["pieces"]["rt"] = True
    return read_route


def load_prior(a, IR, PRE):
    """--load8 1 notorch: pre5's prior (weights exported once per family; tools/pre5/gnn_np.py). No torch here."""
    for _kv in (a.gnn_family or []):
        _d, _f = _kv.split("=")
        IR.V2.FAMILY_FOR_DESIGN.setdefault(_d, _f)
    fam = IR.V2.family_for(a.design)
    sys.path.append(str(ROOT / "tools" / "pre5"))
    import gnn_np
    PRE["prior"] = gnn_np.Prior(str(ROOT / "tools" / "pre5" / "weights" / f"fold_{fam}.npz"))


def use_np_prior():
    return bool(EARLY.get("cold")) and pc("notorch")


def join_all():
    """end of load_all: threads whose results the chain reads right after the join"""
    for nm in ("B2P",):
        f = EARLY.pop(nm, None)
        if f is not None:
            f.get()


def join_bind(B8L):
    f = EARLY.get("BIND")
    if f is not None:
        f.get()


def _method_src(path, name, nxt):
    s = Path(path).read_text()
    a = s.index(f"    def {name}(")
    b = s.index(f"    def {nxt}(", a)
    return textwrap.dedent(s[a:b])


def _rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


def make_sta(FS, FLD):
    """== fast_load.FastStaDesign (same attributes, same values: test_sta8.py); the sorts and takes on threads, no Python per pin"""
    import fast_timer as FT
    FL = sys.modules["fastload_lib"]
    Base = FS.StaDesign
    NTP = K8.NT[0]
    s = _method_src(FS.__file__, "_clocks", "_when_false")
    s = _rep(s, "S = parse_sdc(sdc_path)", "S = L8_SDC(sdc_path)")
    s = _rep(s, 'o = np.argsort(src, kind="stable")', "o = K8.argsort(src)")
    s = _rep(s, "src, dst, inv = src[o], dst[o], inv[o]", "src, dst, inv = K8.take(src, o), K8.take(dst, o), K8.take(inv, o)")
    s = _rep(s, "st = np.searchsorted(src, np.arange(P + 1))", "st = K8.bounds(src, P + 1)")
    s = _rep(s, 'self.ep_pin = np.unique(self.ck_rows["pin"])', 'self.ep_pin = K8.unique_small(self.ck_rows["pin"], P) if self.ck_rows["pin"].dtype == np.int64 else np.unique(self.ck_rows["pin"])')
    ns = dict(FS.__dict__); ns["K8"] = K8; ns["L8_SDC"] = _sdc_get(FS)
    exec(compile(s, FS.__file__ + ":_clocks(load8)", "exec"), ns)
    clocks8 = ns["_clocks"]

    class L8Sta(Base):
        """StaDesign.__init__ (fast_load.FastStaDesign) with the numpy calls on threads"""
        def __init__(self, design, suite="visible", lib=None, kit=FS.KIT, verbose=True):
            t0 = time.time(); tp = now()
            d = os.path.join(kit, suite, design)
            try:
                DF = FL.DefData(os.path.join(d, design + ".def"))
                ND = FLD.get_net(os.path.join(d, design + ".net"))
            except Exception as e:                       # unusual input -> the original loader
                FLD.FALLBACKS.append(repr(e))
                return Base.__init__(self, design, suite, lib, kit, verbose)
            R["t"]["parse_joined_at"] = round(now(), 2); tp = now()
            ex = ThreadPoolExecutor(NTP)
            if pc("preb") and "K_new" in EARLY and "SDC" in EARLY and "PREB" not in EARLY:   # the timer's P x K state, first touched here, not on the bind's path
                def preb(_P=DF.npin):
                    K_ = EARLY["K_new"]()
                    nc = len(EARLY["SDC"][1].get()["order"])
                    if hasattr(K_.L, "csta_inc_pre"):
                        K_.L.csta_inc_pre.argtypes = [__import__("ctypes").c_int64, __import__("ctypes").c_int64]
                        K_.L.csta_inc_pre(_P, 2 * max(nc, 1))
                    return K_
                EARLY["PREB"] = Fut("PREB", preb)
            if "power_pins" not in EARLY and not EARLY.get("no_power"):
                EARLY["power_pins"] = DF.power_pins()
                _k = lambda x: x.replace("\\", "")
                EARLY["power_iid"] = DF.inst_ids({_k(i_) for i_, _p, _v in EARLY["power_pins"] if i_ != "PIN"})
            self.design, self.suite = design, suite
            self.lib = lib or FS.StaLib.load(os.path.join(kit, "NanGate45/lib"))
            self.layer_rc, self.via_r = FT.read_tech(kit)
            t1 = time.time()
            self.net_names = net_names = DF.net_names
            self.net_index = FL.LazyIndex(net_names)
            nn = len(net_names)
            inst_names, cell_names, inst_ct = DF.inst_names, DF.cell_names, DF.inst_ct
            self.inst_names, self.cell_names, self.inst_ct = inst_names, cell_names, inst_ct
            slots = DF.slots
            pin_name = DF.pin_name
            P = len(pin_name)
            self.P = P
            self.pin_name = pin_name
            self.pin_net, self.pin_inst, self.pin_slot = DF.pin_net, DF.pin_inst, DF.pin_slot
            is_port = self.pin_inst < 0
            self.pin_ct = np.where(is_port, -1, inst_ct[np.maximum(self.pin_inst, 0)])
            # ---- from here: StaDesign.__init__ verbatim, except the marked lines ----
            lib = self.lib
            cmods = [lib.cells.get(c) for c in cell_names]
            self.cmods, self.slots = cmods, slots
            unknown = [c for c, m in zip(cell_names, cmods) if m is None]
            sl_off = np.zeros(len(cell_names) + 1, np.int64)
            sl_off[1:] = np.cumsum([len(s) for s in slots])
            ns = int(sl_off[-1])
            sdir = np.zeros(ns + 1, np.int64)
            scap = np.zeros((ns + 1, 2))
            sclk = np.zeros(ns + 1, bool)
            for c, (m, sl) in enumerate(zip(cmods, slots)):
                for pn, s_ in sl.items():
                    pm = m.pin(pn) if m is not None else None
                    if pm is None:
                        continue
                    g = sl_off[c] + s_
                    sdir[g] = 0 if pm[0] == "input" else (1 if pm[0] == "output" else 2)
                    scap[g] = (pm[1], pm[2])
                    sclk[g] = pm[3]
            gs = np.where(is_port, ns, sl_off[np.maximum(self.pin_ct, 0)] + np.maximum(self.pin_slot, 0))
            self.pin_cap = scap[gs]
            pdir = K8.take(sdir, gs)
            pd = DF.pin_pd
            self.pin_isdrvdir = np.where(is_port, pd, (pdir >= 1).astype(np.int64)).astype(bool)
            self.pin_clockpin = K8.take(sclk, gs) & ~is_port
            self.is_port = is_port
            idx = np.arange(P)
            drv = np.full(nn, P, np.int64)
            m_out = self.pin_isdrvdir & ~is_port
            pn_, ii_ = self.pin_net[m_out], idx[m_out]
            drv[pn_[::-1]] = ii_[::-1]                       # == np.minimum.at(drv, pn_, ii_): ii_ rises, so the last write is the smallest
            drv2 = np.full(nn, P, np.int64)
            m_in = self.pin_isdrvdir & is_port
            pn_, ii_ = self.pin_net[m_in], idx[m_in]
            drv2[pn_[::-1]] = ii_[::-1]
            drv = np.where(drv < P, drv, drv2)
            drv[drv == P] = -1
            self.net_drv = drv
            is_drv = np.zeros(P, bool)
            is_drv[drv[drv >= 0]] = True
            self.pin_is_drv = is_drv
            self.pin_is_load = ~self.pin_isdrvdir & (drv[self.pin_net] >= 0)
            loc = np.zeros(len(inst_names), np.int64)
            counts = np.bincount(inst_ct, minlength=len(cell_names))
            order = K8.argsort(inst_ct)
            starts = np.cumsum(counts) - counts
            loc[order] = np.arange(len(inst_names)) - np.repeat(starts, counts)
            mats = []
            ip = np.nonzero(~is_port)[0]
            ctp, locp, slp = self.pin_ct[ip], loc[self.pin_inst[ip]], self.pin_slot[ip]
            for c in range(len(cell_names)):
                mats.append(np.full((counts[c], max(len(slots[c]), 1)), -1, np.int64))
            o = K8.argsort(ctp)
            b = np.r_[0, np.nonzero(np.diff(K8.take(ctp, o)))[0] + 1, len(o)]

            def fill(k):
                sel = o[b[k]:b[k + 1]]
                c = ctp[sel[0]]
                mats[c][locp[sel], slp[sel]] = ip[sel]
            list(ex.map(fill, range(len(b) - 1)))
            del o, b, ctp, locp, slp, ip
            R["t"]["sta_tables"] = round(now() - tp, 2); tp = now()
            # arcs and checks: the groups in the original loop order; masks and copies on threads; `whens` ids by first non-empty group, as the original
            whens = {"": 0}
            GR = []
            for c, (m, sl) in enumerate(zip(cmods, slots)):
                if m is None or counts[c] == 0:
                    continue
                M = mats[c]
                for pn, s_ in sl.items():
                    pm = m.pin(pn)
                    if pm is None:
                        continue
                    if pm[0] != "input":
                        for (rp, kind, sn, tcr, tcf, trt, tft, when) in m.arcs_to(pn):
                            sf = sl.get(rp)
                            if sf is None:
                                continue
                            GR.append((0, M, s_, sf, (kind, sn, tcr, tcf, trt, tft), when, m.lib))
                    for (rp, ck, trc, tfc, when) in m.checks_on(pn):
                        sr = sl.get(rp)
                        if sr is None:
                            continue
                        GR.append((1, M, s_, sr, (ck, trc, tfc), when, None))

            def mask(g):
                _, M, s_, sf, *_r = g
                ok = (M[:, sf] >= 0) & (M[:, s_] >= 0)
                return ok, int(ok.sum())
            MK = list(ex.map(mask, GR))
            na = nc = 0; POS = []
            for g, (ok, n) in zip(GR, MK):
                if n == 0:
                    POS.append(None); continue
                w = whens.setdefault(g[5], len(whens))
                if g[0] == 0:
                    POS.append((na, w)); na += n
                else:
                    POS.append((nc, w)); nc += n
            AK = ("fr", "to", "kind", "sense", "tcr", "tcf", "trt", "tft", "when", "lib")
            CK = ("pin", "cpin", "kind", "trc", "tfc", "when")
            A = {k: np.empty(na, np.int64) for k in AK}
            C = {k: np.empty(nc, np.int64) for k in CK}

            def put(i):
                g = GR[i]; ok, n = MK[i]
                if n == 0:
                    return
                o_, w = POS[i]; e = o_ + n
                t_, M, s_, sf, vals, _, ml = g
                if t_ == 0:
                    A["fr"][o_:e] = M[:, sf][ok]; A["to"][o_:e] = M[:, s_][ok]
                    for k, v in zip(("kind", "sense", "tcr", "tcf", "trt", "tft"), vals):
                        A[k][o_:e] = v
                    A["when"][o_:e] = w; A["lib"][o_:e] = ml
                else:
                    C["pin"][o_:e] = M[:, s_][ok]; C["cpin"][o_:e] = M[:, sf][ok]
                    for k, v in zip(("kind", "trc", "tfc"), vals):
                        C[k][o_:e] = v
                    C["when"][o_:e] = w
            list(ex.map(put, range(len(GR))))
            del GR, MK, POS
            self.arc = A
            self.chk = C
            self.whens = [w for w, _ in sorted(whens.items(), key=lambda kv: kv[1])]
            self.mats, self.inst_loc = mats, loc
            R["t"]["sta_arcs"] = round(now() - tp, 2); tp = now()
            t2 = time.time()
            # ---- .net join in C++ (hash of the DEF pin names / net names): beside the clocks
            def join():
                ptab = DF.pin_table(); ntab = DF.net_table()
                nf_pid = ptab.lookup_ptr(ND.pn_ptr, ND.pname_off, strip=True)
                self.nf_names = ND.names
                nraw = ND.names_raw.encode("utf-8")
                ends = np.flatnonzero(np.frombuffer(nraw, np.uint8) == 10)
                noff = np.zeros(len(ends) + 1, np.int64); noff[1:] = ends + 1
                self.nf_net = ntab.lookup_blob(nraw, noff, strip=True)
                pin_net_nf, pin_slack, ap_cnt, ap = ND.ft_arrays(FT.INF)
                self.nf_pin_net = pin_net_nf
                self.nf_pid = nf_pid
                self.ap_cnt, self.ap = ap_cnt, ap
                self.ap_pin = np.repeat(np.arange(len(nf_pid)), self.ap_cnt)
                self.ap_pid = K8.take(nf_pid, self.ap_pin)
                self.pin_slack_net = np.full(P, np.nan)
                ok = nf_pid >= 0
                self.pin_slack_net[nf_pid[ok]] = pin_slack[ok]
                self._pid = FLD._LazyPid(pin_name)
                self.X = int(self.ap[:, 1].max(initial=0)) + 2
                self.Y = int(self.ap[:, 2].max(initial=0)) + 2
                DF.close()
                ND.release_after_join()
                R["t"]["sta_join_end"] = round(now(), 2)
                if pc("bind"):                                  # beside the graph, not after it
                    EARLY["DA"] = design_arrays(self)
            fj = Fut("JOIN", join)
            self._clocks(os.path.join(d, design + ".sdc"))
            R["t"]["sta_clocks"] = round(now() - tp, 2); tp = now()
            self._graph()
            R["t"]["sta_graph"] = round(now() - tp, 2); tp = now()
            fj.get()
            ex.shutdown(wait=False)
            R["t"]["sta_end"] = round(now(), 2)
            if verbose:
                FS.log(f"{design}[{suite}] (load8): {nn:,} nets ({len(ND.names):,} in .net), {P:,} pins, {len(inst_names):,} instances, "
                       f"{len(self.arc['fr']):,} cell arcs, {len(self.ep_pin):,} endpoints, {len(unknown)} unknown cells; "
                       f"{R['t']}")

        _clocks = clocks8

        def _graph(self):
            """tools/pre5/pre5_kern.graph (== StaDesign._graph) with the levelize, the sorts, the cell edges and the group tables on threads"""
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
                lev, left = K8.levelize(P, efr, eto)
                if not left.any():
                    break
                # cycles: cell arcs inside one strongly connected component; drop latch D->Q ones first, then all
                import pre5_kern as PK
                lab = PK.scc(P, efr, eto)
                sz = np.bincount(lab)
                inside = (lab[fr[ai]] == lab[to[ai]]) & (sz[lab[to[ai]]] > 1)
                bad = inside & latch_dq if (attempt == 0 and (inside & latch_dq).any()) else inside
                FS.log(f"  {int(left.sum())} pins after cycles, {int((sz > 1).sum())} cyclic components: dropping "
                       f"{int(bad.sum())} {'latch D->Q ' if bad is not inside else ''}arcs")
                ai, latch_dq = ai[~bad], latch_dq[~bad]
                efr = np.r_[ne_drv, fr[ai]]
                eto = np.r_[ne_snk, to[ai]]
            self.lev = lev
            self.nlev = int(lev.max(initial=0)) + 1
            # net edges sorted by sink level
            lvs = K8.take(lev, ne_snk)
            o = K8.argsort(lvs)
            self.ne_drv, self.ne_snk = K8.take(ne_drv, o), K8.take(ne_snk, o)
            self.ne_lo = K8.bounds(K8.take(lvs, o), self.nlev + 1)
            del efr, eto, latch_dq, lvs, o
            # cell rf-edges: expand arcs by unateness; sorted by (level, to, orf); groups
            Q = K8.ce_edges(ai, fr, to, kind, A["sense"], A["tcr"], A["tcf"], A["trt"], A["tft"], A["lib"], lev, self.nlev, P)
            assert Q is not None, "load8: composite edge key over 64 bits"
            self.ce_arc, self.ce_irf, self.ce_orf = Q["ce_arc"], Q["ce_irf"], Q["ce_orf"]
            self.ce_td, self.ce_ts = Q["ce_td"], Q["ce_ts"]
            self.ce_from, self.ce_to = Q["ce_from"], Q["ce_to"]
            self.ce_lib = Q["ce_lib"]
            self.ce_lo = Q["ce_lo"]
            self.g_start = Q["g_start"]
            self.g_lo = Q["g_lo"]
            self.g_to = Q["g_to"]
            self.g_orf = Q["g_orf"]
            self.ce_grp = Q["ce_grp"]
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
    return L8Sta


def _sdc_get(FS):
    def get(path):
        e = EARLY.get("SDC")
        if e is not None and os.path.realpath(path) == e[0]:
            return e[1].get()
        return FS.parse_sdc(path)
    return get


def design_arrays(D):
    """== csta.Kernel.design_arrays(D) with the threaded stable sort"""
    nn, P = len(D.net_names), D.P
    ok = np.nonzero(D.ap_pid >= 0)[0]
    net = D.pin_net[D.ap_pid[ok]]
    net_ok = net >= 0
    ok, net = ok[net_ok], net[net_ok]
    o = K8.argsort(net)
    ap_idx = K8.take(ok, o)
    ap_off = np.zeros(nn + 1, np.int64); np.cumsum(np.bincount(net, minlength=nn), out=ap_off[1:])
    lump = np.zeros((nn, 2)); ld = np.nonzero(D.pin_is_load)[0]
    for rf in (0, 1):
        lump[:, rf] = np.bincount(D.pin_net[ld], weights=D.pin_cap[ld, rf], minlength=nn)
    fan = np.bincount(D.pin_net[D.pin_is_load & (D.pin_net >= 0)], minlength=nn)
    return dict(ap_off=ap_off, ap_idx=ap_idx, lumped=lump, fan=fan)


def wrap_sta(FS, G, PRE, BEST, bigm, B8L):
    """big8's wrap_sta with the bind on a thread: the moment the StaDesign is complete: bind + first timer on one thread, activity on another; __init__ returns at once"""
    Cls = FS.StaDesign
    _sp = list(sys.path)
    sys.path.append(str(ROOT / "tools" / "engine"))
    try:
        import csta_inc as CI                                  # here: csta_inc changes sys.path at import
    finally:
        sys.path[:] = _sp
    G["CI"] = CI
    hl = str(BEST / ("libi_kern." + ("dylib" if sys.platform == "darwin" else "so")))
    if pc("i8"):                                               # tools/best11/mk_i8.py: the same kernel, bind and resets on its thread pool
        hl = str(HERE / ("libi8_kern." + ("dylib" if sys.platform == "darwin" else "so")))
        R["pieces"]["i8"] = True

    EARLY["K_new"] = lambda _h=hl, _c=CI: _c.IncKernel(_h)

    class B8Sta(Cls):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk)
            if "K" in B8L.R or "BIND" in EARLY:
                return
            if pc("actx") and "power_pins" in EARLY:
                sys.modules["act_inproc"].CAP["power_pins"] = EARLY["power_pins"]
            if pc("seg") and "CS" in G:
                install_seg(G["CS"])

            def bind():
                t = time.perf_counter()
                f_ = EARLY.get("PREB")
                K = f_.get() if f_ is not None else CI.IncKernel(hl)
                K.bind_design(self)
                B8L.R["inc_info"] = K.bind_rc_design(self, EARLY.pop("DA", None) or design_arrays(self))
                K.L.w3_inc_backward.argtypes = [__import__("ctypes").c_void_p]
                B8L.R["K"] = K
                PRE["cd"], PRE["K"] = self, K
                B8L.R["bind_s"] = round(time.perf_counter() - t, 2)
                B8L.after_bind(G, PRE, bigm)
            EARLY["BIND"] = Fut("BIND", bind)
            _start_act(self)
    FS.StaDesign = B8Sta
    B8L.R["pieces"]["sta"] = True


def _start_act(SD):
    """big5's --big5_act thread start, at the same moment as the bind"""
    import big5_lib as B5L
    if not (B5L.R.get("bigm") and B5L.R.get("act_mode") == "thread"):
        return
    sys.path.append(str(ROOT / "tools" / "pre5")); sys.path.append(str(ROOT / "tools" / "speed4"))
    import pre5_load as P5L
    if P5L.T0 is None:
        P5L.T0 = B5L.R["t0"]
    if "ACT" in P5L.EARLY:
        return
    nt = int(_arg(sys.argv, "--big5_act_threads", "0") or 0) or max(2, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)
    B5L.R["act"] = {"mode": "thread", "threads": nt, "start_s": time.perf_counter() - B5L.R["t0"]}

    def _a(SD=SD, nt=nt):
        v = P5L._act(SD, nt); B5L.R["act"]["done_at_s"] = time.perf_counter() - B5L.R["t0"]; return v
    P5L.EARLY["ACT"] = P5L._Fut("ACT", _a)


def _patch_act():
    """tools/speed4/act_inproc from its source with: the instance table in C++, the stable sorts and bounds on threads. Same arrays."""
    if "act_inproc" in sys.modules:
        return
    f = ROOT / "tools" / "speed4" / "act_inproc.py"
    s = f.read_text()
    s = _rep(s, "iid = {nm: i for i, nm in enumerate(SD.inst_names)}", "iid = L8_POWER_IID()")
    s, n1 = re.subn(r'np\.argsort\(([\w\.]+), kind="stable"\)', r"K8.argsort(\1)", s)
    s, n2 = re.subn(r"np\.searchsorted\(([\w\.]+)\[(\w+)\], np\.arange\(([\w\. ]+?) \+ 1\)\)", r"K8.bounds(K8.take(\1, \2), \3 + 1)", s)
    s = _rep(s, "adj = dst[o].astype(np.int32)", "adj = K8.take(dst, o).astype(np.int32)")
    s = _rep(s, "aadj = adst[o].astype(np.int32);", "aadj = K8.take(adst, o).astype(np.int32);")
    # memory: arrays not read after these lines
    s = _rep(s, "    VP = (np.concatenate(pools) if pools else np.zeros(0, np.int64)).astype(np.int32)\n",
             "    VP = (np.concatenate(pools) if pools else np.zeros(0, np.int64)).astype(np.int32)\n    pools = vp_rows = seq_rows = func_pin = func_id = None\n")
    s = _rep(s, "    t_r = time.perf_counter()\n",
             "    src = dst = asrc = adst = o = aadj = aoff = wsrc = wdst = okw = okw2 = sel = sela = indeg = dis = okc = keep_act = ld_all = wd = net_pins_o = net_off = None\n    t_r = time.perf_counter()\n")
    R["pieces"]["actx"] = [n1, n2]
    m = types.ModuleType("act_inproc"); m.__file__ = str(f)
    m.K8 = K8; m.L8_POWER_IID = lambda: EARLY["power_iid"]
    sys.modules["act_inproc"] = m
    exec(compile(s, str(f) + "(load8)", "exec"), m.__dict__)


def tidx_key(SD):
    """== {_key(nm): i for i, nm in enumerate(SD.nf_names)}; _key(nm) = nm.replace("\\\\", "")"""
    names = SD.nf_names
    raw = names.raw.decode("utf-8") if hasattr(names, "raw") else "\n".join(names) + "\n"
    n = len(names)
    return dict(zip(raw.replace("\\", "").split("\n")[:-1], range(n)))


def state_class(Base):
    """ChainState whose tidx_key (a dict of all .net names) is made on a thread next to the load (it holds the GIL, not the cores) and joined at its first read"""
    class L8State(Base):
        @property
        def tidx_key(self):
            d = self.__dict__
            if "_l8_tk" not in d:
                f = EARLY.get("TK")
                d["_l8_tk"] = f.get() if f is not None else tidx_key(self.SD)
            return d["_l8_tk"]

        @tidx_key.setter
        def tidx_key(self, v):
            self.__dict__["_l8_tk"] = v
    return L8State


def _after_bind():
    f = EARLY.get("BIND")
    if f is not None:
        f.th.join()


def start_dicts(S):
    """the state's and the pin tables' dicts of all names hold the GIL for seconds (not the cores): on a thread AFTER the bind (the bind's Python parts would wait for them);
    no later step of the load waits for them, the chain's first read does"""
    def tk(SD=S.SD):
        _after_bind()
        return tidx_key(SD)
    if "TK" not in EARLY:
        EARLY["TK"] = Fut("TK", tk)


def pin_ranges(SD):
    """== (np.searchsorted(nf_pin_net, np.arange(nf)), np.searchsorted(nf_pin_net, np.arange(nf), side="right")); nf_pin_net rises"""
    nf = len(SD.nf_names)
    cnt = np.bincount(SD.nf_pin_net, minlength=nf)
    end = np.cumsum(cnt)
    return end - cnt, end


def pin_maps(lp, X, Y, NpinsKey, npk_dict):
    """== chain_gb L1 _pin_maps_arrays(lp, X, Y): (pc, npins key object, info)"""
    t = time.perf_counter()
    nd = lp.nd
    nn = int(nd.nn)
    pin_off = np.asarray(nd.pin_off, np.int64); ap_off = np.asarray(nd.ap_off, np.int64); aps = np.asarray(nd.aps)
    assert int(pin_off[nn]) == len(ap_off) - 1 and int(ap_off[-1]) == len(aps), (int(pin_off[nn]), len(ap_off) - 1, int(ap_off[-1]), len(aps))
    blob, off = lp.tab.keep                                     # the '\\n' terminated names without backslashes
    last = lp.tab.lookup_blob(blob, off, strip=False)
    mask = last == np.arange(nn)
    if mask.all():
        ndup, canon = 0, None
    else:
        non = ~mask
        ndup, canon = int(non.sum()) + len(np.unique(last[non])), mask
    pc_, cnt = K8.pin_maps(nn, pin_off, ap_off, aps, canon, X, Y)
    npk = NpinsKey(lp, cnt)
    if npk_dict:                                                # the dict on a thread; get() joins it
        def _d():
            f_ = EARLY.get("TK")
            if f_ is not None:
                f_.th.join()
            _after_bind()
            names = blob.decode("utf-8").split("\n")[:-1]
            c = cnt.tolist()
            npk.d = dict(zip(names, c)) if canon is None else dict(zip(itertools.compress(names, canon), itertools.compress(c, canon)))
        npk.d = None
        EARLY["NPK"] = Fut("NPK", _d)
        cls = type(npk)
        if not hasattr(cls, "_l8"):
            o_get = cls.get

            def get(self, k, d=None, _o=o_get):
                if self.d is None and "NPK" in EARLY:
                    EARLY.pop("NPK").get()
                return _o(self, k, d)
            cls.get = get; cls._l8 = True
    info = {"nets": nn, "dup_name_nets": ndup, "aps": int(len(aps)), "s": time.perf_counter() - t, "npk_dict": bool(npk_dict)}
    return pc_, npk, info


def names_tables(names, SD):
    """Return attach name tables without a Python dict or repeated C++ build."""
    key = (id(names), id(SD.net_names))
    entry = _NAMES_TABLES_CACHE.get(key)
    if entry is not None and entry[0] is names and entry[1] is SD.net_names:
        return entry[2]
    n = len(names)
    b, off = _normalised_blob(names)
    tid = SD.net_names.table().lookup_blob(b, off, strip=False)
    out = (n, names, _TableIndex(table_for_names(names), names), tid, names)
    _NAMES_TABLES_CACHE[key] = (names, SD.net_names, out)
    return out


def install_seg(CS):
    """csta.seg_from_route: names through the C++ table (cached per names tuple), columns on threads; same integers (test_kern8.py). Big routes only."""
    if getattr(CS, "_l8_seg", False):
        return
    FL = sys.modules["fastload_lib"]
    orig = CS.seg_from_route
    cache = {}

    def seg_from_route(cd, rt, grid, _o=orig):
        n = len(rt.x0)
        if n < 2000000:
            return _o(cd, rt, grid)
        names = rt.net_names
        c = cache.get("nid")
        if c is None or c[0] is not names:
            tab = cache.get("tab")
            if tab is None:
                raw = cd.nf_names.raw if hasattr(cd.nf_names, "raw") else ("\n".join(cd.nf_names) + "\n").encode("utf-8")
                ends = np.flatnonzero(np.frombuffer(raw, np.uint8) == 10)
                off = np.zeros(len(ends) + 1, np.int64); off[1:] = ends + 1
                tab = cache["tab"] = FL.Table.from_bytes(raw, off)
                if not hasattr(cd, "_nf_index"):
                    cd._nf_index = FL.LazyIndex(cd.nf_names)
            q = ("\n".join(names) + "\n").encode("utf-8")
            ends = np.flatnonzero(np.frombuffer(q, np.uint8) == 10)
            qo = np.zeros(len(ends) + 1, np.int64); qo[1:] = ends + 1
            cache["nid"] = c = (names, tab.lookup_blob(q, qo, strip=False))
        nid = c[1]
        de = np.where(nid >= 0, cd.nf_net[np.maximum(nid, 0)], -1)
        off = np.asarray(rt.offsets)
        if len(off) != len(de) + 1 or int(off[0]) != 0 or int(off[-1]) != n:
            return _o(cd, rt, grid)
        off = off if off.dtype == np.uint64 else np.asarray(off, np.int64).view(np.uint64)
        s = K8.seg(de, off, (rt.x0, rt.y0, rt.z0, rt.x1, rt.y1, rt.z1), grid)
        return s if s is not None else _o(cd, rt, grid)
    CS.seg_from_route = seg_from_route
    CS._l8_seg = True
    R["pieces"]["seg"] = True


def scan_nets(nets, hot_xy, cur):
    """the candidate scan of escape3d.escape without a Python tuple per net: (i, segs) in increasing i for the nets the loop can add a candidate for: a via segment at a hot (x, y), or i in cur.
    Any other net has an empty set there, so the candidate list is the same, in the same order."""
    Sg = getattr(nets, "segs", None)
    arr = getattr(Sg, "arr", None)
    if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 6):
        for i, (_n, segs, _x) in enumerate(nets):
            yield i, segs
        return
    off = np.asarray(Sg.off, np.int64)
    hx = np.fromiter((p_[0] for p_ in hot_xy), np.int64, len(hot_xy)); hy = np.fromiter((p_[1] for p_ in hot_xy), np.int64, len(hot_xy))
    x1, y1 = arr[:, 0], arr[:, 1]
    X = int(max(x1.max(initial=0), hx.max(initial=0))) + 1; Y = int(max(y1.max(initial=0), hy.max(initial=0))) + 1
    mp = np.zeros((X, Y), bool)
    mp[hx, hy] = True
    rows = np.flatnonzero((arr[:, 2] != arr[:, 5]) & mp[x1, y1])
    ids = set(np.unique(np.searchsorted(off, rows, side="right") - 1).tolist())
    ids.update(cur)
    for i in sorted(ids):
        yield i, Sg[i]


def patch_escape(E):
    """escape3d.escape from its source with the candidate scan of scan_nets (the rest of the function untouched). 64% of the first-pass escape on bsg."""
    if not pc("escscan") or getattr(E, "_l8", False):
        return
    s_ = Path(E.__file__).read_text()
    a = s_.index("def escape(cap, pins, nets,")
    m_ = re.search(r"\n(?=[^\s#])", s_[a + 10:])
    f = s_[a:a + 10 + m_.start()]
    f = _rep(f, "for i, (name, segs, _) in enumerate(nets):\n            if i in cur:", "for i, segs in L8_SCAN(nets, hot_xy, cur):\n            if i in cur:")
    E.__dict__["L8_SCAN"] = scan_nets
    exec(compile(f, E.__file__ + ":escape(load8)", "exec"), E.__dict__)
    E._l8 = True
    R["pieces"]["escscan"] = True


def patch_b2(B2):
    """Patch a baseline builder to use the cached C++ name table."""
    if not pc("names") or getattr(B2, "_l8", False):
        return
    s_ = Path(B2.__file__).read_text()
    a = s_.index("def build_fast2(")
    m_ = re.search(r"\n(?=[^\s#])", s_[a + 10:])            # the next statement at column 0
    b = a + 10 + m_.start() if m_ else len(s_)
    f = s_[a:b]
    f = _rep(f, "pin_idx = {_key(n): i for i, n in enumerate(pn_names)}", "pin_idx = L8_TABLE_INDEX(pn_names)")
    f = _rep(f, "idx = {_key(n): i for i, n in enumerate(net_names_rt)}", "idx = L8_TABLE_INDEX(net_names_rt)")
    B2.__dict__["L8_TABLE_INDEX"] = table_index
    exec(compile(f, B2.__file__ + ":build_fast2(load8)", "exec"), B2.__dict__)
    B2._l8 = True
    R["pieces"]["b2names"] = True


def record():
    out = {k: v for k, v in R.items()}
    try:
        import pre5_load as P5L
        out["lazy_real_imports"] = {k: {"at_s": v.get("at_s"), "s": v.get("s"), "thread": v.get("thread")} for k, v in P5L.R["lazy"].items()}
    except Exception:
        pass
    try:
        out["fastload_fallbacks"] = list(sys.modules["fast_load"].FALLBACKS)
        out["fastload_stats"] = {k: round(v, 3) for k, v in sys.modules["fast_load"].STATS.items()}
    except Exception:
        pass
    return out
