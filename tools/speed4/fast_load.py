# speed4 copy (2026-11-06) of tools/comp2/loader/fast_load.py, unchanged except this line; library built in tools/speed4.
"""fast_load.py: drop-in faster preload for tools/c1hfast/chain_two.py (and chain_one/chain_k users of the
same classes).  install() monkeypatches, in this process only, the slow route-independent loaders with the chunked OpenMP parsers of
libfastload.so; every replacement returns arrays/dicts equal to the original (tools/comp2/loader/test_equal.py asserts it) and falls
back to the original function when the parser reports an unusual input.  Nothing persists between runs (no snapshot, no cache).
  install(): FastStaDesign (DEF + .net + hash join in C++), parse_cap_auto -> build2_fast.parse_cap_fast (C++), activity_file (C++),
             build2_fast.parse_net_fast / ispd25 .net share ONE parse of the .net (lazy per-net access points for the escape step)
"""
import os, sys, time, collections.abc
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STATS = collections.OrderedDict()
FALLBACKS = []


def _t(label, t0):
    STATS[label] = STATS.get(label, 0.0) + time.perf_counter() - t0


_NET = {}                         # path -> NetData (one parse per .net file per process)


def get_net(path):
    import fastload_lib as FL
    path = os.path.abspath(path)
    if path not in _NET:
        t = time.perf_counter(); _NET[path] = FL.NetData(path); _t("net_parse_C++", t)
    return _NET[path]


class _LazyPid:
    """== {pin_name: index} (later duplicate wins), built only when somebody asks (the chain does not)"""
    def __init__(self, names): self._n, self._d = names, None
    def _m(self):
        if self._d is None: self._d = {nm: i for i, nm in enumerate(self._n)}
        return self._d
    def get(self, k, d=None): return self._m().get(k, d)
    def __getitem__(self, k): return self._m()[k]
    def __contains__(self, k): return k in self._m()
    def __len__(self): return len(self._n)
    def __iter__(self): return iter(self._m())
    def items(self): return self._m().items()
    def keys(self): return self._m().keys()
    def values(self): return self._m().values()


class LazyPins(collections.abc.Mapping):
    """== {_key(net name): [[(layer, x, y), ...] per pin with >= 1 access point]} of ispd25_adapter.parse_net_ispd25 (what escape3d.escape reads),
    but each net's list is built on access (escape touches ~1.3k nets of 3.3M)."""
    def __init__(self, nd):
        import fastload_lib as FL
        self.nd = nd
        raw = nd.names_raw.replace("\\", "").encode("utf-8")
        ends = np.flatnonzero(np.frombuffer(raw, np.uint8) == 10)
        off = np.zeros(len(ends) + 1, np.int64); off[1:] = ends + 1
        self.tab = FL.Table.from_bytes(raw, off)
        self.nets = nd.nn
    def _idx(self, k):
        return self.tab.find(k, strip=False) if isinstance(k, str) else -1
    def __contains__(self, k): return self._idx(k) >= 0
    def __getitem__(self, k):
        i = self._idx(k)
        if i < 0: raise KeyError(k)
        nd = self.nd
        out = []
        for p in range(int(nd.pin_off[i]), int(nd.pin_off[i + 1])):
            a, b = int(nd.ap_off[p]), int(nd.ap_off[p + 1])
            if b > a: out.append([tuple(r) for r in nd.aps[a:b].tolist()])
        return out
    def __len__(self): return self.nets
    def __iter__(self):
        raw = self.nd.names_raw.replace("\\", "")
        return iter(raw.split("\n")[:-1])


def make_sta_design():
    import fast_sta_v3 as FS
    import fast_timer as FT
    import fastload_lib as FL
    _Orig = FS.StaDesign
    _key = FS._key
    np_ = np

    class FastStaDesign(_Orig):
        """StaDesign.__init__ with the DEF netlist, the pin loop and the .net join done by libfastload (same arrays)."""
        def __init__(self, design, suite="visible", lib=None, kit=FS.KIT, verbose=True):
            t0 = time.time(); tp = time.perf_counter()
            d = os.path.join(kit, suite, design)
            try:
                DF = FL.DefData(os.path.join(d, design + ".def"))
                ND = get_net(os.path.join(d, design + ".net"))
            except Exception as e:                       # unusual input -> the original loader
                FALLBACKS.append(repr(e))
                return _Orig.__init__(self, design, suite, lib, kit, verbose)
            _t("def_parse_C++", tp)
            tp = time.perf_counter()
            self.design, self.suite = design, suite
            self.lib = lib or FS.StaLib.load(os.path.join(kit, "NanGate45/lib"))
            self.layer_rc, self.via_r = FT.read_tech(kit)
            t1 = time.time()
            self.net_names = net_names = DF.net_names
            self.net_index = {nm: i for i, nm in enumerate(net_names)}
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
            _t("py_lists", tp); tp = time.perf_counter()
            # ---- from here: StaDesign.__init__ verbatim ----
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
                for pn, s in sl.items():
                    pm = m.pin(pn) if m is not None else None
                    if pm is None:
                        continue
                    g = sl_off[c] + s
                    sdir[g] = 0 if pm[0] == "input" else (1 if pm[0] == "output" else 2)
                    scap[g] = (pm[1], pm[2])
                    sclk[g] = pm[3]
            gs = np.where(is_port, ns, sl_off[np.maximum(self.pin_ct, 0)] + np.maximum(self.pin_slot, 0))
            self.pin_cap = scap[gs]
            pdir = sdir[gs]
            pd = DF.pin_pd
            self.pin_isdrvdir = np.where(is_port, pd, (pdir >= 1).astype(np.int64)).astype(bool)
            self.pin_clockpin = sclk[gs] & ~is_port
            self.is_port = is_port
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
                    tp_ = M[:, s]
                    if pm[0] != "input":
                        for (rp, kind, sn, tcr, tcf, trt, tft, when) in m.arcs_to(pn):
                            sf = sl.get(rp)
                            if sf is None:
                                continue
                            fp = M[:, sf]
                            ok = (fp >= 0) & (tp_ >= 0)
                            if not ok.any():
                                continue
                            n = int(ok.sum())
                            A["fr"].append(fp[ok]); A["to"].append(tp_[ok])
                            for k, v in (("kind", kind), ("sense", sn), ("tcr", tcr), ("tcf", tcf), ("trt", trt),
                                         ("tft", tft), ("when", whens.setdefault(when, len(whens))), ("lib", m.lib)):
                                A[k].append(np.full(n, v, np.int64))
                    for (rp, ck, trc, tfc, when) in m.checks_on(pn):
                        sr = sl.get(rp)
                        if sr is None:
                            continue
                        cp = M[:, sr]
                        ok = (cp >= 0) & (tp_ >= 0)
                        if not ok.any():
                            continue
                        n = int(ok.sum())
                        C["pin"].append(tp_[ok]); C["cpin"].append(cp[ok])
                        for k, v in (("kind", ck), ("trc", trc), ("tfc", tfc), ("when", whens.setdefault(when, len(whens)))):
                            C[k].append(np.full(n, v, np.int64))
            cat = lambda l: np.concatenate(l) if l else np.zeros(0, np.int64)
            self.arc = {k: cat(v) for k, v in A.items()}
            self.chk = {k: cat(v) for k, v in C.items()}
            self.whens = [w for w, _ in sorted(whens.items(), key=lambda kv: kv[1])]
            self.mats, self.inst_loc = mats, loc
            _t("arcs_numpy", tp); tp = time.perf_counter()
            t2 = time.time()
            # ---- .net join in C++ (hash of the DEF pin names / net names) ----
            ptab = DF.pin_table(); ntab = DF.net_table()
            nf_pid = ptab.lookup_blob(ND.pnames_blob, ND.pname_off, strip=True)
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
            self.ap_pid = nf_pid[self.ap_pin]
            self.pin_slack_net = np.full(P, np.nan)
            ok = nf_pid >= 0
            self.pin_slack_net[nf_pid[ok]] = pin_slack[ok]
            self._pid = _LazyPid(pin_name)
            self.X = int(self.ap[:, 1].max(initial=0)) + 2
            self.Y = int(self.ap[:, 2].max(initial=0)) + 2
            DF.close()
            t3 = time.time()
            _t("net_join", tp); tp = time.perf_counter()
            self._clocks(os.path.join(d, design + ".sdc"))
            self._graph()
            _t("clocks_graph", tp)
            if verbose:
                FS.log(f"{design}[{suite}] (fast_load): {nn:,} nets ({len(ND.names):,} in .net), {P:,} pins, {len(inst_names):,} instances, "
                       f"{len(self.arc['fr']):,} cell arcs, {len(self.ep_pin):,} endpoints, {len(unknown)} unknown cells; "
                       f"DEF+lists {t1 - t0:.1f}s, arcs {t2 - t1:.1f}s, .net {t3 - t2:.1f}s, clocks+graph {time.time() - t3:.1f}s")

    return FastStaDesign


def fast_activity_file(self, path):
    """== fastmode4.State4.activity_file (last duplicate wins)"""
    import fastload_lib as FL
    t = time.perf_counter()
    try:
        A = FL.ActData(path)
        tab = self.SD_net_table() if hasattr(self, "SD_net_table") else None
        names = self.SD.net_names
        raw = ("\n".join(names) + "\n").encode("utf-8")
        ends = np.flatnonzero(np.frombuffer(raw, np.uint8) == 10)
        off = np.zeros(len(ends) + 1, np.int64); off[1:] = ends + 1
        T = FL.Table.from_bytes(raw, off)
        idx = T.lookup_blob(A.names, A.off, strip=True)
    except Exception as e:
        FALLBACKS.append("activity: " + repr(e))
        return _orig_activity(self, path)
    dens = np.zeros(len(self.SD.net_names))
    ok = idx >= 0
    ii, vv = idx[ok], A.val[ok]
    if len(ii):
        # last occurrence wins (as the sequential python loop)
        _, first_rev = np.unique(ii[::-1], return_index=True)
        last = len(ii) - 1 - first_rev
        dens[ii[last]] = vv[last]
    self.set_activity(dens)
    _t("activity_C++", t)


_orig_activity = None
_installed = False


def install():
    global _orig_activity, _installed
    if _installed:
        return
    import fast_sta_v3 as FS
    import fastmode4 as F4
    import ispd25_adapter as AD
    root = HERE.parents[1]
    sys.path.insert(0, str(root / "tools" / "onepass"))
    import build2_fast as B2
    FS.StaDesign = make_sta_design()
    _orig_activity = F4.State4.activity_file
    F4.State4.activity_file = fast_activity_file
    _oc = AD.parse_cap_auto
    def pc(path, _o=_oc, _f=B2.parse_cap_fast):
        t = time.perf_counter()
        try:
            return _f(path)
        except Exception as e:
            FALLBACKS.append("cap: " + repr(e)); return _o(path)
        finally:
            _t("cap_C++", t)
    AD.parse_cap_auto = pc
    _on = B2.parse_net_fast
    def pn(path, _o=_on):
        try:
            return get_net(path).b2_form()
        except Exception as e:
            FALLBACKS.append("b2net: " + repr(e)); return _o(path)
    B2.parse_net_fast = pn
    _installed = True
