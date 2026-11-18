#!/usr/bin/env python3
"""pre5_kern.py: exact replacements for Python-heavy preload pieces (an internal note).
  levelize(P, efr, eto) == fast_sta_v3._levelize            (C++ Kahn, tools/pre5/pre_kern.cc pre_levelize; integers)
  scc(P, efr, eto)      -> labels with the partition of scipy connected_components(directed, strong) (C++ Tarjan)
  graph(SD)             == fast_sta_v3.StaDesign._graph, copied verbatim except the two calls above (installed on the class)
  pcmap(net, X, Y, key) == ovf_lib.pin_count_map(h_netpins.esc_pins_from_fast(E, net), X, Y) from the build2 .net arrays
                           (per net key: the LAST net of that key, as the dict; unique (x, y) per net; in-bounds count; float64)
"""
import ctypes, os, sys
import numpy as np
import fast_sta_v3 as FS
from fast_sta_v3 import BIG                                          # noqa: F401  (names the copied body uses)
HERE = os.path.dirname(os.path.abspath(__file__))
_L = ctypes.CDLL(os.path.join(HERE, "libpre_kern." + ("dylib" if sys.platform == "darwin" else "so")))
_V, _I = ctypes.c_void_p, ctypes.c_int64
_L.pre_levelize.argtypes = [_I, _I, _V, _V, _V, _V]
_L.pre_scc.argtypes = [_I, _I, _V, _V, _V]; _L.pre_scc.restype = ctypes.c_int


def levelize(P, efr, eto):
    efr = np.ascontiguousarray(efr, np.int64); eto = np.ascontiguousarray(eto, np.int64)
    lev = np.zeros(P, np.int64); left = np.zeros(P, np.uint8)
    _L.pre_levelize(P, len(efr), efr.ctypes.data, eto.ctypes.data, lev.ctypes.data, left.ctypes.data)
    return lev, left.astype(bool)


def scc(P, efr, eto):
    efr = np.ascontiguousarray(efr, np.int64); eto = np.ascontiguousarray(eto, np.int64)
    lab = np.zeros(P, np.int64)
    _L.pre_scc(P, len(efr), efr.ctypes.data, eto.ctypes.data, lab.ctypes.data)
    return lab


def pcmap(net, X, Y, key):
    names, pin_off, ap_off, aps = net
    nn = len(names)
    last = {}
    for i, nm in enumerate(names):
        last[key(nm)] = i
    canon = np.zeros(nn, bool)
    canon[np.fromiter(last.values(), np.int64, len(last))] = True
    pin_off = np.asarray(pin_off, np.int64); ap_off = np.asarray(ap_off, np.int64)
    pin_net = np.repeat(np.arange(nn, dtype=np.int64), np.diff(pin_off))
    ap_net = np.repeat(pin_net, np.diff(ap_off))
    x = np.asarray(aps[:, 1], np.int64); y = np.asarray(aps[:, 2], np.int64)
    ok = (x >= 0) & (x < X) & (y >= 0) & (y < Y) & canon[ap_net]
    XY = int(X) * int(Y)
    code = np.unique(ap_net[ok] * XY + x[ok] * int(Y) + y[ok])
    return np.bincount(code % XY, minlength=XY).astype(np.float64).reshape(int(X), int(Y))


def graph(self):
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
        lev, left = levelize(P, efr, eto)                     # pre5: C++ (exact)
        if not left.any():
            break
        # cycles: cell arcs inside one strongly connected component; drop latch D->Q ones first, then all
        lab = scc(P, efr, eto)                                        # pre5: C++ Tarjan (same partition as scipy)
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
