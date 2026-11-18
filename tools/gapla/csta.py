"""csta.py: ctypes front end of tools/gapla/csta_kern.cc (route-dependent half of fast_sta v1).
API:
  import csta
  K = csta.Kernel()                 # loads libcsta.dylib (OpenMP, uses OMP_NUM_THREADS)
  K.bind_design(D)                  # D = fast_sta.StaDesign  (or csta.load_cache(npz) -> CDesign)
  r = K.analyze(rc)                 # rc = python fast_sta RC (stage 1) ; later K.build_rc(seg) -> rc
"""
import ctypes, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEG, BIG = -1e30, 1e30

def _c(a, dt):
    return np.ascontiguousarray(np.asarray(a), dtype=dt)

class Kernel:
    def __init__(self, lib=os.path.join(HERE, "libcsta.dylib")):
        self.L = ctypes.CDLL(lib)
        self.L.csta_set.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.L.csta_seti.argtypes = [ctypes.c_char_p, ctypes.c_int64]
        self.keep = {}
    def _set(self, name, arr):
        self.keep[name] = arr
        self.L.csta_set(name.encode(), arr.ctypes.data)
    def _seti(self, name, v):
        self.L.csta_seti(name.encode(), int(v))
    def bind_design(self, D):
        S, T = self._set, D.lib.tab
        for n, dt in (("ce_from", "i8"), ("ce_to", "i8"), ("ce_irf", "i8"), ("ce_orf", "i8"), ("ce_lib", "i8"),
                      ("ce_td", "i8"), ("ce_ts", "i8"), ("ce_lo", "i8"), ("g_lo", "i8"), ("g_start", "i8"),
                      ("g_to", "i8"), ("g_orf", "i8"), ("ne_lo", "i8"), ("ne_drv", "i8"), ("ne_snk", "i8"),
                      ("pin_net", "i8"), ("sp_pin", "i8"), ("sp_cls", "i8"), ("sp_rf", "i8")):
            S(n, _c(getattr(D, n), dt))
        for n in ("L_vth", "L_vl", "L_vh", "L_der", "L_vin", "sp_at", "ck_slew"):
            S(n, _c(getattr(D, n), "f8"))
        S("is_port", _c(D.is_port, "u1")); S("reg_ck", _c(D.reg_ck, "u1"))
        R = D.ck_rows
        S("ck_pin", _c(R["pin"], "i8")); S("ck_tr", _c(R["tr"], "i8")); S("ck_tf", _c(R["tf"], "i8"))
        S("ck_type", _c(R["type"], "i8")); S("ck_evt", _c(R["evt"], "i8"))
        S("ck_cks", _c(R["cks"], "f8")); S("ck_off", _c(R["off"], "f8"))
        S("REL", _c(D.REL, "f8"))
        S("tabA", _c(T.A, "f8")); S("tabB", _c(T.B, "f8")); S("tabV", _c(T.Vf, "f8"))
        S("tabNA", _c(T.NA, "i8")); S("tabNB", _c(T.NB, "i8"))
        for n, v in (("tabM", T.M), ("tabT", len(T.NA)), ("P", D.P), ("K", D.K), ("nlev", D.nlev),
                     ("nE", len(D.ce_from)), ("nn", len(D.net_names)), ("def_lib", D.def_lib), ("nsp", len(D.sp_pin)),
                     ("nrow", len(R["pin"])), ("nevt", D.REL.shape[1])):
            self._seti(n, v)
        self.D = D
        self.ep_pin = _c(D.ep_pin, "i8")
    def design_arrays(self, D):
        """route-independent extras (part a): per-net AP lists, lumped caps, fan-out counts"""
        nn, P = len(D.net_names), D.P
        ok = np.nonzero(D.ap_pid >= 0)[0]
        net = D.pin_net[D.ap_pid[ok]]
        net_ok = net >= 0
        ok, net = ok[net_ok], net[net_ok]
        o = np.argsort(net, kind="stable")
        ap_idx = ok[o]
        ap_off = np.zeros(nn + 1, np.int64); np.cumsum(np.bincount(net, minlength=nn), out=ap_off[1:])
        lump = np.zeros((nn, 2)); ld = np.nonzero(D.pin_is_load)[0]
        for rf in (0, 1):
            lump[:, rf] = np.bincount(D.pin_net[ld], weights=D.pin_cap[ld, rf], minlength=nn)
        fan = np.bincount(D.pin_net[D.pin_is_load & (D.pin_net >= 0)], minlength=nn)
        return dict(ap_off=ap_off, ap_idx=ap_idx, lumped=lump, fan=fan)
    def bind_rc_design(self, D, extra=None):
        extra = extra or getattr(D, 'extra', None) or self.design_arrays(D)
        S = self._set
        for n in ("ap_off", "ap_idx", "fan"): S(n, _c(extra[n], "i8"))
        S("lumped", _c(extra["lumped"], "f8"))
        S("ap", _c(D.ap, "i8")); S("ap_pid", _c(D.ap_pid, "i8")); S("net_drv", _c(D.net_drv, "i8"))
        S("pin_cap", _c(D.pin_cap, "f8")); S("layer_rc", _c(D.layer_rc, "f8")); S("via_r", _c(D.via_r, "f8"))
        S("um_per_gc", np.array([_UMGC])); 
        for n in ("pin_is_drv", "pin_is_load", "pin_isdrvdir"): S(n, _c(getattr(D, n), "u1"))
        self._seti("nlr", len(D.layer_rc)); self._seti("nvr", len(D.via_r))
    def build_rc(self, seg, stub_um=1.05):
        D = self.D; P, nn = D.P, len(D.net_names)
        seg = _c(seg, "i8")
        o = dict(c1=np.empty((nn, 2)), c2=np.empty((nn, 2)), rpi=np.empty((nn, 2)), ctot=np.empty((nn, 2)),
                 par=np.empty(nn, np.uint8), elm=np.empty((P, 2)))
        st = np.zeros(4)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        self.L.csta_build_rc.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double] + [ctypes.c_void_p] * 7
        self.L.csta_build_rc(p(seg), len(seg), stub_um, p(o["c1"]), p(o["c2"]), p(o["rpi"]), p(o["ctot"]), p(o["par"]), p(o["elm"]), p(st))
        o["stats"] = st
        return o
    def crit(self, o, rc, npaths=5000, slack_max=0.0):
        """critical-path data, == io_build.read_crit_lines(crit_inmem(D, r, npaths)): (paths, netrc)"""
        D = self.D
        ctot = np.ascontiguousarray(rc["ctot"].max(1)); rd = np.ascontiguousarray(o["net_rd_kohm"])
        st = np.zeros(4)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        self.L.csta_crit.restype = ctypes.c_int64
        self.L.csta_crit.argtypes = [ctypes.c_int64, ctypes.c_double] + [ctypes.c_void_p] * 6
        npth = self.L.csta_crit(npaths, slack_max, p(o["AT"]), p(o["BEST"]), p(o["rmin"]), p(ctot), p(rd), p(st))
        ne = None
        # entry count: poff[-1]; fetch with a 2-step call
        self.L.csta_crit_get.argtypes = [ctypes.c_void_p] * 7
        self.L.csta_crit_n.restype = ctypes.c_int64
        ne = self.L.csta_crit_n()
        poff = np.empty(npth + 1, np.int64); psl = np.empty(npth); net = np.empty(ne, np.int64); fan = np.empty(ne, np.int64)
        cap = np.empty(ne); dl = np.empty(ne); ok = np.empty(ne, np.uint8)
        self.L.csta_crit_get(p(poff), p(psl), p(net), p(fan), p(cap), p(dl), p(ok))
        return dict(npaths=npth, poff=poff, pslack=psl, net=net, fan=fan, cap=cap, dl=dl, ok=ok, t=st[0])
    def analyze(self, rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par):
            D = self.D; P, K, nn, nE = D.P, D.K, len(D.net_names), len(D.ce_from)
            for n, a in (("rc_c1", rc_c1), ("rc_c2", rc_c2), ("rc_rpi", rc_rpi), ("rc_ctot", rc_ctot), ("rc_elm", rc_elm)):
                self._set(n, _c(a, "f8"))
            self._set("rc_par", _c(rc_par, "u1"))
            nrow = len(D.ck_rows["pin"])
            o = dict(AT=np.empty((P, K, 2)), SL=np.empty((P, 2)), gate_d=np.empty(nE), wire_d=np.empty((P, 2)),
                     BEST=np.empty((P, K, 2), np.int64), net_rd=np.empty(nn), rmin=np.empty(nrow), ep_sl_all=np.empty(P),
                     RAT=np.empty((P, K, 2)), pin_slack=np.empty(P), net_slack=np.empty(nn))
            st = np.zeros(8)
            p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            self.L.csta_analyze.argtypes = [ctypes.c_void_p] * 12
            self.L.csta_analyze(p(o["AT"]), p(o["SL"]), p(o["gate_d"]), p(o["wire_d"]), p(o["BEST"]), p(o["net_rd"]),
                                p(o["rmin"]), p(o["ep_sl_all"]), p(o["RAT"]), p(o["pin_slack"]), p(o["net_slack"]), p(st))
            o["stats"] = st
            e = o["ep_sl_all"][D.ep_pin]; has = e < BIG / 2; e = e[has]
            o["ep_slack"] = e; o["ep_pin"] = D.ep_pin[has]
            o["wns"] = float(min(0.0, e.min())) if len(e) else 0.0
            o["tns"] = float(np.minimum(e, 0.0).sum()) if len(e) else 0.0
            o["net_rd_kohm"] = np.clip(o["net_rd"] * 1e3, 0.05, 20.0)
            return o

_UMGC = None
def _init_consts():
    global _UMGC
    import sys
    sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    import fast_timer as FT
    _UMGC = FT.UM_PER_GC
_init_consts()



def to_crit_data(D, cr, names=None):
    """C++ crit arrays -> (paths, netrc) == io_build.read_crit_lines(crit_inmem(...)) (same lists, same dict order)."""
    names = names if names is not None else D.net_names
    na = np.asarray(names, dtype=object) if not isinstance(names, np.ndarray) else names
    nl = na[cr["net"]].tolist()
    po = cr["poff"].tolist(); ps = cr["pslack"].tolist()
    paths = [(ps[i], nl[po[i]:po[i + 1]]) for i in range(cr["npaths"])]
    idx = np.nonzero(cr["ok"])[0]
    nets = cr["net"][idx]
    o = np.lexsort((idx, -cr["cap"][idx], nets))
    nso = nets[o]
    first = np.r_[True, nso[1:] != nso[:-1]]
    bi = idx[o][first]                      # best entry per net (max cap, earliest on ties)
    bn = nso[first]
    # dict order = first appearance of the net among the ok entries
    _, fi = np.unique(nets, return_index=True)
    un = nets[np.sort(fi)]                  # nets in first-appearance order
    pos = np.searchsorted(bn, un)
    sel = bi[pos]
    keys = [na[n].replace("\\", "") for n in un] if any("\\" in x for x in names) else na[un].tolist()
    vals = list(zip(cr["fan"][sel].tolist(), cr["cap"][sel].tolist(), cr["dl"][sel].tolist()))
    if len(set(keys)) != len(keys):         # key collisions: replay entry by entry (slow path)
        rc = {}
        for i in range(cr["npaths"]):
            for t in range(po[i], po[i + 1]):
                if not cr["ok"][t]:
                    continue
                k = na[cr["net"][t]].replace("\\", "")
                last = (int(cr["fan"][t]), float(cr["cap"][t]), float(cr["dl"][t]))
                if k not in rc or last[1] > rc[k][1]:
                    rc[k] = last
        return paths, rc
    return paths, dict(zip(keys, vals))


# ---------------------------------------------------------------------------------------------------------------
# part (a): route-independent cache
# ---------------------------------------------------------------------------------------------------------------
import types
_ARR = ["ce_from", "ce_to", "ce_irf", "ce_orf", "ce_lib", "ce_td", "ce_ts", "ce_lo", "g_lo", "g_start", "g_to", "g_orf",
        "ne_lo", "ne_drv", "ne_snk", "pin_net", "sp_pin", "sp_cls", "sp_rf", "sp_at", "ck_slew", "L_vth", "L_vl", "L_vh",
        "L_der", "L_vin", "is_port", "reg_ck", "REL", "ep_pin", "ap", "ap_pid", "net_drv", "pin_cap", "layer_rc", "via_r",
        "pin_is_drv", "pin_is_load", "pin_isdrvdir", "nf_net"]
_SCAL = ["P", "K", "nlev", "def_lib"]
_ROWS = ["pin", "tr", "tf", "cks", "type", "evt", "off"]
_TAB = ["A", "B", "Vf", "NA", "NB"]

def save_cache(D, path):
    """D = fast_sta(_v3).StaDesign -> npz of the arrays the C++ kernel and the crit export need (no pickle)."""
    K = Kernel.__new__(Kernel)
    extra = Kernel.design_arrays(K, D)
    z = {n: np.asarray(getattr(D, n)) for n in _ARR}
    z.update({"x_" + n: np.asarray(v) for n, v in extra.items()})
    z.update({"s_" + n: np.asarray(int(getattr(D, n))) for n in _SCAL})
    z["s_tabM"] = np.asarray(int(D.lib.tab.M))
    z.update({"ck_" + n: np.asarray(D.ck_rows[n]) for n in _ROWS})
    z.update({"tab_" + n: np.asarray(getattr(D.lib.tab, n)) for n in _TAB})
    z["net_names_blob"] = np.frombuffer("\n".join(D.net_names).encode(), np.uint8)
    z["nf_names_blob"] = np.frombuffer("\n".join(D.nf_names).encode(), np.uint8)
    np.savez(path, **z)

def load_cache(path):
    """-> CDesign with the attributes Kernel.bind_design / bind_rc_design / crit / to_crit_data / seg_from_route use."""
    z = np.load(path)
    d = types.SimpleNamespace()
    for n in _ARR: setattr(d, n, z[n])
    for n in _SCAL: setattr(d, n, int(z["s_" + n]))
    d.ck_rows = {n: z["ck_" + n] for n in _ROWS}
    d.lib = types.SimpleNamespace(tab=types.SimpleNamespace(M=int(z["s_tabM"]), **{n: z["tab_" + n] for n in _TAB}))
    d.net_names = z["net_names_blob"].tobytes().decode().split("\n")
    d.nf_names = z["nf_names_blob"].tobytes().decode().split("\n")
    d.extra = {n[2:]: z[n] for n in z.files if n.startswith("x_")}
    d.sp_pin, d.sp_cls, d.sp_rf = z["sp_pin"], z["sp_cls"], z["sp_rf"]
    return d

def seg_from_route(cd, rt, grid):
    """route_cpp.Route (already in memory) -> int64 [S,7] == fast_sta.read_route(D, route_file)."""
    if not hasattr(cd, "_nf_index"):
        cd._nf_index = {nm: i for i, nm in enumerate(cd.nf_names)}
    idx = cd._nf_index
    nid = np.fromiter((idx.get(n, -1) for n in rt.net_names), np.int64, len(rt.net_names))
    de = np.where(nid >= 0, cd.nf_net[np.maximum(nid, 0)], -1)
    cnt = np.diff(np.asarray(rt.offsets, np.int64))
    seg = np.empty((len(rt.x0), 7), np.int64)
    seg[:, 0] = np.repeat(de, cnt)
    i8 = lambda a: np.asarray(a, np.int64)
    seg[:, 1] = i8(rt.x0) // grid; seg[:, 2] = i8(rt.y0) // grid; seg[:, 3] = i8(rt.z0)
    seg[:, 4] = i8(rt.x1) // grid; seg[:, 5] = i8(rt.y1) // grid; seg[:, 6] = i8(rt.z1)
    return seg[seg[:, 0] >= 0]
