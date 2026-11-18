"""csta_inc.py: ctypes front end of tools/engine/csta_inc_kern.cc, the INCREMENTAL
exact timer (plan item T3 / E1). Same interface as tools/gapla/csta.py Kernel (bind_design, bind_rc_design, build_rc, analyze,
crit) plus:
  K.full(c1, c2, rpi, ctot, elm, par, backward=False, copy=False)  full analysis into the persistent state
  K.begin(); K.build_rc_subset(seg, nets) | K.set_rc(nets, rc); o = K.update(nets); K.commit() | K.reject()
  o has the fields the chain uses: wns, tns, ep_slack, ep_pin, net_rd_kohm, AT, BEST, rmin (+ SL, gate_d, wire_d, net_rd,
  ep_sl_all) and o["inc"] = counters (groups, arcs, edges, rows recomputed ...), o["stats"] = kernel walls.
Views: update() and full(copy=False) return numpy VIEWS of the kernel's state (no copies); they change with the next
update/reject/full. analyze() (the csta.py API) runs full(backward=True, copy=True) and returns copies, so code written for
csta.Kernel keeps working, and the incremental state follows it.
K.rc is the dict of RC arrays owned by the kernel (bound as rc_*); build_rc_subset / set_rc change them in place (logged).
WNS / TNS are computed from the endpoint-slack array exactly as csta.py does.
"""
import ctypes, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "gapla"))
import csta as CS                                                     # noqa: E402

BIG = CS.BIG
EXT = "dylib" if sys.platform == "darwin" else "so"
_V = ctypes.c_void_p
INFO_KEYS = ["P", "K", "nlev", "nE", "nn", "nrow", "groups", "net_edges", "orphan_arcs", "to_mismatch", "neg_net",
             "multi_writer", "order_viol", "startpoints_overwritten"]
CNT_KEYS = ["seed_nets", "groups", "arcs", "edges_rf", "group_out_changed", "sinks_changed", "rows", "ep_pins", "net_rd_nets",
            "late_marks", "levels", "groups_at_only", "edges_rf_at_only", "group_out_slew_changed", "sinks_slew_changed", "pick_resolved"]


def _p(a):
    return a.ctypes.data_as(_V)


class IncKernel(CS.Kernel):
    def __init__(self, lib=None):
        super().__init__(lib or os.path.join(HERE, "libcsta_inc." + EXT))
        L = self.L
        L.csta_inc_bind.argtypes = [_V]; L.csta_inc_bind.restype = ctypes.c_int
        L.csta_inc_full.argtypes = [ctypes.c_int, _V]
        L.csta_inc_update.argtypes = [_V, ctypes.c_int64, _V, _V]; L.csta_inc_update.restype = ctypes.c_int64
        L.csta_inc_reject.restype = ctypes.c_int64
        L.csta_inc_set_rc.argtypes = [_V, ctypes.c_int64] + [_V] * 6
        L.csta_inc_build_rc_subset.argtypes = [_V, ctypes.c_int64, ctypes.c_double, _V, ctypes.c_int64, _V]
        L.csta_inc_build_rc_subset.restype = ctypes.c_int
        L.csta_inc_ptr.argtypes = [ctypes.c_char_p]; L.csta_inc_ptr.restype = _V
        L.csta_inc_dps.argtypes = [_V, _V]
        L.csta_inc_logsize.restype = ctypes.c_int64
        L.csta_inc_ngroups.restype = ctypes.c_int64
        self.inc_ok = False
        self.S = None; self.snap = {}; self.pending_snap = None      # used by the chain (route state of the kernel)

    # ------------------------------------------------------------------ binding
    def bind_rc_design(self, D, extra=None):
        super().bind_rc_design(D, extra)
        P, nn = D.P, len(D.net_names)
        self.rc = dict(c1=np.zeros((nn, 2)), c2=np.zeros((nn, 2)), rpi=np.zeros((nn, 2)), ctot=np.zeros((nn, 2)),
                       par=np.zeros(nn, np.uint8), elm=np.full((P, 2), np.nan))
        for k in ("c1", "c2", "rpi", "ctot", "elm"):
            self._set("rc_" + k, self.rc[k])
        self._set("rc_par", self.rc["par"])
        info = np.zeros(16, np.int64)
        ret = self.L.csta_inc_bind(_p(info))
        self.info = dict(zip(INFO_KEYS, info.tolist()))
        self.inc_ok = ret == 0
        self._views()
        return self.info

    def _rebind_rc(self):
        for k in ("c1", "c2", "rpi", "ctot", "elm"):
            self._set("rc_" + k, self.rc[k])
        self._set("rc_par", self.rc["par"])

    def _arr(self, name, n, dt, shape):
        ptr = self.L.csta_inc_ptr(name.encode())
        ct = {np.float64: ctypes.c_double, np.int64: ctypes.c_int64, np.uint8: ctypes.c_uint8}[dt]
        return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ct)), shape=(n,)).reshape(shape)

    def _views(self):
        D = self.D; P, K, nn, nE = D.P, D.K, len(D.net_names), len(D.ce_from); nrow = len(D.ck_rows["pin"])
        f8, i8 = np.float64, np.int64
        self.V = dict(AT=self._arr("AT", P * K * 2, f8, (P, K, 2)), SL=self._arr("SL", P * 2, f8, (P, 2)),
                      gate_d=self._arr("gate_d", nE, f8, (nE,)), wire_d=self._arr("wire_d", P * 2, f8, (P, 2)),
                      BEST=self._arr("BEST", P * K * 2, i8, (P, K, 2)), net_rd=self._arr("net_rd", nn, f8, (nn,)),
                      rmin=self._arr("rmin", nrow, f8, (nrow,)), ep_sl_all=self._arr("ep_sl", P, f8, (P,)),
                      RAT=self._arr("RAT", P * K * 2, f8, (P, K, 2)), pin_slack=self._arr("pin_slack", P, f8, (P,)),
                      net_slack=self._arr("net_slack", nn, f8, (nn,)))
        ng = int(self.L.csta_inc_ngroups())
        self.Vg = dict(arc_slew=self._arr("arc_slew", nE, f8, (nE,)), grp_rd=self._arr("grp_rd", ng, f8, (ng,)), grp_pk=self._arr("grp_pk", ng, np.uint8, (ng,)),
                       grp_pke=self._arr("grp_pke", ng, i8, (ng,)))

    def reqk(self):
        """copies of S_reqk / S_valid (the required-time rows csta_crit reads)"""
        D = self.D; n = len(D.ck_rows["pin"]) * D.K * 2
        return self._arr("reqk", n, np.float64, (n,)).copy(), self._arr("valid", n, np.uint8, (n,)).copy()

    def dps(self):
        P = self.D.P
        out = np.empty((P * 2, 16)); ai = np.empty((P * 2, 2), np.int64)
        self.L.csta_inc_dps(_p(out), _p(ai))
        return out, ai

    # ------------------------------------------------------------------ results
    def _out(self, st, copy, cnt=None):
        D = self.D
        o = {k: (v.copy() if copy else v) for k, v in self.V.items()}
        o["stats"] = st
        e = o["ep_sl_all"][D.ep_pin]; has = e < BIG / 2; e = e[has]
        o["ep_slack"] = e; o["ep_pin"] = D.ep_pin[has]
        o["wns"] = float(min(0.0, e.min())) if len(e) else 0.0
        o["tns"] = float(np.minimum(e, 0.0).sum()) if len(e) else 0.0
        o["net_rd_kohm"] = np.clip(o["net_rd"] * 1e3, 0.05, 20.0)
        if cnt is not None:
            o["inc"] = dict(zip(CNT_KEYS, cnt.tolist()))
        return o

    def full(self, rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par, backward=False, copy=False):
        R = self.rc
        for k, a in (("c1", rc_c1), ("c2", rc_c2), ("rpi", rc_rpi), ("ctot", rc_ctot), ("elm", rc_elm), ("par", rc_par)):
            if a is not R[k]:
                np.copyto(R[k], np.asarray(a).reshape(R[k].shape), casting="unsafe")
        self._rebind_rc()
        st = np.zeros(4)
        self.L.csta_inc_full(int(bool(backward)), _p(st))
        self._views()                                   # S_reqk may have moved
        if self.S is not None:                          # the chain: the kernel now holds the attached route's state
            self.snap = dict(self.S.new); self.pending_snap = None
        return self._out(st, copy)

    def analyze(self, rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par):
        """csta.Kernel.analyze API (copies, required times included), through the persistent state"""
        return self.full(rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par, backward=True, copy=True)

    def analyze_original(self, rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par):
        """the ORIGINAL csta_analyze of this library (invalidates the incremental state until the next full)"""
        o = super().analyze(rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par)
        self._rebind_rc()
        return o

    # ------------------------------------------------------------------ transactions
    def begin(self):
        self.L.csta_inc_begin()

    def set_rc(self, nets, rc):
        nets = np.ascontiguousarray(nets, np.int64)
        src = [np.ascontiguousarray(rc[k], "f8") for k in ("c1", "c2", "rpi", "ctot")]
        par = np.ascontiguousarray(rc["par"], "u1"); elm = np.ascontiguousarray(rc["elm"], "f8")
        self.L.csta_inc_set_rc(_p(nets), len(nets), *(_p(a) for a in src), _p(par), _p(elm))

    def build_rc_subset(self, seg, nets, stub_um=1.05):
        seg = np.ascontiguousarray(seg, np.int64).reshape(-1, 7)
        nets = np.ascontiguousarray(nets, np.int64)
        st = np.zeros(3)
        n = self.L.csta_inc_build_rc_subset(_p(seg), len(seg), stub_um, _p(nets), len(nets), _p(st))
        return n, st

    def update(self, nets):
        nets = np.ascontiguousarray(nets, np.int64)
        cnt = np.zeros(16, np.int64); st = np.zeros(3)
        ret = self.L.csta_inc_update(_p(nets), len(nets), _p(cnt), _p(st))
        if ret != 0:
            raise RuntimeError(f"csta_inc_update returned {ret} (bind checks: {self.info})")
        return self._out(st, False, cnt)

    def commit(self):
        self.L.csta_inc_commit()
        if self.pending_snap is not None:
            self.snap = self.pending_snap; self.pending_snap = None

    def reject(self):
        n = self.L.csta_inc_reject()
        self.pending_snap = None
        return n

    def flush_pending(self):
        self.L.csta_inc_flush_pending()

    def logsize(self):
        return int(self.L.csta_inc_logsize())


def ref_kernel(cd):
    """reference full analysis in a SEPARATE library (own static state): the original csta_analyze code of libcsta_inc_ref"""
    K = IncKernel(os.path.join(HERE, "libcsta_inc_ref." + EXT))
    K.bind_design(cd); K.bind_rc_design(cd)
    return K
