"""csta5.py: ctypes front end of tools/timer5/csta5_kern.cc with the SAME Python interface as
tools/engine/csta_inc.py (IncKernel: bind_design, bind_rc_design, build_rc, analyze, crit, full, begin, set_rc, build_rc_subset,
update, commit, reject, flush_pending, logsize, reqk, dps, K.L.w3_inc_backward, K.L.csta_hcrit). It subclasses csta_inc.IncKernel;
only the library changes:
  default library            tools/timer5/libcsta5.<so|dylib>
  CSTA5_LIB=<path>           load another library through the same wrapper (e.g. tools/best/libi_kern.so = chain_j's timer)
  CSTA5_REC=<file>           record every state-changing call (full/analyze, begin, build_rc_subset, set_rc, update, commit,
                             reject, flush_pending, w3_inc_backward; first build_rc) as a pickle stream, flushed per call
                             (the chain ends with os._exit). Read by tools/timer5/replay5.py.
"""
import ctypes, os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_sp = list(sys.path)
sys.path.append(os.path.join(ROOT, "tools", "engine"))
try:
    import csta_inc as CI                                             # noqa: E402  (inserts tools/gapla at sys.path[0])
finally:
    sys.path[:] = _sp + [p for p in sys.path if p not in _sp]
CS = CI.CS
BIG = CI.BIG
EXT = CI.EXT
DEFAULT_LIB = os.path.join(HERE, "libcsta5." + EXT)
# idea 0 (spinning pool) dropped by rule R5 of an internal note (ablation: -4.7 % / +5.0 %, no consistent gain): the
# workers sleep at once. Scheduling only; the results cannot change (route md5s with spin off: q5b S runs).
os.environ.setdefault("CSTA5_SPIN_US", "0")


class _Rec:
    def __init__(self, path):
        self.f = open(path, "ab"); self.n = 0; self.nbuild = 0

    def put(self, *ev):
        pickle.dump(ev, self.f, protocol=4); self.f.flush(); self.n += 1


class _Fn:                                   # recorded ctypes function (argtypes / restype pass through)
    def __init__(self, fn, rec, name):
        object.__setattr__(self, "_fn", fn); object.__setattr__(self, "_rec", rec); object.__setattr__(self, "_name", name)

    def __setattr__(self, k, v):
        setattr(self._fn, k, v)

    def __getattr__(self, k):
        return getattr(self._fn, k)

    def __call__(self, *a):
        r = self._fn(*a)
        self._rec.put(self._name)
        return r


class _LProxy:
    def __init__(self, L, rec):
        object.__setattr__(self, "_L", L); object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "_w3", None)

    def __getattr__(self, k):
        if k == "w3_inc_backward":
            if self._w3 is None:
                object.__setattr__(self, "_w3", _Fn(self._L.w3_inc_backward, self._rec, "w3_inc_backward"))
            return self._w3
        return getattr(self._L, k)

    def __setattr__(self, k, v):
        setattr(self._L, k, v)


class IncKernel(CI.IncKernel):
    def __init__(self, lib=None):
        lib = os.environ.get("CSTA5_LIB") or lib or DEFAULT_LIB
        super().__init__(lib)
        self.lib_path = lib
        rp = os.environ.get("CSTA5_REC")
        self._rec = _Rec(rp) if rp else None
        if self._rec:
            self.L = _LProxy(self.L, self._rec)

    def build_rc(self, seg, stub_um=1.05):
        if self._rec and self._rec.nbuild == 0:
            self._rec.nbuild += 1
            self._rec.put("build_rc", np.ascontiguousarray(seg, np.int64), float(stub_um))
        return super().build_rc(seg, stub_um)

    def full(self, rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par, backward=False, copy=False):
        o = super().full(rc_c1, rc_c2, rc_rpi, rc_ctot, rc_elm, rc_par, backward=backward, copy=copy)
        if self._rec:
            self._rec.put("full", {k: v.copy() for k, v in self.rc.items()}, bool(backward))
        return o

    def begin(self):
        super().begin()
        if self._rec:
            self._rec.put("begin")

    def set_rc(self, nets, rc):
        if self._rec:
            self._rec.put("set_rc", np.ascontiguousarray(nets, np.int64), {k: np.array(rc[k], copy=True) for k in ("c1", "c2", "rpi", "ctot", "par", "elm")})
        return super().set_rc(nets, rc)

    def build_rc_subset(self, seg, nets, stub_um=1.05):
        if self._rec:
            self._rec.put("build_rc_subset", np.ascontiguousarray(seg, np.int64).reshape(-1, 7).copy(), np.ascontiguousarray(nets, np.int64).copy(), float(stub_um))
        return super().build_rc_subset(seg, nets, stub_um)

    def update(self, nets):
        if self._rec:
            self._rec.put("update", np.ascontiguousarray(nets, np.int64).copy())
        return super().update(nets)

    def commit(self):
        super().commit()
        if self._rec:
            self._rec.put("commit")

    def reject(self):
        n = super().reject()
        if self._rec:
            self._rec.put("reject")
        return n

    def flush_pending(self):
        super().flush_pending()
        if self._rec:
            self._rec.put("flush_pending")


def ref_kernel(cd):
    K = IncKernel(os.path.join(HERE, "libcsta5_ref." + EXT))
    K.bind_design(cd); K.bind_rc_design(cd)
    return K
