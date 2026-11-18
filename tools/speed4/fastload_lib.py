"""fastload_lib.py: ctypes front of libfastload.so (OpenMP chunked parsers, fastload.cc).
Each function returns the same data as the Python original it replaces (see the docstrings); `None`-returning fallbacks are
handled by the callers (fast_load.py)."""
import ctypes as c, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NT = int(os.environ.get("FASTLOAD_THREADS", "8"))
_L = c.CDLL(os.path.join(HERE, "libfastload.so"))
vp, i64, i32 = c.c_void_p, c.c_int64, c.c_int
_L.net_parse2.argtypes = [c.c_char_p, i32]; _L.net_parse2.restype = vp
for n in ("net2_nnets", "net2_npins", "net2_naps"):
    getattr(_L, n).argtypes = [vp]; getattr(_L, n).restype = i64
_L.net2_bad.argtypes = [vp]; _L.net2_bad.restype = i32
for n in ("net2_names", "net2_pnames"):
    getattr(_L, n).argtypes = [vp, c.POINTER(i64)]; getattr(_L, n).restype = vp
for n in ("net2_pin_off", "net2_pname_off", "net2_ap_off", "net2_slack", "net2_nparen", "net2_aps"):
    getattr(_L, n).argtypes = [vp]; getattr(_L, n).restype = vp
_L.net2_free.argtypes = [vp]
_L.nt_build.argtypes = [vp, vp, i64]; _L.nt_build.restype = vp
_L.nt_free.argtypes = [vp]
_L.nt_find.argtypes = [vp, c.c_char_p, i64, i32]; _L.nt_find.restype = i64
_L.nt_lookup.argtypes = [vp, vp, vp, i64, i32, i32, vp]
_L.def_parse.argtypes = [c.c_char_p, i32]; _L.def_parse.restype = vp
_L.def_bad.argtypes = [vp]; _L.def_bad.restype = i32
_L.def_err.argtypes = [vp]; _L.def_err.restype = c.c_char_p
_L.def_count.argtypes = [vp, i32]; _L.def_count.restype = i64
_L.def_blob.argtypes = [vp, i32, c.POINTER(i64)]; _L.def_blob.restype = vp
_L.def_arr.argtypes = [vp, i32]; _L.def_arr.restype = vp
_L.def_pin_table.argtypes = [vp]; _L.def_pin_table.restype = vp
_L.def_net_table.argtypes = [vp]; _L.def_net_table.restype = vp
_L.def_free.argtypes = [vp]
_L.act_parse.argtypes = [c.c_char_p, i32]; _L.act_parse.restype = vp
_L.act_n.argtypes = [vp]; _L.act_n.restype = i64
_L.act_names.argtypes = [vp, c.POINTER(i64)]; _L.act_names.restype = vp
_L.act_off.argtypes = [vp]; _L.act_off.restype = vp
_L.act_val.argtypes = [vp]; _L.act_val.restype = vp
_L.act_free.argtypes = [vp]


def _arr(ptr, n, ctype, dtype):
    if n == 0 or not ptr:
        return np.empty(0, dtype)
    return np.ctypeslib.as_array(c.cast(ptr, c.POINTER(ctype)), shape=(n,)).copy()


def _blob(ptr, ln):
    return c.string_at(ptr, ln).decode("utf-8") if ln else ""


def _lines(ptr, ln):
    s = _blob(ptr, ln)
    return s.split("\n")[:-1] if s else []


class NetData:
    """parsed .net (all pin lines of the fast_timer pattern); derived forms below. Raw pin names / slack seconds / APs."""
    def __init__(self, path, nt=NT):
        h = _L.net_parse2(os.fsencode(path), nt)
        if not h:
            raise IOError(path)
        try:
            if _L.net2_bad(h):
                raise ValueError("net parser: access point count mismatch")
            self.nn, self.npn, self.nap = (_L.net2_nnets(h), _L.net2_npins(h), _L.net2_naps(h))
            ln = i64(); p = _L.net2_names(h, c.byref(ln)); self.names_raw = _blob(p, ln.value)
            self.names = self.names_raw.split("\n")[:-1] if self.nn else []
            ln = i64(); self.pn_ptr = _L.net2_pnames(h, c.byref(ln)); self.pn_len = ln.value
            self.pnames_blob = c.string_at(self.pn_ptr, self.pn_len)             # raw bytes, '\n' terminated
            self.pin_off = _arr(_L.net2_pin_off(h), self.nn + 1, c.c_int64, np.int64)
            self.pname_off = _arr(_L.net2_pname_off(h), self.npn + 1, c.c_int64, np.int64)
            self.ap_off = _arr(_L.net2_ap_off(h), self.npn + 1, c.c_int64, np.int64)
            self.slack = _arr(_L.net2_slack(h), self.npn, c.c_double, np.float64)
            self.nparen = _arr(_L.net2_nparen(h), self.npn, c.c_int32, np.int32)
            self.aps = _arr(_L.net2_aps(h), self.nap * 3, c.c_int32, np.int32).reshape(self.nap, 3)
        finally:
            _L.net2_free(h)

    # fast_timer.parse_net_fast form (minus the pin_name list: pass it to match_pins instead)
    def ft_arrays(self, INF):
        pin_net = np.repeat(np.arange(self.nn, dtype=np.int64), np.diff(self.pin_off))
        s = self.slack
        with np.errstate(invalid="ignore"):
            slack = np.where(np.abs(s) < 1e20, s * 1e9, INF)
        return pin_net, slack, self.nparen.astype(np.int64), self.aps.astype(np.int64)

    # build2_fast.parse_net_fast form: pins with >= 1 AP only
    def b2_form(self):
        ntrip = np.diff(self.ap_off)
        keep = ntrip > 0
        pin_net = np.repeat(np.arange(self.nn, dtype=np.int64), np.diff(self.pin_off))
        cnt = np.bincount(pin_net[keep], minlength=self.nn)
        pin_off = np.zeros(self.nn + 1, np.int64); pin_off[1:] = np.cumsum(cnt)
        ap_off = np.zeros(int(keep.sum()) + 1, np.int64); ap_off[1:] = np.cumsum(ntrip[keep])
        return self.names, pin_off, ap_off, self.aps.copy()

    def pin_names_list(self):
        return _lines(self.pn_ptr, self.pn_len)


class Table:
    """exact-name hash table over a '\\n' terminated blob; later duplicates win (python dict comprehension semantics)."""
    def __init__(self, blob_ptr, off_ptr, n, owner=None):
        self.h = _L.nt_build(blob_ptr, off_ptr, n); self.owner = owner; self.n = n

    @classmethod
    def from_bytes(cls, blob, off):
        """blob: bytes of '\\n' terminated names; off: int64 offsets (len n+1)"""
        t = cls.__new__(cls); t.keep = (blob, off)
        t.h = _L.nt_build(c.cast(c.c_char_p(blob), vp), off.ctypes.data, len(off) - 1); t.owner = None; t.n = len(off) - 1
        return t

    def find(self, name, strip=True):
        b = name.encode("utf-8")
        return _L.nt_find(self.h, b, len(b), 1 if strip else 0)

    def lookup_blob(self, qblob, qoff, strip=True, nt=NT):
        m = len(qoff) - 1
        out = np.empty(m, np.int64)
        qb = c.c_char_p(qblob)
        _L.nt_lookup(self.h, c.cast(qb, vp), qoff.ctypes.data, m, 1 if strip else 0, nt, out.ctypes.data)
        return out

    def lookup_list(self, names, strip=True):
        bl = ("\n".join(names) + "\n").encode("utf-8") if names else b""
        off = np.zeros(len(names) + 1, np.int64)
        off[1:] = np.cumsum(np.fromiter((len(x) + 1 for x in bl.split(b"\n")[:-1]), np.int64, len(names))) if names else 0
        return self.lookup_blob(bl, off, strip)

    def __del__(self):
        try: _L.nt_free(self.h)
        except Exception: pass


class DefData:
    """== parse_def_netlist + the pin loop of fast_sta_v3.StaDesign.__init__ (keyed names, power nets dropped, slots by first use)."""
    def __init__(self, path, nt=NT):
        h = _L.def_parse(os.fsencode(path), nt)
        if not h:
            raise IOError(path)
        self.h = h
        if _L.def_bad(h):
            err = _L.def_err(h).decode(); _L.def_free(h); self.h = None
            raise ValueError("def parser: " + err)
        cnt = [_L.def_count(h, k) for k in range(5)]
        self.ninst, self.nnet, self.npin, self.ncell, self.nport = cnt
        def blob(k):
            ln = i64(); p = _L.def_blob(h, k, c.byref(ln)); return p, ln.value
        self._b = {k: blob(k) for k in range(5)}
        self.inst_names = _lines(*self._b[0])
        self.cell_names = _lines(*self._b[1])
        self.net_names = _lines(*self._b[2])
        self.pin_name = _lines(*self._b[3])
        sn = _lines(*self._b[4])
        sc = _arr(_L.def_arr(h, 5), self.ncell, c.c_int32, np.int32)
        self.slots = []; k = 0
        for n in sc.tolist():
            self.slots.append({nm: j for j, nm in enumerate(sn[k:k + n])}); k += n
        self.inst_ct = _arr(_L.def_arr(h, 0), self.ninst, c.c_int32, np.int32).astype(np.int64)
        self.pin_net = _arr(_L.def_arr(h, 1), self.npin, c.c_int32, np.int32).astype(np.int64)
        self.pin_inst = _arr(_L.def_arr(h, 2), self.npin, c.c_int32, np.int32).astype(np.int64)
        self.pin_slot = _arr(_L.def_arr(h, 3), self.npin, c.c_int32, np.int32).astype(np.int64)
        self.pin_pd = _arr(_L.def_arr(h, 4), self.npin, c.c_uint8, np.uint8).astype(np.int64)     # 1 for INPUT/INOUT ports (and default)
        self.inst_off = _arr(_L.def_arr(h, 6), self.ninst + 1, c.c_int64, np.int64)
        self.net_off = _arr(_L.def_arr(h, 7), self.nnet + 1, c.c_int64, np.int64)

    def pin_table(self):
        return _Ext(_L.def_pin_table(self.h), self)

    def net_table(self):
        return _Ext(_L.def_net_table(self.h), self)

    def close(self):
        if self.h:
            _L.def_free(self.h); self.h = None


class _Ext(Table):
    """table owned by the DefData handle (freed with it)."""
    def __init__(self, h, owner):
        self.h = h; self.owner = owner; self.n = 0
    def __del__(self):
        pass


class ActData:
    """activity file -> (names blob, offsets, values) for lines accepted by fastmode4.State4.activity_file"""
    def __init__(self, path, nt=NT):
        h = _L.act_parse(os.fsencode(path), nt)
        if not h:
            raise IOError(path)
        try:
            n = _L.act_n(h); ln = i64(); p = _L.act_names(h, c.byref(ln))
            self.names = c.string_at(p, ln.value); self.n = n
            self.off = _arr(_L.act_off(h), n + 1, c.c_int64, np.int64)
            self.val = _arr(_L.act_val(h), n, c.c_double, np.float64)
        finally:
            _L.act_free(h)
