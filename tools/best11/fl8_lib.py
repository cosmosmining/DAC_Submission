#!/usr/bin/env python3
"""fl8_lib.py: ctypes front of libfl8.so (tools/best11/fl8.cc) with the API of tools/comp2/loader/fastload_lib.py, same values, and:
  * the big name lists are LazyLines (the blob + offsets; a Python string only for the name asked for; a full list only if someone iterates);
  * no second copy of the arrays (int32 -> int64 once; the C++ memory is freed right after);
  * DefData.power_text(): the records tools/speed4/act_inproc.power_pins_from_def would scan (no second read of the .def).
"""
import collections.abc, ctypes as c, os, re, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NT = int(os.environ.get("FASTLOAD_THREADS", "8"))
_L = c.CDLL(os.path.join(HERE, "libfl8.so"))
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
_L.net2_release.argtypes = [vp]
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
_L.def_inst_table.argtypes = [vp]; _L.def_inst_table.restype = vp
_L.def_free.argtypes = [vp]
_L.def_power_text.argtypes = [vp, c.POINTER(i64)]; _L.def_power_text.restype = vp
_L.fl8_set_power.argtypes = [i32]


def set_power(v):
    """0: def_parse does not keep the power records (nobody in this process asks for them)"""
    _L.fl8_set_power(int(v))
_L.act_parse.argtypes = [c.c_char_p, i32]; _L.act_parse.restype = vp
_L.act_n.argtypes = [vp]; _L.act_n.restype = i64
_L.act_names.argtypes = [vp, c.POINTER(i64)]; _L.act_names.restype = vp
_L.act_off.argtypes = [vp]; _L.act_off.restype = vp
_L.act_val.argtypes = [vp]; _L.act_val.restype = vp
_L.act_free.argtypes = [vp]


def _view(ptr, n, ctype):
    return np.ctypeslib.as_array(c.cast(ptr, c.POINTER(ctype)), shape=(n,)) if (n and ptr) else np.empty(0, np.ctypeslib.as_ctypes_type and ctype)


def _arr(ptr, n, ctype, dtype):
    if n == 0 or not ptr:
        return np.empty(0, dtype)
    return np.ctypeslib.as_array(c.cast(ptr, c.POINTER(ctype)), shape=(n,)).copy()


def _bytes(ptr, ln):
    return c.string_at(ptr, ln) if ln else b""


class LazyLines(collections.abc.Sequence):
    """== the list of the '\\n' terminated names in `raw` (bytes), offsets `off` (int64, n + 1). One name: decoded on demand. Iteration, a slice, ==: the full list, once."""
    __slots__ = ("raw", "off", "n", "_l", "_t")

    def __init__(self, raw, off):
        self.raw, self.off, self.n, self._l, self._t = raw, off, len(off) - 1, None, None

    def __len__(self):
        return self.n

    def _list(self):
        if self._l is None:
            self._l = self.raw.decode("utf-8").split("\n")[:-1] if self.n else []
        return self._l

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._list()[i]
        i = int(i)
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.raw[int(self.off[i]):int(self.off[i + 1]) - 1].decode("utf-8")

    def __iter__(self):
        return iter(self._list())

    def __eq__(self, o):
        return self._list() == (o._list() if isinstance(o, LazyLines) else o)

    def __hash__(self):
        return id(self)

    def table(self):
        """exact-name table on the raw names (later duplicate wins)"""
        if self._t is None:
            self._t = Table.from_bytes(self.raw, self.off)
        return self._t

    def to_index(self):
        """== {name: i for i, name in enumerate(self)}, at C speed"""
        return dict(zip(self._list(), range(self.n)))


class LazyIndex(collections.abc.Mapping):
    """== {name: i for i, name in enumerate(lines)}: the real dict is made at the first access"""
    __slots__ = ("lines", "_d")

    def __init__(self, lines):
        self.lines, self._d = lines, None

    def _m(self):
        if self._d is None:
            self._d = self.lines.to_index() if isinstance(self.lines, LazyLines) else dict(zip(self.lines, range(len(self.lines))))
        return self._d

    def get(self, k, d=None): return self._m().get(k, d)
    def __getitem__(self, k): return self._m()[k]
    def __contains__(self, k): return k in self._m()
    def __iter__(self): return iter(self._m())
    def __len__(self): return len(self._m())


class TableGet:
    """a dict {name: i} seen through get() only, without the dict: exact-name lookups in a Table (-1 = missing -> d)"""
    __slots__ = ("t",)

    def __init__(self, t): self.t = t
    def get(self, k, d=None):
        i = self.t.find(k, strip=False)
        return d if i < 0 else i


class _Own:
    """a C++ handle freed when no array made by arr() is alive"""
    def __init__(self, free, h): self.free, self.h = free, h

    def __del__(self):
        try:
            self.free(self.h)
        except Exception:
            pass

    def arr(self, ptr, n, ct):
        if n == 0 or not ptr:
            return np.empty(0, np.ctypeslib.as_array((ct * 1)()).dtype)
        a = (ct * n).from_address(ptr); a._own = self
        return np.frombuffer(a, dtype=np.ctypeslib.as_array((ct * 1)()).dtype)


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
            ln = i64(); p = _L.net2_names(h, c.byref(ln)); nb = _bytes(p, ln.value)
            self.names_raw = nb.decode("utf-8")
            ends = np.flatnonzero(np.frombuffer(nb, np.uint8) == 10)
            no = np.zeros(len(ends) + 1, np.int64); no[1:] = ends + 1
            self.names = LazyLines(nb, no)                                       # (the list only if someone iterates)
            ln = i64(); self.pn_ptr = _L.net2_pnames(h, c.byref(ln)); self.pn_len = ln.value   # raw bytes, '\n' terminated
            o = _Own(_L.net2_free, h); h = None                                 # the arrays below are the C++ vectors; freed with the last of them
            self._own = o
            self.pin_off = o.arr(_L.net2_pin_off(o.h), self.nn + 1, c.c_int64)
            self.pname_off = o.arr(_L.net2_pname_off(o.h), self.npn + 1, c.c_int64)
            self.ap_off = o.arr(_L.net2_ap_off(o.h), self.npn + 1, c.c_int64)
            self.slack = o.arr(_L.net2_slack(o.h), self.npn, c.c_double)
            self.nparen = o.arr(_L.net2_nparen(o.h), self.npn, c.c_int32)
            self.aps = o.arr(_L.net2_aps(o.h), self.nap * 3, c.c_int32).reshape(self.nap, 3)
        finally:
            if h:
                _L.net2_free(h)

    @property
    def pnames_blob(self):
        return c.string_at(self.pn_ptr, self.pn_len)                            # a copy; the load8 join uses lookup_ptr

    def release_after_join(self):
        """the pin names, slacks and parenthesis counts are read only by the .net join: freed now"""
        self.pn_ptr = None; self.pn_len = 0; self.pname_off = self.slack = self.nparen = None
        _L.net2_release(self._own.h)

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
        aps = self.aps.view(); aps.flags.writeable = False                       # the original's .copy(): nobody writes it (a write would raise here)
        return self.names, pin_off, ap_off, aps

    def pin_names_list(self):
        return self.pnames_blob.decode("utf-8").split("\n")[:-1] if self.pn_len else []


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

    def lookup_ptr(self, ptr, qoff, strip=True, nt=NT):
        """lookup_blob on a C pointer (no copy of the blob)"""
        m = len(qoff) - 1
        out = np.empty(m, np.int64)
        _L.nt_lookup(self.h, c.cast(ptr, vp), qoff.ctypes.data, m, 1 if strip else 0, nt, out.ctypes.data)
        return out

    def lookup_list(self, names, strip=True):
        bl = ("\n".join(names) + "\n").encode("utf-8") if names else b""
        ends = np.flatnonzero(np.frombuffer(bl, np.uint8) == 10)
        off = np.zeros(len(ends) + 1, np.int64); off[1:] = ends + 1
        return self.lookup_blob(bl, off, strip)

    def __del__(self):
        try: _L.nt_free(self.h)
        except Exception: pass


class DefData:
    """== parse_def_netlist + the pin loop of fast_sta_v3.StaDesign.__init__ (keyed names, power nets dropped, slots by first use). The lists are LazyLines."""
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
            ln = i64(); p = _L.def_blob(h, k, c.byref(ln)); return _bytes(p, ln.value)

        def lines(s):
            return s.decode("utf-8").split("\n")[:-1] if s else []
        self.inst_off = _arr(_L.def_arr(h, 6), self.ninst + 1, c.c_int64, np.int64)
        self.net_off = _arr(_L.def_arr(h, 7), self.nnet + 1, c.c_int64, np.int64)
        self.pin_off = _arr(_L.def_arr(h, 8), self.npin + 1, c.c_int64, np.int64)
        self.inst_names = LazyLines(blob(0), self.inst_off)
        self.cell_names = lines(blob(1))
        self.net_names = LazyLines(blob(2), self.net_off)
        self.pin_name = LazyLines(blob(3), self.pin_off)
        sn = lines(blob(4))
        sc = _arr(_L.def_arr(h, 5), self.ncell, c.c_int32, np.int32)
        self.slots = []; k = 0
        for n in sc.tolist():
            self.slots.append({nm: j for j, nm in enumerate(sn[k:k + n])}); k += n

        def to8(which, n, ct):
            return np.ctypeslib.as_array(c.cast(_L.def_arr(h, which), c.POINTER(ct)), shape=(n,)).astype(np.int64) if n else np.empty(0, np.int64)
        self.inst_ct = to8(0, self.ninst, c.c_int32)
        self.pin_net = to8(1, self.npin, c.c_int32)
        self.pin_inst = to8(2, self.npin, c.c_int32)
        self.pin_slot = to8(3, self.npin, c.c_int32)
        self.pin_pd = to8(4, self.npin, c.c_uint8)                         # 1 for INPUT/INOUT ports (and default)

    def power_pins(self):
        """== tools/speed4/act_inproc.power_pins_from_def(path): [(inst, pin, value)]"""
        ln = i64(); p = _L.def_power_text(self.h, c.byref(ln)); t = _bytes(p, ln.value)
        out = []; i = 0
        rx = re.compile(r"\(\s*(\S+)\s+(\S+)\s*\)")
        while i < len(t):
            v = 1 if t[i:i + 1] == b"1" else 0
            L = struct.unpack("<q", t[i + 1:i + 9])[0]
            s = t[i + 9:i + 9 + L].decode("utf-8", "surrogateescape"); i += 9 + L
            for m in rx.finditer(s):
                if m.group(1) not in ("*", "PIN"):
                    out.append((m.group(1), m.group(2), v))
        return out

    def inst_ids(self, keys):
        """{key: instance index} for the keys that are instances (the DEF's own table)"""
        t = _Ext(_L.def_inst_table(self.h), self)
        out = {}
        for k in keys:
            i = t.find(k, strip=False)
            if i >= 0:
                out[k] = i
        return out

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
