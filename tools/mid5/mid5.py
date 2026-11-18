#!/usr/bin/env python3
"""mid5.py: Python side of the mid5 region drivers of tools/mid5/chain_mid.py (--mid5 1).
Every heavy piece is an extern "C" function of tools/mid5/libmid5.{so,dylib} (mid5_kern.cc) on plain arrays; this file only
binds them and keeps the chain's objects (the incremental kernel K, the state S) out of the C side.

  gate_backward(K)        piece A: after the check's kept incremental update of the prior candidate (K.commit() done), the
                          required-time pass alone (w3_inc_backward = the backward lines of csta_inc_full) instead of a full
                          K.analyze; returns the dict K.analyze would (copies of the kernel views, same keys).
  CPU                     process CPU seconds at the stage boundaries (--mid5_cpu 1; changes nothing)
  state_md5(S, cur)       md5 of S.new, S.la.d and the slack of cur (MID5_STOP dev runs)
"""
import ctypes, hashlib, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "dylib" if sys.platform == "darwin" else "so"
_V = ctypes.c_void_p
_L = [None]


def lib():
    if _L[0] is None:
        _L[0] = ctypes.CDLL(os.path.join(HERE, "libmid5." + EXT))
    return _L[0]


def _p(a):
    return a.ctypes.data_as(_V)


# ------------------------------------------------------------------ piece A
def gate_backward(K):
    st = np.zeros(2)
    K.L.w3_inc_backward(_p(st))
    o = K._out(np.zeros(4), True)                  # same keys and values as K.analyze (copies), stats = zeros
    o["stats"] = np.array([0.0, 0.0, float(st[0]), float(st[0])])
    return o


# ------------------------------------------------------------------ counters
class CPU:
    def __init__(self, on):
        self.on = bool(on); self.rec = {}; self.t = {}

    def start(self, k):
        if self.on:
            self.t[k] = (time.perf_counter(), time.process_time())

    def stop(self, k):
        if self.on and k in self.t:
            w0, c0 = self.t.pop(k)
            r = self.rec.setdefault(k, [0.0, 0.0])
            r[0] += time.perf_counter() - w0; r[1] += time.process_time() - c0

    def out(self):
        return {k: {"wall": v[0], "cpu": v[1], "thr": v[1] / max(v[0], 1e-9)} for k, v in self.rec.items()}


def state_md5(S, cur):
    h = hashlib.md5()
    for i in sorted(S.new):
        h.update(repr((int(i), [tuple(int(v) for v in r) for r in S.new[i]])).encode())
    h.update(np.ascontiguousarray(S.la.d).tobytes())
    if "net_slack" in cur[0]:
        h.update(np.ascontiguousarray(np.asarray(cur[0]["net_slack"], np.float64)).tobytes())
    h.update(repr({k: float(cur[2][k]) for k in ("wns", "tns", "of", "pw", "score")}).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ overflow stage: keys / pin counts / guard (exact, vectorised)
def keys_npins(S, PRE, _k):
    """route-net keys (escape3d._key of S.nets.names) and npk.get(key, 0) per route net, cached for the run"""
    c = PRE.get("mid5_keys")
    if c is None or c[0] is not S.nets.names:
        h = S.__dict__.get("_h_nm")
        keys = h[4] if (h is not None and h[1] == S.nets.names) else [_k(nm) for nm in S.nets.names]
        npk = PRE["ovf_npins_key"]
        npins = np.fromiter((npk.get(k, 0) for k in keys), np.int64, len(keys))
        c = PRE["mid5_keys"] = (S.nets.names, keys, npins)
    return c[1], c[2]


def guarded_keys(S, PRE, OL, _k, slack_r):
    """== frozenset(_k(nm) for i, nm in enumerate(S.nets.names) if slack_r[i] < SLACK_GUARD or npk.get(_k(nm), 0) > MAX_PINS
    or nseg[i] > BIG_NET)"""
    keys, npins = keys_npins(S, PRE, _k)
    nseg = np.diff(S.segs.off)
    cond = (slack_r < OL.SLACK_GUARD) | (npins > OL.MAX_PINS) | (nseg > OL.RB.BIG_NET)
    return frozenset([keys[i] for i in np.flatnonzero(cond).tolist()])


# ------------------------------------------------------------------ overflow stage: escape pass in C++
ESC = {"calls": 0, "fallback": 0, "check": 0, "check_bad": 0, "prep_s": 0.0, "kern_s": 0.0, "post_s": 0.0, "hot_s": 0.0, "segs_s": 0.0, "pins_s": 0.0}
_I, _D = ctypes.c_int64, ctypes.c_double


def _bind_esc(L):
    if getattr(L, "_esc_set", False):
        return
    L.mid5_escape_pass.restype = _V
    L.mid5_escape_pass.argtypes = [_V, _V, _V] + [_I] * 9 + [_V, _V] + [_I, _V, _V, _V, _V, _V, _V, _V, _V] + [_I, _D, _D, _V]
    L.mid5_esc_n.restype = _I; L.mid5_esc_n.argtypes = [_V, _V]
    L.mid5_esc_get.argtypes = [_V, _V, _V, _V]; L.mid5_esc_free.argtypes = [_V]
    L._esc_set = True


def _scan_nets(S, keys, Y):
    """nets with a via row (current route: base rows of unchanged nets + S.new) whose x1 * Y + y1 is in keys, sorted (exact)"""
    arr = np.ascontiguousarray(S.base_arr, np.int64)
    nn = len(S.nets.names)
    chg = np.zeros(nn, np.uint8)
    if S.new:
        chg[np.fromiter(S.new.keys(), np.int64)] = 1
    nk = int(keys.max()) + 1 if len(keys) else 1
    keym = np.zeros(nk, np.uint8)
    if len(keys):
        keym[keys] = 1
    hit = np.zeros(nn, np.uint8)
    L = lib()
    if not getattr(L, "_hn_set", False):
        L.mid5_hot_nets.argtypes = [_V, ctypes.c_int64, _V, _V, ctypes.c_int64, ctypes.c_int64, _V, ctypes.c_int]
        L._hn_set = True
    L.mid5_hot_nets(arr.ctypes.data, len(arr), chg.ctypes.data, keym.ctypes.data, nk, int(Y), hit.ctypes.data,
                    int(os.environ.get("MID5_THREADS", "8")))
    for i, segs in S.new.items():
        for sg in segs:
            if sg[2] != sg[5]:
                k = sg[0] * Y + sg[1]
                if 0 <= k < nk and keym[k]:
                    hit[i] = 1
                    break
    return np.flatnonzero(hit).tolist()


def escape_pass(S, pins, n_hot, guarded, OL, E, PRE, _k, W=12, stats=None, astar="win"):
    """== tools/ovf3/ovf_lib.escape_pass (one escape3d.escape pass on S), the candidate loop in C++ (mid5_escape_pass).
    Falls back to the original when the inputs are not the plain arrays the kernel reads."""
    la = S.la
    E.emit = OL.emit_tuples
    E.dijkstra = OL.astar_win if astar == "win" else OL.RA.astar_c
    d, c = la.d, la.c
    ok = (astar == "win" and OL._CW is not None and isinstance(pins, dict) and d.dtype == np.float64 and c.dtype == np.float64
          and d.strides == c.strides and E.marginal is OL.RA.marginal_fast and E.apply is OL.RA.apply_fast)
    if not ok:
        ESC["fallback"] += 1
        return OL.escape_pass(S, pins, n_hot, guarded, W=W, stats=stats, astar=astar)
    t0 = time.perf_counter()
    ESC["calls"] += 1
    blk = (la.c <= OL.CAP_EPS) & (la.d > 0)
    blk[0] = False
    zz, xx, yy = np.nonzero(blk)
    if len(zz) == 0:
        return {}
    costs = la.u[zz] * np.exp(np.minimum(0.75 * la.d[zz, xx, yy], 700.0))
    order = np.argsort(-costs)[:n_hot]
    hx, hy = xx[order].astype(np.int64), yy[order].astype(np.int64)
    X, Y = int(la.X), int(la.Y)
    keys = np.unique(np.concatenate([hx * Y + hy, (hx + 1) * Y + hy, hx * Y + hy + 1]))
    idx = _scan_nets(S, keys, Y)                                 # == np.unique(cur_arr(S, vias_only)[isin(x1 * Y + y1, keys), 0])
    ta = time.perf_counter(); ESC["hot_s"] += ta - t0
    hot = np.zeros((X + 1) * (Y + 1), np.uint8)                   # escape3d's hot_xy = hot cells and their +x / +y neighbours
    hot[hx * (Y + 1) + hy] = 1; hot[(hx + 1) * (Y + 1) + hy] = 1; hot[hx * (Y + 1) + hy + 1] = 1
    n = len(idx)
    rkeys, _ = keys_npins(S, PRE, _k)
    new = S.new; off = S.segs.off; arr = np.asarray(S.segs.arr, np.int64)
    ia = np.asarray(idx, np.int64)
    cnt = (off[ia + 1] - off[ia]).astype(np.int64) if n else np.zeros(0, np.int64)
    newk = [k for k, i in enumerate(idx) if i in new]
    for k in newk:
        cnt[k] = len(new[idx[k]])
    soff = np.zeros(n + 1, np.int64); soff[1:] = np.cumsum(cnt)
    segs = np.empty((int(soff[-1]), 6), np.int64)
    msk = np.ones(n, bool); msk[newk] = False
    kb = np.flatnonzero(msk & (cnt > 0)); cb = cnt[kb]
    if len(kb):
        ar = np.arange(int(cb.sum())) - np.repeat(np.cumsum(cb) - cb, cb)
        segs[np.repeat(soff[kb], cb) + ar] = arr[np.repeat(off[ia[kb]], cb) + ar]
    for k in newk:
        if cnt[k]:
            segs[soff[k]:soff[k + 1]] = np.asarray(new[idx[k]], np.int64).reshape(-1, 6)
    tb = time.perf_counter(); ESC["segs_s"] += tb - ta
    pidx = np.full(n, -1, np.int64); gfl = np.zeros(n, np.uint8); npin = np.zeros(n, np.int64)
    aps = []; apo = [0]; npn = 0
    for k, i in enumerate(idx):
        key = rkeys[i]
        pl = pins.get(key)
        if pl is None:
            continue
        if key in guarded:
            gfl[k] = 1; pidx[k] = 0
            continue
        pidx[k] = npn; npin[k] = len(pl); npn += len(pl)
        for pin in pl:
            aps.extend(pin); apo.append(len(aps))
    aps_a = np.asarray(aps, np.int64).reshape(-1, 3) if aps else np.zeros((1, 3), np.int64)
    apo_a = np.asarray(apo, np.int64)
    ESC["pins_s"] += time.perf_counter() - tb
    dirs8 = np.ascontiguousarray(np.asarray(la.dirs, np.int8)); dirs32 = np.ascontiguousarray(np.asarray(la.dirs, np.int32))
    u = np.ascontiguousarray(la.u, np.float64)
    st = np.zeros(8, np.int64)
    sd = [s // 8 for s in d.strides]; sc = [s // 8 for s in c.strides]
    d_keep = d.copy()                                            # fallback point (the kernel writes d)
    L = lib(); _bind_esc(L)
    t1 = time.perf_counter(); ESC["prep_s"] += t1 - t0
    h = L.mid5_escape_pass(d.ctypes.data, c.ctypes.data, u.ctypes.data, sd[0], sd[1], sd[2], sc[0], sc[1], sc[2], X, Y, int(la.L),
                           dirs8.ctypes.data, dirs32.ctypes.data, n, soff.ctypes.data, segs.ctypes.data, pidx.ctypes.data,
                           gfl.ctypes.data, npin.ctypes.data, apo_a.ctypes.data, aps_a.ctypes.data, hot.ctypes.data,
                           int(W), 1.0, 0.1, st.ctypes.data)
    t2 = time.perf_counter(); ESC["kern_s"] += t2 - t1
    nr = ctypes.c_int64(0)
    nch = L.mid5_esc_n(h, ctypes.byref(nr))
    if nch < 0:                                                  # a set operation outside the emulation: original code
        L.mid5_esc_free(h); d[...] = d_keep; ESC["fallback"] += 1
        return OL.escape_pass(S, pins, n_hot, guarded, W=W, stats=stats, astar=astar)
    oi = np.zeros(max(nch, 1), np.int64); oo = np.zeros(nch + 1, np.int64); orow = np.zeros((max(nr.value, 1), 6), np.int64)
    L.mid5_esc_get(h, oi.ctypes.data, oo.ctypes.data, orow.ctypes.data)
    L.mid5_esc_free(h)
    undo = {}
    base = S.segs
    rl = orow.tolist()
    for q in range(nch):
        i = idx[int(oi[q])]
        sg = new.get(i)
        undo[i] = sg if sg is not None else base[i]
        new[i] = [tuple(r) for r in rl[int(oo[q]):int(oo[q + 1])]]
        S._dirty.add(int(S.tid[i]))
    OL.ASTAR_CALLS[0] += int(st[7])
    if stats is not None:
        for k_, v_ in zip(("guarded", "skip_pins_at_gcell_0", "skip_pins_at_gcell_many", "skip_tail_is_net", "no_path", "moved", "kept"),
                          st[:7].tolist()):
            if v_:
                stats[k_] = stats.get(k_, 0) + v_
        stats["scan_nets"] = stats.get("scan_nets", 0) + len(idx)
    ESC["post_s"] += time.perf_counter() - t2
    return undo


def escape_pass_checked(S, pins, n_hot, guarded, OL, E, PRE, _k, W=12, stats=None, astar="win"):
    """MID5_CHECK=1: the original pass on a copy of the state first, then the C++ pass; undo, S.new (order and rows), the demand
    bits and the stats must be equal (counted in ESC)"""
    import copy
    d0 = S.la.d.copy(); new0 = dict(S.new); dirty0 = set(S._dirty); ac0 = OL.ASTAR_CALLS[0]
    st_r = {}
    u_r = OL.escape_pass(S, pins, n_hot, guarded, W=W, stats=st_r, astar=astar)
    ref = (list(u_r.items()), list(S.new.items()), S.la.d.copy(), dict(st_r), OL.ASTAR_CALLS[0] - ac0)
    S.la.d[...] = d0; S.new.clear(); S.new.update(new0); S._dirty.clear(); S._dirty.update(dirty0); OL.ASTAR_CALLS[0] = ac0
    st_c = {}
    u_c = escape_pass(S, pins, n_hot, guarded, OL, E, PRE, _k, W=W, stats=st_c, astar=astar)
    ok = (list(u_c.items()) == ref[0] and list(S.new.items()) == ref[1] and np.array_equal(S.la.d.view(np.uint64), ref[2].view(np.uint64))
          and st_c == ref[3] and OL.ASTAR_CALLS[0] - ac0 == ref[4])
    ESC["check"] += 1; ESC["check_bad"] += not ok
    print(f"[mid5 check] escape pass: changed {len(u_c)} (ref {len(ref[0])}) equal={ok} stats {st_c} ref {ref[3]}", flush=True)
    if stats is not None:
        for k_, v_ in st_c.items():
            stats[k_] = stats.get(k_, 0) + v_
    return u_c


# ------------------------------------------------------------------ escape test: blocked share (C++, threads)
def blocked_share(la, official, thresh, nthr, E):
    """the blocked share of chain_j escape_inmem's loop (same cells, same formula; libm exp, long double sums). Returns None
    when the caller must use the numpy loop: share within 1e-6 of the threshold (on/off could differ) or arrays not plain."""
    d, c = la.d, la.c
    if d.dtype != np.float64 or c.dtype != np.float64 or E.CAP_EPS != 1e-3:
        return None
    L = lib()
    if not getattr(L, "_bs_set", False):
        L.mid5_blocked_share.restype = ctypes.c_double
        L.mid5_blocked_share.argtypes = [_V, _V, _V] + [ctypes.c_int64] * 9 + [ctypes.c_int, ctypes.c_int, _V]
        L._bs_set = True
    u = np.ascontiguousarray(la.u, np.float64)
    tb = np.zeros(2)
    sd = [s // 8 for s in d.strides]; sc = [s // 8 for s in c.strides]
    sh = L.mid5_blocked_share(d.ctypes.data, c.ctypes.data, u.ctypes.data, sd[0], sd[1], sd[2], sc[0], sc[1], sc[2],
                              int(la.L), int(la.X), int(la.Y), int(bool(official)), int(nthr), tb.ctypes.data)
    if not np.isfinite(sh) or abs(sh - thresh) < 1e-6:
        return None
    return float(sh)


# ------------------------------------------------------------------ overflow stage: ranked reroute on T threads (same result)
RR = {"calls": 0, "parallel": 0, "batches": 0, "spec_ok": 0, "reeval": 0, "conf_net": 0, "conf_cell": 0, "check": 0, "check_bad": 0}


def reroute_rank(S, pins, chosen, W=14, pin_on=False, pcmap=None, stats=None, astar="win", max_runs=0, hmax=0, timing=None,
                 nthr=8, batch=32):
    """== tools/wave3/ovf2/ovf2_lib.reroute_rank_cpp (same contract, same result), the loop = mid5_rr_rank (mid5_rr.cc)"""
    import gc
    import ovf2_lib as O2
    ge = gc.isenabled(); gc.disable()
    try:
        so = O2._load()
        import ovf_lib as OL
        la = S.la
        if so is None or astar != "win" or OL._CW is None or la.d.dtype != np.float64 or la.c.dtype != np.float64:
            return O2.reroute_rank_cpp(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax, timing)
        return _rr(S, pins, chosen, W, pin_on, pcmap, stats, max_runs, hmax, timing, nthr, batch, O2, OL)
    finally:
        if ge:
            gc.enable()


def _rr(S, pins, chosen, W, pin_on, pcmap, stats, max_runs, hmax, timing, nthr, batch, O2, OL):
    from timing_promote import _key
    la = S.la
    t0 = time.perf_counter()
    if pin_on and pcmap is not None:                              # ovf2_lib._rr_cpp lines verbatim from here to the call
        c_eff = np.array(la.c, dtype=np.float64, copy=True)
        for z in (1, 2):
            if z < la.L:
                c_eff[z] = np.where(c_eff[z] > OL.CAP_EPS, np.maximum(c_eff[z] - OL.PIN_BETA * pcmap, 0.5 * c_eff[z]), c_eff[z])
        cs = c_eff
    else:
        cs = la.c
    if getattr(la, "_c_dirs", None) is None:
        la._c_dirs = np.asarray(la.dirs, np.int32)
        la._c_u = np.ascontiguousarray(la.u, np.float64)
    d, c = la.d, la.c
    strides = np.array([v // 8 for v in d.strides] + [v // 8 for v in c.strides] + [v // 8 for v in cs.strides], np.int64)
    chosen = list(chosen)
    n = len(chosen)
    ridx = np.asarray(chosen, np.int64)
    names = S.nets.names
    arr, off = S.segs.arr, S.segs.off
    parts, cnts = [], np.zeros(n, np.int64)
    for s, i in enumerate(chosen):
        sg = S.new.get(i)
        if sg is None:
            a_ = arr[off[i]:off[i + 1]]
        else:
            a_ = np.asarray(sg, np.int64).reshape(-1, 6) if len(sg) else np.zeros((0, 6), np.int64)
        parts.append(a_); cnts[s] = len(a_)
    segs = np.ascontiguousarray(np.concatenate(parts, 0) if parts else np.zeros((0, 6)), np.int32)
    seg_off = np.zeros(n + 1, np.int64); seg_off[1:] = np.cumsum(cnts)
    pc, ap_parts = [], []
    for i in chosen:
        cnt, aps = O2._pins_of(pins, _key(names[i]))
        pc.append(cnt); ap_parts.append(aps)
    npin = np.array([len(x) for x in pc], np.int64)
    pin_off = np.zeros(n + 1, np.int64); pin_off[1:] = np.cumsum(npin)
    allc = np.concatenate(pc) if pc else np.zeros(0, np.int64)
    ap_off = np.zeros(len(allc) + 1, np.int64); ap_off[1:] = np.cumsum(allc)
    aps = np.ascontiguousarray(np.concatenate(ap_parts, 0) if ap_parts else np.zeros((0, 3)), np.int32)
    if len(aps) == 0:
        aps = np.zeros((1, 3), np.int32)
    st = np.zeros(16, np.int64); info = np.zeros(8, np.int64)
    L = lib()
    if not getattr(L, "_rr_set", False):
        V = _V
        L.mid5_rr_rank.restype = ctypes.c_int
        L.mid5_rr_rank.argtypes = [ctypes.c_int] * 3 + [V] * 6 + [V] + [ctypes.c_int] + [V] * 6 + [ctypes.c_int] * 3 + [V] + [ctypes.c_int] * 2 + [V]
        L.mid5_rr_nseg.restype = ctypes.c_int64
        L.mid5_rr_get.argtypes = [V, V, V]
        L._rr_set = True
    t1 = time.perf_counter()
    nch = L.mid5_rr_rank(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, cs.ctypes.data, _p(la._c_u), _p(la._c_dirs), _p(strides),
                         None, n, _p(ridx), _p(seg_off), _p(segs), _p(pin_off), _p(ap_off), _p(aps), int(W), int(hmax), int(max_runs),
                         _p(st), int(nthr), int(batch), _p(info))
    t2 = time.perf_counter()
    RR["calls"] += 1
    for k, v in zip(("parallel", "batches", "spec_ok", "reeval", "conf_net", "conf_cell"), info[:6].tolist()):
        RR[k] += int(v)
    ns = int(L.mid5_rr_nseg())
    slots = np.zeros(max(nch, 1), np.int32); roff = np.zeros(nch + 1, np.int64); rseg = np.zeros((max(ns, 1), 6), np.int32)
    L.mid5_rr_get(_p(slots), _p(roff), _p(rseg))
    view = OL.CurNets(S)
    undo = {}
    rl = rseg[:ns].tolist()
    for k in range(nch):
        i = chosen[int(slots[k])]
        undo[i] = view.segs(i)
        S.new[i] = [tuple(r) for r in rl[roff[k]:roff[k + 1]]]
        S._dirty.add(int(S.tid[i]))
    keys = ["big_skip", "hmax_skip", "cand_runs", "stale", "pin_end", "spur", "spur_end", "no_path", "reject_invalid",
            "rerouted", "kept", "nets_changed", "astar_calls"]
    if stats is not None:
        for k_, v_ in zip(keys, st.tolist()):
            if v_ or k_ in ("cand_runs", "nets_changed"):
                stats[k_] = int(v_)
    if timing is not None:
        timing.update(prep_s=t1 - t0, loop_s=t2 - t1, out_s=time.perf_counter() - t2)
    return undo


def reroute_rank_checked(S, pins, chosen, *aa, **kw):
    """MID5_CHECK=1: the original C++ loop (ovf2_lib) on a copy of the state first; undo, S.new, demand bits and stats compared"""
    import ovf2_lib as O2
    d0 = S.la.d.copy(); new0 = dict(S.new); dirty0 = set(S._dirty)
    st_r = {}
    kw_r = {k: v for k, v in kw.items() if k not in ("nthr", "batch", "timing", "stats")}
    u_r = O2.reroute_rank_cpp(S, pins, chosen, *aa, stats=st_r, **kw_r)
    ref = (list(u_r.items()), list(S.new.items()), S.la.d.copy(), dict(st_r))
    S.la.d[...] = d0; S.new.clear(); S.new.update(new0); S._dirty.clear(); S._dirty.update(dirty0)
    u_c = reroute_rank(S, pins, chosen, *aa, **kw)
    st_c = kw.get("stats")
    ok = (list(u_c.items()) == ref[0] and list(S.new.items()) == ref[1] and np.array_equal(S.la.d.view(np.uint64), ref[2].view(np.uint64))
          and (st_c is None or {k: st_c[k] for k in ref[3]} == ref[3]))
    RR["check"] += 1; RR["check_bad"] += not ok
    print(f"[mid5 check] reroute: changed {len(u_c)} (ref {len(ref[0])}) equal={ok} info {RR}", flush=True)
    return u_c


# ------------------------------------------------------------------ overflow stage: pretest with the cost map on threads
def pretest_cost(S, OL, nthr=8):
    """== tools/wave3/ovf2/ovf2_lib.pretest_cost (same numbers, same cost array): the elementwise cost map RA._cost(u, d, c)
    is computed layer by layer on threads (numpy releases the GIL; each element is the same expression on the same values);
    the masks and the sums are the original single expressions on the full arrays."""
    from concurrent.futures import ThreadPoolExecutor
    la = S.la
    blk, ovr = OL.ov_masks(la)
    u = la.u
    shp = np.broadcast_shapes(la.d.shape, la.c.shape)
    # the original's array layout (numpy's K order follows the inputs' strides), found on a small block, so that the masked sums
    # below run over memory in the same order (same pairwise summation, same bits)
    sm = OL.RA._cost(u[:, None, None], la.d[:, :2, :3], la.c[:, :2, :3])
    order = sorted(range(3), key=lambda k: -sm.strides[k])
    inv = [order.index(k) for k in range(3)]
    cost = np.empty(tuple(shp[k] for k in order), np.float64).transpose(inv)
    def one(z):
        cost[z] = OL.RA._cost(u[z], la.d[z], la.c[z])
    with ThreadPoolExecutor(max_workers=max(1, int(nthr))) as ex:
        list(ex.map(one, range(la.L)))
    cb, co = float((cost * blk).sum()), float((cost * ovr).sum())
    return {"ov_cells": int(blk.sum() + ovr.sum()), "blocked_cells": int(blk.sum()), "blk_cost": cb, "open_cost": co,
            "bound_score": float(S.w4) * (cb + co), "bound_blk_score": float(S.w4) * cb,
            "blk_share_ov": cb / max(cb + co, 1e-30)}, cost
