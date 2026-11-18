"""n_rr.py: tools/mid5/mid5.py reroute_rank with the kernel tools/best5/n_rr.cc (make_n_rr.py): the overflow stage's ranked
reroute in a new order of work. policy 4 (task graph): same result as the serial loop; policies 1-3 (snapshot batches, no redo): the route changes."""
import ctypes, gc, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "dylib" if sys.platform == "darwin" else "so"
_L = [None]; _V = ctypes.c_void_p
RR = []                                                       # one record per call


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def lib():
    if _L[0] is None:
        L = ctypes.CDLL(os.path.join(HERE, "libn_rr." + EXT))
        V = _V
        L.n_rr_rank.restype = ctypes.c_int
        L.n_rr_rank.argtypes = [ctypes.c_int] * 3 + [V] * 6 + [V] + [ctypes.c_int] + [V] * 6 + [ctypes.c_int] * 3 + [V] + [ctypes.c_int] * 2 + [V] + [ctypes.c_int] * 2 + [V]
        L.n_rr_nseg.restype = ctypes.c_int64
        L.n_rr_get.argtypes = [V, V, V]
        _L[0] = L
    return _L[0]


def reroute_rank(S, pins, chosen, W=14, pin_on=False, pcmap=None, stats=None, astar="win", max_runs=0, hmax=0, timing=None, nthr=8, batch=32, policy=4, shift=2):
    import ovf2_lib as O2
    import ovf_lib as OL
    ge = gc.isenabled(); gc.disable()
    try:
        so = O2._load()
        la = S.la
        if so is None or astar != "win" or OL._CW is None or la.d.dtype != np.float64 or la.c.dtype != np.float64:
            return O2.reroute_rank_cpp(S, pins, chosen, W, pin_on, pcmap, stats, astar, max_runs, hmax, timing)
        return _rr(S, pins, chosen, W, pin_on, pcmap, stats, max_runs, hmax, timing, nthr, batch, O2, OL, policy, shift)
    finally:
        if ge:
            gc.enable()


def _rr(S, pins, chosen, W, pin_on, pcmap, stats, max_runs, hmax, timing, nthr, batch, O2, OL, policy, shift):
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
    L = lib(); dinfo = np.zeros(8)
    t1 = time.perf_counter()
    nch = L.n_rr_rank(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, cs.ctypes.data, _p(la._c_u), _p(la._c_dirs), _p(strides),
                         None, n, _p(ridx), _p(seg_off), _p(segs), _p(pin_off), _p(ap_off), _p(aps), int(W), int(hmax), int(max_runs),
                         _p(st), int(nthr), int(batch), _p(info), int(policy), int(shift), _p(dinfo))
    t2 = time.perf_counter()
    RR.append({"policy": int(policy), "batch": int(batch), "info": info[:6].tolist(), "t": [round(float(v), 4) for v in dinfo[:6]], "changed": int(nch)})
    ns = int(L.n_rr_nseg())
    slots = np.zeros(max(nch, 1), np.int32); roff = np.zeros(nch + 1, np.int64); rseg = np.zeros((max(ns, 1), 6), np.int32)
    L.n_rr_get(_p(slots), _p(roff), _p(rseg))
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
