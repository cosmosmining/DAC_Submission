"""l5_rs.py: tools/reshape/rs_lib.reroute_rs with the C++ loop of tools/late5/l5_rs.cc
(l5_rs_rank: the same loop, candidates evaluated in parallel blocks with an ordered commit; see make_l5_rs.py) and the
input packing done with np.fromiter. Same inputs, same outputs (undo dict, S.new, S._dirty, stats) as rs_lib.reroute_rs."""
import ctypes, os, sys, time
from itertools import chain
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SO = None
C_VIA_LOAD = 2.0
THREADS = int(os.environ.get("L5_RS_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
KEYS = ["big_skip", "hmax_skip", "cand_runs", "stale", "pin_end", "spur", "spur_end", "no_path", "reject_invalid",
        "rerouted", "kept", "nets_changed", "astar_calls", "of_up", "prune_drops", "prune_skip", "budget_reject", "l5_redo", "l5_cands"]


def _load():
    global _SO
    if _SO is None:
        so = ctypes.CDLL(os.path.join(HERE, "libl5_rs." + ("dylib" if sys.platform == "darwin" else "so")))
        V = ctypes.c_void_p
        so.l5_rs_rank.restype = ctypes.c_int
        so.l5_rs_rank.argtypes = ([ctypes.c_int] * 3 + [V] * 6 + [V] + [ctypes.c_int] + [V] * 6 + [ctypes.c_int] * 3 + [V]
                                  + [ctypes.c_int] + [V] * 11 + [ctypes.c_int])
        so.l5_rs_nseg.restype = ctypes.c_int64
        so.l5_rs_get.argtypes = [V, V, V]
        so.l5_rs_times.argtypes = [V]
        _SO = so
    return _SO


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def reroute_rs(S, pins, chosen, mode, tw_r, cl_r, rd_r, pw_r, W=14, hmax=40, max_runs=0, minlen=1, stats=None, skip=0, sb_r=None,
               ss_r=None, soft=0.0, nthreads=None):
    import gc
    import ovf_lib as OL
    import ovf2_lib as O2
    from timing_promote import _key
    ge = gc.isenabled(); gc.disable()
    try:
        so = _load()
        la = S.la
        if getattr(la, "_c_dirs", None) is None:
            la._c_dirs = np.asarray(la.dirs, np.int32)
            la._c_u = np.ascontiguousarray(la.u, np.float64)
        d, c = la.d, la.c
        assert d.dtype == np.float64 and c.dtype == np.float64 and OL._CW is not None
        cs = c
        strides = np.array([v // 8 for v in d.strides] + [v // 8 for v in c.strides] + [v // 8 for v in cs.strides], np.int64)
        chosen = [int(i) for i in chosen]
        n = len(chosen)
        if n == 0:
            return {}
        ridx = np.asarray(chosen, np.int64)
        names = S.nets.names
        arr, off = S.segs.arr, S.segs.off
        new = S.new
        cnts = np.zeros(n, np.int64)
        lists = []
        for s_, i in enumerate(chosen):                  # == rs_lib: S.new lists, else the base rows
            sg = new.get(i)
            if sg is None:
                lists.append(arr[off[i]:off[i + 1]]); cnts[s_] = off[i + 1] - off[i]
            else:
                lists.append(sg); cnts[s_] = len(sg)
        tot = int(cnts.sum())
        segs = np.empty((max(tot, 1), 6), np.int32)
        pos = 0
        run = []
        for s_, sg in enumerate(lists):                   # consecutive base blocks are copied as arrays, lists flattened
            k = int(cnts[s_])
            if k == 0:
                continue
            if isinstance(sg, np.ndarray):
                segs[pos:pos + k] = sg
            else:
                segs[pos:pos + k] = np.fromiter(chain.from_iterable(sg), np.int64, 6 * k).reshape(k, 6)
            pos += k
        segs = np.ascontiguousarray(segs[:max(tot, 0)] if tot else np.zeros((0, 6), np.int32), np.int32)
        if tot == 0:
            segs = np.zeros((0, 6), np.int32)
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
        f64 = lambda x: np.ascontiguousarray(np.asarray(x, np.float64)[ridx])
        ntw, ncl, nrd, npw = f64(tw_r), f64(cl_r), f64(rd_r), f64(pw_r)
        nsb = f64(sb_r) if sb_r is not None else np.full(n, 1e30)
        nss = f64(ss_r) if ss_r is not None else np.zeros(n)
        Rz = np.ascontiguousarray(la.R, np.float64); Cz = np.ascontiguousarray(la.C, np.float64); Rv = np.ascontiguousarray(la.Rv, np.float64)
        Fp = np.array([float(la.w4), float(la.um), C_VIA_LOAD, float(minlen), float(skip), float(soft)], np.float64)
        gain = np.zeros(4)
        st = np.zeros(19, np.int64)
        segs_c = segs if len(segs) else np.zeros((1, 6), np.int32)
        t1 = time.perf_counter()
        nch = so.l5_rs_rank(la.L, la.X, la.Y, d.ctypes.data, c.ctypes.data, cs.ctypes.data, _p(la._c_u), _p(la._c_dirs), _p(strides),
                            None, n, _p(ridx), _p(seg_off), _p(segs_c), _p(pin_off), _p(ap_off), _p(aps), int(W), int(hmax), int(max_runs),
                            _p(st), int(mode), _p(ntw), _p(ncl), _p(nrd), _p(npw), _p(Rz), _p(Cz), _p(Rv), _p(Fp), _p(gain), _p(nsb), _p(nss),
                            int(THREADS if nthreads is None else nthreads))
        t2 = time.perf_counter()
        ns = int(so.l5_rs_nseg())
        slots = np.zeros(max(nch, 1), np.int32); roff = np.zeros(nch + 1, np.int64); rseg = np.zeros((max(ns, 1), 6), np.int32)
        so.l5_rs_get(_p(slots), _p(roff), _p(rseg))
        view = OL.CurNets(S)
        undo = {}
        rl = rseg[:ns].tolist()
        for k in range(nch):
            i = chosen[int(slots[k])]
            undo[i] = view.segs(i)
            S.new[i] = [tuple(r) for r in rl[roff[k]:roff[k + 1]]]
            S._dirty.add(int(S.tid[i]))
        if stats is not None:
            for k_, v_ in zip(KEYS, st.tolist()):
                stats[k_] = stats.get(k_, 0) + int(v_)
            for k_, v_ in zip(("gain_of", "gain_t", "gain_p", "gain_tot"), gain.tolist()):
                stats[k_] = stats.get(k_, 0.0) + float(v_)
            stats["loop_s"] = stats.get("loop_s", 0.0) + (t2 - t1)
            tt = np.zeros(8); so.l5_rs_times(_p(tt))
            stats["l5_t"] = [round(float(x), 4) for x in tt[:5]]      # setup end, copies end, parallel, serial, loop end
        return undo
    finally:
        if ge:
            gc.enable()
