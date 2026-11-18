#!/usr/bin/env python3
"""dgr_stage.py: the descent of an internal note as one stage of tools/best7/chain_q.py (switch --dgr K), an internal note. On the chain's state: (1) timer alone on the GPU at the exact kernel's RC -> per-net sensitivities;
(2) the nets that can matter (score: timer sensitivity x value + overflow cost along the net + power), huge nets held by rule; (3) problem on threads (tools/best7/dgb.cc), device tables; (4) K steps in one kernel call (tools/best7/dg_kern.cu); (5) argmax, rows of the moved nets, one exact test of the whole change (the chain's gate: incremental re-time + overflow update).
A rejected change is split by the linear gain of the objective, as far as the deadline allows. No torch."""
import ctypes, fcntl, os, sys, threading, time
from itertools import chain as _chain
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "best7")); sys.path.insert(0, HERE)
import gnn_f as GN
import dg as DG
import dgb as DB

ST = {"X": None, "th": None, "err": None, "lock": None, "Kt": None}


def begin(gnn=""):
    """on a thread, from the process start: the device (env DGR_LOCK: T4 under a lock, nothing before the stage)"""
    if os.environ.get("DGR_LOCK"):
        return
    def work():
        try:
            ST["X"] = DG.Dg(gpu=os.environ.get("DGR_CPU") != "1", f64=False)
            if gnn:                 # weights on the device, on this thread
                GN.net_of(gnn, os.environ.get("DGR_CPU") != "1", int(os.environ.get("MID5_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))))
        except BaseException as e:      # the stage reports it
            ST["err"] = e
    ST["th"] = threading.Thread(target=work, daemon=True); ST["th"].start()


def tables(cd, w):
    """on a thread, when the design's timing data is there (the first pass runs meanwhile): the timer's tables on the device"""
    if os.environ.get("DGR_LOCK"):
        return
    th0 = ST["th"]
    def work():
        try:
            if th0 is not None:
                th0.join()
            if ST["err"] is None:
                ST["X"].timer_tables(cd, *w)
        except BaseException as e:
            ST["err"] = e
    ST["th"] = threading.Thread(target=work, daemon=True); ST["th"].start()


def names(S, nm):
    """on a thread after the route read: per route net: index into S's pin tables, access point range; the design's arrays of the RC topology"""
    def work():
        try:
            from timing_promote import _key
            tk = S.tidx_key
            J = np.fromiter((tk.get(_key(n), -1) for n in nm), np.int64, len(nm))
            ok = (J >= 0)
            ok[ok] = S.pin_end[J[ok]] > S.pin_start[J[ok]]
            pa0 = np.zeros(len(nm), np.int64); pa1 = np.zeros(len(nm), np.int64)
            Jk = J[ok]
            pa0[ok] = S.ap_start[S.pin_start[Jk]]; pa1[ok] = S.ap_start[S.pin_end[Jk] - 1] + np.asarray(S.D.ap_cnt)[S.pin_end[Jk] - 1]
            ST["names"] = (J, pa0, pa1, len(nm))
        except BaseException as e:
            ST["names_err"] = e
    ST["th_names"] = threading.Thread(target=work, daemon=True); ST["th_names"].start()


def skip_rule(a, S, rt, o, W, now=None, X=None, est=None):
    """True: no first pass. False: a net the descent holds (more than --dgr_hold rows) lies on a violating path: the first pass's layer DP can move it, the descent cannot; or (mode 3) the first pass and one round of the descent fit the deadline"""
    from timing_promote import _key
    cnt = np.diff(np.asarray(rt.offsets, np.int64))
    huge = np.flatnonzero(cnt > a.dgr_hold)
    sl = np.asarray(o["net_slack"], np.float64)
    tids = [S.SD.net_index.get(_key(rt.net_names[int(i)]), -1) for i in huge]
    tids = np.array([t for t in tids if t >= 0], np.int64)
    worst = float(sl[tids].min()) if len(tids) else 0.0
    W["dgr_rule"] = {"huge_nets": int(len(huge)), "worst_slack": worst, "wns": float(o["wns"]), "tns": float(o["tns"])}
    if len(tids) and worst < 0.0:
        return False
    if a.dgr_skip_fp == 3 and now is not None:
        need = sum(float(est.get(k, 0.0)) for k in ("dgr_fp", "dgr_build", "dgr_desc", "dgr_test", "dgr_write"))
        W["dgr_rule"].update(now=round(now, 3), need=round(need, 3), X=X)
        if now + need <= X:
            return False
    return True


def _names(S, nm):
    th = ST.get("th_names")
    if th is None:
        names(S, nm); th = ST["th_names"]
    th.join()
    if ST.get("names_err") is not None:
        raise ST["names_err"]
    return ST["names"]


def topo_tables(cd):
    """the arrays dgb.topo needs (csta.Kernel.bind_rc_design's)"""
    if ST.get("rct") is None:
        import csta as CS
        ex = getattr(cd, "extra", None)
        if ex is None or "ap_idx" not in ex:
            ex = CS.Kernel(lib=os.path.join(ROOT, "tools", "gnnfix", "dtimer", "librc_topo.so")).design_arrays(cd)
        c8 = lambda a: np.ascontiguousarray(np.asarray(a, np.int64))
        cf = lambda a: np.ascontiguousarray(np.asarray(a, np.float64).reshape(-1))
        ST["rct"] = dict(ap_off=c8(ex["ap_off"]), ap_idx=c8(ex["ap_idx"]), ap=c8(cd.ap).reshape(-1), ap_pid=c8(cd.ap_pid), net_drv=c8(cd.net_drv), layer_rc=cf(cd.layer_rc), via_r=cf(cd.via_r), um=float(CS._UMGC))
    return ST["rct"]


def _device(cd, w):
    if ST["th"] is not None:
        ST["th"].join(); ST["th"] = None
    if ST["err"] is not None:
        raise ST["err"]
    if os.environ.get("DGR_LOCK") and ST["lock"] is None:
        ST["lock"] = open(os.environ["DGR_LOCK"], "w"); fcntl.flock(ST["lock"], fcntl.LOCK_EX)
    if ST["X"] is None:
        ST["X"] = DG.Dg(gpu=os.environ.get("DGR_CPU") != "1", f64=False)
    if not ST["X"].timer_on:
        ST["X"].timer_tables(cd, *w)
    return ST["X"]


def end():
    if ST["lock"] is not None:
        ST["lock"].close(); ST["lock"] = None


def cur_rows(S):
    """current rows of every route net (S.new over the base): flat [m,6] int64, off [n+1]"""
    arr, off = np.ascontiguousarray(S.segs.arr, np.int64), np.ascontiguousarray(S.segs.off, np.int64)
    if not S.new:
        return arr, off
    n = len(off) - 1
    ids = np.fromiter(S.new.keys(), np.int64, len(S.new))
    lists = [S.new[i] for i in ids.tolist()]
    ln = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    m = int(ln.sum())
    fl = np.fromiter(_chain.from_iterable(_chain.from_iterable(lists)), np.int64, 6 * m).reshape(m, 6) if m else np.zeros((0, 6), np.int64)
    cnt = np.diff(off); cnt[ids] = ln
    no = np.zeros(n + 1, np.int64); no[1:] = np.cumsum(cnt)
    out = np.empty((int(no[-1]), 6), np.int64)
    keep = np.ones(n, bool); keep[ids] = False
    kn = np.flatnonzero(keep & (cnt > 0)); c = cnt[kn]
    if len(kn):
        s0 = np.repeat(off[kn], c); d0 = np.repeat(no[kn], c); ar = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        out[d0 + ar] = arr[s0 + ar]
    fo = np.zeros(len(ids) + 1, np.int64); fo[1:] = np.cumsum(ln)
    ids_ok = ln > 0
    if ids_ok.any():
        c2 = ln[ids_ok]; s0 = np.repeat(fo[:-1][ids_ok], c2); d0 = np.repeat(no[ids[ids_ok]], c2); ar = np.arange(int(c2.sum())) - np.repeat(np.cumsum(c2) - c2, c2)
        out[d0 + ar] = fl[s0 + ar]
    return out, no


def gather(rows, off, idx):
    """rows of the nets idx, flat, off2"""
    idx = np.asarray(idx, np.int64)
    c = off[idx + 1] - off[idx]
    o2 = np.zeros(len(idx) + 1, np.int64); o2[1:] = np.cumsum(c)
    s0 = np.repeat(off[idx], c); ar = np.arange(int(o2[-1])) - np.repeat(o2[:-1], c)
    return rows[s0 + ar], o2


_NN = None


def normalise(S, rows6, off2, idx, pa0_all, pa1_all, T):
    """normalize_net (tools/best5/n_norm.cc, GCells) of the nets idx with their rows given: flat rows, off"""
    global _NN
    if _NN is None:
        sys.path.append(os.path.join(ROOT, "tools", "best5"))
        import n_norm as NN
        _NN = NN
    if not hasattr(S, "_f3x_ap"):
        S._f3x_ap = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
        S._f3x_pr = {}
    nn = len(idx)
    pa0 = np.ascontiguousarray(pa0_all[idx], np.int64); pa1 = np.ascontiguousarray(pa1_all[idx], np.int64)
    start = np.ascontiguousarray(off2[:-1], np.int64); cnt = np.diff(off2).astype(np.int64)
    segs = np.ascontiguousarray(rows6, np.int32)
    L = _NN._L
    tot = L.n_nw_run(nn, start.ctypes.data, cnt.ctypes.data, segs.ctypes.data, pa0.ctypes.data, pa1.ctypes.data, S._f3x_ap.ctypes.data, 1, T)
    blk = np.empty((tot, 7), np.int64); oo = np.zeros(nn + 1, np.int64)
    ids = np.ascontiguousarray(np.arange(nn), np.int64)
    L.n_nw_fetch(ids.ctypes.data, blk.ctypes.data, oo.ctypes.data, T)
    return blk[:, 1:], oo


class Ctx:
    pass


def sensitivities(S, cur, a, cd, W):
    """(1) timer alone on the device at the exact kernel's RC -> per-net sensitivities, the design's tables; returns the context"""
    t = time.perf_counter()
    def lap(k, t0):
        n = time.perf_counter(); W[k] = W.get(k, 0.0) + n - t0; return n
    C = Ctx()
    C.T_ = int(os.environ.get("MID5_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
    o_, rc_, r_ = cur
    X = _device(cd, (S.w1, S.w2, S.nend))
    t = lap("device_wait_s", t)
    C.X = X
    la = S.la; L = int(la.L); dirs = list(la.dirs)
    H = [z for z in range(1, L) if dirs[z] == 0]; V = [z for z in range(1, L) if dirs[z] == 1]; K = max(len(H), len(V))
    cand2 = np.full((2, K), -1, np.int32); cand2[0, :len(H)] = H; cand2[1, :len(V)] = V
    z2k = np.full((2, L), -1, np.int32)
    for d in (0, 1):
        for k in range(K):
            if cand2[d, k] >= 0:
                z2k[d, cand2[d, k]] = k
    C.H, C.V, C.cand2, C.z2k, C.L, C.dirs = H, V, cand2, z2k, L, dirs
    rcb = {k: np.asarray(rc_[k]) for k in ("c1", "c2", "rpi", "ctot", "elm", "par")}
    C.rcb = rcb
    X.set_base(rcb)
    C.s0 = X.sens(a.dgr_tau)
    t = lap("sens_s", t)
    gc1, gc2, grpi, gctot = (X.get(n, np.float64, (-1, 2)) for n in ("gc1", "gc2", "grpi", "gctot"))
    gelm = X.get("gelm", np.float64, (-1, 2))
    pin_net = np.asarray(cd.pin_net, np.int64); nn = len(cd.net_names); pn = np.where(pin_net < 0, pin_net + nn, pin_net)
    c1, c2, rpi, ctot = (rcb[k].reshape(-1, 2) for k in ("c1", "c2", "rpi", "ctot")); elm = rcb["elm"].reshape(-1, 2)
    s_t = (np.abs(gc1) * c1 + np.abs(gc2) * c2 + np.abs(gctot) * ctot + np.abs(grpi) * rpi).sum(1)
    np.add.at(s_t, pn, (np.abs(gelm) * np.where(np.isfinite(elm), elm, 0.0)).sum(1))
    t = lap("sens_read_s", t)
    C.rows, C.off = cur_rows(S)
    tid = np.asarray(S.tid, np.int64); C.tid = tid
    C.pot_o = S.w4 * DB.ovf_pot(C.rows, C.off, la.d, la.c, np.asarray(la.u, np.float64), C.T_)
    C.kpn = np.asarray(S.kpn, np.float64) if S.kpn is not None else np.zeros(nn)
    ct_ = ctot.max(1)
    tv = tid >= 0; C.tv = tv
    C.pot_t = np.where(tv, s_t[np.maximum(tid, 0)], 0.0); C.pot_p = np.where(tv, S.w3 * C.kpn[np.maximum(tid, 0)] * ct_[np.maximum(tid, 0)], 0.0)
    C.nrows = np.diff(C.off)
    t = lap("select_s", t)
    return C


def prepare(C, S, a, cd, FT, HD, W, nets_n):
    """(2) + (3): the nets, the problem on the device"""
    t = time.perf_counter()
    def lap(k, t0):
        n = time.perf_counter(); W[k] = W.get(k, 0.0) + n - t0; return n
    X, T_, la, tid = C.X, C.T_, S.la, C.tid
    L, cand2, z2k = C.L, C.cand2, C.z2k
    Xg, Yg = int(la.X), int(la.Y)
    hold = C.nrows > a.dgr_hold
    score = C.pot_t * a.dgr_wt + C.pot_o * a.dgr_wo + C.pot_p * a.dgr_wp
    elig = np.flatnonzero(C.tv & (C.nrows > 0) & ~hold)
    o = elig[np.argsort(-score[elig], kind="stable")]
    if nets_n > 0:                    # chosen on the small cases: 8,000 of about 100,000 free nets; a design with over 300,000 nets keeps that share
        o = o[:(nets_n if len(elig) <= 300000 else max(nets_n, int(len(elig) * 0.07)))]
    idx = np.sort(o)
    t = lap("select_s", t)
    C.Jn, C.pa0, C.pa1, _n = _names(S, S.nets.names)
    t = lap("names_s", t)
    rw, ow = gather(C.rows, C.off, idx)
    nr, no = normalise(S, rw, ow, idx, C.pa0, C.pa1, T_)
    t = lap("norm_s", t)
    st_ = getattr(S, "_pf_static", None)
    if st_ is None:
        pin_start = np.ascontiguousarray(S.pin_start, np.int64); pin_end = np.ascontiguousarray(S.pin_end, np.int64)
        ap_start = np.ascontiguousarray(S.ap_start, np.int64); ap_cnt = np.ascontiguousarray(S.D.ap_cnt, np.int64); ap_ = np.ascontiguousarray(np.asarray(S.D.ap, np.int32).reshape(-1, 3))
    else:
        pin_start, pin_end, ap_start, ap_cnt, ap_ = st_["pin_start"], st_["pin_end"], st_["ap_start"], st_["ap_cnt"], st_["ap"]
    net_j = C.Jn[idx]
    t = lap("tables_s", t)
    N = DB.nets(nr, no, net_j, pin_start, pin_end, ap_start, ap_cnt, ap_, C.dirs, C.H, C.V, L, T_)
    t = lap("nets_s", t)
    kept = N["kept"]; run_ = N["run"]; pts = N["pts"]
    C.N, C.idx, C.nr, C.no, C.kept, C.run_ = N, idx, nr, no, kept, run_
    C.elig_n = int(len(elig)); C.held = int((C.tv & hold).sum())
    if len(run_) == 0:
        return False
    i_k = idx[kept]; C.i_k = i_k
    rk_, ok_ = gather(C.rows, C.off, i_k)
    r7 = np.empty((len(rk_), 7), np.int64); r7[:, 0] = np.repeat(i_k, np.diff(ok_)); r7[:, 1:] = rk_
    if not HD._add(la, la.d, r7, -1.0):
        raise RuntimeError("dgr: hd_add")
    t = lap("cells_a_s", t)
    A = DB.cells(N, cand2, la.d, la.c, np.asarray(la.u, np.float64), Xg, Yg, L, T_)
    if a.dgr_gnn:
        GN.capture(A, la.d, la.c, Xg, Yg)
    t = lap("cells_b_s", t)
    if not HD._add(la, la.d, r7, 1.0):
        raise RuntimeError("dgr: hd_add back")
    t = lap("cells_a_s", t)
    C.A = A
    kn = run_[:, 0]; tk_ = tid[i_k][kn]; C.kn = kn
    um = float(FT.GRID) / 2000.0
    lum = (run_[:, 4] - run_[:, 3]).astype(np.float32) * np.float32(um)
    pwk = np.where(tk_ >= 0, C.kpn[np.maximum(tk_, 0)], 0.0).astype(np.float32) * lum
    Cz = np.asarray(a._dgr_Cz, np.float32)
    t = lap("pw_s", t)
    A["prp"], A["prr"] = DB.pr_csr(run_, len(pts))
    X.problem_arrays(run_, pts, A, cand2, z2k, Cz, pwk, S.w3, S.w4, L)
    t = lap("up_problem_s", t)
    keep_m = np.zeros(len(idx), bool); keep_m[kept] = True
    rk2 = np.repeat(keep_m, np.diff(no))
    tnm = np.repeat(tid[idx], np.diff(no))
    seg = np.empty((int(rk2.sum()), 7), np.int64); seg[:, 0] = tnm[rk2]; seg[:, 1:] = nr[rk2]
    rct = topo_tables(cd)
    t = lap("topo_a_s", t)
    tn = DB.topo(seg, rct, T_)
    t = lap("topo_s", t)
    keepm = (seg[:, 0] >= 0) & ((seg[:, 3] != seg[:, 6]) | (seg[:, 1] != seg[:, 4]) | (seg[:, 2] != seg[:, 5]))
    ks = np.nonzero(keepm)[0]; ks = ks[np.argsort(seg[ks, 0], kind="stable")]
    assert len(ks) == tn["ns"], (len(ks), tn["ns"])
    wr_ = seg[ks[tn["wire"]]][:, :6]
    zcur = cand2[run_[:, 1], run_[:, 5]]
    sr, hit = DB.wire_map(wr_, tk_, run_, zcur, T_)
    t = lap("map_s", t)
    if ST.get("crc") is None:
        lrc = np.ascontiguousarray(np.asarray(cd.layer_rc, np.float64).reshape(-1)); 
        ST["crc"] = dict(layer_rc=lrc, via_r=np.ascontiguousarray(np.asarray(cd.via_r, np.float64)), pin_cap=np.ascontiguousarray(np.asarray(cd.pin_cap, np.float64).reshape(-1)), pin_is_drv=np.ascontiguousarray(np.asarray(cd.pin_is_drv).astype(np.uint8)))
    crc = ST["crc"]
    TB = DB.rc_tables(tn, sr, len(run_), crc)
    lr2 = crc["layer_rc"].reshape(-1, 2)
    Lrc = (np.where(cand2 >= 0, lr2[np.maximum(cand2, 0), 0], 0.0), np.where(cand2 >= 0, lr2[np.maximum(cand2, 0), 1], 0.0))
    t = lap("rc_a_s", t)
    X.rc_arrays(TB, tn["nk"], Lrc)
    t = lap("rc_s", t)
    C.nk, C.hit, C.nw = tn["nk"], hit, len(sr)
    return True


def round_rows(C, S, W):
    """argmax, rows of the moved nets; a net whose rows are not rebuilt by the model is left out (dtla0's exact filter, here on the moved nets only). -> nets (kept index), route index, rows, off; undoing is the caller's"""
    t = time.perf_counter()
    def lap(k, t0):
        n = time.perf_counter(); W[k] = W.get(k, 0.0) + n - t0; return n
    X, N, T_, cand2, run_, kn = C.X, C.N, C.T_, C.cand2, C.run_, C.kn
    ch = X.choice(); C.ch = ch
    cur_c = run_[:, 5]; C.cur_c = cur_c
    mv = ch != cur_c; C.mv = mv
    moved_k = np.unique(kn[mv])
    if len(moved_k) == 0:
        return moved_k, moved_k, None, None, 0
    t = lap("rows_a_s", t)
    r_cur, o_cur = DB.rows_of(N, cur_c, moved_k, cand2, T_)
    t = lap("rows_b_s", t)
    i_m = C.i_k[moved_k]
    n_cur, nc_ = normalise(S, r_cur, o_cur, i_m, C.pa0, C.pa1, T_)
    t = lap("rows_c_s", t)
    r_n0, o_n0 = gather(C.nr, C.no, C.kept[moved_k])
    okm = DB.same_rows(n_cur, nc_, r_n0, o_n0, T_)
    t = lap("rows_d_s", t)
    nbad = int((~okm).sum())
    moved_k = moved_k[okm]; i_m = i_m[okm]
    r_new, o_new = DB.rows_of(N, ch, moved_k, cand2, T_)
    t = lap("rows_b_s", t)
    return moved_k, i_m, r_new, o_new, nbad


def run(S, cur, a, cd, gate, FT, HD, DLnow=None, deadline=None, write_est=0.0, test_est=0.0):
    """-> (kept, cur', rec). deadline: seconds on DLnow()'s clock at which the stage must be over (test and write included)"""
    import gc
    ge_ = gc.isenabled(); gc.disable()      # tens of thousands of tuples: no generation scan in the stage
    try:
        return run_(S, cur, a, cd, gate, FT, HD, DLnow, deadline, write_est, test_est)
    finally:
        if ge_:
            gc.enable()


def run_(S, cur, a, cd, gate, FT, HD, DLnow, deadline, write_est, test_est):
    t00 = time.perf_counter(); W = {}
    def lap(k, t0):
        n = time.perf_counter(); W[k] = W.get(k, 0.0) + n - t0; return n
    C = sensitivities(S, cur, a, cd, W)
    X, tau = C.X, a.dgr_tau
    r_ = cur[2]
    if not prepare(C, S, a, cd, FT, HD, W, a.dgr_nets):
        return False, cur, {"stage": "dgr", "skipped": "no runs", "walls": W}
    t = time.perf_counter()
    X.start(a.dgr_init)
    tl = 0.0
    if deadline is not None:
        tl = deadline - DLnow() - write_est - test_est
        if tl <= 0.05:
            return False, cur, {"stage": "dgr", "skipped": "no time", "walls": W, "time_left": tl}
    prof = bool(os.environ.get("DGR_PROF"))
    gpu_ = os.environ.get("DGR_CPU") != "1"
    net_on = bool(a.dgr_gnn) and ST.get("gnn_used", 0) < a.dgr_gnn_rounds
    def plain_run(reuse_, tl_):
        return X.run(a.dgr, lr=a.dgr_lr, t0=a.dgr_t0, t1=a.dgr_t1, tau=tau, stop=a.dgr_stop, patience=a.dgr_pat, dl=a.dgr_dl, dat=a.dgr_dat, reuse=reuse_, time_limit=tl_, lrK=a.dgr_lrk, lrC=a.dgr_lrc, prof=prof)
    gnn_rec = None
    if net_on:
        ST["gnn_used"] = ST.get("gnn_used", 0) + 1
        um_ = float(FT.GRID) / 2000.0
        gnn_rec = GN.start(C, a, X, um_, W, a.dgr_gnn, gpu_, clamp=a.dgr_gnn_k > 0)
        if deadline is not None:
            tl = deadline - DLnow() - write_est - test_est
        t = time.perf_counter()
        if a.dgr_gnn_k > 0 and (deadline is None or tl > 0.05):
            res = X.run(a.dgr_gnn_k, lr=a.dgr_gnn_lr, t0=1.0, t1=a.dgr_gnn_t1, tau=tau, dl=a.dgr_dl, dat=a.dgr_dat, reuse=False, time_limit=tl, lrK=0.0, lrC=0.0, prof=prof)
        else:
            res = {"steps": 0, "vals": np.zeros((1, 8)), "prof": {}}
    else:
        res = plain_run(True, tl)
    t = lap("descent_s", t)
    A, N = C.A, C.N
    rec = {"stage": "dgr", "K": a.dgr, "steps": res["steps"], "nets_eligible": C.elig_n, "nets_set": int(len(C.idx)), "nets_free": int(len(C.kept)), "strict_held": N["strict_held"], "no_tree": N["not_connected"], "held_rows": C.held,
           "runs": int(len(C.run_)), "ent": int(len(A["ec"])), "via_ent": int(len(A["cvg"])), "cells": int(len(A["base"])), "nodes": C.nk, "wires_mapped": C.hit, "wires": C.nw,
           "F": [float(res["vals"][0][4]), float(res["vals"][-1][4])], "F_parts_end": res["vals"][-1][:4].tolist(), "F_timer_start": C.s0, "mem_mb": X.memory_mb(), "time_limit": tl}
    if prof:
        rec["prof"] = {k: v / max(res["steps"], 1) for k, v in res["prof"].items()}
    moved_k, i_m, r_new, o_new, nbad = round_rows(C, S, W)
    rec["runs_moved"] = int(C.mv.sum()); rec["nets_moved"] = int(len(moved_k)); rec["moved_not_rebuilt"] = nbad
    if len(moved_k) == 0:
        rec["walls"] = W; rec["s"] = time.perf_counter() - t00
        return False, cur, rec
    t = time.perf_counter()
    undo = apply(S, i_m, r_new, o_new, C.rows, C.off, HD, W, C)
    t = time.perf_counter()
    keep, c_ = gate(undo, r_, "dgr")
    t = lap("test_s", t)
    tries = [{"nets": int(len(i_m)), "kept": bool(keep), "est": float(c_[2]["score"])}]
    rec["est_in"] = float(r_["score"])
    if not keep and net_on and (deadline is None or DLnow() + W["test_s"] + write_est + 0.1 < deadline):     # the network's change rejected: the plain arm from the same state
        t = time.perf_counter()
        X.start(a.dgr_init)
        tl2 = 0.0 if deadline is None else deadline - DLnow() - write_est - W["test_s"]
        res = plain_run(False, tl2)
        t = lap("fallback_descent_s", t)
        moved_k, i_m, r_new, o_new, nbad = round_rows(C, S, W)
        tries.append({"fallback": "plain", "steps": res["steps"], "nets": int(len(moved_k))})
        if len(moved_k):
            undo = apply(S, i_m, r_new, o_new, C.rows, C.off, HD, W, C)
            t = time.perf_counter()
            keep, c_ = gate(undo, r_, "dgr")
            t = lap("test_s", t)
            tries.append({"nets": int(len(i_m)), "kept": bool(keep), "est": float(c_[2]["score"])})
        rec["runs_moved"] = int(C.mv.sum()); rec["nets_moved"] = int(len(moved_k))
    if not keep and a.dgr_split > 0:
        if X.size.get("g_out") != X.size["x"]:
            X.zeros("g_out", X.size["x"])
        # split: nets by linear gain of the objective at the one-hot start (one more pass on the device), best half, as far as the time allows
        xx = np.zeros((X.I["R"], X.I["K"]), X.RL); xx[np.arange(X.I["R"]), C.cur_c] = 100.0; xx = xx * X.mask; X.put("x", xx)
        ge = X.eval(T=1.0, mode=1, tau=tau, grad=True)["g"].astype(np.float64)
        rr = np.arange(X.I["R"])
        lin = np.where(C.mv, ge[rr, C.ch] - ge[rr, C.cur_c], 0.0)
        ln = np.zeros(len(C.kept)); np.add.at(ln, C.kn, lin)
        order = moved_k[np.argsort(ln[moved_k], kind="stable")]
        cnt = len(order) // 2
        for sp in range(a.dgr_split):
            if cnt < 1:
                break
            if deadline is not None and DLnow() + W["test_s"] + write_est > deadline:
                tries.append({"stop": "deadline"}); break
            sub = np.sort(order[:cnt])
            r_new, o_new = DB.rows_of(C.N, C.ch, sub, C.cand2, C.T_)
            undo = apply(S, C.i_k[sub], r_new, o_new, C.rows, C.off, HD, W, C)
            keep, c_ = gate(undo, r_, "dgr")
            tries.append({"nets": int(len(sub)), "kept": bool(keep), "est": float(c_[2]["score"])})
            if keep:
                break
            cnt //= 2
        t = lap("split_s", t)
    rec["tries"] = tries; rec["kept"] = bool(keep)
    if gnn_rec is not None:
        rec["gnn"] = gnn_rec
    if keep:
        rec["est"] = float(c_[2]["score"]); rec["gain"] = rec["est_in"] - rec["est"]
    rec["walls"] = W; rec["s"] = time.perf_counter() - t00
    return bool(keep), (c_ if keep else cur), rec


def sweep(S, cur, a, cd, retime, K_, FT, HD, out):
    """development only (env DGR_SWEEP = a json list of settings): from one chain state, each setting: descent, rounding, one exact re-time of the whole change, then undone. Keys: N, K, lrk, lr, dl, dat, tau, t1, init. Written to out as it goes."""
    import json
    W = {}
    C = sensitivities(S, cur, a, cd, W)
    X = C.X; r_ = cur[2]
    rec = {"design": S.design, "suite": S.suite, "est_in": float(r_["score"]), "sens_walls": dict(W), "arms": []}
    arms = json.loads(os.environ["DGR_SWEEP"])
    Nprev = None
    for s in arms:
        Wa = {}
        N_ = s.get("N", a.dgr_nets)
        if N_ != Nprev:
            prepare(C, S, a, cd, FT, HD, Wa, N_); Nprev = N_; rec_b = dict(Wa)
        if s.get("prof") == 2:
            os.environ["DG_PROF_TF"] = "1"
        else:
            os.environ.pop("DG_PROF_TF", None)
        t = time.perf_counter()
        X.start(s.get("init", a.dgr_init))
        gw = {}; gi = None
        if s.get("gnn"):
            K0 = s.get("K", a.dgr)
            gi = GN.start(C, a, X, float(FT.GRID) / 2000.0, gw, s.get("gnn_file", a.dgr_gnn), os.environ.get("DGR_CPU") != "1", clamp=K0 > 0, sm=s.get("sm"), x0=s.get("x0"))
        tg = time.perf_counter()
        if s.get("K", a.dgr) > 0:
            res = X.run(s.get("K", a.dgr), lr=s.get("lr", a.dgr_lr), t1=s.get("t1", a.dgr_t1), tau=s.get("tau", a.dgr_tau), dl=s.get("dl", a.dgr_dl), dat=s.get("dat", a.dgr_dat), lrK=s.get("lrk", a.dgr_lrk), lrC=s.get("lrc", a.dgr_lrc), prof=bool(s.get("prof")))
        else:
            res = {"steps": 0, "vals": np.zeros((1, 8)), "prof": {}}
        desc = time.perf_counter() - t; steps_s = time.perf_counter() - tg
        chg = X.get("a_chg", np.uint8); nE_ = len(chg)
        moved_k, i_m, r_new, o_new, nbad = round_rows(C, S, Wa)
        ar = dict(s); ar.update(steps=res["steps"], descent_s=desc, s_per_step=desc / max(res["steps"], 1), arcs_last_step=int(chg.sum()), arcs=nE_, runs=len(C.run_), ent=len(C.A["ec"]), nodes=C.nk, nets_free=len(C.kept), runs_moved=int(C.mv.sum()), nets_moved=int(len(moved_k)), not_rebuilt=nbad, F=[float(res["vals"][0][4]), float(res["vals"][-1][4])])
        if s.get("prof"):
            ar["prof"] = {k: round(v / max(res["steps"], 1), 5) for k, v in res["prof"].items()}
            ar["arcs_per_step"] = res["vals"][:, 7].astype(int).tolist()
        if len(moved_k):
            undo = apply(S, i_m, r_new, o_new, C.rows, C.off, HD, Wa, C)
            t = time.perf_counter()
            o2, rc2, r2 = retime("dgr")
            ar["test_s"] = time.perf_counter() - t
            ar["gain"] = float(r_["score"]) - float(r2["score"])
            ar["est"] = float(r2["score"]); ar["d_of"] = S.w4 * (float(r2["of"]) - float(r_["of"])); ar["d_pw"] = S.w3 * (float(r2["pw"]) - float(r_["pw"]))
            ar["d_wns"] = S.w1 * (float(r2["wns"]) - float(r_["wns"])); ar["d_tns"] = S.w2 * (float(r2["tns"]) - float(r_["tns"])) / S.nend
            t = time.perf_counter()
            undo.go(); K_.reject()
            ar["undo_s"] = time.perf_counter() - t
        ar["walls"] = {k: round(v, 4) for k, v in Wa.items()}
        ar["steps_s"] = steps_s
        if gi is not None:
            ar["gnn"] = gi; ar["gnn_walls"] = {k: round(v, 4) for k, v in gw.items()}
            if os.environ.get("GNN_DUMP") and not rec.get("dumped"):     # tests: the chain's inputs at this state
                import pickle
                net_ = GN.NETS[s.get("gnn_file", a.dgr_gnn)]
                D_ = net_.dump(); z_, _st = net_.forward(want=True)
                pickle.dump(dict(inputs=D_, z=z_, run=C.run_, route_idx=C.i_k, cand2=C.cand2, cell_key=C.A["cell_key"], d0=C.A["d0"], cap=C.A["cap"], cur=C.run_[:, 5], g=None, X=int(S.la.X), Y=int(S.la.Y)), open(os.environ["GNN_DUMP"], "wb"), protocol=5)
                rec["dumped"] = True
        rec["arms"].append(ar)
        print("[dgr_sweep] " + json.dumps({k: ar.get(k) for k in ("tag", "gnn", "N", "K", "lrk", "lr", "t1", "dl", "tau", "steps", "steps_s", "descent_s", "s_per_step", "arcs_last_step", "runs_moved", "nets_moved", "gain", "test_s", "gnn_walls")}, default=float), flush=True)
        with open(out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
        os.replace(out + ".tmp", out)
    end()
    return rec


class Undo:
    """what S.revert needs after a rejected change, in arrays: the chain's gate calls S.revert(undo)"""
    def __init__(self, S, i_m, r7o, r7n, prev, HD):
        self.S, self.i_m, self.r7o, self.r7n, self.prev, self.HD = S, i_m, r7o, r7n, prev, HD

    def go(self):
        S, HD = self.S, self.HD
        if not (HD._add(S.la, S.la.d, self.r7n, -1.0) and HD._add(S.la, S.la.d, self.r7o, 1.0)):
            raise RuntimeError("dgr: hd_add refused on undo")
        new = S.new
        for i in self.i_m.tolist():
            p = self.prev.get(i)
            if p is None:
                new.pop(i, None)
            else:
                new[i] = p
        S._dirty.update(S.tid[self.i_m].tolist())


def wrap_revert(S):
    """S.revert that takes an Undo as well; returns the previous one"""
    r0 = S.revert
    def rv(u):
        if isinstance(u, Undo):
            u.go()
        else:
            r0(u)
    S.revert = rv
    return r0


def apply(S, i_m, r_new, o_new, rows, off, HD, W=None, C=None):
    """the nets i_m take the rows r_new: demand (C++ adder), S.new, dirty marks, the normaliser's cache filled (one C++ call instead of the re-time's list conversion). -> Undo"""
    t = time.perf_counter()
    la = S.la
    i_m = np.asarray(i_m, np.int64)
    ro, oo = gather(rows, off, i_m)
    r7o = np.empty((len(ro), 7), np.int64); r7o[:, 0] = np.repeat(i_m, np.diff(oo)); r7o[:, 1:] = ro
    r7n = np.empty((len(r_new), 7), np.int64); r7n[:, 0] = np.repeat(i_m, np.diff(o_new)); r7n[:, 1:] = r_new
    if not (HD._add(la, la.d, r7o, -1.0) and HD._add(la, la.d, r7n, 1.0)):
        raise RuntimeError("dgr: hd_add refused")
    if W is not None:
        n_ = time.perf_counter(); W["apply_dem_s"] = W.get("apply_dem_s", 0.0) + n_ - t; t = n_
    tup = list(map(tuple, r_new.tolist()))
    ol = o_new.tolist()
    new = S.new; prev = {}
    il = i_m.tolist()
    for q, i in enumerate(il):
        if i in new:
            prev[i] = new[i]
        new[i] = tup[ol[q]:ol[q + 1]]
    S._dirty.update(S.tid[i_m].tolist())
    if W is not None:
        n_ = time.perf_counter(); W["apply_py_s"] = W.get("apply_py_s", 0.0) + n_ - t; t = n_
    if C is not None:                # the re-time's normaliser finds every moved net in its cache
        nr, no = normalise(S, r_new, o_new, i_m, C.pa0, C.pa1, C.T_)
        b7 = np.empty((len(nr), 7), np.int64); b7[:, 0] = np.repeat(i_m, np.diff(no)); b7[:, 1:] = nr
        cache = S.__dict__.setdefault("_f3x_nc", {})
        nl = no.tolist()
        for q, i in enumerate(il):
            cache[i] = (new[i], b7[nl[q]:nl[q + 1]])
        if W is not None:
            W["apply_norm_s"] = W.get("apply_norm_s", 0.0) + time.perf_counter() - t
    return Undo(S, i_m, r7o, r7n, prev, HD)
