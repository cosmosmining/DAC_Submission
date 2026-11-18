#!/usr/bin/env python3
"""dgb.py: ctypes of dgb.cc: the descent's problem as device arrays, no Python per-net loop.  N = build(rows6, noff, net_j, S-tables, la, ...) -> dict of numpy arrays for dg.Dg.problem_arrays; rows_of(N, choice, ks) -> rows of moved nets."""
import ctypes, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_L = None


def lib():
    global _L
    if _L is None:
        _L = ctypes.CDLL(os.path.join(HERE, "libdgb.so"))
        P = ctypes.c_void_p; I = ctypes.c_int64; i = ctypes.c_int
        _L.dgb_nets.argtypes = [I] + [P] * 9 + [P, i, P, i, i, i, P]; _L.dgb_nets.restype = i
        _L.dgb_get.argtypes = [P] * 6
        _L.dgb_cells.argtypes = [I, P, I, P, P, i, P, i, i, i, i, P]; _L.dgb_cells.restype = I
        _L.dgb_cells_get.argtypes = [i] + [P] * 4 + [P, I, I, I, P, I, I, I, P, i] + [P] * 15
        _L.dgb_map.argtypes = [I, P, I, P, P, P, P, i]; _L.dgb_map.restype = I
        _L.dgb_rows.argtypes = [I, P, P, P, P, P, P, P, i, P, i, P, P]; _L.dgb_rows.restype = I
        _L.dgb_ovf_pot.argtypes = [I, P, P, P, I, I, I, P, I, I, I, P, i, P]
        _L.dgb_topo.argtypes = [P, I, ctypes.c_double, I, P, P, P, P, P, P, P, I, I, ctypes.c_double, i, P]; _L.dgb_topo.restype = I
        _L.dgb_topo_get.argtypes = [P, P, P, P, P, i, I, I, I]
        _L.dgb_same.argtypes = [I, P, P, P, P, i, P]
        _L.dgb_rc_count.argtypes = [I, I, I] + [P] * 4 + [I] + [P] * 3
        _L.dgb_rc.argtypes = [I, I, I, I] + [P] * 6 + [I] + [P] * 26
        _L.dgb_pr.argtypes = [I, P, I, P, P]
    return _L


def strides8(a):
    assert a.dtype == np.float64 and all(s % 8 == 0 and s >= 0 for s in a.strides)
    return [s // 8 for s in a.strides]


def nets(rows6, noff, net_j, pin_start, pin_end, ap_start, ap_cnt, ap, dirs, H, V, L, T):
    Lb = lib()
    nn = len(noff) - 1
    rows6 = np.ascontiguousarray(rows6, np.int64); noff = np.ascontiguousarray(noff, np.int64); net_j = np.ascontiguousarray(net_j, np.int64)
    Ha = np.array(H, np.int64); Va = np.array(V, np.int64); cnt = np.zeros(6, np.int64)
    dirs_ = np.ascontiguousarray(np.asarray(dirs), np.int32)
    rc = Lb.dgb_nets(nn, net_j.ctypes.data, noff.ctypes.data, rows6.ctypes.data, pin_start.ctypes.data, pin_end.ctypes.data, ap_start.ctypes.data, ap_cnt.ctypes.data, ap.ctypes.data, dirs_.ctypes.data,
                     Ha.ctypes.data, len(H), Va.ctypes.data, len(V), int(L), int(T), cnt.ctypes.data)
    if rc != 0:
        raise RuntimeError("dgb_nets")
    k, r, p, m = (int(v) for v in cnt[:4])
    N = dict(kept=np.zeros(k, np.int64), node_off=np.zeros(k + 1, np.int64), nodes=np.zeros((m, 4), np.int64), run_off=np.zeros(k + 1, np.int64), run=np.zeros((r, 8), np.int64), pts=np.zeros((p, 4), np.int64),
             strict_held=int(cnt[4]), not_connected=int(cnt[5]), dirs=dirs_)
    Lb.dgb_get(N["kept"].ctypes.data, N["node_off"].ctypes.data, N["nodes"].ctypes.data, N["run_off"].ctypes.data, N["run"].ctypes.data, N["pts"].ctypes.data)
    return N


def cells(N, cand2, d, c, ul, X, Y, L, T, RL=np.float32):
    Lb = lib()
    run, pts = N["run"], N["pts"]
    K = cand2.shape[1]; R = len(run); npt = len(pts)
    cand2 = np.ascontiguousarray(cand2, np.int32); cnt = np.zeros(2, np.int64)
    nc = int(Lb.dgb_cells(R, run.ctypes.data, npt, pts.ctypes.data, cand2.ctypes.data, K, N["dirs"].ctypes.data, int(L), int(X), int(Y), int(T), cnt.ctypes.data))
    E, V = int(cnt[0]), int(cnt[1])
    i32 = lambda n: np.empty(n, np.int32)
    A = dict(ent_ptr=i32(R * K + 1), ec=i32(E), cp=i32(nc + 1), ce=i32(E), cvp=i32(nc + 1), cvi=i32(V), cvg=np.empty(V, RL), base=np.empty(nc, RL), u=np.empty(nc, RL), copen=np.empty(nc, np.uint8),
             pvp=i32(npt + 1), pvz=np.empty(V, np.uint8), pvc=i32(V), pvg=np.empty(V, RL), cell_key=np.empty(nc, np.int64))
    sd = strides8(d); sc = strides8(c); ul_ = np.ascontiguousarray(ul, np.float64)
    Lb.dgb_cells_get(1 if RL == np.float64 else 0, run.ctypes.data, pts.ctypes.data, cand2.ctypes.data, N["dirs"].ctypes.data, d.ctypes.data, sd[0], sd[1], sd[2], c.ctypes.data, sc[0], sc[1], sc[2], ul_.ctypes.data, int(T),
                     *[A[k].ctypes.data for k in ("ent_ptr", "ec", "cp", "ce", "cvp", "cvi", "cvg", "base", "u", "copen", "pvp", "pvz", "pvc", "pvg", "cell_key")])
    return A


def wire_map(wrows, run_tn, run, zcur, T):
    Lb = lib()
    wrows = np.ascontiguousarray(wrows, np.int64); run_tn = np.ascontiguousarray(run_tn, np.int64); zcur = np.ascontiguousarray(zcur, np.int32)
    out = np.empty(len(wrows), np.int64)
    h = Lb.dgb_map(len(wrows), wrows.ctypes.data, len(run), run_tn.ctypes.data, run.ctypes.data, zcur.ctypes.data, out.ctypes.data, int(T))
    return out, int(h)


def rows_of(N, choice, ks, cand2, T):
    """rows [m, 6] GCell of the kept nets ks at the choice (per run: candidate index); off [len(ks) + 1]"""
    Lb = lib()
    ks = np.ascontiguousarray(ks, np.int64); ch = np.ascontiguousarray(choice, np.int64); off = np.zeros(len(ks) + 1, np.int64)
    cand2 = np.ascontiguousarray(cand2, np.int32); K = cand2.shape[1]
    tot = Lb.dgb_rows(len(ks), ks.ctypes.data, ch.ctypes.data, N["run_off"].ctypes.data, N["run"].ctypes.data, N["node_off"].ctypes.data, N["nodes"].ctypes.data, cand2.ctypes.data, K, N["dirs"].ctypes.data, int(T), off.ctypes.data, None)
    rows = np.empty((int(tot), 6), np.int64)
    Lb.dgb_rows(len(ks), ks.ctypes.data, ch.ctypes.data, N["run_off"].ctypes.data, N["run"].ctypes.data, N["node_off"].ctypes.data, N["nodes"].ctypes.data, cand2.ctypes.data, K, N["dirs"].ctypes.data, int(T), off.ctypes.data, rows.ctypes.data)
    return rows, off


def ovf_pot(rows6, noff, d, c, ul, T):
    Lb = lib()
    rows6 = np.ascontiguousarray(rows6, np.int64); noff = np.ascontiguousarray(noff, np.int64)
    pot = np.zeros(len(noff) - 1)
    sd = strides8(d); sc = strides8(c); ul_ = np.ascontiguousarray(ul, np.float64)
    Lb.dgb_ovf_pot(len(pot), noff.ctypes.data, rows6.ctypes.data, d.ctypes.data, sd[0], sd[1], sd[2], c.ctypes.data, sc[0], sc[1], sc[2], ul_.ctypes.data, int(T), pot.ctypes.data)
    return pot


def same_rows(ra, oa, rb, ob, T):
    ok = np.zeros(len(oa) - 1, np.uint8)
    ra = np.ascontiguousarray(ra, np.int64); rb = np.ascontiguousarray(rb, np.int64); oa = np.ascontiguousarray(oa, np.int64); ob = np.ascontiguousarray(ob, np.int64)
    lib().dgb_same(len(ok), ra.ctypes.data, oa.ctypes.data, rb.ctypes.data, ob.ctypes.data, int(T), ok.ctypes.data)
    return ok.astype(bool)


def topo(seg, tb, T, stub_um=1.05):
    """rc_topo.cc csta_topo of the rows seg, on threads; the dict of dg.topo_np without the depth buckets. tb: ap_off ap_idx ap ap_pid net_drv layer_rc via_r um (arrays in the kernel's types)"""
    Lb = lib()
    s = np.ascontiguousarray(seg, np.int64); cnt = np.zeros(2, np.int64)
    nn = len(tb["net_drv"])
    Lb.dgb_topo(s.ctypes.data, len(s), stub_um, nn, tb["ap_off"].ctypes.data, tb["ap_idx"].ctypes.data, tb["ap"].ctypes.data, tb["ap_pid"].ctypes.data, tb["net_drv"].ctypes.data, tb["layer_rc"].ctypes.data, tb["via_r"].ctypes.data, len(tb["layer_rc"]) // 2, len(tb["via_r"]), float(tb["um"]), int(T), cnt.ctypes.data)
    ns, nk = int(cnt[0]), int(cnt[1])
    si = np.empty((6, ns), np.int64); sg = np.empty(ns); ni = np.empty((9, nk), np.int64); nd = np.empty((3, nk)); nt = np.empty((2, nn), np.int64)
    Lb.dgb_topo_get(si.ctypes.data, sg.ctypes.data, ni.ctypes.data, nd.ctypes.data, nt.ctypes.data, int(T), ns, nk, nn)
    return dict(ns=ns, nk=nk, nn=nn, si=si, sg=sg, ni=ni, nd=nd, nt=nt, wire=np.ascontiguousarray(np.nonzero(si[3] == 0)[0], np.int64), s_net=si[0])


def pr_csr(run, npt):
    prp = np.empty(npt + 1, np.int32); prr = np.empty(2 * len(run), np.int32)
    lib().dgb_pr(len(run), run.ctypes.data, npt, prp.ctypes.data, prr.ctypes.data)
    return prp, prr


def rc_tables(t, sr, R, cd_rc):
    """the arrays of dg.Dg.rc from dgb.topo's dict. cd_rc: layer_rc via_r pin_cap pin_is_drv (kernel types)"""
    Lb = lib()
    sr = np.ascontiguousarray(sr, np.int64); cnt = np.zeros(6, np.int64)
    ns, nk, nn = t["ns"], t["nk"], t["nn"]; wire = t["wire"]; si, sg, ni, nd, nt = t["si"], t["sg"], t["ni"], t["nd"], t["nt"]
    Lb.dgb_rc_count(ns, nk, nn, si.ctypes.data, ni.ctypes.data, nt.ctypes.data, wire.ctypes.data, len(wire), sr.ctypes.data, cd_rc["pin_is_drv"].ctypes.data, cnt.ctypes.data)
    c0, c1, c2, c3, c4, c5 = (int(x) for x in cnt)
    i32 = lambda n: np.empty(n, np.int32); f8 = lambda n: np.empty(n, np.float64)
    A = dict(Rp0=f8(nk), rp_run=i32(nk), rp_gl=f8(nk), cap0=f8(2 * nk), reach=np.empty(nk, np.uint8), nsp=i32(nk + 1), nsr=i32(c0), nsg=f8(c0), rsp=i32(R + 1), rsn=i32(c0), rsg=f8(c0), rrp=i32(R + 1), rrn=i32(c1), rrg=f8(c1),
             rnp=i32(c3 + 1), rno=i32(c2), par_n=np.empty(nk, np.int64), roots=i32(c4), rootn=i32(c4), pn_node=i32(c5), pn_pin=i32(c5))
    Lb.dgb_rc(ns, nk, nn, R, si.ctypes.data, sg.ctypes.data, ni.ctypes.data, nd.ctypes.data, nt.ctypes.data, wire.ctypes.data, len(wire), sr.ctypes.data, cd_rc["layer_rc"].ctypes.data, cd_rc["via_r"].ctypes.data, cd_rc["pin_cap"].ctypes.data, cd_rc["pin_is_drv"].ctypes.data,
              *[A[k].ctypes.data for k in ("Rp0", "rp_run", "rp_gl", "cap0", "reach", "nsp", "nsr", "nsg", "rsp", "rsn", "rsg", "rrp", "rrn", "rrg", "rnp", "rno", "par_n", "roots", "rootn", "pn_node", "pn_pin")])
    A["nrn"] = c3
    return A
