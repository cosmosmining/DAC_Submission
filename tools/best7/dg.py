#!/usr/bin/env python3
"""dg.py: numpy driver of dg_kern.cu, no torch. Problem P (tools/e2et/prep.py's dictionary, or tools/best7/dgb.py's) + csta cache data + route segments -> device tables; K steps in one kernel call.
  D = Dg(lib); D.timer_tables(cd, w1, w2, nend)           # the design's tables: once per process, can run on a thread while the first pass runs
  D.problem(P, w)                                        # runs, cells, points
  D.rc(topo_np(...), sr, rc_base)                        # wire segments of the free nets -> runs
  D.start(cur, init); D.run(K, lr, t0, t1, tau); ch = D.choice()"""
import ctypes, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEG, BIG = -1e30, 1e30
PTRS = """Cand Lr Lc Cc z2k rdir pwk rp0 rp1 x p am av l0 l1 rr cr pwv ent_ptr ec cp ce cvp cvi cvg base u copen uc uv prp prr pt_lo pt_hi pvp pvz pvc pvg eF eG eN dv gd
Rp0 rp_run rp_gl cap0 reach nsp nsr nsg Rp y1 y2 y3 elmn gR gy1 gy2 gy3 gelmn rsp rsn rsg rrp rrn rrg roots rootn pn_node pn_pin c1 c2 rpi ctot elm gc1 gc2 grpi gctot gelm
rnp rno par_n a_in a_chg p_chg e_wd e_ls e_el e_pk ecp ecr ep_pin ck_tr ck_tf ck_type ck_evt ck_cks ck_off REL ep_e ep_g AT SL gAT gSL""".split()
TBL = ["tabA", "tabB", "tabV", "tabNA", "tabNB", "ce_from", "ce_to", "ce_irf", "ce_orf", "ce_lib", "ce_td", "ce_ts", "g_start",
       "g_to", "g_orf", "ne_drv", "ne_snk", "pin_net", "L_vth", "L_vl", "L_vh", "L_der", "L_vin", "is_port", "c1", "c2", "rpi",
       "ctot", "elm", "par", "AT", "SL", "a_delay", "a_slew", "a_jd", "a_js", "a_rec", "a_alg", "a_valid", "pk", "e_d", "gAT",
       "gSL", "gREC", "gc1", "gc2", "grpi", "gctot", "gelm", "idx_arc", "idx_grp", "idx_edge",
       "ce_lo_h", "g_lo_h", "ne_lo_h", "cone_arc_lo_h", "cone_grp_lo_h", "cone_edge_lo_h",
       "tabM", "tabT", "P", "K", "nn", "def_lib", "nE", "nNE", "nlev"]
RCT = ["Rp", "y1", "y2", "y3", "elm", "gR", "gy1", "gy2", "gy3", "gelm", "nodes", "par", "plist", "pptr", "lo_h", "plo_h", "maxd"]
IDX = "R K L npt nc nk nn P nr npn nep KT nevt nrn".split()


def csr(keys, n, *vals):
    """stable group by key: ptr [n + 1] (int32), then each value in key order"""
    keys = np.asarray(keys, np.int64)
    o = np.argsort(keys, kind="stable")
    ptr = np.zeros(n + 1, np.int64); ptr[1:] = np.bincount(keys, minlength=n)[:n]; ptr = np.cumsum(ptr).astype(np.int32)
    return (ptr,) + tuple(np.asarray(v)[o] for v in vals)


def topo_np(lib, seg, cd, stub_um=1.05):
    """dtimer.Topo in numpy (rc_topo.cc through librc_topo.so); the depth buckets as gputimer._rc_csr's CSR"""
    s = np.ascontiguousarray(seg, np.int64)
    cnt = np.zeros(2, np.int64)
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.csta_topo.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p]
    lib.csta_topo.restype = ctypes.c_int64
    lib.csta_topo(p(s), len(s), stub_um, p(cnt))
    ns, nk = int(cnt[0]), int(cnt[1])
    nn = len(cd.net_names)
    si = np.empty((6, ns), np.int64); sg = np.empty(ns); ni = np.empty((9, nk), np.int64); nd = np.empty((3, nk)); nt = np.empty((2, nn), np.int64)
    lib.csta_topo_get.argtypes = [ctypes.c_void_p] * 5
    lib.csta_topo_get(p(si), p(sg), p(ni), p(nd), p(nt))
    t = dict(ns=ns, nk=nk, nn=nn, s_net=si[0], s_a=si[1], s_b=si[2], s_via=si[3].astype(bool), s_li=si[4], s_vi=si[5], s_glen=sg,
             n_net=ni[0], n_kind=ni[1], n_pin=ni[2], n_par=ni[3], n_dep=ni[4], n_rk=ni[5], n_ri=ni[6], n_reach=ni[7].astype(bool), n_pg=ni[8],
             n_grv=nd[0], n_sr=nd[1], n_sc=nd[2], root=nt[0], hasd=nt[1].astype(bool))
    t["has_seg"] = np.bincount(si[0], minlength=nn) > 0
    t["wire"] = np.nonzero(~t["s_via"])[0]
    dep, reach, par = t["n_dep"], t["n_reach"], t["n_par"]
    m = reach & (par >= 0)
    order = np.nonzero(m)[0]
    order = order[np.argsort(dep[order], kind="stable")]
    maxd = int(dep[order].max()) if len(order) else 0
    bounds = np.searchsorted(dep[order], np.arange(maxd + 2))
    nodes, plist, pptr, lo, plo = [], [], [], [], []
    n = 0; np_ = 0
    for d in range(maxd + 1):
        lo.append(n); plo.append(np_)
        if d == 0:
            continue
        wv = order[bounds[d]:bounds[d + 1]]
        if len(wv) == 0:
            continue
        o = np.argsort(par[wv], kind="stable")
        wv = wv[o]
        pw = par[wv]
        st = np.nonzero(np.r_[True, pw[1:] != pw[:-1]])[0]
        nodes.append(wv); plist.append(pw[st]); pptr.append(st + n)
        n += len(wv); np_ += len(st)
    lo.append(n); plo.append(np_)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    t.update(maxd=maxd, nodes=np.ascontiguousarray(cat(nodes), np.int64), plist=np.ascontiguousarray(cat(plist), np.int64), pptr=np.ascontiguousarray(np.r_[cat(pptr), n], np.int64),
             lo_h=np.ascontiguousarray(lo, np.int64), plo_h=np.ascontiguousarray(plo, np.int64))
    t["reach_nodes"] = np.nonzero(reach)[0]
    t["pin_nodes"] = np.nonzero((ni[1] == 1) & reach)[0]
    t["attach"] = np.nonzero(ni[1] == 1)[0]
    t["roots"] = np.nonzero(nt[0] >= 0)[0]
    return t


class Dg:
    def __init__(self, lib=None, f64=False, gpu=True):
        self.f64 = f64
        self.RL = np.float64 if f64 else np.float32
        if lib is None:
            lib = os.path.join(HERE, ("libdg_cuda" if gpu else "libdg_cpu") + ("_f64" if f64 else "") + ".so")
            if gpu and not f64 and os.environ.get("DG_LIB_GPU"):   # 11 Nov 2026: a variant build (-DDG_REC32 / -DDG_UM_FALLBACK)
                lib = os.environ["DG_LIB_GPU"]
        L = ctypes.CDLL(lib)
        self.L = L
        try:                                   # bytes per value of the timer-tape record: 8, or 4 for a -DDG_REC32 build; older builds: 8
            L.dg_rec_bytes.restype = ctypes.c_int; self.rec_b = int(L.dg_rec_bytes())
        except AttributeError:
            self.rec_b = 8
        vp = ctypes.c_void_p; I = ctypes.c_int64
        L.dg_alloc.argtypes = [I]; L.dg_alloc.restype = vp
        L.dg_free.argtypes = [vp]
        L.dg_put.argtypes = [vp, vp, I]; L.dg_get.argtypes = [vp, vp, I]; L.dg_zero.argtypes = [vp, I]
        L.dg_run.argtypes = [vp, vp, vp, vp, vp, vp, I, I, vp, vp, vp, ctypes.c_int]
        L.dg_eval.argtypes = [vp, vp, vp, vp, vp, vp, I, ctypes.c_double, ctypes.c_int, vp, vp]
        L.dg_sens.argtypes = [vp, vp, vp, vp, vp, I, vp]
        assert L.dg_nptrs() == len(PTRS), (L.dg_nptrs(), len(PTRS))
        assert L.gt_ntable() == len(TBL) and L.gt_rc_ntable() == len(RCT)
        self.L.dg_init()
        self.p = {}            # name -> device pointer
        self.size = {}
        self.host = {}         # host arrays the kernels read through pointers
        self.I = {k: 0 for k in IDX}
        self.Dv = np.zeros(16)
        self.timer_on = False

    # ---------------------------------------------------------------- memory
    def up(self, name, a):
        a = np.ascontiguousarray(a)
        if name in self.p:
            self.L.dg_free(self.p[name])
        d = self.L.dg_alloc(a.nbytes)
        if d is None:
            raise MemoryError(name)
        if self.L.dg_put(d, a.ctypes.data, a.nbytes) != 0:
            raise RuntimeError("dg_put " + name)
        self.p[name] = d; self.size[name] = a.nbytes
        return d

    def zeros(self, name, nbytes):
        if name in self.p:
            self.L.dg_free(self.p[name])
        d = self.L.dg_alloc(nbytes)
        if d is None:
            raise MemoryError(name)
        self.L.dg_zero(d, nbytes)
        self.p[name] = d; self.size[name] = nbytes
        return d

    def get(self, name, dtype, shape=None):
        n = self.size[name]
        a = np.empty(n // np.dtype(dtype).itemsize, dtype)
        self.L.dg_get(a.ctypes.data, self.p[name], n)
        return a if shape is None else a.reshape(shape)

    def put(self, name, a):
        a = np.ascontiguousarray(a)
        assert a.nbytes == self.size[name], (name, a.nbytes, self.size[name])
        self.L.dg_put(self.p[name], a.ctypes.data, a.nbytes)

    # ---------------------------------------------------------------- the design's timer tables (no route)
    def timer_tables(self, cd, w1, w2, nend):
        T = cd.lib.tab
        i64 = lambda a: np.ascontiguousarray(np.asarray(a, np.int64))
        f64 = lambda a: np.ascontiguousarray(np.asarray(a, np.float64))
        self.tm = {"tabM": int(T.M), "tabT": len(T.NA)}
        self.up("tabA", f64(T.A).reshape(-1)); self.up("tabB", f64(T.B).reshape(-1)); self.up("tabV", f64(T.Vf).reshape(-1))
        self.up("tabNA", i64(T.NA)); self.up("tabNB", i64(T.NB))
        for n in ("ce_from", "ce_to", "ce_irf", "ce_orf", "ce_lib", "ce_td", "ce_ts", "g_start", "g_to", "g_orf", "ne_drv", "ne_snk", "pin_net"):
            self.up(n, i64(getattr(cd, n)))
        for n in ("L_vth", "L_vl", "L_vh", "L_der", "L_vin"):
            self.up(n, f64(getattr(cd, n)).reshape(-1))
        self.up("is_port", np.asarray(cd.is_port).astype(np.uint8))
        self.host.update(ce_lo_h=i64(cd.ce_lo), g_lo_h=i64(cd.g_lo), ne_lo_h=i64(cd.ne_lo))
        P, K = int(cd.P), int(cd.K)
        nE, nNE, nn = len(cd.ce_from), len(cd.ne_drv), len(cd.net_names)
        self.tm.update(P=P, K=K, nn=nn, def_lib=int(cd.def_lib), nE=nE, nNE=nNE, nlev=int(cd.nlev))
        AT = np.full((P, K, 2), NEG, np.float64)
        AT[i64(cd.sp_pin), i64(cd.sp_cls), i64(cd.sp_rf)] = f64(cd.sp_at)
        SL = np.zeros((P, 2), np.float64)
        rg = np.nonzero(np.asarray(cd.reg_ck).astype(bool))[0]
        SL[rg] = f64(cd.ck_slew).reshape(-1, 2)[rg]
        self.AT0, self.SL0 = AT, SL
        self.up("AT", AT); self.up("SL", SL)
        for n, b in (("a_delay", nE * 8), ("a_slew", nE * 8), ("a_jd", nE * 40), ("a_js", nE * 40), ("a_rec", nE * 72 * self.rec_b), ("a_alg", nE), ("a_valid", nE), ("e_d", nNE * 24 * 8),
                     ("gAT", P * K * 2 * 8), ("gSL", P * 2 * 8), ("gREC", P * 2 * 5 * 8), ("gc1", nn * 16), ("gc2", nn * 16), ("grpi", nn * 16), ("gctot", nn * 16), ("gelm", P * 16),
                     ("c1", nn * 16), ("c2", nn * 16), ("rpi", nn * 16), ("ctot", nn * 16), ("elm", P * 16), ("ep_e", 8), ("ep_g", 8)):
            self.zeros(n, b)
        self.up("pk", np.full(P * 2, -1, np.int64))
        for n, b in (("a_in", nE * 40), ("a_chg", nE), ("p_chg", P * 2), ("e_wd", nNE * 16), ("e_ls", nNE * 16), ("e_el", nNE * 16)):
            self.zeros(n, b)
        self.up("e_pk", np.full(nNE * 2, -2, np.int64))
        R = cd.ck_rows
        ep_pin = i64(cd.ep_pin)
        ck_pin = i64(R["pin"])
        self.up("ck_tr", i64(R["tr"])); self.up("ck_tf", i64(R["tf"])); self.up("ck_type", i64(R["type"])); self.up("ck_evt", i64(R["evt"]))
        self.up("ck_cks", f64(R["cks"])); self.up("ck_off", f64(R["off"]))
        REL = f64(cd.REL)
        self.up("REL", REL); self.I["nevt"] = REL.shape[1]; assert REL.shape[0] == K, REL.shape
        o = np.argsort(ck_pin, kind="stable"); sp = ck_pin[o]
        a, b = np.searchsorted(sp, ep_pin), np.searchsorted(sp, ep_pin, side="right")
        n = b - a
        ecp = np.zeros(len(ep_pin) + 1, np.int64); ecp[1:] = np.cumsum(n)
        ecr = np.zeros(int(ecp[-1]), np.int64)
        off = np.repeat(a, n) + (np.arange(int(ecp[-1])) - np.repeat(ecp[:-1], n))
        ecr = o[off]
        self.up("ecp", ecp.astype(np.int32)); self.up("ecr", ecr.astype(np.int32)); self.up("ep_pin", ep_pin)
        self.zeros("ep_e", len(ep_pin) * 8); self.zeros("ep_g", len(ep_pin) * 8)
        self.I.update(nep=len(ep_pin), KT=K, P=P, nn=nn)
        ex_ = getattr(cd, "extra", None)
        self.lumped = f64(ex_["lumped"]).reshape(-1, 2) if ex_ is not None and "lumped" in ex_ else None
        self.w12 = (float(w1), float(w2), float(nend))
        self.cd = cd
        self.timer_on = True

    # ---------------------------------------------------------------- the runs (P: prep.py's keys; w: w3, w4)
    def problem(self, P, w3, w4, layer_rc=None):
        RL = self.RL
        L = int(P["L"]); K = int(P["K"]); dirs = list(P["dirs"])
        H = [z for z in range(1, L) if dirs[z] == 0]; V = [z for z in range(1, L) if dirs[z] == 1]
        Cand = np.full((2, K), -1, np.int32); Cand[0, :len(H)] = H; Cand[1, :len(V)] = V
        z2k = np.full((2, L), -1, np.int32)
        for d in (0, 1):
            for k in range(K):
                if Cand[d, k] >= 0:
                    z2k[d, Cand[d, k]] = k
        rdir = np.asarray(P["run_dir"]).astype(np.uint8); R = len(rdir)
        p0 = np.ascontiguousarray(P["run_p0"], np.int32); p1 = np.ascontiguousarray(P["run_p1"], np.int32)
        npt = int(max(p0.max(), p1.max())) + 1 if R else 0
        rn = np.asarray(P["run_net"], np.int64)
        lum = np.asarray(P["run_len"], RL) * RL(P["um"])
        kpn = np.asarray(P["kPn"], RL)
        self.up("Cand", Cand); self.up("z2k", z2k); self.up("rdir", rdir); self.up("rp0", p0); self.up("rp1", p1)
        self.up("pwk", kpn[rn] * lum)
        Cz = np.asarray(P["Cz"], RL)
        self.up("Cc", np.where(Cand >= 0, Cz[np.maximum(Cand, 0)], RL(0)).astype(RL))
        if layer_rc is not None:
            lr = np.asarray(layer_rc, np.float64).reshape(-1, 2)
            self.up("Lr", np.where(Cand >= 0, lr[np.maximum(Cand, 0), 0], 0.0)); self.up("Lc", np.where(Cand >= 0, lr[np.maximum(Cand, 0), 1], 0.0))
        rk_b = R * K * RL().nbytes
        for n in ("x", "p", "am", "av", "l0", "l1"):
            self.zeros(n, rk_b)
        self.zeros("rr", R * 8); self.zeros("cr", R * 8); self.zeros("pwv", R * RL().nbytes)
        # wire entries
        ent_rk = np.asarray(P["ent_rk"], np.int64); ent_cell = np.asarray(P["ent_cell"], np.int64)
        d0 = np.asarray(P["d0"], RL); cap = np.asarray(P["cap"], RL); u = np.asarray(P["u"], RL)
        nc = len(d0)
        assert (np.diff(ent_rk) >= 0).all()
        self.up("ent_ptr", np.searchsorted(ent_rk, np.arange(R * K + 1)).astype(np.int32))
        self.up("ec", ent_cell.astype(np.int32))
        cp, ce = csr(ent_cell, nc, ent_rk)
        self.up("cp", cp); self.up("ce", ce.astype(np.int32))
        op = cap > RL(1e-3)
        alpha = np.where(op, RL(0.25), RL(0.75)).astype(RL)
        base = (alpha * d0 - np.where(op, RL(0.5) * cap, RL(0))).astype(RL)
        self.up("base", base); self.up("u", u); self.up("copen", op.astype(np.uint8))
        rb = nc * RL().nbytes
        self.zeros("uc", rb); self.zeros("uv", rb)
        # points
        np_pt = np.concatenate([p0, p1]); rr2 = np.concatenate([np.arange(R, dtype=np.int32)] * 2)
        prp, prr = csr(np_pt, npt, rr2)
        self.up("prp", prp); self.up("prr", prr.astype(np.int32))
        self.up("pt_lo", np.asarray(P["pt_lo"], np.int32)); self.up("pt_hi", np.asarray(P["pt_hi"], np.int32))
        via_pt = np.asarray(P["via_pt"], np.int64); via_z = np.asarray(P["via_z"], np.int64); via_cell = np.asarray(P["via_cell"], np.int64)
        via_w = np.asarray(P["via_w"], RL)
        gv = np.expm1(alpha[via_cell] * via_w).astype(RL)
        cvp, cvi, cvg = csr(via_cell, nc, (via_pt * L + via_z).astype(np.int32), gv)
        self.up("cvp", cvp); self.up("cvi", cvi); self.up("cvg", cvg)
        pvp, pvz, pvc, pvg = csr(via_pt, npt, via_z.astype(np.uint8), via_cell.astype(np.int32), gv)
        self.up("pvp", pvp); self.up("pvz", pvz); self.up("pvc", pvc); self.up("pvg", pvg)
        ptb = npt * L * RL().nbytes
        for n in ("eF", "eG", "eN", "dv", "gd"):
            self.zeros(n, ptb)
        self.I.update(R=R, K=K, L=L, npt=npt, nc=nc)
        self.Dv[0] = float(w3); self.Dv[1] = float(w4)
        self.cur = np.asarray(P["run_cur"], np.int64)
        self.mask = (Cand[rdir] >= 0)

    # ---------------------------------------------------------------- the runs from tools/best7/dgb.py (no Python per-net loop). run [R,8]: k d fixed lo hi cur p0 p1; pts [npt,4]; pwk [R]; A: dgb.cells
    def problem_arrays(self, run, pts, A, cand2, z2k, Cz, pwk, w3, w4, L):
        RL = self.RL
        K = cand2.shape[1]; R = len(run); npt = len(pts); nc = len(A["base"])
        rdir = run[:, 1].astype(np.uint8)
        for n, a in (("Cand", cand2.astype(np.int32)), ("z2k", z2k.astype(np.int32)), ("rdir", rdir), ("rp0", run[:, 6].astype(np.int32)), ("rp1", run[:, 7].astype(np.int32)), ("pwk", pwk.astype(RL))):
            self.up(n, a)
        self.up("Cc", np.where(cand2 >= 0, np.asarray(Cz, RL)[np.maximum(cand2, 0)], RL(0)).astype(RL))
        rk_b = R * K * RL().nbytes
        for n in ("x", "p", "am", "av", "l0", "l1"):
            self.zeros(n, rk_b)
        self.zeros("rr", R * 8); self.zeros("cr", R * 8); self.zeros("pwv", R * RL().nbytes)
        for n in ("ent_ptr", "ec", "cp", "ce", "cvp", "cvi", "cvg", "base", "u", "copen", "pvp", "pvz", "pvc", "pvg"):
            self.up(n, A[n])
        rb = nc * RL().nbytes
        self.zeros("uc", rb); self.zeros("uv", rb)
        self.up("prp", A["prp"]); self.up("prr", A["prr"])
        self.up("pt_lo", pts[:, 2].astype(np.int32)); self.up("pt_hi", pts[:, 3].astype(np.int32))
        ptb = npt * L * RL().nbytes
        for n in ("eF", "eG", "eN", "dv", "gd"):
            self.zeros(n, ptb)
        self.I.update(R=R, K=K, L=L, npt=npt, nc=nc)
        self.Dv[0] = float(w3); self.Dv[1] = float(w4)
        self.cur = run[:, 5].copy()
        self.mask = (cand2[rdir] >= 0)

    # ---------------------------------------------------------------- dg.Dg.rc for the arrays of dgb.rc_tables (the base RC stays as set_base left it)
    def rc_arrays(self, T, nk, lrc_cand):
        for n in ("Rp0", "rp_run", "rp_gl", "cap0", "reach", "nsp", "nsr", "nsg", "rsp", "rsn", "rsg", "rrp", "rrn", "rrg", "rnp", "rno", "par_n", "roots", "rootn", "pn_node", "pn_pin"):
            self.up(n, T[n])
        for n in ("Rp", "gR"):
            self.zeros(n, nk * 8)
        for n in ("y1", "y2", "y3", "elmn", "gy1", "gy2", "gy3", "gelmn"):
            self.zeros(n, nk * 16)
        self.up("Lr", lrc_cand[0]); self.up("Lc", lrc_cand[1])
        self.I.update(nk=nk, nr=len(T["roots"]), npn=len(T["pn_node"]), nrn=int(T["nrn"]))
        self._tables()

    # ---------------------------------------------------------------- the exact kernel's RC as the base of every net (c1 c2 rpi ctot elm [..,2], par), then one forward + backward of the timer alone: per-net sensitivities, no problem needed
    def set_base(self, rc):
        for k in ("c1", "c2", "rpi", "ctot", "elm"):
            self.put(k, np.ascontiguousarray(np.asarray(rc[k], np.float64).reshape(-1, 2), np.float64))
        self.up("par", np.ascontiguousarray(np.asarray(rc["par"]).astype(np.uint8)))
        self.I.update(nk=0, nr=0, npn=0, nrn=0, R=0, npt=0, nc=0)
        for n in PTRS:                         # tables not built yet read as null by the timer-only call
            self.p.setdefault(n, None)
        self._tables()

    def sens(self, tau):
        w1, w2, nend = self.w12
        Dv = self.Dv.copy(); Dv[2] = w1; Dv[3] = w2 / nend; Dv[4] = tau
        val = np.zeros(8)
        rc = self.L.dg_sens(self.t_dg.ctypes.data, self.t_I.ctypes.data, Dv.ctypes.data, self.t_gt.ctypes.data, self.gz.ctypes.data, self.gz_n, val.ctypes.data)
        if rc != 0:
            raise RuntimeError("dg_sens")
        return dict(zip("of pw wns tns total wns_raw tns_raw".split(), val[:7].tolist()))

    # ---------------------------------------------------------------- RC trees (t: topo_np) of the nets with free runs; sr: wire segment (t["wire"] order) -> run or -1. base: dict c1 c2 rpi ctot elm par for the nets outside t (None: t covers the design)
    def rc(self, t, sr, base=None):
        cd = self.cd
        lrc = np.asarray(cd.layer_rc, np.float64).reshape(-1, 2); via_r = np.asarray(cd.via_r, np.float64)
        pin_cap = np.asarray(cd.pin_cap, np.float64).reshape(-1, 2)
        pin_is_drv = np.asarray(cd.pin_is_drv).astype(bool)
        cnd = self._cand()
        self.up("Lr", np.where(cnd >= 0, lrc[np.maximum(cnd, 0), 0], 0.0)); self.up("Lc", np.where(cnd >= 0, lrc[np.maximum(cnd, 0), 1], 0.0))
        ns, nk = t["ns"], t["nk"]; wire = t["wire"]
        gl = t["s_glen"][wire]; li = t["s_li"][wire]
        R0, C0 = lrc[li, 0] * gl, lrc[li, 1] * gl
        sr = np.asarray(sr, np.int64); m = sr >= 0
        R = np.zeros(ns); C = np.zeros(ns)
        v = t["s_via"]
        R[v] = via_r[t["s_vi"][v]]
        R[wire] = R0
        C[wire] = np.where(m, 0.0, C0)
        pos = np.full(ns, -1, np.int64); pos[wire] = np.arange(len(wire))
        ncap = np.bincount(t["s_a"], C / 2, nk) + np.bincount(t["s_b"], C / 2, nk)
        cap = np.stack([ncap, ncap], 1)
        pn = t["attach"]; pin = t["n_pin"][pn]; sc = t["n_sc"][pn]
        cap[pn] = pin_cap[pin] + (sc / 2)[:, None]
        ad_ = np.bincount(t["n_pg"][pn], sc / 2, nk)
        cap[:, 0] += ad_; cap[:, 1] += ad_
        Rp = np.zeros(nk)
        m1 = np.nonzero(t["n_rk"] == 1)[0]; m2 = np.nonzero(t["n_rk"] == 2)[0]
        Rp[m1] = R[t["n_ri"][m1]]
        Rp[m2] = t["n_grv"][t["n_ri"][m2]] + t["n_sr"][t["n_ri"][m2]]
        rp_run = np.full(nk, -1, np.int32); rp_gl = np.zeros(nk)
        ps = pos[t["n_ri"][m1]]
        ok = ps >= 0
        ok[ok] = m[ps[ok]]
        rp_run[m1[ok]] = sr[ps[ok]]; rp_gl[m1[ok]] = gl[ps[ok]]
        reach = t["n_reach"]
        cap0 = np.zeros((nk, 2)); cap0[reach] = cap[reach]
        att = np.zeros(nk, bool); att[pn] = True
        wm = np.nonzero(m)[0]; sw = wire[wm]
        n_ab = np.concatenate([t["s_a"][sw], t["s_b"][sw]]); r_ab = np.concatenate([sr[wm], sr[wm]]); g_ab = np.concatenate([gl[wm], gl[wm]]) / 2
        keep = reach[n_ab] & ~att[n_ab]
        n_ab, r_ab, g_ab = n_ab[keep], r_ab[keep], g_ab[keep]
        nsp, nsr, nsg = csr(n_ab, nk, r_ab.astype(np.int32), g_ab)
        Rn = self.I["R"]
        rsp, rsn, rsg = csr(r_ab, Rn, n_ab.astype(np.int32), g_ab)
        wr = np.nonzero(rp_run >= 0)[0]
        rrp, rrn, rrg = csr(rp_run[wr], Rn, wr.astype(np.int32), rp_gl[wr])
        self.up("Rp0", Rp); self.up("rp_run", rp_run); self.up("rp_gl", rp_gl); self.up("cap0", cap0.astype(np.float64)); self.up("reach", reach.astype(np.uint8))
        self.up("nsp", nsp); self.up("nsr", nsr); self.up("nsg", nsg); self.up("rsp", rsp); self.up("rsn", rsn); self.up("rsg", rsg); self.up("rrp", rrp); self.up("rrn", rrn); self.up("rrg", rrg)
        for n in ("Rp", "gR"):
            self.zeros(n, nk * 8)
        for n in ("y1", "y2", "y3", "elmn", "gy1", "gy2", "gy3", "gelmn"):
            self.zeros(n, nk * 16)
        pn_ = np.nonzero(t["n_reach"] & (t["n_par"] >= 0))[0]
        o_ = np.lexsort((t["n_dep"][pn_], t["n_net"][pn_]))
        rno = pn_[o_]; nt_ = t["n_net"][rno]
        st_ = np.nonzero(np.r_[True, nt_[1:] != nt_[:-1]])[0] if len(rno) else np.zeros(0, np.int64)
        self.up("rno", rno.astype(np.int32)); self.up("rnp", np.r_[st_, len(rno)].astype(np.int32)); self.up("par_n", t["n_par"].astype(np.int64))
        self.I["nrn"] = len(st_)
        rn = t["roots"]; pnod = t["pin_nodes"]; pp = t["n_pin"][pnod]; kp = ~pin_is_drv[pp]
        self.up("roots", rn.astype(np.int32)); self.up("rootn", t["root"][rn].astype(np.int32))
        self.up("pn_node", pnod[kp].astype(np.int32)); self.up("pn_pin", pp[kp].astype(np.int32))
        self.I.update(nk=nk, nr=len(rn), npn=int(kp.sum()))
        P_, nn = self.tm["P"], self.tm["nn"]
        if base is None:
            par = t["hasd"] & t["has_seg"]
            b = dict(c1=np.zeros((nn, 2)), c2=np.zeros((nn, 2)), rpi=np.zeros((nn, 2)), ctot=self.lumped.copy(), elm=np.full((P_, 2), np.nan))
        else:
            par = np.asarray(base["par"]).astype(bool)
            b = {k: np.asarray(base[k], np.float64).reshape(-1, 2) for k in ("c1", "c2", "rpi", "ctot", "elm")}
        for k, a in b.items():
            self.put(k, np.ascontiguousarray(a, np.float64))
        self.up("par", par.astype(np.uint8))
        self._tables()

    def _cand(self):
        return self.get("Cand", np.int32, (2, -1))

    def _tables(self):
        p = self.p
        self.t_dg = np.zeros(len(PTRS), np.int64)
        for i, n in enumerate(PTRS):
            self.t_dg[i] = p[n] or 0
        self.t_I = np.array([self.I[k] for k in IDX], np.int64)
        gt = np.zeros(len(TBL), np.int64)
        for i, n in enumerate(TBL):
            if n in self.host:
                gt[i] = self.host[n].ctypes.data
            elif n in self.tm:
                gt[i] = int(self.tm[n])
            elif p.get(n) is not None:
                gt[i] = p[n]
        self.t_gt = gt
        self.t_rc = np.zeros(1, np.int64)
        z = [n for n in ("gAT", "gSL", "gREC", "gc1", "gc2", "grpi", "gctot", "gelm", "gR", "gy1", "gy2", "gy3", "gelmn") if p.get(n) is not None]
        self.gz = np.array([[p[n], self.size[n]] for n in z], np.int64)
        self.gz_n = len(z)

    # ---------------------------------------------------------------- logits
    def start(self, init=4.0, cur=None):
        cur = self.cur if cur is None else cur
        R, K = self.I["R"], self.I["K"]
        x = np.zeros((R, K), self.RL)
        x[np.arange(R), cur] = init
        x *= self.mask
        self.put("x", x)
        self.L.dg_zero(self.p["am"], self.size["am"]); self.L.dg_zero(self.p["av"], self.size["av"])

    def run(self, K, lr=0.1, t0=1.0, t1=0.05, tau=0.002, stop=0.0, patience=5, prof=False, timer=True, dl=0.0, dat=0.0, reuse=False, time_limit=0.0, lrK=0.0, lrC=0.0):
        """K steps; reuse: the timer arrays hold a full pass of the base RC already (dg_sens); time_limit > 0: after two steps the number of steps is set so that K x step seconds fits (schedule and lr follow: lr = max(lr, lrK / K, lrC / K^2))"""
        w1, w2, nend = self.w12
        Dv = self.Dv.copy()
        Dv[2] = w1; Dv[3] = w2 / nend; Dv[4] = tau; Dv[5] = t0; Dv[6] = t1; Dv[7] = lr; Dv[8] = stop; Dv[9] = patience; Dv[10] = dl; Dv[11] = dat; Dv[12] = 1.0 if reuse else 0.0; Dv[13] = time_limit; Dv[14] = lrK; Dv[15] = lrC
        vals = np.zeros((K, 8)); it = np.zeros(1, np.int64); pr = np.zeros(18)
        rc = self.L.dg_run(self.t_dg.ctypes.data, self.t_I.ctypes.data, Dv.ctypes.data, self.t_gt.ctypes.data if timer else None, self.t_rc.ctypes.data, self.gz.ctypes.data, self.gz_n, K, vals.ctypes.data, it.ctypes.data, pr.ctypes.data, 1 if prof else 0)
        if rc != 0:
            raise RuntimeError("dg_run")
        names = "soft pt cell via node rcf root tf ep ep_b tb root_b rcb step host tf_arc tf_grp tf_edge".split()
        return {"steps": int(it[0]), "vals": vals[:int(it[0])], "prof": dict(zip(names, pr.tolist()))}

    def eval(self, T=1.0, mode=1, tau=0.002, grad=False, timer=True):
        w1, w2, nend = self.w12
        Dv = self.Dv.copy(); Dv[2] = w1; Dv[3] = w2 / nend; Dv[4] = tau
        val = np.zeros(8)
        g = None
        if grad:
            if "g_out" not in self.p:
                self.zeros("g_out", self.size["x"])
            g = self.p["g_out"]
        rc = self.L.dg_eval(self.t_dg.ctypes.data, self.t_I.ctypes.data, Dv.ctypes.data, self.t_gt.ctypes.data if timer else None, self.t_rc.ctypes.data, self.gz.ctypes.data, self.gz_n, float(T), mode, g, val.ctypes.data)
        if rc != 0:
            raise RuntimeError("dg_eval")
        out = dict(zip("of pw wns tns total wns_raw tns_raw".split(), val[:7].tolist()))
        if grad:
            out["g"] = self.get("g_out", self.RL, (self.I["R"], self.I["K"]))
        return out

    def choice(self, T_end=0.05):
        x = self.get("x", self.RL, (self.I["R"], self.I["K"]))
        x = np.where(self.mask, x, -np.inf)
        return np.argmax(x, 1)

    def memory_mb(self):
        return sum(self.size.values()) / 1e6
