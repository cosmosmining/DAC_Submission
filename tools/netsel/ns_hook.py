#!/usr/bin/env python3
"""ns_hook.py: NET SELECTION for the later rounds of the DTLA stage (tools/best8/dgr_stage_r.py).
Installed by ns_launch.py before the driver runs (env NS_MODE). It patches the stage module in place (the module object the chain
imports), nothing else of the flow:
  prepare        copy of dgr_stage_r.prepare; only the selection differs. The flow's set (top N of the stage's score, as before) is
                 the candidate pool; from round NS_FROM (default 1) on, mode rule / learned frees a subset of it.
  sensitivities  wrapped: keeps the round's context C;
  round_rows     wrapped: before the rounding, per free net the decisiveness of the descent's end logits (next round's features);
  run            wrapped: the chain's gate is wrapped (every tested change and whether it was kept); after the round: labels, history
                 (moved nets, changed-demand map), log.
  end            wrapped: NS_EXIT=1 ends the process after the stage (log runs: no route written).
Modes (env NS_MODE):
  log      selection unchanged (= the flow's); features + labels of every candidate net of every round written (training data);
  rule     round >= NS_FROM: free = nets moved (applied and kept) in the previous round + the top NS_RULE_K of the flow's ranking
           (fraction of the candidate set if < 1, else a count; default 0.05);
  learned  round >= NS_FROM: free = the top NS_FRAC (default 0.15) of the candidate set by the model's score (NS_MODEL: .npz of
           ns_train.py, MLP on FEATS) + the nets moved in the previous round (NS_FORCE_PREV=1, default).
Labels of a free net: kept (its problem was built), prop (the descent changed a run), app1 (rebuilt and in the first tested change),
app (in the change the gate kept = "moved in this round").
Logs: NS_LOG prefix -> <prefix>.jsonl (one line per round) and, in log mode or with NS_LOG_NETS=1, <prefix>.r<k>.npz (per candidate
net: route net index, features, free, labels; -1 = not free)."""
import json, os, sys, time
import numpy as np

NS = {"on": False, "mode": "", "round": 0, "D": None, "H": None, "C": None, "cap": None, "applies": [], "F": None, "o": None,
      "model": None, "sel": {}}
FEATS = ("rank", "q_pt", "q_po", "q_pp", "l_pt", "l_po", "l_pp", "l_rows", "l_wl", "l_via", "zmean", "zmax", "zmin",
         "p_app", "p_prop", "p_free", "l_hist", "gap", "marg", "age", "l_runs", "dpt", "drank", "l_nb", "nb_rel", "p_frac", "rnd")
BIG = 10.0


def env_f(k, d):
    try:
        return float(os.environ.get(k, d))
    except ValueError:
        return float(d)


def log_line(rec):
    p = os.environ.get("NS_LOG", "")
    if p:
        with open(p + ".jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


def new_hist(n):
    return {"n": n, "prev_app": np.zeros(n, bool), "prev_prop": np.zeros(n, bool), "prev_free": np.zeros(n, bool),
            "cnt": np.zeros(n, np.int16), "gap": np.full(n, BIG, np.float32), "marg": np.full(n, BIG, np.float32),
            "last": np.full(n, -100, np.int32), "runs": np.zeros(n, np.int32), "pt": None, "rank": None, "p_frac": 1.0, "M": None}


def prank(v):
    v = np.asarray(v)
    r = np.empty(len(v), np.float32)
    r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=np.float32) / max(len(v) - 1, 1)
    return r


def lmed(v):
    v = np.asarray(v, np.float64)
    pos = v[v > 0]
    m = float(np.median(pos)) if len(pos) else 1.0
    return np.log10(np.maximum(v, m * 1e-6) / m).astype(np.float32)


def chg_map(r7, Xg, Yg):
    """2-D count map of the GCells covered by the rows r7 [m, 7] (net, x0 y0 z0 x1 y1 z1)"""
    M = np.zeros((Xg, Yg), np.float32)
    if r7 is None or len(r7) == 0:
        return M
    x0, y0, z0, x1, y1, z1 = (np.asarray(r7[:, k], np.int64) for k in range(1, 7))
    h = x0 != x1; v = (~h) & (y0 != y1); p = ~(h | v)
    a, b = np.minimum(x0[h], x1[h]), np.maximum(x0[h], x1[h]); yh = y0[h]
    Dx = (np.bincount(a * Yg + yh, minlength=(Xg + 1) * Yg) - np.bincount((b + 1) * Yg + yh, minlength=(Xg + 1) * Yg)).reshape(Xg + 1, Yg)
    a2, b2 = np.minimum(y0[v], y1[v]), np.maximum(y0[v], y1[v]); xv = x0[v]
    Dy = (np.bincount(xv * (Yg + 1) + a2, minlength=Xg * (Yg + 1)) - np.bincount(xv * (Yg + 1) + b2 + 1, minlength=Xg * (Yg + 1))).reshape(Xg, Yg + 1)
    M += np.bincount(x0[p] * Yg + y0[p], minlength=Xg * Yg).reshape(Xg, Yg).astype(np.float32)
    M += (np.cumsum(Dx, 0)[:Xg] + np.cumsum(Dy, 1)[:, :Yg]).astype(np.float32)
    return M


def row_sums(M, rw):
    """per row [m, 6]: sum of M over the row's GCells (2-D)"""
    Xg, Yg = M.shape
    Cx = np.zeros((Xg + 1, Yg), np.float64); Cx[1:] = np.cumsum(M, 0)
    Cy = np.zeros((Xg, Yg + 1), np.float64); Cy[:, 1:] = np.cumsum(M, 1)
    x0, y0, x1, y1 = rw[:, 0], rw[:, 1], rw[:, 3], rw[:, 4]
    h = x0 != x1; v = (~h) & (y0 != y1); p = ~(h | v)
    out = np.zeros(len(rw), np.float64)
    a, b = np.minimum(x0[h], x1[h]), np.maximum(x0[h], x1[h])
    out[h] = Cx[b + 1, y0[h]] - Cx[a, y0[h]]
    a2, b2 = np.minimum(y0[v], y1[v]), np.maximum(y0[v], y1[v])
    out[v] = Cy[x0[v], b2 + 1] - Cy[x0[v], a2]
    out[p] = M[x0[p], y0[p]]
    return out


def features(C, S, o):
    """per candidate net (o = the flow's ranked set): FEATS, from what is known at selection time"""
    D = NS["D"]; H = NS["H"]; N = len(o); r = NS["round"]
    F = {}
    F["rank"] = np.arange(N, dtype=np.float32) / max(N - 1, 1)
    pt, po, pp = C.pot_t[o], C.pot_o[o], C.pot_p[o]
    F["q_pt"], F["q_po"], F["q_pp"] = prank(pt), prank(po), prank(pp)
    F["l_pt"], F["l_po"], F["l_pp"] = lmed(pt), lmed(po), lmed(pp)
    nrows = C.nrows[o]
    F["l_rows"] = np.log1p(nrows).astype(np.float32)
    rw, ow = D.gather(C.rows, C.off, o)
    seg = np.repeat(np.arange(N), np.diff(ow))
    x0, y0, z0, x1, y1, z1 = (rw[:, k] for k in range(6))
    via = z0 != z1
    ln = (np.abs(x1 - x0) + np.abs(y1 - y0)).astype(np.float64)
    wl = np.bincount(seg, weights=np.where(via, 0.0, ln), minlength=N)
    nv = np.bincount(seg, weights=via.astype(np.float64), minlength=N)
    zs = np.bincount(seg, weights=np.where(via, 0.0, ln * z0), minlength=N)
    F["l_wl"] = np.log1p(wl).astype(np.float32); F["l_via"] = np.log1p(nv).astype(np.float32)
    zw = np.where(wl > 0, zs / np.maximum(wl, 1e-9), 0.0)
    st = ow[:-1]
    ok = np.diff(ow) > 0
    zmx = np.zeros(N); zmn = np.zeros(N)
    if ok.any():
        zhi = np.maximum(z0, z1); zlo = np.minimum(z0, z1)
        zmx[ok] = np.maximum.reduceat(zhi, st[ok]); zmn[ok] = np.minimum.reduceat(zlo, st[ok])
    F["zmean"] = zw.astype(np.float32); F["zmax"] = zmx.astype(np.float32); F["zmin"] = zmn.astype(np.float32)
    if H["M"] is not None and len(rw):
        nb = np.bincount(seg, weights=row_sums(H["M"], rw), minlength=N)
    else:
        nb = np.zeros(N)
    F["l_nb"] = np.log1p(nb).astype(np.float32); F["nb_rel"] = (nb / (wl + nv + 1.0)).astype(np.float32)
    F["p_app"] = H["prev_app"][o].astype(np.float32); F["p_prop"] = H["prev_prop"][o].astype(np.float32); F["p_free"] = H["prev_free"][o].astype(np.float32)
    F["l_hist"] = np.log1p(H["cnt"][o].astype(np.float32))
    F["gap"] = np.clip(H["gap"][o], -BIG, BIG).astype(np.float32); F["marg"] = np.clip(H["marg"][o], -BIG, BIG).astype(np.float32)
    F["age"] = np.clip(r - H["last"][o], 0, 10).astype(np.float32)
    F["l_runs"] = np.log1p(H["runs"][o]).astype(np.float32)
    if H["pt"] is not None:
        p0 = H["pt"][o]; e = max(float(np.median(pt[pt > 0])) if (pt > 0).any() else 1.0, 1e-30) * 1e-3
        F["dpt"] = np.log10((pt + e) / (p0 + e)).astype(np.float32)
    else:
        F["dpt"] = np.zeros(N, np.float32)
    if H["rank"] is not None:
        F["drank"] = ((H["rank"][o] - np.arange(N)) / max(N, 1)).astype(np.float32)
    else:
        F["drank"] = np.zeros(N, np.float32)
    F["p_frac"] = np.full(N, H["p_frac"], np.float32); F["rnd"] = np.full(N, min(r, 8) / 8.0, np.float32)
    return F


def load_model(p):
    z = np.load(p, allow_pickle=False)
    m = {k: z[k] for k in z.files}
    m["feats"] = [str(s) for s in m["feats"]]
    return m


def model_score(F):
    M = NS["model"]
    X = np.stack([np.asarray(F[k], np.float32) for k in M["feats"]], 1)
    X = (X - M["mu"]) / M["sd"]
    h = np.maximum(X @ M["W1"] + M["b1"], 0.0)
    if "W2" in M:
        h = np.maximum(h @ M["W2"] + M["b2"], 0.0)
    return (h @ M["w3"] + M["b3"]).reshape(-1)


def select(C, S, a, o):
    """-> idx (sorted route nets to free) from the flow's ranked set o"""
    t0 = time.perf_counter()
    mode, r = NS["mode"], NS["round"]
    H = NS["H"]; n = len(C.nrows)
    if H is None or H["n"] != n:
        H = NS["H"] = new_hist(n)
    active = mode in ("rule", "learned") and r >= int(env_f("NS_FROM", 1)) and H["pt"] is not None
    need_f = mode == "log" or os.environ.get("NS_LOG_NETS") == "1" or (active and mode == "learned")
    F = features(C, S, o) if need_f else None
    tf = time.perf_counter() - t0
    NS["F"], NS["o"] = F, o
    info = {"n_cand": int(len(o)), "feat_s": round(tf, 4)}
    if not active:
        sel = o; info["sel"] = "flow"
    else:
        prev = o[H["prev_app"][o]]
        if mode == "rule":
            k = env_f("NS_RULE_K", 0.05)
            kn = int(round(k * len(o))) if k < 1 else int(k)
            sel = np.union1d(o[:kn], prev); info.update(sel="rule", k=kn, prev=int(len(prev)))
        else:
            if NS["model"] is None:
                NS["model"] = load_model(os.environ["NS_MODEL"])
            z = model_score(F)
            kn = int(round(env_f("NS_FRAC", 0.15) * len(o)))
            top = o[np.argsort(-z, kind="stable")[:kn]]
            sel = np.union1d(top, prev) if os.environ.get("NS_FORCE_PREV", "1") == "1" else top
            info.update(sel="learned", k=kn, prev=int(len(prev)))
    idx = np.sort(np.asarray(sel, np.int64))
    info["n_free"] = int(len(idx)); info["sel_s"] = round(time.perf_counter() - t0, 4)
    NS["sel"] = info
    return idx


def make_prepare(D):
    DB, GN = D.DB, D.GN

    def prepare(C, S, a, cd, FT, HD, W, nets_n):
        """= dgr_stage_r.prepare; the selection of idx goes through select() (identical in log mode)"""
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
        idx = select(C, S, a, o)          # NETSEL (dgr_stage_r: idx = np.sort(o))
        t = lap("select_s", t)
        C.Jn, C.pa0, C.pa1, _n = D._names(S, S.nets.names)
        t = lap("names_s", t)
        rw, ow = D.gather(C.rows, C.off, idx)
        nr, no = D.normalise(S, rw, ow, idx, C.pa0, C.pa1, T_)
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
        rk_, ok_ = D.gather(C.rows, C.off, i_k)
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
        rct = D.topo_tables(cd)
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
        if D.ST.get("crc") is None:
            lrc = np.ascontiguousarray(np.asarray(cd.layer_rc, np.float64).reshape(-1));
            D.ST["crc"] = dict(layer_rc=lrc, via_r=np.ascontiguousarray(np.asarray(cd.via_r, np.float64)), pin_cap=np.ascontiguousarray(np.asarray(cd.pin_cap, np.float64).reshape(-1)), pin_is_drv=np.ascontiguousarray(np.asarray(cd.pin_is_drv).astype(np.uint8)))
        crc = D.ST["crc"]
        TB = DB.rc_tables(tn, sr, len(run_), crc)
        lr2 = crc["layer_rc"].reshape(-1, 2)
        Lrc = (np.where(cand2 >= 0, lr2[np.maximum(cand2, 0), 0], 0.0), np.where(cand2 >= 0, lr2[np.maximum(cand2, 0), 1], 0.0))
        t = lap("rc_a_s", t)
        X.rc_arrays(TB, tn["nk"], Lrc)
        t = lap("rc_s", t)
        C.nk, C.hit, C.nw = tn["nk"], hit, len(sr)
        return True
    return prepare


def capture(C):
    """per kept net: min over its runs of (logit of the current candidate - best other) and of the argmax's margin; runs per net"""
    X = C.X
    R, K = int(X.I["R"]), int(X.I["K"])
    if R == 0 or K < 2:
        return
    x = X.get("x", X.RL, (R, K)).astype(np.float32)
    m = np.asarray(X.mask, bool)
    xm = np.where(m, x, -np.inf)
    cur = np.asarray(C.run_[:, 5], np.int64); rr = np.arange(R)
    xc = xm[rr, cur]
    xo = xm.copy(); xo[rr, cur] = -np.inf
    gap = xc - xo.max(1)
    t2 = np.partition(xm, K - 2, axis=1)[:, K - 2:]
    marg = t2[:, 1] - t2[:, 0]
    gap = np.where(np.isfinite(gap), gap, BIG).astype(np.float32); marg = np.where(np.isfinite(marg), marg, BIG).astype(np.float32)
    kn = np.asarray(C.kn, np.int64); nk = len(C.i_k)
    g = np.full(nk, BIG, np.float32); mg = np.full(nk, BIG, np.float32)
    nr = np.bincount(kn, minlength=nk).astype(np.int32)
    if R > 1 and np.all(kn[1:] >= kn[:-1]):             # runs grouped by net (as dgb_nets builds them): one reduceat
        st = np.flatnonzero(np.r_[True, kn[1:] != kn[:-1]])
        g[kn[st]] = np.minimum.reduceat(gap, st); mg[kn[st]] = np.minimum.reduceat(marg, st)
    else:
        np.minimum.at(g, kn, gap); np.minimum.at(mg, kn, marg)
    NS["cap"] = (np.asarray(C.i_k, np.int64).copy(), g, mg, nr)


def post(C, res, S, t_run):
    """labels of the round, history for the next one, log"""
    t0 = time.perf_counter()
    H = NS["H"]; r = NS["round"]; o = NS["o"]
    k_, c_, rec = res
    n = H["n"] if H is not None else 0
    E = np.zeros(0, np.int64)
    idx = getattr(C, "idx", None) if C is not None else None
    have = H is not None and idx is not None and o is not None
    app1 = NS["applies"][0][0] if NS["applies"] else E
    app = next((i for (i, kp) in NS["applies"] if kp), E)
    undo_k = next((u for (u, kp) in NS["applies_u"] if kp), None)
    prop = E
    if have and getattr(C, "mv", None) is not None and getattr(C, "kn", None) is not None and len(C.kn) == len(C.mv):
        prop = np.asarray(C.i_k, np.int64)[np.unique(np.asarray(C.kn)[np.asarray(C.mv, bool)])]
    out = {"ns": 1, "mode": NS["mode"], "round": r, "design": getattr(S, "design", None), "suite": getattr(S, "suite", None),
           "kept": bool(k_), "gain": float(rec.get("gain", 0.0)), "nets_free": rec.get("nets_free"), "nets_moved": rec.get("nets_moved"),
           "runs": rec.get("runs"), "runs_moved": rec.get("runs_moved"), "s": rec.get("s"), "steps": rec.get("steps"), "run_s": round(t_run, 4),
           "n_prop": int(len(prop)), "n_app1": int(len(app1)), "n_app": int(len(app)), "tries": rec.get("tries"), "skipped": rec.get("skipped"),
           "walls": {k: round(v, 4) for k, v in (rec.get("walls") or {}).items()}}
    out.update(NS["sel"])
    if have:
        def mk(ids):
            b = np.zeros(n, bool); b[np.asarray(ids, np.int64)] = True
            return b
        fr, kp, pr, a1, ap = mk(idx), mk(getattr(C, "i_k", E)), mk(prop), mk(app1), mk(app)
        lab = lambda b: np.where(fr[o], b[o], -1).astype(np.int8)
        if H["pt"] is not None and len(o):     # the rule on this round (teacher forcing): prev movers + top k of the ranking
            na = int(ap[o].sum()); pv = H["prev_app"][o]
            rr = {}
            for k in (0.02, 0.05, 0.1, 0.2):
                s_ = pv.copy(); s_[:int(round(k * len(o)))] = True
                rr[str(k)] = [round(float(s_.sum()) / len(o), 4), round(float((s_ & ap[o]).sum()) / max(na, 1), 4)]
            out["rule_frac_recall"] = rr; out["prev_frac"] = round(float(pv.sum()) / len(o), 4)
        if NS["mode"] == "log" or os.environ.get("NS_LOG_NETS") == "1":
            p = os.environ.get("NS_LOG", "")
            if p and NS["F"] is not None:
                arrs = {k: v for k, v in NS["F"].items()}
                arrs.update(cand=o.astype(np.int64), free=fr[o], kept=lab(kp), prop=lab(pr), app1=lab(a1), app=lab(ap))
                np.savez_compressed("%s.r%d.npz" % (p, r), **arrs)
        # history for the next round
        H["prev_free"] = fr; H["prev_app"] = ap; H["prev_prop"] = pr
        if len(app):
            H["cnt"][app] = np.minimum(H["cnt"][app] + 1, 1000)
        if NS["cap"] is not None:
            ik, g, mg, nr = NS["cap"]
            H["gap"][ik] = g; H["marg"][ik] = mg; H["last"][ik] = r; H["runs"][ik] = nr
        H["pt"] = np.asarray(C.pot_t, np.float64).copy()
        rk = np.full(n, len(o), np.int64); rk[o] = np.arange(len(o)); H["rank"] = rk
        H["p_frac"] = float(len(app)) / max(len(o), 1)
        if undo_k is not None:
            la = S.la
            H["M"] = chg_map(np.concatenate([undo_k.r7o, undo_k.r7n]), int(la.X), int(la.Y))
        else:
            H["M"] = None
    out["ns_post_s"] = round(time.perf_counter() - t0, 4)
    log_line(out)
    print("[netsel] " + json.dumps({k: out.get(k) for k in ("mode", "round", "sel", "n_cand", "n_free", "nets_free", "n_prop", "n_app1", "n_app", "kept", "gain", "s", "feat_s", "sel_s", "ns_post_s")}, default=float), flush=True)


def install(D):
    """patch the stage module D (dgr_stage_r) in place"""
    mode = os.environ.get("NS_MODE", "")
    if mode not in ("log", "rule", "learned"):
        raise SystemExit("ns_hook: NS_MODE must be log | rule | learned, got %r" % mode)
    if mode == "learned":
        NS["model"] = load_model(os.environ["NS_MODEL"])
    NS.update(on=True, mode=mode, D=D)
    o_sens, o_rr, o_run, o_end = D.sensitivities, D.round_rows, D.run, D.end

    def sensitivities(S, cur, a, cd, W):
        C = o_sens(S, cur, a, cd, W)
        NS["C"] = C
        return C

    def round_rows(C, S, W):
        try:
            capture(C)
        except Exception as e:                   # never break the stage
            NS["cap"] = None; print("[netsel] capture error " + repr(e)[:200], flush=True)
        return o_rr(C, S, W)

    def one(S, cur, a, cd, gate, FT, HD, DLnow, deadline, write_est, test_est):
        NS["applies"] = []; NS["applies_u"] = []; NS["C"] = None; NS["cap"] = None; NS["sel"] = {}; NS["F"] = None; NS["o"] = None
        def gate2(undo, r_, tag):
            keep, c_ = gate(undo, r_, tag)
            im = np.asarray(getattr(undo, "i_m", np.zeros(0, np.int64)), np.int64).copy()
            NS["applies"].append((im, bool(keep))); NS["applies_u"].append((undo, bool(keep)))
            return keep, c_
        t0 = time.perf_counter()
        res = o_run(S, cur, a, cd, gate2, FT, HD, DLnow, deadline, write_est, test_est)
        t_run = time.perf_counter() - t0
        try:
            post(NS["C"], res, S, t_run)
        except Exception as e:
            import traceback
            print("[netsel] post error " + repr(e)[:300] + "\n" + traceback.format_exc()[-1500:], flush=True)
            NS["H"] = None                       # history unusable: the next round frees the flow's set
        NS["round"] += 1
        NS["applies_u"] = []
        return res

    def predict(rec, frac):
        """seconds of the next round if it frees the share frac of the flow's set: fixed parts + frac x the parts that scale with the
        free set + the descent unscaled (the timer pass dominates it) + frac x the check of the moves (+1 s for the selection)"""
        w = rec.get("walls") or {}
        fixed = sum(w.get(k, 0.0) for k in ("device_wait_s", "sens_s", "sens_read_s", "select_s", "names_s"))
        mv = sum(w.get(k, 0.0) for k in ("rows_a_s", "rows_b_s", "rows_c_s", "rows_d_s", "apply_dem_s", "apply_py_s", "apply_norm_s", "test_s", "split_s", "fallback_descent_s"))
        var = sum(v for k, v in w.items() if k.endswith("_s") and k not in ("device_wait_s", "sens_s", "sens_read_s", "select_s", "names_s", "descent_s") and not k.startswith(("rows_", "apply_", "test_", "split_", "fallback_")))
        return fixed + frac * var + w.get("descent_s", 0.0) + max(frac, 0.25) * mv + 1.0

    def run(S, cur, a, cd, gate, FT, HD, DLnow=None, deadline=None, write_est=0.0, test_est=0.0):
        """one round of the flow's loop; with NS_SUB=1 (rule / learned), when the flow's loop would stop on time after a kept round
        (now + this round's seconds + write > deadline) but a selected round is predicted to fit (now + predicted + write + NS_SUB_MARGIN
        <= deadline), further selected rounds run inside this call (sub-rounds), as long as they are kept, gain >= --dgr_round_min and the
        flow's round count allows; a rejected sub-round is settled as the flow's loop settles a rejected round (K.flush_pending)."""
        NS["calls"] = NS.get("calls", 0) + 1
        t_call = time.perf_counter()
        res = one(S, cur, a, cd, gate, FT, HD, DLnow, deadline, write_est, test_est)
        if os.environ.get("NS_SUB") != "1" or NS["mode"] not in ("rule", "learned") or deadline is None or DLnow is None:
            return res
        k_, c_, rec = res
        frac = env_f("NS_FRAC", 0.15) if NS["mode"] == "learned" else min(1.0, env_f("NS_RULE_K", 0.05) + float((NS.get("H") or {}).get("p_frac", 0.0) or 0.0))
        margin = env_f("NS_SUB_MARGIN", 3.0)
        subs = []; gain = float(rec.get("gain", 0.0)) if k_ else 0.0; last = rec
        while k_ and float(last.get("gain", 0.0)) >= a.dgr_round_min and NS.get("H") is not None:
            now = DLnow(); took = time.perf_counter() - t_call
            if NS["calls"] + len(subs) >= max(1, a.dgr_rounds) or len(subs) >= int(env_f("NS_SUB_MAX", 99)):
                break
            if not subs and now + took + write_est <= deadline and os.environ.get("NS_SUB_FORCE") != "1":
                break                            # the flow's loop goes on by itself (its own time test passes): no sub-round
                                                 # (NS_SUB_FORCE=1: tests of the sub-round path in untimed runs only)
            tp = predict(last, frac if NS["sel"].get("sel") == "flow" else 1.0)   # the last round freed the flow's set -> scale; else as is
            if now + tp + write_est + margin > deadline:
                subs.append({"stop": "time", "now": round(now, 2), "pred_s": round(tp, 2)})
                break
            t1 = time.perf_counter()
            k2, c2, r2 = one(S, c_, a, cd, gate, FT, HD, DLnow, deadline, write_est, test_est)
            subs.append({"round": NS["round"] - 1, "kept": bool(k2), "gain": float(r2.get("gain", 0.0)), "pred_s": round(tp, 2), "s": round(time.perf_counter() - t1, 2),
                         "nets_free": r2.get("nets_free"), "nets_moved": r2.get("nets_moved")})
            if not k2:
                try:                             # = the chain's loop after a rejected round: the kept state's required-time rows
                    fv = dict(zip(gate.__code__.co_freevars, (c.cell_contents for c in gate.__closure__ or ())))
                    H_ = fv["H"] if "H" in fv else gate.__globals__.get("H"); K_ = fv["K"] if "K" in fv else gate.__globals__.get("K")
                    subs[-1]["inc"] = bool(H_ and H_.get("inc")); subs[-1]["has_K"] = K_ is not None
                    if H_ and H_.get("inc") and r2.get("tries") and K_ is not None:
                        K_.flush_pending(); subs[-1]["flushed"] = True
                except Exception as e:
                    subs[-1]["flush_error"] = repr(e)[:200]
                break
            c_ = c2; gain += float(r2.get("gain", 0.0)); last = r2
        if subs:
            rec = dict(rec); rec["ns_sub"] = subs; rec["gain"] = gain; rec["ns_sub_s"] = round(time.perf_counter() - t_call, 3)
            log_line({"ns": 1, "sub": subs, "call": NS["calls"], "gain_total": gain})
            print("[netsel] sub-rounds " + json.dumps(subs, default=float), flush=True)
        return (k_, c_, rec)

    def end():
        o_end()
        if os.environ.get("NS_EXIT") == "1":
            log_line({"ns": 1, "exit": "after the DTLA stage", "rounds": NS["round"]})
            print("[netsel] NS_EXIT: end after the DTLA stage", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)

    D.prepare = make_prepare(D)
    D.sensitivities = sensitivities
    D.round_rows = round_rows
    D.run = run
    D.end = end
    print("[netsel] installed mode %s from %s" % (mode, os.environ.get("NS_FROM", "1")), flush=True)
