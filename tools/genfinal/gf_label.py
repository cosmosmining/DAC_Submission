#!/usr/bin/env python3
"""gf_label.py (genfinal 2026-11-12, NEW): target of the generation model = the layer each run of DTLA round 1 has in the FINAL route of
a finished flow (after DTLA, the critical-path and the power rounds), from a dump of tools/genfinal/gf_hook.py.
  gf_label.py <dump.npz> <final .route> <out.npz> [--ref_json <the final run's .json> --dump_json <the dump run's .json>]
A run (k d fixed lo hi cur ..., GCells, tools/best7/dgb.cc) has candidate k on layer cand2[d,k] iff the union of the net's wires on that
layer at that track covers [lo, hi] (the flow's own rule for the current layer). First the rule is checked on the dump's current rows
(must give run[:,5] on >= 99 % of the runs), then applied to the final route (DBU -> GCell = // GRID, as tools/gapla/csta.seg_from_route).
Label y[r] = first covering candidate (as dgb.cc), -1 when none covers (route changed in 2-D, e.g. the overflow stage) -> ignored.
Out npz: y [R] int8, cur_ok [R] bool (the rule on the current rows), stats json; prints the stats."""
import argparse, json, os, sys
import numpy as np
T = "/path/to/workdir"
sys.path.insert(0, os.environ.get("ROUTE_CPP_DIR", T + "/repo/tools/fast_io/route_cpp"))


def wires(seg, nidx):
    """seg [m,6] GCell rows x0 y0 z0 x1 y1 z1, nidx [m] kept-net index -> (key, lo, hi) of the straight wires"""
    x0, y0, z0, x1, y1, z1 = (seg[:, i].astype(np.int64) for i in range(6))
    ok = (z0 == z1) & (nidx >= 0) & ((x0 != x1) ^ (y0 != y1))
    hor = ok & (y0 == y1); ver = ok & (x0 == x1)
    d = np.where(hor, 0, 1); fixed = np.where(hor, y0, x0)
    lo = np.where(hor, np.minimum(x0, x1), np.minimum(y0, y1)); hi = np.where(hor, np.maximum(x0, x1), np.maximum(y0, y1))
    m = hor | ver
    return key_of(nidx[m], z0[m], fixed[m]), lo[m], hi[m]


def key_of(n, z, fixed):
    return (n.astype(np.int64) * 64 + z.astype(np.int64)) * 65536 + (fixed.astype(np.int64) + 1024)


def merged(key, lo, hi):
    """disjoint merged intervals per key (touching or overlapping merge), sorted by (key, lo)"""
    if len(key) == 0:
        return key, lo, hi
    o = np.lexsort((lo, key)); key, lo, hi = key[o], lo[o], hi[o]
    # running max of hi within the key, new block where lo > running max or key changes
    newk = np.ones(len(key), bool); newk[1:] = key[1:] != key[:-1]
    out_k, out_l, out_h = [], [], []
    # vectorised: segment-wise cumulative max of hi
    gid = np.cumsum(newk) - 1
    hi_c = hi.copy()
    # cumulative max per group: use maximum.accumulate on hi offset by group*BIG
    BIG = np.int64(1 << 20)
    hm = np.maximum.accumulate(hi_c + gid * BIG) - gid * BIG
    prev = np.empty(len(key), np.int64); prev[0] = -1; prev[1:] = hm[:-1]
    start = newk | (lo > prev)
    bid = np.cumsum(start) - 1
    nb = int(bid[-1]) + 1
    bk = key[start]; bl = lo[start]
    bh = np.full(nb, -1, np.int64); np.maximum.at(bh, bid, hi)
    return bk, bl, bh


def covers(bk, bl, bh, qk, qlo, qhi):
    """True where some merged interval of key qk contains [qlo, qhi]"""
    if len(bk) == 0:
        return np.zeros(len(qk), bool)
    comb = bk * np.int64(1 << 17) + (bl + 1024)
    q = qk * np.int64(1 << 17) + (qlo + 1024)
    i = np.searchsorted(comb, q, side="right") - 1
    ok = i >= 0
    ii = np.maximum(i, 0)
    return ok & (bk[ii] == qk) & (bl[ii] <= qlo) & (bh[ii] >= qhi)


def label(run, cand2, bk, bl, bh):
    R = len(run); K = cand2.shape[1]
    d = run[:, 1]; fixed = run[:, 2]; lo = run[:, 3]; hi = run[:, 4]; n = run[:, 0]
    cov = np.zeros((R, K), bool)
    for k in range(K):
        z = cand2[d, k]
        v = z >= 0
        c = np.zeros(R, bool)
        c[v] = covers(bk, bl, bh, key_of(n[v], z[v], fixed[v]), lo[v], hi[v])
        cov[:, k] = c
    any_ = cov.any(1)
    y = np.where(any_, cov.argmax(1), -1).astype(np.int8)
    return y, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("route"); ap.add_argument("out")
    ap.add_argument("--ref_json", default=""); ap.add_argument("--dump_json", default=""); ap.add_argument("--require_same", type=int, default=1)
    a = ap.parse_args()
    Z = np.load(a.dump)
    meta = json.loads(Z["meta"].tobytes().decode())
    run = Z["run"].astype(np.int64); cand2 = Z["cand2"].astype(np.int64); R = len(run)
    names = Z["names"].tobytes().decode().split("\n")
    GRID = int(meta["GRID"])
    st = {"dump": a.dump, "route": a.route, "design": meta.get("design"), "suite": meta.get("suite"), "R": R, "nets": len(names), "GRID": GRID,
          "dgr_nets": meta.get("dgr_nets")}
    # 1. the rule on the current rows
    rows = Z["rows"].astype(np.int64); roff = Z["roff"].astype(np.int64)
    nid = np.repeat(np.arange(len(roff) - 1), np.diff(roff))
    bk, bl, bh = merged(*wires(rows, nid))
    yc, covc = label(run, cand2, bk, bl, bh)
    cur = run[:, 5]
    cur_ok = covc[np.arange(R), cur]
    st["cur_rule_covers_cur"] = round(float(cur_ok.mean()), 5)
    st["cur_rule_first_eq_cur"] = round(float((yc == cur).mean()), 5)
    # 2. the final route
    import route_cpp as RC
    rt = RC.read_route(a.route)
    idx = {nm: i for i, nm in enumerate(names)}
    rn = np.fromiter((idx.get(nm, -1) for nm in rt.net_names), np.int64, len(rt.net_names))
    st["final_nets_matched"] = int((rn >= 0).sum())
    cnt = np.diff(np.asarray(rt.offsets, np.int64))
    nidx = np.repeat(rn, cnt)
    sel = nidx >= 0
    seg = np.stack([np.asarray(rt.x0, np.int64)[sel] // GRID, np.asarray(rt.y0, np.int64)[sel] // GRID, np.asarray(rt.z0, np.int64)[sel],
                    np.asarray(rt.x1, np.int64)[sel] // GRID, np.asarray(rt.y1, np.int64)[sel] // GRID, np.asarray(rt.z1, np.int64)[sel]], 1)
    bk, bl, bh = merged(*wires(seg, nidx[sel]))
    y, cov = label(run, cand2, bk, bl, bh)
    lab = y >= 0
    st["labeled"] = round(float(lab.mean()), 5)
    st["multi_cover"] = round(float((cov.sum(1) > 1).mean()), 5)
    st["final_eq_cur"] = round(float((y[lab] == cur[lab]).mean()), 5) if lab.any() else None
    st["changed"] = int((lab & (y != cur)).sum())
    zr = Z["zref"].astype(np.float64); sid = Z["sid"]; mask = Z["mask"]
    o = np.zeros(mask.shape); o[mask] = 6.0 * np.tanh(zr / 6.0)
    for beta in (0.75, 1.0, 2.0, 4.0):
        x = 4.0 * (np.arange(mask.shape[1])[None, :] == cur[:, None]) + beta * o
        p = np.where(mask, x, -1e9).argmax(1)
        ch_p = p != cur; ch_t = lab & (y != cur)
        st["ref_beta%g" % beta] = {"acc": round(float((p[lab] == y[lab]).mean()), 5), "flips": int(ch_p.sum()),
                                   "prec": round(float((ch_p & lab & (p == y)).sum() / max(int((ch_p & lab).sum()), 1)), 4),
                                   "recall": round(float((ch_t & (p == y)).sum() / max(int(ch_t.sum()), 1)), 4)}
    for k, j in (("ref", a.ref_json), ("dump", a.dump_json)):
        if j and os.path.exists(j):
            try:
                L = json.load(open(j)).get("log", [])
                sp = [e for e in L if e.get("stage") == "single_pass"]
                est = sp[0].get("est") if sp else None
                st[k + "_single_pass_est"] = est.get("score") if isinstance(est, dict) else est
            except Exception as e:
                st[k + "_single_pass_est"] = repr(e)[:80]
    if isinstance(st.get("ref_single_pass_est"), float) and isinstance(st.get("dump_single_pass_est"), float):
        st["same_first_pass"] = abs(st["ref_single_pass_est"] - st["dump_single_pass_est"]) < 1e-6
    print(json.dumps(st), flush=True)
    if a.require_same and st.get("same_first_pass") is not True:
        sys.exit(5)                                   # another final run is tried by the driver
    if st["cur_rule_covers_cur"] < 0.99 or st["labeled"] < 0.5:
        sys.exit(6)                                   # geometry rule or route mismatch: no labels
    np.savez(a.out, y=y, cur_ok=cur_ok, stats=np.frombuffer(json.dumps(st).encode(), np.uint8))


if __name__ == "__main__":
    main()
