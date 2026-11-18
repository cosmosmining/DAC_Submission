#!/usr/bin/env python3
"""h_graph.py: dtla_lagnn.graph copied verbatim (dtla_lagnn.py, def graph) except the two
np.unique(X, axis=1) calls on [2, N] int64 index arrays, replaced by _uniq2: one 1-D np.unique on the packed key a*M + b
(M = max(b) + 1), which gives the same columns in the same (lexicographic) order, int64. Negative entries, an empty input or
a key that could overflow int64 fall back to np.unique(axis=1). Same tensors (torch.equal + dtype in chain_h --h_check).
"""
import numpy as np
import torch
from dtla_lagnn import _np


def _uniq2(X):
    X = np.asarray(X)
    if X.dtype != np.int64 or X.ndim != 2 or X.shape[0] != 2 or X.shape[1] == 0:
        return np.unique(X, axis=1)
    a, b = X[0], X[1]
    amin, bmin = int(a.min()), int(b.min())
    if amin < 0 or bmin < 0:
        return np.unique(X, axis=1)
    M = int(b.max()) + 1
    if (int(a.max()) + 1) * M >= 2 ** 62:
        return np.unique(X, axis=1)
    u = np.unique(a * M + b)
    out = np.empty((2, len(u)), np.int64)
    np.floor_divide(u, M, out=out[0]); np.remainder(u, M, out=out[1])
    return out


def graph(P):
    """DTLA problem -> (x_dict, edge_index_dict, slot index (R,K) of each candidate or -1)."""
    R, K, L = len(P["run_net"]), int(P["K"]), int(P["L"])
    cand = _np(P["cand"]).astype(np.int64)
    valid = cand >= 0
    sid = np.full((R, K), -1, np.int64)
    sid[valid] = np.arange(int(valid.sum()))
    rr, kk = np.nonzero(valid)
    run_net = _np(P["run_net"]).astype(np.int64)
    ln = _np(P["run_len"]).astype(np.float64)
    cur = _np(P["run_cur"]).astype(np.int64)
    dco = _np(P["dcoef"]).astype(np.float64) if "dcoef" in P else np.zeros((R, K))
    Cz = _np(P["Cz"]).astype(np.float64)
    nn_ = int(run_net.max()) + 1 if R else 0
    p_net, p_path = _np(P["p_net"]).astype(np.int64), _np(P["p_path"]).astype(np.int64)
    s0 = _np(P["slack0"]).astype(np.float64)
    npaths = np.bincount(p_net, minlength=nn_) if len(p_net) else np.zeros(nn_)
    wns = abs(s0.min()) if len(s0) else 1.0
    worst = np.zeros(nn_)
    if len(p_net):
        np.minimum.at(worst, p_net, s0[p_path] / max(wns, 1e-9))
    kpn = _np(P["kPn"]).astype(np.float64) if "kPn" in P else np.full(nn_, float(P.get("kP", 0.0)))
    z = cand[rr, kk]
    dcur = dco[rr, cur[rr]]
    xc = np.stack([np.log1p(ln[rr]), z / L, (kk == cur[rr]).astype(np.float64), np.log1p(1e3 * dco[rr, kk]),
                   np.arcsinh(1e3 * (dco[rr, kk] - dcur)), np.log1p(Cz[z] * ln[rr] * float(P["um"])),
                   np.log1p(npaths[run_net[rr]]), worst[run_net[rr]], np.log10(np.abs(kpn[run_net[rr]]) + 1e-14) / 14 + 1], 1)
    d0, capv, u = _np(P["d0"]).astype(np.float64), _np(P["cap"]).astype(np.float64), _np(P["u"]).astype(np.float64)
    cz = _np(P["cell_z"]).astype(np.float64)
    xg = np.stack([np.clip(d0 / np.maximum(capv, 1e-3), 0, 10), np.log1p(capv), u, cz / L], 1)
    ent_rk, ent_cell = _np(P["ent_rk"]).astype(np.int64), _np(P["ent_cell"]).astype(np.int64)
    ce = sid.reshape(-1)[ent_rk]
    pt = np.stack([ce, ent_cell])
    pt = _uniq2(pt[:, ce >= 0])
    same = (ent_rk[1:] == ent_rk[:-1])                      # consecutive cells along one (run, slot)
    a, b = ent_cell[:-1][same], ent_cell[1:][same]
    conn = _uniq2(np.concatenate([np.stack([a, b]), np.stack([b, a])], 1)) if len(a) else np.zeros((2, 0), np.int64)
    src, dst = [], []
    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                m = (sid[:, k1] >= 0) & (sid[:, k2] >= 0)
                src.append(sid[m, k1]); dst.append(sid[m, k2])
    comp = np.stack([np.concatenate(src), np.concatenate(dst)]) if src else np.zeros((2, 0), np.int64)
    t = lambda v: torch.as_tensor(v, dtype=torch.float32)
    e = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.long)
    x = {"candidate": t(xc), "grid": t(xg)}
    ei = {("grid", "connects", "grid"): e(conn), ("candidate", "passes_through", "grid"): e(pt),
          ("grid", "influences", "candidate"): e(pt[::-1]), ("candidate", "competes", "candidate"): e(comp)}
    return x, ei, torch.as_tensor(sid)
