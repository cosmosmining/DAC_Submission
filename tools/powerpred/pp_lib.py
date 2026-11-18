#!/usr/bin/env python3
"""pp_lib.py (powerpred 2026-11-12, NEW): per-net labels and features of the power round (power_round_i of tools/day1/chain_d1.py and
tools/final/chain_y.py), and the learned / baseline net selector for it. Reached as the global PPL (builtins) from the patched chain source
(pp_launch.py). Env:
  PP_OUT   prefix of the files written: <PP_OUT>.c<k>.npz (k-th power round of the run) and <PP_OUT>.jsonl
  PP_MODE  log (default; nothing selected) | off (same) | learned (PP_MODEL .npz of pp_train.py, keep the top PP_FRAC) | base (activity x
           wire length, keep the top PP_FRAC)
  PP_FRAC  share of the round's net list kept (default 0.25)
  PP_STOP  1: end the process right after the first power round was written (label runs: nothing later runs, no route is written)
Universe of a round = its net list allr = crit nets o1 (by timing weight) + power nets o2 (top cc = activity x net capacitance), in order.
Labels (exact, from the estimator's own per-net capacitance before and after the round): ctot0, ctot1 (r["net_ctot"]), dc = kp*(ctot0-ctot1)
(the power term is w3*sum kp*ctot, so w3*dc is the net's gain in the score), moved = ctot changed, moved_u = in the round's undo set."""
import json, os, time
import numpy as np

ST = {"calls": 0, "pre": None, "model": None}
LM = 12
NAMES = (["lkp", "lctot", "lcc", "rd", "lw", "slack", "in_o1", "pos", "fan", "lwl", "lvia", "capw", "layer_mean", "layer_top", "act_wl", "r_lkp", "r_lctot", "r_lwl", "r_lcc"]
         + ["wl%d" % z for z in range(LM)] + ["cap%d" % z for z in range(LM)])
IDX = {n: i for i, n in enumerate(NAMES)}


def _prank(v):
    v = np.asarray(v, np.float64); r = np.empty(len(v), np.float32)
    r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=np.float32) / max(len(v) - 1, 1)
    return r


def feats(S, r_, o_, wts3, kp, cc, o1, allr):
    n = len(allr); L = int(S.la.L)
    C = np.zeros(max(L, LM)); Ca = np.asarray(S.la.C, np.float64).ravel(); C[:min(len(Ca), len(C))] = Ca[:len(C)]
    cmin = C[C > 0].min() if (C > 0).any() else 0.0
    ctot = np.asarray(r_["net_ctot"], np.float64); rd = np.asarray(r_["net_rd_kohm"], np.float64)
    try:
        sl = np.asarray(o_["net_slack"], np.float64)
    except Exception:
        sl = None
    WL = np.zeros((n, LM)); VIA = np.zeros(n); FAN = np.zeros(n)
    for j, k in enumerate(allr.tolist()):
        name = S.SD.net_names[k]; i = S.ridx.get(name)
        if i is None:
            continue
        pins = S.pins_of(name); FAN[j] = len(pins) if pins else 0
        for x1, y1, z1, x2, y2, z2 in S.new.get(i, S.segs[i]):
            if z1 == z2:
                WL[j, min(int(z1), LM - 1)] += abs(x2 - x1) + abs(y2 - y1)
            else:
                VIA[j] += abs(int(z2) - int(z1))
    wl = WL.sum(1); CAP = WL * C[:LM][None, :]
    capw = CAP.sum(1)
    zs = np.arange(LM)[None, :]
    lmean = (WL * zs).sum(1) / np.maximum(wl, 1.0)
    ltop = WL[:, LM // 2:].sum(1) / np.maximum(wl, 1.0)
    k_ = kp[allr]
    X = np.zeros((n, len(NAMES)), np.float32)
    X[:, IDX["lkp"]] = np.log10(np.maximum(k_, 1e-30)); X[:, IDX["lctot"]] = np.log10(np.maximum(ctot[allr], 1e-30))
    X[:, IDX["lcc"]] = np.log10(np.maximum(cc[allr], 1e-30)); X[:, IDX["rd"]] = rd[allr]
    X[:, IDX["lw"]] = np.log1p(np.maximum(wts3[allr], 0)); X[:, IDX["slack"]] = 0 if sl is None else sl[allr]
    X[len(o1):, IDX["in_o1"]] = 0; X[:len(o1), IDX["in_o1"]] = 1
    X[:, IDX["pos"]] = np.arange(n) / max(n - 1, 1); X[:, IDX["fan"]] = FAN
    X[:, IDX["lwl"]] = np.log1p(wl); X[:, IDX["lvia"]] = np.log1p(VIA); X[:, IDX["capw"]] = capw
    X[:, IDX["layer_mean"]] = lmean; X[:, IDX["layer_top"]] = ltop
    X[:, IDX["act_wl"]] = np.log10(np.maximum(k_ * wl, 1e-30))
    X[:, IDX["r_lkp"]] = _prank(X[:, IDX["lkp"]]); X[:, IDX["r_lctot"]] = _prank(X[:, IDX["lctot"]])
    X[:, IDX["r_lwl"]] = _prank(wl); X[:, IDX["r_lcc"]] = _prank(X[:, IDX["lcc"]])
    X[:, IDX["wl0"]:IDX["wl0"] + LM] = np.log1p(WL); X[:, IDX["cap0"]:IDX["cap0"] + LM] = CAP
    return X


def pre(S, r_, o_, wts3, kp, cc, o1, allr, a):
    ST["calls"] += 1
    t = time.perf_counter()
    X = feats(S, r_, o_, wts3, kp, cc, o1, allr)
    ST["pre"] = {"ids": np.asarray(allr).copy(), "X": X, "ctot0": np.asarray(r_["net_ctot"], np.float64)[allr].copy(), "kp": np.asarray(kp, np.float64)[allr].copy(),
                 "n_o1": int(len(o1)), "w3": float(S.w3), "score0": float(r_["score"]), "pw0": float(r_["pw"]), "feat_s": time.perf_counter() - t, "S": S, "sel": None}


def _mlp(X, M):
    Z = (X - M["mu"]) / M["sd"]
    H = np.maximum(Z @ M["W1"] + M["b1"], 0)
    return (H @ M["W2"] + M["b2"]).ravel()


def select(allr):
    P = ST["pre"]; mode = os.environ.get("PP_MODE", "log")
    if P is None or mode in ("log", "off"):
        return allr
    frac = float(os.environ.get("PP_FRAC", "0.25")); n = len(allr); k = max(1, int(np.ceil(frac * n)))
    if mode == "base":
        sc = P["X"][:, IDX["act_wl"]]
    else:
        if ST["model"] is None:
            z = np.load(os.environ["PP_MODEL"]); ST["model"] = {q: z[q] for q in z.files}
        sc = _mlp(P["X"], ST["model"])
    keep = np.zeros(n, bool); keep[np.argsort(-sc, kind="stable")[:k]] = True
    P["sel"] = keep
    return np.asarray(allr)[keep]


def post(res):
    P = ST["pre"]
    if P is None:
        return
    kept2, undo2, r3, dtp = res
    ids = P["ids"]; c1 = np.asarray(r3["net_ctot"], np.float64)[ids]; c0 = P["ctot0"]
    moved = np.abs(c1 - c0) > 1e-12 * np.maximum(1.0, np.abs(c0))
    S = P["S"]; und = set(undo2.keys()) if undo2 else set()
    mu = np.array([S.ridx.get(S.SD.net_names[k]) in und for k in ids.tolist()])
    dc = P["kp"] * (c0 - c1)
    meta = {"call": ST["calls"], "kept": bool(kept2), "round_s": float(dtp), "n": int(len(ids)), "n_o1": P["n_o1"], "moved": int(moved.sum()), "moved_u": int(mu.sum()),
            "w3": P["w3"], "score0": P["score0"], "score1": float(r3["score"]), "pw0": P["pw0"], "pw1": float(r3["pw"]), "mode": os.environ.get("PP_MODE", "log"),
            "n_sel": int(P["sel"].sum()) if P["sel"] is not None else int(len(ids)), "feat_s": P["feat_s"], "gain_sum_w3dc": float(P["w3"] * dc[moved].sum())}
    out = os.environ.get("PP_OUT", "")
    if out:
        np.savez_compressed("%s.c%d.npz" % (out, ST["calls"]), ids=ids, X=P["X"], ctot0=c0, ctot1=c1, kp=P["kp"], dc=dc, moved=moved, moved_u=mu, n_o1=P["n_o1"],
                            sel=P["sel"] if P["sel"] is not None else np.ones(len(ids), bool), w3=P["w3"], names=np.array(NAMES))
        with open(out + ".jsonl", "a") as fh:
            fh.write(json.dumps(meta) + "\n")
    print("PPL power round", json.dumps(meta), flush=True)
    ST["pre"] = None
    if os.environ.get("PP_STOP") == "1":
        print("PPL stop after the power round", flush=True)
        os._exit(0)


def wrap(f, cur):
    ST["pre"] = None
    res = f(cur)
    post(res)
    return res
