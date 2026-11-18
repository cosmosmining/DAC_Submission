#!/usr/bin/env python3
"""ns_train.py: offline study of net selection for the later DTLA rounds, from ns_hook log runs
(NS_MODE=log: per round and candidate net the features FEATS and the labels). Teacher forcing: every round is the log run's own state.
  eval  --logs DIR [--test GLOB] [--extra_train DIR ...]
        (1) rule = the previous round's movers + the top k of the flow's ranking, k set so that the freed share is q;
        (2) learned = MLP on FEATS, one model per held-out design family (never trained on the tested design: all logs of the other
            families + --extra_train logs), score's top q (+ the previous round's movers: "learned+prev");
        per round and summed per log: recall of the nets that moved (applied and kept; app1 for a rejected round) at q = 5/10/15/20/30 %,
        and the round time predicted from the log's own walls (fixed part + share q of the free-set part + the recalled moves' part).
        --model M: the given model for every log instead (e.g. Group logs with the model trained on the small designs only)
  fit   --logs DIR [--extra_train DIR ...] [--exclude SUBSTR ...] --out model.npz     model for the flow (ns_hook NS_MODEL)
  time  --logs DIR                                                                     per round walls of runs with NS_MODE rule/learned
Labels: y = app if the round was kept, else app1 (what the descent moved and the rows rebuilt). Rounds >= 1 only (round 0 has no
history and is always the flow's full set)."""
import argparse, fnmatch, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ns_hook import FEATS

QS = (0.05, 0.1, 0.15, 0.2, 0.3)
FIXED = ("device_wait_s", "sens_s", "sens_read_s", "select_s", "names_s")
MOVE = ("rows_a_s", "rows_b_s", "rows_c_s", "rows_d_s", "apply_dem_s", "apply_py_s", "apply_norm_s", "test_s", "split_s", "fallback_descent_s")


def runs_in(d):
    out = []
    for j in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        key = os.path.basename(j)[:-6]
        recs = [json.loads(l) for l in open(j) if l.strip()]
        recs = [r for r in recs if "round" in r and r.get("ns") == 1]
        out.append((key, recs, j[:-6]))
    return out


def family(key):
    return key.split(".")[1]


def load_rounds(d, r_min=1):
    """-> list of dict(key, fam, round, X, y, rank, prev, rec)"""
    R = []
    for key, recs, pre in runs_in(d):
        for rec in recs:
            r = rec["round"]
            p = "%s.r%d.npz" % (pre, r)
            if r < r_min or not os.path.exists(p):
                continue
            z = np.load(p)
            yk = z["app"] if rec.get("kept") else z["app1"]
            ok = z["free"].astype(bool) & (yk >= 0)
            X = np.stack([z[f].astype(np.float32) for f in FEATS], 1)
            R.append(dict(key=key, fam=family(key), round=r, X=X, y=(yk > 0), ok=ok, prev=z["p_app"] > 0, rec=rec, n=len(yk)))
    return R


def torch_fit(X, y, w, epochs=8, seed=0, hid=(64, 32), lr=2e-3, bs=8192, threads=4):
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed); np.random.seed(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); yt = torch.from_numpy(y.astype(np.float32)); wt = torch.from_numpy(w.astype(np.float32))
    d = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hid[0]), torch.nn.ReLU(), torch.nn.Linear(hid[0], hid[1]), torch.nn.ReLU(), torch.nn.Linear(hid[1], 1))
    p = float((y * w).sum() / w.sum())
    pw = torch.tensor(min(max((1 - p) / max(p, 1e-6), 1.0), 50.0))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            z = net(Xn[b]).squeeze(1)
            l = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[b], weight=wt[b], pos_weight=pw)
            opt.zero_grad(); l.backward(); opt.step()
    L = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {"feats": np.array(FEATS), "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
            "W1": L[0].weight.detach().numpy().T.copy(), "b1": L[0].bias.detach().numpy().copy(),
            "W2": L[1].weight.detach().numpy().T.copy(), "b2": L[1].bias.detach().numpy().copy(),
            "w3": L[2].weight.detach().numpy().T.copy(), "b3": L[2].bias.detach().numpy().copy()}


def score(m, X):
    Xn = (X - m["mu"]) / m["sd"]
    h = np.maximum(Xn @ m["W1"] + m["b1"], 0.0); h = np.maximum(h @ m["W2"] + m["b2"], 0.0)
    return (h @ m["w3"] + m["b3"]).reshape(-1)


def stack(rounds):
    X = np.concatenate([r["X"][r["ok"]] for r in rounds]); y = np.concatenate([r["y"][r["ok"]] for r in rounds])
    w = np.concatenate([np.full(int(r["ok"].sum()), 1.0 / max(int(r["ok"].sum()), 1)) for r in rounds])
    return X, y, w * len(w) / w.sum()


def rule_set(n, prev, q):
    """prev movers + top k of the ranking (candidates are in rank order), k such that the set has round(q n) nets (as near as possible)"""
    target = int(round(q * n))
    pc = np.cumsum(~prev)                     # number of non-prev nets among the first k
    npv = int(prev.sum())
    k = int(np.searchsorted(pc, max(target - npv, 0), side="left")) + 1 if target > npv else 0
    s = prev.copy(); s[:min(k, n)] = True
    return s


def time_model(rec):
    w = rec.get("walls") or {}
    fx = sum(w.get(k, 0.0) for k in FIXED); mv = sum(w.get(k, 0.0) for k in MOVE); de = w.get("descent_s", 0.0)
    var = sum(v for k, v in w.items() if k not in FIXED and k not in MOVE and k != "descent_s")
    return fx, var, de, mv


def evaluate(a):
    R = load_rounds(a.logs)
    extra = [r for d in (a.extra_train or []) for r in load_rounds(d)]
    fams = sorted(set(r["fam"] for r in R))
    test_keys = sorted(set(r["key"] for r in R if not a.test or any(fnmatch.fnmatch(r["key"], t) for t in a.test)))
    models = {}
    if getattr(a, "model", ""):                   # a given model for every tested log (e.g. Group with all_small.npz: never trained on it)
        z_ = np.load(a.model); m_ = {k: z_[k] for k in z_.files}
        for f in fams:
            models[f] = m_
        print("[model] %s for %s" % (a.model, ", ".join(fams)), flush=True)
    for f in ([] if getattr(a, "model", "") else fams):
        tr = [r for r in R if r["fam"] != f] + extra
        if not tr:
            continue
        X, y, w = stack(tr)
        models[f] = torch_fit(X, y, w, epochs=a.epochs, seed=a.seed)
        print("[fit] held out %s: %d rows, %.2f %% positive" % (f, len(y), 100.0 * y.mean()), flush=True)
    rows = []; summ = {}
    for key in test_keys:
        rs = [r for r in R if r["key"] == key]
        f = rs[0]["fam"]; m = models.get(f)
        acc = {}
        for r in rs:
            n = r["n"]; y = r["y"] & r["ok"]; ny = int(y.sum()); prev = r["prev"]
            fx, var, de, mv = time_model(r["rec"])
            line = {"key": key, "round": r["round"], "n": n, "movers": ny, "prev": int(prev.sum()), "kept": r["rec"].get("kept"),
                    "s": r["rec"].get("s"), "fixed_s": round(fx, 3), "var_s": round(var, 3), "descent_s": round(de, 3), "move_s": round(mv, 3)}
            z = score(m, r["X"]) if m is not None else None
            order = np.argsort(-z, kind="stable") if z is not None else None
            for q in QS:
                kq = int(round(q * n))
                rs_ = rule_set(n, prev, q)
                rec_rule = (rs_ & y).sum() / max(ny, 1)
                out = {"rule": [round(rs_.mean(), 4), round(float(rec_rule), 4)]}
                if order is not None:
                    s1 = np.zeros(n, bool); s1[order[:kq]] = True
                    s2 = s1 | prev
                    out["learned"] = [round(s1.mean(), 4), round(float((s1 & y).sum() / max(ny, 1)), 4)]
                    out["learned+prev"] = [round(s2.mean(), 4), round(float((s2 & y).sum() / max(ny, 1)), 4)]
                for arm, (fr, rc) in out.items():
                    t = fx + fr * (var + de) + rc * mv            # descent scaled with the share (upper bound of the saving)
                    t_lo = fx + fr * var + de + rc * mv           # descent not scaled (lower bound)
                    out[arm] = [fr, rc, round(t, 3), round(t_lo, 3)]
                    A = acc.setdefault((arm, q), [0, 0, 0.0, 0.0, 0.0, 0])
                    A[0] += int(round(rc * ny)); A[1] += ny; A[2] += t; A[3] += t_lo; A[4] += fr; A[5] += 1
                line["q%g" % q] = out
            line["flow_s_model"] = round(fx + var + de + mv, 3)
            rows.append(line)
        summ[key] = {"%s@%g" % (arm, q): {"recall": round(A[0] / max(A[1], 1), 4), "freed": round(A[4] / max(A[5], 1), 4), "t_round_hi": round(A[2] / max(A[5], 1), 3),
                                          "t_round_lo": round(A[3] / max(A[5], 1), 3)} for (arm, q), A in acc.items()}
        flow_t = [l["flow_s_model"] for l in rows if l["key"] == key]
        summ[key]["flow_round_s"] = round(float(np.mean(flow_t)), 3) if flow_t else None
        summ[key]["rounds"] = len(rs); summ[key]["movers"] = int(sum(int((r["y"] & r["ok"]).sum()) for r in rs)); summ[key]["n_cand"] = rs[0]["n"]
    out = {"rows": rows, "summary": summ}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    for key, s in summ.items():
        print("%-40s rounds %d n %d movers %d flow_round_s %s" % (key, s["rounds"], s["n_cand"], s["movers"], s["flow_round_s"]))
        for q in (0.1, 0.2):
            print("   q=%.2f " % q + "  ".join("%s rec %.3f freed %.3f t %.2f..%.2f" % (arm, s.get("%s@%g" % (arm, q), {}).get("recall", -1), s.get("%s@%g" % (arm, q), {}).get("freed", -1),
                                                                                       s.get("%s@%g" % (arm, q), {}).get("t_round_lo", -1), s.get("%s@%g" % (arm, q), {}).get("t_round_hi", -1))
                                             for arm in ("rule", "learned", "learned+prev")))
    return out


def fit(a):
    R = [r for d in [a.logs] + (a.extra_train or []) for r in load_rounds(d)]
    if a.exclude:
        R = [r for r in R if not any(s in r["key"] for s in a.exclude)]
    X, y, w = stack(R)
    m = torch_fit(X, y, w, epochs=a.epochs, seed=a.seed)
    np.savez(a.out, **m)
    print("[fit] %d rows from %d rounds of %d logs, %.2f %% positive -> %s" % (len(y), len(R), len(set(r["key"] for r in R)), 100.0 * y.mean(), a.out))


def times(a):
    for key, recs, pre in runs_in(a.logs):
        print(key)
        for r in recs:
            w = r.get("walls") or {}
            fx, var, de, mv = time_model(r)
            print("   r%d %-7s cand %s free %s nets_free %s app %s kept %s gain %.4f s %.2f (fixed %.2f var %.2f desc %.2f move %.2f) sel_s %s" % (
                r["round"], r.get("sel"), r.get("n_cand"), r.get("n_free"), r.get("nets_free"), r.get("n_app"), r.get("kept"), r.get("gain", 0.0), r.get("s") or 0.0, fx, var, de, mv, r.get("sel_s")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("eval", "fit", "time"))
    ap.add_argument("--logs", required=True); ap.add_argument("--extra_train", nargs="*"); ap.add_argument("--test", nargs="*")
    ap.add_argument("--exclude", nargs="*"); ap.add_argument("--out", default=""); ap.add_argument("--model", default="", help="eval: use this model for every log (no leave-one-out fits)"); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"eval": evaluate, "fit": fit, "time": times}[a.cmd](a)


if __name__ == "__main__":
    main()
