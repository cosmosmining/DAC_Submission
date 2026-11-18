#!/usr/bin/env python3
"""dtla_gnn.py — end-to-end trained GNN for timing-driven layer assignment (no teacher).

The network maps a DTLA problem (dtla.build: the critical nets' runs, their candidate
layers, the congestion each candidate would see, the STA paths through each net) to layer
logits for every run. It is trained END-TO-END through the differentiable ISPD'25 score of
dtla.Objective (overflow + power + smooth WNS/TNS), with straight-through Gumbel-softmax so
the forward pass is a HARD (rounded) assignment — the thing that gets routed. No teacher
labels, no imitation of DGR/greedy outputs.

Graph: run nodes <-> net nodes (a run belongs to one net), 2 message-passing rounds.
  run features   : log length, direction, per candidate slot [valid, is-current, C_z, log R_z,
                   mean / max overflow d0-2c along the run, blocked fraction]
  net features   : log(1+#violating paths through it), worst slack / |WNS|, log C_n0, log #runs
  output         : one logit per candidate slot (masked)

Protocol (dtla_gnn.py --train ... --test ...): train on problems of some designs, report the
rounded DTLA score on held-out designs for init in {current, uniform, gnn} after
0 / 25 / 100 / 300 optimizer steps (same Objective, same budget).

  python dtla_gnn.py --train experiments/dtla_problems/ariane.*.pt ... --test <held-out .pt> \
                     --epochs 200 --out models/dtla_gnn.pt
  python dtla_gnn.py --self_test
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import dtla as DT
from timing_promote import RC

HID = 64


def log(m):
    print(f"[dtla_gnn {time.strftime('%H:%M:%S')}] {m}", flush=True)


def features(P):
    """numpy -> (run_x [R, Fr], slot_x [R, K, Fs], net_x [N, Fn])."""
    R, K, L = len(P["run_net"]), P["K"], P["L"]
    cand = P["cand"]
    valid = (cand >= 0).astype(np.float32)
    Cz = np.array([RC[f"metal{z+1}"][1] for z in range(L)], np.float32)
    Rz = np.array([RC[f"metal{z+1}"][0] for z in range(L)], np.float32)
    cz = np.where(cand >= 0, Cz[np.clip(cand, 0, L - 1)], 0.0) / 0.1
    rz = np.where(cand >= 0, np.log(Rz[np.clip(cand, 0, L - 1)]) / 10.0, 0.0)
    cur = np.zeros((R, K), np.float32)
    cur[np.arange(R), P["run_cur"]] = 1.0
    # per-(run, slot) congestion aggregates over the cells the slot would occupy
    ov = (P["d0"] - 2.0 * P["cap"]).astype(np.float32)
    blk = (P["cap"] <= 1e-3).astype(np.float32)
    cnt = np.zeros(R * K, np.float32)
    s_ov = np.zeros(R * K, np.float32)
    m_ov = np.full(R * K, -50.0, np.float32)
    s_bk = np.zeros(R * K, np.float32)
    rk, cell = P["ent_rk"], P["ent_cell"]
    np.add.at(cnt, rk, 1.0)
    np.add.at(s_ov, rk, ov[cell])
    np.maximum.at(m_ov, rk, ov[cell])
    np.add.at(s_bk, rk, blk[cell])
    cnt = np.maximum(cnt, 1.0)
    mean_ov = (s_ov / cnt).reshape(R, K) / 10.0
    max_ov = np.clip(m_ov.reshape(R, K), -50, 50) / 10.0
    frac_bk = (s_bk / cnt).reshape(R, K)
    slot_x = np.stack([valid, cur, cz, rz, mean_ov, max_ov, frac_bk], axis=2).astype(np.float32)
    run_x = np.stack([np.log1p(P["run_len"]), P["run_dir"].astype(np.float32)], axis=1).astype(np.float32)
    N = int(P["run_net"].max()) + 1
    k_n = np.zeros(N, np.float32)
    np.add.at(k_n, P["p_net"], 1.0)
    wns = min(float(P["slack0"].min()), -1e-6) if len(P["slack0"]) else -1.0
    worst = np.zeros(N, np.float32)
    np.minimum.at(worst, P["p_net"], P["slack0"][P["p_path"]])
    nruns = np.zeros(N, np.float32)
    np.add.at(nruns, P["run_net"], 1.0)
    clen = np.zeros(N, np.float32)
    np.add.at(clen, P["run_net"], P["run_len"])
    net_x = np.stack([np.log1p(k_n), worst / abs(wns), np.log1p(clen), np.log1p(nruns)], 1).astype(np.float32)
    return run_x, slot_x, net_x


class LAGNN(nn.Module):
    def __init__(self, fr=2, fs=7, fn=4, hid=HID, rounds=2):
        super().__init__()
        self.enc_run = nn.Sequential(nn.Linear(fr + 5 * fs, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.enc_net = nn.Sequential(nn.Linear(fn, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.net_upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, hid))
                                      for _ in range(rounds)])
        self.run_upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, hid))
                                      for _ in range(rounds)])
        self.slot = nn.Sequential(nn.Linear(hid + fs, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, run_x, slot_x, net_x, run_net):
        R, K, Fs = slot_x.shape
        pad = torch.zeros(R, 5, Fs, device=slot_x.device)
        pad[:, :K] = slot_x
        h = self.enc_run(torch.cat([run_x, pad.reshape(R, -1)], 1))
        g = self.enc_net(net_x)
        N = net_x.shape[0]
        cntn = torch.zeros(N, device=h.device).index_add_(0, run_net, torch.ones(R, device=h.device)).clamp(min=1)
        for nu, ru in zip(self.net_upd, self.run_upd):
            agg = torch.zeros(N, h.shape[1], device=h.device).index_add_(0, run_net, h) / cntn[:, None]
            g = g + nu(torch.cat([g, agg], 1))
            h = h + ru(torch.cat([h, g[run_net]], 1))
        logits = self.slot(torch.cat([h[:, None, :].expand(R, K, h.shape[1]), slot_x], 2)).squeeze(2)
        return logits.masked_fill(slot_x[:, :, 0] < 0.5, DT.NEG)


def tensors(P, device):
    rx, sx, nx = features(P)
    t = lambda a: torch.as_tensor(a, device=device)
    return t(rx), t(sx), t(nx), t(P["run_net"].astype(np.int64))


def load_problem(path, device="cpu"):
    P = torch.load(path, weights_only=False)
    obj = DT.Objective(P, P.get("a", 6e-4), P.get("kP", 0.0), P.get("tau", 0.02), device=device)
    with torch.no_grad():
        C0 = obj.terms(DT.current_logits(P, device))["Cn"].detach()
    return P, obj, C0, tensors(P, device)


def st_gumbel(logits, tau=1.0):
    """straight-through Gumbel-softmax: one-hot forward, softmax gradient backward."""
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)))
    y = torch.softmax((logits + g) / tau, 1)
    hard = Fnn.one_hot(y.argmax(1), logits.shape[1]).float()
    return (hard - y).detach() + y


def hard_score(obj, logits, C0):
    with torch.no_grad():
        h = Fnn.one_hot(logits.argmax(1), obj.K).float() * 50.0
        return float(obj.score(h, C0)[0])


def train(train_paths, epochs=200, lr=2e-3, device="cpu", out=None, test_paths=(), mode="relaxed",
          t_min=0.2, eval_every=10, val_paths=()):
    """Relaxed training can drift to near-equal logits: spreading probability lowers the EXPECTED
    overflow (convex cost), but the rounded assignment — the thing that is routed — gets worse
    (seen with 5x data or 5x epochs: hard d-score -1.74 -> +0.65). So every `eval_every` epochs the
    rounded score is measured on the REAL training problems (synthetic '.syn' ones excluded) and
    the best checkpoint by that true objective is kept."""
    data = [load_problem(p, device) for p in train_paths]
    log(f"training on {len(data)} problems ({sum(len(d[0]['run_net']) for d in data):,} runs)")
    model = LAGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = [hard_score(obj, DT.current_logits(P, device), C0) for P, obj, C0, _ in data]
    sel = [k for k, p in enumerate(train_paths) if ".syn" not in os.path.basename(p)] or list(range(len(data)))
    best = (float("inf"), -1, None)
    # checkpoint selection on a held-out VALIDATION split when given (selecting on the training problems
    # picked a checkpoint that was worse than doing nothing on unseen problems for one seed)
    vdata = [load_problem(p, device) for p in val_paths]
    vbase = [hard_score(obj, DT.current_logits(P, device), C0) for P, obj, C0, _ in vdata]
    if vdata:
        log(f"checkpoint selection on {len(vdata)} validation problems")

    def hard_eval(idx):
        with torch.no_grad():
            if vdata:
                return float(np.mean([hard_score(o, model(*x), c) - b for (P, o, c, x), b in zip(vdata, vbase)]))
            return float(np.mean([hard_score(data[k][1], model(*data[k][3]), data[k][2]) - base[k] for k in idx]))
    for ep in range(epochs):
        tot = 0.0
        for (P, obj, C0, x), b in zip(data, base):
            logits = model(*x)
            if mode == "st":                             # straight-through Gumbel (noisy)
                y = st_gumbel(logits, tau=max(0.3, 1.0 - ep / max(1, epochs)))
            elif mode == "sth":                          # deterministic straight-through: the
                T = max(t_min, 1.0 - ep / max(1, epochs))  # forward pass IS the rounded (routed)
                soft = torch.softmax(logits.masked_fill(obj.cand < 0, DT.NEG) / T, 1)   # assignment,
                hard = Fnn.one_hot(soft.argmax(1), soft.shape[1]).float()            # so spreading
                y = (hard - soft).detach() + soft                                    # cannot pay off
            else:                                        # deterministic relaxed objective,
                T = max(t_min, 1.0 - ep / max(1, epochs))  # annealed toward one-hot
                y = torch.softmax(logits.masked_fill(obj.cand < 0, DT.NEG) / T, 1)
            s, _ = obj.score_p(y, C0)
            loss = s - b                                 # per-problem baseline: current layers
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
        if ep % eval_every == 0 or ep == epochs - 1:
            h = hard_eval(sel)
            if h < best[0]:
                best = (h, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            log(f"  epoch {ep:4d} train loss {tot / len(data):+.4f}  rounded d-score vs current "
                f"({'validation' if vdata else 'real train'} problems) {h:+.4f}  best {best[0]:+.4f} @ {best[1]}")
    if best[2] is not None:
        model.load_state_dict(best[2])
        log(f"kept the epoch-{best[1]} checkpoint (best rounded score {best[0]:+.4f})")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(model.state_dict(), out)
        log(f"saved {out}")
    return model


def predict_logits(ckpt, P, obj=None, device="cpu"):
    model = LAGNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        return model(*tensors(P, device))


def evaluate(model, test_paths, budgets=(0, 25, 100, 300), device="cpu"):
    rows = []
    for path in test_paths:
        P, obj, C0, x = load_problem(path, device)
        inits = {"current": DT.current_logits(P, device), "uniform": torch.zeros(len(P["run_net"]), P["K"])}
        with torch.no_grad():
            inits["gnn"] = model(*x).clamp(min=-20)
        base = hard_score(obj, inits["current"], C0)
        for name, th0 in inits.items():
            res = []
            for b in budgets:
                if b == 0:
                    res.append(hard_score(obj, th0, C0) - base)
                else:
                    th, sh, _ = DT.optimize(obj, th0, steps=b, lr=0.1, log_every=10 ** 9)
                    res.append(sh - base)
            rows.append((os.path.basename(path), name, res))
            log(f"  {os.path.basename(path):45s} {name:8s} " +
                " ".join(f"{b}:{r:+.4f}" for b, r in zip(budgets, res)))
    return rows


def self_test():
    """Synthetic problem with a feature-dependent optimum: every run's fastest layer is its
    last slot, but for a random half of the runs that slot is blocked with a large overflow
    penalty. The GNN must read the per-slot congestion features to choose; it is trained
    end-to-end on the objective (no labels) and must (i) beat the current layers, (ii) match
    the DTLA optimizer on the same problem, (iii) avoid the blocked slot."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    R, K, T = 80, 5, 8
    cand = np.array([[2, 4, 6, 8, -1] if r % 2 == 0 else [1, 3, 5, 7, 9] for r in range(R)])
    last = np.where(cand[:, 4] >= 0, 4, 3)                   # fastest slot per run
    blocked = rng.random(R) < 0.5
    ent_rk, ent_cell, capv = [], [], []
    for r in range(R):
        for k in range(K):
            if cand[r, k] < 0:
                continue
            for t in range(T):
                ent_rk.append(r * K + k)
                ent_cell.append(len(capv))
                capv.append(0.0 if (blocked[r] and k == last[r]) else 6.0)
    n = len(capv)
    Rz = np.array([RC[f"metal{z+1}"][0] for z in range(10)])
    Cz = np.array([RC[f"metal{z+1}"][1] for z in range(10)])
    l = T * 2.1
    dcoef = np.array([[l * (Rz[z] * 50.0 + Cz[z] * 2.0) / 1000 + Rz[z] * Cz[z] * l * l / 2000 if z >= 0 else 0.0
                       for z in row] for row in cand], np.float32)
    P = {"case": 1, "L": 10, "K": K, "run_net": np.arange(R), "run_len": np.full(R, T, np.float32),
         "run_cur": np.zeros(R, np.int64), "run_p0": np.arange(R) * 2, "run_p1": np.arange(R) * 2 + 1,
         "run_dir": (np.arange(R) % 2).astype(np.int64), "cand": cand, "dcoef": dcoef,
         "ent_rk": np.array(ent_rk), "ent_cell": np.array(ent_cell),
         "d0": np.zeros(n, np.float32), "cap": np.array(capv, np.float32),
         "u": np.full(n, 3e5, np.float32),
         "pt_lo": np.full(R * 2, 10), "pt_hi": np.full(R * 2, -1),
         "p_net": np.arange(R), "p_path": np.arange(R), "slack0": np.full(R, -0.3, np.float32),
         "Cz": Cz.astype(np.float32), "um": 2.1, "a": 0.0, "kP": 0.0, "tau": 0.02}
    d = "/tmp/dtla_gnn_selftest"
    os.makedirs(d, exist_ok=True)
    torch.save(P, f"{d}/p.pt")
    model = train([f"{d}/p.pt"], epochs=300, lr=5e-3)
    Pp, obj, C0, x = load_problem(f"{d}/p.pt")
    cur = hard_score(obj, DT.current_logits(Pp), C0)
    with torch.no_grad():
        lg = model(*x)
    gnn = hard_score(obj, lg, C0)
    th, opt_s, _ = DT.optimize(obj, DT.current_logits(Pp), steps=300, lr=0.2, log_every=10 ** 9)
    ch = lg.argmax(1).numpy()
    used_blocked = int(((ch == last) & blocked).sum())
    fast_free = float(((ch == last) & ~blocked).sum() / max(1, (~blocked).sum()))
    ok = gnn < cur and gnn <= opt_s + 0.05 * abs(cur - opt_s) and used_blocked == 0
    print(f"  current {cur:+.4f} | DTLA optimizer {opt_s:+.4f} | e2e GNN (0 steps) {gnn:+.4f}; "
          f"blocked slot used {used_blocked}, fastest free slot taken {fast_free:.0%}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="models/dtla_gnn.pt")
    ap.add_argument("--load", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="relaxed", choices=["relaxed", "st", "sth"])
    ap.add_argument("--t_min", type=float, default=0.2, help="final softmax temperature of the relaxation")
    ap.add_argument("--eval_every", type=int, default=10, help="epochs between rounded-score checkpoints")
    ap.add_argument("--seed", type=int, default=None, help="torch/numpy seed for training (repeat runs)")
    ap.add_argument("--val", nargs="*", default=[], help="validation problems for checkpoint selection")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
    tr = sorted({p for g in a.train for p in glob.glob(g)})
    te = sorted({p for g in a.test for p in glob.glob(g)})
    if a.load:
        model = LAGNN().to(a.device)
        model.load_state_dict(torch.load(a.load, map_location=a.device))
    else:
        va = sorted({p for g in a.val for p in glob.glob(g)})
        model = train(tr, a.epochs, a.lr, a.device, a.out, mode=a.mode, t_min=a.t_min, eval_every=a.eval_every,
                      val_paths=va)
    if te:
        log("held-out evaluation: rounded DTLA score minus current-layer score (lower = better)")
        evaluate(model, te, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
