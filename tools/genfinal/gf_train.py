#!/usr/bin/env python3
"""gf_train.py (genfinal 2026-11-12, NEW): the generation model. SAME network, graph and features as the GNN start of
tools/groupgnn (tools/best8/gnn_kern.cu: input Linear+ReLU per node type, 3 x per destination type two SAGE convolutions
lin_l(mean of sources) + lin_r(destination), summed, ReLU; output Linear(64, 1); offsets o = 6 tanh(z / 6)), trained on the dumps of
tools/genfinal/gf_hook.py (the kernel's own inputs of DTLA round 1). ONLY the target changes: per candidate (run, layer) the layer the
FINAL route of a finished flow uses (tools/genfinal/gf_label.py). Loss = cross-entropy of softmax over the run's candidates of
4 x onehot(current) + o (the hook's start at beta = 1, gate 1) toward that layer; runs without a final label are left out.
Leave-one-family-out: --train / --test name lists (design keys suite.design[.nNNNN]).
Gaps only: every 5 s tools/genfinal/flowbusy.sh t gen1 (a flow runs, or a waiting flow could start without our reservation);
busy -> checkpoint and exit 3 (the driver resumes later). Checkpoint every
--ckpt_s (90) s too. GPU memory capped at --gpu_gb (30) GB (torch per-process fraction).
  gf_train.py --data DIR --train a,b,.. --test c,.. --out DIR [--steps 2400] [--lr 2e-3]
Out DIR: ckpt.pt (resume), model.pt (final), train.log (json lines), eval.json (per design: accuracy, change precision / recall at
betas 0.6 / 1 / 2 / 4, keep-all baseline, the same for the reference network's offsets), DONE."""
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = (0.6, 1.0, 2.0, 4.0)


_MM = {"coo": False}


def _mm(A, h):
    """CSR x dense (cuSPARSE); if this build refuses it, COO x dense from then on"""
    if not _MM["coo"]:
        try:
            return torch.sparse.mm(A, h)
        except (RuntimeError, NotImplementedError) as e:
            _MM["coo"] = True; print("[gf_train] CSR mm failed, COO fallback: " + repr(e)[:200], flush=True)
    return torch.sparse.mm(A.to_sparse_coo(), h)


class SpMM(torch.autograd.Function):
    """mean aggregation y = A h with A (CSR, rows = destinations, 1/deg values); backward A^T g"""
    @staticmethod
    def forward(ctx, A, AT, h):
        ctx.AT = AT
        return _mm(A, h.contiguous())

    @staticmethod
    def backward(ctx, g):
        return None, None, _mm(ctx.AT, g.contiguous())


class Net(nn.Module):
    """= gnn_kern.cu; parameter names as tools/groupgnn/export11.py reads them"""
    def __init__(self, H=64, layers=3, gin=5, cin=10):
        super().__init__()
        self.H, self.layers, self.gin, self.cin = H, layers, gin, cin
        self.input_proj = nn.ModuleDict({"grid": nn.Linear(gin, H), "candidate": nn.Linear(cin, H)})
        R = ("grid__connects__grid", "candidate__passes_through__grid", "grid__influences__candidate", "candidate__competes__candidate")
        self.rels = R
        self.convs = nn.ModuleList([nn.ModuleDict({r: nn.ModuleDict({"lin_l": nn.Linear(H, H), "lin_r": nn.Linear(H, H, bias=False)}) for r in R})
                                    for _ in range(layers)])
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def layer(self, cv, G, hg, hc):
        r0, r1, r2, r3 = self.rels
        mg0 = SpMM.apply(G["A0"], G["A0T"], hg); mg1 = SpMM.apply(G["A1"], G["A1T"], hc)
        hg2 = F.relu(cv[r0]["lin_l"](mg0) + cv[r0]["lin_r"](hg) + cv[r1]["lin_l"](mg1) + cv[r1]["lin_r"](hg))
        mc2 = SpMM.apply(G["A2"], G["A2T"], hg); mc3 = SpMM.apply(G["A3"], G["A3T"], hc)
        hc2 = F.relu(cv[r2]["lin_l"](mc2) + cv[r2]["lin_r"](hc) + cv[r3]["lin_l"](mc3) + cv[r3]["lin_r"](hc))
        return hg2, hc2

    def forward(self, G, ckpt=False):
        hg = F.relu(self.input_proj["grid"](G["xg"])); hc = F.relu(self.input_proj["candidate"](G["xc"]))
        for cv in self.convs:
            if ckpt and torch.is_grad_enabled():           # activation checkpointing per layer (large graphs)
                from torch.utils.checkpoint import checkpoint
                hg, hc = checkpoint(lambda a_, b_, cv_=cv: self.layer(cv_, G, a_, b_), hg, hc, use_reentrant=False)
            else:
                hg, hc = self.layer(cv, G, hg, hc)
        return self.out(hc).squeeze(-1)


def csr(off, src, n_src, dev):
    off = torch.as_tensor(off.astype(np.int64)); src = torch.as_tensor(src.astype(np.int64))
    n_dst = len(off) - 1
    deg = (off[1:] - off[:-1]).clamp(min=1).float()
    val = torch.repeat_interleave(1.0 / deg, off[1:] - off[:-1])
    A = torch.sparse_csr_tensor(off, src, val, size=(n_dst, n_src))
    dst = torch.repeat_interleave(torch.arange(n_dst), off[1:] - off[:-1])
    AT = torch.sparse_coo_tensor(torch.stack([src, dst]), val, size=(n_src, n_dst)).coalesce().to_sparse_csr()
    return A.to(dev), AT.to(dev)


def load(data, key, dev):
    """one design dump + label -> tensors on dev (CPU tensors cached in <data>/cache/<key>.pt for fast resumes)"""
    cp = os.path.join(data, "cache", key + ".pt")
    if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(os.path.join(data, "labels", key + ".npz")):
        C = torch.load(cp, map_location="cpu", weights_only=False)
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in C.items()}
    G = load_npz(data, key, torch.device("cpu"))
    try:
        os.makedirs(os.path.dirname(cp), exist_ok=True); torch.save(G, cp + ".tmp"); os.replace(cp + ".tmp", cp)
    except Exception:
        pass
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in G.items()}


def load_npz(data, key, dev):
    Z = np.load(os.path.join(data, "dumps", key + ".npz")); Y = np.load(os.path.join(data, "labels", key + ".npz"))
    xg, xc = Z["xg"], Z["xc"]; ng, nc = len(xg), len(xc)
    sid = Z["sid"]; run = Z["run"]; R, K = sid.shape
    rr, kk = np.nonzero(sid >= 0)
    order = sid[rr, kk]; cr = np.empty(nc, np.int64); ck = np.empty(nc, np.int64); cr[order] = rr; ck[order] = kk
    cur = run[:, 5].astype(np.int64); y = Y["y"].astype(np.int64)
    G = {"key": key, "ng": ng, "nc": nc, "R": R, "xg": torch.as_tensor(xg).to(dev), "xc": torch.as_tensor(xc).to(dev)}
    G["A0"], G["A0T"] = csr(Z["off0"], Z["src0"], ng, dev)
    G["A1"], G["A1T"] = csr(Z["off1"], Z["src1"], nc, dev)
    G["A2"], G["A2T"] = csr(Z["off2"], Z["src2"], ng, dev)
    G["A3"], G["A3T"] = csr(Z["off3"], Z["src3"], nc, dev)
    G["cr"] = torch.as_tensor(cr).to(dev); G["iscur"] = torch.as_tensor((ck == cur[cr]).astype(np.float32)).to(dev)
    lab = y >= 0
    G["islab"] = torch.as_tensor((ck == y[cr]) & lab[cr]).to(dev)            # the label candidate of a labeled run
    G["lab"] = torch.as_tensor(lab).to(dev); G["y"] = torch.as_tensor(y).to(dev); G["cur"] = torch.as_tensor(cur).to(dev)
    G["ck"] = torch.as_tensor(ck).to(dev)
    G["zref"] = torch.as_tensor(Z["zref"].astype(np.float32)).to(dev)
    G["n_lab"] = int(lab.sum()); G["n_chg"] = int((lab & (y != cur)).sum())
    return G


def run_logits(G, o, beta=1.0):
    return 4.0 * G["iscur"] + beta * o


def lse_per_run(G, x):
    R = G["R"]; cr = G["cr"]
    mx = torch.full((R,), -1e30, device=x.device).scatter_reduce(0, cr, x, "amax", include_self=True)
    s = torch.zeros(R, device=x.device).index_add(0, cr, torch.exp(x - mx[cr]))
    return mx + torch.log(s.clamp(min=1e-30))


def loss_of(G, z):
    o = 6.0 * torch.tanh(z / 6.0)
    x = run_logits(G, o)
    lse = lse_per_run(G, x)
    tgt = torch.zeros(G["R"], device=x.device).index_add(0, G["cr"], torch.where(G["islab"], x, torch.zeros_like(x)))
    l = (lse - tgt)[G["lab"]]
    return l.mean()


@torch.no_grad()
def metrics(G, z, betas=BETAS, ref=False):
    o = 6.0 * torch.tanh(z / 6.0)
    R = G["R"]; cr = G["cr"]; lab = G["lab"]; y = G["y"]; cur = G["cur"]
    out = {"runs": R, "labeled": G["n_lab"], "changed": G["n_chg"], "keep_all_acc": round(float((cur[lab] == y[lab]).float().mean()), 5)}
    for b in betas:
        x = run_logits(G, o, b)
        mx = torch.full((R,), -1e30, device=x.device).scatter_reduce(0, cr, x, "amax", include_self=True)
        hit = x >= mx[cr]
        # first maximal candidate per run (argmax), in slot order
        idx = torch.full((R,), 1 << 30, device=x.device, dtype=torch.long).scatter_reduce(0, cr, torch.where(hit, G["ck"], torch.full_like(G["ck"], 1 << 30)), "amin", include_self=True)
        flip = idx != cur; chg = lab & (y != cur); good = flip & lab & (idx == y)
        out["b%g" % b] = {"acc": round(float((idx[lab] == y[lab]).float().mean()), 5), "flips": int(flip.sum()),
                          "prec": round(float(good.sum()) / max(int((flip & lab).sum()), 1), 4),
                          "recall": round(float((chg & (idx == y)).sum()) / max(int(chg.sum()), 1), 4)}
    return out


def busy():
    try:
        r = subprocess.run(["bash", os.path.join(HERE, "flowbusy.sh"), "t", "gen1"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception as e:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--train", required=True); ap.add_argument("--test", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=2400); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ckpt_s", type=float, default=90.0)
    ap.add_argument("--gpu_gb", type=float, default=30.0); ap.add_argument("--no_busy", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt_nodes", type=int, default=2500000, help="activation checkpointing per layer on graphs with more nodes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda")
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb / tot), 0)
    logf = open(os.path.join(a.out, "train.log"), "a")
    def log(rec):
        rec["t"] = time.strftime("%F %T"); logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)
    if not a.no_busy:
        b = busy()
        if b:
            log({"preempt_at_start": b}); sys.exit(3)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    step = 0; ck = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)        # RNG state must stay a CPU ByteTensor
        net.load_state_dict(s["state"]); opt.load_state_dict(s["opt"]); step = int(s["step"])
        random.setstate(s["py_rng"]); torch.set_rng_state(s["torch_rng"].cpu())
        log({"resume": step})
    def save(path):
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "step": step, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                    "train": a.train, "test": a.test, "version": "genfinal1", "H": 64, "layers": 3, "gin": 5, "cin": 10}, path + ".tmp")
        os.replace(path + ".tmp", path)
    t_load = time.perf_counter()
    trainK = [k for k in a.train.split(",") if k]; testK = [k for k in a.test.split(",") if k]
    TR = [load(a.data, k, dev) for k in trainK]
    log({"loaded_train": [(G["key"], G["R"], G["nc"], G["ng"], G["n_lab"], G["n_chg"]) for G in TR], "load_s": round(time.perf_counter() - t_load, 1),
         "gpu_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 2)})
    sched = lambda s: a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    t_ck = time.perf_counter(); t_busy = time.perf_counter(); run_l = []
    order = []; SKIP = set()
    while step < a.steps:
        if not order:
            order = list(range(len(TR))); random.shuffle(order)
        G = TR[order.pop()]
        if G["key"] in SKIP:
            continue
        for pg in opt.param_groups:
            pg["lr"] = sched(step)
        net.train()
        try:
            z = net(G, ckpt=G["ng"] + G["nc"] > a.ckpt_nodes); l = loss_of(G, z)
            opt.zero_grad(set_to_none=True); l.backward()
        except torch.cuda.OutOfMemoryError:
            z = l = None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            SKIP.add(G["key"]); log({"oom_skip": G["key"], "ng": G["ng"], "nc": G["nc"]})
            if len(SKIP) >= len(TR):
                sys.exit(5)
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step(); step += 1; run_l.append(float(l))
        if step % 50 == 0:
            log({"step": step, "loss": round(float(np.mean(run_l)), 5), "lr": round(sched(step), 6), "gnorm": round(gn, 4),
                 "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}); run_l = []
        now = time.perf_counter()
        if now - t_ck > a.ckpt_s:
            save(ck); t_ck = now
        if not a.no_busy and now - t_busy > 5.0:
            t_busy = now; b = busy()
            if b:
                save(ck); log({"preempt": b, "step": step}); sys.exit(3)
    save(ck); save(os.path.join(a.out, "model.pt"))
    # evaluation: training designs (fit) and held-out designs
    net.eval(); ev = {"train": {}, "test": {}, "steps": step, "skipped_oom": sorted(SKIP),
                      "data": {G["key"]: {"runs": G["R"], "labeled": G["n_lab"], "changed": G["n_chg"], "cand": G["nc"], "grid": G["ng"]} for G in TR}}
    ev["data_total"] = {f: int(sum(v[f] for v in ev["data"].values())) for f in ("runs", "labeled", "changed", "cand", "grid")}
    with torch.no_grad():
        for G in TR:
            try:
                ev["train"][G["key"]] = metrics(G, net(G))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); ev["train"][G["key"]] = {"error": "oom"}
        del TR, G; torch.cuda.empty_cache()
        for k in testK:
            try:
                G = load(a.data, k, dev)
                m = metrics(G, net(G)); m["ref_f1"] = metrics(G, G["zref"], betas=(0.75, 1.0, 2.0, 4.0))
                m["loss"] = round(float(loss_of(G, net(G))), 5)
                m["size"] = {"runs": G["R"], "cand": G["nc"], "grid": G["ng"]}
            except Exception as e:
                m = {"error": repr(e)[:200]}
            ev["test"][k] = m; G = None; torch.cuda.empty_cache()
    json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    log({"eval": ev})
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
