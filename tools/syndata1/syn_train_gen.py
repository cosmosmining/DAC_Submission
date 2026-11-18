#!/usr/bin/env python3
"""syn_train_gen.py (syndata1, NEW 14 Nov 2026): S-GEN of RULES_SYN1 (Amendment A3), torch on the GPU.
Network, graph, features, loss and settings of tools/genfinal/gf_train.py as genfinal2 ran it (gf_train.Net: H 64, 3 layers; loss
gf_train.loss_of; AdamW lr 2e-3 wd 1e-4, cosine to 5% over 2400 steps, one graph per step, grad clip 1.0, activation checkpointing
above 2.5M nodes, a graph that runs out of GPU memory is skipped, seed 0, GPU memory capped at 24 GB, 4 threads), trained ONCE on every
listed synthetic graph (no contest graph). gf_train.main() moves every training graph to the GPU before step 1; thousands of synthetic
graphs do not fit, so this file runs the same loop out of core: the visiting order is gf_train's (random.seed(0); order = a shuffled
index list, pop() from its end, reshuffled when empty; a graph in the OOM-skip set is passed over as there), each graph is read by
gf_train.load_npz (unchanged) on worker threads a few steps ahead of its step and moved to the GPU for it. The arithmetic per step is
gf_train's. Checkpoint every 90 s (net, AdamW, step, position in the order, skip set): a resumed run continues the same sequence.
After training: model.pt (gf_train's checkpoint format; gf_export.py reads it), eval.json: "test" = every contest graph of
tools/genfinal2's data (eight H100 designs; metrics, ref_f1 and loss as gf_train) plus per-family sums; "train_sample" = the first
--train_eval graphs of the order (gf_train evaluates every training graph; here a sample, diagnostic only).
  syn_train_gen.py --list GEN_KEYS.tsv --data FARM --test CONTEST_GEN.txt --out DIR --gf TOOLS_GENFINAL [--steps 2400] [--smoke]
  --smoke: --steps 50 unless given (a test of the code path, never a result)."""
import argparse, collections, json, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True); ap.add_argument("--data", required=True); ap.add_argument("--test", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--gf", required=True)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ckpt_s", type=float, default=90.0); ap.add_argument("--gpu_gb", type=float, default=24.0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--ckpt_nodes", type=int, default=2500000)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--ahead", type=int, default=8); ap.add_argument("--train_eval", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.steps is None:
        a.steps = 50 if a.smoke else 2400
    sys.path.insert(0, os.path.abspath(a.gf))
    import torch
    import gf_train as GT
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda")
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb / tot), 0)
    logf = open(os.path.join(a.out, "train.log"), "a")

    def log(rec):
        rec["t"] = time.strftime("%F %T"); logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)
    keys = [l.split("\t")[0] for l in open(a.list) if l.strip()]
    testK = [l.strip() for l in open(a.test) if l.strip()] if a.test else []
    if not keys:
        raise SystemExit("syn_train_gen: empty list")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    net = GT.Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    step = 0; pos = 0; SKIP = set(); BAD = set(); ck = os.path.join(a.out, "ckpt.pt"); sig = [len(keys), keys[0], keys[-1], a.steps, a.lr, a.wd, a.seed]
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        if s.get("sig") == sig:
            net.load_state_dict(s["state"]); opt.load_state_dict(s["opt"]); step = int(s["step"]); pos = int(s["pos"])
            SKIP = set(s["skip"]); BAD = set(s.get("bad", []))
            log({"resume": step, "pos": pos})
        else:
            log({"ckpt_ignored": "signature differs"})

    def save(path):
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "step": step, "pos": pos, "skip": sorted(SKIP), "bad": sorted(BAD), "sig": sig,
                    "train": "%s (%d graphs)" % (os.path.abspath(a.list), len(keys)), "test": ",".join(testK), "version": "genfinal1",
                    "H": 64, "layers": 3, "gin": 5, "cin": 10}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def sequence():                                     # gf_train's visiting order: random.shuffle of range(N), pop() from the end
        order = []
        while True:
            if not order:
                order = list(range(len(keys))); random.shuffle(order)
            yield order.pop()
    sq = sequence()
    for _ in range(pos):                                # resume: the same sequence, the consumed part passed over
        next(sq)
    cpu = torch.device("cpu")

    def load_cpu(i):
        try:
            return GT.load_npz(a.data, keys[i], cpu)
        except Exception as e:
            return e
    sched = lambda s: a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    log({"start": True, "smoke": int(a.smoke), "graphs": len(keys), "test": len(testK), "args": vars(a), "step": step, "pos": pos})
    ex = ThreadPoolExecutor(a.workers); Q = collections.deque(); first = []
    t_ck = time.perf_counter(); t0 = time.perf_counter(); run_l = []; wait_s = 0.0; seen = 0
    while step < a.steps:
        while len(Q) < a.ahead:
            i = next(sq); Q.append((i, ex.submit(load_cpu, i)))
        i, fu = Q.popleft(); pos += 1
        if keys[i] in SKIP or keys[i] in BAD:
            continue
        tw = time.perf_counter(); Gc = fu.result(); wait_s += time.perf_counter() - tw
        if isinstance(Gc, Exception):
            BAD.add(keys[i]); log({"load_error": keys[i], "error": repr(Gc)[:300]}); continue
        if len(first) < a.train_eval:
            first.append(i)
        G = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in Gc.items()}; Gc = None
        for pg in opt.param_groups:
            pg["lr"] = sched(step)
        net.train()
        try:
            z = net(G, ckpt=G["ng"] + G["nc"] > a.ckpt_nodes); l = GT.loss_of(G, z)
            opt.zero_grad(set_to_none=True); l.backward()
        except torch.cuda.OutOfMemoryError:
            z = l = None; G = None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            SKIP.add(keys[i]); log({"oom_skip": keys[i]})
            if len(SKIP) >= len(keys):
                sys.exit(5)
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step(); step += 1; run_l.append(float(l)); seen += 1; G = z = l = None
        if step % 50 == 0:
            log({"step": step, "loss": round(float(np.mean(run_l)), 5), "lr": round(sched(step), 6), "gnorm": round(gn, 4),
                 "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "pos": pos, "wait_s": round(wait_s, 1),
                 "s": round(time.perf_counter() - t0, 1)}); run_l = []
        now = time.perf_counter()
        if now - t_ck > a.ckpt_s:
            save(ck); t_ck = now
    if run_l:
        log({"step": step, "loss": round(float(np.mean(run_l)), 5), "tail_steps": len(run_l)})
    ex.shutdown(wait=True, cancel_futures=True); Q.clear()
    save(ck); save(os.path.join(a.out, "model.pt"))
    log({"trained": step, "pos": pos, "skipped_oom": len(SKIP), "load_errors": len(BAD), "train_s": round(time.perf_counter() - t0, 1), "load_wait_s": round(wait_s, 1)})
    net.eval(); ev = {"steps": step, "graphs": len(keys), "skipped_oom": sorted(SKIP), "load_errors": sorted(BAD), "train_sample": {}, "test": {}, "test_family": {}}
    fam = lambda k: {"ariane": "ariane", "bsg_chip": "bsg", "NV_NVDLA_partition_c": "nvdla", "mempool_tile_wrap": "mempool"}.get(k.split(".")[1], k.split(".")[1])
    with torch.no_grad():
        for i in first:                                  # empty when the run was resumed after its last step

            try:
                G = GT.load_npz(a.data, keys[i], dev); ev["train_sample"][keys[i]] = GT.metrics(G, net(G))
            except Exception as e:
                ev["train_sample"][keys[i]] = {"error": repr(e)[:200]}
            G = None; torch.cuda.empty_cache()
        agg = {}
        for k in testK:
            try:
                G = GT.load_npz(a.data, k, dev)
                m = GT.metrics(G, net(G)); m["ref_f1"] = GT.metrics(G, G["zref"], betas=(0.75, 1.0, 2.0, 4.0))
                m["loss"] = round(float(GT.loss_of(G, net(G))), 5)
                m["size"] = {"runs": G["R"], "cand": G["nc"], "grid": G["ng"]}
                f = fam(k); A = agg.setdefault(f, {"labeled": 0, "changed": 0, "hit": 0.0, "keep": 0.0, "rec": 0.0, "graphs": 0})
                A["labeled"] += m["labeled"]; A["changed"] += m["changed"]; A["graphs"] += 1
                A["hit"] += m["b1"]["acc"] * m["labeled"]; A["keep"] += m["keep_all_acc"] * m["labeled"]; A["rec"] += m["b1"]["recall"] * m["changed"]
            except Exception as e:
                m = {"error": repr(e)[:200]}
            ev["test"][k] = m; G = None; torch.cuda.empty_cache()
        for f, A in agg.items():
            ev["test_family"][f] = {"graphs": A["graphs"], "labeled": A["labeled"], "changed": A["changed"], "acc": round(A["hit"] / max(A["labeled"], 1), 4),
                                    "keep_all_acc": round(A["keep"] / max(A["labeled"], 1), 4), "recall_changed": round(A["rec"] / max(A["changed"], 1), 4)}
    json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    log({"test_family": ev["test_family"]})
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%F %T") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
