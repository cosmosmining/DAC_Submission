#!/usr/bin/env python3
"""gf2_hook.py (genfinal2, NEW 14 Nov 2026): the generation model of tools/genfinal inside every DTLA round of the final flow.
Module GF2 of the patched stage (gf2_patch.py: `res = GF2.descend(C, S, a, X, plain_run, tl, W, FT)` replaces the plain descent).
Env GF2_MODE:
  unset/off  the stage's own call plain_run(True, tl), nothing else (identity control GF0, plain arm P8 with --dgr 8)
  dump       the plain descent first (decisions unchanged), then the network's exact inputs of this round, as tools/genfinal/gf_hook.py
             writes them (xg, xc, sid, four CSR edge lists, run, cand2, mask, g, zref, rows/roff, names, meta), to
             $GF2_OUT/<suite>.<design>.r<NN>.npz; x is saved before and put back after, so the rounding sees the descent's x.
             GF2_BIN0 = any kernel .bin (the dump needs a network handle; its zref is only a reference).
  gen        GD1: one forward of the held-out model $GF2_MODELS/<family>/gen1.bin; proposal = argmax over the run's legal
             candidates of 4 x onehot(current) + 6 tanh(z / 6) (the trained logits, beta 1); x = init x onehot(proposal), 0 steps.
             Rounding, the exact gate and the split of a rejected change are the stage's own code.
  gens       GS8: x = init x onehot(current) + 6 tanh(z / 6), Adam state zero, then the stage's descent with its own settings and
             a.dgr steps (run with --dgr 8), reuse=False (the timer pass is recomputed after the extra evaluation).
Inputs exactly as gf_hook.dump (gradient at the one-hot start, tau GF2_TAU 0.01, gnn_f.Net.inputs). Any error in gen/gens: the plain
start and the plain descent, and a line "[gf2] FALLBACK ..." (such a run is not counted). One json line per call to $GF2_LOG."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
ST = {"calls": 0, "fallback": 0, "dumps": 0, "gen": 0}
FAM = {"ariane": "ariane", "bsg_chip": "bsg", "NV_NVDLA_partition_c": "nvdla", "mempool_tile_wrap": "mempool", "mempool_group": "mempool"}


def mode():
    m = os.environ.get("GF2_MODE", "")
    return "" if m in ("", "off") else m


def on():
    return mode() in ("dump", "gen", "gens")


def _log(rec):
    print("[gf2] " + json.dumps(rec, default=float), flush=True)
    p = os.environ.get("GF2_LOG")
    if p:
        with open(p, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


def _gn():
    b8 = os.path.join(ROOT, "tools", "best8")
    if b8 not in sys.path:
        sys.path.insert(0, b8)
    import gnn_f
    return gnn_f


def _gather(rows, off, idx):
    idx = np.asarray(idx, np.int64)
    c = off[idx + 1] - off[idx]
    o2 = np.zeros(len(idx) + 1, np.int64); o2[1:] = np.cumsum(c)
    s0 = np.repeat(off[idx], c); ar = np.arange(int(o2[-1])) - np.repeat(o2[:-1], c)
    return rows[s0 + ar], o2


def _inputs(X, C, a, FT, binp):
    """the GNN's inputs of this round, as gf_hook.dump builds them (x is overwritten: caller restores or replaces it)"""
    R, Kc = int(X.I["R"]), int(X.I["K"]); run = np.asarray(C.run_, np.int64); L = int(C.L); mask = np.asarray(X.mask, bool)
    cur = run[:, 5].astype(np.int64); ri = np.arange(R)
    oh = np.zeros((R, Kc), np.float32); oh[ri, cur] = 1.0
    X.put("x", ((100.0 * oh) * mask).astype(X.RL))
    if X.size.get("g_out") != X.size["x"]:
        X.zeros("g_out", X.size["x"])
    e = X.eval(T=1.0, mode=1, tau=float(os.environ.get("GF2_TAU", "0.01")), grad=True)
    g = np.asarray(e["g"], np.float32).reshape(R, Kc); dv = X.get("dv", X.RL)
    GN = _gn()
    net = GN.net_of(binp, os.environ.get("GF2_CUDA", "1") == "1", int(os.environ.get("GF2_THREADS") or 8))
    net.L.gnn_opts(1.0, 0)
    tk = C.tid[C.i_k][C.kn]
    kpn_r = np.where(tk >= 0, C.kpn[np.maximum(tk, 0)], 0.0).astype(np.float32).astype(np.float64)
    Cz = np.asarray(a._dgr_Cz, np.float32).astype(np.float64)
    cand2 = np.asarray(C.cand2, np.int32)
    cnt = net.inputs(R, Kc, L, run, cand2, C.A, kpn_r, Cz, float(FT.GRID) / 2000.0, C.A["d0"], C.A["cap"], C.A["cell_z"], dv, g)
    return dict(net=net, cnt=cnt, e=e, g=g, run=run, cand2=cand2, mask=mask, cur=cur, oh=oh, R=R, K=Kc, L=L)


def _dump(X, C, a, S, FT, W):
    t0 = time.perf_counter(); out = os.environ["GF2_OUT"]
    xs = X.get("x", X.RL).copy()
    try:
        I = _inputs(X, C, a, FT, os.environ["GF2_BIN0"])
        net = I["net"]; D = net.dump()
        try:
            z, st = net.forward(want=True)
        except Exception as e_:
            z = np.zeros(int(I["cnt"][0]), np.float32)
        names = [str(S.nets.names[int(i)]) for i in np.asarray(C.i_k, np.int64)]
        rows, roff = _gather(np.asarray(C.rows, np.int64), np.asarray(C.off, np.int64), np.asarray(C.i_k, np.int64))
        r = ST["dumps"]
        meta = {"design": str(getattr(a, "design", "")), "suite": str(getattr(a, "suite", "")), "round": r, "R": I["R"], "K": I["K"], "L": I["L"],
                "GRID": int(FT.GRID), "nc": int(I["cnt"][0]), "ng": int(net.ng), "E": [int(v) for v in I["cnt"][1:5]], "scale": float(net.scale),
                "F_onehot": float(I["e"]["total"]), "dgr_nets": int(getattr(a, "dgr_nets", -1)), "nets_kept": int(len(names)),
                "bin_ref": os.environ["GF2_BIN0"], "after_descent": True}
        os.makedirs(out, exist_ok=True)
        fn = os.path.join(out, "%s.%s.r%02d.npz" % (meta["suite"], meta["design"], r))
        np.savez(fn + ".tmp.npz", xg=D["xg"], xc=D["xc"], sid=D["sid"], keep=D["keep"],
                 off0=D["off0"], src0=D["src0"], off1=D["off1"], src1=D["src1"], off2=D["off2"], src2=D["src2"], off3=D["off3"], src3=D["src3"],
                 run=I["run"], cand2=I["cand2"], mask=I["mask"], g=I["g"], zref=z.astype(np.float32), rows=rows.astype(np.int32), roff=roff,
                 names=np.frombuffer("\n".join(names).encode(), np.uint8), meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        os.replace(fn + ".tmp.npz", fn)
        ST["dumps"] += 1
        meta["dump_s"] = round(time.perf_counter() - t0, 3); meta["file"] = fn; meta["mode"] = "dump"
        _log(meta)
    except Exception as e:
        import traceback
        _log({"mode": "dump", "dump_error": repr(e)[:300], "tb": traceback.format_exc()[-1500:]})
    finally:
        X.put("x", xs)                                                  # the descent's x for the stage's rounding
    if W is not None:
        W["gf2_dump_s"] = W.get("gf2_dump_s", 0.0) + time.perf_counter() - t0


def _offsets(X, C, a, S, FT):
    """forward of the held-out model -> (o [R,K] masked, proposal [R], record)"""
    design = str(getattr(a, "design", "")); fam = FAM.get(design, design)
    binp = os.path.join(os.environ["GF2_MODELS"], fam, "gen1.bin")
    if not os.path.isfile(binp):
        raise FileNotFoundError("model " + binp)
    t0 = time.perf_counter()
    I = _inputs(X, C, a, FT, binp)
    t1 = time.perf_counter()
    z, st = I["net"].forward(want=True)
    t2 = time.perf_counter()
    sid = np.asarray(I["net"].dump()["sid"]).reshape(I["R"], I["K"])
    rr, kk = np.nonzero(sid >= 0)
    if len(rr) != len(z) or not np.all(I["mask"][rr, kk]) or int(I["mask"].sum()) != len(z):
        raise RuntimeError("slot map: sid %d z %d mask %d" % (len(rr), len(z), int(I["mask"].sum())))
    o = np.zeros((I["R"], I["K"]), np.float32)
    o[rr, kk] = (6.0 * np.tanh(z[sid[rr, kk]].astype(np.float64) / 6.0)).astype(np.float32)
    xl = 4.0 * I["oh"] + o
    prop = np.where(I["mask"], xl, -np.inf).argmax(1)
    rec = {"design": design, "suite": str(getattr(a, "suite", "")), "family": fam, "model": binp, "R": I["R"], "nc": int(len(z)),
           "flips": int((prop != I["cur"]).sum()), "off_abs_mean": float(np.abs(o[I["mask"]]).mean()) if I["mask"].any() else 0.0,
           "inputs_s": round(t1 - t0, 3), "forward_s": round(t2 - t1, 3), "kern": [float(v) for v in st]}
    return o, prop, I, rec


def descend(C, S, a, X, plain_run, tl, W=None, FT=None):
    m = mode(); ST["calls"] += 1
    if m == "":
        return plain_run(True, tl)
    if m == "dump":
        res = plain_run(True, tl)
        _dump(X, C, a, S, FT, W)
        return res
    t = time.perf_counter(); init = float(a.dgr_init)
    try:
        o, prop, I, rec = _offsets(X, C, a, S, FT)
    except Exception as e:                                              # never break the stage; the run is not counted
        import traceback
        ST["fallback"] += 1
        X.start(a.dgr_init)
        _log({"mode": m, "call": ST["calls"], "fallback": True, "error": repr(e)[:300], "tb": traceback.format_exc()[-1200:]})
        print("[gf2] FALLBACK %s call %d: %s" % (m, ST["calls"], repr(e)[:200]), flush=True)
        return plain_run(True, tl)
    R, Kc, mask = I["R"], I["K"], I["mask"]
    if m == "gen":
        x = np.zeros((R, Kc), np.float32); x[np.arange(R), prop] = init
        X.put("x", (x * mask).astype(X.RL))
        X.L.dg_zero(X.p["am"], X.size["am"]); X.L.dg_zero(X.p["av"], X.size["av"])
        res = {"steps": 0, "vals": np.zeros((1, 8)), "prof": {}}
    elif m == "gens":
        X.put("x", ((init * I["oh"] + o) * mask).astype(X.RL))
        X.L.dg_zero(X.p["am"], X.size["am"]); X.L.dg_zero(X.p["av"], X.size["av"])
        res = plain_run(False, tl)
    else:
        raise ValueError("GF2_MODE " + m)
    ST["gen"] += 1
    rec.update(mode=m, call=ST["calls"], fallback=False, init=init, steps=int(res["steps"]), s=round(time.perf_counter() - t, 3))
    if W is not None:
        W["gf2_s"] = W.get("gf2_s", 0.0) + time.perf_counter() - t
    _log(rec)
    return res
