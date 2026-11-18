#!/usr/bin/env python3
"""gnn_np.py: the learned prior of the check without torch.
  graph_np(P)              == tools/fast3x/h_graph.graph (dtla_lagnn.graph without np.unique(axis=1)) with numpy arrays
                              in place of the tensors (same float32 / int64 values; the numpy lines are copied verbatim)
  Prior(npz).logits(P, nt) == the chain's
                                  lg = DD.to_logits(model(x, ei), sid).clamp(-60, 60); out = lg.numpy().astype(float64);
                                  out[~(cand >= 0)] = -inf
                              with the forward pass in C++ (tools/pre5/gnn_kern.cc; float32, deterministic, any thread
                              count gives the same bits; NOT bit-identical to torch: the chain's --pre5_gnn check compares)
  export(ckpt, out_npz)    one-time export of a model checkpoint's state_dict (needs torch; run outside the timed run;
                           the file depends on our program only, not on a case)
"""
import ctypes, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEG = -1e4                      # dtla_lagnn.NEG
TYPES = (("grid", "connects", "grid"), ("candidate", "passes_through", "grid"),
         ("grid", "influences", "candidate"), ("candidate", "competes", "candidate"))
_L = None


def _lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(os.path.join(HERE, "libgnn_kern." + ("dylib" if sys.platform == "darwin" else "so")))
        V, I = ctypes.c_void_p, ctypes.c_int64
        L.gnn_forward.restype = ctypes.c_int
        L.gnn_forward.argtypes = [I, I, V, V] + [I, V, V] * 4 + [V, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, V, V]
        _L = L
    return _L


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _uniq2(X):                                           # == tools/fast3x/h_graph._uniq2
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


def graph_np(P):
    """== h_graph.graph(P) as numpy: (x {type: float32 [n, f]}, ei {edge type: int64 [2, E]}, sid int64 [R, K])"""
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
    same = (ent_rk[1:] == ent_rk[:-1])
    a, b = ent_cell[:-1][same], ent_cell[1:][same]
    conn = _uniq2(np.concatenate([np.stack([a, b]), np.stack([b, a])], 1)) if len(a) else np.zeros((2, 0), np.int64)
    src, dst = [], []
    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                m = (sid[:, k1] >= 0) & (sid[:, k2] >= 0)
                src.append(sid[m, k1]); dst.append(sid[m, k2])
    comp = np.stack([np.concatenate(src), np.concatenate(dst)]) if src else np.zeros((2, 0), np.int64)
    e = lambda v: np.ascontiguousarray(v, dtype=np.int64)
    x = {"candidate": np.ascontiguousarray(xc, dtype=np.float32), "grid": np.ascontiguousarray(xg, dtype=np.float32)}
    ei = {TYPES[0]: e(conn), TYPES[1]: e(pt), TYPES[2]: e(pt[::-1]), TYPES[3]: e(comp)}
    return x, ei, sid


def _sd_key(t):
    return "<" + "___".join(t) + ">"           # torch_geometric HeteroConv ModuleDict key


def pack(sd, H=64, nlayers=3):
    """state_dict (name -> float array) -> one float32 vector in gnn_kern.cc's order"""
    f = lambda k: np.asarray(sd[k], np.float32).reshape(-1)
    parts = [f("input_proj.grid.weight"), f("input_proj.grid.bias"), f("input_proj.candidate.weight"), f("input_proj.candidate.bias")]
    for l in range(nlayers):
        for t in TYPES:
            b = f"convs.{l}.convs.{_sd_key(t)}"
            parts += [f(b + ".lin_l.weight"), f(b + ".lin_l.bias"), f(b + ".lin_r.weight")]
    parts += [f("out.weight"), f("out.bias")]
    return np.ascontiguousarray(np.concatenate(parts), np.float32)


class Prior:
    def __init__(self, npz):
        t = time.perf_counter()
        z = np.load(npz)
        self.sd = {k: z[k] for k in z.files}
        self.H = int(self.sd["input_proj.grid.weight"].shape[0])
        self.gin = int(self.sd["input_proj.grid.weight"].shape[1]); self.cin = int(self.sd["input_proj.candidate.weight"].shape[1])
        self.nl = 1 + max(int(k.split(".")[1]) for k in self.sd if k.startswith("convs."))
        self.W = pack(self.sd, self.H, self.nl)
        self.load_s = time.perf_counter() - t
        _lib()

    def forward(self, x, ei, nthreads=8):
        xg, xc = x["grid"], x["candidate"]
        ng, nc = len(xg), len(xc)
        out = np.zeros(nc, np.float32); st = np.zeros(4)
        args = [ng, nc, xg.ctypes.data, xc.ctypes.data]
        keep = []
        for t in TYPES:
            E = ei[t]
            s_, d_ = np.ascontiguousarray(E[0]), np.ascontiguousarray(E[1])
            keep += [s_, d_]
            args += [E.shape[1], s_.ctypes.data, d_.ctypes.data]
        rc = _lib().gnn_forward(*args, self.W.ctypes.data, self.H, self.nl, self.gin, self.cin, int(nthreads), out.ctypes.data, st.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"gnn_forward rc {rc}")
        self.last_s = float(st[0])
        return out

    def logits(self, P, nthreads=8):
        """-> (out float64 [R, K] as the chain's P['gnn_logits'], info)"""
        t = time.perf_counter()
        x, ei, sid = graph_np(P)
        tg = time.perf_counter() - t
        z = self.forward(x, ei, nthreads)
        Lg = np.full(sid.shape, np.float32(NEG), np.float32)
        m = sid >= 0
        Lg[m] = z[sid[m]]
        np.clip(Lg, np.float32(-60.0), np.float32(60.0), out=Lg)
        out = Lg.astype(np.float64)
        out[~(np.asarray(P["cand"]) >= 0)] = -np.inf
        info = {"graph_s": tg, "kern_s": self.last_s, "n_grid": int(len(x["grid"])), "n_cand": int(len(x["candidate"])),
                "edges": [int(ei[t_].shape[1]) for t_ in TYPES], "total_s": time.perf_counter() - t}
        return out, info, (x, ei, sid)


def export(ckpt, out_npz):
    """one-time: checkpoint -> npz of the state_dict (float32), with the same loader the chain uses (dtla_lagnn.load)"""
    import dtla_lagnn as DD
    m = DD.load(ckpt, device="cpu")
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(out_npz, **sd)
    return sorted(sd)


if __name__ == "__main__":                                     # python gnn_np.py export <models dir> <out dir>
    if sys.argv[1] == "export":
        from pathlib import Path
        ROOT = Path(HERE).parents[1]
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "fast_la"))
        import gapla_dp_v2 as V2
        V2._fallback_torch_scatter()
        os.makedirs(sys.argv[3], exist_ok=True)
        for f in sorted(os.listdir(sys.argv[2])):
            if f.startswith("fold_") and f.endswith(".pt"):
                ks = export(os.path.join(sys.argv[2], f), os.path.join(sys.argv[3], f[:-3] + ".npz"))
                print(f, len(ks), ks[:4])
