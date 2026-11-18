#!/usr/bin/env python3
"""gnn_f.py: ctypes of tools/best8/gnn_kern.cu + the network's start of the descent (an internal note): gradient at the one-hot start from the kernels, inputs and graph on threads, forward on the device, logits on the device. No torch."""
import ctypes, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_L = {}


def lib(gpu=True):
    key = "cuda" if gpu else "cpu"
    if key not in _L:
        L = ctypes.CDLL(os.path.join(HERE, f"libgnn_{key}.so"))
        P = ctypes.c_void_p; I = ctypes.c_int64; D = ctypes.c_double
        L.gnn_threads.argtypes = [ctypes.c_int]
        L.gnn_opts.argtypes = [D, ctypes.c_int]
        L.gnn_load.argtypes = [ctypes.c_char_p]; L.gnn_load.restype = P
        L.gnn_unload.argtypes = [P]
        L.gnn_inputs.argtypes = [I, I, I, P, P, P, P, I, P, P, P, P, P, P, P, D, P, P, P, P, P, P, P, P, P]; L.gnn_inputs.restype = ctypes.c_int
        L.gnn_inputs_get.argtypes = [P] * 12
        L.gnn_inputs_set.argtypes = [I, I] + [P] * 10
        L.gnn_forward.argtypes = [P, P, P]; L.gnn_forward.restype = ctypes.c_int
        L.gnn_logits.argtypes = [P, D, D, P]
        _L[key] = L
    return _L[key]


def _a(x, t):
    return np.ascontiguousarray(np.asarray(x), t)


class Net:
    def __init__(self, path, gpu=True, T=8):
        self.L = lib(gpu)
        self.L.gnn_threads(int(T))
        self.h = self.L.gnn_load(os.fsencode(path))
        if not self.h:
            raise RuntimeError("gnn_load " + path)
        self.path = path

    def inputs(self, R, K, L, run, cand2, A, kpn, Cz, um, d0, cap, cell_z, dv, g):
        """A: dgb.cells' dictionary (ent_ptr ec cp ce cvp cvi cvg u copen); -> counts"""
        cnt = np.zeros(6, np.int64); sc = np.zeros(1)
        hold = [_a(run, np.int64), _a(cand2, np.int32), _a(A["ent_ptr"], np.int32), _a(A["ec"], np.int32), _a(A["cp"], np.int32), _a(A["ce"], np.int32), _a(A["cvp"], np.int32), _a(A["cvi"], np.int32), _a(A["cvg"], np.float32), _a(kpn, np.float64), _a(Cz, np.float64),
                _a(d0, np.float32), _a(cap, np.float32), _a(A["u"], np.float32), _a(cell_z, np.int32), _a(A["copen"], np.uint8), _a(dv, np.float32), _a(g, np.float32)]
        ng = len(hold[11])
        rc = self.L.gnn_inputs(R, K, L, hold[0].ctypes.data, hold[1].ctypes.data, hold[2].ctypes.data, hold[3].ctypes.data, ng, hold[4].ctypes.data, hold[5].ctypes.data, hold[6].ctypes.data, hold[7].ctypes.data, hold[8].ctypes.data, hold[9].ctypes.data, hold[10].ctypes.data, float(um),
                               hold[11].ctypes.data, hold[12].ctypes.data, hold[13].ctypes.data, hold[14].ctypes.data, hold[15].ctypes.data, hold[16].ctypes.data, hold[17].ctypes.data, cnt.ctypes.data, sc.ctypes.data)
        if rc != 0:
            raise RuntimeError("gnn_inputs")
        self.R, self.K, self.ng, self.nc, self.E = R, K, int(cnt[5]), int(cnt[0]), cnt[1:5].copy()
        self.scale = float(sc[0])
        return cnt

    def dump(self):
        """the inputs as arrays (tests)"""
        ng, nc, E = self.ng, self.nc, self.E
        xg = np.empty((ng, 5), np.float32); xc = np.empty((nc, 10), np.float32); sid = np.empty(self.R * self.K, np.int32)
        o = [np.empty(ng + 1, np.int32), np.empty(int(E[0]), np.int32), np.empty(ng + 1, np.int32), np.empty(int(E[1]), np.int32), np.empty(nc + 1, np.int32), np.empty(int(E[2]), np.int32), np.empty(nc + 1, np.int32), np.empty(int(E[3]), np.int32)]
        keep = np.empty(ng, np.int32)
        self.L.gnn_inputs_get(keep.ctypes.data, xg.ctypes.data, xc.ctypes.data, sid.ctypes.data, *[v.ctypes.data for v in o])
        return dict(keep=keep, xg=xg, xc=xc, sid=sid.reshape(self.R, self.K), off0=o[0], src0=o[1], off1=o[2], src1=o[3], off2=o[4], src2=o[5], off3=o[6], src3=o[7])

    def set_inputs(self, xg, xc, csr):
        """csr: four (off, src) pairs by destination, in the edge types' order (tests: E2ET's arrays)"""
        xg = _a(xg, np.float32); xc = _a(xc, np.float32); fl = [_a(v, np.int32) for pair in csr for v in pair]
        self.ng, self.nc = len(xg), len(xc)
        self.L.gnn_inputs_set(self.ng, self.nc, xg.ctypes.data, xc.ctypes.data, *[v.ctypes.data for v in fl])

    def forward(self, want=False):
        st = np.zeros(3)
        z = np.empty(self.nc, np.float32) if want else None
        rc = self.L.gnn_forward(self.h, z.ctypes.data if want else None, st.ctypes.data)
        if rc != 0:
            raise RuntimeError("gnn_forward")
        return (z, st) if want else st

    def logits(self, run, init, clamp_c, RL=np.float32):
        run = _a(run, np.int64)
        x = np.empty((self.R, self.K), np.float32)
        self.L.gnn_logits(run.ctypes.data, float(init), float(clamp_c), x.ctypes.data)
        return x.astype(RL, copy=False)


def capture(A, d, c, X, Y):
    """inside the stage's problem build, while the free nets are out of the demand map: the cells' background demand, capacity, layer"""
    k = A["cell_key"]
    z = k // (X * Y); x = (k // Y) % X; y = k % Y
    A["d0"] = d[z, x, y].astype(np.float32); A["cap"] = c[z, x, y].astype(np.float32); A["cell_z"] = z.astype(np.int32)


NETS = {}


def net_of(path, gpu, T):
    if path not in NETS:
        NETS[path] = Net(path, gpu=gpu, T=T)
    return NETS[path]


def start(C, a, X, um, W, path, gpu, clamp=True, init=4.0, sm=None, x0=None):
    """the problem is on the device (dgr_stage.prepare): x = the network's start logits, Adam's state zero. W gets the pieces' seconds"""
    t = time.perf_counter()
    def lap(k, t0):
        n = time.perf_counter(); W[k] = W.get(k, 0.0) + n - t0; return n
    R, Kc = X.I["R"], X.I["K"]; run_ = C.run_; L = C.L
    xx = np.zeros((R, Kc), X.RL); xx[np.arange(R), run_[:, 5]] = 100.0; xx *= X.mask
    X.put("x", xx)
    if X.size.get("g_out") != X.size["x"]:
        X.zeros("g_out", X.size["x"])
    e = X.eval(T=1.0, mode=1, tau=a.dgr_gnn_tau, grad=True)
    g = e["g"]; dv = X.get("dv", X.RL)
    t = lap("gnn_grad_s", t)
    net = net_of(path, gpu, C.T_)
    net.L.gnn_opts(float(getattr(a, "dgr_gnn_sm", 1.0) if sm is None else sm), int(getattr(a, "dgr_gnn_x0", 0) if x0 is None else x0))
    t = lap("gnn_load_s", t)
    A = C.A
    tk = C.tid[C.i_k][C.kn]
    kpn_r = np.where(tk >= 0, C.kpn[np.maximum(tk, 0)], 0.0).astype(np.float32).astype(np.float64)    # float32 as E2ET's problem
    Cz = np.asarray(a._dgr_Cz, np.float32).astype(np.float64)
    cnt = net.inputs(R, Kc, L, run_, C.cand2, A, kpn_r, Cz, um, A["d0"], A["cap"], A["cell_z"], dv, g)
    t = lap("gnn_inputs_s", t)
    st = net.forward()
    W["gnn_up_s"] = W.get("gnn_up_s", 0.0) + float(st[0]); W["gnn_fwd_s"] = W.get("gnn_fwd_s", 0.0) + float(st[1])
    t = lap("gnn_forward_s", t)
    x0 = net.logits(run_, init, 5.0 if clamp else 0.0, X.RL)
    X.put("x", x0)
    X.L.dg_zero(X.p["am"], X.size["am"]); X.L.dg_zero(X.p["av"], X.size["av"])
    t = lap("gnn_put_s", t)
    return dict(F_grad=float(e["total"]), nc=int(cnt[0]), ng=int(net.ng), ng_cells=int(len(A["d0"])), E=[int(v) for v in cnt[1:5]], scale=net.scale)
