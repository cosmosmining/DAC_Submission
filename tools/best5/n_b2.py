"""n_b2.py: tools/onepass/build2_fast.py build_fast2 with its per-net loop in C++ on threads (n_b2.cc); same P. install(B2) re-defines B2.build_fast2 from the module's own source with
the loop's text replaced (one place to change it), so the module's globals and the chain's patches on them stay in force. Env N_B2_CHECK=1: the Python loop runs too and every list is compared."""
import ctypes, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_L = ctypes.CDLL(str(HERE / ("libn_b2." + ("dylib" if sys.platform == "darwin" else "so"))))
_P = ctypes.c_void_p; _I = ctypes.c_int64; _i = ctypes.c_int
_L.n_b2_run.argtypes = [_I] + [_P] * 8 + [_P, _i, _P, _i, _i, _i, _P]; _L.n_b2_run.restype = _i
_L.n_b2_get.argtypes = [_P] * 5
T = int(os.environ.get("MID5_THREADS", "8"))
STAT = {"cpp": 0, "fallback": 0, "check": 0, "check_bad": 0}


def run(chosen, idx, pin_idx, net_names_rt, G6, off, pin_off, ap_off, aps, la, H, V, netrc, net_rc_params):
    nn = len(chosen)
    net_i = np.fromiter((idx[k] for k in chosen), np.int64, nn); net_j = np.fromiter((pin_idx[k] for k in chosen), np.int64, nn)
    seg = np.ascontiguousarray(G6, np.int64); off_ = np.ascontiguousarray(off, np.int64)
    po = np.ascontiguousarray(pin_off, np.int64); ao = np.ascontiguousarray(ap_off, np.int64); ap = np.ascontiguousarray(aps, np.int32)
    dirs = np.ascontiguousarray(np.asarray(la.dirs), np.int32)
    Ha = np.array(H, np.int64); Va = np.array(V, np.int64); cnt = np.zeros(4, np.int64)
    rc = _L.n_b2_run(nn, net_i.ctypes.data, net_j.ctypes.data, off_.ctypes.data, seg.ctypes.data, po.ctypes.data, ao.ctypes.data, ap.ctypes.data, dirs.ctypes.data,
                     Ha.ctypes.data, len(H), Va.ctypes.data, len(V), int(la.L), T, cnt.ctypes.data)
    if rc != 0:
        STAT["fallback"] += 1
        return None
    k, r, p, m = (int(v) for v in cnt)
    kept = np.zeros(k, np.int64); noff = np.zeros(k + 1, np.int64); nodes = np.zeros((m, 4), np.int64); runs = np.zeros((r, 8), np.int64); pts = np.zeros((p, 4), np.int64)
    _L.n_b2_get(kept.ctypes.data, noff.ctypes.data, nodes.ctypes.data, runs.ctypes.data, pts.ctypes.data)
    STAT["cpp"] += 1
    kl = kept.tolist(); nl = nodes.tolist(); ol = noff.tolist()
    net_names, net_route_idx, net_nodes, net_cl, net_rd = [], [], [], [], []
    il = net_i.tolist()
    for t, c in enumerate(kl):
        kname = chosen[c]; i = il[c]
        cl, rd = net_rc_params(netrc.get(kname), 5.0, 2.0)
        net_cl.append(cl); net_rd.append(rd)
        net_names.append(net_names_rt[i]); net_route_idx.append(i)
        net_nodes.append({(a, b): (lo, hi) for a, b, lo, hi in nl[ol[t]:ol[t + 1]]})
    return (kept, net_i, runs, pts, net_names, net_route_idx, net_nodes, net_cl, net_rd)


_LOOP_A = "    for kname in chosen:\n        i = idx[kname]\n        name = net_names_rt[i]\n"
_LOOP_B = "        pt_id = {}\n"
_NEW = '''    _n9 = _N_B2.run(chosen, idx, pin_idx, net_names_rt, G6, off, pin_off, ap_off, aps, la, H, V, netrc, net_rc_params)
    if _n9 is None or _N_B2_CHECK:
%s
    if _n9 is not None:
        _k9, _ni9, _r9, _p9, _nn9, _nri9, _nnd9, _cl9, _rd9 = _n9
        if _N_B2_CHECK:
            _ok9 = (net_names == _nn9 and net_route_idx == _nri9 and net_nodes == _nnd9 and [list(x.items()) for x in net_nodes] == [list(x.items()) for x in _nnd9] and net_cl == _cl9 and net_rd == _rd9 and
                    run_net == _r9[:, 0].tolist() and run_dir == _r9[:, 1].tolist() and run_fixed == _r9[:, 2].tolist() and run_lo == _r9[:, 3].tolist() and run_hi == _r9[:, 4].tolist() and
                    run_cur == _r9[:, 5].tolist() and run_p0 == _r9[:, 6].tolist() and run_p1 == _r9[:, 7].tolist() and
                    pt_xy == list(zip(_p9[:, 0].tolist(), _p9[:, 1].tolist())) and pt_lo == _p9[:, 2].tolist() and pt_hi == _p9[:, 3].tolist() and np.array_equal(mask[_ni9[_k9]], np.zeros(len(_k9), np.uint8)))
            _N_B2.STAT["check"] += 1; _N_B2.STAT["check_bad"] += not _ok9
            print("[n_b2 check] equal=%%s nets %%d runs %%d" %% (_ok9, len(_nn9), len(_r9)), flush=True)
        mask[_ni9[_k9]] = 0
        net_names, net_route_idx, net_nodes, net_cl, net_rd = _nn9, _nri9, _nnd9, _cl9, _rd9
        run_net, run_dir, run_fixed, run_lo, run_hi, run_cur, run_p0, run_p1 = (_r9[:, q] for q in range(8))
        run_len = run_hi - run_lo
        pt_xy = list(zip(_p9[:, 0].tolist(), _p9[:, 1].tolist())); pt_lo = _p9[:, 2].tolist(); pt_hi = _p9[:, 3].tolist()
'''


def install(B2):
    if getattr(B2, "_N_B2", None) is not None:
        return
    src = Path(B2.__file__).read_text()
    a = src.index("def build_fast2(")
    b = src.index("build_fast = build_fast2", a)
    f = src[a:b]
    la = f.index(_LOOP_A); lb = f.index(_LOOP_B, la) + len(_LOOP_B)
    loop = "".join("    " + l + "\n" for l in f[la:lb].rstrip("\n").split("\n"))
    f = f[:la] + (_NEW % loop.rstrip("\n")) + f[lb:]
    B2._N_B2 = sys.modules[__name__]; B2._N_B2_CHECK = os.environ.get("N_B2_CHECK") == "1"
    exec(compile(f, B2.__file__, "exec"), B2.__dict__)
