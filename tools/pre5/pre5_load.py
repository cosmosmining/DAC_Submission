#!/usr/bin/env python3
"""pre5_load.py: chain_pre.py --pre5 1, everything before the route is read (an internal note).
Same objects as chain_j's preload (tools/best2/chain_j.py load_all + its __main__ wrapper with --h_pre 1), computed COLD from the
raw case files (no timer cache, no Liberty pickle, no prepared activity file) and in parallel:
  early(ROOT, argv)  at the top of the chain, right after numpy: gc off; lazy stand-ins for `torch` and `dtla` (the real module is
                     imported at the first attribute access, so nothing changes where they are used; when and from where is
                     recorded); thread P starts the C++ parsers of the DEF and the .net (tools/speed4/libfastload.so, OpenMP,
                     --pre5_threads) before the remaining imports.
  load(a, NS)        the preload thread (chain_j load_all order): Liberty (parsed, thread LIB, from t0) | imports | C++ design
                     loader (tools/speed4/fast_load.install: FastStaDesign, .cap, one .net parse shared) | prior weights (numpy) |
                     tech | build2 pre-parse | ChainState | timer bind on the in-memory StaDesign | thread ACT: switching activity in
                     the process (tools/speed4/act_inproc.run with the level-parallel kernel tools/pre5/libact_par) | thread PIN:
                     escape + overflow pin tables | Flute.  The chain waits for ACT only at the first read of S.kpn.
Every piece is the same function chain_j calls (or speed4's checked C++ twin); the order of threads changes no value.
"""
import gc, os, sys, threading, time, types, importlib, traceback
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = {"lazy": {}, "t": {}, "cpu": {}, "thr": {}}              # records -> PRE["pre5"]
EARLY = {}
T0 = None


def _arg(argv, k, d=None):
    for i, v in enumerate(argv):
        if v == k and i + 1 < len(argv):
            return argv[i + 1]
    return d


DEFER = {"torch": ("set_num_threads",)}                     # calls recorded and replayed (in order) when the module is imported
FROMIMP = {"dtla": ("build",)}
PKG = ("torch",)                                             # packages: __path__ (submodule import) imports the real one                               # `from dtla import build` (io_build: reference only) -> lazy attribute


class _LazyAttr:
    def __init__(self, mod, k): self._m, self._k = mod, k
    def _r(self): return getattr(self._m._p5_load(), self._k)
    def __call__(self, *aa, **kk): return self._r()(*aa, **kk)
    def __getattr__(self, k): return getattr(self._r(), k)


class _Lazy(types.ModuleType):
    """module stand-in: the real module is imported at the first attribute access (then every access is forwarded)"""
    def __init__(self, name):
        super().__init__(name)
        self.__dict__["_p5_real"] = None
        self.__dict__["_p5_lock"] = threading.RLock()
        self.__dict__["_p5_calls"] = []

    def _p5_load(self):
        d = self.__dict__
        if d["_p5_real"] is None:
            with d["_p5_lock"]:
                if d["_p5_real"] is None:
                    nm = d["__name__"]
                    t = time.perf_counter()
                    if sys.modules.get(nm) is self:
                        del sys.modules[nm]
                    real = importlib.import_module(nm)
                    for k_, aa_, kk_ in d["_p5_calls"]:          # replay the deferred calls in their order
                        getattr(real, k_)(*aa_, **kk_)
                    d["_p5_real"] = real
                    R["lazy"][nm] = {"at_s": t - T0, "s": time.perf_counter() - t, "thread": threading.current_thread().name,
                                     "by": traceback.format_stack(limit=8)[:-2]}
        return d["_p5_real"]

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__") and not (k == "__path__" and self.__dict__["__name__"] in PKG):   # generic scans (inspect.getmodule: __file__) must not import it
            raise AttributeError(k)
        d = self.__dict__
        if d["_p5_real"] is None:
            nm = d["__name__"]
            if k in DEFER.get(nm, ()):
                def rec(*aa, _k=k, **kk):
                    if d["_p5_real"] is not None:
                        return getattr(d["_p5_real"], _k)(*aa, **kk)
                    d["_p5_calls"].append((_k, aa, kk))
                    R["lazy_deferred"] = R.get("lazy_deferred", []) + [f"{nm}.{_k}{aa}"]
                return rec
            if k in FROMIMP.get(nm, ()):
                return _LazyAttr(self, k)
        return getattr(self._p5_load(), k)

    def __setattr__(self, k, v):
        setattr(self._p5_load(), k, v)


def _ru():
    try:
        import resource
        r = resource.getrusage(getattr(resource, "RUSAGE_THREAD", resource.RUSAGE_SELF))
        return r.ru_stime, r.ru_minflt + r.ru_majflt
    except Exception:
        return 0.0, 0


def _timed(name, fn, *aa, **kk):
    t = time.perf_counter(); c = time.thread_time(); s0, f0 = _ru()
    try:
        return fn(*aa, **kk)
    finally:
        R["t"][name] = R["t"].get(name, 0.0) + time.perf_counter() - t
        R["cpu"][name] = R["cpu"].get(name, 0.0) + time.thread_time() - c
        s1, f1 = _ru()
        R.setdefault("sys", {})[name] = (round(s1 - s0, 3), f1 - f0)
        R.setdefault("at", {})[name] = (round(t - T0, 3), round(time.perf_counter() - T0, 3))


class _Fut:
    def __init__(self, name, fn, *aa):
        self.name, self.v, self.e = name, None, None
        self.th = threading.Thread(target=self._run, args=(fn,) + aa, name=name, daemon=True)
        self.th.start()

    def _run(self, fn, *aa):
        t = time.perf_counter(); c = time.thread_time()
        R["thr"][self.name] = {"start_s": t - T0}
        try:
            self.v = fn(*aa)
        except BaseException as e:
            traceback.print_exc()
            self.e = e
        R["thr"][self.name].update(end_s=time.perf_counter() - T0, wall_s=time.perf_counter() - t, cpu_thread_s=time.thread_time() - c)

    def get(self):
        if self.th.is_alive():
            t = time.perf_counter(); self.th.join()
            R["t"]["wait_" + self.name] = R["t"].get("wait_" + self.name, 0.0) + time.perf_counter() - t
        if self.e is not None:
            raise self.e
        return self.v


def early(root, argv, t0):
    """called once at the top of chain_pre.py when --pre5 1 (before the chain's own imports)"""
    global T0
    T0 = t0
    R["t_early_s"] = time.perf_counter() - T0
    R["cpu_at_early"] = time.process_time()
    gc.disable()                                             # = chain_j --h_pre 1 (gc off in the preload, frozen after)
    for nm in ("torch", "dtla"):                            # lazy: imported at the first use
        if nm not in sys.modules:
            sys.modules[nm] = _Lazy(nm)
    nt = int(_arg(argv, "--pre5_threads", os.environ.get("OMP_NUM_THREADS", "8")))
    os.environ["FASTLOAD_THREADS"] = str(nt)
    EARLY["nt"] = nt
    d, s = _arg(argv, "--design"), _arg(argv, "--suite", "visible")
    base = Path(os.environ.get("ONEPASS_DATA_ROOT") or root) / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks" / s / d
    EARLY["dir"] = base
    sys.path.append(str(root / "tools" / "speed4"))

    def parse():
        import fast_load as FLD                              # tools/speed4 (light imports: numpy, fastload_lib)
        import fastload_lib as FL
        gn = FLD.get_net
        t = time.perf_counter(); df = FL.DefData(str(base / (d + ".def")), nt); R["t"]["P_def_C++"] = time.perf_counter() - t
        t = time.perf_counter(); gn(str(base / (d + ".net"))); R["t"]["P_net_C++"] = time.perf_counter() - t
        return df
    EARLY["P"] = _Fut("P", parse)

    def lib():
        import fast_sta_v3 as FS                             # cold: parsed from the .lib files (no pickle)
        return FS.StaLib.load(os.path.join(FS.KIT, "NanGate45/lib"), cache="/dev/null/pre5_no_sta_lib.pkl")
    EARLY["LIB"] = _Fut("LIB", lib)
    if _arg(argv, "--pre5_act", "child") == "child":         # activity in a child process from t0 (own GIL)
        import subprocess
        t = time.perf_counter()
        nta = int(_arg(argv, "--pre5_act_threads", "0")) or max(2, nt // 2)
        R["act_child_threads"] = nta
        pr = subprocess.Popen([sys.executable, "-u", str(HERE / "act_child.py"), d, s, str(nta)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        R["t"]["act_child_spawn"] = time.perf_counter() - t

        def child():
            import json as _j
            out, err = pr.communicate()
            if pr.returncode != 0 or not out.startswith(b"P5ACT"):
                raise RuntimeError("act_child failed rc=%s: %s" % (pr.returncode, err.decode(errors="replace")[-2000:]))
            n = int(np.frombuffer(out[5:13], np.int64)[0])
            dens = np.frombuffer(out[13:13 + 8 * n], np.float64).copy()
            for ln in err.decode(errors="replace").splitlines():
                if ln.startswith("P5ACTINFO "):
                    R["act_info"] = _j.loads(ln[10:])
            return dens
        EARLY["ACT"] = _Fut("ACT", child)


def _install_fast(NS, a):
    """tools/speed4/fast_load.install + the pre-parsed DEF / .net / Liberty handed to the C++ design loader"""
    import fast_sta_v3 as FS
    sys.modules["fast_sta"] = FS
    import fast_load as FLD
    import fastload_lib as FL
    FLD.install()
    defp = os.path.realpath(str(EARLY["dir"] / (a.design + ".def")))
    _oD = FL.DefData

    def DefData(path, nt=FL.NT, _o=_oD):
        if os.path.realpath(path) == defp and "P" in EARLY:
            df = EARLY.pop("P").get()
            return df
        return _o(path, nt)
    FL.DefData = DefData
    _og = FLD.get_net

    def get_net(path, _o=_og):
        f = EARLY.get("P")
        if f is not None and f.th is not threading.current_thread():   # the early thread fills fast_load._NET[path]
            f.th.join()
        return _o(path)
    FLD.get_net = get_net
    _ol = FS.StaLib.load
    libdir = os.path.realpath(os.path.join(FS.KIT, "NanGate45/lib"))

    def load(lib_dir=os.path.join(FS.KIT, "NanGate45/lib"), cache=None, _o=_ol):
        if os.path.realpath(lib_dir) == libdir and "LIB" in EARLY:
            return EARLY["LIB"].get()                       # the same parse, made once in this process
        return _o(lib_dir, cache="/dev/null/pre5_no_sta_lib.pkl")
    FS.StaLib.load = staticmethod(load)
    sys.path.append(str(HERE))
    import pre5_kern as PK
    EARLY["graph_orig"] = FS.StaDesign._graph
    FS.StaDesign._graph = PK.graph                           # FastStaDesign only: C++ levelize / SCC (exact)
    import ispd25_adapter as AD                              # ChainState's .cap parse (C++, GIL free) started now, used once
    _pc = AD.parse_cap_auto
    capp = os.path.realpath(str(EARLY["dir"] / (a.design + ".cap")))
    EARLY["CAP"] = _Fut("CAP", _pc, str(EARLY["dir"] / (a.design + ".cap")))

    def parse_cap_auto(path, _o=_pc):
        if os.path.realpath(path) == capp and "CAP" in EARLY:
            return EARLY.pop("CAP").get()
        return _o(path)
    AD.parse_cap_auto = parse_cap_auto
    return FLD


def _cold_state(Base, mode):
    """kpn read lazily from the activity thread (as chain_s4 --s4_act / cold_j lazy); no prepared activity file"""
    if mode == "file":
        return Base

    class ColdState(Base):
        def activity_file(self, path):                       # the prepared file is NOT read (cold); recorded only
            R["act_file_skipped"] = str(path)

        def _get_kpn(self):
            d = self.__dict__
            if d.get("_kpn_real") is not None:
                return d["_kpn_real"]
            t = time.perf_counter()
            dens = EARLY["ACT"].get()
            R.setdefault("act_needed_at_s", t - T0); R.setdefault("act_wait_s", time.perf_counter() - t)
            d["_kpn_real"] = None
            self.set_activity(dens)
            return d["_kpn_real"]

        def _set_kpn(self, v):
            self.__dict__["_kpn_real"] = v
        kpn = property(_get_kpn, _set_kpn)
    return ColdState


def _act(SD, nt):
    import act_inproc as AI                                  # tools/speed4 (exact arrays), kernel: tools/pre5/libact_par
    import ctypes
    if not getattr(AI, "_p5", False):
        L = ctypes.CDLL(str(HERE / ("libact_par." + ("dylib" if sys.platform == "darwin" else "so"))))
        V = ctypes.c_void_p
        L.act_run.restype = ctypes.c_int
        L.act_run.argtypes = [ctypes.c_int64] + [V] * 10 + [ctypes.c_int64, V] + [V] * 10 + [ctypes.c_int64] + [V] * 6 + [ctypes.c_int] + [V] * 4
        L.act_clkmask.restype = ctypes.c_int
        L.act_clkmask.argtypes = [ctypes.c_int64, V, V, V]
        L.act_density.argtypes = [ctypes.c_int64, V, V, V]
        L.act_set_threads.argtypes = [ctypes.c_int]
        AI._LIB = L; AI._p5 = True
    AI._LIB.act_set_threads(int(nt))
    dens, info = AI.run(SD)
    R["act_info"] = {k: v for k, v in info.items() if not k.startswith("_")}
    return dens


def load(a, NS):
    """chain_j load_all (+ its --h_pre wrapper) for --pre5 1; fills NS['PRE'] / NS['G'] with the same keys"""
    if os.environ.get("PRE5_PROF"):                          # test: cProfile of the preload thread -> $PRE5_PROF
        import cProfile
        pr = cProfile.Profile(); pr.enable()
        try:
            return _load(a, NS)
        finally:
            pr.disable(); pr.dump_stats(os.environ["PRE5_PROF"])
    return _load(a, NS)


def _load(a, NS):
    PRE, G, H, ITEMS = NS["PRE"], NS["G"], NS["H"], NS["ITEMS"]
    ROOT, BEST = NS["ROOT"], NS["BEST"]
    t = time.perf_counter(); c0 = time.process_time()
    R["preload_start_s"] = t - T0
    nt = EARLY["nt"]
    try:
        FLD = _timed("install_fastload", _install_fast, NS, a)       # imports fastmode4 / ispd25_adapter / build2_fast
        PRE["s4_fastload"] = FLD
        ChainState = _timed("make_state_class", NS["make_state_class"])
        ChainState = _cold_state(ChainState, a.pre5_act)
        NS["ChainState"] = ChainState
        import fast_sta_v3
        sys.modules["fast_sta"] = fast_sta_v3

        def imports():
            import io_run as IR
            G["IR"] = IR; G["B"], G["V1"], G["V2"] = IR.B, IR.V1, IR.V2
            if os.environ.get("ONEPASS_DATA_ROOT"):
                _r = Path(os.environ["ONEPASS_DATA_ROOT"])
                IR.V1.DEFAULT_DATA_ROOT = _r; IR.V2.DEFAULT_DATA_ROOT = _r; IR.V2.DEFAULT_GNN_ROOT = _r / "experiments" / "fastgnn_lofo" / "models"
            import dp_split_api as DS
            G["DS"] = DS
            import fastmode4 as F4, fastmode as FM
            G["F4"], G["FM"] = F4, FM
            import csta as CS, fast_timer as FT
            G["CS"], G["FT"] = CS, FT
            import promote_cpp as PC
            G["PC"] = PC
            from route_cpp import read_route
            G["read_route"] = read_route
            if H["read"]:
                import h_route_read as HRR
                G["read_route"] = lambda p, _f=read_route: HRR.read_route(p, 8, fallback=_f)
                G["read_route_orig"] = read_route
            return IR
        IR = _timed("imports", imports)
        PRE["t_imports"] = time.perf_counter() - t
        bench = IR.V1.DEFAULT_DATA_ROOT / "ispd_bench" / "ISPD2025" / "ISPD2025_benchmarks"
        if a.lam > 0 or a.gate > 0:
            for _kv in (a.gnn_family or []):
                _d_, _f_ = _kv.split("=")
                IR.V2.FAMILY_FOR_DESIGN.setdefault(_d_, _f_)
            fam = IR.V2.family_for(a.design)
            if a.pre5_gnn in ("np", "check"):
                sys.path.append(str(HERE))
                import gnn_np
                PRE["prior"] = _timed("prior_weights", gnn_np.Prior, str(HERE / "weights" / f"fold_{fam}.npz"))
            if a.pre5_gnn in ("torch", "check"):                # chain_j's path (torch + torch_geometric)
                def tmodel():
                    IR.V2._fallback_torch_scatter()
                    import torch
                    torch.set_num_threads(8)
                    import dtla_lagnn as DD
                    PRE["DD"] = DD
                    PRE["model"] = DD.load(str(IR.V2.DEFAULT_GNN_ROOT / f"fold_{fam}.pt"), device="cpu")
                _timed("torch_model", tmodel)
        PRE["tech"] = _timed("tech", G["FT"].read_tech, str(bench))
        if a.build == "b2":
            def b2():
                sys.path.insert(0, str(ROOT / "tools" / "onepass"))
                import build2_fast as B2
                G["B2"] = B2
                _cp = os.path.join(B2.BENCH_ROOT, a.suite, a.design, a.design + ".cap"); _np = os.path.join(B2.BENCH_ROOT, a.suite, a.design, a.design + ".net")
                t2 = time.perf_counter(); _cap = B2.parse_cap_fast(_cp); _net = B2.parse_net_fast(_np)
                if H["pre"]:
                    PRE["h_net"] = (_np, _net)
                _la = B2.TimingLA(_cap, 0.0, B2.GRID / 2000.0, 5.0, 2.0)
                EARLY["XY"] = (_la.X, _la.Y)
                PRE["t_preparse"] = time.perf_counter() - t2
                _OTL = B2.TimingLA
                def _TL(cap, *aa, _c=_cap, _l=[_la], _o=_OTL):
                    return _l.pop() if (cap is _c and _l) else _o(cap, *aa)
                B2.TimingLA = _TL
                _oc, _on = B2.parse_cap_fast, B2.parse_net_fast
                B2.parse_cap_fast = lambda p, _c=_cap, _p=_cp, _o=_oc: _c if p == _p else _o(p)
                B2.parse_net_fast = lambda p, _c=_net, _p=_np, _o=_on: _c if p == _p else _o(p)
            _timed("b2_preparse", b2)
        t_i2 = time.perf_counter()
        if H["inc"] or H["crit"] or a.ovf_timer == "inc":       # import with chain_j's sys.path change before the escape imports (chain_j order)
            _sp = list(sys.path)
            sys.path.append(str(ROOT / "tools" / "engine"))
            try:
                import csta_inc as CI
            finally:
                sys.path[:] = _sp
            G["CI"] = CI

        if a.escape != "off" or a.ovf != "off":                 # module imports here (this thread), the tables in thread PIN
            import escape3d_fast                                   # patches E.dijkstra/marginal/apply
            import escape3d as E
            G["E"] = E
            import h_netpins                                       # noqa: F401
            if a.ovf != "off":
                import ovf_lib as OL
                G["OL"] = OL
                if a.ovf_impl == "cpp":
                    import ovf2_lib as OVF2
                    G["OVF2"] = OVF2
            import pre5_kern as PK                                 # noqa: F401
        R["t"]["imports2"] = time.perf_counter() - t_i2

        def pins():
            if not (a.escape != "off" or a.ovf != "off"):
                return
            t3 = time.perf_counter()
            E = G["E"]
            if 5 not in ITEMS or a.ovf != "off":
                _nf = os.path.join(E.BENCH_ROOT, a.suite, a.design, a.design + ".net")
                if H["pre"] and "h_net" in PRE and os.path.samefile(_nf, PRE["h_net"][0]):
                    PRE["esc_pins"] = sys.modules["h_netpins"].esc_pins_from_fast(E, PRE["h_net"][1])
                else:
                    PRE["esc_pins"] = {E._key(n["name"]): n["pins"] for n in E.parse_net_ispd25(_nf)}
            PRE["t_esc_pre"] = time.perf_counter() - t3
            if a.ovf != "off":
                t4 = time.perf_counter()
                OL = G["OL"]
                X_, Y_ = EARLY["XY"]                               # == S.la.X, S.la.Y (same .cap; checked with --pre5_check)
                if a.pre5_pcmap and "h_net" in PRE:
                    PRE["ovf_pcmap"] = sys.modules["pre5_kern"].pcmap(PRE["h_net"][1], X_, Y_, E._key)
                else:
                    PRE["ovf_pcmap"] = OL.pin_count_map(PRE["esc_pins"], X_, Y_)
                PRE["ovf_npins_key"] = {k: len(v) for k, v in PRE["esc_pins"].items()}
                PRE["t_ovf_pre"] = time.perf_counter() - t4
        PRE["t_csta"] = time.perf_counter() - t
        S = _timed("state", ChainState, a.design, a.suite)       # FastStaDesign (C++), cap, TimingLA, tables
        PRE["S"] = S
        fp = _Fut("PIN", pins)                                    # after the state: a CPU-bound Python thread next to the
                                                                  # state's many short GIL releases costs ~5 ms per release
        if 5 not in ITEMS and a.escape != "off":                 # (chain_j: demand model of the escape in every run; off: item 5)
            from timing_promote import TimingLA as _TLA, GRID as _GR
            PRE["esc_la"] = _TLA(PRE["S"].cap, 0.0, _GR / 2000.0, 5.0, 2.0)
        if a.pre5_act == "thread":
            EARLY["ACT"] = _Fut("ACT", _act, S.SD, nt)
        cd = S.SD                                                  # cold: the timer binds the in-memory StaDesign

        def kernel():
            CS = G["CS"]
            if H["inc"] or H["crit"] or a.ovf_timer == "inc":
                CI = G["CI"]
                _hl = str(BEST / ("libi_kern." + ("dylib" if sys.platform == "darwin" else "so")))
                K = CI.IncKernel(_hl); K.bind_design(cd); PRE["inc_info"] = K.bind_rc_design(cd)
                K.L.w3_inc_backward.argtypes = [__import__("ctypes").c_void_p]
                if (H["inc"] or a.ovf_timer == "inc") and not K.inc_ok:
                    raise RuntimeError(f"incremental timer refused this design: {PRE['inc_info']}")
            else:
                K = CS.Kernel(); K.bind_design(cd); K.bind_rc_design(cd)
            PRE["cd"], PRE["K"] = cd, K
        _timed("kernel_bind", kernel)
        from rsmt_route import Flute
        PRE["fl"] = _timed("flute", Flute)
        PRE["t_state"] = time.perf_counter() - t
        fp.get()
        if a.pre5_check:                                          # test: the replaced pieces against the originals
            ck = R.setdefault("check", {})
            import copy
            sd2 = copy.copy(S.SD); EARLY["graph_orig"](sd2)
            keys = [k for k in vars(sd2) if k not in ("lib",)]
            bad = [k for k in keys if isinstance(getattr(sd2, k), np.ndarray) and not (getattr(sd2, k).dtype == getattr(S.SD, k).dtype and np.array_equal(getattr(sd2, k), getattr(S.SD, k)))]
            bad += [k for k in keys if not isinstance(getattr(sd2, k), np.ndarray) and isinstance(getattr(sd2, k), (int, float, list)) and getattr(sd2, k) != getattr(S.SD, k)]
            ck["graph_bad"] = bad
            ck["xy"] = [list(EARLY["XY"]), [S.la.X, S.la.Y]]
            if "ACT" in EARLY and PRE.get("act_file"):             # activity vs the prepared file chain_j reads
                class _D:
                    pass
                o_ = _D(); o_.SD = S.SD; o_.set_activity = lambda v, _o=o_: setattr(_o, "dens", np.asarray(v, float))
                FLD.fast_activity_file(o_, PRE["act_file"])
                dn = EARLY["ACT"].get()
                ck["act_vs_file"] = {"n": int(len(dn)), "n_diff": int((dn != o_.dens).sum()),
                                     "max_rel": float((np.abs(dn - o_.dens) / np.maximum(np.abs(o_.dens), 1e-300)).max()) if len(dn) else 0.0}
            if "ovf_pcmap" in PRE:
                import ovf_lib as OL
                ref = OL.pin_count_map(PRE["esc_pins"], S.la.X, S.la.Y)
                ck["pcmap_equal"] = bool(ref.dtype == PRE["ovf_pcmap"].dtype and np.array_equal(ref, PRE["ovf_pcmap"]))
        if a.pre5_preimp:                                      # post-route imports moved before the route (no value changes)
            def preimp():
                import route_export, timing_promote, fastmode3, fast_apply   # noqa: F401
                sys.path.append(str(ROOT / "tools" / "offix"))
                import of_official                                           # noqa: F401
                if a.promote_impl == "full":
                    import promote_full                                      # noqa: F401
                import h_dem, h_asm                                          # noqa: F401
            _timed("preimports", preimp)
    except BaseException as e:
        traceback.print_exc()
        PRE["err"] = repr(e)
    PRE["preload_s"] = time.perf_counter() - t
    try:
        R["fastload_stats"] = {k: round(v, 3) for k, v in sys.modules["fast_load"].STATS.items()}
        R["fastload_fallbacks"] = list(sys.modules["fast_load"].FALLBACKS)
    except Exception:
        pass
    PRE.pop("h_net", None)
    gc.freeze(); gc.enable()
    PRE["preload_end"] = time.perf_counter() - T0
    R["preload_cpu_s"] = time.process_time() - c0
    R["preload_threads_used"] = R["preload_cpu_s"] / max(PRE["preload_s"], 1e-9)
    PRE["pre5"] = R
