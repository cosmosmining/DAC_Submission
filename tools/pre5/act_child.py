#!/usr/bin/env python3
"""act_child.py: the switching activity of one case in a CHILD process (own interpreter, so its
Python work does not share the chain's GIL). Started by tools/pre5/pre5_load.early() at t0 with the raw case files only:
  python act_child.py <design> <suite> <threads>   -> stdout: b"P5ACT" + int64 n + n float64 densities (per DEF net), then
                                                      one JSON line of counters on stderr
Same code path as the in-thread variant (chain_pre --pre5_act thread): fast_sta_v3.StaDesign through the C++ loader
(tools/speed4/fast_load.install + tools/pre5/pre5_kern.graph), Liberty parsed (no pickle), tools/speed4/act_inproc.run
with the level-parallel kernel tools/pre5/libact_par; deterministic, so the densities are the same bits."""
import json, os, sys, time
T0 = time.perf_counter()
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ["FAST_STA_V3"] = "1"
for p in (ROOT / "tools" / "c1hfast", ROOT / "tools" / "bsgfast", ROOT / "tools" / "tilefix", ROOT / "tools" / "speedcut", ROOT / "tools" / "gapla",
          ROOT / "tools" / "fast_io" / "route_cpp", ROOT / "tools" / "fast_la", ROOT, ROOT / "tools" / "fast_path", ROOT / "tools" / "fast_path"):
    sys.path.insert(0, str(p))                                   # = tools/best2/chain_j.py lines 91-93
for p in ("best", "fast3x", "wave3/ovf2", "ovf3"):
    sys.path.append(str(ROOT / "tools" / p))
sys.path.append(str(ROOT / "tools" / "onepass")); sys.path.append(str(ROOT / "tools" / "speed4")); sys.path.append(str(HERE))
d, s, nt = sys.argv[1], sys.argv[2], int(sys.argv[3])
_OUT = os.fdopen(os.dup(1), "wb")                              # the binary result; everything printed goes to stderr
os.dup2(2, 1); sys.stdout = sys.stderr
os.environ["FASTLOAD_THREADS"] = str(nt)
import numpy as np
import pre5_load as P5L                                         # lazy torch / dtla stand-ins (nothing here uses them)
P5L.T0 = T0
for nm in ("torch", "dtla"):
    sys.modules.setdefault(nm, P5L._Lazy(nm))
import fast_sta_v3 as FS
sys.modules["fast_sta"] = FS
import fast_load as FLD
FLD.install()
import pre5_kern as PK
FS.StaDesign._graph = PK.graph
_ol = FS.StaLib.load
FS.StaLib.load = staticmethod(lambda lib_dir=os.path.join(FS.KIT, "NanGate45/lib"), cache=None, _o=_ol: _o(lib_dir, cache="/dev/null/pre5_no_sta_lib.pkl"))
t1 = time.perf_counter()
SD = FS.StaDesign(d, s, verbose=False)
t2 = time.perf_counter()
dens = P5L._act(SD, nt)
t3 = time.perf_counter()
out = np.ascontiguousarray(dens, np.float64)
_OUT.write(b"P5ACT" + np.int64(len(out)).tobytes() + out.tobytes())
_OUT.flush()
info = dict(P5L.R.get("act_info", {}))
info.update(child_import_s=t1 - T0, child_sd_s=t2 - t1, child_act_s=t3 - t2, child_total_s=time.perf_counter() - T0,
            lazy=list(P5L.R["lazy"]), cpu_s=time.process_time())
sys.stderr.write("P5ACTINFO " + json.dumps(info, default=str) + "\n")
