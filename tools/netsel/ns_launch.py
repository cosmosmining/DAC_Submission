#!/usr/bin/env python3
"""ns_launch.py: ns_launch.py <driver.py> [driver args...]. Installs the net-selection hook
(tools/netsel/ns_hook.py) on the DTLA stage module tools/best8/dgr_stage_r.py when env NS_MODE is set (log | rule | learned), the
GNN-start hook tools/groupgnn/gg_hook3.py when env GG_GNN is set, then runs the driver as __main__ (as tools/groupgnn/gg_launch3.py
and tools/genroute/ls_launch.py do). With neither variable set it only runs the driver (the "current" arm goes through the same
launcher). Env NS_REPO: the repository root."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NS_REPO", "/path/to/workdir/repo")
P0 = list(sys.path)                     # restored below: the driver sees the path a plain run sees (modules stay imported)
if os.environ.get("GG_GNN"):
    sys.path.insert(0, os.path.join(ROOT, "tools", "groupgnn"))
    import gg_hook3
    gg_hook3.install()
if os.environ.get("NS_MODE"):
    sys.path.insert(0, os.path.join(ROOT, "tools", "best7")); sys.path.insert(0, os.path.join(ROOT, "tools", "best8"))
    import dgr_stage_r
    sys.path.insert(0, HERE)
    import ns_hook
    ns_hook.install(dgr_stage_r)
sys.path[:] = P0
drv = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(drv)
runpy.run_path(drv, run_name="__main__")
