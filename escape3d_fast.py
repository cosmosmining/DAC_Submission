#!/usr/bin/env python3
"""escape3d_fast.py — escape3d.py with repair3d_astar.py's exact fast pieces (new file; escape3d.py unchanged):
A* window search (same optimal costs), the via-demand rule tested on the via's own track (same demand, linear in
net size) and numpy-summed marginal costs of big nets. Same arguments and output as escape3d.py; fastmode8.py
--escape_fast uses it (ariane_b: escape pass 1 took 240 s with escape3d.py).

  python escape3d_fast.py --design ariane --suite blind --route <in> --out <out> --passes 2
"""
import sys

import escape3d as E                                         # READ-ONLY reuse
import repair3d_astar as RA

E.dijkstra, E.marginal, E.apply = RA.astar_c, RA.marginal_fast, RA.apply_fast   # names escape3d looks up

if __name__ == "__main__":
    sys.exit(E.main())
