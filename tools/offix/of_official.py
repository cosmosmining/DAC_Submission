#!/usr/bin/env python3
"""of_official.py: the overflow cost with the OFFICIAL rule for blocked cells (new file, 5 Nov 2026).

Finding (5 Nov 2026, experiments/congmap arrays, six small cases): fastmode.overflow_cost charges exp(0.75*0) = 1 for every
blocked cell (capacity 0) that carries NO demand. The organisers' evaluator charges such a cell nothing
(ispd_bench/ISPD2025/ISPD2025_benchmarks/evaluation/evaluator.cpp lines 631-636: the blocked branch requires demand > 0).
With the official rule our demand model reproduces the official overflow number to 4 significant digits on all six cases
(for example ariane visible c4: 6.794 M against the evaluator's 6.795 M; the old function gave 11.674 M).
Effect of the old rule on the in-run estimate:
  * a constant floor of u_z per idle blocked cell (1.0 to 11.1 M cost units, i.e. 0.7 to 1.7 score units), harmless by itself;
  * a BIAS in score changes: a move that puts the first demand on an idle blocked cell is under-charged by u_z, and a move
    that empties a blocked cell is under-rewarded by u_z. Summed over the c4 recipe the estimator under-counted the overflow
    our moves add by 0.003 to 0.032 score units per case.
The marginal costs inside the DP, promotion and escape kernels already use the official rule (base 0 for an idle blocked
cell); only the total used by the accept test was off.

Use:
    import of_official; of_official.patch()      # after `import fastmode`; replaces fastmode.overflow_cost
or call of_official.overflow_cost(la) directly. Callers that did `from fastmode import overflow_cost` before the patch keep
the old function; patch before those imports or call this one explicitly.
"""
import numpy as np

CAP_EPS = 1e-3


def overflow_cost(la):
    """official evaluator overflow cost from the demand model (layers 1..L-1; demand in half units)."""
    tot = 0.0
    for z in range(1, la.L):
        d, c = la.d[z], la.c[z]
        with np.errstate(over="ignore"):
            open_ = np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0))
            blocked = np.where(d > 0, np.exp(np.minimum(0.75 * d, 700.0)), 0.0)
            loss = np.where(c > CAP_EPS, open_, blocked)
        tot += float(la.u[z]) * float(loss.sum())
    return tot


def patch():
    import fastmode
    if getattr(fastmode, "_of_official_patched", False):
        return
    fastmode._overflow_cost_floor = fastmode.overflow_cost
    fastmode.overflow_cost = overflow_cost
    fastmode._of_official_patched = True


if __name__ == "__main__":
    import glob, os, sys, types
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "congmap", "*.npz"))):
        z = np.load(f)
        la = types.SimpleNamespace(d=z["c4_d"].astype(np.float64), c=z["c"].astype(np.float64), u=z["u"], L=z["c4_d"].shape[0])
        print("%-34s official-rule overflow of the c4 route: %.6g" % (os.path.basename(f)[:-4], overflow_cost(la)))
