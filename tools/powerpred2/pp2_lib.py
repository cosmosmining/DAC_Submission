#!/usr/bin/env python3
"""pp2_lib.py (powerpred2, numpy only): ranking, capture and selection helpers for the power round, built on the label/feature format of
tools/powerpred/pp_lib.py (feature names NAMES, labels moved / dc / sel). Nothing here reads a server file or a flow object (see pp2_sel.py).

Unit facts used (checked 13 Nov 2026):
  * unit capacitance per layer, fF/um, metal1..metal10 = NG45_C below, copied from timing_promote.py:53-57 (RC, second entry), which is what the
    flow's TimingLA uses as S.la.C (timing_promote.py:158-160); one table for every ISPD 2025 design (NanGate45 setRC.tcl).
  * routing direction per layer index z (0 = metal1): 0,1,0,1,... (0 = horizontal) = the .cap 'dir' field, exposed by the flow as S.la.dirs
    (route_export.py LayerAssigner).
  * the power-round kernel assigns only z >= 1 (tools/tilefix/promote_pw_kern.cc:74), so the cheapest layer of a direction is taken over z >= 1.
Order of sources for C and dirs: the flow's own S.la.C / S.la.dirs (pp2_sel.layer_table), else the NG45 table, else (last resort) an estimate
from the feature columns cap<z> / expm1(wl<z>). Every function that ranks returns or prints which source it used.

'potential' = activity x reducible capacitance. It is the power-term drop of a net if every wire moved to the cheapest layer (z >= 1) of its own
direction, topology unchanged. It is NOT an upper bound of the label dc: the round can also change the topology (FLUTE 'rsmt' candidate), and dc
(kp x net_ctot change) includes pin capacitance; a net with potential 0 can still have dc != 0."""
import numpy as np

LM = 12  # layers kept in the old features (pp_lib.py:17)
NAMES = (["lkp", "lctot", "lcc", "rd", "lw", "slack", "in_o1", "pos", "fan", "lwl", "lvia", "capw", "layer_mean", "layer_top", "act_wl", "r_lkp", "r_lctot", "r_lwl", "r_lcc"]
         + ["wl%d" % z for z in range(LM)] + ["cap%d" % z for z in range(LM)])  # copy of pp_lib.py:18-19

# copy of timing_promote.py:53-57 RC[...][1] (fF/um), metal1..metal10; pp2_selftest checks it against that file's text
NG45_C = np.array([7.41819e-2, 6.74606e-2, 8.88758e-2, 1.07121e-1, 1.08964e-1, 1.02044e-1, 1.10436e-1, 9.69714e-2, 3.6864e-2, 2.8042e-2])
NG45_DIR = np.arange(10) % 2
NG45_SRC = "NanGate45 table (copy of timing_promote.py:53-57; dirs alternating from metal1=H)"
ZMIN = 1  # lowest layer the power-round kernel assigns


def default_layer_dir(n=LM):
    """Alternating 0 (horizontal), 1 (vertical) from index 0 = metal1; matches the .cap files of all ISPD 2025 designs."""
    return np.arange(n) % 2


def cheapest_by_dir(C, dirs, zmin=ZMIN):
    """Per layer: the smallest unit C among the layers z >= zmin with C > 0 (and finite) of the same direction. Layers with C <= 0 or not finite
    are padding / unknown and are never a minimum. A direction with no allowed layer gets cmin = C of the layer itself (no reduction)."""
    c = np.asarray(C, np.float64).ravel(); d = np.asarray(dirs).ravel()
    assert len(c) == len(d), "C and dirs differ in length"
    ok = (np.arange(len(c)) >= zmin) & np.isfinite(c) & (c > 0)
    cmin = c.copy()
    for v in np.unique(d):
        m = d == v; mm = m & ok
        if mm.any():
            cmin[m] = c[mm].min()
    return cmin


def reducible_cap(len_by_layer, C, dirs, zmin=ZMIN):
    """Per net: sum over layers z of len[z] * max(C[z] - cmin[dir z], 0), cmin over layers z >= zmin of the same direction (cheapest_by_dir).
    len_by_layer (n, L') with L' >= L = len(C): extra columns (padding of the old 12-column features) must be all zero; layers with C <= 0
    (padding) must carry no length. Result >= 0, in len units x C units."""
    Lm = np.asarray(len_by_layer, np.float64)
    if Lm.ndim == 1:
        Lm = Lm[None, :]
    c = np.asarray(C, np.float64).ravel(); L = len(c)
    assert Lm.shape[1] >= L, "fewer length columns (%d) than layers (%d)" % (Lm.shape[1], L)
    if Lm.shape[1] > L:
        assert not Lm[:, L:].any(), "length on layer columns >= %d, which the layer table does not have" % L
        Lm = Lm[:, :L]
    pad = ~(np.isfinite(c) & (c > 0))
    if pad.any():
        assert not Lm[:, pad].any(), "length on layers with C <= 0 / unknown: %s" % np.nonzero(pad)[0].tolist()
    cmin = cheapest_by_dir(c, dirs, zmin)
    dlt = np.where(pad, 0.0, np.maximum(c - cmin, 0.0))
    return Lm @ dlt


def lengths_from_features(X, names):
    names = [str(s) for s in names]; X = np.asarray(X, np.float64)
    iw = [names.index("wl%d" % z) for z in range(LM) if "wl%d" % z in names]
    ic = [names.index("cap%d" % z) for z in range(LM) if "cap%d" % z in names]
    L = min(len(iw), len(ic))
    return np.expm1(X[:, iw[:L]]), X[:, ic[:L]]


def estimate_C(WL, CAP):
    """Last-resort estimate of the unit C per layer: sum cap / sum length over the rows. Layers with no length get nan (unknown: never a
    minimum, see cheapest_by_dir) and are listed in the returned note."""
    tl = WL.sum(0); tc = CAP.sum(0)
    c = np.where(tl > 0, tc / np.maximum(tl, 1e-300), np.nan)
    miss = np.nonzero(~(tl > 0))[0].tolist()
    return c, ("estimated from rows; layers without length (unknown C, excluded from the minimum): %s" % miss)


def reducible_cap_from_features(X, names, C="ng45", dirs=None):
    """Reducible cap from the feature columns wl<z> (log1p length) and cap<z> (length x C[z], pp_lib.py:49,66).
    C: an array (exact, e.g. S.la.C saved at label time), 'ng45' (default: NG45_C, the ISPD 2025 table) or 'estimate'.
    dirs: array, default NG45_DIR / alternating. Returns (red, source note). Units: feature length units x fF/um."""
    WL, CAP = lengths_from_features(X, names)
    if isinstance(C, str) and C == "estimate":
        c, src = estimate_C(WL, CAP)                  # unknown layers (nan) carry no length by construction
    elif isinstance(C, str) and C == "ng45":
        c = NG45_C.copy(); src = NG45_SRC
    else:
        c = np.asarray(C, np.float64).ravel()
        nz = np.nonzero(c != 0)[0]
        c = c[:int(nz.max()) + 1] if len(nz) else c   # strip the zero padding of pp_lib (C padded to 12 layers, pp_lib.py:31)
        src = "exact C given (%d layers)" % len(c)
    d = default_layer_dir(len(c)) if dirs is None else np.asarray(dirs).ravel()[:len(c)]
    return reducible_cap(WL, c, d), src


def potential(activity, red_cap):
    """activity x reducible capacitance."""
    return np.asarray(activity, np.float64) * np.asarray(red_cap, np.float64)


def activity_from_features(X, names):
    """activity kp = 10**lkp (pp_lib.py:56; kp = 0 is stored as 1e-30)."""
    return 10.0 ** np.asarray(X, np.float64)[:, [str(s) for s in names].index("lkp")]


def potential_from_features(X, names, C="ng45", dirs=None):
    red, src = reducible_cap_from_features(X, names, C, dirs)
    return potential(activity_from_features(X, names), red), src


def k_at_most(frac, n):
    """Largest net count that is at most frac of n (floor), used by every 'at most x % of the list' check and by the fixed-fraction modes."""
    return int(min(n, max(0, np.floor(frac * n + 1e-9)))) if n else 0


def _order(score):
    return np.argsort(-np.asarray(score, np.float64), kind="stable")


def oracle_recall(label_moved, frac):
    """Largest recall of the moved nets any ranking reaches in the top floor(frac*n) of the list."""
    y = np.asarray(label_moved).astype(bool); n = len(y); m = int(y.sum())
    if m == 0:
        return float("nan")
    return float(min(1.0, k_at_most(frac, n) / m))


def capture_curve(score, weight, fracs):
    """Share of sum(weight) held by the top floor(frac*n) nets by descending score (ties: original order). Weights may be negative (signed gain);
    shares can then exceed 1 or fall. nan when the total weight is <= 0."""
    s = np.asarray(score, np.float64); w = np.asarray(weight, np.float64); n = len(s)
    fr = list(fracs)
    if n == 0:
        return np.full(len(fr), np.nan)
    cw = np.concatenate([[0.0], np.cumsum(w[_order(s)])]); tot = cw[-1]
    if tot <= 0:
        return np.full(len(fr), np.nan)
    return np.array([cw[k_at_most(f, n)] / tot for f in fr])


def smallest_fraction_for(score, weight, share):
    """Smallest k/n such that the top-k by descending score holds >= share of sum(weight) (signed weights allowed: the first k where the running
    sum reaches the target, not a searchsorted on a non-monotone sum). nan if the total is <= 0."""
    assert 0.0 <= share <= 1.0, "share must be in [0, 1]"
    s = np.asarray(score, np.float64); w = np.asarray(weight, np.float64); n = len(s)
    if n == 0:
        return float("nan")
    cw = np.cumsum(w[_order(s)]); tot = cw[-1]
    if tot <= 0:
        return float("nan")
    hit = cw >= share * tot - 1e-12 * abs(tot)
    return (int(np.argmax(hit)) + 1) / n if hit.any() else float("nan")


def choose_prefix(pred, target, forced=None):
    """Selection used by the hook. pred = predicted gain per listed net (>= 0). Nets in 'forced' (bool mask) are always kept; among the others,
    keep the smallest prefix by descending pred whose summed pred >= target x their total (target >= 1: keep all). At least one net is kept.
    Returns (kept indices in ascending original order, fraction of the whole list kept, predicted share kept among the non-forced)."""
    p = np.asarray(pred, np.float64).ravel(); n = len(p)
    if n == 0:
        return np.zeros(0, int), 0.0, float("nan")
    assert np.isfinite(p).all() and (p >= 0).all(), "predicted gains must be finite and >= 0"
    f = np.zeros(n, bool) if forced is None else np.asarray(forced, bool).ravel()
    free = np.nonzero(~f)[0]
    if target >= 1.0 or len(free) == 0:
        return np.arange(n), 1.0, 1.0
    pf = p[free]; o = free[np.argsort(-pf, kind="stable")]; cp = np.cumsum(p[o]); tot = cp[-1]
    if tot <= 0:
        k = 0 if f.any() else 1
        sh = float("nan")
    else:
        k = int(np.argmax(cp >= target * tot - 1e-12 * tot)) + 1
        sh = float(cp[k - 1] / tot)
    keep = f.copy(); keep[o[:k]] = True
    idx = np.nonzero(keep)[0]
    return idx, len(idx) / n, sh


def selection_eval(pred, gain, target, forced=None):
    """Offline proxy of the TIME rule for one round: apply choose_prefix to pred and measure the TRUE gain kept.
    Returns dict(frac=kept share of the list, g_signed=sum gain kept / total signed gain, g_clip=same with gain clipped at 0, neg_share=
    share of |gain| that is negative)."""
    g = np.asarray(gain, np.float64)
    idx, frac, _ = choose_prefix(pred, target, forced)
    tot = g.sum(); gc = np.maximum(g, 0); totc = gc.sum(); ab = np.abs(g).sum()
    return dict(frac=frac, g_signed=float(g[idx].sum() / tot) if tot > 0 else float("nan"),
                g_clip=float(gc[idx].sum() / totc) if totc > 0 else float("nan"),
                neg_share=float(-np.minimum(g, 0).sum() / ab) if ab > 0 else 0.0)
