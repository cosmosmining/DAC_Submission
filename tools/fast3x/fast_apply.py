"""fast_apply.py: linear-time versions of timing_promote.apply and
route_export.LayerAssigner.commit_net.

The original functions test every via against EVERY wire of the net on the via's layer (quadratic on giant nets).
A wire (z, f, a, b) can only cover a via at (x, y, z) when f is the via's own track (f == y on a horizontal layer,
f == x on a vertical one), and every wire of track (z, f) lives in wires[(z, f)]. So the test is restricted to that
one list (bisect on long lists). The demand updates are made in exactly the same order with the same values, so
la.d is bit-identical (all updates are +-1/+-2 on float64 anyway).
"""
from bisect import bisect_right

_LONG = 8          # lists longer than this get a sorted (lo, prefix-max hi) index


class _Cover:
    """closed-interval cover test on the via's own track, lazily indexed per (z, fixed)."""
    __slots__ = ("wires", "idx")

    def __init__(self, wires):
        self.wires = wires
        self.idx = {}

    def hit(self, key, c):
        ivs = self.wires.get(key)
        if not ivs:
            return False
        if len(ivs) <= _LONG:
            for lo, hi in ivs:
                if lo <= c <= hi:
                    return True
            return False
        ix = self.idx.get(key)
        if ix is None:
            s = sorted(ivs)
            los = [lo for lo, _ in s]
            pm, m = [], None
            for _, hi in s:
                m = hi if m is None or hi > m else m
                pm.append(m)
            ix = self.idx[key] = (los, pm)
        los, pm = ix
        k = bisect_right(los, c)                    # intervals with lo <= c are s[:k]
        return k > 0 and pm[k - 1] >= c


def apply(la, wires, vias, sign):
    """timing_promote.apply, linear via test (identical demand updates in identical order)."""
    X, Y, d, dirs = la.X, la.Y, la.d, la.dirs
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            if dirs[z] == 0:
                d[z, a:b, fixed] += 2 * sign
            else:
                d[z, fixed, a:b] += 2 * sign
    cov = None
    for (x, y, z) in vias:
        if z == 0:
            continue
        dz = dirs[z]
        if dz == 0:
            if cov is None: cov = _Cover(wires)
            if cov.hit((z, y), x):
                continue
            if 0 < x < X - 1:
                d[z, x - 1, y] += sign
                d[z, x, y] += sign
            elif x > 0:
                d[z, x - 1, y] += 2 * sign
            elif x < X - 1:
                d[z, x, y] += 2 * sign
        elif dz == 1:
            if cov is None: cov = _Cover(wires)
            if cov.hit((z, x), y):
                continue
            if 0 < y < Y - 1:
                d[z, x, y - 1] += sign
                d[z, x, y] += sign
            elif y > 0:
                d[z, x, y - 1] += 2 * sign
            elif y < Y - 1:
                d[z, x, y] += 2 * sign


def commit_net(self, wires, vias):
    """route_export.LayerAssigner.commit_net, linear via test."""
    X, Y, d, dirs = self.X, self.Y, self.d, self.dirs
    for (z, fixed), ivs in wires.items():
        for a, b in ivs:
            if dirs[z] == 0:
                d[z, a:b, fixed] += 2
            else:
                d[z, fixed, a:b] += 2
    cov = None
    for (x, y, z) in vias:
        if z == 0:
            continue
        dz = dirs[z]
        if dz == 0:
            if cov is None: cov = _Cover(wires)
            if cov.hit((z, y), x):
                continue
            if 0 < x < X - 1:
                d[z, x - 1, y] += 1
                d[z, x, y] += 1
            elif x > 0:
                d[z, x - 1, y] += 2
            elif x < X - 1:
                d[z, x, y] += 2
        elif dz == 1:
            if cov is None: cov = _Cover(wires)
            if cov.hit((z, x), y):
                continue
            if 0 < y < Y - 1:
                d[z, x, y - 1] += 1
                d[z, x, y] += 1
            elif y > 0:
                d[z, x, y - 1] += 2
            elif y < Y - 1:
                d[z, x, y] += 2


class _Shadow:
    """test mode: a second demand array updated with the ORIGINAL functions; compared with la.d after every round."""
    def __init__(self, la):
        self.X, self.Y, self.dirs, self.L = la.X, la.Y, la.dirs, la.L
        self.d = la.d.copy()


def install(TP, RE, modules, shadow=None):
    """patch timing_promote.apply (and the copies imported by name into `modules`) and LayerAssigner.commit_net.
    shadow: a _Shadow; when given, every call is mirrored with the original function on shadow.d."""
    o_apply, o_commit = TP.apply, RE.LayerAssigner.commit_net
    if shadow is None:
        new_apply, new_commit = apply, commit_net
    else:
        def new_apply(la, wires, vias, sign, _o=o_apply, _s=shadow):
            if la.d is not _s.d and getattr(_s, "la", None) is la:
                _o(_s, wires, vias, sign)
            apply(la, wires, vias, sign)

        def new_commit(self, wires, vias, _o=o_commit, _s=shadow):
            if getattr(_s, "la", None) is self:
                _o(_s, wires, vias)
            commit_net(self, wires, vias)
    TP.apply = new_apply
    for m in modules:
        if getattr(m, "apply", None) is o_apply:
            m.apply = new_apply
    RE.LayerAssigner.commit_net = new_commit
    return o_apply, o_commit
