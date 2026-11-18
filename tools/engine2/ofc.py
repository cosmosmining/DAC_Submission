"""ofc.py: the official-rule overflow cost (tools/offix/of_official.overflow_cost) with a
per-cell cache. The cache holds the demand map and the per-cell loss of the LAST call; a call recomputes the loss only at the
cells whose demand changed since then (same elementwise formula, same numpy ufuncs) and sums each layer's loss array as the
original does (float(loss.sum()) per layer, layers 1..L-1 in order). The loss arrays are elementwise equal to the original's,
so the sums are the same bits. check=True also runs the original and counts mismatches (exact float equality)."""
import numpy as np

CAP_EPS = 1e-3


class OFC:
    def __init__(self, la, ref=None, check=False):
        self.la = la; self.ref = ref; self.check = check
        self.d = None; self.loss = None
        self.calls = 0; self.cells = 0; self.bad = 0

    def _loss(self, d, c):
        with np.errstate(over="ignore"):
            open_ = np.exp(np.minimum(0.25 * d - 0.5 * c, 700.0))
            blocked = np.where(d > 0, np.exp(np.minimum(0.75 * d, 700.0)), 0.0)
            return np.where(c > CAP_EPS, open_, blocked)

    def __call__(self, la=None):
        la = la or self.la
        assert la is self.la
        self.calls += 1
        if self.d is None:
            self.d = np.array(la.d, copy=True)
            self.loss = [None] + [self._loss(la.d[z], la.c[z]) for z in range(1, la.L)]
        else:
            for z in range(1, la.L):
                dz = la.d[z]
                ch = np.nonzero(dz != self.d[z])
                if len(ch[0]):
                    self.cells += len(ch[0])
                    self.loss[z][ch] = self._loss(dz[ch], la.c[z][ch])
                    self.d[z][ch] = dz[ch]
        tot = 0.0
        for z in range(1, la.L):
            tot += float(la.u[z]) * float(self.loss[z].sum())
        if self.check and self.ref is not None:
            r = self.ref(la)
            self.bad += int(r != tot)
        return tot


class OFC2(OFC):
    """--e_ofc 2: the same per-cell loss cache, but the cells to recompute come from touch(rows) (the segments that were
    removed / added since the last call: a superset of the changed cells), and the layer totals are kept as running sums
    (exact full sums at the first call, then += the loss change of the touched cells). Not bit-identical to the full sum:
    the difference is rounding (about 1e-15 relative); check=True compares with the original at every call and records
    the largest relative difference."""
    def __init__(self, la, ref=None, check=False):
        super().__init__(la, ref, check)
        self.pend = []; self.tz = None; self.maxrel = 0.0
        self.dirs = np.asarray(la.dirs, np.int64)

    def touch(self, rows):
        """rows: [m,6] or [m,7] GCell segments (x1, y1, z1, x2, y2, z2) whose demand was added or removed"""
        if rows is None or len(rows) == 0:
            return
        r = np.asarray(rows, np.int64)
        if r.shape[1] == 7:
            r = r[:, 1:]
        self.pend.append(r)

    def _cells(self):
        la = self.la; L, X, Y = la.L, la.X, la.Y
        r = np.concatenate(self.pend, 0); self.pend = []
        x1, y1, z1, x2, y2, z2 = (r[:, k] for k in range(6))
        out = []
        w = z1 == z2
        if w.any():
            zw = z1[w]; dz = self.dirs[np.clip(zw, 0, L - 1)]
            hz = (dz == 0) & (y1[w] == y2[w]) & (x1[w] != x2[w]); vz = (dz == 1) & (x1[w] == x2[w]) & (y1[w] != y2[w])
            for m, lo_, hi_, fx, horiz in ((hz, np.minimum(x1[w], x2[w]), np.maximum(x1[w], x2[w]), y1[w], True),
                                           (vz, np.minimum(y1[w], y2[w]), np.maximum(y1[w], y2[w]), x1[w], False)):
                if not m.any():
                    continue
                lo, hi, f, z = lo_[m], hi_[m], fx[m], zw[m]
                n = hi - lo
                rep = np.repeat(np.arange(len(lo)), n)
                t = lo[rep] + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
                out.append(np.stack([z[rep], t, f[rep]] if horiz else [z[rep], f[rep], t], 1))
        v = ~w
        if v.any():
            zl, zh = np.minimum(z1[v], z2[v]), np.maximum(z1[v], z2[v]); n = zh - zl
            rep = np.repeat(np.arange(len(zl)), n)
            z = zl[rep] + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
            xv, yv = x1[v][rep], y1[v][rep]
            k = z > 0
            z, xv, yv = z[k], xv[k], yv[k]
            dz = self.dirs[np.clip(z, 0, L - 1)]
            a = np.stack([z, np.where(dz == 0, xv - 1, xv), np.where(dz == 0, yv, yv - 1)], 1)
            b = np.stack([z, xv, yv], 1)
            out += [a, b]
        if not out:
            return None
        c = np.concatenate(out, 0)
        ok = (c[:, 0] >= 1) & (c[:, 0] < L) & (c[:, 1] >= 0) & (c[:, 1] < X) & (c[:, 2] >= 0) & (c[:, 2] < Y)
        c = c[ok]
        lin = np.unique((c[:, 0] * X + c[:, 1]) * Y + c[:, 2])
        return lin

    def __call__(self, la=None):
        la = la or self.la
        assert la is self.la
        self.calls += 1
        L, X, Y = la.L, la.X, la.Y
        if self.tz is None:
            self.pend = []
            self.loss = np.zeros((L, X, Y))
            for z in range(1, L):
                self.loss[z] = self._loss(la.d[z], la.c[z])
            self.tz = [0.0] + [float(self.loss[z].sum()) for z in range(1, L)]
        elif self.pend:
            lin = self._cells()
            if lin is not None and len(lin):
                self.cells += len(lin)
                if getattr(self, "cflat", None) is None:
                    self.cflat = np.ascontiguousarray(la.c, np.float64).reshape(-1)     # capacity is constant
                fl = self.loss.reshape(-1)                                          # own contiguous array: a view
                zi = lin // (X * Y); xi = (lin // Y) % X; yi = lin % Y
                new = self._loss(la.d[zi, xi, yi], self.cflat[lin]); old = fl[lin]
                dl = new - old
                ds = np.bincount(zi, weights=dl, minlength=L)
                for z in range(1, L):
                    if ds[z] != 0.0:
                        self.tz[z] += float(ds[z])
                fl[lin] = new
        tot = 0.0
        for z in range(1, L):
            tot += float(la.u[z]) * self.tz[z]
        if self.check and self.ref is not None:
            r = self.ref(la)
            rel = abs(r - tot) / max(abs(r), 1e-300)
            self.maxrel = max(self.maxrel, rel); self.bad += int(rel > 1e-9)
        return tot
