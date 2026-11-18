// astar3d_win.cpp: tools/astar3d.cpp (repair3d_astar.astar in C) with the same costs,
// the same heap order and therefore the same path, made faster for the overflow stage of tools/ovf3/chain_ovf.py:
//  * window-local buffers (dist / prev / stamps indexed inside the search window, not over the whole L x X x Y grid):
//    the explored nodes are contiguous in memory instead of scattered over a 40 MB array;
//  * per-call caches of the wire-edge cost of a cell and of the via-edge cost of a (x, y, z) (each was recomputed by
//    every neighbour, 12 exp() per pop); the cached value is the same double, so results are bit-identical.
// Falls back to grid-global indexing when a source lies outside the window (never in escape / repair use).
//   macOS: clang++ -O3 -std=c++17 -shared -fPIC -o tools/ovf3/libastar3d_win_darwin.so tools/ovf3/astar3d_win.cpp
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>
#include <algorithm>
#include <limits>

namespace {
constexpr double CAP_EPS = 1e-3;
constexpr double CLIP = 700.0;

struct Grid {
    int L, X, Y;
    const double *d, *c, *u;
    const int32_t *dirs;
    int64_t dz, dx, dy, cz, cx, cy;
    inline double D(int z, int x, int y) const { return d[z * dz + x * dx + y * dy]; }
    inline double C(int z, int x, int y) const { return c[z * cz + x * cx + y * cy]; }
};

inline double cell_cost(double u, double d, double c) {
    if (c > CAP_EPS) return u * std::exp(std::min(0.25 * d - 0.5 * c, CLIP));
    return d > 0 ? u * std::exp(std::min(0.75 * d, CLIP)) : 0.0;
}

inline double edge_wire(const Grid &g, int z, int x, int y) {
    double d = g.D(z, x, y), c = g.C(z, x, y), u = g.u[z];
    return cell_cost(u, d + 2, c) - cell_cost(u, d, c);
}

inline double edge_via(const Grid &g, int x, int y, int z) {
    if (z == 0) return 0.0;
    int X = g.X, Y = g.Y;
    int cx[2], cy[2], dd[2], n = 0;
    if (g.dirs[z] == 0) {
        if (0 < x && x < X - 1) { cx[0] = x - 1; cy[0] = y; dd[0] = 1; cx[1] = x; cy[1] = y; dd[1] = 1; n = 2; }
        else if (x > 0) { cx[0] = x - 1; cy[0] = y; dd[0] = 2; n = 1; }
        else { cx[0] = x; cy[0] = y; dd[0] = 2; n = 1; }
    } else {
        if (0 < y && y < Y - 1) { cx[0] = x; cy[0] = y - 1; dd[0] = 1; cx[1] = x; cy[1] = y; dd[1] = 1; n = 2; }
        else if (y > 0) { cx[0] = x; cy[0] = y - 1; dd[0] = 2; n = 1; }
        else { cx[0] = x; cy[0] = y; dd[0] = 2; n = 1; }
    }
    double tot = 0.0, u = g.u[z];
    for (int k = 0; k < n; ++k) {
        double d = g.D(z, cx[k], cy[k]), c = g.C(z, cx[k], cy[k]);
        tot += cell_cost(u, d + dd[k], c) - cell_cost(u, d, c);
    }
    return tot;
}

struct Item {
    double f, g;
    int z, x, y;
    bool operator>(const Item &o) const {
        if (f != o.f) return f > o.f;
        if (g != o.g) return g > o.g;
        if (z != o.z) return z > o.z;
        if (x != o.x) return x > o.x;
        return y > o.y;
    }
};

std::vector<double> g_dist, g_ew, g_ev;
std::vector<int64_t> g_prev;
std::vector<uint32_t> g_seen, g_tgt, g_ews, g_evs;
std::vector<Item> g_heap;
uint32_t g_stamp = 0;

void ensure(int64_t N) {
    if ((int64_t)g_dist.size() < N) {
        g_dist.assign(N, 0.0); g_prev.assign(N, -1); g_seen.assign(N, 0); g_tgt.assign(N, 0);
        g_ew.assign(N, 0.0); g_ev.assign(N, 0.0); g_ews.assign(N, 0); g_evs.assign(N, 0); g_stamp = 0;
    }
    if (++g_stamp == 0) {
        std::fill(g_seen.begin(), g_seen.end(), 0); std::fill(g_tgt.begin(), g_tgt.end(), 0);
        std::fill(g_ews.begin(), g_ews.end(), 0); std::fill(g_evs.begin(), g_evs.end(), 0); g_stamp = 1;
    }
}
}  // namespace

extern "C" {
int astar3d_win(int L, int X, int Y, const double *d, const double *c, const double *u, const int32_t *dirs,
                const int64_t *strides, const int32_t *src, int ns, const int32_t *tgt, int nt, int x0, int y0, int W,
                double step_pen, double via_pen, int32_t *path, int max_path, double *cost) {
    Grid g{L, X, Y, d, c, u, dirs, strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]};
    int xlo = std::max(0, x0 - W), xhi = std::min(X - 1, x0 + W);
    int ylo = std::max(0, y0 - W), yhi = std::min(Y - 1, y0 + W);
    if (nt == 0) { *cost = INFINITY; return 0; }
    int zmin = L, zmax = -1, txmin = X, txmax = -1, tymin = Y, tymax = -1;
    for (int k = 0; k < nt; ++k) {
        int z = tgt[3 * k], x = tgt[3 * k + 1], y = tgt[3 * k + 2];
        zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        txmin = std::min(txmin, x); txmax = std::max(txmax, x);
        tymin = std::min(tymin, y); tymax = std::max(tymax, y);
    }
    bool local = xlo <= xhi && ylo <= yhi;
    for (int k = 0; k < ns && local; ++k) {
        int x = src[3 * k + 1], y = src[3 * k + 2];
        if (x < xlo || x > xhi || y < ylo || y > yhi) local = false;
    }
    const int64_t bx0 = local ? xlo : 0, by0 = local ? ylo : 0;
    const int64_t BX = local ? (xhi - xlo + 1) : X, BY = local ? (yhi - ylo + 1) : Y;
    const int64_t bx1 = bx0 + BX - 1, by1 = by0 + BY - 1;
    const int64_t N = (int64_t)L * BX * BY;
    ensure(N);
    const uint32_t st = g_stamp;
    auto idx = [&](int z, int x, int y) -> int64_t { return ((int64_t)z * BX + (x - bx0)) * BY + (y - by0); };
    auto inb = [&](int x, int y) -> bool { return x >= bx0 && x <= bx1 && y >= by0 && y <= by1; };
    auto h = [&](int z, int x, int y) -> double {
        int hx = std::max(0, std::max(txmin - x, x - txmax));
        int hy = std::max(0, std::max(tymin - y, y - tymax));
        int hz = std::max(0, std::max(zmin - z, z - zmax));
        return step_pen * (double)(hx + hy) + via_pen * (double)hz;
    };
    // wire edge leaving cell (z, x, y) in + direction == edge_wire(z, x, y); cached per call
    auto ew = [&](int z, int x, int y) -> double {
        int64_t j = idx(z, x, y);
        if (g_ews[j] != st) { g_ews[j] = st; g_ew[j] = edge_wire(g, z, x, y); }
        return g_ew[j];
    };
    auto ev = [&](int x, int y, int z) -> double {           // edge_via(x, y, z): via between z and z + 1
        int64_t j = idx(z, x, y);
        if (g_evs[j] != st) { g_evs[j] = st; g_ev[j] = edge_via(g, x, y, z); }
        return g_ev[j];
    };
    for (int k = 0; k < nt; ++k) {
        int x = tgt[3 * k + 1], y = tgt[3 * k + 2];
        if (inb(x, y)) g_tgt[idx(tgt[3 * k], x, y)] = st;    // a target outside the buffer is never reached
    }
    g_heap.clear();
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq(std::greater<Item>(), std::move(g_heap));
    for (int k = 0; k < ns; ++k) {
        int z = src[3 * k], x = src[3 * k + 1], y = src[3 * k + 2];
        int64_t i = idx(z, x, y);
        g_seen[i] = st; g_dist[i] = 0.0; g_prev[i] = -1;
        pq.push(Item{h(z, x, y), 0.0, z, x, y});
    }
    int ret = 0;
    *cost = INFINITY;
    while (!pq.empty()) {
        Item it = pq.top(); pq.pop();
        int64_t i = idx(it.z, it.x, it.y);
        if (it.g > g_dist[i]) continue;
        if (g_tgt[i] == st) {
            int n = 0;
            for (int64_t j = i; j >= 0; j = g_prev[j]) ++n;
            if (n > max_path) { *cost = it.g; ret = -n; break; }
            int k = n - 1;
            for (int64_t j = i; j >= 0; j = g_prev[j]) {
                int64_t r = j;
                int y = (int)(r % BY + by0); r /= BY;
                int x = (int)(r % BX + bx0); r /= BX;
                path[3 * k] = (int32_t)r; path[3 * k + 1] = x; path[3 * k + 2] = y;
                --k;
            }
            *cost = it.g;
            ret = n;
            break;
        }
        int z = it.z, x = it.x, y = it.y;
        double dcur = it.g;
        auto relax = [&](int nz, int nx, int ny, double w) {
            double nd = dcur + w;
            int64_t j = idx(nz, nx, ny);
            if (g_seen[j] != st || nd < g_dist[j]) {
                g_seen[j] = st; g_dist[j] = nd; g_prev[j] = i;
                pq.push(Item{nd + h(nz, nx, ny), nd, nz, nx, ny});
            }
        };
        if (z + 1 < L) relax(z + 1, x, y, ev(x, y, z) + via_pen);
        if (z - 1 >= 0) relax(z - 1, x, y, ev(x, y, z - 1) + via_pen);
        if (z >= 1) {
            if (dirs[z] == 0) {
                if (x + 1 <= xhi && x + 1 <= bx1) relax(z, x + 1, y, ew(z, x, y) + step_pen);
                if (x - 1 >= xlo && x - 1 >= bx0) relax(z, x - 1, y, ew(z, x - 1, y) + step_pen);
            } else {
                if (y + 1 <= yhi && y + 1 <= by1) relax(z, x, y + 1, ew(z, x, y) + step_pen);
                if (y - 1 >= ylo && y - 1 >= by0) relax(z, x, y - 1, ew(z, x, y - 1) + step_pen);
            }
        }
    }
    return ret;
}
}
