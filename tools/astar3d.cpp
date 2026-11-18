// astar3d.cpp — C version of repair3d_astar.astar (the A* window search of escape / repair), same costs, same
// tie-breaking (heap key = (f, g, z, x, y), as Python's heapq orders (f, g, (z, x, y))), so it returns the same path.
// Built as a shared library and called through ctypes by repair3d_astar.py (falls back to Python when absent):
//   macOS:  clang++ -O3 -std=c++17 -shared -fPIC -o tools/libastar3d_darwin.so tools/astar3d.cpp
//   Linux:  g++     -O3 -std=c++17 -shared -fPIC -o tools/libastar3d_linux.so  tools/astar3d.cpp
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
    int64_t dz, dx, dy, cz, cx, cy;                    // element strides of d and c (any numpy layout)
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
    bool operator>(const Item &o) const {          // min-heap on (f, g, z, x, y)
        if (f != o.f) return f > o.f;
        if (g != o.g) return g > o.g;
        if (z != o.z) return z > o.z;
        if (x != o.x) return x > o.x;
        return y > o.y;
    }
};

// buffers reused across calls (stamped, so no re-initialisation)
std::vector<double> g_dist;
std::vector<int64_t> g_prev;  // grid index of the predecessor
std::vector<uint32_t> g_seen, g_tgt;
uint32_t g_stamp = 0;
}  // namespace

extern "C" {
// returns path length (nodes) written to path[3*k] = (z, x, y), or 0 if no path; *cost = path cost
int astar3d(int L, int X, int Y, const double *d, const double *c, const double *u, const int32_t *dirs,
            const int64_t *strides, const int32_t *src, int ns, const int32_t *tgt, int nt, int x0, int y0, int W,
            double step_pen, double via_pen, int32_t *path, int max_path, double *cost) {
    Grid g{L, X, Y, d, c, u, dirs, strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]};
    int xlo = std::max(0, x0 - W), xhi = std::min(X - 1, x0 + W);
    int ylo = std::max(0, y0 - W), yhi = std::min(Y - 1, y0 + W);
    if (nt == 0) { *cost = INFINITY; return 0; }
    // stamped buffers over the whole grid, allocated once per grid size: a call costs only the nodes it explores
    int zmin = L, zmax = -1, txmin = X, txmax = -1, tymin = Y, tymax = -1;
    for (int k = 0; k < nt; ++k) {
        int z = tgt[3 * k], x = tgt[3 * k + 1], y = tgt[3 * k + 2];
        zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        txmin = std::min(txmin, x); txmax = std::max(txmax, x);
        tymin = std::min(tymin, y); tymax = std::max(tymax, y);
    }
    const int64_t BX = X, BY = Y, bx0 = 0, by0 = 0, bx1 = X - 1, by1 = Y - 1;
    const int64_t N = (int64_t)L * BX * BY;
    if ((int64_t)g_dist.size() != N) {
        g_dist.assign(N, 0.0); g_prev.assign(N, -1); g_seen.assign(N, 0); g_tgt.assign(N, 0); g_stamp = 0;
    }
    if (++g_stamp == 0) { std::fill(g_seen.begin(), g_seen.end(), 0); std::fill(g_tgt.begin(), g_tgt.end(), 0); g_stamp = 1; }
    const uint32_t st = g_stamp;
    auto idx = [&](int z, int x, int y) -> int64_t { return ((int64_t)z * BX + (x - bx0)) * BY + (y - by0); };
    auto h = [&](int z, int x, int y) -> double {
        int hx = std::max(0, std::max(txmin - x, x - txmax));
        int hy = std::max(0, std::max(tymin - y, y - tymax));
        int hz = std::max(0, std::max(zmin - z, z - zmax));
        return step_pen * (double)(hx + hy) + via_pen * (double)hz;
    };
    for (int k = 0; k < nt; ++k) g_tgt[idx(tgt[3 * k], tgt[3 * k + 1], tgt[3 * k + 2])] = st;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (int k = 0; k < ns; ++k) {
        int z = src[3 * k], x = src[3 * k + 1], y = src[3 * k + 2];
        int64_t i = idx(z, x, y);
        g_seen[i] = st; g_dist[i] = 0.0; g_prev[i] = -1;
        pq.push(Item{h(z, x, y), 0.0, z, x, y});
    }
    while (!pq.empty()) {
        Item it = pq.top(); pq.pop();
        int64_t i = idx(it.z, it.x, it.y);
        if (it.g > g_dist[i]) continue;                    // stale (g_seen[i] == st for every pushed node)
        if (g_tgt[i] == st) {
            int n = 0;
            for (int64_t j = i; j >= 0; j = g_prev[j]) ++n;
            if (n > max_path) { *cost = it.g; return -n; }
            int k = n - 1;
            for (int64_t j = i; j >= 0; j = g_prev[j]) {
                int64_t r = j;
                int y = (int)(r % BY) + by0; r /= BY;
                int x = (int)(r % BX) + bx0; r /= BX;
                path[3 * k] = (int32_t)r; path[3 * k + 1] = x; path[3 * k + 2] = y;
                --k;
            }
            *cost = it.g;
            return n;
        }
        int z = it.z, x = it.x, y = it.y;
        double dcur = it.g;
        // neighbours in astar()'s order: up, down, wire +, wire -
        auto relax = [&](int nz, int nx, int ny, double w) {
            double nd = dcur + w;
            int64_t j = idx(nz, nx, ny);
            if (g_seen[j] != st || nd < g_dist[j]) {
                g_seen[j] = st; g_dist[j] = nd; g_prev[j] = i;
                pq.push(Item{nd + h(nz, nx, ny), nd, nz, nx, ny});
            }
        };
        if (z + 1 < L) relax(z + 1, x, y, edge_via(g, x, y, z) + via_pen);
        if (z - 1 >= 0) relax(z - 1, x, y, edge_via(g, x, y, z - 1) + via_pen);
        if (z >= 1) {
            if (dirs[z] == 0) {
                if (x + 1 <= xhi && x + 1 <= bx1) relax(z, x + 1, y, edge_wire(g, z, x, y) + step_pen);
                if (x - 1 >= xlo && x - 1 >= bx0) relax(z, x - 1, y, edge_wire(g, z, x - 1, y) + step_pen);
            } else {
                if (y + 1 <= yhi && y + 1 <= by1) relax(z, x, y + 1, edge_wire(g, z, x, y) + step_pen);
                if (y - 1 >= ylo && y - 1 >= by0) relax(z, x, y - 1, edge_wire(g, z, x, y - 1) + step_pen);
            }
        }
    }
    *cost = INFINITY;
    return 0;
}
}
