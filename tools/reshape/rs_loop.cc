// rs_loop.cc:
// the overflow stage's per-run re-route loop with a timing + power price and two more candidate modes (see make_rs_loop.py).
// Build (no -ffast-math):  Linux: g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -shared -fPIC -o librs_loop.so rs_loop.cc
//                          macOS: clang++ ... -o librs_loop.dylib rs_loop.cc
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

typedef int64_t i64;
typedef int (*AstarFn)(int, int, int, const double *, const double *, const double *, const int32_t *, const int64_t *,
                       const int32_t *, int, const int32_t *, int, int, int, int, double, double, int32_t *, int, double *);

namespace {
constexpr double CAP_EPS = 1e-3;
constexpr int BIG = 64;           // repair3d_astar.BIG
constexpr int BIG_NET = 4000;     // repair3d_big.BIG_NET

inline double cell_cost(double u, double d, double c) {
    if (c > CAP_EPS) return u * std::exp(std::min(0.25 * d - 0.5 * c, 700.0));
    return d > 0 ? u * std::exp(std::min(0.75 * d, 700.0)) : 0.0;
}

// ------------------------------------------------------------------ CPython 3.12 set of 3-int tuples
inline uint64_t int_hash(i64 v) { return v == -1 ? (uint64_t)-2 : (uint64_t)v; }
inline uint64_t tuple_hash3(i64 a, i64 b, i64 c) {
    const uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P5 = 2870177450012600261ULL;
    uint64_t acc = P5;
    i64 it[3] = {a, b, c};
    for (int k = 0; k < 3; ++k) {
        acc += int_hash(it[k]) * P2;
        acc = (acc << 31) | (acc >> 33);
        acc *= P1;
    }
    acc += (uint64_t)3 ^ (P5 ^ 3527539UL);
    if (acc == (uint64_t)-1) return 1546275796ULL;
    return acc;
}
struct SE { uint64_t h; int32_t x, y, z; bool k; };
struct PySet {
    std::vector<SE> t;
    size_t mask = 7, fill = 0, used = 0;
    PySet() : t(8, SE{0, 0, 0, 0, false}) {}
    static void insert_clean(std::vector<SE> &tab, size_t m, const SE &e) {
        size_t perturb = (size_t)e.h, i = (size_t)e.h & m;
        while (true) {
            if (!tab[i].k) { tab[i] = e; return; }
            if (i + 9 <= m) {
                for (size_t j = 1; j <= 9; ++j) if (!tab[i + j].k) { tab[i + j] = e; return; }
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & m;
        }
    }
    void resize(size_t minused) {
        size_t ns = 8;
        while (ns <= minused) ns <<= 1;
        std::vector<SE> nt(ns, SE{0, 0, 0, 0, false});
        for (const SE &e : t) if (e.k) insert_clean(nt, ns - 1, e);
        t.swap(nt); mask = ns - 1; fill = used;
    }
    void add_entry(const SE &e) {
        size_t m = mask, i = (size_t)e.h & m, perturb = (size_t)e.h;
        while (true) {
            size_t j = i;
            int probes = (i + 9 <= m) ? 9 : 0;
            do {
                SE &s = t[j];
                if (!s.k) {
                    s = e; s.k = true; ++fill; ++used;
                    if (fill * 5 < m * 3) return;
                    resize(used > 50000 ? used * 2 : used * 4);
                    return;
                }
                if (s.h == e.h && s.x == e.x && s.y == e.y && s.z == e.z) return;
                ++j;
            } while (probes--);
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & m;
        }
    }
    void add(int x, int y, int z) { add_entry(SE{tuple_hash3(x, y, z), x, y, z, true}); }
    void merge(const PySet &o) {                                     // set_merge(self, o)
        if (&o == this || o.used == 0) return;
        if ((fill + o.used) * 5 >= mask * 3) resize((used + o.used) * 2);
        if (fill == 0 && mask == o.mask && o.fill == o.used) { t = o.t; fill = o.fill; used = o.used; return; }
        if (fill == 0) {
            fill = o.used; used = o.used;
            for (const SE &e : o.t) if (e.k) insert_clean(t, mask, e);
            return;
        }
        for (const SE &e : o.t) if (e.k) add_entry(e);
    }
    PySet copy() const { PySet s; s.merge(*this); return s; }       // set(v) / set_copy
    size_t size() const { return used; }
};
PySet set_or(const PySet &a, const PySet &b) { PySet r = a.copy(); if (&a != &b) r.merge(b); return r; }

// ------------------------------------------------------------------ grid
struct Grid {
    int L, X, Y;
    double *d; const double *c; const double *u; const int32_t *dirs;
    i64 dz, dx, dy, cz, cx, cy;
    inline double &D(int z, int x, int y) const { return d[z * dz + x * dx + y * dy]; }
    inline double C(int z, int x, int y) const { return c[z * cz + x * cx + y * cy]; }
};
inline uint64_t pk(int z, int x, int y) { return ((uint64_t)(uint32_t)z << 42) | ((uint64_t)(uint32_t)x << 21) | (uint64_t)(uint32_t)y; }

struct Track { int z, f; std::vector<std::pair<int, int>> iv; };
typedef std::vector<Track> Wires;                                       // dict order
inline uint64_t tk(int z, int f) { return ((uint64_t)(uint32_t)z << 32) | (uint32_t)f; }

struct WIdx {                                                           // dict lookup of a Wires
    std::unordered_map<uint64_t, int> m;
    explicit WIdx(const Wires &w) { m.reserve(w.size() * 2 + 1); for (int k = 0; k < (int)w.size(); ++k) m[tk(w[k].z, w[k].f)] = k; }
    int get(int z, int f) const { auto it = m.find(tk(z, f)); return it == m.end() ? -1 : it->second; }
};

void merge_ivs(std::vector<std::pair<int, int>> &ivs) {
    std::sort(ivs.begin(), ivs.end());
    std::vector<std::pair<int, int>> m;
    m.reserve(ivs.size());
    for (auto &p : ivs) {
        if (!m.empty() && p.first <= m.back().second) m.back().second = std::max(m.back().second, p.second);
        else m.push_back(p);
    }
    ivs.swap(m);
}

// timing_promote.to_wires_vias
void to_wires_vias(const int32_t *sg, i64 n, const int32_t *dirs, Wires &w, PySet &v) {
    std::unordered_map<uint64_t, int> idx;
    for (i64 s = 0; s < n; ++s) {
        int x1 = sg[6 * s], y1 = sg[6 * s + 1], z1 = sg[6 * s + 2], x2 = sg[6 * s + 3], y2 = sg[6 * s + 4], z2 = sg[6 * s + 5];
        if (z1 != z2) {
            for (int z = std::min(z1, z2); z < std::max(z1, z2); ++z) v.add(x1, y1, z);
        } else if (dirs[z1] == 0 && y1 == y2 && x1 != x2) {
            auto it = idx.find(tk(z1, y1));
            int k;
            if (it == idx.end()) { k = (int)w.size(); idx[tk(z1, y1)] = k; w.push_back(Track{z1, y1, {}}); } else k = it->second;
            w[k].iv.push_back({std::min(x1, x2), std::max(x1, x2)});
        } else if (dirs[z1] == 1 && x1 == x2 && y1 != y2) {
            auto it = idx.find(tk(z1, x1));
            int k;
            if (it == idx.end()) { k = (int)w.size(); idx[tk(z1, x1)] = k; w.push_back(Track{z1, x1, {}}); } else k = it->second;
            w[k].iv.push_back({std::min(y1, y2), std::max(y1, y2)});
        }
    }
    for (auto &t : w) merge_ivs(t.iv);
}

// escape3d.merge_wires(a, b)
Wires merge_wires(const Wires &a, const Wires &b) {
    Wires out = a;
    WIdx ix(out);
    std::unordered_map<uint64_t, int> extra;
    for (const Track &t : b) {
        int k = ix.get(t.z, t.f);
        if (k < 0) {
            auto it = extra.find(tk(t.z, t.f));
            if (it == extra.end()) { k = (int)out.size(); extra[tk(t.z, t.f)] = k; out.push_back(Track{t.z, t.f, {}}); } else k = it->second;
        }
        out[k].iv.insert(out[k].iv.end(), t.iv.begin(), t.iv.end());
    }
    for (auto &t : out) merge_ivs(t.iv);
    return out;
}

// repair3d.drop_run
Wires drop_run(const Wires &w, int z, int f, int a, int b) {
    Wires out;
    out.reserve(w.size());
    for (const Track &t : w) {
        if (t.z == z && t.f == f) {
            Track n{t.z, t.f, {}};
            for (auto &p : t.iv) if (!(p.first == a && p.second == b)) n.iv.push_back(p);
            if (!n.iv.empty()) out.push_back(std::move(n));
        } else out.push_back(t);
    }
    return out;
}

inline bool covered(const Grid &g, const Wires &w, const WIdx &ix, int x, int y, int z) {
    int k = g.dirs[z] == 0 ? ix.get(z, y) : ix.get(z, x);
    if (k < 0) return false;
    int pos = g.dirs[z] == 0 ? x : y;
    for (auto &p : w[k].iv) if (p.first <= pos && pos <= p.second) return true;
    return false;
}

inline int via_cells(const Grid &g, int x, int y, int z, int *cx, int *cy, int *dd) {
    if (g.dirs[z] == 0) {
        if (0 < x && x < g.X - 1) { cx[0] = x - 1; cy[0] = y; dd[0] = 1; cx[1] = x; cy[1] = y; dd[1] = 1; return 2; }
        if (x > 0) { cx[0] = x - 1; cy[0] = y; dd[0] = 2; return 1; }
        if (x < g.X - 1) { cx[0] = x; cy[0] = y; dd[0] = 2; return 1; }
        return 0;
    }
    if (0 < y && y < g.Y - 1) { cx[0] = x; cy[0] = y - 1; dd[0] = 1; cx[1] = x; cy[1] = y; dd[1] = 1; return 2; }
    if (y > 0) { cx[0] = x; cy[0] = y - 1; dd[0] = 2; return 1; }
    if (y < g.Y - 1) { cx[0] = x; cy[0] = y; dd[0] = 2; return 1; }
    return 0;
}

// repair3d_astar.apply_fast
void apply(const Grid &g, const Wires &w, const PySet &v, int sign) {
    WIdx ix(w);
    for (const Track &t : w) {
        bool h = g.dirs[t.z] == 0;
        for (auto &p : t.iv)
            for (int s = p.first; s < p.second; ++s) {
                if (h) g.D(t.z, s, t.f) += (double)(2 * sign); else g.D(t.z, t.f, s) += (double)(2 * sign);
            }
    }
    int cx[2], cy[2], dd[2];
    for (const SE &e : v.t) {
        if (!e.k) continue;
        if (e.z == 0 || covered(g, w, ix, e.x, e.y, e.z)) continue;
        int n = via_cells(g, e.x, e.y, e.z, cx, cy, dd);
        for (int k = 0; k < n; ++k) g.D(e.z, cx[k], cy[k]) += (double)(dd[k] * sign);
    }
}

double pairwise_sum(const double *a, i64 n) {                           // numpy pairwise_sum_DOUBLE
    if (n < 8) {
        double res = 0.;
        for (i64 i = 0; i < n; ++i) res += a[i];
        return res;
    } else if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        i64 i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// repair3d_astar.marginal_fast
double marginal(const Grid &g, const Wires &w, const PySet &v) {
    WIdx ix(w);
    i64 wl = 0;
    for (const Track &t : w) for (auto &p : t.iv) wl += p.second - p.first;
    int cx[2], cy[2], dd[2];
    if ((i64)w.size() <= BIG && wl + 2 * (i64)v.size() <= 2 * BIG) {
        std::vector<std::pair<uint64_t, int>> out;
        std::unordered_map<uint64_t, int> pos;
        auto put = [&](int z, int x, int y, int q) {
            uint64_t key = pk(z, x, y);
            auto it = pos.find(key);
            if (it == pos.end()) { pos[key] = (int)out.size(); out.push_back({key, q}); } else out[it->second].second += q;
        };
        for (const Track &t : w) {
            bool h = g.dirs[t.z] == 0;
            for (auto &p : t.iv) for (int s = p.first; s < p.second; ++s) { if (h) put(t.z, s, t.f, 2); else put(t.z, t.f, s, 2); }
        }
        for (const SE &e : v.t) {
            if (!e.k) continue;
            if (e.z == 0 || covered(g, w, ix, e.x, e.y, e.z)) continue;
            int n = via_cells(g, e.x, e.y, e.z, cx, cy, dd);
            for (int k = 0; k < n; ++k) put(e.z, cx[k], cy[k], dd[k]);
        }
        double tot = 0.0;
        for (auto &o : out) {
            int z = (int)(o.first >> 42), x = (int)((o.first >> 21) & 0x1FFFFF), y = (int)(o.first & 0x1FFFFF);
            if (z == 0) continue;
            double d = g.D(z, x, y), c = g.C(z, x, y), u = g.u[z];
            tot += cell_cost(u, d + (double)o.second, c) - cell_cost(u, d, c);
        }
        return tot;
    }
    std::vector<std::pair<i64, double>> cells;
    const i64 X = g.X, Y = g.Y;
    for (const Track &t : w) {
        if (t.z <= 0) continue;
        bool h = g.dirs[t.z] == 0;
        for (auto &p : t.iv) {
            if (p.second <= p.first) continue;
            for (int s = p.first; s < p.second; ++s) {
                i64 x = h ? s : t.f, y = h ? t.f : s;
                cells.push_back({((i64)t.z * X + x) * Y + y, 2.0});
            }
        }
    }
    for (const SE &e : v.t) {
        if (!e.k) continue;
        if (e.z == 0 || covered(g, w, ix, e.x, e.y, e.z)) continue;
        int n = via_cells(g, e.x, e.y, e.z, cx, cy, dd);
        for (int k = 0; k < n; ++k) cells.push_back({((i64)e.z * X + cx[k]) * Y + cy[k], (double)dd[k]});
    }
    if (cells.empty()) return 0.0;
    std::stable_sort(cells.begin(), cells.end(), [](const std::pair<i64, double> &a, const std::pair<i64, double> &b) { return a.first < b.first; });
    std::vector<double> terms;
    terms.reserve(cells.size());
    for (size_t k = 0; k < cells.size();) {
        i64 lin = cells[k].first;
        double q = 0.0;
        size_t j = k;
        for (; j < cells.size() && cells[j].first == lin; ++j) q += cells[j].second;
        int z = (int)(lin / (X * Y)); i64 r = lin % (X * Y); int x = (int)(r / Y), y = (int)(r % Y);
        double d = g.D(z, x, y), c = g.C(z, x, y), u = g.u[z];
        terms.push_back(cell_cost(u, d + q, c) - cell_cost(u, d, c));
        k = j;
    }
    return pairwise_sum(terms.data(), (i64)terms.size());
}

// ------------------------------------------------------------------ interval union-find (ovf_lib._iv_components)
struct Comp {
    const Grid *g = nullptr;
    struct Node { bool wire; int z, f, a, b; int x, y; };                // wire: z, f, [a, b]; point: z, x, y
    std::vector<Node> nodes;
    std::vector<int> par;
    std::unordered_map<uint64_t, std::pair<int, int>> track;           // (z, f) -> (first node, count)
    std::unordered_map<uint64_t, int> pnode;
    int find(int k) {
        while (par[k] != k) { par[k] = par[par[k]]; k = par[k]; }
        return k;
    }
    int lookup(int z, int x, int y) const {                             // node of a point or -1 (no creation)
        bool h = g->dirs[z] == 0;
        auto it = track.find(h ? tk(z, y) : tk(z, x));
        int pos = h ? x : y;
        if (it != track.end()) {
            int lo = 0, hi = it->second.second, b0 = it->second.first;
            while (lo < hi) { int m = (lo + hi) >> 1; if (nodes[b0 + m].a <= pos) lo = m + 1; else hi = m; }
            int j = lo - 1;
            if (j >= 0 && nodes[b0 + j].b >= pos) return b0 + j;
        }
        auto p = pnode.find(pk(z, x, y));
        return p == pnode.end() ? -1 : p->second;
    }
    int locate(int z, int x, int y) {
        int k = lookup(z, x, y);
        if (k >= 0) return k;
        k = (int)nodes.size();
        nodes.push_back(Node{false, z, 0, 0, 0, x, y}); par.push_back(k); pnode[pk(z, x, y)] = k;
        return k;
    }
    void build(const Grid &g_, const Wires &w, const PySet &v) {
        g = &g_;
        for (const Track &t : w) {
            std::vector<std::pair<int, int>> lst = t.iv;                // sorted(ivs)
            std::sort(lst.begin(), lst.end());
            int first = (int)nodes.size();
            bool have = false; int prev_b = 0, prev_k = 0;
            for (auto &p : lst) {
                int k = (int)nodes.size();
                nodes.push_back(Node{true, t.z, t.f, p.first, p.second, 0, 0}); par.push_back(k);
                if (have && p.first <= prev_b) par[k] = prev_k;
                if (!have || p.second > prev_b) { prev_b = p.second; prev_k = k; have = true; }
            }
            track[tk(t.z, t.f)] = {first, (int)lst.size()};
        }
        for (const SE &e : v.t) {
            if (!e.k) continue;
            int k1 = find(locate(e.z, e.x, e.y));
            int k2 = find(locate(e.z + 1, e.x, e.y));
            if (k1 != k2) par[k1] = k2;
        }
    }
};

// ovf_lib.net_valid_fast
bool net_valid(const Grid &g, const Wires &w, const PySet &v, const int32_t *ap, const i64 *pin_off, int npins) {
    Comp cm; cm.build(g, w, v);
    if (cm.nodes.empty()) return false;
    int r0 = cm.find(0);
    for (int k = 1; k < (int)cm.nodes.size(); ++k) if (cm.find(k) != r0) return false;
    for (int p = 0; p < npins; ++p) {
        bool ok = false;
        for (i64 q = pin_off[p]; q < pin_off[p + 1] && !ok; ++q) {
            int l = ap[3 * q], x = ap[3 * q + 1], y = ap[3 * q + 2];
            if (cm.pnode.count(pk(l, x, y))) { ok = true; break; }
            bool h = g.dirs[l] == 0;
            auto it = cm.track.find(h ? tk(l, y) : tk(l, x));
            if (it != cm.track.end()) {
                int pos = h ? x : y;
                for (int j = 0; j < it->second.second; ++j) {
                    auto &nd = cm.nodes[it->second.first + j];
                    if (nd.a <= pos && pos <= nd.b) { ok = true; break; }
                }
            }
        }
        if (!ok) return false;
    }
    return true;
}

struct NetSt { Wires w; PySet v; };
struct Cand { double nsc; i64 i; int z, f, a, b, slot; };

// results of the last call
std::vector<int> R_changed;          // slots (positions in the chosen list), sorted by route index
std::vector<i64> R_off;
std::vector<int32_t> R_seg;
}  // namespace

extern "C" {
// stats out (int64[16]): 0 big_skip 1 hmax_skip 2 cand_runs 3 stale 4 pin_end 5 spur 6 spur_end 7 no_path 8 reject_invalid
//                        9 rerouted 10 kept 11 nets_changed 12 astar_calls
// extra stats: 13 accepted although the overflow marginal rose, 14 prune drops, 15 mode-2 skips (would need a reconnection)
// gain out (double[4]): sum over accepted runs of the price fall split as overflow / timing / power (score units), total
int rs_rank(int L, int X, int Y, double *d, const double *c, const double *cs, const double *u, const int32_t *dirs,
               const int64_t *strides,              // d (3), c (3), cs (3), in elements
               void *astar_ptr, int n, const int64_t *ridx, const int64_t *seg_off, const int32_t *segs,
               const int64_t *pin_off, const int64_t *ap_off, const int32_t *aps,
               int W, int hmax, int max_runs, int64_t *stats,
               int mode, const double *ntw, const double *ncl, const double *nrd, const double *npw,
               const double *Rz, const double *Cz, const double *Rvz, const double *Fp, double *gain, const double *nsb, const double *nss) {
    const double w4 = Fp[0], um = Fp[1], cvia = Fp[2]; const int minlen = (int)Fp[3];
    for (int k = 0; k < 4; ++k) gain[k] = 0.0;
    // timing / power part of the price of one net (score units)
    double last_dl = 0.0;                   // lumped delay (ns) of the last tp_price call (slack-budget guard)
    auto tp_price = [&](int s, const Wires &w, const PySet &v, double &tpart, double &ppart) {
        double tw = ntw[s], pw = npw[s], cl = ncl[s], rd = nrd[s], dl = 0.0, cap = 0.0;
        for (const Track &t : w) {
            if (t.z < 0 || t.z >= L) continue;
            double R = Rz[t.z], C = Cz[t.z];
            for (auto &p : t.iv) {
                double l = (double)(p.second - p.first) * um;
                dl += l * (R * cl + C * rd) / 1000.0 + R * C * l * l / 2.0 / 1000.0;
                cap += C * l;
            }
        }
        if (tw > 0.0) for (const SE &e : v.t) if (e.k && e.z >= 0 && e.z < L) dl += Rvz[e.z] * cvia / 1000.0;
        if (tw <= 0.0) for (const SE &e : v.t) if (e.k && e.z >= 0 && e.z < L) dl += Rvz[e.z] * cvia / 1000.0;
        last_dl = dl;
        tpart = tw * dl; ppart = pw * cap;
    };
    AstarFn astar = (AstarFn)astar_ptr;
    Grid g{L, X, Y, d, c, u, dirs, strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]};
    const int64_t sstr[6] = {strides[0], strides[1], strides[2], strides[6], strides[7], strides[8]};
    for (int k = 0; k < 17; ++k) stats[k] = 0;
    // ov mask and per-track prefix of m = (cost(d) - cost(max(d - 2, 0))) * ov, as numpy (sequential cumsum along the track)
    auto ovc = [&](int z, int x, int y) -> bool {
        if (z == 0) return false;
        double dd_ = g.D(z, x, y), cc = g.C(z, x, y);
        return (cc <= CAP_EPS && dd_ > 0) || (cc > CAP_EPS && dd_ > 2.0 * cc);
    };
    std::unordered_map<uint64_t, std::pair<std::vector<double>, std::vector<int32_t>>> pref;
    auto get_pref = [&](int z, int f) -> std::pair<std::vector<double>, std::vector<int32_t>> & {
        auto it = pref.find(tk(z, f));
        if (it != pref.end()) return it->second;
        bool h = dirs[z] == 0;
        int n_ = h ? X : Y;
        std::vector<double> P(n_ + 1, 0.0); std::vector<int32_t> Cn(n_ + 1, 0);
        double uz = u[z];
        for (int t = 0; t < n_; ++t) {
            int x = h ? t : f, y = h ? f : t;
            bool o = ovc(z, x, y);
            double dd_ = g.D(z, x, y), cc = g.C(z, x, y);
            double m = (cell_cost(uz, dd_, cc) - cell_cost(uz, std::max(dd_ - 2.0, 0.0), cc)) * (o ? 1.0 : 0.0);
            P[t + 1] = P[t] + m; Cn[t + 1] = Cn[t] + (o ? 1 : 0);
        }
        return pref.emplace(tk(z, f), std::make_pair(std::move(P), std::move(Cn))).first->second;
    };
    std::vector<NetSt> cur(n);
    std::vector<char> live(n, 0);
    std::vector<Cand> cand;
    for (int s = 0; s < n; ++s) {
        NetSt &ns = cur[s];
        to_wires_vias(segs + 6 * seg_off[s], seg_off[s + 1] - seg_off[s], dirs, ns.w, ns.v);
        i64 tot = (i64)ns.v.size();
        for (auto &t : ns.w) tot += (i64)t.iv.size();
        if (tot > BIG_NET) { stats[0]++; ns.w.clear(); ns.v = PySet(); continue; }
        live[s] = 1;
        if (mode != 0) {
            for (auto &t : ns.w) {
                if (t.z < 1 || t.z >= L) continue;
                for (auto &p : t.iv) {
                    int a = p.first, b = p.second;
                    if (b - a < std::max(minlen, 1)) continue;
                    if (hmax && (b - a) / 2 + W > hmax) { stats[1]++; continue; }
                    double l = (double)(b - a) * um, R = Rz[t.z], C = Cz[t.z];
                    double sc = mode == 2 ? (double)(b - a)
                                          : ntw[s] * (l * (R * ncl[s] + C * nrd[s]) / 1000.0 + R * C * l * l / 2000.0) + npw[s] * C * l;
                    cand.push_back(Cand{-sc, ridx[s], t.z, t.f, a, b, s});
                }
            }
            continue;
        }
        for (auto &t : ns.w) {
            if (t.z < 1 || t.z >= L) continue;
            auto &pr = get_pref(t.z, t.f);
            for (auto &p : t.iv) {
                int a = p.first, b = p.second;
                int k = pr.second[b] - pr.second[a];
                if (!k) continue;
                if (hmax && (b - a) / 2 + W > hmax) { stats[1]++; continue; }
                double sc = pr.first[b] - pr.first[a];
                cand.push_back(Cand{-sc, ridx[s], t.z, t.f, a, b, s});
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Cand &p, const Cand &q) {
        if (p.nsc != q.nsc) return p.nsc < q.nsc;
        if (p.i != q.i) return p.i < q.i;
        if (p.z != q.z) return p.z < q.z;
        if (p.f != q.f) return p.f < q.f;
        if (p.a != q.a) return p.a < q.a;
        return p.b < q.b;
    });
    stats[2] = (i64)cand.size();
    {   // reshape: batches of a ranking (skip the first Fp[4] candidates)
        i64 sk = (i64)Fp[4];
        if (sk > 0) cand.erase(cand.begin(), cand.begin() + std::min<i64>(sk, (i64)cand.size()));
    }
    if (max_runs && (i64)cand.size() > max_runs) cand.resize(max_runs);
    std::vector<char> changed(n, 0);
    std::vector<int32_t> pathbuf((size_t)3 << 16, 0);
    int path_cap = 1 << 16;
    std::vector<int32_t> src, tgt;
    for (const Cand &cd : cand) {
        int s = cd.slot;
        NetSt &ns = cur[s];
        const Wires &w = ns.w;
        const PySet &v = ns.v;
        {   // stale
            WIdx ix(w);
            int k = ix.get(cd.z, cd.f);
            bool found = false;
            if (k >= 0) for (auto &p : w[k].iv) if (p.first == cd.a && p.second == cd.b) { found = true; break; }
            if (!found) { stats[3]++; continue; }
        }
        const int32_t *ap = aps + 3 * ap_off[pin_off[s]];
        std::vector<i64> poff_local;
        int npins = (int)(pin_off[s + 1] - pin_off[s]);
        poff_local.resize(npins + 1);
        for (int p = 0; p <= npins; ++p) poff_local[p] = ap_off[pin_off[s] + p] - ap_off[pin_off[s]];
        apply(g, w, v, -1);
        double old = marginal(g, w, v);
        double t_old = 0.0, p_old = 0.0;
        tp_price(s, w, v, t_old, p_old);
        double dl_old = last_dl;
        Wires w_rest = drop_run(w, cd.z, cd.f, cd.a, cd.b);
        int z = cd.z, fx = cd.f;
        bool h = dirs[z] == 0;
        int A[3] = {z, h ? cd.a : fx, h ? fx : cd.a};
        int B[3] = {z, h ? cd.b : fx, h ? fx : cd.b};
        Comp cm; cm.build(g, w_rest, v);
        // pin-end points
        auto is_pin = [&](const int *q) {
            i64 na = poff_local[npins];
            for (i64 k = 0; k < na; ++k) if (ap[3 * k] == q[0] && ap[3 * k + 1] == q[1] && ap[3 * k + 2] == q[2]) return true;
            return false;
        };
        int nodeA = cm.lookup(A[0], A[1], A[2]), nodeB = cm.lookup(B[0], B[1], B[2]);
        bool pinA = false, pinB = false;
        if (nodeA < 0 && is_pin(A)) { pinA = true; stats[4]++; }
        if (nodeB < 0 && is_pin(B)) { pinB = true; stats[4]++; }
        bool inA = nodeA >= 0 || pinA, inB = nodeB >= 0 || pinB;
        // group ids: by first node index; pin-end singletons last (A then B)
        int nn = (int)cm.nodes.size();
        std::vector<int> gid_of_root(nn, -1), node_gid(nn);
        int ng = 0;
        for (int k = 0; k < nn; ++k) { int r = cm.find(k); if (gid_of_root[r] < 0) gid_of_root[r] = ng++; node_gid[k] = gid_of_root[r]; }
        int gA_pin = -1, gB_pin = -1;
        if (pinA) gA_pin = ng++;
        if (pinB) gB_pin = ng++;
        auto gid = [&](const int *q, int node, bool pin, int gp) { (void)q; return pin ? gp : node_gid[node]; };
        Wires w2; PySet v2;
        bool reconnect = false, simple = false;
        if (!inA && !inB) { stats[5]++; simple = true; }
        else if (inA && inB && gid(A, nodeA, pinA, gA_pin) == gid(B, nodeB, pinB, gB_pin)) simple = true;
        else reconnect = true;
        bool fail = false;
        if (simple) { w2 = w_rest; v2 = v; }
        else if (reconnect) {
            if (!(inA && inB)) stats[6]++;
            const int *E0 = inA ? A : B;
            int nodeE = inA ? nodeA : nodeB; bool pinE = inA ? pinA : pinB; int gpE = inA ? gA_pin : gB_pin;
            int gE = gid(E0, nodeE, pinE, gpE);
            int x0 = (A[1] + B[1]) / 2, y0 = (A[2] + B[2]) / 2;
            int half = std::max(std::abs(A[1] - B[1]), std::abs(A[2] - B[2])) / 2 + W;
            int ex = E0[1], ey = E0[2];
            auto inwin = [&](int x, int y) { return std::abs(x - x0) <= half && std::abs(y - y0) <= half; };
            // per group: min distance to E over all points, in-window points
            std::vector<int> gmin(ng, INT32_MAX);
            std::vector<char> gwin(ng, 0);
            std::vector<std::vector<int>> gnodes(ng);
            for (int k = 0; k < nn; ++k) gnodes[node_gid[k]].push_back(k);
            auto node_minmax = [&](int k, int &dmin, bool &win) {
                const auto &nd = cm.nodes[k];
                if (!nd.wire) { dmin = std::abs(nd.x - ex) + std::abs(nd.y - ey); win = inwin(nd.x, nd.y); return; }
                bool hh = dirs[nd.z] == 0;
                if (hh) {
                    int t = std::min(std::max(ex, nd.a), nd.b);
                    dmin = std::abs(t - ex) + std::abs(nd.f - ey);
                    int lo = std::max(nd.a, x0 - half), hi = std::min(nd.b, x0 + half);
                    win = lo <= hi && std::abs(nd.f - y0) <= half;
                } else {
                    int t = std::min(std::max(ey, nd.a), nd.b);
                    dmin = std::abs(nd.f - ex) + std::abs(t - ey);
                    int lo = std::max(nd.a, y0 - half), hi = std::min(nd.b, y0 + half);
                    win = lo <= hi && std::abs(nd.f - x0) <= half;
                }
            };
            for (int k = 0; k < nn; ++k) {
                int dm; bool wn; node_minmax(k, dm, wn);
                int q = node_gid[k];
                gmin[q] = std::min(gmin[q], dm); if (wn) gwin[q] = 1;
            }
            if (pinA) { gmin[gA_pin] = std::abs(A[1] - ex) + std::abs(A[2] - ey); gwin[gA_pin] = inwin(A[1], A[2]); }
            if (pinB) { gmin[gB_pin] = std::abs(B[1] - ex) + std::abs(B[2] - ey); gwin[gB_pin] = inwin(B[1], B[2]); }
            auto win_points = [&](int q, std::vector<int32_t> &out) {     // (z, x, y) triples
                out.clear();
                if (q == gA_pin) { out.insert(out.end(), {A[0], A[1], A[2]}); return; }
                if (q == gB_pin) { out.insert(out.end(), {B[0], B[1], B[2]}); return; }
                for (int k : gnodes[q]) {
                    const auto &nd = cm.nodes[k];
                    if (!nd.wire) { if (inwin(nd.x, nd.y)) out.insert(out.end(), {nd.z, nd.x, nd.y}); continue; }
                    bool hh = dirs[nd.z] == 0;
                    if (hh) {
                        if (std::abs(nd.f - y0) > half) continue;
                        for (int t = std::max(nd.a, x0 - half); t <= std::min(nd.b, x0 + half); ++t) out.insert(out.end(), {nd.z, t, nd.f});
                    } else {
                        if (std::abs(nd.f - x0) > half) continue;
                        for (int t = std::max(nd.a, y0 - half); t <= std::min(nd.b, y0 + half); ++t) out.insert(out.end(), {nd.z, nd.f, t});
                    }
                }
            };
            std::vector<int> others;
            for (int q = 0; q < ng; ++q) if (q != gE && gwin[q]) others.push_back(q);
            std::stable_sort(others.begin(), others.end(), [&](int p, int q) { return gmin[p] < gmin[q]; });
            if (mode == 2 && !others.empty()) { others.clear(); fail = true; stats[15]++; stats[7]--; }   // prune: no A*
            std::unordered_set<uint64_t> tset;
            win_points(gE, src);
            for (size_t k = 0; k < src.size(); k += 3) tset.insert(pk(src[k], src[k + 1], src[k + 2]));
            w2 = w_rest; v2 = v.copy();
            for (int q : others) {
                win_points(q, tgt);
                if (src.empty() || tgt.empty()) { fail = true; break; }
                int np_;
                double cost = 0.0;
                while (true) {
                    np_ = astar(L, X, Y, d, cs, u, dirs, sstr, src.data(), (int)(src.size() / 3), tgt.data(), (int)(tgt.size() / 3),
                                x0, y0, half, 1.0, 0.1, pathbuf.data(), path_cap, &cost);
                    if (np_ >= 0) break;
                    path_cap = -np_ * 2; pathbuf.assign((size_t)path_cap * 3, 0);
                }
                stats[12]++;
                if (np_ == 0) { fail = true; break; }
                Wires pw; PySet pv;
                {
                    std::unordered_map<uint64_t, int> idx;
                    for (int k = 0; k + 1 < np_; ++k) {
                        int z1 = pathbuf[3 * k], x1 = pathbuf[3 * k + 1], y1 = pathbuf[3 * k + 2];
                        int z2 = pathbuf[3 * k + 3], x2 = pathbuf[3 * k + 4], y2 = pathbuf[3 * k + 5];
                        if (z1 != z2) { pv.add(x1, y1, std::min(z1, z2)); continue; }
                        int f_ = dirs[z1] == 0 ? y1 : x1;
                        std::pair<int, int> ivp = dirs[z1] == 0 ? std::make_pair(std::min(x1, x2), std::max(x1, x2))
                                                                : std::make_pair(std::min(y1, y2), std::max(y1, y2));
                        auto it = idx.find(tk(z1, f_));
                        int kk;
                        if (it == idx.end()) { kk = (int)pw.size(); idx[tk(z1, f_)] = kk; pw.push_back(Track{z1, f_, {}}); } else kk = it->second;
                        pw[kk].iv.push_back(ivp);
                    }
                }
                w2 = merge_wires(w2, pw);
                v2 = set_or(v2, pv);
                for (int k = 0; k < np_; ++k) {
                    uint64_t key = pk(pathbuf[3 * k], pathbuf[3 * k + 1], pathbuf[3 * k + 2]);
                    if (tset.insert(key).second) src.insert(src.end(), {pathbuf[3 * k], pathbuf[3 * k + 1], pathbuf[3 * k + 2]});
                }
                for (size_t k = 0; k < tgt.size(); k += 3) {
                    uint64_t key = pk(tgt[k], tgt[k + 1], tgt[k + 2]);
                    if (tset.insert(key).second) src.insert(src.end(), {tgt[k], tgt[k + 1], tgt[k + 2]});
                }
            }
            if (fail) { apply(g, w, v, +1); stats[7]++; continue; }
        }
        if (!net_valid(g, w2, v2, ap, poff_local.data(), npins)) { apply(g, w, v, +1); stats[8]++; continue; }
        double nw = marginal(g, w2, v2);
        double t_new = 0.0, p_new = 0.0;
        tp_price(s, w2, v2, t_new, p_new);
        if (last_dl - dl_old > nsb[s]) { apply(g, w, v, +1); stats[10]++; stats[16]++; continue; }   // slack budget
        // soft TNS price (score units): Fp[5] x the part of the delay increase beyond the net's positive slack
        double pen = Fp[5] > 0.0 ? Fp[5] * std::max(0.0, (last_dl - dl_old) - nss[s]) : 0.0;
        double pr_old = old + (t_old + p_old) / w4, pr_new = nw + (t_new + p_new + pen) / w4;
        if (pr_new < pr_old - 1e-9) {
            apply(g, w2, v2, +1);
            ns.w = std::move(w2); ns.v = std::move(v2);
            if (nw > old) stats[13]++;
            if (mode == 2) stats[14]++;
            gain[0] += w4 * (old - nw); gain[1] += t_old - t_new; gain[2] += p_old - p_new; gain[3] += w4 * (pr_old - pr_new);
            changed[s] = 1; stats[9]++;
        } else {
            apply(g, w, v, +1); stats[10]++;
        }
    }
    // output: changed slots sorted by route index; emit_tuples order (wires in dict order, then sorted vias)
    R_changed.clear(); R_off.assign(1, 0); R_seg.clear();
    std::vector<int> sl;
    for (int s = 0; s < n; ++s) if (changed[s]) sl.push_back(s);
    std::sort(sl.begin(), sl.end(), [&](int p, int q) { return ridx[p] < ridx[q]; });
    for (int s : sl) {
        const NetSt &ns = cur[s];
        for (const Track &t : ns.w)
            for (auto &p : t.iv) {
                if (dirs[t.z] == 0) R_seg.insert(R_seg.end(), {p.first, t.f, t.z, p.second, t.f, t.z});
                else R_seg.insert(R_seg.end(), {t.f, p.first, t.z, t.f, p.second, t.z});
            }
        std::vector<std::array<int32_t, 3>> vv;
        for (const SE &e : ns.v.t) if (e.k) vv.push_back({e.x, e.y, e.z});
        std::sort(vv.begin(), vv.end());
        for (auto &q : vv) R_seg.insert(R_seg.end(), {q[0], q[1], q[2], q[0], q[1], q[2] + 1});
        R_changed.push_back(s);
        R_off.push_back((i64)(R_seg.size() / 6));
    }
    stats[11] = (i64)R_changed.size();
    return (int)R_changed.size();
}

i64 rs_nseg() { return (i64)(R_seg.size() / 6); }
void rs_get(int32_t *slots, int64_t *off, int32_t *seg) {
    std::memcpy(slots, R_changed.data(), R_changed.size() * sizeof(int));
    std::memcpy(off, R_off.data(), R_off.size() * sizeof(i64));
    std::memcpy(seg, R_seg.data(), R_seg.size() * sizeof(int32_t));
}

}
