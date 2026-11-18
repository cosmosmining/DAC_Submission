// l5_kern.cc: C++ drivers of the late region of tools/late5/chain_late.py (--late5 1).
// Build (host): g++ -O3 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread -o libl5_kern.so l5_kern.cc
// Every entry point is extern "C" on plain arrays (no Python object), so a C++ main can call it.
//
// l5_swap: the demand change of fastmode.State.revert for a set of nets = subtract the build_demand_fast contribution of
//   their current rows and add that of their undo rows (the rules of tools/fast3x/h_dem.cc hd_add, copied: merged wire runs
//   per net/layer/track add 2 per cell, unique unit vias not covered by a run of the same net add 1+1 / 2). timing_promote
//   .to_wires_vias + fast_apply.apply give the same contributions; all are small integers, so the float64 map is the same
//   for any order of additions. The caller checks that every index is inside the map first (then nothing can fail).
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <thread>

static int add_rows(double* d, int64_t L, int64_t X, int64_t Y, int64_t s0, int64_t s1, int64_t s2, const int64_t* dirs,
                    const int64_t* seg, int64_t m, double sign) {
  std::vector<int64_t> idx(m);
  for (int64_t i = 0; i < m; i++) idx[i] = i;
  bool grouped = true;
  for (int64_t i = 1; i < m && grouped; i++) if (seg[7 * i] < seg[7 * (i - 1)]) grouped = false;
  if (!grouped) std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return seg[7 * a] < seg[7 * b]; });
  auto D = [&](int64_t z, int64_t x, int64_t y) -> double& { return d[z * s0 + x * s1 + y * s2]; };
  struct Wr { int64_t z, f, a, b; };
  std::vector<Wr> w[2], mg[2];
  std::vector<std::tuple<int64_t, int64_t, int64_t>> vp;
  const double two = 2.0 * sign, one = 1.0 * sign;
  int64_t s = 0;
  while (s < m) {
    int64_t n = seg[7 * idx[s]], e = s;
    while (e < m && seg[7 * idx[e]] == n) e++;
    w[0].clear(); w[1].clear(); mg[0].clear(); mg[1].clear(); vp.clear();
    for (int64_t t = s; t < e; t++) {
      const int64_t* r = seg + 7 * idx[t];
      int64_t x1 = r[1], y1 = r[2], z1 = r[3], x2 = r[4], y2 = r[5], z2 = r[6];
      if (z1 != z2) {
        int64_t zl = std::min(z1, z2), zh = std::max(z1, z2);
        if (zl < 0 || zh > L || x1 < 0 || x1 >= X || y1 < 0 || y1 >= Y) return -1;
        for (int64_t z = zl; z < zh; z++) vp.emplace_back(z, x1, y1);
        continue;
      }
      int64_t zc = std::min(std::max(z1, (int64_t)0), L - 1);
      int64_t dz = dirs[zc];
      if (dz == 0 && y1 == y2 && x1 != x2) w[0].push_back({z1, y1, std::min(x1, x2), std::max(x1, x2)});
      else if (dz == 1 && x1 == x2 && y1 != y2) w[1].push_back({z1, x1, std::min(y1, y2), std::max(y1, y2)});
    }
    for (int dd = 0; dd < 2; dd++) {
      auto& v = w[dd];
      std::stable_sort(v.begin(), v.end(), [](const Wr& p, const Wr& q) { return std::tie(p.z, p.f, p.a) < std::tie(q.z, q.f, q.a); });
      const int64_t T = dd == 0 ? X : Y, F = dd == 0 ? Y : X;
      for (size_t i = 0; i < v.size();) {
        size_t j = i + 1; int64_t ma = v[i].a, mb = v[i].b;
        while (j < v.size() && v[j].z == v[i].z && v[j].f == v[i].f && v[j].a <= mb) { mb = std::max(mb, v[j].b); j++; }
        int64_t z = v[i].z, f = v[i].f;
        if (z < 0 || z >= L || f < 0 || f >= F || ma < 0 || mb > T) return -1;
        mg[dd].push_back({z, f, ma, mb});
        for (int64_t c = ma; c < mb; c++) { if (dd == 0) D(z, c, f) += two; else D(z, f, c) += two; }
        i = j;
      }
    }
    std::sort(vp.begin(), vp.end());
    vp.erase(std::unique(vp.begin(), vp.end()), vp.end());
    for (auto& p : vp) {
      int64_t z = std::get<0>(p), x = std::get<1>(p), y = std::get<2>(p);
      if (z == 0) continue;
      if (z < 0 || z >= L) return -1;
      int64_t dd = dirs[z];
      if (dd != 0 && dd != 1) continue;
      int64_t f = dd == 0 ? y : x, t = dd == 0 ? x : y;
      auto& M = mg[dd];
      auto it = std::upper_bound(M.begin(), M.end(), std::make_tuple(z, f, t),
                                 [](const std::tuple<int64_t, int64_t, int64_t>& q, const Wr& g) { return q < std::tie(g.z, g.f, g.a); });
      bool cov = false;
      if (it != M.begin()) { --it; cov = it->z == z && it->f == f && it->a <= t && t <= it->b; }
      if (cov) continue;
      const int64_t T = dd == 0 ? X : Y;
      bool mid = t > 0 && t < T - 1, hi = !mid && t > 0, lo = !mid && !hi && t < T - 1;
      if (dd == 0) {
        if (mid) { D(z, x - 1, y) += one; D(z, x, y) += one; }
        else if (hi) D(z, x - 1, y) += two;
        else if (lo) D(z, x, y) += two;
      } else {
        if (mid) { D(z, x, y - 1) += one; D(z, x, y) += one; }
        else if (hi) D(z, x, y - 1) += two;
        else if (lo) D(z, x, y) += two;
      }
    }
    s = e;
  }
  return 0;
}

// rows: [m,7] int64 (net, x1, y1, z1, x2, y2, z2). Returns 1 when every row is inside the map (z in [0,L-1], x in [0,X),
// y in [0,Y) for both ends), else 0 (the caller uses the Python revert).
static int rows_ok(const int64_t* r, int64_t m, int64_t L, int64_t X, int64_t Y) {
  for (int64_t i = 0; i < m; i++) {
    const int64_t* q = r + 7 * i;
    if (q[3] < 0 || q[6] < 0 || q[3] > L - 1 || q[6] > L - 1) return 0;
    if (q[1] < 0 || q[4] < 0 || q[1] >= X || q[4] >= X) return 0;
    if (q[2] < 0 || q[5] < 0 || q[2] >= Y || q[5] >= Y) return 0;
  }
  return 1;
}

extern "C" int l5_swap(double* d, int64_t L, int64_t X, int64_t Y, int64_t s0, int64_t s1, int64_t s2, const int64_t* dirs,
                       const int64_t* rold, int64_t mold, const int64_t* rnew, int64_t mnew) {
  if (!rows_ok(rold, mold, L, X, Y) || !rows_ok(rnew, mnew, L, X, Y)) return 1;   // nothing changed
  if (add_rows(d, L, X, Y, s0, s1, s2, dirs, rold, mold, -1.0) != 0) return -1;    // cannot happen after rows_ok
  if (add_rows(d, L, X, Y, s0, s1, s2, dirs, rnew, mnew, 1.0) != 0) return -2;
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// l5_ovf_score: the per-route-net score of chain_j.ovf_order (chain_t3.ovf_order + --ovfround_open) on given marginal costs
//   Hm[L][X][Y] (contiguous; Hm[0] = 0) and the current wire rows (m x 7, route net first, in cur_arr order). Same float
//   operations in the same order: per layer z = 1..L-1 (skipped when it has no rows or Hm[z] is all zero) the prefix sums
//   along the layer direction (sequential, == np.cumsum), v = P[b+1][f] - P[a][f], sc[net] += v in row order (== np.add.at).
// Returns 0, or 1 when an index is outside the arrays (the caller then runs the numpy code).
extern "C" int l5_ovf_score(int64_t L, int64_t X, int64_t Y, const double* Hm, const int64_t* dirs, const int64_t* rows,
                            int64_t m, int64_t nnet, double* sc) {
  for (int64_t i = 0; i < m; i++) {
    const int64_t* r = rows + 7 * i;
    if (r[0] < 0 || r[0] >= nnet) return 1;
    int64_t z = r[3];
    if (z < 1 || z >= L) continue;
    int64_t dz = dirs[z];
    if (dz == 0) { if (r[1] < 0 || r[4] < 0 || r[1] >= X || r[4] >= X || r[2] < 0 || r[2] >= Y) return 1; }
    else { if (r[2] < 0 || r[5] < 0 || r[2] >= Y || r[5] >= Y || r[1] < 0 || r[1] >= X) return 1; }
  }
  std::vector<double> P;
  std::vector<int64_t> lay;
  for (int64_t z = 1; z < L; z++) {
    lay.clear();
    for (int64_t i = 0; i < m; i++) if (rows[7 * i + 3] == z) lay.push_back(i);
    if (lay.empty()) continue;
    const double* H = Hm + z * X * Y;
    bool any = false;
    for (int64_t k = 0; k < X * Y; k++) if (H[k] != 0.0) { any = true; break; }
    if (!any) continue;
    if (dirs[z] == 0) {                       // P[(X+1) x Y], P[x+1][y] = P[x][y] + H[x][y]
      P.assign((X + 1) * Y, 0.0);
      for (int64_t y = 0; y < Y; y++) P[1 * Y + y] = H[y];
      for (int64_t x = 1; x < X; x++) for (int64_t y = 0; y < Y; y++) P[(x + 1) * Y + y] = P[x * Y + y] + H[x * Y + y];
      for (int64_t i : lay) {
        const int64_t* r = rows + 7 * i;
        int64_t a = std::min(r[1], r[4]), b = std::max(r[1], r[4]), f = r[2];
        sc[r[0]] += P[(b + 1) * Y + f] - P[a * Y + f];
      }
    } else {                                  // P[X x (Y+1)], P[x][y+1] = P[x][y] + H[x][y]
      P.assign(X * (Y + 1), 0.0);
      for (int64_t x = 0; x < X; x++) {
        double* p = &P[x * (Y + 1)];
        const double* h = H + x * Y;
        p[1] = h[0];
        for (int64_t y = 1; y < Y; y++) p[y + 1] = p[y] + h[y];
      }
      for (int64_t i : lay) {
        const int64_t* r = rows + 7 * i;
        int64_t a = std::min(r[2], r[5]), b = std::max(r[2], r[5]), f = r[1];
        sc[r[0]] += P[f * (Y + 1) + b + 1] - P[f * (Y + 1) + a];
      }
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// l5_selc_pairs / l5_selc_greedy: ovf2_lib.select_c_fast (== ovf_lib.select_c) in two calls.
//  pairs: wire rows (m x 7) -> the sorted unique (net, cell) pairs with an overflowed cell (ov[L*X*Y] uint8), then the pool
//   filter (slack_r[net] >= guard and npins_r[net] <= maxp). Results are kept in static buffers: counts are returned in
//   out[0..3] = (nets crossing, pool nets, pool pairs, unique pool cells); l5_selc_cells writes the unique pool cells (sorted)
//   so the caller can evaluate the cost expression on them with numpy (same elementwise bits as on the full map).
//  greedy: cell costs (aligned with the unique cell list) and slack -> select_c's weighted greedy hitting set with a binary
//   heap on (-value, net) (the same total order as Python's heapq on the same tuples, so the same pops), the same sums in
//   the same order, the same lazy re-push test. Writes the chosen nets sorted; returns their count.
#include <unordered_map>
static std::vector<int64_t> SC_pn, SC_pc, SC_pool, SC_pool_off, SC_cells, SC_pc_cell;

extern "C" int l5_selc_pairs(int64_t L, int64_t X, int64_t Y, const uint8_t* ov, const int64_t* dirs, const int64_t* rows, int64_t m,
                             const double* slack_r, const int64_t* npins_r, int64_t nnet, double guard, int64_t maxp, int64_t* out) {
  const int64_t N = L * X * Y;
  std::vector<int64_t> keys;
  for (int64_t i = 0; i < m; i++) {
    const int64_t* r = rows + 7 * i;
    int64_t n = r[0], x1 = r[1], y1 = r[2], z1 = r[3], x2 = r[4], y2 = r[5];
    int64_t zc = std::min(std::max(z1, (int64_t)0), L - 1);
    bool hor = dirs[zc] == 0 && y1 == y2 && x1 != x2 && z1 > 0;
    bool ver = dirs[zc] == 1 && x1 == x2 && y1 != y2 && z1 > 0;
    if (!hor && !ver) continue;
    if (n < 0 || n >= nnet || z1 >= L) return 1;
    if (hor) {
      int64_t lo = std::min(x1, x2), hi = std::max(x1, x2);
      if (lo < 0 || hi > X || y1 < 0 || y1 >= Y) return 1;
      for (int64_t t = lo; t < hi; t++) { int64_t lin = (z1 * X + t) * Y + y1; if (ov[lin]) keys.push_back(n * N + lin); }
    } else {
      int64_t lo = std::min(y1, y2), hi = std::max(y1, y2);
      if (lo < 0 || hi > Y || x1 < 0 || x1 >= X) return 1;
      for (int64_t t = lo; t < hi; t++) { int64_t lin = (z1 * X + x1) * Y + t; if (ov[lin]) keys.push_back(n * N + lin); }
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  int64_t ncross = 0, last = -1;
  SC_pn.clear(); SC_pc.clear();
  for (int64_t k : keys) {
    int64_t n = k / N, c = k % N;
    if (n != last) { ncross++; last = n; }
    if (slack_r[n] >= guard && npins_r[n] <= maxp) { SC_pn.push_back(n); SC_pc.push_back(c); }
  }
  SC_pool.clear(); SC_pool_off.clear();
  for (size_t j = 0; j < SC_pn.size(); j++) {
    if (SC_pool.empty() || SC_pool.back() != SC_pn[j]) { SC_pool.push_back(SC_pn[j]); SC_pool_off.push_back((int64_t)j); }
  }
  SC_pool_off.push_back((int64_t)SC_pn.size());
  SC_cells = SC_pc;
  std::sort(SC_cells.begin(), SC_cells.end());
  SC_cells.erase(std::unique(SC_cells.begin(), SC_cells.end()), SC_cells.end());
  SC_pc_cell.resize(SC_pc.size());
  for (size_t j = 0; j < SC_pc.size(); j++)
    SC_pc_cell[j] = std::lower_bound(SC_cells.begin(), SC_cells.end(), SC_pc[j]) - SC_cells.begin();
  out[0] = ncross; out[1] = (int64_t)SC_pool.size(); out[2] = (int64_t)SC_pn.size(); out[3] = (int64_t)SC_cells.size();
  return 0;
}

extern "C" void l5_selc_cells(int64_t* cells) { for (size_t j = 0; j < SC_cells.size(); j++) cells[j] = SC_cells[j]; }

// CPython 3.12 builtin sum() of floats (start 0): the first item as is, then Neumaier's compensated summation, the
// compensation added at the end when it is non-zero and finite (Python/bltinmodule.c builtin_sum_impl).
struct PySum {
  double f = 0.0, c = 0.0; bool any = false;
  inline void add(double x) {
    if (!any) { f = 0.0 + x; any = true; return; }
    double t = f + x;
    if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x; else c += (x - t) + f;
    f = t;
  }
  inline double get() const { double r = f; if (c != 0.0 && std::isfinite(c)) r += c; return any ? r : 0.0; }
};
struct HE { double v; int64_t i; };
static inline bool he_less(const HE& a, const HE& b) { return a.v < b.v || (!(b.v < a.v) && a.i < b.i); }   // tuple order

// min-heap with Python heapq's tuple order
struct MinHeap {
  std::vector<HE> h;
  void push(HE e) { h.push_back(e); std::push_heap(h.begin(), h.end(), [](const HE& a, const HE& b) { return he_less(b, a); }); }
  HE pop() { std::pop_heap(h.begin(), h.end(), [](const HE& a, const HE& b) { return he_less(b, a); }); HE e = h.back(); h.pop_back(); return e; }
  const HE& top() const { return h.front(); }
  bool empty() const { return h.empty(); }
};

extern "C" int64_t l5_selc_greedy(const double* ccost, const double* slack_r, int64_t cap_k, int64_t* chosen, double* info) {
  const size_t np_ = SC_pool.size();
  std::vector<double> wt(np_);
  std::vector<uint8_t> unc(SC_cells.size(), 1);
  int64_t nunc = (int64_t)SC_cells.size();
  MinHeap hq; hq.h.reserve(np_);
  std::vector<HE> init(np_);
  std::unordered_map<int64_t, size_t> pos; pos.reserve(np_ * 2);
  for (size_t j = 0; j < np_; j++) {
    int64_t i = SC_pool[j];
    double x = 0.5 - slack_r[i];
    double mx = x > 0.0 ? x : 0.0;
    wt[j] = 1.0 + mx / 0.5;
    PySum ps;
    for (int64_t k = SC_pool_off[j]; k < SC_pool_off[j + 1]; k++) ps.add(ccost[SC_pc_cell[k]]);
    double s = ps.get();
    init[j] = {(-s) / wt[j], i};
    pos[i] = j;
  }
  hq.h = init;
  std::make_heap(hq.h.begin(), hq.h.end(), [](const HE& a, const HE& b) { return he_less(b, a); });
  int64_t nch = 0;
  while (!hq.empty() && nunc > 0 && nch < cap_k) {
    HE e = hq.pop();
    size_t j = pos[e.i];
    PySum ps;
    for (int64_t k = SC_pool_off[j]; k < SC_pool_off[j + 1]; k++) if (unc[SC_pc_cell[k]]) ps.add(ccost[SC_pc_cell[k]]);
    double val = ps.get() / wt[j];
    if (val <= 0) continue;
    if (!hq.empty() && val < -hq.top().v - 1e-12) { hq.push({-val, e.i}); continue; }
    chosen[nch++] = e.i;
    for (int64_t k = SC_pool_off[j]; k < SC_pool_off[j + 1]; k++) if (unc[SC_pc_cell[k]]) { unc[SC_pc_cell[k]] = 0; nunc--; }
  }
  std::sort(chosen, chosen + nch);
  PySum cu_;
  for (size_t c = 0; c < SC_cells.size(); c++) if (unc[c]) cu_.add(ccost[c]);
  double cu = cu_.get();
  info[0] = (double)nunc; info[1] = cu;
  return nch;
}
