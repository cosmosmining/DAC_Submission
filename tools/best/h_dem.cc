// h_dem.cc: fastmode.build_demand_fast in C++ (same rules, per net):
//  * wires: rows with z1 == z2, layer direction dirs[clip(z1)] == 0 and y1 == y2, x1 != x2 (horizontal) or == 1 and
//    x1 == x2, y1 != y2 (vertical); per (net, z, track) the intervals [min, max] sorted by start are merged while
//    start <= running max end; every merged run adds 2 to the cells [ma, mb) along the track;
//  * vias: rows with z1 != z2 give the points (net, z, x1, y1) for z in [min z, max z), unique per net; z == 0 skipped; a
//    point covered by a merged run of the same net on layer z (track = its y (dir 0) / x (dir 1), ma <= t <= mb) skipped;
//    the others add 1 + 1 at t-1, t (0 < t < T-1), 2 at t-1 (t >= T-1, t > 0) or 2 at t (t == 0, T > 1) along dirs[z].
// Every addition is a small integer, so the float64 map equals build_demand_fast's for any order of additions.
// sign = +1 adds, -1 subtracts the contributions. Returns 0, or -1 when an index would be outside the map (the caller
// then uses the Python function; numpy would wrap negative indices).
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>

extern "C" int hd_add(double* d, int64_t L, int64_t X, int64_t Y, int64_t s0, int64_t s1, int64_t s2, const int64_t* dirs, const int64_t* seg,
                      int64_t m, double sign) {   // s0..s2: element strides of d (any memory layout)
  // rows grouped by net: process maximal blocks of equal net id in order; nets may repeat in separate blocks only if the
  // net column is not grouped -> handled by sorting an index (stable) when needed
  std::vector<int64_t> idx(m);
  for (int64_t i = 0; i < m; i++) idx[i] = i;
  bool grouped = true;
  for (int64_t i = 1; i < m && grouped; i++) if (seg[7 * i] < seg[7 * (i - 1)]) grouped = false;
  if (!grouped) std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return seg[7 * a] < seg[7 * b]; });
  auto D = [&](int64_t z, int64_t x, int64_t y) -> double& { return d[z * s0 + x * s1 + y * s2]; };
  struct Wr { int64_t z, f, a, b; };
  struct Mg { int64_t z, f, a, b; };
  std::vector<Wr> w[2]; std::vector<Mg> mg[2];
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
        if (zl < 0 || zh > L || x1 < 0 || x1 >= X || y1 < 0 || y1 >= Y) return -1;   // numpy would wrap / collide keys
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
        size_t j = i; int64_t ma = v[i].a, mb = v[i].b;
        j++;
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
      // covered by a merged run of this net on (z, f) with ma <= t <= mb
      auto& M = mg[dd];
      auto it = std::upper_bound(M.begin(), M.end(), std::make_tuple(z, f, t),
                                 [](const std::tuple<int64_t, int64_t, int64_t>& q, const Mg& g) { return q < std::tie(g.z, g.f, g.a); });
      bool cov = false;
      if (it != M.begin()) { --it; cov = it->z == z && it->f == f && it->a <= t && t <= it->b; }
      if (cov) continue;
      const int64_t T = dd == 0 ? X : Y;
      if (x < 0 || x >= X || y < 0 || y >= Y) return -1;
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
