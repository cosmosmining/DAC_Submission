// s4_esc.cc: item 4c, exact C++ twins of repair3d_astar.apply_fast and
// repair3d_astar.marginal_fast (both paths: the Python small-net loop with its dict insertion order and sequential float sum,
// and the numpy big-net path: sorted unique cells, np.sum pairwise summation). The pin-escape search (escape3d.escape) calls
// them twice per candidate. Same expressions, libm exp, -ffp-contract=off. Build: see tools/speed4/build_s4.sh.
#include <cmath>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
typedef int64_t i64;

namespace {
const double CAP_EPS = 1e-3;
struct LA { double *d; const double *c; const double *u; i64 sz, sx, sy, X, Y, L; const int8_t *dirs; };

inline i64 at(const LA &la, i64 z, i64 x, i64 y) { return z * la.sz + x * la.sx + y * la.sy; }

struct Wires {   // (z, fixed) -> intervals, from the flattened dict
  std::unordered_map<i64, std::vector<std::pair<i64, i64>>> m;
  Wires(const LA &la, i64 n, const i64 *z, const i64 *f, const i64 *a, const i64 *b) {
    for (i64 i = 0; i < n; i++) m[z[i] * (la.X + la.Y + 1) + f[i]].push_back({a[i], b[i]});
  }
  bool covered(const LA &la, i64 x, i64 y, i64 z) const {
    i64 fx = la.dirs[z] == 0 ? y : x, pos = la.dirs[z] == 0 ? x : y;
    auto it = m.find(z * (la.X + la.Y + 1) + fx);
    if (it == m.end()) return false;
    for (auto &p : it->second) if (p.first <= pos && pos <= p.second) return true;
    return false;
  }
};

inline int via_cells(const LA &la, i64 x, i64 y, i64 z, i64 out[2][3]) {
  if (la.dirs[z] == 0) {
    if (0 < x && x < la.X - 1) { out[0][0] = x - 1; out[0][1] = y; out[0][2] = 1; out[1][0] = x; out[1][1] = y; out[1][2] = 1; return 2; }
    if (x > 0) { out[0][0] = x - 1; out[0][1] = y; out[0][2] = 2; return 1; }
    if (x < la.X - 1) { out[0][0] = x; out[0][1] = y; out[0][2] = 2; return 1; }
    return 0;
  }
  if (0 < y && y < la.Y - 1) { out[0][0] = x; out[0][1] = y - 1; out[0][2] = 1; out[1][0] = x; out[1][1] = y; out[1][2] = 1; return 2; }
  if (y > 0) { out[0][0] = x; out[0][1] = y - 1; out[0][2] = 2; return 1; }
  if (y < la.Y - 1) { out[0][0] = x; out[0][1] = y; out[0][2] = 2; return 1; }
  return 0;
}

inline double cell_cost(double u, double d, double c) {          // escape3d.cell_cost (Python floats)
  if (c > CAP_EPS) { double e = 0.25 * d - 0.5 * c; if (700.0 < e) e = 700.0; return u * std::exp(e); }
  if (d > 0) { double e = 0.75 * d; if (700.0 < e) e = 700.0; return u * std::exp(e); }
  return 0.0;
}

inline double cost_np(double u, double d, double c) {            // repair3d_astar._cost elementwise
  bool open_ = c > CAP_EPS;
  double e = open_ ? 0.25 * d - 0.5 * c : 0.75 * d;
  if (e > 700.0) e = 700.0;                                     // np.minimum(e, 700.0)
  return (open_ || d > 0) ? u * std::exp(e) : 0.0;
}

double pw(const double *a, i64 n) {                              // numpy DOUBLE_pairwise_sum
  if (n < 8) { double r = 0.; for (i64 i = 0; i < n; i++) r += a[i]; return r; }
  if (n <= 128) {
    double r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
    i64 i;
    for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  i64 n2 = n / 2; n2 -= n2 % 8;
  return pw(a, n2) + pw(a + n2, n - n2);
}
}  // namespace

extern "C" {
void s4_apply(double *d, i64 sz, i64 sx, i64 sy, i64 X, i64 Y, i64 L, const int8_t *dirs,
              i64 niv, const i64 *iz, const i64 *ifx, const i64 *ia, const i64 *ib,
              i64 nv, const i64 *vx, const i64 *vy, const i64 *vz, i64 sign) {
  LA la{d, nullptr, nullptr, sz, sx, sy, X, Y, L, dirs};
  for (i64 i = 0; i < niv; i++) {
    i64 z = iz[i], f = ifx[i];
    if (dirs[z] == 0) for (i64 t = ia[i]; t < ib[i]; t++) d[at(la, z, t, f)] += 2 * sign;
    else for (i64 t = ia[i]; t < ib[i]; t++) d[at(la, z, f, t)] += 2 * sign;
  }
  Wires W(la, niv, iz, ifx, ia, ib);
  i64 cells[2][3];
  for (i64 k = 0; k < nv; k++) {
    i64 x = vx[k], y = vy[k], z = vz[k];
    if (z == 0 || W.covered(la, x, y, z)) continue;
    int nc = via_cells(la, x, y, z, cells);
    for (int j = 0; j < nc; j++) d[at(la, z, cells[j][0], cells[j][1])] += cells[j][2] * sign;
  }
}

double s4_marginal(double *d, const double *c, const double *u, i64 sz, i64 sx, i64 sy, i64 X, i64 Y, i64 L, const int8_t *dirs,
                   i64 nkeys, i64 niv, const i64 *iz, const i64 *ifx, const i64 *ia, const i64 *ib,
                   i64 nv, const i64 *vx, const i64 *vy, const i64 *vz) {
  LA la{d, c, u, sz, sx, sy, X, Y, L, dirs};
  Wires W(la, niv, iz, ifx, ia, ib);
  i64 tot_len = 0;
  for (i64 i = 0; i < niv; i++) tot_len += ib[i] - ia[i];
  i64 cells[2][3];
  if (nkeys <= 64 && tot_len + 2 * nv <= 2 * 64) {             // _marginal_small
    std::vector<i64> keys; std::vector<i64> val; std::unordered_map<i64, size_t> idx;
    auto add = [&](i64 z, i64 x, i64 y, i64 dd) {
      i64 k = (z * X + x) * Y + y;
      auto it = idx.find(k);
      if (it == idx.end()) { idx[k] = keys.size(); keys.push_back(k); val.push_back(dd); }
      else val[it->second] += dd;
    };
    for (i64 i = 0; i < niv; i++) {
      i64 z = iz[i], f = ifx[i];
      for (i64 t = ia[i]; t < ib[i]; t++) { if (dirs[z] == 0) add(z, t, f, 2); else add(z, f, t, 2); }
    }
    for (i64 k = 0; k < nv; k++) {
      i64 x = vx[k], y = vy[k], z = vz[k];
      if (z == 0 || W.covered(la, x, y, z)) continue;
      int nc = via_cells(la, x, y, z, cells);
      for (int j = 0; j < nc; j++) add(z, cells[j][0], cells[j][1], cells[j][2]);
    }
    double tot = 0.0;
    for (size_t q = 0; q < keys.size(); q++) {
      i64 k = keys[q]; i64 z = k / (X * Y), r = k % (X * Y), x = r / Y, y = r % Y;
      if (z == 0) continue;
      double dv = d[at(la, z, x, y)], cv = c[at(la, z, x, y)], uv = u[z];
      tot += cell_cost(uv, dv + (double)val[q], cv) - cell_cost(uv, dv, cv);
    }
    return tot;
  }
  std::vector<std::pair<i64, double>> lin;                      // numpy path
  for (i64 i = 0; i < niv; i++) {
    i64 z = iz[i], f = ifx[i];
    if (!(ib[i] > ia[i] && z > 0)) continue;
    for (i64 t = ia[i]; t < ib[i]; t++) {
      i64 x = dirs[z] == 0 ? t : f, y = dirs[z] == 0 ? f : t;
      lin.push_back({(z * X + x) * Y + y, 2.0});
    }
  }
  for (i64 k = 0; k < nv; k++) {
    i64 x = vx[k], y = vy[k], z = vz[k];
    if (z == 0 || W.covered(la, x, y, z)) continue;
    int nc = via_cells(la, x, y, z, cells);
    for (int j = 0; j < nc; j++) lin.push_back({(z * X + cells[j][0]) * Y + cells[j][1], (double)cells[j][2]});
  }
  if (lin.empty()) return 0.0;
  std::stable_sort(lin.begin(), lin.end(), [](const std::pair<i64, double> &a, const std::pair<i64, double> &b) { return a.first < b.first; });
  std::vector<double> vals;
  for (size_t i = 0; i < lin.size();) {
    size_t j = i; double dd = 0.0;
    while (j < lin.size() && lin[j].first == lin[i].first) { dd += lin[j].second; j++; }
    i64 k = lin[i].first; i64 z = k / (X * Y), r = k % (X * Y), x = r / Y, y = r % Y;
    double dv = d[at(la, z, x, y)], cv = c[at(la, z, x, y)], uv = u[z];
    vals.push_back(cost_np(uv, dv + dd, cv) - cost_np(uv, dv, cv));
    i = j;
  }
  return pw(vals.data(), (i64)vals.size());
}
}
