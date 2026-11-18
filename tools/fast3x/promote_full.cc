// promote_full.cc: one C++ call per promotion round.
// Exact port of the per-net loop of tools/c1hfast/promote_cpp.py::promote_cpp and tools/tilefix/promote_pw.py::promote_cpp
// (to_wires_vias, apply, used_access_points, nodes_from_aps, segments_to_polys, tree_connects, FLUTE edges + L choice,
// the layer DP of promote_kern.cc / promote_pw_kern.cc copied verbatim, route_net, commit_net, segs_of).
// Same iteration orders as the Python (dict insertion order, sorted via sets), same floating-point operation order.
// Build: tools/fast3x/build_full.sh  (clang++ -O2 -ffp-contract=off -shared -fPIC; no fast-math, no FMA)
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <thread>
#include <atomic>
#include <chrono>

typedef int (*flute_fn)(int, int*, int*, int*, int*, int*);

static const double CAP_EPS = 1e-3, EXP_CLIP = 700.0;

// ───────────────────────────── layer DP kernel (verbatim from promote_kern.cc / promote_pw_kern.cc) ─────────────────────────────
struct Ctx {
  const double *d, *c, *u, *R, *C, *Rv; const int32_t* dirs;
  const int64_t *sd, *sc; int L, X, Y; double w4, um, tw, cl, rd, E05M1, E025M1, E15, pn;
};
static inline double pw_sum(const double* a, long n) {   // numpy pairwise_sum
  if (n < 8) { double r = 0.; for (long i = 0; i < n; i++) r += a[i]; return r; }
  if (n <= 128) {
    double r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
    long i; for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  long n2 = n / 2; n2 -= n2 % 8;
  return pw_sum(a, n2) + pw_sum(a + n2, n - n2);
}
static inline double wm(const Ctx& k, int z, long x, long y) {
  double dz = k.d[z*k.sd[0]+x*k.sd[1]+y*k.sd[2]], cz = k.c[z*k.sc[0]+x*k.sc[1]+y*k.sc[2]], u = k.u[z];
  if (cz > CAP_EPS) {
    return u * std::exp(std::min(0.25 * dz - 0.5 * cz, EXP_CLIP)) * k.E05M1;
  } else if (dz > 0) {
    return u * (std::exp(std::min(0.75 * (dz + 2), EXP_CLIP)) - std::exp(std::min(0.75 * dz, EXP_CLIP)));
  }
  return u * k.E15;
}
template <bool PW>
static double run_cost(const Ctx& k, std::vector<double>& buf, int dr, int fixed, int a, int b, int z) {
  long n = b - a;
  for (long i = 0; i < n; i++) buf[i] = dr == 0 ? wm(k, z, a + i, fixed) : wm(k, z, fixed, a + i);
  double ov = pw_sum(buf.data(), n);
  double l = (double)n * k.um;
  double dly = l * (k.R[z] * k.cl + k.C[z] * k.rd) / 1000.0 + k.R[z] * k.C[z] * l * l / 2.0 / 1000.0;
  if (PW) return k.w4 * ov + k.tw * dly + k.pn * k.C[z] * l;
  return k.w4 * ov + k.tw * dly;
}
static double via_cost(const Ctx& k, int x, int y, int za, int zb) {
  int lo = za <= zb ? za : zb, hi = za <= zb ? zb : za;
  double tot = 0.0;
  for (int z = lo; z < hi; z++) {
    tot += k.tw * k.Rv[z] * 2.0 / 1000.0;
    if (z != 0 && (k.dirs[z] == 0 || k.dirs[z] == 1)) {
      double dz = k.d[z*k.sd[0]+x*k.sd[1]+y*k.sd[2]], cz = k.c[z*k.sc[0]+x*k.sc[1]+y*k.sc[2]], u = k.u[z], m;
      if (cz > CAP_EPS) m = 2.0 * u * std::exp(std::min(0.25 * dz - 0.5 * cz, EXP_CLIP)) * k.E025M1;
      else {
        double base = dz > 0 ? std::exp(std::min(0.75 * dz, EXP_CLIP)) : 0.0;
        m = 2.0 * u * (std::exp(std::min(0.75 * (dz + 1), EXP_CLIP)) - base);
      }
      tot += k.w4 * m;
    }
  }
  return tot;
}
static double stack_cost(const Ctx& k, int x, int y, int lo, int hi, int z) {
  if (lo < 0) return 0.0;
  if (z > hi) return via_cost(k, x, y, hi, z);
  if (z < lo) return via_cost(k, x, y, z, lo);
  return 0.0;
}
typedef std::array<int32_t, 8> Run;   // dir, fixed, lo, hi, sx, sy, ex, ey
// one polyline (assign_batch body for one p); layers written to lay[0..Rn)
template <bool PW>
static double dp_poly(const Ctx& k, const std::vector<int>& H, const std::vector<int>& V, std::vector<double>& buf,
                      const Run* runs, int Rn, const int32_t* sp, int32_t* lay) {
  if (Rn == 0) return 0.0;
  std::vector<std::vector<double>> rc(Rn);
  std::vector<const std::vector<int>*> cand(Rn);
  for (int i = 0; i < Rn; i++) {
    const int32_t* r = runs[i].data();
    cand[i] = r[0] == 0 ? &H : &V;
    for (int z : *cand[i]) rc[i].push_back(run_cost<PW>(k, buf, r[0], r[1], r[2], r[3], z));
  }
  const int32_t* r = runs[0].data();
  std::vector<double> best;
  for (size_t kk = 0; kk < cand[0]->size(); kk++)
    best.push_back(rc[0][kk] + stack_cost(k, r[4], r[5], sp[0], sp[1], (*cand[0])[kk]));
  std::vector<std::vector<int>> back;
  for (int i = 1; i < Rn; i++) {
    const int32_t* ri = runs[i].data();
    std::vector<double> nb; std::vector<int> bk;
    for (size_t kk = 0; kk < cand[i]->size(); kk++) {
      int z = (*cand[i])[kk]; int bj = 0; double bv = 0;
      for (size_t j = 0; j < cand[i - 1]->size(); j++) {
        double v = best[j] + via_cost(k, ri[4], ri[5], (*cand[i - 1])[j], z);
        if (j == 0 || v < bv) { bv = v; bj = (int)j; }
      }
      nb.push_back(bv + rc[i][kk]); bk.push_back(bj);
    }
    best = nb; back.push_back(bk);
  }
  const int32_t* rl = runs[Rn - 1].data();
  int ak = 0; double fv = 0;
  for (size_t kk = 0; kk < cand[Rn - 1]->size(); kk++) {
    double v = best[kk] + stack_cost(k, rl[6], rl[7], sp[2], sp[3], (*cand[Rn - 1])[kk]);
    if (kk == 0 || v < fv) { fv = v; ak = (int)kk; }
  }
  int kk = ak; lay[Rn - 1] = (*cand[Rn - 1])[kk];
  for (int i = Rn - 1; i > 0; i--) { kk = back[i - 1][kk]; lay[i - 1] = (*cand[i - 1])[kk]; }
  return fv;
}

// ───────────────────────────── geometry helpers ─────────────────────────────
typedef std::array<int64_t, 6> Seg;
static inline uint64_t pkey(int64_t x, int64_t y) { return ((uint64_t)(uint32_t)(int32_t)x << 32) | (uint64_t)(uint32_t)(int32_t)y; }
static inline uint64_t zkey(int64_t z, int64_t f) { return ((uint64_t)(uint32_t)(int32_t)z << 32) | (uint64_t)(uint32_t)(int32_t)f; }

// insertion-ordered map key -> vector of intervals (python dict of lists)
struct OrdIv {
  std::unordered_map<uint64_t, int> idx;
  std::vector<uint64_t> keys;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> ivs;
  void clear() { idx.clear(); keys.clear(); ivs.clear(); }
  std::vector<std::pair<int64_t, int64_t>>& at(uint64_t k) {
    auto it = idx.find(k);
    if (it != idx.end()) return ivs[it->second];
    idx.emplace(k, (int)keys.size()); keys.push_back(k); ivs.emplace_back();
    return ivs.back();
  }
  // sort + merge every list in place (to_wires_vias / route_net merge: a <= last hi -> extend)
  void merge_all() {
    for (auto& v : ivs) {
      std::sort(v.begin(), v.end());
      std::vector<std::pair<int64_t, int64_t>> m; m.push_back(v[0]);
      for (size_t t = 1; t < v.size(); t++) {
        if (v[t].first <= m.back().second) m.back().second = std::max(m.back().second, v[t].second);
        else m.push_back(v[t]);
      }
      v.swap(m);
    }
  }
};
static inline int kz(uint64_t k) { return (int)(int32_t)(uint32_t)(k >> 32); }
static inline int64_t kf(uint64_t k) { return (int64_t)(int32_t)(uint32_t)(k & 0xffffffffu); }

struct Grid {
  double* d; const double* c; const int32_t* dirs; const int64_t* sd; int L, X, Y;
  inline double& D(int z, long x, long y) const { return d[z*sd[0]+x*sd[1]+y*sd[2]]; }
};

// timing_promote.apply / route_export.commit_net (wires merged per (z, fixed), vias a sorted unique list)
static void apply_net(const Grid& g, const OrdIv& w, const std::vector<std::array<int64_t, 3>>& vias, int sign) {
  for (size_t t = 0; t < w.keys.size(); t++) {
    int z = kz(w.keys[t]); int64_t f = kf(w.keys[t]);
    for (auto& ab : w.ivs[t]) {
      int64_t a = ab.first, b = ab.second;
      if (g.dirs[z] == 0) { int64_t bb = std::min<int64_t>(b, g.X); for (int64_t x = a; x < bb; x++) g.D(z, x, f) += 2 * sign; }
      else { int64_t bb = std::min<int64_t>(b, g.Y); for (int64_t y = a; y < bb; y++) g.D(z, f, y) += 2 * sign; }
    }
  }
  const int X = g.X, Y = g.Y;
  for (auto& v : vias) {
    int64_t x = v[0], y = v[1]; int z = (int)v[2];
    if (z == 0) continue;
    int dz = g.dirs[z];
    if (dz == 0 || dz == 1) {
      auto it = w.idx.find(zkey(z, dz == 0 ? y : x));
      if (it != w.idx.end()) {
        int64_t q = dz == 0 ? x : y; bool hit = false;
        for (auto& ab : w.ivs[it->second]) if (ab.first <= q && q <= ab.second) { hit = true; break; }
        if (hit) continue;
      }
    }
    if (dz == 0) {
      if (0 < x && x < X - 1) { g.D(z, x - 1, y) += sign; g.D(z, x, y) += sign; }
      else if (x > 0) g.D(z, x - 1, y) += 2 * sign;
      else if (x < X - 1) g.D(z, x, y) += 2 * sign;
    } else if (dz == 1) {
      if (0 < y && y < Y - 1) { g.D(z, x, y - 1) += sign; g.D(z, x, y) += sign; }
      else if (y > 0) g.D(z, x, y - 1) += 2 * sign;
      else if (y < Y - 1) g.D(z, x, y) += 2 * sign;
    }
  }
}

// timing_promote.to_wires_vias
static void to_wires_vias(const Seg* s, long n, const int32_t* dirs, OrdIv& w, std::vector<std::array<int64_t, 3>>& vias) {
  w.clear(); vias.clear();
  for (long t = 0; t < n; t++) {
    int64_t x1 = s[t][0], y1 = s[t][1], z1 = s[t][2], x2 = s[t][3], y2 = s[t][4], z2 = s[t][5];
    if (z1 != z2) { for (int64_t z = std::min(z1, z2); z < std::max(z1, z2); z++) vias.push_back({x1, y1, z}); }
    else if (dirs[z1] == 0 && y1 == y2 && x1 != x2) w.at(zkey(z1, y1)).push_back({std::min(x1, x2), std::max(x1, x2)});
    else if (dirs[z1] == 1 && x1 == x2 && y1 != y2) w.at(zkey(z1, x1)).push_back({std::min(y1, y2), std::max(y1, y2)});
  }
  w.merge_all();
  std::sort(vias.begin(), vias.end()); vias.erase(std::unique(vias.begin(), vias.end()), vias.end());
}

struct Node { int64_t x, y; int lo, hi; };
struct Nodes {           // python dict {(x, y): (lo, hi)} in insertion order
  std::vector<Node> v; std::unordered_map<uint64_t, int> idx;
  void clear() { v.clear(); idx.clear(); }
  const Node* get(int64_t x, int64_t y) const { auto it = idx.find(pkey(x, y)); return it == idx.end() ? nullptr : &v[it->second]; }
};

struct Poly { int n; int64_t x[3], y[3]; };

// route_export.to_runs
static void to_runs(const Poly& p, std::vector<Run>& out) {
  out.clear();
  for (int t = 0; t + 1 < p.n; t++) {
    int64_t x0 = p.x[t], y0 = p.y[t], x1 = p.x[t + 1], y1 = p.y[t + 1];
    if (x0 != x1 && y0 != y1) {
      out.push_back({0, (int32_t)y0, (int32_t)std::min(x0, x1), (int32_t)std::max(x0, x1), (int32_t)x0, (int32_t)y0, (int32_t)x1, (int32_t)y0});
      out.push_back({1, (int32_t)x1, (int32_t)std::min(y0, y1), (int32_t)std::max(y0, y1), (int32_t)x1, (int32_t)y0, (int32_t)x1, (int32_t)y1});
    } else if (x0 != x1) {
      out.push_back({0, (int32_t)y0, (int32_t)std::min(x0, x1), (int32_t)std::max(x0, x1), (int32_t)x0, (int32_t)y0, (int32_t)x1, (int32_t)y1});
    } else if (y0 != y1) {
      out.push_back({1, (int32_t)x0, (int32_t)std::min(y0, y1), (int32_t)std::max(y0, y1), (int32_t)x0, (int32_t)y0, (int32_t)x1, (int32_t)y1});
    }
  }
}

// route_export.segments_to_polys
static void segments_to_polys(const Seg* s, long n, const Nodes& nodes, std::vector<Poly>& polys) {
  polys.clear();
  OrdIv runs;
  for (long t = 0; t < n; t++) {
    int64_t xl = s[t][0], yl = s[t][1], zl = s[t][2], xh = s[t][3], yh = s[t][4], zh = s[t][5];
    if (zl != zh) continue;
    if (yl == yh && xh > xl) runs.at(zkey(0, yl)).push_back({xl, xh});
    else if (xl == xh && yh > yl) runs.at(zkey(1, xl)).push_back({yl, yh});
  }
  struct M { int d; int64_t f, lo, hi; };
  std::vector<M> merged;
  for (size_t t = 0; t < runs.keys.size(); t++) {
    auto& iv = runs.ivs[t]; std::sort(iv.begin(), iv.end());
    int d = kz(runs.keys[t]); int64_t f = kf(runs.keys[t]);
    int64_t lo = iv[0].first, hi = iv[0].second;
    for (size_t q = 1; q < iv.size(); q++) {
      if (iv[q].first <= hi) hi = std::max(hi, iv[q].second);
      else { merged.push_back({d, f, lo, hi}); lo = iv[q].first; hi = iv[q].second; }
    }
    merged.push_back({d, f, lo, hi});
  }
  std::unordered_set<uint64_t> keys;
  for (auto& nd : nodes.v) keys.insert(pkey(nd.x, nd.y));
  for (auto& m : merged) {
    if (m.d == 0) { keys.insert(pkey(m.lo, m.f)); keys.insert(pkey(m.hi, m.f)); }
    else { keys.insert(pkey(m.f, m.lo)); keys.insert(pkey(m.f, m.hi)); }
  }
  std::vector<const M*> hs, vs;
  for (auto& m : merged) (m.d == 0 ? hs : vs).push_back(&m);
  for (auto* h : hs) for (auto* v : vs) {
    int64_t y = h->f, x = v->f;
    if (h->lo <= x && x <= h->hi && v->lo <= y && y <= v->hi) keys.insert(pkey(x, y));
  }
  std::unordered_map<int64_t, std::vector<int64_t>> by_row, by_col;
  for (uint64_t k : keys) { int64_t x = kz(k), y = kf(k); by_row[y].push_back(x); by_col[x].push_back(y); }
  std::vector<int64_t> cuts;
  for (auto& m : merged) {
    cuts.clear(); cuts.push_back(m.lo); cuts.push_back(m.hi);
    auto it = (m.d == 0 ? by_row : by_col).find(m.f);
    if (it != (m.d == 0 ? by_row : by_col).end()) for (int64_t c : it->second) if (m.lo < c && c < m.hi) cuts.push_back(c);
    std::sort(cuts.begin(), cuts.end()); cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    for (size_t q = 0; q + 1 < cuts.size(); q++) {
      Poly p; p.n = 2;
      if (m.d == 0) { p.x[0] = cuts[q]; p.y[0] = m.f; p.x[1] = cuts[q + 1]; p.y[1] = m.f; }
      else { p.x[0] = m.f; p.y[0] = cuts[q]; p.x[1] = m.f; p.y[1] = cuts[q + 1]; }
      polys.push_back(p);
    }
  }
}

// route_export.tree_connects
static bool tree_connects(const Nodes& nodes, const std::vector<Poly>& polys) {
  if (nodes.v.size() <= 1) return true;
  std::unordered_map<uint64_t, int> id; std::vector<int> par;
  auto gid = [&](int64_t x, int64_t y) { auto k = pkey(x, y); auto it = id.find(k); if (it != id.end()) return it->second; int i = (int)par.size(); id.emplace(k, i); par.push_back(i); return i; };
  auto find = [&](int a) { while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; } return a; };
  for (auto& p : polys) { if (p.n == 0) continue; int a = find(gid(p.x[0], p.y[0])); int b = find(gid(p.x[p.n - 1], p.y[p.n - 1])); par[a] = b; }
  int r0 = -1;
  for (auto& nd : nodes.v) { int r = find(gid(nd.x, nd.y)); if (r0 < 0) r0 = r; else if (r != r0) return false; }
  return true;
}

// ───────────────────────────── round ─────────────────────────────
// Phase 1 (parallel, demand independent): for the first occurrence of every route net in the order, everything the Python
//   loop computes before the first cost evaluation: old wires/vias, used access points, nodes, base polylines + tree_connects,
//   FLUTE edges and the runs of both L shapes. Valid because a net's segments only change when that net itself is processed.
// Phase 2 (sequential, in order): apply(-1), layer DP of the candidates on the live demand, pick, route_net, commit_net, segs_of.
struct Edge2 { std::vector<Run> r[2]; int32_t sp[4]; };
struct Pre {
  int status = 0;                 // 0 skip (not_in_route), 1 ready, -1 empty access points, -2 bad layer
  OrdIv ow; std::vector<std::array<int64_t, 3>> ov;
  Nodes nodes;
  bool base_on = false; std::vector<std::vector<Run>> bruns; std::vector<std::array<int32_t, 4>> bsp;
  bool rsmt_on = false; std::vector<Edge2> edges;
};
struct In {
  const int32_t* dirs; const int64_t *net_i, *net_j, *pin_start, *pin_end, *ap_start, *ap_cnt; const int32_t* ap;
  const Seg* base; const int64_t* boff; const int64_t* new_ptr; const int64_t* new_cnt; const int64_t* unused; flute_fn flute;
};

static void prep_net(const In& in, int64_t n, const Seg* segs, long ns, Pre& pr) {
  int64_t i = in.net_i[n], j = in.net_j[n];
  if (i < 0 || j < 0 || in.pin_end[j] <= in.pin_start[j]) { pr.status = 0; return; }
  for (long t = 0; t < ns; t++) if (segs[t][2] < 0 || segs[t][2] > 63 || segs[t][5] < 0 || segs[t][5] > 63) { pr.status = -2; return; }
  const int32_t* ap = in.ap;
  to_wires_vias(segs, ns, in.dirs, pr.ow, pr.ov);
  // la_swap.used_access_points
  const long p0 = in.pin_start[j], p1 = in.pin_end[j];
  std::unordered_set<uint64_t> ap_gc;
  for (long p = p0; p < p1; p++) for (long q = in.ap_start[p]; q < in.ap_start[p] + in.ap_cnt[p]; q++) ap_gc.insert(pkey(ap[3 * q + 1], ap[3 * q + 2]));
  std::unordered_map<int64_t, std::vector<int64_t>> rows, cols;
  for (uint64_t kk : ap_gc) { int64_t x = kz(kk), y = kf(kk); rows[y].push_back(x); cols[x].push_back(y); }
  std::unordered_map<uint64_t, uint64_t> wire_at;
  std::unordered_map<uint64_t, std::vector<std::pair<int64_t, int64_t>>> via_at;
  for (long t = 0; t < ns; t++) {
    int64_t xl = segs[t][0], yl = segs[t][1], zl = segs[t][2], xh = segs[t][3], yh = segs[t][4], zh = segs[t][5];
    if (zl != zh) {
      if (xl == xh && yl == yh && ap_gc.count(pkey(xl, yl))) via_at[pkey(xl, yl)].push_back({zl, zh});
    } else if (yl == yh) {
      auto it = rows.find(yl);
      if (it != rows.end()) for (int64_t x : it->second) if (xl <= x && x <= xh) wire_at[pkey(x, yl)] |= (1ull << zl);
    } else {
      auto it = cols.find(xl);
      if (it != cols.end()) for (int64_t y : it->second) if (yl <= y && y <= yh) wire_at[pkey(xl, y)] |= (1ull << zl);
    }
  }
  // la_swap.nodes_from_aps([chosen or min(aps)])
  Nodes& nodes = pr.nodes; nodes.clear();
  for (long p = p0; p < p1; p++) {
    long q0 = in.ap_start[p], q1 = in.ap_start[p] + in.ap_cnt[p];
    long hit = -1;
    for (long q = q0; q < q1 && hit < 0; q++) {
      int64_t z = ap[3 * q], x = ap[3 * q + 1], y = ap[3 * q + 2];
      uint64_t gk = pkey(x, y);
      auto wi = wire_at.find(gk);
      if (wi != wire_at.end() && z >= 0 && z <= 63 && ((wi->second >> z) & 1ull)) { hit = q; break; }
      auto vi = via_at.find(gk);
      if (vi != via_at.end()) for (auto& ab : vi->second) if (ab.first <= z && z <= ab.second) { hit = q; break; }
    }
    if (hit < 0) {
      if (q1 <= q0) { pr.status = -1; return; }
      hit = q0;
      for (long q = q0 + 1; q < q1; q++)
        if (std::lexicographical_compare(ap + 3 * q, ap + 3 * q + 3, ap + 3 * hit, ap + 3 * hit + 3)) hit = q;
    }
    int l0 = ap[3 * hit]; int64_t x0 = ap[3 * hit + 1], y0 = ap[3 * hit + 2];
    auto it = nodes.idx.find(pkey(x0, y0));
    if (it == nodes.idx.end()) { nodes.idx.emplace(pkey(x0, y0), (int)nodes.v.size()); nodes.v.push_back({x0, y0, l0, l0}); }
    else { Node& nd = nodes.v[it->second]; nd.lo = std::min(nd.lo, l0); nd.hi = std::max(nd.hi, l0); }
  }
  auto spans_of = [&](int64_t x, int64_t y, int32_t* sp) { const Node* nd = nodes.get(x, y); if (nd) { sp[0] = nd->lo; sp[1] = nd->hi; } else { sp[0] = -1; sp[1] = -1; } };
  std::vector<Poly> bpolys; std::vector<Run> tr;
  segments_to_polys(segs, ns, nodes, bpolys);
  if (tree_connects(nodes, bpolys)) {
    pr.base_on = true;
    for (auto& p : bpolys) {
      to_runs(p, tr);
      if (tr.empty()) continue;
      std::array<int32_t, 4> sp; spans_of(p.x[0], p.y[0], sp.data()); spans_of(p.x[p.n - 1], p.y[p.n - 1], sp.data() + 2);
      pr.bruns.push_back(tr); pr.bsp.push_back(sp);
    }
  }
  pr.rsmt_on = in.flute && nodes.v.size() > 1;
  pr.status = 1;
}


// FLUTE results are a pure function of the point sequence (LUT read-only, no other state): memoised across rounds and calls.
struct VecHash { size_t operator()(const std::vector<int32_t>& v) const { uint64_t h = 1469598103934665603ull; for (int32_t x : v) { h ^= (uint32_t)x; h *= 1099511628211ull; } return (size_t)h; } };
struct FluteOut { std::vector<int32_t> ox, oy, on; };
static std::unordered_map<std::vector<int32_t>, FluteOut, VecHash> FCACHE;
static flute_fn FCACHE_FN = nullptr; static long FCACHE_INTS = 0, F_HITS = 0, F_MISS = 0;

// rsmt_route.Flute.edges(list(nodes.keys())) + the two L shapes per edge (runs and spans)
static void prep_rsmt(const In& in, Pre& pr) {
  const Nodes& nodes = pr.nodes;
  auto spans_of = [&](int64_t x, int64_t y, int32_t* sp) { const Node* nd = nodes.get(x, y); if (nd) { sp[0] = nd->lo; sp[1] = nd->hi; } else { sp[0] = -1; sp[1] = -1; } };
  std::vector<std::array<int64_t, 4>> edges;
  const int dd = (int)nodes.v.size();
  if (dd == 2) edges.push_back({nodes.v[0].x, nodes.v[0].y, nodes.v[1].x, nodes.v[1].y});
  else {
    std::vector<int32_t> key(2 * dd);
    for (int t = 0; t < dd; t++) { key[2 * t] = (int32_t)nodes.v[t].x; key[2 * t + 1] = (int32_t)nodes.v[t].y; }
    if (FCACHE_FN != in.flute) { FCACHE.clear(); FCACHE_INTS = 0; FCACHE_FN = in.flute; }
    auto it = FCACHE.find(key);
    if (it == FCACHE.end()) {
      F_MISS++;
      std::vector<int> fx(dd), fy(dd), ox(2 * dd + 2), oy(2 * dd + 2), on(2 * dd + 2);
      for (int t = 0; t < dd; t++) { fx[t] = key[2 * t]; fy[t] = key[2 * t + 1]; }
      int m = in.flute(dd, fx.data(), fy.data(), ox.data(), oy.data(), on.data());
      FluteOut fo; fo.ox.assign(ox.begin(), ox.begin() + m); fo.oy.assign(oy.begin(), oy.begin() + m); fo.on.assign(on.begin(), on.begin() + m);
      fo.ox.resize(2 * dd + 2, 0); fo.oy.resize(2 * dd + 2, 0); fo.on.resize(m);       // branch targets index < 2d-2
      if (FCACHE_INTS > 50000000) { FCACHE.clear(); FCACHE_INTS = 0; }
      FCACHE_INTS += 2 * dd + 3 * m;
      it = FCACHE.emplace(std::move(key), std::move(fo)).first;
    } else F_HITS++;
    const FluteOut& fo = it->second;
    const int m = (int)fo.on.size();
    std::set<std::array<int64_t, 4>> seen;
    for (int t = 0; t < m; t++) {
      int jj = fo.on[t];
      int64_t ax = fo.ox[t], ay = fo.oy[t], bx = fo.ox[jj], by = fo.oy[jj];
      if (ax == bx && ay == by) continue;
      std::array<int64_t, 4> k4 = (std::make_pair(ax, ay) < std::make_pair(bx, by)) ? std::array<int64_t, 4>{ax, ay, bx, by} : std::array<int64_t, 4>{bx, by, ax, ay};
      if (!seen.insert(k4).second) continue;
      edges.push_back({ax, ay, bx, by});
    }
  }
  pr.edges.resize(edges.size());
  for (size_t e = 0; e < edges.size(); e++) {
    int64_t px = edges[e][0], py = edges[e][1], cx = edges[e][2], cy = edges[e][3];
    Edge2& E = pr.edges[e];
    spans_of(px, py, E.sp); spans_of(cx, cy, E.sp + 2);
    for (int jv = 0; jv < 2; jv++) {
      Poly p; p.n = 3; p.x[0] = px; p.y[0] = py; p.x[2] = cx; p.y[2] = cy;
      if (jv == 0) { p.x[1] = cx; p.y[1] = py; } else { p.x[1] = px; p.y[1] = cy; }
      to_runs(p, E.r[jv]);
    }
  }
}

static std::vector<int64_t> O_changed, O_off;
static std::vector<int64_t> O_segs;
static int64_t O_stats[8];      // count, first position for not_in_route, kept, topo_base, topo_rsmt
static int64_t O_last = -1;
static double O_t[8];           // phase 1, phase 2 DP, phase 2 rest, threads, phase 1a (parallel), flute hits, flute misses, cache ints

extern "C" {
// P (pointers): 0 d, 1 c, 2 u, 3 R, 4 C, 5 Rv, 6 cst, 7 dirs(i32), 8 sd(i64), 9 sc(i64),
//   10 order(i64), 11 w(f64), 12 cl(f64), 13 rd(f64), 14 pwn(f64 or 0), 15 net_i(i64), 16 net_j(i64),
//   17 pin_start(i64), 18 pin_end(i64), 19 ap_start(i64), 20 ap_cnt(i64), 21 ap(i32 [A,3]),
//   22 base_arr(i64 [S,6]), 23 base_off(i64), 24 new_ptr(i64 address of an i64 [k,6] block per route net), 25 new_cnt(i64 per route net, -1 = base segments), 26 unused, 27 flute fn
// I: 0 L, 1 X, 2 Y, 3 n_order, 4 pw_mode, 5 n_route, 6 threads
// F: 0 w4, 1 um, 2 tw_scale
// returns 0 ok, -1 empty access-point list (python min([]) ValueError; outputs so far are valid), -2 layer index > 63
long pf_round(void** P, const int64_t* I, const double* F) {
  using clk = std::chrono::steady_clock;
  auto T0 = clk::now();
  const double* cst = (const double*)P[6];
  Ctx k{(const double*)P[0], (const double*)P[1], (const double*)P[2], (const double*)P[3], (const double*)P[4], (const double*)P[5],
        (const int32_t*)P[7], (const int64_t*)P[8], (const int64_t*)P[9], (int)I[0], (int)I[1], (int)I[2], F[0], F[1], 0.0, 0.0, 0.0,
        cst[0], cst[1], cst[2], 0.0};
  Grid g{(double*)P[0], (const double*)P[1], (const int32_t*)P[7], (const int64_t*)P[8], (int)I[0], (int)I[1], (int)I[2]};
  In in{(const int32_t*)P[7], (const int64_t*)P[15], (const int64_t*)P[16], (const int64_t*)P[17], (const int64_t*)P[18],
        (const int64_t*)P[19], (const int64_t*)P[20], (const int32_t*)P[21], (const Seg*)P[22], (const int64_t*)P[23],
        (const int64_t*)P[24], (const int64_t*)P[25], (const int64_t*)P[26], (flute_fn)P[27]};
  const int32_t* dirs = in.dirs;
  const int64_t* order = (const int64_t*)P[10]; const double* w = (const double*)P[11]; const double* cl = (const double*)P[12];
  const double* rd = (const double*)P[13]; const double* pwn = (const double*)P[14];
  const long NO = (long)I[3]; const bool PWM = I[4] != 0; const int NT = std::max<int>(1, (int)I[6]);
  const double tw_scale = F[2];
  std::vector<int> H, V;
  for (int z = 1; z < k.L; z++) (dirs[z] == 0 ? H : V).push_back(z);
  O_changed.clear(); O_off.clear(); O_segs.clear(); O_off.push_back(0);
  for (int t = 0; t < 8; t++) O_stats[t] = t % 2 ? -1 : 0;
  O_last = -1;
  auto start_segs = [&](int64_t i, const Seg*& s, long& ns) {
    if (in.new_cnt[i] >= 0) { s = (const Seg*)(uintptr_t)in.new_ptr[i]; ns = (long)in.new_cnt[i]; }
    else { s = in.base + in.boff[i]; ns = (long)(in.boff[i + 1] - in.boff[i]); }
  };
  // ---- phase 1 ----
  std::vector<Pre> pre(NO);
  std::vector<char> first(NO, 0);
  {
    std::unordered_set<int64_t> seen_i;
    for (long oi = 0; oi < NO; oi++) {
      int64_t i = in.net_i[order[oi]];
      if (i < 0 || seen_i.insert(i).second) first[oi] = 1;       // nets not in the route are cheap skips, mark them too
    }
  }
  std::atomic<long> nxt(0);
  auto worker = [&]() {
    for (;;) {
      long oi = nxt.fetch_add(1);
      if (oi >= NO) break;
      if (!first[oi]) continue;
      int64_t n = order[oi], i = in.net_i[n];
      const Seg* s = nullptr; long ns = 0;
      if (i >= 0) start_segs(i, s, ns);
      prep_net(in, n, s, ns, pre[oi]);
    }
  };
  {
    std::vector<std::thread> th;
    for (int t = 1; t < NT; t++) th.emplace_back(worker);
    worker();
    for (auto& t : th) t.join();
  }
  auto T1a = clk::now();
  for (long oi = 0; oi < NO; oi++) if (first[oi] && pre[oi].status == 1 && pre[oi].rsmt_on) prep_rsmt(in, pre[oi]);
  auto T1 = clk::now();
  double t_dp = 0;
  // ---- phase 2 ----
  std::vector<double> buf(std::max(k.X, k.Y) + 8);
  std::unordered_map<int64_t, std::vector<Seg>> cur;          // nets re-routed earlier in this round
  Pre tmp;
  auto stat = [&](int c, long pos) { if (O_stats[2 * c] == 0) O_stats[2 * c + 1] = pos; O_stats[2 * c]++; };
  struct Cand { bool on = false; std::vector<const std::vector<Run>*> runs; std::vector<std::vector<int32_t>> lay; std::vector<double> cost; };
  for (long oi = 0; oi < NO; oi++) {
    int64_t n = order[oi];
    Pre* pr = &pre[oi];
    if (!first[oi]) {                                           // repeated route net: prepare on its current segments
      int64_t i = in.net_i[n];
      tmp = Pre();
      auto cit = cur.find(i);
      if (cit != cur.end()) prep_net(in, n, cit->second.data(), (long)cit->second.size(), tmp);
      else { const Seg* s; long ns; start_segs(i, s, ns); prep_net(in, n, s, ns, tmp); }
      if (tmp.status == 1 && tmp.rsmt_on) prep_rsmt(in, tmp);
      pr = &tmp;
    }
    if (pr->status == 0) { stat(0, oi); continue; }
    if (pr->status == -2) return -2;
    int64_t i = in.net_i[n];
    apply_net(g, pr->ow, pr->ov, -1);
    if (pr->status == -1) return -1;
    k.tw = w[n] * tw_scale;
    k.pn = (PWM && pwn) ? pwn[n] : 0.0;
    k.cl = cl[n]; k.rd = rd[n];
    O_last = n;
    auto td = clk::now();
    Cand cb, cr;
    if (pr->base_on) {
      cb.on = true;
      for (size_t q = 0; q < pr->bruns.size(); q++) {
        const auto& tr = pr->bruns[q];
        std::vector<int32_t> ly(tr.size());
        double c = PWM ? dp_poly<true>(k, H, V, buf, tr.data(), (int)tr.size(), pr->bsp[q].data(), ly.data())
                       : dp_poly<false>(k, H, V, buf, tr.data(), (int)tr.size(), pr->bsp[q].data(), ly.data());
        cb.runs.push_back(&tr); cb.lay.push_back(std::move(ly)); cb.cost.push_back(c);
      }
    }
    if (pr->rsmt_on) {
      cr.on = true;
      for (auto& E : pr->edges) {
        double bc = 0; int bj = -1; std::vector<int32_t> bly;
        for (int jv = 0; jv < 2; jv++) {
          const auto& tr = E.r[jv];
          std::vector<int32_t> ly(tr.size());
          double c = tr.empty() ? 0.0 : (PWM ? dp_poly<true>(k, H, V, buf, tr.data(), (int)tr.size(), E.sp, ly.data())
                                             : dp_poly<false>(k, H, V, buf, tr.data(), (int)tr.size(), E.sp, ly.data()));
          if (bj < 0 || c < bc) { bc = c; bj = jv; bly = std::move(ly); }   // strict <, first wins
        }
        if (E.r[bj].empty()) continue;                           // the evaluation skips polylines without runs
        cr.runs.push_back(&E.r[bj]); cr.lay.push_back(std::move(bly)); cr.cost.push_back(bc);
      }
    }
    t_dp += std::chrono::duration<double>(clk::now() - td).count();
    if (!cb.on && !cr.on) { apply_net(g, pr->ow, pr->ov, +1); stat(1, oi); continue; }
    const Cand* best = nullptr; double bestc = 0; int btag = 0;
    if (cb.on) { double s = 0.0; for (double c : cb.cost) s += c; best = &cb; bestc = s; btag = 2; }
    if (cr.on) { double s = 0.0; for (double c : cr.cost) s += c; if (best == nullptr || s < bestc) { best = &cr; bestc = s; btag = 3; } }
    // route_net with the computed layers
    const Nodes& nodes = pr->nodes;
    OrdIv wr;
    std::unordered_map<uint64_t, std::pair<int, int>> nl;   // node_layers: only min / max matter
    std::vector<uint64_t> nl_keys;
    auto nl_add = [&](int64_t x, int64_t y, int z) {
      auto kk = pkey(x, y); auto it = nl.find(kk);
      if (it == nl.end()) { nl.emplace(kk, std::make_pair(z, z)); nl_keys.push_back(kk); }
      else { it->second.first = std::min(it->second.first, z); it->second.second = std::max(it->second.second, z); }
    };
    for (size_t pi = 0; pi < best->runs.size(); pi++) {
      const auto& rr = *best->runs[pi]; const auto& ly = best->lay[pi];
      for (size_t q = 0; q < rr.size(); q++) {
        int z = ly[q];
        wr.at(zkey(z, rr[q][1])).push_back({rr[q][2], rr[q][3]});
        nl_add(rr[q][4], rr[q][5], z); nl_add(rr[q][6], rr[q][7], z);
      }
    }
    for (auto& nd : nodes.v) { nl_add(nd.x, nd.y, nd.lo); nl_add(nd.x, nd.y, nd.hi); }
    std::vector<std::array<int64_t, 3>> vias;
    for (uint64_t kk : nl_keys) { auto& mm = nl[kk]; for (int z = mm.first; z < mm.second; z++) vias.push_back({kz(kk), kf(kk), z}); }
    if (wr.keys.empty() && vias.empty() && !nodes.v.empty()) {
      const Node& nd = nodes.v[0];
      vias.push_back({nd.x, nd.y, nd.lo < k.L - 1 ? nd.lo : nd.lo - 1});
    }
    std::sort(vias.begin(), vias.end()); vias.erase(std::unique(vias.begin(), vias.end()), vias.end());
    wr.merge_all();
    apply_net(g, wr, vias, +1);
    // fastmode.segs_of
    std::vector<Seg> out;
    for (size_t t = 0; t < wr.keys.size(); t++) {
      int z = kz(wr.keys[t]); int64_t f = kf(wr.keys[t]);
      for (auto& ab : wr.ivs[t]) out.push_back(dirs[z] == 0 ? Seg{ab.first, f, z, ab.second, f, z} : Seg{f, ab.first, z, f, ab.second, z});
    }
    for (auto& v : vias) out.push_back({v[0], v[1], v[2], v[0], v[1], v[2] + 1});
    O_changed.push_back(i);
    for (auto& s_ : out) for (int q = 0; q < 6; q++) O_segs.push_back(s_[q]);
    O_off.push_back((int64_t)(O_segs.size() / 6));
    cur[i] = std::move(out);
    stat(btag, oi);
  }
  auto T2 = clk::now();
  O_t[0] = std::chrono::duration<double>(T1 - T0).count(); O_t[1] = t_dp; O_t[2] = std::chrono::duration<double>(T2 - T1).count() - t_dp; O_t[3] = NT;
  O_t[4] = std::chrono::duration<double>(T1a - T0).count(); O_t[5] = (double)F_HITS; O_t[6] = (double)F_MISS; O_t[7] = (double)FCACHE_INTS;
  return 0;
}
long pf_nchanged() { return (long)O_changed.size(); }
long pf_nsegs() { return (long)(O_segs.size() / 6); }
void pf_fetch(int64_t* changed, int64_t* off, int64_t* segs, int64_t* stats, int64_t* last) {
  if (!O_changed.empty()) std::memcpy(changed, O_changed.data(), O_changed.size() * 8);
  std::memcpy(off, O_off.data(), O_off.size() * 8);
  if (!O_segs.empty()) std::memcpy(segs, O_segs.data(), O_segs.size() * 8);
  std::memcpy(stats, O_stats, 8 * 8);
  *last = O_last;
}
void pf_times(double* t) { std::memcpy(t, O_t, 8 * 8); }
}
