// mid5_kern.cc: C++ drivers of the mid5 region of tools/mid5/chain_mid.py (--mid5 1).
// Every entry point is extern "C" on plain arrays (no Python object), so a later C++ main can call them without an interpreter.
//
//  mid5_escape_pass(...)   ONE pass of escape3d.escape as tools/ovf3/ovf_lib.escape_pass calls it in the overflow stage
//                          (pin escape out of the hottest blocked cells), the whole candidate loop in C++: to_wires_vias,
//                          la_swap.used_access_points, the tail rule, apply / marginal (the exact C++ twins of
//                          tools/speed4/s4_esc.cc, included), window A* (tools/ovf3/astar3d_win.cpp, included), merge_wires,
//                          emit_tuples. Python's semantics are kept bit for bit, including the iteration order of the Python
//                          sets that decides the order of the floating-point sums of a net's marginal cost: the vias sets are
//                          an EMULATION of CPython 3.12's set table (tuple hash, linear probes, perturbation, resize, merge,
//                          difference), checked against the interpreter by tools/mid5/test_pyset.py. A set operation the
//                          emulation does not cover (an insert into a table with dummies) sets an error flag and the pass is
//                          redone by the original Python code (never seen in the tests).
//  mid5_blocked_share(...) the escape test's blocked share of the overflow cost (chain_j escape_inmem), on T threads.
//                          libm exp and a different summation order than numpy: the caller falls back to the numpy test
//                          when the share is within 1e-6 of the threshold, so the on/off decision is the same.
// Build: tools/mid5/build_mid5.sh  (g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -shared -pthread).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <limits>
#include <utility>
#include <thread>
#include <array>

namespace s4n {
#include "../speed4/s4_esc.cc"
}
namespace awn {
#include "../ovf3/astar3d_win.cpp"
}

typedef int64_t i64;

// ===================================================================== CPython 3.12 set emulation (tuples of ints)
namespace pyset {
static const uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P5 = 2870177450012600261ULL;
static const uint64_t MODP = (1ULL << 61) - 1;
static inline int64_t int_hash(i64 v) {
  if (v >= 0) { int64_t h = (int64_t)((uint64_t)v % MODP); return h; }
  int64_t h = -(int64_t)((uint64_t)(-v) % MODP);
  return h == -1 ? -2 : h;
}
static inline int64_t tuple_hash(const int32_t* v, int n) {
  uint64_t acc = P5;
  for (int i = 0; i < n; i++) {
    uint64_t lane = (uint64_t)int_hash(v[i]);
    acc += lane * P2; acc = (acc << 31) | (acc >> 33); acc *= P1;
  }
  acc += (uint64_t)n ^ (P5 ^ 3527539ULL);
  if (acc == (uint64_t)-1) return 1546275796;
  return (int64_t)acc;
}
struct Ent { int64_t h; int32_t k[3]; uint8_t s; };            // s: 0 unused, 1 active, 2 dummy
struct Set {
  std::vector<Ent> t; size_t mask; i64 fill, used; int n; bool bad;
  explicit Set(int n_ = 3) : t(8), mask(7), fill(0), used(0), n(n_), bad(false) { for (auto& e : t) e.s = 0; }
  inline bool eq(const Ent& e, const int32_t* k) const { for (int i = 0; i < n; i++) if (e.k[i] != k[i]) return false; return true; }
  i64 lookup(const int32_t* k, int64_t h) const {
    size_t perturb = (size_t)h, i = (size_t)h & mask;
    while (true) {
      size_t probes = (i + 9 <= mask) ? 9 : 0, j = i;
      while (true) {
        const Ent& e = t[j];
        if (e.s == 0) return -1;
        if (e.s == 1 && e.h == h && eq(e, k)) return (i64)j;
        j++;
        if (probes-- == 0) break;
      }
      perturb >>= 5; i = (i * 5 + 1 + perturb) & mask;
    }
  }
  bool contains(const int32_t* k) const { return lookup(k, tuple_hash(k, n)) >= 0; }
  static void insert_clean(std::vector<Ent>& tb, size_t m, const Ent& src) {
    size_t perturb = (size_t)src.h, i = (size_t)src.h & m;
    while (true) {
      if (tb[i].s == 0) { tb[i] = src; tb[i].s = 1; return; }
      if (i + 9 <= m) for (size_t j = 1; j <= 9; j++) if (tb[i + j].s == 0) { tb[i + j] = src; tb[i + j].s = 1; return; }
      perturb >>= 5; i = (i * 5 + 1 + perturb) & m;
    }
  }
  void resize(i64 minused) {
    size_t ns = 8; while (ns <= (size_t)minused) ns <<= 1;
    std::vector<Ent> nt(ns); for (auto& e : nt) e.s = 0;
    for (auto& e : t) if (e.s == 1) insert_clean(nt, ns - 1, e);
    t.swap(nt); mask = ns - 1; fill = used;
  }
  void add_h(const int32_t* k, int64_t h) {
    size_t perturb = (size_t)h, i = (size_t)h & mask;
    while (true) {
      size_t probes = (i + 9 <= mask) ? 9 : 0, j = i;
      while (true) {
        Ent& e = t[j];
        if (e.s == 0) {
          e.s = 1; e.h = h; for (int q = 0; q < 3; q++) e.k[q] = q < n ? k[q] : 0;
          fill++; used++;
          if ((size_t)fill * 5 < mask * 3) return;
          resize(used > 50000 ? used * 2 : used * 4);
          return;
        }
        if (e.s == 1 && e.h == h && eq(e, k)) return;
        if (e.s == 2) bad = true;                            // freeslot semantics not emulated (never used by escape)
        j++;
        if (probes-- == 0) break;
      }
      perturb >>= 5; i = (i * 5 + 1 + perturb) & mask;
    }
  }
  void add(const int32_t* k) { add_h(k, tuple_hash(k, n)); }
  void add3(int a, int b, int c) { int32_t k[3] = {a, b, c}; add(k); }
  void discard_h(const int32_t* k, int64_t h) {
    i64 j = lookup(k, h);
    if (j >= 0) { t[j].s = 2; t[j].h = -1; used--; }
  }
  void merge(const Set& o) {                                 // set_merge(this, o)
    if (&o == this || o.used == 0) return;
    if ((size_t)(fill + o.used) * 5 >= mask * 3) resize((used + o.used) * 2);
    if (fill == 0 && mask == o.mask && o.fill == o.used) {
      for (size_t i = 0; i <= o.mask; i++) if (o.t[i].s == 1) t[i] = o.t[i];
      fill = o.fill; used = o.used; return;
    }
    if (fill == 0) {
      fill = o.used; used = o.used;
      for (size_t i = 0; i <= o.mask; i++) if (o.t[i].s == 1) insert_clean(t, mask, o.t[i]);
      return;
    }
    for (size_t i = 0; i <= o.mask; i++) if (o.t[i].s == 1) add_h(o.t[i].k, o.t[i].h);
    if (o.bad) bad = true;
  }
  Set copy() const { Set r(n); r.merge(*this); r.bad = r.bad || bad; return r; }
  Set difference(const Set& o) const {                       // set_difference(this, o), o a set
    if ((used >> 2) > o.used) {
      Set r = copy();
      for (size_t i = 0; i <= o.mask; i++) if (o.t[i].s == 1) r.discard_h(o.t[i].k, o.t[i].h);
      return r;
    }
    Set r(n);
    for (size_t i = 0; i <= mask; i++) if (t[i].s == 1 && o.lookup(t[i].k, t[i].h) < 0) r.add_h(t[i].k, t[i].h);
    r.bad = r.bad || bad;
    return r;
  }
  Set uni(const Set& o) const { Set r = copy(); if (&o != this) r.merge(o); return r; }   // set_or
  template <class F> void each(F f) const { for (size_t i = 0; i <= mask; i++) if (t[i].s == 1) f(t[i].k); }
};
}  // namespace pyset

// ===================================================================== escape pass
namespace esc {
using pyset::Set;
typedef std::pair<i64, i64> IV;
struct WV {                                                   // ({(z, f): [(a, b)]} in dict order, vias set)
  std::vector<std::pair<i64, i64>> keys;
  std::vector<std::vector<IV>> ivs;
  Set vias{3};
};
struct Grid {
  double* d; const double* c; const double* u; i64 sz, sx, sy; i64 cz, cx, cy; i64 X, Y, L;
  const int8_t* dirs8; const int32_t* dirs32;
};
static inline i64 kz(i64 z, i64 f) { return z * (1LL << 32) + f; }

static void merge_list(std::vector<IV>& v, std::vector<IV>& out) {          // ivs.sort(); merge touching / overlapping
  std::sort(v.begin(), v.end());
  out.clear();
  out.push_back(v[0]);
  for (size_t q = 1; q < v.size(); q++) {
    if (v[q].first <= out.back().second) out.back().second = std::max(out.back().second, v[q].second);
    else out.push_back(v[q]);
  }
}

static void to_wires_vias(const i64* s, i64 nr, const Grid& g, WV& o) {
  o.keys.clear(); o.ivs.clear(); o.vias = Set(3);
  std::vector<std::vector<IV>> raw; std::unordered_map<i64, size_t> pos;
  for (i64 r = 0; r < nr; r++) {
    const i64* q = s + 6 * r;
    i64 x1 = q[0], y1 = q[1], z1 = q[2], x2 = q[3], y2 = q[4], z2 = q[5];
    if (z1 != z2) {
      for (i64 z = std::min(z1, z2); z < std::max(z1, z2); z++) o.vias.add3((int)x1, (int)y1, (int)z);
    } else if (g.dirs8[z1] == 0 && y1 == y2 && x1 != x2) {
      i64 k = kz(z1, y1); auto it = pos.find(k);
      if (it == pos.end()) { pos[k] = raw.size(); o.keys.push_back({z1, y1}); raw.emplace_back(); it = pos.find(k); }
      raw[it->second].push_back({std::min(x1, x2), std::max(x1, x2)});
    } else if (g.dirs8[z1] == 1 && x1 == x2 && y1 != y2) {
      i64 k = kz(z1, x1); auto it = pos.find(k);
      if (it == pos.end()) { pos[k] = raw.size(); o.keys.push_back({z1, x1}); raw.emplace_back(); it = pos.find(k); }
      raw[it->second].push_back({std::min(y1, y2), std::max(y1, y2)});
    }
  }
  o.ivs.resize(raw.size());
  for (size_t q = 0; q < raw.size(); q++) merge_list(raw[q], o.ivs[q]);
}

struct Flat { std::vector<i64> iz, ifx, ia, ib, vx, vy, vz; i64 nkeys; };
static void flatten(const WV& w, Flat& f) {
  f.iz.clear(); f.ifx.clear(); f.ia.clear(); f.ib.clear(); f.vx.clear(); f.vy.clear(); f.vz.clear();
  f.nkeys = (i64)w.keys.size();
  for (size_t q = 0; q < w.keys.size(); q++)
    for (auto& p : w.ivs[q]) { f.iz.push_back(w.keys[q].first); f.ifx.push_back(w.keys[q].second); f.ia.push_back(p.first); f.ib.push_back(p.second); }
  w.vias.each([&](const int32_t* k) { f.vx.push_back(k[0]); f.vy.push_back(k[1]); f.vz.push_back(k[2]); });
}
static void apply(const Grid& g, const WV& w, i64 sign, Flat& f) {
  flatten(w, f);
  s4n::s4_apply(g.d, g.sz, g.sx, g.sy, g.X, g.Y, g.L, g.dirs8, (i64)f.iz.size(), f.iz.data(), f.ifx.data(), f.ia.data(), f.ib.data(),
                (i64)f.vx.size(), f.vx.data(), f.vy.data(), f.vz.data(), sign);
}
static double marginal(const Grid& g, const WV& w, Flat& f) {
  flatten(w, f);
  return s4n::s4_marginal(g.d, g.c, g.u, g.sz, g.sx, g.sy, g.X, g.Y, g.L, g.dirs8, f.nkeys, (i64)f.iz.size(), f.iz.data(), f.ifx.data(),
                          f.ia.data(), f.ib.data(), (i64)f.vx.size(), f.vx.data(), f.vy.data(), f.vz.data());
}

// la_swap.used_access_points(segs, pins): per pin the first access point the route touches, or none (-1)
static void used_aps(const i64* s, i64 nr, i64 p0, i64 np_, const i64* ap_off, const i64* aps, std::vector<i64>& used) {
  std::unordered_set<i64> ap_gc; std::unordered_map<i64, std::vector<i64>> rows, cols;
  auto xy = [](i64 x, i64 y) { return x * (1LL << 32) + y; };
  for (i64 p = 0; p < np_; p++)
    for (i64 a = ap_off[p0 + p]; a < ap_off[p0 + p + 1]; a++) {
      i64 x = aps[3 * a + 1], y = aps[3 * a + 2];
      if (ap_gc.insert(xy(x, y)).second) { rows[y].push_back(x); cols[x].push_back(y); }
    }
  std::unordered_map<i64, uint64_t> wire_at; std::unordered_map<i64, std::vector<IV>> via_at;
  for (i64 r = 0; r < nr; r++) {
    const i64* q = s + 6 * r;
    i64 xl = q[0], yl = q[1], zl = q[2], xh = q[3], yh = q[4], zh = q[5];
    if (zl != zh) {
      if (xl == xh && yl == yh && ap_gc.count(xy(xl, yl))) via_at[xy(xl, yl)].push_back({zl, zh});
    } else if (yl == yh) {
      auto it = rows.find(yl);
      if (it != rows.end()) for (i64 x : it->second) if (xl <= x && x <= xh) wire_at[xy(x, yl)] |= (1ULL << zl);
    } else {
      auto it = cols.find(xl);
      if (it != cols.end()) for (i64 y : it->second) if (yl <= y && y <= yh) wire_at[xy(xl, y)] |= (1ULL << zl);
    }
  }
  used.assign(np_, -1);
  for (i64 p = 0; p < np_; p++)
    for (i64 a = ap_off[p0 + p]; a < ap_off[p0 + p + 1]; a++) {
      i64 z = aps[3 * a], x = aps[3 * a + 1], y = aps[3 * a + 2], g = xy(x, y);
      bool hit = false;
      auto wi = wire_at.find(g);
      if (wi != wire_at.end() && z >= 0 && z < 64 && ((wi->second >> z) & 1ULL)) hit = true;
      if (!hit) { auto vi = via_at.find(g); if (vi != via_at.end()) for (auto& ab : vi->second) if (ab.first <= z && z <= ab.second) { hit = true; break; } }
      if (hit) { used[p] = a; break; }
    }
}

struct Out { std::vector<i64> order; std::vector<std::vector<i64>> rows; std::unordered_map<i64, size_t> pos; i64 err = 0; };

static void emit(const Grid& g, const WV& w, std::vector<i64>& rows) {
  rows.clear();
  for (size_t q = 0; q < w.keys.size(); q++) {
    i64 z = w.keys[q].first, f = w.keys[q].second;
    for (auto& p : w.ivs[q]) {
      if (g.dirs8[z] == 0) { i64 r[6] = {p.first, f, z, p.second, f, z}; rows.insert(rows.end(), r, r + 6); }
      else { i64 r[6] = {f, p.first, z, f, p.second, z}; rows.insert(rows.end(), r, r + 6); }
    }
  }
  std::vector<std::array<i64, 3>> vs;
  w.vias.each([&](const int32_t* k) { vs.push_back({k[0], k[1], k[2]}); });
  std::sort(vs.begin(), vs.end());
  for (auto& v : vs) { i64 r[6] = {v[0], v[1], v[2], v[0], v[1], v[2] + 1}; rows.insert(rows.end(), r, r + 6); }
}
}  // namespace esc

extern "C" {
// One escape3d.escape pass (passes=1, la given, guarded given) on the local nets k = 0..n-1 of ovf_lib.escape_pass.
//   d (written), c, u: the demand model; strides in elements of d and c; X, Y, L; dirs (int8 and int32 copies)
//   soff[n+1], segs[*,6]: the current GCell segments of each local net (S.new or the base rows), in order
//   pidx[n]: -1 the net's key has no pins entry; else the global index of the net's first pin; gflag[n]: key is guarded
//   ap_off[*], aps[*,3]: pins CSR (pin j of net k = pidx[k] + j; its access points (l, x, y) = aps[ap_off[p]..ap_off[p+1]])
//   npin[n]: number of pins of the net; hot[(X+1)*(Y+1)]: escape3d's hot_xy set
//   W, step_pen, via_pen: search window and penalties; stats[8] (+=): guarded, skip_pins_at_gcell_0, _many,
//   skip_tail_is_net, no_path, moved, kept, astar_calls
// returns a handle (mid5_esc_n / mid5_esc_get / mid5_esc_free): changed nets in first-move order with their emitted rows.
void* mid5_escape_pass(double* d, const double* c, const double* u, i64 sz, i64 sx, i64 sy, i64 cz, i64 cx, i64 cy,
                       i64 X, i64 Y, i64 L, const int8_t* dirs8, const int32_t* dirs32,
                       i64 n, const i64* soff, const i64* segs, const i64* pidx, const uint8_t* gflag, const i64* npin,
                       const i64* ap_off, const i64* aps, const uint8_t* hot,
                       i64 W, double step_pen, double via_pen, i64* stats) {
  using namespace esc;
  Grid g{d, c, u, sz, sx, sy, cz, cx, cy, X, Y, L, dirs8, dirs32};
  Out* out = new Out();
  const i64 HY = Y + 1;
  auto ishot = [&](i64 x, i64 y) { return x >= 0 && y >= 0 && x <= X && y <= Y && hot[x * HY + y]; };
  // candidate (net, GCell) pairs in escape3d's order: nets in order, per net the set {(x1, y1)} of hot via stacks
  std::vector<std::pair<i64, std::array<int32_t, 2>>> cand;
  for (i64 k = 0; k < n; k++) {
    pyset::Set gs(2);
    for (i64 r = soff[k]; r < soff[k + 1]; r++) {
      const i64* q = segs + 6 * r;
      if (q[2] != q[5] && ishot(q[0], q[1])) { int32_t kk[2] = {(int32_t)q[0], (int32_t)q[1]}; gs.add(kk); }
    }
    if (gs.bad) out->err = 1;
    gs.each([&](const int32_t* kk) { cand.push_back({k, {kk[0], kk[1]}}); });
  }
  std::unordered_map<i64, WV> cur;
  std::vector<i64> used; Flat fl;
  std::vector<int32_t> pathbuf(3 * (1 << 16)), src, tgt;
  i64 strides6[6] = {sz, sx, sy, cz, cx, cy};
  for (auto& cd : cand) {
    i64 k = cd.first; i64 gx = cd.second[0], gy = cd.second[1];
    if (pidx[k] == -1) continue;
    if (gflag[k]) { stats[0]++; continue; }
    WV wtmp; const WV* wp;
    auto it = cur.find(k);
    if (it != cur.end()) wp = &it->second;
    else { to_wires_vias(segs + 6 * soff[k], soff[k + 1] - soff[k], g, wtmp); wp = &wtmp; }
    const WV& w = *wp;
    i64 zmax = -1; bool any = false;
    w.vias.each([&](const int32_t* q) { if (q[0] == gx && q[1] == gy) { any = true; zmax = std::max<i64>(zmax, q[2]); } });
    if (!any) continue;
    const i64 p0 = pidx[k], np_ = npin[k];
    used_aps(segs + 6 * soff[k], soff[k + 1] - soff[k], p0, np_, ap_off, aps, used);
    i64 nhere = 0, hp = -1, ha = -1;
    for (i64 p = 0; p < np_; p++) {
      i64 a = used[p];
      if (a >= 0 && aps[3 * a + 1] == gx && aps[3 * a + 2] == gy) { nhere++; if (nhere == 1) { hp = p; ha = a; } }
    }
    if (nhere != 1) { stats[nhere == 0 ? 1 : 2]++; continue; }
    i64 lp = aps[3 * ha];
    i64 zw = std::numeric_limits<i64>::max(); bool wl = false;
    for (size_t q = 0; q < w.keys.size(); q++) {
      i64 z = w.keys[q].first, f = w.keys[q].second;
      for (auto& ab : w.ivs[q])
        if ((dirs8[z] == 0 && f == gy && ab.first <= gx && gx <= ab.second) || (dirs8[z] == 1 && f == gx && ab.first <= gy && gy <= ab.second)) {
          wl = true; zw = std::min(zw, z);
        }
    }
    if (!wl) zw = zmax + 1;
    Set tail(3);
    for (i64 z = lp; z < zw; z++) { int32_t kk[3] = {(int32_t)gx, (int32_t)gy, (int32_t)z}; if (w.vias.contains(kk)) tail.add(kk); }
    if (tail.used == 0) continue;
    apply(g, w, -1, fl);
    double old = marginal(g, w, fl);
    Set v_rest = w.vias.difference(tail);
    tgt.clear();
    for (size_t q = 0; q < w.keys.size(); q++) {
      i64 z = w.keys[q].first, f = w.keys[q].second;
      for (auto& ab : w.ivs[q]) for (i64 t = ab.first; t <= ab.second; t++) {
        if (dirs8[z] == 0) { tgt.push_back((int32_t)z); tgt.push_back((int32_t)t); tgt.push_back((int32_t)f); }
        else { tgt.push_back((int32_t)z); tgt.push_back((int32_t)f); tgt.push_back((int32_t)t); }
      }
    }
    v_rest.each([&](const int32_t* q) {
      tgt.push_back(q[2]); tgt.push_back(q[0]); tgt.push_back(q[1]);
      tgt.push_back(q[2] + 1); tgt.push_back(q[0]); tgt.push_back(q[1]);
    });
    if (tgt.empty()) { apply(g, w, +1, fl); stats[3]++; continue; }
    src.clear();
    const i64 pp = p0 + hp;
    for (i64 a = ap_off[pp]; a < ap_off[pp + 1]; a++) {
      i64 l = aps[3 * a], x = aps[3 * a + 1], y = aps[3 * a + 2];
      if (std::llabs(x - gx) <= W && std::llabs(y - gy) <= W) { src.push_back((int32_t)l); src.push_back((int32_t)x); src.push_back((int32_t)y); }
    }
    double cost = 0.0; int np2;
    stats[7]++;
    while (true) {
      np2 = awn::astar3d_win((int)L, (int)X, (int)Y, d, c, u, dirs32, strides6, src.data(), (int)(src.size() / 3), tgt.data(),
                             (int)(tgt.size() / 3), (int)gx, (int)gy, (int)W, step_pen, via_pen, pathbuf.data(), (int)(pathbuf.size() / 3), &cost);
      if (np2 >= 0) break;
      pathbuf.assign((size_t)(-np2) * 2 * 3, 0);
    }
    if (np2 == 0) { apply(g, w, +1, fl); stats[4]++; continue; }
    // path_to_wires_vias + merge_wires(w, pw) + v_rest | pv
    WV pw; std::unordered_map<i64, size_t> ppos;
    for (int q = 0; q + 1 < np2; q++) {
      i64 z1 = pathbuf[3 * q], x1 = pathbuf[3 * q + 1], y1 = pathbuf[3 * q + 2];
      i64 z2 = pathbuf[3 * q + 3], x2 = pathbuf[3 * q + 4], y2 = pathbuf[3 * q + 5];
      if (z1 != z2) { pw.vias.add3((int)x1, (int)y1, (int)std::min(z1, z2)); continue; }
      i64 f = dirs8[z1] == 0 ? y1 : x1;
      IV ab = dirs8[z1] == 0 ? IV{std::min(x1, x2), std::max(x1, x2)} : IV{std::min(y1, y2), std::max(y1, y2)};
      i64 kk = kz(z1, f); auto pi = ppos.find(kk);
      if (pi == ppos.end()) { ppos[kk] = pw.keys.size(); pw.keys.push_back({z1, f}); pw.ivs.emplace_back(); pi = ppos.find(kk); }
      pw.ivs[pi->second].push_back(ab);
    }
    WV w2;
    {
      std::vector<std::vector<IV>> raw; std::unordered_map<i64, size_t> rpos;
      auto addk = [&](const WV& x) {
        for (size_t q = 0; q < x.keys.size(); q++) {
          i64 kk = kz(x.keys[q].first, x.keys[q].second); auto ri = rpos.find(kk);
          if (ri == rpos.end()) { rpos[kk] = raw.size(); w2.keys.push_back(x.keys[q]); raw.emplace_back(); ri = rpos.find(kk); }
          raw[ri->second].insert(raw[ri->second].end(), x.ivs[q].begin(), x.ivs[q].end());
        }
      };
      addk(w); addk(pw);
      w2.ivs.resize(raw.size());
      for (size_t q = 0; q < raw.size(); q++) merge_list(raw[q], w2.ivs[q]);
    }
    w2.vias = v_rest.uni(pw.vias);
    if (w2.vias.bad || v_rest.bad || w.vias.bad) out->err = 1;
    double nw = marginal(g, w2, fl);
    if (nw < old - 1e-9) {
      apply(g, w2, +1, fl);
      auto po = out->pos.find(k);
      if (po == out->pos.end()) { out->pos[k] = out->rows.size(); out->order.push_back(k); out->rows.emplace_back(); po = out->pos.find(k); }
      emit(g, w2, out->rows[po->second]);
      cur[k] = std::move(w2);                                  // (w, wtmp may alias cur[k]: no use of w after this line)
      stats[5]++;
    } else {
      apply(g, w, +1, fl);
      stats[6]++;
    }
  }
  return out;
}
i64 mid5_esc_n(void* h, i64* nrows) {
  esc::Out* o = (esc::Out*)h; i64 t = 0;
  for (auto& r : o->rows) t += (i64)r.size() / 6;
  *nrows = t; return o->err ? -1 : (i64)o->order.size();
}
void mid5_esc_get(void* h, i64* idx, i64* off, i64* rows) {
  esc::Out* o = (esc::Out*)h; i64 t = 0;
  off[0] = 0;
  for (size_t q = 0; q < o->order.size(); q++) {
    idx[q] = o->order[q];
    std::memcpy(rows + 6 * t, o->rows[q].data(), o->rows[q].size() * sizeof(i64));
    t += (i64)o->rows[q].size() / 6; off[q + 1] = t;
  }
}
void mid5_esc_free(void* h) { delete (esc::Out*)h; }

// ---------------------------------------------------------------- CPython set emulation test driver (tools/mid5/test_pyset.py)
// ops: per trial: nv triples of v (adds in order), nt triples of tail, np triples of pv; out: iteration orders of v, v - tail,
// (v - tail) | pv concatenated (3 ints each), and for arity 2 the same with pairs (z ignored).
i64 mid5_pyset_trial(int arity, i64 nv, const int32_t* v, i64 nt, const int32_t* t, i64 np_, const int32_t* p, int32_t* out, i64 chain) {
  pyset::Set V(arity);
  for (i64 i = 0; i < nv; i++) V.add(v + 3 * i);
  i64 m = 0;
  auto dump = [&](const pyset::Set& s) { s.each([&](const int32_t* k) { for (int q = 0; q < 3; q++) out[m++] = q < arity ? k[q] : 0; }); };
  dump(V);
  pyset::Set T(arity); for (i64 i = 0; i < nt; i++) T.add(t + 3 * i);
  pyset::Set P(arity); for (i64 i = 0; i < np_; i++) P.add(p + 3 * i);
  pyset::Set R = V.difference(T); dump(R);
  pyset::Set U = R.uni(P); dump(U);
  for (i64 c = 0; c < chain; c++) {                          // repeat: U - T, then | P (cur net reused by a later candidate)
    pyset::Set R2 = U.difference(T); dump(R2);
    U = R2.uni(P); dump(U);
  }
  return (V.bad || R.bad || U.bad) ? -m : m;
}

// ---------------------------------------------------------------- escape test: blocked share of the overflow cost
// d, c: [L, X, Y] with element strides; u[L]; official: idle blocked cells cost nothing. Sums in long double per thread.
double mid5_blocked_share(const double* d, const double* c, const double* u, i64 sz, i64 sx, i64 sy, i64 cz, i64 cx, i64 cy,
                          i64 L, i64 X, i64 Y, int official, int nthr, double* totblk) {
  const double EPS = 1e-3;
  if (nthr < 1) nthr = 1;
  i64 rows = (L - 1) * X;
  std::vector<long double> T(nthr, 0.0L), B(nthr, 0.0L);
  auto work = [&](int w) {
    long double tt = 0.0L, bb = 0.0L;
    for (i64 r = w; r < rows; r += nthr) {
      i64 z = 1 + r / X, x = r % X;
      long double ts = 0.0L, bs = 0.0L;
      for (i64 y = 0; y < Y; y++) {
        double dv = d[z * sz + x * sx + y * sy], cv = c[z * cz + x * cx + y * cy];
        double a;
        if (cv > EPS) a = std::exp(std::min(0.25 * dv - 0.5 * cv, 700.0));
        else a = std::exp(std::min(0.75 * dv, 700.0));
        if (official && cv <= EPS && dv <= 0) a = 0.0;
        ts += a;
        if (cv <= EPS && dv > 0) bs += std::exp(std::min(0.75 * dv, 700.0)) - 1.0;
      }
      tt += (long double)u[z] * ts; bb += (long double)u[z] * bs;
    }
    T[w] = tt; B[w] = bb;
  };
  std::vector<std::thread> th;
  for (int w = 1; w < nthr; w++) th.emplace_back(work, w);
  work(0);
  for (auto& x : th) x.join();
  long double tt = 0.0L, bb = 0.0L;
  for (int w = 0; w < nthr; w++) { tt += T[w]; bb += B[w]; }
  totblk[0] = (double)tt; totblk[1] = (double)bb;
  return tt > 0 ? (double)(bb / tt) : 0.0;
}
}  // extern "C"

extern "C" {
// escape pass: the nets ovf_lib.escape_pass scans, rows part: base rows arr[nr, 7] (net, x1, y1, z1, x2, y2, z2) of nets not in
// `chg` (uint8 per net), via rows (z1 != z2) whose key x1 * Y + y1 is in `keym` (uint8 over key values 0..nk-1, the aliasing of
// escape_pass's key arithmetic kept); out: hit[net] = 1. T threads (each net's flag is only ever set to 1: no ordering issue).
void mid5_hot_nets(const int64_t* arr, int64_t nr, const uint8_t* chg, const uint8_t* keym, int64_t nk, int64_t Y, uint8_t* hit, int nthr) {
  if (nthr < 1) nthr = 1;
  auto work = [&](int w) {
    int64_t lo = nr * w / nthr, hi = nr * (w + 1) / nthr;
    for (int64_t r = lo; r < hi; r++) {
      const int64_t* q = arr + 7 * r;
      if (q[3] == q[6] || chg[q[0]]) continue;
      int64_t k = q[1] * Y + q[2];
      if (k >= 0 && k < nk && keym[k]) hit[q[0]] = 1;
    }
  };
  std::vector<std::thread> th;
  for (int w = 1; w < nthr; w++) th.emplace_back(work, w);
  work(0);
  for (auto& t : th) t.join();
}
}
