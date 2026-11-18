// mid5_rr.cc: the overflow stage's ranked reroute loop (tools/wave3/ovf2/rr_loop.cc w3_rr_rank) on
// T threads with the SAME result: speculative evaluation of a batch of candidate runs in parallel, each on a private copy of the
// demand map equal to the committed state at the start of the batch, then an in-order commit; a candidate whose read set (its
// net's cells and its A* windows, +-1 cell) meets a cell written by an earlier reroute committed in the same batch, or whose net
// was changed in the batch, is re-evaluated serially on the committed state (= exactly what the serial loop computes for it).
// Exactness rests on the demand map holding integers (every apply adds +-1 / +-2): then apply / restore are exact and the order
// of commits never changes a value; the driver checks this and runs the serial loop (w3_rr_rank's body) when it does not hold.
// The per-candidate code is rr_loop.cc's loop body verbatim (lines 490-652), with the A* of tools/mid5/astar_tl.inc (the same
// function, thread-local buffers). rr_loop.cc and astar_tl.inc are included unchanged.
// Build: tools/mid5/build_mid5.sh (with mid5_kern.cc into libmid5).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <limits>
#include <thread>
#include <atomic>

namespace rrn {
#include "../wave3/ovf2/rr_loop.cc"
}
namespace atl {
#include "astar_tl.inc"
}

namespace mrr {
using namespace rrn;
typedef int64_t i64;

struct Ctx {                                    // read-only inputs of one call
  int L, X, Y; const double *c, *cs, *u; const int32_t* dirs; i64 cz, cx, cy, csz, csx, csy;
  const i64 *pin_off, *ap_off; const int32_t* aps; int W;
};
struct Res {                                    // one evaluated candidate
  int kind = 0;                                 // 0 stale, 1 fail (no path), 2 invalid, 3 rerouted, 4 kept
  i64 st[16] = {0};
  Wires w2; PySet v2;
  std::vector<std::array<int, 3>> win;          // (x0, y0, half) of each A* window
  std::vector<int32_t> rd;                      // xy pairs of every cell the A* searches read (exact read set)
  bool astar = false;
};
struct Scratch { std::vector<int32_t> pathbuf; int path_cap; std::vector<int32_t> src, tgt;
  Scratch() : pathbuf((size_t)3 << 16, 0), path_cap(1 << 16) {} };

// rr_loop.cc lines 490-652 (the body of the candidate loop), with `continue` -> return and the stats into r.st.
// commit = true: leaves the grid as the serial loop does (w2 applied when rerouted, else w); false: restores the grid.
static void eval(const Ctx& x, Grid& g, const i64* sstr, const Cand& cd, const NetSt& ns, Scratch& S, Res& r, bool commit) {
  const int L = x.L, X = x.X, Y = x.Y; const int32_t* dirs = x.dirs; const double* cs = x.cs; const double* u = x.u;
  double* d = g.d; const int W = x.W; i64* stats = r.st;
  int s = cd.slot;
  const Wires& w = ns.w;
  const PySet& v = ns.v;
  {   // stale
    WIdx ix(w);
    int k = ix.get(cd.z, cd.f);
    bool found = false;
    if (k >= 0) for (auto& p : w[k].iv) if (p.first == cd.a && p.second == cd.b) { found = true; break; }
    if (!found) { stats[3]++; r.kind = 0; return; }
  }
  const int32_t* ap = x.aps + 3 * x.ap_off[x.pin_off[s]];
  std::vector<i64> poff_local;
  int npins = (int)(x.pin_off[s + 1] - x.pin_off[s]);
  poff_local.resize(npins + 1);
  for (int p = 0; p <= npins; ++p) poff_local[p] = x.ap_off[x.pin_off[s] + p] - x.ap_off[x.pin_off[s]];
  apply(g, w, v, -1);
  double old = marginal(g, w, v);
  Wires w_rest = drop_run(w, cd.z, cd.f, cd.a, cd.b);
  int z = cd.z, fx = cd.f;
  bool h = dirs[z] == 0;
  int A[3] = {z, h ? cd.a : fx, h ? fx : cd.a};
  int B[3] = {z, h ? cd.b : fx, h ? fx : cd.b};
  Comp cm; cm.build(g, w_rest, v);
  auto is_pin = [&](const int* q) {
    i64 na = poff_local[npins];
    for (i64 k = 0; k < na; ++k) if (ap[3 * k] == q[0] && ap[3 * k + 1] == q[1] && ap[3 * k + 2] == q[2]) return true;
    return false;
  };
  int nodeA = cm.lookup(A[0], A[1], A[2]), nodeB = cm.lookup(B[0], B[1], B[2]);
  bool pinA = false, pinB = false;
  if (nodeA < 0 && is_pin(A)) { pinA = true; stats[4]++; }
  if (nodeB < 0 && is_pin(B)) { pinB = true; stats[4]++; }
  bool inA = nodeA >= 0 || pinA, inB = nodeB >= 0 || pinB;
  int nn = (int)cm.nodes.size();
  std::vector<int> gid_of_root(nn, -1), node_gid(nn);
  int ng = 0;
  for (int k = 0; k < nn; ++k) { int rr = cm.find(k); if (gid_of_root[rr] < 0) gid_of_root[rr] = ng++; node_gid[k] = gid_of_root[rr]; }
  int gA_pin = -1, gB_pin = -1;
  if (pinA) gA_pin = ng++;
  if (pinB) gB_pin = ng++;
  auto gid = [&](const int* q, int node, bool pin, int gp) { (void)q; return pin ? gp : node_gid[node]; };
  Wires w2; PySet v2;
  bool reconnect = false, simple = false;
  if (!inA && !inB) { stats[5]++; simple = true; }
  else if (inA && inB && gid(A, nodeA, pinA, gA_pin) == gid(B, nodeB, pinB, gB_pin)) simple = true;
  else reconnect = true;
  bool fail = false;
  std::vector<int32_t>& src = S.src; std::vector<int32_t>& tgt = S.tgt;
  if (simple) { w2 = w_rest; v2 = v; }
  else if (reconnect) {
    if (!(inA && inB)) stats[6]++;
    const int* E0 = inA ? A : B;
    int nodeE = inA ? nodeA : nodeB; bool pinE = inA ? pinA : pinB; int gpE = inA ? gA_pin : gB_pin;
    int gE = gid(E0, nodeE, pinE, gpE);
    int x0 = (A[1] + B[1]) / 2, y0 = (A[2] + B[2]) / 2;
    int half = std::max(std::abs(A[1] - B[1]), std::abs(A[2] - B[2])) / 2 + W;
    int ex = E0[1], ey = E0[2];
    auto inwin = [&](int xx, int yy) { return std::abs(xx - x0) <= half && std::abs(yy - y0) <= half; };
    std::vector<int> gmin(ng, INT32_MAX);
    std::vector<char> gwin(ng, 0);
    std::vector<std::vector<int>> gnodes(ng);
    for (int k = 0; k < nn; ++k) gnodes[node_gid[k]].push_back(k);
    auto node_minmax = [&](int k, int& dmin, bool& win) {
      const auto& nd = cm.nodes[k];
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
    auto win_points = [&](int q, std::vector<int32_t>& out) {
      out.clear();
      if (q == gA_pin) { out.insert(out.end(), {A[0], A[1], A[2]}); return; }
      if (q == gB_pin) { out.insert(out.end(), {B[0], B[1], B[2]}); return; }
      for (int k : gnodes[q]) {
        const auto& nd = cm.nodes[k];
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
    std::unordered_set<uint64_t> tset;
    win_points(gE, src);
    for (size_t k = 0; k < src.size(); k += 3) tset.insert(pk(src[k], src[k + 1], src[k + 2]));
    w2 = w_rest; v2 = v.copy();
    for (int q : others) {
      win_points(q, tgt);
      if (src.empty() || tgt.empty()) { fail = true; break; }
      int np_;
      double cost = 0.0;
      r.win.push_back({x0, y0, half}); r.astar = true;
      atl::g_rd.clear(); atl::g_rd_on = true;
      while (true) {
        np_ = atl::astar3d_win_tl(L, X, Y, d, cs, u, dirs, sstr, src.data(), (int)(src.size() / 3), tgt.data(), (int)(tgt.size() / 3),
                                  x0, y0, half, 1.0, 0.1, S.pathbuf.data(), S.path_cap, &cost);
        if (np_ >= 0) break;
        S.path_cap = -np_ * 2; S.pathbuf.assign((size_t)S.path_cap * 3, 0);
      }
      atl::g_rd_on = false; r.rd.insert(r.rd.end(), atl::g_rd.begin(), atl::g_rd.end());
      stats[12]++;
      if (np_ == 0) { fail = true; break; }
      const std::vector<int32_t>& pathbuf = S.pathbuf;
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
    if (fail) { apply(g, w, v, +1); stats[7]++; r.kind = 1; return; }
  }
  if (!net_valid(g, w2, v2, ap, poff_local.data(), npins)) { apply(g, w, v, +1); stats[8]++; r.kind = 2; return; }
  double nw = marginal(g, w2, v2);
  if (nw < old - 1e-9) {
    apply(g, w2, v2, +1);
    if (!commit) { apply(g, w2, v2, -1); apply(g, w, v, +1); }
    r.w2 = std::move(w2); r.v2 = std::move(v2);
    stats[9]++; r.kind = 3;
  } else {
    apply(g, w, v, +1); stats[10]++; r.kind = 4;
  }
}

// xy cells a net's demand touches (wire cells [a, b] of its tracks, via cells (x, y), (x - 1, y), (x, y - 1))
template <class F> static void net_cells(const Wires& w, const PySet& v, const int32_t* dirs, F f) {
  for (const Track& t : w)
    for (auto& p : t.iv)
      for (int q = p.first; q <= p.second; ++q) { if (dirs[t.z] == 0) f(q, t.f); else f(t.f, q); }
  for (const SE& e : v.t) if (e.k) { f(e.x, e.y); f(e.x - 1, e.y); f(e.x, e.y - 1); }
}
}  // namespace mrr

extern "C" {
// == w3_rr_rank (same arguments; the A* pointer is ignored: the thread-local copy of the same function is used) + nthr, batch;
// info[8] out: 0 parallel(1)/serial(0), 1 batches, 2 speculative accepted, 3 re-evaluated, 4 conflicts by net, 5 by cells.
// The changed slots and their rows are read with w3_rr_nseg / w3_rr_get of the included rr_loop (same globals).
int mid5_rr_rank(int L, int X, int Y, double* d, const double* c, const double* cs, const double* u, const int32_t* dirs,
                 const int64_t* strides, void* astar_ptr, int n, const int64_t* ridx, const int64_t* seg_off, const int32_t* segs,
                 const int64_t* pin_off, const int64_t* ap_off, const int32_t* aps, int W, int hmax, int max_runs, int64_t* stats,
                 int nthr, int batch, int64_t* info) {
  using namespace rrn;
  using namespace mrr;
  for (int k = 0; k < 8; ++k) info[k] = 0;
  // integer demand check (else the serial loop of rr_loop.cc, unchanged)
  bool integral = true;
  for (int z = 0; z < L && integral; ++z)
    for (int x = 0; x < X && integral; ++x)
      for (int y = 0; y < Y; ++y) {
        double v = d[z * strides[0] + x * strides[1] + y * strides[2]];
        if (!(v == std::floor(v)) || std::fabs(v) > 1e15) { integral = false; break; }
      }
  if (!integral || nthr <= 1)
    return w3_rr_rank(L, X, Y, d, c, cs, u, dirs, strides, (void*)&atl::astar3d_win_tl, n, ridx, seg_off, segs, pin_off, ap_off, aps,
                      W, hmax, max_runs, stats);
  info[0] = 1;
  Grid g{L, X, Y, d, c, u, dirs, strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]};
  const int64_t sstr[6] = {strides[0], strides[1], strides[2], strides[6], strides[7], strides[8]};
  for (int k = 0; k < 16; ++k) stats[k] = 0;
  // ---- rr_loop.cc lines 429-484 verbatim (prefix sums, candidate runs, rank order, budget)
  auto ovc = [&](int z, int x, int y) -> bool {
    if (z == 0) return false;
    double dd_ = g.D(z, x, y), cc = g.C(z, x, y);
    return (cc <= CAP_EPS && dd_ > 0) || (cc > CAP_EPS && dd_ > 2.0 * cc);
  };
  std::unordered_map<uint64_t, std::pair<std::vector<double>, std::vector<int32_t>>> pref;
  auto get_pref = [&](int z, int f) -> std::pair<std::vector<double>, std::vector<int32_t>>& {
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
    NetSt& ns = cur[s];
    to_wires_vias(segs + 6 * seg_off[s], seg_off[s + 1] - seg_off[s], dirs, ns.w, ns.v);
    i64 tot = (i64)ns.v.size();
    for (auto& t : ns.w) tot += (i64)t.iv.size();
    if (tot > BIG_NET) { stats[0]++; ns.w.clear(); ns.v = PySet(); continue; }
    live[s] = 1;
    for (auto& t : ns.w) {
      if (t.z < 1 || t.z >= L) continue;
      auto& pr = get_pref(t.z, t.f);
      for (auto& p : t.iv) {
        int a = p.first, b = p.second;
        int k = pr.second[b] - pr.second[a];
        if (!k) continue;
        if (hmax && (b - a) / 2 + W > hmax) { stats[1]++; continue; }
        double sc = pr.first[b] - pr.first[a];
        cand.push_back(Cand{-sc, ridx[s], t.z, t.f, a, b, s});
      }
    }
  }
  std::sort(cand.begin(), cand.end(), [](const Cand& p, const Cand& q) {
    if (p.nsc != q.nsc) return p.nsc < q.nsc;
    if (p.i != q.i) return p.i < q.i;
    if (p.z != q.z) return p.z < q.z;
    if (p.f != q.f) return p.f < q.f;
    if (p.a != q.a) return p.a < q.a;
    return p.b < q.b;
  });
  stats[2] = (i64)cand.size();
  if (max_runs && (i64)cand.size() > max_runs) cand.resize(max_runs);
  std::vector<char> changed(n, 0);
  // ---- speculative batches
  Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W};
  const int T = nthr;
  const i64 PZ = (i64)X * Y, PX = Y, PY = 1;
  std::vector<std::vector<double>> priv(T, std::vector<double>((size_t)L * X * Y));
  {
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
      double* p = priv[t].data();
      for (int z = 0; z < L; ++z) for (int x = 0; x < X; ++x) for (int y = 0; y < Y; ++y)
        p[z * PZ + x * PX + y * PY] = d[z * strides[0] + x * strides[1] + y * strides[2]];
    });
    for (auto& t : th) t.join();
  }
  std::vector<Grid> pg(T);
  std::vector<std::array<int64_t, 6>> pstr(T);
  for (int t = 0; t < T; ++t) {
    pg[t] = Grid{L, X, Y, priv[t].data(), c, u, dirs, PZ, PX, PY, strides[3], strides[4], strides[5]};
    pstr[t] = {PZ, PX, PY, strides[6], strides[7], strides[8]};
  }
  std::vector<Scratch> scr(T + 1);
  const int Bsz = std::max(1, batch);
  std::vector<Res> res(Bsz);
  std::vector<uint32_t> dirty((size_t)(X + 2) * (Y + 2), 0);
  uint32_t stamp = 0;
  auto mark = [&](int x, int y) { if (x >= -1 && y >= -1 && x <= X && y <= Y) dirty[(size_t)(x + 1) * (Y + 2) + (y + 1)] = stamp; };
  auto isd = [&](int x, int y) { return x >= -1 && y >= -1 && x <= X && y <= Y && dirty[(size_t)(x + 1) * (Y + 2) + (y + 1)] == stamp; };
  std::vector<uint32_t> slot_stamp(n, 0);
  struct Chg { int s; Wires w; PySet v; Wires w2; PySet v2; };   // committed reroutes of a batch (old, new), for the private copies
  std::vector<Chg> chg;
  i64 nc = (i64)cand.size();
  for (i64 b0 = 0; b0 < nc; b0 += Bsz) {
    i64 b1 = std::min(nc, b0 + Bsz), nb = b1 - b0;
    info[1]++;
    for (i64 j = 0; j < nb; ++j) res[j] = Res();
    std::atomic<i64> next(0);
    {
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
        while (true) {
          i64 j = next.fetch_add(1);
          if (j >= nb) break;
          const Cand& cd = cand[b0 + j];
          eval(cx, pg[t], pstr[t].data(), cd, cur[cd.slot], scr[t], res[j], false);
        }
      });
      for (auto& t : th) t.join();
    }
    ++stamp; chg.clear();
    for (i64 j = 0; j < nb; ++j) {
      const Cand& cd = cand[b0 + j];
      int s = cd.slot;
      Res& r = res[j];
      bool conflict = slot_stamp[s] == stamp;
      if (conflict) info[4]++;
      if (!conflict && r.kind != 0) {          // the net's own cells and the A* windows (+-1) against this batch's writes
        net_cells(cur[s].w, cur[s].v, dirs, [&](int x, int y) { if (!conflict && isd(x, y)) conflict = true; });
        for (size_t q = 0; q + 1 < r.rd.size() && !conflict; q += 2) if (isd(r.rd[q], r.rd[q + 1])) conflict = true;
        if (conflict) info[5]++;
      }
      if (conflict) {                           // serial evaluation on the committed state (the serial loop's step)
        Res r2;
        eval(cx, g, sstr, cd, cur[s], scr[T], r2, true);
        r = std::move(r2); info[3]++;
        if (r.kind == 3) {
          chg.push_back(Chg{s, cur[s].w, cur[s].v, r.w2, r.v2});
          net_cells(cur[s].w, cur[s].v, dirs, mark); net_cells(r.w2, r.v2, dirs, mark);
          cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; slot_stamp[s] = stamp;
        }
      } else {
        info[2]++;
        if (r.kind == 3) {
          apply(g, cur[s].w, cur[s].v, -1); apply(g, r.w2, r.v2, +1);
          chg.push_back(Chg{s, cur[s].w, cur[s].v, r.w2, r.v2});
          net_cells(cur[s].w, cur[s].v, dirs, mark); net_cells(r.w2, r.v2, dirs, mark);
          cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; slot_stamp[s] = stamp;
        }
      }
      for (int k = 0; k < 16; ++k) stats[k] += r.st[k];
    }
    if (!chg.empty()) {                         // private copies follow the committed state
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
        for (auto& ch : chg) { apply(pg[t], ch.w, ch.v, -1); apply(pg[t], ch.w2, ch.v2, +1); }
      });
      for (auto& t : th) t.join();
    }
  }
  // ---- output: rr_loop.cc lines 654-674 verbatim
  R_changed.clear(); R_off.assign(1, 0); R_seg.clear();
  std::vector<int> sl;
  for (int s = 0; s < n; ++s) if (changed[s]) sl.push_back(s);
  std::sort(sl.begin(), sl.end(), [&](int p, int q) { return ridx[p] < ridx[q]; });
  for (int s : sl) {
    const NetSt& ns = cur[s];
    for (const Track& t : ns.w)
      for (auto& p : t.iv) {
        if (dirs[t.z] == 0) R_seg.insert(R_seg.end(), {p.first, t.f, t.z, p.second, t.f, t.z});
        else R_seg.insert(R_seg.end(), {t.f, p.first, t.z, t.f, p.second, t.z});
      }
    std::vector<std::array<int32_t, 3>> vv;
    for (const SE& e : ns.v.t) if (e.k) vv.push_back({e.x, e.y, e.z});
    std::sort(vv.begin(), vv.end());
    for (auto& q : vv) R_seg.insert(R_seg.end(), {q[0], q[1], q[2], q[0], q[1], q[2] + 1});
    R_changed.push_back(s);
    R_off.push_back((i64)(R_seg.size() / 6));
  }
  stats[11] = (i64)R_changed.size();
  return (int)R_changed.size();
}
int64_t mid5_rr_nseg() { return rrn::w3_rr_nseg(); }
void mid5_rr_get(int32_t* slots, int64_t* off, int32_t* seg) { rrn::w3_rr_get(slots, off, seg); }
}
