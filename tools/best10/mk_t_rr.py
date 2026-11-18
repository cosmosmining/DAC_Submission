#!/usr/bin/env python3
"""mk_t_rr.py: tools/best10/t_rr.cc + astar_t.inc from tools/best5/n_rr.cc + tools/mid5/astar_tl.inc; every patch must match exactly once.
New (all off by default: then policy 4 is n_rr's task graph, same route):
  rect (H, W2)   a run with half = len/2 + W > H searches a rectangle: +-half along the run, +-W2 across (footprint and A* window), instead of the square +-half
  long (Hd, Nd)  runs with half > Hd go after all other runs of the budget, in rank order; Nd > 0: only the first Nd of them are kept
  popcap N       an A* with more than N pops gives up (the run keeps its path)
  T_RR_DUMP=file one line per run: rank half pops dur kind gain level (diagnostics, changes nothing)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, "..", "best5", "n_rr.cc")).read()
a = open(os.path.join(HERE, "..", "mid5", "astar_tl.inc")).read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:90])
    s = s.replace(old, new)


def repa(old, new, cnt=1):
    global a
    assert a.count(old) == cnt, (a.count(old), old[:90])
    a = a.replace(old, new)


# ---- A*: rectangle and pop cap through thread-locals (signature unchanged)
repa('''thread_local bool g_rd_on = false;
''', '''thread_local bool g_rd_on = false;
thread_local int g_hx = -1, g_hy = -1;                 // speed10: >= 0: half widths of the window instead of W
thread_local int64_t g_popcap = 0, g_pops = 0;         // speed10: pop cap (0 = none); pops of the last call
''')
repa('''    int xlo = std::max(0, x0 - W), xhi = std::min(X - 1, x0 + W);
    int ylo = std::max(0, y0 - W), yhi = std::min(Y - 1, y0 + W);
''', '''    const int WX = g_hx >= 0 ? g_hx : W, WY = g_hy >= 0 ? g_hy : W;
    int xlo = std::max(0, x0 - WX), xhi = std::min(X - 1, x0 + WX);
    int ylo = std::max(0, y0 - WY), yhi = std::min(Y - 1, y0 + WY);
''')
repa('''    while (!pq.empty()) {
        Item it = pq.top(); pq.pop();
        int64_t i = idx(it.z, it.x, it.y);
        if (it.g > g_dist[i]) continue;
''', '''    g_pops = 0;
    while (!pq.empty()) {
        Item it = pq.top(); pq.pop();
        int64_t i = idx(it.z, it.x, it.y);
        if (it.g > g_dist[i]) continue;
        if (++g_pops > g_popcap && g_popcap > 0) break;
''')
open(os.path.join(HERE, "astar_t.inc"), "w").write(a)
# ---- kernel
rep('''#include "n_dag.h"
''', '''#include <cstdio>
#include <cstdlib>
#include "../best5/n_dag.h"
''')
rep('''#include "../mid5/astar_tl.inc"
''', '''#include "astar_t.inc"
''')
rep('''  const i64 *pin_off, *ap_off; const int32_t* aps; int W;
};''', '''  const i64 *pin_off, *ap_off; const int32_t* aps; int W;
  int rH = 0, rW = 0; i64 popcap = 0;           // speed10: rectangle above half rH (across +-rW), A* pop cap
};''')
rep('''  bool astar = false;
};''', '''  bool astar = false;
  double gain = 0; i64 pops = 0;                // speed10: old - new marginal cost, A* pops
};''')
rep('''    int half = std::max(std::abs(A[1] - B[1]), std::abs(A[2] - B[2])) / 2 + W;
    int ex = E0[1], ey = E0[2];
    auto inwin = [&](int xx, int yy) { return std::abs(xx - x0) <= half && std::abs(yy - y0) <= half; };
''', '''    int half = std::max(std::abs(A[1] - B[1]), std::abs(A[2] - B[2])) / 2 + W;
    int hx = half, hy = half;                   // speed10: rectangle for a long run (else the square, n_rr's values)
    if (x.rH > 0 && half > x.rH) { if (h) hy = std::min(hy, x.rW); else hx = std::min(hx, x.rW); }
    int ex = E0[1], ey = E0[2];
    auto inwin = [&](int xx, int yy) { return std::abs(xx - x0) <= hx && std::abs(yy - y0) <= hy; };
''')
rep('''        int lo = std::max(nd.a, x0 - half), hi = std::min(nd.b, x0 + half);
        win = lo <= hi && std::abs(nd.f - y0) <= half;
''', '''        int lo = std::max(nd.a, x0 - hx), hi = std::min(nd.b, x0 + hx);
        win = lo <= hi && std::abs(nd.f - y0) <= hy;
''')
rep('''        int lo = std::max(nd.a, y0 - half), hi = std::min(nd.b, y0 + half);
        win = lo <= hi && std::abs(nd.f - x0) <= half;
''', '''        int lo = std::max(nd.a, y0 - hy), hi = std::min(nd.b, y0 + hy);
        win = lo <= hi && std::abs(nd.f - x0) <= hx;
''')
rep('''          if (std::abs(nd.f - y0) > half) continue;
          for (int t = std::max(nd.a, x0 - half); t <= std::min(nd.b, x0 + half); ++t) out.insert(out.end(), {nd.z, t, nd.f});
''', '''          if (std::abs(nd.f - y0) > hy) continue;
          for (int t = std::max(nd.a, x0 - hx); t <= std::min(nd.b, x0 + hx); ++t) out.insert(out.end(), {nd.z, t, nd.f});
''')
rep('''          if (std::abs(nd.f - x0) > half) continue;
          for (int t = std::max(nd.a, y0 - half); t <= std::min(nd.b, y0 + half); ++t) out.insert(out.end(), {nd.z, nd.f, t});
''', '''          if (std::abs(nd.f - x0) > hx) continue;
          for (int t = std::max(nd.a, y0 - hy); t <= std::min(nd.b, y0 + hy); ++t) out.insert(out.end(), {nd.z, nd.f, t});
''')
rep('''      if (NREC) { atl::g_rd.clear(); atl::g_rd_on = true; }
      while (true) {
''', '''      if (NREC) { atl::g_rd.clear(); atl::g_rd_on = true; }
      atl::g_hx = hx; atl::g_hy = hy; atl::g_popcap = x.popcap;
      while (true) {
''')
rep('''      stats[12]++;
      if (np_ == 0) { fail = true; break; }
''', '''      stats[12]++; r.pops += atl::g_pops;
      if (np_ == 0) { fail = true; break; }
''')
rep('''  double nw = marginal(g, w2, v2);
  if (nw < old - 1e-9) {
''', '''  double nw = marginal(g, w2, v2);
  r.gain = old - nw;
  if (nw < old - 1e-9) {
''')
# signature, names
rep('''int n_rr_rank(int L, int X, int Y, double* d,''', '''int t_rr_rank(int L, int X, int Y, double* d,''')
rep('''                 int nthr, int batch, int64_t* info, int policy, int blk_shift, double* dinfo) {
''', '''                 int nthr, int batch, int64_t* info, int policy, int blk_shift, double* dinfo, int rH, int rW, int dH, int dN, int64_t popcap, int64_t areaB) {
''')
rep('''int64_t n_rr_nseg() { return rrn::w3_rr_nseg(); }
void n_rr_get(int32_t* slots, int64_t* off, int32_t* seg) { rrn::w3_rr_get(slots, off, seg); }
''', '''int64_t t_rr_nseg() { return rrn::w3_rr_nseg(); }
void t_rr_get(int32_t* slots, int64_t* off, int32_t* seg) { rrn::w3_rr_get(slots, off, seg); }
''')
# ---- long runs last, after the budget
rep('''  if (max_runs && (i64)cand.size() > max_runs) cand.resize(max_runs);
  std::vector<char> changed(n, 0);
''', '''  if (max_runs && (i64)cand.size() > max_runs) cand.resize(max_runs);
  auto half_of = [&](const Cand& q) { return (q.b - q.a) / 2 + W; };
  {                                             // speed10: work budget: cells of the runs' windows, in rank order (deterministic); dinfo[6] = cells of the runs kept, dinfo[7] = runs cut
    i64 acc = 0, keep = 0;
    for (const Cand& q : cand) {
      i64 hf = half_of(q), wd = (rH > 0 && hf > rH) ? std::min<i64>(hf, rW) : hf;
      i64 area = (2 * hf + 1) * (2 * wd + 1);
      if (areaB > 0 && acc + area > areaB) break;
      acc += area; keep++;
    }
    dinfo[7] = (double)((i64)cand.size() - keep); dinfo[6] = (double)acc;
    cand.resize((size_t)keep);
  }
  if (dH > 0) {                                 // speed10: long runs after the others, rank order kept in each part
    std::stable_partition(cand.begin(), cand.end(), [&](const Cand& q) { return half_of(q) <= dH; });
    if (dN > 0) {
      i64 k = 0, keep = 0;
      for (const Cand& q : cand) { if (half_of(q) <= dH || k < dN) keep++; if (half_of(q) > dH) k++; }
      cand.resize((size_t)keep);
    }
  }
  std::vector<char> changed(n, 0);
''')
# ---- task graph: rectangle footprint, dump
rep('''      int ax = h ? cd.a : cd.f, ay = h ? cd.f : cd.a, bx = h ? cd.b : cd.f, by = h ? cd.f : cd.b;
      int x0 = (ax + bx) / 2, y0 = (ay + by) / 2, half = std::max(std::abs(ax - bx), std::abs(ay - by)) / 2 + W;
      rects[s].push_back({x0 - half - 1, x0 + half + 1, y0 - half - 1, y0 + half + 1});
''', '''      int ax = h ? cd.a : cd.f, ay = h ? cd.f : cd.a, bx = h ? cd.b : cd.f, by = h ? cd.f : cd.b;
      int x0 = (ax + bx) / 2, y0 = (ay + by) / 2, half = std::max(std::abs(ax - bx), std::abs(ay - by)) / 2 + W;
      int hx = half, hy = half;
      if (rH > 0 && half > rH) { if (h) hy = std::min(hy, rW); else hx = std::min(hx, rW); }
      rects[s].push_back({x0 - hx - 1, x0 + hx + 1, y0 - hy - 1, y0 + hy + 1});
''')
rep('''    Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W};
    std::vector<Scratch> scr(T);
    std::vector<std::array<i64, 16>> tst(T);
    for (auto& q : tst) q.fill(0);
    ndag::Info di;
''', '''    Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W, rH, rW, popcap};
    std::vector<Scratch> scr(T);
    std::vector<std::array<i64, 16>> tst(T);
    for (auto& q : tst) q.fill(0);
    ndag::Info di;
    const char* dmp = getenv("T_RR_DUMP");
    struct DR { i64 pops; float dur, gain; short kind; };
    std::vector<DR> drec(dmp ? cand.size() : 0);
''')
rep('''      Res r;
      eval(cx, g, sstr, cd, cur[s], scr[t], r, true);
      if (r.kind == 3) { cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; }
      for (int k = 0; k < 16; ++k) tst[t][k] += r.st[k];
    }, di);
''', '''      Res r;
      double t0_ = dmp ? ndag::now_s() : 0.0;
      eval(cx, g, sstr, cd, cur[s], scr[t], r, true);
      if (dmp) drec[j] = DR{r.pops, (float)(ndag::now_s() - t0_), (float)r.gain, (short)r.kind};
      if (r.kind == 3) { cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; }
      for (int k = 0; k < 16; ++k) tst[t][k] += r.st[k];
    }, di);
    if (dmp) {
      FILE* f_ = fopen(dmp, "a");
      if (f_) {
        fprintf(f_, "# call nc %d run_s %.4f busy_s %.4f crit_s %.4f levels %ld edges %ld\\n", (int)cand.size(), di.run_s, di.busy_s, di.crit_s, (long)di.levels, (long)di.edges);
        for (size_t j = 0; j < cand.size(); ++j) fprintf(f_, "%zu %.5g %d %d %ld %.6f %d %.6g %d\\n", j, -cand[j].nsc, half_of(cand[j]), cand[j].z, (long)drec[j].pops, drec[j].dur, drec[j].kind, drec[j].gain, cand[j].slot);
        fclose(f_);
      }
    }
''')
open(os.path.join(HERE, "t_rr.cc"), "w").write(s)
print("t_rr.cc", len(s), "astar_t.inc", len(a))
