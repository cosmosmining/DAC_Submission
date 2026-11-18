#!/usr/bin/env python3
"""make_n_rr.py: writes tools/best5/n_rr.cc from tools/mid5/mid5_rr.cc (read only): the overflow stage's ranked
reroute with four new orders of work, chosen by `policy`:
  0 mid5's speculative loop with redo (same route as the serial loop)
  1 snapshot batches: searched on T threads against one demand map; rank-order commit of every reroute (only a second run of one net in the batch is dropped)
  2 snapshot batches: rank-order commit, a run whose read cells changed in the batch is dropped (no redo)
  3 snapshot batches: rank-order commit, each reroute re-tested on the committed map (marginal cost, no search), dropped if it no longer pays
  4 task graph (n_dag.h): a run waits for the earlier runs that share cells; same result as the serial loop, one shared map
  5 sliding snapshot: run j is searched on the state after the commits of runs <= j - batch (no barrier: a long search holds nobody back until the others
    are `batch` runs ahead), then committed in rank order after a re-test on the committed map, dropped if it no longer pays or its net changed meanwhile.
    Deterministic: what a run sees and the commit order do not depend on thread timing.
Run: python3 tools/best5/make_n_rr.py; then build_n.sh."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
s = (ROOT / "tools/mid5/mid5_rr.cc").read_text()
def rep(a, b, n=1):
    global s
    assert s.count(a) >= 1 and (n == 0 or s.count(a) == n), a
    s = s.replace(a, b)
rep('#include "astar_tl.inc"', '#include "../mid5/astar_tl.inc"')
rep('namespace mrr {\nusing namespace rrn;', 'static bool NREC = true;\nnamespace mrr {\nusing namespace rrn;')
rep('atl::g_rd.clear(); atl::g_rd_on = true;', 'if (NREC) { atl::g_rd.clear(); atl::g_rd_on = true; }')
rep('atl::g_rd_on = false; r.rd.insert(r.rd.end(), atl::g_rd.begin(), atl::g_rd.end());', 'if (NREC) { atl::g_rd_on = false; r.rd.insert(r.rd.end(), atl::g_rd.begin(), atl::g_rd.end()); }')
rep('#include <atomic>\n', '#include <atomic>\n#include <memory>\n#include <chrono>\n#include "n_dag.h"\n')
rep('int nthr, int batch, int64_t* info) {', 'int nthr, int batch, int64_t* info, int policy, int blk_shift, double* dinfo) {')
rep('int mid5_rr_rank(int L', 'int n_rr_rank(int L')
rep('int64_t mid5_rr_nseg()', 'int64_t n_rr_nseg()')
rep('void mid5_rr_get(', 'void n_rr_get(')
rep('''  bool integral = true;
  for (int z = 0; z < L && integral; ++z)''', '''  bool integral = true;
  for (int z = 0; z < L && integral && policy == 0; ++z)''')
rep('  if (!integral || nthr <= 1)\n', '  if (policy == 0 && (!integral || nthr <= 1))\n')
rep('  for (int k = 0; k < 8; ++k) info[k] = 0;\n', '  for (int k = 0; k < 8; ++k) info[k] = 0;\n  for (int k = 0; k < 8; ++k) dinfo[k] = 0;\n  double t_all0 = ndag::now_s();\n')
# the set-up on threads
SETUP = r'''
  // ---- speed9: the loop below on threads (policy >= 1), same candidates after the sort
  if (policy >= 1) {
    const int T0 = std::max(1, nthr);
    auto par_for = [&](i64 n_items, auto fn) {
      std::atomic<i64> nx(0);
      auto w_ = [&](int t) { for (;;) { i64 k = nx.fetch_add(64); if (k >= n_items) return; for (i64 q = k; q < std::min(n_items, k + 64); ++q) fn(q, t); } };
      std::vector<std::thread> th_; for (int t = 0; t + 1 < T0; ++t) th_.emplace_back(w_, t); w_(T0 - 1); for (auto& x : th_) x.join();
    };
    std::vector<std::vector<uint64_t>> tkeys(T0);
    std::vector<i64> tbig(T0, 0);
    par_for((i64)n, [&](i64 s_, int t) {
      int s = (int)s_;
      NetSt& ns = cur[s];
      to_wires_vias(segs + 6 * seg_off[s], seg_off[s + 1] - seg_off[s], dirs, ns.w, ns.v);
      i64 tot = (i64)ns.v.size();
      for (auto& q : ns.w) tot += (i64)q.iv.size();
      if (tot > BIG_NET) { tbig[t]++; ns.w.clear(); ns.v = PySet(); return; }
      live[s] = 1;
      for (auto& q : ns.w) if (q.z >= 1 && q.z < L) tkeys[t].push_back(tk(q.z, q.f));
    });
    for (auto v : tbig) stats[0] += v;
    std::unordered_map<uint64_t, std::pair<std::vector<double>, std::vector<int32_t>>> pm;
    { i64 tot = 0; for (auto& v : tkeys) tot += (i64)v.size(); pm.reserve((size_t)tot); }
    std::vector<std::pair<std::pair<int, int>, std::pair<std::vector<double>, std::vector<int32_t>>*>> plist;
    for (int s = 0; s < n; ++s) if (live[s]) for (auto& q : cur[s].w) if (q.z >= 1 && q.z < L) {
      auto ins = pm.emplace(tk(q.z, q.f), std::pair<std::vector<double>, std::vector<int32_t>>());
      if (ins.second) plist.push_back({{q.z, q.f}, &ins.first->second});
    }
    par_for((i64)plist.size(), [&](i64 k, int) {
      int z = plist[k].first.first, f = plist[k].first.second;
      auto& pr = *plist[k].second;
      bool h = dirs[z] == 0;
      int n_ = h ? X : Y;
      pr.first.assign(n_ + 1, 0.0); pr.second.assign(n_ + 1, 0);
      double uz = u[z];
      for (int t = 0; t < n_; ++t) {
        int x = h ? t : f, y = h ? f : t;
        double dd_ = g.D(z, x, y), cc = g.C(z, x, y);
        bool o = ovc(z, x, y);
        double m = o ? (cell_cost(uz, dd_, cc) - cell_cost(uz, std::max(dd_ - 2.0, 0.0), cc)) : 0.0;   // (...) * 0.0 == 0.0: no exp where the mask is off
        pr.first[t + 1] = pr.first[t] + m; pr.second[t + 1] = pr.second[t] + (o ? 1 : 0);
      }
    });
    std::vector<std::vector<Cand>> tc(T0);
    std::vector<i64> thm(T0, 0);
    par_for((i64)n, [&](i64 s_, int th_) {
      int s = (int)s_;
      if (!live[s]) return;
      NetSt& ns = cur[s];
      for (auto& t : ns.w) {
        if (t.z < 1 || t.z >= L) continue;
        auto& pr = pm.find(tk(t.z, t.f))->second;
        for (auto& p : t.iv) {
          int a = p.first, b = p.second;
          int k = pr.second[b] - pr.second[a];
          if (!k) continue;
          if (hmax && (b - a) / 2 + W > hmax) { thm[th_]++; continue; }
          double sc = pr.first[b] - pr.first[a];
          tc[th_].push_back(Cand{-sc, ridx[s], t.z, t.f, a, b, s});
        }
      }
    });
    for (auto v : thm) stats[1] += v;
    for (auto& v : tc) cand.insert(cand.end(), v.begin(), v.end());
  } else
'''
a0 = "  for (int s = 0; s < n; ++s) {\n    NetSt& ns = cur[s];\n    to_wires_vias("
assert s.count(a0) == 1
s = s.replace(a0, SETUP + a0)
# the batches
a = s.index("  // ---- speculative batches\n")
b = s.index("  // ---- output: rr_loop.cc lines 654-674 verbatim\n")
spec = s[a:b]
new = r'''  // ---- speed9 orders of work
  dinfo[0] = ndag::now_s() - t_all0;
  if (policy == 4) {                            // task graph: one shared map, a run waits for the earlier runs on its blocks
    NREC = false;
    const int T = std::max(1, nthr);
    const int sh = std::max(0, blk_shift);
    const int BX = ((X + 2) >> sh) + 1, BY = ((Y + 2) >> sh) + 1;
    struct Rc { int x0, x1, y0, y1; };
    std::vector<std::vector<Rc>> rects(n);
    std::vector<char> rects_set(n, 0);
    auto add_rc = [&](const Rc& q, std::vector<int>& out) {
      int bx0 = (std::max(q.x0, -1) + 1) >> sh, bx1 = (std::min(q.x1, X) + 1) >> sh, by0 = (std::max(q.y0, -1) + 1) >> sh, by1 = (std::min(q.y1, Y) + 1) >> sh;
      for (int bx = bx0; bx <= bx1; ++bx) for (int by = by0; by <= by1; ++by) out.push_back(bx * BY + by);
    };
    auto fp = [&](int j, std::vector<int>& out) {
      const Cand& cd = cand[j]; int s = cd.slot;
      if (!rects_set[s]) {                      // the net's cells (+-1: via cells) at the start
        rects_set[s] = 1;
        for (const Track& t : cur[s].w) for (auto& p : t.iv) {
          if (dirs[t.z] == 0) rects[s].push_back({p.first - 1, p.second + 1, t.f - 1, t.f + 1}); else rects[s].push_back({t.f - 1, t.f + 1, p.first - 1, p.second + 1});
        }
        for (const SE& e : cur[s].v.t) if (e.k) rects[s].push_back({e.x - 1, e.x + 1, e.y - 1, e.y + 1});
      }
      bool h = dirs[cd.z] == 0;                 // the A* window of this run (+-1), kept for the later runs of the net (a new path lies inside it)
      int ax = h ? cd.a : cd.f, ay = h ? cd.f : cd.a, bx = h ? cd.b : cd.f, by = h ? cd.f : cd.b;
      int x0 = (ax + bx) / 2, y0 = (ay + by) / 2, half = std::max(std::abs(ax - bx), std::abs(ay - by)) / 2 + W;
      rects[s].push_back({x0 - half - 1, x0 + half + 1, y0 - half - 1, y0 + half + 1});
      for (const Rc& q : rects[s]) add_rc(q, out);
    };
    Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W};
    std::vector<Scratch> scr(T);
    std::vector<std::array<i64, 16>> tst(T);
    for (auto& q : tst) q.fill(0);
    ndag::Info di;
    ndag::run((int)cand.size(), BX * BY, T, fp, [&](int j, int t) {
      const Cand& cd = cand[j]; int s = cd.slot;
      Res r;
      eval(cx, g, sstr, cd, cur[s], scr[t], r, true);
      if (r.kind == 3) { cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; }
      for (int k = 0; k < 16; ++k) tst[t][k] += r.st[k];
    }, di);
    for (auto& q : tst) for (int k = 0; k < 16; ++k) stats[k] += q[k];
    info[0] = 4; info[1] = di.levels; info[2] = di.edges; info[3] = di.tasks;
    dinfo[1] = di.build_s; dinfo[2] = di.run_s; dinfo[3] = di.busy_s; dinfo[4] = di.crit_s;
  } else if (policy == 5) {                     // sliding snapshot, no barrier, no redo
    NREC = false;
    Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W};
    const int T = std::max(1, nthr);
    // private maps in the master's memory order (one memcpy; the layers of la.d are x-inner on the hosts)
    const bool mem_c = strides[0] == (i64)X * Y && ((strides[1] == 1 && strides[2] == X) || (strides[1] == Y && strides[2] == 1));
    const i64 PZ = mem_c ? strides[0] : (i64)X * Y, PX = mem_c ? strides[1] : Y, PY = mem_c ? strides[2] : 1;
    std::vector<std::unique_ptr<double[]>> priv(T);
    {
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
        priv[t].reset(new double[(size_t)L * X * Y]);
        double* p = priv[t].get();
        if (mem_c) std::memcpy(p, d, sizeof(double) * (size_t)L * X * Y);
        else for (int z = 0; z < L; ++z) for (int x = 0; x < X; ++x) for (int y = 0; y < Y; ++y)
          p[z * PZ + x * PX + y * PY] = d[z * strides[0] + x * strides[1] + y * strides[2]];
      });
      for (auto& t : th) t.join();
    }
    dinfo[1] = ndag::now_s() - t_all0;
    std::vector<Grid> pg(T);
    std::vector<std::array<int64_t, 6>> pstr(T);
    for (int t = 0; t < T; ++t) {
      pg[t] = Grid{L, X, Y, priv[t].get(), c, u, dirs, PZ, PX, PY, strides[3], strides[4], strides[5]};
      pstr[t] = {PZ, PX, PY, strides[6], strides[7], strides[8]};
    }
    std::vector<Scratch> scr(T);
    const size_t NCd = cand.size();
    const long WW = std::max(1, batch);
    struct Dl { long j; std::shared_ptr<const NetSt> o, nw; };
    std::vector<Dl> lg(NCd);
    std::atomic<size_t> nlog(0), done(0), nxt(0);
    std::vector<std::shared_ptr<const NetSt>> st(n);
    for (const Cand& cd : cand) if (!st[cd.slot]) st[cd.slot] = std::make_shared<const NetSt>(cur[cd.slot]);
    std::vector<Res> res(NCd);
    std::unique_ptr<std::atomic<int>[]> ready(new std::atomic<int>[NCd]);
    for (size_t q = 0; q < NCd; ++q) ready[q].store(0);
    std::vector<long> lasts(n, -1);
    auto wait_a_bit = [](int& spins) { if (++spins < 200) std::this_thread::yield(); else std::this_thread::sleep_for(std::chrono::microseconds(50)); };
    auto worker = [&](int t) {
      size_t seen = 0;
      for (;;) {
        size_t j = nxt.fetch_add(1);
        if (j >= NCd) return;
        long lim = (long)j - WW;                 // this run's view: the commits of runs <= lim
        if (lim >= 0) { int sp = 0; while (done.load(std::memory_order_acquire) < (size_t)lim + 1) wait_a_bit(sp); }
        size_t cn = nlog.load(std::memory_order_acquire);
        for (; seen < cn && lg[seen].j <= lim; ++seen) { const Dl& q = lg[seen]; apply(pg[t], q.o->w, q.o->v, -1); apply(pg[t], q.nw->w, q.nw->v, +1); }
        const Cand& cd = cand[j];
        std::shared_ptr<const NetSt> ns = std::atomic_load(&st[cd.slot]);
        eval(cx, pg[t], pstr[t].data(), cd, *ns, scr[t], res[j], false);
        ready[j].store(1, std::memory_order_release);
      }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(worker, t);
    i64 dropped = 0, drop_net = 0;
    for (size_t j = 0; j < NCd; ++j) {
      { int sp = 0; while (!ready[j].load(std::memory_order_acquire)) wait_a_bit(sp); }
      const Cand& cd = cand[j]; int s = cd.slot;
      Res& r = res[j];
      if (r.kind == 3) {
        bool drop = false;
        if (lasts[s] > (long)j - WW) { drop = true; drop_net++; }
        else {
          apply(g, cur[s].w, cur[s].v, -1);
          double old = marginal(g, cur[s].w, cur[s].v), nw = marginal(g, r.w2, r.v2);
          if (nw < old - 1e-9) apply(g, r.w2, r.v2, +1);
          else { apply(g, cur[s].w, cur[s].v, +1); drop = true; }
        }
        if (drop) { dropped++; r.st[9]--; r.st[10]++; }
        else {
          Dl dl; dl.j = (long)j; dl.o = std::atomic_load(&st[s]);
          cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2);
          dl.nw = std::make_shared<const NetSt>(cur[s]);
          std::atomic_store(&st[s], dl.nw);
          changed[s] = 1; lasts[s] = (long)j;
          size_t cn = nlog.load(std::memory_order_relaxed);
          lg[cn] = std::move(dl);
          nlog.store(cn + 1, std::memory_order_release);
        }
      }
      for (int k = 0; k < 16; ++k) stats[k] += r.st[k];
      r = Res();
      done.store(j + 1, std::memory_order_release);
    }
    for (auto& x : th) x.join();
    info[0] = 5; info[2] = dropped; info[3] = drop_net;
  } else if (policy >= 1 && policy <= 3) {      // snapshot batches, no redo
    NREC = policy == 2;
    Ctx cx{L, X, Y, c, cs, u, dirs, strides[3], strides[4], strides[5], strides[6], strides[7], strides[8], pin_off, ap_off, aps, W};
    const int T = std::max(1, nthr);
    // private maps in the master's memory order (one memcpy; the layers of la.d are x-inner on the hosts)
    const bool mem_c = strides[0] == (i64)X * Y && ((strides[1] == 1 && strides[2] == X) || (strides[1] == Y && strides[2] == 1));
    const i64 PZ = mem_c ? strides[0] : (i64)X * Y, PX = mem_c ? strides[1] : Y, PY = mem_c ? strides[2] : 1;
    std::vector<std::unique_ptr<double[]>> priv(T);
    {
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
        priv[t].reset(new double[(size_t)L * X * Y]);
        double* p = priv[t].get();
        if (mem_c) std::memcpy(p, d, sizeof(double) * (size_t)L * X * Y);
        else for (int z = 0; z < L; ++z) for (int x = 0; x < X; ++x) for (int y = 0; y < Y; ++y)
          p[z * PZ + x * PX + y * PY] = d[z * strides[0] + x * strides[1] + y * strides[2]];
      });
      for (auto& t : th) t.join();
    }
    std::vector<Grid> pg(T);
    std::vector<std::array<int64_t, 6>> pstr(T);
    for (int t = 0; t < T; ++t) {
      pg[t] = Grid{L, X, Y, priv[t].get(), c, u, dirs, PZ, PX, PY, strides[3], strides[4], strides[5]};
      pstr[t] = {PZ, PX, PY, strides[6], strides[7], strides[8]};
    }
    std::vector<Scratch> scr(T);
    const int Bsz = std::max(1, batch);
    std::vector<Res> res(Bsz);
    std::vector<uint32_t> dirty((size_t)(X + 2) * (Y + 2), 0);
    uint32_t stamp = 0;
    auto mark = [&](int x, int y) { if (x >= -1 && y >= -1 && x <= X && y <= Y) dirty[(size_t)(x + 1) * (Y + 2) + (y + 1)] = stamp; };
    auto isd = [&](int x, int y) { return x >= -1 && y >= -1 && x <= X && y <= Y && dirty[(size_t)(x + 1) * (Y + 2) + (y + 1)] == stamp; };
    std::vector<uint32_t> slot_stamp(n, 0);
    struct Chg { int s; Wires w; PySet v; Wires w2; PySet v2; };
    std::vector<Chg> chg;
    i64 nc = (i64)cand.size();
    i64 dropped = 0, drop_net = 0;
    ndag::Pool pool(T);
    for (i64 b0 = 0; b0 < nc; b0 += Bsz) {
      i64 b1 = std::min(nc, b0 + Bsz), nb = b1 - b0;
      info[1]++;
      for (i64 j = 0; j < nb; ++j) res[j] = Res();
      std::atomic<i64> next(0);
      auto wk = [&](int t) {
        while (true) {
          i64 j = next.fetch_add(1);
          if (j >= nb) break;
          const Cand& cd = cand[b0 + j];
          eval(cx, pg[t], pstr[t].data(), cd, cur[cd.slot], scr[t], res[j], false);
        }
      };
      pool.run(wk);
      ++stamp; chg.clear();
      for (i64 j = 0; j < nb; ++j) {
        const Cand& cd = cand[b0 + j];
        int s = cd.slot;
        Res& r = res[j];
        if (r.kind == 3) {
          bool drop = false;
          if (slot_stamp[s] == stamp) { drop = true; drop_net++; }   // a second run of a net changed in this batch: its route is gone
          if (!drop && policy == 2) {
            net_cells(cur[s].w, cur[s].v, dirs, [&](int x, int y) { if (!drop && isd(x, y)) drop = true; });
            for (size_t q = 0; q + 1 < r.rd.size() && !drop; q += 2) if (isd(r.rd[q], r.rd[q + 1])) drop = true;
          }
          if (!drop && policy == 3) {
            apply(g, cur[s].w, cur[s].v, -1);
            double old = marginal(g, cur[s].w, cur[s].v), nw = marginal(g, r.w2, r.v2);
            if (nw < old - 1e-9) apply(g, r.w2, r.v2, +1);
            else { apply(g, cur[s].w, cur[s].v, +1); drop = true; }
          } else if (!drop) { apply(g, cur[s].w, cur[s].v, -1); apply(g, r.w2, r.v2, +1); }
          if (drop) { dropped++; r.st[9]--; r.st[10]++; }
          else {
            chg.push_back(Chg{s, cur[s].w, cur[s].v, r.w2, r.v2});
            if (policy == 2) { net_cells(cur[s].w, cur[s].v, dirs, mark); net_cells(r.w2, r.v2, dirs, mark); }
            cur[s].w = std::move(r.w2); cur[s].v = std::move(r.v2); changed[s] = 1; slot_stamp[s] = stamp;
          }
        }
        for (int k = 0; k < 16; ++k) stats[k] += r.st[k];
      }
      if (!chg.empty()) {
        auto ap1 = [&](int t) { for (auto& ch : chg) { apply(pg[t], ch.w, ch.v, -1); apply(pg[t], ch.w2, ch.v2, +1); } };
        pool.run(ap1);
      }
    }
    info[0] = policy; info[2] = dropped; info[3] = drop_net;
  } else {
    NREC = true;
''' + spec.replace("  // ---- speculative batches\n", "") + "  }\n  dinfo[5] = ndag::now_s() - t_all0;\n"
s = s[:a] + new + s[b:]
s = s.replace("// mid5_rr.cc:", "// n_rr.cc: GENERATED by tools/best5/make_n_rr.py from tools/mid5/mid5_rr.cc.\n// mid5_rr.cc:", 1)
(ROOT / "tools/best5/n_rr.cc").write_text(s)
print("wrote n_rr.cc", len(s))
