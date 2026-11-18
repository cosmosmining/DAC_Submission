// n_b2.cc: the per-net loop of tools/onepass/build2_fast.py build_fast2 (used_access_points, nodes_from_aps, segments_to_polys, tree_connects, to_wires_vias, the runs and
// their current layer, the net's point table) on T threads, with the exact ports of tools/par7/p7_pf.cc (prep_net: same iteration orders as the Python). Build: tools/best5/build_n.sh.
#include "../par7/p7_pf.cc"
#include <atomic>
namespace {
struct B2Out { int ok = 0; std::vector<std::array<int64_t, 4>> nodes, pts; std::vector<std::array<int64_t, 7>> runs; };   // run: d, fixed, lo, hi, cur, p0, p1 (net-local)
std::vector<B2Out> BO;
}
extern "C" {
// returns 0, or 1 when a net needs the Python loop (no pins, empty access points, layer out of range). counts: kept nets, runs, points, nodes.
int n_b2_run(int64_t nn, const int64_t* net_i, const int64_t* net_j, const int64_t* off, const int64_t* seg, const int64_t* pin_off, const int64_t* ap_off, const int32_t* aps, const int32_t* dirs,
             const int64_t* H, int nH, const int64_t* V, int nV, int L, int T, int64_t* counts) {
  BO.assign(nn, B2Out());
  int64_t npin = 0;
  for (int64_t n = 0; n < nn; ++n) npin = std::max(npin, pin_off[net_j[n] + 1]);
  std::vector<int64_t> ap_cnt(npin);
  for (int64_t p = 0; p < npin; ++p) ap_cnt[p] = ap_off[p + 1] - ap_off[p];
  In in{dirs, net_i, net_j, pin_off, pin_off + 1, ap_off, ap_cnt.data(), aps, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::atomic<int64_t> nx(0); std::atomic<int> bad(0);
  auto work = [&]() {
    std::vector<Poly> polys;
    for (;;) {
      int64_t n = nx.fetch_add(1);
      if (n >= nn) return;
      B2Out& o = BO[n];
      int64_t i = net_i[n];
      const Seg* s = (const Seg*)(seg + 6 * off[i]); long ns = (long)(off[i + 1] - off[i]);
      Pre pr;
      prep_net(in, n, s, ns, pr);
      if (pr.status != 1) { bad = 1; continue; }
      segments_to_polys(s, ns, pr.nodes, polys);
      if (polys.empty() || !tree_connects(pr.nodes, polys)) continue;
      o.ok = 1;
      for (auto& nd : pr.nodes.v) o.nodes.push_back({nd.x, nd.y, nd.lo, nd.hi});
      std::unordered_map<uint64_t, int> pid;
      for (auto& p : polys) {
        int64_t x0 = p.x[0], y0 = p.y[0], x1 = p.x[p.n - 1], y1 = p.y[p.n - 1];
        int d = y0 == y1 ? 0 : 1;
        int64_t fixed = d == 0 ? y0 : x0;
        int64_t lo = d == 0 ? std::min(x0, x1) : std::min(y0, y1), hi = d == 0 ? std::max(x0, x1) : std::max(y0, y1);
        const int64_t* cands = d == 0 ? H : V; int nc = d == 0 ? nH : nV;
        int64_t cur = 0;
        for (int k = 0; k < nc; ++k) {
          auto it = pr.ow.idx.find(zkey((int)cands[k], fixed));
          bool hit = false;
          if (it != pr.ow.idx.end()) for (auto& ab : pr.ow.ivs[it->second]) if (ab.first <= lo && hi <= ab.second) { hit = true; break; }
          if (hit) { cur = k; break; }
        }
        int64_t ids[2]; int64_t xs[2] = {x0, x1}, ys[2] = {y0, y1};
        for (int q = 0; q < 2; ++q) {
          uint64_t kk = pkey(xs[q], ys[q]);
          auto f = pid.find(kk);
          if (f == pid.end()) {
            int id = (int)o.pts.size(); pid.emplace(kk, id);
            const Node* nd = pr.nodes.get(xs[q], ys[q]);
            o.pts.push_back({xs[q], ys[q], nd ? (int64_t)nd->lo : (int64_t)L, nd ? (int64_t)nd->hi : (int64_t)-1});
            ids[q] = id;
          } else ids[q] = f->second;
        }
        o.runs.push_back({d, fixed, lo, hi, cur, ids[0], ids[1]});
      }
    }
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join(); }
  if (bad) return 1;
  int64_t k = 0, r = 0, p = 0, m = 0;
  for (auto& o : BO) if (o.ok) { k++; r += (int64_t)o.runs.size(); p += (int64_t)o.pts.size(); m += (int64_t)o.nodes.size(); }
  counts[0] = k; counts[1] = r; counts[2] = p; counts[3] = m;
  return 0;
}
// kept[k] = index in the chosen list; node_off[k+1], nodes[m,4]; run[r,8]: n_id d fixed lo hi cur p0 p1 (global point ids); pts[p,4]: x y lo hi
void n_b2_get(int64_t* kept, int64_t* node_off, int64_t* nodes, int64_t* run, int64_t* pts) {
  int64_t k = 0, r = 0, p = 0, m = 0;
  node_off[0] = 0;
  for (size_t n = 0; n < BO.size(); ++n) {
    B2Out& o = BO[n];
    if (!o.ok) continue;
    kept[k] = (int64_t)n;
    for (auto& v : o.nodes) { for (int q = 0; q < 4; ++q) nodes[4 * m + q] = v[q]; m++; }
    node_off[k + 1] = m;
    for (auto& v : o.runs) {
      int64_t* w = run + 8 * r;
      w[0] = k; w[1] = v[0]; w[2] = v[1]; w[3] = v[2]; w[4] = v[3]; w[5] = v[4]; w[6] = p + v[5]; w[7] = p + v[6]; r++;
    }
    for (auto& v : o.pts) { for (int q = 0; q < 4; ++q) pts[4 * (p) + q] = v[q]; p++; }
    k++;
  }
  std::vector<B2Out>().swap(BO);
}
}
