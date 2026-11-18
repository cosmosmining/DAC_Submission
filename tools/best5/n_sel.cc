// n_sel.cc: reshape's net selection (tools/late5/l5_kern.cc l5_selc_pairs) on T threads, same pairs (sorted unique keys) and
// the same statics for l5_selc_cells / l5_selc_greedy; the overflow mask on threads. Rows: the base wire rows without the changed nets + the changed nets' wire rows.
// Build: tools/best5/build_n.sh.
#include "../late5/l5_kern.cc"
#include <atomic>
extern "C" {
// ov[L*X*Y] = official overflow mask (ovf_lib.ov_masks: blk | ovr, layer 0 off) of C-contiguous d, c.
void n_ov_mask(int64_t L, int64_t X, int64_t Y, const double* d, const int64_t* sd, const double* c, const int64_t* sc, uint8_t* ov, int T) {
  const int64_t N0 = X * Y;
  std::vector<std::thread> th;
  for (int k = 0; k < N0; ++k) ov[k] = 0;
  for (int t = 0; t < T; ++t) th.emplace_back([=]() {
    int64_t N = (L - 1) * X;
    for (int64_t q = N * t / T; q < N * (t + 1) / T; ++q) {
      int64_t z = 1 + q / X, x = q % X;
      const double* dp = d + z * sd[0] + x * sd[1]; const double* cp = c + z * sc[0] + x * sc[1];
      uint8_t* o = ov + (z * X + x) * Y;
      for (int64_t y = 0; y < Y; ++y) {
        double dd = dp[y * sd[2]], cc = cp[y * sc[2]];
        o[y] = (cc <= 1e-3 && dd > 0) || (cc > 1e-3 && dd > 2.0 * cc);
      }
    }
  });
  for (auto& x : th) x.join();
}
// flat positions z*X*Y + x*Y + y of the overflowed cells (blocked first flag): counts in cnt[0..1]; lists kept until n_ov_get
static std::vector<int64_t> OVB, OVO;
void n_ov_cells(int64_t L, int64_t X, int64_t Y, const double* d, const int64_t* sd, const double* c, const int64_t* sc, int64_t* cnt, int T) {
  std::vector<std::vector<int64_t>> tb(T), to(T);
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
    int64_t N = (L - 1) * X;
    for (int64_t q = N * t / T; q < N * (t + 1) / T; ++q) {
      int64_t z = 1 + q / X, x = q % X;
      const double* dp = d + z * sd[0] + x * sd[1]; const double* cp = c + z * sc[0] + x * sc[1];
      int64_t b0 = (z * X + x) * Y;
      for (int64_t y = 0; y < Y; ++y) {
        double dd = dp[y * sd[2]], cc = cp[y * sc[2]];
        if (cc <= 1e-3) { if (dd > 0) tb[t].push_back(b0 + y); }
        else if (dd > 2.0 * cc) to[t].push_back(b0 + y);
      }
    }
  });
  for (auto& x : th) x.join();
  OVB.clear(); OVO.clear();
  for (auto& v : tb) OVB.insert(OVB.end(), v.begin(), v.end());
  for (auto& v : to) OVO.insert(OVO.end(), v.begin(), v.end());
  cnt[0] = (int64_t)OVB.size(); cnt[1] = (int64_t)OVO.size();
}
void n_ov_get(int64_t* b, int64_t* o) { std::copy(OVB.begin(), OVB.end(), b); std::copy(OVO.begin(), OVO.end(), o); }
// changed nets: ids[me] route nets, ptr[me] addresses of int64 [cnt, 6] blocks (x1, y1, z1, x2, y2, z2), wire rows only used
int n_selc_pairs(int64_t L, int64_t X, int64_t Y, const uint8_t* ov, const int64_t* dirs, const int64_t* base, int64_t mb, const uint8_t* chm,
                 const int64_t* ids, const int64_t* ptr, const int64_t* cnt, int64_t me,
                 const double* slack_r, const int64_t* npins_r, int64_t nnet, double guard, int64_t maxp, int64_t* out, int T) {
  const int64_t N = L * X * Y;
  std::vector<std::vector<int64_t>> tk(T);
  std::atomic<int> bad(0);
  auto walk = [&](int t) {
    std::vector<int64_t>& keys = tk[t];
    auto row = [&](int64_t n, const int64_t* r) {
      int64_t x1 = r[0], y1 = r[1], z1 = r[2], x2 = r[3], y2 = r[4], z2 = r[5];
      if (z1 != z2) return;                                   // wire rows only (select_c: rows with z1 == z2)
      int64_t zc = std::min(std::max(z1, (int64_t)0), L - 1);
      bool hor = dirs[zc] == 0 && y1 == y2 && x1 != x2 && z1 > 0;
      bool ver = dirs[zc] == 1 && x1 == x2 && y1 != y2 && z1 > 0;
      if (!hor && !ver) return;
      if (n < 0 || n >= nnet || z1 >= L) { bad = 1; return; }
      if (hor) {
        int64_t lo = std::min(x1, x2), hi = std::max(x1, x2);
        if (lo < 0 || hi > X || y1 < 0 || y1 >= Y) { bad = 1; return; }
        for (int64_t q = lo; q < hi; q++) { int64_t lin = (z1 * X + q) * Y + y1; if (ov[lin]) keys.push_back(n * N + lin); }
      } else {
        int64_t lo = std::min(y1, y2), hi = std::max(y1, y2);
        if (lo < 0 || hi > Y || x1 < 0 || x1 >= X) { bad = 1; return; }
        for (int64_t q = lo; q < hi; q++) { int64_t lin = (z1 * X + x1) * Y + q; if (ov[lin]) keys.push_back(n * N + lin); }
      }
    };
    for (int64_t i = mb * t / T; i < mb * (t + 1) / T; i++) { const int64_t* r = base + 7 * i; if (!chm[r[0]]) row(r[0], r + 1); }
    for (int64_t i = me * t / T; i < me * (t + 1) / T; i++) { const int64_t* q = (const int64_t*)ptr[i]; for (int64_t k = 0; k < cnt[i]; ++k) row(ids[i], q + 6 * k); }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(walk, t); walk(T - 1); for (auto& x : th) x.join(); }
  if (bad) return 1;
  // merge the sorted lists two by two on threads
  while (tk.size() > 1) {
    std::vector<std::vector<int64_t>> nx((tk.size() + 1) / 2);
    std::vector<std::thread> th;
    for (size_t k = 0; k < nx.size(); ++k) th.emplace_back([&, k]() {
      if (2 * k + 1 < tk.size()) {
        nx[k].resize(tk[2 * k].size() + tk[2 * k + 1].size());
        auto e = std::set_union(tk[2 * k].begin(), tk[2 * k].end(), tk[2 * k + 1].begin(), tk[2 * k + 1].end(), nx[k].begin());
        nx[k].resize(e - nx[k].begin());
      } else nx[k] = std::move(tk[2 * k]);
    });
    for (auto& x : th) x.join();
    tk = std::move(nx);
  }
  const std::vector<int64_t>& keys = tk[0];
  // l5_selc_pairs' tail, unchanged
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
}
