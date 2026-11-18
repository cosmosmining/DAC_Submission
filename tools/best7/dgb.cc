// dgb.cc: the descent's problem from the chain's rows, on threads: tools/best5/n_b2.cc's per-net loop (runs, points, nodes: the exact ports of tools/par7/p7_pf.cc) with dtla0's strict filter,
// the cell and via tables of tools/e2et/prep.py (dtla_via.add_vias) as the CSR arrays of dg_kern.cu, the wire segment -> run map (tools/e2et/descend.py seg_run_map), the rows of a choice (dtla0.net_rows).
// Build: g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared dgb.cc -o libdgb.so
#include "../par7/p7_pf.cc"
#include <atomic>
#include <cmath>
namespace {
struct BOut { int ok = 0; std::vector<std::array<int64_t, 4>> nodes, pts; std::vector<std::array<int64_t, 7>> runs; };
std::vector<BOut> BO;
std::vector<int32_t> CID;       // dense key -> cell id; -1 free
std::vector<int64_t> CKEY;      // cell id -> key
template <class F> void par_for(int T, int64_t n, int64_t chunk, F f) {
  std::atomic<int64_t> nx(0);
  auto w = [&]() { for (;;) { int64_t a = nx.fetch_add(chunk); if (a >= n) return; int64_t b = std::min(n, a + chunk); for (int64_t i = a; i < b; ++i) f(i); } };
  std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(w); w(); for (auto& x : th) x.join();
}
}
extern "C" {
// nets n = 0..nn-1: rows [noff[n], noff[n+1]) of rows6; net_j = index into pin_start / pin_end. counts: kept nets, runs, points, nodes, strict-held nets, not connected
int dgb_nets(int64_t nn, const int64_t* net_j, const int64_t* noff, const int64_t* rows6, const int64_t* pin_start, const int64_t* pin_end, const int64_t* ap_start, const int64_t* ap_cnt, const int32_t* ap, const int32_t* dirs,
             const int64_t* H, int nH, const int64_t* V, int nV, int L, int T, int64_t* counts) {
  BO.assign(nn, BOut());
  std::vector<int64_t> net_i(nn); for (int64_t n = 0; n < nn; ++n) net_i[n] = n;
  In in{dirs, net_i.data(), net_j, pin_start, pin_end, ap_start, ap_cnt, ap, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::atomic<int64_t> nx(0), nstrict(0), nconn(0);
  auto work = [&]() {
    std::vector<Poly> polys;
    for (;;) {
      int64_t n = nx.fetch_add(1);
      if (n >= nn) return;
      BOut& o = BO[n];
      const Seg* s = (const Seg*)(rows6 + 6 * noff[n]); long ns = (long)(noff[n + 1] - noff[n]);
      Pre pr;
      prep_net(in, n, s, ns, pr);
      if (pr.status != 1) { nconn++; continue; }
      segments_to_polys(s, ns, pr.nodes, polys);
      if (polys.empty() || !tree_connects(pr.nodes, polys)) { nconn++; continue; }
      std::unordered_map<uint64_t, int> pid;
      bool strict_ok = true;
      for (auto& p : polys) {
        int64_t x0 = p.x[0], y0 = p.y[0], x1 = p.x[p.n - 1], y1 = p.y[p.n - 1];
        int d = y0 == y1 ? 0 : 1;
        int64_t fixed = d == 0 ? y0 : x0;
        int64_t lo = d == 0 ? std::min(x0, x1) : std::min(y0, y1), hi = d == 0 ? std::max(x0, x1) : std::max(y0, y1);
        const int64_t* cands = d == 0 ? H : V; int nc = d == 0 ? nH : nV;
        int64_t cur = -1;
        for (int k = 0; k < nc && cur < 0; ++k) {
          auto it = pr.ow.idx.find(zkey((int)cands[k], fixed));
          if (it != pr.ow.idx.end()) for (auto& ab : pr.ow.ivs[it->second]) if (ab.first <= lo && hi <= ab.second) { cur = k; break; }
        }
        if (cur < 0) { strict_ok = false; break; }
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
      if (!strict_ok) { o.runs.clear(); o.pts.clear(); nstrict++; continue; }
      o.ok = 1;
      for (auto& nd : pr.nodes.v) o.nodes.push_back({nd.x, nd.y, nd.lo, nd.hi});
    }
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join(); }
  int64_t k = 0, r = 0, p = 0, m = 0;
  for (auto& o : BO) if (o.ok) { k++; r += (int64_t)o.runs.size(); p += (int64_t)o.pts.size(); m += (int64_t)o.nodes.size(); }
  counts[0] = k; counts[1] = r; counts[2] = p; counts[3] = m; counts[4] = nstrict.load(); counts[5] = nconn.load();
  return 0;
}
// kept[k] = n; node_off[k+1], nodes[m,4]; run[r,8]: k d fixed lo hi cur p0 p1 (global point ids); run_off[k+1]; pts[p,4]: x y lo hi
void dgb_get(int64_t* kept, int64_t* node_off, int64_t* nodes, int64_t* run_off, int64_t* run, int64_t* pts) {
  int64_t k = 0, r = 0, p = 0, m = 0;
  node_off[0] = 0; run_off[0] = 0;
  for (size_t n = 0; n < BO.size(); ++n) {
    BOut& o = BO[n];
    if (!o.ok) continue;
    kept[k] = (int64_t)n;
    for (auto& v : o.nodes) { for (int q = 0; q < 4; ++q) nodes[4 * m + q] = v[q]; m++; }
    node_off[k + 1] = m;
    for (auto& v : o.runs) {
      int64_t* w = run + 8 * r;
      w[0] = k; w[1] = v[0]; w[2] = v[1]; w[3] = v[2]; w[4] = v[3]; w[5] = v[4]; w[6] = p + v[5]; w[7] = p + v[6]; r++;
    }
    run_off[k + 1] = r;
    for (auto& v : o.pts) { for (int q = 0; q < 4; ++q) pts[4 * p + q] = v[q]; p++; }
    k++;
  }
  std::vector<BOut>().swap(BO);
}
// cells touched by the runs (all candidates) and the vias of the points; returns the number of cells; counts[0] wire entries, counts[1] via entries. The tables stay here until dgb_cells_get.
static int64_t C_R, C_K, C_L, C_X, C_Y, C_E, C_V, C_NC, C_NPT;
int64_t dgb_cells(int64_t R, const int64_t* run, int64_t npt, const int64_t* pts, const int32_t* cand2, int K, const int32_t* dirs, int L, int X, int Y, int T, int64_t* counts) {
  C_R = R; C_K = K; C_L = L; C_X = X; C_Y = Y; C_NPT = npt;
  size_t D = (size_t)L * X * Y;
  if (CID.size() != D) CID.assign(D, -1);
  int32_t* cid = CID.data();
  int64_t E = 0, Vv = 0;
  for (int64_t r = 0; r < R; ++r) { const int64_t* w = run + 8 * r; int d = (int)w[1]; int nc = 0; for (int k = 0; k < K; ++k) if (cand2[d * K + k] >= 0) nc++; E += nc * std::max<int64_t>(w[4] - w[3], 0); }
  par_for(T, R, 512, [&](int64_t r) {
    const int64_t* w = run + 8 * r; int d = (int)w[1]; int64_t fx = w[2], lo = w[3], hi = w[4];
    for (int k = 0; k < K; ++k) { int z = cand2[d * K + k]; if (z < 0) continue;
      for (int64_t t = lo; t < hi; ++t) { int64_t x = d == 0 ? t : fx, y = d == 0 ? fx : t; cid[((int64_t)z * X + x) * Y + y] = -2; } }
  });
  std::atomic<int64_t> nv(0);
  par_for(T, npt, 2048, [&](int64_t p) {
    int64_t px = pts[4 * p], py = pts[4 * p + 1]; int64_t cnt = 0;
    for (int z = 1; z < L; ++z) { int dz = dirs[z]; if (dz != 0 && dz != 1) continue;
      int64_t coord = dz == 0 ? px : py, size = dz == 0 ? X : Y;
      bool inner = coord > 0 && coord < size - 1;
      double wl = inner ? 1.0 : ((coord == size - 1 && coord > 0) ? 2.0 : 0.0), wr = inner ? 1.0 : ((coord == 0 && size > 1) ? 2.0 : 0.0);
      if (wl > 0) { int64_t ex = dz == 0 ? px - 1 : px, ey = dz == 0 ? py : py - 1; cid[((int64_t)z * X + ex) * Y + ey] = -2; cnt++; }
      if (wr > 0) { cid[((int64_t)z * X + px) * Y + py] = -2; cnt++; }
    }
    nv.fetch_add(cnt, std::memory_order_relaxed);
  });
  Vv = nv.load();
  // ids in key order (np.unique's order)
  int nb = std::max(T, 1); std::vector<int64_t> cn(nb + 1, 0);
  { std::vector<std::thread> th; auto w = [&](int b) { int64_t a0 = (int64_t)D * b / nb, a1 = (int64_t)D * (b + 1) / nb, c = 0; for (int64_t i = a0; i < a1; ++i) c += cid[i] == -2; cn[b + 1] = c; };
    for (int b = 1; b < nb; ++b) th.emplace_back(w, b); w(0); for (auto& x : th) x.join(); }
  for (int b = 0; b < nb; ++b) cn[b + 1] += cn[b];
  int64_t nc = cn[nb];
  CKEY.resize(nc);
  { std::vector<std::thread> th; auto w = [&](int b) { int64_t a0 = (int64_t)D * b / nb, a1 = (int64_t)D * (b + 1) / nb, c = cn[b]; for (int64_t i = a0; i < a1; ++i) if (cid[i] == -2) { cid[i] = (int32_t)c; CKEY[c] = i; c++; } };
    for (int b = 1; b < nb; ++b) th.emplace_back(w, b); w(0); for (auto& x : th) x.join(); }
  C_E = E; C_V = Vv; C_NC = nc;
  counts[0] = E; counts[1] = Vv;
  return nc;
}
// fills: ent_ptr[R*K+1], ec[E], cp[nc+1], ce[E], cvp[nc+1], cvi[V], cvg[V], base[nc], u[nc], copen[nc], pvp[npt+1], pvz[V], pvc[V], pvg[V] (RL = float, or double when rl64); d: demand map with the free nets taken out, c: capacity, stride in doubles; ul[L] per-layer unit cost
} // extern C
template <class RL> static void cells_get_t(const int64_t* run, const int64_t* pts, const int32_t* cand2, const int32_t* dirs, const double* d, int64_t s0, int64_t s1, int64_t s2, const double* cc, int64_t c0, int64_t c1, int64_t c2, const double* ul, int T,
                  int32_t* ent_ptr, int32_t* ec, int32_t* cp, int32_t* ce, int32_t* cvp, int32_t* cvi, RL* cvg, RL* base, RL* uu, unsigned char* copen, int32_t* pvp, unsigned char* pvz, int32_t* pvc, RL* pvg) {
  const int K = (int)C_K, L = (int)C_L, X = (int)C_X, Y = (int)C_Y; const int64_t R = C_R, NC = C_NC, E = C_E, NPT = C_NPT;
  const int32_t* cid = CID.data();
  // per cell
  par_for(T, NC, 4096, [&](int64_t i) {
    int64_t key = CKEY[i]; int64_t y = key % Y, xz = key / Y, x = xz % X, z = xz / X;
    double dd = d[z * s0 + x * s1 + y * s2], cv = cc[z * c0 + x * c1 + y * c2];
    RL dr = (RL)dd, cr = (RL)cv;
    bool op = cr > (RL)1e-3;
    RL al = op ? (RL)0.25 : (RL)0.75;
    base[i] = al * dr - (op ? (RL)0.5 * cr : (RL)0);
    uu[i] = (RL)ul[z]; copen[i] = op ? 1 : 0;
  });
  // wire entries
  int64_t e = 0;
  for (int64_t r = 0; r < R; ++r) {
    const int64_t* w = run + 8 * r; int d_ = (int)w[1]; int64_t n = std::max<int64_t>(w[4] - w[3], 0);
    for (int k = 0; k < K; ++k) { ent_ptr[r * K + k] = (int32_t)e; if (cand2[d_ * K + k] >= 0) e += n; }
  }
  ent_ptr[R * K] = (int32_t)e;
  par_for(T, R, 512, [&](int64_t r) {
    const int64_t* w = run + 8 * r; int d_ = (int)w[1]; int64_t fx = w[2], lo = w[3], hi = w[4];
    for (int k = 0; k < K; ++k) { int z = cand2[d_ * K + k]; if (z < 0) continue; int32_t q = ent_ptr[r * K + k];
      for (int64_t t = lo; t < hi; ++t) { int64_t x = d_ == 0 ? t : fx, y = d_ == 0 ? fx : t; ec[q++] = cid[((int64_t)z * X + x) * Y + y]; } }
  });
  // by cell
  std::fill(cp, cp + NC + 1, 0);
  for (int64_t q = 0; q < E; ++q) cp[ec[q] + 1]++;
  for (int64_t i = 0; i < NC; ++i) cp[i + 1] += cp[i];
  { std::vector<int32_t> fp(cp, cp + NC);
    for (int64_t r = 0; r < R; ++r) for (int k = 0; k < K; ++k) for (int32_t q = ent_ptr[r * K + k]; q < ent_ptr[r * K + k + 1]; ++q) ce[fp[ec[q]]++] = (int32_t)(r * K + k); }
  // via entries, in point order
  RL gw[2][2];                      // [open][w - 1]
  gw[1][0] = (RL)std::expm1((double)((RL)0.25 * (RL)1)); gw[1][1] = (RL)std::expm1((double)((RL)0.25 * (RL)2));
  gw[0][0] = (RL)std::expm1((double)((RL)0.75 * (RL)1)); gw[0][1] = (RL)std::expm1((double)((RL)0.75 * (RL)2));
  pvp[0] = 0;
  for (int64_t p = 0; p < NPT; ++p) {
    int64_t px = pts[4 * p], py = pts[4 * p + 1]; int cnt = 0;
    for (int z = 1; z < L; ++z) { int dz = dirs[z]; if (dz != 0 && dz != 1) continue;
      int64_t coord = dz == 0 ? px : py, size = dz == 0 ? X : Y;
      bool inner = coord > 0 && coord < size - 1;
      cnt += (inner || (coord == size - 1 && coord > 0)) ? 1 : 0; cnt += (inner || (coord == 0 && size > 1)) ? 1 : 0; }
    pvp[p + 1] = pvp[p] + cnt;
  }
  par_for(T, NPT, 2048, [&](int64_t p) {
    int64_t px = pts[4 * p], py = pts[4 * p + 1]; int32_t q = pvp[p];
    for (int z = 1; z < L; ++z) { int dz = dirs[z]; if (dz != 0 && dz != 1) continue;
      int64_t coord = dz == 0 ? px : py, size = dz == 0 ? X : Y;
      bool inner = coord > 0 && coord < size - 1;
      int wl = inner ? 1 : ((coord == size - 1 && coord > 0) ? 2 : 0), wr = inner ? 1 : ((coord == 0 && size > 1) ? 2 : 0);
      if (wl > 0) { int64_t ex = dz == 0 ? px - 1 : px, ey = dz == 0 ? py : py - 1; int32_t c = cid[((int64_t)z * X + ex) * Y + ey]; pvz[q] = (unsigned char)z; pvc[q] = c; pvg[q] = gw[copen[c]][wl - 1]; q++; }
      if (wr > 0) { int32_t c = cid[((int64_t)z * X + px) * Y + py]; pvz[q] = (unsigned char)z; pvc[q] = c; pvg[q] = gw[copen[c]][wr - 1]; q++; }
    }
  });
  int64_t Vv = C_V;
  std::fill(cvp, cvp + NC + 1, 0);
  for (int64_t q = 0; q < Vv; ++q) cvp[pvc[q] + 1]++;
  for (int64_t i = 0; i < NC; ++i) cvp[i + 1] += cvp[i];
  { std::vector<int32_t> fp(cvp, cvp + NC);
    for (int64_t p = 0; p < NPT; ++p) for (int32_t q = pvp[p]; q < pvp[p + 1]; ++q) { int32_t s = fp[pvc[q]]++; cvi[s] = (int32_t)(p * L + pvz[q]); cvg[s] = pvg[q]; } }
  // the dense map is free again
  int32_t* cidw = CID.data();
  for (int64_t i = 0; i < NC; ++i) cidw[CKEY[i]] = -1;
}
extern "C" {
void dgb_cells_get(int rl64, const int64_t* run, const int64_t* pts, const int32_t* cand2, const int32_t* dirs, const double* d, int64_t s0, int64_t s1, int64_t s2, const double* cc, int64_t c0, int64_t c1, int64_t c2, const double* ul, int T,
                   int32_t* ent_ptr, int32_t* ec, int32_t* cp, int32_t* ce, int32_t* cvp, int32_t* cvi, void* cvg, void* base, void* uu, unsigned char* copen, int32_t* pvp, unsigned char* pvz, int32_t* pvc, void* pvg, int64_t* keys) {
  if (keys) for (int64_t i = 0; i < C_NC; ++i) keys[i] = CKEY[i];
  if (rl64) cells_get_t<double>(run, pts, cand2, dirs, d, s0, s1, s2, cc, c0, c1, c2, ul, T, ent_ptr, ec, cp, ce, cvp, cvi, (double*)cvg, (double*)base, (double*)uu, copen, pvp, pvz, pvc, (double*)pvg);
  else cells_get_t<float>(run, pts, cand2, dirs, d, s0, s1, s2, cc, c0, c1, c2, ul, T, ent_ptr, ec, cp, ce, cvp, cvi, (float*)cvg, (float*)base, (float*)uu, copen, pvp, pvz, pvc, (float*)pvg);
}
// wire segment -> run (descend.py seg_run_map): wrows[nw,6]: timer net, x1 y1 z x2 y2 of the segment; run key from run_tn (timer net), dir, fixed, current layer
int64_t dgb_map(int64_t nw, const int64_t* wrows, int64_t R, const int64_t* run_tn, const int64_t* run, const int32_t* zcur, int64_t* out, int T) {
  struct KR { uint64_t k; int32_t r; };
  std::vector<KR> kr((size_t)R);
  auto mk = [](int64_t n, int64_t d, int64_t f, int64_t z) { return ((uint64_t)n << 24) | ((uint64_t)d << 23) | ((uint64_t)f << 5) | (uint64_t)z; };
  for (int64_t r = 0; r < R; ++r) { const int64_t* w = run + 8 * r; kr[r] = {run_tn[r] >= 0 ? mk(run_tn[r], w[1], w[2], zcur[r]) : ~0ull, (int32_t)r}; }
  std::stable_sort(kr.begin(), kr.end(), [](const KR& a, const KR& b) { return a.k < b.k; });
  std::atomic<int64_t> hit(0);
  par_for(T, nw, 1024, [&](int64_t j) {
    const int64_t* s = wrows + 6 * j; int64_t n = s[0], x1 = s[1], y1 = s[2], z = s[3], x2 = s[4], y2 = s[5];
    uint64_t key; int64_t lo, hi;
    if (y1 == y2) { key = mk(n, 0, y1, z); lo = std::min(x1, x2); hi = std::max(x1, x2); } else { key = mk(n, 1, x1, z); lo = std::min(y1, y2); hi = std::max(y1, y2); }
    auto it = std::lower_bound(kr.begin(), kr.end(), key, [](const KR& a, uint64_t k) { return a.k < k; });
    int64_t o = -1;
    for (; it != kr.end() && it->k == key; ++it) { const int64_t* w = run + 8 * it->r; if (w[3] <= lo && hi <= w[4]) { o = it->r; break; } }
    out[j] = o;
    if (o >= 0) hit.fetch_add(1, std::memory_order_relaxed);
  });
  return hit.load();
}
// dtla0.net_rows for the kept nets ks (k index): rows [x y z x y z] at the choice; off[nk+1]; returns total. Two calls: rows = null counts only.
int64_t dgb_rows(int64_t nk, const int64_t* ks, const int64_t* choice, const int64_t* run_off, const int64_t* run, const int64_t* node_off, const int64_t* nodes, const int32_t* cand2, int K, const int32_t* dirs, int T, int64_t* off, int64_t* rows) {
  std::vector<std::vector<int64_t>> out((size_t)nk);
  par_for(T, nk, 64, [&](int64_t q) {
    int64_t k = ks[q]; std::vector<int64_t>& b = out[q];
    std::vector<std::array<int64_t, 3>> wg;      // (z, f, run) in the dict's insertion order of (z, f)
    std::unordered_map<uint64_t, int> order; std::vector<std::vector<std::pair<int64_t, int64_t>>> ivs; std::vector<std::pair<int, int64_t>> gkey;
    std::unordered_map<uint64_t, int> nid; std::vector<std::array<int64_t, 4>> nl;   // x y minz maxz, insertion order
    auto touch = [&](int64_t x, int64_t y, int64_t z1, int64_t z2) { uint64_t kk = pkey(x, y); auto f = nid.find(kk); if (f == nid.end()) { nid.emplace(kk, (int)nl.size()); nl.push_back({x, y, std::min(z1, z2), std::max(z1, z2)}); } else { auto& v = nl[f->second]; v[2] = std::min(v[2], std::min(z1, z2)); v[3] = std::max(v[3], std::max(z1, z2)); } };
    for (int64_t r = run_off[k]; r < run_off[k + 1]; ++r) {
      const int64_t* w = run + 8 * r; int d = (int)w[1]; int z = cand2[d * K + choice[r]]; if (z < 0) continue;
      int64_t f = w[2], lo = w[3], hi = w[4];
      uint64_t gk = zkey(z, f); auto it = order.find(gk);
      if (it == order.end()) { order.emplace(gk, (int)ivs.size()); ivs.push_back({}); gkey.push_back({z, f}); it = order.find(gk); }
      ivs[it->second].push_back({lo, hi});
      if (d == 0) { touch(lo, f, z, z); touch(hi, f, z, z); } else { touch(f, lo, z, z); touch(f, hi, z, z); }
    }
    for (int64_t m = node_off[k]; m < node_off[k + 1]; ++m) { const int64_t* v = nodes + 4 * m; touch(v[0], v[1], v[2], v[3]); }
    for (size_t g = 0; g < ivs.size(); ++g) { int z = gkey[g].first; int64_t f = gkey[g].second;
      for (auto& iv : ivs[g]) { if (dirs[z] == 0) { b.insert(b.end(), {iv.first, f, (int64_t)z, iv.second, f, (int64_t)z}); } else { b.insert(b.end(), {f, iv.first, (int64_t)z, f, iv.second, (int64_t)z}); } } }
    for (auto& v : nl) for (int64_t z = v[2]; z < v[3]; ++z) b.insert(b.end(), {v[0], v[1], z, v[0], v[1], z + 1});
  });
  int64_t tot = 0; off[0] = 0;
  for (int64_t q = 0; q < nk; ++q) { tot += (int64_t)out[q].size() / 6; off[q + 1] = tot; }
  if (rows) par_for(T, nk, 256, [&](int64_t q) { if (!out[q].empty()) memcpy(rows + 6 * off[q], out[q].data(), out[q].size() * 8); });
  return tot;
}
// overflow cost per net along its wires, official rule: u exp(0.25 d - 0.5 c) (open), u exp(0.75 d) (blocked, d > 0): the potential of nets for the cut. rows6 of every candidate net (GCell), wires only.
void dgb_ovf_pot(int64_t nn, const int64_t* noff, const int64_t* rows6, const double* d, int64_t s0, int64_t s1, int64_t s2, const double* cc, int64_t c0, int64_t c1, int64_t c2, const double* ul, int T, double* pot) {
  par_for(T, nn, 256, [&](int64_t n) {
    double a = 0;
    for (int64_t q = noff[n]; q < noff[n + 1]; ++q) {
      const int64_t* s = rows6 + 6 * q; int64_t z = s[2];
      if (z != s[5]) continue;
      int64_t x0 = std::min(s[0], s[3]), x1 = std::max(s[0], s[3]), y0 = std::min(s[1], s[4]), y1 = std::max(s[1], s[4]);
      bool hz = y0 == y1; int64_t m = hz ? x1 - x0 : y1 - y0;
      for (int64_t t = 0; t < m; ++t) {
        int64_t x = hz ? x0 + t : x0, y = hz ? y0 : y0 + t;
        double dd = d[z * s0 + x * s1 + y * s2], cv = cc[z * c0 + x * c1 + y * c2];
        double e = cv > 1e-3 ? 0.25 * dd - 0.5 * cv : 0.75 * dd;
        if (e > -6.0) a += ul[z] * std::exp(std::min(e, 60.0)) * (cv > 1e-3 ? 0.6487212707 : 3.4816890703);
      }
    }
    pot[n] = a;
  });
}
}

// ---------------------------------------------------------------- the RC topology of the given nets on threads: tools/gnnfix/dtimer/rc_topo.cc csta_topo's per-net body, same order and values (per-net outputs merged in net order)
namespace {
inline int64_t pack3t(int64_t z, int64_t x, int64_t y) { return (z * 65536 + x) * 65536 + y; }
struct PinT { int64_t pin, gnode; double grv, sr, sc; };
struct TNet { int64_t n = -1, root = -1, nnodes = 0, nsegs = 0;
  std::vector<int64_t> sa, sb, svia, sli, svi, nkind, npin, npar, ndep, nrk, nri, nreach, pg; std::vector<double> sglen, pgrv, psr, psc; };
std::vector<TNet> TN; std::vector<int64_t> TROOT, THASD;
}
extern "C" {
int64_t dgb_topo(const int64_t* seg, int64_t S, double stub_um, int64_t nn, const int64_t* ap_off, const int64_t* ap_idx, const int64_t* ap, const int64_t* ap_pid, const int64_t* net_drv, const double* layer_rc, const double* via_r, int64_t nlr, int64_t nvr, double UMGC, int T, int64_t* counts) {
  const int L = 16;
  std::vector<int64_t> cnt(nn + 1, 0);
  std::vector<unsigned char> keep(S);
  for (int64_t s = 0; s < S; s++) { const int64_t* r = seg + s * 7; bool k = r[0] >= 0 && ((r[3] != r[6]) || (r[1] != r[4]) || (r[2] != r[5])); keep[s] = k; if (k) cnt[r[0] + 1]++; }
  for (int64_t i = 0; i < nn; i++) cnt[i + 1] += cnt[i];
  std::vector<int64_t> sidx(cnt[nn]), fillp(cnt.begin(), cnt.end() - 1);
  for (int64_t s = 0; s < S; s++) if (keep[s]) sidx[fillp[seg[s * 7]]++] = s;
  std::vector<int64_t> nets; for (int64_t n = 0; n < nn; n++) if (cnt[n + 1] > cnt[n]) nets.push_back(n);
  TN.assign(nets.size(), TNet());
  TROOT.assign(nn, -1); THASD.resize(nn);
  for (int64_t n = 0; n < nn; n++) THASD[n] = net_drv[n] >= 0;
  std::atomic<int64_t> nx(0);
  auto work = [&]() {
    std::vector<int64_t> uk, ia, ib, adjo, adje, adjn, par, depth, eu, ev, bucket, ekind, eidx, stN, stI, order;
    std::vector<double> Rs, eR; std::vector<PinT> pins, pins2; std::vector<unsigned char> reach;
    for (;;) {
      int64_t q = nx.fetch_add(1); if (q >= (int64_t)nets.size()) return;
      int64_t n = nets[q]; TNet& o = TN[q]; o.n = n;
      const int64_t m = cnt[n + 1] - cnt[n]; const int64_t* sg = sidx.data() + cnt[n];
      uk.resize(2 * m);
      for (int64_t j = 0; j < m; j++) { const int64_t* r = seg + sg[j] * 7; uk[2 * j] = pack3t(r[3], r[1], r[2]); uk[2 * j + 1] = pack3t(r[6], r[4], r[5]); }
      std::sort(uk.begin(), uk.end()); uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
      const int64_t NN = (int64_t)uk.size();
      auto find = [&](int64_t key) -> int64_t { auto it = std::lower_bound(uk.begin(), uk.end(), key); return (it != uk.end() && *it == key) ? it - uk.begin() : -1; };
      ia.resize(m); ib.resize(m); Rs.resize(m);
      o.sa.resize(m); o.sb.resize(m); o.svia.resize(m); o.sli.resize(m); o.svi.resize(m); o.sglen.resize(m);
      for (int64_t j = 0; j < m; j++) {
        const int64_t* r = seg + sg[j] * 7;
        int64_t x1 = r[1], y1_ = r[2], z1 = r[3], x2 = r[4], y2_ = r[5], z2 = r[6];
        ia[j] = find(pack3t(z1, x1, y1_)); ib[j] = find(pack3t(z2, x2, y2_));
        bool isvia = z1 != z2;
        double glen = (double)(std::llabs(x2 - x1) + std::llabs(y2_ - y1_)) * UMGC;
        int64_t zl = std::min(z1, z2);
        int64_t vi = std::min(std::max(zl, (int64_t)0), nvr - 1), li = std::min(std::max(z1, (int64_t)0), nlr - 1);
        Rs[j] = isvia ? via_r[vi] : layer_rc[li * 2] * glen;
        o.sa[j] = ia[j]; o.sb[j] = ib[j]; o.svia[j] = isvia; o.sli[j] = li; o.svi[j] = vi; o.sglen[j] = glen;
      }
      pins.clear();
      for (int64_t t = ap_off[n]; t < ap_off[n + 1]; t++) {
        int64_t a = ap_idx[t];
        int64_t apl = ap[a * 3], apx = ap[a * 3 + 1], apy = ap[a * 3 + 2];
        int64_t ku = find(pack3t(std::min(apl + 1, (int64_t)L - 1), apx, apy));
        int64_t ks = (ku >= 0) ? -1 : find(pack3t(apl, apx, apy));
        bool oku = ku >= 0;
        int64_t gnode = oku ? ku : ks;
        if (gnode < 0) continue;
        PinT pr; pr.pin = ap_pid[a]; pr.gnode = gnode;
        pr.grv = oku ? via_r[std::min(std::max(apl, (int64_t)0), nvr - 1)] : 0.0;
        int64_t lz = oku ? std::min(std::max(apl + 1, (int64_t)0), nlr - 1) : std::min(std::max(apl, (int64_t)0), nlr - 1);
        pr.sr = layer_rc[lz * 2] * stub_um; pr.sc = layer_rc[lz * 2 + 1] * stub_um;
        pins.push_back(pr);
      }
      std::stable_sort(pins.begin(), pins.end(), [](const PinT& a, const PinT& b) { return a.pin < b.pin; });
      pins2.clear();
      for (size_t t = 0; t < pins.size(); t++) if (t == 0 || pins[t].pin != pins[t - 1].pin) pins2.push_back(pins[t]);
      const int64_t npn = (int64_t)pins2.size(), NT = NN + npn, E = m + npn;
      eR.resize(E); eu.resize(E); ev.resize(E);
      for (int64_t j = 0; j < m; j++) { eu[j] = ia[j]; ev[j] = ib[j]; eR[j] = Rs[j]; }
      for (int64_t k = 0; k < npn; k++) { eu[m + k] = NN + k; ev[m + k] = pins2[k].gnode; eR[m + k] = pins2[k].grv + pins2[k].sr; }
      adjo.assign(NT + 2, 0);
      for (int64_t e = 0; e < E; e++) { adjo[eu[e] + 1]++; adjo[ev[e] + 1]++; }
      for (int64_t i = 0; i < NT; i++) adjo[i + 1] += adjo[i];
      adje.resize(2 * E); adjn.resize(2 * E);
      bucket.assign(adjo.begin(), adjo.begin() + NT);
      for (int64_t e = 0; e < E; e++) { int64_t p = bucket[eu[e]]++; adje[p] = e; adjn[p] = ev[e]; p = bucket[ev[e]]++; adje[p] = e; adjn[p] = eu[e]; }
      o.nkind.assign(NT, 0); o.npin.assign(NT, -1); o.pgrv.assign(NT, 0.0); o.psr.assign(NT, 0.0); o.psc.assign(NT, 0.0); o.pg.assign(NT, -1);
      for (int64_t k = 0; k < npn; k++) { o.nkind[NN + k] = 1; o.npin[NN + k] = pins2[k].pin; o.pgrv[NN + k] = pins2[k].grv; o.psr[NN + k] = pins2[k].sr; o.psc[NN + k] = pins2[k].sc; o.pg[NN + k] = pins2[k].gnode; }
      int64_t drv = net_drv[n], root = -1;
      if (drv >= 0) { auto it = std::lower_bound(pins2.begin(), pins2.end(), drv, [](const PinT& a, int64_t v) { return a.pin < v; });
        if (it != pins2.end() && it->pin == drv) root = NN + (it - pins2.begin()); }
      par.assign(NT, -1); depth.assign(NT, 0); reach.assign(NT, 0); ekind.assign(NT, 0); eidx.assign(NT, -1);
      if (root >= 0) {
        o.root = root;
        stN.clear(); stI.clear(); order.clear();
        stN.push_back(root); stI.push_back(adjo[root]); reach[root] = 1; order.push_back(root);
        while (!stN.empty()) {
          int64_t u = stN.back(), i = stI.back();
          if (i >= adjo[u + 1]) { stN.pop_back(); stI.pop_back(); continue; }
          stI.back() = i + 1;
          int64_t w = adjn[i];
          if (reach[w]) continue;
          reach[w] = 1; par[w] = u; depth[w] = depth[u] + 1; order.push_back(w);
          stN.push_back(w); stI.push_back(adjo[w]);
        }
        for (int64_t w : order) {
          if (par[w] < 0) continue;
          double best = 0; bool f = false; int64_t be = -1;
          for (int64_t i = adjo[w]; i < adjo[w + 1]; i++) if (adjn[i] == par[w]) { double r = eR[adje[i]]; if (!f || r < best) { best = r; f = true; be = adje[i]; } }
          if (be < m) { ekind[w] = 1; eidx[w] = be; } else { ekind[w] = 2; eidx[w] = NN + (be - m); }
        }
      }
      o.npar = par; o.ndep = depth; o.nrk = ekind; o.nri = eidx; o.nreach.assign(reach.begin(), reach.end());
      o.nnodes = NT; o.nsegs = m;
    }
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join(); }
  int64_t ns = 0, nk = 0; for (auto& o : TN) { ns += o.nsegs; nk += o.nnodes; }
  counts[0] = ns; counts[1] = nk;
  return 0;
}
void dgb_topo_get(int64_t* si, double* sg, int64_t* ni, double* nd, int64_t* nt, int T, int64_t ns, int64_t nk, int64_t nn) {
  int64_t K_ = (int64_t)TN.size();
  std::vector<int64_t> nb(K_ + 1, 0), sb(K_ + 1, 0);
  for (int64_t q = 0; q < K_; ++q) { nb[q + 1] = nb[q] + TN[q].nnodes; sb[q + 1] = sb[q] + TN[q].nsegs; }
  par_for(T, K_, 64, [&](int64_t q) {
    TNet& o = TN[q]; int64_t b = nb[q], s = sb[q];
    for (int64_t j = 0; j < o.nsegs; ++j) {
      si[0 * ns + s + j] = o.n; si[1 * ns + s + j] = b + o.sa[j]; si[2 * ns + s + j] = b + o.sb[j]; si[3 * ns + s + j] = o.svia[j]; si[4 * ns + s + j] = o.sli[j]; si[5 * ns + s + j] = o.svi[j]; sg[s + j] = o.sglen[j];
    }
    for (int64_t i = 0; i < o.nnodes; ++i) {
      int64_t k = b + i;
      ni[0 * nk + k] = o.n; ni[1 * nk + k] = o.nkind[i]; ni[2 * nk + k] = o.npin[i]; ni[3 * nk + k] = o.npar[i] >= 0 ? b + o.npar[i] : -1; ni[4 * nk + k] = o.ndep[i]; ni[5 * nk + k] = o.nrk[i];
      ni[6 * nk + k] = o.nrk[i] == 1 ? s + o.nri[i] : (o.nrk[i] == 2 ? b + o.nri[i] : -1); ni[7 * nk + k] = o.nreach[i]; ni[8 * nk + k] = o.pg[i] >= 0 ? b + o.pg[i] : -1;
      nd[0 * nk + k] = o.pgrv[i]; nd[1 * nk + k] = o.psr[i]; nd[2 * nk + k] = o.psc[i];
    }
  });
  for (int64_t n = 0; n < nn; ++n) { nt[n] = -1; nt[nn + n] = THASD[n]; }
  for (int64_t q = 0; q < K_; ++q) if (TN[q].root >= 0) nt[TN[q].n] = nb[q] + TN[q].root;
  std::vector<TNet>().swap(TN);
}
// per net: the two row lists hold the same rows (order-free)
void dgb_same(int64_t n, const int64_t* ra, const int64_t* oa, const int64_t* rb, const int64_t* ob, int T, unsigned char* ok) {
  par_for(T, n, 64, [&](int64_t q) {
    int64_t ca = oa[q + 1] - oa[q], cb = ob[q + 1] - ob[q];
    if (ca != cb) { ok[q] = 0; return; }
    std::vector<std::array<int64_t, 6>> A(ca), B(cb);
    for (int64_t j = 0; j < ca; ++j) { memcpy(A[j].data(), ra + 6 * (oa[q] + j), 48); memcpy(B[j].data(), rb + 6 * (ob[q] + j), 48); }
    std::sort(A.begin(), A.end()); std::sort(B.begin(), B.end());
    ok[q] = A == B ? 1 : 0;
  });
}
}
// ---------------------------------------------------------------- dg.Dg.rc's tables (gputimer._rc_core with the runs' R and C free) from the topology, in C++
extern "C" {
// counts: 0 entries node-cap, 1 entries run-R, 2 reached non-root nodes, 3 nets with such nodes, 4 roots, 5 pins read
void dgb_rc_count(int64_t ns, int64_t nk, int64_t nn, const int64_t* si, const int64_t* ni, const int64_t* nt, const int64_t* wire, int64_t nw, const int64_t* sr, const unsigned char* pin_is_drv, int64_t* counts) {
  const int64_t* s_a = si + ns, *s_b = si + 2 * ns; const int64_t* n_net = ni, *n_kind = ni + nk, *n_pin = ni + 2 * nk, *n_par = ni + 3 * nk, *n_rk = ni + 5 * nk, *n_ri = ni + 6 * nk, *n_reach = ni + 7 * nk;
  std::vector<int64_t> pos(ns, -1); for (int64_t w = 0; w < nw; ++w) pos[wire[w]] = w;
  int64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, last = -1;
  for (int64_t w = 0; w < nw; ++w) if (sr[w] >= 0) { int64_t s = wire[w], a = s_a[s], b = s_b[s]; c0 += (n_reach[a] && n_kind[a] != 1); c0 += (n_reach[b] && n_kind[b] != 1); }
  for (int64_t n = 0; n < nk; ++n) {
    if (n_rk[n] == 1 && pos[n_ri[n]] >= 0 && sr[pos[n_ri[n]]] >= 0) c1++;
    if (n_reach[n] && n_par[n] >= 0) { c2++; if (n_net[n] != last) { c3++; last = n_net[n]; } }
    if (n_kind[n] == 1 && n_reach[n] && !pin_is_drv[n_pin[n]]) c5++;
  }
  for (int64_t n = 0; n < nn; ++n) c4 += nt[n] >= 0;
  counts[0] = c0; counts[1] = c1; counts[2] = c2; counts[3] = c3; counts[4] = c4; counts[5] = c5;
}
void dgb_rc(int64_t ns, int64_t nk, int64_t nn, int64_t R, const int64_t* si, const double* sg, const int64_t* ni, const double* nd, const int64_t* nt, const int64_t* wire, int64_t nw, const int64_t* sr, const double* lrc, const double* via_r, const double* pin_cap, const unsigned char* pin_is_drv,
            double* Rp0, int32_t* rp_run, double* rp_gl, double* cap0, unsigned char* reach, int32_t* nsp, int32_t* nsr, double* nsg, int32_t* rsp, int32_t* rsn, double* rsg, int32_t* rrp, int32_t* rrn, double* rrg, int32_t* rnp, int32_t* rno, int64_t* par_n, int32_t* roots, int32_t* rootn, int32_t* pn_node, int32_t* pn_pin) {
  const int64_t* s_a = si + ns, *s_b = si + 2 * ns, *s_via = si + 3 * ns, *s_li = si + 4 * ns, *s_vi = si + 5 * ns;
  const int64_t* n_net = ni, *n_kind = ni + nk, *n_pin = ni + 2 * nk, *n_par = ni + 3 * nk, *n_dep = ni + 4 * nk, *n_rk = ni + 5 * nk, *n_ri = ni + 6 * nk, *n_reach = ni + 7 * nk, *n_pg = ni + 8 * nk;
  const double* n_grv = nd, *n_sr = nd + nk, *n_sc = nd + 2 * nk;
  std::vector<double> Rs(ns, 0.0), ncap(nk, 0.0);
  std::vector<int64_t> pos(ns, -1);
  for (int64_t s = 0; s < ns; ++s) if (s_via[s]) Rs[s] = via_r[s_vi[s]];
  for (int64_t w = 0; w < nw; ++w) {
    int64_t s = wire[w]; pos[s] = w; double gl = sg[s], r0 = lrc[s_li[s] * 2] * gl, c0 = lrc[s_li[s] * 2 + 1] * gl;
    Rs[s] = r0;
    if (sr[w] < 0) { ncap[s_a[s]] += c0 / 2; ncap[s_b[s]] += c0 / 2; }
  }
  for (int64_t n = 0; n < nk; ++n) { reach[n] = n_reach[n] ? 1 : 0; cap0[2 * n] = ncap[n]; cap0[2 * n + 1] = ncap[n]; par_n[n] = n_par[n]; }
  for (int64_t n = 0; n < nk; ++n) if (n_kind[n] == 1) { double sc = n_sc[n]; cap0[2 * n] = pin_cap[2 * n_pin[n]] + sc / 2; cap0[2 * n + 1] = pin_cap[2 * n_pin[n] + 1] + sc / 2; }
  for (int64_t n = 0; n < nk; ++n) if (n_kind[n] == 1) { double sc = n_sc[n]; cap0[2 * n_pg[n]] += sc / 2; cap0[2 * n_pg[n] + 1] += sc / 2; }
  for (int64_t n = 0; n < nk; ++n) if (!reach[n]) { cap0[2 * n] = 0; cap0[2 * n + 1] = 0; }
  for (int64_t r = 0; r <= R; ++r) { rsp[r] = 0; rrp[r] = 0; }
  for (int64_t n = 0; n <= nk; ++n) nsp[n] = 0;
  for (int64_t n = 0; n < nk; ++n) {
    Rp0[n] = 0.0; rp_run[n] = -1; rp_gl[n] = 0.0;
    if (n_rk[n] == 1) { int64_t s = n_ri[n]; Rp0[n] = Rs[s]; int64_t w = pos[s]; if (w >= 0 && sr[w] >= 0) { rp_run[n] = (int32_t)sr[w]; rp_gl[n] = sg[s]; rrp[sr[w] + 1]++; } }
    else if (n_rk[n] == 2) { int64_t q = n_ri[n]; Rp0[n] = n_grv[q] + n_sr[q]; }
  }
  auto ok_node = [&](int64_t a) { return reach[a] && n_kind[a] != 1; };
  for (int64_t w = 0; w < nw; ++w) if (sr[w] >= 0) { int64_t s = wire[w]; for (int64_t a : {s_a[s], s_b[s]}) if (ok_node(a)) { nsp[a + 1]++; rsp[sr[w] + 1]++; } }
  for (int64_t n = 0; n < nk; ++n) nsp[n + 1] += nsp[n];
  for (int64_t r = 0; r < R; ++r) { rsp[r + 1] += rsp[r]; rrp[r + 1] += rrp[r]; }
  { std::vector<int32_t> f1(nsp, nsp + nk), f2(rsp, rsp + R);
    for (int64_t w = 0; w < nw; ++w) if (sr[w] >= 0) { int64_t s = wire[w]; double gh = sg[s] / 2; for (int64_t a : {s_a[s], s_b[s]}) if (ok_node(a)) { int32_t q = f1[a]++; nsr[q] = (int32_t)sr[w]; nsg[q] = gh; int32_t t = f2[sr[w]]++; rsn[t] = (int32_t)a; rsg[t] = gh; } }
    std::vector<int32_t> f3(rrp, rrp + R);
    for (int64_t n = 0; n < nk; ++n) if (rp_run[n] >= 0) { int32_t t = f3[rp_run[n]]++; rrn[t] = (int32_t)n; rrg[t] = rp_gl[n]; } }
  // nodes per net by depth
  int64_t c = 0, nc = 0, last = -1;
  for (int64_t n = 0; n < nk; ++n) if (n_reach[n] && n_par[n] >= 0) {
    if (n_net[n] != last) { rnp[nc++] = (int32_t)c; last = n_net[n]; }
    rno[c++] = (int32_t)n;
  }
  rnp[nc] = (int32_t)c;
  for (int64_t q = 0; q < nc; ++q) std::stable_sort(rno + rnp[q], rno + rnp[q + 1], [&](int32_t a, int32_t b) { return n_dep[a] < n_dep[b]; });
  int64_t k = 0; for (int64_t n = 0; n < nn; ++n) if (nt[n] >= 0) { roots[k] = (int32_t)n; rootn[k] = (int32_t)nt[n]; k++; }
  k = 0; for (int64_t n = 0; n < nk; ++n) if (n_kind[n] == 1 && n_reach[n] && !pin_is_drv[n_pin[n]]) { pn_node[k] = (int32_t)n; pn_pin[k] = (int32_t)n_pin[n]; k++; }
}
// points -> incident runs (p0 then p1, as dtla_via's two index_adds): prp[npt+1], prr[2R]
void dgb_pr(int64_t R, const int64_t* run, int64_t npt, int32_t* prp, int32_t* prr) {
  for (int64_t p = 0; p <= npt; ++p) prp[p] = 0;
  for (int64_t r = 0; r < R; ++r) { prp[run[8 * r + 6] + 1]++; prp[run[8 * r + 7] + 1]++; }
  for (int64_t p = 0; p < npt; ++p) prp[p + 1] += prp[p];
  std::vector<int32_t> f(prp, prp + npt);
  for (int64_t r = 0; r < R; ++r) prr[f[run[8 * r + 6]]++] = (int32_t)r;
  for (int64_t r = 0; r < R; ++r) prr[f[run[8 * r + 7]]++] = (int32_t)r;
}
}
