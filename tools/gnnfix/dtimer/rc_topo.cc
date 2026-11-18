// rc_topo.cc: the exact csta kernel (included unchanged) + one extra entry point that
// exports the RC-tree TOPOLOGY that csta_build_rc builds for every net (same node keys, same pin attachment, same
// depth-first tree, same min-resistance parallel edge), so that tools/gnnfix/dtimer/dtimer.py can recompute the
// moments y1/y2/y3, the Elmore delays and the pi model in torch with differentiable segment R and C.
// Build (host): g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -shared -fPIC -pthread rc_topo.cc -o librc_topo.so
#include "../../gapla/csta_kern.cc"

// outputs (static, fetched with csta_topo_get)
static std::vector<i64> T_snet, T_sa, T_sb, T_svia, T_sli, T_svi;      // kept segments
static std::vector<double> T_sglen;
static std::vector<i64> T_nnet, T_nkind, T_npin, T_npar, T_ndep, T_nrk, T_nri, T_nreach;   // nodes (wire + pin)
static std::vector<double> T_pgrv, T_psr, T_psc;                          // per pin node (indexed by node)
static std::vector<i64> T_pg;                                              // pin node -> its wire node (global)
static std::vector<i64> T_root, T_hasd;                                    // per net

extern "C" i64 csta_topo(const i64* seg, i64 S, double stub_um, i64* counts) {
  const i64 nn = gi("nn");
  const double UMGC = *g<double>("um_per_gc");
  const i64 *ap_off = g<i64>("ap_off"), *ap_idx = g<i64>("ap_idx"), *ap = g<i64>("ap"), *ap_pid = g<i64>("ap_pid"),
            *net_drv = g<i64>("net_drv");
  const double *layer_rc = g<double>("layer_rc"), *via_r = g<double>("via_r");
  const i64 nlr = gi("nlr"), nvr = gi("nvr");
  const int L = 16;
  T_snet.clear(); T_sa.clear(); T_sb.clear(); T_svia.clear(); T_sli.clear(); T_svi.clear(); T_sglen.clear();
  T_nnet.clear(); T_nkind.clear(); T_npin.clear(); T_npar.clear(); T_ndep.clear(); T_nrk.clear(); T_nri.clear(); T_nreach.clear();
  T_pgrv.clear(); T_psr.clear(); T_psc.clear(); T_pg.clear(); T_root.assign(nn, -1); T_hasd.assign(nn, 0);
  std::vector<i64> cnt(nn + 1, 0);
  std::vector<unsigned char> keep(S);
  for (i64 s = 0; s < S; s++) {
    const i64* r = seg + s * 7;
    bool k = r[0] >= 0 && ((r[3] != r[6]) || (r[1] != r[4]) || (r[2] != r[5]));
    keep[s] = k;
    if (k) cnt[r[0] + 1]++;
  }
  for (i64 i = 0; i < nn; i++) cnt[i + 1] += cnt[i];
  std::vector<i64> sidx(cnt[nn]), fillp(cnt.begin(), cnt.end() - 1);
  for (i64 s = 0; s < S; s++) if (keep[s]) sidx[fillp[seg[s * 7]]++] = s;
  std::vector<i64> uk, ia, ib, adjo, adje, adjn, par, depth, eu, ev, bucket, ekind, eidx;
  std::vector<double> Rs, eR;
  std::vector<PinRec> pins, pins2;
  std::vector<unsigned char> reach;
  for (i64 n = 0; n < nn; n++) {
    T_hasd[n] = net_drv[n] >= 0;
    const i64 m = cnt[n + 1] - cnt[n];
    if (m <= 0) continue;
    const i64* sg = sidx.data() + cnt[n];
    uk.resize(2 * m);
    for (i64 j = 0; j < m; j++) {
      const i64* r = seg + sg[j] * 7;
      uk[2 * j] = pack3(r[3], r[1], r[2]); uk[2 * j + 1] = pack3(r[6], r[4], r[5]);
    }
    std::sort(uk.begin(), uk.end());
    uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
    const i64 NN = (i64)uk.size();
    auto find = [&](i64 key) -> i64 { auto it = std::lower_bound(uk.begin(), uk.end(), key); return (it != uk.end() && *it == key) ? it - uk.begin() : -1; };
    const i64 nbase = (i64)T_nnet.size(), sbase = (i64)T_snet.size();
    ia.resize(m); ib.resize(m); Rs.resize(m);
    for (i64 j = 0; j < m; j++) {
      const i64* r = seg + sg[j] * 7;
      i64 x1 = r[1], y1_ = r[2], z1 = r[3], x2 = r[4], y2_ = r[5], z2 = r[6];
      ia[j] = find(pack3(z1, x1, y1_)); ib[j] = find(pack3(z2, x2, y2_));
      bool isvia = z1 != z2;
      double glen = (double)(std::llabs(x2 - x1) + std::llabs(y2_ - y1_)) * UMGC;
      i64 zl = std::min(z1, z2);
      i64 vi = std::min(std::max(zl, (i64)0), nvr - 1), li = std::min(std::max(z1, (i64)0), nlr - 1);
      Rs[j] = isvia ? via_r[vi] : layer_rc[li * 2] * glen;
      T_snet.push_back(n); T_sa.push_back(nbase + ia[j]); T_sb.push_back(nbase + ib[j]); T_svia.push_back(isvia);
      T_sli.push_back(li); T_svi.push_back(vi); T_sglen.push_back(glen);
    }
    pins.clear();
    for (i64 t = ap_off[n]; t < ap_off[n + 1]; t++) {
      i64 a = ap_idx[t];
      i64 apl = ap[a * 3], apx = ap[a * 3 + 1], apy = ap[a * 3 + 2];
      i64 ku = find(pack3(std::min(apl + 1, (i64)L - 1), apx, apy));
      i64 ks = (ku >= 0) ? -1 : find(pack3(apl, apx, apy));
      bool oku = ku >= 0;
      i64 gnode = oku ? ku : ks;
      if (gnode < 0) continue;
      PinRec pr;
      pr.pin = ap_pid[a]; pr.gnode = gnode;
      pr.grv = oku ? via_r[std::min(std::max(apl, (i64)0), nvr - 1)] : 0.0;
      i64 lz = oku ? std::min(std::max(apl + 1, (i64)0), nlr - 1) : std::min(std::max(apl, (i64)0), nlr - 1);
      pr.sr = layer_rc[lz * 2] * stub_um; pr.sc = layer_rc[lz * 2 + 1] * stub_um;
      pins.push_back(pr);
    }
    std::stable_sort(pins.begin(), pins.end(), [](const PinRec& a, const PinRec& b) { return a.pin < b.pin; });
    pins2.clear();
    for (size_t t = 0; t < pins.size(); t++) if (t == 0 || pins[t].pin != pins[t - 1].pin) pins2.push_back(pins[t]);
    const i64 npn = (i64)pins2.size(), NT = NN + npn;
    const i64 E = m + npn;
    eR.resize(E); eu.resize(E); ev.resize(E);
    for (i64 j = 0; j < m; j++) { eu[j] = ia[j]; ev[j] = ib[j]; eR[j] = Rs[j]; }
    for (i64 k = 0; k < npn; k++) { eu[m + k] = NN + k; ev[m + k] = pins2[k].gnode; eR[m + k] = pins2[k].grv + pins2[k].sr; }
    adjo.assign(NT + 2, 0);
    for (i64 e = 0; e < E; e++) { adjo[eu[e] + 1]++; adjo[ev[e] + 1]++; }
    for (i64 i = 0; i < NT; i++) adjo[i + 1] += adjo[i];
    adje.resize(2 * E); adjn.resize(2 * E);
    bucket.assign(adjo.begin(), adjo.begin() + NT);
    for (i64 e = 0; e < E; e++) {
      i64 p = bucket[eu[e]]++; adje[p] = e; adjn[p] = ev[e];
      p = bucket[ev[e]]++; adje[p] = e; adjn[p] = eu[e];
    }
    // nodes
    for (i64 i = 0; i < NN; i++) { T_nnet.push_back(n); T_nkind.push_back(0); T_npin.push_back(-1); T_pgrv.push_back(0); T_psr.push_back(0);
      T_psc.push_back(0); T_pg.push_back(-1); }
    for (i64 k = 0; k < npn; k++) { T_nnet.push_back(n); T_nkind.push_back(1); T_npin.push_back(pins2[k].pin); T_pgrv.push_back(pins2[k].grv);
      T_psr.push_back(pins2[k].sr); T_psc.push_back(pins2[k].sc); T_pg.push_back(nbase + pins2[k].gnode); }
    i64 drv = net_drv[n];
    i64 root = -1;
    if (drv >= 0) { auto it = std::lower_bound(pins2.begin(), pins2.end(), drv, [](const PinRec& a, i64 v) { return a.pin < v; });
      if (it != pins2.end() && it->pin == drv) root = NN + (it - pins2.begin()); }
    par.assign(NT, -1); depth.assign(NT, 0); reach.assign(NT, 0); ekind.assign(NT, 0); eidx.assign(NT, -1);
    if (root >= 0) {
      T_root[n] = nbase + root;
      std::vector<i64> stN, stI, order;
      stN.push_back(root); stI.push_back(adjo[root]); reach[root] = 1; order.push_back(root);
      while (!stN.empty()) {
        i64 u = stN.back(), i = stI.back();
        if (i >= adjo[u + 1]) { stN.pop_back(); stI.pop_back(); continue; }
        stI.back() = i + 1;
        i64 w = adjn[i];
        if (reach[w]) continue;
        reach[w] = 1; par[w] = u; depth[w] = depth[u] + 1; order.push_back(w);
        stN.push_back(w); stI.push_back(adjo[w]);
      }
      for (i64 w : order) {
        if (par[w] < 0) continue;
        double best = BIG * 1e300; bool f = false; i64 be = -1;
        for (i64 i = adjo[w]; i < adjo[w + 1]; i++) if (adjn[i] == par[w]) { double r = eR[adje[i]]; if (!f || r < best) { best = r; f = true; be = adje[i]; } }
        if (be < m) { ekind[w] = 1; eidx[w] = sbase + be; } else { ekind[w] = 2; eidx[w] = nbase + NN + (be - m); }
      }
    }
    for (i64 i = 0; i < NT; i++) { T_npar.push_back(par[i] >= 0 ? nbase + par[i] : -1); T_ndep.push_back(depth[i]); T_nrk.push_back(ekind[i]);
      T_nri.push_back(eidx[i]); T_nreach.push_back(reach[i]); }
  }
  counts[0] = (i64)T_snet.size(); counts[1] = (i64)T_nnet.size();
  return 0;
}
// seg arrays: net a b via li vi (i64 x6, each [ns]) + glen [ns];  node arrays: net kind pin par dep rk ri reach pg (i64 x9, [nnode])
// + grv sr sc [nnode];  net arrays: root hasd [nn]
extern "C" void csta_topo_get(i64* si, double* sg, i64* ni, double* nd, i64* nt) {
  const i64 ns = (i64)T_snet.size(), nk = (i64)T_nnet.size(), nn = (i64)T_root.size();
  const std::vector<i64>* S6[6] = {&T_snet, &T_sa, &T_sb, &T_svia, &T_sli, &T_svi};
  for (int c = 0; c < 6; c++) memcpy(si + c * ns, S6[c]->data(), ns * 8);
  memcpy(sg, T_sglen.data(), ns * 8);
  const std::vector<i64>* N9[9] = {&T_nnet, &T_nkind, &T_npin, &T_npar, &T_ndep, &T_nrk, &T_nri, &T_nreach, &T_pg};
  for (int c = 0; c < 9; c++) memcpy(ni + c * nk, N9[c]->data(), nk * 8);
  const std::vector<double>* N3[3] = {&T_pgrv, &T_psr, &T_psc};
  for (int c = 0; c < 3; c++) memcpy(nd + c * nk, N3[c]->data(), nk * 8);
  memcpy(nt, T_root.data(), nn * 8); memcpy(nt + nn, T_hasd.data(), nn * 8);
}
