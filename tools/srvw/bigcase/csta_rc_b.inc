// csta_rc_b.inc: copy of tools/gapla/csta_rc.inc, item M3 below; included by csta_kern_b.cc.
// csta_rc.inc -- included by csta_kern.cc.  build_rc (v3 semantics: OpenSTA depth-first tree, identical to the
// BFS tree on loop-free nets) and the critical-path extraction (fastmode4.write_crit + io_build.read_crit_lines).
static const i64 PK = 65536;
static inline i64 pack3(i64 z, i64 x, i64 y) { return (z * PK + x) * PK + y; }

struct PinRec { i64 pin; i64 gnode; double grv, sr, sc; };

extern "C" int csta_build_rc(const i64* seg, i64 S, double stub_um, double* c1o, double* c2o, double* rpio, double* ctoto,
                             unsigned char* paro, double* elmo, double* stats) {
  double t0w = now();
  const i64 P = gi("P"), nn = gi("nn");
  const double UMGC = *g<double>("um_per_gc");
  const i64 *ap_off = g<i64>("ap_off"), *ap_idx = g<i64>("ap_idx"), *ap = g<i64>("ap"), *ap_pid = g<i64>("ap_pid"),
            *net_drv = g<i64>("net_drv");
  const double *pin_cap = g<double>("pin_cap"), *layer_rc = g<double>("layer_rc"), *via_r = g<double>("via_r"),
               *lumped = g<double>("lumped");
  const unsigned char* pin_is_drv = g<unsigned char>("pin_is_drv");
  const i64 nlr = gi("nlr"), nvr = gi("nvr");
  const int L = 16;
  for (i64 i = 0; i < nn; i++) { c1o[i * 2] = c1o[i * 2 + 1] = c2o[i * 2] = c2o[i * 2 + 1] = rpio[i * 2] = rpio[i * 2 + 1] = 0.0;
    ctoto[i * 2] = lumped[i * 2]; ctoto[i * 2 + 1] = lumped[i * 2 + 1]; paro[i] = 0; }
  for (i64 i = 0; i < P * 2; i++) elmo[i] = NaN;
  // bucket kept segments by net (stable)
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
  std::vector<i64> nets;
  for (i64 i = 0; i < nn; i++) if (cnt[i + 1] > cnt[i]) nets.push_back(i);
  std::sort(nets.begin(), nets.end(), [&](i64 a, i64 b) { return (cnt[a + 1] - cnt[a]) > (cnt[b + 1] - cnt[b]); });
  double t1w = now();
  const i64 nnets = (i64)nets.size();
  pfor(nnets, 4, [&](i64 qlo, i64 qhi) {
    thread_local std::vector<i64> uk, ia, ib, adjo, adje, adjn, par, depth, order, stackn, stacki, bucket;
    thread_local std::vector<double> Rs, Cs, ncap, cap, eR, Rp, y1, y2, y3, elm;
    thread_local std::vector<PinRec> pins, pins2;
    thread_local std::vector<unsigned char> reach;
    for (i64 qi = qlo; qi < qhi; qi++) {
      const i64 n = nets[qi];
      const i64 m = cnt[n + 1] - cnt[n];
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
      ia.resize(m); ib.resize(m); Rs.resize(m); Cs.resize(m);
      ncap.assign(NN, 0.0);
      for (i64 j = 0; j < m; j++) {
        const i64* r = seg + sg[j] * 7;
        i64 x1 = r[1], y1_ = r[2], z1 = r[3], x2 = r[4], y2_ = r[5], z2 = r[6];
        ia[j] = find(pack3(z1, x1, y1_)); ib[j] = find(pack3(z2, x2, y2_));
        bool isvia = z1 != z2;
        double glen = (double)(std::llabs(x2 - x1) + std::llabs(y2_ - y1_)) * UMGC;
        i64 zl = std::min(z1, z2);
        i64 vi = std::min(std::max(zl, (i64)0), nvr - 1), li = std::min(std::max(z1, (i64)0), nlr - 1);
        Rs[j] = isvia ? via_r[vi] : layer_rc[li * 2] * glen;
        Cs[j] = isvia ? 0.0 : layer_rc[li * 2 + 1] * glen;
      }
      for (i64 j = 0; j < m; j++) ncap[ia[j]] += Cs[j] / 2;
      for (i64 j = 0; j < m; j++) ncap[ib[j]] += Cs[j] / 2;
      // pins -> nodes
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
      cap.assign(NT * 2, 0.0);
      for (i64 i = 0; i < NN; i++) { cap[i * 2] = ncap[i]; cap[i * 2 + 1] = ncap[i]; }
      for (i64 k = 0; k < npn; k++) {
        const PinRec& pr = pins2[k];
        cap[(NN + k) * 2] = pin_cap[pr.pin * 2] + pr.sc / 2;
        cap[(NN + k) * 2 + 1] = pin_cap[pr.pin * 2 + 1] + pr.sc / 2;
      }
      for (i64 k = 0; k < npn; k++) { cap[pins2[k].gnode * 2] += pins2[k].sc / 2; cap[pins2[k].gnode * 2 + 1] += pins2[k].sc / 2; }
      // edges + adjacency (CSR, entries in edge-creation order)
      const i64 E = m + npn;
      eR.resize(E);
      std::vector<i64>& eu = stackn; std::vector<i64>& ev = stacki;  // reuse buffers
      eu.resize(E); ev.resize(E);
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
      // root
      i64 drv = net_drv[n];
      bool hasd = drv >= 0;
      paro[n] = hasd;
      i64 root = -1;
      if (hasd) { auto it = std::lower_bound(pins2.begin(), pins2.end(), drv, [](const PinRec& a, i64 v) { return a.pin < v; });
        if (it != pins2.end() && it->pin == drv) root = NN + (it - pins2.begin()); }
      if (root < 0) { continue; }  // Y = 0 -> c1 = c2 = rpi = 0, ctot = Y1 = 0 (par true)
      // depth-first tree (OpenSTA ReduceToPiElmore order)
      par.assign(NT, -1); depth.assign(NT, 0); reach.assign(NT, 0);
      std::vector<i64> stN, stI;
      stN.push_back(root); stI.push_back(adjo[root]); reach[root] = 1;
      order.clear(); order.push_back(root);
      while (!stN.empty()) {
        i64 u = stN.back(), i = stI.back();
        if (i >= adjo[u + 1]) { stN.pop_back(); stI.pop_back(); continue; }
        stI.back() = i + 1;
        i64 w = adjn[i];
        if (reach[w]) continue;
        reach[w] = 1; par[w] = u; depth[w] = depth[u] + 1; order.push_back(w);
        stN.push_back(w); stI.push_back(adjo[w]);
      }
      Rp.assign(NT, 0.0);
      for (i64 w : order) {
        if (par[w] < 0) continue;
        double best = BIG * 1e300; bool f = false;
        for (i64 i = adjo[w]; i < adjo[w + 1]; i++) if (adjn[i] == par[w]) { double r = eR[adje[i]]; if (!f || r < best) { best = r; f = true; } }
        Rp[w] = best;
      }
      // level order: (depth, node index)
      i64 maxd = 0; for (i64 w : order) maxd = std::max(maxd, depth[w]);
      std::vector<i64> lstart(maxd + 2, 0);
      for (i64 w : order) lstart[depth[w] + 1]++;
      for (i64 d = 0; d <= maxd; d++) lstart[d + 1] += lstart[d];
      std::vector<i64> nodes_sorted(order.size());
      std::sort(order.begin(), order.end());
      { std::vector<i64> fp(lstart.begin(), lstart.end() - 1); for (i64 w : order) nodes_sorted[fp[depth[w]]++] = w; }
      y1.assign(NT * 2, 0.0); y2.assign(NT * 2, 0.0); y3.assign(NT * 2, 0.0);
      for (i64 w : order) { y1[w * 2] = cap[w * 2]; y1[w * 2 + 1] = cap[w * 2 + 1]; }
      for (i64 d = maxd; d >= 1; d--) {
        for (i64 t = lstart[d]; t < lstart[d + 1]; t++) {
          i64 w = nodes_sorted[t], p = par[w]; double R = Rp[w];
          for (int rf = 0; rf < 2; rf++) {
            double a1 = y1[w * 2 + rf], a2 = y2[w * 2 + rf], a3 = y3[w * 2 + rf];
            y1[p * 2 + rf] += a1;
            y2[p * 2 + rf] += a2 - R * a1 * a1;
            y3[p * 2 + rf] += a3 - 2 * R * a1 * a2 + R * R * std::pow(a1, 3.0);
          }
        }
      }
      elm.assign(NT * 2, 0.0);
      for (i64 d = 1; d <= maxd; d++)
        for (i64 t = lstart[d]; t < lstart[d + 1]; t++) {
          i64 w = nodes_sorted[t], p = par[w];
          for (int rf = 0; rf < 2; rf++) elm[w * 2 + rf] = elm[p * 2 + rf] + Rp[w] * y1[w * 2 + rf] * 1e-3;
        }
      for (int rf = 0; rf < 2; rf++) {
        double Y1 = y1[root * 2 + rf], Y2 = y2[root * 2 + rf], Y3 = y3[root * 2 + rf];
        bool capld = (Y2 == 0) && (Y3 == 0);
        double c1 = capld ? Y1 : Y2 * Y2 / Y3;
        double c2 = capld ? 0.0 : Y1 - Y2 * Y2 / Y3;
        double rpi = capld ? 0.0 : -Y3 * Y3 / (Y2 * Y2 * Y2);
        c1o[n * 2 + rf] = nn0(c1); c2o[n * 2 + rf] = nn0(c2); rpio[n * 2 + rf] = nn0(rpi);
        ctoto[n * 2 + rf] = Y1;
      }
      for (i64 k = 0; k < npn; k++) {
        i64 node = NN + k, pin = pins2[k].pin;
        if (!reach[node]) continue;
        if (pin_is_drv[pin]) continue;
        elmo[pin * 2] = elm[node * 2]; elmo[pin * 2 + 1] = elm[node * 2 + 1];
      }
    }
  });
  double t2w = now();
  stats[0] = t1w - t0w; stats[1] = t2w - t1w; stats[2] = t2w - t0w;
  return 0;
}

// ------------------------------------------------------------ critical paths (write_crit + read_crit_lines)
static std::vector<i64> C_poff, C_net, C_fan; static std::vector<double> C_slack, C_cap, C_dl; static std::vector<unsigned char> C_ok;
static double r5(double v) { char b[400]; snprintf(b, sizeof b, "%.5f", v); return strtod(b, nullptr); }
static double r10_5(double v) { char b[400]; snprintf(b, sizeof b, "%10.5f", v); return strtod(b, nullptr); }

extern "C" i64 csta_crit(i64 npaths, double slack_max, const double* AT, const i64* BEST, const double* rmin,
                         const double* net_ctot, const double* net_rd_kohm, double* stats) {
  double ta = now();
  const i64 P = gi("P"), K = gi("K"), nE = gi("nE"), nn = gi("nn"), nrow = gi("nrow");
  const i64 *ck_pin = g<i64>("ck_pin"), *pin_net = g<i64>("pin_net"), *net_drv = g<i64>("net_drv"), *ce_from = g<i64>("ce_from"),
            *ce_irf = g<i64>("ce_irf"), *fan = g<i64>("fan");
  const unsigned char *pin_is_load = g<unsigned char>("pin_is_load"), *is_port = g<unsigned char>("is_port"),
                      *pin_isdrvdir = g<unsigned char>("pin_isdrvdir"), *pin_is_drv = g<unsigned char>("pin_is_drv");
  std::vector<i64> rows;
  for (i64 r = 0; r < nrow; r++) if (rmin[r] < slack_max) rows.push_back(r);
  std::stable_sort(rows.begin(), rows.end(), [&](i64 a, i64 b) { if (ck_pin[a] != ck_pin[b]) return ck_pin[a] < ck_pin[b]; return rmin[a] < rmin[b]; });
  std::vector<i64> u;
  for (size_t t = 0; t < rows.size(); t++) if (t == 0 || ck_pin[rows[t]] != ck_pin[rows[t - 1]]) u.push_back(rows[t]);
  std::stable_sort(u.begin(), u.end(), [&](i64 a, i64 b) { return rmin[a] < rmin[b]; });
  if ((i64)u.size() > npaths) u.resize(npaths);
  C_poff.assign(1, 0); C_net.clear(); C_fan.clear(); C_slack.clear(); C_cap.clear(); C_dl.clear(); C_ok.clear();
  std::vector<std::vector<i64>> paths(u.size());
  pfor((i64)u.size(), 16, [&](i64 lo_, i64 hi_) { for (i64 qi = lo_; qi < hi_; qi++) {
    i64 row = u[qi], e = ck_pin[row];
    double best = 0; i64 j = 0;
    for (i64 q = 0; q < K * 2; q++) {
      double sl = S_valid[row * K * 2 + q] ? S_reqk[row * K * 2 + q] - AT[(e * K) * 2 + q] : BIG;  // AT index (e*K+k)*2+rf = e*K*2 + k*2+rf
      if (q == 0 || sl < best) { best = sl; j = q; }
    }
    i64 k = j / 2, rf = j % 2, p = e;
    std::vector<i64>& path = paths[qi];
    path.push_back(e);
    for (int it = 0; it < 100000; it++) {
      if ((pin_is_load[p] && !is_port[p]) || (is_port[p] && !pin_isdrvdir[p])) {
        i64 nt = pin_net[p]; if (nt < 0) nt += nn;
        i64 q = net_drv[nt];
        if (q < 0 || AT[(q * K + k) * 2 + rf] <= NEG / 2) break;
        p = q;
      } else {
        i64 eb = BEST[(p * K + k) * 2 + rf];
        if (eb < 0 || eb >= nE) break;
        p = ce_from[eb]; rf = ce_irf[eb];
      }
      path.push_back(p);
    }
    std::reverse(path.begin(), path.end());
  } });
  // M3: the per-path entries are built per path on the pool (same expressions, same order inside a path),
  // then concatenated in path order: identical C_* arrays.
  struct PE { std::vector<i64> net, fan; std::vector<double> cap, dl; std::vector<unsigned char> ok; double sl; };
  std::vector<PE> pe(u.size());
  pfor((i64)u.size(), 16, [&](i64 lo_, i64 hi_) { for (i64 qi = lo_; qi < hi_; qi++) {
    PE& E = pe[qi];
    E.sl = r5(rmin[u[qi]]);
    double t = 0.0;
    for (i64 q : paths[qi]) {
      i64 n = pin_net[q];
      if (!pin_is_drv[q] || n < 0) continue;
      double cap = mx(net_ctot[n], 1e-3);
      double dl = net_rd_kohm[n] * cap / 1000.0;
      t += dl;
      E.net.push_back(n); E.fan.push_back(fan[n]);
      E.cap.push_back(r10_5(cap)); E.dl.push_back(r10_5(dl));
      E.ok.push_back(std::isfinite(cap) && std::isfinite(dl) && std::isfinite(t) && cap >= 0 && dl >= 0);
    }
  } });
  size_t tot = 0; for (auto& E : pe) tot += E.net.size();
  C_net.reserve(tot); C_fan.reserve(tot); C_cap.reserve(tot); C_dl.reserve(tot); C_ok.reserve(tot); C_slack.reserve(u.size());
  for (size_t qi = 0; qi < u.size(); qi++) {
    PE& E = pe[qi];
    C_slack.push_back(E.sl);
    C_net.insert(C_net.end(), E.net.begin(), E.net.end()); C_fan.insert(C_fan.end(), E.fan.begin(), E.fan.end());
    C_cap.insert(C_cap.end(), E.cap.begin(), E.cap.end()); C_dl.insert(C_dl.end(), E.dl.begin(), E.dl.end());
    C_ok.insert(C_ok.end(), E.ok.begin(), E.ok.end());
    C_poff.push_back((i64)C_net.size());
  }
  stats[0] = now() - ta;
  (void)P; (void)nE;
  return (i64)u.size();
}
extern "C" void csta_crit_get(i64* poff, double* pslack, i64* net, i64* fan, double* cap, double* dl, unsigned char* ok) {
  memcpy(poff, C_poff.data(), C_poff.size() * 8); memcpy(pslack, C_slack.data(), C_slack.size() * 8);
  memcpy(net, C_net.data(), C_net.size() * 8); memcpy(fan, C_fan.data(), C_fan.size() * 8);
  memcpy(cap, C_cap.data(), C_cap.size() * 8); memcpy(dl, C_dl.data(), C_dl.size() * 8); memcpy(ok, C_ok.data(), C_ok.size());
}

extern "C" i64 csta_crit_n() { return (i64)C_net.size(); }
