// t_csta.cc: tools/timer5/csta5_kern.cc with csta_crit's second loop on threads (same values: every entry is a pure function of the
// path, written to its own row; snprintf + strtod per entry was the whole of K.crit on ariane v and tile v). The original body stays as csta_crit_orig for the check.
// Build: tools/best10/build_t.sh (flags of tools/timer5/build5.sh).
#define csta_crit csta_crit_orig
#include "../timer5/csta5_kern.cc"
#undef csta_crit

extern "C" i64 csta_crit(i64 npaths, double slack_max, const double* AT, const i64* BEST, const double* rmin,
                         const double* net_ctot, const double* net_rd_kohm, double* stats) {
  double ta = now();
  const i64 K = gi("K"), nE = gi("nE"), nn = gi("nn"), nrow = gi("nrow");
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
  const size_t nu = u.size();
  struct PR { std::vector<i64> net, fan; std::vector<double> cap, dl; std::vector<unsigned char> ok; double slack = 0; };
  std::vector<PR> pr(nu);
  pfor((i64)nu, 16, [&](i64 lo_, i64 hi_) {
    std::vector<i64> path;
    for (i64 qi = lo_; qi < hi_; qi++) {
      i64 row = u[qi], e = ck_pin[row];
      double best = 0; i64 j = 0;
      for (i64 q = 0; q < K * 2; q++) {
        double sl = S_valid[row * K * 2 + q] ? S_reqk[row * K * 2 + q] - AT[(e * K) * 2 + q] : BIG;
        if (q == 0 || sl < best) { best = sl; j = q; }
      }
      i64 k = j / 2, rf = j % 2, p = e;
      path.clear();
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
      PR& o = pr[qi];
      o.slack = r5(rmin[row]);
      double t = 0.0;
      for (i64 q : path) {
        i64 n = pin_net[q];
        if (!pin_is_drv[q] || n < 0) continue;
        double cap = mx(net_ctot[n], 1e-3);
        double dl = net_rd_kohm[n] * cap / 1000.0;
        t += dl;
        o.net.push_back(n); o.fan.push_back(fan[n]);
        o.cap.push_back(r10_5(cap)); o.dl.push_back(r10_5(dl));
        o.ok.push_back(std::isfinite(cap) && std::isfinite(dl) && std::isfinite(t) && cap >= 0 && dl >= 0);
      }
    }
  });
  C_poff.resize(nu + 1); C_poff[0] = 0;
  for (size_t qi = 0; qi < nu; qi++) C_poff[qi + 1] = C_poff[qi] + (i64)pr[qi].net.size();
  const size_t ne = (size_t)C_poff[nu];
  C_net.resize(ne); C_fan.resize(ne); C_slack.resize(nu); C_cap.resize(ne); C_dl.resize(ne); C_ok.resize(ne);
  pfor((i64)nu, 64, [&](i64 lo_, i64 hi_) {
    for (i64 qi = lo_; qi < hi_; qi++) {
      const PR& o = pr[qi]; const size_t b = (size_t)C_poff[qi], m = o.net.size();
      C_slack[qi] = o.slack;
      if (m) {
        memcpy(C_net.data() + b, o.net.data(), m * 8); memcpy(C_fan.data() + b, o.fan.data(), m * 8);
        memcpy(C_cap.data() + b, o.cap.data(), m * 8); memcpy(C_dl.data() + b, o.dl.data(), m * 8); memcpy(C_ok.data() + b, o.ok.data(), m);
      }
    }
  });
  stats[0] = now() - ta;
  return (i64)nu;
}
