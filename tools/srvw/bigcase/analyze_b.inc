// analyze_b.inc: csta_analyze of tools/gapla/csta_kern.cc with the exact memo (M1) and the
// pool-parallel setup / required-time / slack sweeps (M2) described at the top of csta_kern_b.cc.  Included by csta_kern_b.cc in
// place of the original function; every value is computed by the same expression in the same order per output element.
static std::vector<double> S_reqk; static std::vector<unsigned char> S_valid;

static inline bool beq(double a, double b) { uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8); return x == y; }

// ---- M1 state (valid while M_epoch == EPOCH)
static i64 M_epoch = -1;
static bool M_load_ok = false;                 // design passes the single-writer / write-before-read check for the driver model
static std::vector<Dmp> M_res;                 // [nE] last dmp_gate1 result per arc
static std::vector<double> M_key;              // [nE*5] sin c1 c2 rpi ctot it was computed from
static std::vector<unsigned char> M_kpar, M_ok;
static std::vector<DP> M_dps;                  // [P*2] driver model (persistent; written by the group phase)
static std::vector<unsigned char> M_dpc;       // [P*2] driver model changed in this call
static std::vector<double> M_lelm, M_lwd, M_lls;   // [nNE*2] load-delay memo
static std::vector<unsigned char> M_lok;
// ---- M2 grouping of the backward pass (per level: indices ordered by written pin, stable)
static std::vector<i64> G_neo, G_ner, G_ceo, G_cer;    // order and run starts (global arrays)
static std::vector<i64> G_ner_lo, G_cer_lo;            // per level: [lo, hi) into the run arrays
static std::vector<unsigned char> G_ne_par, G_ce_par;  // per level: parallel allowed (writers disjoint from readers)

static void memo_init(i64 P, i64 nE, i64 nlev, i64 nNE, const i64* ce_lo, const i64* ce_from, const i64* ce_to,
                      const i64* g_lo, const i64* g_to, const i64* g_orf, const i64* ne_lo, const i64* ne_drv,
                      const i64* ne_snk, const unsigned char* is_port) {
  M_res.assign((size_t)nE, Dmp());
  M_key.assign((size_t)nE * 5, 0.0); M_kpar.assign((size_t)nE, 0); M_ok.assign((size_t)nE, 0);
  DP d0; memset(&d0, 0, sizeof(DP)); d0.alg = -1;
  M_dps.resize((size_t)P * 2);
  for (auto& d : M_dps) memcpy(&d, &d0, sizeof(DP));
  M_dpc.assign((size_t)P * 2, 0);
  M_lelm.assign((size_t)nNE * 2, 0.0); M_lwd.assign((size_t)nNE * 2, 0.0); M_lls.assign((size_t)nNE * 2, 0.0);
  M_lok.assign((size_t)nNE * 2, 0);
  // driver-model check: each (pin, rf) written by at most one group, at a level <= every level that reads it
  std::vector<i64> wl((size_t)P * 2, -1);
  bool ok = true;
  for (i64 lv = 1; lv < nlev && ok; lv++)
    for (i64 gg = g_lo[lv]; gg < g_lo[lv + 1]; gg++) {
      i64 q = g_to[gg] * 2 + g_orf[gg];
      if (wl[q] >= 0) { ok = false; break; }
      wl[q] = lv;
    }
  for (i64 lv = 1; lv < nlev && ok; lv++)
    for (i64 i = ne_lo[lv]; i < ne_lo[lv + 1]; i++) {
      if (is_port[ne_drv[i]]) continue;
      for (int rf = 0; rf < 2; rf++) { i64 w = wl[ne_drv[i] * 2 + rf]; if (w > lv) { ok = false; break; } }
      if (!ok) break;
    }
  M_load_ok = ok;
  // backward grouping
  std::vector<i64> stamp((size_t)P, -1);
  G_neo.clear(); G_ner.clear(); G_ceo.clear(); G_cer.clear();
  G_ner_lo.assign((size_t)(nlev + 1) * 2, 0); G_cer_lo.assign((size_t)(nlev + 1) * 2, 0);
  G_ne_par.assign((size_t)nlev + 1, 0); G_ce_par.assign((size_t)nlev + 1, 0);
  for (i64 lv = 1; lv < nlev; lv++) {
    for (int ph = 0; ph < 2; ph++) {
      i64 lo = ph == 0 ? ne_lo[lv] : ce_lo[lv], hi = ph == 0 ? ne_lo[lv + 1] : ce_lo[lv + 1];
      const i64* wr = ph == 0 ? ne_drv : ce_from;   // written pin
      const i64* rd = ph == 0 ? ne_snk : ce_to;     // read pin
      std::vector<i64>& O = ph == 0 ? G_neo : G_ceo;
      std::vector<i64>& Rn = ph == 0 ? G_ner : G_cer;
      std::vector<i64>& RL = ph == 0 ? G_ner_lo : G_cer_lo;
      std::vector<unsigned char>& PAR = ph == 0 ? G_ne_par : G_ce_par;
      i64 mark = lv * 2 + ph;
      for (i64 i = lo; i < hi; i++) stamp[wr[i]] = mark;
      bool disj = true;
      for (i64 i = lo; i < hi; i++) if (stamp[rd[i]] == mark) { disj = false; break; }
      PAR[lv] = disj;
      size_t o0 = O.size();
      for (i64 i = lo; i < hi; i++) O.push_back(i);
      std::stable_sort(O.begin() + o0, O.end(), [&](i64 x, i64 y) { return wr[x] < wr[y]; });
      RL[lv * 2] = (i64)Rn.size();
      for (size_t t = o0; t < O.size(); t++) if (t == o0 || wr[O[t]] != wr[O[t - 1]]) Rn.push_back((i64)t);
      RL[lv * 2 + 1] = (i64)Rn.size();
    }
  }
  G_ner.push_back((i64)G_neo.size()); G_cer.push_back((i64)G_ceo.size());   // sentinel: run r spans [R[r], R[r+1])
  // NOTE: the sentinels are global; a level's last run ends at the next level's first run start, which equals the level's end
  // because the order arrays are filled level by level.
}

// outputs are caller-allocated numpy arrays (see csta.py); stats[0..3] = wall of forward / endpoints / rat / total
extern "C" int csta_analyze(double* AT, double* SL, double* gate_d, double* wire_d, i64* BEST, double* net_rd,
                            double* rmin_out, double* ep_sl_out, double* RAT, double* pin_slack, double* net_slack,
                            double* stats) {
  double tA = now();
  TB.A = g<double>("tabA"); TB.B = g<double>("tabB"); TB.V = g<double>("tabV");
  TB.NA = g<i64>("tabNA"); TB.NB = g<i64>("tabNB"); TB.M = gi("tabM"); TB.T = gi("tabT");
  const i64 P = gi("P"), K = gi("K"), nlev = gi("nlev"), nE = gi("nE"), nn = gi("nn"), def_lib = gi("def_lib");
  const i64 *ce_from = g<i64>("ce_from"), *ce_to = g<i64>("ce_to"), *ce_irf = g<i64>("ce_irf"), *ce_orf = g<i64>("ce_orf"),
            *ce_lib = g<i64>("ce_lib"), *ce_td = g<i64>("ce_td"), *ce_ts = g<i64>("ce_ts"), *ce_lo = g<i64>("ce_lo"),
            *g_lo = g<i64>("g_lo"), *g_start = g<i64>("g_start"), *g_to = g<i64>("g_to"), *g_orf = g<i64>("g_orf"),
            *ne_lo = g<i64>("ne_lo"), *ne_drv = g<i64>("ne_drv"), *ne_snk = g<i64>("ne_snk"), *pin_net = g<i64>("pin_net");
  const double *L_vth = g<double>("L_vth"), *L_vl = g<double>("L_vl"), *L_vh = g<double>("L_vh"), *L_der = g<double>("L_der"),
               *L_vin = g<double>("L_vin");
  const unsigned char* is_port = g<unsigned char>("is_port");
  const unsigned char* reg_ck = g<unsigned char>("reg_ck");
  const double* ck_slew = g<double>("ck_slew");
  const i64 nsp = gi("nsp");
  const i64 *sp_pin = g<i64>("sp_pin"), *sp_cls = g<i64>("sp_cls"), *sp_rf = g<i64>("sp_rf");
  const double* sp_at = g<double>("sp_at");
  const double *c1a = g<double>("rc_c1"), *c2a = g<double>("rc_c2"), *rpia = g<double>("rc_rpi"), *ctota = g<double>("rc_ctot"),
               *elma = g<double>("rc_elm");
  const unsigned char* para = g<unsigned char>("rc_par");
  const i64 nNE = ne_lo[nlev];
  const char* me = getenv("CSTA_MEMO");
  const bool memo_on = !(me && me[0] == '0');
  if (M_epoch != EPOCH) {
    memo_init(P, nE, nlev, nNE, ce_lo, ce_from, ce_to, g_lo, g_to, g_orf, ne_lo, ne_drv, ne_snk, is_port);
    M_epoch = EPOCH;
    if (getenv("CSTA_VERBOSE")) fprintf(stderr, "[csta_b] memo init: P %lld nE %lld nNE %lld load_memo %d\n", (long long)P,
                                        (long long)nE, (long long)nNE, (int)M_load_ok);
  }
  const bool memo_arc = memo_on, memo_load = memo_on && M_load_ok;
  const i64 KK = K * 2;
  pfor(P, 4096, [&](i64 lo_, i64 hi_) {
    for (i64 i = lo_ * KK; i < hi_ * KK; i++) { AT[i] = NEG; BEST[i] = -1; }
    for (i64 p = lo_; p < hi_; p++) {
      SL[p * 2] = 0.0; SL[p * 2 + 1] = 0.0; wire_d[p * 2] = 0.0; wire_d[p * 2 + 1] = 0.0;
      M_dpc[p * 2] = 0; M_dpc[p * 2 + 1] = 0;
    }
  });
  pfor(nE, 1 << 16, [&](i64 lo_, i64 hi_) { for (i64 i = lo_; i < hi_; i++) gate_d[i] = 0.0; });
  pfor(nn, 1 << 16, [&](i64 lo_, i64 hi_) { for (i64 i = lo_; i < hi_; i++) net_rd[i] = 0.0; });
  for (i64 i = 0; i < nsp; i++) AT[(sp_pin[i] * K + sp_cls[i]) * 2 + sp_rf[i]] = sp_at[i];
  pfor(P, 1 << 14, [&](i64 lo_, i64 hi_) {
    for (i64 p = lo_; p < hi_; p++) if (reg_ck[p]) { SL[p * 2] = ck_slew[p * 2]; SL[p * 2 + 1] = ck_slew[p * 2 + 1]; }
  });
  double tS = now(), t_arc = 0.0;
  std::atomic<i64> n_arc{0}, n_load{0};
  i64 maxg = 0;
  for (i64 lv = 1; lv < nlev; lv++) maxg = std::max(maxg, g_lo[lv + 1] - g_lo[lv]);
  std::vector<i64> pick((size_t)maxg);
  Dmp* res = M_res.data();
  for (i64 lv = 1; lv < nlev; lv++) {
    i64 a = ce_lo[lv], b = ce_lo[lv + 1];
    if (b > a) {
      double t0 = now();
      pfor(b - a, 64, [&](i64 lo_, i64 hi_) { i64 cnt = 0; for (i64 e = a + lo_; e < a + hi_; e++) {
        i64 frm = ce_from[e], to = ce_to[e], irf = ce_irf[e], orf = ce_orf[e], li = ce_lib[e];
        i64 net = pin_net[to];
        Arc A;
        A.td = ce_td[e]; A.ts = ce_ts[e]; A.sin = SL[frm * 2 + irf];
        A.c2 = c2a[net * 2 + orf]; A.rpi = rpia[net * 2 + orf]; A.c1 = c1a[net * 2 + orf];
        A.vth = L_vth[li * 2 + orf]; A.vl = L_vl[li * 2 + orf]; A.vh = L_vh[li * 2 + orf]; A.der = L_der[li];
        bool par = para[net] != 0; double ct = ctota[net * 2 + orf];
        double* kk = &M_key[(size_t)e * 5];
        if (!(memo_arc && M_ok[e] && M_kpar[e] == (unsigned char)par && beq(kk[0], A.sin) && beq(kk[1], A.c1) &&
              beq(kk[2], A.c2) && beq(kk[3], A.rpi) && beq(kk[4], ct))) {
          dmp_gate1(A, par, ct, res[e]);
          kk[0] = A.sin; kk[1] = A.c1; kk[2] = A.c2; kk[3] = A.rpi; kk[4] = ct; M_kpar[e] = par; M_ok[e] = 1; cnt++;
        }
        gate_d[e] = res[e].delay;
      } n_arc += cnt; });
      t_arc += now() - t0;
      i64 ga = g_lo[lv], gb = g_lo[lv + 1];
      pfor(gb - ga, 256, [&](i64 lo_, i64 hi_) { for (i64 gg = ga + lo_; gg < ga + hi_; gg++) {
        i64 s0 = g_start[gg], s1 = (gg + 1 < gb) ? g_start[gg + 1] : b;  // arcs [s0,s1)
        i64 gto = g_to[gg], gorf = g_orf[gg];
        double smax = res[s0].slew;
        for (i64 e = s0 + 1; e < s1; e++) smax = mx(smax, res[e].slew);
        SL[gto * 2 + gorf] = smax;
        i64 pk = -1;
        for (i64 e = s0; e < s1; e++) if (res[e].slew >= smax) { pk = e; break; }
        if (pk < 0) pk = b;  // python would index out of range; never expected
        pick[gg - ga] = pk;
        DP nd; memset(&nd, 0, sizeof(DP)); nd.alg = -1;
        if (pk < b) {
          const Dmp& r = res[pk];
          nd.slew = r.slew; nd.t0 = r.t0; nd.dt = r.dt; nd.k0 = r.k0; nd.k1 = r.k1; nd.k2 = r.k2; nd.k3 = r.k3; nd.k4 = r.k4;
          nd.p1 = r.p1; nd.p2 = r.p2; nd.vub = r.vub; nd.vod = r.vod; nd.vth = r.vth; nd.vl = r.vl; nd.vh = r.vh; nd.der = r.der;
          nd.alg = r.alg; nd.valid = r.valid;
        }
        DP& d = M_dps[gto * 2 + gorf];
        if (memcmp(&d, &nd, sizeof(DP)) != 0) { memcpy(&d, &nd, sizeof(DP)); M_dpc[gto * 2 + gorf] = 1; }
        for (i64 k = 0; k < K; k++) {
          double amax = NEG; bool first = true;
          for (i64 e = s0; e < s1; e++) {
            double c = AT[(ce_from[e] * K + k) * 2 + ce_irf[e]] + res[e].delay;
            amax = first ? c : mx(amax, c); first = false;
          }
          AT[(gto * K + k) * 2 + gorf] = amax;
          i64 best = nE;
          for (i64 e = s0; e < s1; e++) {
            double c = AT[(ce_from[e] * K + k) * 2 + ce_irf[e]] + res[e].delay;
            if (c >= amax - 1e-12) { best = e; break; }
          }
          BEST[(gto * K + k) * 2 + gorf] = best;
        }
      } });
      for (i64 gg = ga; gg < gb; gg++) {  // net_rd (np.maximum.at, sequential)
        i64 pk = pick[gg - ga];
        if (pk < b) { i64 net = pin_net[ce_to[pk]]; net_rd[net] = mx(net_rd[net], res[pk].rd); }
      }
    }
    a = ne_lo[lv]; b = ne_lo[lv + 1];
    if (b > a) {
      i64 nb = b - a;
      pfor(2 * nb, 128, [&](i64 lo_, i64 hi_) { i64 cnt = 0; for (i64 j = lo_; j < hi_; j++) {
        i64 rf = j / nb, i = a + (j % nb);
        i64 drv = ne_drv[i], snk = ne_snk[i];
        double elm = elma[snk * 2 + rf], wd, ls;
        if (is_port[drv]) {      // the original computes dmp_load1 here too and overwrites both outputs: skipped
          bool okp = std::isfinite(elm);
          double vin = L_vin[def_lib * 2 + rf], vlo = L_vl[def_lib * 2 + rf], vhi = L_vh[def_lib * 2 + rf];
          wd = okp ? -elm * std::log(1.0 - vin) : 0.0;
          ls = SL[drv * 2 + rf] + (okp ? elm * std::log((1 - vlo) / (1 - vhi)) / L_der[def_lib] : 0.0);
        } else {
          size_t q = (size_t)i * 2 + rf;
          if (memo_load && M_lok[q] && !M_dpc[drv * 2 + rf] && beq(M_lelm[q], elm)) { wd = M_lwd[q]; ls = M_lls[q]; }
          else {
            dmp_load1(M_dps[drv * 2 + rf], elm, &wd, &ls);
            M_lelm[q] = elm; M_lwd[q] = wd; M_lls[q] = ls; M_lok[q] = 1; cnt++;
          }
        }
        wire_d[snk * 2 + rf] = wd;
        SL[snk * 2 + rf] = ls;
        for (i64 k = 0; k < K; k++) AT[(snk * K + k) * 2 + rf] = AT[(drv * K + k) * 2 + rf] + wd;
      } n_load += cnt; });
    }
  }
  double tB = now();
  // ---- endpoints
  const i64 nrow = gi("nrow");
  const i64 *ck_pin = g<i64>("ck_pin"), *ck_tr = g<i64>("ck_tr"), *ck_tf = g<i64>("ck_tf"), *ck_type = g<i64>("ck_type"),
            *ck_evt = g<i64>("ck_evt");
  const double *ck_cks = g<double>("ck_cks"), *ck_off = g<double>("ck_off"), *REL = g<double>("REL");  // REL [K][nevt]
  const i64 nevt = gi("nevt");
  S_reqk.assign((size_t)nrow * K * 2, 0.0); S_valid.assign((size_t)nrow * K * 2, 0);
  std::vector<double>& reqk = S_reqk; std::vector<unsigned char>& valid = S_valid;
  pfor(nrow, 4096, [&](i64 lo_, i64 hi_) { for (i64 r = lo_; r < hi_; r++) {
    i64 p = ck_pin[r];
    double setup[2];
    for (int rf = 0; rf < 2; rf++) {
      i64 tid = rf == 0 ? ck_tr[r] : ck_tf[r];
      setup[rf] = 0.0;
      if (tid >= 0) setup[rf] = lut(tid, ck_cks[r], SL[p * 2 + rf]);
      else if (ck_type[r] <= 1) setup[rf] = NaN;
    }
    double rmin = BIG;
    for (i64 k = 0; k < K; k++) {
      double req = REL[k * nevt + ck_evt[r]];
      for (int rf = 0; rf < 2; rf++) {
        double rk = req - setup[rf] - ck_off[r];
        double at = AT[(p * K + k) * 2 + rf];
        bool v = (at > NEG / 2) && (req < BIG / 2) && std::isfinite(rk);
        reqk[(r * K + k) * 2 + rf] = rk;
        valid[(r * K + k) * 2 + rf] = v;
        double sl = v ? rk - at : BIG;
        rmin = mn(rmin, sl);
      }
    }
    rmin_out[r] = rmin;
  } });
  pfor(P, 1 << 16, [&](i64 lo_, i64 hi_) { for (i64 p = lo_; p < hi_; p++) ep_sl_out[p] = BIG; });
  for (i64 r = 0; r < nrow; r++) ep_sl_out[ck_pin[r]] = mn(ep_sl_out[ck_pin[r]], rmin_out[r]);
  double tC = now();
  // ---- required times
  pfor(P, 4096, [&](i64 lo_, i64 hi_) { for (i64 i = lo_ * KK; i < hi_ * KK; i++) RAT[i] = BIG; });
  for (i64 r = 0; r < nrow; r++)
    for (i64 k = 0; k < K; k++)
      for (int rf = 0; rf < 2; rf++) {
        i64 ix = (ck_pin[r] * K + k) * 2 + rf;
        RAT[ix] = mn(RAT[ix], valid[(r * K + k) * 2 + rf] ? reqk[(r * K + k) * 2 + rf] : BIG);
      }
  for (i64 lv = nlev - 1; lv >= 1; lv--) {
    if (G_ne_par[lv]) {
      i64 r0 = G_ner_lo[lv * 2], r1 = G_ner_lo[lv * 2 + 1];
      pfor(r1 - r0, 64, [&](i64 lo_, i64 hi_) { for (i64 r = r0 + lo_; r < r0 + hi_; r++) {
        for (i64 t = G_ner[r]; t < G_ner[r + 1]; t++) {
          i64 i = G_neo[t], drv = ne_drv[i], snk = ne_snk[i];
          for (i64 k = 0; k < K; k++)
            for (int rf = 0; rf < 2; rf++) {
              i64 ix = (drv * K + k) * 2 + rf;
              RAT[ix] = mn(RAT[ix], RAT[(snk * K + k) * 2 + rf] - wire_d[snk * 2 + rf]);
            }
        }
      } });
    } else {
      for (i64 i = ne_lo[lv]; i < ne_lo[lv + 1]; i++) {
        i64 drv = ne_drv[i], snk = ne_snk[i];
        for (i64 k = 0; k < K; k++)
          for (int rf = 0; rf < 2; rf++) {
            i64 ix = (drv * K + k) * 2 + rf;
            RAT[ix] = mn(RAT[ix], RAT[(snk * K + k) * 2 + rf] - wire_d[snk * 2 + rf]);
          }
      }
    }
    if (G_ce_par[lv]) {
      i64 r0 = G_cer_lo[lv * 2], r1 = G_cer_lo[lv * 2 + 1];
      pfor(r1 - r0, 64, [&](i64 lo_, i64 hi_) { for (i64 r = r0 + lo_; r < r0 + hi_; r++) {
        for (i64 t = G_cer[r]; t < G_cer[r + 1]; t++) {
          i64 e = G_ceo[t];
          i64 to = ce_to[e], orf = ce_orf[e], fr = ce_from[e], irf = ce_irf[e];
          for (i64 k = 0; k < K; k++) {
            double v = RAT[(to * K + k) * 2 + orf] - gate_d[e];
            i64 ix = (fr * K + k) * 2 + irf;
            RAT[ix] = mn(RAT[ix], v);
          }
        }
      } });
    } else {
      for (i64 e = ce_lo[lv]; e < ce_lo[lv + 1]; e++) {
        i64 to = ce_to[e], orf = ce_orf[e], fr = ce_from[e], irf = ce_irf[e];
        for (i64 k = 0; k < K; k++) {
          double v = RAT[(to * K + k) * 2 + orf] - gate_d[e];
          i64 ix = (fr * K + k) * 2 + irf;
          RAT[ix] = mn(RAT[ix], v);
        }
      }
    }
  }
  pfor(nn, 1 << 16, [&](i64 lo_, i64 hi_) { for (i64 i = lo_; i < hi_; i++) net_slack[i] = BIG; });
  pfor(P, 4096, [&](i64 lo_, i64 hi_) { for (i64 p = lo_; p < hi_; p++) {
    double m = BIG;
    for (i64 k = 0; k < K; k++)
      for (int rf = 0; rf < 2; rf++) {
        i64 ix = (p * K + k) * 2 + rf;
        bool ok = (AT[ix] > NEG / 2) && (RAT[ix] < BIG / 2);
        m = mn(m, ok ? RAT[ix] - AT[ix] : BIG);
      }
    pin_slack[p] = m;
  } });
  for (i64 p = 0; p < P; p++) {
    i64 n = pin_net[p]; if (n < 0) n += nn;
    net_slack[n] = mn(net_slack[n], pin_slack[p]);
  }
  double tD = now();
  stats[0] = tB - tA; stats[1] = tC - tB; stats[2] = tD - tC; stats[3] = tD - tA;
  stats[4] = tS - tA; stats[5] = t_arc; stats[6] = (double)n_arc.load(); stats[7] = (double)n_load.load();
  return 0;
}
