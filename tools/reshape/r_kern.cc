// r_kern.cc: the chain_h incremental timer library, built in tools/reshape for the hosts.
// = tools/fast3x/h_csta_kern.cc verbatim (included) + w3_inc_backward() copied from tools/wave3/ovf2/w3_inc_kern.cc (the
// required-time pass on the current committed state; chain_h never calls it, so chain_h's results are unchanged).
// Build: tools/reshape/build_r.sh (-O3 -ffp-contract=off -fno-fast-math).
#include "../fast3x/h_csta_kern.cc"
extern "C" void w3_inc_backward(double* stats) {
  double tC = now();
  DA d = da();
  const i64 P = d.P, K = d.K, nn = d.nn;
    const i64 *ce_lo = d.ce_lo, *ne_lo = d.ne_lo;
    for (i64 i = 0; i < P * K * 2; i++) I_RAT[i] = BIG;
    for (i64 r = 0; r < d.nrow; r++)
      for (i64 k = 0; k < K; k++)
        for (int rf = 0; rf < 2; rf++) {
          i64 ix = (d.ck_pin[r] * K + k) * 2 + rf;
          I_RAT[ix] = mn(I_RAT[ix], S_valid[(r * K + k) * 2 + rf] ? S_reqk[(r * K + k) * 2 + rf] : BIG);
        }
    for (i64 lv = d.nlev - 1; lv >= 1; lv--) {
      for (i64 i = ne_lo[lv]; i < ne_lo[lv + 1]; i++) {
        i64 drv = d.ne_drv[i], snk = d.ne_snk[i];
        for (i64 k = 0; k < K; k++)
          for (int rf = 0; rf < 2; rf++) {
            i64 ix = (drv * K + k) * 2 + rf;
            I_RAT[ix] = mn(I_RAT[ix], I_RAT[(snk * K + k) * 2 + rf] - I_wd[snk * 2 + rf]);
          }
      }
      for (i64 e = ce_lo[lv]; e < ce_lo[lv + 1]; e++) {
        i64 to = d.ce_to[e], orf = d.ce_orf[e], fr = d.ce_from[e], irf = d.ce_irf[e];
        for (i64 k = 0; k < K; k++) {
          double v = I_RAT[(to * K + k) * 2 + orf] - I_gd[e];
          i64 ix = (fr * K + k) * 2 + irf;
          I_RAT[ix] = mn(I_RAT[ix], v);
        }
      }
    }
    for (i64 i = 0; i < nn; i++) I_nsl[i] = BIG;
    for (i64 p = 0; p < P; p++) {
      double m = BIG;
      for (i64 k = 0; k < K; k++)
        for (int rf = 0; rf < 2; rf++) {
          i64 ix = (p * K + k) * 2 + rf;
          bool ok = (I_AT[ix] > NEG / 2) && (I_RAT[ix] < BIG / 2);
          m = mn(m, ok ? I_RAT[ix] - I_AT[ix] : BIG);
        }
      I_psl[p] = m;
      i64 n = d.pin_net[p]; if (n < 0) n += nn;
      I_nsl[n] = mn(I_nsl[n], m);
    }
  stats[0] = now() - tC;
}
