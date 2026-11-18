// promote_kern.cc: exact C++ port of route_export.LayerAssigner.assign_path with timing_promote.TimingLA costs
// (run_cost = w4*sum(wire_marginal) + tw*delay_ns ; via_cost with timing term), batched over many polylines.
// Build: clang++ -O2 -ffp-contract=off -shared -fPIC -o libpromote_kern.dylib promote_kern.cc   (no fast-math, no FMA)
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
extern "C" {
static const double CAP_EPS = 1e-3, EXP_CLIP = 700.0;
struct Ctx {
  const double *d, *c, *u, *R, *C, *Rv; const int32_t* dirs;
  const int64_t *sd, *sc; int L, X, Y; double w4, um, tw, cl, rd, E05M1, E025M1, E15;
};
static inline double pw(const double* a, long n) {   // numpy pairwise_sum
  if (n < 8) { double r = 0.; for (long i = 0; i < n; i++) r += a[i]; return r; }
  if (n <= 128) {
    double r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
    long i; for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  long n2 = n / 2; n2 -= n2 % 8;
  return pw(a, n2) + pw(a + n2, n - n2);
}
double np_sum(const double* a, long n) { return pw(a, n); }   // verified vs numpy: plain pairwise
static inline double wm(const Ctx& k, int z, long x, long y) {
  double dz = k.d[z*k.sd[0]+x*k.sd[1]+y*k.sd[2]], cz = k.c[z*k.sc[0]+x*k.sc[1]+y*k.sc[2]], u = k.u[z];
  if (cz > CAP_EPS) {
    return u * std::exp(std::min(0.25 * dz - 0.5 * cz, EXP_CLIP)) * k.E05M1;
  } else if (dz > 0) {
    return u * (std::exp(std::min(0.75 * (dz + 2), EXP_CLIP)) - std::exp(std::min(0.75 * dz, EXP_CLIP)));
  }
  return u * k.E15;
}
static double run_cost(const Ctx& k, std::vector<double>& buf, int dr, int fixed, int a, int b, int z) {
  long n = b - a;
  for (long i = 0; i < n; i++) buf[i] = dr == 0 ? wm(k, z, a + i, fixed) : wm(k, z, fixed, a + i);
  double ov = np_sum(buf.data(), n);
  double l = (double)n * k.um;
  double dly = l * (k.R[z] * k.cl + k.C[z] * k.rd) / 1000.0 + k.R[z] * k.C[z] * l * l / 2.0 / 1000.0;
  return k.w4 * ov + k.tw * dly;
}
static double via_cost(const Ctx& k, int x, int y, int za, int zb) {
  int lo = za <= zb ? za : zb, hi = za <= zb ? zb : za;
  double tot = 0.0;
  for (int z = lo; z < hi; z++) {
    tot += k.tw * k.Rv[z] * 2.0 / 1000.0;
    if (z != 0 && (k.dirs[z] == 0 || k.dirs[z] == 1)) {
      double dz = k.d[z*k.sd[0]+x*k.sd[1]+y*k.sd[2]], cz = k.c[z*k.sc[0]+x*k.sc[1]+y*k.sc[2]], u = k.u[z], m;
      if (cz > CAP_EPS) m = 2.0 * u * std::exp(std::min(0.25 * dz - 0.5 * cz, EXP_CLIP)) * k.E025M1;
      else {
        double base = dz > 0 ? std::exp(std::min(0.75 * dz, EXP_CLIP)) : 0.0;
        m = 2.0 * u * (std::exp(std::min(0.75 * (dz + 1), EXP_CLIP)) - base);
      }
      tot += k.w4 * m;
    }
  }
  return tot;
}
static double stack_cost(const Ctx& k, int x, int y, int lo, int hi, int z) {
  if (lo < 0) return 0.0;
  if (z > hi) return via_cost(k, x, y, hi, z);
  if (z < lo) return via_cost(k, x, y, z, lo);
  return 0.0;
}
// runs: int32[R,8] (dir,fixed,lo,hi,sx,sy,ex,ey); poff[P+1]; span int32[P,4] (slo,shi,elo,ehi; slo<0 = None)
// out: layers int32[R], cost double[P]
void assign_batch(const double* d, const double* c, const double* u, const double* R, const double* C, const double* Rv,
                  const double* cst, const int32_t* dirs, const int64_t* sd, const int64_t* sc, int L, int X, int Y, double w4, double um, double tw, double cl, double rd,
                  const int32_t* runs, const int32_t* poff, const int32_t* span, int P, int32_t* layers, double* cost) {
  Ctx k{d, c, u, R, C, Rv, dirs, sd, sc, L, X, Y, w4, um, tw, cl, rd, cst[0], cst[1], cst[2]};
  std::vector<int> H, V;
  for (int z = 1; z < L; z++) (dirs[z] == 0 ? H : V).push_back(z);
  std::vector<double> buf(std::max(X, Y) + 8);
  for (int p = 0; p < P; p++) {
    int r0 = poff[p], r1 = poff[p + 1], Rn = r1 - r0;
    if (Rn == 0) { cost[p] = 0.0; continue; }
    std::vector<std::vector<double>> rc(Rn);
    std::vector<const std::vector<int>*> cand(Rn);
    for (int i = 0; i < Rn; i++) {
      const int32_t* r = runs + 8 * (r0 + i);
      cand[i] = r[0] == 0 ? &H : &V;
      for (int z : *cand[i]) rc[i].push_back(run_cost(k, buf, r[0], r[1], r[2], r[3], z));
    }
    const int32_t* sp = span + 4 * p;
    const int32_t* r = runs + 8 * r0;
    std::vector<double> best;
    for (size_t kk = 0; kk < cand[0]->size(); kk++)
      best.push_back(rc[0][kk] + stack_cost(k, r[4], r[5], sp[0], sp[1], (*cand[0])[kk]));
    std::vector<std::vector<int>> back;
    for (int i = 1; i < Rn; i++) {
      const int32_t* ri = runs + 8 * (r0 + i);
      std::vector<double> nb; std::vector<int> bk;
      for (size_t kk = 0; kk < cand[i]->size(); kk++) {
        int z = (*cand[i])[kk]; int bj = 0; double bv = 0;
        for (size_t j = 0; j < cand[i - 1]->size(); j++) {
          double v = best[j] + via_cost(k, ri[4], ri[5], (*cand[i - 1])[j], z);
          if (j == 0 || v < bv) { bv = v; bj = (int)j; }
        }
        nb.push_back(bv + rc[i][kk]); bk.push_back(bj);
      }
      best = nb; back.push_back(bk);
    }
    const int32_t* rl = runs + 8 * (r1 - 1);
    int ak = 0; double fv = 0;
    for (size_t kk = 0; kk < cand[Rn - 1]->size(); kk++) {
      double v = best[kk] + stack_cost(k, rl[6], rl[7], sp[2], sp[3], (*cand[Rn - 1])[kk]);
      if (kk == 0 || v < fv) { fv = v; ak = (int)kk; }
    }
    int kk = ak; layers[r0 + Rn - 1] = (*cand[Rn - 1])[kk];
    for (int i = Rn - 1; i > 0; i--) { kk = back[i - 1][kk]; layers[r0 + i - 1] = (*cand[i - 1])[kk]; }
    cost[p] = fv;
  }
}
}
