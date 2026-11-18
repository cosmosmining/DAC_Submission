// gt_kern.cu: the exact csta timer (tools/gapla/csta_kern.cc, analyze part) as
// one-thread-per-element kernels, with the local derivatives of every gate arc and wire edge computed in the same
// thread by forward-mode dual numbers, and a reverse level sweep that gives d(AT, SL) -> d(c1, c2, rpi, ctot, elm).
// Values: the kernel's own scalar code (same formulas and order; build without FMA contraction).
// Derivatives: the rules of tools/gnnfix/dtimer/dtimer.py (implicit gradients of the Newton and root solves with the
// true Jacobian at the solution; torch conventions for clamp / abs / minimum / where / nan_to_num; max/min ties split
// evenly; the driver record follows the first arc reaching the group's slew).
// Same source, two builds:
//   GPU: nvcc -O3 -arch=sm_75 -fmad=false -Xcompiler -fPIC -shared gt_kern.cu -o libgt_cuda.so
//   CPU: g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -fPIC -shared -pthread -x c++ gt_kern.cu -o libgt_cpu.so
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <algorithm>
#include <chrono>
typedef int64_t i64;
typedef unsigned char u8;
#ifdef __CUDACC__
#define HD __host__ __device__ inline
#else
#define HD inline
#endif

#define NEGV (-1e30)
#define BIGV (1e30)
HD double mknan() { unsigned long long u = 0x7ff8000000000000ULL; double d; memcpy(&d, &u, 8); return d; }
HD bool fin(double x) { return (x - x) == 0.0; }
HD double mx(double a, double b) { return (a != a || b != b) ? mknan() : (a >= b ? a : b); }
HD double mn(double a, double b) { return (a != a || b != b) ? mknan() : (a <= b ? a : b); }
HD double nn0d(double v) {
  if (v != v) return 0.0;
  if (!fin(v)) return v > 0 ? 1.7976931348623157e308 : -1.7976931348623157e308;
  return v;
}

// ---------------------------------------------------------------- dual numbers (N slots; arcs 5, wire edges 6, implicit solves 8)
template <int N> struct Dn {
  static const int NS = N;
  double v; double d[N];
  HD Dn() {}
  HD Dn(double x) { v = x; for (int i = 0; i < N; i++) d[i] = 0.0; }
};
typedef Dn<8> D8; typedef Dn<6> D6; typedef Dn<5> D5;
template <int N> HD Dn<N> seed(double x, int i) { Dn<N> r(x); r.d[i] = 1.0; return r; }
template <int N> HD Dn<N> operator+(const Dn<N>& a, const Dn<N>& b) { Dn<N> r; r.v = a.v + b.v; for (int i = 0; i < N; i++) r.d[i] = a.d[i] + b.d[i]; return r; }
template <int N> HD Dn<N> operator-(const Dn<N>& a, const Dn<N>& b) { Dn<N> r; r.v = a.v - b.v; for (int i = 0; i < N; i++) r.d[i] = a.d[i] - b.d[i]; return r; }
template <int N> HD Dn<N> operator-(const Dn<N>& a) { Dn<N> r; r.v = -a.v; for (int i = 0; i < N; i++) r.d[i] = -a.d[i]; return r; }
template <int N> HD Dn<N> operator*(const Dn<N>& a, const Dn<N>& b) { Dn<N> r; r.v = a.v * b.v; for (int i = 0; i < N; i++) r.d[i] = a.d[i] * b.v + a.v * b.d[i]; return r; }
template <int N> HD Dn<N> operator/(const Dn<N>& a, const Dn<N>& b) {
  Dn<N> r; r.v = a.v / b.v; double ib = 1.0 / b.v;
  for (int i = 0; i < N; i++) r.d[i] = a.d[i] * ib - a.v * b.d[i] * ib * ib;
  return r;
}
template <int N> HD Dn<N> operator+(const Dn<N>& a, double b) { Dn<N> r = a; r.v = a.v + b; return r; }
template <int N> HD Dn<N> operator+(double a, const Dn<N>& b) { Dn<N> r = b; r.v = a + b.v; return r; }
template <int N> HD Dn<N> operator-(const Dn<N>& a, double b) { Dn<N> r = a; r.v = a.v - b; return r; }
template <int N> HD Dn<N> operator-(double a, const Dn<N>& b) { Dn<N> r; r.v = a - b.v; for (int i = 0; i < N; i++) r.d[i] = -b.d[i]; return r; }
template <int N> HD Dn<N> operator*(const Dn<N>& a, double b) { Dn<N> r; r.v = a.v * b; for (int i = 0; i < N; i++) r.d[i] = a.d[i] * b; return r; }
template <int N> HD Dn<N> operator*(double a, const Dn<N>& b) { Dn<N> r; r.v = a * b.v; for (int i = 0; i < N; i++) r.d[i] = a * b.d[i]; return r; }
template <int N> HD Dn<N> operator/(const Dn<N>& a, double b) { Dn<N> r; r.v = a.v / b; for (int i = 0; i < N; i++) r.d[i] = a.d[i] / b; return r; }
template <int N> HD Dn<N> operator/(double a, const Dn<N>& b) {
  Dn<N> r; r.v = a / b.v; double ib = 1.0 / b.v;
  for (int i = 0; i < N; i++) r.d[i] = -a * b.d[i] * ib * ib;
  return r;
}
HD double VAL(double x) { return x; }
template <int N> HD double VAL(const Dn<N>& x) { return x.v; }
HD double EXP(double x) { return exp(x); }
template <int N> HD Dn<N> EXP(const Dn<N>& a) { double e = exp(a.v); Dn<N> r; r.v = e; for (int i = 0; i < N; i++) r.d[i] = a.d[i] * e; return r; }
HD double LOG(double x) { return log(x); }
template <int N> HD Dn<N> LOG(const Dn<N>& a) { Dn<N> r; r.v = log(a.v); for (int i = 0; i < N; i++) r.d[i] = a.d[i] / a.v; return r; }
HD double SQRT(double x) { return sqrt(x); }
template <int N> HD Dn<N> SQRT(const Dn<N>& a) { double s = sqrt(a.v); Dn<N> r; r.v = s; for (int i = 0; i < N; i++) r.d[i] = a.d[i] / (2.0 * s); return r; }
HD double ABS(double x) { return fabs(x); }
template <int N> HD Dn<N> ABS(const Dn<N>& a) {
  double sg = a.v > 0 ? 1.0 : (a.v < 0 ? -1.0 : 0.0);
  Dn<N> r; r.v = fabs(a.v); for (int i = 0; i < N; i++) r.d[i] = a.d[i] * sg; return r;
}
// torch.clamp(x, min=m): gradient passes where x >= m
HD double CLMIN(double x, double m) { return mx(x, m); }
template <int N> HD Dn<N> CLMIN(const Dn<N>& a, double m) {
  if (a.v != a.v) return Dn<N>(mknan());
  if (a.v >= m) return a;
  return Dn<N>(m);
}
// torch.minimum(a, b): ties split the gradient in half
HD double TMIN(double a, double b) { return mn(a, b); }
template <int N> HD Dn<N> TMIN(const Dn<N>& a, const Dn<N>& b) {
  if (a.v != a.v || b.v != b.v) return Dn<N>(mknan());
  if (a.v < b.v) return a;
  if (b.v < a.v) return b;
  Dn<N> r; r.v = a.v; for (int i = 0; i < N; i++) r.d[i] = 0.5 * (a.d[i] + b.d[i]); return r;
}
HD double NN0(double v) { return nn0d(v); }
template <int N> HD Dn<N> NN0(const Dn<N>& a) { if (!fin(a.v)) return Dn<N>(nn0d(a.v)); return a; }
// widen 5 slots to 8 (slots 5.. zero)
HD D8 W8(const D5& a) { D8 r(a.v); for (int i = 0; i < 5; i++) r.d[i] = a.d[i]; return r; }
template <class S> HD S ex(const S& x) { return EXP(CLMIN(x, -700.0)); }

// ---------------------------------------------------------------- tables (lut of the kernel)
struct Tab { const double *A, *B, *V; const i64 *NA, *NB; i64 M, T; };
template <class S> HD S lut(const Tab& TB, i64 tid, const S& x, const S& y) {
  if (tid < 0) tid += TB.T;
  const double* A = TB.A + tid * TB.M;
  const double* B = TB.B + tid * TB.M;
  i64 na = TB.NA[tid], nb = TB.NB[tid];
  i64 ka = 0, kb = 0;
  double xv = VAL(x), yv = VAL(y);
  for (i64 j = 1; j < na; j++) ka += (xv > A[j]);
  for (i64 j = 1; j < nb; j++) kb += (yv > B[j]);
  if (ka > na - 2) ka = na - 2;
  if (kb > nb - 2) kb = nb - 2;
  double a0 = A[ka], a1 = A[ka + 1], b0 = B[kb], b1 = B[kb + 1];
  S fa = (x - a0) / (a1 != a0 ? a1 - a0 : 1.0);
  S fb = (y - b0) / (b1 != b0 ? b1 - b0 : 1.0);
  const double* V = TB.V;
  i64 M = TB.M, base = tid * (M * M) + ka * M + kb;
  return (1.0 - fa) * (1.0 - fb) * V[base] + (1.0 - fa) * fb * V[base + 1] + fa * (1.0 - fb) * V[base + M] +
         fa * fb * V[base + M + 1];
}

// ---------------------------------------------------------------- waveforms
template <class S> HD S y0f(const S& u, const S& rc) { return u - rc * (1.0 - ex(-u / rc)); }
template <class S> HD S yramp(const S& t, const S& t0, const S& dt, const S& rc) {
  S t1 = t - t0;
  S a = y0f(CLMIN(t1, 0.0), rc), b = y0f(CLMIN(t1 - dt, 0.0), rc);
  return VAL(t1) <= 0 ? S(0.0) : (VAL(t1) <= VAL(dt) ? a / dt : (a - b) / dt);
}
HD void dyramp(double t, double t0, double dt, double cl, double rd, double* o) {
  double rc = rd * cl, t1 = t - t0, u1 = mx(t1, 0.0), u2 = mx(t1 - dt, 0.0);
  double y0a = y0f(u1, rc), y0b = y0f(u2, rc);
  double dta = 1.0 - ex(-u1 / rc), dtb = 1.0 - ex(-u2 / rc);
  double dca = rd * ((1.0 + u1 / rc) * ex(-u1 / rc) - 1.0);
  double dcb = rd * ((1.0 + u2 / rc) * ex(-u2 / rc) - 1.0);
  bool in_ramp = t1 <= dt;
  double d_t0 = in_ramp ? -dta / dt : -(dta - dtb) / dt;
  double d_dt = in_ramp ? -y0a / (dt * dt) : -(y0a + y0b) / (dt * dt) + dtb / dt;
  double d_cl = in_ramp ? dca / dt : (dca - dcb) / dt;
  bool z = t1 <= 0;
  o[0] = z ? 0.0 : d_t0; o[1] = z ? 0.0 : d_dt; o[2] = z ? 0.0 : d_cl;
}
template <class S> struct VoP {
  S t0, dt, k0, k1, k2, k3, k4, p1, p2;
  HD void V0(const S& u, S* vv, S* dd) const {
    S e1 = ex(-p1 * u), e2 = ex(-p2 * u);
    *vv = k0 * (k1 + k2 * u + k3 * e1 + k4 * e2);
    *dd = k0 * (k2 - k3 * p1 * e1 - k4 * p2 * e2);
  }
  HD void f(const S& t, S* v, S* d) const {
    S t1 = t - t0, va, da, vb, db;
    V0(CLMIN(t1, 0.0), &va, &da);
    V0(CLMIN(t1 - dt, 0.0), &vb, &db);
    *v = VAL(t1) <= 0 ? S(0.0) : (VAL(t1) <= VAL(dt) ? va / dt : (va - vb) / dt);
    *d = VAL(t1) <= 0 ? S(0.0) : (VAL(t1) <= VAL(dt) ? da / dt : (da - db) / dt);
  }
};
template <class S> struct VlP {
  S t0, dt, p1, p2, p3, D1, D3, D4, D5;
  HD void init(const S& t0_, const S& dt_, const S& k0, const S& k1, const S& k2, const S& k3, const S& k4, const S& p1_,
               const S& p2_, const S& p3_) {
    t0 = t0_; dt = dt_; p1 = p1_; p2 = p2_; p3 = p3_;
    D1 = k0 * (k1 - k2 / p3);
    D3 = -p3 * k0 * k3 / (p1 - p3);
    D4 = -p3 * k0 * k4 / (p2 - p3);
    D5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3) + p3 * k4 / (p2 - p3));
  }
  HD void V0(const S& u, S* vv, S* dd) const {
    S e1 = ex(-p1 * u), e2 = ex(-p2 * u), e3 = ex(-p3 * u);
    *vv = D1 + u + D3 * e1 + D4 * e2 + D5 * e3;
    *dd = 1.0 - D3 * p1 * e1 - D4 * p2 * e2 - D5 * p3 * e3;
  }
  HD void f(const S& t, S* v, S* d) const {
    S t1 = t - t0, va, da, vb, db;
    V0(CLMIN(t1, 0.0), &va, &da);
    V0(CLMIN(t1 - dt, 0.0), &vb, &db);
    *v = VAL(t1) <= 0 ? S(0.0) : (VAL(t1) <= VAL(dt) ? va / dt : (va - vb) / dt);
    *d = VAL(t1) <= 0 ? S(0.0) : (VAL(t1) <= VAL(dt) ? da / dt : (da - db) / dt);
  }
};
// safeguarded Newton (the kernel's rootf)
template <class P> HD bool rootf(const P& q, double target, double lo, double hi, double x0, double* xo) {
  double v, d;
  q.f(lo, &v, &d); double flo = v - target;
  q.f(hi, &v, &d); double fhi = v - target;
  bool ok = (flo <= 0) && (fhi >= 0) && fin(flo) && fin(fhi);
  double x = fin(x0) ? mn(mx(x0, lo), hi) : 0.5 * (lo + hi);
  if (ok) {
    for (int it = 0; it < 40; it++) {
      double fx, dfx;
      q.f(x, &fx, &dfx);
      fx = fx - target;
      bool pos = fx > 0;
      if (pos) hi = x; else lo = x;
      double xn = x - fx / dfx;
      bool done = (fabs(xn - x) <= 1e-12 * (1.0 + fabs(x))) || (fx == 0);
      bool bad = (!fin(xn) || (xn <= lo) || (xn >= hi)) && !done;
      if (bad) xn = 0.5 * (lo + hi);
      x = xn;
      if (done) break;
    }
  }
  *xo = x;
  return ok && fin(x);
}
// implicit gradient of a root x* of q.f(x) = target (dtimer rootf): dx = -dF/dq dq / F'(x*), zero where not ok
template <class S, template <class> class PT> HD S root_imp(const PT<S>& qd, double x, bool ok, double dval) {
  S r(x);
  if (!ok) return r;
  S v, dd; qd.f(S(x), &v, &dd);
  double d = (dval != 0 && fin(dval)) ? dval : 1.0;
  for (int i = 0; i < S::NS; i++) r.d[i] = -v.d[i] / d;
  return r;
}
template <class S> HD void gate_delays(const Tab& TB, i64 td, i64 ts, const S& s, const S& ce, double vth, double vl, double vh,
                                       double der, S* tv, S* tl, S* sl) {
  *tv = lut(TB, td, s, ce);
  *sl = CLMIN(lut(TB, ts, s, ce), 0.0) * der;
  *tl = *tv - *sl * (vth - vl) / (vh - vl);
}
HD void lu_solve(int n, double J[3][3], const double* F, double* p, double* det) {
  double a[3][3]; double b[3];
  for (int i = 0; i < n; i++) { b[i] = -F[i]; for (int j = 0; j < n; j++) a[i][j] = J[i][j]; }
  double dt = 1.0; int perm_sign = 1;
  for (int c = 0; c < n; c++) {
    int pv = c; double m = fabs(a[c][c]);
    for (int r = c + 1; r < n; r++) if (fabs(a[r][c]) > m) { m = fabs(a[r][c]); pv = r; }
    if (pv != c) {
      for (int j = 0; j < n; j++) { double t = a[c][j]; a[c][j] = a[pv][j]; a[pv][j] = t; }
      double t = b[c]; b[c] = b[pv]; b[pv] = t; perm_sign = -perm_sign;
    }
    if (a[c][c] == 0.0) { *det = 0.0; for (int i = 0; i < n; i++) p[i] = mknan(); return; }
    for (int r = c + 1; r < n; r++) {
      double f = a[r][c] / a[c][c];
      a[r][c] = f;
      for (int j = c + 1; j < n; j++) a[r][j] -= f * a[c][j];
      b[r] -= f * b[c];
    }
  }
  for (int i = 0; i < n; i++) dt *= a[i][i];
  *det = dt * perm_sign;
  for (int i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (int j = i + 1; j < n; j++) s -= a[i][j] * p[j];
    p[i] = s / a[i][i];
  }
}
// X* of F(X, P) = 0 with F evaluated in duals (X seeded in slots 5.., P tangents in slots 0..4):
// dX = -J^-1 dF/dP (J = true Jacobian at X*); zero when J^-1 is not finite (dtimer _implicit)
HD void implicit_dx(int n, const D8* F, double dX[3][5]) {
  double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, Ji[3][3];
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) J[i][j] = F[i].d[5 + j];
  bool good = true;
  for (int i = 0; i < n; i++) {
    double e[3] = {0, 0, 0}, p[3], det; e[i] = -1.0;
    lu_solve(n, J, e, p, &det);
    for (int j = 0; j < n; j++) { Ji[j][i] = p[j]; good = good && fin(p[j]); }
  }
  for (int j = 0; j < n; j++)
    for (int s = 0; s < 5; s++) {
      double acc = 0.0;
      if (good) for (int i = 0; i < n; i++) acc += Ji[j][i] * F[i].d[s];
      dX[j][s] = good ? -acc : 0.0;
    }
}

// ---------------------------------------------------------------- gate arc: values (kernel) + duals (dtimer rules)
// record fields (driver record read by the wire edges): 0 slew 1 t0 2 dt 3 k0 4 k1 5 k2 6 k3 7 k4 8 p1 9 p2 10 vub 11 vod
#define NREC 12
#define RECW (NREC * 6)
// 11 Nov 2026: -DDG_REC32 keeps the per-arc record of the timer tape (a_rec: NREC values and 5 derivatives each, per arc) in single
// precision, which halves its memory (MemPool-Cluster: 82.1 -> 41.0 GB); the values are computed in double and rounded on store.
// Default builds: double, unchanged.
#ifdef DG_REC32
typedef float RECT;
#else
typedef double RECT;
#endif
struct GateSt { double X[3]; bool ok, good, o1, o2, o3; double d50, tl, th, dv1, dv2, dv3; };

template <class S> HD void pi_F(const Tab& TB, i64 td, i64 ts, const S& s, double vth, double vl, double vh, double der,
                                const S& rd, const S& p1, const S& p2, const S& Aa, const S& Bb, const S& Dd,
                                const S& t0, const S& dt, const S& ce, S* F) {
  S cs = CLMIN(ce, 1e-12);
  S dts = VAL(dt) > 0 ? dt : S(1e-12);
  S tv, tl, sl;
  gate_delays(TB, td, ts, s, cs, vth, vl, vh, der, &tv, &tl, &sl);
  S ct = TMIN(sl / (vh - vl), 1.4 * dts);
  S rce = rd * cs;
  S y50 = yramp(tv, t0, dts, rce), y20 = yramp(tl, t0, dts, rce);
  S ce1 = ex(-p1 * ct), ce2 = ex(-p2 * ct), cerc = ex(-ct / rce);
  S ipi = (Aa * ct + (Bb / p1) * (1.0 - ce1) + (Dd / p2) * (1.0 - ce2)) / (rd * ct * dts);
  S iceff = (rce * ct - rce * rce * (1.0 - cerc)) / (rd * ct * dts);
  F[0] = y20 - vl; F[1] = y50 - vth; F[2] = ipi - iceff;
}

struct GOut { double delay, slew, rec[NREC]; int alg; bool valid; };

// kernel dmp_pi (values), state kept for the duals
HD void pi_val(const Tab& TB, i64 td, i64 ts, double s, double c1, double c2, double rpi, double rd, double vth, double vl,
               double vh, double der, GOut& o, GateSt& st) {
  double z1 = 1.0 / (rpi * c1);
  double k0 = 1.0 / (rd * c2);
  double a = rpi * rd * c1 * c2;
  double b = rd * (c1 + c2) + rpi * c1;
  double sq = sqrt(mx(b * b - 4 * a, 0.0));
  double p1 = (b + sq) / (2 * a), p2 = (b - sq) / (2 * a);
  double p1p2 = p1 * p2;
  double k2 = z1 / p1p2;
  double k1 = (1.0 - k2 * (p1 + p2)) / p1p2;
  double k4 = (k1 * p1 + k2) / (p2 - p1);
  double k3 = -k1 - k4;
  double zz = (c1 + c2) / (rpi * c1 * c2);
  double Aa = zz / p1p2;
  double Bb = (zz - p1) / (p1 * (p1 - p2));
  double Dd = (zz - p2) / (p2 * (p2 - p1));
  double X[3];
  auto newton = [&](double* X) -> bool {
    for (int it = 0; it < 100; it++) {
      double t0 = X[0], dt = X[1], ce = X[2];
      bool bad = (ce < 0) || (ce > c1 + c2) || (dt <= 0);
      double cs = mx(ce, 1e-12);
      double dts = dt > 0 ? dt : 1e-12;
      double tv, tl, sl;
      gate_delays(TB, td, ts, s, cs, vth, vl, vh, der, &tv, &tl, &sl);
      bad |= (sl == 0);
      double ct = mn(sl / (vh - vl), 1.4 * dts);
      double rce = rd * cs;
      double e1 = ex(-p1 * dts), e2 = ex(-p2 * dts), erc = ex(-dts / rce);
      double y50 = yramp(tv, t0, dts, rce), y20 = yramp(tl, t0, dts, rce);
      double ce1 = ex(-p1 * ct), ce2 = ex(-p2 * ct), cerc = ex(-ct / rce);
      double ipi = (Aa * ct + (Bb / p1) * (1 - ce1) + (Dd / p2) * (1 - ce2)) / (rd * ct * dts);
      double iceff = (rce * ct - rce * rce * (1 - cerc)) / (rd * ct * dts);
      double F[3] = {y20 - vl, y50 - vth, ipi - iceff};
      double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      dyramp(tl, t0, dts, cs, rd, J[0]);
      dyramp(tv, t0, dts, cs, rd, J[1]);
      J[2][1] = (-Aa * dts + Bb * dts * e1 - (2 * Bb / p1) * (1 - e1) + Dd * dts * e2 - (2 * Dd / p2) * (1 - e2) +
                 rce * (dts + dts * erc - 2 * rce * (1 - erc))) / (rd * pow(dts, 3.0));
      J[2][2] = (2 * rce - dts - (2 * rce + dts) * ex(-dts / rce)) / (dts * dts);
      double p[3], det;
      lu_solve(3, J, F, p, &det);
      bool finF = fin(F[0]) && fin(F[1]) && fin(F[2]);
      bool sing = !fin(det) || (fabs(det) < 1e-300) || bad || !finF;
      if (sing) return false;
      bool conv = (fabs(p[0]) <= fabs(X[0]) * 0.01) && (fabs(p[1]) <= fabs(X[1]) * 0.01) && (fabs(p[2]) <= fabs(X[2]) * 0.01);
      X[0] += p[0]; X[1] += p[1]; X[2] += p[2];
      if (conv) return true;
    }
    return false;
  };
  auto start = [&](double ce0, double* X) {
    double tv, tl, sl;
    gate_delays(TB, td, ts, s, ce0, vth, vl, vh, der, &tv, &tl, &sl);
    double dt = sl / (vh - vl);
    X[0] = tv + log(1.0 - vth) * rd * ce0 - vth * dt; X[1] = dt; X[2] = ce0;
  };
  start(c1 + c2, X);
  bool ok = newton(X);
  if (!ok) { start(c2, X); ok = newton(X); }
  double ceff = ok ? X[2] : c1 + c2;
  double t0 = X[0], dt = X[1];
  double delay = lut(TB, td, s, ceff);
  double tslew = ts >= 0 ? mx(lut(TB, ts, s, ceff), 0.0) : 0.0;
  double vub = t0 + dt + (c1 + c2) * (rd + rpi) * 2.0;
  bool good = ok && fin(k0 + k1 + k2 + k3 + k4 + p1 + p2) && (dt > 0);
  bool valid = false; double vod = 0.0, slew = tslew;
  st.X[0] = X[0]; st.X[1] = X[1]; st.X[2] = X[2]; st.ok = ok; st.good = good; st.o1 = st.o2 = st.o3 = false;
  if (good) {
    VoP<double> q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    double span = tslew * der / (vh - vl);
    double d50, tl_, th_, vv;
    bool o1 = rootf(q, vth, t0, vub, delay, &d50);
    bool o2 = rootf(q, vl, t0, d50, d50 - span * (vth - vl), &tl_);
    bool o3 = rootf(q, vh, d50, vub, d50 + span * (vh - vth), &th_);
    valid = o1 && o2 && o3;
    vod = d50;
    slew = valid ? (th_ - tl_) / der : tslew;
    st.o1 = o1; st.o2 = o2; st.o3 = o3; st.d50 = d50; st.tl = tl_; st.th = th_;
    q.f(d50, &vv, &st.dv1); q.f(tl_, &vv, &st.dv2); q.f(th_, &vv, &st.dv3);
  }
  o.alg = ok ? 2 : 0; o.delay = delay; o.slew = slew; o.valid = valid && ok;
  double R[NREC] = {slew, t0, dt, nn0d(k0), nn0d(k1), nn0d(k2), nn0d(k3), nn0d(k4), nn0d(p1), nn0d(p2), nn0d(vub), vod};
  for (int i = 0; i < NREC; i++) o.rec[i] = R[i];
}

HD void pi_dual(const Tab& TB, i64 td, i64 ts, const D5& s, const D5& c1, const D5& c2, const D5& rpi, const D5& rd,
                double vth, double vl, double vh, double der, const GateSt& st, D5* delay, D5* slew, D5* R) {
  D5 z1 = 1.0 / (rpi * c1);
  D5 k0 = 1.0 / (rd * c2);
  D5 a = rpi * rd * c1 * c2;
  D5 b = rd * (c1 + c2) + rpi * c1;
  D5 sq = SQRT(CLMIN(b * b - 4.0 * a, 0.0));
  D5 p1 = (b + sq) / (2.0 * a), p2 = (b - sq) / (2.0 * a);
  D5 p1p2 = p1 * p2;
  D5 k2 = z1 / p1p2;
  D5 k1 = (1.0 - k2 * (p1 + p2)) / p1p2;
  D5 k4 = (k1 * p1 + k2) / (p2 - p1);
  D5 k3 = -k1 - k4;
  D5 zz = (c1 + c2) / (rpi * c1 * c2);
  D5 Aa = zz / p1p2;
  D5 Bb = (zz - p1) / (p1 * (p1 - p2));
  D5 Dd = (zz - p2) / (p2 * (p2 - p1));
  D5 t0(st.X[0]), dt(st.X[1]), ce(st.X[2]);
  if (st.ok) {
    D8 F[3];
    pi_F(TB, td, ts, W8(s), vth, vl, vh, der, W8(rd), W8(p1), W8(p2), W8(Aa), W8(Bb), W8(Dd), seed<8>(st.X[0], 5), seed<8>(st.X[1], 6), seed<8>(st.X[2], 7), F);
    double dX[3][5];
    implicit_dx(3, F, dX);
    for (int q = 0; q < 5; q++) { t0.d[q] = dX[0][q]; dt.d[q] = dX[1][q]; ce.d[q] = dX[2][q]; }
  }
  D5 ceff = st.ok ? ce : c1 + c2;
  D5 dl = lut(TB, td, s, ceff);
  D5 tslew = ts >= 0 ? CLMIN(lut(TB, ts, s, ceff), 0.0) : D5(0.0);
  D5 vub = t0 + dt + (c1 + c2) * (rd + rpi) * 2.0;
  D5 vod(0.0), sl = tslew;
  if (st.good) {
    VoP<D5> q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    D5 d50 = root_imp(q, st.d50, st.o1, st.dv1);
    D5 tl_ = root_imp(q, st.tl, st.o2, st.dv2);
    D5 th_ = root_imp(q, st.th, st.o3, st.dv3);
    vod = d50;
    if (st.o1 && st.o2 && st.o3) sl = (th_ - tl_) / der;
  }
  *delay = dl; *slew = sl;
  R[0] = sl; R[1] = t0; R[2] = dt; R[3] = NN0(k0); R[4] = NN0(k1); R[5] = NN0(k2); R[6] = NN0(k3); R[7] = NN0(k4);
  R[8] = NN0(p1); R[9] = NN0(p2); R[10] = NN0(vub); R[11] = vod;
}

// kernel dmp_zc2 (values)
HD void zc_val(const Tab& TB, i64 td, i64 ts, double s, double c1, double rpi, double rd, double vth, double vl, double vh,
               double der, GOut& o, GateSt& st) {
  double z1 = 1.0 / (rpi * c1);
  double p1 = 1.0 / (c1 * (rd + rpi));
  double k0 = p1 / z1;
  double k2 = 1.0 / k0;
  double k1 = (p1 - z1) / (p1 * p1);
  double k3 = -k1;
  double tv0, tl0, sl0;
  gate_delays(TB, td, ts, s, c1, vth, vl, vh, der, &tv0, &tl0, &sl0);
  double rce = rd * c1;
  double dt0 = sl0 / (vh - vl);
  double X[2] = {tv0 + log(1.0 - vth) * rce - vth * dt0, dt0};
  bool ok = false;
  for (int it = 0; it < 100; it++) {
    double t0 = X[0], dt = X[1];
    dt = (dt <= 0) ? (tl0 - tv0) / 100.0 : dt;
    X[1] = dt;
    double dts = dt > 0 ? dt : 1e-12;
    double F[2] = {yramp(tl0, t0, dts, rce) - vl, yramp(tv0, t0, dts, rce) - vth};
    double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, a3[3], b3[3];
    dyramp(tl0, t0, dts, c1, rd, a3);
    dyramp(tv0, t0, dts, c1, rd, b3);
    J[0][0] = a3[0]; J[0][1] = a3[1]; J[1][0] = b3[0]; J[1][1] = b3[1];
    bool bad = !(dt > 0);
    double p[3], det;
    lu_solve(2, J, F, p, &det);
    bool finF = fin(F[0]) && fin(F[1]);
    bool sing = !fin(det) || (fabs(det) < 1e-300) || bad || !finF;
    if (sing) break;
    bool conv = (fabs(p[0]) <= fabs(X[0]) * 0.01) && (fabs(p[1]) <= fabs(X[1]) * 0.01);
    X[0] += p[0]; X[1] += p[1];
    if (conv) { ok = true; break; }
  }
  double t0 = X[0], dt = X[1];
  double vub = t0 + dt + c1 * (rd + rpi) * 2.0;
  double delay = lut(TB, td, s, c1);
  double tslew = ts >= 0 ? mx(lut(TB, ts, s, c1), 0.0) : 0.0;
  double slew = tslew, vod = 0.0, k4 = 0.0, p2 = 1.0;
  bool valid = false;
  st.X[0] = X[0]; st.X[1] = X[1]; st.X[2] = 0; st.ok = ok; st.o1 = st.o2 = st.o3 = false;
  st.good = ok && fin(k0 + k1 + k2 + k3 + p1) && (dt > 0);
  if (st.good) {
    VoP<double> q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    double span = tslew * der / (vh - vl);
    double d50, tl_, th_, vv;
    bool o1 = rootf(q, vth, t0, vub, tv0, &d50);
    bool o2 = rootf(q, vl, t0, d50, d50 - span * (vth - vl), &tl_);
    bool o3 = rootf(q, vh, d50, vub, d50 + span * (vh - vth), &th_);
    valid = o1 && o2 && o3;
    vod = d50;
    if (valid) { delay = d50; slew = (th_ - tl_) / der; }
    st.o1 = o1; st.o2 = o2; st.o3 = o3; st.d50 = d50; st.tl = tl_; st.th = th_;
    q.f(d50, &vv, &st.dv1); q.f(tl_, &vv, &st.dv2); q.f(th_, &vv, &st.dv3);
  }
  o.alg = valid ? 1 : 0; o.delay = delay; o.slew = slew; o.valid = valid;
  double R[NREC] = {slew, t0, dt, nn0d(k0), nn0d(k1), nn0d(k2), nn0d(k3), nn0d(k4), nn0d(p1), nn0d(p2), nn0d(vub), vod};
  for (int i = 0; i < NREC; i++) o.rec[i] = R[i];
}

template <class S> HD void zc_F(const S& tl0, const S& tv0, const S& rce, double vl, double vth, const S& t0, const S& dt, S* F) {
  S dts = VAL(dt) > 0 ? dt : S(1e-12);
  F[0] = yramp(tl0, t0, dts, rce) - vl;
  F[1] = yramp(tv0, t0, dts, rce) - vth;
}

HD void zc_dual(const Tab& TB, i64 td, i64 ts, const D5& s, const D5& c1, const D5& rpi, const D5& rd, double vth, double vl,
                double vh, double der, const GateSt& st, D5* delay, D5* slew, D5* R) {
  D5 z1 = 1.0 / (rpi * c1);
  D5 p1 = 1.0 / (c1 * (rd + rpi));
  D5 k0 = p1 / z1;
  D5 k2 = 1.0 / k0;
  D5 k1 = (p1 - z1) / (p1 * p1);
  D5 k3 = -k1;
  D5 tv0, tl0, sl0;
  gate_delays(TB, td, ts, s, c1, vth, vl, vh, der, &tv0, &tl0, &sl0);
  D5 rce = rd * c1;
  D5 t0(st.X[0]), dt(st.X[1]);
  if (st.ok) {
    D8 F[2];
    zc_F(W8(tl0), W8(tv0), W8(rce), vl, vth, seed<8>(st.X[0], 5), seed<8>(st.X[1], 6), F);
    double dX[3][5];
    implicit_dx(2, F, dX);
    for (int q = 0; q < 5; q++) { t0.d[q] = dX[0][q]; dt.d[q] = dX[1][q]; }
  }
  D5 vub = t0 + dt + c1 * (rd + rpi) * 2.0;
  D5 dl = lut(TB, td, s, c1);
  D5 tslew = ts >= 0 ? CLMIN(lut(TB, ts, s, c1), 0.0) : D5(0.0);
  D5 sl = tslew, vod(0.0);
  if (st.good) {
    VoP<D5> q{t0, dt, k0, k1, k2, k3, D5(0.0), p1, D5(1.0)};
    D5 d50 = root_imp(q, st.d50, st.o1, st.dv1);
    D5 tl_ = root_imp(q, st.tl, st.o2, st.dv2);
    D5 th_ = root_imp(q, st.th, st.o3, st.dv3);
    vod = d50;
    if (st.o1 && st.o2 && st.o3) { dl = d50; sl = (th_ - tl_) / der; }
  }
  *delay = dl; *slew = sl;
  R[0] = sl; R[1] = t0; R[2] = dt; R[3] = NN0(k0); R[4] = NN0(k1); R[5] = NN0(k2); R[6] = NN0(k3); R[7] = D5(0.0);
  R[8] = NN0(p1); R[9] = D5(1.0); R[10] = NN0(vub); R[11] = vod;
}

// ---------------------------------------------------------------- context
struct Ctx {
  Tab TB;
  const i64 *ce_from, *ce_to, *ce_irf, *ce_orf, *ce_lib, *ce_td, *ce_ts, *g_start, *g_to, *g_orf, *ne_drv, *ne_snk, *pin_net;
  const double *L_vth, *L_vl, *L_vh, *L_der, *L_vin;
  const u8* is_port;
  const double *c1, *c2, *rpi, *ctot, *elm; const u8* par;
  double *AT, *SL;
  double *a_delay, *a_slew, *a_jd, *a_js; RECT *a_rec; signed char* a_alg; u8* a_valid; i64* pk;
  double* e_d;
  double *gAT, *gSL, *gREC, *gc1, *gc2, *grpi, *gctot, *gelm;
  const i64 *idx_arc, *idx_grp, *idx_edge;   // cone mode: compacted index lists (null = full ranges)
  i64 P, K, nn, def_lib, nE, nNE;
  int mode; double tau;
};
// pointer table order (python gputimer.py PTR list must match)
enum {
  T_tabA, T_tabB, T_tabV, T_tabNA, T_tabNB, T_ce_from, T_ce_to, T_ce_irf, T_ce_orf, T_ce_lib, T_ce_td, T_ce_ts, T_g_start,
  T_g_to, T_g_orf, T_ne_drv, T_ne_snk, T_pin_net, T_L_vth, T_L_vl, T_L_vh, T_L_der, T_L_vin, T_is_port, T_c1, T_c2, T_rpi,
  T_ctot, T_elm, T_par, T_AT, T_SL, T_a_delay, T_a_slew, T_a_jd, T_a_js, T_a_rec, T_a_alg, T_a_valid, T_pk, T_e_d, T_gAT,
  T_gSL, T_gREC, T_gc1, T_gc2, T_grpi, T_gctot, T_gelm, T_idx_arc, T_idx_grp, T_idx_edge,
  T_ce_lo_h, T_g_lo_h, T_ne_lo_h, T_cone_arc_lo_h, T_cone_grp_lo_h, T_cone_edge_lo_h,
  T_tabM, T_tabT, T_P, T_K, T_nn, T_def_lib, T_nE, T_nNE, T_nlev, T_N
};
static Ctx make_ctx(const i64* t, int mode, double tau) {
  Ctx c;
#define PT(type, name) c.name = (type)(t[T_##name])
  c.TB.A = (const double*)t[T_tabA]; c.TB.B = (const double*)t[T_tabB]; c.TB.V = (const double*)t[T_tabV];
  c.TB.NA = (const i64*)t[T_tabNA]; c.TB.NB = (const i64*)t[T_tabNB]; c.TB.M = t[T_tabM]; c.TB.T = t[T_tabT];
  PT(const i64*, ce_from); PT(const i64*, ce_to); PT(const i64*, ce_irf); PT(const i64*, ce_orf); PT(const i64*, ce_lib);
  PT(const i64*, ce_td); PT(const i64*, ce_ts); PT(const i64*, g_start); PT(const i64*, g_to); PT(const i64*, g_orf);
  PT(const i64*, ne_drv); PT(const i64*, ne_snk); PT(const i64*, pin_net);
  PT(const double*, L_vth); PT(const double*, L_vl); PT(const double*, L_vh); PT(const double*, L_der); PT(const double*, L_vin);
  PT(const u8*, is_port); PT(const double*, c1); PT(const double*, c2); PT(const double*, rpi); PT(const double*, ctot);
  PT(const double*, elm); PT(const u8*, par); PT(double*, AT); PT(double*, SL);
  PT(double*, a_delay); PT(double*, a_slew); PT(double*, a_jd); PT(double*, a_js); PT(RECT*, a_rec);
  PT(signed char*, a_alg); PT(u8*, a_valid); PT(i64*, pk); PT(double*, e_d);
  PT(double*, gAT); PT(double*, gSL); PT(double*, gREC); PT(double*, gc1); PT(double*, gc2); PT(double*, grpi);
  PT(double*, gctot); PT(double*, gelm); PT(const i64*, idx_arc); PT(const i64*, idx_grp); PT(const i64*, idx_edge);
  c.P = t[T_P]; c.K = t[T_K]; c.nn = t[T_nn]; c.def_lib = t[T_def_lib]; c.nE = t[T_nE]; c.nNE = t[T_nNE];
  c.mode = mode; c.tau = tau;
#undef PT
  return c;
}

HD void aadd(double* p, double v) {
  if (v == 0.0) return;
#if defined(__CUDA_ARCH__)
  atomicAdd(p, v);
#else
  unsigned long long* a = (unsigned long long*)p;
  unsigned long long old = __atomic_load_n(a, __ATOMIC_RELAXED), nw;
  for (;;) {
    double o; memcpy(&o, &old, 8); double n = o + v; memcpy(&nw, &n, 8);
    if (__atomic_compare_exchange_n(a, &old, nw, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
  }
#endif
}

// ---------------------------------------------------------------- forward elements
HD void arc_fwd(const Ctx& C, i64 e) {
  i64 frm = C.ce_from[e], to = C.ce_to[e], irf = C.ce_irf[e], orf = C.ce_orf[e], li = C.ce_lib[e];
  i64 net = C.pin_net[to]; if (net < 0) net += C.nn;
  i64 td = C.ce_td[e], ts = C.ce_ts[e];
  double sin = C.SL[frm * 2 + irf], c1 = C.c1[net * 2 + orf], c2 = C.c2[net * 2 + orf], rr = C.rpi[net * 2 + orf];
  double ctot = C.ctot[net * 2 + orf];
  bool par = C.par[net] != 0;
  double vth = C.L_vth[li * 2 + orf], vl = C.L_vl[li * 2 + orf], vh = C.L_vh[li * 2 + orf], der = C.L_der[li];
  const Tab& TB = C.TB;
  // values (kernel dmp_gate1)
  double rpi = rr * 1e-3;
  double cpi = par ? c1 + c2 : ctot;
  double d1 = lut(TB, td, sin, cpi), d2 = lut(TB, td, sin, cpi + 1.0);
  double rd = -log(vth) * fabs(d1 - d2) / 1.0;
  bool capm = (!par) || (rd < 1e-8) || (rpi < rd * 1e-3) || (c1 == 0) || (c1 < c2 * 1e-3) || (rpi == 0);
  bool zc2 = !capm && (c2 < c1 * 1e-3);
  bool pim = !capm && !zc2;
  GOut o; GateSt st;
  o.alg = 0; o.valid = false; o.delay = d1;
  o.slew = ts >= 0 ? mx(lut(TB, ts, sin, cpi), 0.0) : 0.0;
  for (int i = 0; i < NREC; i++) o.rec[i] = 0.0;
  o.rec[0] = o.slew; o.rec[9] = 1.0;
  if (pim) pi_val(TB, td, ts, sin, c1, c2, rpi, rd, vth, vl, vh, der, o, st);
  else if (zc2) zc_val(TB, td, ts, sin, c1, rpi, rd, vth, vl, vh, der, o, st);
  // duals (dtimer gate1): slots 0 sin 1 c1 2 c2 3 rpi (raw) 4 ctot
  D5 S_s = seed<5>(sin, 0), S_c1 = seed<5>(c1, 1), S_c2 = seed<5>(c2, 2), S_rr = seed<5>(rr, 3), S_ct = seed<5>(ctot, 4);
  D5 S_rpi = S_rr * 1e-3;
  D5 S_cpi = par ? S_c1 + S_c2 : S_ct;
  D5 S_d1 = lut(TB, td, S_s, S_cpi), S_d2 = lut(TB, td, S_s, S_cpi + 1.0);
  D5 S_rd = -log(vth) * ABS(S_d1 - S_d2) / 1.0;
  D5 dl = S_d1, sl = ts >= 0 ? CLMIN(lut(TB, ts, S_s, S_cpi), 0.0) : D5(0.0);
  D5 R[NREC];
  for (int i = 0; i < NREC; i++) R[i] = D5(0.0);
  R[9] = D5(1.0);
  if (pim) pi_dual(TB, td, ts, S_s, S_c1, S_c2, S_rpi, S_rd, vth, vl, vh, der, st, &dl, &sl, R);
  else if (zc2) zc_dual(TB, td, ts, S_s, S_c1, S_rpi, S_rd, vth, vl, vh, der, st, &dl, &sl, R);
  else R[0] = sl;
  C.a_delay[e] = o.delay; C.a_slew[e] = o.slew;
  for (int q = 0; q < 5; q++) { C.a_jd[e * 5 + q] = dl.d[q]; C.a_js[e * 5 + q] = sl.d[q]; }
  RECT* rp = C.a_rec + e * RECW;
  for (int f = 0; f < NREC; f++) { rp[f * 6] = o.rec[f]; for (int q = 0; q < 5; q++) rp[f * 6 + 1 + q] = R[f].d[q]; }
  C.a_alg[e] = (signed char)o.alg; C.a_valid[e] = o.valid ? 1 : 0;
}

HD void grp_fwd(const Ctx& C, i64 gg, i64 gb, i64 b) {
  i64 s0 = C.g_start[gg], s1 = (gg + 1 < gb) ? C.g_start[gg + 1] : b;
  i64 gto = C.g_to[gg], gorf = C.g_orf[gg], K = C.K;
  double smax = C.a_slew[s0];
  for (i64 e = s0 + 1; e < s1; e++) smax = mx(smax, C.a_slew[e]);
  C.SL[gto * 2 + gorf] = smax;
  i64 pk = -1;
  for (i64 e = s0; e < s1; e++) if (C.a_slew[e] >= smax) { pk = e; break; }
  C.pk[gto * 2 + gorf] = pk;
  for (i64 k = 0; k < K; k++) {
    double amax = NEGV; bool first = true;
    for (i64 e = s0; e < s1; e++) {
      double c = C.AT[(C.ce_from[e] * K + k) * 2 + C.ce_irf[e]] + C.a_delay[e];
      amax = first ? c : mx(amax, c); first = false;
    }
    double at = amax;
    if (C.mode == 1) {
      double s = 0.0;
      for (i64 e = s0; e < s1; e++) {
        double af = C.AT[(C.ce_from[e] * K + k) * 2 + C.ce_irf[e]];
        if (af > NEGV / 2) s += exp((af + C.a_delay[e] - amax) / C.tau);
      }
      if (s > 0) at = amax + C.tau * log(s);
    }
    C.AT[(gto * K + k) * 2 + gorf] = at;
  }
}

// wire edge: values (kernel dmp_load1 + port rule) and duals; j in [0, 2 nb): rf = j / nb, i = ea + j % nb
HD void edge_fwd(const Ctx& C, i64 i, i64 rf) {
  i64 drv = C.ne_drv[i], snk = C.ne_snk[i], K = C.K;
  double elm = C.elm[snk * 2 + rf];
  i64 pk = C.pk[drv * 2 + rf];
  double rv[NREC]; for (int f = 0; f < NREC; f++) rv[f] = 0.0;
  int alg = -1; bool valid = false; double vth = 0, vl = 0, vh = 0, der = 0;
  const RECT* rp = nullptr;
  if (pk >= 0) {
    rp = C.a_rec + pk * RECW;
    for (int f = 0; f < NREC; f++) rv[f] = rp[f * 6];
    alg = C.a_alg[pk]; valid = C.a_valid[pk] != 0;
    i64 li = C.ce_lib[pk], orf = C.ce_orf[pk];
    vth = C.L_vth[li * 2 + orf]; vl = C.L_vl[li * 2 + orf]; vh = C.L_vh[li * 2 + orf]; der = C.L_der[li];
  }
  double slew = rv[0];
  double wd = fin(elm) ? elm : 0.0, ls = slew;
  bool use = (alg >= 1) && valid && fin(elm) && (elm > 0) && !(elm < slew * 1e-3);
  double ld = 0, tl = 0, th = 0, dv1 = 0, dv2 = 0, dv3 = 0; bool o1 = false, o2 = false, o3 = false, ok = false;
  if (use) {
    double p3 = 1.0 / elm;
    VlP<double> q; q.init(rv[1], rv[2], rv[3], rv[4], rv[5], rv[6], rv[7], rv[8], rv[9], p3);
    double lo = rv[1], hi = rv[10] + 2.0 * elm;
    double sp = (slew * der + 2.2 * elm) / (vh - vl), vv;
    o1 = rootf(q, vth, lo, hi, rv[11] + 0.7 * elm, &ld);
    o2 = rootf(q, vl, lo, ld, ld - sp * (vth - vl), &tl);
    o3 = rootf(q, vh, ld, hi, ld + sp * (vh - vth), &th);
    ok = o1 && o2 && o3;
    double d1 = ld - rv[11], s1 = (th - tl) / der;
    d1 = d1 < 0 ? elm : d1;
    s1 = s1 < slew ? slew : s1;
    wd = ok ? d1 : elm;
    ls = ok ? s1 : slew;
    q.f(ld, &vv, &dv1); q.f(tl, &vv, &dv2); q.f(th, &vv, &dv3);
  }
  double* ed = C.e_d + (i * 2 + rf) * 12;
  bool port = C.is_port[drv] != 0;
  if (port) {
    bool okp = fin(elm);
    double vin = C.L_vin[C.def_lib * 2 + rf], vlo = C.L_vl[C.def_lib * 2 + rf], vhi = C.L_vh[C.def_lib * 2 + rf];
    wd = okp ? -elm * log(1.0 - vin) : 0.0;
    double g = log((1 - vlo) / (1 - vhi)) / C.L_der[C.def_lib];
    ls = C.SL[drv * 2 + rf] + (okp ? elm * g : 0.0);
    for (int q = 0; q < 12; q++) ed[q] = 0.0;
    ed[5] = okp ? -log(1.0 - vin) : 0.0;
    ed[11] = okp ? g : 0.0;
  } else {
    // duals: slots 0..4 = the picked arc's inputs (record tangents), slot 5 = elm
    D6 R[NREC];
    for (int f = 0; f < NREC; f++) { R[f] = D6(rv[f]); if (rp) for (int q = 0; q < 5; q++) R[f].d[q] = rp[f * 6 + 1 + q]; }
    D6 E = seed<6>(elm, 5);
    D6 elz = fin(elm) ? E : D6(0.0);
    D6 Wd = elz, Ls = R[0];
    if (use) {
      D6 e = elz, p3 = 1.0 / e;
      VlP<D6> q; q.init(R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], p3);
      D6 Ld = root_imp(q, ld, o1, dv1), Tl = root_imp(q, tl, o2, dv2), Th = root_imp(q, th, o3, dv3);
      D6 d1 = Ld - R[11], s1 = (Th - Tl) / der;
      d1 = (ld - rv[11]) < 0 ? e : d1;
      s1 = ((th - tl) / der) < slew ? R[0] : s1;
      Wd = ok ? d1 : e;
      Ls = ok ? s1 : R[0];
    }
    for (int q = 0; q < 6; q++) { ed[q] = Wd.d[q]; ed[6 + q] = Ls.d[q]; }
  }
  C.SL[snk * 2 + rf] = ls;
  for (i64 k = 0; k < K; k++) C.AT[(snk * K + k) * 2 + rf] = C.AT[(drv * K + k) * 2 + rf] + wd;
}

// ---------------------------------------------------------------- backward elements
HD void edge_bwd(const Ctx& C, i64 i, i64 rf) {
  i64 drv = C.ne_drv[i], snk = C.ne_snk[i], K = C.K;
  double gls = C.gSL[snk * 2 + rf], gwd = 0.0;
  for (i64 k = 0; k < K; k++) {
    double g = C.gAT[(snk * K + k) * 2 + rf];
    gwd += g;
    aadd(&C.gAT[(drv * K + k) * 2 + rf], g);
  }
  if (gwd == 0.0 && gls == 0.0) return;
  const double* ed = C.e_d + (i * 2 + rf) * 12;
  C.gelm[snk * 2 + rf] += gwd * ed[5] + gls * ed[11];
  if (C.is_port[drv]) { aadd(&C.gSL[drv * 2 + rf], gls); return; }
  double* gr = C.gREC + (drv * 2 + rf) * 5;
  for (int q = 0; q < 5; q++) aadd(&gr[q], gwd * ed[q] + gls * ed[6 + q]);
}

HD void grp_bwd(const Ctx& C, i64 gg, i64 gb, i64 b) {
  i64 s0 = C.g_start[gg], s1 = (gg + 1 < gb) ? C.g_start[gg + 1] : b;
  i64 gto = C.g_to[gg], gorf = C.g_orf[gg], K = C.K;
  double gS = C.gSL[gto * 2 + gorf];
  const double* gR = C.gREC + (gto * 2 + gorf) * 5;
  i64 pk = C.pk[gto * 2 + gorf];
  double smax = C.SL[gto * 2 + gorf];
  int ns = 0;
  for (i64 e = s0; e < s1; e++) ns += (C.a_slew[e] == smax);
  double gse = ns > 0 ? gS / ns : 0.0;
  // arrival weights per k: hard = ties split evenly; smooth = softmax (exp((cand - at) / tau)) when a valid input exists
  double wk_cnt[64], amk[64]; bool soft[64];
  for (i64 k = 0; k < K; k++) {
    double am = NEGV; bool first = true; int nv = 0;
    for (i64 e = s0; e < s1; e++) {
      double af = C.AT[(C.ce_from[e] * K + k) * 2 + C.ce_irf[e]];
      double c = af + C.a_delay[e];
      am = first ? c : mx(am, c); first = false;
      nv += (af > NEGV / 2);
    }
    soft[k] = (C.mode == 1) && nv > 0;
    int nt = 0;
    for (i64 e = s0; e < s1; e++) nt += ((C.AT[(C.ce_from[e] * K + k) * 2 + C.ce_irf[e]] + C.a_delay[e]) == am);
    wk_cnt[k] = nt > 0 ? 1.0 / nt : 0.0;
    amk[k] = am;
  }
  for (i64 e = s0; e < s1; e++) {
    i64 frm = C.ce_from[e], irf = C.ce_irf[e];
    double gd = 0.0;
    for (i64 k = 0; k < K; k++) {
      double gk = C.gAT[(gto * K + k) * 2 + gorf];
      if (gk == 0.0) continue;
      double af = C.AT[(frm * K + k) * 2 + irf];
      double c = af + C.a_delay[e];
      double w;
      if (soft[k]) {
        w = (af > NEGV / 2) ? exp((c - C.AT[(gto * K + k) * 2 + gorf]) / C.tau) : 0.0;
      } else {
        w = (c == amk[k]) ? wk_cnt[k] : 0.0;
      }
      double g = gk * w;
      gd += g;
      aadd(&C.gAT[(frm * K + k) * 2 + irf], g);
    }
    double gs = (C.a_slew[e] == smax) ? gse : 0.0;
    double gin[5];
    for (int q = 0; q < 5; q++) gin[q] = gd * C.a_jd[e * 5 + q] + gs * C.a_js[e * 5 + q] + (e == pk ? gR[q] : 0.0);
    i64 to = C.ce_to[e], orf = C.ce_orf[e];
    i64 net = C.pin_net[to]; if (net < 0) net += C.nn;
    aadd(&C.gSL[frm * 2 + irf], gin[0]);
    aadd(&C.gc1[net * 2 + orf], gin[1]); aadd(&C.gc2[net * 2 + orf], gin[2]);
    aadd(&C.grpi[net * 2 + orf], gin[3]); aadd(&C.gctot[net * 2 + orf], gin[4]);
  }
}

// ---------------------------------------------------------------- drivers
#ifdef __CUDACC__
__global__ void k_arc(Ctx C, i64 a, i64 n, const i64* idx) {
  i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t >= n) return;
  arc_fwd(C, idx ? idx[a + t] : a + t);
}
__global__ void k_grp(Ctx C, i64 ga, i64 n, i64 gb, i64 b, const i64* idx, int bwd) {
  i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t >= n) return;
  i64 gg = idx ? idx[ga + t] : ga + t;
  if (bwd) grp_bwd(C, gg, gb, b); else grp_fwd(C, gg, gb, b);
}
__global__ void k_edge(Ctx C, i64 ea, i64 nb, const i64* idx, int bwd) {
  i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t >= 2 * nb) return;
  i64 rf = t / nb, j = t % nb;
  i64 i = idx ? idx[ea + j] : ea + j;
  if (bwd) edge_bwd(C, i, rf); else edge_fwd(C, i, rf);
}
static int BS = 128;
#define LAUNCH(n) (unsigned)(((n) + BS - 1) / BS), BS
#else
struct Pool {
  std::vector<std::thread> th; std::mutex m; std::condition_variable cv, cvd;
  const std::function<void(i64, i64)>* job = nullptr; std::atomic<i64> next{0}; i64 n = 0, chunk = 1; int gen = 0, active = 0;
  int nthr = 0;
  void init() {
    if (nthr) return;
    const char* e = getenv("GT_THREADS"); if (!e) e = getenv("OMP_NUM_THREADS");
    nthr = e ? atoi(e) : 8; if (nthr < 1) nthr = 1;
    for (int i = 1; i < nthr; i++) { th.emplace_back([this]() { loop(); }); th.back().detach(); }
  }
  void work() { for (;;) { i64 s = next.fetch_add(chunk); if (s >= n) break; (*job)(s, std::min(n, s + chunk)); } }
  void loop() {
    int seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, [&] { return gen != seen; });
      seen = gen; lk.unlock();
      work();
      lk.lock(); if (--active == 0) cvd.notify_one();
    }
  }
  void run(i64 n_, i64 chunk_, const std::function<void(i64, i64)>& f) {
    init();
    if (n_ <= 0) return;
    if (nthr == 1 || n_ <= chunk_) { f(0, n_); return; }
    { std::unique_lock<std::mutex> lk(m); job = &f; n = n_; chunk = chunk_; next = 0; active = nthr - 1; gen++; }
    cv.notify_all();
    work();
    std::unique_lock<std::mutex> lk(m); cvd.wait(lk, [&] { return active == 0; });
  }
};
static Pool POOL;
template <class F> static void pfor(i64 n, i64 chunk, F f) { std::function<void(i64, i64)> fn = f; POOL.run(n, chunk, fn); }
#endif

// ---------------------------------------------------------------- RC build: the depth sweeps of dtimer.build_rc with their adjoint
// up: y1[p] += y1[w]; y2[p] += y2[w] - R y1[w]^2; y3[p] += y3[w] - 2 R y1[w] y2[w] + R^2 y1[w]^3 (deepest level first)
// down: elm[w] = elm[p] + R y1[w] 1e-3 (shallowest level first). One thread per parent (up, reverse of down) or per child, no atomics.
struct RcCtx { const double* Rp; double *y1, *y2, *y3, *elm, *gR, *gy1, *gy2, *gy3, *gelm; const i64 *nodes, *par, *plist, *pptr; };
enum { R_Rp, R_y1, R_y2, R_y3, R_elm, R_gR, R_gy1, R_gy2, R_gy3, R_gelm, R_nodes, R_par, R_plist, R_pptr, R_lo_h, R_plo_h, R_maxd, R_N };
static RcCtx make_rc(const i64* t) {
  RcCtx c;
  c.Rp = (const double*)t[R_Rp]; c.y1 = (double*)t[R_y1]; c.y2 = (double*)t[R_y2]; c.y3 = (double*)t[R_y3]; c.elm = (double*)t[R_elm];
  c.gR = (double*)t[R_gR]; c.gy1 = (double*)t[R_gy1]; c.gy2 = (double*)t[R_gy2]; c.gy3 = (double*)t[R_gy3]; c.gelm = (double*)t[R_gelm];
  c.nodes = (const i64*)t[R_nodes]; c.par = (const i64*)t[R_par]; c.plist = (const i64*)t[R_plist]; c.pptr = (const i64*)t[R_pptr];
  return c;
}
HD void rc_up(const RcCtx& C, i64 i) {
  i64 p = C.plist[i], j0 = C.pptr[i], j1 = C.pptr[i + 1];
  for (int c = 0; c < 2; c++) {
    double s1 = C.y1[p * 2 + c], s2 = C.y2[p * 2 + c], s3 = C.y3[p * 2 + c];
    for (i64 j = j0; j < j1; j++) {
      i64 w = C.nodes[j]; double R = C.Rp[w], a1 = C.y1[w * 2 + c], a2 = C.y2[w * 2 + c], a3 = C.y3[w * 2 + c];
      s1 += a1; s2 += a2 - R * a1 * a1; s3 += a3 - 2 * R * a1 * a2 + R * R * (a1 * a1 * a1);
    }
    C.y1[p * 2 + c] = s1; C.y2[p * 2 + c] = s2; C.y3[p * 2 + c] = s3;
  }
}
HD void rc_down(const RcCtx& C, i64 j) {
  i64 w = C.nodes[j], p = C.par[w]; double R = C.Rp[w];
  for (int c = 0; c < 2; c++) C.elm[w * 2 + c] = C.elm[p * 2 + c] + R * C.y1[w * 2 + c] * 1e-3;
}
HD void rc_down_b(const RcCtx& C, i64 i) {
  i64 p = C.plist[i], j0 = C.pptr[i], j1 = C.pptr[i + 1];
  for (int c = 0; c < 2; c++) {
    double acc = 0.0;
    for (i64 j = j0; j < j1; j++) {
      i64 w = C.nodes[j]; double ge = C.gelm[w * 2 + c], R = C.Rp[w];
      acc += ge;
      C.gy1[w * 2 + c] += ge * R * 1e-3;
      C.gR[w] += ge * C.y1[w * 2 + c] * 1e-3;
    }
    C.gelm[p * 2 + c] += acc;
  }
}
HD void rc_up_b(const RcCtx& C, i64 j) {
  i64 w = C.nodes[j], p = C.par[w]; double R = C.Rp[w], gr = 0.0;
  for (int c = 0; c < 2; c++) {
    double g1 = C.gy1[p * 2 + c], g2 = C.gy2[p * 2 + c], g3 = C.gy3[p * 2 + c];
    double a1 = C.y1[w * 2 + c], a2 = C.y2[w * 2 + c];
    C.gy1[w * 2 + c] += g1 + g2 * (-2 * R * a1) + g3 * (-2 * R * a2 + 3 * R * R * a1 * a1);
    C.gy2[w * 2 + c] += g2 + g3 * (-2 * R * a1);
    C.gy3[w * 2 + c] += g3;
    gr += g2 * (-a1 * a1) + g3 * (-2 * a1 * a2 + 2 * R * a1 * a1 * a1);
  }
  C.gR[w] += gr;
}
#ifdef __CUDACC__
__global__ void k_rc(RcCtx C, i64 a, i64 n, int which) {
  i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t >= n) return;
  if (which == 0) rc_up(C, a + t); else if (which == 1) rc_down(C, a + t); else if (which == 2) rc_down_b(C, a + t); else rc_up_b(C, a + t);
}
#define RC_RUN(a, n, which) do { if ((n) > 0) k_rc<<<LAUNCH(n)>>>(C, a, n, which); } while (0)
#else
template <class F> static void rc_cpu(i64 a, i64 n, F f) { pfor(n, 512, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) f(a + x); }); }
#define RC_RUN(a, n, which) do { if ((n) > 0) { \
    if ((which) == 0) rc_cpu(a, n, [&](i64 x) { rc_up(C, x); }); else if ((which) == 1) rc_cpu(a, n, [&](i64 x) { rc_down(C, x); }); \
    else if ((which) == 2) rc_cpu(a, n, [&](i64 x) { rc_down_b(C, x); }); else rc_cpu(a, n, [&](i64 x) { rc_up_b(C, x); }); } } while (0)
#endif
extern "C" int gt_rc_ntable() { return R_N; }
extern "C" int gt_rc_forward(const i64* t) {
  RcCtx C = make_rc(t); const i64* lo = (const i64*)t[R_lo_h]; const i64* plo = (const i64*)t[R_plo_h]; i64 maxd = t[R_maxd];
  for (i64 d = maxd; d >= 1; d--) RC_RUN(plo[d], plo[d + 1] - plo[d], 0);
  for (i64 d = 1; d <= maxd; d++) RC_RUN(lo[d], lo[d + 1] - lo[d], 1);
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "gt_rc_forward: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}
extern "C" int gt_rc_backward(const i64* t) {
  RcCtx C = make_rc(t); const i64* lo = (const i64*)t[R_lo_h]; const i64* plo = (const i64*)t[R_plo_h]; i64 maxd = t[R_maxd];
  for (i64 d = maxd; d >= 1; d--) RC_RUN(plo[d], plo[d + 1] - plo[d], 2);
  for (i64 d = 1; d <= maxd; d++) RC_RUN(lo[d], lo[d + 1] - lo[d], 3);
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "gt_rc_backward: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}

// level ranges: full mode from ce_lo / g_lo / ne_lo; cone mode from the compacted lists' level offsets
static void ranges(const i64* t, i64 lv, bool cone, i64* a, i64* na, i64* ga, i64* ng, i64* ea, i64* ne, i64* gb, i64* b) {
  const i64 *ce_lo = (const i64*)t[T_ce_lo_h], *g_lo = (const i64*)t[T_g_lo_h], *ne_lo = (const i64*)t[T_ne_lo_h];
  *gb = g_lo[lv + 1]; *b = ce_lo[lv + 1];
  if (!cone) {
    *a = ce_lo[lv]; *na = ce_lo[lv + 1] - ce_lo[lv]; *ga = g_lo[lv]; *ng = g_lo[lv + 1] - g_lo[lv];
    *ea = ne_lo[lv]; *ne = ne_lo[lv + 1] - ne_lo[lv];
  } else {
    const i64 *ca = (const i64*)t[T_cone_arc_lo_h], *cg = (const i64*)t[T_cone_grp_lo_h], *ce = (const i64*)t[T_cone_edge_lo_h];
    *a = ca[lv]; *na = ca[lv + 1] - ca[lv]; *ga = cg[lv]; *ng = cg[lv + 1] - cg[lv]; *ea = ce[lv]; *ne = ce[lv + 1] - ce[lv];
  }
}

extern "C" int gt_forward(const i64* t, int mode, double tau) {
  Ctx C = make_ctx(t, mode, tau);
  bool cone = C.idx_arc != nullptr;
  i64 nlev = t[T_nlev];
  bool prof = getenv("GT_PROF") != nullptr; double pa = 0, pg = 0, pe = 0; (void)prof; (void)pa; (void)pg; (void)pe;
  for (i64 lv = 1; lv < nlev; lv++) {
    i64 a, na, ga, ng, ea, ne, gb, b;
    ranges(t, lv, cone, &a, &na, &ga, &ng, &ea, &ne, &gb, &b);
#ifdef __CUDACC__
    if (prof) {   // GT_PROF=1: every launch waits, split per element kind (adds launch latency to each)
      auto now = [] { cudaDeviceSynchronize(); return std::chrono::steady_clock::now(); };
      auto sec = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
      auto t0 = now();
      if (na > 0) k_arc<<<LAUNCH(na)>>>(C, a, na, C.idx_arc);
      auto t1 = now();
      if (ng > 0) k_grp<<<LAUNCH(ng)>>>(C, ga, ng, gb, b, C.idx_grp, 0);
      auto t2 = now();
      if (ne > 0) k_edge<<<LAUNCH(2 * ne)>>>(C, ea, ne, C.idx_edge, 0);
      auto t3 = now();
      pa += sec(t0, t1); pg += sec(t1, t2); pe += sec(t2, t3);
      if (getenv("GT_PROF_LV")) fprintf(stderr, "LV %ld %ld %ld %ld %.6f %.6f %.6f\n", (long)lv, (long)na, (long)ng, (long)ne, sec(t0, t1), sec(t1, t2), sec(t2, t3));
      continue;
    }
    if (na > 0) k_arc<<<LAUNCH(na)>>>(C, a, na, C.idx_arc);
    if (ng > 0) k_grp<<<LAUNCH(ng)>>>(C, ga, ng, gb, b, C.idx_grp, 0);
    if (ne > 0) k_edge<<<LAUNCH(2 * ne)>>>(C, ea, ne, C.idx_edge, 0);
#else
    pfor(na, 64, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) arc_fwd(C, cone ? C.idx_arc[a + x] : a + x); });
    pfor(ng, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) grp_fwd(C, cone ? C.idx_grp[ga + x] : ga + x, gb, b); });
    pfor(2 * ne, 128, [&](i64 lo, i64 hi) {
      for (i64 x = lo; x < hi; x++) { i64 rf = x / ne, j = x % ne; edge_fwd(C, cone ? C.idx_edge[ea + j] : ea + j, rf); } });
#endif
  }
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "gt_forward: %s\n", cudaGetErrorString(err)); return 1; }
  if (prof) fprintf(stderr, "GT_PROF arcs %.4f groups %.4f edges %.4f\n", pa, pg, pe);
#endif
  return 0;
}

extern "C" int gt_backward(const i64* t, int mode, double tau) {
  Ctx C = make_ctx(t, mode, tau);
  bool cone = C.idx_arc != nullptr;
  i64 nlev = t[T_nlev];
  for (i64 lv = nlev - 1; lv >= 1; lv--) {
    i64 a, na, ga, ng, ea, ne, gb, b;
    ranges(t, lv, cone, &a, &na, &ga, &ng, &ea, &ne, &gb, &b);
#ifdef __CUDACC__
    if (ne > 0) k_edge<<<LAUNCH(2 * ne)>>>(C, ea, ne, C.idx_edge, 1);
    if (ng > 0) k_grp<<<LAUNCH(ng)>>>(C, ga, ng, gb, b, C.idx_grp, 1);
#else
    pfor(2 * ne, 128, [&](i64 lo, i64 hi) {
      for (i64 x = lo; x < hi; x++) { i64 rf = x / ne, j = x % ne; edge_bwd(C, cone ? C.idx_edge[ea + j] : ea + j, rf); } });
    pfor(ng, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) grp_bwd(C, cone ? C.idx_grp[ga + x] : ga + x, gb, b); });
#endif
  }
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "gt_backward: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}
extern "C" int gt_sync() {
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "gt_sync: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}
extern "C" int gt_ntable() { return T_N; }
