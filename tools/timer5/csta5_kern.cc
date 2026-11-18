// csta5_kern.cc: the exact incremental timer of chain_j (tools/best/libi_kern = i_kern.cc =
// tools/fast3x/h_csta_kern.cc + w3_inc_backward; h_csta_kern.cc = tools/engine/csta_inc_kern.cc + csta_hcrit; csta_inc_kern.cc =
// tools/gapla/csta_kern.cc + csta_rc.inc) FLATTENED into one file so the thread pool and the level loops can change.
// Rule: the arithmetic inside every node (arc, group, net edge, row, net RC) is the original source, in the original order;
// only the scheduling (which thread, when) and the bookkeeping change. Changes are marked "T5:".
// Build: tools/timer5/build5.sh (g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread, no -march: hosts lack AVX2).
// ======================================================================== tools/gapla/csta_kern.cc (verbatim but marked T5)
// csta_kern.cc -- C++ port of the route-dependent half of fast_sta.py (v1, the timer the one-pass flow runs):
//   build_rc (net RC from route segments)  +  StaTimer.analyze (OpenSTA dmp_ceff_elmore, levelized AT/SL propagation,
//   endpoint slacks, required times)  +  the critical-path extraction of fastmode4.write_crit / io_build.read_crit_lines.
// Same arithmetic order as the numpy code (build with -ffp-contract=off, no -ffast-math); only OpenMP over independent
// elements.  Build: tools/gapla/csta_build.sh.  Called through ctypes by tools/gapla/csta.py.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <limits>
typedef int64_t i64;
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <array>
// ---------------------------------------------------------------- tiny persistent thread pool (no OpenMP: torch's libomp would clash)
// T5 (idea 0): spin-then-block pool. Workers spin up to CSTA5_SPIN_US (default 100) microseconds on the job counter before they
// sleep on the condition variable; the caller spins for the end of the job. A job costs ~1-3 us instead of a futex wake per worker
// (the timer issues ~2 jobs per level, 70-190 levels per full pass, more in an update). The work split itself is unchanged in kind:
// dynamic chunks, each element computed by exactly one thread with the original code, so the results cannot change.
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}
static double now();
struct Pool {
  std::vector<std::thread> th; std::mutex m; std::condition_variable cv;
  const std::function<void(i64, i64)>* job = nullptr; std::atomic<i64> next{0}; i64 n = 0, chunk = 1;
  std::atomic<int> gen{0}, active{0}, sleepers{0}; std::atomic<bool> stop{false};
  int nthr = 0; double spin_s = 100e-6;
  void init() {
    if (nthr) return;
    const char* e = getenv("CSTA_THREADS"); if (!e) e = getenv("OMP_NUM_THREADS");
    nthr = e ? atoi(e) : 8; if (nthr < 1) nthr = 1;
    const char* s = getenv("CSTA5_SPIN_US"); if (s) spin_s = atof(s) * 1e-6;
    for (int i = 1; i < nthr; i++) { th.emplace_back([this]() { loop(); }); th.back().detach(); }
  }
  void work() { for (;;) { i64 s = next.fetch_add(chunk); if (s >= n) break; (*job)(s, std::min(n, s + chunk)); } }
  void loop() {
    int seen = 0;
    for (;;) {
      int g = gen.load(std::memory_order_acquire);
      if (g == seen && !stop.load()) {
        double t0 = now();
        for (int it = 1; ; it++) {
          cpu_relax();
          g = gen.load(std::memory_order_acquire);
          if (g != seen || stop.load()) break;
          if ((it & 255) == 0 && now() - t0 > spin_s) break;
        }
      }
      if (g == seen && !stop.load()) {
        std::unique_lock<std::mutex> lk(m);
        sleepers.fetch_add(1);
        while (gen.load() == seen && !stop.load()) cv.wait(lk);
        sleepers.fetch_sub(1);
        g = gen.load();
      }
      if (stop.load()) return;
      seen = g;
      work();
      active.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  void run(i64 n_, i64 chunk_, const std::function<void(i64, i64)>& f) {
    init();
    if (n_ <= 0) return;
    if (nthr == 1 || n_ <= chunk_) { f(0, n_); return; }
    job = &f; n = n_; chunk = chunk_; next.store(0); active.store(nthr - 1);
    gen.fetch_add(1);                                                        // seq_cst: pairs with sleepers below (Dekker)
    if (sleepers.load() > 0) { std::lock_guard<std::mutex> lk(m); cv.notify_all(); }
    work();
    while (active.load(std::memory_order_acquire) != 0) cpu_relax();
  }
};
static Pool& POOL = *new Pool;                                             // T5: never destroyed (workers may be waiting at exit)
template <class F> static void pfor(i64 n, i64 chunk, F f) { std::function<void(i64, i64)> fn = f; POOL.run(n, chunk, fn); }
// T5 (idea 1): chunk sized by the work: about 8 chunks per thread, at most cmax elements, at least 1
static inline i64 ach(i64 n, i64 cmax) { POOL.init(); i64 c = n / (8 * (i64)POOL.nthr); return c < 1 ? 1 : (c > cmax ? cmax : c); }
static const double NEG = -1e30, BIG = 1e30;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------- registry
static std::map<std::string, const void*> REG;
static std::map<std::string, i64> SCL;
extern "C" void csta_set(const char* name, const void* p) { REG[name] = p; }
extern "C" void csta_seti(const char* name, i64 v) { SCL[name] = v; }
template <class T> static const T* g(const char* n) {
  auto it = REG.find(n);
  if (it == REG.end()) { fprintf(stderr, "csta: missing array %s\n", n); abort(); }
  return (const T*)it->second;
}
static i64 gi(const char* n) {
  auto it = SCL.find(n);
  if (it == SCL.end()) { fprintf(stderr, "csta: missing scalar %s\n", n); abort(); }
  return it->second;
}

// np.maximum / np.minimum semantics (NaN propagates)
static inline double mx(double a, double b) { return (a != a || b != b) ? NaN : (a >= b ? a : b); }
static inline double mn(double a, double b) { return (a != a || b != b) ? NaN : (a <= b ? a : b); }
#ifdef T5_COUNT                                                              // T5: diagnostic build only (-DT5_COUNT)
static std::atomic<i64> CNT5[16];
#define C5(i) CNT5[i].fetch_add(1, std::memory_order_relaxed)
#include <x86intrin.h>
#define TS5(v) unsigned long long v = __rdtsc()
#define TE5(i, v) CNT5[i].fetch_add((i64)(__rdtsc() - v), std::memory_order_relaxed)
extern "C" void csta5_counts(i64* o) { for (int i = 0; i < 16; i++) o[i] = CNT5[i].load(); }
#else
#define C5(i)
#define TS5(v)
#define TE5(i, v)
#endif
static inline double ex(double x) { C5(0); return std::exp(mx(x, -700.0)); }
static inline double nn0(double v) {  // np.nan_to_num
  if (v != v) return 0.0;
  if (std::isinf(v)) return v > 0 ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
  return v;
}

// ---------------------------------------------------------------- tables
struct Tab {
  const double *A, *B, *V;
  const i64 *NA, *NB;
  i64 M, T;
};
static Tab TB;
static inline double lut(i64 tid, double x, double y) {
  C5(1);
  if (tid < 0) tid += TB.T;  // numpy negative index
  const double* A = TB.A + tid * TB.M;
  const double* B = TB.B + tid * TB.M;
  i64 na = TB.NA[tid], nb = TB.NB[tid];
  i64 ka = 0, kb = 0;
  for (i64 j = 1; j < na; j++) ka += (x > A[j]);
  for (i64 j = 1; j < nb; j++) kb += (y > B[j]);
  if (ka > na - 2) ka = na - 2;
  if (kb > nb - 2) kb = nb - 2;
  double a0 = A[ka], a1 = A[ka + 1], b0 = B[kb], b1 = B[kb + 1];
  double fa = (x - a0) / (a1 != a0 ? a1 - a0 : 1.0);
  double fb = (y - b0) / (b1 != b0 ? b1 - b0 : 1.0);
  const double* V = TB.V;
  i64 M = TB.M, base = tid * (M * M) + ka * M + kb;
  return (1 - fa) * (1 - fb) * V[base] + (1 - fa) * fb * V[base + 1] + fa * (1 - fb) * V[base + M] +
         fa * fb * V[base + M + 1];
}

// ---------------------------------------------------------------- waveforms
static inline double y0f(double u, double rc) { return u - rc * (1.0 - ex(-u / rc)); }
// T5 (idea 6, per node, exact): the waveform helpers evaluate only the terms the selected branch returns (the original computes
// both ramp terms and then picks one) and reuse ex(-u/rc) where the original evaluates the same expression up to three times.
// Every returned value is the same expression on the same operands, so the bits are equal; NaN inputs take the same branch.
static inline double yramp(double t, double t0, double dt, double rc) {
  double t1 = t - t0;
  if (t1 <= 0) return 0.0;
  double a = y0f(mx(t1, 0.0), rc);
  if (t1 <= dt) return a / dt;
  double b = y0f(mx(t1 - dt, 0.0), rc);
  return (a - b) / dt;
}
// yramp(t, t0, dt, rd * cl) and dyramp(t, t0, dt, cl, rd, o) in one pass (both form rc = rd * cl from the same operands)
static inline double ydy(double t, double t0, double dt, double cl, double rd, double* o) {
  double rc = rd * cl, t1 = t - t0;
  if (t1 <= 0) { o[0] = 0.0; o[1] = 0.0; o[2] = 0.0; return 0.0; }
  double u1 = mx(t1, 0.0), E1 = ex(-u1 / rc);
  double y0a = u1 - rc * (1.0 - E1);
  double dta = 1.0 - E1;
  double dca = rd * ((1.0 + u1 / rc) * E1 - 1.0);
  if (t1 <= dt) { o[0] = -dta / dt; o[1] = -y0a / (dt * dt); o[2] = dca / dt; return y0a / dt; }
  double u2 = mx(t1 - dt, 0.0), E2 = ex(-u2 / rc);
  double y0b = u2 - rc * (1.0 - E2);
  double dtb = 1.0 - E2;
  double dcb = rd * ((1.0 + u2 / rc) * E2 - 1.0);
  o[0] = -(dta - dtb) / dt; o[1] = -(y0a + y0b) / (dt * dt) + dtb / dt; o[2] = (dca - dcb) / dt;
  return (y0a - y0b) / dt;
}
static inline void dyramp(double t, double t0, double dt, double cl, double rd, double* o) {
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

struct VoP { double t0, dt, k0, k1, k2, k3, k4, p1, p2; };
static inline void vo_f(const VoP& q, double t, double* v, double* d) {
  auto V0 = [&](double u, double* vv, double* dd) {
    double e1 = ex(-q.p1 * u), e2 = ex(-q.p2 * u);
    *vv = q.k0 * (q.k1 + q.k2 * u + q.k3 * e1 + q.k4 * e2);
    *dd = q.k0 * (q.k2 - q.k3 * q.p1 * e1 - q.k4 * q.p2 * e2);
  };
  double t1 = t - q.t0, va, da, vb, db;                                     // T5: lazy branches (see yramp)
  if (t1 <= 0) { *v = 0.0; *d = 0.0; return; }
  V0(mx(t1, 0.0), &va, &da);
  if (t1 <= q.dt) { *v = va / q.dt; *d = da / q.dt; return; }
  V0(mx(t1 - q.dt, 0.0), &vb, &db);
  *v = (va - vb) / q.dt; *d = (da - db) / q.dt;
}
struct VlP { double t0, dt, p1, p2, p3, D1, D3, D4, D5; };
static inline void vl_init(VlP& q, double t0, double dt, double k0, double k1, double k2, double k3, double k4, double p1,
                           double p2, double p3) {
  q.t0 = t0; q.dt = dt; q.p1 = p1; q.p2 = p2; q.p3 = p3;
  q.D1 = k0 * (k1 - k2 / p3);
  q.D3 = -p3 * k0 * k3 / (p1 - p3);
  q.D4 = -p3 * k0 * k4 / (p2 - p3);
  q.D5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3) + p3 * k4 / (p2 - p3));
}
static inline void vl_f(const VlP& q, double t, double* v, double* d) {
  auto V0 = [&](double u, double* vv, double* dd) {
    double e1 = ex(-q.p1 * u), e2 = ex(-q.p2 * u), e3 = ex(-q.p3 * u);
    *vv = q.D1 + u + q.D3 * e1 + q.D4 * e2 + q.D5 * e3;
    *dd = 1.0 - q.D3 * q.p1 * e1 - q.D4 * q.p2 * e2 - q.D5 * q.p3 * e3;
  };
  double t1 = t - q.t0, va, da, vb, db;                                     // T5: lazy branches (see yramp)
  if (t1 <= 0) { *v = 0.0; *d = 0.0; return; }
  V0(mx(t1, 0.0), &va, &da);
  if (t1 <= q.dt) { *v = va / q.dt; *d = da / q.dt; return; }
  V0(mx(t1 - q.dt, 0.0), &vb, &db);
  *v = (va - vb) / q.dt; *d = (da - db) / q.dt;
}
// safeguarded Newton (rtsafe), scalar copy of _root
// T5 (idea 8): the three root searches of one waveform share end points (t0/vub/d50 resp. lo/hi/ld); the waveform value at a
// shared end point is computed once (WV: value known or not). The value is target-independent; flo = v - target as before.
struct WV { bool has = false; double v = 0.0; };
template <class P, void (*F)(const P&, double, double*, double*)>
static bool rootfc(const P& q, double target, double lo, double hi, double x0, double* xo, WV& wlo, WV& whi) {
  double v, d;
  if (!wlo.has) { F(q, lo, &v, &d); wlo.v = v; wlo.has = true; }
  double flo = wlo.v - target;
  if (!whi.has) { F(q, hi, &v, &d); whi.v = v; whi.has = true; }
  double fhi = whi.v - target;
  bool ok = (flo <= 0) && (fhi >= 0) && std::isfinite(flo) && std::isfinite(fhi);
  double x = std::isfinite(x0) ? mn(mx(x0, lo), hi) : 0.5 * (lo + hi);
  if (ok) {
    for (int it = 0; it < 40; it++) {
      double fx, dfx;
      F(q, x, &fx, &dfx);
      fx = fx - target;
      bool pos = fx > 0;
      if (pos) hi = x; else lo = x;
      double xn = x - fx / dfx;
      bool done = (std::fabs(xn - x) <= 1e-12 * (1.0 + std::fabs(x))) || (fx == 0);
      bool bad = (!std::isfinite(xn) || (xn <= lo) || (xn >= hi)) && !done;
      if (bad) xn = 0.5 * (lo + hi);
      x = xn;
      if (done) break;
    }
  }
  *xo = x;
  return ok && std::isfinite(x);
}
template <class P, void (*F)(const P&, double, double*, double*)>
static bool rootf(const P& q, double target, double lo, double hi, double x0, double* xo) {
  double v, d;
  F(q, lo, &v, &d); double flo = v - target;
  F(q, hi, &v, &d); double fhi = v - target;
  bool ok = (flo <= 0) && (fhi >= 0) && std::isfinite(flo) && std::isfinite(fhi);
  double x = std::isfinite(x0) ? mn(mx(x0, lo), hi) : 0.5 * (lo + hi);
  if (ok) {
    C5(3);
    for (int it = 0; it < 40; it++) {
      C5(2);
      double fx, dfx;
      F(q, x, &fx, &dfx);
      fx = fx - target;
      bool pos = fx > 0;
      if (pos) hi = x; else lo = x;
      double xn = x - fx / dfx;
      bool done = (std::fabs(xn - x) <= 1e-12 * (1.0 + std::fabs(x))) || (fx == 0);  // v2: converged -> keep the iterate
      bool bad = (!std::isfinite(xn) || (xn <= lo) || (xn >= hi)) && !done;
      if (bad) xn = 0.5 * (lo + hi);
      x = xn;
      if (done) break;
    }
  }
  *xo = x;
  return ok && std::isfinite(x);
}

// ---------------------------------------------------------------- gate delay (dmp_gate), one arc
struct Dmp {
  double delay, slew, rd, ceff, t0, dt, vod, k0, k1, k2, k3, k4, p1, p2, vub, vth, vl, vh, der;
  int alg; bool valid; double dbg1, dbg2;
};
static inline void gate_delays(i64 td, i64 ts, double s, double ce, double vth, double vl, double vh, double der,
                               double* tv, double* tl, double* sl) {
  *tv = lut(td, s, ce);
  *sl = mx(lut(ts, s, ce), 0.0) * der;
  *tl = *tv - *sl * (vth - vl) / (vh - vl);
}
// solve J p = -F (n = 2, 3) by LU with partial pivoting; det from the same LU
static inline void lu_solve(int n, double J[3][3], const double* F, double* p, double* det) {
  double a[3][3]; double b[3];
  for (int i = 0; i < n; i++) { b[i] = -F[i]; for (int j = 0; j < n; j++) a[i][j] = J[i][j]; }
  double dt = 1.0; int perm_sign = 1;
  for (int c = 0; c < n; c++) {
    int pv = c; double m = std::fabs(a[c][c]);
    for (int r = c + 1; r < n; r++) if (std::fabs(a[r][c]) > m) { m = std::fabs(a[r][c]); pv = r; }
    if (pv != c) { for (int j = 0; j < n; j++) std::swap(a[c][j], a[pv][j]); std::swap(b[c], b[pv]); perm_sign = -perm_sign; }
    if (a[c][c] == 0.0) { *det = 0.0; for (int i = 0; i < n; i++) p[i] = NaN; return; }
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

struct Arc { i64 td, ts; double sin, c1, c2, rpi, rd, vth, vl, vh, der; };

// T5 (idea 9): the delay / slew table values at ce = c1 + c2 (computed by dmp_gate1 as d1 and its slew lookup) are reused
// where dmp_pi looks them up again at a bit-equal ce (start point, first Newton step, final values when Newton failed)
struct LM { double ce, tv, ts; };
static inline bool beq5(double a, double b) { uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8); return x == y; }
static inline void gate_delays_m(i64 td, i64 ts, double s, double ce, double vth, double vl, double vh, double der,
                                 double* tv, double* tl, double* sl, const LM& m) {
  bool hit = beq5(ce, m.ce);
  *tv = hit ? m.tv : lut(td, s, ce);
  *sl = mx(hit ? m.ts : lut(ts, s, ce), 0.0) * der;
  *tl = *tv - *sl * (vth - vl) / (vh - vl);
}
static void dmp_pi(const Arc& A, Dmp& o, double lt_, double ls_, bool has_ls) {
  C5(5);
  double c1 = A.c1, c2 = A.c2, rpi = A.rpi, rd = A.rd, vth = A.vth, vl = A.vl, vh = A.vh, der = A.der, s = A.sin;
  i64 td = A.td, ts = A.ts;
  const LM lm{c1 + c2, lt_, has_ls ? ls_ : lut(ts, s, c1 + c2)};
  double z1 = 1.0 / (rpi * c1);
  double k0 = 1.0 / (rd * c2);
  double a = rpi * rd * c1 * c2;
  double b = rd * (c1 + c2) + rpi * c1;
  double sq = std::sqrt(mx(b * b - 4 * a, 0.0));
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
  // newton
  auto newton = [&](double X[3]) -> bool {
    for (int it = 0; it < 100; it++) {
      C5(4);
      double t0 = X[0], dt = X[1], ce = X[2];
      bool bad = (ce < 0) || (ce > c1 + c2) || (dt <= 0);
      double cs = mx(ce, 1e-12);
      double dts = dt > 0 ? dt : 1e-12;
      double tv, tl, sl;
      gate_delays_m(td, ts, s, cs, vth, vl, vh, der, &tv, &tl, &sl, lm);
      bad |= (sl == 0);
      double ct = mn(sl / (vh - vl), 1.4 * dts);
      double rce = rd * cs;
      double e1 = ex(-p1 * dts), e2 = ex(-p2 * dts), erc = ex(-dts / rce);
      double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      double y50 = ydy(tv, t0, dts, cs, rd, J[1]), y20 = ydy(tl, t0, dts, cs, rd, J[0]);   // T5: = yramp(., rce) + dyramp
      double ce1 = ex(-p1 * ct), ce2 = ex(-p2 * ct), cerc = ex(-ct / rce);
      double ipi = (Aa * ct + (Bb / p1) * (1 - ce1) + (Dd / p2) * (1 - ce2)) / (rd * ct * dts);
      double iceff = (rce * ct - rce * rce * (1 - cerc)) / (rd * ct * dts);
      double F[3] = {y20 - vl, y50 - vth, ipi - iceff};
      J[2][1] = (-Aa * dts + Bb * dts * e1 - (2 * Bb / p1) * (1 - e1) + Dd * dts * e2 - (2 * Dd / p2) * (1 - e2) +
                 rce * (dts + dts * erc - 2 * rce * (1 - erc))) / (rd * std::pow(dts, 3.0));
      J[2][2] = (2 * rce - dts - (2 * rce + dts) * erc) / (dts * dts);       // T5: erc == ex(-dts / rce)
      double p[3], det;
      lu_solve(3, J, F, p, &det);
      bool fin = std::isfinite(F[0]) && std::isfinite(F[1]) && std::isfinite(F[2]);
      bool sing = !std::isfinite(det) || (std::fabs(det) < 1e-300) || bad || !fin;
      if (sing) return false;
      bool conv = (std::fabs(p[0]) <= std::fabs(X[0]) * 0.01) && (std::fabs(p[1]) <= std::fabs(X[1]) * 0.01) &&
                  (std::fabs(p[2]) <= std::fabs(X[2]) * 0.01);
      X[0] += p[0]; X[1] += p[1]; X[2] += p[2];
      if (conv) return true;
    }
    return false;
  };
  auto start = [&](double ce0, double X[3]) {
    double tv, tl, sl;
    gate_delays_m(td, ts, s, ce0, vth, vl, vh, der, &tv, &tl, &sl, lm);
    double dt = sl / (vh - vl);
    X[0] = tv + std::log(1.0 - vth) * rd * ce0 - vth * dt; X[1] = dt; X[2] = ce0;
  };
  TS5(tn_);
  double X[3];
  start(c1 + c2, X);
  bool ok = newton(X);
  if (!ok) { start(c2, X); ok = newton(X); }
  double ceff = ok ? X[2] : c1 + c2;
  double t0 = X[0], dt = X[1];
  const bool hit_ = beq5(ceff, lm.ce);
  double delay = hit_ ? lm.tv : lut(td, s, ceff);
  double tslew = ts >= 0 ? mx(hit_ ? lm.ts : lut(ts, s, ceff), 0.0) : 0.0;
  double vub = t0 + dt + (c1 + c2) * (rd + rpi) * 2.0;
  TE5(11, tn_);
  TS5(tr_);
  bool good = ok && std::isfinite(k0 + k1 + k2 + k3 + k4 + p1 + p2) && (dt > 0);
  bool valid = false; double vod = 0.0, slew = tslew;
  if (good) {
    VoP q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    double span = tslew * der / (vh - vl);
    double d50, tl_, th_;
    WV wt0, wub, wd50;                                                      // T5: shared end points
    bool o1 = rootfc<VoP, vo_f>(q, vth, t0, vub, delay, &d50, wt0, wub);
    bool o2 = rootfc<VoP, vo_f>(q, vl, t0, d50, d50 - span * (vth - vl), &tl_, wt0, wd50);
    bool o3 = rootfc<VoP, vo_f>(q, vh, d50, vub, d50 + span * (vh - vth), &th_, wd50, wub);
    valid = o1 && o2 && o3;
    vod = d50;
    slew = valid ? (th_ - tl_) / der : tslew;
    o.dbg1 = tl_; o.dbg2 = th_;
  }
  TE5(12, tr_);
  o.alg = ok ? 2 : 0; o.ceff = ceff; o.delay = delay; o.slew = slew;
  o.t0 = t0; o.dt = dt; o.vod = vod; o.valid = valid && ok;
  o.k0 = nn0(k0); o.k1 = nn0(k1); o.k2 = nn0(k2); o.k3 = nn0(k3); o.k4 = nn0(k4); o.p1 = nn0(p1); o.p2 = nn0(p2);
  o.vub = nn0(vub);
}

static void dmp_zc2(const Arc& A, Dmp& o) {
  C5(6);
  double c1 = A.c1, rpi = A.rpi, rd = A.rd, vth = A.vth, vl = A.vl, vh = A.vh, der = A.der, s = A.sin;
  i64 td = A.td, ts = A.ts;
  double z1 = 1.0 / (rpi * c1);
  double p1 = 1.0 / (c1 * (rd + rpi));
  double k0 = p1 / z1;
  double k2 = 1.0 / k0;
  double k1 = (p1 - z1) / (p1 * p1);
  double k3 = -k1;
  double tv0, tl0, sl0;
  gate_delays(td, ts, s, c1, vth, vl, vh, der, &tv0, &tl0, &sl0);
  double rce = rd * c1;
  double dt0 = sl0 / (vh - vl);
  double X[2] = {tv0 + std::log(1.0 - vth) * rce - vth * dt0, dt0};
  bool ok = false;
  for (int it = 0; it < 100; it++) {
    C5(7);
    double t0 = X[0], dt = X[1];
    dt = (dt <= 0) ? (tl0 - tv0) / 100.0 : dt;
    X[1] = dt;
    double dts = dt > 0 ? dt : 1e-12;
    double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, a3[3], b3[3];
    double ya_ = ydy(tl0, t0, dts, c1, rd, a3), yb_ = ydy(tv0, t0, dts, c1, rd, b3);   // T5: = yramp(., rce) + dyramp
    double F[2] = {ya_ - vl, yb_ - vth};
    J[0][0] = a3[0]; J[0][1] = a3[1]; J[1][0] = b3[0]; J[1][1] = b3[1];
    bool bad = !(dt > 0);
    double p[3], det;
    lu_solve(2, J, F, p, &det);
    bool fin = std::isfinite(F[0]) && std::isfinite(F[1]);
    bool sing = !std::isfinite(det) || (std::fabs(det) < 1e-300) || bad || !fin;
    if (sing) break;
    bool conv = (std::fabs(p[0]) <= std::fabs(X[0]) * 0.01) && (std::fabs(p[1]) <= std::fabs(X[1]) * 0.01);
    X[0] += p[0]; X[1] += p[1];
    if (conv) { ok = true; break; }
  }
  double t0 = X[0], dt = X[1];
  double vub = t0 + dt + c1 * (rd + rpi) * 2.0;
  double delay = lut(td, s, c1);
  double tslew = ts >= 0 ? mx(lut(ts, s, c1), 0.0) : 0.0;
  double slew = tslew, vod = 0.0, k4 = 0.0, p2 = 1.0;
  bool valid = false;
  if (ok && std::isfinite(k0 + k1 + k2 + k3 + p1) && (dt > 0)) {
    VoP q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    double span = tslew * der / (vh - vl);
    double d50, tl_, th_;
    WV wt0, wub, wd50;                                                      // T5: shared end points
    bool o1 = rootfc<VoP, vo_f>(q, vth, t0, vub, tv0, &d50, wt0, wub);
    bool o2 = rootfc<VoP, vo_f>(q, vl, t0, d50, d50 - span * (vth - vl), &tl_, wt0, wd50);
    bool o3 = rootfc<VoP, vo_f>(q, vh, d50, vub, d50 + span * (vh - vth), &th_, wd50, wub);
    valid = o1 && o2 && o3;
    vod = d50;
    if (valid) { delay = d50; slew = (th_ - tl_) / der; }
  }
  o.alg = valid ? 1 : 0; o.ceff = c1; o.delay = delay; o.slew = slew;
  o.t0 = t0; o.dt = dt; o.vod = vod; o.valid = valid;
  o.k0 = nn0(k0); o.k1 = nn0(k1); o.k2 = nn0(k2); o.k3 = nn0(k3); o.k4 = nn0(k4); o.p1 = nn0(p1); o.p2 = nn0(p2);
  o.vub = nn0(vub);
}

static inline void dmp_gate1(const Arc& Ain, bool par, double ctot, Dmp& o) {
  C5(8);
  Arc A = Ain;
  double rpi = A.rpi * 1e-3;
  A.rpi = rpi;
  double cpi = par ? A.c1 + A.c2 : ctot;
  double d1 = lut(A.td, A.sin, cpi), d2 = lut(A.td, A.sin, cpi + 1.0);
  double rd = -std::log(A.vth) * std::fabs(d1 - d2) / 1.0;
  A.rd = rd;
  bool capm = (!par) || (rd < 1e-8) || (rpi < rd * 1e-3) || (A.c1 == 0) || (A.c1 < A.c2 * 1e-3) || (rpi == 0);
  bool zc2 = !capm && (A.c2 < A.c1 * 1e-3);
  bool pim = !capm && !zc2;
  o.rd = rd; o.vth = A.vth; o.vl = A.vl; o.vh = A.vh; o.der = A.der;
  o.alg = 0; o.valid = false; o.ceff = cpi; o.delay = d1;
  const double ls_ = A.ts >= 0 ? lut(A.ts, A.sin, cpi) : 0.0;              // T5: kept for dmp_pi (same lookup)
  o.slew = A.ts >= 0 ? mx(ls_, 0.0) : 0.0;
  o.t0 = o.dt = o.vod = o.k0 = o.k1 = o.k2 = o.k3 = o.k4 = o.p1 = o.vub = 0.0; o.p2 = 1.0;
  TS5(tg_);
  if (pim) dmp_pi(A, o, d1, ls_, A.ts >= 0);
  else if (zc2) dmp_zc2(A, o);
  TE5(13, tg_);
}

// ---------------------------------------------------------------- load delay / slew (dmp_load)
struct DP { double slew, t0, dt, k0, k1, k2, k3, k4, p1, p2, vub, vod, vth, vl, vh, der; int alg; unsigned char valid; };
static inline void dmp_load1(const DP& d, double elm, double* wd, double* ls) {
  *wd = std::isfinite(elm) ? elm : 0.0;
  *ls = d.slew;
  bool use = (d.alg >= 1) && d.valid && std::isfinite(elm) && (elm > 0) && !(elm < d.slew * 1e-3);
  C5(9);
  if (!use) return;
  C5(10);
  TS5(tl_);
  double p3 = 1.0 / elm;
  VlP q; vl_init(q, d.t0, d.dt, d.k0, d.k1, d.k2, d.k3, d.k4, d.p1, d.p2, p3);
  double lo = d.t0, hi = d.vub + 2.0 * elm;
  double sp = (d.slew * d.der + 2.2 * elm) / (d.vh - d.vl);
  double ld, tl, th;
  WV wlo, whi, wld;                                                         // T5: shared end points
  bool o1 = rootfc<VlP, vl_f>(q, d.vth, lo, hi, d.vod + 0.7 * elm, &ld, wlo, whi);
  bool o2 = rootfc<VlP, vl_f>(q, d.vl, lo, ld, ld - sp * (d.vth - d.vl), &tl, wlo, wld);
  bool o3 = rootfc<VlP, vl_f>(q, d.vh, ld, hi, ld + sp * (d.vh - d.vth), &th, wld, whi);
  TE5(14, tl_);
  bool ok = o1 && o2 && o3;
  double d1 = ld - d.vod, s1 = (th - tl) / d.der;
  d1 = d1 < 0 ? elm : d1;
  s1 = s1 < d.slew ? d.slew : s1;
  *wd = ok ? d1 : elm;
  *ls = ok ? s1 : d.slew;
}

// ---------------------------------------------------------------- analyze
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<double> S_reqk; static std::vector<unsigned char> S_valid;
static bool T5_ana = false;   // T5: the original csta_analyze overwrote S_reqk -> the next csta_inc_full starts from scratch
// outputs are caller-allocated numpy arrays (see csta.py); stats[0..3] = wall of forward / endpoints / rat / total
extern "C" int csta_analyze(double* AT, double* SL, double* gate_d, double* wire_d, i64* BEST, double* net_rd,
                            double* rmin_out, double* ep_sl_out, double* RAT, double* pin_slack, double* net_slack,
                            double* stats) {
  double tA = now();
  T5_ana = true;
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

  for (i64 i = 0; i < P * K * 2; i++) { AT[i] = NEG; BEST[i] = -1; }
  for (i64 i = 0; i < P * 2; i++) { SL[i] = 0.0; wire_d[i] = 0.0; }
  for (i64 i = 0; i < nE; i++) gate_d[i] = 0.0;
  for (i64 i = 0; i < nn; i++) net_rd[i] = 0.0;
  for (i64 i = 0; i < nsp; i++) AT[(sp_pin[i] * K + sp_cls[i]) * 2 + sp_rf[i]] = sp_at[i];
  for (i64 p = 0; p < P; p++) if (reg_ck[p]) { SL[p * 2] = ck_slew[p * 2]; SL[p * 2 + 1] = ck_slew[p * 2 + 1]; }

  std::vector<DP> dps((size_t)P * 2);
  for (auto& d : dps) { memset(&d, 0, sizeof(DP)); d.alg = -1; }
  i64 maxarc = 0, maxg = 0;
  for (i64 lv = 1; lv < nlev; lv++) { maxarc = std::max(maxarc, ce_lo[lv + 1] - ce_lo[lv]); maxg = std::max(maxg, g_lo[lv + 1] - g_lo[lv]); }
  std::vector<Dmp> res((size_t)maxarc);
  std::vector<i64> pick((size_t)maxg);
  std::vector<i64> picknet_all;  // for net_rd
  for (i64 lv = 1; lv < nlev; lv++) {
    i64 a = ce_lo[lv], b = ce_lo[lv + 1];
    if (b > a) {
      pfor(b - a, 64, [&](i64 lo_, i64 hi_) { for (i64 e = a + lo_; e < a + hi_; e++) {
        i64 frm = ce_from[e], to = ce_to[e], irf = ce_irf[e], orf = ce_orf[e], li = ce_lib[e];
        i64 net = pin_net[to];
        Arc A;
        A.td = ce_td[e]; A.ts = ce_ts[e]; A.sin = SL[frm * 2 + irf];
        A.c2 = c2a[net * 2 + orf]; A.rpi = rpia[net * 2 + orf]; A.c1 = c1a[net * 2 + orf];
        A.vth = L_vth[li * 2 + orf]; A.vl = L_vl[li * 2 + orf]; A.vh = L_vh[li * 2 + orf]; A.der = L_der[li];
        dmp_gate1(A, para[net] != 0, ctota[net * 2 + orf], res[e - a]);
        gate_d[e] = res[e - a].delay;
      } });
      i64 ga = g_lo[lv], gb = g_lo[lv + 1];
      pfor(gb - ga, 256, [&](i64 lo_, i64 hi_) { for (i64 gg = ga + lo_; gg < ga + hi_; gg++) {
        i64 s0 = g_start[gg], s1 = (gg + 1 < gb) ? g_start[gg + 1] : b;  // arcs [s0,s1)
        i64 gto = g_to[gg], gorf = g_orf[gg];
        double smax = res[s0 - a].slew;
        for (i64 e = s0 + 1; e < s1; e++) smax = mx(smax, res[e - a].slew);
        SL[gto * 2 + gorf] = smax;
        i64 pk = -1;
        for (i64 e = s0; e < s1; e++) if (res[e - a].slew >= smax) { pk = e; break; }
        if (pk < 0) pk = b;  // python would index out of range; never expected
        pick[gg - ga] = pk;
        DP& d = dps[gto * 2 + gorf];
        if (pk < b) {
          const Dmp& r = res[pk - a];
          d.slew = r.slew; d.t0 = r.t0; d.dt = r.dt; d.k0 = r.k0; d.k1 = r.k1; d.k2 = r.k2; d.k3 = r.k3; d.k4 = r.k4;
          d.p1 = r.p1; d.p2 = r.p2; d.vub = r.vub; d.vod = r.vod; d.vth = r.vth; d.vl = r.vl; d.vh = r.vh; d.der = r.der;
          d.alg = r.alg; d.valid = r.valid;
        }
        for (i64 k = 0; k < K; k++) {
          double amax = NEG; bool first = true;
          for (i64 e = s0; e < s1; e++) {
            double c = AT[(ce_from[e] * K + k) * 2 + ce_irf[e]] + res[e - a].delay;
            amax = first ? c : mx(amax, c); first = false;
          }
          AT[(gto * K + k) * 2 + gorf] = amax;
          i64 best = nE;
          for (i64 e = s0; e < s1; e++) {
            double c = AT[(ce_from[e] * K + k) * 2 + ce_irf[e]] + res[e - a].delay;
            if (c >= amax - 1e-12) { best = e; break; }
          }
          BEST[(gto * K + k) * 2 + gorf] = best;
        }
      } });
      for (i64 gg = ga; gg < gb; gg++) {  // net_rd (np.maximum.at, sequential)
        i64 pk = pick[gg - ga];
        if (pk < b) { i64 net = pin_net[ce_to[pk]]; net_rd[net] = mx(net_rd[net], res[pk - a].rd); }
      }
    }
    a = ne_lo[lv]; b = ne_lo[lv + 1];
    if (b > a) {
      i64 nb = b - a;
      pfor(2 * nb, 128, [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) {
        i64 rf = j / nb, i = a + (j % nb);
        i64 drv = ne_drv[i], snk = ne_snk[i];
        double elm = elma[snk * 2 + rf], wd, ls;
        dmp_load1(dps[drv * 2 + rf], elm, &wd, &ls);
        if (is_port[drv]) {
          bool okp = std::isfinite(elm);
          double vin = L_vin[def_lib * 2 + rf], vlo = L_vl[def_lib * 2 + rf], vhi = L_vh[def_lib * 2 + rf];
          wd = okp ? -elm * std::log(1.0 - vin) : 0.0;
          ls = SL[drv * 2 + rf] + (okp ? elm * std::log((1 - vlo) / (1 - vhi)) / L_der[def_lib] : 0.0);
        }
        wire_d[snk * 2 + rf] = wd;
        SL[snk * 2 + rf] = ls;
        for (i64 k = 0; k < K; k++) AT[(snk * K + k) * 2 + rf] = AT[(drv * K + k) * 2 + rf] + wd;
      } });
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
  for (i64 p = 0; p < P; p++) ep_sl_out[p] = BIG;
  for (i64 r = 0; r < nrow; r++) ep_sl_out[ck_pin[r]] = mn(ep_sl_out[ck_pin[r]], rmin_out[r]);
  double tC = now();
  // ---- required times
  for (i64 i = 0; i < P * K * 2; i++) RAT[i] = BIG;
  for (i64 r = 0; r < nrow; r++)
    for (i64 k = 0; k < K; k++)
      for (int rf = 0; rf < 2; rf++) {
        i64 ix = (ck_pin[r] * K + k) * 2 + rf;
        RAT[ix] = mn(RAT[ix], valid[(r * K + k) * 2 + rf] ? reqk[(r * K + k) * 2 + rf] : BIG);
      }
  for (i64 lv = nlev - 1; lv >= 1; lv--) {
    for (i64 i = ne_lo[lv]; i < ne_lo[lv + 1]; i++) {
      i64 drv = ne_drv[i], snk = ne_snk[i];
      for (i64 k = 0; k < K; k++)
        for (int rf = 0; rf < 2; rf++) {
          i64 ix = (drv * K + k) * 2 + rf;
          RAT[ix] = mn(RAT[ix], RAT[(snk * K + k) * 2 + rf] - wire_d[snk * 2 + rf]);
        }
    }
    for (i64 e = ce_lo[lv]; e < ce_lo[lv + 1]; e++) {
      i64 to = ce_to[e], orf = ce_orf[e], fr = ce_from[e], irf = ce_irf[e];
      for (i64 k = 0; k < K; k++) {
        double v = RAT[(to * K + k) * 2 + orf] - gate_d[e];
        i64 ix = (fr * K + k) * 2 + irf;
        RAT[ix] = mn(RAT[ix], v);
      }
    }
  }
  for (i64 i = 0; i < nn; i++) net_slack[i] = BIG;
  for (i64 p = 0; p < P; p++) {
    double m = BIG;
    for (i64 k = 0; k < K; k++)
      for (int rf = 0; rf < 2; rf++) {
        i64 ix = (p * K + k) * 2 + rf;
        bool ok = (AT[ix] > NEG / 2) && (RAT[ix] < BIG / 2);
        m = mn(m, ok ? RAT[ix] - AT[ix] : BIG);
      }
    pin_slack[p] = m;
    i64 n = pin_net[p]; if (n < 0) n += nn;
    net_slack[n] = mn(net_slack[n], m);
  }
  double tD = now();
  stats[0] = tB - tA; stats[1] = tC - tB; stats[2] = tD - tC; stats[3] = tD - tA;
  return 0;
}

// debug / unit-test entry: one arc -> out[19] = delay slew rd ceff t0 dt vod k0 k1 k2 k3 k4 p1 p2 vub alg valid
extern "C" void csta_gate1(i64 td, i64 ts, double sin, double c1, double c2, double rpi, double ctot, int par, double vth,
                           double vl, double vh, double der, double* out) {
  Arc A; A.td = td; A.ts = ts; A.sin = sin; A.c1 = c1; A.c2 = c2; A.rpi = rpi; A.rd = 0; A.vth = vth; A.vl = vl; A.vh = vh; A.der = der;
  Dmp o; dmp_gate1(A, par != 0, ctot, o);
  double v[19] = {o.delay, o.slew, o.rd, o.ceff, o.t0, o.dt, o.vod, o.k0, o.k1, o.k2, o.k3, o.k4, o.p1, o.p2, o.vub, (double)o.alg, (double)o.valid, o.dbg1, o.dbg2};
  for (int i = 0; i < 19; i++) out[i] = v[i];
}
extern "C" int csta_root_vo(const double* q9, double target, double lo, double hi, double x0, double* x, double* trace) {
  VoP q{q9[0], q9[1], q9[2], q9[3], q9[4], q9[5], q9[6], q9[7], q9[8]};
  double v, d; vo_f(q, lo, &v, &d); trace[0] = v; vo_f(q, hi, &v, &d); trace[1] = v; vo_f(q, x0, &v, &d); trace[2] = v; trace[3] = d;
  return rootf<VoP, vo_f>(q, target, lo, hi, x0, x);
}

#include "../gapla/csta_rc.inc"
// ======================================================================== tools/engine/csta_inc_kern.cc
// csta_inc_kern.cc: INCREMENTAL exact C++ timer (plan item T3 / E1).
// The whole original kernel (tools/gapla/csta_kern.cc + csta_rc.inc) is compiled into this library by inclusion, so every
// original entry point (csta_set, csta_analyze, csta_build_rc, csta_crit, ...) is unchanged and the per-arc / per-edge
// arithmetic of the update is literally the same source (dmp_gate1, dmp_load1, lut). New entry points (csta_inc_*):
//   csta_inc_bind(info)         reverse indices of the bound design + safety checks (unique writer per pin/rf, level order)
//   csta_inc_full(bwd, stats)   full analysis into PERSISTENT state (AT, SL, gate_d, wire_d, BEST, net_rd, rmin, ep_sl,
//                               driver-model records per (pin, rf), picked-arc rd per gate group, S_reqk/S_valid);
//                               bwd = 0 skips the required-time pass (RAT / pin slack / net slack not written)
//   csta_inc_begin()            start a transaction (applies required-time rows kept visible by the last reject)
//   csta_inc_set_rc(...)        copy the RC of some nets from given arrays into the bound rc_* arrays (logged)
//   csta_inc_build_rc_subset()  rebuild the RC of some nets from their segments into the bound rc_* arrays (logged)
//   csta_inc_update(nets)       re-time the fan-out cone of the changed nets (dirty lists per level, stop where the
//                               values are bit-identical to the stored ones); bit-identical to csta_inc_full
//   csta_inc_commit() / csta_inc_reject()   drop / undo the transaction. reject keeps S_reqk/S_valid of the rejected
//                               state visible to csta_crit until the next begin (= what the chain sees today: K.crit reads
//                               the required times of the LAST analyze), then restores them.
// Calling the ORIGINAL csta_analyze in this library overwrites S_reqk: call csta_inc_full again afterwards.
// Build (host): tools/engine/build_inc.sh  (g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -shared -fPIC).

static inline bool beq(double a, double b) { uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8); return x == y; }
static inline bool dp_eq(const DP& a, const DP& b) {
  return beq(a.slew, b.slew) && beq(a.t0, b.t0) && beq(a.dt, b.dt) && beq(a.k0, b.k0) && beq(a.k1, b.k1) && beq(a.k2, b.k2) &&
         beq(a.k3, b.k3) && beq(a.k4, b.k4) && beq(a.p1, b.p1) && beq(a.p2, b.p2) && beq(a.vub, b.vub) && beq(a.vod, b.vod) &&
         beq(a.vth, b.vth) && beq(a.vl, b.vl) && beq(a.vh, b.vh) && beq(a.der, b.der) && a.alg == b.alg && a.valid == b.valid;
}
static inline DP dp_init() { DP d; memset(&d, 0, sizeof(DP)); d.alg = -1; return d; }

// ---------------------------------------------------------------- persistent state
static i64 I_P = 0, I_K = 0, I_nE = 0, I_nn = 0, I_nlev = 0, I_nrow = 0, I_g0 = 0, I_g1 = 0, I_e0 = 0, I_e1 = 0;
static std::vector<double> I_AT, I_SL, I_gd, I_as, I_wd, I_nrd, I_rmin, I_ep, I_grd, I_RAT, I_psl, I_nsl;   // I_as: arc output slew
static std::vector<i64> I_BEST, I_gpke;
static std::vector<unsigned char> I_gpk;
static std::vector<DP> I_dps;
static bool I_ready = false, I_full = false;
// T5 (idea 7): shadow of the RC the persistent state was computed from (per net; written by the full pass and, for the seed
// nets, by each update; its writes have their own undo log so reject restores it and the reject count stays as before)
static std::vector<double> SH_c1, SH_c2, SH_rpi, SH_ctot, SH_elm; static std::vector<unsigned char> SH_par;
static std::vector<i64> pn_o, pn;                                  // pins per net
static i64 T5_inc_full = 0, T5_scratch_full = 0, T5_inc_nets = 0;
// indices
static std::vector<i64> I_glv, I_gs1, I_elv;                       // group level, group arc end, net-edge level
static std::vector<i64> netg_o, netg, nete_o, nete, drve_o, drve, rdg_o, rdg, rda_o, rda, prow_o, prow, apin_o, apin, I_agrp;
// dirty lists
static std::vector<std::vector<i64>> Gd, Ed;
static std::vector<unsigned char> gflag, eflag, rflag, pflag, nflag, rdflag, aflag;
static std::vector<i64> rlist, plist, nlist, rdlist;
static i64 cur_phase = 0, late_marks = 0;                          // phase = 2*lv (groups) or 2*lv+1 (edges)
// undo logs
struct LD { double* p; double v; }; struct LI { i64* p; i64 v; }; struct LU { unsigned char* p; unsigned char v; };
struct LP { DP* p; DP v; };
static std::vector<LD> lD, lDq, pDq; static std::vector<LI> lI; static std::vector<LU> lU, lUq, pUq; static std::vector<LP> lP;
static inline bool wD(double* p, double v) { if (beq(*p, v)) return false; lD.push_back({p, *p}); *p = v; return true; }
static inline bool wI(i64* p, i64 v) { if (*p == v) return false; lI.push_back({p, *p}); *p = v; return true; }
static inline bool wU(unsigned char* p, unsigned char v) { if (*p == v) return false; lU.push_back({p, *p}); *p = v; return true; }
static inline bool wP(DP* p, const DP& v) { if (dp_eq(*p, v)) return false; lP.push_back({p, *p}); *p = v; return true; }
static std::vector<LD> lSH; static std::vector<LU> lSHu;          // T5: undo log of the RC shadow
static inline void wSH(double* p, double v) { if (!beq(*p, v)) { lSH.push_back({p, *p}); *p = v; } }
static inline void wSHu(unsigned char* p, unsigned char v) { if (*p != v) { lSHu.push_back({p, *p}); *p = v; } }
static void sh_clear() { lSH.clear(); lSHu.clear(); }
static void sh_undo() {
  for (i64 i = (i64)lSH.size() - 1; i >= 0; i--) *lSH[i].p = lSH[i].v;
  for (i64 i = (i64)lSHu.size() - 1; i >= 0; i--) *lSHu[i].p = lSHu[i].v;
  sh_clear();
}

// ---------------------------------------------------------------- design arrays (fetched once per call)
struct DA {
  i64 P, K, nlev, nE, nn, def_lib, nsp, nrow, nevt;
  const i64 *ce_from, *ce_to, *ce_irf, *ce_orf, *ce_lib, *ce_td, *ce_ts, *ce_lo, *g_lo, *g_start, *g_to, *g_orf, *ne_lo, *ne_drv,
      *ne_snk, *pin_net, *sp_pin, *sp_cls, *sp_rf, *ck_pin, *ck_tr, *ck_tf, *ck_type, *ck_evt;
  const double *L_vth, *L_vl, *L_vh, *L_der, *L_vin, *ck_slew, *sp_at, *ck_cks, *ck_off, *REL;
  const unsigned char *is_port, *reg_ck;
  double *c1, *c2, *rpi, *ctot, *elm; unsigned char* par;
};
static DA da() {
  DA d;
  TB.A = g<double>("tabA"); TB.B = g<double>("tabB"); TB.V = g<double>("tabV");
  TB.NA = g<i64>("tabNA"); TB.NB = g<i64>("tabNB"); TB.M = gi("tabM"); TB.T = gi("tabT");
  d.P = gi("P"); d.K = gi("K"); d.nlev = gi("nlev"); d.nE = gi("nE"); d.nn = gi("nn"); d.def_lib = gi("def_lib"); d.nsp = gi("nsp");
  d.nrow = gi("nrow"); d.nevt = gi("nevt");
  d.ce_from = g<i64>("ce_from"); d.ce_to = g<i64>("ce_to"); d.ce_irf = g<i64>("ce_irf"); d.ce_orf = g<i64>("ce_orf");
  d.ce_lib = g<i64>("ce_lib"); d.ce_td = g<i64>("ce_td"); d.ce_ts = g<i64>("ce_ts"); d.ce_lo = g<i64>("ce_lo");
  d.g_lo = g<i64>("g_lo"); d.g_start = g<i64>("g_start"); d.g_to = g<i64>("g_to"); d.g_orf = g<i64>("g_orf");
  d.ne_lo = g<i64>("ne_lo"); d.ne_drv = g<i64>("ne_drv"); d.ne_snk = g<i64>("ne_snk"); d.pin_net = g<i64>("pin_net");
  d.sp_pin = g<i64>("sp_pin"); d.sp_cls = g<i64>("sp_cls"); d.sp_rf = g<i64>("sp_rf");
  d.ck_pin = g<i64>("ck_pin"); d.ck_tr = g<i64>("ck_tr"); d.ck_tf = g<i64>("ck_tf"); d.ck_type = g<i64>("ck_type"); d.ck_evt = g<i64>("ck_evt");
  d.L_vth = g<double>("L_vth"); d.L_vl = g<double>("L_vl"); d.L_vh = g<double>("L_vh"); d.L_der = g<double>("L_der");
  d.L_vin = g<double>("L_vin"); d.ck_slew = g<double>("ck_slew"); d.sp_at = g<double>("sp_at");
  d.ck_cks = g<double>("ck_cks"); d.ck_off = g<double>("ck_off"); d.REL = g<double>("REL");
  d.is_port = g<unsigned char>("is_port"); d.reg_ck = g<unsigned char>("reg_ck");
  d.c1 = (double*)g<double>("rc_c1"); d.c2 = (double*)g<double>("rc_c2"); d.rpi = (double*)g<double>("rc_rpi");
  d.ctot = (double*)g<double>("rc_ctot"); d.elm = (double*)g<double>("rc_elm"); d.par = (unsigned char*)g<unsigned char>("rc_par");
  return d;
}

// ---------------------------------------------------------------- one gate group (same code as csta_analyze lines 469-511)
struct GOut { double sl; DP dp; unsigned char pk; i64 pke; double rd; i64 pick_recomputed; };
static inline void arc_compute(const DA& d, i64 e, Dmp& r) {
  i64 frm = d.ce_from[e], to = d.ce_to[e], irf = d.ce_irf[e], orf = d.ce_orf[e], li = d.ce_lib[e];
  i64 net = d.pin_net[to];
  Arc A;
  A.td = d.ce_td[e]; A.ts = d.ce_ts[e]; A.sin = I_SL[frm * 2 + irf];
  A.c2 = d.c2[net * 2 + orf]; A.rpi = d.rpi[net * 2 + orf]; A.c1 = d.c1[net * 2 + orf];
  A.vth = d.L_vth[li * 2 + orf]; A.vl = d.L_vl[li * 2 + orf]; A.vh = d.L_vh[li * 2 + orf]; A.der = d.L_der[li];
  dmp_gate1(A, d.par[net] != 0, d.ctot[net * 2 + orf], r);
}
static const int STORED = -9;   // res[].alg marker: arc not re-solved, slew / delay taken from the stored I_as / I_gd
// all = true: every arc re-solved (full pass; RC of the output net changed). all = false: only the arcs flagged in aflag
// (their input slew changed); the others are exactly the stored results of the same inputs (dmp_gate1 is deterministic).
static void group_compute(const DA& d, i64 gg, bool all, Dmp* res, double* at_o, i64* best_o, GOut& o) {
  const i64 lv = I_glv[gg], s0 = d.g_start[gg], s1 = I_gs1[gg], b = d.ce_lo[lv + 1], K = d.K;
  for (i64 e = s0; e < s1; e++) {
    if (all || aflag[e]) arc_compute(d, e, res[e - s0]);
    else { res[e - s0].slew = I_as[e]; res[e - s0].delay = I_gd[e]; res[e - s0].alg = STORED; }
  }
  double smax = res[0].slew;
  for (i64 e = s0 + 1; e < s1; e++) smax = mx(smax, res[e - s0].slew);
  o.sl = smax; o.pick_recomputed = 0;
  i64 pk = -1;
  for (i64 e = s0; e < s1; e++) if (res[e - s0].slew >= smax) { pk = e; break; }
  if (pk < 0) pk = b;
  o.pke = pk; o.pk = pk < b; o.rd = 0.0;
  if (pk < b && res[pk - s0].alg == STORED && I_gpk[gg] && I_gpke[gg] == pk) {   // same picked arc, same inputs: same record
    o.dp = I_dps[d.g_to[gg] * 2 + d.g_orf[gg]]; o.rd = I_grd[gg];
  } else if (pk < b) {
    if (res[pk - s0].alg == STORED) { arc_compute(d, pk, res[pk - s0]); o.pick_recomputed = 1; }
    const Dmp& r = res[pk - s0];
    DP& dd = o.dp; memset(&dd, 0, sizeof(DP));
    dd.slew = r.slew; dd.t0 = r.t0; dd.dt = r.dt; dd.k0 = r.k0; dd.k1 = r.k1; dd.k2 = r.k2; dd.k3 = r.k3; dd.k4 = r.k4;
    dd.p1 = r.p1; dd.p2 = r.p2; dd.vub = r.vub; dd.vod = r.vod; dd.vth = r.vth; dd.vl = r.vl; dd.vh = r.vh; dd.der = r.der;
    dd.alg = r.alg; dd.valid = r.valid;
    o.rd = r.rd;
  } else o.dp = dp_init();
  for (i64 k = 0; k < K; k++) {
    double amax = NEG; bool first = true;
    for (i64 e = s0; e < s1; e++) {
      double c = I_AT[(d.ce_from[e] * K + k) * 2 + d.ce_irf[e]] + res[e - s0].delay;
      amax = first ? c : mx(amax, c); first = false;
    }
    at_o[k] = amax;
    i64 best = d.nE;
    for (i64 e = s0; e < s1; e++) {
      double c = I_AT[(d.ce_from[e] * K + k) * 2 + d.ce_irf[e]] + res[e - s0].delay;
      if (c >= amax - 1e-12) { best = e; break; }
    }
    best_o[k] = best;
  }
}
// ---------------------------------------------------------------- one net edge, one transition (csta_analyze lines 521-533)
static void edge_compute(const DA& d, i64 i, i64 rf, double* wd_o, double* ls_o, double* at_o) {
  i64 drv = d.ne_drv[i], snk = d.ne_snk[i];
  double elm = d.elm[snk * 2 + rf], wd, ls;
  dmp_load1(I_dps[drv * 2 + rf], elm, &wd, &ls);
  if (d.is_port[drv]) {
    bool okp = std::isfinite(elm);
    double vin = d.L_vin[d.def_lib * 2 + rf], vlo = d.L_vl[d.def_lib * 2 + rf], vhi = d.L_vh[d.def_lib * 2 + rf];
    wd = okp ? -elm * std::log(1.0 - vin) : 0.0;
    ls = I_SL[drv * 2 + rf] + (okp ? elm * std::log((1 - vlo) / (1 - vhi)) / d.L_der[d.def_lib] : 0.0);
  }
  *wd_o = wd; *ls_o = ls;
  for (i64 k = 0; k < d.K; k++) at_o[k] = I_AT[(drv * d.K + k) * 2 + rf] + wd;
}
// ---------------------------------------------------------------- one endpoint row (csta_analyze lines 547-568)
static double row_compute(const DA& d, i64 r, double* reqk_o, unsigned char* valid_o) {
  const i64 K = d.K;
  i64 p = d.ck_pin[r];
  double setup[2];
  for (int rf = 0; rf < 2; rf++) {
    i64 tid = rf == 0 ? d.ck_tr[r] : d.ck_tf[r];
    setup[rf] = 0.0;
    if (tid >= 0) setup[rf] = lut(tid, d.ck_cks[r], I_SL[p * 2 + rf]);
    else if (d.ck_type[r] <= 1) setup[rf] = NaN;
  }
  double rmin = BIG;
  for (i64 k = 0; k < K; k++) {
    double req = d.REL[k * d.nevt + d.ck_evt[r]];
    for (int rf = 0; rf < 2; rf++) {
      double rk = req - setup[rf] - d.ck_off[r];
      double at = I_AT[(p * K + k) * 2 + rf];
      bool v = (at > NEG / 2) && (req < BIG / 2) && std::isfinite(rk);
      reqk_o[k * 2 + rf] = rk;
      valid_o[k * 2 + rf] = v;
      double sl = v ? rk - at : BIG;
      rmin = mn(rmin, sl);
    }
  }
  return rmin;
}

static void csr(i64 n, const std::vector<std::pair<i64, i64>>& kv, std::vector<i64>& off, std::vector<i64>& val) {
  off.assign(n + 1, 0);
  for (auto& q : kv) off[q.first + 1]++;
  for (i64 i = 0; i < n; i++) off[i + 1] += off[i];
  val.resize(kv.size());
  std::vector<i64> f(off.begin(), off.end() - 1);
  for (auto& q : kv) val[f[q.first]++] = q.second;   // kv given in increasing value order -> lists sorted
}

// ---------------------------------------------------------------- T5 (idea 1b): parallel required-time pass
// Per level, the net edges are grouped by driver pin and the arcs by (from pin, rf); each group is one task that applies its
// updates in the original (ascending edge / arc) order, so every RAT entry sees the same sequence of mn() as in the serial loop.
// Tasks of one phase write disjoint RAT entries; reads hit entries no task of the same phase writes, PROVIDED no pin is both a
// net-edge driver and a net-edge sink, and no pin is both an arc start and an arc end (checked here; else the serial loop runs).
static std::vector<i64> bwE, bwEt, bwEl, bwA, bwAt, bwAl;   // permuted ids, task starts, first task per level (CSR)
static bool bw_par = false;
static i64 bw_conf = 0;
static void bw_bind(const DA& d) {
  const i64 nlev = d.nlev;
  bwE.clear(); bwEt.clear(); bwA.clear(); bwAt.clear(); bwEl.assign(nlev + 1, 0); bwAl.assign(nlev + 1, 0);
  std::vector<i64> tmp;
  for (i64 lv = 0; lv < nlev; lv++) {
    bwEl[lv] = (i64)bwEt.size(); bwAl[lv] = (i64)bwAt.size();
    if (lv == 0) continue;
    tmp.clear(); for (i64 i = d.ne_lo[lv]; i < d.ne_lo[lv + 1]; i++) tmp.push_back(i);
    std::stable_sort(tmp.begin(), tmp.end(), [&](i64 a, i64 b) { return d.ne_drv[a] < d.ne_drv[b]; });
    for (size_t t = 0; t < tmp.size(); t++) { if (t == 0 || d.ne_drv[tmp[t]] != d.ne_drv[tmp[t - 1]]) bwEt.push_back((i64)bwE.size()); bwE.push_back(tmp[t]); }
    tmp.clear(); for (i64 e = d.ce_lo[lv]; e < d.ce_lo[lv + 1]; e++) tmp.push_back(e);
    auto key = [&](i64 e) { return d.ce_from[e] * 2 + d.ce_irf[e]; };
    std::stable_sort(tmp.begin(), tmp.end(), [&](i64 a, i64 b) { return key(a) < key(b); });
    for (size_t t = 0; t < tmp.size(); t++) { if (t == 0 || key(tmp[t]) != key(tmp[t - 1])) bwAt.push_back((i64)bwA.size()); bwA.push_back(tmp[t]); }
  }
  bwEl[nlev] = (i64)bwEt.size(); bwAl[nlev] = (i64)bwAt.size();
  bwEt.push_back((i64)bwE.size()); bwAt.push_back((i64)bwA.size());
  std::vector<unsigned char> f((size_t)d.P, 0);
  for (i64 i = d.ne_lo[1]; i < d.ne_lo[nlev]; i++) f[d.ne_drv[i]] |= 1, f[d.ne_snk[i]] |= 2;
  for (i64 e = d.ce_lo[1]; e < d.ce_lo[nlev]; e++) f[d.ce_from[e]] |= 4, f[d.ce_to[e]] |= 8;
  bw_conf = 0;
  for (i64 p = 0; p < d.P; p++) bw_conf += ((f[p] & 3) == 3) + ((f[p] & 12) == 12);
  bw_par = bw_conf == 0 && !getenv("CSTA5_SERIAL_BWD");
}
// == the backward block of csta_inc_full / w3_inc_backward (RAT, pin slack, net slack), scheduled in parallel
static void bwd_par(const DA& d) {
  const i64 P = d.P, K = d.K, nn = d.nn;
  // RAT = BIG, then the rows of each pin in ascending row order (prow lists them so) = the serial row loop per entry
  pfor(P, ach(P, 1 << 14), [&](i64 lo_, i64 hi_) {
    for (i64 p = lo_; p < hi_; p++) {
      for (i64 q = p * K * 2; q < (p + 1) * K * 2; q++) I_RAT[q] = BIG;
      for (i64 t = prow_o[p]; t < prow_o[p + 1]; t++) {
        i64 r = prow[t];
        for (i64 k = 0; k < K; k++)
          for (int rf = 0; rf < 2; rf++) {
            i64 ix = (p * K + k) * 2 + rf;
            I_RAT[ix] = mn(I_RAT[ix], S_valid[(r * K + k) * 2 + rf] ? S_reqk[(r * K + k) * 2 + rf] : BIG);
          }
      }
    } });
  for (i64 lv = d.nlev - 1; lv >= 1; lv--) {
    i64 t0 = bwEl[lv], t1 = bwEl[lv + 1];
    pfor(t1 - t0, ach(t1 - t0, 512), [&](i64 lo_, i64 hi_) {
      for (i64 t = t0 + lo_; t < t0 + hi_; t++)
        for (i64 j = bwEt[t]; j < bwEt[t + 1]; j++) {
          i64 i = bwE[j], drv = d.ne_drv[i], snk = d.ne_snk[i];
          for (i64 k = 0; k < K; k++)
            for (int rf = 0; rf < 2; rf++) {
              i64 ix = (drv * K + k) * 2 + rf;
              I_RAT[ix] = mn(I_RAT[ix], I_RAT[(snk * K + k) * 2 + rf] - I_wd[snk * 2 + rf]);
            }
        } });
    t0 = bwAl[lv]; t1 = bwAl[lv + 1];
    pfor(t1 - t0, ach(t1 - t0, 512), [&](i64 lo_, i64 hi_) {
      for (i64 t = t0 + lo_; t < t0 + hi_; t++)
        for (i64 j = bwAt[t]; j < bwAt[t + 1]; j++) {
          i64 e = bwA[j], to = d.ce_to[e], orf = d.ce_orf[e], fr = d.ce_from[e], irf = d.ce_irf[e];
          for (i64 k = 0; k < K; k++) {
            double v = I_RAT[(to * K + k) * 2 + orf] - I_gd[e];
            i64 ix = (fr * K + k) * 2 + irf;
            I_RAT[ix] = mn(I_RAT[ix], v);
          }
        } });
  }
  pfor(P, ach(P, 1 << 14), [&](i64 lo_, i64 hi_) {
    for (i64 p = lo_; p < hi_; p++) {
      double m = BIG;
      for (i64 k = 0; k < K; k++)
        for (int rf = 0; rf < 2; rf++) {
          i64 ix = (p * K + k) * 2 + rf;
          bool ok = (I_AT[ix] > NEG / 2) && (I_RAT[ix] < BIG / 2);
          m = mn(m, ok ? I_RAT[ix] - I_AT[ix] : BIG);
        }
      I_psl[p] = m;
    } });
  for (i64 i = 0; i < nn; i++) I_nsl[i] = BIG;
  for (i64 p = 0; p < P; p++) { i64 n = d.pin_net[p]; if (n < 0) n += nn; I_nsl[n] = mn(I_nsl[n], I_psl[p]); }
}

// info[0..15]: P K nlev nE nn nrow nG nNE orphan_arcs to_mismatch neg_net multi_writer order_viol nsp_overwritten 0 0
extern "C" int csta_inc_bind(i64* info) {
  DA d = da();
  I_P = d.P; I_K = d.K; I_nE = d.nE; I_nn = d.nn; I_nlev = d.nlev; I_nrow = d.nrow;
  I_g0 = d.g_lo[1]; I_g1 = d.g_lo[d.nlev]; I_e0 = d.ne_lo[1]; I_e1 = d.ne_lo[d.nlev];
  const i64 nG = I_g1, nNE = I_e1;
  i64 orphan = 0, tomis = 0, negnet = 0, multi = 0, viol = 0, spov = 0;
  I_glv.assign(nG, 0); I_gs1.assign(nG, 0); I_elv.assign(nNE, 0);
  for (i64 lv = 1; lv < d.nlev; lv++) {
    i64 a = d.ce_lo[lv], b = d.ce_lo[lv + 1], ga = d.g_lo[lv], gb = d.g_lo[lv + 1];
    orphan += (gb > ga) ? (d.g_start[ga] - a) : (b - a);
    for (i64 gg = ga; gg < gb; gg++) {
      I_glv[gg] = lv; I_gs1[gg] = (gg + 1 < gb) ? d.g_start[gg + 1] : b;
      if (I_gs1[gg] <= d.g_start[gg]) tomis++;                       // empty group (python would fail)
      for (i64 e = d.g_start[gg]; e < I_gs1[gg]; e++) if (d.ce_to[e] != d.g_to[gg] || d.ce_orf[e] != d.g_orf[gg]) tomis++;
      if (d.pin_net[d.g_to[gg]] < 0) negnet++;
    }
    for (i64 i = d.ne_lo[lv]; i < d.ne_lo[lv + 1]; i++) { I_elv[i] = lv; if (d.pin_net[d.ne_snk[i]] < 0) negnet++; }
  }
  // writers of (pin, rf): step 2*lv (group) or 2*lv+1 (net edge)
  std::vector<i64> ws((size_t)d.P * 2, -1);
  for (i64 gg = I_g0; gg < nG; gg++) { i64 key = d.g_to[gg] * 2 + d.g_orf[gg]; if (ws[key] >= 0) multi++; ws[key] = 2 * I_glv[gg]; }
  for (i64 i = I_e0; i < nNE; i++) for (int rf = 0; rf < 2; rf++) {
    i64 key = d.ne_snk[i] * 2 + rf; if (ws[key] >= 0) multi++; ws[key] = 2 * I_elv[i] + 1; }
  for (i64 i = 0; i < d.nsp; i++) if (ws[d.sp_pin[i] * 2 + d.sp_rf[i]] >= 0) spov++;    // fine (writer overwrites), reported
  for (i64 gg = I_g0; gg < nG; gg++)
    for (i64 e = d.g_start[gg]; e < I_gs1[gg]; e++) { i64 w = ws[d.ce_from[e] * 2 + d.ce_irf[e]]; if (w >= 2 * I_glv[gg]) viol++; }
  for (i64 i = I_e0; i < nNE; i++) for (int rf = 0; rf < 2; rf++) { i64 w = ws[d.ne_drv[i] * 2 + rf]; if (w >= 2 * I_elv[i] + 1) viol++; }
  // reverse indices
  std::vector<std::pair<i64, i64>> kv;
  kv.clear(); for (i64 gg = I_g0; gg < nG; gg++) { i64 n = d.pin_net[d.g_to[gg]]; if (n >= 0) kv.push_back({n, gg}); }
  csr(d.nn, kv, netg_o, netg);
  kv.clear(); for (i64 i = I_e0; i < nNE; i++) { i64 n = d.pin_net[d.ne_snk[i]]; if (n >= 0) kv.push_back({n, i}); }
  csr(d.nn, kv, nete_o, nete);
  kv.clear(); for (i64 i = I_e0; i < nNE; i++) kv.push_back({d.ne_drv[i], i});
  csr(d.P, kv, drve_o, drve);
  kv.clear();
  { std::vector<i64> last((size_t)d.P * 2, -1);
    for (i64 gg = I_g0; gg < nG; gg++)
      for (i64 e = d.g_start[gg]; e < I_gs1[gg]; e++) { i64 key = d.ce_from[e] * 2 + d.ce_irf[e]; if (last[key] != gg) { last[key] = gg; kv.push_back({key, gg}); } } }
  csr(d.P * 2, kv, rdg_o, rdg);
  kv.clear(); I_agrp.assign(d.nE, -1);
  for (i64 gg = I_g0; gg < nG; gg++) for (i64 e = d.g_start[gg]; e < I_gs1[gg]; e++) I_agrp[e] = gg;
  for (i64 e = 0; e < d.nE; e++) if (I_agrp[e] >= 0) kv.push_back({d.ce_from[e] * 2 + d.ce_irf[e], e});
  csr(d.P * 2, kv, rda_o, rda);
  kv.clear(); for (i64 r = 0; r < d.nrow; r++) kv.push_back({d.ck_pin[r], r});
  csr(d.P, kv, prow_o, prow);
  bw_bind(d);                                                        // T5: tasks of the parallel required-time pass
  { kv.clear(); for (i64 p = 0; p < d.P; p++) if (d.pin_net[p] >= 0) kv.push_back({d.pin_net[p], p});   // T5: pins per net
    csr(d.nn, kv, pn_o, pn);
    SH_c1.assign((size_t)d.nn * 2, 0.0); SH_c2 = SH_c1; SH_rpi = SH_c1; SH_ctot = SH_c1; SH_par.assign(d.nn, 0); SH_elm.assign((size_t)d.P * 2, 0.0); sh_clear(); }
  // AP pins per net (the pins whose Elmore value build_rc of that net can write)
  { const i64 *ap_off = g<i64>("ap_off"), *ap_idx = g<i64>("ap_idx"), *ap_pid = g<i64>("ap_pid");
    apin_o.assign(d.nn + 1, 0); apin.clear();
    for (i64 n = 0; n < d.nn; n++) { for (i64 t = ap_off[n]; t < ap_off[n + 1]; t++) apin.push_back(ap_pid[ap_idx[t]]); apin_o[n + 1] = (i64)apin.size(); } }
  // state
  I_AT.assign((size_t)d.P * d.K * 2, NEG); I_BEST.assign((size_t)d.P * d.K * 2, -1);
  I_SL.assign((size_t)d.P * 2, 0.0); I_wd.assign((size_t)d.P * 2, 0.0); I_gd.assign(d.nE, 0.0); I_as.assign(d.nE, 0.0); I_nrd.assign(d.nn, 0.0);
  I_rmin.assign(d.nrow, 0.0); I_ep.assign(d.P, BIG); I_grd.assign(nG, 0.0); I_gpk.assign(nG, 0); I_gpke.assign(nG, 0);
  I_dps.assign((size_t)d.P * 2, dp_init());
  I_RAT.assign((size_t)d.P * d.K * 2, BIG); I_psl.assign(d.P, BIG); I_nsl.assign(d.nn, BIG);
  Gd.assign(d.nlev + 1, {}); Ed.assign(d.nlev + 1, {});
  gflag.assign(nG, 0); eflag.assign((size_t)nNE * 2, 0); rflag.assign(d.nrow, 0); pflag.assign(d.P, 0); nflag.assign(d.nn, 0);
  rdflag.assign(d.nn, 0); aflag.assign(d.nE, 0);
  lD.clear(); lDq.clear(); pDq.clear(); lI.clear(); lU.clear(); lUq.clear(); pUq.clear(); lP.clear();
  I_ready = (orphan == 0 && tomis == 0 && negnet == 0 && multi == 0 && viol == 0);
  I_full = false;
  i64 v[16] = {d.P, d.K, d.nlev, d.nE, d.nn, d.nrow, nG - I_g0, nNE - I_e0, orphan, tomis, negnet, multi, viol, spov, 0, 0};
  for (int i = 0; i < 16; i++) info[i] = v[i];
  return I_ready ? 0 : 1;
}

// ---------------------------------------------------------------- full analysis into the persistent state
// stats: [0] forward [1] endpoints [2] backward [3] total
static std::vector<std::array<double, 4>> LVP;                            // T5: per-level walls (CSTA5_LVPROF), diagnostics
extern "C" i64 csta5_lvprof(double* o) { if (o) memcpy(o, LVP.data(), LVP.size() * 32); return (i64)LVP.size(); }
static int full_scratch(int backward, double* stats) {                     // T5: = the original csta_inc_full (+ RC shadow)
  double tA = now();
  static const bool lvp = getenv("CSTA5_LVPROF") != nullptr; LVP.clear();
  DA d = da();
  const i64 P = d.P, K = d.K, nE = d.nE, nn = d.nn;
  // T5: the resets in parallel (pure stores; the sequential overwrites by startpoints / clock pins follow as before)
  pfor(P, ach(P, 1 << 16), [&](i64 lo_, i64 hi_) {
    for (i64 i = lo_ * K * 2; i < hi_ * K * 2; i++) { I_AT[i] = NEG; I_BEST[i] = -1; }
    for (i64 i = lo_ * 2; i < hi_ * 2; i++) { I_SL[i] = 0.0; I_wd[i] = 0.0; }
    const DP z = dp_init();
    for (i64 i = lo_ * 2; i < hi_ * 2; i++) I_dps[i] = z; });
  pfor(nE, ach(nE, 1 << 16), [&](i64 lo_, i64 hi_) { for (i64 i = lo_; i < hi_; i++) { I_gd[i] = 0.0; I_as[i] = 0.0; } });
  for (i64 i = 0; i < nn; i++) I_nrd[i] = 0.0;
  for (i64 i = 0; i < d.nsp; i++) I_AT[(d.sp_pin[i] * K + d.sp_cls[i]) * 2 + d.sp_rf[i]] = d.sp_at[i];
  for (i64 p = 0; p < P; p++) if (d.reg_ck[p]) { I_SL[p * 2] = d.ck_slew[p * 2]; I_SL[p * 2 + 1] = d.ck_slew[p * 2 + 1]; }
  for (i64 gg = 0; gg < (i64)I_gpk.size(); gg++) { I_gpk[gg] = 0; I_grd[gg] = 0.0; I_gpke[gg] = 0; }
  double tq = now();
  if (lvp) LVP.push_back({0.0, 4.0, 0.0, tq - tA});
  for (i64 lv = 1; lv < d.nlev; lv++) {
    i64 ga = d.g_lo[lv], gb = d.g_lo[lv + 1];
    if (gb > ga) {
      pfor(gb - ga, ach(gb - ga, 64), [&](i64 lo_, i64 hi_) {                // T5: chunk by work (was 64)
        thread_local std::vector<Dmp> res; thread_local std::vector<double> at; thread_local std::vector<i64> bs;
        at.resize(K); bs.resize(K);
        for (i64 gg = ga + lo_; gg < ga + hi_; gg++) {
          i64 s0 = d.g_start[gg], s1 = I_gs1[gg];
          if ((i64)res.size() < s1 - s0) res.resize(s1 - s0);
          GOut o; group_compute(d, gg, true, res.data(), at.data(), bs.data(), o);
          for (i64 e = s0; e < s1; e++) { I_gd[e] = res[e - s0].delay; I_as[e] = res[e - s0].slew; }
          i64 gto = d.g_to[gg], gorf = d.g_orf[gg];
          I_SL[gto * 2 + gorf] = o.sl;
          if (o.pk) I_dps[gto * 2 + gorf] = o.dp;
          for (i64 k = 0; k < K; k++) { I_AT[(gto * K + k) * 2 + gorf] = at[k]; I_BEST[(gto * K + k) * 2 + gorf] = bs[k]; }
          I_gpk[gg] = o.pk; I_grd[gg] = o.rd; I_gpke[gg] = o.pke;
        }
      });
      if (lvp) LVP.push_back({(double)lv, 0.0, (double)(gb - ga), now() - tq}), tq = now();
      for (i64 gg = ga; gg < gb; gg++) if (I_gpk[gg]) { i64 net = d.pin_net[d.ce_to[I_gpke[gg]]]; I_nrd[net] = mx(I_nrd[net], I_grd[gg]); }
      if (lvp) LVP.push_back({(double)lv, 1.0, (double)(gb - ga), now() - tq}), tq = now();
    }
    i64 a = d.ne_lo[lv], b = d.ne_lo[lv + 1];
    if (b > a) {
      i64 nb = b - a;
      pfor(2 * nb, ach(2 * nb, 128), [&](i64 lo_, i64 hi_) {                // T5: chunk by work (was 128)
        thread_local std::vector<double> at; at.resize(K);
        for (i64 j = lo_; j < hi_; j++) {
          i64 rf = j / nb, i = a + (j % nb), snk = d.ne_snk[i];
          double wd, ls; edge_compute(d, i, rf, &wd, &ls, at.data());
          I_wd[snk * 2 + rf] = wd; I_SL[snk * 2 + rf] = ls;
          for (i64 k = 0; k < K; k++) I_AT[(snk * K + k) * 2 + rf] = at[k];
        }
      });
      if (lvp) LVP.push_back({(double)lv, 2.0, (double)(2 * nb), now() - tq}), tq = now();
    }
  }
  double tB = now();
  if (lvp) LVP.push_back({-1.0, 3.0, 0.0, tB - tA});
  S_reqk.assign((size_t)d.nrow * K * 2, 0.0); S_valid.assign((size_t)d.nrow * K * 2, 0);
  pfor(d.nrow, ach(d.nrow, 4096), [&](i64 lo_, i64 hi_) { for (i64 r = lo_; r < hi_; r++)
    I_rmin[r] = row_compute(d, r, S_reqk.data() + r * K * 2, S_valid.data() + r * K * 2); });
  for (i64 p = 0; p < P; p++) I_ep[p] = BIG;
  for (i64 r = 0; r < d.nrow; r++) I_ep[d.ck_pin[r]] = mn(I_ep[d.ck_pin[r]], I_rmin[r]);
  double tC = now();
  if (backward && bw_par) bwd_par(d);                                       // T5: parallel, same mn() sequence per entry
  else if (backward) {   // same code as csta_analyze lines 574-611
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
  }
  double tD = now();
  lD.clear(); lDq.clear(); pDq.clear(); lI.clear(); lU.clear(); lUq.clear(); pUq.clear(); lP.clear();
  I_full = true;
  memcpy(SH_c1.data(), d.c1, sizeof(double) * nn * 2); memcpy(SH_c2.data(), d.c2, sizeof(double) * nn * 2);   // T5: shadow
  memcpy(SH_rpi.data(), d.rpi, sizeof(double) * nn * 2); memcpy(SH_ctot.data(), d.ctot, sizeof(double) * nn * 2);
  memcpy(SH_par.data(), d.par, nn); memcpy(SH_elm.data(), d.elm, sizeof(double) * P * 2); sh_clear();
  T5_ana = false; T5_scratch_full++;
  stats[0] = tB - tA; stats[1] = tC - tB; stats[2] = tD - tC; stats[3] = tD - tA;
  return 0;
}

// T5 (idea 7): the full pass as an incremental update when the persistent state is valid. The nets whose bound RC differs (bit
// for bit) from the shadow are the seeds; csta_inc_update re-times their fan-out cones and stops where the values are bit-equal,
// which is the same state the full pass computes (the property the update was built and tested for, and checked here by replay
// against the old library's full pass after every call). Pending required-time rows of a reject are applied first (the full pass
// recomputes all rows from the restored state). Then the backward pass as in the full pass, all logs dropped (as the full pass).
// From scratch when: no valid state, the original csta_analyze ran, more than 1/4 of the nets changed, CSTA5_SCRATCH is set,
// or the update reports a late mark (not expected; the lists are cleared first).
extern "C" i64 csta_inc_update(const i64* nets, i64 nnet, i64* cnt, double* stats);
extern "C" void csta_inc_begin();
extern "C" int csta_inc_full(int backward, double* stats) {
  double tA = now();
  static const bool off = getenv("CSTA5_SCRATCH") != nullptr;
  if (off || !I_ready || !I_full || T5_ana || (backward && !bw_par)) return full_scratch(backward, stats);
  DA d = da();
  const i64 P = d.P, K = d.K, nn = d.nn;
  static std::vector<unsigned char> chn; chn.assign(nn, 0);
  pfor(nn, ach(nn, 1 << 14), [&](i64 lo_, i64 hi_) {
    for (i64 n = lo_; n < hi_; n++) {
      bool c = d.par[n] != SH_par[n];
      for (int rf = 0; rf < 2 && !c; rf++) {
        i64 q = n * 2 + rf;
        c = !beq(d.c1[q], SH_c1[q]) || !beq(d.c2[q], SH_c2[q]) || !beq(d.rpi[q], SH_rpi[q]) || !beq(d.ctot[q], SH_ctot[q]);
      }
      for (i64 t = pn_o[n]; t < pn_o[n + 1] && !c; t++) { i64 p = pn[t]; c = !beq(d.elm[p * 2], SH_elm[p * 2]) || !beq(d.elm[p * 2 + 1], SH_elm[p * 2 + 1]); }
      chn[n] = c;
    } });
  std::vector<i64> nl;
  for (i64 n = 0; n < nn; n++) if (chn[n]) nl.push_back(n);
  if ((i64)nl.size() * 4 > nn) return full_scratch(backward, stats);
  csta_inc_begin();                                                        // pending rows applied, logs dropped
  i64 cnt[16]; double st[3];
  i64 ret = csta_inc_update(nl.data(), (i64)nl.size(), cnt, st);
  if (ret != 0) {                                                          // not expected: clean the dirty lists, from scratch
    for (auto& v : Gd) { for (i64 gg : v) gflag[gg] = 0; v.clear(); }
    for (auto& v : Ed) { for (i64 k : v) eflag[k] = 0; v.clear(); }
    for (i64 r : rlist) rflag[r] = 0;
    rlist.clear(); std::fill(aflag.begin(), aflag.end(), 0); std::fill(rdflag.begin(), rdflag.end(), 0);
    return full_scratch(backward, stats);
  }
  double tC = now();
  if (backward) bwd_par(d);
  double tD = now();
  lD.clear(); lDq.clear(); pDq.clear(); lI.clear(); lU.clear(); lUq.clear(); pUq.clear(); lP.clear(); sh_clear();
  I_full = true; T5_inc_full++; T5_inc_nets += (i64)nl.size();
  (void)P; (void)K;
  stats[0] = st[0]; stats[1] = st[1]; stats[2] = tD - tC; stats[3] = tD - tA;
  return 0;
}
extern "C" void csta5_fullcnt(i64* o) { o[0] = T5_scratch_full; o[1] = T5_inc_full; o[2] = T5_inc_nets; }

// ---------------------------------------------------------------- transactions
extern "C" void csta_inc_begin() {
  for (i64 i = (i64)pDq.size() - 1; i >= 0; i--) *pDq[i].p = pDq[i].v;
  for (i64 i = (i64)pUq.size() - 1; i >= 0; i--) *pUq[i].p = pUq[i].v;
  pDq.clear(); pUq.clear();
  lD.clear(); lDq.clear(); lI.clear(); lU.clear(); lUq.clear(); lP.clear(); sh_clear();
}
extern "C" void csta_inc_commit() { lD.clear(); lDq.clear(); lI.clear(); lU.clear(); lUq.clear(); lP.clear(); sh_clear(); }
extern "C" i64 csta_inc_reject() {
  i64 n = (i64)(lD.size() + lI.size() + lU.size() + lP.size());
  for (i64 i = (i64)lD.size() - 1; i >= 0; i--) *lD[i].p = lD[i].v;
  for (i64 i = (i64)lI.size() - 1; i >= 0; i--) *lI[i].p = lI[i].v;
  for (i64 i = (i64)lU.size() - 1; i >= 0; i--) *lU[i].p = lU[i].v;
  for (i64 i = (i64)lP.size() - 1; i >= 0; i--) *lP[i].p = lP[i].v;
  sh_undo();                                                               // T5
  pDq.swap(lDq); pUq.swap(lUq);   // required-time rows of the rejected state stay visible until the next begin
  lD.clear(); lDq.clear(); lI.clear(); lU.clear(); lUq.clear(); lP.clear();
  return n;
}
extern "C" void csta_inc_flush_pending() {
  for (i64 i = (i64)pDq.size() - 1; i >= 0; i--) *pDq[i].p = pDq[i].v;
  for (i64 i = (i64)pUq.size() - 1; i >= 0; i--) *pUq[i].p = pUq[i].v;
  pDq.clear(); pUq.clear();
}

// RC of nets copied from given arrays (test driver; logged)
extern "C" int csta_inc_set_rc(const i64* nets, i64 k, const double* c1, const double* c2, const double* rpi, const double* ctot,
                               const unsigned char* par, const double* elm) {
  DA d = da();
  for (i64 q = 0; q < k; q++) {
    i64 n = nets[q];
    for (int rf = 0; rf < 2; rf++) {
      wD(&d.c1[n * 2 + rf], c1[n * 2 + rf]); wD(&d.c2[n * 2 + rf], c2[n * 2 + rf]); wD(&d.rpi[n * 2 + rf], rpi[n * 2 + rf]);
      wD(&d.ctot[n * 2 + rf], ctot[n * 2 + rf]);
    }
    wU(&d.par[n], par[n]);
    for (i64 t = apin_o[n]; t < apin_o[n + 1]; t++) { i64 p = apin[t]; wD(&d.elm[p * 2], elm[p * 2]); wD(&d.elm[p * 2 + 1], elm[p * 2 + 1]); }
  }
  return 0;
}

// ---------------------------------------------------------------- RC of one net (verbatim body of the csta_build_rc lambda)
struct RCC { i64 P, nn, nlr, nvr; double UMGC; const i64 *ap_off, *ap_idx, *ap, *ap_pid, *net_drv; const double *pin_cap, *layer_rc, *via_r, *lumped;
             const unsigned char* pin_is_drv; };
// (only change: the per-net outputs c1/c2/rpi/ctot/par go to row oi of the caller's scratch instead of row n)
static void rc_one(const RCC& C, i64 n, i64 oi, const i64* seg, const i64* sg, i64 m, double stub_um, double* c1o, double* c2o, double* rpio,
                   double* ctoto, unsigned char* paro, double* elmo) {
  const i64 *ap_off = C.ap_off, *ap_idx = C.ap_idx, *ap = C.ap, *ap_pid = C.ap_pid, *net_drv = C.net_drv;
  const double *pin_cap = C.pin_cap, *layer_rc = C.layer_rc, *via_r = C.via_r;
  const unsigned char* pin_is_drv = C.pin_is_drv;
  const i64 nlr = C.nlr, nvr = C.nvr; const double UMGC = C.UMGC;
  const int L = 16;
  thread_local std::vector<i64> uk, ia, ib, adjo, adje, adjn, par, depth, order, stackn, stacki, bucket;
  thread_local std::vector<double> Rs, Cs, ncap, cap, eR, Rp, y1, y2, y3, elm;
  thread_local std::vector<PinRec> pins, pins2;
  thread_local std::vector<unsigned char> reach;
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
  const i64 E = m + npn;
  eR.resize(E);
  std::vector<i64>& eu = stackn; std::vector<i64>& ev = stacki;
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
  i64 drv = net_drv[n];
  bool hasd = drv >= 0;
  paro[oi] = hasd;
  i64 root = -1;
  if (hasd) { auto it = std::lower_bound(pins2.begin(), pins2.end(), drv, [](const PinRec& a, i64 v) { return a.pin < v; });
    if (it != pins2.end() && it->pin == drv) root = NN + (it - pins2.begin()); }
  if (root < 0) return;
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
  i64 maxd = 0; for (i64 w : order) maxd = std::max(maxd, depth[w]);
  std::vector<i64> lstart(maxd + 2, 0);
  for (i64 w : order) lstart[depth[w] + 1]++;
  for (i64 dd = 0; dd <= maxd; dd++) lstart[dd + 1] += lstart[dd];
  std::vector<i64> nodes_sorted(order.size());
  std::sort(order.begin(), order.end());
  { std::vector<i64> fp(lstart.begin(), lstart.end() - 1); for (i64 w : order) nodes_sorted[fp[depth[w]]++] = w; }
  y1.assign(NT * 2, 0.0); y2.assign(NT * 2, 0.0); y3.assign(NT * 2, 0.0);
  for (i64 w : order) { y1[w * 2] = cap[w * 2]; y1[w * 2 + 1] = cap[w * 2 + 1]; }
  for (i64 dd = maxd; dd >= 1; dd--) {
    for (i64 t = lstart[dd]; t < lstart[dd + 1]; t++) {
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
  for (i64 dd = 1; dd <= maxd; dd++)
    for (i64 t = lstart[dd]; t < lstart[dd + 1]; t++) {
      i64 w = nodes_sorted[t], p = par[w];
      for (int rf = 0; rf < 2; rf++) elm[w * 2 + rf] = elm[p * 2 + rf] + Rp[w] * y1[w * 2 + rf] * 1e-3;
    }
  for (int rf = 0; rf < 2; rf++) {
    double Y1 = y1[root * 2 + rf], Y2 = y2[root * 2 + rf], Y3 = y3[root * 2 + rf];
    bool capld = (Y2 == 0) && (Y3 == 0);
    double c1 = capld ? Y1 : Y2 * Y2 / Y3;
    double c2 = capld ? 0.0 : Y1 - Y2 * Y2 / Y3;
    double rpi = capld ? 0.0 : -Y3 * Y3 / (Y2 * Y2 * Y2);
    c1o[oi * 2 + rf] = nn0(c1); c2o[oi * 2 + rf] = nn0(c2); rpio[oi * 2 + rf] = nn0(rpi);
    ctoto[oi * 2 + rf] = Y1;
  }
  for (i64 k = 0; k < npn; k++) {
    i64 node = NN + k, pin = pins2[k].pin;
    if (!reach[node]) continue;
    if (pin_is_drv[pin]) continue;
    elmo[pin * 2] = elm[node * 2]; elmo[pin * 2 + 1] = elm[node * 2 + 1];
  }
}

// RC of the nets `nets` from the rows of `seg` that belong to them (other rows ignored), into the bound rc_* arrays.
// stats: [0] prep (log+reset+bucket) [1] per-net work [2] total
extern "C" int csta_inc_build_rc_subset(const i64* seg, i64 S, double stub_um, const i64* nets, i64 k, double* stats) {
  double t0w = now();
  DA d = da();
  RCC C;
  C.P = gi("P"); C.nn = gi("nn"); C.UMGC = *g<double>("um_per_gc");
  C.ap_off = g<i64>("ap_off"); C.ap_idx = g<i64>("ap_idx"); C.ap = g<i64>("ap"); C.ap_pid = g<i64>("ap_pid"); C.net_drv = g<i64>("net_drv");
  C.pin_cap = g<double>("pin_cap"); C.layer_rc = g<double>("layer_rc"); C.via_r = g<double>("via_r"); C.lumped = g<double>("lumped");
  C.pin_is_drv = g<unsigned char>("pin_is_drv"); C.nlr = gi("nlr"); C.nvr = gi("nvr");
  // log + reset to the defaults of csta_build_rc
  std::vector<i64> nl; nl.reserve(k);
  static std::vector<i64> pos; if ((i64)pos.size() != C.nn) pos.assign(C.nn, -1);
  for (i64 q = 0; q < k; q++) {
    i64 n = nets[q]; if (n < 0 || n >= C.nn || pos[n] >= 0) continue;
    pos[n] = (i64)nl.size(); nl.push_back(n);
    for (int rf = 0; rf < 2; rf++) {
      wD(&d.c1[n * 2 + rf], 0.0); wD(&d.c2[n * 2 + rf], 0.0); wD(&d.rpi[n * 2 + rf], 0.0); wD(&d.ctot[n * 2 + rf], C.lumped[n * 2 + rf]);
    }
    wU(&d.par[n], 0);
    // Elmore values are written below without logging (in parallel): log the old value unconditionally here
    for (i64 t = apin_o[n]; t < apin_o[n + 1]; t++) { i64 p = apin[t];
      lD.push_back({&d.elm[p * 2], d.elm[p * 2]}); lD.push_back({&d.elm[p * 2 + 1], d.elm[p * 2 + 1]}); d.elm[p * 2] = NaN; d.elm[p * 2 + 1] = NaN; }
  }
  // c1/c2/rpi/ctot/par: reset logged above (when it changed a value), final values logged against the reset value below
  const i64 nk = (i64)nl.size();
  std::vector<i64> cnt(nk + 1, 0);
  std::vector<unsigned char> keep(S);
  for (i64 s = 0; s < S; s++) {
    const i64* r = seg + s * 7;
    bool kp = r[0] >= 0 && r[0] < C.nn && pos[r[0]] >= 0 && ((r[3] != r[6]) || (r[1] != r[4]) || (r[2] != r[5]));
    keep[s] = kp;
    if (kp) cnt[pos[r[0]] + 1]++;
  }
  for (i64 i = 0; i < nk; i++) cnt[i + 1] += cnt[i];
  std::vector<i64> sidx(cnt[nk]), fillp(cnt.begin(), cnt.end() - 1);
  for (i64 s = 0; s < S; s++) if (keep[s]) sidx[fillp[pos[seg[s * 7]]]++] = s;
  std::vector<i64> work;
  for (i64 i = 0; i < nk; i++) if (cnt[i + 1] > cnt[i]) work.push_back(i);
  std::sort(work.begin(), work.end(), [&](i64 a, i64 b) { return (cnt[a + 1] - cnt[a]) > (cnt[b + 1] - cnt[b]); });
  // outputs go to scratch copies so that the writes can be logged serially afterwards
  std::vector<double> c1s(nk * 2), c2s(nk * 2), rps(nk * 2), cts(nk * 2); std::vector<unsigned char> pas(nk);
  for (i64 i = 0; i < nk; i++) { i64 n = nl[i]; for (int rf = 0; rf < 2; rf++) { c1s[i * 2 + rf] = 0.0; c2s[i * 2 + rf] = 0.0; rps[i * 2 + rf] = 0.0;
    cts[i * 2 + rf] = C.lumped[n * 2 + rf]; } pas[i] = 0; }
  double t1w = now();
  pfor((i64)work.size(), 4, [&](i64 qlo, i64 qhi) {
    for (i64 qi = qlo; qi < qhi; qi++) {
      i64 i = work[qi], n = nl[i];
      rc_one(C, n, i, seg, sidx.data() + cnt[i], cnt[i + 1] - cnt[i], stub_um, c1s.data(), c2s.data(), rps.data(), cts.data(), pas.data(), d.elm);
    }
  });
  for (i64 i = 0; i < nk; i++) {
    i64 n = nl[i];
    for (int rf = 0; rf < 2; rf++) { wD(&d.c1[n * 2 + rf], c1s[i * 2 + rf]); wD(&d.c2[n * 2 + rf], c2s[i * 2 + rf]);
      wD(&d.rpi[n * 2 + rf], rps[i * 2 + rf]); wD(&d.ctot[n * 2 + rf], cts[i * 2 + rf]); }
    wU(&d.par[n], pas[i]);
    pos[n] = -1;
  }
  double t2w = now();
  stats[0] = t1w - t0w; stats[1] = t2w - t1w; stats[2] = t2w - t0w;
  return (int)nk;
}

// ---------------------------------------------------------------- the update
// T5: phase seconds of all updates since load: [0] seed marks [1] sort/split [2] group compute [3] group commit+propagate
//     [4] edge compute [5] edge commit+propagate [6] net_rd [7] endpoint rows [8] calls
static double PF[16];
extern "C" void csta5_prof(double* out) { for (int i = 0; i < 16; i++) out[i] = PF[i]; }
// dirty levels of a gate group: 1 = only an input ARRIVAL changed (arc delays, slews and driver records are unchanged: the
// arrival is re-derived from the stored delays, no Newton solve), 2 = the input slew of some arcs changed (those arcs are
// re-solved, aflag), 3 = RC of the output net changed (all arcs re-solved). Net edges: 1 = arrival only, 2 = full.
static inline void markG(i64 gg, unsigned char lvl) {
  if (!gflag[gg]) { if (2 * I_glv[gg] <= cur_phase) late_marks++; Gd[I_glv[gg]].push_back(gg); }
  if (gflag[gg] < lvl) gflag[gg] = lvl;
}
static inline void markE(i64 i, i64 rf, unsigned char lvl) {
  i64 key = i * 2 + rf;
  if (!eflag[key]) { if (2 * I_elv[i] + 1 <= cur_phase) late_marks++; Ed[I_elv[i]].push_back(key); }
  if (eflag[key] < lvl) eflag[key] = lvl;
}
static inline void markR(i64 r) { if (!rflag[r]) { rflag[r] = 1; rlist.push_back(r); } }
static inline void propagate(i64 q, i64 rf, bool slew, bool drec) {   // (pin, rf) changed: slew / driver record / arrival
  if (slew) { for (i64 t = rda_o[q * 2 + rf]; t < rda_o[q * 2 + rf + 1]; t++) { i64 e = rda[t]; aflag[e] = 1; markG(I_agrp[e], 2); } }
  else for (i64 t = rdg_o[q * 2 + rf]; t < rdg_o[q * 2 + rf + 1]; t++) markG(rdg[t], 1);
  for (i64 t = drve_o[q]; t < drve_o[q + 1]; t++) markE(drve[t], rf, (slew || drec) ? 2 : 1);
  for (i64 t = prow_o[q]; t < prow_o[q + 1]; t++) markR(prow[t]);
}
// T5 (idea 3b): the commit of one level in parallel. Each recomputed node writes only its own outputs (unique writer per pin/rf,
// checked at bind), so the writes are race-free; the undo entries go to per-thread logs appended after the level (each address
// is logged at most once per update, so the restore order inside one update does not matter); the marks of the fan-out use
// atomic flags and per-thread lists that are merged and SORTED before use, as the serial lists are; counters are summed.
// Same final state, same counters; only the order of entries inside one update's undo log can differ.
struct TC5 {
  std::vector<LD> lD; std::vector<LI> lI; std::vector<LU> lU; std::vector<LP> lP;
  std::vector<std::vector<i64>> Gd, Ed; std::vector<i64> gl, el, rl, rdl; i64 c[16]; i64 late;
};
static TC5 TCS[64];
static thread_local int t5_tid = -1; static std::atomic<int> t5_ntid{0};
static inline TC5& tc5() { if (t5_tid < 0) t5_tid = t5_ntid.fetch_add(1); return TCS[t5_tid]; }
static inline bool wDt(TC5& t, double* p, double v) { if (beq(*p, v)) return false; t.lD.push_back({p, *p}); *p = v; return true; }
static inline bool wIt(TC5& t, i64* p, i64 v) { if (*p == v) return false; t.lI.push_back({p, *p}); *p = v; return true; }
static inline bool wUt(TC5& t, unsigned char* p, unsigned char v) { if (*p == v) return false; t.lU.push_back({p, *p}); *p = v; return true; }
static inline bool wPt(TC5& t, DP* p, const DP& v) { if (dp_eq(*p, v)) return false; t.lP.push_back({p, *p}); *p = v; return true; }
static inline bool cas_list(unsigned char* f, unsigned char lvl) {   // true when this call listed the node (flag was 0)
  unsigned char cur = __atomic_load_n(f, __ATOMIC_RELAXED);
  if (cur == 0 && __atomic_compare_exchange_n(f, &cur, lvl, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return true;
  while (cur < lvl && !__atomic_compare_exchange_n(f, &cur, lvl, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
  return false;
}
static inline void markG_t(TC5& t, i64 gg, unsigned char lvl) {
  if (cas_list(&gflag[gg], lvl)) { i64 lv = I_glv[gg]; if (2 * lv <= cur_phase) t.late++;
    if ((i64)t.Gd.size() <= lv) t.Gd.resize(lv + 1); if (t.Gd[lv].empty()) t.gl.push_back(lv); t.Gd[lv].push_back(gg); }
}
static inline void markE_t(TC5& t, i64 i, i64 rf, unsigned char lvl) {
  i64 key = i * 2 + rf;
  if (cas_list(&eflag[key], lvl)) { i64 lv = I_elv[i]; if (2 * lv + 1 <= cur_phase) t.late++;
    if ((i64)t.Ed.size() <= lv) t.Ed.resize(lv + 1); if (t.Ed[lv].empty()) t.el.push_back(lv); t.Ed[lv].push_back(key); }
}
static inline void propagate_t(TC5& t, i64 q, i64 rf, bool slew, bool drec) {
  if (slew) { for (i64 u = rda_o[q * 2 + rf]; u < rda_o[q * 2 + rf + 1]; u++) { i64 e = rda[u]; __atomic_store_n(&aflag[e], (unsigned char)1, __ATOMIC_RELAXED); markG_t(t, I_agrp[e], 2); } }
  else for (i64 u = rdg_o[q * 2 + rf]; u < rdg_o[q * 2 + rf + 1]; u++) markG_t(t, rdg[u], 1);
  for (i64 u = drve_o[q]; u < drve_o[q + 1]; u++) markE_t(t, drve[u], rf, (slew || drec) ? 2 : 1);
  for (i64 u = prow_o[q]; u < prow_o[q + 1]; u++) { i64 r = prow[u]; unsigned char z = 0;
    if (__atomic_compare_exchange_n(&rflag[r], &z, (unsigned char)1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) t.rl.push_back(r); }
}
static void tc5_reset() { for (int i = 0; i < 64; i++) { TC5& t = TCS[i]; for (int k = 0; k < 16; k++) t.c[k] = 0; t.late = 0; } }
static void tc5_merge(i64* cnt) {                                          // after a parallel commit (all workers joined)
  int nt = std::min(64, t5_ntid.load());
  for (int i = 0; i < nt; i++) {
    TC5& t = TCS[i];
    lD.insert(lD.end(), t.lD.begin(), t.lD.end()); lI.insert(lI.end(), t.lI.begin(), t.lI.end());
    lU.insert(lU.end(), t.lU.begin(), t.lU.end()); lP.insert(lP.end(), t.lP.begin(), t.lP.end());
    t.lD.clear(); t.lI.clear(); t.lU.clear(); t.lP.clear();
    for (i64 lv : t.gl) { Gd[lv].insert(Gd[lv].end(), t.Gd[lv].begin(), t.Gd[lv].end()); t.Gd[lv].clear(); }
    for (i64 lv : t.el) { Ed[lv].insert(Ed[lv].end(), t.Ed[lv].begin(), t.Ed[lv].end()); t.Ed[lv].clear(); }
    t.gl.clear(); t.el.clear();
    rlist.insert(rlist.end(), t.rl.begin(), t.rl.end()); t.rl.clear();
    rdlist.insert(rdlist.end(), t.rdl.begin(), t.rdl.end()); t.rdl.clear();
    for (int k = 0; k < 16; k++) { cnt[k] += t.c[k]; t.c[k] = 0; }
    late_marks += t.late; t.late = 0;
  }
}
static const i64 PAR_COMMIT_MIN = 256;                                     // smaller levels: the serial commit (original code)
static const bool SERIAL_COMMIT = getenv("CSTA5_SERIAL_COMMIT") != nullptr;   // idea switch (measurements)

// arrival / BEST of a group from the STORED arc delays (== group_compute when no input slew and no RC changed)
static void group_at_only(const DA& d, i64 gg, double* at_o, i64* best_o) {
  const i64 s0 = d.g_start[gg], s1 = I_gs1[gg], K = d.K;
  for (i64 k = 0; k < K; k++) {
    double amax = NEG; bool first = true;
    for (i64 e = s0; e < s1; e++) {
      double c = I_AT[(d.ce_from[e] * K + k) * 2 + d.ce_irf[e]] + I_gd[e];
      amax = first ? c : mx(amax, c); first = false;
    }
    at_o[k] = amax;
    i64 best = d.nE;
    for (i64 e = s0; e < s1; e++) {
      double c = I_AT[(d.ce_from[e] * K + k) * 2 + d.ce_irf[e]] + I_gd[e];
      if (c >= amax - 1e-12) { best = e; break; }
    }
    best_o[k] = best;
  }
}

// counters: [0] seed nets [1] groups recomputed in full [2] arcs recomputed [3] net edges (x rf) recomputed in full
//           [4] group outputs changed [5] sinks changed [6] rows recomputed [7] endpoint pins [8] nets net_rd recomputed
//           [9] late marks [10] levels touched [11] groups arrival-only [12] net edges arrival-only [13] group outputs whose
//           slew changed [14] sinks whose slew changed [15] picked arcs re-solved only to rebuild the driver record
// stats: [0] forward [1] endpoints [2] total
extern "C" i64 csta_inc_update(const i64* nets, i64 nnet, i64* cnt, double* stats) {
  for (int i = 0; i < 16; i++) cnt[i] = 0;
  if (!I_ready || !I_full) return -1;
  double tA = now();
  DA d = da();
  const i64 K = d.K;
  late_marks = 0; cur_phase = 0;
  nlist.clear();
  for (i64 q = 0; q < nnet; q++) {
    i64 n = nets[q]; if (n < 0 || n >= d.nn || nflag[n]) continue;
    nflag[n] = 1; nlist.push_back(n);
    for (i64 t = netg_o[n]; t < netg_o[n + 1]; t++) markG(netg[t], 3);
    for (i64 t = nete_o[n]; t < nete_o[n + 1]; t++) { markE(nete[t], 0, 2); markE(nete[t], 1, 2); }
  }
  for (i64 n : nlist) nflag[n] = 0;
  for (i64 n : nlist) {                                                    // T5: the state of these nets now follows their RC
    for (int rf = 0; rf < 2; rf++) { i64 q = n * 2 + rf; wSH(&SH_c1[q], d.c1[q]); wSH(&SH_c2[q], d.c2[q]); wSH(&SH_rpi[q], d.rpi[q]); wSH(&SH_ctot[q], d.ctot[q]); }
    wSHu(&SH_par[n], d.par[n]);
    for (i64 t = pn_o[n]; t < pn_o[n + 1]; t++) { i64 p = pn[t]; wSH(&SH_elm[p * 2], d.elm[p * 2]); wSH(&SH_elm[p * 2 + 1], d.elm[p * 2 + 1]); }
  }
  cnt[0] = (i64)nlist.size();
  static std::vector<Dmp> res; static std::vector<i64> roff; static std::vector<GOut> go; static std::vector<double> at; static std::vector<i64> bs;
  static std::vector<double> ewd, els; static std::vector<i64> GF, GA, EF, EA;
  rdlist.clear();
  double tq = now(); PF[0] += tq - tA;                                     // T5: phase timers (csta5_prof)
  for (i64 lv = 1; lv < d.nlev; lv++) {
    std::vector<i64>& GL = Gd[lv];
    if (!GL.empty()) {
      cnt[10]++;
      cur_phase = 2 * lv;
      std::sort(GL.begin(), GL.end());
      GF.clear(); GA.clear();
      for (i64 gg : GL) (gflag[gg] >= 2 ? GF : GA).push_back(gg);
      const i64 m = (i64)GF.size(), ma = (i64)GA.size();
      roff.resize(m + 1); roff[0] = 0;
      for (i64 j = 0; j < m; j++) roff[j + 1] = roff[j] + (I_gs1[GF[j]] - d.g_start[GF[j]]);
      res.resize(roff[m]); go.resize(m); at.resize((m + ma) * K); bs.resize((m + ma) * K);
      { double t_ = now(); PF[1] += t_ - tq; tq = t_; }
      pfor(m, ach(m, 16), [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) group_compute(d, GF[j], gflag[GF[j]] == 3, res.data() + roff[j], at.data() + j * K, bs.data() + j * K, go[j]); });
      pfor(ma, ach(ma, 256), [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) group_at_only(d, GA[j], at.data() + (m + j) * K, bs.data() + (m + j) * K); });
      { double t_ = now(); PF[2] += t_ - tq; tq = t_; }
      cnt[1] += m; cnt[11] += ma;
      if (m + ma >= PAR_COMMIT_MIN && POOL.nthr > 1 && POOL.nthr < 64 && !SERIAL_COMMIT) {   // T5: parallel commit
        pfor(m + ma, std::max<i64>(32, ach(m + ma, 512)), [&](i64 lo_, i64 hi_) {
          TC5& t = tc5();
          for (i64 j = lo_; j < hi_; j++) {
            if (j < m) {
              i64 gg = GF[j], s0 = d.g_start[gg], s1 = I_gs1[gg], gto = d.g_to[gg], gorf = d.g_orf[gg];
              for (i64 e = s0; e < s1; e++) {
                const Dmp& r = res[roff[j] + e - s0];
                if (r.alg != STORED) { t.c[2]++; wDt(t, &I_gd[e], r.delay); wDt(t, &I_as[e], r.slew); }
                aflag[e] = 0;
              }
              t.c[15] += go[j].pick_recomputed;
              bool chs = wDt(t, &I_SL[gto * 2 + gorf], go[j].sl);
              bool chd = wPt(t, &I_dps[gto * 2 + gorf], go[j].dp);
              bool cha = false;
              for (i64 k = 0; k < K; k++) { cha |= wDt(t, &I_AT[(gto * K + k) * 2 + gorf], at[j * K + k]); wIt(t, &I_BEST[(gto * K + k) * 2 + gorf], bs[j * K + k]); }
              bool rch = wUt(t, &I_gpk[gg], go[j].pk); rch |= wDt(t, &I_grd[gg], go[j].rd); rch |= wIt(t, &I_gpke[gg], go[j].pke);
              if (rch) { i64 n = d.pin_net[gto]; unsigned char z = 0;
                if (__atomic_compare_exchange_n(&rdflag[n], &z, (unsigned char)1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) t.rdl.push_back(n); }
              if (chs || chd || cha) { t.c[4]++; t.c[13] += chs; propagate_t(t, gto, gorf, chs, chd); }
              gflag[gg] = 0;
            } else {
              const i64 ja = j - m; i64 gg = GA[ja], gto = d.g_to[gg], gorf = d.g_orf[gg];
              bool cha = false;
              for (i64 k = 0; k < K; k++) { cha |= wDt(t, &I_AT[(gto * K + k) * 2 + gorf], at[(m + ja) * K + k]); wIt(t, &I_BEST[(gto * K + k) * 2 + gorf], bs[(m + ja) * K + k]); }
              if (cha) { t.c[4]++; propagate_t(t, gto, gorf, false, false); }
              gflag[gg] = 0;
            }
          } });
        tc5_merge(cnt);
      } else {
      for (i64 j = 0; j < m; j++) {
        i64 gg = GF[j], s0 = d.g_start[gg], s1 = I_gs1[gg], gto = d.g_to[gg], gorf = d.g_orf[gg];
        for (i64 e = s0; e < s1; e++) {
          const Dmp& r = res[roff[j] + e - s0];
          if (r.alg != STORED) { cnt[2]++; wD(&I_gd[e], r.delay); wD(&I_as[e], r.slew); }
          aflag[e] = 0;
        }
        cnt[15] += go[j].pick_recomputed;
        bool chs = wD(&I_SL[gto * 2 + gorf], go[j].sl);
        bool chd = wP(&I_dps[gto * 2 + gorf], go[j].dp);
        bool cha = false;
        for (i64 k = 0; k < K; k++) { cha |= wD(&I_AT[(gto * K + k) * 2 + gorf], at[j * K + k]); wI(&I_BEST[(gto * K + k) * 2 + gorf], bs[j * K + k]); }
        bool rch = wU(&I_gpk[gg], go[j].pk); rch |= wD(&I_grd[gg], go[j].rd); rch |= wI(&I_gpke[gg], go[j].pke);
        if (rch) { i64 n = d.pin_net[gto]; if (!rdflag[n]) { rdflag[n] = 1; rdlist.push_back(n); } }
        if (chs || chd || cha) { cnt[4]++; cnt[13] += chs; propagate(gto, gorf, chs, chd); }
        gflag[gg] = 0;
      }
      for (i64 j = 0; j < ma; j++) {
        i64 gg = GA[j], gto = d.g_to[gg], gorf = d.g_orf[gg];
        bool cha = false;
        for (i64 k = 0; k < K; k++) { cha |= wD(&I_AT[(gto * K + k) * 2 + gorf], at[(m + j) * K + k]); wI(&I_BEST[(gto * K + k) * 2 + gorf], bs[(m + j) * K + k]); }
        if (cha) { cnt[4]++; propagate(gto, gorf, false, false); }
        gflag[gg] = 0;
      }
      }
      GL.clear();
      { double t_ = now(); PF[3] += t_ - tq; tq = t_; }
    }
    std::vector<i64>& EL = Ed[lv];
    if (!EL.empty()) {
      cur_phase = 2 * lv + 1;
      std::sort(EL.begin(), EL.end());
      EF.clear(); EA.clear();
      for (i64 key : EL) (eflag[key] == 2 ? EF : EA).push_back(key);
      const i64 m = (i64)EF.size(), ma = (i64)EA.size();
      ewd.resize(m); els.resize(m); at.resize((m + ma) * K);
      { double t_ = now(); PF[1] += t_ - tq; tq = t_; }
      pfor(m, ach(m, 64), [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) edge_compute(d, EF[j] / 2, EF[j] % 2, &ewd[j], &els[j], at.data() + j * K); });
      pfor(ma, ach(ma, 1024), [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) {
        i64 i = EA[j] / 2, rf = EA[j] % 2, drv = d.ne_drv[i], snk = d.ne_snk[i]; double wd = I_wd[snk * 2 + rf];
        for (i64 k = 0; k < K; k++) at[(m + j) * K + k] = I_AT[(drv * K + k) * 2 + rf] + wd; } });
      { double t_ = now(); PF[4] += t_ - tq; tq = t_; }
      cnt[3] += m; cnt[12] += ma;
      if (m + ma >= PAR_COMMIT_MIN && POOL.nthr > 1 && POOL.nthr < 64 && !SERIAL_COMMIT) {   // T5: parallel commit
        pfor(m + ma, std::max<i64>(32, ach(m + ma, 512)), [&](i64 lo_, i64 hi_) {
          TC5& t = tc5();
          for (i64 j = lo_; j < hi_; j++) {
            if (j < m) {
              i64 i = EF[j] / 2, rf = EF[j] % 2, snk = d.ne_snk[i];
              wDt(t, &I_wd[snk * 2 + rf], ewd[j]);
              bool chs = wDt(t, &I_SL[snk * 2 + rf], els[j]), cha = false;
              for (i64 k = 0; k < K; k++) cha |= wDt(t, &I_AT[(snk * K + k) * 2 + rf], at[j * K + k]);
              if (chs || cha) { t.c[5]++; t.c[14] += chs; propagate_t(t, snk, rf, chs, false); }
              eflag[EF[j]] = 0;
            } else {
              const i64 ja = j - m; i64 i = EA[ja] / 2, rf = EA[ja] % 2, snk = d.ne_snk[i];
              bool cha = false;
              for (i64 k = 0; k < K; k++) cha |= wDt(t, &I_AT[(snk * K + k) * 2 + rf], at[(m + ja) * K + k]);
              if (cha) { t.c[5]++; propagate_t(t, snk, rf, false, false); }
              eflag[EA[ja]] = 0;
            }
          } });
        tc5_merge(cnt);
      } else {
      for (i64 j = 0; j < m; j++) {
        i64 i = EF[j] / 2, rf = EF[j] % 2, snk = d.ne_snk[i];
        wD(&I_wd[snk * 2 + rf], ewd[j]);
        bool chs = wD(&I_SL[snk * 2 + rf], els[j]), cha = false;
        for (i64 k = 0; k < K; k++) cha |= wD(&I_AT[(snk * K + k) * 2 + rf], at[j * K + k]);
        if (chs || cha) { cnt[5]++; cnt[14] += chs; propagate(snk, rf, chs, false); }
        eflag[EF[j]] = 0;
      }
      for (i64 j = 0; j < ma; j++) {
        i64 i = EA[j] / 2, rf = EA[j] % 2, snk = d.ne_snk[i];
        bool cha = false;
        for (i64 k = 0; k < K; k++) cha |= wD(&I_AT[(snk * K + k) * 2 + rf], at[(m + j) * K + k]);
        if (cha) { cnt[5]++; propagate(snk, rf, false, false); }
        eflag[EA[j]] = 0;
      }
      }
      EL.clear();
      { double t_ = now(); PF[5] += t_ - tq; tq = t_; }
    }
  }
  // net_rd of the nets whose groups changed their pick (same order as the full pass: groups in index order)
  for (i64 n : rdlist) {
    double v = 0.0;
    for (i64 t = netg_o[n]; t < netg_o[n + 1]; t++) { i64 gg = netg[t]; if (I_gpk[gg]) v = mx(v, I_grd[gg]); }
    wD(&I_nrd[n], v); rdflag[n] = 0;
  }
  cnt[8] = (i64)rdlist.size();
  double tB = now();
  // endpoint rows
  std::sort(rlist.begin(), rlist.end());
  const i64 nr = (i64)rlist.size();
  static std::vector<double> rq, rm; static std::vector<unsigned char> rv;
  rq.resize(nr * K * 2); rv.resize(nr * K * 2); rm.resize(nr);
  pfor(nr, ach(nr, 256), [&](i64 lo_, i64 hi_) { for (i64 j = lo_; j < hi_; j++) rm[j] = row_compute(d, rlist[j], rq.data() + j * K * 2, rv.data() + j * K * 2); });
  plist.clear();
  for (i64 j = 0; j < nr; j++) {
    i64 r = rlist[j];
    for (i64 q = 0; q < K * 2; q++) {
      double* pq = &S_reqk[r * K * 2 + q]; if (!beq(*pq, rq[j * K * 2 + q])) { lDq.push_back({pq, *pq}); *pq = rq[j * K * 2 + q]; }
      unsigned char* pv = &S_valid[r * K * 2 + q]; if (*pv != rv[j * K * 2 + q]) { lUq.push_back({pv, *pv}); *pv = rv[j * K * 2 + q]; }
    }
    wD(&I_rmin[r], rm[j]);
    i64 p = d.ck_pin[r]; if (!pflag[p]) { pflag[p] = 1; plist.push_back(p); }
    rflag[r] = 0;
  }
  rlist.clear();
  for (i64 p : plist) {
    double v = BIG;
    for (i64 t = prow_o[p]; t < prow_o[p + 1]; t++) v = mn(v, I_rmin[prow[t]]);
    wD(&I_ep[p], v); pflag[p] = 0;
  }
  cnt[6] = nr; cnt[7] = (i64)plist.size(); cnt[9] = late_marks;
  double tC = now();
  PF[6] += tB - tq; PF[7] += tC - tB; PF[8] += 1;
  stats[0] = tB - tA; stats[1] = tC - tB; stats[2] = tC - tA;
  return late_marks ? -2 : 0;
}

// ---------------------------------------------------------------- access
extern "C" void* csta_inc_ptr(const char* name) {
  std::string s(name);
  if (s == "AT") return I_AT.data(); if (s == "SL") return I_SL.data(); if (s == "gate_d") return I_gd.data();
  if (s == "wire_d") return I_wd.data(); if (s == "arc_slew") return I_as.data(); if (s == "BEST") return I_BEST.data(); if (s == "net_rd") return I_nrd.data();
  if (s == "rmin") return I_rmin.data(); if (s == "ep_sl") return I_ep.data(); if (s == "grp_rd") return I_grd.data();
  if (s == "grp_pk") return I_gpk.data(); if (s == "grp_pke") return I_gpke.data(); if (s == "RAT") return I_RAT.data();
  if (s == "pin_slack") return I_psl.data(); if (s == "net_slack") return I_nsl.data();
  if (s == "reqk") return S_reqk.data(); if (s == "valid") return S_valid.data();
  return nullptr;
}
extern "C" i64 csta_inc_ngroups() { return I_g1; }
// driver-model records -> out[P*2][16] doubles + ai[P*2][2] (alg, valid)
extern "C" void csta_inc_dps(double* out, i64* ai) {
  for (size_t i = 0; i < I_dps.size(); i++) {
    const DP& q = I_dps[i];
    double v[16] = {q.slew, q.t0, q.dt, q.k0, q.k1, q.k2, q.k3, q.k4, q.p1, q.p2, q.vub, q.vod, q.vth, q.vl, q.vh, q.der};
    memcpy(out + i * 16, v, sizeof v); ai[i * 2] = q.alg; ai[i * 2 + 1] = q.valid;
  }
}
extern "C" i64 csta_inc_logsize() { return (i64)(lD.size() + lDq.size() + lI.size() + lU.size() + lUq.size() + lP.size()); }
// ======================================================================== tools/fast3x/h_csta_kern.cc (after its include)
// h_csta_kern.cc: the incremental exact timer of tools/engine/csta_inc_kern.cc (which
// includes tools/gapla/csta_kern.cc + csta_rc.inc verbatim, so csta_analyze / csta_build_rc / csta_crit are the same
// source) plus ONE new entry point for the round weights:
//   csta_hcrit(...)  == csta_crit restricted to the outputs the chain's weights() reads (path offsets, net per entry, ok
//                       flag per entry). The pslack/cap/dl roundings (snprintf + strtod per entry, single thread) are not
//                       computed; the ok flag uses the UNROUNDED cap/dl/t exactly as csta_crit does. The per-path entry
//                       loop runs in the thread pool (per-path vectors, concatenated in path order).
// Build: tools/fast3x/h_build.sh (clang++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -mcpu=native, as csta_build.sh).

static std::vector<i64> H_poff, H_net; static std::vector<unsigned char> H_ok;

extern "C" i64 csta_hcrit(i64 npaths, double slack_max, const double* AT, const i64* BEST, const double* rmin,
                          const double* net_ctot, const double* net_rd_kohm, double* stats) {
  double ta = now();
  const i64 K = gi("K"), nE = gi("nE"), nn = gi("nn"), nrow = gi("nrow");
  const i64 *ck_pin = g<i64>("ck_pin"), *pin_net = g<i64>("pin_net"), *net_drv = g<i64>("net_drv"), *ce_from = g<i64>("ce_from"),
            *ce_irf = g<i64>("ce_irf");
  const unsigned char *pin_is_load = g<unsigned char>("pin_is_load"), *is_port = g<unsigned char>("is_port"),
                      *pin_isdrvdir = g<unsigned char>("pin_isdrvdir"), *pin_is_drv = g<unsigned char>("pin_is_drv");
  // ---- verbatim from csta_crit: endpoint rows, one per clock pin, ordered by slack
  std::vector<i64> rows;
  for (i64 r = 0; r < nrow; r++) if (rmin[r] < slack_max) rows.push_back(r);
  std::stable_sort(rows.begin(), rows.end(), [&](i64 a, i64 b) { if (ck_pin[a] != ck_pin[b]) return ck_pin[a] < ck_pin[b]; return rmin[a] < rmin[b]; });
  std::vector<i64> u;
  for (size_t t = 0; t < rows.size(); t++) if (t == 0 || ck_pin[rows[t]] != ck_pin[rows[t - 1]]) u.push_back(rows[t]);
  std::stable_sort(u.begin(), u.end(), [&](i64 a, i64 b) { return rmin[a] < rmin[b]; });
  if ((i64)u.size() > npaths) u.resize(npaths);
  std::vector<std::vector<i64>> pn(u.size());
  std::vector<std::vector<unsigned char>> pk(u.size());
  pfor((i64)u.size(), 16, [&](i64 lo_, i64 hi_) {
    std::vector<i64> path;
    for (i64 qi = lo_; qi < hi_; qi++) {
      // ---- verbatim path trace from csta_crit
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
      // ---- the entry loop of csta_crit for this path (net and ok only; same arithmetic for ok)
      std::vector<i64>& on = pn[qi]; std::vector<unsigned char>& ok = pk[qi];
      double t = 0.0;
      for (i64 q : path) {
        i64 n = pin_net[q];
        if (!pin_is_drv[q] || n < 0) continue;
        double cap = mx(net_ctot[n], 1e-3);
        double dl = net_rd_kohm[n] * cap / 1000.0;
        t += dl;
        on.push_back(n);
        ok.push_back(std::isfinite(cap) && std::isfinite(dl) && std::isfinite(t) && cap >= 0 && dl >= 0);
      }
    }
  });
  H_poff.assign(u.size() + 1, 0);
  for (size_t qi = 0; qi < u.size(); qi++) H_poff[qi + 1] = H_poff[qi] + (i64)pn[qi].size();
  H_net.resize(H_poff.back()); H_ok.resize(H_poff.back());
  for (size_t qi = 0; qi < u.size(); qi++) {
    if (!pn[qi].empty()) {
      memcpy(H_net.data() + H_poff[qi], pn[qi].data(), pn[qi].size() * 8);
      memcpy(H_ok.data() + H_poff[qi], pk[qi].data(), pk[qi].size());
    }
  }
  stats[0] = now() - ta;
  return (i64)u.size();
}
extern "C" i64 csta_hcrit_n() { return (i64)H_net.size(); }
extern "C" void csta_hcrit_get(i64* poff, i64* net, unsigned char* ok) {
  memcpy(poff, H_poff.data(), H_poff.size() * 8); memcpy(net, H_net.data(), H_net.size() * 8); memcpy(ok, H_ok.data(), H_ok.size());
}
// ======================================================================== tools/best/i_kern.cc (after its include)
// i_kern.cc: tools/fast3x/h_csta_kern.cc (incremental exact timer + csta_hcrit) included
// verbatim + the w3_inc_backward() entry point of tools/wave3/ovf2/w3_inc_kern.cc (its lines after the include, copied).
// Build: tools/best/build_i.sh (-O3 -ffp-contract=off -fno-fast-math, as h_build.sh / w3_inc_kern.cc).

extern "C" void w3_inc_backward(double* stats) {
  double tC = now();
  DA d = da();
  if (bw_par) { bwd_par(d); stats[0] = now() - tC; return; }              // T5: parallel, same mn() sequence per entry
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
