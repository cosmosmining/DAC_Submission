// csta_kern_b.cc: copy of tools/gapla/csta_kern.cc with the SAME arithmetic, faster on large
// designs between rounds that change few nets:
//   M1. exact memo of the per-arc gate delay (dmp_gate1): an arc is recomputed only when one of its inputs (input slew, the four RC
//       numbers of its net, the pi/lumped flag) differs bit for bit from the previous analyze; the per-pin driver model (DP) and the
//       per-sink load delay (dmp_load1) likewise.  dmp_gate1/dmp_load1 are pure functions of those inputs (and the fixed tables), so
//       every output is byte-identical (tools/srvw/bigcase/bench_sta.py --lib2 checks every array).  CSTA_MEMO=0 turns it off.
//   M2. the O(P*K) initialisations and the slack sweep run on the thread pool; the backward required-time pass runs per level
//       grouped by the written pin, each pin taking its contributions in the original index order (same mn() sequence).
//   M3. (csta_rc_b.inc) the per-path rounding loop of csta_crit runs per path on the pool, concatenated in path order.
// stats[4..7] of analyze = setup s, arc-delay s, arcs recomputed, loads recomputed.
// Build (host): g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -shared -fPIC -pthread csta_kern_b.cc -o libcsta_b.so
// Original header follows.
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
// ---------------------------------------------------------------- tiny persistent thread pool (no OpenMP: torch's libomp would clash)
struct Pool {
  std::vector<std::thread> th; std::mutex m; std::condition_variable cv, cvd;
  const std::function<void(i64, i64)>* job = nullptr; std::atomic<i64> next{0}; i64 n = 0, chunk = 1; int gen = 0, active = 0; bool stop = false;
  int nthr = 0;
  void init() {
    if (nthr) return;
    const char* e = getenv("CSTA_THREADS"); if (!e) e = getenv("OMP_NUM_THREADS");
    nthr = e ? atoi(e) : 8; if (nthr < 1) nthr = 1;
    for (int i = 1; i < nthr; i++) { th.emplace_back([this]() { loop(); }); th.back().detach(); }
  }
  void work() { for (;;) { i64 s = next.fetch_add(chunk); if (s >= n) break; (*job)(s, std::min(n, s + chunk)); } }
  void loop() {
    int seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, [&] { return stop || gen != seen; });
      if (stop) return;
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
static const double NEG = -1e30, BIG = 1e30;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------- registry
static std::map<std::string, const void*> REG;
static std::map<std::string, i64> SCL;
static i64 EPOCH = 0;  // bumped when a design array (not an rc_* input) is (re)bound: the memo is then invalid
extern "C" void csta_set(const char* name, const void* p) { if (strncmp(name, "rc_", 3) != 0) EPOCH++; REG[name] = p; }
extern "C" void csta_seti(const char* name, i64 v) { EPOCH++; SCL[name] = v; }
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
static inline double ex(double x) { return std::exp(mx(x, -700.0)); }
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
static inline double yramp(double t, double t0, double dt, double rc) {
  double t1 = t - t0;
  double a = y0f(mx(t1, 0.0), rc), b = y0f(mx(t1 - dt, 0.0), rc);
  return t1 <= 0 ? 0.0 : (t1 <= dt ? a / dt : (a - b) / dt);
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
  double t1 = t - q.t0, va, da, vb, db;
  V0(mx(t1, 0.0), &va, &da);
  V0(mx(t1 - q.dt, 0.0), &vb, &db);
  *v = t1 <= 0 ? 0.0 : (t1 <= q.dt ? va / q.dt : (va - vb) / q.dt);
  *d = t1 <= 0 ? 0.0 : (t1 <= q.dt ? da / q.dt : (da - db) / q.dt);
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
  double t1 = t - q.t0, va, da, vb, db;
  V0(mx(t1, 0.0), &va, &da);
  V0(mx(t1 - q.dt, 0.0), &vb, &db);
  *v = t1 <= 0 ? 0.0 : (t1 <= q.dt ? va / q.dt : (va - vb) / q.dt);
  *d = t1 <= 0 ? 0.0 : (t1 <= q.dt ? da / q.dt : (da - db) / q.dt);
}
// safeguarded Newton (rtsafe), scalar copy of _root
template <class P, void (*F)(const P&, double, double*, double*)>
static bool rootf(const P& q, double target, double lo, double hi, double x0, double* xo) {
  double v, d;
  F(q, lo, &v, &d); double flo = v - target;
  F(q, hi, &v, &d); double fhi = v - target;
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

static void dmp_pi(const Arc& A, Dmp& o) {
  double c1 = A.c1, c2 = A.c2, rpi = A.rpi, rd = A.rd, vth = A.vth, vl = A.vl, vh = A.vh, der = A.der, s = A.sin;
  i64 td = A.td, ts = A.ts;
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
      double t0 = X[0], dt = X[1], ce = X[2];
      bool bad = (ce < 0) || (ce > c1 + c2) || (dt <= 0);
      double cs = mx(ce, 1e-12);
      double dts = dt > 0 ? dt : 1e-12;
      double tv, tl, sl;
      gate_delays(td, ts, s, cs, vth, vl, vh, der, &tv, &tl, &sl);
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
                 rce * (dts + dts * erc - 2 * rce * (1 - erc))) / (rd * std::pow(dts, 3.0));
      J[2][2] = (2 * rce - dts - (2 * rce + dts) * ex(-dts / rce)) / (dts * dts);
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
    gate_delays(td, ts, s, ce0, vth, vl, vh, der, &tv, &tl, &sl);
    double dt = sl / (vh - vl);
    X[0] = tv + std::log(1.0 - vth) * rd * ce0 - vth * dt; X[1] = dt; X[2] = ce0;
  };
  double X[3];
  start(c1 + c2, X);
  bool ok = newton(X);
  if (!ok) { start(c2, X); ok = newton(X); }
  double ceff = ok ? X[2] : c1 + c2;
  double t0 = X[0], dt = X[1];
  double delay = lut(td, s, ceff);
  double tslew = ts >= 0 ? mx(lut(ts, s, ceff), 0.0) : 0.0;
  double vub = t0 + dt + (c1 + c2) * (rd + rpi) * 2.0;
  bool good = ok && std::isfinite(k0 + k1 + k2 + k3 + k4 + p1 + p2) && (dt > 0);
  bool valid = false; double vod = 0.0, slew = tslew;
  if (good) {
    VoP q{t0, dt, k0, k1, k2, k3, k4, p1, p2};
    double span = tslew * der / (vh - vl);
    double d50, tl_, th_;
    bool o1 = rootf<VoP, vo_f>(q, vth, t0, vub, delay, &d50);
    bool o2 = rootf<VoP, vo_f>(q, vl, t0, d50, d50 - span * (vth - vl), &tl_);
    bool o3 = rootf<VoP, vo_f>(q, vh, d50, vub, d50 + span * (vh - vth), &th_);
    valid = o1 && o2 && o3;
    vod = d50;
    slew = valid ? (th_ - tl_) / der : tslew;
    o.dbg1 = tl_; o.dbg2 = th_;
  }
  o.alg = ok ? 2 : 0; o.ceff = ceff; o.delay = delay; o.slew = slew;
  o.t0 = t0; o.dt = dt; o.vod = vod; o.valid = valid && ok;
  o.k0 = nn0(k0); o.k1 = nn0(k1); o.k2 = nn0(k2); o.k3 = nn0(k3); o.k4 = nn0(k4); o.p1 = nn0(p1); o.p2 = nn0(p2);
  o.vub = nn0(vub);
}

static void dmp_zc2(const Arc& A, Dmp& o) {
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
    bool o1 = rootf<VoP, vo_f>(q, vth, t0, vub, tv0, &d50);
    bool o2 = rootf<VoP, vo_f>(q, vl, t0, d50, d50 - span * (vth - vl), &tl_);
    bool o3 = rootf<VoP, vo_f>(q, vh, d50, vub, d50 + span * (vh - vth), &th_);
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
  o.slew = A.ts >= 0 ? mx(lut(A.ts, A.sin, cpi), 0.0) : 0.0;
  o.t0 = o.dt = o.vod = o.k0 = o.k1 = o.k2 = o.k3 = o.k4 = o.p1 = o.vub = 0.0; o.p2 = 1.0;
  if (pim) dmp_pi(A, o);
  else if (zc2) dmp_zc2(A, o);
}

// ---------------------------------------------------------------- load delay / slew (dmp_load)
struct DP { double slew, t0, dt, k0, k1, k2, k3, k4, p1, p2, vub, vod, vth, vl, vh, der; int alg; unsigned char valid; };
static inline void dmp_load1(const DP& d, double elm, double* wd, double* ls) {
  *wd = std::isfinite(elm) ? elm : 0.0;
  *ls = d.slew;
  bool use = (d.alg >= 1) && d.valid && std::isfinite(elm) && (elm > 0) && !(elm < d.slew * 1e-3);
  if (!use) return;
  double p3 = 1.0 / elm;
  VlP q; vl_init(q, d.t0, d.dt, d.k0, d.k1, d.k2, d.k3, d.k4, d.p1, d.p2, p3);
  double lo = d.t0, hi = d.vub + 2.0 * elm;
  double sp = (d.slew * d.der + 2.2 * elm) / (d.vh - d.vl);
  double ld, tl, th;
  bool o1 = rootf<VlP, vl_f>(q, d.vth, lo, hi, d.vod + 0.7 * elm, &ld);
  bool o2 = rootf<VlP, vl_f>(q, d.vl, lo, ld, ld - sp * (d.vth - d.vl), &tl);
  bool o3 = rootf<VlP, vl_f>(q, d.vh, ld, hi, ld + sp * (d.vh - d.vth), &th);
  bool ok = o1 && o2 && o3;
  double d1 = ld - d.vod, s1 = (th - tl) / d.der;
  d1 = d1 < 0 ? elm : d1;
  s1 = s1 < d.slew ? d.slew : s1;
  *wd = ok ? d1 : elm;
  *ls = ok ? s1 : d.slew;
}

// ---------------------------------------------------------------- analyze
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#include "analyze_b.inc"  // M1 + M2 (replaces the original csta_analyze)

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

#include "csta_rc_b.inc"
