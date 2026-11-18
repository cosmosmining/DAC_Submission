// dg_kern.cu: the whole objective of tools/e2et/descend.py (expected overflow with vias, per-net power, smooth WNS + TNS through tools/gputimer's timer) and its gradient and
// Adam in kernels, no torch: logits -> q -> objective -> adjoint -> logits, K steps in one C++ call. Same formulas as tools/objfix/objective2.py (mgf overflow, via_terms) and tools/gputimer/gputimer.py
// (_rc_core, analyze). Same source, two builds (the timer's own rules: no FMA contraction):
//   GPU: nvcc -O3 -std=c++17 -arch=sm_75 -fmad=false -Xcompiler -fPIC -shared dg_kern.cu -o libdg_cuda.so      (H100: -arch=sm_90)
//   CPU: g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -fPIC -shared -pthread -x c++ dg_kern.cu -o libdg_cpu.so
//   -DDG_F64: overflow + power in double (the reference check against Objective2's float64 values).
#include "../weights/gt_kern_w.cu"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#ifdef DG_F64
typedef double RL;
#define DG_EPS 1e-12
#define DG_CLIP 600.0
#else
typedef float RL;
#define DG_EPS 1e-6f
#define DG_CLIP 60.0f
#endif

// ---------------------------------------------------------------- device memory
extern "C" void* dg_alloc(i64 n) {
#ifdef __CUDACC__
  void* p = nullptr; cudaError_t e = cudaErrorMemoryAllocation;
#ifndef DG_UM_FALLBACK
  e = cudaMalloc(&p, (size_t)(n > 0 ? n : 8));
#else
  // 11 Nov 2026 (build flag only): when an array would leave less than DG_UM_HEADROOM_GB (default 6) of device memory free, or does
  // not fit, it goes to CUDA managed memory, which the driver pages between the GPU and the host (MemPool-Cluster: the timer tables
  // exceed the 94 GB of one H100). The headroom keeps room on the GPU for the pages of the managed arrays. Default builds: unchanged.
  size_t fr_ = 0, tot_ = 0; cudaMemGetInfo(&fr_, &tot_);
  const char* hs_ = getenv("DG_UM_HEADROOM_GB"); double hr_ = (hs_ ? atof(hs_) : 6.0) * 1073741824.0;
  if ((double)fr_ >= (double)(n > 0 ? n : 8) + hr_) e = cudaMalloc(&p, (size_t)(n > 0 ? n : 8));
  if (e != cudaSuccess) {
    cudaGetLastError();
    e = cudaMallocManaged(&p, (size_t)(n > 0 ? n : 8));
    if (e == cudaSuccess) { fprintf(stderr, "dg_alloc %lld: device memory full, managed memory\n", (long long)n); return p; }
  }
#endif
  if (e != cudaSuccess) { fprintf(stderr, "dg_alloc %lld: %s\n", (long long)n, cudaGetErrorString(e)); return nullptr; }
  return p;
#else
  return malloc((size_t)(n > 0 ? n : 8));
#endif
}
extern "C" int dg_rec_bytes() { return (int)sizeof(RECT); }   // bytes per value of the timer-tape record (8; 4 with -DDG_REC32)
extern "C" void dg_free(void* p) {
#ifdef __CUDACC__
  cudaFree(p);
#else
  free(p);
#endif
}
extern "C" int dg_put(void* dev, const void* host, i64 n) {
  if (n <= 0) return 0;
#ifdef __CUDACC__
  cudaError_t e = cudaMemcpy(dev, host, (size_t)n, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) { fprintf(stderr, "dg_put: %s\n", cudaGetErrorString(e)); return 1; }
#else
  memcpy(dev, host, (size_t)n);
#endif
  return 0;
}
extern "C" int dg_get(void* host, const void* dev, i64 n) {
  if (n <= 0) return 0;
#ifdef __CUDACC__
  cudaError_t e = cudaMemcpy(host, dev, (size_t)n, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) { fprintf(stderr, "dg_get: %s\n", cudaGetErrorString(e)); return 1; }
#else
  memcpy(host, dev, (size_t)n);
#endif
  return 0;
}
extern "C" int dg_zero(void* dev, i64 n) {
  if (n <= 0) return 0;
#ifdef __CUDACC__
  cudaError_t e = cudaMemset(dev, 0, (size_t)n);
  if (e != cudaSuccess) { fprintf(stderr, "dg_zero: %s\n", cudaGetErrorString(e)); return 1; }
#else
  memset(dev, 0, (size_t)n);
#endif
  return 0;
}
// starts the device on the calling thread (context, first allocation); the kernels' code loads with the first launch
extern "C" int dg_init() {
#ifdef __CUDACC__
  int n = 0; cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n < 1) { fprintf(stderr, "dg_init: %s\n", cudaGetErrorString(e)); return 1; }
  void* p = nullptr; e = cudaMalloc(&p, 64); if (e != cudaSuccess) return 1; cudaFree(p);
#endif
  return 0;
}

// ---------------------------------------------------------------- table (python dg.py PTRS: the same order)
#define DG_PTRS(X) \
  X(const int*, Cand) X(const double*, Lr) X(const double*, Lc) X(const RL*, Cc) X(const int*, z2k) X(const u8*, rdir) X(const RL*, pwk) X(const int*, rp0) X(const int*, rp1) \
  X(RL*, x) X(RL*, p) X(RL*, am) X(RL*, av) X(RL*, l0) X(RL*, l1) X(double*, rr) X(double*, cr) X(RL*, pwv) \
  X(const int*, ent_ptr) X(const int*, ec) X(const int*, cp) X(const int*, ce) X(const int*, cvp) X(const int*, cvi) X(const RL*, cvg) X(const RL*, base) X(const RL*, u) X(const u8*, copen) X(RL*, uc) X(RL*, uv) \
  X(const int*, prp) X(const int*, prr) X(const int*, pt_lo) X(const int*, pt_hi) X(const int*, pvp) X(const u8*, pvz) X(const int*, pvc) X(const RL*, pvg) X(RL*, eF) X(RL*, eG) X(RL*, eN) X(RL*, dv) X(RL*, gd) \
  X(const double*, Rp0) X(const int*, rp_run) X(const double*, rp_gl) X(const double*, cap0) X(const u8*, reach) X(const int*, nsp) X(const int*, nsr) X(const double*, nsg) X(double*, Rp) X(double*, y1) X(double*, y2) X(double*, y3) X(double*, elmn) \
  X(double*, gR) X(double*, gy1) X(double*, gy2) X(double*, gy3) X(double*, gelmn) X(const int*, rsp) X(const int*, rsn) X(const double*, rsg) X(const int*, rrp) X(const int*, rrn) X(const double*, rrg) \
  X(const int*, roots) X(const int*, rootn) X(const int*, pn_node) X(const int*, pn_pin) X(double*, c1) X(double*, c2) X(double*, rpi) X(double*, ctot) X(double*, elm) X(double*, gc1) X(double*, gc2) X(double*, grpi) X(double*, gctot) X(double*, gelm) \
  X(const int*, rnp) X(const int*, rno) X(const i64*, par_n) X(double*, a_in) X(u8*, a_chg) X(u8*, p_chg) X(double*, e_wd) X(double*, e_ls) X(double*, e_el) X(i64*, e_pk) \
  X(const int*, ecp) X(const int*, ecr) X(const i64*, ep_pin) X(const i64*, ck_tr) X(const i64*, ck_tf) X(const i64*, ck_type) X(const i64*, ck_evt) X(const double*, ck_cks) X(const double*, ck_off) X(const double*, REL) X(double*, ep_e) X(double*, ep_g) X(double*, AT) X(double*, SL) X(double*, gAT) X(double*, gSL)
struct DgC {
#define X_(t, n) t n;
  DG_PTRS(X_)
#undef X_
  Tab TB;
  i64 R, K, L, npt, nc, nk, nn, P, nr, npn, nep, KT, nevt;
  RL gwo, gwb, T, w3, w4, b1t, b2t, lr, b1, b2, aeps;
  double tau; double dl, dat; int mode; int use_timer; int first; i64 nrn;
};
enum { DGI_R, DGI_K, DGI_L, DGI_npt, DGI_nc, DGI_nk, DGI_nn, DGI_P, DGI_nr, DGI_npn, DGI_nep, DGI_KT, DGI_nevt, DGI_nrn, DGI_N };
extern "C" int dg_nptrs() {
  int n = 0;
#define X_(t, nm) n++;
  DG_PTRS(X_)
#undef X_
  return n;
}

// ---------------------------------------------------------------- loops
#ifdef __CUDACC__
template <class F> __global__ void k_for(F f, i64 n) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < n) f(t); }
template <class F> static inline void RUNF(const F& f, i64 n, int bs = 128) { if (n > 0) k_for<F><<<(unsigned)((n + bs - 1) / bs), (unsigned)bs>>>(f, n); }
static inline void SYNC() { cudaDeviceSynchronize(); }
#else
template <class F> static inline void RUNF(const F& f, i64 n, int bs = 128) { pfor(n, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) f(x); }); (void)bs; }
static inline void SYNC() {}
#endif
#define KERNEL(name) struct name { DgC c; HD void operator()(i64 i) const; }; HD void name::operator()(i64 i) const

HD RL rclamp(RL v, RL lo, RL hi) { return v < lo ? lo : (v > hi ? hi : v); }

// q = softmax(x / T), the two wire logs per candidate, the run's expected R and C per unit length, power per run
KERNEL(K_soft) {
  i64 K = c.K; int d = c.rdir[i];
  const RL* x = c.x + i * K; RL* p = c.p + i * K; const int* cand = c.Cand + d * K;
  RL mm = 0; bool first = true;
  RL v[8];
  for (i64 k = 0; k < K; k++) if (cand[k] >= 0) { v[k] = x[k] / c.T; if (first || v[k] > mm) mm = v[k]; first = false; }
  RL s = 0;
  for (i64 k = 0; k < K; k++) { if (cand[k] >= 0) { v[k] = exp(v[k] - mm); s += v[k]; } else v[k] = 0; }
  double rr = 0, cr = 0; RL pw = 0;
  const double* Lr = c.Lr + d * K; const double* Lc = c.Lc + d * K; const RL* Cc = c.Cc + d * K;
  for (i64 k = 0; k < K; k++) {
    RL q = v[k] / s; p[k] = q;
    c.l0[i * K + k] = log1p(q * c.gwo); c.l1[i * K + k] = log1p(q * c.gwb);
    double qd = (double)q; rr += qd * Lr[k]; cr += qd * Lc[k]; pw += q * Cc[k];
  }
  c.rr[i] = rr; c.cr[i] = cr; c.pwv[i] = pw * c.pwk[i];
}
// per point: via_terms (objective2): lF, lG, lN over the incident runs; dvia kept raw (its clamp at 0 is read by the users)
KERNEL(K_pt) {
  i64 K = c.K, L = c.L; int j0 = c.prp[i], j1 = c.prp[i + 1];
  RL lF[16], lG[16], lN[16];
  for (i64 z = 0; z < L; z++) { lF[z] = 0; lG[z] = 0; lN[z] = 0; }
  for (int j = j0; j < j1; j++) {
    i64 r = c.prr[j]; int d = c.rdir[r]; RL cum = 0; const RL* p = c.p + r * K; const int* zk = c.z2k + d * L;
    for (i64 z = 0; z < L; z++) {
      int k = zk[z]; RL pl = k >= 0 ? p[k] : (RL)0;
      cum += pl;
      RL F = rclamp(cum, DG_EPS, (RL)1), G = rclamp((RL)1 - F, DG_EPS, (RL)1), N = rclamp((RL)1 - pl, DG_EPS, (RL)1);
      lF[z] += log(F); lG[z] += log(G); lN[z] += log(N);
    }
  }
  RL* eF = c.eF + i * L; RL* eG = c.eG + i * L; RL* eN = c.eN + i * L; RL* dv = c.dv + i * L;
  int lo = c.pt_lo[i], hi = c.pt_hi[i];
  for (i64 z = 0; z < L; z++) { eF[z] = exp(lF[z]); eG[z] = exp(lG[z]); eN[z] = exp(lN[z]); }
  for (i64 z = 0; z < L; z++) {
    RL pmin = (z < lo) ? eG[z] : (RL)0, below = (z > 0 && z >= hi) ? eF[z - 1] : (RL)0;
    dv[z] = eN[z] - pmin - below;
  }
}
// per cell: lg, cost; uc = the gradient weight (0 above the clip, torch.clamp), uv = the value
KERNEL(K_cell) {
  bool op = c.copen[i] != 0;
  const RL* lw = op ? c.l0 : c.l1;
  RL lg = c.base[i];
  for (int e = c.cp[i]; e < c.cp[i + 1]; e++) lg += lw[c.ce[e]];
  for (int v = c.cvp[i]; v < c.cvp[i + 1]; v++) { RL pv = c.dv[c.cvi[v]]; if (pv > 0) lg += log1p(pv * c.cvg[v]); }
  RL cost = exp(lg <= DG_CLIP ? lg : (RL)DG_CLIP);
  RL uv = c.u[i] * cost;
  c.uv[i] = uv; c.uc[i] = lg <= DG_CLIP ? uv : (RL)0;
}
// per point: d(overflow)/d(dvia)
KERNEL(K_via_adj) {
  i64 L = c.L; RL g[16];
  for (i64 z = 0; z < L; z++) g[z] = 0;
  const RL* dv = c.dv + i * L;
  for (int v = c.pvp[i]; v < c.pvp[i + 1]; v++) {
    int z = c.pvz[v]; RL pv = dv[z]; RL gv = c.pvg[v];
    RL q = pv > 0 ? pv : (RL)0;
    g[z] += c.uc[c.pvc[v]] * gv / ((RL)1 + q * gv);
  }
  for (i64 z = 0; z < L; z++) c.gd[i * L + z] = dv[z] >= 0 ? g[z] : (RL)0;
}
// RC inputs per node
KERNEL(K_node) {
  c.y2[i * 2] = 0; c.y2[i * 2 + 1] = 0; c.y3[i * 2] = 0; c.y3[i * 2 + 1] = 0; c.elmn[i * 2] = 0; c.elmn[i * 2 + 1] = 0;
  int rr = c.rp_run[i];
  c.Rp[i] = rr >= 0 ? c.rr[rr] * c.rp_gl[i] : c.Rp0[i];
  double a = 0;
  for (int j = c.nsp[i]; j < c.nsp[i + 1]; j++) a += c.cr[c.nsr[j]] * c.nsg[j];
  c.y1[i * 2] = c.cap0[i * 2] + a; c.y1[i * 2 + 1] = c.cap0[i * 2 + 1] + a;
}
HD bool isf(double x) { return (x - x) == 0.0; }
// pi model at the root (gputimer._rc_core), per root net
KERNEL(K_root) {
  i64 n = c.roots[i], rt = c.rootn[i];
  for (int q = 0; q < 2; q++) {
    double Y1 = c.y1[rt * 2 + q], Y2 = c.y2[rt * 2 + q], Y3 = c.y3[rt * 2 + q];
    bool capld = (Y2 == 0) && (Y3 == 0);
    double c1r, c2r, rpr;
    if (capld) { c1r = Y1; c2r = 0.0; rpr = 0.0; }
    else { double r_ = Y2 * Y2 / Y3; c1r = r_; c2r = Y1 - r_; rpr = -Y3 * Y3 / (Y2 * Y2 * Y2); }
    c.c1[n * 2 + q] = nn0d(c1r); c.c2[n * 2 + q] = nn0d(c2r); c.rpi[n * 2 + q] = nn0d(rpr); c.ctot[n * 2 + q] = Y1;
  }
}
KERNEL(K_pinelm) { i64 nd = c.pn_node[i], pn = c.pn_pin[i]; c.elm[pn * 2] = c.elmn[nd * 2]; c.elm[pn * 2 + 1] = c.elmn[nd * 2 + 1]; }

// endpoints (gputimer.analyze): one thread per endpoint (rows of its pin, clocks, transitions)
HD double ep_setup(const DgC& c, i64 row, int rf, double sl) {
  i64 tid = rf == 0 ? c.ck_tr[row] : c.ck_tf[row];
  if (tid >= 0) return lut(c.TB, tid, c.ck_cks[row], sl);
  if (c.ck_type[row] <= 1) return mknan();
  return 0.0;
}
KERNEL(K_ep) {
  i64 pin = c.ep_pin[i], KT = c.KT;
  double m = BIGV, s = 0.0;
  double SLr = c.SL[pin * 2], SLf = c.SL[pin * 2 + 1];
  for (int pass = 0; pass < (c.mode == 1 ? 2 : 1); pass++) {
    for (int j = c.ecp[i]; j < c.ecp[i + 1]; j++) {
      i64 row = c.ecr[j]; double off = c.ck_off[row]; i64 evt = c.ck_evt[row];
      for (int rf = 0; rf < 2; rf++) {
        double st = ep_setup(c, row, rf, rf == 0 ? SLr : SLf);
        for (i64 k = 0; k < KT; k++) {
          double req = c.REL[k * c.nevt + evt];
          double rk = req - st - off, at = c.AT[(pin * KT + k) * 2 + rf];
          bool v = (at > NEGV / 2) && (req < BIGV / 2) && fin(rk);
          if (!v) continue;
          double sl = rk - at;
          if (pass == 0) { if (sl < m) m = sl; }
          else s += exp(-(sl - m) / c.tau);
        }
      }
    }
  }
  c.ep_e[i] = (c.mode == 1 && s > 0) ? m - c.tau * log(s) : m;
}
KERNEL(K_ep_b) {
  double ge = c.ep_g[i];
  if (ge == 0.0) return;
  i64 pin = c.ep_pin[i], KT = c.KT;
  double m = BIGV, s = 0.0;
  double SLr = c.SL[pin * 2], SLf = c.SL[pin * 2 + 1];
  for (int pass = 0; pass < 3; pass++) {
    for (int j = c.ecp[i]; j < c.ecp[i + 1]; j++) {
      i64 row = c.ecr[j]; double off = c.ck_off[row]; i64 evt = c.ck_evt[row];
      for (int rf = 0; rf < 2; rf++) {
        D5 ss = seed<5>(rf == 0 ? SLr : SLf, 0);
        i64 tid = rf == 0 ? c.ck_tr[row] : c.ck_tf[row];
        double st, dst = 0.0;
        if (tid >= 0) { D5 r_ = lut(c.TB, tid, D5(c.ck_cks[row]), ss); st = r_.v; dst = r_.d[0]; }
        else st = c.ck_type[row] <= 1 ? mknan() : 0.0;
        for (i64 k = 0; k < KT; k++) {
          double req = c.REL[k * c.nevt + evt];
          double rk = req - st - off, at = c.AT[(pin * KT + k) * 2 + rf];
          bool v = (at > NEGV / 2) && (req < BIGV / 2) && fin(rk);
          if (!v) continue;
          double sl = rk - at;
          if (pass == 0) { if (sl < m) m = sl; }
          else if (pass == 1) s += exp(-(sl - m) / c.tau);
          else {
            double w = exp(-(sl - m) / c.tau) / s * ge;
            aadd(&c.gAT[(pin * KT + k) * 2 + rf], -w);
            aadd(&c.gSL[pin * 2 + rf], -w * dst);
          }
        }
      }
    }
  }
}
// adjoint at the roots and at the pins
KERNEL(K_root_b) {
  i64 n = c.roots[i], rt = c.rootn[i];
  for (int q = 0; q < 2; q++) {
    double Y1 = c.y1[rt * 2 + q], Y2 = c.y2[rt * 2 + q], Y3 = c.y3[rt * 2 + q];
    double g1 = c.gc1[n * 2 + q], g2 = c.gc2[n * 2 + q], gr = c.grpi[n * 2 + q], gt = c.gctot[n * 2 + q];
    bool capld = (Y2 == 0) && (Y3 == 0);
    double a1 = gt, a2 = 0, a3 = 0;
    if (capld) { if (isf(Y1)) a1 += g1; }
    else {
      double c1r = Y2 * Y2 / Y3, c2r = Y1 - c1r, rpr = -Y3 * Y3 / (Y2 * Y2 * Y2);
      if (!isf(c1r)) g1 = 0.0;
      if (!isf(c2r)) g2 = 0.0;
      if (!isf(rpr)) gr = 0.0;
      double dY2 = 2.0 * Y2 / Y3, dY3 = -Y2 * Y2 / (Y3 * Y3);
      a1 += g2;
      a2 += (g1 - g2) * dY2 + gr * (3.0 * Y3 * Y3 / (Y2 * Y2 * Y2 * Y2));
      a3 += (g1 - g2) * dY3 + gr * (-2.0 * Y3 / (Y2 * Y2 * Y2));
    }
    c.gy1[rt * 2 + q] = a1; c.gy2[rt * 2 + q] = a2; c.gy3[rt * 2 + q] = a3;
  }
}
KERNEL(K_pinelm_b) { i64 nd = c.pn_node[i], pn = c.pn_pin[i]; c.gelmn[nd * 2] = c.gelm[pn * 2]; c.gelmn[nd * 2 + 1] = c.gelm[pn * 2 + 1]; }

// per run: the whole adjoint (wire entries, via terms, power, R and C), softmax's adjoint, Adam
HD void run_grad(const DgC& c, i64 i, RL* gq) {
  i64 K = c.K, L = c.L; int d = c.rdir[i];
  const RL* p = c.p + i * K; const int* cand = c.Cand + d * K;
  // wire entries, in su
  for (i64 k = 0; k < K; k++) {
    gq[k] = 0;
    if (cand[k] < 0) continue;
    RL q = p[k], ao = c.gwo / ((RL)1 + q * c.gwo), ab = c.gwb / ((RL)1 + q * c.gwb), g = 0;
    i64 rk = i * K + k;
    for (int e = c.ent_ptr[rk]; e < c.ent_ptr[rk + 1]; e++) { int ce = c.ec[e]; g += c.uc[ce] * (c.copen[ce] ? ao : ab); }
    gq[k] = g;
  }
  // via terms (via_terms adjoint): from the two points
  RL AF[16], AN[16];
  for (i64 z = 0; z < L; z++) { AF[z] = 0; AN[z] = 0; }
  const int* zk = c.z2k + d * L;
  RL cum = 0;
  RL Fv[16], Gv[16], Nv[16]; bool g1[16], g2[16], g3[16];
  for (i64 z = 0; z < L; z++) {
    int k = zk[z]; RL pl = k >= 0 ? p[k] : (RL)0;
    cum += pl;
    g1[z] = cum >= (RL)DG_EPS && cum <= (RL)1;
    Fv[z] = rclamp(cum, DG_EPS, (RL)1);
    RL gg = (RL)1 - Fv[z];
    g2[z] = gg >= (RL)DG_EPS && gg <= (RL)1;
    Gv[z] = rclamp(gg, DG_EPS, (RL)1);
    RL nn = (RL)1 - pl;
    g3[z] = nn >= (RL)DG_EPS && nn <= (RL)1;
    Nv[z] = rclamp(nn, DG_EPS, (RL)1);
  }
  for (int q = 0; q < 2; q++) {
    i64 pt = q == 0 ? c.rp0[i] : c.rp1[i];
    int lo = c.pt_lo[pt], hi = c.pt_hi[pt];
    const RL* gd = c.gd + pt * L; const RL* eF = c.eF + pt * L; const RL* eG = c.eG + pt * L; const RL* eN = c.eN + pt * L;
    for (i64 z = 0; z < L; z++) {
      RL gF = (z + 1 < L && (z + 1) >= hi) ? -gd[z + 1] * eF[z] : (RL)0;
      RL gG = (z < lo) ? -gd[z] * eG[z] : (RL)0;
      RL gN = gd[z] * eN[z];
      RL a = gF / Fv[z];
      if (g2[z]) a += -gG / Gv[z];
      AF[z] += a;
      if (g3[z]) AN[z] += -gN / Nv[z];
    }
  }
  RL acc = 0;
  for (i64 z = L - 1; z >= 0; z--) {
    if (g1[z]) acc += AF[z];
    int k = zk[z];
    if (k >= 0) gq[k] += acc + AN[z];
  }
  for (i64 k = 0; k < K; k++) if (cand[k] >= 0) gq[k] = c.w4 * gq[k] + c.w3 * c.pwk[i] * c.Cc[d * K + k];
  // R and C
  if (c.use_timer) {
    double gc = 0, gr = 0;
    for (int j = c.rsp[i]; j < c.rsp[i + 1]; j++) { i64 nd = c.rsn[j]; gc += (c.gy1[nd * 2] + c.gy1[nd * 2 + 1]) * c.rsg[j]; }
    for (int j = c.rrp[i]; j < c.rrp[i + 1]; j++) gr += c.gR[c.rrn[j]] * c.rrg[j];
    const double* Lr = c.Lr + d * K; const double* Lc = c.Lc + d * K;
    for (i64 k = 0; k < K; k++) if (cand[k] >= 0) gq[k] += (RL)(gr * Lr[k] + gc * Lc[k]);
  }
}
KERNEL(K_step) {
  i64 K = c.K; int d = c.rdir[i]; const int* cand = c.Cand + d * K;
  RL gq[8];
  run_grad(c, i, gq);
  const RL* p = c.p + i * K;
  RL gs = 0;
  for (i64 k = 0; k < K; k++) if (cand[k] >= 0) gs += p[k] * gq[k];
  for (i64 k = 0; k < K; k++) {
    if (cand[k] < 0) continue;
    RL g = p[k] * (gq[k] - gs) / c.T;
    RL* m = c.am + i * K + k; RL* v = c.av + i * K + k;
    *m = c.b1 * *m + ((RL)1 - c.b1) * g;
    *v = c.b2 * *v + ((RL)1 - c.b2) * g * g;
    RL den = sqrt(*v) / sqrt(c.b2t) + c.aeps;
    c.x[i * K + k] -= (c.lr / c.b1t) * *m / den;
  }
}
// gradient per candidate, no update (tests, the linear order of a split)
struct K_grad { DgC c; RL* out; HD void operator()(i64 i) const { RL gq[8]; run_grad(c, i, gq); for (i64 k = 0; k < c.K; k++) out[i * c.K + k] = gq[k]; } };
// sums in chunks (the values are reported, not used by the gradient)
struct K_sum { const RL* a; double* o; i64 n, ch; HD void operator()(i64 t) const { double s = 0; i64 e = (t + 1) * ch; if (e > n) e = n; for (i64 j = t * ch; j < e; j++) s += a[j]; o[t] = s; } };

// ---------------------------------------------------------------- RC per net: one thread per net, nodes of the net by depth (one launch per direction; the depth buckets of gt_rc_* need 4 x depth launches)
KERNEL(K_rcn_f) {
  const int j0 = c.rnp[i], j1 = c.rnp[i + 1];
  for (int j = j1 - 1; j >= j0; j--) {
    i64 w = c.rno[j], p = c.par_n[w]; double R = c.Rp[w];
    for (int q = 0; q < 2; q++) {
      double a1 = c.y1[w * 2 + q], a2 = c.y2[w * 2 + q], a3 = c.y3[w * 2 + q];
      c.y1[p * 2 + q] += a1; c.y2[p * 2 + q] += a2 - R * a1 * a1; c.y3[p * 2 + q] += a3 - 2 * R * a1 * a2 + R * R * (a1 * a1 * a1);
    }
  }
  for (int j = j0; j < j1; j++) {
    i64 w = c.rno[j], p = c.par_n[w]; double R = c.Rp[w];
    for (int q = 0; q < 2; q++) c.elmn[w * 2 + q] = c.elmn[p * 2 + q] + R * c.y1[w * 2 + q] * 1e-3;
  }
}
KERNEL(K_rcn_b) {
  const int j0 = c.rnp[i], j1 = c.rnp[i + 1];
  for (int j = j1 - 1; j >= j0; j--) {
    i64 w = c.rno[j], p = c.par_n[w]; double R = c.Rp[w];
    for (int q = 0; q < 2; q++) {
      double ge = c.gelmn[w * 2 + q];
      c.gy1[w * 2 + q] += ge * R * 1e-3; c.gR[w] += ge * c.y1[w * 2 + q] * 1e-3; c.gelmn[p * 2 + q] += ge;
    }
  }
  for (int j = j0; j < j1; j++) {
    i64 w = c.rno[j], p = c.par_n[w]; double R = c.Rp[w], gr = 0.0;
    for (int q = 0; q < 2; q++) {
      double g1 = c.gy1[p * 2 + q], g2 = c.gy2[p * 2 + q], g3 = c.gy3[p * 2 + q];
      double a1 = c.y1[w * 2 + q], a2 = c.y2[w * 2 + q];
      c.gy1[w * 2 + q] += g1 + g2 * (-2 * R * a1) + g3 * (-2 * R * a2 + 3 * R * R * a1 * a1);
      c.gy2[w * 2 + q] += g2 + g3 * (-2 * R * a1);
      c.gy3[w * 2 + q] += g3;
      gr += g2 * (-a1 * a1) + g3 * (-2 * a1 * a2 + 2 * R * a1 * a1 * a1);
    }
    c.gR[w] += gr;
  }
}
// ---------------------------------------------------------------- the timer's forward with reuse: an arc whose five inputs moved by at most dl (relative) since it was last computed keeps its outputs and local derivatives;
// a group none of whose arcs or arrivals changed, a wire edge whose driver record and Elmore did not change, keep theirs. dl = 0, dat = 0: only bitwise repeats are reused (values as the full sweep).
struct Ev { double* a_in; u8 *a_chg, *p_chg; double *e_wd, *e_ls, *e_el; i64* e_pk; double dl, dat; int first; };
HD bool near_(double x, double r, double dl) { if (x == r) return true; double d = x - r; if (d < 0) d = -d; double a = r < 0 ? -r : r; return d <= dl * a; }
HD void arc_ev(const Ctx& C, const Ev& E, i64 e) {
  i64 frm = C.ce_from[e], to = C.ce_to[e], irf = C.ce_irf[e], orf = C.ce_orf[e];
  i64 net = C.pin_net[to]; if (net < 0) net += C.nn;
  double in[5] = {C.SL[frm * 2 + irf], C.c1[net * 2 + orf], C.c2[net * 2 + orf], C.rpi[net * 2 + orf], C.ctot[net * 2 + orf]};
  double* r = E.a_in + e * 5;
  if (!E.first && near_(in[0], r[0], E.dl) && near_(in[1], r[1], E.dl) && near_(in[2], r[2], E.dl) && near_(in[3], r[3], E.dl) && near_(in[4], r[4], E.dl)) { E.a_chg[e] = 0; return; }
  arc_fwd(C, e);
  for (int q = 0; q < 5; q++) r[q] = in[q];
  E.a_chg[e] = 1;
}
HD void grp_ev(const Ctx& C, const Ev& E, i64 gg, i64 gb, i64 b) {
  i64 s0 = C.g_start[gg], s1 = (gg + 1 < gb) ? C.g_start[gg + 1] : b;
  i64 gto = C.g_to[gg], gorf = C.g_orf[gg], K = C.K;
  u8 any = E.first ? 1 : 0;
  if (!any) for (i64 e = s0; e < s1; e++) if (E.a_chg[e] || E.p_chg[C.ce_from[e] * 2 + C.ce_irf[e]]) { any = 1; break; }
  if (!any) { E.p_chg[gto * 2 + gorf] = 0; return; }
  double old[32]; for (i64 k = 0; k < K && k < 32; k++) old[k] = C.AT[(gto * K + k) * 2 + gorf];
  grp_fwd(C, gg, gb, b);
  u8 ch = E.first ? 1 : 0;
  for (i64 k = 0; k < K; k++) { double d = C.AT[(gto * K + k) * 2 + gorf] - (k < 32 ? old[k] : 0.0); if (d > E.dat || d < -E.dat || k >= 32) ch = 1; }
  E.p_chg[gto * 2 + gorf] = ch;
}
// tools/weights/gt_kern_w.cu edge_fwd + the delay and slew kept for the next step
HD void edge_ev(const Ctx& C, const Ev& E, i64 i, i64 rf) {
  i64 drv = C.ne_drv[i], snk = C.ne_snk[i], K = C.K;
  i64 j = i * 2 + rf;
  double elm = C.elm[snk * 2 + rf];
  i64 pk = C.pk[drv * 2 + rf];
  bool same = !E.first && pk == E.e_pk[j] && (pk < 0 || !E.a_chg[pk]) && !C.is_port[drv];
  if (same) {
    double r = E.e_el[j];
    if (fin(elm) != fin(r)) same = false;
    else if (fin(elm) && !near_(elm, r, E.dl)) same = false;
  }
  if (same) {
    u8 ch = E.p_chg[drv * 2 + rf];
    if (ch) { double wd = E.e_wd[j]; for (i64 k = 0; k < K; k++) C.AT[(snk * K + k) * 2 + rf] = C.AT[(drv * K + k) * 2 + rf] + wd; }
    E.p_chg[snk * 2 + rf] = ch;
    return;
  }
  edge_fwd(C, i, rf);
  // the delay of this edge: from the sink's arrival written by edge_fwd when a real arrival exists; else the unknown delay is not read (NEG + delay stays NEG in double)
  double wd = 0.0; bool got = false;
  for (i64 k = 0; k < K && !got; k++) { double ad = C.AT[(drv * K + k) * 2 + rf]; if (ad > NEGV / 2) { wd = C.AT[(snk * K + k) * 2 + rf] - ad; got = true; } }
  E.e_wd[j] = wd; E.e_el[j] = elm; E.e_pk[j] = pk;
  E.p_chg[snk * 2 + rf] = 1;
}
HD void grp_bev(const Ctx& C, i64 gg, i64 gb, i64 b) {
  i64 gto = C.g_to[gg], gorf = C.g_orf[gg], K = C.K;
  bool nz = C.gSL[gto * 2 + gorf] != 0.0;
  const double* gR = C.gREC + (gto * 2 + gorf) * 5;
  for (int q = 0; q < 5 && !nz; q++) nz = gR[q] != 0.0;
  for (i64 k = 0; k < K && !nz; k++) nz = C.gAT[(gto * K + k) * 2 + gorf] != 0.0;
  if (nz) grp_bwd(C, gg, gb, b);
}
#ifdef __CUDACC__
__global__ void k_arc2(Ctx C, Ev E, i64 a, i64 n) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < n) arc_ev(C, E, a + t); }
__global__ void k_grp2(Ctx C, Ev E, i64 ga, i64 n, i64 gb, i64 b) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < n) grp_ev(C, E, ga + t, gb, b); }
__global__ void k_edge2(Ctx C, Ev E, i64 ea, i64 nb) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < 2 * nb) { i64 rf = t / nb, j = t % nb; edge_ev(C, E, ea + j, rf); } }
__global__ void k_grp2b(Ctx C, i64 ga, i64 n, i64 gb, i64 b) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < n) grp_bev(C, ga + t, gb, b); }
#endif
static double PROF_TF[3];
static int dg_forward(const i64* t, const Ev& E, int mode, double tau, bool prof) {
  Ctx C = make_ctx(t, mode, tau);
  i64 nlev = t[T_nlev];
  (void)prof;
  for (i64 lv = 1; lv < nlev; lv++) {
    i64 a, na, ga, ng, ea, ne, gb, b;
    ranges(t, lv, false, &a, &na, &ga, &ng, &ea, &ne, &gb, &b);
#ifdef __CUDACC__
    if (prof) {
      double t0 = (cudaDeviceSynchronize(), std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
      if (na > 0) k_arc2<<<LAUNCH(na)>>>(C, E, a, na);
      double t1 = (cudaDeviceSynchronize(), std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
      if (ng > 0) k_grp2<<<LAUNCH(ng)>>>(C, E, ga, ng, gb, b);
      double t2 = (cudaDeviceSynchronize(), std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
      if (ne > 0) k_edge2<<<LAUNCH(2 * ne)>>>(C, E, ea, ne);
      double t3 = (cudaDeviceSynchronize(), std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
      PROF_TF[0] += t1 - t0; PROF_TF[1] += t2 - t1; PROF_TF[2] += t3 - t2;
      continue;
    }
    if (na > 0) k_arc2<<<LAUNCH(na)>>>(C, E, a, na);
    if (ng > 0) k_grp2<<<LAUNCH(ng)>>>(C, E, ga, ng, gb, b);
    if (ne > 0) k_edge2<<<LAUNCH(2 * ne)>>>(C, E, ea, ne);
#else
    pfor(na, 64, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) arc_ev(C, E, a + x); });
    pfor(ng, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) grp_ev(C, E, ga + x, gb, b); });
    pfor(2 * ne, 128, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) { i64 rf = x / ne, j = x % ne; edge_ev(C, E, ea + j, rf); } });
#endif
  }
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "dg_forward: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}
static int dg_backward(const i64* t, int mode, double tau) {
  Ctx C = make_ctx(t, mode, tau);
  i64 nlev = t[T_nlev];
  for (i64 lv = nlev - 1; lv >= 1; lv--) {
    i64 a, na, ga, ng, ea, ne, gb, b;
    ranges(t, lv, false, &a, &na, &ga, &ng, &ea, &ne, &gb, &b);
#ifdef __CUDACC__
    if (ne > 0) k_edge<<<LAUNCH(2 * ne)>>>(C, ea, ne, nullptr, 1);
    if (ng > 0) k_grp2b<<<LAUNCH(ng)>>>(C, ga, ng, gb, b);
#else
    pfor(2 * ne, 128, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) { i64 rf = x / ne, j = x % ne; edge_bwd(C, ea + j, rf); } });
    pfor(ng, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) grp_bev(C, ga + x, gb, b); });
#endif
  }
#ifdef __CUDACC__
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "dg_backward: %s\n", cudaGetErrorString(err)); return 1; }
#endif
  return 0;
}

// ---------------------------------------------------------------- one call
struct Prof { double soft, pt, cell, via, node, rcf, root, tf, ep, ep_b, tb, root_b, rcb, step, host; };
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static DgC make_dg(const i64* t, const i64* I, const double* Dv, const i64* gt) {
  DgC c; memset((void*)&c, 0, sizeof(c)); i64 q = 0;
  if (gt) {
    c.TB.A = (const double*)gt[T_tabA]; c.TB.B = (const double*)gt[T_tabB]; c.TB.V = (const double*)gt[T_tabV];
    c.TB.NA = (const i64*)gt[T_tabNA]; c.TB.NB = (const i64*)gt[T_tabNB]; c.TB.M = gt[T_tabM]; c.TB.T = gt[T_tabT];
  }
#define X_(tp, nm) c.nm = (tp)t[q++];
  DG_PTRS(X_)
#undef X_
  c.R = I[DGI_R]; c.K = I[DGI_K]; c.L = I[DGI_L]; c.npt = I[DGI_npt]; c.nc = I[DGI_nc]; c.nk = I[DGI_nk]; c.nn = I[DGI_nn]; c.P = I[DGI_P]; c.nr = I[DGI_nr]; c.npn = I[DGI_npn]; c.nep = I[DGI_nep]; c.KT = I[DGI_KT]; c.nevt = I[DGI_nevt]; c.nrn = I[DGI_nrn];
  c.gwo = (RL)expm1(2.0 * 0.25); c.gwb = (RL)expm1(2.0 * 0.75);
  c.w3 = (RL)Dv[0]; c.w4 = (RL)Dv[1];
  c.b1 = (RL)0.9; c.b2 = (RL)0.999; c.aeps = (RL)1e-8;
  return c;
}
static double dev_sum(const RL* a, i64 n, double* dv, i64 cap) {
  i64 ch = (n + cap - 1) / (cap > 0 ? cap : 1); if (ch < 1024) ch = 1024;
  i64 m = (n + ch - 1) / ch;
  K_sum f{a, dv, n, ch};
#ifdef __CUDACC__
  if (m > 0) k_for<K_sum><<<(unsigned)((m + 31) / 32), 32>>>(f, m);
  cudaDeviceSynchronize();
  std::vector<double> h(m); dg_get(h.data(), dv, m * 8);
#else
  pfor(m, 1, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) f(x); });
  std::vector<double> h(dv, dv + m);
#endif
  double s = 0; for (double v : h) s += v;
  return s;
}

// forward of everything at the current x, temperature T; values[0..4] = of pw wns tns total (su); mode 1 smooth, 0 hard (hard: no gradient possible)
static int forward_all(DgC& c, const i64* gt, i64* grc, double w1, double w2n, double* sumdv, i64 sumcap, double* val, std::vector<double>& eh, Prof* pr, bool want_values) {
  double t = now_s();
  auto tick = [&](double* a) { if (pr) { SYNC(); double n = now_s(); *a += n - t; t = n; } };
  RUNF(K_soft{c}, c.R); tick(pr ? &pr->soft : nullptr);
  RUNF(K_pt{c}, c.npt); tick(pr ? &pr->pt : nullptr);
  RUNF(K_cell{c}, c.nc); tick(pr ? &pr->cell : nullptr);
  if (c.use_timer) {
    RUNF(K_node{c}, c.nk); tick(pr ? &pr->node : nullptr);
    RUNF(K_rcn_f{c}, c.nrn, 32);
    tick(pr ? &pr->rcf : nullptr);
    RUNF(K_root{c}, c.nr); RUNF(K_pinelm{c}, c.npn); tick(pr ? &pr->root : nullptr);
    { Ev E{c.a_in, c.a_chg, c.p_chg, c.e_wd, c.e_ls, c.e_el, c.e_pk, c.dl, c.dat, c.first};
      if (dg_forward(gt, E, c.mode, c.tau, pr != nullptr && getenv("DG_PROF_TF") != nullptr) != 0) return 1; }
    tick(pr ? &pr->tf : nullptr);
    RUNF(K_ep{c}, c.nep);
    SYNC();
    eh.resize(c.nep);
    dg_get(eh.data(), c.ep_e, c.nep * 8);
    tick(pr ? &pr->ep : nullptr);
  }
  double t_of = 0, t_pw = 0;
  if (want_values) { t_of = dev_sum(c.uv, c.nc, sumdv, sumcap); t_pw = dev_sum(c.pwv, c.R, sumdv, sumcap); }
  val[0] = (double)c.w4 * t_of; val[1] = (double)c.w3 * t_pw; val[2] = 0; val[3] = 0;
  if (c.use_timer) {
    double wns = 0, tns = 0;
    if (c.mode == 1) {
      double tau = c.tau, mxe = 0.0;
      for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) { double a = -eh[i] / tau; if (a > mxe) mxe = a; }
      double s = exp(0.0 - mxe);
      for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) {
        double a = -eh[i] / tau;
        s += exp(a - mxe);
        // softplus(a) = log1p(exp(a)) (a > 0: a + log1p(exp(-a)))
        tns -= tau * (a > 0 ? a + log1p(exp(-a)) : log1p(exp(a)));
      }
      wns = -tau * (mxe + log(s));
    } else {
      bool any = false; double mn_ = 0.0;
      for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) { if (!any || eh[i] < mn_) mn_ = eh[i]; any = true; if (eh[i] < 0) tns += eh[i]; }
      wns = any && mn_ < 0 ? mn_ : 0.0;
    }
    val[2] = w1 * wns; val[3] = w2n * tns;
    val[5] = wns; val[6] = tns;
  }
  val[4] = val[0] + val[1] + val[2] + val[3];
  return 0;
}

static int backward_all(DgC& c, const i64* gt, i64* grc, double w1, double w2n, const std::vector<double>& eh, std::vector<double>& gh, const i64* gzero, i64 nzero, Prof* pr) {
  double t = now_s();
  auto tick = [&](double* a) { if (pr) { SYNC(); double n = now_s(); *a += n - t; t = n; } };
  RUNF(K_via_adj{c}, c.npt); tick(pr ? &pr->via : nullptr);
  if (c.use_timer) {
    for (i64 j = 0; j < nzero; j++) dg_zero((void*)gzero[2 * j], gzero[2 * j + 1]);
    double tau = c.tau, mxe = 0.0;
    gh.resize(c.nep);
    for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) { double a = -eh[i] / tau; if (a > mxe) mxe = a; }
    double s = exp(0.0 - mxe);
    for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) s += exp(-eh[i] / tau - mxe);
    for (i64 i = 0; i < c.nep; i++) {
      if (!(eh[i] < BIGV / 2)) { gh[i] = 0.0; continue; }
      double a = -eh[i] / tau;
      double sig = a > 0 ? 1.0 / (1.0 + exp(-a)) : exp(a) / (1.0 + exp(a));
      gh[i] = w1 * exp(a - mxe) / s + w2n * sig;
    }
    dg_put(c.ep_g, gh.data(), c.nep * 8);
    RUNF(K_ep_b{c}, c.nep); tick(pr ? &pr->ep_b : nullptr);
    if (dg_backward(gt, c.mode, c.tau) != 0) return 1;
    tick(pr ? &pr->tb : nullptr);
    RUNF(K_root_b{c}, c.nr); RUNF(K_pinelm_b{c}, c.npn); tick(pr ? &pr->root_b : nullptr);
    RUNF(K_rcn_b{c}, c.nrn, 32);
    tick(pr ? &pr->rcb : nullptr);
  }
  return 0;
}

// x from the host [R*K], or left as it is when x0 is null; every array of the table is in device memory.
// Dv: 0 w3, 1 w4, 2 w1, 3 w2 / nend, 4 tau, 5 T start, 6 T end, 7 lr, 8 stop rule: relative fall (0 = off), 9 stop patience in steps, 10 dl (relative input move an arc ignores), 11 dat (ns: arrival move not passed on)
// out_val[K * 7] = per step of tot pw... (of, pw, wns, tns, total, wns raw, tns raw); ret_it[0] = steps run; prof[15] seconds (pieces, only when prof_on).
extern "C" int dg_run(const i64* t, const i64* I, const double* Dv, const i64* gt, i64* grc, const i64* gzero, i64 nzero, i64 K, double* out_val, i64* ret_it, double* prof, int prof_on) {
  DgC c = make_dg(t, I, Dv, gt);
  c.mode = 1; c.tau = Dv[4]; c.use_timer = (gt != nullptr); c.lr = (RL)Dv[7]; c.dl = Dv[10]; c.dat = Dv[11];
  double w1 = Dv[2], w2n = Dv[3];
  void* sumdv = dg_alloc(8 * 8192);
  std::vector<double> eh, gh; Prof pr; memset(&pr, 0, sizeof(pr)); Prof* ppr = prof_on ? &pr : nullptr;
  double b1p = 1.0, b2p = 1.0;
  double best = 1e300; i64 last_best = 0;
  i64 it = 0;
  double t_loop = now_s(), lr0 = Dv[7], lrK = Dv[14], lrC = Dv[15];
  auto lr_of = [&](i64 K_) { double l = lr0; if (lrK > 0 && lrK / (double)K_ > l) l = lrK / (double)K_; if (lrC > 0 && lrC / ((double)K_ * (double)K_) > l) l = lrC / ((double)K_ * (double)K_); return (RL)l; };
  c.lr = lr_of(K);
  for (; it < K; it++) {
    if (it == 2 && Dv[13] > 0) {   // two steps measured: how many fit
      SYNC();
      double per = (now_s() - t_loop) / 2.0;
      i64 fit = (i64)((Dv[13] - (now_s() - t_loop)) / per) + 2;
      if (fit < 3) fit = 3;
      if (fit < K) { K = fit; c.lr = lr_of(K); }
    }
    c.T = (RL)(Dv[5] * pow(Dv[6] / Dv[5], (double)it / (double)(K > 1 ? K - 1 : 1)));
    c.first = (it == 0 && Dv[12] == 0) ? 1 : 0;
    double val[7] = {0, 0, 0, 0, 0, 0, 0};
    if (forward_all(c, gt, grc, w1, w2n, (double*)sumdv, 8192, val, eh, ppr, true) != 0) return 1;
    for (int q = 0; q < 7; q++) out_val[it * 8 + q] = val[q];
    out_val[it * 8 + 7] = 0;
    if (prof_on && gt) { i64 nE_ = gt[T_nE]; std::vector<u8> h_(nE_); dg_get(h_.data(), c.a_chg, nE_); i64 s_ = 0; for (i64 q = 0; q < nE_; q++) s_ += h_[q]; out_val[it * 8 + 7] = (double)s_; }
    if (backward_all(c, gt, grc, w1, w2n, eh, gh, gzero, nzero, ppr) != 0) return 1;
    double ts = now_s();
    b1p *= 0.9; b2p *= 0.999;
    c.b1t = (RL)(1.0 - b1p); c.b2t = (RL)(1.0 - b2p);
    RUNF(K_step{c}, c.R);
    SYNC();
    if (prof_on) pr.step += now_s() - ts;
    if (Dv[8] > 0) {          // stop rule: the objective has not fallen by the fraction Dv[8] of its fall so far for Dv[9] steps
      double f0 = out_val[4];
      if (val[4] < best - Dv[8] * fabs(f0 - best)) { best = val[4]; last_best = it; }
      else if (it - last_best >= (i64)Dv[9] && it + 1 < K) { it++; break; }
    }
  }
  ret_it[0] = it;
  double* pp = (double*)&pr; for (int q = 0; q < 15; q++) prof[q] = pp[q];
  prof[15] = PROF_TF[0]; prof[16] = PROF_TF[1]; prof[17] = PROF_TF[2]; PROF_TF[0] = PROF_TF[1] = PROF_TF[2] = 0;
  dg_free(sumdv);
  return 0;
}
// one forward at the current x (T, mode, tau given) without touching x: values only, or with the gradient per candidate into gout (device, R*K of RL)
extern "C" int dg_eval(const i64* t, const i64* I, const double* Dv, const i64* gt, i64* grc, const i64* gzero, i64 nzero, double T, int mode, void* gout, double* val) {
  DgC c = make_dg(t, I, Dv, gt);
  c.mode = mode; c.tau = Dv[4]; c.use_timer = (gt != nullptr); c.T = (RL)T; c.first = 1; c.dl = 0; c.dat = 0;
  double w1 = Dv[2], w2n = Dv[3];
  void* sumdv = dg_alloc(8 * 8192);
  std::vector<double> eh, gh;
  if (forward_all(c, gt, grc, w1, w2n, (double*)sumdv, 8192, val, eh, nullptr, true) != 0) return 1;
  if (gout) {
    if (mode != 1) { fprintf(stderr, "dg_eval: gradient needs the smooth mode\n"); return 1; }
    if (backward_all(c, gt, grc, w1, w2n, eh, gh, gzero, nzero, nullptr) != 0) return 1;
    K_grad f{c, (RL*)gout};
#ifdef __CUDACC__
    if (c.R > 0) k_for<K_grad><<<(unsigned)((c.R + 127) / 128), 128>>>(f, c.R);
    cudaDeviceSynchronize();
#else
    pfor(c.R, 256, [&](i64 lo, i64 hi) { for (i64 x = lo; x < hi; x++) f(x); });
#endif
  }
  SYNC();
  dg_free(sumdv);
  return 0;
}
// the timer alone at the RC in the device arrays: every element computed (first = 1, the kept inputs are written), endpoints, adjoint: gc1 gc2 grpi gctot gelm per net / pin, no problem needed
extern "C" int dg_sens(const i64* t, const i64* I, const double* Dv, const i64* gt, const i64* gzero, i64 nzero, double* val) {
  DgC c = make_dg(t, I, Dv, gt);
  c.mode = 1; c.tau = Dv[4]; c.use_timer = 1; c.first = 1; c.dl = 0; c.dat = 0;
  double w1 = Dv[2], w2n = Dv[3];
  Ev E{c.a_in, c.a_chg, c.p_chg, c.e_wd, c.e_ls, c.e_el, c.e_pk, c.dl, c.dat, c.first};
  if (dg_forward(gt, E, c.mode, c.tau, false) != 0) return 1;
  RUNF(K_ep{c}, c.nep);
  SYNC();
  std::vector<double> eh(c.nep), gh(c.nep);
  dg_get(eh.data(), c.ep_e, c.nep * 8);
  double tau = c.tau, mxe = 0.0, tns = 0.0;
  for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) { double a = -eh[i] / tau; if (a > mxe) mxe = a; }
  double s = exp(0.0 - mxe);
  for (i64 i = 0; i < c.nep; i++) if (eh[i] < BIGV / 2) { double a = -eh[i] / tau; s += exp(a - mxe); tns -= tau * (a > 0 ? a + log1p(exp(-a)) : log1p(exp(a))); }
  double wns = -tau * (mxe + log(s));
  val[0] = 0; val[1] = 0; val[2] = w1 * wns; val[3] = w2n * tns; val[4] = val[2] + val[3]; val[5] = wns; val[6] = tns;
  for (i64 j = 0; j < nzero; j++) dg_zero((void*)gzero[2 * j], gzero[2 * j + 1]);
  for (i64 i = 0; i < c.nep; i++) {
    if (!(eh[i] < BIGV / 2)) { gh[i] = 0.0; continue; }
    double a = -eh[i] / tau;
    double sig = a > 0 ? 1.0 / (1.0 + exp(-a)) : exp(a) / (1.0 + exp(a));
    gh[i] = w1 * exp(a - mxe) / s + w2n * sig;
  }
  dg_put(c.ep_g, gh.data(), c.nep * 8);
  RUNF(K_ep_b{c}, c.nep);
  if (dg_backward(gt, c.mode, c.tau) != 0) return 1;
  SYNC();
  return 0;
}
