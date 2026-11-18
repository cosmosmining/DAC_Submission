// gnn_kern.cu: the end-to-end network of an internal note (tools/e2e25/e2elib.GNN: input Linear + ReLU per node type, 3 x (per destination type the sum of two
// SAGE convolutions: lin_l(mean of the sources) + lin_r(destination), ReLU), output Linear(H, 1)) as kernels on plain arrays, float32, no torch; and its inputs and graph built on threads from the descent's own tables
// (tools/best7/dgb.py cells, dg_kern's gradient at the one-hot start). Same result up to rounding as torch (test: tools/best8/test_fwd.py).
// Same source, two builds:
//   GPU: nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -fPIC -shared gnn_kern.cu -lcublas -o libgnn_cuda.so     (H100: -arch=sm_90): the linear layers through cublasSgemm (a per-row kernel loop was 6 x slower than torch on the T4)
//   CPU: g++ -O3 -std=c++17 -fno-fast-math -fPIC -shared -pthread -x c++ gnn_kern.cu -o libgnn_cpu.so
// Edge types: 0 grid-connects->grid, 1 candidate-passes_through->grid, 2 grid-influences->candidate, 3 candidate-competes->candidate.
// Weights file (tools/best8/export_w.py): int32 H, layers, gin, cin, then float32: proj_g W[H][gin] b[H], proj_c W[H][cin] b[H], per layer, per type: lin_l W[H][H] b[H], lin_r W[H][H]; out W[H] b[1].
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
typedef int64_t i64;
typedef int32_t i32;
typedef unsigned char u8;
static const int HMAX = 64;
static inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int G_T = 8;
static double G_SM = 1.0; static int G_X0 = 0;    // inputs at a cut of the nets: multiplier of the gradient's scale, background demand column zero
template <class Fn> static void par(i64 n, i64 chunk, Fn f) {
  std::atomic<i64> nx(0);
  auto w = [&]() { for (;;) { i64 a = nx.fetch_add(chunk); if (a >= n) return; i64 b = std::min(n, a + chunk); for (i64 i = a; i < b; ++i) f(i); } };
  std::vector<std::thread> th; for (int t = 0; t + 1 < G_T; ++t) th.emplace_back(w); w(); for (auto& x : th) x.join();
}
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define HD __host__ __device__
template <class Fn> __global__ void k_for(Fn f, i64 n) { i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (t < n) f(t); }
template <class Fn> static inline void RUNF(const Fn& f, i64 n, int bs = 64) { if (n > 0) k_for<Fn><<<(unsigned)((n + bs - 1) / bs), (unsigned)bs>>>(f, n); }
static inline void SYNC() { cudaDeviceSynchronize(); }
static void* d_alloc(i64 n) { void* p = nullptr; if (cudaMalloc(&p, (size_t)(n > 0 ? n : 8)) != cudaSuccess) { fprintf(stderr, "gnn: alloc %lld failed\n", (long long)n); return nullptr; } return p; }
static void d_free(void* p) { if (p) cudaFree(p); }
static void d_put(void* d, const void* h, i64 n) { if (n > 0) cudaMemcpy(d, h, (size_t)n, cudaMemcpyHostToDevice); }
static void d_get(void* h, const void* d, i64 n) { if (n > 0) cudaMemcpy(h, d, (size_t)n, cudaMemcpyDeviceToHost); }
#else
#define HD
template <class Fn> static inline void RUNF(const Fn& f, i64 n, int bs = 64) { (void)bs; par(n, 4096, [&](i64 i) { f(i); }); }
static inline void SYNC() {}
static void* d_alloc(i64 n) { return malloc((size_t)(n > 0 ? n : 8)); }
static void d_free(void* p) { free(p); }
static void d_put(void* d, const void* h, i64 n) { if (n > 0) memcpy(d, h, (size_t)n); }
static void d_get(void* h, const void* d, i64 n) { if (n > 0) memcpy(h, d, (size_t)n); }
#endif

// one thread per (row, column), H = 64 threads per block: a warp reads neighbouring floats of one row (a per-row thread with an inner loop over the columns was 30 x slower on the T4)
struct K_mean {         // m[i] = mean of xs over the sources of i (0 without sources)
  const i32* off; const i32* src; const float* xs; float* m;
  HD void operator()(i64 t) const {
    const i64 i = t >> 6; const int j = (int)(t & 63); const i32 a = off[i], b = off[i + 1];
    float v = 0;
    for (i32 e = a; e < b; e++) v += xs[((i64)src[e] << 6) + j];
    m[t] = b > a ? v / (float)(b - a) : (float)0;
  }
};
struct K_fin {          // y[i] = relu(y[i] + b1 + b2)
  float* y; const float* b1; const float* b2;
  HD void operator()(i64 t) const { const int j = (int)(t & 63); const float v = y[t] + b1[j] + (b2 ? b2[j] : (float)0); y[t] = v > 0 ? v : (float)0; }
};
struct K_add1 {         // z[i] += b
  float* z; float b;
  HD void operator()(i64 i) const { z[i] += b; }
};
#ifdef __CUDACC__
static cublasHandle_t CB = nullptr;
// Y[n][H] = X[n][in] W^T (W stored [H][in]), + beta Y
static void lin(i64 n, int in, int H, const float* W, const float* X, float beta, float* Y) { if (n <= 0) return; const float one = 1; cublasSgemm(CB, CUBLAS_OP_T, CUBLAS_OP_N, H, (int)n, in, &one, W, in, X, in, &beta, Y, H); }
#else
static void lin(i64 n, int in, int H, const float* W, const float* X, float beta, float* Y) {
  par(n, 256, [&](i64 r) { const float* xr = X + r * in; float* yr = Y + r * H; for (int j = 0; j < H; j++) { float s = 0; const float* w = W + (i64)j * in; for (int k = 0; k < in; k++) s += xr[k] * w[k]; yr[j] = beta != 0 ? yr[j] + s : s; } });
}
#endif

namespace {
struct Net { int H = 0, layers = 0, gin = 0, cin = 0; i64 nw = 0; float* W = nullptr; };      // weights on the device
struct In {            // the last inputs (host)
  i64 R = 0, K = 0, nc = 0, ng = 0, ng_full = 0; double scale = 0;
  std::vector<i32> keep;      // kept cell -> cell of the tables (cells without a candidate have no edge: left out)
  std::vector<i32> sid, off0, src0, off1, src1, off2, src2, off3, src3, cand_r; std::vector<float> xc, xg, z;
  std::vector<u8> cand_k;
} GI;
}  // namespace

extern "C" {
void gnn_threads(int T) { G_T = T > 0 ? T : 1; }
void gnn_opts(double scale_mult, int zero_d0) { G_SM = scale_mult; G_X0 = zero_d0; }
// weights -> device; returns a handle (nullptr on failure)
void* gnn_load(const char* path) {
  FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "gnn: cannot open %s\n", path); return nullptr; }
  i32 hd[4]; if (fread(hd, 4, 4, f) != 4) { fclose(f); return nullptr; }
#ifdef __CUDACC__
  if (!CB && cublasCreate(&CB) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "gnn: cublasCreate failed\n"); fclose(f); return nullptr; }
#endif
  Net* n = new Net; n->H = hd[0]; n->layers = hd[1]; n->gin = hd[2]; n->cin = hd[3];
  if (n->H != HMAX) { fprintf(stderr, "gnn: H %d, only %d is built\n", n->H, HMAX); fclose(f); delete n; return nullptr; }
  i64 H = n->H;
  n->nw = H * n->gin + H + H * n->cin + H + n->layers * 4 * (H * H + H + H * H) + H + 1;
  std::vector<float> w((size_t)n->nw);
  size_t got = fread(w.data(), 4, (size_t)n->nw, f); fclose(f);
  if ((i64)got != n->nw) { fprintf(stderr, "gnn: %s: %lld floats, %lld wanted\n", path, (long long)got, (long long)n->nw); delete n; return nullptr; }
  n->W = (float*)d_alloc(n->nw * 4); if (!n->W) { delete n; return nullptr; }
  d_put(n->W, w.data(), n->nw * 4);
  return n;
}
void gnn_unload(void* h) { Net* n = (Net*)h; if (n) { d_free(n->W); delete n; } }

// inputs + graph, from the descent's tables. run[R,8]: k d fixed lo hi cur p0 p1. cand2[2,K]: layer or -1. ent_ptr[R*K+1], ec[E]: cells of a candidate (dgb.cells). cp[ng+1], ce[E]: entries of a cell (R*K index). cvp[ng+1], cvi[V] (point * L + z), cvg[V] (e^{alpha w} - 1).
// kpn[R]: the net's power weight, Cz[L], um: dtla0's; d0, cap, uu[ng], cell_z[ng]: background demand, capacity, unit cost, layer of the cells; copen[ng]; dv[npt*L]: via demand at the one-hot start; g[R*K]: dF/dp at the one-hot start.
// counts: nc, E0, E1, E2, E3; *scale: the median of |g - g_cur| over the candidates that are not current.
int gnn_inputs(i64 R, i64 K, i64 L, const i64* run, const i32* cand2, const i32* ent_ptr, const i32* ec, i64 ng, const i32* cp, const i32* ce, const i32* cvp, const i32* cvi, const float* cvg, const double* kpn, const double* Cz, double um,
               const float* d0, const float* cap, const float* uu, const i32* cell_z, const u8* copen, const float* dv, const float* g, i64* counts, double* scale) {
  In& G = GI; G.R = R; G.K = K; G.ng_full = ng;
  G.sid.assign((size_t)(R * K), -1); G.cand_r.clear(); G.cand_k.clear();
  i64 nc = 0;
  { std::vector<i32> o((size_t)(R + 1)); for (i64 r = 0; r < R; r++) { int d = (int)run[r * 8 + 1]; o[r] = (i32)nc; for (i64 k = 0; k < K; k++) if (cand2[d * K + k] >= 0) G.sid[r * K + k] = (i32)nc++; } o[R] = (i32)nc;
    G.nc = nc; G.cand_r.resize(nc); G.cand_k.resize(nc);
    par(R, 2048, [&](i64 r) { int d = (int)run[r * 8 + 1]; for (i64 k = 0; k < K; k++) { i32 s = G.sid[r * K + k]; if (s >= 0) { G.cand_r[s] = (i32)r; G.cand_k[s] = (u8)k; } } }); }
  // the gradient input: g minus the run's current candidate, arcsinh of the design's median
  std::vector<double> dg((size_t)nc);
  { std::vector<float> av; av.reserve((size_t)nc);
    for (i64 i = 0; i < nc; i++) { i64 r = G.cand_r[i], k = G.cand_k[i], cur = run[r * 8 + 5]; double v = (double)g[r * K + k] - (double)g[r * K + cur]; dg[i] = v; if (k != cur) av.push_back((float)fabs(v)); }
    double m = 1e-30;
    if (!av.empty()) { size_t q = (av.size() - 1) / 2; std::nth_element(av.begin(), av.begin() + q, av.end()); m = (double)av[q]; if (m < 1e-30) m = 1e-30; }
    else m = 1.0;
    m *= G_SM; G.scale = m; *scale = m; }
  G.xc.resize((size_t)nc * 10);
  par(nc, 4096, [&](i64 i) {
    i64 r = G.cand_r[i], k = G.cand_k[i]; const i64* w = run + r * 8; int d = (int)w[1], z = cand2[d * K + k]; double ln = (double)(w[4] - w[3]);
    float* x = G.xc.data() + i * 10;
    x[0] = (float)log1p(ln); x[1] = (float)((double)z / (double)L); x[2] = k == w[5] ? 1.0f : 0.0f; x[3] = 0; x[4] = 0;
    x[5] = (float)log1p(Cz[z] * ln * um); x[6] = 0; x[7] = 0; x[8] = (float)(log10(fabs(kpn[r]) + 1e-14) / 14.0 + 1.0); x[9] = (float)asinh(dg[i] / G.scale);
  });
  G.keep.clear(); std::vector<i32> newid((size_t)ng, -1);
  for (i64 i = 0; i < ng; i++) if (cp[i + 1] > cp[i]) { newid[i] = (i32)G.keep.size(); G.keep.push_back((i32)i); }
  const i64 ngk = (i64)G.keep.size(); G.ng = ngk;
  G.xg.resize((size_t)ngk * 5);
  par(ngk, 4096, [&](i64 q) {
    const i64 i = G.keep[q];
    double dem = (double)d0[i], c = std::max((double)cap[i], 1e-3);
    float* x = G.xg.data() + q * 5;
    x[0] = G_X0 ? 0.0f : (float)std::min(std::max(dem / c, 0.0), 10.0); x[1] = (float)log1p((double)cap[i]); x[2] = uu[i]; x[3] = (float)((double)cell_z[i] / (double)L);
    for (i32 e = cp[i]; e < cp[i + 1]; e++) { i64 rk = ce[e], r = rk / K, k = rk % K; if (run[r * 8 + 5] == k) dem += 2.0; }
    const double al = copen[i] ? 0.25 : 0.75;
    for (i32 v = cvp[i]; v < cvp[i + 1]; v++) { float pv = dv[cvi[v]]; if (pv > 0) dem += std::round(log1p((double)cvg[v]) / al) * (double)pv; }
    x[4] = (float)std::min(std::max(dem / c, 0.0), 10.0);
  });
  // graph: 2 = a candidate's cells, 1 = a cell's candidates, 0 = neighbours along a run (at most prev and next), 3 = the other candidates of the run
  G.off2.resize((size_t)nc + 1); G.src2.resize(ent_ptr[R * K]);
  par(ent_ptr[R * K], 65536, [&](i64 q) { G.src2[q] = newid[ec[q]]; });
  for (i64 i = 0; i < nc; i++) G.off2[i] = ent_ptr[G.cand_r[i] * K + G.cand_k[i]];
  G.off2[nc] = ent_ptr[R * K];
  G.off1.resize((size_t)ngk + 1); G.off1[0] = 0; G.src1.resize(G.off2[nc]);
  for (i64 q = 0; q < ngk; q++) G.off1[q + 1] = G.off1[q] + (cp[G.keep[q] + 1] - cp[G.keep[q]]);
  par(ngk, 4096, [&](i64 q) { const i64 i = G.keep[q]; i32 o = G.off1[q]; for (i32 e = cp[i]; e < cp[i + 1]; e++) G.src1[o++] = G.sid[ce[e]]; });
  std::vector<i32> nxt((size_t)ngk, -1), prv((size_t)ngk, -1);
  par(nc, 512, [&](i64 i) { for (i32 q = G.off2[i]; q + 1 < G.off2[i + 1]; q++) { i32 a = G.src2[q], b = G.src2[q + 1]; __atomic_store_n(&nxt[a], b, __ATOMIC_RELAXED); __atomic_store_n(&prv[b], a, __ATOMIC_RELAXED); } });
  G.off0.resize((size_t)ngk + 1); G.off0[0] = 0;
  for (i64 i = 0; i < ngk; i++) G.off0[i + 1] = G.off0[i] + (prv[i] >= 0) + (nxt[i] >= 0);
  G.src0.resize(G.off0[ngk]);
  par(ngk, 4096, [&](i64 i) { i32 q = G.off0[i]; if (prv[i] >= 0) G.src0[q++] = prv[i]; if (nxt[i] >= 0) G.src0[q++] = nxt[i]; });
  G.off3.resize((size_t)nc + 1); G.off3[0] = 0;
  std::vector<i32> nv((size_t)R, 0);
  for (i64 i = 0; i < nc; i++) nv[G.cand_r[i]]++;
  for (i64 i = 0; i < nc; i++) G.off3[i + 1] = G.off3[i] + nv[G.cand_r[i]] - 1;
  G.src3.resize(G.off3[nc]);
  par(nc, 4096, [&](i64 i) { i64 r = G.cand_r[i]; i32 q = G.off3[i]; for (i64 k = 0; k < K; k++) { i32 s = G.sid[r * K + k]; if (s >= 0 && s != (i32)i) G.src3[q++] = s; } });
  counts[0] = nc; counts[1] = G.off0[ngk]; counts[2] = G.off1[ngk]; counts[3] = G.off2[nc]; counts[4] = G.off3[nc]; counts[5] = ngk;
  return 0;
}
// the inputs as the tests read them: any pointer may be null. off/src: arrays in the order 0, 1, 2, 3
void gnn_inputs_get(i32* keep, float* xg, float* xc, i32* sid, i32* off0, i32* src0, i32* off1, i32* src1, i32* off2, i32* src2, i32* off3, i32* src3) {
  In& G = GI;
  auto cp = [](auto* d, const auto& v) { if (d && !v.empty()) memcpy(d, v.data(), v.size() * sizeof(v[0])); };
  cp(keep, G.keep); cp(xg, G.xg); cp(xc, G.xc); cp(sid, G.sid); cp(off0, G.off0); cp(src0, G.src0); cp(off1, G.off1); cp(src1, G.src1); cp(off2, G.off2); cp(src2, G.src2); cp(off3, G.off3); cp(src3, G.src3);
}
// inputs given (tests: E2ET's arrays): replaces the last inputs. offN[n_dst + 1], srcN[EN]
void gnn_inputs_set(i64 ng, i64 nc, const float* xg, const float* xc, const i32* off0, const i32* src0, const i32* off1, const i32* src1, const i32* off2, const i32* src2, const i32* off3, const i32* src3) {
  In& G = GI; G.ng = ng; G.nc = nc;
  G.xg.assign(xg, xg + ng * 5); G.xc.assign(xc, xc + nc * 10);
  G.off0.assign(off0, off0 + ng + 1); G.src0.assign(src0, src0 + off0[ng]); G.off1.assign(off1, off1 + ng + 1); G.src1.assign(src1, src1 + off1[ng]);
  G.off2.assign(off2, off2 + nc + 1); G.src2.assign(src2, src2 + off2[nc]); G.off3.assign(off3, off3 + nc + 1); G.src3.assign(src3, src3 + off3[nc]);
}
// forward on the last inputs: out[nc] (host; may be null: kept for gnn_logits). st[0] upload s, st[1] kernels s, st[2] download s
int gnn_forward(void* h, float* out, double* st) {
  Net* n = (Net*)h; In& G = GI; const int H = n->H; const i64 ng = G.ng, nc = G.nc;
  double t0 = now_s();
  auto up = [&](const auto& v) { void* d = d_alloc((i64)(v.size() * sizeof(v[0]))); if (d) d_put(d, v.data(), (i64)(v.size() * sizeof(v[0]))); return d; };
  i32* o0 = (i32*)up(G.off0); i32* s0 = (i32*)up(G.src0); i32* o1 = (i32*)up(G.off1); i32* s1 = (i32*)up(G.src1);
  i32* o2 = (i32*)up(G.off2); i32* s2 = (i32*)up(G.src2); i32* o3 = (i32*)up(G.off3); i32* s3 = (i32*)up(G.src3);
  float* xg = (float*)up(G.xg); float* xc = (float*)up(G.xc);
  float* hg = (float*)d_alloc(ng * H * 4); float* hg2 = (float*)d_alloc(ng * H * 4); float* hc = (float*)d_alloc(nc * H * 4); float* hc2 = (float*)d_alloc(nc * H * 4); float* z = (float*)d_alloc(nc * 4);
  if (!o0 || !s0 || !o1 || !s1 || !o2 || !s2 || !o3 || !s3 || !xg || !xc || !hg || !hg2 || !hc || !hc2 || !z) return 2;
  double t1 = now_s();
  const float* p = n->W;
  auto take = [&](i64 m) { const float* q = p; p += m; return q; };
  const float* Wg = take((i64)H * n->gin); const float* bg = take(H); const float* Wc = take((i64)H * n->cin); const float* bc = take(H);
  lin(ng, n->gin, H, Wg, xg, 0.0f, hg); RUNF(K_fin{hg, bg, nullptr}, ng * H, H);
  lin(nc, n->cin, H, Wc, xc, 0.0f, hc); RUNF(K_fin{hc, bc, nullptr}, nc * H, H);
  float* mg = (float*)d_alloc(ng * H * 4); float* mc = (float*)d_alloc(nc * H * 4);
  if (!mg || !mc) return 2;
  const bool prof = getenv("GNN_PROF") != nullptr; double tm = 0, tl = 0, tf = 0, tt = now_s();
  auto tick = [&](double& a) { if (prof) { SYNC(); double nn = now_s(); a += nn - tt; tt = nn; } };
  for (int l = 0; l < n->layers; l++) {
    const float* Wl[4]; const float* bl[4]; const float* Wr[4];
    for (int t = 0; t < 4; t++) { Wl[t] = take((i64)H * H); bl[t] = take(H); Wr[t] = take((i64)H * H); }
    RUNF(K_mean{o0, s0, hg, mg}, ng * H, H); tick(tm); lin(ng, H, H, Wl[0], mg, 0.0f, hg2); lin(ng, H, H, Wr[0], hg, 1.0f, hg2); tick(tl);
    RUNF(K_mean{o1, s1, hc, mg}, ng * H, H); tick(tm); lin(ng, H, H, Wl[1], mg, 1.0f, hg2); lin(ng, H, H, Wr[1], hg, 1.0f, hg2); tick(tl);
    RUNF(K_fin{hg2, bl[0], bl[1]}, ng * H, H); tick(tf);
    RUNF(K_mean{o2, s2, hg, mc}, nc * H, H); tick(tm); lin(nc, H, H, Wl[2], mc, 0.0f, hc2); lin(nc, H, H, Wr[2], hc, 1.0f, hc2); tick(tl);
    RUNF(K_mean{o3, s3, hc, mc}, nc * H, H); tick(tm); lin(nc, H, H, Wl[3], mc, 1.0f, hc2); lin(nc, H, H, Wr[3], hc, 1.0f, hc2); tick(tl);
    RUNF(K_fin{hc2, bl[2], bl[3]}, nc * H, H); tick(tf);
    std::swap(hg, hg2); std::swap(hc, hc2);
  }
  if (prof) fprintf(stderr, "gnn_prof mean %.3f lin %.3f fin %.3f ng %lld nc %lld\n", tm, tl, tf, (long long)ng, (long long)nc);
  d_free(mg); d_free(mc);
  const float* Wo = take(H); const float* bo = take(1);
  { float bv = 0; d_get(&bv, bo, 4); lin(nc, H, 1, Wo, hc, 0.0f, z); RUNF(K_add1{z, bv}, nc); }
  SYNC();
  double t2 = now_s();
  G.z.resize((size_t)nc);
  d_get(G.z.data(), z, nc * 4);
  if (out) memcpy(out, G.z.data(), (size_t)nc * 4);
  double t3 = now_s();
  void* all[] = {o0, s0, o1, s1, o2, s2, o3, s3, xg, xc, hg, hg2, hc, hc2, z};
  for (void* a : all) d_free(a);
  st[0] = t1 - t0; st[1] = t2 - t1; st[2] = t3 - t2;
  return 0;
}
// start logits [R*K] from the last forward: init x one-hot(current) + network, centred per run, clamped to [-c, c] when clamp_c > 0, 0 on invalid candidates (E2ET's gk_t)
void gnn_logits(const i64* run, double init, double clamp_c, float* x) {
  In& G = GI; const i64 R = G.R, K = G.K;
  par(R, 2048, [&](i64 r) {
    float* xr = x + r * K; float s = 0; int nv = 0; const i64 cur = run[r * 8 + 5];
    for (i64 k = 0; k < K; k++) { i32 q = G.sid[r * K + k]; if (q >= 0) { float v = (k == cur ? (float)init : 0.0f) + G.z[q]; xr[k] = v; s += v; nv++; } else xr[k] = 0; }
    if (nv) { const float mean = s / (float)nv;
      for (i64 k = 0; k < K; k++) if (G.sid[r * K + k] >= 0) { float v = xr[k] - mean; if (clamp_c > 0) { if (v > (float)clamp_c) v = (float)clamp_c; if (v < -(float)clamp_c) v = -(float)clamp_c; } xr[k] = v; } }
  });
}
}  // extern "C"
