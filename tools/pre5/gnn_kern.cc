// gnn_kern.cc: forward pass of the learned prior (the learned-start GNN class: input Linear+ReLU
// per node type, 3 x HeteroConv(aggr=sum) of SAGEConv((-1,-1), 64) (mean aggregation, lin_l with bias on the mean of the
// sources, lin_r without bias on the destination), ReLU, output Linear(64,1)) on plain arrays, float32, no torch.
// The result is deterministic and independent of the thread count (each output row is computed by one thread, sums in a
// fixed order: edges of one destination in input order, k = 0..in-1 in every dot product). It is NOT bit-identical to
// torch (MKL sgemm / scatter_reduce order): tools/pre5/gnn_np.py compares both in test mode.
// Edge types (fixed order, = dtla_lagnn.graph / HeteroConv dict order):
//   0 grid -connects-> grid, 1 candidate -passes_through-> grid, 2 grid -influences-> candidate, 3 candidate -competes-> candidate
// Weights (float32, torch layout W[out][in]), packed by gnn_np.pack():
//   proj_g W[H][4] b[H], proj_c W[H][9] b[H], then per layer l, per type t: lin_l W[H][H] b[H], lin_r W[H][H]; out W[1][H] b[1]
// Build (tools/pre5/build_pre5.sh): g++ -std=c++17 -O3 -fno-fast-math [-mavx2 -mfma on x86 hosts that have them] -fopenmp -fPIC -shared
// (FMA contraction allowed here: the prior is not bit-identical to torch anyway; the result is still the same at any thread count)
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>

typedef int64_t i64;

namespace {
struct CSR { std::vector<i64> off; std::vector<i64> src; };

CSR by_dst(i64 n_dst, i64 ne, const i64 *src, const i64 *dst) {
  CSR c; c.off.assign(n_dst + 1, 0); c.src.resize(ne);
  for (i64 e = 0; e < ne; e++) c.off[dst[e] + 1]++;
  for (i64 i = 0; i < n_dst; i++) c.off[i + 1] += c.off[i];
  std::vector<i64> pos(c.off.begin(), c.off.end() - 1);
  for (i64 e = 0; e < ne; e++) c.src[pos[dst[e]]++] = src[e];   // stable: input order per destination
  return c;
}

// mean of the source rows (H wide) per destination row; empty -> 0
void mean_agg(const CSR &c, i64 n_dst, const float *xs, int H, float *out) {
#pragma omp parallel for schedule(static, 1024)
  for (i64 i = 0; i < n_dst; i++) {
    float *o = out + i * H;
    for (int j = 0; j < H; j++) o[j] = 0.0f;
    i64 a = c.off[i], b = c.off[i + 1];
    for (i64 e = a; e < b; e++) {
      const float *x = xs + c.src[e] * H;
      for (int j = 0; j < H; j++) o[j] += x[j];
    }
    if (b > a) {
      float cnt = (float)(b - a);
      for (int j = 0; j < H; j++) o[j] = o[j] / cnt;
    }
  }
}

// y[r][j] (+)= b[j] + sum_k x[r][k] * W[j][k]   (WT = W transposed, [in][out]); acc: add to y
void linear(i64 n, int in, int H, const float *x, const float *WT, const float *b, float *y, bool acc) {
#pragma omp parallel for schedule(static, 256)
  for (i64 r = 0; r < n; r++) {
    float t[256];
    for (int j = 0; j < H; j++) t[j] = b ? b[j] : 0.0f;
    const float *xr = x + r * in;
    for (int k = 0; k < in; k++) {
      float xk = xr[k];
      const float *w = WT + (i64)k * H;
      for (int j = 0; j < H; j++) t[j] += xk * w[j];
    }
    float *yr = y + r * H;
    if (acc) for (int j = 0; j < H; j++) yr[j] += t[j];
    else for (int j = 0; j < H; j++) yr[j] = t[j];
  }
}

std::vector<float> transpose(const float *W, int out, int in) {
  std::vector<float> T((size_t)out * in);
  for (int j = 0; j < out; j++) for (int k = 0; k < in; k++) T[(size_t)k * out + j] = W[(size_t)j * in + k];
  return T;
}

void relu(float *x, i64 n) {
#pragma omp parallel for schedule(static, 65536)
  for (i64 i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}
}  // namespace

extern "C" {

// returns 0; st[0] = seconds, st[1] = threads used
int gnn_forward(i64 ng, i64 nc, const float *xg, const float *xc,
                i64 ne0, const i64 *s0, const i64 *d0, i64 ne1, const i64 *s1, const i64 *d1,
                i64 ne2, const i64 *s2, const i64 *d2, i64 ne3, const i64 *s3, const i64 *d3,
                const float *W, int H, int nlayers, int gin, int cin, int nthreads, float *out, double *st) {
  auto t0 = std::chrono::steady_clock::now();
  if (nthreads > 0) omp_set_num_threads(nthreads);
  if (H > 256) return 2;
  const float *p = W;
  auto take = [&](size_t n) { const float *q = p; p += n; return q; };
  const float *Wpg = take((size_t)H * gin), *bpg = take(H), *Wpc = take((size_t)H * cin), *bpc = take(H);
  i64 nd[4] = {ng, ng, nc, nc}, nsrc[4] = {ng, nc, ng, nc};
  const i64 *S[4] = {s0, s1, s2, s3}, *D[4] = {d0, d1, d2, d3};
  i64 NE[4] = {ne0, ne1, ne2, ne3};
  (void)nsrc;
  CSR C[4];
#pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < 4; t++) C[t] = by_dst(nd[t], NE[t], S[t], D[t]);
  std::vector<float> hg((size_t)ng * H), hc((size_t)nc * H), ng_((size_t)ng * H), nc_((size_t)nc * H);
  std::vector<float> m((size_t)(ng > nc ? ng : nc) * H);
  { auto T = transpose(Wpg, H, gin); linear(ng, gin, H, xg, T.data(), bpg, hg.data(), false); relu(hg.data(), (i64)ng * H); }
  { auto T = transpose(Wpc, H, cin); linear(nc, cin, H, xc, T.data(), bpc, hc.data(), false); relu(hc.data(), (i64)nc * H); }
  for (int l = 0; l < nlayers; l++) {
    const float *Wl[4], *bl[4], *Wr[4];
    for (int t = 0; t < 4; t++) { Wl[t] = take((size_t)H * H); bl[t] = take(H); Wr[t] = take((size_t)H * H); }
    // destination grid: type 0 (src grid) + type 1 (src candidate); candidate: type 2 (src grid) + type 3 (src candidate)
    for (int t = 0; t < 4; t++) {
      const float *xs = (t == 0 || t == 2) ? hg.data() : hc.data();
      const float *xd = (t < 2) ? hg.data() : hc.data();
      float *y = (t < 2) ? ng_.data() : nc_.data();
      i64 n = nd[t];
      mean_agg(C[t], n, xs, H, m.data());
      auto TL = transpose(Wl[t], H, H), TR = transpose(Wr[t], H, H);
      // conv_t = lin_l(mean) + lin_r(x_dst); destination sum: first type assigns, second adds (as stack().sum(0) of 2)
      std::vector<float> tmp((size_t)n * H);
      linear(n, H, H, m.data(), TL.data(), bl[t], tmp.data(), false);
      linear(n, H, H, xd, TR.data(), nullptr, tmp.data(), true);
      if (t == 0 || t == 2) std::memcpy(y, tmp.data(), sizeof(float) * (size_t)n * H);
      else {
#pragma omp parallel for schedule(static, 65536)
        for (i64 i = 0; i < n * H; i++) y[i] += tmp[i];
      }
    }
    hg.swap(ng_); hc.swap(nc_);
    relu(hg.data(), (i64)ng * H); relu(hc.data(), (i64)nc * H);
  }
  const float *Wo = take(H), *bo = take(1);
#pragma omp parallel for schedule(static, 4096)
  for (i64 r = 0; r < nc; r++) {
    float s = bo[0];
    const float *x = hc.data() + r * H;
    for (int k = 0; k < H; k++) s += x[k] * Wo[k];
    out[r] = s;
  }
  st[0] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  st[1] = omp_get_max_threads();
  return 0;
}
}
