// t_fpk.cc: small exact kernels of tools/best10/t_fp.py. std::thread only. Build: tools/best10/build_t.sh (no -ffast-math).
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
typedef int64_t i64;
template <class F> static void pfor(i64 n, i64 chunk, int T, F f) {
  std::atomic<i64> nx(0);
  auto w = [&]() { for (;;) { i64 k = nx.fetch_add(chunk); if (k >= n) return; i64 e = k + chunk < n ? k + chunk : n; for (i64 q = k; q < e; ++q) f(q); } };
  if (T <= 1 || n <= chunk) { w(); return; }
  std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(w); w(); for (auto& x : th) x.join();
}
extern "C" {
// keep[k] = 0 when the rows the round gave net changed[k] equal its current rows (byte for byte, same order): segs[off[k]..off[k+1]) against ptr[i] / cnt[i] (cnt < 0: base rows barr[boff[i]..boff[i+1]))
void tf_same(i64 nc, const i64* changed, const i64* off, const i64* segs, const i64* ptr, const i64* cnt, const i64* boff, const i64* barr, unsigned char* keep, int T) {
  pfor(nc, 64, T, [&](i64 k) {
    i64 i = changed[k], m = off[k + 1] - off[k];
    const i64* cur; i64 mc;
    if (cnt[i] >= 0) { cur = reinterpret_cast<const i64*>(ptr[i]); mc = cnt[i]; } else { cur = barr + 6 * boff[i]; mc = boff[i + 1] - boff[i]; }
    keep[k] = !(m == mc && (m == 0 || std::memcmp(segs + 6 * off[k], cur, (size_t)m * 48) == 0));
  });
}
// a6[n,6] = (x0 // G, y0 // G, z0, x1 // G, y1 // G, z1) of int32 columns (floor division, as numpy on int64)
static inline i64 fdiv(i64 a, i64 b) { i64 q = a / b; return ((a % b != 0) && ((a < 0) != (b < 0))) ? q - 1 : q; }
void tf_a6(i64 n, const int32_t* x0, const int32_t* y0, const int32_t* z0, const int32_t* x1, const int32_t* y1, const int32_t* z1, i64 G, i64* a6, int T) {
  pfor(n, 1 << 14, T, [&](i64 q) {
    i64* o = a6 + 6 * q;
    o[0] = fdiv(x0[q], G); o[1] = fdiv(y0[q], G); o[2] = z0[q]; o[3] = fdiv(x1[q], G); o[4] = fdiv(y1[q], G); o[5] = z1[q];
  });
}
// base[S,7] = (net, a6 row): np.concatenate([np.repeat(np.arange(n), diff(off))[:, None], a6], 1)
void tf_base(i64 n, const i64* off, const i64* a6, i64* base, int T) {
  pfor(n, 1 << 10, T, [&](i64 i) {
    for (i64 r = off[i]; r < off[i + 1]; ++r) { i64* o = base + 7 * r; o[0] = i; std::memcpy(o + 1, a6 + 6 * r, 48); }
  });
}
// the kept nets' rows of the assembly: cols[k][noff[i]..] = src[k][off[i]..] for the nets kn[]
void tf_keep(i64 m, const i64* kn, const i64* cnt, const i64* off, const i64* noff, const int32_t** src, int32_t** dst, int T) {
  pfor(m, 256, T, [&](i64 q) {
    i64 i = kn[q], c = cnt[i];
    for (int k = 0; k < 6; ++k) std::memcpy(dst[k] + noff[i], src[k] + off[i], (size_t)c * 4);
  });
}
}
