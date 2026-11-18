// n_seg.cc: tools/gapla/csta.py seg_from_route's array on T threads: the same int64 [S,7] rows (route net -> timer net, coordinates // grid, rows of unknown nets dropped). Build: tools/best5/build_n.sh.
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>
extern "C" {
// de[nnet]: timer net of each route net or -1; off[nnet+1] rows per net; x0..z1 int32 [S]. out [kept,7]; returns kept
int64_t n_seg_fill(int64_t nnet, const int64_t* off, const int64_t* de, const int32_t* x0, const int32_t* y0, const int32_t* z0, const int32_t* x1, const int32_t* y1, const int32_t* z1, int64_t grid, int64_t* out, int T, int count_only) {
  std::vector<int64_t> pre(nnet + 1, 0);
  for (int64_t n = 0; n < nnet; ++n) pre[n + 1] = pre[n] + (de[n] >= 0 ? off[n + 1] - off[n] : 0);
  if (count_only) return pre[nnet];
  auto fl = [&](int64_t a) { return a >= 0 ? a / grid : -((-a + grid - 1) / grid); };
  std::atomic<int64_t> nx(0);
  const int64_t CH = 4096;
  auto work = [&]() {
    for (;;) {
      int64_t c = nx.fetch_add(CH);
      if (c >= nnet) return;
      for (int64_t n = c; n < std::min(nnet, c + CH); ++n) {
        if (de[n] < 0) continue;
        int64_t* o = out + 7 * pre[n];
        for (int64_t r = off[n]; r < off[n + 1]; ++r, o += 7) { o[0] = de[n]; o[1] = fl(x0[r]); o[2] = fl(y0[r]); o[3] = z0[r]; o[4] = fl(x1[r]); o[5] = fl(y1[r]); o[6] = z1[r]; }
      }
    }
  };
  std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join();
  return pre[nnet];
}
}
