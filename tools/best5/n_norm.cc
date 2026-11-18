// n_norm.cc: tools/c1hfast/norm_write.cc nw_run (seg_normalize.normalize_net per net, integer only) on T threads, same rows in the same order; the
// output block [m,7] int64 (route net, x1, y1, z1, x2, y2, z2) written directly (tools/late5/l5.py normalise builds it from the 6 columns). Build: tools/best5/build_n.sh.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <tuple>
#include <vector>
namespace {
struct Chunk { std::vector<int32_t> r; std::vector<int64_t> ends; };
std::vector<Chunk> CH; const int64_t NCH = 64; long NN = 0; const int64_t* IDS = nullptr;
}
extern "C" {
long n_nw_run(long nn, const int64_t* start, const int64_t* cnt, const int32_t* segs, const int64_t* pa0, const int64_t* pa1, const int32_t* ap, int grid, int T) {
  const int64_t nch = (nn + NCH - 1) / NCH;
  CH.assign(nch, Chunk());
  NN = nn;
  const int h = grid / 2;
  std::atomic<int64_t> nx(0);
  auto work = [&]() {
    struct W { int x1, y1, x2, y2, z; }; struct V { int x, y, zl, zh; };
    std::vector<W> wires; std::vector<V> vias;
    std::vector<std::tuple<int, int, int>> prow, pcol;
    for (;;) {
      int64_t c = nx.fetch_add(1);
      if (c >= nch) return;
      Chunk& C = CH[c];
      for (long i = c * NCH; i < std::min<long>(nn, (c + 1) * NCH); i++) {
        wires.clear(); vias.clear(); prow.clear(); pcol.clear();
        for (int64_t k = 0; k < cnt[i]; k++) {
          const int32_t* s = segs + 6 * (start[i] + k);
          int x1 = s[0], y1 = s[1], z1 = s[2], x2 = s[3], y2 = s[4], z2 = s[5];
          if (z1 != z2) vias.push_back({x1, y1, std::min(z1, z2), std::max(z1, z2)});
          else if (x1 != x2 || y1 != y2) wires.push_back({x1, y1, x2, y2, z1});
        }
        auto addp = [&](int x, int y, int z) { prow.emplace_back(z, y, x); pcol.emplace_back(z, x, y); };
        for (int64_t a = pa0[i]; a < pa1[i]; a++) addp(ap[3 * a + 1], ap[3 * a + 2], ap[3 * a]);
        for (auto& w : wires) { addp(w.x1, w.y1, w.z); addp(w.x2, w.y2, w.z); }
        for (auto& v : vias) for (int z = v.zl; z <= v.zh; z++) addp(v.x, v.y, z);
        std::sort(prow.begin(), prow.end()); prow.erase(std::unique(prow.begin(), prow.end()), prow.end());
        std::sort(pcol.begin(), pcol.end()); pcol.erase(std::unique(pcol.begin(), pcol.end()), pcol.end());
        auto emit = [&](int x1, int y1, int z1, int x2, int y2, int z2) {
          C.r.push_back(x1 * grid + h); C.r.push_back(y1 * grid + h); C.r.push_back(z1);
          C.r.push_back(x2 * grid + h); C.r.push_back(y2 * grid + h); C.r.push_back(z2); };
        for (auto& w : wires) {
          if (w.y1 == w.y2) {
            int lo = std::min(w.x1, w.x2), hi = std::max(w.x1, w.x2);
            auto it = std::upper_bound(prow.begin(), prow.end(), std::make_tuple(w.z, w.y1, lo));
            int prev = lo;
            for (; it != prow.end() && std::get<0>(*it) == w.z && std::get<1>(*it) == w.y1 && std::get<2>(*it) < hi; ++it) { emit(prev, w.y1, w.z, std::get<2>(*it), w.y1, w.z); prev = std::get<2>(*it); }
            emit(prev, w.y1, w.z, hi, w.y1, w.z);
          } else if (w.x1 == w.x2) {
            int lo = std::min(w.y1, w.y2), hi = std::max(w.y1, w.y2);
            auto it = std::upper_bound(pcol.begin(), pcol.end(), std::make_tuple(w.z, w.x1, lo));
            int prev = lo;
            for (; it != pcol.end() && std::get<0>(*it) == w.z && std::get<1>(*it) == w.x1 && std::get<2>(*it) < hi; ++it) { emit(w.x1, prev, w.z, w.x1, std::get<2>(*it), w.z); prev = std::get<2>(*it); }
            emit(w.x1, prev, w.z, w.x1, hi, w.z);
          } else emit(w.x1, w.y1, w.z, w.x2, w.y2, w.z);
        }
        for (auto& v : vias) emit(v.x, v.y, v.zl, v.x, v.y, v.zh);
        C.ends.push_back((int64_t)(C.r.size() / 6));
      }
    }
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join(); }
  int64_t tot = 0;
  for (auto& C : CH) tot += (int64_t)(C.r.size() / 6);
  return (long)tot;
}
// blk [tot,7] int64, off [nn+1]; ids[nn] = route net of each listed net
void n_nw_fetch(const int64_t* ids, int64_t* blk, int64_t* off, int T) {
  const int64_t nch = (int64_t)CH.size();
  std::vector<int64_t> base(nch + 1, 0);
  for (int64_t c = 0; c < nch; ++c) base[c + 1] = base[c] + (int64_t)(CH[c].r.size() / 6);
  off[0] = 0;
  for (int64_t c = 0; c < nch; ++c) { int64_t i = c * NCH; for (int64_t e : CH[c].ends) off[++i] = base[c] + e; }
  std::atomic<int64_t> nx(0);
  auto work = [&]() {
    for (;;) {
      int64_t c = nx.fetch_add(1);
      if (c >= nch) return;
      const Chunk& C = CH[c];
      int64_t i = c * NCH, b = 0;
      for (int64_t e : C.ends) {
        for (int64_t k = b; k < e; ++k) {
          int64_t* o = blk + 7 * (base[c] + k); const int32_t* r = C.r.data() + 6 * k;
          o[0] = ids[i]; o[1] = r[0]; o[2] = r[1]; o[3] = r[2]; o[4] = r[3]; o[5] = r[4]; o[6] = r[5];
        }
        b = e; ++i;
      }
    }
  };
  { std::vector<std::thread> th; for (int t = 0; t + 1 < T; ++t) th.emplace_back(work); work(); for (auto& x : th) x.join(); }
  std::vector<Chunk>().swap(CH);
}
}
