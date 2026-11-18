// norm_write.cc: seg_normalize.normalize_net for every net (same output order), in C++. Integer-only, exact.
// in: per-net start/count into segs int32 [S,6] (x1,y1,z1,x2,y2,z2 GCells), per-net pin ap range pa0/pa1 into ap int32 [M,3] (l,x,y)
// out (via nw_fetch): per-net offsets and 6 int32 columns in DBU (x*4200+2100), zero-based z  == what blocks_write consumes.
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
static std::vector<int32_t> OC[6]; static std::vector<uint64_t> OOFF;
extern "C" {
long nw_run(long nn, const int64_t* start, const int64_t* cnt, const int32_t* segs, const int64_t* pa0, const int64_t* pa1, const int32_t* ap, int grid) {
  for (auto& v : OC) v.clear(); OOFF.assign(nn + 1, 0);
  const int h = grid / 2;
  struct W { int x1, y1, x2, y2, z; }; struct V { int x, y, zl, zh; };
  std::vector<W> wires; std::vector<V> vias;
  std::vector<std::tuple<int,int,int>> prow, pcol;   // (z,y,x) and (z,x,y)
  for (long i = 0; i < nn; i++) {
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
      OC[0].push_back(x1 * grid + h); OC[1].push_back(y1 * grid + h); OC[2].push_back(z1);
      OC[3].push_back(x2 * grid + h); OC[4].push_back(y2 * grid + h); OC[5].push_back(z2); };
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
    OOFF[i + 1] = OC[0].size();
  }
  return (long)OC[0].size();
}
void nw_fetch(int32_t* c0, int32_t* c1, int32_t* c2, int32_t* c3, int32_t* c4, int32_t* c5, uint64_t* off) {
  int32_t* o[6] = {c0, c1, c2, c3, c4, c5};
  for (int k = 0; k < 6; k++) std::copy(OC[k].begin(), OC[k].end(), o[k]);
  std::copy(OOFF.begin(), OOFF.end(), off);
}
}
