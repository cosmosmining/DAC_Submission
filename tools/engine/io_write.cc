// io_write.cc: tools/c1hfast/norm_write.cc (nw_run: seg_normalize.normalize_net of
// every net) + tools/fast_path/io_write.cc (blocks_write: the contest text) fused, parallel over nets, one final write.
// Each thread takes a contiguous range of nets (balanced by segment count), normalises each net with the SAME code as
// nw_run and formats it exactly as blocks_write ("%d" numbers, "metal<z+1>"); the per-thread texts are written in net
// order with one fopen. Same bytes as nw_run + nw_fetch + blocks_write.
// Build (Linux): g++ -O2 -std=c++17 -fPIC -shared -pthread io_write.cc -o libio_write.so      (no -ffast-math)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <tuple>
#include <chrono>
static double g_t[2];                                       // seconds: normalise+format (threads), file write
extern "C" void iow_times(double* o) { o[0] = g_t[0]; o[1] = g_t[1]; }

namespace {
struct W { int x1, y1, x2, y2, z; };
struct V { int x, y, zl, zh; };

inline void put_int(std::string& b, int32_t v) {          // == snprintf("%d")
  char t[16]; int n = 0;
  uint32_t u = v < 0 ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
  do { t[n++] = char('0' + u % 10); u /= 10; } while (u);
  if (v < 0) t[n++] = '-';
  while (n) b.push_back(t[--n]);
}

void work(long a, long b, const int64_t* start, const int64_t* cnt, const int32_t* segs, const int64_t* pa0,
          const int64_t* pa1, const int32_t* ap, int grid, const char* const* nm, const int64_t* nl,
          std::string& out, long& nseg) {
  const int h = grid / 2;
  std::vector<W> wires; std::vector<V> vias;
  std::vector<std::tuple<int,int,int>> prow, pcol;
  nseg = 0;
  auto emit = [&](int x1, int y1, int z1, int x2, int y2, int z2) {
    put_int(out, x1 * grid + h); out.push_back(' '); put_int(out, y1 * grid + h); out.append(" metal"); put_int(out, z1 + 1);
    out.push_back(' ');
    put_int(out, x2 * grid + h); out.push_back(' '); put_int(out, y2 * grid + h); out.append(" metal"); put_int(out, z2 + 1);
    out.push_back('\n'); nseg++;
  };
  for (long i = a; i < b; i++) {
    out.append(nm[i], nl[i]); out.append("\n(\n");
    wires.clear(); vias.clear(); prow.clear(); pcol.clear();
    for (int64_t k = 0; k < cnt[i]; k++) {
      const int32_t* s = segs + 6 * (start[i] + k);
      int x1 = s[0], y1 = s[1], z1 = s[2], x2 = s[3], y2 = s[4], z2 = s[5];
      if (z1 != z2) vias.push_back({x1, y1, std::min(z1, z2), std::max(z1, z2)});
      else if (x1 != x2 || y1 != y2) wires.push_back({x1, y1, x2, y2, z1});
    }
    auto addp = [&](int x, int y, int z) { prow.emplace_back(z, y, x); pcol.emplace_back(z, x, y); };
    for (int64_t q = pa0[i]; q < pa1[i]; q++) addp(ap[3 * q + 1], ap[3 * q + 2], ap[3 * q]);
    for (auto& w : wires) { addp(w.x1, w.y1, w.z); addp(w.x2, w.y2, w.z); }
    for (auto& v : vias) for (int z = v.zl; z <= v.zh; z++) addp(v.x, v.y, z);
    std::sort(prow.begin(), prow.end()); prow.erase(std::unique(prow.begin(), prow.end()), prow.end());
    std::sort(pcol.begin(), pcol.end()); pcol.erase(std::unique(pcol.begin(), pcol.end()), pcol.end());
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
    out.append(")\n");
  }
}
}  // namespace

extern "C" long iow_write(const char* path, const char* names, long nn, const int64_t* start, const int64_t* cnt,
                          const int32_t* segs, const int64_t* pa0, const int64_t* pa1, const int32_t* ap, int grid,
                          int nthreads) {
  std::vector<const char*> nm(nn); std::vector<int64_t> nl(nn);
  const char* p = names;
  for (long i = 0; i < nn; i++) { const char* e = std::strchr(p, '\n'); nm[i] = p; nl[i] = e - p; p = e + 1; }
  int T = std::max(1, nthreads);
  int64_t tot = 0; for (long i = 0; i < nn; i++) tot += cnt[i] + 1;
  std::vector<long> cut(T + 1, nn); cut[0] = 0;
  { int64_t acc = 0; int k = 1;
    for (long i = 0; i < nn && k < T; i++) { acc += cnt[i] + 1; if (acc * T >= tot * k) { cut[k++] = i + 1; } }
    for (; k < T; k++) cut[k] = nn; }
  std::vector<std::string> buf(T); std::vector<long> ns(T, 0);
  for (int t = 0; t < T; t++) buf[t].reserve((size_t)((cut[t + 1] - cut[t]) * 16 + 1024));
  auto c0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (int t = 1; t < T; t++)
    th.emplace_back(work, cut[t], cut[t + 1], start, cnt, segs, pa0, pa1, ap, grid, nm.data(), nl.data(), std::ref(buf[t]), std::ref(ns[t]));
  work(cut[0], cut[1], start, cnt, segs, pa0, pa1, ap, grid, nm.data(), nl.data(), buf[0], ns[0]);
  for (auto& x : th) x.join();
  auto c1 = std::chrono::steady_clock::now();
  FILE* f = std::fopen(path, "wb");
  if (!f) return -1;
  long n = 0;
  for (int t = 0; t < T; t++) { if (!buf[t].empty() && std::fwrite(buf[t].data(), 1, buf[t].size(), f) != buf[t].size()) { std::fclose(f); return -2; } n += ns[t]; }
  if (std::fclose(f) != 0) return -3;
  g_t[0] = std::chrono::duration<double>(c1 - c0).count();
  g_t[1] = std::chrono::duration<double>(std::chrono::steady_clock::now() - c1).count();
  return n;
}
