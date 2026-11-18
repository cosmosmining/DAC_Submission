// h_asm.cc: the per-net loop of chain_g.assemble (io_run.write_fast's arithmetic) in C++.
// For every DP net (given in the order Python's defaultdict(list) would hold them = first appearance in run_net), with its runs
// in increasing run index:
//   wires: (z, f) keys in first-insertion order, each with its (lo, hi) intervals in run order (runs with z < 0 skipped);
//   node layers: (x, y) keys in first-insertion order (run end a, run end b, then the net's net_nodes items in dict order),
//   only min and max of the layer set are used;
//   output: every wire (lo,f,z,hi,f,z) if dirs[z]==0 else (f,lo,z,f,hi,z), then per node (x,y,z,x,y,z+1) for z in [min,max).
// Integer only. Build: tools/fast3x/h_build.sh.
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>

static std::vector<int64_t> OUT; static std::vector<int64_t> CNT;

extern "C" {
// nets[nn] (DP net ids in processing order), roff[nn+1]/runs[]: run indices of each net (increasing),
// z[R] chosen layer per run, dir/fix/lo/hi[R], dirs[L], noff[nn+1]/nodes[*4]: (x, y, s0, s1) per net in dict order
int64_t ha_run(int64_t nn, const int64_t* roff, const int64_t* runs, const int64_t* z, const int64_t* dir, const int64_t* fix,
               const int64_t* lo, const int64_t* hi, const int64_t* dirs, const int64_t* nofs, const int64_t* nodes) {
  OUT.clear(); CNT.assign(nn, 0);
  std::unordered_map<uint64_t, int64_t> wk, nk; wk.reserve(64); nk.reserve(64);
  std::vector<int64_t> wkz, wkf; std::vector<std::vector<std::pair<int64_t, int64_t>>> wiv;
  std::vector<int64_t> nx, ny, nmin, nmax;
  auto pk = [](int64_t a, int64_t b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
  for (int64_t q = 0; q < nn; q++) {
    wk.clear(); nk.clear(); wkz.clear(); wkf.clear(); nx.clear(); ny.clear(); nmin.clear(); nmax.clear();
    size_t nw = 0;
    auto node = [&](int64_t x, int64_t y, int64_t zz) {
      auto it = nk.find(pk(x, y));
      if (it == nk.end()) { nk.emplace(pk(x, y), (int64_t)nx.size()); nx.push_back(x); ny.push_back(y); nmin.push_back(zz); nmax.push_back(zz); }
      else { int64_t j = it->second; nmin[j] = std::min(nmin[j], zz); nmax[j] = std::max(nmax[j], zz); }
    };
    for (int64_t t = roff[q]; t < roff[q + 1]; t++) {
      int64_t r = runs[t], zz = z[r];
      if (zz < 0) continue;
      int64_t d = dir[r], f = fix[r], l = lo[r], h = hi[r];
      uint64_t key = pk(zz, f);
      auto it = wk.find(key); int64_t j;
      if (it == wk.end()) { j = (int64_t)nw++; wk.emplace(key, j); wkz.push_back(zz); wkf.push_back(f); if (wiv.size() < nw) wiv.emplace_back(); wiv[j].clear(); }
      else j = it->second;
      wiv[j].push_back({l, h});
      if (d == 0) { node(l, f, zz); node(h, f, zz); } else { node(f, l, zz); node(f, h, zz); }
    }
    for (int64_t t = nofs[q]; t < nofs[q + 1]; t++) {
      const int64_t* e = nodes + 4 * t;
      node(e[0], e[1], e[2]); node(e[0], e[1], e[3]);
    }
    int64_t c = 0;
    for (size_t j = 0; j < nw; j++) {
      int64_t zz = wkz[j], f = wkf[j];
      for (auto& iv : wiv[j]) {
        if (dirs[zz] == 0) { int64_t s[6] = {iv.first, f, zz, iv.second, f, zz}; OUT.insert(OUT.end(), s, s + 6); }
        else { int64_t s[6] = {f, iv.first, zz, f, iv.second, zz}; OUT.insert(OUT.end(), s, s + 6); }
        c++;
      }
    }
    for (size_t j = 0; j < nx.size(); j++)
      for (int64_t zz = nmin[j]; zz < nmax[j]; zz++) { int64_t s[6] = {nx[j], ny[j], zz, nx[j], ny[j], zz + 1}; OUT.insert(OUT.end(), s, s + 6); c++; }
    CNT[q] = c;
  }
  return (int64_t)(OUT.size() / 6);
}
void ha_fetch(int64_t* out, int64_t* cnt) {
  std::copy(OUT.begin(), OUT.end(), out); std::copy(CNT.begin(), CNT.end(), cnt);
}
}
