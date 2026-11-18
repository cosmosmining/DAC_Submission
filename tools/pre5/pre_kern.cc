// pre_kern.cc: integer graph kernels of the preload (exact replacements, no floating point).
//   pre_levelize : == fast_sta_v3._levelize (longest-path levels by Kahn; pins never reached (on / after a cycle) keep
//                  level 0 and are flagged left = 1). Duplicate edges count as in np.bincount / np.subtract.at.
//   pre_scc      : strongly connected components (iterative Tarjan); the labels differ from scipy's numbering, but the
//                  partition is the same, which is all fast_sta_v3.StaDesign._graph uses (same-label test and sizes).
// Build (tools/pre5/build_pre5.sh): g++ -std=c++17 -O3 -fPIC -shared
#include <cstdint>
#include <vector>

typedef int64_t i64;

extern "C" {

int pre_levelize(i64 P, i64 ne, const i64 *efr, const i64 *eto, i64 *lev, uint8_t *left) {
  std::vector<i64> off(P + 1, 0), adj(ne), indeg(P, 0);
  for (i64 e = 0; e < ne; e++) { off[efr[e] + 1]++; indeg[eto[e]]++; }
  for (i64 i = 0; i < P; i++) off[i + 1] += off[i];
  std::vector<i64> pos(off.begin(), off.end() - 1);
  for (i64 e = 0; e < ne; e++) adj[pos[efr[e]]++] = eto[e];
  std::vector<i64> q; q.reserve(P);
  std::vector<uint8_t> done(P, 0);
  for (i64 i = 0; i < P; i++) { lev[i] = 0; if (!indeg[i]) { q.push_back(i); done[i] = 1; } }
  size_t h = 0;
  while (h < q.size()) {
    i64 i = q[h++];
    for (i64 e = off[i]; e < off[i + 1]; e++) {
      i64 j = adj[e];
      if (lev[j] < lev[i] + 1) lev[j] = lev[i] + 1;
      if (--indeg[j] == 0) { q.push_back(j); done[j] = 1; }
    }
  }
  for (i64 i = 0; i < P; i++) { if (!done[i]) lev[i] = 0; left[i] = !done[i]; }
  return 0;
}

int pre_scc(i64 P, i64 ne, const i64 *efr, const i64 *eto, i64 *lab) {
  std::vector<i64> off(P + 1, 0), adj(ne);
  for (i64 e = 0; e < ne; e++) off[efr[e] + 1]++;
  for (i64 i = 0; i < P; i++) off[i + 1] += off[i];
  std::vector<i64> pos(off.begin(), off.end() - 1);
  for (i64 e = 0; e < ne; e++) adj[pos[efr[e]]++] = eto[e];
  std::vector<i64> idx(P, -1), low(P, 0), stk; stk.reserve(P);
  std::vector<uint8_t> on(P, 0);
  std::vector<std::pair<i64, i64>> cs;            // (node, next edge)
  i64 counter = 0, ncomp = 0;
  for (i64 r = 0; r < P; r++) {
    if (idx[r] >= 0) continue;
    cs.push_back({r, off[r]}); idx[r] = low[r] = counter++; stk.push_back(r); on[r] = 1;
    while (!cs.empty()) {
      i64 v = cs.back().first;
      i64 &e = cs.back().second;
      if (e < off[v + 1]) {
        i64 w = adj[e++];
        if (idx[w] < 0) { idx[w] = low[w] = counter++; stk.push_back(w); on[w] = 1; cs.push_back({w, off[w]}); }
        else if (on[w] && idx[w] < low[v]) low[v] = idx[w];
      } else {
        if (low[v] == idx[v]) {
          while (true) { i64 w = stk.back(); stk.pop_back(); on[w] = 0; lab[w] = ncomp; if (w == v) break; }
          ncomp++;
        }
        cs.pop_back();
        if (!cs.empty()) { i64 u = cs.back().first; if (low[v] < low[u]) low[u] = low[v]; }
      }
    }
  }
  return (int)ncomp;
}
}
