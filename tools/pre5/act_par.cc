// act_par.cc: tools/speed4/act_kern.cc with the per-level BFS run on several threads
// (act_set_threads(n), default 1 = the original loop). Exact by construction: inside one BFS level every visit reads only
// pins of LOWER levels (checked per pin: the driver of a load pin and the function inputs of a driver pin; a level with any
// pin that reads a pin of its own or a higher level runs sequentially in the original order) and writes only its own pin, so
// the stored activities after each level are the same bits in any order; the follow-up (register marks, queueing of the
// fanout) runs after the level in the original pop order. Original header follows.
#include <omp.h>
// act_kern.cc: switching-activity propagation of OpenSTA's Power::ensureActivities
// (contest fork, src/sta/power/Power.cc; the array replica tools/wave3/cold/Power_fast4.cc) on OUR timing graph
// (tools/speed4/act_inproc.py builds the arrays from fast_sta_v3.StaDesign + the Liberty functions).
// Same semantics: seeds (levelization roots that are not clock sources: 0.1 / duty 0.5, origin input), per-level BFS,
// PropActivityVisitor::visit (load pins copy the driver's stored activity, drivers evaluate their Liberty function on
// complement-edge ROBDDs: duty recursion and Boolean-difference activity with the float/double expressions of
// evalBddDuty / evalBddActivity), the 0.001 change test, register passes (D function -> IQ/IQN, flops capped at 1,
// IQN duty 1 - duty) up to max_activity_passes_ = 100, findActivity (constants 0/0, clock pins 2.0 / clock duty).
// Order within one BFS level does not change the result (every adjacency edge goes to a higher level), so our own
// topological levels give the same end state as OpenSTA's levels.  Built with -ffp-contract=off (no FMA contraction).
// Then findClkedActivity: activity / period of the pin's clock (else the instance clock, else unscaled), printed as
// "%.5e" and read back (what the chain reads from the .act file).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

typedef int64_t i64;
typedef int32_t i32;

enum { O_UNKNOWN = 0, O_INPUT = 1, O_PROP = 2, O_CLOCK = 3, O_CONST = 4 };

namespace {
struct PA { float act; float duty; int org; };

// node pool: ptr = 2 * id + complement; id 0 = constant ONE
const i32 *N_var, *N_T, *N_E;
// functions
const i32 *F_root, *F_nv, *F_diff, *F_pos, *F_direct;   // F_diff/F_pos: offsets into DIFF / POS pools
const i32 *DIFF, *POS;
// pins
i64 NP;
const int8_t *KIND; const i32 *WFROM; const uint8_t *ISLOAD; const i32 *FUNC; const i64 *VPOFF; const i32 *VP; const i32 *SEQI;
const float *CDUTY;
// seq instances
i64 NS;
const i32 *S_FUNC; const i64 *S_VPOFF; const uint8_t *S_ISREG; const uint8_t *S_HASINV; const i64 *S_QOFF; const i32 *S_Q;

std::vector<PA> A;          // per pin stored activity (activity_map_)
int G_THREADS = 1;
std::vector<PA> SQ;         // per seq instance x {IQ, IQN}  (seq_activity_map_)

inline PA findActivity(i32 k) {
  const PA &a = A[k];
  if (KIND[k] == 1) return PA{0.0f, 0.0f, O_CONST};
  if (KIND[k] == 2) {
    if (a.org != O_UNKNOWN) return a;
    return PA{2.0f, CDUTY[k], O_CLOCK};
  }
  if (a.org != O_UNKNOWN) return a;
  return PA{0.0f, 0.0f, O_UNKNOWN};
}

inline PA findSeq(i32 s, int which) {
  if (s < 0) return PA{0.0f, 0.0f, O_UNKNOWN};
  const PA &q = SQ[2 * (size_t)s + which];
  if (q.org != O_UNKNOWN) return q;
  return PA{0.0f, 0.0f, O_UNKNOWN};
}

// evalBddDuty on pointer p; vp = pins of the vars in sum order, pos = BDD var index -> sum position; s = seq instance
float duty(i32 p, const i32 *vp, const i32 *pos, i32 s) {
  i32 id = p >> 1;
  bool comp = p & 1;
  if (id == 0) return comp ? 0.0f : 1.0f;
  float duty0 = duty(N_E[id], vp, pos, s);
  float duty1 = duty(N_T[id], vp, pos, s);
  i32 k = vp[pos[N_var[id]]];
  if (k <= -10)                                 // internal port (IQ = -10, IQN = -11): findSeqActivity(...).duty()
    return findSeq(s, -10 - k).duty;
  if (k >= 0) {
    PA va = findActivity(k);
    float var_duty = va.duty;
    float d = duty0 * (1.0 - var_duty) + duty1 * var_duty;
    if (comp) d = 1.0 - d;
    return d;
  }
  return 0.0f;
}

PA evalActivity(i32 f, const i32 *vp, i32 s) {
  if (F_direct[f] >= 0) return findSeq(s, F_direct[f]);
  const i32 *pos = POS + F_pos[f];
  const i32 *dif = DIFF + F_diff[f];
  float d = duty(F_root[f], vp, pos, s);
  float activity = 0.0f;
  for (i32 i = 0; i < F_nv[f]; i++) {
    i32 k = vp[i];
    if (k >= 0) {
      PA va = findActivity(k);
      float diff_duty = duty(dif[i], vp, pos, s);
      float var_act = va.act * diff_duty;
      activity += var_act;
    }
  }
  return PA{activity, d, O_PROP};
}

inline bool setCheck(i32 k, PA a) {
  PA &prev = A[k];
  float ad = std::fabs(a.act - prev.act);
  float dd = std::fabs(a.duty - prev.duty);
  if (ad > 0.001f || dd > 0.001f || a.org != prev.org) {
    prev = a;
    return true;
  }
  return false;
}
}  // namespace

extern "C" {

void act_set_threads(int n) { G_THREADS = n > 0 ? n : 1; }

// returns 0 ok, 1 cycle in the adjacency; st[0..5] = passes, visits, levels, seconds, max_level_size, cyc_pins
int act_run(i64 np, const int8_t *kind, const i32 *wfrom, const uint8_t *isload, const i32 *func, const i64 *vpoff, const i32 *vp,
            const i32 *seqi, const float *cduty,
            const i64 *adj_off, const i32 *adj, i64 nroots, const i32 *roots,
            const i32 *n_var, const i32 *n_t, const i32 *n_e,
            const i32 *f_root, const i32 *f_nv, const i32 *f_diff, const i32 *f_pos, const i32 *f_direct, const i32 *diffp, const i32 *posp,
            i64 ns, const i32 *s_func, const i64 *s_vpoff, const uint8_t *s_isreg, const uint8_t *s_hasinv, const i64 *s_qoff, const i32 *s_q,
            int max_passes,
            float *o_act, float *o_duty, int8_t *o_org, double *st) {
  auto t0 = std::chrono::steady_clock::now();
  NP = np; KIND = kind; WFROM = wfrom; ISLOAD = isload; FUNC = func; VPOFF = vpoff; VP = vp; SEQI = seqi; CDUTY = cduty;
  N_var = n_var; N_T = n_t; N_E = n_e; F_root = f_root; F_nv = f_nv; F_diff = f_diff; F_pos = f_pos; F_direct = f_direct;
  DIFF = diffp; POS = posp;
  NS = ns; S_FUNC = s_func; S_VPOFF = s_vpoff; S_ISREG = s_isreg; S_HASINV = s_hasinv; S_QOFF = s_qoff; S_Q = s_q;
  A.assign(np, PA{0.0f, 0.0f, O_UNKNOWN});
  SQ.assign(2 * (size_t)ns, PA{0.0f, 0.0f, O_UNKNOWN});
  // loops: back edges of a DFS from the roots (pins without fanin, in pin order, then the rest) are disabled, as
  // Levelize marks disabled-loop edges (ActivitySrchPred skips them); only built when Kahn finds a cycle
  std::vector<i64> boff(adj_off, adj_off + np + 1);
  std::vector<i32> badj(adj, adj + adj_off[np]);
  int nback = 0;
  for (int attempt = 0; attempt < 2; attempt++) {
    std::vector<i32> indeg(np, 0);
    for (i64 e = 0; e < boff[np]; e++) indeg[badj[e]]++;
    std::vector<i32> qq; qq.reserve(np);
    for (i64 i = 0; i < np; i++) if (!indeg[i]) qq.push_back((i32)i);
    size_t hh = 0;
    while (hh < qq.size()) { i32 i = qq[hh++]; for (i64 e = boff[i]; e < boff[i + 1]; e++) if (--indeg[badj[e]] == 0) qq.push_back(badj[e]); }
    if ((i64)qq.size() == np) break;
    if (attempt == 1) { st[5] = (double)(np - (i64)qq.size()); return 1; }
    std::vector<i32> indeg0(np, 0);
    for (i64 e = 0; e < boff[np]; e++) indeg0[badj[e]]++;
    std::vector<char> color(np, 0);                 // 0 new, 1 on stack, 2 done
    std::vector<char> back(boff[np], 0);
    std::vector<std::pair<i32, i64>> stk;
    auto dfs = [&](i32 r) {
      if (color[r]) return;
      color[r] = 1; stk.push_back({r, boff[r]});
      while (!stk.empty()) {
        auto &t = stk.back();
        if (t.second < boff[t.first + 1]) {
          i64 e = t.second++;
          i32 j = badj[e];
          if (color[j] == 1) back[e] = 1;
          else if (color[j] == 0) { color[j] = 1; stk.push_back({j, boff[j]}); }
        } else { color[t.first] = 2; stk.pop_back(); }
      }
    };
    for (i64 i = 0; i < np; i++) if (!indeg0[i]) dfs((i32)i);
    for (i64 i = 0; i < np; i++) dfs((i32)i);
    std::vector<i64> no(np + 1, 0); std::vector<i32> na;
    for (i64 i = 0; i < np; i++) { for (i64 e = boff[i]; e < boff[i + 1]; e++) if (!back[e]) na.push_back(badj[e]); else nback++; no[i + 1] = (i64)na.size(); }
    boff.swap(no); badj.swap(na);
  }
  adj_off = boff.data(); adj = badj.data();
  st[6] = nback;
  // topological levels (Kahn) over the adjacency
  std::vector<i32> indeg(np, 0), level(np, 0);
  for (i64 e = 0; e < adj_off[np]; e++) indeg[adj[e]]++;
  std::vector<i32> q; q.reserve(np);
  for (i64 i = 0; i < np; i++) if (!indeg[i]) q.push_back((i32)i);
  size_t h = 0;
  while (h < q.size()) {
    i32 i = q[h++];
    for (i64 e = adj_off[i]; e < adj_off[i + 1]; e++) {
      i32 j = adj[e];
      if (level[j] < level[i] + 1) level[j] = level[i] + 1;
      if (--indeg[j] == 0) q.push_back(j);
    }
  }
  if ((i64)q.size() != np) { st[5] = (double)(np - (i64)q.size()); return 1; }
  int maxl = 0;
  for (i64 i = 0; i < np; i++) if (level[i] > maxl) maxl = level[i];
  std::vector<std::vector<i32>> Q(maxl + 1);
  std::vector<char> inq(np, 0);
  int first = 1 << 30, last = 0;
  auto enq = [&](i32 i) {
    if (!inq[i]) { inq[i] = 1; int l = level[i]; Q[l].push_back(i); if (last < l) last = l; if (l < first) first = l; }
  };
  auto enq_adj = [&](i32 i) { for (i64 e = adj_off[i]; e < adj_off[i + 1]; e++) enq(adj[e]); };
  std::vector<char> regmark(ns, 0);
  std::vector<i32> regs, nregs;
  long visits = 0;
  auto visit = [&](i32 i) {
    visits++;
    bool changed = false;
    if (ISLOAD[i] && WFROM[i] >= 0) {
      const PA &f = A[WFROM[i]];
      changed = setCheck(i, PA{f.act, f.duty, O_PROP});
    }
    if (FUNC[i] >= 0) {
      PA a = evalActivity(FUNC[i], VP + VPOFF[i], SEQI[i]);
      changed = setCheck(i, a);
    }
    if (changed) {
      if (ISLOAD[i] && SEQI[i] >= 0 && !regmark[SEQI[i]]) { regmark[SEQI[i]] = 1; nregs.push_back(SEQI[i]); }
      enq_adj(i);
    }
  };
  // pre5: pins whose reads are all in lower levels (safe for a parallel level)
  std::vector<char> safe(np, 1);
  long n_unsafe = 0;
  for (i64 i = 0; i < np; i++) {
    bool ok = true;
    if (ISLOAD[i] && WFROM[i] >= 0 && level[WFROM[i]] >= level[i]) ok = false;
    if (FUNC[i] >= 0 && F_direct[FUNC[i]] < 0) {
      const i32 *vp = VP + VPOFF[i];
      for (i32 v = 0; v < F_nv[FUNC[i]]; v++) { i32 k = vp[v]; if (k >= 0 && level[k] >= level[i]) ok = false; }
    }
    if (!ok) { safe[i] = 0; n_unsafe++; }
  }
  st[7] = (double)n_unsafe;
  std::vector<char> chg(np, 0);
  long par_levels = 0;
  auto bfs = [&]() {
    while (first <= last && first <= maxl) {
      std::vector<i32> &lv = Q[first];
      first++;
      bool par = G_THREADS > 1 && (i64)lv.size() >= 512;
      if (par) for (i32 i : lv) if (!safe[i]) { par = false; break; }
      if (!par) {
        while (!lv.empty()) { i32 i = lv.back(); lv.pop_back(); inq[i] = 0; visit(i); }
        continue;
      }
      par_levels++;
      const i64 n = (i64)lv.size();
      const i32 *L = lv.data();
#pragma omp parallel for num_threads(G_THREADS) schedule(dynamic, 256)
      for (i64 t = 0; t < n; t++) {
        i32 i = L[t];
        bool changed = false;
        if (ISLOAD[i] && WFROM[i] >= 0) {
          const PA &f = A[WFROM[i]];
          changed = setCheck(i, PA{f.act, f.duty, O_PROP});
        }
        if (FUNC[i] >= 0) {
          PA a = evalActivity(FUNC[i], VP + VPOFF[i], SEQI[i]);
          changed = setCheck(i, a);
        }
        chg[i] = changed;
      }
      visits += n;
      for (i64 t = n - 1; t >= 0; t--) {               // the original pop order (back to front)
        i32 i = L[t];
        inq[i] = 0;
        if (chg[i]) {
          if (ISLOAD[i] && SEQI[i] >= 0 && !regmark[SEQI[i]]) { regmark[SEQI[i]] = 1; nregs.push_back(SEQI[i]); }
          enq_adj(i);
        }
      }
      lv.clear();
    }
  };
  for (i64 r = 0; r < nroots; r++) {
    i32 k = roots[r];
    A[k] = PA{0.1f, 0.5f, O_INPUT};
    enq_adj(k);
  }
  bfs();
  regs.swap(nregs); nregs.clear();
  for (i32 s : regs) regmark[s] = 0;
  int pass = 1;
  while (!regs.empty() && pass < max_passes) {
    for (i32 s : regs) {
      for (int inv = 0; inv < 2; inv++) {
        if (inv && !S_HASINV[s]) continue;
        PA a = evalActivity(S_FUNC[s], VP + S_VPOFF[s], s);
        if (S_ISREG[s] && a.act > 1.0f) a.act = 1.0f;
        if (inv) a.duty = 1.0 - a.duty;
        a.org = O_PROP;
        SQ[2 * (size_t)s + inv] = a;
      }
      for (i64 e = S_QOFF[s]; e < S_QOFF[s + 1]; e++) enq(S_Q[e]);
    }
    bfs();
    regs.swap(nregs); nregs.clear();
    for (i32 s : regs) regmark[s] = 0;
    pass++;
  }
  for (i64 i = 0; i < np; i++) { o_act[i] = A[i].act; o_duty[i] = A[i].duty; o_org[i] = (int8_t)A[i].org; }
  st[0] = pass; st[1] = (double)visits; st[2] = maxl + 1; st[4] = (double)par_levels;
  st[3] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return 0;
}

// clock reach masks (bit c = clock c reaches the pin by arrivals): monotone OR worklist over the arrival adjacency
int act_clkmask(i64 np, const i64 *off, const i32 *adj, uint64_t *mask) {
  std::vector<i32> q; q.reserve(np);
  std::vector<char> inq(np, 0);
  for (i64 i = 0; i < np; i++) if (mask[i]) { q.push_back((i32)i); inq[i] = 1; }
  size_t h = 0;
  while (h < q.size()) {
    i32 i = q[h++]; inq[i] = 0;
    for (i64 e = off[i]; e < off[i + 1]; e++) {
      i32 j = adj[e];
      uint64_t m = mask[j] | mask[i];
      if (m != mask[j]) { mask[j] = m; if (!inq[j]) { inq[j] = 1; q.push_back(j); } }
    }
    if (h > (1u << 20) && h * 2 > q.size()) { q.erase(q.begin(), q.begin() + h); h = 0; }
  }
  return 0;
}

// density = act / period (float), "%.5e" round trip (what the chain parses from the .act file)
void act_density(i64 n, const float *act, const float *period, double *out) {
  char buf[64];
  for (i64 i = 0; i < n; i++) {
    float v = period[i] > 0.0f ? act[i] / period[i] : act[i];   // float / float (double gave 284 / 193 differences on ariane / NVDLA v)
    snprintf(buf, sizeof buf, "%.5e", (double)v);
    out[i] = strtod(buf, nullptr);
  }
}
}
