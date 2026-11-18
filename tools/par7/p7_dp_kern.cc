// p7_dp_kern.cc: tools/speedcut/dp_split_kern.cc with the per-net body moved into net_dp()
// (unchanged) and dp_run_par() (level schedule on T threads, same results). Build: g++ -O3 -std=c++17 -ffp-contract=off -fPIC -shared -pthread
// dp_split_kern.cc: C++ port of DP v2 (gapla_dp_v2.assign_v2 with
// gapla_dp._CostModel / _tree_dp), same arithmetic order, built with -ffp-contract=off so the
// chosen candidates are bit-identical to the numpy/Python version.
//   dp_run(): process the nets in `order` (net ids), one tree DP per net, committing demand after
//   each net.  `demand` is in/out (so a second call continues where the first stopped = the split).
//   crit_eff[n] is the criticality used in the delay term of net n (0 -> congestion-only cost).
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>

typedef int64_t i64;
static const double INF = std::numeric_limits<double>::infinity();

struct Ctx {
    i64 R, K, L, npts;
    const i64 *run_net, *cand, *p0, *p1, *ent_rk, *ent_cell, *pt_lo, *pt_hi;
    i64 n_ent, n_pt_lo, n_pt_hi;
    const double *dcoef, *cap, *u, *Cz, *run_len, *gnn_logp;
    double um, w4, dw, pw, via_cost, gnn_lambda, exp_half, cap_eps;
    // CSR: runs per net (ascending run id), entries per run (ascending entry id), entries per rk
    std::vector<i64> net_off, net_runs, run_eoff, run_ent, rk_off, rk_ent;
};

static inline double via(const Ctx &c, i64 a, i64 b) {
    i64 d = a > b ? a - b : b - a;
    return c.via_cost * (double)d;
}

static inline double pin_stack(const Ctx &c, i64 point, i64 layer) {
    i64 lo = c.n_pt_lo ? c.pt_lo[point] : c.L;
    i64 hi = c.n_pt_hi ? c.pt_hi[point] : -1;
    if (hi < 0 || lo >= c.L) return 0.0;
    if (layer < lo) return via(c, layer, lo);
    if (layer > hi) return via(c, hi, layer);
    return 0.0;
}

static inline i64 argmin_np(const double *v, i64 n) {
    // numpy argmin: first minimum, first NaN wins
    i64 k = 0;
    double best = v[0];
    if (best != best) return 0;
    for (i64 i = 1; i < n; i++) {
        double x = v[i];
        if (x != x) return i;
        if (x < best) { best = x; k = i; }
    }
    return k;
}

// par7: the per-net body of dp_run, unchanged (scratch per caller; lc / fb accumulated by the caller in net order)
struct Scr { std::vector<double> unary, msg, edge_cost, vals, root_cost; std::vector<i64> pt_local, pt_ids; };
static void net_dp(const Ctx &c, const i64 n, const double *crit_eff, double *demand, i64 *choice, Scr &S,
                   i64 &fallback, double &local_cost) {
    const i64 K = c.K, L = c.L;
    std::vector<double> &unary = S.unary, &msg = S.msg, &vals = S.vals, &root_cost = S.root_cost;
    std::vector<i64> &pt_local = S.pt_local, &pt_ids = S.pt_ids;
    const i64 *runs = &c.net_runs[c.net_off[n]];
    const i64 m = c.net_off[n + 1] - c.net_off[n];
    if (m == 0) return;
    // ---------------- unary ----------------
    unary.assign(m * K, 0.0);
    for (i64 j = 0; j < m; j++) {
        const i64 r = runs[j];
        double *bins = &unary[j * K];   // overflow sums first
        for (i64 e = c.run_eoff[r]; e < c.run_eoff[r + 1]; e++) {
            const i64 i = c.run_ent[e];
            const i64 cell = c.ent_cell[i];
            const i64 k = c.ent_rk[i] - r * K;
            const double d = demand[cell], cp = c.cap[cell], uu = c.u[cell];
            double v;
            if (cp > c.cap_eps) {
                v = uu * std::exp(std::fmin(0.25 * (d + 2.0) - 0.5 * cp, 60.0)) * c.exp_half;
            } else {
                v = uu * (std::exp(std::fmin(0.75 * (d + 2.0), 60.0)) - std::exp(std::fmin(0.75 * d, 60.0)));
            }
            bins[k] += v;
        }
    }
    const double cw = c.dw * crit_eff[n];
    for (i64 j = 0; j < m; j++) {
        const i64 r = runs[j];
        for (i64 k = 0; k < K; k++) {
            const i64 z = c.cand[r * K + k];
            double ov = unary[j * K + k];
            double dly = cw * c.dcoef[r * K + k];
            i64 zc = z < 0 ? 0 : z;
            double cp = ((c.pw * c.Cz[zc]) * c.run_len[r]) * c.um;
            double o = c.w4 * ov;
            o = o + dly;
            o = o + cp;
            if (z < 0) o = INF;
            else if (c.gnn_logp && c.gnn_lambda != 0.0) o = o - c.gnn_lambda * c.gnn_logp[r * K + k];
            unary[j * K + k] = o;
        }
    }
    // ---------------- tree DP ----------------
    // adjacency in dict-insertion order: for each run (in order) p0->(r,p1), p1->(r,p0)
    pt_ids.clear();
    auto local = [&](i64 pt) -> i64 {
        if (pt_local[pt] < 0) { pt_local[pt] = (i64)pt_ids.size(); pt_ids.push_back(pt); }
        return pt_local[pt];
    };
    std::vector<std::vector<std::pair<i64, i64>>> adj;  // (run row j, neighbor local)
    for (i64 j = 0; j < m; j++) {
        const i64 r = runs[j];
        i64 a = local(c.p0[r]), b = local(c.p1[r]);
        if ((i64)adj.size() < (i64)pt_ids.size()) adj.resize(pt_ids.size());
        adj[a].push_back({j, b});
        adj[b].push_back({j, a});
    }
    const i64 np = (i64)pt_ids.size();
    std::vector<i64> parent_node(np, -2), parent_row(np, -1), order_v;
    order_v.reserve(np);
    order_v.push_back(0);
    parent_node[0] = -1;
    i64 n_tree_edges = 0;
    for (size_t oi2 = 0; oi2 < order_v.size(); oi2++) {
        i64 uu = order_v[oi2];
        for (auto &pr : adj[uu]) {
            i64 v = pr.second;
            if (parent_node[v] != -2) continue;
            parent_node[v] = uu; parent_row[v] = pr.first;
            n_tree_edges++;
            order_v.push_back(v);
        }
    }
    bool ok = ((i64)order_v.size() == np) && (n_tree_edges == m);
    i64 *ch = choice;  // write by run id
    if (!ok) {
        fallback++;
        for (i64 j = 0; j < m; j++) {
            i64 k = argmin_np(&unary[j * K], K);
            ch[runs[j]] = k;
            local_cost += unary[j * K + k];
        }
    } else {
        msg.assign(m * K, 0.0);
        for (i64 idx = (i64)order_v.size() - 1; idx >= 1; idx--) {
            const i64 v = order_v[idx];
            const i64 jp = parent_row[v];
            const i64 rp = runs[jp];
            double *ec = &msg[jp * K];
            for (i64 k = 0; k < K; k++) ec[k] = unary[jp * K + k];
            for (i64 zs = 0; zs < K; zs++) {
                const i64 z = c.cand[rp * K + zs];
                if (z < 0) { ec[zs] = INF; continue; }
                double total = pin_stack(c, c.pt_lo ? pt_ids[v] : pt_ids[v], z);
                for (auto &pr : adj[v]) {
                    if (pr.first == jp) continue;
                    const i64 jc = pr.first, rc = runs[jc];
                    const double *cm = &msg[jc * K];
                    for (i64 q = 0; q < K; q++) {
                        const i64 zz = c.cand[rc * K + q];
                        vals[q] = cm[q] + (zz >= 0 ? via(c, z, zz) : INF);
                    }
                    i64 ck = argmin_np(vals.data(), K);
                    total += vals[ck];
                }
                ec[zs] += total;
            }
        }
        // root = local node 0
        std::vector<i64> rch;  // run rows adjacent to root
        for (auto &pr : adj[0]) rch.push_back(pr.first);
        for (i64 anchor = 0; anchor < L; anchor++) {
            double total = 0.0;
            for (i64 jr : rch) {
                const i64 rr = runs[jr];
                double mn = INF; bool first = true;
                for (i64 q = 0; q < K; q++) {
                    const i64 zz = c.cand[rr * K + q];
                    double x = msg[jr * K + q] + (zz >= 0 ? via(c, anchor, zz) : INF);
                    if (first || x < mn || x != x) { mn = x; first = false; }
                }
                total += mn;
            }
            root_cost[anchor] = total;
        }
        const i64 anchor = argmin_np(root_cost.data(), L);
        // descend (BFS from root; incoming layer per node)
        std::vector<i64> in_layer(np, 0);
        std::vector<i64> stack_nodes;
        // process nodes in BFS order: for node v with incoming layer, choose each child edge
        in_layer[0] = anchor;
        for (size_t oi3 = 0; oi3 < order_v.size(); oi3++) {
            const i64 v = order_v[oi3];
            const bool is_root = (v == order_v[0]);
            for (auto &pr : adj[v]) {
                const i64 jr = pr.first, child = pr.second;
                if (parent_node[child] != v || parent_row[child] != jr) continue;  // child edges only
                const i64 rr = runs[jr];
                for (i64 q = 0; q < K; q++) {
                    const i64 zz = c.cand[rr * K + q];
                    double add;
                    if (is_root) add = (zz >= 0) ? via(c, in_layer[v], zz) : 0.0;
                    else add = (zz >= 0) ? via(c, in_layer[v], zz) : INF;
                    vals[q] = msg[jr * K + q] + add;
                }
                i64 k = argmin_np(vals.data(), K);
                ch[rr] = k;
                in_layer[child] = c.cand[rr * K + k];
            }
        }
        double lc = 0.0;
        for (i64 j = 0; j < m; j++) lc += unary[j * K + ch[runs[j]]];
        local_cost += lc;
    }
    for (i64 q = 0; q < np; q++) pt_local[pt_ids[q]] = -1;
    // ---------------- commit (fancy-index += : duplicates count once) ----------------
    for (i64 j = 0; j < m; j++) {
        const i64 r = runs[j];
        const i64 rk = r * K + ch[r];
        const i64 a = c.rk_off[rk], b = c.rk_off[rk + 1];
        if (b - a == 1) { demand[c.ent_cell[c.rk_ent[a]]] += 2.0; continue; }
        // dedupe cells in this candidate (small lists): mark by sorting a copy
        static thread_local std::vector<i64> tmp;
        tmp.clear();
        for (i64 e = a; e < b; e++) tmp.push_back(c.ent_cell[c.rk_ent[e]]);
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        for (i64 cell : tmp) demand[cell] += 2.0;
    }
}

extern "C" {

void *dp_ctx_new(i64 R, i64 K, i64 L, i64 nnet, i64 npts,
                 const i64 *run_net, const i64 *cand, const i64 *p0, const i64 *p1,
                 i64 n_ent, const i64 *ent_rk, const i64 *ent_cell,
                 i64 n_pt_lo, const i64 *pt_lo, i64 n_pt_hi, const i64 *pt_hi,
                 const double *dcoef, const double *cap, const double *u, const double *Cz,
                 const double *run_len, const double *gnn_logp,
                 double um, double w4, double dw, double pw, double via_cost, double gnn_lambda,
                 double exp_half, double cap_eps) {
    Ctx *c = new Ctx();
    c->R = R; c->K = K; c->L = L; c->npts = npts;
    c->run_net = run_net; c->cand = cand; c->p0 = p0; c->p1 = p1;
    c->ent_rk = ent_rk; c->ent_cell = ent_cell; c->n_ent = n_ent;
    c->pt_lo = pt_lo; c->pt_hi = pt_hi; c->n_pt_lo = n_pt_lo; c->n_pt_hi = n_pt_hi;
    c->dcoef = dcoef; c->cap = cap; c->u = u; c->Cz = Cz; c->run_len = run_len; c->gnn_logp = gnn_logp;
    c->um = um; c->w4 = w4; c->dw = dw; c->pw = pw; c->via_cost = via_cost; c->gnn_lambda = gnn_lambda;
    c->exp_half = exp_half; c->cap_eps = cap_eps;
    // runs per net
    c->net_off.assign(nnet + 1, 0);
    for (i64 r = 0; r < R; r++) c->net_off[run_net[r] + 1]++;
    for (i64 n = 0; n < nnet; n++) c->net_off[n + 1] += c->net_off[n];
    c->net_runs.resize(R);
    { std::vector<i64> pos(c->net_off.begin(), c->net_off.end() - 1);
      for (i64 r = 0; r < R; r++) c->net_runs[pos[run_net[r]]++] = r; }
    // entries per run / per rk (stable)
    c->run_eoff.assign(R + 1, 0);
    for (i64 i = 0; i < n_ent; i++) c->run_eoff[ent_rk[i] / K + 1]++;
    for (i64 r = 0; r < R; r++) c->run_eoff[r + 1] += c->run_eoff[r];
    c->run_ent.resize(n_ent);
    { std::vector<i64> pos(c->run_eoff.begin(), c->run_eoff.end() - 1);
      for (i64 i = 0; i < n_ent; i++) c->run_ent[pos[ent_rk[i] / K]++] = i; }
    c->rk_off.assign(R * K + 1, 0);
    for (i64 i = 0; i < n_ent; i++) c->rk_off[ent_rk[i] + 1]++;
    for (i64 j = 0; j < R * K; j++) c->rk_off[j + 1] += c->rk_off[j];
    c->rk_ent.resize(n_ent);
    { std::vector<i64> pos(c->rk_off.begin(), c->rk_off.end() - 1);
      for (i64 i = 0; i < n_ent; i++) c->rk_ent[pos[ent_rk[i]]++] = i; }
    return c;
}

void dp_ctx_free(void *p) { delete (Ctx *)p; }

// returns number of fallback nets; local_cost (approximate sum, stats only) in out[0]
i64 dp_run(void *p, const i64 *order, i64 n_order, const double *crit_eff,
           double *demand, i64 *choice, double *out) {
    const Ctx &c = *(Ctx *)p;
    i64 fallback = 0;
    double local_cost = 0.0;
    Scr S; S.vals.resize(c.K); S.root_cost.resize(c.L); S.pt_local.assign(c.npts, -1);
    for (i64 oi = 0; oi < n_order; oi++) net_dp(c, order[oi], crit_eff, demand, choice, S, fallback, local_cost);
    out[0] = local_cost;
    return fallback;
}

// par7: the same nets on T threads. Level of a net = 1 + the largest level of an earlier net (in `order`) that reads or
// writes one of its demand cells (its candidates' entry cells = everything net_dp reads from or adds to `demand`). Nets of
// one level share no cell, and every net reads its cells after all earlier nets that touch them committed and before any
// later one did: the values each net reads are those of the sequential loop, so choice / demand / fallback are identical;
// local_cost is summed per net in `order`. st (8 doubles): items, levels, max level size, t_levels, then nets in levels of
// size 1, 2-7, 8-63, >= 64.
i64 dp_run_par(void *p, const i64 *order, i64 n_order, const double *crit_eff, double *demand, i64 *choice, double *out,
               i64 n_cells, i64 T, double *st) {
    const Ctx &c = *(Ctx *)p;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> last(n_cells, 0), lev(n_order, 0);
    int32_t nlev = 0;
    for (i64 oi = 0; oi < n_order; oi++) {
        const i64 n = order[oi];
        int32_t L0 = 0;
        for (i64 q = c.net_off[n]; q < c.net_off[n + 1]; q++) {
            const i64 r = c.net_runs[q];
            for (i64 e = c.run_eoff[r]; e < c.run_eoff[r + 1]; e++) L0 = std::max(L0, last[c.ent_cell[c.run_ent[e]]]);
        }
        const int32_t lv = L0 + 1; lev[oi] = lv; nlev = std::max(nlev, lv);
        for (i64 q = c.net_off[n]; q < c.net_off[n + 1]; q++) {
            const i64 r = c.net_runs[q];
            for (i64 e = c.run_eoff[r]; e < c.run_eoff[r + 1]; e++) last[c.ent_cell[c.run_ent[e]]] = lv;
        }
    }
    std::vector<i64> lstart(nlev + 2, 0), items(n_order);
    for (i64 oi = 0; oi < n_order; oi++) lstart[lev[oi] + 1]++;
    for (int32_t l = 0; l <= nlev; l++) lstart[l + 1] += lstart[l];
    { std::vector<i64> pos(lstart.begin(), lstart.end() - 1); for (i64 oi = 0; oi < n_order; oi++) items[pos[lev[oi]]++] = oi; }
    double mx = 0, b[4] = {0, 0, 0, 0};
    for (int32_t l = 1; l <= nlev; l++) {
        double sz = (double)(lstart[l + 1] - lstart[l]); mx = std::max(mx, sz);
        b[sz < 2 ? 0 : sz < 8 ? 1 : sz < 64 ? 2 : 3] += sz;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::vector<i64> fbn(n_order, 0); std::vector<double> lcn(n_order, 0.0);
    std::atomic<i64> nxt(0), done(0);
    auto worker = [&]() {
        Scr S; S.vals.resize(c.K); S.root_cost.resize(c.L); S.pt_local.assign(c.npts, -1);
        for (;;) {
            const i64 k = nxt.fetch_add(1);
            if (k >= n_order) break;
            const i64 oi = items[k];
            const i64 need = lstart[lev[oi]];
            int spins = 0;
            while (done.load(std::memory_order_acquire) < need) { if (++spins > 64) { std::this_thread::yield(); spins = 0; } }
            net_dp(c, order[oi], crit_eff, demand, choice, S, fbn[oi], lcn[oi]);
            done.fetch_add(1, std::memory_order_release);
        }
    };
    { std::vector<std::thread> th; for (i64 t = 1; t < T; t++) th.emplace_back(worker); worker(); for (auto &t : th) t.join(); }
    i64 fallback = 0; double local_cost = 0.0;
    for (i64 oi = 0; oi < n_order; oi++) { fallback += fbn[oi]; local_cost += lcn[oi]; }
    out[0] = local_cost;
    auto t2 = std::chrono::steady_clock::now();
    if (st) { st[0] = (double)n_order; st[1] = nlev; st[2] = mx; st[3] = std::chrono::duration<double>(t1 - t0).count();
              st[4] = b[0]; st[5] = b[1]; st[6] = b[2]; st[7] = b[3]; st[8] = std::chrono::duration<double>(t2 - t1).count(); }
    return fallback;
}

}  // extern "C"
