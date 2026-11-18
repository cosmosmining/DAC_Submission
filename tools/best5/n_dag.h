// n_dag.h: task graph of ranked re-route runs. Each run has a footprint (blocks of cells it can read or
// write, known before the loop); run j waits for the latest earlier run on each of its blocks. Runs without a path between them touch
// disjoint cells, so they may run at once, on one shared demand map, and the result is the serial loop's (no private maps, no redo).
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ndag {
// keeps T threads alive between the batches of one call: a thread's A* buffers (thread_local, as large as the largest window) survive
struct Pool {
  int T; std::vector<std::thread> th; std::mutex m; std::condition_variable cv, cvd; long gen = 0; int running = 0; bool quit = false; std::function<void(int)> job;
  explicit Pool(int T_) : T(T_) {
    for (int t = 0; t + 1 < T; ++t) th.emplace_back([this, t]() {
      long seen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]() { return gen != seen || quit; });
        if (quit) return;
        seen = gen; auto f = job;
        lk.unlock(); f(t); lk.lock();
        if (--running == 0) cvd.notify_all();
      }
    });
  }
  void run(std::function<void(int)> f) {
    { std::lock_guard<std::mutex> lk(m); job = f; running = T - 1; gen++; }
    cv.notify_all();
    f(T - 1);
    std::unique_lock<std::mutex> lk(m);
    cvd.wait(lk, [&]() { return running == 0; });
  }
  ~Pool() { { std::lock_guard<std::mutex> lk(m); quit = true; } cv.notify_all(); for (auto& x : th) x.join(); }
};
inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Info { double build_s = 0, run_s = 0, busy_s = 0, crit_s = 0; int64_t edges = 0, levels = 0, tasks = 0; };

// fp(j, out): append the block ids of run j to out (duplicates allowed). task(j, t): run j on thread t. T threads (the caller is one).
template <class FP, class TASK>
void run(int nc, int nblk, int T, FP fp, TASK task, Info& info) {
  double t0 = now_s();
  std::vector<int> last((size_t)nblk, -1), seen((size_t)nc, -1), indeg((size_t)nc, 0), lev((size_t)nc, 0), buf;
  std::vector<std::vector<int>> succ((size_t)nc);
  int64_t edges = 0, maxlev = 0;
  for (int j = 0; j < nc; ++j) {
    buf.clear(); fp(j, buf);
    int lv = 1;
    for (int b : buf) {
      int i = last[(size_t)b];
      if (i >= 0 && i != j && seen[(size_t)i] != j) { seen[(size_t)i] = j; succ[(size_t)i].push_back(j); indeg[(size_t)j]++; edges++; lv = std::max(lv, lev[(size_t)i] + 1); }
      last[(size_t)b] = j;
    }
    lev[(size_t)j] = lv; maxlev = std::max<int64_t>(maxlev, lv);
  }
  info.build_s = now_s() - t0; info.edges = edges; info.levels = maxlev; info.tasks = nc;
  std::vector<double> dur((size_t)nc, 0.0);
  double t1 = now_s();
  std::mutex m; std::condition_variable cv;
  std::priority_queue<int, std::vector<int>, std::greater<int>> rq;
  for (int j = 0; j < nc; ++j) if (indeg[(size_t)j] == 0) rq.push(j);
  int done = 0;
  auto worker = [&](int t) {
    std::unique_lock<std::mutex> lk(m);
    for (;;) {
      while (rq.empty() && done < nc) cv.wait(lk);
      if (rq.empty()) return;
      int j = rq.top(); rq.pop();
      lk.unlock();
      double a = now_s();
      task(j, t);
      dur[(size_t)j] = now_s() - a;
      lk.lock();
      done++;
      int woke = 0;
      for (int k : succ[(size_t)j]) if (--indeg[(size_t)k] == 0) { rq.push(k); woke++; }
      if (done == nc) cv.notify_all();
      else for (int q = 1; q < woke; ++q) cv.notify_one();
    }
  };
  std::vector<std::thread> th;
  for (int t = 0; t + 1 < T; ++t) th.emplace_back(worker, t);
  worker(T - 1);
  for (auto& x : th) x.join();
  info.run_s = now_s() - t1;
  // measured: busy thread-seconds, and the longest dependency chain in seconds
  std::vector<double> fin((size_t)nc, 0.0);
  double busy = 0, crit = 0;
  for (int j = 0; j < nc; ++j) {
    busy += dur[(size_t)j];
    double f = fin[(size_t)j] + dur[(size_t)j];
    crit = std::max(crit, f);
    for (int k : succ[(size_t)j]) fin[(size_t)k] = std::max(fin[(size_t)k], f);
  }
  info.busy_s = busy; info.crit_s = crit;
}
}  // namespace ndag
