// n_ofc.cc: the two whole-map steps of tools/speed4/s4_ovf.py on T threads, same bits:
//  n_ofc_diff: per layer, the cells whose demand differs from the kept copy (flat memory positions of the C-contiguous arrays), copy updated;
//  n_ofc_sum:  numpy's pairwise sum of a contiguous double array (numpy pairwise_sum_DOUBLE: same splits, so the same float), the top of the
//              recursion on threads. The exp of the changed cells stays in numpy (tools/best5/n_ofc.py), so it is numpy's exp on every machine.
// Build: tools/best5/build_n.sh (no -ffast-math, no AVX2 flags).
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>
typedef int64_t i64;
static double pw(const double* a, i64 n) {
  if (n < 8) { double res = 0.; for (i64 i = 0; i < n; ++i) res += a[i]; return res; }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    i64 i;
    for (i = 8; i < n - (n % 8); i += 8) { for (int j = 0; j < 8; ++j) r[j] += a[i + j]; }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  i64 n2 = n / 2; n2 -= n2 % 8;
  return pw(a, n2) + pw(a + n2, n - n2);
}
static double pw_par(const double* a, i64 n, int depth) {
  if (depth <= 0 || n <= 4096) return pw(a, n);
  i64 n2 = n / 2; n2 -= n2 % 8;
  double l = 0;
  std::thread t([&]() { l = pw_par(a, n2, depth - 1); });
  double r = pw_par(a + n2, n - n2, depth - 1);
  t.join();
  return l + r;
}
extern "C" {
// layers z = 1 .. L-1: d[z], dprev[z] contiguous arrays of n doubles (pointers in dp / pp). out[z]: positions (i64, capacity n each, caller's), cnt[z] counts.
void n_ofc_diff(int L, i64 n, const double** dp, double** pp, i64** out, i64* cnt, int T) {
  std::vector<std::thread> th;
  int nl = L - 1, T1 = std::max(1, std::min(T, nl));
  for (int t = 0; t < T1; ++t) th.emplace_back([=]() {
    for (int q = t; q < nl; q += T1) {
      int z = q + 1;
      const double* a = dp[z]; double* b = pp[z]; i64* o = out[z]; i64 m = 0;
      for (i64 i = 0; i < n; ++i) if (a[i] != b[i]) { o[m++] = i; b[i] = a[i]; }
      cnt[z] = m;
    }
  });
  for (auto& x : th) x.join();
}
// out[z] = numpy's sum of lp[z] (n doubles), z = 1 .. L-1
void n_ofc_sum(int L, i64 n, const double** lp, double* out, int T) {
  std::vector<std::thread> th;
  int nl = L - 1, T1 = std::max(1, std::min(T, nl));
  int depth = T >= 2 * nl ? 1 : 0;
  for (int t = 0; t < T1; ++t) th.emplace_back([=]() {
    for (int q = t; q < nl; q += T1) out[q + 1] = pw_par(lp[q + 1], n, depth);
  });
  for (auto& x : th) x.join();
}
}
