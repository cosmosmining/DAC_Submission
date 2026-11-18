// h_route_read.cc: contest .route reader with a hand-written number parser over the whole
// file in memory, split over threads at net-block boundaries. Same arrays as tools/fast_io/route_cpp (read_route).
// STRICT: it accepts only the canonical layout the router and our writers produce:
//     <name>\n (\n <int> <int> metal<k> <int> <int> metal<k>\n ... )\n      (k >= 1, single spaces, '\n' line ends)
// Any other byte layout (blank lines, '\r', tabs, extra fields, a missing final '\n', ...) returns -2 and the Python side
// falls back to route_cpp.read_route on the whole file, so the result is the original reader's in every case.
// Numbers: as route_cpp, int64 parse then static_cast<int32_t>; a number of more than 18 digits -> fallback.
// Build: clang++ -std=c++17 -O3 -DNDEBUG -fPIC -dynamiclib (as tools/fast_io/route_cpp/build.sh), -pthread.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct Part {
  std::vector<int32_t> c[6];
  std::vector<uint64_t> cnt;        // segments per net
  std::vector<int64_t> nm0, nm1;    // name byte ranges in the file buffer
  bool ok = true;
};

inline bool pint(const char*& p, const char* e, int64_t& v) {    // -?[0-9]{1,18} then ' ' or '\n' (not consumed)
  bool neg = false;
  if (p < e && *p == '-') { neg = true; ++p; }
  const char* s = p; int64_t x = 0;
  while (p < e && (unsigned)(*p - '0') < 10u) { x = x * 10 + (*p - '0'); ++p; }
  if (p == s || p - s > 18) return false;
  v = neg ? -x : x;
  return true;
}
inline bool player(const char*& p, const char* e, int32_t& z) {  // metal[0-9]{1,9}, value >= 1
  if (e - p < 6 || std::memcmp(p, "metal", 5) != 0) return false;
  p += 5;
  const char* s = p; int64_t x = 0;
  while (p < e && (unsigned)(*p - '0') < 10u) { x = x * 10 + (*p - '0'); ++p; }
  if (p == s || p - s > 9 || x <= 0) return false;
  z = (int32_t)(x - 1);
  return true;
}

// parse [b, e): a sequence of whole net blocks
void parse(const char* b, const char* e, Part& P) {
  const char* p = b;
  P.cnt.reserve(1 << 14); P.nm0.reserve(1 << 14); P.nm1.reserve(1 << 14);
  for (auto& v : P.c) v.reserve((size_t)((e - b) / 36 + 16));
  while (p < e) {
    // name line: no leading/trailing space/tab/CR/LF, not "(" or ")"
    const char* q = (const char*)std::memchr(p, '\n', e - p);
    if (!q) { P.ok = false; return; }
    if (q == p) { P.ok = false; return; }
    for (const char* t = p; t < q; ++t) { char ch = *t; if (ch == '\r' || ch == '\t' || ch == ' ' || ch == '\v' || ch == '\f') { if (t == p || t == q - 1) { P.ok = false; return; } } }
    if (q - p == 1 && (*p == '(' || *p == ')')) { P.ok = false; return; }
    P.nm0.push_back(p - b); P.nm1.push_back(q - b);
    p = q + 1;
    if (e - p < 2 || p[0] != '(' || p[1] != '\n') { P.ok = false; return; }
    p += 2;
    uint64_t n = 0;
    for (;;) {
      if (p >= e) { P.ok = false; return; }
      if (*p == ')') {
        if (e - p < 2 || p[1] != '\n') { P.ok = false; return; }
        p += 2; break;
      }
      int64_t ax, ay, bx, by; int32_t za, zb;
      if (!pint(p, e, ax) || p >= e || *p != ' ') { P.ok = false; return; } ++p;
      if (!pint(p, e, ay) || p >= e || *p != ' ') { P.ok = false; return; } ++p;
      if (!player(p, e, za) || p >= e || *p != ' ') { P.ok = false; return; } ++p;
      if (!pint(p, e, bx) || p >= e || *p != ' ') { P.ok = false; return; } ++p;
      if (!pint(p, e, by) || p >= e || *p != ' ') { P.ok = false; return; } ++p;
      if (!player(p, e, zb) || p >= e || *p != '\n') { P.ok = false; return; } ++p;
      P.c[0].push_back((int32_t)ax); P.c[1].push_back((int32_t)ay); P.c[2].push_back(za);
      P.c[3].push_back((int32_t)bx); P.c[4].push_back((int32_t)by); P.c[5].push_back(zb);
      ++n;
    }
    P.cnt.push_back(n);
  }
}

std::vector<char> g_buf;
std::vector<Part> g_parts;
size_t g_nn = 0, g_ns = 0;
double g_t[3];
}  // namespace

extern "C" {
// returns 0 ok, -1 cannot read, -2 not canonical (caller falls back). Fills counts.
int hrr_read(const char* path, int nthreads, int64_t* out_nn, int64_t* out_ns) {
  g_parts.clear(); g_nn = g_ns = 0;
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st; if (fstat(fd, &st) != 0) { ::close(fd); return -1; }
  size_t sz = (size_t)st.st_size;
  g_buf.resize(sz);
  size_t got = 0;
  while (got < sz) { ssize_t r = ::read(fd, g_buf.data() + got, sz - got); if (r <= 0) { ::close(fd); return -1; } got += (size_t)r; }
  ::close(fd);
  const char* b = g_buf.data(); const char* e = b + sz;
  if (sz == 0) { *out_nn = 0; *out_ns = 0; return 0; }
  if (e[-1] != '\n') return -2;
  int T = nthreads < 1 ? 1 : nthreads;
  // chunk starts: just after a "\n)\n" (a net block's end), so every chunk starts at a name line
  std::vector<const char*> cut; cut.push_back(b);
  for (int k = 1; k < T; k++) {
    const char* p = b + sz * k / T;
    if (p <= cut.back()) continue;
    const char* hit = nullptr;
    while (p + 3 <= e) {
      const char* q = (const char*)std::memchr(p, ')', e - p);
      if (!q) break;
      if (q > b && q[-1] == '\n' && q + 1 < e && q[1] == '\n') { hit = q + 2; break; }
      p = q + 1;
    }
    if (!hit || hit >= e) break;
    if (hit > cut.back()) cut.push_back(hit);
  }
  cut.push_back(e);
  size_t np = cut.size() - 1;
  g_parts.assign(np, Part());
  std::vector<std::thread> th;
  for (size_t i = 1; i < np; i++) th.emplace_back(parse, cut[i], cut[i + 1], std::ref(g_parts[i]));
  parse(cut[0], cut[1], g_parts[0]);
  for (auto& t : th) t.join();
  for (size_t i = 0; i < np; i++) {
    if (!g_parts[i].ok) { g_parts.clear(); return -2; }
    // name offsets are relative to the chunk start: make them absolute
    int64_t o = cut[i] - b;
    for (auto& v : g_parts[i].nm0) v += o;
    for (auto& v : g_parts[i].nm1) v += o;
    g_nn += g_parts[i].cnt.size(); g_ns += g_parts[i].c[0].size();
  }
  *out_nn = (int64_t)g_nn; *out_ns = (int64_t)g_ns;
  return 0;
}

// copies into caller arrays: net_id/x0..z1 [ns] int32, offsets [nn+1] uint64; names blob '\n'-joined (+ final '\n'), returns its size
int64_t hrr_names_size() {
  int64_t s = 0;
  for (auto& P : g_parts) for (size_t i = 0; i < P.nm0.size(); i++) s += P.nm1[i] - P.nm0[i] + 1;
  return s;
}
void hrr_fetch(int32_t* net_id, int32_t* x0, int32_t* y0, int32_t* z0, int32_t* x1, int32_t* y1, int32_t* z1, uint64_t* off,
               char* names) {
  int32_t* o[6] = {x0, y0, z0, x1, y1, z1};
  size_t ps = 0, pn = 0; char* w = names; off[0] = 0;
  for (auto& P : g_parts) {
    size_t m = P.c[0].size();
    for (int k = 0; k < 6; k++) std::memcpy(o[k] + ps, P.c[k].data(), m * sizeof(int32_t));
    for (size_t i = 0; i < P.cnt.size(); i++) {
      uint64_t c = P.cnt[i];
      for (uint64_t j = 0; j < c; j++) net_id[off[pn] + j] = (int32_t)pn;
      off[pn + 1] = off[pn] + c; pn++;
      size_t L = (size_t)(P.nm1[i] - P.nm0[i]);
      std::memcpy(w, g_buf.data() + P.nm0[i], L); w += L; *w++ = '\n';
    }
    ps += m;
  }
  g_parts.clear(); std::vector<Part>().swap(g_parts);
  std::vector<char>().swap(g_buf);
}
}
