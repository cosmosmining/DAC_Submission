// io_write_b.cc: blocks_write of tools/fast_path/io_write.cc, same bytes, formatted in parallel.
// The nets are cut into contiguous ranges; each thread formats its range with the SAME snprintf("%d") calls into its own buffer;
// the buffers are written at their offsets (prefix sums, pwrite on the same threads).  So the file is the concatenation of exactly the
// strings the original builds; the return value is 0 on success as fclose's was.
// Threads: IOW_THREADS, else OMP_NUM_THREADS, else 8.
// Build (host): g++ -std=c++17 -O3 -shared -fPIC -pthread io_write_b.cc -o libio_write_b.so
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

extern "C" int blocks_write(const char* path, const char* names, std::size_t nnets, const std::uint64_t* off,
                            const std::int32_t* x0, const std::int32_t* y0, const std::int32_t* z0,
                            const std::int32_t* x1, const std::int32_t* y1, const std::int32_t* z1) {
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    const char* e = getenv("IOW_THREADS"); if (!e) e = getenv("OMP_NUM_THREADS");
    int T = e ? atoi(e) : 8; if (T < 1) T = 1;
    // name start of every net (sequential scan, as the original's strchr walk)
    std::vector<const char*> nm(nnets + 1);
    const char* p = names;
    for (std::size_t n = 0; n < nnets; ++n) { nm[n] = p; p = std::strchr(p, '\n') + 1; }
    nm[nnets] = p;
    // ranges balanced by segment count (+ one per net)
    const std::size_t NCH = (std::size_t)T * 8;
    std::vector<std::size_t> cut(1, 0);
    const std::uint64_t tot = off[nnets] + nnets;
    for (std::size_t c = 1; c < NCH; c++) {
        std::uint64_t want = tot * c / NCH;
        std::size_t lo = cut.back(), hi = nnets;           // first n with off[n] + n >= want
        while (lo < hi) { std::size_t m = (lo + hi) / 2; if (off[m] + m < want) lo = m + 1; else hi = m; }
        cut.push_back(lo);
    }
    cut.push_back(nnets);
    std::vector<std::string> buf(cut.size() - 1);
    std::vector<std::thread> th;
    auto work = [&](std::size_t c) {
        std::string& b = buf[c];
        char tmp[16];
        auto num = [&](std::int32_t v) { int n = std::snprintf(tmp, sizeof tmp, "%d", v); b.append(tmp, n); };
        for (std::size_t n = cut[c]; n < cut[c + 1]; ++n) {
            b.append(nm[n], nm[n + 1] - 1 - nm[n]); b.append("\n(\n");
            for (std::uint64_t i = off[n]; i < off[n + 1]; ++i) {
                num(x0[i]); b.push_back(' '); num(y0[i]); b.append(" metal"); num(z0[i] + 1); b.push_back(' ');
                num(x1[i]); b.push_back(' '); num(y1[i]); b.append(" metal"); num(z1[i] + 1); b.push_back('\n');
            }
            b.append(")\n");
        }
    };
    std::atomic_size_t* nx = new std::atomic_size_t(0);
    for (int t = 0; t < T; t++) th.emplace_back([&, nx]() { for (;;) { std::size_t c = nx->fetch_add(1); if (c >= buf.size()) break; work(c); } });
    for (auto& t : th) t.join();
    delete nx;
    // the buffers go to their offsets in the file (prefix sums of the sizes) with pwrite from the same threads: same bytes
    std::vector<std::size_t> at(buf.size() + 1, 0);
    for (std::size_t c = 0; c < buf.size(); c++) at[c + 1] = at[c] + buf[c].size();
    std::atomic<int> bad{0};
    std::atomic_size_t* nw = new std::atomic_size_t(0);
    th.clear();
    for (int t = 0; t < T; t++) th.emplace_back([&, nw]() {
        for (;;) {
            std::size_t c = nw->fetch_add(1); if (c >= buf.size()) break;
            const char* d = buf[c].data(); std::size_t left = buf[c].size(); off_t o = (off_t)at[c];
            while (left > 0) { ssize_t w = ::pwrite(fd, d, left, o); if (w <= 0) { bad = 1; break; } d += w; left -= (std::size_t)w; o += w; }
            std::string().swap(buf[c]);
        }
    });
    for (auto& t : th) t.join();
    delete nw;
    int rc = ::close(fd);
    return (bad || rc != 0) ? -1 : 0;
}
