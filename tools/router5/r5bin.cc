// r5bin.cc (router5, 2026-11-06, NEW): reader of the compact binary route R5B1 written by InstantGR_r5 -bin <file>.
// Returns exactly what route_cpp.read_route returns for the contest route that route_cpp.convert_instantgr makes from
// the text route of the same run: x/y in DBU (GCell * 4200 + 2100), z = zero-based metal, InstantGR via rows split
// into one-layer segments (lo..hi-1), net_id = net block index, offsets[n_nets + 1], names in file order.
//
// R5B1 layout (little endian, no padding; written by graph.hpp r5_write_bin):
//   char[4] "R5B1" | uint32 version (1) | uint64 n_nets | uint64 n_rows | uint64 names_bytes
//   uint64 name_off[n_nets + 1] | uint64 row_off[n_nets + 1] | char names[names_bytes] | int32 rows[n_rows][6]
//   rows: x0 y0 l0 x1 y1 l1 of each text line (GCell x, y; InstantGR layer index), nets in the order of the text route.
//
// Build: g++ -O3 -std=c++17 -fPIC -shared -pthread r5bin.cc -o libr5bin.so   (C entry points below; Python: r5bin.py)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
const int64_t kDbu = 4200, kHalf = 2100;
thread_local std::string last_error;

struct Doc {
    uint64_t n_nets = 0, n_segs = 0;
    std::vector<uint64_t> offsets;
    std::vector<int32_t> net_id, x0, y0, z0, x1, y1, z1;
    std::string names_nl;   // names joined by '\n'
};

template <class F>
void pfor(uint64_t n, int T, F f) {
    if (T <= 1 || n < 4096) { f(0, n); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) th.emplace_back([=, &f] { f(n * t / T, n * (t + 1) / T); });
    for (auto &x : th) x.join();
}

Doc *read_doc(const char *path, int T) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { last_error = std::string("cannot open ") + path; return nullptr; }
    struct stat st;
    fstat(fd, &st);
    size_t fs = st.st_size;
    char *b = (char *)malloc(fs ? fs : 1);
    size_t got = 0;
    while (got < fs) {
        ssize_t r = pread(fd, b + got, fs - got, got);
        if (r <= 0) { close(fd); free(b); last_error = "read error"; return nullptr; }
        got += r;
    }
    close(fd);
    if (fs < 32 || memcmp(b, "R5B1", 4) != 0) { free(b); last_error = "not an R5B1 file"; return nullptr; }
    uint64_t nn, nr, nb;
    memcpy(&nn, b + 8, 8); memcpy(&nr, b + 16, 8); memcpy(&nb, b + 24, 8);
    size_t need = 32 + 16 * (nn + 1) + nb + 24 * nr;
    if (fs != need) { free(b); last_error = "R5B1 size mismatch"; return nullptr; }
    const uint64_t *name_off = (const uint64_t *)(b + 32);
    const uint64_t *row_off = name_off + (nn + 1);
    const char *names = (const char *)(row_off + (nn + 1));
    const int32_t *rows = (const int32_t *)(names + nb);
    auto *d = new Doc();
    d->n_nets = nn;
    // segments per net (a via row of l0 != l1 gives |l1 - l0| one-layer segments)
    std::vector<uint64_t> cnt(nn + 1, 0);
    pfor(nn, T, [&](uint64_t a, uint64_t e) {
        for (uint64_t k = a; k < e; k++) {
            uint64_t c = 0;
            for (uint64_t r = row_off[k]; r < row_off[k + 1]; r++) {
                int32_t l0 = rows[6 * r + 2], l1 = rows[6 * r + 5];
                c += l0 != l1 ? (uint64_t)(l0 < l1 ? l1 - l0 : l0 - l1) : 1;
            }
            cnt[k + 1] = c;
        }
    });
    d->offsets.assign(nn + 1, 0);
    for (uint64_t k = 0; k < nn; k++) d->offsets[k + 1] = d->offsets[k] + cnt[k + 1];
    uint64_t S = d->offsets[nn];
    d->n_segs = S;
    for (auto *v : {&d->net_id, &d->x0, &d->y0, &d->z0, &d->x1, &d->y1, &d->z1}) v->resize(S);
    pfor(nn, T, [&](uint64_t a, uint64_t e) {
        for (uint64_t k = a; k < e; k++) {
            uint64_t s = d->offsets[k];
            for (uint64_t r = row_off[k]; r < row_off[k + 1]; r++) {
                const int32_t *w = rows + 6 * r;
                int32_t X0 = (int32_t)(w[0] * kDbu + kHalf), Y0 = (int32_t)(w[1] * kDbu + kHalf);
                int32_t X1 = (int32_t)(w[3] * kDbu + kHalf), Y1 = (int32_t)(w[4] * kDbu + kHalf);
                int32_t zl = w[2], zh = w[5];
                if (zl != zh) {
                    int32_t lo = zl < zh ? zl : zh, hi = zl < zh ? zh : zl;
                    for (int32_t z = lo; z < hi; z++, s++) {
                        d->net_id[s] = (int32_t)k; d->x0[s] = X0; d->y0[s] = Y0; d->z0[s] = z;
                        d->x1[s] = X1; d->y1[s] = Y1; d->z1[s] = z + 1;
                    }
                } else {
                    d->net_id[s] = (int32_t)k; d->x0[s] = X0; d->y0[s] = Y0; d->z0[s] = zl;
                    d->x1[s] = X1; d->y1[s] = Y1; d->z1[s] = zh;
                    s++;
                }
            }
        }
    });
    d->names_nl.reserve(nb + nn);
    for (uint64_t k = 0; k < nn; k++) {
        if (k) d->names_nl.push_back('\n');
        d->names_nl.append(names + name_off[k], name_off[k + 1] - name_off[k]);
    }
    free(b);
    return d;
}
}  // namespace

extern "C" {
void *r5b_read(const char *path, int threads) { return read_doc(path, threads); }
void r5b_free(void *h) { delete (Doc *)h; }
const char *r5b_error() { return last_error.c_str(); }
uint64_t r5b_num_nets(void *h) { return ((Doc *)h)->n_nets; }
uint64_t r5b_num_segments(void *h) { return ((Doc *)h)->n_segs; }
const uint64_t *r5b_offsets(void *h) { return ((Doc *)h)->offsets.data(); }
const int32_t *r5b_col(void *h, int c) {
    Doc *d = (Doc *)h;
    const std::vector<int32_t> *v[7] = {&d->net_id, &d->x0, &d->y0, &d->z0, &d->x1, &d->y1, &d->z1};
    return (c >= 0 && c < 7) ? v[c]->data() : nullptr;
}
const char *r5b_names(void *h) { return ((Doc *)h)->names_nl.c_str(); }
uint64_t r5b_names_len(void *h) { return ((Doc *)h)->names_nl.size(); }
}
