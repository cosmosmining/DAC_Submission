// Generic ISPD'25 route block writer (new file, agent io). Same bytes as route_cpp's writer,
// but the number of segments per net may change: offsets are passed in.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
extern "C" int blocks_write(const char* path, const char* names, std::size_t nnets, const std::uint64_t* off,
                            const std::int32_t* x0, const std::int32_t* y0, const std::int32_t* z0,
                            const std::int32_t* x1, const std::int32_t* y1, const std::int32_t* z1) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return -1;
    std::string b; b.reserve(1 << 21);
    const char* p = names;
    char tmp[16];
    auto num = [&](std::int32_t v) { int n = std::snprintf(tmp, sizeof tmp, "%d", v); b.append(tmp, n); };
    for (std::size_t n = 0; n < nnets; ++n) {
        const char* e = std::strchr(p, '\n');
        b.append(p, e - p); b.append("\n(\n"); p = e + 1;
        for (std::uint64_t i = off[n]; i < off[n + 1]; ++i) {
            num(x0[i]); b.push_back(' '); num(y0[i]); b.append(" metal"); num(z0[i] + 1); b.push_back(' ');
            num(x1[i]); b.push_back(' '); num(y1[i]); b.append(" metal"); num(z1[i] + 1); b.push_back('\n');
        }
        b.append(")\n");
        if (b.size() > (1 << 20)) { std::fwrite(b.data(), 1, b.size(), f); b.clear(); }
    }
    std::fwrite(b.data(), 1, b.size(), f);
    return std::fclose(f);
}
