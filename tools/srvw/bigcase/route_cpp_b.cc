// route_cpp_b.cc: copy of tools/fast_io/route_cpp/route_cpp.cc with a faster route_read (same
// document, see read_route_file below; ROUTE_CPP_SLOW=1 = the original reader). Build (host):
//   g++ -std=c++17 -O3 -DNDEBUG -fPIC -shared route_cpp_b.cc -o libroute_cpp_b.so
// C++17 route I/O and InstantGR route conversion.
//
// The C ABI is intentionally small: Python owns numpy copies of the arrays,
// while this library owns the parsed document and its net-block metadata.

#include <cstddef>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr std::int64_t kGcellDbu = 4200;
constexpr std::int64_t kGcellHalf = 2100;

thread_local std::string last_error;

std::string trim(const std::string& input) {
    const std::size_t first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

struct RouteDocument {
    std::vector<std::string> names;
    // offsets has one entry per net plus the final segment count.
    std::vector<std::uint64_t> offsets;
    std::vector<std::int32_t> net_id;
    std::vector<std::int32_t> x0, y0, z0, x1, y1, z1;
};

bool parse_layer(const std::string& token, std::int32_t& layer) {
    if (token.size() < 6 || token.compare(0, 5, "metal") != 0) return false;
    try {
        std::size_t used = 0;
        const int one_based = std::stoi(token.substr(5), &used);
        if (used != token.size() - 5 || one_based <= 0) return false;
        layer = static_cast<std::int32_t>(one_based - 1);
        return true;
    } catch (...) {
        return false;
    }
}

RouteDocument* read_route_file_slow(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open route: ") + path);

    auto* doc = new RouteDocument();
    doc->offsets.reserve(1024);
    doc->offsets.push_back(0);
    bool in_net = false;
    std::string raw;
    while (std::getline(in, raw)) {
        const std::string line = trim(raw);
        if (line.empty()) continue;
        if (!in_net) {
            if (line == "(" || line == ")") {
                throw std::runtime_error("route: unexpected block delimiter");
            }
            doc->names.push_back(line);
            in_net = true;
            continue;
        }
        if (line == "(") continue;
        if (line == ")") {
            doc->offsets.push_back(doc->x0.size());
            in_net = false;
            continue;
        }

        std::istringstream fields(line);
        std::int64_t ax, ay, bx, by;
        std::string az, bz, extra;
        if (!(fields >> ax >> ay >> az >> bx >> by >> bz) || (fields >> extra)) {
            throw std::runtime_error("route: malformed segment line: " + line);
        }
        std::int32_t layer_a, layer_b;
        if (!parse_layer(az, layer_a) || !parse_layer(bz, layer_b)) {
            throw std::runtime_error("route: expected metalN layers: " + line);
        }
        doc->net_id.push_back(static_cast<std::int32_t>(doc->names.size() - 1));
        doc->x0.push_back(static_cast<std::int32_t>(ax));
        doc->y0.push_back(static_cast<std::int32_t>(ay));
        doc->z0.push_back(layer_a);
        doc->x1.push_back(static_cast<std::int32_t>(bx));
        doc->y1.push_back(static_cast<std::int32_t>(by));
        doc->z1.push_back(layer_b);
    }
    if (in_net) {
        doc->offsets.push_back(doc->x0.size());
    }
    if (doc->offsets.size() != doc->names.size() + 1) {
        throw std::runtime_error("route: missing closing delimiter");
    }
    return doc;
}

// ---- agent srvw_bigcase: fast reader, same document.  The file is read in one piece and split on '\n' as getline does; each
// trimmed line takes the original branch; a segment line whose six tokens are plain ([-]digits x2, metal<digits>, x2) separated by
// blanks/tabs is converted directly (the values istream >> int64 and stoi give for such tokens); any other segment line goes through
// the original istringstream code, so errors and odd inputs behave exactly as before.
static inline bool is_blank(char c) { return c == ' ' || c == '\t'; }
static inline bool tok_int(const char*& p, const char* e, std::int64_t& v) {
    bool neg = false; const char* q = p;
    if (q < e && *q == '-') { neg = true; ++q; }
    const char* d0 = q; std::int64_t x = 0;
    while (q < e && *q >= '0' && *q <= '9') { x = x * 10 + (*q - '0'); ++q; }
    std::size_t nd = (std::size_t)(q - d0);
    if (nd == 0 || nd > 18) return false;
    if (q < e && !is_blank(*q)) return false;
    v = neg ? -x : x; p = q; return true;
}
static inline bool tok_layer(const char*& p, const char* e, std::int32_t& layer) {
    const char* q = p;
    if (e - q < 6 || std::memcmp(q, "metal", 5) != 0) return false;
    q += 5; const char* d0 = q; std::int64_t x = 0;
    while (q < e && *q >= '0' && *q <= '9') { x = x * 10 + (*q - '0'); ++q; }
    std::size_t nd = (std::size_t)(q - d0);
    if (nd == 0 || nd > 9 || x <= 0) return false;
    if (q < e && !is_blank(*q)) return false;
    layer = static_cast<std::int32_t>(x - 1); p = q; return true;
}
static inline void skip_blank(const char*& p, const char* e) { while (p < e && is_blank(*p)) ++p; }

RouteDocument* read_route_file(const char* path) {
    if (std::getenv("ROUTE_CPP_SLOW")) return read_route_file_slow(path);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open route: ") + path);
    std::vector<char> data;
    {
        std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        if (sz < 0) { std::fclose(f); return read_route_file_slow(path); }
        data.resize((std::size_t)sz);
        std::size_t got = sz ? std::fread(data.data(), 1, (std::size_t)sz, f) : 0;
        std::fclose(f);
        if (got != (std::size_t)sz) return read_route_file_slow(path);
    }
    auto* doc = new RouteDocument();
    try {
        std::size_t est = data.size() / 40 + 16;
        doc->offsets.reserve(1024); doc->offsets.push_back(0);
        for (auto* v : {&doc->net_id, &doc->x0, &doc->y0, &doc->z0, &doc->x1, &doc->y1, &doc->z1}) v->reserve(est);
        bool in_net = false;
        const char* p = data.data(); const char* end = p + data.size();
        while (p < end) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', (std::size_t)(end - p)));
            const char* le = nl ? nl : end;
            const char* b = p; const char* e = le;
            p = nl ? nl + 1 : end;
            while (b < e && (*b == ' ' || *b == '\t' || *b == '\r' || *b == '\n')) ++b;
            while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) --e;
            if (b == e) continue;
            const std::size_t n = (std::size_t)(e - b);
            if (!in_net) {
                if (n == 1 && (*b == '(' || *b == ')')) throw std::runtime_error("route: unexpected block delimiter");
                doc->names.emplace_back(b, n);
                in_net = true;
                continue;
            }
            if (n == 1 && *b == '(') continue;
            if (n == 1 && *b == ')') { doc->offsets.push_back(doc->x0.size()); in_net = false; continue; }
            std::int64_t ax, ay, bx, by; std::int32_t la, lb;
            const char* q = b;
            bool ok = tok_int(q, e, ax); if (ok) { skip_blank(q, e); ok = tok_int(q, e, ay); }
            if (ok) { skip_blank(q, e); ok = tok_layer(q, e, la); }
            if (ok) { skip_blank(q, e); ok = tok_int(q, e, bx); }
            if (ok) { skip_blank(q, e); ok = tok_int(q, e, by); }
            if (ok) { skip_blank(q, e); ok = tok_layer(q, e, lb); }
            ok = ok && q == e;
            if (!ok) {                                   // the original code for this line
                const std::string line(b, n);
                std::istringstream fields(line);
                std::string az, bz, extra;
                if (!(fields >> ax >> ay >> az >> bx >> by >> bz) || (fields >> extra)) {
                    throw std::runtime_error("route: malformed segment line: " + line);
                }
                if (!parse_layer(az, la) || !parse_layer(bz, lb)) {
                    throw std::runtime_error("route: expected metalN layers: " + line);
                }
            }
            doc->net_id.push_back(static_cast<std::int32_t>(doc->names.size() - 1));
            doc->x0.push_back(static_cast<std::int32_t>(ax));
            doc->y0.push_back(static_cast<std::int32_t>(ay));
            doc->z0.push_back(la);
            doc->x1.push_back(static_cast<std::int32_t>(bx));
            doc->y1.push_back(static_cast<std::int32_t>(by));
            doc->z1.push_back(lb);
        }
        if (in_net) doc->offsets.push_back(doc->x0.size());
        if (doc->offsets.size() != doc->names.size() + 1) throw std::runtime_error("route: missing closing delimiter");
    } catch (...) { delete doc; throw; }
    return doc;
}

template <typename T>
void check_ptr(const T* ptr, std::size_t count, const char* label) {
    if (count != 0 && ptr == nullptr) {
        throw std::runtime_error(std::string("route_write: null ") + label);
    }
}

template <typename T>
void append_integer(std::string& buffer, T value) {
    char digits[32];
    const auto result = std::to_chars(digits, digits + sizeof(digits), value);
    if (result.ec != std::errc()) throw std::runtime_error("integer formatting failed");
    buffer.append(digits, result.ptr);
}

void flush_buffer(std::ofstream& out, std::string& buffer) {
    if (!buffer.empty()) {
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        buffer.clear();
    }
}

void write_route_file(const RouteDocument& doc, const char* path,
                      const std::int32_t* x0, const std::int32_t* y0,
                      const std::int32_t* z0, const std::int32_t* x1,
                      const std::int32_t* y1, const std::int32_t* z1,
                      std::size_t count) {
    if (count != doc.x0.size()) throw std::runtime_error("route_write: segment count changed");
    check_ptr(x0, count, "x0"); check_ptr(y0, count, "y0");
    check_ptr(z0, count, "z0"); check_ptr(x1, count, "x1");
    check_ptr(y1, count, "y1"); check_ptr(z1, count, "z1");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("cannot open route output: ") + path);
    std::string buffer;
    buffer.reserve(1 << 20);
    for (std::size_t n = 0; n < doc.names.size(); ++n) {
        buffer.append(doc.names[n]);
        buffer.append("\n(\n");
        const std::size_t begin = static_cast<std::size_t>(doc.offsets[n]);
        const std::size_t end = static_cast<std::size_t>(doc.offsets[n + 1]);
        for (std::size_t i = begin; i < end; ++i) {
            append_integer(buffer, x0[i]); buffer.push_back(' ');
            append_integer(buffer, y0[i]); buffer.append(" metal");
            append_integer(buffer, z0[i] + 1); buffer.push_back(' ');
            append_integer(buffer, x1[i]); buffer.push_back(' ');
            append_integer(buffer, y1[i]); buffer.append(" metal");
            append_integer(buffer, z1[i] + 1); buffer.push_back('\n');
            if (buffer.size() >= (1 << 20)) flush_buffer(out, buffer);
        }
        buffer.append(")\n");
    }
    flush_buffer(out, buffer);
    if (!out) throw std::runtime_error("route_write: write failed");
}

std::size_t convert_instantgr_file(const char* input_path, const char* output_path) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open InstantGR route: ") + input_path);
    std::ofstream out(output_path, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("cannot open conversion output: ") + output_path);
    std::string buffer;
    buffer.reserve(1 << 20);

    std::size_t count = 0;
    std::string raw;
    while (std::getline(in, raw)) {
        std::istringstream fields(raw);
        std::int64_t xl, yl, zl, xh, yh, zh;
        std::string extra;
        const bool six_ints = (fields >> xl >> yl >> zl >> xh >> yh >> zh) && !(fields >> extra);
        if (!six_ints) {
            buffer.append(raw);
            if (!in.eof()) buffer.push_back('\n');
            if (buffer.size() >= (1 << 20)) flush_buffer(out, buffer);
            continue;
        }
        const std::int64_t x0 = xl * kGcellDbu + kGcellHalf;
        const std::int64_t y0 = yl * kGcellDbu + kGcellHalf;
        const std::int64_t x1 = xh * kGcellDbu + kGcellHalf;
        const std::int64_t y1 = yh * kGcellDbu + kGcellHalf;
        if (zl != zh) {
            const std::int64_t lo = zl < zh ? zl : zh;
            const std::int64_t hi = zl < zh ? zh : zl;
            for (std::int64_t z = lo; z < hi; ++z) {
                append_integer(buffer, x0); buffer.push_back(' ');
                append_integer(buffer, y0); buffer.append(" metal");
                append_integer(buffer, z + 1); buffer.push_back(' ');
                append_integer(buffer, x1); buffer.push_back(' ');
                append_integer(buffer, y1); buffer.append(" metal");
                append_integer(buffer, z + 2); buffer.push_back('\n');
                ++count;
            }
        } else {
            append_integer(buffer, x0); buffer.push_back(' ');
            append_integer(buffer, y0); buffer.append(" metal");
            append_integer(buffer, zl + 1); buffer.push_back(' ');
            append_integer(buffer, x1); buffer.push_back(' ');
            append_integer(buffer, y1); buffer.append(" metal");
            append_integer(buffer, zh + 1); buffer.push_back('\n');
            ++count;
        }
        if (buffer.size() >= (1 << 20)) flush_buffer(out, buffer);
    }
    flush_buffer(out, buffer);
    if (!out) throw std::runtime_error("route_convert: write failed");
    return count;
}

}  // namespace

extern "C" {

void* route_read(const char* path) {
    try {
        last_error.clear();
        return read_route_file(path);
    } catch (const std::exception& exc) {
        last_error = exc.what();
        return nullptr;
    }
}

void route_free(void* opaque) { delete static_cast<RouteDocument*>(opaque); }

const char* route_error() { return last_error.c_str(); }

std::size_t route_num_nets(const void* opaque) {
    return static_cast<const RouteDocument*>(opaque)->names.size();
}
std::size_t route_num_segments(const void* opaque) {
    return static_cast<const RouteDocument*>(opaque)->x0.size();
}
const std::int32_t* route_net_id(const void* opaque) {
    return static_cast<const RouteDocument*>(opaque)->net_id.data();
}
const std::int32_t* route_x0(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->x0.data(); }
const std::int32_t* route_y0(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->y0.data(); }
const std::int32_t* route_z0(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->z0.data(); }
const std::int32_t* route_x1(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->x1.data(); }
const std::int32_t* route_y1(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->y1.data(); }
const std::int32_t* route_z1(const void* opaque) { return static_cast<const RouteDocument*>(opaque)->z1.data(); }
const std::uint64_t* route_offsets(const void* opaque) {
    return static_cast<const RouteDocument*>(opaque)->offsets.data();
}
const char* route_net_name(const void* opaque, std::size_t index) {
    const auto& names = static_cast<const RouteDocument*>(opaque)->names;
    return index < names.size() ? names[index].c_str() : nullptr;
}

int route_write(const void* opaque, const char* path,
                const std::int32_t* x0, const std::int32_t* y0,
                const std::int32_t* z0, const std::int32_t* x1,
                const std::int32_t* y1, const std::int32_t* z1,
                std::size_t count) {
    try {
        last_error.clear();
        write_route_file(*static_cast<const RouteDocument*>(opaque), path, x0, y0, z0, x1, y1, z1, count);
        return 0;
    } catch (const std::exception& exc) {
        last_error = exc.what();
        return -1;
    }
}

std::int64_t route_convert_instantgr(const char* input_path, const char* output_path) {
    try {
        last_error.clear();
        return static_cast<std::int64_t>(convert_instantgr_file(input_path, output_path));
    } catch (const std::exception& exc) {
        last_error = exc.what();
        return -1;
    }
}

}  // extern "C"
