#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t RADIX = 1u << 16;

template <typename Function>
void parallel_for(std::int64_t count, int workers, Function function) {
    const std::int64_t nw = std::max<std::int64_t>(
        1, std::min<std::int64_t>(count, std::max(1, std::min(workers, 32))));
    if (nw == 1) {
        function(0, count, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nw));
    for (std::int64_t w = 0; w < nw; ++w) {
        const std::int64_t begin = count * w / nw;
        const std::int64_t end = count * (w + 1) / nw;
        threads.emplace_back([&, begin, end, w]() { function(begin, end, w); });
    }
    for (auto& thread : threads) thread.join();
}

inline std::uint64_t biased(std::int64_t value) {
    return static_cast<std::uint64_t>(value) ^ UINT64_C(0x8000000000000000);
}

}  // namespace

// Stable LSD radix passes implement lexicographic ordering without combining
// keys.  key0 is the highest-priority key; key3 is the lowest when present.
extern "C" int sort_i64_lex(const std::int64_t* key0,
                             const std::int64_t* key1,
                             const std::int64_t* key2,
                             const std::int64_t* key3,
                             std::int64_t* output,
                             std::int64_t count,
                             int nkeys,
                             int workers) {
    const std::int64_t* keys[4] = {key0, key1, key2, key3};
    if (count < 0 || output == nullptr || nkeys < 1 || nkeys > 4) return 1;
    for (int k = 0; k < nkeys; ++k) {
        if (keys[k] == nullptr) return 2;
    }
    if (count == 0) return 0;

    const std::int64_t nw = std::max<std::int64_t>(
        1, std::min<std::int64_t>(count, std::max(1, std::min(workers, 32))));
    std::vector<std::int64_t> source(static_cast<std::size_t>(count));
    std::vector<std::int64_t> destination(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) source[static_cast<std::size_t>(i)] = i;

    std::vector<std::uint64_t> histogram(static_cast<std::size_t>(nw) * RADIX);
    std::vector<std::uint64_t> position(static_cast<std::size_t>(nw) * RADIX);
    std::vector<std::uint64_t> bucket_start(RADIX);

    // LSD key order is the reverse of the requested priority order.
    for (int key_no = nkeys - 1; key_no >= 0; --key_no) {
        const std::int64_t* key = keys[key_no];
        for (int digit_pass = 0; digit_pass < 4; ++digit_pass) {
            std::fill(histogram.begin(), histogram.end(), 0);
            parallel_for(count, static_cast<int>(nw), [&](std::int64_t begin,
                                                           std::int64_t end,
                                                           std::int64_t worker) {
                std::uint64_t* local = histogram.data() +
                    static_cast<std::size_t>(worker) * RADIX;
                for (std::int64_t i = begin; i < end; ++i) {
                    const std::int64_t row = source[static_cast<std::size_t>(i)];
                    const std::uint64_t digit =
                        (biased(key[static_cast<std::size_t>(row)]) >> (digit_pass * 16)) &
                        (RADIX - 1);
                    ++local[static_cast<std::size_t>(digit)];
                }
            });

            std::uint64_t total = 0;
            for (std::size_t digit = 0; digit < RADIX; ++digit) {
                bucket_start[digit] = total;
                std::uint64_t within_bucket = 0;
                for (std::int64_t worker = 0; worker < nw; ++worker) {
                    const std::size_t slot = static_cast<std::size_t>(worker) * RADIX + digit;
                    position[slot] = total + within_bucket;
                    within_bucket += histogram[slot];
                }
                total += within_bucket;
            }

            parallel_for(count, static_cast<int>(nw), [&](std::int64_t begin,
                                                           std::int64_t end,
                                                           std::int64_t worker) {
                std::uint64_t* local = position.data() +
                    static_cast<std::size_t>(worker) * RADIX;
                for (std::int64_t i = begin; i < end; ++i) {
                    const std::int64_t row = source[static_cast<std::size_t>(i)];
                    const std::uint64_t digit =
                        (biased(key[static_cast<std::size_t>(row)]) >> (digit_pass * 16)) &
                        (RADIX - 1);
                    destination[static_cast<std::size_t>(local[static_cast<std::size_t>(digit)]++)] = row;
                }
            });
            source.swap(destination);
        }
    }
    std::copy(source.begin(), source.end(), output);
    return 0;
}
