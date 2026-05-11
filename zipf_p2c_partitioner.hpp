#ifndef ZIPF_P2C_PARTITIONER_HPP
#define ZIPF_P2C_PARTITIONER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include <atomic>
#include <tbb/task_group.h>

class zipf_p2c_partitioner {
public:
    struct options {
        std::size_t shard_count = 64;
        std::size_t grain_size = 2048;
        std::size_t flush_threshold = 0;
        unsigned int num_threads = 0;
        bool assume_power_of_two_shards = true;
        bool reuse_threads = true;          // более не используется
        bool use_dual_hash = true;          // true = h1 и h2 из 64‑битного хеша
    };

    explicit zipf_p2c_partitioner(options opt) : opt_(std::move(opt)) {
        if (opt_.num_threads == 0)
            opt_.num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    ~zipf_p2c_partitioner() = default;

    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void run(const Items& items,
             KeyExtractor&& key_of,
             Hasher&& hasher,
             ShardSink&& sink) const {
        const std::size_t n = items.size();
        if (n == 0) return;

        const std::size_t S = normalize_shards(opt_.shard_count);
        const unsigned int P = opt_.num_threads;

        auto ranges = split_range(n, P);

        std::vector<std::function<void()>> tasks;
        tasks.reserve(P);
        for (unsigned int t = 0; t < P; ++t) {
            tasks.emplace_back([this, t, &ranges, &items, &key_of, &hasher, S, &sink]() {
                process_and_flush(ranges[t].first, ranges[t].second,
                                  items, key_of, hasher, S, sink);
            });
        }

        tbb::task_group tg;
        for (auto& task : tasks)
            tg.run(std::move(task));
        tg.wait();

        if constexpr (requires(ShardSink s) { s.finalize(); }) {
            sink.finalize();
        }
    }

private:
    options opt_;

    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void process_and_flush(std::size_t begin, std::size_t end,
                           const Items& items,
                           KeyExtractor& key_of,
                           Hasher& hasher,
                           std::size_t S,
                           ShardSink& sink) const {
        const std::size_t range_len = end - begin;
        if (range_len == 0) return;

        std::vector<std::vector<std::size_t>> local_bufs(S);
        constexpr std::size_t INITIAL_CAP = 256;
        for (auto& buf : local_bufs)
            buf.reserve(INITIAL_CAP);

        std::vector<std::size_t> cnt(S, 0);
        std::size_t* cnt_ptr = cnt.data();

        const bool pow2 = opt_.assume_power_of_two_shards;
        const bool dual_hash = opt_.use_dual_hash;
        const std::size_t mask = S - 1;

        if (pow2) {
            if (dual_hash) {
                for (std::size_t i = begin; i < end; ++i) {
                    const auto& item = items[i];
                    auto key = key_of(item);
                    auto full_hash = static_cast<std::size_t>(hasher(key));
                    std::size_t h1 = full_hash;
                    std::size_t h2 = full_hash >> 32;
                    std::size_t s1 = h1 & mask;
                    std::size_t s2 = h2 & mask;
                    std::size_t chosen = (cnt_ptr[s1] <= cnt_ptr[s2]) ? s1 : s2;
                    ++cnt_ptr[chosen];
                    auto& buf = local_bufs[chosen];
                    if (__builtin_expect(buf.size() == buf.capacity(), 0))
                        buf.reserve(buf.capacity() * 2);
                    buf.push_back(i);
                }
            } else {
                for (std::size_t i = begin; i < end; ++i) {
                    const auto& item = items[i];
                    auto key = key_of(item);
                    std::size_t h1 = static_cast<std::size_t>(hasher(key));
                    std::size_t h2 = mix_bits(h1);
                    std::size_t s1 = h1 & mask;
                    std::size_t s2 = h2 & mask;
                    std::size_t chosen = (cnt_ptr[s1] <= cnt_ptr[s2]) ? s1 : s2;
                    ++cnt_ptr[chosen];
                    auto& buf = local_bufs[chosen];
                    if (__builtin_expect(buf.size() == buf.capacity(), 0))
                        buf.reserve(buf.capacity() * 2);
                    buf.push_back(i);
                }
            }
        } else {
            if (dual_hash) {
                for (std::size_t i = begin; i < end; ++i) {
                    const auto& item = items[i];
                    auto key = key_of(item);
                    auto full_hash = static_cast<std::size_t>(hasher(key));
                    std::size_t h1 = full_hash;
                    std::size_t h2 = full_hash >> 32;
                    std::size_t s1 = h1 % S;
                    std::size_t s2 = h2 % S;
                    std::size_t chosen = (cnt_ptr[s1] <= cnt_ptr[s2]) ? s1 : s2;
                    ++cnt_ptr[chosen];
                    auto& buf = local_bufs[chosen];
                    if (__builtin_expect(buf.size() == buf.capacity(), 0))
                        buf.reserve(buf.capacity() * 2);
                    buf.push_back(i);
                }
            } else {
                for (std::size_t i = begin; i < end; ++i) {
                    const auto& item = items[i];
                    auto key = key_of(item);
                    std::size_t h1 = static_cast<std::size_t>(hasher(key));
                    std::size_t h2 = mix_bits(h1);
                    std::size_t s1 = h1 % S;
                    std::size_t s2 = h2 % S;
                    std::size_t chosen = (cnt_ptr[s1] <= cnt_ptr[s2]) ? s1 : s2;
                    ++cnt_ptr[chosen];
                    auto& buf = local_bufs[chosen];
                    if (__builtin_expect(buf.size() == buf.capacity(), 0))
                        buf.reserve(buf.capacity() * 2);
                    buf.push_back(i);
                }
            }
        }

        for (std::size_t s = 0; s < S; ++s) {
            auto& buf = local_bufs[s];
            if (!buf.empty()) {
                sink.consume(s, items, buf.data(), buf.data() + buf.size());
                buf.clear();
            }
        }
    }

    static std::size_t normalize_shards(std::size_t s) { return s < 1 ? 1 : s; }

    static std::vector<std::pair<std::size_t, std::size_t>>
    split_range(std::size_t total, unsigned int parts) {
        std::vector<std::pair<std::size_t, std::size_t>> ranges(parts);
        std::size_t base = total / parts;
        std::size_t rem  = total % parts;
        std::size_t start = 0;
        for (unsigned int i = 0; i < parts; ++i) {
            std::size_t size = base + (i < rem ? 1 : 0);
            ranges[i] = {start, start + size};
            start += size;
        }
        return ranges;
    }

    static std::size_t mix_bits(std::size_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
};

#endif // ZIPF_P2C_PARTITIONER_HPP