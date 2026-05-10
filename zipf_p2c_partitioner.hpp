// zipf_p2c_partitioner.hpp
#ifndef ZIPF_P2C_PARTITIONER_HPP
#define ZIPF_P2C_PARTITIONER_HPP

#include <cstddef>
#include <vector>
#include <utility>
#include <thread>
#include <algorithm>
#include <future>

class zipf_p2c_partitioner {
public:
    struct options {
        std::size_t shard_count = 64;
        std::size_t grain_size = 2048;          // не используется (бенчмарк переопределяет)
        std::size_t flush_threshold = 0;        // не используется
        unsigned int num_threads = 0;
        bool assume_power_of_two_shards = true;
        bool parallel_final_flush = true;       // не используется
        bool reuse_threads = true;              // не используется

        // Дополнительный seed для второго хеша (можно менять при необходимости)
        std::uint64_t salt_for_second_hash = 0x9E3779B97F4A7C15ULL;
    };

    explicit zipf_p2c_partitioner(options opt) : opt_(std::move(opt)) {
        if (opt_.num_threads == 0)
            opt_.num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

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
        std::vector<std::future<void>> futures;
        futures.reserve(P);

        for (unsigned int t = 0; t < P; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                process_p2c_range(ranges[t].first, ranges[t].second,
                                  items, key_of, hasher, S, sink);
            }));
        }

        for (auto& f : futures) f.get();

        if constexpr (requires(ShardSink s) { s.finalize(); }) {
            sink.finalize();
        }
    }

private:
    options opt_;

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

    // Быстрый миксер бит для получения второго хеша из первого
    static std::size_t mix_bits(std::size_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void process_p2c_range(std::size_t begin, std::size_t end,
                           const Items& items,
                           KeyExtractor&& key_of,
                           Hasher&& hasher,
                           std::size_t S,
                           ShardSink& sink) const {
        // Локальный массив счётчиков занятости шардов и буферы индексов
        std::vector<std::size_t> shard_sizes(S, 0);
        std::vector<std::vector<std::size_t>> local_buffers(S);
        std::size_t estimated = (end - begin) / S + 1;
        for (auto& buf : local_buffers) buf.reserve(estimated);

        for (std::size_t i = begin; i < end; ++i) {
            const auto& item = items[i];
            const auto key = key_of(item);
            std::size_t h1 = static_cast<std::size_t>(hasher(key));

            // Второй хеш, независимый от первого
            std::size_t h2 = mix_bits(h1 ^ opt_.salt_for_second_hash);

            // Индексы двух шардов
            std::size_t s1 = (opt_.assume_power_of_two_shards) ? (h1 & (S - 1)) : (h1 % S);
            std::size_t s2 = (opt_.assume_power_of_two_shards) ? (h2 & (S - 1)) : (h2 % S);

            // Выбор менее заполненного шарда в локальном буфере
            std::size_t chosen = (shard_sizes[s1] <= shard_sizes[s2]) ? s1 : s2;

            local_buffers[chosen].push_back(i);
            ++shard_sizes[chosen];
        }

        // Сброс накопленных индексов в приёмник
        for (std::size_t s = 0; s < S; ++s) {
            auto& buf = local_buffers[s];
            if (!buf.empty()) {
                sink.consume(s, items, buf.data(), buf.data() + buf.size());
                buf.clear();
            }
        }
    }
};

#endif // ZIPF_P2C_PARTITIONER_HPP