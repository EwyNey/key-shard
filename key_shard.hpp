#ifndef KEY_SHARD_PARTITIONER_HPP
#define KEY_SHARD_PARTITIONER_HPP

#include <cstddef>
#include <vector>
#include <type_traits>
#include <utility>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>

class key_shard_partitioner {
public:
    struct options {
        std::size_t shard_count = 64;          // лучше степенью двойки
        std::size_t grain_size  = 2048;        // размер блока итераций для parallel_for
        std::size_t flush_threshold = 256;     // сколько элементов накопить в локальном буфере шара, прежде чем сливать
        bool assume_power_of_two_shards = true; // если shard_count степень 2, можно быстрее считать shard id
    };

    explicit key_shard_partitioner(options opt ) : opt_(opt) {}

    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void run(const Items& items,
             KeyExtractor&& key_of,
             Hasher&& hasher,
             ShardSink&& sink) const
    {
        static_assert(
            std::is_invocable_v<KeyExtractor, typename Items::value_type const&>,
            "KeyExtractor must be callable: key_of(item) -> key"
        );

        const std::size_t n = items.size();
        if (n == 0) return;

        const std::size_t S = normalize_shards_(opt_.shard_count);

        struct local_state {
            std::vector<std::vector<std::size_t>> idx_by_shard; // храним индексы (дешево копировать/передавать)
            explicit local_state(std::size_t S) : idx_by_shard(S) {}
        };

        oneapi::tbb::enumerable_thread_specific<local_state> tls([&]{
            return local_state(S);
        });

        auto flush_one = [&](std::size_t shard, std::vector<std::size_t>& buf) {
            if (buf.empty()) return;
            // sink.consume(shard, items, indices_begin, indices_end)
            sink.consume(shard, items, buf.data(), buf.data() + buf.size());
            buf.clear();
        };

        auto flush_all = [&](local_state& st) {
            for (std::size_t s = 0; s < S; ++s) {
                flush_one(s, st.idx_by_shard[s]);
            }
        };

        // 1) parallel_for по индексам с static_partitioner:
        //    - меньше overhead на сплиты
        //    - предсказуемая работа каждого worker-а
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, n, opt_.grain_size),
            [&](const oneapi::tbb::blocked_range<std::size_t>& r) {
                auto& st = tls.local();

                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& item = items[i];
                    const auto key   = key_of(item);

                    std::size_t h = static_cast<std::size_t>(hasher(key));
                    std::size_t shard = shard_id_(h, S);

                    auto& buf = st.idx_by_shard[shard];
                    buf.push_back(i);

                    if (buf.size() >= opt_.flush_threshold) {
                        flush_one(shard, buf);
                    }
                }

                // Можно не вызывать flush_all тут, чтобы не делать много мелких flush-ов.
                // Но тогда буферы будут жить до конца — это нормально.
            },
            oneapi::tbb::static_partitioner{}
        );

        // 2) финальный слив всех локальных буферов
        tls.combine_each([&](local_state& st) {
            flush_all(st);
        });

        // 3) (опционально) финализация sink-а
        if constexpr (requires(ShardSink s) { s.finalize(); }) {
            sink.finalize();
        }
    }

private:
    options opt_;

    static std::size_t normalize_shards_(std::size_t s) {
        if (s < 1) return 1;
        return s;
    }

    std::size_t shard_id_(std::size_t h, std::size_t S) const {
        if (opt_.assume_power_of_two_shards) {
            // если S — степень 2, shard = h & (S-1)
            return h & (S - 1);
        }
        return h % S;
    }
};

#endif // KEY_SHARD_PARTITIONER_HPP