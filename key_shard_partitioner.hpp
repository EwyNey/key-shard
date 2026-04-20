// key_shard_partitioner


#ifndef KEY_SHARD_PARTITIONER_HPP
#define KEY_SHARD_PARTITIONER_HPP

#include <cstddef>
#include <vector>
#include <type_traits>
#include <utility>
#include <thread>
#include <algorithm>
#include <atomic>
#include <memory>
#include <functional>
#include <future>


// ============================================================================
// key_shard_partitioner (версия с thread_local буферами и прямым сбросом)
// ============================================================================
class key_shard_partitioner {
public:
    struct options {
        std::size_t shard_count = 64;
        std::size_t grain_size = 2048;
        std::size_t flush_threshold = 0;        // не используется
        bool assume_power_of_two_shards = true;
        unsigned int num_threads = 0;
        bool parallel_final_flush = true;       // игнорируется, всегда параллельно
        bool reuse_threads = true;
    };

    explicit key_shard_partitioner(options opt) : opt_(opt) {
        if (opt_.num_threads == 0) {
            opt_.num_threads = std::max(1u, std::thread::hardware_concurrency());
        }
        // Потоки будут создаваться для каждого вызова run() (проще и нет оверхеда на синхронизацию)
    }

    // Основной метод
    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void run(const Items& items,
             KeyExtractor&& key_of,
             Hasher&& hasher,
             ShardSink&& sink) const {
        const std::size_t n = items.size();
        if (n == 0) return;

        const std::size_t S = normalize_shards_(opt_.shard_count);
        const unsigned int P = opt_.num_threads;

        // Разбиваем диапазон индексов
        std::vector<std::pair<std::size_t, std::size_t>> ranges = split_range(n, P);

        // Запускаем потоки
        std::vector<std::future<void>> futures;
        futures.reserve(P);
        for (unsigned int t = 0; t < P; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                process_range_and_flush(ranges[t].first, ranges[t].second,
                                        items, key_of, hasher, S, sink);
            }));
        }

        // Ожидаем завершения всех потоков
        for (auto& f : futures) {
            f.get();
        }

        if constexpr (requires(ShardSink s) { s.finalize(); }) {
            sink.finalize();
        }
    }

private:
    options opt_;

    static std::size_t normalize_shards_(std::size_t s) {
        return s < 1 ? 1 : s;
    }

    std::size_t shard_id_(std::size_t h, std::size_t S) const {
        if (opt_.assume_power_of_two_shards) {
            return h & (S - 1);
        }
        return h % S;
    }

    static std::vector<std::pair<std::size_t, std::size_t>> split_range(std::size_t total, unsigned int parts) {
        std::vector<std::pair<std::size_t, std::size_t>> ranges(parts);
        std::size_t base = total / parts;
        std::size_t rem = total % parts;
        std::size_t start = 0;
        for (unsigned int i = 0; i < parts; ++i) {
            std::size_t size = base + (i < rem ? 1 : 0);
            ranges[i] = {start, start + size};
            start += size;
        }
        return ranges;
    }

    // Основная функция, выполняемая в каждом потоке
    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void process_range_and_flush(std::size_t begin, std::size_t end,
                                 const Items& items,
                                 KeyExtractor&& key_of,
                                 Hasher&& hasher,
                                 std::size_t S,
                                 ShardSink& sink) const {
        // Локальные буферы для этого потока
        std::vector<std::vector<std::size_t>> local_buffers(S);
        // Предварительное резервирование
        std::size_t estimated = (end - begin) / S + 1;
        for (auto& buf : local_buffers) {
            buf.reserve(estimated);
        }

        // Фаза сбора
        for (std::size_t block_start = begin; block_start < end; block_start += opt_.grain_size) {
            std::size_t block_end = std::min(block_start + opt_.grain_size, end);
            for (std::size_t i = block_start; i < block_end; ++i) {
                const auto& item = items[i];
                const auto key = key_of(item);
                std::size_t h = static_cast<std::size_t>(hasher(key));
                std::size_t shard = shard_id_(h, S);
                local_buffers[shard].push_back(i);
            }
        }

        // Фаза сброса (без дополнительной синхронизации между потоками)
        for (std::size_t s = 0; s < S; ++s) {
            auto& buf = local_buffers[s];
            if (!buf.empty()) {
                // Передаём сразу массив индексов, sink сам извлечёт ключи и захватит мьютекс
                sink.consume(s, items, buf.data(), buf.data() + buf.size());
                buf.clear();   // не обязательно, но для чистоты
            }
        }
    }
};


#endif // KEY_SHARD_PARTITIONER_HPP


