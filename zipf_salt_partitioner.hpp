// zipf_salt_partitioner.hpp
#ifndef ZIPF_SALT_PARTITIONER_HPP
#define ZIPF_SALT_PARTITIONER_HPP

#include <cstddef>
#include <vector>
#include <utility>
#include <thread>
#include <algorithm>
#include <future>
#include <unordered_map>
#include <cmath>

/**
 * @brief Партиционер для данных с распределением Ципфа (тяжёлые ключи).
 *
 * Основная идея: «соление» (salting) горячих ключей.
 * 1. Определяются ключи с аномально высокой частотой.
 * 2. Каждый такой ключ разбивается на несколько виртуальных под-ключей (key + salt),
 *    которые затем хешируются в разные шарды.
 * 3. Количество солей пропорционально частоте ключа, что даёт практически
 *    идеально равномерное заполнение всех шардов.
 *
 * Партиционер не хранит данные между вызовами; `unordered_map` используется
 * только как временная структура для анализа частот.
 */
class zipf_salt_partitioner {
public:
    struct options {
        std::size_t shard_count = 64;          // количество выходных шардов
        std::size_t grain_size = 2048;         // размер блока итераций (не используется в этой версии)
        unsigned int num_threads = 0;          // 0 = автоматически (число ядер)
        bool use_power_of_two_shards = true;   // использовать & (S-1) вместо %

        // Параметры обнаружения горячих ключей
        double hot_key_threshold_ratio = 0.01; // ключ "горячий", если его доля >= 1%
        double sample_rate = 0.1;              // доля элементов для оценки частот (0 = полный проход)
        std::size_t max_sample_size = 1'000'000;

        // Параметры соления
        bool deterministic_salt = false;       // true = соль от хеша элемента (не реализовано),
                                              // false = round-robin
        std::uint64_t salt_seed = 0x9E3779B9; // seed для комбинирования хешей
    };

    explicit zipf_salt_partitioner(options opt) : opt_(std::move(opt)) {
        if (opt_.num_threads == 0)
            opt_.num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    /**
     * Основной метод:
     * - items       : контейнер элементов (должен поддерживать size() и operator[])
     * - key_of      : функция извлечения ключа из элемента
     * - hasher      : хеш-функция для ключа (std::size_t hasher(const Key&))
     * - sink        : приёмник шардов (метод consume(shard, items, begin, end))
     *
     * Шардирование выполняется многопоточно, без блокировок между потоками.
     */
    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void run(const Items& items,
             KeyExtractor&& key_of,
             Hasher&& hasher,
             ShardSink&& sink) const {
        const std::size_t n = items.size();
        if (n == 0) return;

        using Key = std::decay_t<decltype(key_of(items[0]))>;

        // 1. Сбор частот ключей (с сэмплированием)
        auto freq = collect_frequencies(items, key_of);

        // 2. Определение горячих ключей и количества солей для каждого
        auto hot_info = build_hot_keys_info(freq, n);

        // 3. Параллельная раскладка по шардам
        const std::size_t S = normalize_shards(opt_.shard_count);
        const unsigned int P = opt_.num_threads;

        auto ranges = split_range(n, P);
        std::vector<std::future<void>> futures;
        futures.reserve(P);

        for (unsigned int t = 0; t < P; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                process_salted_range(ranges[t].first, ranges[t].second,
                                     items, key_of, hasher, S, hot_info, sink);
            }));
        }

        // Ожидание окончания всех потоков
        for (auto& f : futures) f.get();

        // Даём приёмнику возможность финализировать состояние
        if constexpr (requires(ShardSink s) { s.finalize(); }) {
            sink.finalize();
        }
    }

private:
    options opt_;

    // ----- Утилиты -----
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

    // Алгоритм смешивания двух хешей (аналог boost::hash_combine)
    static std::size_t combine_hashes(std::size_t h1, std::size_t h2, std::size_t seed) {
        h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        h1 ^= seed;
        // Дополнительное перемешивание для равномерности
        h1 = (h1 ^ (h1 >> 33)) * 0xff51afd7ed558ccdULL;
        h1 = (h1 ^ (h1 >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        h1 = h1 ^ (h1 >> 33);
        return h1;
    }

    // ----- Сбор частот ключей -----
    template <class Items, class KeyExtractor>
    auto collect_frequencies(const Items& items, KeyExtractor&& key_of) const {
        using Key = std::decay_t<decltype(key_of(items[0]))>;
        std::unordered_map<Key, std::size_t> freq;
        const std::size_t n = items.size();
        if (n == 0) return freq;

        std::size_t sample_size = (opt_.sample_rate > 0.0)
            ? std::min(static_cast<std::size_t>(n * opt_.sample_rate), opt_.max_sample_size)
            : n;
        if (sample_size == 0) sample_size = n;

        std::size_t step = std::max(std::size_t(1), n / sample_size);
        for (std::size_t i = 0; i < n; i += step) {
            ++freq[key_of(items[i])];
        }

        // Экстраполяция частот до полного объёма (если был шаг > 1)
        if (step > 1) {
            for (auto& [_, cnt] : freq) cnt *= step;
        }
        return freq;
    }

    // ----- Построение карты горячих ключей -----
    template <typename Key>
    auto build_hot_keys_info(const std::unordered_map<Key, std::size_t>& freq,
                             std::size_t total_count) const {
        std::unordered_map<Key, std::size_t> hot_salts;   // ключ -> количество солей

        if (total_count == 0) return hot_salts;

        const std::size_t S = normalize_shards(opt_.shard_count);
        const double avg_per_shard = static_cast<double>(total_count) / S;
        const std::size_t threshold = std::max(
            std::size_t(1),
            static_cast<std::size_t>(total_count * opt_.hot_key_threshold_ratio));

        for (const auto& [key, count] : freq) {
            if (count < threshold) continue;

            // Сколько солей нужно, чтобы записи этого ключа равномерно
            // распределились по шардам?
            std::size_t salts = static_cast<std::size_t>(std::ceil(count / avg_per_shard));
            if (salts > S) salts = S;   // не больше числа шардов
            if (salts < 1) salts = 1;
            hot_salts[key] = salts;
        }
        return hot_salts;
    }

    // ----- Потоковая обработка диапазона -----
    template <class Items, class KeyExtractor, class Hasher, class ShardSink>
    void process_salted_range(
        std::size_t begin, std::size_t end,
        const Items& items,
        KeyExtractor&& key_of,
        Hasher&& hasher,                // используем переданный хешер
        std::size_t S,
        const std::unordered_map<std::decay_t<decltype(key_of(items[0]))>, std::size_t>& hot_salts,
        ShardSink& sink) const {

        using Key = std::decay_t<decltype(key_of(items[0]))>;
        std::vector<std::vector<std::size_t>> local_buffers(S);
        std::size_t estimated = (end - begin) / S + 1;
        for (auto& buf : local_buffers) buf.reserve(estimated);

        // Round-robin счётчики соли для каждого горячего ключа (если не deterministic)
        std::unordered_map<Key, std::size_t> salt_counters;

        for (std::size_t i = begin; i < end; ++i) {
            const auto& item = items[i];
            const Key key = key_of(item);
            std::size_t shard;

            auto it = hot_salts.find(key);
            if (it == hot_salts.end()) {
                // Обычный ключ – простое хеш-шардирование
                std::size_t h = static_cast<std::size_t>(hasher(key));
                shard = (opt_.use_power_of_two_shards) ? (h & (S - 1)) : (h % S);
            } else {
                // Горячий ключ – разбиваем на it->second кусков
                std::size_t salt_count = it->second;
                std::size_t salt = 0;

                if (opt_.deterministic_salt) {
                    // Детерминированная соль: можно хешировать сам элемент, но здесь для простоты
                    // используем адрес элемента – подойдёт для демонстрации.
                    // В реальном коде замените на hasher(item) или другой стабильный хеш.
                    salt = reinterpret_cast<std::size_t>(&item) % salt_count;
                } else {
                    // Round-robin
                    salt = salt_counters[key]++ % salt_count;
                }

                // Получаем хеш ключа через предоставленную функцию
                std::size_t h_key = static_cast<std::size_t>(hasher(key));
                std::size_t h_salted = combine_hashes(h_key, salt, opt_.salt_seed);
                shard = (opt_.use_power_of_two_shards) ? (h_salted & (S - 1)) : (h_salted % S);
            }

            local_buffers[shard].push_back(i);
        }

        // Отправка накопленных индексов в приёмник
        for (std::size_t s = 0; s < S; ++s) {
            auto& buf = local_buffers[s];
            if (!buf.empty()) {
                sink.consume(s, items, buf.data(), buf.data() + buf.size());
                buf.clear();   // освобождаем память
            }
        }
    }
};

#endif // ZIPF_SALT_PARTITIONER_HPP