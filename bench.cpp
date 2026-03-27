// перепеши этот бенч на гугл бенчмарк. размер массива сделай как в предыдущем // sharded_partitioner_bench.cpp
// g++ -O3 -DNDEBUG -std=c++17 sharded_partitioner_bench.cpp -ltbb -fopenmp -o bench
// ./bench --n 20000000 --keys 1048576 --shards 64 --buckets 4096 --zipf 1.1 --iters 6
//
// Примечание: для корректного сравнения выставь одинаковое число потоков:
//   oneTBB: export TBB_NUM_THREADS=16
//   OpenMP: export OMP_NUM_THREADS=16

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>
// #include <oneapi/tbb/auto_partitioner.h>
// #include <oneapi/tbb/simple_partitioner.h>
// #include <oneapi/tbb/static_partitioner.h>
#include <oneapi/tbb/spin_mutex.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>



// ---------------------- utils ----------------------
static inline uint64_t ns_now() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static inline void clobber() {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#endif
}

// SplitMix64 for fast hashing
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Args {
    size_t n = 20'000'000;         // количество элементов
    size_t key_space = 1 << 20;    // число возможных ключей
    size_t shards = 64;            // число шардов (лучше степень 2)
    size_t buckets = 4096;         // размер массива счётчиков внутри каждого шарда
    double zipf = 1.10;            // параметр "горячести" ключей (>=1.0)
    int iters = 6;                 // повторов, берём best/median
    size_t grain = 4096;           // grain_size для TBB в baseline
    size_t flush = 256;            // flush_threshold для sharded
};

// ---------------------- workload ----------------------
// Sharded counters with lock (симулируем contention)
struct ShardedCounters {
    struct Shard {
        oneapi::tbb::spin_mutex m;
        std::vector<uint64_t> counters;
        Shard() = default;
    };

    std::vector<Shard> sh;
    size_t buckets;

    ShardedCounters(size_t shard_count, size_t buckets_)
        : sh(shard_count), buckets(buckets_) {
        for (auto& s : sh) s.counters.assign(buckets, 0);
    }

    inline void update_locked(size_t shard, uint32_t v) {
        auto& S = sh[shard];
        oneapi::tbb::spin_mutex::scoped_lock lock(S.m);
        // Простая функция обновления: инкремент в бакет
        S.counters[v & (buckets - 1)]++;
    }

    inline void update_locked_batch(size_t shard, const uint32_t* vals, size_t count) {
        auto& S = sh[shard];
        oneapi::tbb::spin_mutex::scoped_lock lock(S.m);
        for (size_t i = 0; i < count; ++i) {
            S.counters[vals[i] & (buckets - 1)]++;
        }
    }

    uint64_t checksum() const {
        uint64_t sum = 0;
        for (auto const& S : sh)
            for (auto v : S.counters) sum += v;
        return sum;
    }

    void reset() {
        for (auto& S : sh) std::fill(S.counters.begin(), S.counters.end(), 0);
    }
};

// ---------------------- data generator ----------------------
// Генерируем ключи с Zipf-подобным распределением (hot keys)
static std::vector<uint32_t> gen_zipf_keys(size_t n, size_t key_space, double s, uint64_t seed) {
    // Делаем таблицу CDF по рангу 1..key_space: p(k) ~ 1/k^s
    // Это O(key_space) по памяти и времени — нормально для microbench.
    std::vector<double> cdf(key_space);
    double H = 0.0;
    for (size_t k = 1; k <= key_space; ++k) H += 1.0 / std::pow((double)k, s);
    double acc = 0.0;
    for (size_t k = 1; k <= key_space; ++k) {
        acc += (1.0 / std::pow((double)k, s)) / H;
        cdf[k - 1] = acc;
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        double u = U(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        size_t rank = (size_t)std::distance(cdf.begin(), it) + 1; // 1..key_space
        // Преобразуем rank в key (можно просто rank-1)
        out[i] = (uint32_t)(rank - 1);
    }

    // Перемешаем, чтобы baseline partitioner-ы не получали случайно хорошую локальность по диапазону
    std::shuffle(out.begin(), out.end(), rng);
    return out;
}

// ---------------------- our "sharded partitioner" API ----------------------
template <typename ShardOf, typename BatchFunc>
static void ParallelForSharded_TBB(
    size_t from, size_t to,
    ShardOf&& shardOf,
    BatchFunc&& batch,
    size_t shardCount,
    size_t grainSize,
    size_t flushThreshold)
{
    if (to <= from) return;
    const size_t n = to - from;

    struct Local {
        std::vector<std::vector<uint32_t>> vals_by_shard; // храним значения (ключ) или то, что нужно обновлять
        explicit Local(size_t S) : vals_by_shard(S) {}
    };

    oneapi::tbb::enumerable_thread_specific<Local> tls([&] { return Local(shardCount); });

    auto flush_one = [&](size_t shard, std::vector<uint32_t>& buf) {
        if (buf.empty()) return;
        batch(shard, buf.data(), buf.size());
        buf.clear();
    };

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, n, grainSize),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            auto& st = tls.local();
            for (size_t off = r.begin(); off != r.end(); ++off) {
                const size_t i = from + off;
                const size_t shard = (size_t)shardOf(i);

                auto& buf = st.vals_by_shard[shard];
                // Здесь кладём полезную нагрузку (в нашем бенче это key)
                buf.push_back((uint32_t)shardOf.value(i)); // see wrapper below
                if (buf.size() >= flushThreshold) flush_one(shard, buf);
            }
        },
        oneapi::tbb::static_partitioner{}
    );

    tls.combine_each([&](Local& st) {
        for (size_t s = 0; s < shardCount; ++s) flush_one(s, st.vals_by_shard[s]);
    });
}

// Чтобы не вызывать shardOf(i) дважды (для shard id и для value),
// сделаем маленький wrapper для нашего бенча.
struct ShardOfKey {
    const uint32_t* keys;
    size_t shard_mask;
    inline size_t operator()(size_t i) const {
        uint64_t h = splitmix64((uint64_t)keys[i]);
        return (size_t)(h & shard_mask);
    }
    inline uint32_t value(size_t i) const { return keys[i]; }
};

// ---------------------- baseline: TBB parallel_for ----------------------
template <typename Partitioner>
static uint64_t run_tbb_baseline(
    const Args& a,
    const std::vector<uint32_t>& keys,
    ShardedCounters& counters,
    Partitioner&& part)
{
    const size_t mask = a.shards - 1;

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, keys.size(), a.grain),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                uint64_t h = splitmix64((uint64_t)keys[i]);
                size_t shard = (size_t)(h & mask);
                counters.update_locked(shard, keys[i]);
            }
        },
        part
    );

    return counters.checksum();
}

// ---------------------- our: TBB sharded batch ----------------------
static uint64_t run_tbb_sharded_batch(
    const Args& a,
    const std::vector<uint32_t>& keys,
    ShardedCounters& counters)
{
    const size_t mask = a.shards - 1;

    ShardOfKey shardOf{ keys.data(), mask };

    // Вызовем нашу схему: в batch берём локальную пачку и одним lock обновляем шард.
    // (важно: shardCount==a.shards)
    // Реализацию делаем напрямую, без "двойного shardOf".
    struct Local {
        std::vector<std::vector<uint32_t>> buf;
        explicit Local(size_t S) : buf(S) {}
    };

    oneapi::tbb::enumerable_thread_specific<Local> tls([&]{ return Local(a.shards); });

    auto flush_one = [&](size_t shard, std::vector<uint32_t>& b) {
        if (b.empty()) return;
        counters.update_locked_batch(shard, b.data(), b.size());
        b.clear();
    };

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, keys.size(), a.grain),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            auto& st = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                size_t shard = shardOf(i);
                auto& b = st.buf[shard];
                b.push_back(keys[i]);
                if (b.size() >= a.flush) flush_one(shard, b);
            }
        },
        oneapi::tbb::static_partitioner{}
    );

    tls.combine_each([&](Local& st){
        for (size_t s = 0; s < a.shards; ++s) flush_one(s, st.buf[s]);
    });

    return counters.checksum();
}

// ---------------------- OpenMP baseline ----------------------
static uint64_t run_omp_baseline(
    const Args& a,
    const std::vector<uint32_t>& keys,
    ShardedCounters& counters)
{
    const size_t mask = a.shards - 1;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)keys.size(); ++i) {
        uint64_t h = splitmix64((uint64_t)keys[(size_t)i]);
        size_t shard = (size_t)(h & mask);
        counters.update_locked(shard, keys[(size_t)i]);
    }

    return counters.checksum();
}

// ---------------------- OpenMP sharded batch (local buffers) ----------------------
static uint64_t run_omp_sharded_batch(
    const Args& a,
    const std::vector<uint32_t>& keys,
    ShardedCounters& counters)
{
    const size_t mask = a.shards - 1;

    #pragma omp parallel
    {
        std::vector<std::vector<uint32_t>> buf(a.shards);

        auto flush_one = [&](size_t shard) {
            auto& b = buf[shard];
            if (b.empty()) return;
            counters.update_locked_batch(shard, b.data(), b.size());
            b.clear();
        };

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)keys.size(); ++i) {
            uint32_t k = keys[(size_t)i];
            size_t shard = (size_t)(splitmix64((uint64_t)k) & mask);
            auto& b = buf[shard];
            b.push_back(k);
            if (b.size() >= a.flush) flush_one(shard);
        }

        // финальный слив
        for (size_t s = 0; s < a.shards; ++s) flush_one(s);
    }

    return counters.checksum();
}

// ---------------------- benchmark harness ----------------------
template <typename Fn>
static double measure_ms_best(const Args& a, ShardedCounters& counters, Fn&& fn, uint64_t& out_checksum) {
    std::vector<double> times;
    times.reserve(a.iters);

    uint64_t chk = 0;
    for (int it = 0; it < a.iters; ++it) {
        counters.reset();
        clobber();
        uint64_t t0 = ns_now();
        chk = fn();
        uint64_t t1 = ns_now();
        clobber();
        double ms = (double)(t1 - t0) / 1e6;
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    out_checksum = chk;
    // берём медиану (устойчивее), но можно best: times.front()
    return times[times.size()/2];
}

static void print_line(const std::string& name, double ms, double base_ms, uint64_t chk) {
    double speedup = base_ms / ms;
    std::cout << name << ": " << ms << " ms"
              << " | speedup vs base: " << speedup
              << " | checksum: " << chk
              << "\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto get = [&](const char* k) -> const char* {
            if (std::strcmp(argv[i], k) == 0 && i + 1 < argc) return argv[++i];
            return nullptr;
        };
        if (auto v = get("--n")) a.n = std::stoull(v);
        else if (auto v = get("--keys")) a.key_space = std::stoull(v);
        else if (auto v = get("--shards")) a.shards = std::stoull(v);
        else if (auto v = get("--buckets")) a.buckets = std::stoull(v);
        else if (auto v = get("--zipf")) a.zipf = std::stod(v);
        else if (auto v = get("--iters")) a.iters = std::stoi(v);
        else if (auto v = get("--grain")) a.grain = std::stoull(v);
        else if (auto v = get("--flush")) a.flush = std::stoull(v);
    }
    // sanity
    auto is_pow2 = [](size_t x){ return x && ((x & (x-1)) == 0); };
    if (!is_pow2(a.shards) || !is_pow2(a.buckets)) {
        std::cerr << "ERROR: --shards and --buckets should be power of two\n";
        std::exit(1);
    }
    return a;
}



// ---------------------- TBB parallel_reduce (локальные копии) ----------------------
class HistogramReducer {
public:
    std::vector<uint64_t> hist;
    size_t shards;
    size_t buckets;
    size_t shard_mask;
    size_t bucket_mask;
    const std::vector<uint32_t>* keys;  // указатель вместо ссылки

    HistogramReducer(size_t shards_, size_t buckets_, size_t shard_mask_, size_t bucket_mask_,
                     const std::vector<uint32_t>& keys_)
        : hist(shards_ * buckets_, 0), shards(shards_), buckets(buckets_),
          shard_mask(shard_mask_), bucket_mask(bucket_mask_), keys(&keys_) {}

    // Конструктор для создания "пустого" reducer-а при редукции (необходим для split)
    HistogramReducer(const HistogramReducer& other, oneapi::tbb::split)
        : hist(other.shards * other.buckets, 0), shards(other.shards), buckets(other.buckets),
          shard_mask(other.shard_mask), bucket_mask(other.bucket_mask), keys(other.keys) {}

    // Оператор обработки диапазона
    void operator()(const oneapi::tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            uint32_t key = (*keys)[i];
            size_t shard = splitmix64(key) & shard_mask;
            size_t bucket = key & bucket_mask;
            ++hist[shard * buckets + bucket];
        }
    }

    // Объединение двух reducer-ов
    void join(const HistogramReducer& other) {
        for (size_t i = 0; i < hist.size(); ++i) {
            hist[i] += other.hist[i];
        }
    }
};

#include <oneapi/tbb/parallel_reduce.h>   // добавлено


static uint64_t run_tbb_parallel_reduce(
    const Args& a,
    const std::vector<uint32_t>& keys,
    ShardedCounters& counters)
{
    const size_t shard_mask = a.shards - 1;
    const size_t bucket_mask = a.buckets - 1;

    HistogramReducer reducer(a.shards, a.buckets, shard_mask, bucket_mask, keys);
    // Вызываем parallel_reduce без присваивания — reducer изменится внутри
    oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<size_t>(0, keys.size(), a.grain),
        reducer,
        oneapi::tbb::auto_partitioner()
    );

    // Переносим результат в структуру ShardedCounters
    counters.reset();
    for (size_t s = 0; s < a.shards; ++s) {
        auto& shard_counters = counters.sh[s].counters;
        for (size_t b = 0; b < a.buckets; ++b) {
            shard_counters[b] = reducer.hist[s * a.buckets + b];
        }
    }

    return counters.checksum();
}


int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::cout << "n=" << a.n
              << " key_space=" << a.key_space
              << " shards=" << a.shards
              << " buckets=" << a.buckets
              << " zipf=" << a.zipf
              << " iters=" << a.iters
              << " grain=" << a.grain
              << " flush=" << a.flush
              << "\n";

    std::cout << "TBB_NUM_THREADS=" << (std::getenv("TBB_NUM_THREADS") ? std::getenv("TBB_NUM_THREADS") : "(unset)")
              << " | OMP_NUM_THREADS=" << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "(unset)")
              << "\n";

    auto keys = gen_zipf_keys(a.n, a.key_space, a.zipf, /*seed=*/12345);

    ShardedCounters counters(a.shards, a.buckets);

    // BASELINE: TBB auto_partitioner (часто дефолтный ориентир)
    uint64_t chk_base = 0;
    double base_ms = measure_ms_best(a, counters, [&]{
        oneapi::tbb::auto_partitioner p;
        return run_tbb_baseline(a, keys, counters, p);
    }, chk_base);

    std::cout << "\n--- Results (median of " << a.iters << " runs), base = TBB auto_partitioner ---\n";
    print_line("TBB auto_partitioner (BASE)", base_ms, base_ms, chk_base);

    // Остальные TBB partitioner-ы
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            oneapi::tbb::simple_partitioner p;
            return run_tbb_baseline(a, keys, counters, p);
        }, chk);
        print_line("TBB simple_partitioner", ms, base_ms, chk);
    }
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            oneapi::tbb::static_partitioner p;
            return run_tbb_baseline(a, keys, counters, p);
        }, chk);
        print_line("TBB static_partitioner", ms, base_ms, chk);
    }
    {
        oneapi::tbb::affinity_partitioner ap;
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            return run_tbb_baseline(a, keys, counters, ap);
        }, chk);
        print_line("TBB affinity_partitioner", ms, base_ms, chk);
    }

    // Наш “sharded batch”
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            return run_tbb_sharded_batch(a, keys, counters);
        }, chk);
        print_line("TBB SHARDED-BATCH (ours)", ms, base_ms, chk);
    }

    // OpenMP baseline
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            return run_omp_baseline(a, keys, counters);
        }, chk);
        print_line("OpenMP baseline (static)", ms, base_ms, chk);
    }

    // OpenMP sharded batch (для честности)
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            return run_omp_sharded_batch(a, keys, counters);
        }, chk);
        print_line("OpenMP SHARDED-BATCH", ms, base_ms, chk);
    }


     // TBB parallel_reduce (локальные копии)
    {
        uint64_t chk = 0;
        double ms = measure_ms_best(a, counters, [&]{
            return run_tbb_parallel_reduce(a, keys, counters);
        }, chk);
        print_line("TBB parallel_reduce (local histograms)", ms, base_ms, chk);
    }

    std::cout << "\nNotes:\n"
              << "- If zipf is high (e.g., 1.1..1.3) => more hot keys => more contention => SHARDED-BATCH should win.\n"
              << "- Increase --flush to reduce lock acquisitions (often helps until cache pressure grows).\n"
              << "- Ensure shards and buckets are power-of-two for fast masking.\n";

    return 0;
}