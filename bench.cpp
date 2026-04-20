#include "key_shard_partitioner.hpp"

#include <benchmark/benchmark.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/spin_mutex.h>

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------- utils ----------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Глобальные параметры (кроме n)
struct GlobalArgs {
    size_t key_space = 1 << 20;
    size_t shards = 64;
    size_t buckets = 4096;
    double zipf = 1.10;
    size_t grain = 4096;
    size_t flush = 0;
} g_args;

// ---------------------- workload: шардированные счётчики ----------------------
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

// ---------------------- генерация ключей ----------------------
static std::vector<uint32_t> gen_zipf_keys(size_t n, size_t key_space, double s, uint64_t seed) {
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
        size_t rank = (size_t)std::distance(cdf.begin(), it) + 1;
        out[i] = (uint32_t)(rank - 1);
    }

    std::shuffle(out.begin(), out.end(), rng);
    return out;
}

// ---------------------- единый приёмник для всех тестов ----------------------
struct BatchSink {
    ShardedCounters& counters;

    void consume(std::size_t shard,
                 const std::vector<uint32_t>& items,
                 const std::size_t* idx_begin,
                 const std::size_t* idx_end) {
        if (idx_begin == idx_end) return;
        size_t count = idx_end - idx_begin;
        alignas(64) static thread_local std::vector<uint32_t> key_buf;
        key_buf.clear();
        key_buf.reserve(count);
        for (auto it = idx_begin; it != idx_end; ++it) {
            key_buf.push_back(items[*it]);
        }
        counters.update_locked_batch(shard, key_buf.data(), count);
    }
};

// ---------------------- функции запуска (исправленные) ----------------------

template <typename Partitioner>
static uint64_t run_tbb_baseline(const GlobalArgs& args,
                                 const std::vector<uint32_t>& keys,
                                 ShardedCounters& counters,
                                 Partitioner&& part) {
    const size_t mask = args.shards - 1;
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, keys.size(), args.grain),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                uint64_t h = splitmix64((uint64_t)keys[i]);
                size_t shard = (size_t)(h & mask);
                auto& S = counters.sh[shard];
                oneapi::tbb::spin_mutex::scoped_lock lock(S.m);
                S.counters[keys[i] & (counters.buckets - 1)]++;
            }
        },
        part
    );
    return counters.checksum();
}

static uint64_t run_tbb_sharded_batch(const GlobalArgs& args,
                                      const std::vector<uint32_t>& keys,
                                      ShardedCounters& counters) {
    const size_t mask = args.shards - 1;
    struct Local {
        std::vector<std::vector<std::size_t>> idx_buf;
        explicit Local(size_t S) : idx_buf(S) {}
    };

    oneapi::tbb::enumerable_thread_specific<Local> tls([&]{ return Local(args.shards); });
    BatchSink sink{counters};

    auto flush_one = [&](size_t shard, std::vector<std::size_t>& b) {
        if (b.empty()) return;
        sink.consume(shard, keys, b.data(), b.data() + b.size());
        b.clear();
    };

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, keys.size(), args.grain),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            auto& st = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                size_t shard = (size_t)(splitmix64((uint64_t)keys[i]) & mask);
                auto& b = st.idx_buf[shard];
                b.push_back(i);
            }
        },
        oneapi::tbb::static_partitioner{}
    );

    tls.combine_each([&](Local& st) {
        for (size_t s = 0; s < args.shards; ++s) {
            flush_one(s, st.idx_buf[s]);
        }
    });

    return counters.checksum();
}

static uint64_t run_omp_baseline(const GlobalArgs& args,
                                 const std::vector<uint32_t>& keys,
                                 ShardedCounters& counters) {
    const size_t mask = args.shards - 1;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)keys.size(); ++i) {
        uint64_t h = splitmix64((uint64_t)keys[(size_t)i]);
        size_t shard = (size_t)(h & mask);
        auto& S = counters.sh[shard];
        oneapi::tbb::spin_mutex::scoped_lock lock(S.m);
        S.counters[keys[(size_t)i] & (counters.buckets - 1)]++;
    }
    return counters.checksum();
}

static uint64_t run_omp_sharded_batch(const GlobalArgs& args,
                                      const std::vector<uint32_t>& keys,
                                      ShardedCounters& counters) {
    const size_t mask = args.shards - 1;
    BatchSink sink{counters};

    #pragma omp parallel
    {
        std::vector<std::vector<std::size_t>> idx_buf(args.shards);

        auto flush_one = [&](size_t shard) {
            auto& b = idx_buf[shard];
            if (b.empty()) return;
            sink.consume(shard, keys, b.data(), b.data() + b.size());
            b.clear();
        };

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)keys.size(); ++i) {
            size_t shard = (size_t)(splitmix64((uint64_t)keys[i]) & mask);
            idx_buf[shard].push_back((size_t)i);
        }

        for (size_t s = 0; s < args.shards; ++s) {
            flush_one(s);
        }
    }
    return counters.checksum();
}

static uint64_t run_key_shard_partitioner(const GlobalArgs& args,
                                          const std::vector<uint32_t>& keys,
                                          ShardedCounters& counters) {
    key_shard_partitioner::options opt;
    opt.shard_count = args.shards;
    opt.grain_size = args.grain;
    opt.flush_threshold = args.flush;
    opt.assume_power_of_two_shards = true;
    opt.parallel_final_flush = true;

    key_shard_partitioner part(opt);

    auto key_extractor = [](const uint32_t& key) -> uint32_t {
        return key;
    };
    auto hasher = [](uint32_t key) -> std::size_t {
        return static_cast<std::size_t>(splitmix64(key));
    };

    BatchSink sink{counters};
    part.run(keys, key_extractor, hasher, sink);

    return counters.checksum();
}

// ---------------------- бенчмарки с параметризацией по n ----------------------

// Вспомогательный макрос для регистрации с диапазоном
#define BENCHMARK_RANGE(func) \
    benchmark::RegisterBenchmark(#func, func) \
        ->ArgsProduct({benchmark::CreateRange(1000, 20000000, /*multi=*/10)}) \
        ->Iterations(20)

// Функции бенчмарков принимают benchmark::State&
static void BM_TBB_Auto(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        oneapi::tbb::auto_partitioner p;
        uint64_t chk = run_tbb_baseline(g_args, keys, counters, p);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_TBB_Simple(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        oneapi::tbb::simple_partitioner p;
        uint64_t chk = run_tbb_baseline(g_args, keys, counters, p);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_TBB_Static(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        oneapi::tbb::static_partitioner p;
        uint64_t chk = run_tbb_baseline(g_args, keys, counters, p);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_TBB_Affinity(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    oneapi::tbb::affinity_partitioner ap;
    for (auto _ : state) {
        counters.reset();
        uint64_t chk = run_tbb_baseline(g_args, keys, counters, ap);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_TBB_ShardedBatch(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        uint64_t chk = run_tbb_sharded_batch(g_args, keys, counters);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_OpenMP_Baseline(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        uint64_t chk = run_omp_baseline(g_args, keys, counters);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_OpenMP_ShardedBatch(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        uint64_t chk = run_omp_sharded_batch(g_args, keys, counters);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

static void BM_KeyShardPartitioner(benchmark::State& state) {
    size_t n = state.range(0);
    auto keys = gen_zipf_keys(n, g_args.key_space, g_args.zipf, 12345);
    ShardedCounters counters(g_args.shards, g_args.buckets);
    for (auto _ : state) {
        counters.reset();
        uint64_t chk = run_key_shard_partitioner(g_args, keys, counters);
        benchmark::DoNotOptimize(chk);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint32_t));
}

// ---------------------- main ----------------------
int main(int argc, char** argv) {
    // Чтение глобальных параметров из переменных окружения
    if (const char* env = std::getenv("KEY_SPACE")) g_args.key_space = std::stoull(env);
    if (const char* env = std::getenv("SHARDS")) g_args.shards = std::stoull(env);
    if (const char* env = std::getenv("BUCKETS")) g_args.buckets = std::stoull(env);
    if (const char* env = std::getenv("ZIPF")) g_args.zipf = std::stod(env);
    if (const char* env = std::getenv("GRAIN")) g_args.grain = std::stoull(env);
    if (const char* env = std::getenv("FLUSH")) g_args.flush = std::stoull(env);

    // Регистрируем бенчмарки с варьируемым n
    // Используем набор значений: 1000, 10k, 100k, 1M, 2M, 5M, 10M, 20M
    std::vector<int64_t> range_n = {1000, 10000, 100000, 1000000,
                                    2000000, 5000000, 10000000, 20000000};

    // Регистрация с фиксированным числом итераций (20)
    for (auto n : range_n) {
        benchmark::RegisterBenchmark("BM_TBB_Auto", BM_TBB_Auto)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_TBB_Simple", BM_TBB_Simple)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_TBB_Static", BM_TBB_Static)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_TBB_Affinity", BM_TBB_Affinity)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_TBB_ShardedBatch", BM_TBB_ShardedBatch)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_OpenMP_Baseline", BM_OpenMP_Baseline)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_OpenMP_ShardedBatch", BM_OpenMP_ShardedBatch)
            ->Args({n})->Iterations(20);
        benchmark::RegisterBenchmark("BM_KeyShardPartitioner", BM_KeyShardPartitioner)
            ->Args({n})->Iterations(20);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}