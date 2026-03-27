#include "key_shard.hpp"

#include <iostream>
#include <vector>
#include <functional>   // для std::hash
#include <cassert>

// Пример sink-функтора: собирает элементы каждого шарда в отдельный вектор
template<typename T>
struct vector_sink {
    std::vector<std::vector<T>> sharded_data;

    vector_sink(std::size_t num_shards) : sharded_data(num_shards) {}

    // consume вызывается для каждого блока индексов одного шарда
    template<typename Items>
    void consume(std::size_t shard, const Items& items,
                 const std::size_t* begin, const std::size_t* end) const
    {
        // Здесь мы можем модифицировать sharded_data, хотя метод константный.
        // В реальном коде можно сделать sharded_data mutable, либо передавать указатель.
        // Для простоты примера сделаем небольшое отступление: используем const_cast,
        // но в продакшене лучше передавать sink по ссылке с неконстантным методом.
        auto& target = const_cast<std::vector<T>&>(sharded_data[shard]);
        target.reserve(target.size() + (end - begin));
        for (auto it = begin; it != end; ++it) {
            target.push_back(items[*it]);
        }
    }

    // Опциональная финализация
    void finalize() const {
        std::cout << "Finalizing sink. Shard sizes:" << std::endl;
        for (std::size_t s = 0; s < sharded_data.size(); ++s) {
            std::cout << "  shard " << s << ": " << sharded_data[s].size() << " elements" << std::endl;
        }
    }
};

int main() {
    // 1. Подготовка данных: вектор целых чисел
    std::vector<int> data;
    for (int i = 0; i < 10000; ++i) {
        data.push_back(i);
    }

    // 2. Настройки partitioner: 16 шардов (степень двойки), порог сброса 128
    key_shard_partitioner::options opt;
    opt.shard_count = 16;
    opt.grain_size = 512;
    opt.flush_threshold = 128;
    opt.assume_power_of_two_shards = true;

    key_shard_partitioner partitioner(opt);

    // 3. Создаём sink (будет заполнен элементами пошардно)
    vector_sink<int> sink(opt.shard_count);

    // 4. Запускаем разбиение
    partitioner.run(data,
                    [](const int& x) { return x; },                    // KeyExtractor – сам элемент
                    [](int key) { return std::hash<int>{}(key); },     // Hasher
                    sink);                                              // ShardSink

    // 5. Проверяем, что все элементы распределились без потерь
    std::size_t total = 0;
    for (const auto& vec : sink.sharded_data) {
        total += vec.size();
    }
    assert(total == data.size());
    std::cout << "Total elements processed: " << total << std::endl;

    return 0;
}