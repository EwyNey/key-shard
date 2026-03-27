cmake_minimum_required(VERSION 3.15)
project(KeyShardPartitionerExample LANGUAGES CXX)

# Найти пакет oneTBB
find_package(TBB REQUIRED)

# Исполняемый файл примера
add_executable(example main.cpp)

# Подключаем заголовочный файл (он лежит рядом)
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Линкуемся с TBB
target_link_libraries(example PRIVATE TBB::tbb)

# Включить поддержку C++17 (или новее) для requires в constexpr if
set_target_properties(example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)



# Исполняемый файл примера
add_executable(bench bench.cpp)

# Подключаем заголовочный файл (он лежит рядом)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Линкуемся с TBB
target_link_libraries(bench PRIVATE TBB::tbb)

# Включить поддержку C++17 (или новее) для requires в constexpr if
set_target_properties(bench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)