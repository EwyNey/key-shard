import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# -------------------------------------------------------------------
# НАСТРОЙКИ
# -------------------------------------------------------------------
USE_JSON = False               # True = читать results.json, False = парсить консольный вывод
JSON_FILE = 'results.json'
CONSOLE_FILE = 'out.txt'    # Файл с выводом команды ./partitioner_benchmark

# -------------------------------------------------------------------
# 1. ЧТЕНИЕ ДАННЫХ
# -------------------------------------------------------------------
records = []

if USE_JSON:
    import json
    with open(JSON_FILE) as f:
        data = json.load(f)
    for bm in data.get('benchmarks', []):
        name = bm['name']
        # Ожидаем формат "BM_SomeAlg/1000"
        parts = name.split('/')
        if len(parts) < 2:
            continue
        alg = parts[0]
        try:
            size = int(parts[1])
        except ValueError:
            continue
        time_ns = bm['real_time']  # в наносекундах? Google Benchmark по умолчанию возвращает наносекунды
        time_ms = time_ns / 1_000_000.0
        records.append({
            'algorithm': alg,
            'size': size,
            'time_ms': time_ms
        })
else:
    with open(CONSOLE_FILE) as f:
        lines = f.readlines()

    # Паттерн для нового формата:
    # BM_TBB_Auto/1000/iterations:20   2374407 ns   549727 ns   20   bytes_per_second=...
    pattern = re.compile(
        r'^(BM_\w+)/(\d+)/iterations:\d+\s+(\d+)\s+ns\s+\d+\s+ns\s+\d+'
    )
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        alg = m.group(1)
        size = int(m.group(2))
        time_ns = int(m.group(3))
        time_ms = time_ns / 1_000_000.0
        records.append({
            'algorithm': alg,
            'size': size,
            'time_ms': time_ms
        })

if not records:
    print("Ошибка: не удалось извлечь ни одной записи. Проверьте формат файла.")
    sys.exit(1)

df = pd.DataFrame(records)
# Округлим время для читаемости
df['time_ms'] = df['time_ms'].round(3)
print(f"Загружено {len(df)} записей.")

# -------------------------------------------------------------------
# 2. ЭКСПОРТ CSV ДЛЯ EXCEL
# -------------------------------------------------------------------
df.to_csv('raw_data.csv', index=False, float_format='%.3f')

# Сводная таблица: строки = алгоритм, столбцы = размер данных, значения = время
pivot = df.pivot_table(index='algorithm', columns='size', values='time_ms', aggfunc='mean')
pivot.to_csv('pivot_for_excel.csv', float_format='%.3f')
print("CSV сохранены: raw_data.csv, pivot_for_excel.csv")

# -------------------------------------------------------------------
# 3. ПОСТРОЕНИЕ ГРАФИКОВ
# -------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Список алгоритмов и размеров
algorithms = sorted(df['algorithm'].unique())
sizes = sorted(df['size'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

# 3.1. Групповая столбчатая диаграмма: сравнение алгоритмов для каждого размера
plt.figure(figsize=(12, 7))
# Расположение столбцов
x = np.arange(len(sizes))
width = 0.8 / len(algorithms)

for i, alg in enumerate(algorithms):
    alg_data = df[df['algorithm'] == alg].sort_values('size')
    # если какого-то размера нет, заполняем NaN, чтобы не сдвигать столбцы
    vals = []
    for s in sizes:
        row = alg_data[alg_data['size'] == s]
        if not row.empty:
            vals.append(row['time_ms'].values[0])
        else:
            vals.append(np.nan)
    plt.bar(x + i * width, vals, width, label=alg, color=colors[i])

plt.xlabel('Входной размер (элементов)')
plt.ylabel('Время (ms)')
plt.title('Сравнение времени выполнения алгоритмов')
plt.xticks(x + width * (len(algorithms)-1)/2, sizes)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('bar_comparison_by_size.png')
plt.close()
print("График 'bar_comparison_by_size.png' сохранён.")

# 3.2. Отдельные графики для каждого алгоритма (время от размера)
plt.figure(figsize=(10, 6))
for i, alg in enumerate(algorithms):
    alg_data = df[df['algorithm'] == alg].sort_values('size')
    plt.plot(alg_data['size'], alg_data['time_ms'],
             marker='o', label=alg, color=colors[i], linewidth=2)

plt.xlabel('Входной размер (элементов)')
plt.ylabel('Время (ms)')
plt.title('Зависимость времени от размера данных')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('line_time_vs_size.png')
plt.close()
print("График 'line_time_vs_size.png' сохранён.")

# 3.3. Тепловая карта: алгоритм × размер = время
pivot_heatmap = df.pivot_table(index='algorithm', columns='size', values='time_ms')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap, annot=True, fmt='.2f', cmap='viridis_r', cbar_kws={'label': 'Time (ms)'})
plt.title('Тепловая карта времени выполнения')
plt.xlabel('Входной размер')
plt.ylabel('Алгоритм')
plt.tight_layout()
plt.savefig('heatmap_algorithms.png')
plt.close()
print("График 'heatmap_algorithms.png' сохранён.")

print("\n=== ГОТОВО ===")