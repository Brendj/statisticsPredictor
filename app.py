import pandas as pd
import numpy as np

# Количество записей в датасете
num_entries = 30000

# Создаем словарь с данными
data = {
    "Вид спорта": np.random.choice(["Футбол", "Легкая атлетика"], num_entries),
    "Вес": np.random.randint(40, 100, num_entries),
    "Рост": np.random.randint(140, 200, num_entries),
    "Разряд": np.random.randint(1, 6, num_entries),
    "Возраст": np.random.randint(10, 18, num_entries),
    "Текущее место": np.random.randint(1, 20, num_entries),
    "Предыдущее место": np.random.randint(1, 20, num_entries),
    "Количество голов": np.random.randint(0, 5, num_entries),
    "Коэффициент статистики": np.random.uniform(0, 1, num_entries),
    "Позиция": np.random.choice(["Вратарь", "Защитник", "Полузащитник", "Нападающий"], num_entries),
    "Количество соревнований": np.random.randint(1, 50, num_entries)  # Добавляем количество соревнований
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Заменяем количество голов и позицию на NaN для легкой атлетики
df.loc[df["Вид спорта"] == "Легкая атлетика", ["Количество голов", "Позиция", "Количество соревнований"]] = np.nan

# Сохраняем DataFrame в файл
df.to_csv('dataset.csv', index=False)

print("Обновленный датасет сохранен в файл 'dataset.csv'")
