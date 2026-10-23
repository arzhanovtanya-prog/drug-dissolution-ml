
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ЗАГОЛОВКОВ РЕЗУЛЬТАТОВ
# ============================================================

def print_heading(number, title):
    print(f"\n{'=' * 65}")
    print(f"{number}. {title}")
    print("=" * 65)


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

# 1.1. Определение пути к файлу dataset_616.xlsl
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "dataset_616.xlsx")

# 1.2. Загрузка таблицы
df = pd.read_excel(file_path)

print(f"Файл загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")


# ============================================================
# 2. ПЕРВИЧНАЯ ДИАГНОСТИКА ДАТАСЕТА
# ============================================================

# 2.1. Проверка пропущенных значений
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print_heading("2.1", "Пропущенные значения")

if missing.empty:
    print("Пропущенных значений нет.")
else:
    print(missing.to_string())


# 2.2. Типы данных и количество уникальных значений
diagnostics = pd.DataFrame({
    "Тип данных": df.dtypes.astype(str),
    "Количество уникальных значений": df.nunique(dropna=False),
})

print_heading("2.2", "Типы данных и уникальные значения")
print(diagnostics.to_string())


# 2.3. Проверка целевой переменной
if "Dissolution" not in df.columns:
    raise ValueError("В датасете отсутствует столбец 'Dissolution'.")

print_heading("2.3", "Распределение целевой переменной")
print(df["Dissolution"].describe().to_string())


# 2.4. Проверка границ целевой переменной
print_heading("2.4", "Проверка границ целевой переменной")

d = pd.to_numeric(df["Dissolution"], errors="coerce")

print(f"Минимум: {d.min():.6f}")
print(f"Максимум: {d.max():.6f}")
print(f"Отрицательных значений: {(d < 0).sum()}")
print(f"Значений выше 1: {(d > 1).sum()}")


# 2.5. Количество экспериментальных серий и групп препаратов
if "Experimental_index" not in df.columns:
    raise ValueError(
        "В датасете отсутствует столбец 'Experimental_index'."
    )

if "Drug_Group" not in df.columns:
    raise ValueError("В датасете отсутствует столбец 'Drug_Group'.")

print_heading("2.5", "Экспериментальные серии и группы препаратов")
print(f"Количество экспериментальных серий: "
      f"{df['Experimental_index'].nunique()}")
print(f"Количество групп препаратов: {df['Drug_Group'].nunique()}")


# 2.6. Проверка повторяющихся строк
duplicate_count = df.duplicated().sum()

print_heading("2.6", "Повторяющиеся строки")
print(f"Количество полностью повторяющихся строк: {duplicate_count}")



# ============================================================
# 3. ПРОВЕРКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

print_heading("3", "Проверка и подготовка данных")

# 3.1. Проверка обязательных столбцов
print("\n3.1. Проверка обязательных столбцов")

required_columns = [
    "Dissolution",
    "Experimental_index",
    "Drug_Group",
]

missing_required = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        f"В датасете отсутствуют обязательные столбцы: "
        f"{missing_required}"
    )

print("Все обязательные столбцы присутствуют.")


# 3.2. Проверка целевой переменной
print("\n3.2. Проверка целевой переменной")

df["Dissolution"] = pd.to_numeric(
    df["Dissolution"],
    errors="coerce"
)

if df["Dissolution"].isna().any():
    invalid_count = df["Dissolution"].isna().sum()
    raise ValueError(
        f"В целевой переменной обнаружено {invalid_count} "
        f"пропущенных или нечисловых значений."
    )

print("Целевая переменная содержит только числовые значения.")
print(f"Количество наблюдений: {len(df)}")



# 3.3. Формирование списка исключаемых столбцов
print("\n3.3. Определение столбцов, исключаемых из признаков")

excluded_columns = [
    "Dissolution",
    "Experimental_index",
    "Drug_Group",
    "_API_candidate",
]

excluded_columns = [
    column for column in excluded_columns
    if column in df.columns
]

print(f"Исключаемые столбцы: {excluded_columns}")


# 3.4. Формирование матрицы признаков и целевой переменной
print("\n3.4. Формирование матрицы признаков")

X = df.drop(columns=excluded_columns).copy()
y = df["Dissolution"].copy()
groups = df["Experimental_index"].copy()

print(f"Количество строк: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Количество экспериментальных серий: {groups.nunique()}")


# 3.5. Проверка типов данных признаков
print("\n3.5. Проверка типов данных признаков")

non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric_columns:
    raise TypeError(
        "В матрице признаков обнаружены нечисловые столбцы: "
        f"{non_numeric_columns}. "
        "Необходимо определить способ их обработки."
    )

print("Все признаки имеют числовой тип.")
print("Подготовка данных завершена.")



# ============================================================
# 4. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================================

print_heading("4", "Разделение данных на обучающую и тестовую выборки")

# 4.1. Разделение по экспериментальным сериям
print("\n4.1. Разделение по экспериментальным сериям")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

print("Разделение выполнено.")
print(f"Обучающая выборка: {len(X_train)} строк")
print(f"Тестовая выборка: {len(X_test)} строк")


# 4.2. Проверка разделения экспериментальных серий
print("\n4.2. Проверка пересечения экспериментальных серий")

train_groups = set(groups_train.unique())
test_groups = set(groups_test.unique())

overlapping_groups = train_groups.intersection(test_groups)

if overlapping_groups:
    raise ValueError(
        "Ошибка разделения: некоторые экспериментальные серии "
        "присутствуют одновременно в обучающей и тестовой выборках."
    )

print(f"Серий в обучающей выборке: {len(train_groups)}")
print(f"Серий в тестовой выборке: {len(test_groups)}")
print(f"Пересекающихся серий: {len(overlapping_groups)}")
print("Проверка разделения завершена.")



# ============================================================
# 5. ОБУЧЕНИЕ МОДЕЛИ LIGHTGBM
# ============================================================

print_heading("5", "Обучение модели LightGBM")

# 5.1. Задание параметров модели
print("\n5.1. Параметры модели")

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_child_samples=20,
    random_state=42,
    verbosity=-1
)

print("Количество деревьев: 100")
print("Скорость обучения: 0.05")
print("Максимальная глубина дерева: 3")
print("Минимальное число объектов в листе: 20")
print("Случайное состояние: 42")


# 5.2. Обучение модели
print("\n5.2. Обучение на обучающей выборке")

model.fit(X_train, y_train)

print("Обучение модели LightGBM завершено.")
print(f"Использовано строк: {len(X_train)}")
print(f"Использовано признаков: {X_train.shape[1]}")



# ============================================================
# 6. ОЦЕНКА КАЧЕСТВА МОДЕЛИ
# ============================================================

# 6.1. Расчёт предсказаний
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# 6.2. Расчёт метрик качества
train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)
test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


# 6.3. Вывод метрик качества
print_heading("6", "Качество модели")

print(f"{'Метрика':<30}{'Обучающая выборка':>20}"
      f"{'Тестовая выборка':>20}")
print("-" * 70)
print(f"{'RMSE':<30}{train_rmse:>20.4f}{test_rmse:>20.4f}")
print(f"{'R²':<30}{train_r2:>20.4f}{test_r2:>20.4f}")


# ============================================================
# 7. АНАЛИЗ ОШИБОК МОДЕЛИ
# ============================================================

# 7.1. Формирование таблицы ошибок
test_results = df.iloc[test_idx].copy()

test_results["Predicted"] = y_test_pred
test_results["Error"] = (
    test_results["Predicted"] - test_results["Dissolution"]
)
test_results["Absolute_Error"] = test_results["Error"].abs()


# 7.2. Анализ ошибок по экспериментальным сериям
group_errors = (
    test_results
    .groupby(
        ["Experimental_index", "Drug_Group"],
        dropna=False
    )
    .agg(
        Number_of_points=("Dissolution", "size"),
        Mean_Absolute_Error=("Absolute_Error", "mean"),
        RMSE=(
            "Error",
            lambda values: np.sqrt(np.mean(values ** 2))
        ),
        Mean_Actual_Dissolution=("Dissolution", "mean"),
        Mean_Predicted_Dissolution=("Predicted", "mean"),
    )
    .reset_index()
    .sort_values("Mean_Absolute_Error", ascending=False)
)

# 7.2.1. Вывод серий с наибольшей средней абсолютной ошибкой
print_heading("7.2.1", "Серии с наибольшей средней абсолютной ошибкой")
print(group_errors.head(10).to_string(index=False))


# 7.3. Анализ характеристик пяти наиболее проблемных серий
worst_series = group_errors.head(5)["Experimental_index"].tolist()

# 7.3.1. Выбор характеристик экспериментальных серий
characteristic_columns = [
    "Experimental_index",
    "Drug_Group",
    "pH",
    "wDrug",
    "wExci1",
    "Particle_size",
    "Preparation_Method",
    "Preparation_T",
    "Equivalent_dosage",
]

available_characteristic_columns = [
    column for column in characteristic_columns
    if column in df.columns
]

# 7.3.2. Формирование таблицы характеристик
worst_series_data = (
    df.loc[
        df["Experimental_index"].isin(worst_series),
        available_characteristic_columns
    ]
    .drop_duplicates()
)

# 7.3.3. Вывод характеристик проблемных серий
print_heading("7.3", "Характеристики пяти наиболее проблемных серий")

if worst_series_data.empty:
    print("Данные для выбранных серий не найдены.")
else:
    print(worst_series_data.to_string(index=False))


# 7.4. Анализ отдельных предсказаний с наибольшей ошибкой
# 7.4.1. Выбор двадцати точек с наибольшей абсолютной ошибкой
top_20_errors = test_results.sort_values(
    "Absolute_Error",
    ascending=False
).head(20)

print_heading("7.4", "20 предсказаний с наибольшей абсолютной ошибкой")

print(
    top_20_errors[
        [
            "Experimental_index",
            "Drug_Group",
            "Dissolution",
            "Predicted",
            "Error",
            "Absolute_Error",
        ]
    ].to_string(index=False)
)


# 7.5. Анализ предсказаний для экспериментальных серий 71 и 73
# 7.5.1. Отбор серий из тестовой выборки
selected_test_series = [71, 73]

selected_test_results = test_results.loc[
    test_results["Experimental_index"].isin(selected_test_series)
].copy()

# 7.5.2. Вывод фактических и предсказанных значений
print_heading("7.5", "Предсказания для экспериментальных серий 71 и 73")

if selected_test_results.empty:
    print("Серии 71 и 73 не обнаружены в тестовой выборке.")
else:
    print(
        selected_test_results[
            [
                "Experimental_index",
                "Drug_Group",
                "Dissolution",
                "Predicted",
                "Error",
                "Absolute_Error",
            ]
        ].sort_values(
            ["Experimental_index", "Dissolution"]
        ).to_string(index=False)
    )


# 7.6. Анализ предсказаний для экспериментальной серии 72
# 7.6.1. Формирование предсказаний для обучающей выборки
train_results = df.iloc[train_idx].copy()

train_results["Predicted"] = y_train_pred
train_results["Error"] = (
    train_results["Predicted"] - train_results["Dissolution"]
)
train_results["Absolute_Error"] = train_results["Error"].abs()

# 7.6.2. Вывод фактических и предсказанных значений для серии 72
series_72_train = train_results.loc[
    train_results["Experimental_index"] == 72
]

print_heading("7.6", "Предсказания для экспериментальной серии 72")

if series_72_train.empty:
    print("Серия 72 не обнаружена в обучающей выборке.")
else:
    print(
        series_72_train[
            [
                "Experimental_index",
                "Drug_Group",
                "Dissolution",
                "Predicted",
                "Error",
                "Absolute_Error",
            ]
        ].sort_values("Dissolution").to_string(index=False)
    )


# ============================================================
# 8. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ЭКСПЕРИМЕНТАЛЬНЫХ СЕРИЙ
# ============================================================

# 8.1. Сравнение экспериментальных серий 71–74
series_71_74 = df.loc[
    df["Experimental_index"].isin([71, 72, 73, 74])
].copy()

comparison_columns = [
    "pH",
    "wDrug",
    "wExci1",
    "Particle_size",
    "Preparation_Method",
    "Preparation_T",
    "Equivalent_dosage",
    "Dissolution",
]

available_comparison_columns = [
    column for column in comparison_columns
    if column in series_71_74.columns
]

if series_71_74.empty:
    print("\nСерии 71–74 не найдены в датасете.")
else:
    # Числовые характеристики усредняем, категориальные сохраняем.
    aggregation = {}

    for column in available_comparison_columns:
        if pd.api.types.is_numeric_dtype(series_71_74[column]):
            aggregation[column] = "mean"
        else:
            aggregation[column] = "first"

    experiment_comparison = (
        series_71_74
        .groupby(
            ["Experimental_index", "Drug_Group"],
            dropna=False
        )
        .agg(aggregation)
        .reset_index()
    )

    # 8.1.1. Определение принадлежности серий к выборкам
    experiment_comparison["Dataset"] = (
        experiment_comparison["Experimental_index"]
        .map(
            lambda series_id:
            "Train" if series_id in train_groups else "Test"
        )
    )

    # 8.1.2. Вывод таблицы сравнения
    print_heading("8.1", "Сравнение экспериментальных серий 71–74")
    print(experiment_comparison.to_string(index=False))


# 8.2. Сравнение характеристик экспериментальных серий 71–76
series_71_76 = df.loc[
    df["Experimental_index"].isin([71, 72, 73, 74, 75, 76])
].copy()

features_to_compare = [
    "Experimental_index",
    "Drug_Group",
    "pH",
    "wDrug",
    "wExci1",
    "Particle_size",
    "Preparation_Method",
    "Preparation_T",
    "Equivalent_dosage",
    "Drug_XLogP3",
    "Drug_HBD",
    "Drug_HBA",
    "Drug_RB",
    "Drug_LogS",
    "Exci_Mw1",
    "Exci_sigement1",
    "Exci_u1",
    "Exci_e1",
    "Exci_Mw2",
    "Dissolution",
]

available_features_to_compare = [
    column for column in features_to_compare
    if column in series_71_76.columns
]

if series_71_76.empty:
    print("\nСерии 71–76 не найдены в датасете.")
else:
    # 8.2.1. Определение принадлежности серий к выборкам
    series_71_76["Dataset"] = (
        series_71_76["Experimental_index"]
        .map(
            lambda series_id:
            "Train" if series_id in train_groups else "Test"
        )
    )

    # 8.2.2. Вывод таблицы характеристик
    output_columns = available_features_to_compare.copy()

    if "Dataset" not in output_columns:
        output_columns.append("Dataset")

    print_heading("8.2", "Характеристики экспериментальных серий 71–76")
    print(series_71_76[output_columns].to_string(index=False))


# 8.3. Поиск аналогов для серий 71–73 в обучающей выборке
for subsection, series_id in enumerate([71, 72, 73], start=1):

    # 8.3.1–8.3.3. Отбор серии и поиск аналогов
    target_series = df.loc[
        df["Experimental_index"] == series_id
    ]

    if target_series.empty:
        print(f"\nСерия {series_id} не найдена.")
        continue

    drug_groups_for_series = (
        target_series["Drug_Group"].dropna().unique()
    )

    if len(drug_groups_for_series) == 0:
        print(
            f"\nДля серии {series_id} "
            "не удалось определить группу препарата."
        )
        continue

    target_drug_group = drug_groups_for_series[0]

    analogs = df.loc[
        (df["Drug_Group"] == target_drug_group)
        & (df["Experimental_index"].isin(train_groups))
        & (df["Experimental_index"] != series_id)
    ].copy()

    # 8.3.4. Вывод найденных аналогов
    print_heading(
        f"8.3.{subsection}",
        f"Аналоги для экспериментальной серии {series_id}"
    )

    if analogs.empty:
        print(
            "В обучающей выборке не найдены другие серии "
            "этой группы препарата."
        )
    else:
        analog_columns = [
            "Experimental_index",
            "Drug_Group",
            "pH",
            "wDrug",
            "wExci1",
            "Particle_size",
            "Preparation_Method",
            "Preparation_T",
            "Equivalent_dosage",
            "Dissolution",
        ]

        available_analog_columns = [
            column for column in analog_columns
            if column in analogs.columns
        ]

        print(analogs[available_analog_columns].to_string(index=False))


# 8.4. Анализ экспериментальных серий с итраконазолом
itraconazole_mask = (
    df["Drug_Group"]
    .astype(str)
    .str.contains("itraconazole", case=False, na=False)
)

itraconazole_data = df.loc[itraconazole_mask].copy()

if itraconazole_data.empty:
    print(
        "\nВ столбце Drug_Group не найдены записи, "
        "содержащие 'itraconazole'."
    )
else:
    # 8.4.1. Определение принадлежности серий к выборкам
    itraconazole_data["Dataset"] = (
        itraconazole_data["Experimental_index"]
        .map(
            lambda series_id:
            "Train" if series_id in train_groups else "Test"
        )
    )

    # 8.4.2. Вывод отдельных строк экспериментов
    itraconazole_columns = [
        "Experimental_index",
        "Drug_Group",
        "pH",
        "wDrug",
        "wExci1",
        "Particle_size",
        "Preparation_Method",
        "Preparation_T",
        "Equivalent_dosage",
        "Dissolution",
        "Dataset",
    ]

    available_itraconazole_columns = [
        column for column in itraconazole_columns
        if column in itraconazole_data.columns
    ]

    print_heading("8.4", "Экспериментальные серии с итраконазолом")
    print(
        itraconazole_data[available_itraconazole_columns]
        .to_string(index=False)
    )

    # 8.4.3. Формирование сводной таблицы по сериям
    summary_columns = [
        column for column in [
            "pH",
            "wDrug",
            "wExci1",
            "Particle_size",
            "Preparation_T",
            "Equivalent_dosage",
            "Dissolution",
        ]
        if column in itraconazole_data.columns
    ]

    if summary_columns:
        itraconazole_summary = (
            itraconazole_data
            .groupby(
                ["Experimental_index", "Drug_Group", "Dataset"],
                dropna=False
            )[summary_columns]
            .mean()
            .reset_index()
        )

        # 8.4.4. Вывод сводных средних значений
        print_heading(
            "8.4.1",
            "Сводные средние значения по сериям итраконазола"
        )

        print(itraconazole_summary.to_string(index=False))


# ============================================================
# 9. АНАЛИЗ ОШИБОК В ЗАВИСИМОСТИ ОТ pH
# ============================================================

# 9.1. Проверка наличия столбца pH
if "pH" not in test_results.columns:
    print("\nАнализ по pH невозможен: столбец pH отсутствует.")
else:
    # 9.2. Подготовка данных для анализа
    test_results["pH_numeric"] = pd.to_numeric(
        test_results["pH"],
        errors="coerce"
    )

    ph_data = test_results.dropna(subset=["pH_numeric"]).copy()

    if ph_data.empty:
        print("\nАнализ по pH невозможен: нет числовых значений pH.")
    else:
        # 9.3. Формирование таблицы ошибок по pH
        ph_errors = (
            ph_data
            .groupby("pH_numeric")
            .agg(
                Number_of_points=("Dissolution", "size"),
                Mean_Absolute_Error=("Absolute_Error", "mean"),
                RMSE=(
                    "Error",
                    lambda values: np.sqrt(np.mean(values ** 2))
                ),
                Mean_Actual_Dissolution=("Dissolution", "mean"),
                Mean_Predicted_Dissolution=("Predicted", "mean"),
            )
            .reset_index()
            .sort_values("pH_numeric")
        )

        # 9.4. Вывод результатов анализа по pH
        print_heading("9", "Ошибки модели при разных значениях pH")
        print(ph_errors.to_string(index=False))


# ============================================================
# 10. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================

# 10.1. Формирование таблицы важности признаков
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

# 10.2. Вывод двадцати наиболее важных признаков
print_heading("10", "20 наиболее важных признаков")
print(feature_importance.head(20).to_string(index=False))


# ============================================================
# 11. ПРИЗНАКИ, ИСПОЛЬЗУЕМЫЕ МОДЕЛЬЮ
# ============================================================

# 11.1. Вывод полного списка признаков
print_heading("11", "Признаки, используемые моделью")
print(f"Всего признаков: {len(X.columns)}")

for number, column in enumerate(X.columns, start=1):
    print(f"{number}. {column}")



