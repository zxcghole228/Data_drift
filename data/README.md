# Демонстрационные данные

Исходные и сгенерированные датасеты не коммитятся. Каталоги `data/generated/` и `data/raw/`, а также файлы CSV/Parquet исключены через `.gitignore`.

## Схема

Генератор создаёт две **независимые** выборки: историческую `Reference` и текущую `Current`. Контрольный `Current` не является копией `Reference`.

| Колонка | Семантический тип | pandas dtype при генерации | Смысл и единицы |
| --- | --- | --- | --- |
| `age` | `numeric` | `float64` | Возраст в полных годах, диапазон 18–100 |
| `income` | `numeric` | `float64` | Месячный доход в условных денежных единицах, диапазон 0–1 000 000 |
| `region` | `categorical` | `object` | Один из `central`, `northwest`, `south`, `volga`, `siberia` |

`age` намеренно хранится как `float64`: так nullable-числа одинаково представлены в исходном DataFrame и после чтения CSV. Физические dtype CSV и Parquet могут различаться, но должны соответствовать семантическим типам из `configs/default.yaml`.

## Сценарии

| `--scenario` | Изменение только в Current |
| --- | --- |
| `none` | Дрейф не вносится; обе выборки независимо получены из одних распределений |
| `numeric` | К `age` добавляется 8 лет с ограничением диапазоном 18–100 |
| `categorical` | Доля `central` увеличивается с 0.30 до 0.55, доли остальных регионов уменьшаются |
| `missingness` | Доля пропусков `income` увеличивается с 2% до 8%, то есть на 6 процентных пунктов |
| `combined` | Одновременно применяются все три изменения |

Величину сдвига возраста и обе доли пропусков можно изменить аргументами CLI. Фактически применённые параметры сохраняются в `metadata.json`.

## Быстрый запуск для интеграции с загрузчиком

Из корня репозитория в активированном окружении выполнить:

```bash
python scripts/generate_demo_data.py --scenario none --format both
python scripts/generate_demo_data.py --scenario combined --format both
```

Будут созданы:

```text
data/generated/none/reference.csv
data/generated/none/current.csv
data/generated/none/reference.parquet
data/generated/none/current.parquet
data/generated/none/metadata.json
data/generated/combined/reference.csv
data/generated/combined/current.csv
data/generated/combined/reference.parquet
data/generated/combined/current.parquet
data/generated/combined/metadata.json
```

Эти пути можно сразу передать функциям загрузки Павла. CSV и Parquet внутри одного сценария содержат одни и те же строки.

## Параметры

```bash
python scripts/generate_demo_data.py --help
```

Основные аргументы:

- `--seed` — seed генератора, по умолчанию `42`;
- `--n-reference` — число строк Reference, по умолчанию `5000`;
- `--n-current` — число строк Current, по умолчанию `3000`;
- `--scenario` — один из пяти сценариев из таблицы;
- `--format` — `csv`, `parquet` или `both`;
- `--output-dir` — корневой каталог результата, по умолчанию `data/generated`;
- `--age-shift-years` — сдвиг возраста для `numeric`/`combined`, по умолчанию `8`;
- `--reference-income-missing-fraction` — базовая доля пропусков, по умолчанию `0.02`;
- `--drifted-income-missing-fraction` — доля пропусков при дрейфе, по умолчанию `0.08`.

Пример с другими размерами и seed:

```bash
python scripts/generate_demo_data.py \
  --scenario categorical \
  --seed 2026 \
  --n-reference 10000 \
  --n-current 4000 \
  --format parquet
```

Повторный запуск с одинаковыми параметрами создаёт одинаковые таблицы. Изменение scenario при неизменных seed и размерах не изменяет Reference, что позволяет сравнивать сценарии с общей исторической выборкой.

## Проверка

Запустить тесты генератора:

```bash
python -m pytest tests/unit/test_generate_demo_data.py
```

Тесты используют временный каталог и не добавляют данные в репозиторий.
