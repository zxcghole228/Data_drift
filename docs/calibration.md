# Воспроизводимая demo-калибровка

Калибровочный runner исследует поведение уже реализованного публичного
`analyze(reference, current, config)`. Он не содержит отдельной статистической
логики и не подменяет pipeline.

## Полный запуск

Из корня репозитория в активированном Python 3.13 окружении:

```bash
python scripts/run_calibration.py
```

По умолчанию выполняются 20 фиксированных seed для каждого случая:

- размеры Current: 500, 1 200 и 3 000 строк;
- Reference: 5 000 строк;
- контроль без внесённого дрейфа;
- сдвиг `age`: 2, 4 и 8 лет;
- доля пропусков `income` в Current: 4%, 8% и 15% при 2% в Reference;
- фиксированный категориальный сдвиг `region`;
- комбинированный приёмочный сценарий.

Это 27 случаев и 540 вызовов `analyze`. Adversarial Validation включён, поэтому
полный прогон может занять несколько минут. Флаг `--disable-adversarial`
предназначен только для быстрой диагностики runner и не заменяет полный прогон.

Быстрый smoke test с отдельным, игнорируемым Git output-каталогом:

```bash
python scripts/run_calibration.py \
  --seeds 3 7 \
  --current-sizes 500 \
  --age-shifts 2 8 \
  --missing-fractions 0.04 0.15 \
  --n-reference 1000 \
  --disable-adversarial \
  --output-dir outputs/calibration-smoke \
  --overwrite
```

## Артефакты

Полный запуск атомарно создаёт в `report/experiments/`:

| Файл | Содержимое |
| --- | --- |
| `calibration_runs.csv` | Одна строка на решение метода в каждом seed/случае |
| `calibration_summary.csv` | Наблюдаемые доли алертов и средние/медианные значения |
| `calibration_metadata.json` | Сетка, версии, время, commit SHA, config SHA-256 и признак dirty worktree |

Существующие файлы защищены от случайной замены. Для осознанного повторного
запуска используется `--overwrite`.

В `calibration_summary.csv` различаются три типа наблюдаемой доли:

- `false_alert_rate` — алерты в независимом контроле без внесённого дрейфа;
- `detection_rate` — срабатывания метода на целевом для него сдвиге;
- `non_target_alert_rate` — срабатывания проверки, которой данный сценарий не
  адресован напрямую.

У `alert=None` решения нет, поэтому такая строка не входит в знаменатель
`n_decisions`. Это особенно важно для неприменимых или отключённых проверок.

## Как читать результат

При одинаковой величине сдвига больший Current обычно повышает статистическую
мощность. При фиксированном размере более сильный сдвиг обычно повышает долю
обнаружений. Это ожидаемое направление, а не требование монотонности для каждой
отдельной случайной выборки.

Двадцать seed дают первичную инженерную оценку поведения demo-сценария, но не
точную оценку будущего false-positive rate. Например, при истинной доле около
5% один дополнительный алерт заметно меняет результат на выборке из 20
повторов. Production-калибровка требует исторических данных конкретного
источника, реальных размеров батчей, анализа сезонности и цены ложных и
пропущенных тревог.

`configs/demo_calibrated.yaml` поэтому явно помечен как demo-specific. Общие
безразмерные пороги PSI/JS не превращаются от этого в универсальные правила, а
Wasserstein для `age` задаётся отдельно в годах и не применяется к `income`.

## Зафиксированный приёмочный сценарий для Павла

Одинаковые входные файлы и конфигурация для CLI, HTML и Streamlit:

```bash
python scripts/generate_demo_data.py \
  --scenario combined \
  --seed 42 \
  --n-reference 5000 \
  --n-current 1200 \
  --age-shift-years 8 \
  --reference-income-missing-fraction 0.02 \
  --drifted-income-missing-fraction 0.08 \
  --format both

python -m data_drift_guardian \
  --reference data/generated/combined/reference.parquet \
  --current data/generated/combined/current.parquet \
  --config configs/demo_calibrated.yaml \
  --json-output outputs/acceptance/combined.json \
  --html-output outputs/acceptance/combined.html \
  --overwrite
```

В Streamlit нужно загрузить те же два Parquet-файла и
`configs/demo_calibrated.yaml`, не меняя Adversarial override. Значения
`effective_config`, Data Quality, drift-метрик, Cramér's V, OOF ROC-AUC,
статусов и алертов должны совпасть с JSON и HTML. Время выполнения и визуальное
форматирование частью числового равенства не являются.
