# Data Drift Guardian

Учебный проект системы проверки качества табличных данных и обнаружения сдвигов распределений.

## Текущее состояние

Реализованы воспроизводимый генератор независимых Reference/Current-выборок,
загрузка CSV/Parquet, валидация конфигурации и схемы, Data Quality, KS,
Wasserstein, χ², Cramér's V, PSI, Jensen–Shannon divergence,
Benjamini–Hochberg и общий `analyze()` pipeline. Контракт 0.2 поддерживает
персональные distance-пороги признаков и возвращает `effective_config`.
Adversarial Validation возвращает OOF ROC-AUC по LightGBM, значения фолдов,
агрегированные важности и опциональный групповой split. Streamlit-интерфейс
загружает две таблицы и YAML-конфигурацию, показывает статусы, проверки,
алерты, ML-результат и сопоставимые распределения. Командный интерфейс запускает
тот же анализ для локальных CSV/Parquet и атомарно сохраняет полный результат
контракта 0.2 в JSON и необязательный автономный HTML с теми же статусами,
порогами и интерактивными распределениями. Итоговый текст проекта, Docker-smoke
test и скринкаст пока остаются незавершёнными требованиями к 27.09.

Цель первой недели — первый совместный проход данных через систему и проверка основных модулей. Это промежуточный результат, а не финальная сдача.

## Что должно быть в итоговом решении

- Загрузка CSV/Parquet и проверка схемы.
- Контроль типов, пропусков, дубликатов и диапазонов.
- KS-тест, Wasserstein, χ², PSI и Jensen–Shannon Divergence.
- Adversarial Validation на LightGBM с ROC-AUC на отложенных данных и важностями признаков.
- Единый интерфейс `analyze(reference, current, config) -> dict`.
- Streamlit-дашборд, графики распределений и объяснимые алерты.
- Воспроизводимый генератор демонстрационных данных, notebook и unit-тесты.
- Итоговый HTML- или PDF-отчёт, Dockerfile, инструкция запуска и скринкаст 2–5 минут.

Описание возможностей и ограничений: [docs/architecture.md](docs/architecture.md).

## Команда и ветки

| Ветка | Назначение | План |
| --- | --- | --- |
| `main` | Общая основа и проверенные изменения | [Порядок работы](CONTRIBUTING.md) |
| `mikhail` | Статистика, Adversarial Validation, эксперименты | [План Михаила](docs/plans/mikhail.md) |
| `pavel` | Загрузка, качество данных, объединение модулей и интерфейс | [План Павла](docs/plans/pavel.md) |

Оба файла планов входят в общую основу и доступны в каждой ветке. Ответственный за модуль также пишет его тесты и документацию. Изменения общего контракта согласуются вдвоём.

## Структура

| Путь | Содержимое |
| --- | --- |
| `src/data_drift_guardian/` | Основной Python-пакет |
| `src/data_drift_guardian/drift/` | Статистические методы и подготовка распределений |
| `app/` | Streamlit-приложение |
| `configs/` | Пример схемы и параметров проверок |
| `scripts/` | Воспроизводимая генерация данных и калибровочная сетка |
| `notebooks/` | Демонстрационный notebook |
| `tests/` | Места для unit- и интеграционных тестов |
| `docs/` | Архитектура, контракт, планы и заметки об экспериментах |
| `report/` | Место для итогового отчёта и ссылки на скринкаст |
| `data/` | Только инструкция; сгенерированные данные не коммитятся |

## Подготовка окружения

Для общей среды команды выбран Python 3.13; пакет ограничивает поддерживаемую ветку диапазоном `>=3.13,<3.14`. Зафиксированный процесс проверки: [docs/environment.md](docs/environment.md).

```bash
python -m venv .venv
```

Активация в Linux/macOS:

```bash
source .venv/bin/activate
```

Активация в Git Bash на Windows:

```bash
source .venv/Scripts/activate
```

Активация в Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установка зависимостей в активированное окружение:

```bash
python -m pip install -r requirements.txt
```

В `pyproject.toml` заданы допустимые диапазоны, в
`requirements-runtime.txt` — точные прямые runtime-зависимости, а
`requirements.txt` добавляет инструменты разработки. Проверка импортов
выполнена на macOS, Windows и в чистом Linux-окружении с Python 3.13.

## Запуск

### Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

После запуска:

1. Загрузите Reference и Current в формате CSV или Parquet.
2. Оставьте встроенную `configs/default.yaml` либо выберите собственный YAML-файл.
3. При необходимости включите override Adversarial Validation и задайте порог
   ROC-AUC. Override применяется к копии загруженной конфигурации и отражается в
   `effective_config` результата.
4. Нажмите **«Запустить анализ»**. Изменение входного файла, конфигурации или
   override-настроек помечает сохранённый результат как неактуальный и требует
   нового запуска.
5. Просмотрите итоговый статус, длительность, алерты, проверки схемы и качества,
   drift-метрики, источник порогов, Cramér's V, категориальные диагностики,
   стратегию ML-split и график выбранного признака.
6. Фильтруйте готовые таблицы по статусу и признаку без повторного анализа и
   скачайте тот же результат в JSON или автономном HTML.

Статусы `warning`, `critical`, `error` и `skipped` показываются отдельно. Ошибки
чтения или конфигурации выводятся в интерфейсе и не маскируются успешным
результатом. Пропуски исключаются из графиков распределений с указанием их
количества; бесконечности также исключаются из числовых графиков. JSON и HTML
готовятся один раз после анализа и сохраняются в состоянии сессии, поэтому
фильтры и кнопки скачивания не запускают статистику повторно.

### Командный интерфейс

Контрольный сценарий без внесённого дрейфа:

```bash
python -m data_drift_guardian \
  --reference data/generated/none/reference.csv \
  --current data/generated/none/current.csv \
  --config configs/default.yaml \
  --json-output outputs/html/none.json \
  --html-output outputs/html/none.html
```

Сценарий с комбинированным сдвигом и ненулевым кодом при алерте:

```bash
python -m data_drift_guardian \
  --reference data/generated/combined/reference.parquet \
  --current data/generated/combined/current.parquet \
  --config configs/default.yaml \
  --json-output outputs/html/combined.json \
  --html-output outputs/html/combined.html \
  --fail-on-alert
```

CLI возвращает `0`, если анализ выполнен; `1` при ошибке запуска; `2`, если
анализ выполнен с алертами и передан `--fail-on-alert`. Без этого флага
бизнес-алерт не считается ошибкой программы. Существующий JSON защищён от
случайной замены; та же политика действует для HTML, а для явной перезаписи
используется `--overwrite`. HTML включает Plotly JS внутрь файла, открывается
офлайн без Python-сервера и в проверенных demo-сценариях занимает около 4.8 МБ.
Полный контракт и политика атомарной записи описаны в
[docs/cli.md](docs/cli.md).

Основной библиотечный API:

```python
from data_drift_guardian import analyze
from data_drift_guardian.config import load_config

config = load_config("configs/default.yaml")
result = analyze(reference, current, config=config)
```

Pipeline проверяет схему и качество данных, рассчитывает настроенные drift-метрики и возвращает единый JSON-безопасный словарь. Неприменимые к конкретным данным проверки получают `skipped` с причиной. При `config=None` семантические типы признаков выводятся только из Reference.

### Калибровочные эксперименты

Полная воспроизводимая demo-сетка запускается одной командой:

```bash
python scripts/run_calibration.py
```

Runner выполняет контроль и несколько уровней сдвига на 20 фиксированных seed
при размерах Current 500, 1 200 и 3 000. Длинная таблица решений, агрегированные
наблюдаемые доли и metadata с версиями/commit/config сохраняются в
`report/experiments/`. Методика, ограничения и общий приёмочный сценарий для
CLI, HTML и Streamlit: [docs/calibration.md](docs/calibration.md).

YAML 0.1 по-прежнему принимается и нормализуется до контракта 0.2. Фактически
использованная конфигурация доступна в `result["effective_config"]`. Пример
разных Wasserstein-порогов в единицах каждого признака:

```yaml
drift:
  distance_thresholds:
    wasserstein: null
    psi: null
    js: null
  feature_thresholds:
    age: {wasserstein: 3.0}
    income: {wasserstein: 10000.0}
```

Adversarial Validation по умолчанию отключён, чтобы тяжёлый ML-блок не
запускался неожиданно. Для запуска укажите в YAML:

```yaml
adversarial:
  enabled: true
  n_splits: 3
  roc_auc_threshold: 0.70  # исследовательский пример, не универсальная граница
  exclude_columns: []     # сюда добавить ID и target при их наличии
  group_column: null      # либо имя entity ID для StratifiedGroupKFold
```

При заданном `group_column` колонка автоматически исключается из model
features; пропущенные ID и недостаток групп дают объяснимый `skipped`.
Описание результата: [docs/contracts.md](docs/contracts.md).

Запуск тестов:

```bash
python -m pytest
```

Unit- и integration-тесты покрывают генератор, загрузку, конфигурацию, схему,
Data Quality, доступные статистические методы, общий pipeline, CLI, автономный
HTML, графики и основные сценарии Streamlit-интерфейса. План проверок находится в
[tests/README.md](tests/README.md).

Демонстрационный notebook запускается из корня проекта:

```bash
python -m jupyter notebook notebooks/demo.ipynb
```

Он генерирует и загружает данные, выполняет все четыре сценария, повторяет
контроль на пяти seed и показывает те же результаты, которые читает дашборд.
Фактические наблюдения записаны в [docs/experiments.md](docs/experiments.md).

Docker-образ со Streamlit собирается и запускается командами:

```bash
docker build --tag data-drift-guardian:local .
docker run --detach --name data-drift-guardian \
  --publish 8501:8501 data-drift-guardian:local
docker inspect --format '{{.State.Health.Status}}' data-drift-guardian
```

После статуса `healthy` интерфейс доступен на `http://localhost:8501`.
Приложение работает от непривилегированного пользователя. Полная инструкция,
диагностика и состав автоматических проверок:
[docs/deployment.md](docs/deployment.md). Pull request в `main` также запускает
полные тесты Python 3.13 и Docker build/healthcheck через GitHub Actions.

## Данные, результаты и публикация

Демонстрационные данные генерируются с фиксированным seed по инструкции [data/README.md](data/README.md). Внешние датасеты указываются ссылками; CSV/Parquet с исходными данными в Git не добавляются. Результаты экспериментов записаны в [docs/experiments.md](docs/experiments.md).

Инструкция по размещению подготовленных веток: [docs/github_setup.md](docs/github_setup.md).

Перечень обязательных итоговых артефактов без планирования следующих недель: [docs/deliverables.md](docs/deliverables.md).
