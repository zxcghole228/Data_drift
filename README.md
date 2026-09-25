<p align="center">
  <img src="app/assets/data_drift_guardian.png" width="128" alt="Data Drift Guardian">
</p>

# Data Drift Guardian

Data Drift Guardian — прототип системы контроля качества табличных данных и
обнаружения data drift. Одно статистическое ядро используется в библиотечном
API, CLI, автономном HTML-отчёте, Streamlit и online micro-batch мониторинге.

Проект не использует нейросети. Статистические проверки реализованы на
NumPy/SciPy, а дополнительная Adversarial Validation — на scikit-learn и
LightGBM.

**Итоговый отчёт:** [`report/final_report.html`](report/final_report.html) —
постановка задачи, архитектура, методы, результаты экспериментов, проверка
online-мониторинга и инструкция по воспроизведению.

## Возможности

- загрузка Reference и Current из CSV/Parquet;
- строгая проверка колонок и семантических типов;
- контроль пропусков, дубликатов, Infinity и границ Min/Max;
- KS-test и Wasserstein-1 для числовых признаков;
- χ² и Cramér's V для категориальных признаков;
- PSI и Jensen–Shannon divergence на общем пространстве бинов/категорий;
- поправка Benjamini–Hochberg на множественные проверки;
- персональные пороги метрик для отдельных признаков;
- Adversarial Validation с OOF ROC-AUC и Feature Importance;
- единый JSON-безопасный `AnalysisResult` версии `0.2`;
- тёмный Streamlit-дашборд и автономный HTML с Plotly;
- FastAPI для событий, батчей, Reference и запаздывающего Feedback;
- tumbling windows, идемпотентность, SQLite-история и доставка алертов;
- Docker Compose, CI, unit-, integration- и acceptance-тесты.

## Быстрый старт

Требуется Python `>=3.13,<3.14`.

Windows Git Bash:

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

`requirements-runtime.txt` содержит проверенные прямые runtime-зависимости,
`requirements.txt` дополнительно устанавливает инструменты разработки.

## Демонстрационные данные

Данные создаются локально с фиксированным seed и не хранятся в Git:

```bash
python scripts/generate_demo_data.py --scenario none --format both
python scripts/generate_demo_data.py --scenario combined --format both
```

Результат появится в `data/generated/none/` и
`data/generated/combined/`. Доступны сценарии `none`, `numeric`,
`categorical`, `missingness` и `combined`. Подробности находятся в
[`data/README.md`](data/README.md).

## Offline-анализ

### CLI

Контроль без намеренного дрейфа:

```bash
python -m data_drift_guardian \
  --reference data/generated/none/reference.csv \
  --current data/generated/none/current.csv \
  --config configs/demo_calibrated.yaml \
  --json-output outputs/demo/none.json \
  --html-output outputs/demo/none.html
```

Комбинированный сценарий:

```bash
python -m data_drift_guardian \
  --reference data/generated/combined/reference.parquet \
  --current data/generated/combined/current.parquet \
  --config configs/demo_calibrated.yaml \
  --json-output outputs/demo/combined.json \
  --html-output outputs/demo/combined.html \
  --fail-on-alert
```

Коды завершения:

| Код | Значение |
| ---: | --- |
| `0` | анализ завершён; алертов нет либо `--fail-on-alert` не передан |
| `1` | ошибка аргументов, входов, конфигурации или записи |
| `2` | анализ завершён с алертами при `--fail-on-alert` |

Существующие результаты защищены от случайной замены. Для явной перезаписи
используется `--overwrite`. Полный контракт CLI: [`docs/cli.md`](docs/cli.md).

### Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

В режиме **Офлайн-анализ**:

1. загрузите Reference и Current в CSV или Parquet;
2. используйте встроенную конфигурацию либо загрузите YAML;
3. нажмите «Запустить анализ»;
4. изучите сводку, Data Quality, drift-метрики, ML-результат и распределения;
5. скачайте полный JSON или автономный HTML.

Фильтры интерфейса не запускают pipeline повторно. После изменения входов или
конфигурации результат помечается как неактуальный.

### Python API

```python
from data_drift_guardian import analyze
from data_drift_guardian.config import load_config
from data_drift_guardian.ingestion import load_table

reference = load_table("data/generated/none/reference.parquet")
current = load_table("data/generated/none/current.parquet")
config = load_config("configs/demo_calibrated.yaml")

result = analyze(reference, current, config=config)
print(result["summary"])
```

Функция не изменяет входные DataFrame. Неприменимые проверки получают
`skipped` с причиной; значения NaN/Infinity не попадают в итоговый JSON.

## Online-мониторинг

Online-слой принимает готовые батчи или отдельные события. События сохраняются
в SQLite и формируют непересекающиеся окна фиксированного размера. Каждое
готовое окно передаётся в тот же `analyze`, что и offline-режим.

Терминал 1 — API:

```bash
python -m data_drift_guardian.online \
  --config configs/online.yaml \
  --state data/online/monitoring.sqlite3
```

Терминал 2 — dashboard:

```bash
DDG_ONLINE_CONFIG="configs/online.yaml" \
DDG_ONLINE_STATE="data/online/monitoring.sqlite3" \
python -m streamlit run app/streamlit_app.py
```

Регистрация активного Reference:

```bash
curl --fail --request POST \
  "http://127.0.0.1:8000/api/v1/references?format=csv&reference_id=demo-ref&name=DemoReference&activate=true" \
  --header "Content-Type: text/csv" \
  --data-binary @data/generated/none/reference.csv
```

Проверки состояния:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

- API: `http://127.0.0.1:8000`;
- Swagger UI: `http://127.0.0.1:8000/docs`;
- Streamlit: `http://127.0.0.1:8501`.

Система хранит Reference, буфер, Window/Batch Run, полные результаты, алерты,
доставки и Feedback. Повтор идентичного `event_id` или `batch_id` не запускает
повторный анализ; конфликтующее содержимое возвращает HTTP `409`.

Полный HTTP-контракт: [`docs/online_monitoring.md`](docs/online_monitoring.md).

## Docker Compose

```bash
docker compose up --detach --build
docker compose ps
```

После перехода обоих сервисов в `healthy`:

- API и Swagger: `http://localhost:8000/docs`;
- Streamlit: `http://localhost:8501`.

Compose запускает API и dashboard от UID/GID `10001`, использует общий
persistent volume для SQLite и отдельный volume для JSONL-алертов. Обычный
`docker compose down` сохраняет данные. Подробная инструкция:
[`docs/deployment.md`](docs/deployment.md).

## Конфигурации

| Файл | Назначение |
| --- | --- |
| `configs/default.yaml` | базовый offline-пример |
| `configs/demo_calibrated.yaml` | воспроизводимая синтетическая демонстрация |
| `configs/online.yaml` | анализ и параметры online-слоя |

YAML `0.1` автоматически нормализуется до `0.2`. Фактически применённые
настройки сохраняются в `result["effective_config"]`. Порог Wasserstein задаётся
в единицах признака; demo-пороги PSI, JS и ROC-AUC не являются универсальными
production-границами.

## Тесты и воспроизводимость

```bash
python -m pytest -q
```

Финальный локальный прогон после визуальных изменений: `391 passed`. Тесты
покрывают отдельные методы, общий pipeline, CLI/HTML, Streamlit, Online API,
SQLite, окна, идемпотентность, Feedback, Docker-файлы и сквозные сценарии.

Демонстрационный notebook:

```bash
python -m jupyter notebook notebooks/demo.ipynb
```

Калибровочная сетка:

```bash
python scripts/run_calibration.py
```

Она выполняет 540 анализов и сохраняет машинно-читаемые результаты в
`report/experiments/`. Методика и значения:
[`docs/experiments.md`](docs/experiments.md).

## Структура репозитория

| Путь | Содержимое |
| --- | --- |
| `src/data_drift_guardian/` | статистическое ядро, CLI и online-сервис |
| `app/` | Streamlit и визуальные ресурсы |
| `configs/` | YAML-конфигурации |
| `scripts/` | генератор, калибровка и проверка окружения |
| `notebooks/` | воспроизводимая демонстрация |
| `tests/` | unit-, integration- и acceptance-тесты |
| `docs/` | архитектура, контракты, методы и запуск |
| `report/final_report.html` | автономный итоговый отчёт с печатью в PDF |
| `report/experiments/` | результаты калибровки |

## Документация

| Документ | Содержание |
| --- | --- |
| [`architecture.md`](docs/architecture.md) | компоненты, поток данных и границы |
| [`contracts.md`](docs/contracts.md) | входы, результат, статусы и пороги |
| [`methods.md`](docs/methods.md) | статистика, Data Quality и ML-метод |
| [`experiments.md`](docs/experiments.md) | сценарии, калибровка и результаты |
| [`online_monitoring.md`](docs/online_monitoring.md) | Online API, окна и Feedback |
| [`cli.md`](docs/cli.md) | аргументы, exit codes и экспорт |
| [`deployment.md`](docs/deployment.md) | Python, Docker Compose и CI |

## Ограничения

- синтетические пороги требуют повторной калибровки на реальных данных;
- SQLite-конфигурация рассчитана на один API worker;
- автоматическое переобучение и автоматическая замена Reference не выполняются;
- учебный API не реализует TLS, аутентификацию и rate limiting;
- изменение входных распределений не является причинным доказательством
  concept drift или необходимости переобучения.
