# Тестирование

Тесты разделены по уровню ответственности:

| Каталог | Назначение |
| --- | --- |
| `unit/` | формулы, конфигурация, схема, Data Quality, хранилище и форматирование |
| `integration/` | pipeline, CLI/HTML, Streamlit, FastAPI, notebook и deployment-файлы |
| `acceptance/` | сквозной публичный контракт online-мониторинга |

## Полный прогон

Из корня репозитория в активированном окружении:

```bash
python -m pytest -q
```

Используйте именно `python -m pytest`: так корень проекта гарантированно
попадает в путь импорта на Windows. Финальный локальный прогон после
визуальных изменений: `391 passed`.

## Выборочные проверки

```bash
# Статистическое и ML-ядро
python -m pytest -q \
  tests/unit/test_numeric_drift.py \
  tests/unit/test_categorical_drift.py \
  tests/unit/test_stability.py \
  tests/unit/test_adversarial.py

# Offline pipeline, CLI, HTML и Streamlit
python -m pytest -q \
  tests/integration/test_pipeline.py \
  tests/integration/test_cli.py \
  tests/unit/test_reporting.py \
  tests/integration/test_streamlit_app.py

# Online API, monitor и dashboard
python -m pytest -q \
  tests/integration/test_online_api.py \
  tests/integration/test_online_monitor.py \
  tests/integration/test_online_dashboard.py
```

## Online acceptance

Полная сквозная проверка:

```bash
python -m pytest -q tests/acceptance/test_online_acceptance.py
```

Быстрый persistence-smoke, также используемый CI:

```bash
python -m pytest -q -m online_smoke tests/acceptance
```

Acceptance-сценарии используют временную SQLite-базу и подтверждают:

- регистрацию и восстановление Reference;
- сохранение неполного окна после рестарта;
- отсутствие ожидаемых алертов в контрольном сценарии;
- обнаружение комбинированного сдвига;
- идемпотентность событий и батчей;
- сохранение Run при ошибке webhook;
- пересчёт качества после запаздывающего Feedback.

## Принципы тестов

- временные файлы и базы создаются через `tmp_path`;
- входные DataFrame не должны изменяться;
- JSON проверяется с `allow_nan=False`;
- `skipped` проверяется вместе с причиной;
- детерминированные формулы отделяются от статистических экспериментов;
- независимый контроль без внесённого дрейфа не обязан для каждого seed давать
  все p-value выше `0.05`;
- тесты не записывают сгенерированные CSV/Parquet в репозиторий.
