# Установка и запуск

Проект поддерживает локальный запуск в Python 3.13 и контейнерный запуск через
Docker Compose. Compose поднимает Online API и Streamlit с общей persistent
SQLite-базой.

## Локальное окружение

Из корня репозитория:

```bash
python -m venv .venv
```

Активация в Git Bash на Windows:

```bash
source .venv/Scripts/activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установка и проверка:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m pytest -q
```

`pyproject.toml` задаёт допустимые диапазоны, `requirements-runtime.txt` —
проверенные прямые runtime-зависимости, `requirements.txt` добавляет dev-
инструменты. Ошибку импорта LightGBM нельзя игнорировать: в Linux она обычно
означает отсутствие системной библиотеки OpenMP (`libgomp1`).

Проверенное окружение разработки: Python 3.13, NumPy 2.5.3, pandas 3.0.5,
SciPy 1.18.1, scikit-learn 1.9.0, LightGBM 4.7.0, Plotly 6.9.0,
Streamlit 1.63.0, PyYAML 6.0.3, PyArrow 23.0.1 и pytest 9.1.1.

## Локальные интерфейсы

Streamlit:

```bash
python -m streamlit run app/streamlit_app.py
```

Online API:

```bash
python -m data_drift_guardian.online \
  --config configs/online.yaml \
  --state data/online/monitoring.sqlite3
```

После запуска доступны:

- Streamlit: `http://127.0.0.1:8501`;
- FastAPI: `http://127.0.0.1:8000`;
- Swagger UI: `http://127.0.0.1:8000/docs`.

## Docker Compose

```bash
docker compose config
docker compose up --detach --build
docker compose ps
```

Сервисы `api` и `dashboard` должны перейти в состояние `healthy`. По умолчанию
API опубликован на `http://localhost:8000`, Streamlit — на
`http://localhost:8501`.

Liveness доступен сразу, readiness возвращает `503`, пока не зарегистрирован
активный Reference:

```bash
curl --fail http://localhost:8000/health/live
curl --include http://localhost:8000/health/ready

curl --fail --request POST \
  "http://localhost:8000/api/v1/references?format=csv&reference_id=compose-ref&name=ComposeReference&activate=true" \
  --header "Content-Type: text/csv" \
  --data-binary @data/generated/none/reference.csv

curl --fail http://localhost:8000/health/ready
```

Compose создаёт named volumes для SQLite/WAL и JSONL-алертов. API является
единственным writer, dashboard читает ту же базу. Оба процесса работают от
непривилегированного UID/GID `10001`:

```bash
docker compose exec api python -c "import os; print(os.getuid(), os.getgid())"
docker compose exec dashboard python -c "import os; print(os.getuid(), os.getgid())"
```

Проверка сохранения состояния:

```bash
docker compose restart
curl --fail http://localhost:8000/api/v1/references/current
```

Остановка без удаления данных:

```bash
docker compose down
```

`docker compose down --volumes` безвозвратно удаляет демонстрационную SQLite-
базу и JSONL внутри volumes, поэтому команда используется только для явного
сброса состояния.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `DDG_API_PORT` | `8000` | порт API на host |
| `DDG_DASHBOARD_PORT` | `8501` | порт Streamlit на host |
| `DDG_ONLINE_CONFIG` | `configs/online.yaml` | online-конфигурация |
| `DDG_ONLINE_STATE` | путь из YAML | SQLite-файл состояния |
| `DDG_ALERT_WEBHOOK_URL` | пусто | необязательный webhook алертов |

Для Compose необязательные значения можно поместить в локальный `.env` по
образцу `.env.example`. Секреты не добавляются в Git.

## Самостоятельный Docker-образ

```bash
docker build --tag data-drift-guardian:local .
docker run --detach \
  --name data-drift-guardian \
  --publish 8501:8501 \
  data-drift-guardian:local
```

Проверка пользователя и healthcheck:

```bash
docker run --rm --entrypoint python data-drift-guardian:local \
  -c "import os; print(f'uid={os.getuid()}'); assert os.getuid() != 0"
docker inspect --format '{{.State.Health.Status}}' data-drift-guardian
```

## Continuous Integration

`.github/workflows/ci.yml` выполняет два последовательных job:

1. создаёт чистое окружение Python 3.13, устанавливает зависимости, проверяет
   импорты и запускает весь `pytest`;
2. собирает Docker-образ, проверяет непривилегированного пользователя,
   запускает контейнер и ожидает успешный healthcheck.

Локальный полный прогон после финальных визуальных изменений: `391 passed`.
