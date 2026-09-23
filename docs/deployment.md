# Docker Compose, standalone Docker и CI

Один образ используется двумя сервисами: `api` запускает single-worker FastAPI,
а `dashboard` — Streamlit с офлайн- и online-режимами. В образ копируются только
пакет, приложение и YAML-конфигурации: локальные данные, результаты, notebook,
тесты, Git-история и `.env` исключены через `.dockerignore`.

## Online stack через Docker Compose

Из корня репозитория:

```bash
docker compose config
docker compose up --detach --build
docker compose ps
```

Сервисы должны перейти в состояние `healthy`:

- Online API и Swagger UI: `http://localhost:8000` и
  `http://localhost:8000/docs`;
- Streamlit: `http://localhost:8501`.

Liveness API доступен сразу, а readiness возвращает `503`, пока не
зарегистрирован активный Reference:

```bash
curl --fail http://localhost:8000/health/live
curl --include http://localhost:8000/health/ready
curl --fail --request POST \
  "http://localhost:8000/api/v1/references?format=csv&reference_id=compose-ref&name=ComposeReference&activate=true" \
  --header "Content-Type: text/csv" \
  --data-binary @data/generated/none/reference.csv
curl --fail http://localhost:8000/health/ready
```

Compose создаёт `online-state` для SQLite/WAL и `online-outputs` для JSONL.
Оба сервиса монтируют их по одинаковым путям; API является единственным writer,
dashboard читает сохранённую историю. Проверка непривилегированного запуска:

```bash
docker compose exec api python -c "import os; print(os.getuid(), os.getgid())"
docker compose exec dashboard python -c "import os; print(os.getuid(), os.getgid())"
```

Ожидается `10001 10001` для обоих процессов. Состояние должно сохраняться после
перезапуска:

```bash
docker compose restart
curl --fail http://localhost:8000/api/v1/references/current
```

Обычная остановка не удаляет volumes:

```bash
docker compose down
```

Команда `docker compose down --volumes` удаляет SQLite и JSONL без возможности
восстановления из Compose, поэтому её следует использовать только для явного
сброса демонстрационного состояния.

Необязательные настройки копируются из `.env.example` в локальный `.env`:

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `DDG_API_PORT` | `8000` | Порт API на host |
| `DDG_DASHBOARD_PORT` | `8501` | Порт Streamlit на host |
| `DDG_ALERT_WEBHOOK_URL` | пусто | URL webhook; пустое значение отключает канал |

Внутри Compose зафиксированы `DDG_ONLINE_CONFIG=/app/configs/online.yaml` и
`DDG_ONLINE_STATE=/app/data/online/monitoring.sqlite3`, чтобы API и dashboard
гарантированно читали одну конфигурацию и одну базу.

## Standalone Streamlit-контейнер

Из корня репозитория:

```bash
docker build --tag data-drift-guardian:local .
docker run --detach \
  --name data-drift-guardian \
  --publish 8501:8501 \
  data-drift-guardian:local
```

Приложение доступно на `http://localhost:8501`. Контейнерный процесс работает
от пользователя `app` с UID 10001, а не от `root`. LightGBM получает системную
библиотеку `libgomp1`.

Проверка пользователя и состояния контейнера:

```bash
docker run --rm --entrypoint python data-drift-guardian:local \
  -c "import os; print(f'uid={os.getuid()}'); assert os.getuid() != 0"
docker inspect --format '{{.State.Health.Status}}' data-drift-guardian
```

PowerShell использует двойные кавычки вокруг шаблона inspect:

```powershell
docker inspect --format "{{.State.Health.Status}}" data-drift-guardian
(Invoke-WebRequest http://localhost:8501/_stcore/health).Content
```

Ожидаются `healthy` и ответ `ok`. Если контейнер не стал здоровым:

```bash
docker ps --all --filter name=data-drift-guardian
docker logs data-drift-guardian
docker inspect data-drift-guardian
```

После ручного smoke test загрузите в интерфейс одинаковую пару Reference и
Current вместе с `configs/demo_calibrated.yaml` и выполните анализ. Остановить и
удалить тестовый контейнер:

```bash
docker stop data-drift-guardian
docker rm data-drift-guardian
```

## Что проверяет GitHub Actions

Workflow `.github/workflows/ci.yml` запускается для pull request в `main`, push
в рабочие ветки и вручную. Он не использует секреты и выполняет два job:

1. создаёт чистое окружение Python 3.13, устанавливает `requirements.txt`,
   проверяет импорты и запускает весь `pytest`;
2. после успешных тестов собирает Docker-образ, проверяет непривилегированного
   пользователя, запускает контейнер и ждёт встроенный healthcheck Streamlit.

Merge выполняется только после зелёных `Python 3.13 tests` и
`Docker build and healthcheck`. Если GitHub Actions отключён в настройках
репозитория, его необходимо включить до приёмки Docker-задачи.

## Фиксация контрольного прогона

В описании PR нужно записать факты, а не ожидаемые значения:

- commit SHA и чистоту рабочей директории;
- версию Docker (`docker version`);
- результат полного `pytest`;
- image ID собранного образа и digest, если он доступен (`docker image inspect`);
- UID процесса и итог `healthy`;
- результат ручного анализа demo-сценария;
- ссылки на оба зелёных job GitHub Actions.
