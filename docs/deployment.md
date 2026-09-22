# Docker и CI

Контейнер запускает тот же Streamlit-интерфейс, что и локальная команда. В образ
копируются только пакет, приложение и YAML-конфигурации: локальные данные,
результаты, notebook, тесты, Git-история и `.env` исключены через
`.dockerignore`.

## Сборка и запуск

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
