# Общее окружение

Решение команды от 07.09.2026:

- Python: ветка 3.12 (`>=3.12,<3.13`).
- Управление окружением: стандартный `venv` и `pip`.
- Источник зависимостей: `pyproject.toml`.
- Проверка: `scripts/check_environment.py` действительно импортирует библиотеки и выводит версии.
- Точные проверенные версии прямых зависимостей фиксируются ниже после запуска на компьютерах Михаила и Павла.

## Создание среды

Windows PowerShell из корня репозитория:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Git Bash на Windows:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Если команда проверки возвращает код 1, в таблице указан отсутствующий или не импортирующийся пакет. Не игнорировать ошибку импорта LightGBM: она часто связана не с Python-кодом, а с системной библиотекой OpenMP.

## Фактически проверенные версии

Заполнить только по выводу скрипта, а не вручную по ожидаемым версиям.

| Компонент | Михаил | Павел | Совпадает |
| --- | --- | --- | --- |
| Python | — | — | — |
| NumPy | — | — | — |
| pandas | — | — | — |
| SciPy | — | — | — |
| scikit-learn | — | — | — |
| LightGBM | — | — | — |
| Plotly | — | — | — |
| Streamlit | — | — | — |
| PyYAML | — | — | — |
| PyArrow | — | — | — |
| pytest | — | — | — |

Если версии различаются, команда выбирает один рабочий набор и фиксирует его отдельным lock-файлом после проверки установки на обеих машинах. Не использовать `pip freeze` из глобального или давно используемого окружения.
