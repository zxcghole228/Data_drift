# Общее окружение

Решение команды от 07.09.2026:

- Python: ветка 3.13 (`>=3.13,<3.14`).
- Управление окружением: стандартный `venv` и `pip`.
- Источник зависимостей: `pyproject.toml`.
- Проверка: `scripts/check_environment.py` действительно импортирует библиотеки и выводит версии.
- Точные проверенные версии прямых зависимостей фиксируются ниже после запуска на компьютерах Михаила и Павла.

## Создание среды

Windows PowerShell из корня репозитория:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

Git Bash на Windows:

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

Если команда проверки возвращает код 1, в таблице указан отсутствующий или не импортирующийся пакет. Не игнорировать ошибку импорта LightGBM: она часто связана не с Python-кодом, а с системной библиотекой OpenMP.

## Фактически проверенные версии

Заполнить только по выводу скрипта, а не вручную по ожидаемым версиям.

| Компонент | Михаил | Павел | Совпадает |
| --- | --- | --- | --- |
| Python | — | 3.13.7 | — |
| NumPy | — | 2.5.3 | — |
| pandas | — | 3.0.5 | — |
| SciPy | — | 1.18.1 | — |
| scikit-learn | — | 1.9.0 | — |
| LightGBM | — | 4.7.0 | — |
| Plotly | — | 6.9.0 | — |
| Streamlit | — | 1.63.0 | — |
| PyYAML | — | 6.0.3 | — |
| PyArrow | — | 23.0.1 | — |
| pytest | — | 9.1.1 | — |

Если версии различаются, команда выбирает один рабочий набор и фиксирует его отдельным lock-файлом после проверки установки на обеих машинах. Не использовать `pip freeze` из глобального или давно используемого окружения.
