# Общее окружение

Решение команды от 07.09.2026:

- Python: ветка 3.13 (`>=3.13,<3.14`).
- Управление окружением: стандартный `venv` и `pip`.
- Источник зависимостей: `pyproject.toml`.
- Проверка: `scripts/check_environment.py` действительно импортирует библиотеки и выводит версии.
- Точные проверенные runtime-версии прямых зависимостей зафиксированы ниже и в
  `requirements-runtime.txt`; `requirements.txt` добавляет dev-инструменты.
  Транзитивные зависимости пока не закреплены отдельным lock-файлом.

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
| Python | 3.13.0 | 3.13.7 | Да, ветка 3.13; patch отличается |
| NumPy | 2.5.3 | 2.5.3 | Да |
| pandas | 3.0.5 | 3.0.5 | Да |
| SciPy | 1.18.1 | 1.18.1 | Да |
| scikit-learn | 1.9.0 | 1.9.0 | Да |
| LightGBM | 4.7.0 | 4.7.0 | Да |
| Plotly | 6.9.0 | 6.9.0 | Да |
| Streamlit | 1.63.0 | 1.63.0 | Да |
| PyYAML | 6.0.3 | 6.0.3 | Да |
| PyArrow | 23.0.1 | 23.0.1 | Да |
| pytest | 9.1.1 | 9.1.1 | Да |

Чистая дополнительная проверка 14.09 выполнена на Linux с Python 3.13.15 и теми
же версиями прямых библиотек: установка из `requirements.txt`, импорт всех
пакетов и полный прогон `233 passed`. Не использовать `pip freeze` из
глобального или давно используемого окружения. Для финальной сдачи ещё
потребуется полноценный lock транзитивных зависимостей либо повторная проверка
установки непосредственно перед релизом.
