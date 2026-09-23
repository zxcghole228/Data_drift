# Online acceptance

Набор проверяет публичный FastAPI-контракт поверх временной persistent
SQLite-базы и воспроизводимых сценариев генератора.

Полный запуск:

```bash
python -m pytest -q tests/acceptance/test_online_acceptance.py
```

Быстрый persistence-smoke, который отдельно выполняется в CI:

```bash
python -m pytest -q -m online_smoke tests/acceptance
```
