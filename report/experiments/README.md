# Калибровочные артефакты

Файлы в этой папке создаются командой:

```bash
python scripts/run_calibration.py
```

Ожидаются `calibration_runs.csv`, `calibration_summary.csv` и
`calibration_metadata.json`. CSV разрешены в `.gitignore` только для этой
папки; исходные сгенерированные таблицы по-прежнему не коммитятся.

Методика, схема полей, результаты и ограничения описаны в
[`docs/experiments.md`](../../docs/experiments.md).
