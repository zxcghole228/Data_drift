# Передача изменений Михаила за 15.09–18.09

Реализация построена поверх `origin/main` `c72c481`
(`Plan work for 15-20 September`), который уже содержит принятый
change set `636fbe4`.

## Что изменилось

| Дата | Результат |
| --- | --- |
| 15.09 | Контракт 0.2, миграция YAML 0.1, `effective_config`, согласованные диагностические поля |
| 16.09 | Единый resolver персональных порогов Wasserstein/PSI/JS с глобальным fallback |
| 17.09 | Cramér's V и сведения о новых, исчезнувших и pooled-категориях |
| 18.09 | Опциональный `StratifiedGroupKFold`, исключение group ID из модели и проверки применимости |

HTML/PDF-экспорт не входит в этот change set. Пороги PSI, JS и ROC-AUC остаются
исследовательскими; Cramér's V не создаёт алерт без отдельной калибровки.

## Что должен читать Павел

UI, CLI и HTML не должны повторять правила выбора порогов. Они читают готовые
поля результата:

- `result.contract_version == "0.2"`;
- `result.effective_config` — полная нормализованная конфигурация запуска;
- `drift.features.<feature>.checks.<method>.details.threshold_source` —
  `feature`, `global` или `not_configured`;
- `details.resolved_threshold` — число либо `null`;
- для χ²: `details.cramers_v`, `new_category_count`,
  `disappeared_category_count`, `pooled_category_count` и
  `pooled_observation_fraction`;
- для ML: `adversarial.split_strategy`, `group_column`, `n_groups` и `reason`.

Минимальный YAML-фрагмент:

```yaml
contract_version: "0.2"
drift:
  distance_thresholds:
    wasserstein: null
    psi: 0.20
    js: 0.10
  feature_thresholds:
    age: {wasserstein: 3.0}
    income:
      wasserstein: 10000.0
      psi: null
adversarial:
  enabled: true
  n_splits: 4
  roc_auc_threshold: null
  exclude_columns: [target]
  group_column: entity_id
```

Фрагмент реального `CheckResult` имеет вид:

```json
{
  "name": "wasserstein",
  "value": 4.2,
  "threshold": 3.0,
  "alert": true,
  "details": {
    "resolved_threshold": 3.0,
    "threshold_source": "feature",
    "decision": "upper_threshold"
  }
}
```

## Проверенные крайние случаи

- старый YAML 0.1 нормализуется до 0.2 без изменения прежних решений;
- неизвестный признак/метод, отрицательный, бесконечный или булев порог дают
  `ValueError`;
- явный feature-level `null` перекрывает общий порог; при отсутствии обоих
  `alert=None`;
- равенство метрики порогу не срабатывает, используется строгое `>`;
- единственная категория и вырожденная таблица χ² дают JSON-safe `skipped`;
- отсутствующий/пустой group ID и недостаток групп дают `skipped`;
- тест каждого группового fold подтверждает отсутствие пересечения entity ID;
- режим без `group_column` сохраняет прежний `StratifiedKFold`.

На синтетическом leakage-примере с seed `20260918` обычный OOF ROC-AUC равен
`1.0`, групповой — `0.5`. Это иллюстрация утечки повторяющейся сущности, а не
оценка production-данных и не доказательство падения качества основной модели.

## Командная операция после применения

После merge PR записать новый базовый SHA:

```bash
git fetch origin
git rev-parse --short origin/main
git log -1 --oneline origin/main
```

Затем синхронизировать обе рабочие ветки с `origin/main` и повторить полный
`python -m pytest`.
