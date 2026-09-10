# PSI и Jensen–Shannon divergence

Модуль `data_drift_guardian.drift.binning` переводит числовые и
категориальные признаки в сопоставимые векторы вероятностей. Модуль
`data_drift_guardian.drift.stability` рассчитывает по этим векторам PSI и
Jensen–Shannon divergence. Вычисление метрики отделено от решения об alert.

## Общее разбиение

Нельзя строить отдельные бины для Reference и Current: ячейка с одним индексом
должна обозначать одну и ту же часть пространства в обеих выборках.

Для числового признака внутренние границы — квантили конечных значений
Reference. К обеим выборкам применяется один набор границ. Крайние границы
равны `-inf` и `+inf`, поэтому значения Current за пределами диапазона Reference
не теряются. В JSON они записываются строками `"-inf"` и `"+inf"`.

Если Reference константный, обычные квантильные границы совпадают. В этом случае
строятся три ячейки: ниже эталонного значения, равно ему и выше него. Такая
политика позволяет увидеть отклонение в любую сторону.

Для категориального признака используется упорядоченное объединение: сначала
категории Reference в порядке появления, затем новые категории Current. Поэтому
новая категория получает нулевую вероятность в Reference, но не исчезает.

`NaN`/`None` исключаются из этих распределений и анализируются Data Quality.
Для числовых признаков также исключаются `+inf` и `-inf`. Исходные и фактически
использованные размеры сохраняются в `details`.

## Population Stability Index

Для вероятностей Reference \(p_i\) и Current \(q_i\):

\[
\operatorname{PSI}(P,Q)=\sum_i(q_i-p_i)\ln\frac{q_i}{p_i}.
\]

При нулевой вероятности логарифм не определён. В проекте применяется аддитивное
сглаживание с параметром \(\varepsilon\), после которого каждый вектор заново
нормируется:

\[
\tilde p_i=\frac{p_i+\varepsilon}{1+k\varepsilon},\qquad
\tilde q_i=\frac{q_i+\varepsilon}{1+k\varepsilon},
\]

где \(k\) — число ячеек. По умолчанию \(\varepsilon=10^{-6}\), значение берётся
из `config["drift"]["psi_smoothing"]`. PSI использует натуральный логарифм.

Результат зависит от числа бинов и сглаживания. Часто встречающиеся ориентиры
PSI нельзя считать универсальными доказанными границами: их нужно проверять на
данных и бизнес-риске конкретного признака.

## Jensen–Shannon divergence

Для \(M=(P+Q)/2\):

\[
\operatorname{JSD}(P,Q)=\frac12D_{KL}(P\Vert M)+
\frac12D_{KL}(Q\Vert M).
\]

SciPy `jensenshannon` возвращает Jensen–Shannon **distance**, то есть квадратный
корень из divergence. Поэтому реализация возводит результат SciPy в квадрат.
При основании логарифма 2 divergence находится в диапазоне от 0 до 1. Нулевые
ячейки допустимы и не требуют PSI-сглаживания.

## Вызов из pipeline

```python
from data_drift_guardian.drift import (
    build_probabilities,
    js_divergence,
    psi,
)

feature = "age"
feature_config = config["schema"]["features"][feature]
drift_config = config["drift"]

reference_probabilities, current_probabilities, binning_details = (
    build_probabilities(
        reference[feature],
        current[feature],
        {
            "kind": feature_config["kind"],
            "n_bins": drift_config["n_bins"],
        },
    )
)

psi_result = psi(
    reference_probabilities,
    current_probabilities,
    smoothing=drift_config["psi_smoothing"],
)
js_result = js_divergence(
    reference_probabilities,
    current_probabilities,
    base=drift_config["js_base"],
)

psi_result["details"]["binning"] = binning_details
js_result["details"]["binning"] = binning_details
```

Обе метрики возвращают `CheckResult`. Пока порог не применён общим pipeline,
`threshold` и `alert` равны `None`; это не означает подтверждённое отсутствие
дрейфа.

## Воспроизводимый пример

Для `seed=42`, `n_reference=5000`, `n_current=3000`, десяти квантильных бинов
и сглаживания `1e-6` получены следующие значения:

| Сценарий | Признак | PSI | JS divergence, base 2 |
| --- | --- | ---: | ---: |
| `none` | `age` | 0.002573 | 0.000464 |
| `numeric` | `age` | 0.456097 | 0.078878 |
| `none` | `region` | 0.001071 | 0.000193 |
| `categorical` | `region` | 0.270713 | 0.048116 |

Значения фиксируют проверенный пример, но не объявляются универсальными
порогами дрейфа.
