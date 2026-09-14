"""Out-of-fold adversarial validation with fold-local preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .contracts import AdversarialResult, FeatureType, Status

_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "n_splits",
        "roc_auc_threshold",
        "exclude_columns",
        "random_seed",
        "feature_types",
    }
)


def _result(
    status: Status,
    reason: str | None,
    *,
    roc_auc: float | None = None,
    threshold: float | None = None,
    fold_auc: list[float] | None = None,
    feature_importance: dict[str, float] | None = None,
    importance_type: str | None = None,
    alert: bool | None = None,
) -> AdversarialResult:
    return {
        "status": status,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "fold_auc": [] if fold_auc is None else fold_auc,
        "feature_importance": (
            {} if feature_importance is None else feature_importance
        ),
        "importance_type": importance_type,
        "alert": alert,
        "reason": reason,
    }


def _require_dataframes(reference: object, current: object) -> None:
    invalid: list[str] = []
    if not isinstance(reference, pd.DataFrame):
        invalid.append(f"reference={type(reference).__name__}")
    if not isinstance(current, pd.DataFrame):
        invalid.append(f"current={type(current).__name__}")
    if invalid:
        raise TypeError(
            "reference и current должны иметь тип pandas.DataFrame; "
            f"получено: {', '.join(invalid)}"
        )


def _config_values(
    config: Mapping[str, Any],
) -> tuple[int, float | None, list[str], int, dict[str, FeatureType]]:
    if not isinstance(config, Mapping):
        raise TypeError("config должен быть отображением ключ-значение")
    unknown = sorted(str(key) for key in set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Неизвестные параметры Adversarial Validation: " + ", ".join(unknown)
        )

    n_splits = config.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise ValueError("n_splits должен иметь тип int")
    normalized_splits = int(n_splits)
    if normalized_splits < 2:
        raise ValueError("n_splits должен быть не меньше 2")

    threshold_value = config.get("roc_auc_threshold")
    threshold: float | None
    if threshold_value is None:
        threshold = None
    else:
        if isinstance(threshold_value, bool) or not isinstance(threshold_value, Real):
            raise ValueError("roc_auc_threshold должен быть числом из [0, 1] или None")
        threshold = float(threshold_value)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("roc_auc_threshold должен быть числом из [0, 1] или None")

    excluded = config.get("exclude_columns", [])
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) or not name for name in excluded
    ):
        raise ValueError("exclude_columns должен быть списком непустых строк")
    if len(excluded) != len(set(excluded)):
        raise ValueError("exclude_columns не должен содержать повторы")

    seed = config.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("random_seed должен быть неотрицательным int")

    raw_types = config.get("feature_types", {})
    if not isinstance(raw_types, Mapping):
        raise ValueError("feature_types должен быть отображением")
    feature_types: dict[str, FeatureType] = {}
    for name, kind in raw_types.items():
        if not isinstance(name, str) or kind not in {"numeric", "categorical"}:
            raise ValueError(
                "feature_types должен сопоставлять имена с numeric/categorical"
            )
        feature_types[name] = cast(FeatureType, kind)

    return normalized_splits, threshold, list(excluded), int(seed), feature_types


def _infer_type(series: pd.Series) -> FeatureType | None:
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        return "numeric"
    if (
        series.dtype == "object"
        or isinstance(series.dtype, pd.StringDtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or is_bool_dtype(series.dtype)
    ):
        return "categorical"
    return None


def _category_value(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    python_value = value.item() if isinstance(value, np.generic) else value
    return f"{type(python_value).__name__}:{python_value!r}"


def _prepare_features(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    excluded: list[str],
    configured_types: Mapping[str, FeatureType],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, FeatureType]]:
    common_columns = [
        str(name)
        for name in reference.columns
        if name in current.columns and str(name) not in excluded
    ]
    types: dict[str, FeatureType] = {}
    prepared_columns: dict[str, pd.Series] = {}
    combined = pd.concat([reference, current], ignore_index=True).copy(deep=True)

    for name in common_columns:
        expected = configured_types.get(name)
        reference_type = _infer_type(reference[name])
        current_type = _infer_type(current[name])
        feature_type = expected or reference_type
        if (
            feature_type is None
            or reference_type != feature_type
            or current_type != feature_type
        ):
            continue

        if feature_type == "numeric":
            values = combined[name].astype("float64").replace([np.inf, -np.inf], np.nan)
            prepared_columns[name] = values
        else:
            try:
                normalized = [
                    _category_value(value) for value in combined[name].tolist()
                ]
            except (TypeError, ValueError):
                continue
            prepared_columns[name] = pd.Series(normalized, dtype="object")
        types[name] = feature_type

    labels = np.concatenate(
        [
            np.zeros(len(reference), dtype=np.int8),
            np.ones(len(current), dtype=np.int8),
        ]
    )
    return pd.DataFrame(prepared_columns), labels, types


def _build_pipeline(
    feature_types: Mapping[str, FeatureType],
    *,
    random_seed: int,
) -> tuple[Pipeline, dict[str, str]]:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    transformer_features: dict[str, str] = {}
    for index, (feature, feature_type) in enumerate(feature_types.items()):
        transformer_name = f"feature_{index}"
        transformer_features[transformer_name] = feature
        if feature_type == "numeric":
            transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                            keep_empty_features=True,
                        ),
                    )
                ]
            )
        else:
            transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="<MISSING>",
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                ]
            )
        transformers.append((transformer_name, transformer, [feature]))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=random_seed,
        n_jobs=1,
        importance_type="gain",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", model)]
    ), transformer_features


def _fold_importance(
    pipeline: Pipeline,
    transformer_features: Mapping[str, str],
) -> dict[str, float]:
    preprocessor = cast(ColumnTransformer, pipeline.named_steps["preprocess"])
    model = cast(LGBMClassifier, pipeline.named_steps["model"])
    gains = np.asarray(
        model.booster_.feature_importance(importance_type="gain"),
        dtype=np.float64,
    )
    by_feature: dict[str, float] = {}
    for transformer, feature in transformer_features.items():
        output_slice = preprocessor.output_indices_[transformer]
        by_feature[feature] = float(gains[output_slice].sum())
    total = sum(by_feature.values())
    if total > 0.0:
        return {feature: value / total for feature, value in by_feature.items()}
    return {feature: 0.0 for feature in by_feature}


def adversarial_validate(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any],
) -> AdversarialResult:
    """Estimate Reference/Current separability using out-of-fold ROC-AUC.

    The source indicator is kept only in ``y``. Learned imputers, category
    encoders and the LightGBM classifier are fitted independently inside each
    training fold. Columns named in ``exclude_columns`` (for example IDs and a
    production target) are never passed to the model.
    """

    _require_dataframes(reference, current)
    n_splits, threshold, excluded, seed, configured_types = _config_values(config)
    if reference.empty or current.empty:
        return _result(
            "skipped",
            "Adversarial Validation требует непустые Reference и Current",
            threshold=threshold,
        )
    if len(reference) < n_splits or len(current) < n_splits:
        return _result(
            "skipped",
            f"Для n_splits={n_splits} требуется не меньше {n_splits} строк "
            "каждого источника",
            threshold=threshold,
        )

    features, labels, feature_types = _prepare_features(
        reference,
        current,
        excluded=excluded,
        configured_types=configured_types,
    )
    if features.shape[1] == 0:
        return _result(
            "skipped",
            "После исключений и проверки типов не осталось подходящих признаков",
            threshold=threshold,
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_predictions = np.full(len(labels), np.nan, dtype=np.float64)
    fold_auc: list[float] = []
    importance_accumulator = {feature: 0.0 for feature in feature_types}

    try:
        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(features, labels)
        ):
            pipeline, transformer_features = _build_pipeline(
                feature_types,
                random_seed=seed + fold_index,
            )
            pipeline.fit(features.iloc[train_indices], labels[train_indices])
            validation_predictions = pipeline.predict_proba(
                features.iloc[validation_indices]
            )[:, 1]
            oof_predictions[validation_indices] = validation_predictions
            score = float(
                roc_auc_score(labels[validation_indices], validation_predictions)
            )
            fold_auc.append(score)
            for feature, importance in _fold_importance(
                pipeline, transformer_features
            ).items():
                importance_accumulator[feature] += importance
    except (LightGBMError, TypeError, ValueError) as exc:
        return _result(
            "error",
            f"Не удалось выполнить Adversarial Validation: {type(exc).__name__}: {exc}",
            threshold=threshold,
        )

    if not np.isfinite(oof_predictions).all():
        return _result(
            "error",
            "Не для всех строк получено отложенное предсказание",
            threshold=threshold,
        )

    overall_auc = float(roc_auc_score(labels, oof_predictions))
    averaged_importance = {
        feature: float(value / n_splits)
        for feature, value in importance_accumulator.items()
    }
    ordered_importance = dict(
        sorted(averaged_importance.items(), key=lambda item: (-item[1], item[0]))
    )

    alert: bool | None = None
    status: Status = "ok"
    reason: str | None = None
    if threshold is not None:
        alert = overall_auc > threshold and not isclose(
            overall_auc,
            threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        if alert:
            status = "warning"
            reason = (
                f"Отложенный ROC-AUC={overall_auc:.6g} превышает "
                f"настроенный порог {threshold:.6g}"
            )

    return _result(
        status,
        reason,
        roc_auc=overall_auc,
        threshold=threshold,
        fold_auc=[float(value) for value in fold_auc],
        feature_importance=ordered_importance,
        importance_type="mean_normalized_gain_across_folds",
        alert=alert,
    )
