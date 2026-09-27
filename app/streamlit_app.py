"""Интерактивный интерфейс Data Drift Guardian."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Protocol, cast

import pandas as pd
import streamlit as st

from data_drift_guardian import analyze
from data_drift_guardian.config import load_config, validate_config
from data_drift_guardian.contracts import AnalysisConfig, AnalysisResult, Status
from data_drift_guardian.ingestion import load_table
from data_drift_guardian.online.dashboard import render_online_dashboard
from data_drift_guardian.reporting import distribution_figure, export_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ICON_PATH = PROJECT_ROOT / "app" / "assets" / "data_drift_guardian.png"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_ONLINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "online.yaml"
APP_MODE_KEY = "application_mode"
ANALYSIS_STATE_KEY = "analysis_payload"
FEATURE_WIDGET_KEY = "distribution_feature"
STATUS_FILTER_KEY = "result_status_filter"
FEATURE_FILTER_KEY = "result_feature_filter"
ADVERSARIAL_OVERRIDE_KEY = "override_adversarial"
ADVERSARIAL_ENABLED_KEY = "adversarial_enabled"
USE_AUC_THRESHOLD_KEY = "use_auc_threshold"
AUC_THRESHOLD_KEY = "auc_threshold"

RESULT_WIDGET_KEYS = (
    FEATURE_WIDGET_KEY,
    STATUS_FILTER_KEY,
    FEATURE_FILTER_KEY,
)

STATUS_VIEW: dict[Status, tuple[str, str]] = {
    "ok": ("✅", "Успешно"),
    "warning": ("⚠️", "Предупреждение"),
    "critical": ("🚨", "Критическая проблема"),
    "skipped": ("⏭️", "Проверка пропущена"),
    "error": ("❌", "Ошибка"),
}


class UploadedFileLike(Protocol):
    """Минимальный интерфейс загруженного через Streamlit файла."""

    name: str

    def getvalue(self) -> bytes:
        """Вернуть полное содержимое файла."""


def load_uploaded_table(uploaded_file: UploadedFileLike) -> pd.DataFrame:
    """Передать загруженный файл существующему файловому адаптеру."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with TemporaryDirectory(prefix="data-drift-table-") as temporary_directory:
        temporary_path = Path(temporary_directory) / f"uploaded{suffix}"
        temporary_path.write_bytes(uploaded_file.getvalue())
        return load_table(temporary_path)


def load_uploaded_config(uploaded_file: UploadedFileLike) -> AnalysisConfig:
    """Прочитать загруженный YAML через общий валидатор конфигурации."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with TemporaryDirectory(prefix="data-drift-config-") as temporary_directory:
        temporary_path = Path(temporary_directory) / f"uploaded{suffix}"
        temporary_path.write_bytes(uploaded_file.getvalue())
        return load_config(temporary_path)


def _clear_result_widgets() -> None:
    for key in RESULT_WIDGET_KEYS:
        st.session_state.pop(key, None)


def clear_analysis() -> None:
    """Удалить сохранённый результат перед явным повторным анализом."""
    st.session_state.pop(ANALYSIS_STATE_KEY, None)
    _clear_result_widgets()


def mark_analysis_stale() -> None:
    """Пометить результат неактуальным после изменения входов или настроек."""
    payload = st.session_state.get(ANALYSIS_STATE_KEY)
    if isinstance(payload, dict):
        payload["stale"] = True
        st.session_state[ANALYSIS_STATE_KEY] = payload
    _clear_result_widgets()


def _status_text(status: Status) -> str:
    icon, label = STATUS_VIEW[status]
    return f"{icon} {label}"


def _show_status(status: Status, *, prefix: str = "Итоговый статус") -> None:
    message = f"{prefix}: {_status_text(status)}"
    if status == "ok":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status in {"critical", "error"}:
        st.error(message)
    else:
        st.info(message)


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _check_row(
    *,
    feature: str | None,
    check: dict[str, Any],
) -> dict[str, object]:
    status = cast(Status, check["status"])
    details = check.get("details")
    if not isinstance(details, dict):
        details = {}
    return {
        "_status": status,
        "_feature": feature,
        "_reason": check["reason"],
        "_p_value": check["p_value"],
        "_adjusted_p_value": check["adjusted_p_value"],
        "_details": details,
        "Признак": feature or "Вся таблица",
        "Проверка": str(check["name"]),
        "Статус": _status_text(status),
        "Значение": _format_value(check["value"]),
        "Порог": _format_value(check["threshold"]),
        "Источник порога": str(details.get("threshold_source", "—")),
    }


def _quality_rows(result: AnalysisResult) -> list[dict[str, object]]:
    rows = [
        _check_row(feature=None, check=check)
        for check in result["quality"]["dataset_checks"]
    ]
    for feature, checks in result["quality"]["feature_checks"].items():
        rows.extend(
            _check_row(feature=feature, check=check)
            for check in checks
        )
    return rows


def _drift_rows(result: AnalysisResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature, feature_result in result["drift"]["features"].items():
        rows.extend(
            _check_row(feature=feature, check=check)
            for check in feature_result["checks"].values()
        )
    return rows


def _feature_rows(result: AnalysisResult) -> list[dict[str, object]]:
    return [
        {
            "_status": feature_result["status"],
            "_feature": feature,
            "Признак": feature,
            "Тип": feature_result["feature_type"],
            "Статус": _status_text(feature_result["status"]),
            "Reference valid": feature_result["n_reference_valid"],
            "Current valid": feature_result["n_current_valid"],
            "Причина": feature_result["reason"] or "—",
        }
        for feature, feature_result in result["drift"]["features"].items()
    ]


def _matching_rows(
    rows: list[dict[str, object]],
    statuses: set[Status],
    feature: str | None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["_status"] in statuses
        and (feature is None or row["_feature"] == feature)
    ]


def _visible_rows(
    rows: list[dict[str, object]],
    statuses: set[Status],
    feature: str | None,
) -> list[dict[str, object]]:
    visible = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in _matching_rows(rows, statuses, feature)
    ]
    if not visible:
        return visible

    empty_values = (None, "", "—")
    empty_columns = {
        key
        for key in visible[0]
        if all(row.get(key) in empty_values for row in visible)
    }
    return [
        {key: value for key, value in row.items() if key not in empty_columns}
        for row in visible
    ]


def _table_height(row_count: int, *, maximum_rows: int = 10) -> int:
    visible_rows = min(max(row_count, 1), maximum_rows)
    return 39 + visible_rows * 35


def _show_dataframe(frame: pd.DataFrame) -> None:
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=_table_height(len(frame)),
    )


def _render_check_details(
    rows: list[dict[str, object]],
    statuses: set[Status],
    feature_filter: str | None,
    *,
    show_statistical_significance: bool,
    show_category_diagnostics: bool,
) -> None:
    matching = _matching_rows(rows, statuses, feature_filter)

    reason_rows = [
        {
            "Признак": row["Признак"],
            "Проверка": row["Проверка"],
            "Статус": row["Статус"],
            "Пояснение": row["_reason"],
        }
        for row in matching
        if row["_reason"] not in (None, "")
    ]
    if reason_rows:
        with st.expander(f"Пояснения к проверкам ({len(reason_rows)})"):
            _show_dataframe(pd.DataFrame(reason_rows))

    if show_statistical_significance:
        significance_rows = [
            {
                "Признак": row["Признак"],
                "Проверка": row["Проверка"],
                "p-value": _format_value(row["_p_value"]),
                "Скорр. p-value": _format_value(row["_adjusted_p_value"]),
                "Статус": row["Статус"],
            }
            for row in matching
            if row["_p_value"] is not None
            or row["_adjusted_p_value"] is not None
        ]
        if significance_rows:
            with st.expander(
                f"Статистическая значимость ({len(significance_rows)})"
            ):
                _show_dataframe(pd.DataFrame(significance_rows))

    if not show_category_diagnostics:
        return

    category_rows: list[dict[str, object]] = []
    diagnostic_keys = {
        "cramers_v",
        "new_category_count",
        "disappeared_category_count",
        "pooled_category_count",
        "pooled_observation_fraction",
    }
    for row in matching:
        details = cast(dict[str, Any], row["_details"])
        if not diagnostic_keys.intersection(details):
            continue
        pooled_fraction = details.get("pooled_observation_fraction")
        if isinstance(pooled_fraction, dict):
            pooled_fraction = pooled_fraction.get("combined")
        category_rows.append(
            {
                "Признак": row["Признак"],
                "Проверка": row["Проверка"],
                "Cramér's V": _format_value(details.get("cramers_v")),
                "Новых": _format_value(details.get("new_category_count")),
                "Исчезнувших": _format_value(
                    details.get("disappeared_category_count")
                ),
                "Pooled": _format_value(details.get("pooled_category_count")),
                "Доля pooled": _format_value(pooled_fraction),
            }
        )
    if category_rows:
        with st.expander(f"Категориальная диагностика ({len(category_rows)})"):
            _show_dataframe(pd.DataFrame(category_rows))


def _render_filters(result: AnalysisResult) -> tuple[set[Status], str | None]:
    features = list(result["drift"]["features"])
    with st.expander("Фильтры результата", expanded=False):
        selected_statuses = st.multiselect(
            "Статусы",
            options=list(STATUS_VIEW),
            default=list(STATUS_VIEW),
            format_func=_status_text,
            key=STATUS_FILTER_KEY,
            help="Фильтр применяется к готовому результату и не запускает анализ повторно.",
        )
        selected_feature = st.selectbox(
            "Признак",
            options=["Все признаки", *features],
            key=FEATURE_FILTER_KEY,
        )
    return set(cast(list[Status], selected_statuses)), (
        None if selected_feature == "Все признаки" else selected_feature
    )


def _render_alerts(
    result: AnalysisResult,
    statuses: set[Status],
    feature_filter: str | None,
) -> None:
    st.subheader("Алерты")
    alerts = [
        alert
        for alert in result["alerts"]
        if alert["severity"] in statuses
        and (feature_filter is None or alert["feature"] == feature_filter)
    ]
    if not alerts:
        if result["alerts"]:
            st.info("Нет алертов, соответствующих выбранным фильтрам.")
        else:
            st.success("Алертов нет.")
        return

    for alert in alerts:
        feature = f" · признак `{alert['feature']}`" if alert["feature"] else ""
        message = (
            f"**{alert['source']} / {alert['check']}**{feature}: {alert['message']}"
        )
        if alert["severity"] == "critical":
            st.error(message)
        else:
            st.warning(message)


def _render_schema(result: AnalysisResult) -> None:
    with st.expander("Проверка схемы", expanded=result["schema"]["status"] != "ok"):
        _show_status(result["schema"]["status"], prefix="Схема")
        if result["schema"]["reason"]:
            st.write(result["schema"]["reason"])
        st.json(result["schema"])


def _render_quality(
    result: AnalysisResult,
    statuses: set[Status],
    feature_filter: str | None,
) -> None:
    st.subheader("Data Quality")
    _show_status(result["quality"]["status"], prefix="Качество данных")
    if result["quality"]["reason"]:
        st.write(result["quality"]["reason"])
    all_rows = _quality_rows(result)
    rows = _visible_rows(all_rows, statuses, feature_filter)
    if rows:
        _show_dataframe(pd.DataFrame(rows))
        _render_check_details(
            all_rows,
            statuses,
            feature_filter,
            show_statistical_significance=False,
            show_category_diagnostics=False,
        )
    else:
        st.info("Нет проверок качества, соответствующих выбранным фильтрам.")


def _render_drift(
    result: AnalysisResult,
    statuses: set[Status],
    feature_filter: str | None,
) -> None:
    st.subheader("Data Drift")
    _show_status(result["drift"]["status"], prefix="Drift-проверки")
    if result["drift"]["reason"]:
        st.write(result["drift"]["reason"])

    feature_rows = _visible_rows(_feature_rows(result), statuses, feature_filter)
    if feature_rows:
        st.markdown("#### Сводка по признакам")
        _show_dataframe(pd.DataFrame(feature_rows))

    all_check_rows = _drift_rows(result)
    check_rows = _visible_rows(all_check_rows, statuses, feature_filter)
    if check_rows:
        st.markdown("#### Метрики")
        _show_dataframe(pd.DataFrame(check_rows))
        _render_check_details(
            all_check_rows,
            statuses,
            feature_filter,
            show_statistical_significance=True,
            show_category_diagnostics=True,
        )
    elif not feature_rows:
        st.info("Нет drift-метрик, соответствующих выбранным фильтрам.")


def _render_adversarial(result: AnalysisResult) -> None:
    adversarial = result["adversarial"]
    with st.expander(
        "Adversarial Validation",
        expanded=adversarial["status"] not in {"ok", "skipped"},
    ):
        _show_status(adversarial["status"], prefix="ML-проверка")
        st.caption(
            "Стратегия split: "
            f"{adversarial['split_strategy']} · group column: "
            f"{_format_value(adversarial['group_column'])} · групп: "
            f"{_format_value(adversarial['n_groups'])}"
        )
        if adversarial["reason"]:
            st.write(adversarial["reason"])
        if adversarial["roc_auc"] is not None:
            st.metric("Отложенный ROC-AUC", f"{adversarial['roc_auc']:.4f}")
            st.caption(
                "Порог алерта: "
                f"{_format_value(adversarial['threshold'])} · "
                f"решение: {_format_value(adversarial['alert'])}"
            )
        if adversarial["fold_auc"]:
            st.markdown("#### ROC-AUC по фолдам")
            _show_dataframe(
                pd.DataFrame(
                    {
                        "Фолд": range(1, len(adversarial["fold_auc"]) + 1),
                        "ROC-AUC": adversarial["fold_auc"],
                    }
                )
            )
        if adversarial["feature_importance"]:
            st.markdown("#### Важности признаков")
            importance = pd.DataFrame(
                [
                    {"Признак": feature, "Важность": value}
                    for feature, value in adversarial["feature_importance"].items()
                ]
            )
            _show_dataframe(importance)
            st.caption(
                f"Тип: {adversarial['importance_type']}. Важность показывает "
                "вклад в различение выборок, но не причинный эффект."
            )


def _render_distribution(
    result: AnalysisResult,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_filter: str | None,
) -> None:
    available_features = [
        feature
        for feature in result["drift"]["features"]
        if feature in reference.columns
        and feature in current.columns
        and (feature_filter is None or feature == feature_filter)
    ]
    if not available_features:
        st.info("Нет общего корректного признака для построения графика.")
        return

    if st.session_state.get(FEATURE_WIDGET_KEY) not in available_features:
        st.session_state.pop(FEATURE_WIDGET_KEY, None)
    st.subheader("Сравнение распределений")
    selected_feature = st.selectbox(
        "Признак для графика",
        available_features,
        key=FEATURE_WIDGET_KEY,
    )
    feature_type = result["drift"]["features"][selected_feature]["feature_type"]
    try:
        figure = distribution_figure(
            reference[selected_feature],
            current[selected_feature],
            feature_type,
        )
    except (TypeError, ValueError) as exc:
        st.warning(f"Не удалось построить график: {exc}")
        return
    st.plotly_chart(figure, width="stretch")


def _render_downloads(json_content: bytes, html_content: bytes) -> None:
    st.subheader("Выгрузка результата")
    first, second = st.columns(2)
    first.download_button(
        "Скачать JSON",
        data=json_content,
        file_name="data_drift_result.json",
        mime="application/json",
        key="download_json",
        width="stretch",
    )
    second.download_button(
        "Скачать автономный HTML",
        data=html_content,
        file_name="data_drift_report.html",
        mime="text/html",
        key="download_html",
        width="stretch",
    )
    st.caption(
        "Оба файла созданы из сохранённого результата; скачивание и фильтрация "
        "не запускают статистический анализ повторно."
    )


def render_result(
    result: AnalysisResult,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    elapsed_seconds: float,
    json_content: bytes,
    html_content: bytes,
) -> None:
    """Показать единый сохранённый результат без повторного вычисления метрик."""
    st.header("Результат анализа")
    _show_status(result["summary"]["status"])

    first, second, third, fourth = st.columns(4)
    first.metric("Строк Reference", result["metadata"]["reference_rows"])
    second.metric("Строк Current", result["metadata"]["current_rows"])
    third.metric("Проанализировано признаков", result["summary"]["analyzed_features"])
    fourth.metric("Алертов", result["summary"]["n_alerts"])
    st.caption(
        f"Версия контракта: {result['contract_version']} · "
        f"seed: {result['metadata']['random_seed']} · "
        f"загрузка и анализ: {elapsed_seconds:.3f} с"
    )

    _render_downloads(json_content, html_content)
    with st.expander("Фактически применённая конфигурация", expanded=False):
        st.json(result["effective_config"])

    statuses, feature_filter = _render_filters(result)
    _render_alerts(result, statuses, feature_filter)
    _render_schema(result)
    _render_quality(result, statuses, feature_filter)
    _render_drift(result, statuses, feature_filter)
    _render_adversarial(result)
    _render_distribution(result, reference, current, feature_filter)


def _apply_adversarial_override(
    config: AnalysisConfig,
    *,
    override: bool,
    enabled: bool,
    use_auc_threshold: bool,
    auc_threshold: float,
) -> AnalysisConfig:
    effective_config = deepcopy(config)
    if not override:
        return effective_config

    effective_config["adversarial"]["enabled"] = enabled
    effective_config["adversarial"]["roc_auc_threshold"] = (
        float(auc_threshold) if use_auc_threshold else None
    )
    return validate_config(effective_config)


def _build_download_artifacts(
    result: AnalysisResult,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[bytes, bytes]:
    json_content = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    with TemporaryDirectory(prefix="data-drift-report-") as temporary_directory:
        html_path = Path(temporary_directory) / "report.html"
        export_html(
            result,
            html_path,
            reference=reference,
            current=current,
        )
        html_content = html_path.read_bytes()
    return json_content, html_content


def _run_analysis(
    reference_file: UploadedFileLike | None,
    current_file: UploadedFileLike | None,
    *,
    use_default_config: bool,
    config_file: UploadedFileLike | None,
    override_adversarial: bool,
    adversarial_enabled: bool,
    use_auc_threshold: bool,
    auc_threshold: float,
) -> None:
    if reference_file is None or current_file is None:
        raise ValueError("Загрузите оба файла: Reference и Current.")
    if not use_default_config and config_file is None:
        raise ValueError("Загрузите YAML-конфигурацию или выберите встроенную.")

    started_at = perf_counter()
    reference = load_uploaded_table(reference_file)
    current = load_uploaded_table(current_file)
    config = (
        load_config(DEFAULT_CONFIG_PATH)
        if use_default_config
        else load_uploaded_config(config_file)
    )
    effective_config = _apply_adversarial_override(
        config,
        override=override_adversarial,
        enabled=adversarial_enabled,
        use_auc_threshold=use_auc_threshold,
        auc_threshold=auc_threshold,
    )
    result = analyze(reference, current, config=effective_config)
    elapsed_seconds = perf_counter() - started_at
    json_content, html_content = _build_download_artifacts(
        result,
        reference,
        current,
    )
    st.session_state[ANALYSIS_STATE_KEY] = {
        "result": result,
        "reference": reference,
        "current": current,
        "elapsed_seconds": elapsed_seconds,
        "json_content": json_content,
        "html_content": html_content,
        "stale": False,
    }


def _render_app_header(title: str) -> None:
    """Показать общий заголовок приложения с фирменной иконкой."""

    logo_column, title_column = st.columns(
        [1, 15],
        gap=None,
        vertical_alignment="center",
    )
    with logo_column:
        st.image(str(APP_ICON_PATH), width="stretch")
    with title_column:
        st.title(title)


def main() -> None:
    st.set_page_config(
        page_title="Data Drift Guardian",
        page_icon=str(APP_ICON_PATH),
        layout="wide",
    )
    with st.sidebar:
        application_mode = st.radio(
            "Режим",
            ["Офлайн-анализ", "Онлайн-мониторинг"],
            key=APP_MODE_KEY,
        )

    if application_mode == "Онлайн-мониторинг":
        _render_app_header("Data Drift Guardian · Онлайн")
        render_online_dashboard(DEFAULT_ONLINE_CONFIG_PATH)
        return

    _render_app_header("Data Drift Guardian")
    st.write(
        "Сравните эталонную и текущую таблицы, проверьте качество данных "
        "и доступные метрики дрейфа."
    )

    with st.sidebar:
        st.header("Параметры анализа")
        reference_file = st.file_uploader(
            "Reference",
            type=["csv", "parquet"],
            key="reference_file",
            on_change=mark_analysis_stale,
            help="Эталонная выборка в формате CSV или Parquet.",
        )
        current_file = st.file_uploader(
            "Current",
            type=["csv", "parquet"],
            key="current_file",
            on_change=mark_analysis_stale,
            help="Текущая выборка в формате CSV или Parquet.",
        )
        config_source = st.radio(
            "Конфигурация",
            ["Встроенная", "Загрузить YAML"],
            key="config_source",
            on_change=mark_analysis_stale,
        )
        config_file = None
        if config_source == "Загрузить YAML":
            config_file = st.file_uploader(
                "YAML-конфигурация",
                type=["yaml", "yml"],
                key="config_file",
                on_change=mark_analysis_stale,
            )
        else:
            st.caption("Используется `configs/default.yaml`.")

        st.markdown("#### Adversarial Validation")
        override_adversarial = st.checkbox(
            "Переопределить настройки YAML",
            key=ADVERSARIAL_OVERRIDE_KEY,
            on_change=mark_analysis_stale,
            help="Изменения применяются к копии загруженной конфигурации.",
        )
        adversarial_enabled = False
        use_auc_threshold = False
        auc_threshold = 0.70
        if override_adversarial:
            adversarial_enabled = st.checkbox(
                "Включить ML-проверку",
                value=True,
                key=ADVERSARIAL_ENABLED_KEY,
                on_change=mark_analysis_stale,
            )
            use_auc_threshold = st.checkbox(
                "Использовать порог ROC-AUC",
                value=True,
                key=USE_AUC_THRESHOLD_KEY,
                on_change=mark_analysis_stale,
            )
            if use_auc_threshold:
                auc_threshold = float(
                    st.number_input(
                        "Порог ROC-AUC",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.70,
                        step=0.01,
                        key=AUC_THRESHOLD_KEY,
                        on_change=mark_analysis_stale,
                    )
                )
            else:
                st.caption("Результат AUC будет показан без решения об алерте.")
        else:
            st.caption("Используются настройки секции `adversarial` из YAML.")

        run_requested = st.button(
            "Запустить анализ",
            key="run_analysis",
            type="primary",
            width="stretch",
        )
        st.caption("Анализ запускается только после нажатия кнопки.")

    if run_requested:
        clear_analysis()
        try:
            with st.spinner(
                "Читаем данные, рассчитываем проверки и готовим выгрузки…"
            ):
                _run_analysis(
                    reference_file,
                    current_file,
                    use_default_config=config_source == "Встроенная",
                    config_file=config_file,
                    override_adversarial=override_adversarial,
                    adversarial_enabled=adversarial_enabled,
                    use_auc_threshold=use_auc_threshold,
                    auc_threshold=auc_threshold,
                )
        except Exception as exc:  # UI обязан превратить ошибку входа в сообщение.
            st.error(f"Не удалось выполнить анализ: {type(exc).__name__}: {exc}")

    payload = st.session_state.get(ANALYSIS_STATE_KEY)
    if payload is None:
        st.info("Загрузите две таблицы и явно запустите анализ.")
        return
    if payload.get("stale"):
        st.warning(
            "Входные файлы или настройки изменились. Сохранённый результат "
            "помечен как неактуальный; запустите анализ повторно."
        )
        return

    render_result(
        payload["result"],
        payload["reference"],
        payload["current"],
        elapsed_seconds=payload["elapsed_seconds"],
        json_content=payload["json_content"],
        html_content=payload["html_content"],
    )


if __name__ == "__main__":
    main()
