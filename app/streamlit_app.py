"""Интерактивный интерфейс Data Drift Guardian."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import pandas as pd
import streamlit as st

from data_drift_guardian import analyze
from data_drift_guardian.config import load_config
from data_drift_guardian.contracts import AnalysisConfig, AnalysisResult, Status
from data_drift_guardian.ingestion import load_table
from data_drift_guardian.reporting import distribution_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ANALYSIS_STATE_KEY = "analysis_payload"
FEATURE_WIDGET_KEY = "distribution_feature"

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


def clear_analysis() -> None:
    """Удалить устаревший результат после изменения входных файлов."""
    st.session_state.pop(ANALYSIS_STATE_KEY, None)
    st.session_state.pop(FEATURE_WIDGET_KEY, None)


def _status_text(status: Status) -> str:
    icon, label = STATUS_VIEW[status]
    return f"{icon} {label} (`{status}`)"


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
    source: str,
    feature: str | None,
    check: dict[str, Any],
) -> dict[str, str]:
    status = check["status"]
    return {
        "Источник": source,
        "Признак": feature or "—",
        "Проверка": str(check["name"]),
        "Статус": _status_text(status),
        "Значение": _format_value(check["value"]),
        "Порог": _format_value(check["threshold"]),
        "p-value": _format_value(check["p_value"]),
        "Алерт": _format_value(check["alert"]),
        "Причина": str(check["reason"] or "—"),
    }


def _quality_rows(result: AnalysisResult) -> list[dict[str, str]]:
    rows = [
        _check_row(source="quality", feature=None, check=check)
        for check in result["quality"]["dataset_checks"]
    ]
    for feature, checks in result["quality"]["feature_checks"].items():
        rows.extend(
            _check_row(source="quality", feature=feature, check=check)
            for check in checks
        )
    return rows


def _drift_rows(result: AnalysisResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for feature, feature_result in result["drift"]["features"].items():
        rows.extend(
            _check_row(source="drift", feature=feature, check=check)
            for check in feature_result["checks"].values()
        )
    return rows


def _feature_rows(result: AnalysisResult) -> list[dict[str, object]]:
    return [
        {
            "Признак": feature,
            "Тип": feature_result["feature_type"],
            "Статус": _status_text(feature_result["status"]),
            "Reference valid": feature_result["n_reference_valid"],
            "Current valid": feature_result["n_current_valid"],
            "Алерт": _format_value(feature_result["alert"]),
            "Причина": feature_result["reason"] or "—",
        }
        for feature, feature_result in result["drift"]["features"].items()
    ]


def _render_alerts(result: AnalysisResult) -> None:
    st.subheader("Алерты")
    if not result["alerts"]:
        st.success("Алертов нет.")
        return

    for alert in result["alerts"]:
        feature = f" · признак `{alert['feature']}`" if alert["feature"] else ""
        message = (
            f"**{alert['source']} / {alert['check']}**{feature}: "
            f"{alert['message']}"
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


def _render_quality(result: AnalysisResult) -> None:
    st.subheader("Data Quality")
    _show_status(result["quality"]["status"], prefix="Качество данных")
    if result["quality"]["reason"]:
        st.write(result["quality"]["reason"])
    rows = _quality_rows(result)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Проверки качества не выполнялись.")


def _render_drift(result: AnalysisResult) -> None:
    st.subheader("Data Drift")
    _show_status(result["drift"]["status"], prefix="Drift-проверки")
    if result["drift"]["reason"]:
        st.write(result["drift"]["reason"])

    feature_rows = _feature_rows(result)
    if feature_rows:
        st.markdown("#### Сводка по признакам")
        st.dataframe(
            pd.DataFrame(feature_rows), hide_index=True, width="stretch"
        )

    check_rows = _drift_rows(result)
    if check_rows:
        st.markdown("#### Метрики")
        st.dataframe(
            pd.DataFrame(check_rows), hide_index=True, width="stretch"
        )
    elif not feature_rows:
        st.info("Drift-метрики не рассчитывались.")


def _render_adversarial(result: AnalysisResult) -> None:
    with st.expander(
        "Adversarial Validation",
        expanded=result["adversarial"]["status"] not in {"ok", "skipped"},
    ):
        _show_status(result["adversarial"]["status"], prefix="ML-проверка")
        if result["adversarial"]["reason"]:
            st.write(result["adversarial"]["reason"])
        if result["adversarial"]["roc_auc"] is not None:
            st.metric("ROC-AUC", f"{result['adversarial']['roc_auc']:.4f}")


def _render_distribution(
    result: AnalysisResult,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    available_features = [
        feature
        for feature in result["drift"]["features"]
        if feature in reference.columns and feature in current.columns
    ]
    if not available_features:
        st.info("Нет общего корректного признака для построения графика.")
        return

    st.subheader("Сравнение распределений")
    selected_feature = st.selectbox(
        "Признак",
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


def render_result(
    result: AnalysisResult,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    """Показать единый результат анализа без повторного вычисления метрик."""
    st.header("Результат анализа")
    _show_status(result["summary"]["status"])

    first, second, third, fourth = st.columns(4)
    first.metric("Строк Reference", result["metadata"]["reference_rows"])
    second.metric("Строк Current", result["metadata"]["current_rows"])
    third.metric("Проанализировано признаков", result["summary"]["analyzed_features"])
    fourth.metric("Алертов", result["summary"]["n_alerts"])
    st.caption(
        f"Версия контракта: {result['contract_version']} · "
        f"seed: {result['metadata']['random_seed']}"
    )

    _render_alerts(result)
    _render_schema(result)
    _render_quality(result)
    _render_drift(result)
    _render_adversarial(result)
    _render_distribution(result, reference, current)


def _run_analysis(
    reference_file: UploadedFileLike | None,
    current_file: UploadedFileLike | None,
    *,
    use_default_config: bool,
    config_file: UploadedFileLike | None,
) -> None:
    if reference_file is None or current_file is None:
        raise ValueError("Загрузите оба файла: Reference и Current.")
    if not use_default_config and config_file is None:
        raise ValueError("Загрузите YAML-конфигурацию или выберите встроенную.")

    reference = load_uploaded_table(reference_file)
    current = load_uploaded_table(current_file)
    config = (
        load_config(DEFAULT_CONFIG_PATH)
        if use_default_config
        else load_uploaded_config(config_file)
    )
    result = analyze(reference, current, config=config)
    st.session_state[ANALYSIS_STATE_KEY] = {
        "result": result,
        "reference": reference,
        "current": current,
    }


def main() -> None:
    st.set_page_config(page_title="Data Drift Guardian", page_icon="🛡️", layout="wide")
    st.title("🛡️ Data Drift Guardian")
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
            on_change=clear_analysis,
            help="Эталонная выборка в формате CSV или Parquet.",
        )
        current_file = st.file_uploader(
            "Current",
            type=["csv", "parquet"],
            key="current_file",
            on_change=clear_analysis,
            help="Текущая выборка в формате CSV или Parquet.",
        )
        config_source = st.radio(
            "Конфигурация",
            ["Встроенная", "Загрузить YAML"],
            key="config_source",
            on_change=clear_analysis,
        )
        config_file = None
        if config_source == "Загрузить YAML":
            config_file = st.file_uploader(
                "YAML-конфигурация",
                type=["yaml", "yml"],
                key="config_file",
                on_change=clear_analysis,
            )
        else:
            st.caption("Используется `configs/default.yaml`.")

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
            with st.spinner("Читаем данные и рассчитываем проверки…"):
                _run_analysis(
                    reference_file,
                    current_file,
                    use_default_config=config_source == "Встроенная",
                    config_file=config_file,
                )
        except Exception as exc:  # UI обязан превратить ошибку входа в сообщение.
            st.error(f"Не удалось выполнить анализ: {type(exc).__name__}: {exc}")

    payload = st.session_state.get(ANALYSIS_STATE_KEY)
    if payload is None:
        st.info("Загрузите две таблицы и явно запустите анализ.")
        return

    render_result(
        payload["result"],
        payload["reference"],
        payload["current"],
    )


if __name__ == "__main__":
    main()
