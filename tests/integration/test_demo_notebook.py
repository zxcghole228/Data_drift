"""Execute every code cell of the demo notebook in repository order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from plotly.graph_objects import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "demo.ipynb"


def test_demo_notebook_code_cells_execute_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr(Figure, "show", lambda self, *args, **kwargs: None)

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        exec(  # noqa: S102 - the repository-owned notebook is the test target.
            compile(source, f"notebooks/demo.ipynb:{cell['id']}", "exec"),
            namespace,
        )

    experiment_table = namespace["experiment_table"]
    assert len(experiment_table) == 9
    assert set(experiment_table["scenario"]) == {
        "none",
        "numeric",
        "categorical",
        "missingness",
        "combined",
    }
    assert namespace["combined_result"]["adversarial"]["roc_auc"] is not None
    assert (
        namespace["combined_result"]["adversarial"]["split_strategy"]
        == "stratified_kfold"
    )
    assert namespace["combined_result"]["summary"]["has_alerts"] is True
    assert "threshold_source" in namespace["drift_rows"][0]
    assert "cramers_v" in namespace["drift_rows"][0]
    assert "calibration_summary" in namespace
