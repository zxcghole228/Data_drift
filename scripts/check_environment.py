"""Печатает версии общего окружения и проверяет реальные импорты.

Запуск из корня репозитория:
    python scripts/check_environment.py
    python scripts/check_environment.py --json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from typing import TypedDict


class PackageStatus(TypedDict):
    distribution: str
    module: str
    version: str | None
    import_ok: bool
    error: str | None


PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("lightgbm", "lightgbm"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
    ("PyYAML", "yaml"),
    ("pyarrow", "pyarrow"),
    ("pytest", "pytest"),
)


def inspect_package(distribution: str, module: str) -> PackageStatus:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None

    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - диагностируем и двигаемся дальше
        return {
            "distribution": distribution,
            "module": module,
            "version": version,
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "distribution": distribution,
        "module": module,
        "version": version,
        "import_ok": True,
        "error": None,
    }


def collect_environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_3_12": sys.version_info[:2] == (3, 12),
        "packages": [inspect_package(*package) for package in PACKAGES],
    }


def print_table(report: dict[str, object]) -> None:
    print(f"Python: {report['python']} ({report['implementation']})")
    print(f"Platform: {report['platform']}")
    print(f"Python 3.12: {'OK' if report['python_3_12'] else 'FAIL'}")
    print()
    print(f"{'Package':<18} {'Version':<18} Import")
    print("-" * 56)
    for package in report["packages"]:
        assert isinstance(package, dict)
        version = package["version"] or "not installed"
        status = "OK" if package["import_ok"] else f"FAIL ({package['error']})"
        print(f"{package['distribution']:<18} {version:<18} {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="вывести JSON")
    args = parser.parse_args()

    report = collect_environment()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_table(report)

    imports_ok = all(
        package["import_ok"]
        for package in report["packages"]
        if isinstance(package, dict)
    )
    return 0 if report["python_3_12"] and imports_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
