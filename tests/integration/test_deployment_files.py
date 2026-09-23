"""Static safeguards for the Docker and CI deployment contract."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_runs_streamlit_as_non_root_with_healthcheck() -> None:
    dockerfile = _text("Dockerfile")

    assert dockerfile.startswith("FROM python:3.13-slim")
    assert "libgomp1" in dockerfile
    assert "useradd --uid 10001" in dockerfile
    assert "USER app" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "mkdir -p data/online outputs/online" in dockerfile
    assert "chown -R app:app data outputs" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8501/_stcore/health" in dockerfile
    assert 'CMD ["python", "-m", "streamlit", "run"' in dockerfile
    assert "COPY . " not in dockerfile


def test_runtime_requirements_exclude_development_tools() -> None:
    runtime_requirements = _text("requirements-runtime.txt")
    development_requirements = _text("requirements.txt")

    assert "streamlit==" in runtime_requirements
    assert "lightgbm==" in runtime_requirements
    assert "fastapi==" in runtime_requirements
    assert "uvicorn==" in runtime_requirements
    assert "httpx2==" not in runtime_requirements
    assert "pytest==" not in runtime_requirements
    assert "jupyter==" not in runtime_requirements
    assert "-r requirements-runtime.txt" in development_requirements
    assert "httpx2==" in development_requirements


def test_dockerignore_excludes_local_and_generated_artifacts() -> None:
    ignored = {
        line.strip()
        for line in _text(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".git", ".venv", ".env", "data", "outputs", "report", "tests"} <= ignored


def test_compose_runs_api_and_dashboard_with_healthchecks() -> None:
    compose = yaml.safe_load(_text("compose.yaml"))
    services = compose["services"]
    api = services["api"]
    dashboard = services["dashboard"]

    assert set(services) == {"api", "dashboard"}
    assert api["image"] == dashboard["image"] == "data-drift-guardian:online"
    assert "data_drift_guardian.online" in api["command"]
    assert "streamlit" in dashboard["command"]
    assert "${DDG_API_PORT:-8000}:8000" in api["ports"]
    assert "${DDG_DASHBOARD_PORT:-8501}:8501" in dashboard["ports"]
    assert "/health/live" in api["healthcheck"]["test"][-1]
    assert "/_stcore/health" in dashboard["healthcheck"]["test"][-1]
    assert dashboard["depends_on"]["api"]["condition"] == "service_healthy"


def test_compose_uses_shared_persistent_state_and_non_root_user() -> None:
    compose = yaml.safe_load(_text("compose.yaml"))
    services = compose["services"]
    expected_mounts = {
        "online-state:/app/data/online",
        "online-outputs:/app/outputs/online",
    }

    assert set(compose["volumes"]) == {"online-state", "online-outputs"}
    for service in services.values():
        assert service["user"] == "10001:10001"
        assert set(service["volumes"]) == expected_mounts
        assert (
            service["environment"]["DDG_ONLINE_STATE"]
            == "/app/data/online/monitoring.sqlite3"
        )


def test_compose_environment_example_documents_public_settings() -> None:
    example = _text(".env.example")

    assert "DDG_API_PORT=8000" in example
    assert "DDG_DASHBOARD_PORT=8501" in example
    assert "DDG_ALERT_WEBHOOK_URL=" in example


def test_ci_runs_environment_tests_docker_and_healthcheck() -> None:
    workflow = _text(".github/workflows/ci.yml")

    assert 'python-version: "3.13"' in workflow
    assert "python scripts/check_environment.py" in workflow
    assert "python -m pytest -q -m online_smoke tests/acceptance" in workflow
    assert "python -m pytest -q" in workflow
    assert "docker build --tag data-drift-guardian:ci ." in workflow
    assert "Verify non-root runtime user" in workflow
    assert "Docker build and healthcheck" in workflow
    assert "_stcore/health" in workflow
