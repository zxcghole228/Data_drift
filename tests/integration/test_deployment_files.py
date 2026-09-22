"""Static safeguards for the Docker and CI deployment contract."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_runs_streamlit_as_non_root_with_healthcheck() -> None:
    dockerfile = _text("Dockerfile")

    assert dockerfile.startswith("FROM python:3.13-slim")
    assert "libgomp1" in dockerfile
    assert "useradd --uid 10001" in dockerfile
    assert "USER app" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8501/_stcore/health" in dockerfile
    assert 'CMD ["python", "-m", "streamlit", "run"' in dockerfile
    assert "COPY . " not in dockerfile


def test_runtime_requirements_exclude_development_tools() -> None:
    runtime_requirements = _text("requirements-runtime.txt")
    development_requirements = _text("requirements.txt")

    assert "streamlit==" in runtime_requirements
    assert "lightgbm==" in runtime_requirements
    assert "pytest==" not in runtime_requirements
    assert "jupyter==" not in runtime_requirements
    assert "-r requirements-runtime.txt" in development_requirements


def test_dockerignore_excludes_local_and_generated_artifacts() -> None:
    ignored = {
        line.strip()
        for line in _text(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".git", ".venv", ".env", "data", "outputs", "report", "tests"} <= ignored


def test_ci_runs_environment_tests_docker_and_healthcheck() -> None:
    workflow = _text(".github/workflows/ci.yml")

    assert 'python-version: "3.13"' in workflow
    assert "python scripts/check_environment.py" in workflow
    assert "python -m pytest -q" in workflow
    assert "docker build --tag data-drift-guardian:ci ." in workflow
    assert "Verify non-root runtime user" in workflow
    assert "Docker build and healthcheck" in workflow
    assert "_stcore/health" in workflow
