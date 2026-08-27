"""Deployment contract tests for the Cloud Run production boundary."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_runtime_is_pinned_to_frozen_version() -> None:
    """Deployment must preserve the frozen Python runtime version."""

    path = PROJECT_ROOT / ".python-version"

    assert path.is_file()

    assert path.read_text(
        encoding="utf-8"
    ) == "3.11.15\n"


def test_dockerfile_uses_frozen_python_runtime() -> None:
    """Cloud Run container must use the frozen Python runtime."""

    path = PROJECT_ROOT / "Dockerfile"

    assert path.is_file()

    text = path.read_text(
        encoding="utf-8"
    )

    assert text.startswith(
        "FROM python:3.11.15-slim\n"
    )


def test_dockerfile_uses_canonical_build_lifecycle() -> None:
    """Image build must compose the verified build.sh lifecycle."""

    text = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    assert "COPY . ." in text

    assert "RUN ./build.sh" in text

    assert (
        "python -m rag.index build"
        not in text
    )

    assert (
        "python -m rag.index publish"
        not in text
    )


def test_dockerfile_uses_cloud_run_runtime_contract() -> None:
    """Container runtime must bind Uvicorn to Cloud Run's port."""

    text = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    assert "0.0.0.0" in text

    assert "PORT" in text

    assert "uvicorn" in text

    assert "app.main:app" in text


def test_dockerfile_does_not_rebuild_index_at_runtime() -> None:
    """Runtime command must not invoke the offline index lifecycle."""

    text = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    runtime_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(
            (
                "CMD ",
                "ENTRYPOINT ",
            )
        )
    ]

    assert runtime_lines

    runtime = "\n".join(
        runtime_lines
    )

    assert "rag.index" not in runtime

    assert "build.sh" not in runtime


def test_dockerignore_excludes_local_and_generated_state() -> None:
    """Container build context must exclude local/generated state."""

    path = PROJECT_ROOT / ".dockerignore"

    assert path.is_file()

    ignored = {
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            line.strip()
            and not line.lstrip().startswith("#")
        )
    }

    required = {
        ".git",
        ".venv",
        "venv",
        ".env",
        "__pycache__",
        ".pytest_cache",
        "logs",
        "chroma_db",
        ".chroma_db.build",
        ".chroma_db.backup",
    }

    assert required <= ignored


def test_build_script_uses_canonical_index_lifecycle() -> None:
    """Deployment build must compose the verified build/publish CLI."""

    path = PROJECT_ROOT / "build.sh"

    assert path.is_file()

    text = path.read_text(
        encoding="utf-8"
    )

    assert text.startswith(
        "#!/usr/bin/env bash\n"
    )

    assert "set -euo pipefail" in text

    install = text.index(
        "python -m pip install -r requirements.txt"
    )

    build = text.index(
        "python -m rag.index build"
    )

    publish = text.index(
        "python -m rag.index publish"
    )

    assert install < build < publish


def test_build_script_verifies_published_index_contract() -> None:
    """Deployment build must fail closed on count/freshness mismatch."""

    text = (
        PROJECT_ROOT
        / "build.sh"
    ).read_text(
        encoding="utf-8"
    )

    assert "collection.count()" in text

    assert "count != 400" in text

    assert "is_index_current(" in text

    assert "if not current:" in text

    assert (
        "deployment_index_verification=PASS"
        in text
    )


def test_container_configuration_commits_no_secrets() -> None:
    """Docker deployment artifacts must contain no secret values."""

    paths = (
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / ".dockerignore",
    )

    text = "\n".join(
        path.read_text(
            encoding="utf-8"
        )
        for path in paths
        if path.exists()
    )

    assert "sk-" not in text

    assert "OPENROUTER_API_KEY=" not in text

    assert "LLM_API_KEY=" not in text


def test_dockerfile_preinstalls_frozen_cpu_torch() -> None:
    """Container must resolve frozen Torch from the CPU-only index."""
    text = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    cpu_index = (
        "https://download.pytorch.org/whl/cpu"
    )

    assert cpu_index in text
    assert '"torch==2.13.0"' in text

    cpu_index_position = text.index(cpu_index)
    build_position = text.index("RUN ./build.sh")

    assert cpu_index_position < build_position


def test_dockerfile_configures_offline_embedding_runtime() -> None:
    """Runtime image must use the baked embedding model without Hub access."""
    dockerfile = Path("Dockerfile").read_text(
        encoding="utf-8",
    )

    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile


def test_dockerfile_preserves_chroma_runtime_configuration() -> None:
    """Offline model configuration must not replace the Chroma runtime path."""
    dockerfile = Path("Dockerfile").read_text(
        encoding="utf-8",
    )

    assert "CHROMA_DIR=chroma_db" in dockerfile


def test_dockerfile_enables_embedding_offline_mode_only_after_build() -> None:
    """Build may acquire model artifacts; deployed runtime must be offline."""

    dockerfile = Path(
        "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    lines = [
        line.strip()
        for line in dockerfile.splitlines()
    ]

    build_index = lines.index(
        "RUN ./build.sh"
    )

    hf_offline_index = lines.index(
        "ENV HF_HUB_OFFLINE=1"
    )

    transformers_offline_index = lines.index(
        "ENV TRANSFORMERS_OFFLINE=1"
    )

    assert build_index < hf_offline_index
    assert (
        build_index
        < transformers_offline_index
    )
