"""Deployment contract tests for the Render production boundary."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_runtime_is_pinned_to_frozen_version() -> None:
    """Render must use the frozen Python runtime version."""

    path = PROJECT_ROOT / ".python-version"

    assert path.is_file()
    assert path.read_text(
        encoding="utf-8"
    ) == "3.11.15\n"


def test_render_blueprint_uses_frozen_web_service_contract() -> None:
    """Render Blueprint must preserve the V1 deployment boundary."""

    path = PROJECT_ROOT / "render.yaml"

    assert path.is_file()

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(payload, dict)

    services = payload.get("services")

    assert isinstance(services, list)
    assert len(services) == 1

    service = services[0]

    assert service["type"] == "web"
    assert service["runtime"] == "python"
    assert service["plan"] == "free"
    assert service["buildCommand"] == "./build.sh"
    assert service["startCommand"] == (
        "uvicorn app.main:app "
        "--host 0.0.0.0 "
        "--port $PORT"
    )
    assert service["healthCheckPath"] == "/health"


def test_render_blueprint_preserves_environment_contract_without_secrets() -> None:
    """Blueprint declares configuration names without committed secrets."""

    path = PROJECT_ROOT / "render.yaml"

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    service = payload["services"][0]
    env_vars = service["envVars"]

    by_key = {
        item["key"]: item
        for item in env_vars
    }

    assert by_key["CHROMA_DIR"] == {
        "key": "CHROMA_DIR",
        "value": "chroma_db",
    }

    assert by_key["CORPUS_DIR"] == {
        "key": "CORPUS_DIR",
        "value": "corpus",
    }

    for key in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
    ):
        assert by_key[key] == {
            "key": key,
            "sync": False,
        }

    serialized = path.read_text(
        encoding="utf-8"
    )

    assert "sk-" not in serialized
    assert "OPENROUTER_API_KEY" not in serialized


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
    """Deployment build must fail closed on count or freshness mismatch."""

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
    assert "deployment_index_verification=PASS" in text
