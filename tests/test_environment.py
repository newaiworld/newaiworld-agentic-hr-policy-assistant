import sys


def test_python_version() -> None:
    """The project runtime is frozen at Python 3.11."""
    assert sys.version_info[:2] == (3, 11)

def test_rag_chunk_import_does_not_load_heavy_ml_runtime() -> None:
    """Importing chunk configuration must not load Torch/Transformers."""

    import subprocess
    import sys

    code = """
import sys

import rag.chunk

for family in (
    "torch",
    "transformers",
    "sentence_transformers",
):
    loaded = (
        family in sys.modules
        or any(
            name.startswith(family + ".")
            for name in sys.modules
        )
    )

    if loaded:
        raise SystemExit(
            f"unexpected heavy import: {family}"
        )
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )


def test_rag_embed_import_does_not_load_heavy_ml_runtime() -> None:
    """Importing embedding contracts must not eagerly load model libraries."""

    import subprocess
    import sys

    code = """
import sys

import rag.embed

for family in (
    "torch",
    "transformers",
    "sentence_transformers",
):
    loaded = (
        family in sys.modules
        or any(
            name.startswith(family + ".")
            for name in sys.modules
        )
    )

    if loaded:
        raise SystemExit(
            f"unexpected heavy import: {family}"
        )
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )


def test_app_import_does_not_load_heavy_ml_runtime() -> None:
    """Web-process import must not load the local embedding runtime."""

    import subprocess
    import sys

    code = """
import sys

import app.main

for family in (
    "torch",
    "transformers",
    "sentence_transformers",
):
    loaded = (
        family in sys.modules
        or any(
            name.startswith(family + ".")
            for name in sys.modules
        )
    )

    if loaded:
        raise SystemExit(
            f"unexpected heavy import: {family}"
        )
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
