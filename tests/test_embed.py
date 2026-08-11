"""Focused tests for the S4 embedding-model lifecycle."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import rag.embed as embed_module
from rag.embed import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_SEQUENCE_LENGTH,
    QUERY_INSTRUCTION,
    EmbeddingModelLoadError,
    EmbeddingModelValidationError,
    _validate_embedding_model,
    get_embedding_model,
)


def test_embedding_configuration_matches_frozen_spec() -> None:
    """Embedding constants must match the approved S4 configuration."""

    assert embed_module.EMBEDDING_MODEL_NAME == (
        "BAAI/bge-small-en-v1.5"
    )
    assert EMBEDDING_DIMENSION == 384
    assert EMBEDDING_MAX_SEQUENCE_LENGTH == 512
    assert DEFAULT_EMBEDDING_BATCH_SIZE == 32
    assert QUERY_INSTRUCTION == (
        "Represent this sentence for searching relevant passages: "
    )


def test_validate_embedding_model_accepts_frozen_contract() -> None:
    """A compatible model must pass lifecycle validation."""

    model = Mock()
    model.get_embedding_dimension.return_value = 384
    model.max_seq_length = 512

    _validate_embedding_model(model)


@pytest.mark.parametrize(
    ("dimension", "expected_message"),
    [
        (383, "383"),
        (768, "768"),
        (None, "None"),
    ],
)
def test_validate_embedding_model_rejects_wrong_dimension(
    dimension: object,
    expected_message: str,
) -> None:
    """Any non-384 embedding dimension must fail fast."""

    model = Mock()
    model.get_embedding_dimension.return_value = dimension
    model.max_seq_length = 512

    with pytest.raises(
        EmbeddingModelValidationError,
        match="dimension",
    ) as exc_info:
        _validate_embedding_model(model)

    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    "max_seq_length",
    [
        None,
        0,
        450,
        511,
        True,
    ],
)
def test_validate_embedding_model_rejects_insufficient_context(
    max_seq_length: object,
) -> None:
    """The embedding model must expose the approved 512-token context."""

    model = Mock()
    model.get_embedding_dimension.return_value = 384
    model.max_seq_length = max_seq_length

    with pytest.raises(
        EmbeddingModelValidationError,
        match="sequence length",
    ):
        _validate_embedding_model(model)


def test_validate_embedding_model_wraps_dimension_lookup_failure() -> None:
    """Dimension-inspection failures must become project exceptions."""

    model = Mock()
    model.get_embedding_dimension.side_effect = RuntimeError(
        "dimension unavailable"
    )
    model.max_seq_length = 512

    with pytest.raises(
        EmbeddingModelValidationError,
        match="determine the embedding model dimension",
    ):
        _validate_embedding_model(model)


def test_get_embedding_model_loads_expected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifecycle loader must request the frozen BGE model."""

    model = Mock()
    model.get_embedding_dimension.return_value = 384
    model.max_seq_length = 512

    constructor = Mock(return_value=model)

    monkeypatch.setattr(
        embed_module,
        "SentenceTransformer",
        constructor,
    )

    get_embedding_model.cache_clear()

    try:
        result = get_embedding_model()
    finally:
        get_embedding_model.cache_clear()

    assert result is model
    constructor.assert_called_once_with(
        "BAAI/bge-small-en-v1.5"
    )


def test_get_embedding_model_is_cached_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated lifecycle calls must reuse one loaded model."""

    model = Mock()
    model.get_embedding_dimension.return_value = 384
    model.max_seq_length = 512

    constructor = Mock(return_value=model)

    monkeypatch.setattr(
        embed_module,
        "SentenceTransformer",
        constructor,
    )

    get_embedding_model.cache_clear()

    try:
        first = get_embedding_model()
        second = get_embedding_model()
    finally:
        get_embedding_model.cache_clear()

    assert first is model
    assert second is model
    assert first is second
    constructor.assert_called_once()


def test_get_embedding_model_wraps_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-loading failures must expose a clean project exception."""

    constructor = Mock(
        side_effect=RuntimeError("model unavailable")
    )

    monkeypatch.setattr(
        embed_module,
        "SentenceTransformer",
        constructor,
    )

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    get_embedding_model.cache_clear()

    try:
        with pytest.raises(
            EmbeddingModelLoadError,
            match="Failed to load embedding model",
        ):
            get_embedding_model()
    finally:
        get_embedding_model.cache_clear()


def test_get_embedding_model_explains_offline_cache_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline load failures should provide an actionable diagnostic."""

    constructor = Mock(
        side_effect=RuntimeError("model unavailable")
    )

    monkeypatch.setattr(
        embed_module,
        "SentenceTransformer",
        constructor,
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    get_embedding_model.cache_clear()

    try:
        with pytest.raises(
            EmbeddingModelLoadError,
            match="offline mode is enabled",
        ):
            get_embedding_model()
    finally:
        get_embedding_model.cache_clear()
