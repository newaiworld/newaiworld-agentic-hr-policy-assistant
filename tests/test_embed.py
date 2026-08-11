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
    EmbeddingError,
    _validate_document_embeddings,
    embed_documents,
    _validate_query,
    embed_query,
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

def test_embed_documents_preserves_order_and_encoder_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document embedding must preserve ordered input-to-row mapping."""

    texts = (
        "First policy passage.",
        "Second policy passage.",
    )

    expected = embed_module.np.array(
        [
            [1.0] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,
        ],
        dtype=float,
    )

    model = Mock()
    model.encode.return_value = expected

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    result = embed_documents(
        texts,
        batch_size=2,
    )

    assert result is expected

    model.encode.assert_called_once_with(
        list(texts),
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


@pytest.mark.parametrize(
    "texts",
    [
        [],
        ["Policy text."],
        "Policy text.",
        None,
    ],
)
def test_embed_documents_rejects_non_tuple_input(
    texts: object,
) -> None:
    """The document-embedding API requires deterministic tuple input."""

    with pytest.raises(
        TypeError,
        match="tuple",
    ):
        embed_documents(texts)  # type: ignore[arg-type]


def test_embed_documents_rejects_empty_tuple() -> None:
    """At least one policy text is required."""

    with pytest.raises(
        ValueError,
        match="at least one document",
    ):
        embed_documents(())


@pytest.mark.parametrize(
    "texts",
    [
        ("",),
        ("   ",),
        ("Valid policy.", ""),
        ("Valid policy.", "\n\t"),
    ],
)
def test_embed_documents_rejects_blank_text(
    texts: tuple[str, ...],
) -> None:
    """Blank policy chunks must fail before model encoding."""

    with pytest.raises(
        ValueError,
        match="blank",
    ):
        embed_documents(texts)


@pytest.mark.parametrize(
    "texts",
    [
        ("Valid policy.", 123),
        ("Valid policy.", None),
        ("Valid policy.", b"bytes"),
    ],
)
def test_embed_documents_rejects_non_string_members(
    texts: tuple[object, ...],
) -> None:
    """Every document in the ordered collection must be text."""

    with pytest.raises(
        TypeError,
        match="only strings",
    ):
        embed_documents(texts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        True,
        1.5,
        "32",
        None,
    ],
)
def test_embed_documents_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    """Batch size must be a positive non-boolean integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        embed_documents(
            ("Policy text.",),
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_embed_documents_wraps_encoder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model failures must become clean embedding-pipeline errors."""

    model = Mock()
    model.encode.side_effect = RuntimeError(
        "encoder failed"
    )

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="Failed to embed document texts",
    ):
        embed_documents(
            ("Policy text.",),
        )


def test_validate_document_embeddings_rejects_wrong_result_type() -> None:
    """The encoder must return a NumPy array."""

    with pytest.raises(
        EmbeddingError,
        match="unexpected result type",
    ):
        _validate_document_embeddings(
            [[0.0] * 384],
            expected_rows=1,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (384,),
        (1, 383),
        (1, 385),
        (2, 384),
    ],
)
def test_validate_document_embeddings_rejects_wrong_shape(
    shape: tuple[int, ...],
) -> None:
    """The embedding matrix must exactly match rows × 384."""

    embeddings = embed_module.np.zeros(
        shape,
        dtype=float,
    )

    with pytest.raises(
        EmbeddingError,
        match="unexpected shape",
    ):
        _validate_document_embeddings(
            embeddings,
            expected_rows=1,
        )


def test_validate_document_embeddings_rejects_non_float_dtype() -> None:
    """Dense semantic embeddings must be floating point."""

    embeddings = embed_module.np.zeros(
        (1, 384),
        dtype=int,
    )

    with pytest.raises(
        EmbeddingError,
        match="floating-point",
    ):
        _validate_document_embeddings(
            embeddings,
            expected_rows=1,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_document_embeddings_rejects_non_finite_values(
    bad_value: float,
) -> None:
    """NaN and infinite values must never enter the vector store."""

    embeddings = embed_module.np.zeros(
        (1, 384),
        dtype=float,
    )
    embeddings[0, 0] = bad_value

    with pytest.raises(
        EmbeddingError,
        match="non-finite",
    ):
        _validate_document_embeddings(
            embeddings,
            expected_rows=1,
        )


@pytest.mark.parametrize(
    "query",
    [
        None,
        123,
        3.14,
        True,
        ["PTO"],
    ],
)
def test_embed_query_rejects_non_string_input(
    query: object,
) -> None:
    """Query embedding accepts strings only."""

    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        embed_query(query)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
        "\n\t ",
    ],
)
def test_embed_query_rejects_blank_input(
    query: str,
) -> None:
    """Blank search queries must fail before model encoding."""

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        embed_query(query)


def test_embed_query_applies_retrieval_instruction_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BGE retrieval instruction must prefix the user query once."""

    embedding = embed_module.np.zeros(
        (384,),
        dtype=float,
    )
    embedding[0] = 1.0

    model = Mock()
    model.encode.return_value = embedding

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    result = embed_query(
        "What is the PTO approval process?"
    )

    assert result is embedding

    model.encode.assert_called_once_with(
        QUERY_INSTRUCTION
        + "What is the PTO approval process?",
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_query_does_not_modify_user_query_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query body must be appended exactly as supplied."""

    embedding = embed_module.np.zeros(
        (384,),
        dtype=float,
    )
    embedding[0] = 1.0

    model = Mock()
    model.encode.return_value = embedding

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    query = "  PTO policy?  "

    embed_query(query)

    model.encode.assert_called_once_with(
        QUERY_INSTRUCTION + query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_query_wraps_encoder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model errors must become clean query embedding errors."""

    model = Mock()
    model.encode.side_effect = RuntimeError(
        "encoder failed"
    )

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="Failed to embed query text",
    ):
        embed_query(
            "What is the remote work policy?"
        )


def test_embed_query_rejects_wrong_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query encoder must return a NumPy array."""

    model = Mock()
    model.encode.return_value = [0.0] * 384

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="unexpected result type",
    ):
        embed_query(
            "What is the PTO policy?"
        )


@pytest.mark.parametrize(
    "shape",
    [
        (383,),
        (385,),
        (1, 384),
        (2, 384),
    ],
)
def test_embed_query_rejects_wrong_shape(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
) -> None:
    """One query must produce exactly one 384-dimensional vector."""

    model = Mock()
    model.encode.return_value = embed_module.np.zeros(
        shape,
        dtype=float,
    )

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="unexpected shape",
    ):
        embed_query(
            "What is the PTO policy?"
        )


def test_embed_query_rejects_non_float_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query vectors must contain floating-point values."""

    model = Mock()
    model.encode.return_value = embed_module.np.zeros(
        (384,),
        dtype=int,
    )

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="floating-point",
    ):
        embed_query(
            "What is the PTO policy?"
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_embed_query_rejects_non_finite_values(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    """NaN and infinite query vectors must never reach retrieval."""

    embedding = embed_module.np.zeros(
        (384,),
        dtype=float,
    )
    embedding[0] = bad_value

    model = Mock()
    model.encode.return_value = embedding

    monkeypatch.setattr(
        embed_module,
        "get_embedding_model",
        Mock(return_value=model),
    )

    with pytest.raises(
        EmbeddingError,
        match="non-finite",
    ):
        embed_query(
            "What is the PTO policy?"
        )
