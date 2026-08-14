"""Focused tests for the S4 retrieval-domain contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rag.embed import EmbeddingError
from rag.store import ChromaStoreError

from rag.retrieve import (
    DEFAULT_RETRIEVAL_K,
    RetrievalError,
    RetrievalResult,
    _compile_chroma_where,
    _get_active_policy_collection,
    _query_policy_collection_raw,
    _validate_retrieval_filters,
    _validate_retrieval_k,
    _validate_retrieval_query,
)


def make_result(
    *,
    chunk_id: str = "HR-POL-004__0000__abcdef0123456789",
    doc_id: str = "HR-POL-004",
    title: str = "Remote and Flexible Work Policy",
    section: str = "5.3 International approval",
    section_path: tuple[str, ...] = (
        "Remote and Flexible Work Policy",
        "5. Procedures or Application",
        "5.3 International approval",
    ),
    snippet: str = (
        "International remote work requires written approval."
    ),
    source_format: str = "md",
    distance: float = 0.25,
    similarity: float = 0.75,
) -> RetrievalResult:
    """Build one valid retrieval result for focused contract tests."""

    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        section=section,
        section_path=section_path,
        snippet=snippet,
        source_format=source_format,
        distance=distance,
        similarity=similarity,
    )


def test_retrieval_configuration_matches_frozen_contract() -> None:
    """Default retrieval depth must remain the approved top-k value."""

    assert DEFAULT_RETRIEVAL_K == 5


def test_retrieval_error_is_runtime_error() -> None:
    """Retrieval failures must cross a project-owned runtime boundary."""

    assert issubclass(
        RetrievalError,
        RuntimeError,
    )


def test_retrieval_result_preserves_citation_contract() -> None:
    """A valid result must preserve all citation-ready fields."""

    result = make_result()

    assert result.chunk_id == (
        "HR-POL-004__0000__abcdef0123456789"
    )
    assert result.doc_id == "HR-POL-004"
    assert result.title == "Remote and Flexible Work Policy"
    assert result.section == "5.3 International approval"
    assert result.section_path == (
        "Remote and Flexible Work Policy",
        "5. Procedures or Application",
        "5.3 International approval",
    )
    assert result.snippet == (
        "International remote work requires written approval."
    )
    assert result.source_format == "md"
    assert result.distance == 0.25
    assert result.similarity == 0.75


def test_retrieval_result_is_frozen() -> None:
    """Retrieval results must be immutable after validation."""

    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.section = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "chunk_id",
        "doc_id",
        "title",
        "section",
        "snippet",
        "source_format",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_retrieval_result_rejects_blank_string_fields(
    field_name: str,
    invalid_value: str,
) -> None:
    """Citation string fields must never be empty or whitespace-only."""

    kwargs = {
        field_name: invalid_value,
    }

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        make_result(
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "section_path",
    [
        (),
        ("",),
        (" ",),
        ("Policy", ""),
        ("Policy", " "),
    ],
)
def test_retrieval_result_rejects_invalid_section_path(
    section_path: tuple[str, ...],
) -> None:
    """Section provenance must contain only non-empty headings."""

    with pytest.raises(
        ValueError,
        match="section_path",
    ):
        make_result(
            section_path=section_path,
        )


def test_retrieval_result_rejects_non_tuple_section_path() -> None:
    """The internal section path must remain immutable."""

    with pytest.raises(
        ValueError,
        match="section_path",
    ):
        make_result(
            section_path=[  # type: ignore[arg-type]
                "Remote and Flexible Work Policy",
                "5.3 International approval",
            ],
        )


def test_retrieval_result_requires_title_to_match_path_root() -> None:
    """Policy title and full provenance path must remain aligned."""

    with pytest.raises(
        ValueError,
        match="title must match",
    ):
        make_result(
            title="Wrong title",
        )


def test_retrieval_result_requires_section_to_match_path_leaf() -> None:
    """Public citation section must be the canonical leaf heading."""

    with pytest.raises(
        ValueError,
        match="section must match",
    ):
        make_result(
            section="Wrong section",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("distance", float("nan")),
        ("distance", float("inf")),
        ("distance", float("-inf")),
        ("similarity", float("nan")),
        ("similarity", float("inf")),
        ("similarity", float("-inf")),
    ],
)
def test_retrieval_result_rejects_non_finite_scores(
    field_name: str,
    invalid_value: float,
) -> None:
    """Retrieval scores must always be finite before publication."""

    kwargs = {
        field_name: invalid_value,
    }

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        make_result(
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("distance", "similarity"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (1.0, 0.0),
        (2.0, -1.0),
    ],
)
def test_retrieval_result_accepts_cosine_score_mapping(
    distance: float,
    similarity: float,
) -> None:
    """Similarity must use the frozen higher-is-better orientation."""

    result = make_result(
        distance=distance,
        similarity=similarity,
    )

    assert result.distance == distance
    assert result.similarity == similarity


def test_retrieval_result_rejects_inconsistent_similarity() -> None:
    """A caller must not publish a score inconsistent with distance."""

    with pytest.raises(
        ValueError,
        match="similarity must equal",
    ):
        make_result(
            distance=0.25,
            similarity=0.25,
        )


def test_validate_retrieval_query_accepts_non_empty_string() -> None:
    """A non-empty retrieval query must pass request validation."""

    _validate_retrieval_query(
        "Can I work remotely overseas?"
    )


@pytest.mark.parametrize(
    "query",
    [
        None,
        123,
        True,
        [],
        {},
    ],
)
def test_validate_retrieval_query_rejects_non_string(
    query: object,
) -> None:
    """Retrieval queries must use the public string contract."""

    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        _validate_retrieval_query(
            query,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\t",
        "\n",
        "   \t\n",
    ],
)
def test_validate_retrieval_query_rejects_blank_string(
    query: str,
) -> None:
    """Whitespace-only retrieval queries must fail before embedding."""

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",
    ):
        _validate_retrieval_query(
            query
        )


@pytest.mark.parametrize(
    "k",
    [
        1,
        3,
        5,
        8,
        100,
    ],
)
def test_validate_retrieval_k_accepts_positive_integer(
    k: int,
) -> None:
    """Positive retrieval depths must pass without an invented maximum."""

    _validate_retrieval_k(
        k
    )


@pytest.mark.parametrize(
    "k",
    [
        None,
        1.0,
        "5",
        [],
        {},
    ],
)
def test_validate_retrieval_k_rejects_non_integer(
    k: object,
) -> None:
    """Retrieval depth must use the integer API contract."""

    with pytest.raises(
        TypeError,
        match="k must be an integer",
    ):
        _validate_retrieval_k(
            k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "k",
    [
        True,
        False,
    ],
)
def test_validate_retrieval_k_rejects_boolean(
    k: bool,
) -> None:
    """Booleans must not pass Python's integer subclass relationship."""

    with pytest.raises(
        TypeError,
        match="k must be an integer",
    ):
        _validate_retrieval_k(
            k
        )


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        -10,
    ],
)
def test_validate_retrieval_k_rejects_non_positive_integer(
    k: int,
) -> None:
    """Retrieval depth must request at least one result."""

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        _validate_retrieval_k(
            k
        )


@pytest.mark.parametrize(
    "filters",
    [
        None,
        {},
        {"doc_id": "HR-POL-004"},
        {"source_format": "md"},
        {"source_format": "pdf"},
        {
            "doc_id": "HR-POL-004",
            "source_format": "md",
        },
    ],
)
def test_validate_retrieval_filters_accepts_supported_contract(
    filters: dict[str, str] | None,
) -> None:
    """Supported public retrieval-filter combinations must pass."""

    _validate_retrieval_filters(
        filters
    )


@pytest.mark.parametrize(
    "filters",
    [
        "doc_id=HR-POL-004",
        [],
        (),
        {"doc_id", "HR-POL-004"},
    ],
)
def test_validate_retrieval_filters_rejects_non_dictionary(
    filters: object,
) -> None:
    """Retrieval filters must use a dictionary or None."""

    with pytest.raises(
        TypeError,
        match="filters must be a dictionary or None",
    ):
        _validate_retrieval_filters(
            filters,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "filters",
    [
        {"department": "Engineering"},
        {"title": "Remote and Flexible Work Policy"},
        {
            "doc_id": "HR-POL-004",
            "department": "Engineering",
        },
    ],
)
def test_validate_retrieval_filters_rejects_unsupported_keys(
    filters: dict[str, str],
) -> None:
    """Only the intentionally exposed retrieval filters may be used."""

    with pytest.raises(
        ValueError,
        match="Unsupported retrieval filter key",
    ):
        _validate_retrieval_filters(
            filters
        )


@pytest.mark.parametrize(
    "filters",
    [
        {"doc_id": 4},
        {"doc_id": None},
        {"source_format": 1},
        {"source_format": True},
    ],
)
def test_validate_retrieval_filters_rejects_non_string_values(
    filters: dict[str, object],
) -> None:
    """Filter values must remain explicit strings."""

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        _validate_retrieval_filters(
            filters,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "filters",
    [
        {"doc_id": ""},
        {"doc_id": " "},
        {"source_format": ""},
        {"source_format": "\t"},
    ],
)
def test_validate_retrieval_filters_rejects_blank_values(
    filters: dict[str, str],
) -> None:
    """Whitespace-only filter values must not reach Chroma."""

    with pytest.raises(
        ValueError,
        match="must be a non-empty string",
    ):
        _validate_retrieval_filters(
            filters
        )


@pytest.mark.parametrize(
    "source_format",
    [
        "html",
        "txt",
        "MD",
        " md ",
    ],
)
def test_validate_retrieval_filters_rejects_unsupported_source_format(
    source_format: str,
) -> None:
    """Source-format filters must match the canonical corpus formats."""

    with pytest.raises(
        ValueError,
        match="source_format.*unsupported",
    ):
        _validate_retrieval_filters(
            {
                "source_format": source_format,
            }
        )


@pytest.mark.parametrize(
    "filters",
    [
        {1: "value"},
        {True: "value"},
        {None: "value"},
    ],
)
def test_validate_retrieval_filters_rejects_non_string_keys(
    filters: dict[object, str],
) -> None:
    """Public retrieval-filter keys must be strings."""

    with pytest.raises(
        TypeError,
        match="filter keys must be strings",
    ):
        _validate_retrieval_filters(
            filters,  # type: ignore[arg-type]
        )


def test_get_active_policy_collection_returns_validated_collection(
    tmp_path: Path,
) -> None:
    """Runtime access must compose the existing CP8 store primitives."""

    active = tmp_path / "chroma_db"
    active.mkdir()

    client = Mock()
    collection = Mock()

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            return_value=active,
        ) as resolve_mock,
        patch(
            "rag.retrieve.get_chroma_client",
            return_value=client,
        ) as client_mock,
        patch(
            "rag.retrieve.get_policy_collection",
            return_value=collection,
        ) as collection_mock,
    ):
        result = _get_active_policy_collection()

    assert result is collection
    resolve_mock.assert_called_once_with()
    client_mock.assert_called_once_with(active)
    collection_mock.assert_called_once_with(client)


def test_get_active_policy_collection_rejects_missing_index_before_client(
    tmp_path: Path,
) -> None:
    """A missing published index must not reach PersistentClient."""

    active = tmp_path / "missing_chroma"

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            return_value=active,
        ),
        patch(
            "rag.retrieve.get_chroma_client",
        ) as client_mock,
        patch(
            "rag.retrieve.get_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Active Chroma index directory does not exist",
        ):
            _get_active_policy_collection()

    client_mock.assert_not_called()
    collection_mock.assert_not_called()
    assert not active.exists()


def test_get_active_policy_collection_rejects_non_directory_before_client(
    tmp_path: Path,
) -> None:
    """The active Chroma path must refer to a directory."""

    active = tmp_path / "chroma_db"
    active.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            return_value=active,
        ),
        patch(
            "rag.retrieve.get_chroma_client",
        ) as client_mock,
        patch(
            "rag.retrieve.get_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Active Chroma index path is not a directory",
        ):
            _get_active_policy_collection()

    client_mock.assert_not_called()
    collection_mock.assert_not_called()


def test_get_active_policy_collection_wraps_resolution_failure() -> None:
    """Storage-layer path-resolution failures become retrieval failures."""

    cause = ChromaStoreError(
        "resolution failed"
    )

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            side_effect=cause,
        ),
        patch(
            "rag.retrieve.get_chroma_client",
        ) as client_mock,
        patch(
            "rag.retrieve.get_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Failed to resolve the active Chroma index directory",
        ) as exc_info:
            _get_active_policy_collection()

    assert exc_info.value.__cause__ is cause
    client_mock.assert_not_called()
    collection_mock.assert_not_called()


def test_get_active_policy_collection_wraps_client_failure(
    tmp_path: Path,
) -> None:
    """Persistent-client failures must not leak ChromaStoreError."""

    active = tmp_path / "chroma_db"
    active.mkdir()

    cause = ChromaStoreError(
        "client failed"
    )

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            return_value=active,
        ),
        patch(
            "rag.retrieve.get_chroma_client",
            side_effect=cause,
        ),
        patch(
            "rag.retrieve.get_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Failed to open the active policy retrieval index",
        ) as exc_info:
            _get_active_policy_collection()

    assert exc_info.value.__cause__ is cause
    collection_mock.assert_not_called()


def test_get_active_policy_collection_wraps_collection_failure(
    tmp_path: Path,
) -> None:
    """Existing-collection failures must cross the retrieval boundary."""

    active = tmp_path / "chroma_db"
    active.mkdir()

    client = Mock()
    cause = ChromaStoreError(
        "collection failed"
    )

    with (
        patch(
            "rag.retrieve.resolve_chroma_dir",
            return_value=active,
        ),
        patch(
            "rag.retrieve.get_chroma_client",
            return_value=client,
        ) as client_mock,
        patch(
            "rag.retrieve.get_policy_collection",
            side_effect=cause,
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Failed to open the active policy retrieval index",
        ) as exc_info:
            _get_active_policy_collection()

    assert exc_info.value.__cause__ is cause
    client_mock.assert_called_once_with(active)
    collection_mock.assert_called_once_with(client)


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (None, None),
        ({}, None),
        (
            {"doc_id": "HR-POL-004"},
            {"doc_id": "HR-POL-004"},
        ),
        (
            {"source_format": "md"},
            {"source_format": "md"},
        ),
        (
            {"source_format": "pdf"},
            {"source_format": "pdf"},
        ),
    ],
)
def test_compile_chroma_where_accepts_zero_or_one_filter(
    filters: dict[str, str] | None,
    expected: dict[str, object] | None,
) -> None:
    """Zero or one public filter must map directly to Chroma syntax."""

    result = _compile_chroma_where(
        filters
    )

    assert result == expected


def test_compile_chroma_where_combines_two_filters_with_and() -> None:
    """Both supported filters must use Chroma's explicit $and form."""

    result = _compile_chroma_where(
        {
            "doc_id": "HR-POL-004",
            "source_format": "md",
        }
    )

    assert result == {
        "$and": [
            {
                "doc_id": "HR-POL-004",
            },
            {
                "source_format": "md",
            },
        ],
    }


def test_compile_chroma_where_is_deterministic_across_input_order() -> None:
    """Caller dictionary insertion order must not affect Chroma syntax."""

    forward = _compile_chroma_where(
        {
            "doc_id": "HR-POL-004",
            "source_format": "md",
        }
    )

    reversed_input = _compile_chroma_where(
        {
            "source_format": "md",
            "doc_id": "HR-POL-004",
        }
    )

    assert forward == reversed_input
    assert forward == {
        "$and": [
            {
                "doc_id": "HR-POL-004",
            },
            {
                "source_format": "md",
            },
        ],
    }


@pytest.mark.parametrize(
    "filters",
    [
        {"department": "Engineering"},
        {"doc_id": ""},
        {"source_format": "html"},
    ],
)
def test_compile_chroma_where_rejects_invalid_public_filters(
    filters: dict[str, str],
) -> None:
    """Malformed public filters must fail before Chroma translation."""

    with pytest.raises(
        ValueError,
    ):
        _compile_chroma_where(
            filters
        )


@pytest.mark.parametrize(
    "filters",
    [
        "doc_id=HR-POL-004",
        [],
        {"doc_id": 4},
    ],
)
def test_compile_chroma_where_rejects_wrong_filter_types(
    filters: object,
) -> None:
    """Wrong public filter types must preserve request validation."""

    with pytest.raises(
        TypeError,
    ):
        _compile_chroma_where(
            filters,  # type: ignore[arg-type]
        )


def test_query_policy_collection_raw_executes_unfiltered_query() -> None:
    """Unfiltered retrieval must use the frozen Chroma query contract."""

    embedding = np.zeros(
        384,
        dtype=np.float32,
    )

    collection = Mock()

    raw_response = {
        "ids": [["chunk-a"]],
        "documents": [["Result text."]],
        "metadatas": [[
            {
                "doc_id": "HR-POL-004",
            }
        ]],
        "distances": [[0.25]],
    }

    collection.query.return_value = raw_response

    with (
        patch(
            "rag.retrieve.embed_query",
            return_value=embedding,
        ) as embed_mock,
        patch(
            "rag.retrieve._get_active_policy_collection",
            return_value=collection,
        ) as collection_mock,
    ):
        result = _query_policy_collection_raw(
            "remote work",
            k=3,
        )

    assert result is raw_response

    embed_mock.assert_called_once_with(
        "remote work"
    )
    collection_mock.assert_called_once_with()

    collection.query.assert_called_once_with(
        query_embeddings=[
            embedding,
        ],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


def test_query_policy_collection_raw_executes_filtered_query() -> None:
    """Validated public filters must reach Chroma through the adapter."""

    embedding = np.zeros(
        384,
        dtype=np.float32,
    )

    collection = Mock()

    raw_response = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    collection.query.return_value = raw_response

    with (
        patch(
            "rag.retrieve.embed_query",
            return_value=embedding,
        ),
        patch(
            "rag.retrieve._get_active_policy_collection",
            return_value=collection,
        ),
    ):
        result = _query_policy_collection_raw(
            "remote work",
            k=5,
            filters={
                "doc_id": "HR-POL-004",
                "source_format": "md",
            },
        )

    assert result is raw_response

    collection.query.assert_called_once_with(
        query_embeddings=[
            embedding,
        ],
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
        where={
            "$and": [
                {
                    "doc_id": "HR-POL-004",
                },
                {
                    "source_format": "md",
                },
            ],
        },
    )


@pytest.mark.parametrize(
    ("query", "k", "filters", "expected_error"),
    [
        ("", 5, None, ValueError),
        (123, 5, None, TypeError),
        ("remote work", 0, None, ValueError),
        ("remote work", True, None, TypeError),
        (
            "remote work",
            5,
            {"department": "Engineering"},
            ValueError,
        ),
    ],
)
def test_query_policy_collection_raw_validates_before_runtime_work(
    query: object,
    k: object,
    filters: object,
    expected_error: type[Exception],
) -> None:
    """Invalid retrieval requests must fail before model or index work."""

    with (
        patch(
            "rag.retrieve.embed_query",
        ) as embed_mock,
        patch(
            "rag.retrieve._get_active_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            expected_error,
        ):
            _query_policy_collection_raw(
                query,  # type: ignore[arg-type]
                k=k,  # type: ignore[arg-type]
                filters=filters,  # type: ignore[arg-type]
            )

    embed_mock.assert_not_called()
    collection_mock.assert_not_called()


def test_query_policy_collection_raw_wraps_embedding_failure() -> None:
    """Embedding runtime failures must cross the retrieval boundary."""

    cause = EmbeddingError(
        "model failure"
    )

    with (
        patch(
            "rag.retrieve.embed_query",
            side_effect=cause,
        ),
        patch(
            "rag.retrieve._get_active_policy_collection",
        ) as collection_mock,
    ):
        with pytest.raises(
            RetrievalError,
            match="Failed to embed policy retrieval query",
        ) as exc_info:
            _query_policy_collection_raw(
                "remote work"
            )

    assert exc_info.value.__cause__ is cause
    collection_mock.assert_not_called()


def test_query_policy_collection_raw_propagates_active_index_failure() -> None:
    """Existing retrieval-index failures must remain RetrievalError."""

    embedding = np.zeros(
        384,
        dtype=np.float32,
    )

    cause = RetrievalError(
        "index unavailable"
    )

    with (
        patch(
            "rag.retrieve.embed_query",
            return_value=embedding,
        ),
        patch(
            "rag.retrieve._get_active_policy_collection",
            side_effect=cause,
        ),
    ):
        with pytest.raises(
            RetrievalError,
            match="index unavailable",
        ) as exc_info:
            _query_policy_collection_raw(
                "remote work"
            )

    assert exc_info.value is cause


def test_query_policy_collection_raw_wraps_chroma_query_failure() -> None:
    """Low-level Chroma query failures must become retrieval failures."""

    embedding = np.zeros(
        384,
        dtype=np.float32,
    )

    collection = Mock()

    cause = RuntimeError(
        "query failed"
    )

    collection.query.side_effect = cause

    with (
        patch(
            "rag.retrieve.embed_query",
            return_value=embedding,
        ),
        patch(
            "rag.retrieve._get_active_policy_collection",
            return_value=collection,
        ),
    ):
        with pytest.raises(
            RetrievalError,
            match="Failed to execute policy retrieval query",
        ) as exc_info:
            _query_policy_collection_raw(
                "remote work"
            )

    assert exc_info.value.__cause__ is cause


def test_query_policy_collection_raw_returns_zero_result_response() -> None:
    """A structurally raw zero-match response is valid at transport layer."""

    embedding = np.zeros(
        384,
        dtype=np.float32,
    )

    collection = Mock()

    raw_response = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    collection.query.return_value = raw_response

    with (
        patch(
            "rag.retrieve.embed_query",
            return_value=embedding,
        ),
        patch(
            "rag.retrieve._get_active_policy_collection",
            return_value=collection,
        ),
    ):
        result = _query_policy_collection_raw(
            "remote work"
        )

    assert result is raw_response
