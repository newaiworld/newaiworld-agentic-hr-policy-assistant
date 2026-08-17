"""Focused tests for the S4 retrieval-domain contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rag.embed import EmbeddingError
from rag.ingest import ParsedSection
from rag.store import ChromaStoreError

from rag.retrieve import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_RETRIEVAL_K,
    PolicySection,
    RetrievalError,
    RetrievalResult,
    ValidatedRetrievalRows,
    _build_policy_section_catalogue,
    _build_retrieval_results,
    _compile_chroma_where,
    _convert_parsed_section,
    _extract_section_number,
    _get_active_policy_collection,
    _get_cached_policy_section_catalogue,
    _query_policy_collection_raw,
    _validate_raw_retrieval_response,
    _validate_retrieval_filters,
    _validate_retrieval_k,
    _validate_policy_section_lookup,
    _validate_retrieval_query,
    get_policy_section_catalogue,
    resolve_corpus_dir,
    retrieve_policy,
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


def make_raw_retrieval_response() -> dict[str, object]:
    """Return one fresh structurally valid raw Chroma query response."""

    document = (
        "International remote work requires written approval."
    )

    return {
        "ids": [[
            "HR-POL-004__0000__abcdef0123456789",
        ]],
        "documents": [[
            document,
        ]],
        "metadatas": [[
            {
                "doc_id": "HR-POL-004",
                "title": "Remote and Flexible Work Policy",
                "section_path": [
                    "Remote and Flexible Work Policy",
                    "5. Procedures or Application",
                    "5.3 International approval",
                ],
                "snippet": document,
                "source_format": "md",
            }
        ]],
        "distances": [[
            0.25,
        ]],
    }


def test_validate_raw_retrieval_response_returns_validated_rows() -> None:
    """One valid Chroma response must unwrap into aligned tuples."""

    response = make_raw_retrieval_response()

    rows = _validate_raw_retrieval_response(
        response
    )

    assert isinstance(
        rows,
        ValidatedRetrievalRows,
    )

    assert rows.ids == (
        "HR-POL-004__0000__abcdef0123456789",
    )

    assert rows.documents == (
        "International remote work requires written approval.",
    )

    assert rows.distances == (
        0.25,
    )

    assert len(
        rows.metadatas
    ) == 1


def test_validate_raw_retrieval_response_accepts_extra_top_level_keys() -> None:
    """Chroma bookkeeping fields outside retrieval data are allowed."""

    response = make_raw_retrieval_response()

    response["embeddings"] = None
    response["included"] = [
        "documents",
        "metadatas",
        "distances",
    ]
    response["uris"] = None
    response["data"] = None

    rows = _validate_raw_retrieval_response(
        response
    )

    assert len(
        rows.ids
    ) == 1


def test_validate_raw_retrieval_response_accepts_extra_metadata_keys() -> None:
    """Additional metadata must not break the required retrieval contract."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    assert isinstance(
        metadatas,
        list,
    )

    metadata = metadatas[0][0]
    metadata["future_field"] = "allowed"

    rows = _validate_raw_retrieval_response(
        response
    )

    assert (
        rows.metadatas[0]["future_field"]
        == "allowed"
    )


def test_validate_raw_retrieval_response_accepts_zero_rows() -> None:
    """A valid zero-match Chroma response must remain successful."""

    rows = _validate_raw_retrieval_response(
        {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
    )

    assert rows.ids == ()
    assert rows.documents == ()
    assert rows.metadatas == ()
    assert rows.distances == ()


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        (),
        "invalid",
        123,
    ],
)
def test_validate_raw_retrieval_response_rejects_non_dictionary(
    response: object,
) -> None:
    """The pinned Chroma response contract must be dictionary-shaped."""

    with pytest.raises(
        RetrievalError,
        match="must be a dictionary",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "ids",
        "documents",
        "metadatas",
        "distances",
    ],
)
def test_validate_raw_retrieval_response_rejects_missing_required_field(
    field_name: str,
) -> None:
    """Every retrieval-owned Chroma field must be present."""

    response = make_raw_retrieval_response()
    del response[field_name]

    with pytest.raises(
        RetrievalError,
        match="invalid.*structure",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        None,
        "invalid",
        {},
        [],
        [1],
        [[], []],
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_outer_structure(
    bad_value: object,
) -> None:
    """Required response fields must use one nested query-result list."""

    response = make_raw_retrieval_response()
    response["ids"] = bad_value

    with pytest.raises(
        RetrievalError,
        match="invalid 'ids' structure",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "documents",
        "metadatas",
        "distances",
    ],
)
def test_validate_raw_retrieval_response_rejects_misalignment(
    field_name: str,
) -> None:
    """All raw result fields must align one-to-one with returned IDs."""

    response = make_raw_retrieval_response()
    response[field_name] = [[]]

    with pytest.raises(
        RetrievalError,
        match=f"misaligned {field_name}",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "chunk_id",
    [
        None,
        1,
        True,
        "",
        "   ",
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_chunk_id(
    chunk_id: object,
) -> None:
    """Returned Chroma IDs must be non-empty strings."""

    response = make_raw_retrieval_response()
    response["ids"] = [[chunk_id]]

    with pytest.raises(
        RetrievalError,
        match="invalid chunk ID",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "document",
    [
        None,
        1,
        True,
        "",
        "   ",
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_document(
    document: object,
) -> None:
    """Returned Chroma documents must be non-empty strings."""

    response = make_raw_retrieval_response()
    response["documents"] = [[document]]

    with pytest.raises(
        RetrievalError,
        match="invalid document",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "invalid",
        [],
        1,
    ],
)
def test_validate_raw_retrieval_response_rejects_non_dictionary_metadata(
    metadata: object,
) -> None:
    """Each Chroma result must contain dictionary metadata."""

    response = make_raw_retrieval_response()
    response["metadatas"] = [[metadata]]

    with pytest.raises(
        RetrievalError,
        match="invalid metadata",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "doc_id",
        "title",
        "section_path",
        "snippet",
        "source_format",
    ],
)
def test_validate_raw_retrieval_response_rejects_missing_metadata_field(
    field_name: str,
) -> None:
    """Citation-critical metadata fields must always be present."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    assert isinstance(
        metadatas,
        list,
    )

    metadata = metadatas[0][0]
    del metadata[field_name]

    with pytest.raises(
        RetrievalError,
        match="missing required field",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("doc_id", None),
        ("doc_id", ""),
        ("title", 1),
        ("title", "   "),
        ("snippet", None),
        ("snippet", ""),
        ("source_format", 1),
        ("source_format", ""),
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_metadata_string(
    field_name: str,
    bad_value: object,
) -> None:
    """Required scalar metadata values must be non-empty strings."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    assert isinstance(
        metadatas,
        list,
    )

    metadata = metadatas[0][0]
    metadata[field_name] = bad_value

    with pytest.raises(
        RetrievalError,
        match=f"invalid '{field_name}'",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "section_path",
    [
        None,
        "Policy",
        (),
        [],
        [""],
        ["   "],
        ["Policy", ""],
        ["Policy", 1],
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_section_path(
    section_path: object,
) -> None:
    """Section provenance must be a non-empty list of non-empty strings."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    assert isinstance(
        metadatas,
        list,
    )

    metadata = metadatas[0][0]
    metadata["section_path"] = section_path

    with pytest.raises(
        RetrievalError,
        match="invalid 'section_path'",
    ):
        _validate_raw_retrieval_response(
            response
        )


def test_validate_raw_retrieval_response_rejects_title_path_mismatch() -> None:
    """Policy title must equal the root of its persisted heading path."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    metadata = metadatas[0][0]

    metadata["title"] = "Different Policy"

    with pytest.raises(
        RetrievalError,
        match="title does not match section_path root",
    ):
        _validate_raw_retrieval_response(
            response
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
def test_validate_raw_retrieval_response_rejects_unsupported_source_format(
    source_format: str,
) -> None:
    """Persisted source format must remain within the corpus contract."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    metadata = metadatas[0][0]

    metadata["source_format"] = source_format

    with pytest.raises(
        RetrievalError,
        match="unsupported source_format",
    ):
        _validate_raw_retrieval_response(
            response
        )


def test_validate_raw_retrieval_response_rejects_snippet_document_mismatch() -> None:
    """Citation snippet must remain identical to the indexed document."""

    response = make_raw_retrieval_response()

    metadatas = response["metadatas"]
    metadata = metadatas[0][0]

    metadata["snippet"] = "Different text."

    with pytest.raises(
        RetrievalError,
        match="document and citation snippet do not match",
    ):
        _validate_raw_retrieval_response(
            response
        )


@pytest.mark.parametrize(
    "distance",
    [
        None,
        1,
        True,
        "0.25",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_raw_retrieval_response_rejects_invalid_distance(
    distance: object,
) -> None:
    """Distances must be finite Python floats from the pinned runtime."""

    response = make_raw_retrieval_response()
    response["distances"] = [[distance]]

    with pytest.raises(
        RetrievalError,
        match="invalid distance",
    ):
        _validate_raw_retrieval_response(
            response
        )


def test_validate_raw_retrieval_response_rejects_duplicate_chunk_ids() -> None:
    """One Chroma query must not return the same canonical chunk twice."""

    document = (
        "International remote work requires written approval."
    )

    metadata = {
        "doc_id": "HR-POL-004",
        "title": "Remote and Flexible Work Policy",
        "section_path": [
            "Remote and Flexible Work Policy",
            "5. Procedures or Application",
            "5.3 International approval",
        ],
        "snippet": document,
        "source_format": "md",
    }

    response = {
        "ids": [[
            "HR-POL-004__0000__abcdef0123456789",
            "HR-POL-004__0000__abcdef0123456789",
        ]],
        "documents": [[
            document,
            document,
        ]],
        "metadatas": [[
            dict(metadata),
            dict(metadata),
        ]],
        "distances": [[
            0.25,
            0.30,
        ]],
    }

    with pytest.raises(
        RetrievalError,
        match="duplicate chunk IDs",
    ):
        _validate_raw_retrieval_response(
            response
        )


def make_validated_retrieval_rows(
    *,
    ids: tuple[str, ...] = (
        "HR-POL-004__0000__abcdef0123456789",
    ),
    distances: tuple[float, ...] = (
        0.25,
    ),
) -> ValidatedRetrievalRows:
    """Return one fresh validated-row fixture for R4 conversion tests."""

    document = (
        "International remote work requires written approval."
    )

    metadata = {
        "doc_id": "HR-POL-004",
        "title": "Remote and Flexible Work Policy",
        "section_path": [
            "Remote and Flexible Work Policy",
            "5. Procedures or Application",
            "5.3 International approval",
        ],
        "snippet": document,
        "source_format": "md",
    }

    return ValidatedRetrievalRows(
        ids=ids,
        documents=tuple(
            document
            for _ in ids
        ),
        metadatas=tuple(
            dict(metadata)
            for _ in ids
        ),
        distances=distances,
    )


def test_build_retrieval_results_returns_citation_ready_tuple() -> None:
    """Validated rows must map into the frozen RetrievalResult contract."""

    rows = make_validated_retrieval_rows()

    results = _build_retrieval_results(
        rows
    )

    assert isinstance(
        results,
        tuple,
    )

    assert len(results) == 1

    result = results[0]

    assert isinstance(
        result,
        RetrievalResult,
    )

    assert (
        result.chunk_id
        == "HR-POL-004__0000__abcdef0123456789"
    )
    assert result.doc_id == "HR-POL-004"
    assert (
        result.title
        == "Remote and Flexible Work Policy"
    )
    assert (
        result.section
        == "5.3 International approval"
    )
    assert result.section_path == (
        "Remote and Flexible Work Policy",
        "5. Procedures or Application",
        "5.3 International approval",
    )
    assert (
        result.snippet
        == "International remote work requires written approval."
    )
    assert result.source_format == "md"
    assert result.distance == 0.25
    assert result.similarity == 0.75


def test_build_retrieval_results_converts_section_path_to_tuple() -> None:
    """Persisted Chroma section-path lists must become immutable paths."""

    rows = make_validated_retrieval_rows()

    raw_path = rows.metadatas[0][
        "section_path"
    ]

    assert isinstance(
        raw_path,
        list,
    )

    result = _build_retrieval_results(
        rows
    )[0]

    assert isinstance(
        result.section_path,
        tuple,
    )

    assert result.section == result.section_path[-1]


def test_build_retrieval_results_preserves_input_ranking_order() -> None:
    """R4 must not rerank or sort already-ranked Chroma rows."""

    rows = ValidatedRetrievalRows(
        ids=(
            "rank-1",
            "rank-2",
            "rank-3",
        ),
        documents=(
            "First.",
            "Second.",
            "Third.",
        ),
        metadatas=(
            {
                "doc_id": "HR-POL-001",
                "title": "Policy",
                "section_path": [
                    "Policy",
                    "First",
                ],
                "snippet": "First.",
                "source_format": "md",
            },
            {
                "doc_id": "HR-POL-001",
                "title": "Policy",
                "section_path": [
                    "Policy",
                    "Second",
                ],
                "snippet": "Second.",
                "source_format": "md",
            },
            {
                "doc_id": "HR-POL-001",
                "title": "Policy",
                "section_path": [
                    "Policy",
                    "Third",
                ],
                "snippet": "Third.",
                "source_format": "md",
            },
        ),
        distances=(
            0.30,
            0.10,
            0.20,
        ),
    )

    results = _build_retrieval_results(
        rows
    )

    assert tuple(
        result.chunk_id
        for result in results
    ) == (
        "rank-1",
        "rank-2",
        "rank-3",
    )

    assert tuple(
        result.distance
        for result in results
    ) == (
        0.30,
        0.10,
        0.20,
    )


@pytest.mark.parametrize(
    ("distance", "expected_similarity"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (1.0, 0.0),
        (2.0, -1.0),
    ],
)
def test_build_retrieval_results_uses_frozen_cosine_score_mapping(
    distance: float,
    expected_similarity: float,
) -> None:
    """Conversion must use similarity=1-distance without clamping."""

    rows = make_validated_retrieval_rows(
        distances=(
            distance,
        ),
    )

    result = _build_retrieval_results(
        rows
    )[0]

    assert result.distance == distance
    assert result.similarity == expected_similarity


def test_build_retrieval_results_accepts_zero_rows() -> None:
    """A valid zero-match retrieval must convert to an empty tuple."""

    rows = ValidatedRetrievalRows(
        ids=(),
        documents=(),
        metadatas=(),
        distances=(),
    )

    assert (
        _build_retrieval_results(
            rows
        )
        == ()
    )


@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [],
        (),
        "invalid",
    ],
)
def test_build_retrieval_results_rejects_wrong_input_type(
    rows: object,
) -> None:
    """The converter accepts only the R3D validated-row contract."""

    with pytest.raises(
        TypeError,
        match="ValidatedRetrievalRows",
    ):
        _build_retrieval_results(
            rows,  # type: ignore[arg-type]
        )


def test_build_retrieval_results_wraps_missing_metadata_field() -> None:
    """Impossible post-R3D metadata failures must become RetrievalError."""

    rows = make_validated_retrieval_rows()

    del rows.metadatas[0][
        "doc_id"
    ]

    with pytest.raises(
        RetrievalError,
        match="Failed to convert validated retrieval row 0",
    ) as exc_info:
        _build_retrieval_results(
            rows
        )

    assert isinstance(
        exc_info.value.__cause__,
        KeyError,
    )


def test_build_retrieval_results_wraps_invalid_section_path() -> None:
    """Impossible empty section paths must cross the retrieval boundary."""

    rows = make_validated_retrieval_rows()

    rows.metadatas[0][
        "section_path"
    ] = []

    with pytest.raises(
        RetrievalError,
        match="Failed to convert validated retrieval row 0",
    ) as exc_info:
        _build_retrieval_results(
            rows
        )

    assert isinstance(
        exc_info.value.__cause__,
        IndexError,
    )


def test_build_retrieval_results_wraps_non_list_section_path() -> None:
    """R4 must reject a violated validated-row section-path contract."""

    rows = make_validated_retrieval_rows()

    rows.metadatas[0][
        "section_path"
    ] = (
        "Remote and Flexible Work Policy",
        "5.3 International approval",
    )

    with pytest.raises(
        RetrievalError,
        match="Failed to convert validated retrieval row 0",
    ) as exc_info:
        _build_retrieval_results(
            rows
        )

    assert isinstance(
        exc_info.value.__cause__,
        TypeError,
    )


def test_build_retrieval_results_wraps_result_domain_failure() -> None:
    """RetrievalResult invariant failures must not leak as ValueError."""

    rows = make_validated_retrieval_rows()

    rows.metadatas[0][
        "title"
    ] = "Different Policy"

    with pytest.raises(
        RetrievalError,
        match="Failed to convert validated retrieval row 0",
    ) as exc_info:
        _build_retrieval_results(
            rows
        )

    assert isinstance(
        exc_info.value.__cause__,
        ValueError,
    )


def test_build_retrieval_results_wraps_misaligned_validated_rows() -> None:
    """Manual violation of R3D alignment must become RetrievalError."""

    rows = ValidatedRetrievalRows(
        ids=(
            "chunk-a",
            "chunk-b",
        ),
        documents=(
            "A.",
            "B.",
        ),
        metadatas=(
            {
                "doc_id": "HR-POL-001",
                "title": "Policy",
                "section_path": [
                    "Policy",
                    "A",
                ],
                "snippet": "A.",
                "source_format": "md",
            },
        ),
        distances=(
            0.1,
            0.2,
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="validated retrieval row 1",
    ) as exc_info:
        _build_retrieval_results(
            rows
        )

    assert isinstance(
        exc_info.value.__cause__,
        IndexError,
    )


def test_retrieve_policy_composes_retrieval_pipeline() -> None:
    """Public retrieval must compose the three verified lower layers."""

    raw_response = {
        "raw": "response",
    }

    rows = ValidatedRetrievalRows(
        ids=(),
        documents=(),
        metadatas=(),
        distances=(),
    )

    results = (
        RetrievalResult(
            chunk_id="chunk-a",
            doc_id="HR-POL-004",
            title="Remote and Flexible Work Policy",
            section="5.3 International approval",
            section_path=(
                "Remote and Flexible Work Policy",
                "5. Procedures or Application",
                "5.3 International approval",
            ),
            snippet=(
                "International remote work requires written approval."
            ),
            source_format="md",
            distance=0.25,
            similarity=0.75,
        ),
    )

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            return_value=raw_response,
        ) as query_mock,
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
            return_value=rows,
        ) as validate_mock,
        patch(
            "rag.retrieve._build_retrieval_results",
            return_value=results,
        ) as build_mock,
    ):
        actual = retrieve_policy(
            "Can I work remotely overseas?",
            k=3,
            filters={
                "doc_id": "HR-POL-004",
            },
        )

    assert actual is results

    query_mock.assert_called_once_with(
        "Can I work remotely overseas?",
        k=3,
        filters={
            "doc_id": "HR-POL-004",
        },
    )

    validate_mock.assert_called_once_with(
        raw_response
    )

    build_mock.assert_called_once_with(
        rows
    )


def test_retrieve_policy_uses_default_k_and_filters() -> None:
    """The public API must preserve the frozen defaults."""

    raw_response = {
        "raw": "response",
    }

    rows = ValidatedRetrievalRows(
        ids=(),
        documents=(),
        metadatas=(),
        distances=(),
    )

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            return_value=raw_response,
        ) as query_mock,
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
            return_value=rows,
        ),
        patch(
            "rag.retrieve._build_retrieval_results",
            return_value=(),
        ),
    ):
        result = retrieve_policy(
            "remote work"
        )

    assert result == ()

    query_mock.assert_called_once_with(
        "remote work",
        k=DEFAULT_RETRIEVAL_K,
        filters=None,
    )


def test_retrieve_policy_preserves_zero_result_tuple() -> None:
    """A valid no-match retrieval must remain an empty result tuple."""

    raw_response = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    rows = ValidatedRetrievalRows(
        ids=(),
        documents=(),
        metadatas=(),
        distances=(),
    )

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            return_value=raw_response,
        ),
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
            return_value=rows,
        ),
        patch(
            "rag.retrieve._build_retrieval_results",
            return_value=(),
        ),
    ):
        result = retrieve_policy(
            "no-match query"
        )

    assert result == ()


@pytest.mark.parametrize(
    "error",
    [
        TypeError(
            "query must be a string."
        ),
        ValueError(
            "k must be positive."
        ),
        RetrievalError(
            "index unavailable"
        ),
    ],
)
def test_retrieve_policy_propagates_query_stage_errors(
    error: Exception,
) -> None:
    """Public retrieval must not re-wrap existing query-stage errors."""

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            side_effect=error,
        ),
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
        ) as validate_mock,
        patch(
            "rag.retrieve._build_retrieval_results",
        ) as build_mock,
    ):
        with pytest.raises(
            type(error),
        ) as exc_info:
            retrieve_policy(
                "remote work"
            )

    assert exc_info.value is error
    validate_mock.assert_not_called()
    build_mock.assert_not_called()


def test_retrieve_policy_propagates_response_validation_failure() -> None:
    """Malformed Chroma output must remain the R3D RetrievalError."""

    raw_response = {
        "invalid": "response",
    }

    error = RetrievalError(
        "malformed response"
    )

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            return_value=raw_response,
        ),
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
            side_effect=error,
        ),
        patch(
            "rag.retrieve._build_retrieval_results",
        ) as build_mock,
    ):
        with pytest.raises(
            RetrievalError,
        ) as exc_info:
            retrieve_policy(
                "remote work"
            )

    assert exc_info.value is error
    build_mock.assert_not_called()


def test_retrieve_policy_propagates_result_conversion_failure() -> None:
    """R4 conversion failures must pass through the public boundary."""

    raw_response = {
        "raw": "response",
    }

    rows = ValidatedRetrievalRows(
        ids=(),
        documents=(),
        metadatas=(),
        distances=(),
    )

    error = RetrievalError(
        "conversion failed"
    )

    with (
        patch(
            "rag.retrieve._query_policy_collection_raw",
            return_value=raw_response,
        ),
        patch(
            "rag.retrieve._validate_raw_retrieval_response",
            return_value=rows,
        ),
        patch(
            "rag.retrieve._build_retrieval_results",
            side_effect=error,
        ),
    ):
        with pytest.raises(
            RetrievalError,
        ) as exc_info:
            retrieve_policy(
                "remote work"
            )

    assert exc_info.value is error


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("5.3 International approval", "5.3"),
        ("10.1 Is this policy authoritative?", "10.1"),
        ("6. Exceptions and Escalation", "6"),
        ("8. Decision Rules", "8"),
        ("11. Related Documents", "11"),
        ("Employee Handbook", None),
        ("5.3foo", None),
        ("10.1FAQ", None),
        ("5.", None),
        ("5. ", None),
        ("5", "5"),
        ("5.3", "5.3"),
    ],
)
def test_extract_section_number_uses_exact_heading_grammar(
    section: str,
    expected: str | None,
) -> None:
    """Section-number extraction must follow the frozen heading grammar."""

    assert (
        _extract_section_number(
            section
        )
        == expected
    )


@pytest.mark.parametrize(
    "section",
    [
        None,
        5,
        True,
        [],
        {},
    ],
)
def test_extract_section_number_rejects_non_string(
    section: object,
) -> None:
    """Section-number extraction accepts only strings."""

    with pytest.raises(
        TypeError,
        match="section must be a string",
    ):
        _extract_section_number(
            section,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "section",
    [
        "",
        "   ",
    ],
)
def test_extract_section_number_rejects_blank_string(
    section: str,
) -> None:
    """Blank section values must fail deterministically."""

    with pytest.raises(
        ValueError,
        match="section must be a non-empty string",
    ):
        _extract_section_number(
            section
        )


def make_policy_section(
    *,
    doc_id: str = "HR-POL-004",
    title: str = "Remote and Flexible Work Policy",
    section: str = "5.3 International approval",
    section_path: tuple[str, ...] = (
        "Remote and Flexible Work Policy",
        "5. Procedures or Application",
        "5.3 International approval",
    ),
    section_number: str | None = "5.3",
    text: str = (
        "International remote work requires written approval "
        "before travel is booked."
    ),
    source_format: str = "md",
    section_order: int = 15,
) -> PolicySection:
    """Return one fresh valid exact-section domain object."""

    return PolicySection(
        doc_id=doc_id,
        title=title,
        section=section,
        section_path=section_path,
        section_number=section_number,
        text=text,
        source_format=source_format,
        section_order=section_order,
    )


def test_policy_section_accepts_numbered_section() -> None:
    """A normal numbered policy section must satisfy the domain contract."""

    result = make_policy_section()

    assert result.doc_id == "HR-POL-004"
    assert (
        result.title
        == "Remote and Flexible Work Policy"
    )
    assert (
        result.section
        == "5.3 International approval"
    )
    assert result.section_number == "5.3"
    assert result.section == result.section_path[-1]
    assert result.source_format == "md"
    assert result.section_order == 15


def test_policy_section_accepts_unnumbered_root_section() -> None:
    """Document-root sections may legitimately have no section number."""

    result = make_policy_section(
        doc_id="HR-POL-001",
        title="Employee Handbook",
        section="Employee Handbook",
        section_path=(
            "Employee Handbook",
        ),
        section_number=None,
        text="Synthetic HR policy document.",
        section_order=0,
    )

    assert result.section_number is None
    assert result.section_path == (
        "Employee Handbook",
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("doc_id", ""),
        ("doc_id", "   "),
        ("title", ""),
        ("section", ""),
        ("text", ""),
        ("source_format", ""),
    ],
)
def test_policy_section_rejects_blank_string_fields(
    field_name: str,
    bad_value: str,
) -> None:
    """Required scalar section fields must be non-empty strings."""

    kwargs = {
        "doc_id": "HR-POL-004",
        "title": "Remote and Flexible Work Policy",
        "section": "5.3 International approval",
        "section_path": (
            "Remote and Flexible Work Policy",
            "5. Procedures or Application",
            "5.3 International approval",
        ),
        "section_number": "5.3",
        "text": "Policy text.",
        "source_format": "md",
        "section_order": 15,
    }

    kwargs[field_name] = bad_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a non-empty string",
    ):
        PolicySection(
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "section_path",
    [
        (),
        ("",),
        ("Policy", ""),
        ("Policy", 1),
        ["Policy"],  # type: ignore[list-item]
    ],
)
def test_policy_section_rejects_invalid_section_path(
    section_path: object,
) -> None:
    """Exact-section provenance must use a non-empty tuple of strings."""

    with pytest.raises(
        ValueError,
        match="section_path must be a non-empty tuple",
    ):
        make_policy_section(
            section_path=section_path,  # type: ignore[arg-type]
        )


def test_policy_section_rejects_title_path_mismatch() -> None:
    """The policy title must equal the root of the heading hierarchy."""

    with pytest.raises(
        ValueError,
        match="title must match the first section_path element",
    ):
        make_policy_section(
            title="Different Policy"
        )


def test_policy_section_rejects_leaf_section_mismatch() -> None:
    """The public section value must equal the final path element."""

    with pytest.raises(
        ValueError,
        match="section must match the final section_path element",
    ):
        make_policy_section(
            section="5.2 Different section"
        )


@pytest.mark.parametrize(
    "source_format",
    [
        "html",
        "txt",
        "MD",
        " pdf ",
    ],
)
def test_policy_section_rejects_unsupported_source_format(
    source_format: str,
) -> None:
    """Exact-section provenance must use the canonical corpus formats."""

    with pytest.raises(
        ValueError,
        match="source_format must be a supported corpus format",
    ):
        make_policy_section(
            source_format=source_format
        )


@pytest.mark.parametrize(
    "section_order",
    [
        True,
        -1,
    ],
)
def test_policy_section_rejects_invalid_section_order(
    section_order: object,
) -> None:
    """Section order must be a non-negative integer, excluding Boolean."""

    with pytest.raises(
        ValueError,
        match="section_order",
    ):
        make_policy_section(
            section_order=section_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "section_number",
    [
        "",
        "   ",
        5,
    ],
)
def test_policy_section_rejects_invalid_section_number_type_or_value(
    section_number: object,
) -> None:
    """Section number must be a non-empty string or None."""

    with pytest.raises(
        ValueError,
        match="section_number must be a non-empty string or None",
    ):
        make_policy_section(
            section_number=section_number,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("section", "section_number"),
    [
        ("5.3 International approval", "5.2"),
        ("6. Exceptions and Escalation", "6.1"),
        ("Employee Handbook", "1"),
    ],
)
def test_policy_section_rejects_section_number_mismatch(
    section: str,
    section_number: str,
) -> None:
    """Stored section number must exactly match the heading prefix."""

    if section == "Employee Handbook":
        section_path = (
            "Employee Handbook",
        )
        title = "Employee Handbook"
    else:
        section_path = (
            "Remote and Flexible Work Policy",
            section,
        )
        title = "Remote and Flexible Work Policy"

    with pytest.raises(
        ValueError,
        match="section_number must match the numeric prefix",
    ):
        make_policy_section(
            title=title,
            section=section,
            section_path=section_path,
            section_number=section_number,
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("doc_id", None),
        ("doc_id", 1),
        ("title", None),
        ("title", 1),
        ("section", None),
        ("section", 1),
        ("text", None),
        ("text", 1),
        ("source_format", None),
        ("source_format", 1),
    ],
)
def test_policy_section_rejects_non_string_required_fields(
    field_name: str,
    bad_value: object,
) -> None:
    """Required scalar section fields must be strings."""

    kwargs = {
        "doc_id": "HR-POL-004",
        "title": "Remote and Flexible Work Policy",
        "section": "5.3 International approval",
        "section_path": (
            "Remote and Flexible Work Policy",
            "5. Procedures or Application",
            "5.3 International approval",
        ),
        "section_number": "5.3",
        "text": "Policy text.",
        "source_format": "md",
        "section_order": 15,
    }

    kwargs[field_name] = bad_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a non-empty string",
    ):
        PolicySection(
            **kwargs,  # type: ignore[arg-type]
        )


def test_policy_section_is_frozen() -> None:
    """Exact policy-section domain objects must remain immutable."""

    result = make_policy_section()

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.section = "Different section"  # type: ignore[misc]


def test_resolve_corpus_dir_uses_default_when_unconfigured() -> None:
    """Unconfigured corpus resolution must use the project default."""

    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        result = resolve_corpus_dir()

    assert result == (
        DEFAULT_CORPUS_DIR
        .expanduser()
        .resolve()
    )
    assert result.is_absolute()


def test_resolve_corpus_dir_resolves_relative_override() -> None:
    """Configured relative corpus paths must resolve absolutely."""

    with patch.dict(
        "os.environ",
        {
            "CORPUS_DIR": "custom_corpus",
        },
        clear=True,
    ):
        result = resolve_corpus_dir()

    assert result == Path(
        "custom_corpus"
    ).resolve()
    assert result.is_absolute()


def test_resolve_corpus_dir_trims_configured_value() -> None:
    """Configured corpus paths must ignore surrounding whitespace."""

    with patch.dict(
        "os.environ",
        {
            "CORPUS_DIR": "  custom_corpus  ",
        },
        clear=True,
    ):
        result = resolve_corpus_dir()

    assert result == Path(
        "custom_corpus"
    ).resolve()


def test_resolve_corpus_dir_expands_user_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured home markers must be expanded before resolution."""

    fake_home = Path("/tmp/r6-home")

    monkeypatch.setenv(
        "HOME",
        str(fake_home),
    )

    with patch.dict(
        "os.environ",
        {
            "CORPUS_DIR": "~/policy-corpus",
        },
        clear=False,
    ):
        result = resolve_corpus_dir()

    assert result == (
        fake_home
        / "policy-corpus"
    ).resolve()


def test_resolve_corpus_dir_rejects_blank_override() -> None:
    """Explicit blank corpus configuration must fail deterministically."""

    with patch.dict(
        "os.environ",
        {
            "CORPUS_DIR": "   ",
        },
        clear=True,
    ):
        with pytest.raises(
            RetrievalError,
            match="CORPUS_DIR must not be blank when configured",
        ):
            resolve_corpus_dir()


def test_resolve_corpus_dir_does_not_require_existing_path(
    tmp_path: Path,
) -> None:
    """Resolution must not perform catalogue existence validation."""

    missing = (
        tmp_path
        / "does_not_exist"
    )

    with patch.dict(
        "os.environ",
        {
            "CORPUS_DIR": str(missing),
        },
        clear=True,
    ):
        result = resolve_corpus_dir()

    assert result == missing.resolve()
    assert not result.exists()


def make_parsed_policy_section(
    *,
    doc_id: str = "HR-POL-004",
    title: str = "Remote and Flexible Work Policy",
    section_path: tuple[str, ...] = (
        "Remote and Flexible Work Policy",
        "5. Procedures or Application",
        "5.3 International approval",
    ),
    section_order: int = 15,
    text: str = (
        "International remote work requires written approval."
    ),
    source_format: str = "md",
) -> ParsedSection:
    """Return one valid ParsedSection for exact-section conversion tests."""

    return ParsedSection(
        doc_id=doc_id,
        title=title,
        section_path=section_path,
        section_order=section_order,
        text=text,
        source_format=source_format,
    )


def test_convert_parsed_section_builds_numbered_policy_section() -> None:
    """A numbered parsed section must map completely into lookup form."""

    parsed = make_parsed_policy_section()

    result = _convert_parsed_section(
        parsed
    )

    assert isinstance(
        result,
        PolicySection,
    )
    assert result.doc_id == parsed.doc_id
    assert result.title == parsed.title
    assert result.section == parsed.section_path[-1]
    assert result.section_path == parsed.section_path
    assert result.section_number == "5.3"
    assert result.text == parsed.text
    assert result.source_format == parsed.source_format
    assert result.section_order == parsed.section_order


def test_convert_parsed_section_builds_unnumbered_root_section() -> None:
    """A document-root parsed section must retain no section number."""

    parsed = make_parsed_policy_section(
        doc_id="HR-POL-001",
        title="Employee Handbook",
        section_path=(
            "Employee Handbook",
        ),
        section_order=0,
        text="Synthetic HR policy document.",
    )

    result = _convert_parsed_section(
        parsed
    )

    assert result.section == "Employee Handbook"
    assert result.section_path == (
        "Employee Handbook",
    )
    assert result.section_number is None


def test_convert_parsed_section_extracts_major_section_number() -> None:
    """Major headings such as ``6. ...`` must map to number ``6``."""

    parsed = make_parsed_policy_section(
        section_path=(
            "Remote and Flexible Work Policy",
            "6. Exceptions and Escalation",
        ),
        section_order=20,
    )

    result = _convert_parsed_section(
        parsed
    )

    assert result.section == "6. Exceptions and Escalation"
    assert result.section_number == "6"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_convert_parsed_section_rejects_empty_text(
    text: str,
) -> None:
    """Empty structural parser sections must not become lookup records."""

    parsed = make_parsed_policy_section(
        text=text
    )

    with pytest.raises(
        ValueError,
        match="section text must be non-empty for exact policy lookup",
    ):
        _convert_parsed_section(
            parsed
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "section",
        1,
        True,
        (),
        {},
    ],
)
def test_convert_parsed_section_rejects_wrong_input_type(
    value: object,
) -> None:
    """Conversion accepts only the ingestion ParsedSection domain."""

    with pytest.raises(
        TypeError,
        match="section must be a ParsedSection instance",
    ):
        _convert_parsed_section(
            value,  # type: ignore[arg-type]
        )


def test_convert_parsed_section_preserves_complete_text() -> None:
    """Conversion must not truncate complete normalized section text."""

    text = (
        "First policy paragraph.\n\n"
        "Second policy paragraph containing the complete section."
    )

    parsed = make_parsed_policy_section(
        text=text
    )

    result = _convert_parsed_section(
        parsed
    )

    assert result.text == text


def test_build_policy_section_catalogue_builds_real_corpus() -> None:
    """The real corpus must materialize the frozen exact-section counts."""

    corpus_dir = (
        Path("corpus")
        .resolve()
    )

    catalogue = _build_policy_section_catalogue(
        corpus_dir
    )

    assert len(catalogue) == 400

    assert len(
        {
            section.doc_id
            for section in catalogue
        }
    ) == 13

    assert sum(
        section.source_format == "md"
        for section in catalogue
    ) == 277

    assert sum(
        section.source_format == "pdf"
        for section in catalogue
    ) == 123

    assert sum(
        section.section_number is not None
        for section in catalogue
    ) == 391

    assert sum(
        section.section_number is None
        for section in catalogue
    ) == 9


def test_build_policy_section_catalogue_returns_policy_sections() -> None:
    """Every catalogue record must satisfy the PolicySection contract."""

    catalogue = _build_policy_section_catalogue(
        Path("corpus").resolve()
    )

    assert catalogue
    assert all(
        isinstance(
            section,
            PolicySection,
        )
        for section in catalogue
    )


def test_build_policy_section_catalogue_is_deterministically_ordered() -> None:
    """Repeated uncached builds must preserve identical catalogue order."""

    corpus_dir = Path(
        "corpus"
    ).resolve()

    first = _build_policy_section_catalogue(
        corpus_dir
    )

    second = _build_policy_section_catalogue(
        corpus_dir
    )

    first_keys = tuple(
        (
            section.doc_id,
            section.section_order,
            section.section_path,
        )
        for section in first
    )

    second_keys = tuple(
        (
            section.doc_id,
            section.section_order,
            section.section_path,
        )
        for section in second
    )

    assert first_keys == second_keys


def test_build_policy_section_catalogue_preserves_complete_text() -> None:
    """Catalogue records must retain complete normalized section text."""

    catalogue = _build_policy_section_catalogue(
        Path("corpus").resolve()
    )

    assert all(
        section.text.strip()
        for section in catalogue
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "corpus",
        1,
        True,
        (),
        {},
    ],
)
def test_build_policy_section_catalogue_rejects_wrong_path_type(
    value: object,
) -> None:
    """Catalogue construction accepts only pathlib.Path corpus roots."""

    with pytest.raises(
        TypeError,
        match="corpus_dir must be a pathlib.Path instance",
    ):
        _build_policy_section_catalogue(
            value,  # type: ignore[arg-type]
        )


def test_build_policy_section_catalogue_rejects_relative_path() -> None:
    """Catalogue construction requires an explicit absolute corpus root."""

    with pytest.raises(
        ValueError,
        match="corpus_dir must be absolute",
    ):
        _build_policy_section_catalogue(
            Path("corpus")
        )


def test_cached_policy_section_catalogue_reuses_same_identity() -> None:
    """Identical path/version keys must reuse one catalogue instance."""

    corpus_dir = Path("/tmp/r6-cache-corpus")
    catalogue = ("catalogue",)

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with patch(
            "rag.retrieve._build_policy_section_catalogue",
            return_value=catalogue,
        ) as build_mock:
            first = _get_cached_policy_section_catalogue(
                corpus_dir,
                "1.0",
            )

            second = _get_cached_policy_section_catalogue(
                corpus_dir,
                "1.0",
            )

        assert first is catalogue
        assert second is first
        build_mock.assert_called_once_with(
            corpus_dir
        )

        info = _get_cached_policy_section_catalogue.cache_info()

        assert info.misses == 1
        assert info.hits == 1
        assert info.currsize == 1
        assert info.maxsize == 1
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


def test_cached_policy_section_catalogue_invalidates_on_version_change() -> None:
    """Changing corpus version must produce a new catalogue build."""

    corpus_dir = Path("/tmp/r6-cache-corpus")
    first_catalogue = ("v1",)
    second_catalogue = ("v2",)

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with patch(
            "rag.retrieve._build_policy_section_catalogue",
            side_effect=(
                first_catalogue,
                second_catalogue,
            ),
        ) as build_mock:
            first = _get_cached_policy_section_catalogue(
                corpus_dir,
                "1.0",
            )

            second = _get_cached_policy_section_catalogue(
                corpus_dir,
                "1.1",
            )

        assert first is first_catalogue
        assert second is second_catalogue
        assert build_mock.call_count == 2

        info = _get_cached_policy_section_catalogue.cache_info()

        assert info.misses == 2
        assert info.hits == 0
        assert info.currsize == 1
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


def test_cached_policy_section_catalogue_invalidates_on_path_change() -> None:
    """Changing the absolute corpus root must change cache identity."""

    first_dir = Path("/tmp/r6-corpus-a")
    second_dir = Path("/tmp/r6-corpus-b")

    first_catalogue = ("first",)
    second_catalogue = ("second",)

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with patch(
            "rag.retrieve._build_policy_section_catalogue",
            side_effect=(
                first_catalogue,
                second_catalogue,
            ),
        ) as build_mock:
            first = _get_cached_policy_section_catalogue(
                first_dir,
                "1.0",
            )

            second = _get_cached_policy_section_catalogue(
                second_dir,
                "1.0",
            )

        assert first is first_catalogue
        assert second is second_catalogue
        assert build_mock.call_count == 2
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "corpus",
        1,
        True,
        (),
    ],
)
def test_cached_policy_section_catalogue_rejects_wrong_path_type(
    value: object,
) -> None:
    """Cached catalogue keys require pathlib.Path corpus roots."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with pytest.raises(
            TypeError,
            match="corpus_dir must be a pathlib.Path instance",
        ):
            _get_cached_policy_section_catalogue(
                value,  # type: ignore[arg-type]
                "1.0",
            )
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


def test_cached_policy_section_catalogue_rejects_relative_path() -> None:
    """The cached catalogue must not accept a relative corpus root."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with pytest.raises(
            ValueError,
            match="corpus_dir must be absolute",
        ):
            _get_cached_policy_section_catalogue(
                Path("corpus"),
                "1.0",
            )
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        True,
        (),
    ],
)
def test_cached_policy_section_catalogue_rejects_wrong_version_type(
    value: object,
) -> None:
    """Corpus-version cache keys must be strings."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with pytest.raises(
            TypeError,
            match="corpus_version must be a string",
        ):
            _get_cached_policy_section_catalogue(
                Path("/tmp/r6-cache-corpus"),
                value,  # type: ignore[arg-type]
            )
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_cached_policy_section_catalogue_rejects_blank_version(
    value: str,
) -> None:
    """Blank corpus versions must not participate in cache identity."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with pytest.raises(
            ValueError,
            match="corpus_version must be a non-empty string",
        ):
            _get_cached_policy_section_catalogue(
                Path("/tmp/r6-cache-corpus"),
                value,
            )
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


def test_get_policy_section_catalogue_composes_active_corpus_identity() -> None:
    """Public catalogue access must resolve path and validated version."""

    corpus_dir = Path("/tmp/r6-active-corpus")
    manifest = Mock()
    manifest.version = "7.4"
    catalogue = ("catalogue",)

    with (
        patch(
            "rag.retrieve.resolve_corpus_dir",
            return_value=corpus_dir,
        ) as resolve_mock,
        patch(
            "rag.retrieve.load_manifest",
            return_value=manifest,
        ) as manifest_mock,
        patch(
            "rag.retrieve._get_cached_policy_section_catalogue",
            return_value=catalogue,
        ) as cache_mock,
    ):
        result = get_policy_section_catalogue()

    assert result is catalogue

    resolve_mock.assert_called_once_with()

    manifest_mock.assert_called_once_with(
        corpus_dir
        / "version.json"
    )

    cache_mock.assert_called_once_with(
        corpus_dir,
        "7.4",
    )


def test_get_policy_section_catalogue_returns_real_cached_catalogue() -> None:
    """Public access must lazily reuse the real active-corpus catalogue."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        first = get_policy_section_catalogue()
        second = get_policy_section_catalogue()

        assert len(first) == 400
        assert first is second

        info = _get_cached_policy_section_catalogue.cache_info()

        assert info.misses == 1
        assert info.hits == 1
        assert info.currsize == 1
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


@pytest.mark.parametrize(
    ("corpus_dir", "corpus_version"),
    [
        ({}, "1.0"),
        (Path("/tmp/r6-cache-corpus"), {}),
    ],
)
def test_cached_policy_section_catalogue_rejects_unhashable_cache_keys(
    corpus_dir: object,
    corpus_version: object,
) -> None:
    """Unhashable cache keys must fail safely before cache execution."""

    _get_cached_policy_section_catalogue.cache_clear()

    try:
        with pytest.raises(
            TypeError,
        ):
            _get_cached_policy_section_catalogue(
                corpus_dir,  # type: ignore[arg-type]
                corpus_version,  # type: ignore[arg-type]
            )

        info = _get_cached_policy_section_catalogue.cache_info()

        assert info.hits == 0
        assert info.misses == 0
        assert info.currsize == 0
    finally:
        _get_cached_policy_section_catalogue.cache_clear()


@pytest.mark.parametrize(
    (
        "doc_id",
        "section",
        "expected",
    ),
    [
        (
            "HR-POL-004",
            "5.3 International approval",
            (
                "HR-POL-004",
                "5.3 International approval",
            ),
        ),
        (
            "  HR-POL-004  ",
            "  5.3 International approval  ",
            (
                "HR-POL-004",
                "5.3 International approval",
            ),
        ),
        (
            "HR-POL-004",
            "5.3",
            (
                "HR-POL-004",
                "5.3",
            ),
        ),
        (
            " HR-POL-001 ",
            " Employee Handbook ",
            (
                "HR-POL-001",
                "Employee Handbook",
            ),
        ),
    ],
)
def test_validate_policy_section_lookup_returns_trimmed_values(
    doc_id: str,
    section: str,
    expected: tuple[str, str],
) -> None:
    """Valid exact-section lookup inputs must be trimmed deterministically."""

    result = _validate_policy_section_lookup(
        doc_id,
        section,
    )

    assert result == expected


@pytest.mark.parametrize(
    "doc_id",
    [
        None,
        1,
        5.3,
        True,
        (),
        {},
    ],
)
def test_validate_policy_section_lookup_rejects_non_string_doc_id(
    doc_id: object,
) -> None:
    """Exact lookup document IDs must be strings."""

    with pytest.raises(
        TypeError,
        match="doc_id must be a string",
    ):
        _validate_policy_section_lookup(
            doc_id,  # type: ignore[arg-type]
            "5.3",
        )


@pytest.mark.parametrize(
    "section",
    [
        None,
        1,
        5.3,
        True,
        (),
        {},
    ],
)
def test_validate_policy_section_lookup_rejects_non_string_section(
    section: object,
) -> None:
    """Exact lookup section values must be strings."""

    with pytest.raises(
        TypeError,
        match="section must be a string",
    ):
        _validate_policy_section_lookup(
            "HR-POL-004",
            section,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "doc_id",
    [
        "",
        " ",
        "   ",
        "\n\t",
    ],
)
def test_validate_policy_section_lookup_rejects_blank_doc_id(
    doc_id: str,
) -> None:
    """Blank document identifiers must fail before catalogue matching."""

    with pytest.raises(
        ValueError,
        match="doc_id must be a non-empty string",
    ):
        _validate_policy_section_lookup(
            doc_id,
            "5.3",
        )


@pytest.mark.parametrize(
    "section",
    [
        "",
        " ",
        "   ",
        "\n\t",
    ],
)
def test_validate_policy_section_lookup_rejects_blank_section(
    section: str,
) -> None:
    """Blank section values must fail before catalogue matching."""

    with pytest.raises(
        ValueError,
        match="section must be a non-empty string",
    ):
        _validate_policy_section_lookup(
            "HR-POL-004",
            section,
        )


def test_validate_policy_section_lookup_preserves_doc_id_case() -> None:
    """Document identifiers must not be silently case-normalized."""

    result = _validate_policy_section_lookup(
        "hr-pol-004",
        "5.3",
    )

    assert result == (
        "hr-pol-004",
        "5.3",
    )


def test_validate_policy_section_lookup_preserves_section_case() -> None:
    """Validation trims section text but leaves matching semantics downstream."""

    result = _validate_policy_section_lookup(
        "HR-POL-004",
        "5.3 INTERNATIONAL APPROVAL",
    )

    assert result == (
        "HR-POL-004",
        "5.3 INTERNATIONAL APPROVAL",
    )
