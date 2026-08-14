"""Focused tests for the S4 retrieval-domain contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rag.retrieve import (
    DEFAULT_RETRIEVAL_K,
    RetrievalError,
    RetrievalResult,
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
