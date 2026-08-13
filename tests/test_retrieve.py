"""Focused tests for the S4 retrieval-domain contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rag.retrieve import (
    DEFAULT_RETRIEVAL_K,
    RetrievalError,
    RetrievalResult,
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
