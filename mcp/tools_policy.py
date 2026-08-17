"""Policy-backed MCP tool adapters."""

from __future__ import annotations

from rag.retrieve import RetrievalResult


def _convert_retrieval_results(
    results: tuple[RetrievalResult, ...],
) -> list[dict[str, str | float]]:
    """Convert retrieval results to the frozen MCP response shape.

    Conversion preserves retrieval ranking and copies only fields
    defined by the MCP policy-search contract. Retrieval scoring,
    section derivation, snippet construction, and validation remain
    owned by ``rag.retrieve``.

    Args:
        results:
            Validated retrieval-domain results in ranking order.

    Returns:
        Plain JSON-compatible MCP response records.

    Raises:
        TypeError:
            If ``results`` is not a tuple or contains values that are
            not ``RetrievalResult`` instances.
    """

    if not isinstance(
        results,
        tuple,
    ):
        raise TypeError(
            "results must be a tuple."
        )

    if any(
        not isinstance(
            result,
            RetrievalResult,
        )
        for result in results
    ):
        raise TypeError(
            "results must contain only RetrievalResult instances."
        )

    return [
        {
            "doc_id": result.doc_id,
            "title": result.title,
            "section": result.section,
            "snippet": result.snippet,
            "score": result.similarity,
        }
        for result in results
    ]
