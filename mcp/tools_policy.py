"""Policy-backed MCP tool adapters."""

from __future__ import annotations

from rag.retrieve import (
    RetrievalResult,
    retrieve_policy,
)


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


def search_policy_documents(
    query: str,
    k: int = 5,
) -> list[dict[str, str | float]]:
    """Search indexed policy evidence and return the MCP response shape.

    Validation, embedding, vector-store access, ranking, and retrieval
    result construction remain owned by ``rag.retrieve``. This function
    only composes retrieval with the existing MCP response adapter.

    Args:
        query:
            Policy search query forwarded unchanged to retrieval.
        k:
            Number of ranked policy results requested. The frozen MCP
            contract fixes the public default at 5.

    Returns:
        Plain JSON-compatible policy evidence records in retrieval order.

    Raises:
        TypeError:
            If the delegated retrieval request violates its type contract.
        ValueError:
            If the delegated retrieval request violates its value contract.
        RetrievalError:
            If the delegated retrieval pipeline fails.
    """

    results = retrieve_policy(
        query,
        k=k,
    )

    return _convert_retrieval_results(
        results
    )
