"""Policy-backed MCP tool adapters."""

from __future__ import annotations

from rag.retrieve import (
    PolicySection,
    RetrievalResult,
    get_policy_section as retrieve_policy_section,
    retrieve_policy,
)


def _convert_policy_section(
    result: PolicySection,
) -> dict[str, str]:
    """Convert one exact policy section to the frozen MCP response shape.

    Exact lookup, matching, normalization, corpus access, and section
    validation remain owned by ``rag.retrieve``. This adapter performs
    only the public MCP contract projection.

    Args:
        result:
            One validated exact-section domain object.

    Returns:
        Plain JSON-compatible exact-section evidence containing only
        ``title``, ``section``, and complete ``text``.

    Raises:
        TypeError:
            If ``result`` is not a ``PolicySection`` instance.
    """

    if not isinstance(
        result,
        PolicySection,
    ):
        raise TypeError(
            "result must be a PolicySection instance."
        )

    return {
        "title": result.title,
        "section": result.section,
        "text": result.text,
    }


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


def get_policy_section(
    doc_id: str,
    section: str,
) -> dict[str, str]:
    """Return one exact policy section in the frozen MCP response shape.

    Input validation, corpus access, section matching, ambiguity
    handling, and complete section-text retrieval remain owned by
    ``rag.retrieve``. This function only composes the existing exact
    lookup with the MCP response projection.

    Args:
        doc_id:
            Stable policy document identifier forwarded unchanged to
            exact retrieval.
        section:
            Complete section heading or canonical numeric identifier
            forwarded unchanged to exact retrieval.

    Returns:
        Plain JSON-compatible exact policy section containing only
        ``title``, ``section``, and complete ``text``.

    Raises:
        TypeError:
            If delegated exact lookup rejects an argument type.
        ValueError:
            If delegated exact lookup rejects an argument value.
        RetrievalError:
            If the delegated policy document or section cannot be
            resolved uniquely.
    """

    result = retrieve_policy_section(
        doc_id,
        section,
    )

    return _convert_policy_section(
        result
    )


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
