"""Citation-ready retrieval contracts for the S4 RAG pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from rag.ingest import SUPPORTED_SOURCE_FORMATS
from rag.store import (
    ChromaStoreError,
    get_chroma_client,
    get_policy_collection,
    resolve_chroma_dir,
)


DEFAULT_RETRIEVAL_K: Final[int] = 5
ALLOWED_RETRIEVAL_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "doc_id",
        "source_format",
    }
)


class RetrievalError(RuntimeError):
    """Base exception for retrieval-runtime failures."""


@dataclass(frozen=True)
class RetrievalResult:
    """Represent one validated citation-ready retrieval result.

    The retrieval layer preserves both the concise section value needed
    by downstream MCP/API contracts and the complete section path needed
    for provenance, debugging, and evaluation.

    ``distance`` stores the raw Chroma cosine distance. ``similarity``
    uses the project-facing orientation where higher is better:

        similarity = 1.0 - distance

    Attributes:
        chunk_id:
            Deterministic canonical chunk identifier.
        doc_id:
            Stable policy document identifier.
        title:
            Human-readable policy title.
        section:
            Public citation section, defined as the leaf heading of
            ``section_path``.
        section_path:
            Complete immutable heading hierarchy.
        snippet:
            Non-empty supporting policy text.
        source_format:
            Original corpus source format.
        distance:
            Finite raw cosine distance returned by Chroma.
        similarity:
            Finite similarity score derived as ``1.0 - distance``.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_path: tuple[str, ...]
    snippet: str
    source_format: str
    distance: float
    similarity: float

    def __post_init__(self) -> None:
        """Validate invariants intrinsic to one retrieval result."""

        for field_name in (
            "chunk_id",
            "doc_id",
            "title",
            "section",
            "snippet",
            "source_format",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if (
            not isinstance(self.section_path, tuple)
            or not self.section_path
            or any(
                not isinstance(part, str) or not part.strip()
                for part in self.section_path
            )
        ):
            raise ValueError(
                "section_path must be a non-empty tuple of "
                "non-empty strings."
            )

        if self.title != self.section_path[0]:
            raise ValueError(
                "title must match the first section_path element."
            )

        if self.section != self.section_path[-1]:
            raise ValueError(
                "section must match the final section_path element."
            )

        for field_name in (
            "distance",
            "similarity",
        ):
            value = getattr(self, field_name)

            if (
                not isinstance(value, float)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{field_name} must be a finite float."
                )

        expected_similarity = 1.0 - self.distance

        if not math.isclose(
            self.similarity,
            expected_similarity,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "similarity must equal 1.0 - distance."
            )

def _validate_retrieval_query(
    query: str,
) -> None:
    """Validate one public retrieval query.

    Retrieval owns its request boundary independently from the embedding
    layer so callers receive deterministic validation before any model
    or vector-store work begins.

    Args:
        query:
            Raw policy-search query.

    Raises:
        TypeError:
            If ``query`` is not a string.
        ValueError:
            If ``query`` is empty or contains only whitespace.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "query must be a string."
        )

    if not query.strip():
        raise ValueError(
            "query must be a non-empty string."
        )


def _validate_retrieval_k(
    k: int,
) -> None:
    """Validate one requested retrieval depth.

    Args:
        k:
            Number of nearest policy chunks requested by the caller.

    Raises:
        TypeError:
            If ``k`` is not an integer or is a Boolean.
        ValueError:
            If ``k`` is not positive.
    """

    if (
        not isinstance(
            k,
            int,
        )
        or isinstance(
            k,
            bool,
        )
    ):
        raise TypeError(
            "k must be an integer."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )


def _validate_retrieval_filters(
    filters: dict[str, str] | None,
) -> None:
    """Validate optional public metadata filters for retrieval.

    Only the retrieval fields intentionally exposed by the CP9 contract
    are accepted. Chroma-specific ``where`` construction belongs to a
    later adapter step and is deliberately not performed here.

    Args:
        filters:
            Optional mapping containing ``doc_id`` and/or
            ``source_format`` equality filters.

    Raises:
        TypeError:
            If ``filters`` is not ``None`` or a dictionary, or if a
            filter value is not a string.
        ValueError:
            If a filter key is unsupported, a filter value is blank, or
            ``source_format`` is not a supported corpus format.
    """

    if filters is None:
        return

    if not isinstance(
        filters,
        dict,
    ):
        raise TypeError(
            "filters must be a dictionary or None."
        )

    for field_name in filters:
        if not isinstance(
            field_name,
            str,
        ):
            raise TypeError(
                "retrieval filter keys must be strings."
            )

    unsupported_keys = (
        set(filters)
        - ALLOWED_RETRIEVAL_FILTERS
    )

    if unsupported_keys:
        unsupported = ", ".join(
            sorted(
                str(key)
                for key in unsupported_keys
            )
        )

        raise ValueError(
            "Unsupported retrieval filter key(s): "
            f"{unsupported}."
        )

    for field_name, value in filters.items():
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "retrieval filter "
                f"{field_name!r} must be a string."
            )

        if not value.strip():
            raise ValueError(
                "retrieval filter "
                f"{field_name!r} must be a non-empty string."
            )

    source_format = filters.get(
        "source_format"
    )

    if (
        source_format is not None
        and source_format
        not in SUPPORTED_SOURCE_FORMATS
    ):
        supported = ", ".join(
            sorted(
                SUPPORTED_SOURCE_FORMATS
            )
        )

        raise ValueError(
            "retrieval filter 'source_format' is unsupported: "
            f"{source_format!r}; expected one of {supported}."
        )


def _get_active_policy_collection() -> object:
    """Open the validated active policy collection for runtime retrieval.

    Runtime retrieval is read-only with respect to index lifecycle.
    The configured persistence directory must already exist before
    Chroma client construction so an absent published index is not
    accidentally materialized as an empty database directory.

    Freshness checking and rebuilding belong to application startup and
    deployment lifecycle logic rather than the per-query retrieval path.

    Returns:
        Existing Chroma policy collection satisfying the frozen storage
        contract.

    Raises:
        RetrievalError:
            If the configured active index is missing, is not a
            directory, or the storage layer cannot open and validate
            the existing policy collection.
    """

    try:
        chroma_dir = resolve_chroma_dir()
    except ChromaStoreError as exc:
        raise RetrievalError(
            "Failed to resolve the active Chroma index directory."
        ) from exc

    if not chroma_dir.exists():
        raise RetrievalError(
            "Active Chroma index directory does not exist: "
            f"{str(chroma_dir)!r}."
        )

    if not chroma_dir.is_dir():
        raise RetrievalError(
            "Active Chroma index path is not a directory: "
            f"{str(chroma_dir)!r}."
        )

    try:
        client = get_chroma_client(
            chroma_dir
        )

        return get_policy_collection(
            client
        )
    except ChromaStoreError as exc:
        raise RetrievalError(
            "Failed to open the active policy retrieval index."
        ) from exc
