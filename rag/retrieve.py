"""Citation-ready retrieval contracts for the S4 RAG pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


DEFAULT_RETRIEVAL_K: Final[int] = 5


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
