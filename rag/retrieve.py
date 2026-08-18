"""Citation-ready retrieval contracts for the S4 RAG pipeline."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from rag.embed import EmbeddingError, embed_query
from rag.ingest import (
    ParsedSection,
    SUPPORTED_SOURCE_FORMATS,
    load_manifest,
    normalize_section,
    parse_markdown_document,
    parse_pdf_document,
    resolve_manifest_sources,
)
from rag.store import (
    ChromaStoreError,
    get_chroma_client,
    get_policy_collection,
    resolve_chroma_dir,
)


DEFAULT_CORPUS_DIR: Final[Path] = Path("corpus")
DEFAULT_RETRIEVAL_K: Final[int] = 5
ALLOWED_RETRIEVAL_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "doc_id",
        "source_format",
    }
)


class RetrievalError(RuntimeError):
    """Base exception for retrieval-runtime failures."""


class _CliArgumentParser(argparse.ArgumentParser):
    """Argument parser that preserves the frozen CLI error contract."""

    def error(
        self,
        message: str,
    ) -> None:
        """Raise a normal validation error instead of exiting with code 2."""

        raise ValueError(
            f"invalid command-line arguments: {message}"
        )


def _parse_cli_filter(
    value: str,
) -> tuple[str, str]:
    """Parse one CLI ``KEY=VALUE`` retrieval filter.

    This helper validates only the CLI transport shape. Supported filter
    names and filter-value semantics remain owned by the existing
    retrieval validation layer.

    Args:
        value:
            Raw filter argument supplied after ``--filter``.

    Returns:
        Normalized ``(key, value)`` pair.

    Raises:
        ValueError:
            If the argument does not contain ``=``, or if either side is
            blank after trimming whitespace.
    """

    if "=" not in value:
        raise ValueError(
            "filter must use KEY=VALUE format."
        )

    key, filter_value = value.split(
        "=",
        1,
    )

    key = key.strip()
    filter_value = filter_value.strip()

    if not key:
        raise ValueError(
            "filter key must be non-empty."
        )

    if not filter_value:
        raise ValueError(
            "filter value must be non-empty."
        )

    return (
        key,
        filter_value,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the frozen semantic-retrieval command-line interface.

    The CLI exposes semantic retrieval only. Exact policy-section lookup
    remains a separate Python API and is intentionally not exposed here.

    Returns:
        Configured argument parser for ``python -m rag.retrieve``.
    """

    parser = _CliArgumentParser(
        prog="python -m rag.retrieve",
        description=(
            "Retrieve citation-ready HR policy evidence "
            "from the active RAG index."
        ),
    )

    parser.add_argument(
        "query",
        help="Policy retrieval query.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help=(
            "Number of ranked policy results to return "
            f"(default: {DEFAULT_RETRIEVAL_K})."
        ),
    )

    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Optional retrieval filter. Repeat for multiple filters. "
            "Supported semantic keys are validated by the retrieval layer."
        ),
    )

    return parser


def _build_cli_filters(
    values: list[str],
) -> dict[str, str] | None:
    """Build retrieval filters from repeated CLI ``--filter`` values.

    Each raw value is parsed by ``_parse_cli_filter``. This helper owns
    only repeated-argument assembly and duplicate-key rejection.
    Supported filter names and value semantics remain owned by the
    retrieval validation layer.

    Args:
        values:
            Raw ``KEY=VALUE`` values collected by argparse.

    Returns:
        ``None`` when no filters were supplied, otherwise a dictionary
        preserving the supplied key/value pairs.

    Raises:
        TypeError:
            If ``values`` is not a list or contains a non-string member.
        ValueError:
            If one filter has invalid structural syntax or a key is
            repeated.
    """

    if not isinstance(
        values,
        list,
    ):
        raise TypeError(
            "CLI filters must be a list."
        )

    filters: dict[str, str] = {}

    for value in values:
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "CLI filter values must be strings."
            )

        key, filter_value = _parse_cli_filter(
            value
        )

        if key in filters:
            raise ValueError(
                f"duplicate filter key: {key!r}."
            )

        filters[key] = filter_value

    if not filters:
        return None

    return filters


def _format_cli_retrieval_result(
    result: RetrievalResult,
    rank: int,
) -> str:
    """Format one retrieval result for deterministic CLI display.

    Formatting exposes only citation-ready fields already owned by the
    retrieval-domain contract. It does not alter ranking, scores,
    snippets, or citation provenance.

    Args:
        result:
            Validated retrieval-domain result.
        rank:
            One-based ranking position assigned by the caller.

    Returns:
        Human-readable deterministic text block.

    Raises:
        TypeError:
            If ``result`` is not a ``RetrievalResult`` or ``rank`` is
            not an integer.
        ValueError:
            If ``rank`` is not positive.
    """

    if not isinstance(
        result,
        RetrievalResult,
    ):
        raise TypeError(
            "result must be a RetrievalResult."
        )

    if (
        not isinstance(
            rank,
            int,
        )
        or isinstance(
            rank,
            bool,
        )
    ):
        raise TypeError(
            "rank must be an integer."
        )

    if rank <= 0:
        raise ValueError(
            "rank must be positive."
        )

    return "\n".join(
        (
            f"Rank: {rank}",
            f"Similarity: {result.similarity:.6f}",
            f"Document ID: {result.doc_id}",
            f"Title: {result.title}",
            f"Section: {result.section}",
            f"Snippet: {result.snippet}",
        )
    )


def resolve_corpus_dir() -> Path:
    """Return the configured policy corpus directory.

    The ``CORPUS_DIR`` environment variable overrides the project
    default. User-home markers are expanded and the result is resolved
    to an absolute path so retrieval code does not depend on the
    caller's current path representation.

    This helper performs configuration resolution only. Existence and
    corpus-structure validation belong to later catalogue-construction
    stages.

    Returns:
        Absolute path to the configured policy corpus directory.

    Raises:
        RetrievalError:
            If ``CORPUS_DIR`` is defined but contains only whitespace.
    """

    configured = os.getenv(
        "CORPUS_DIR"
    )

    if configured is None:
        path = DEFAULT_CORPUS_DIR
    else:
        if not configured.strip():
            raise RetrievalError(
                "CORPUS_DIR must not be blank when configured."
            )

        path = Path(
            configured.strip()
        )

    return path.expanduser().resolve()


SECTION_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\d+(?:\.\d+)*)(?=\.\s|\s|$)"
)


def _extract_section_number(
    section: str,
) -> str | None:
    """Return the exact numeric prefix from one section heading.

    Section numbers are derived only from a leading canonical numeric
    prefix such as ``5`` or ``5.3``. Document-root headings without a
    numeric prefix return ``None``.

    Args:
        section:
            Non-empty section or leaf-heading value.

    Returns:
        The leading section number, or ``None`` when the heading is
        unnumbered.

    Raises:
        TypeError:
            If ``section`` is not a string.
        ValueError:
            If ``section`` is empty or contains only whitespace.
    """

    if not isinstance(
        section,
        str,
    ):
        raise TypeError(
            "section must be a string."
        )

    section = section.strip()

    if not section:
        raise ValueError(
            "section must be a non-empty string."
        )

    match = SECTION_NUMBER_PATTERN.match(
        section
    )

    if match is None:
        return None

    return match.group(1)


def _validate_policy_section_lookup(
    doc_id: str,
    section: str,
) -> tuple[str, str]:
    """Validate and canonicalize one exact policy-section lookup.

    Exact lookup accepts stable policy document identifiers and either
    a complete section heading or canonical numeric section identifier.
    Surrounding whitespace is ignored, but document identifiers remain
    case-sensitive.

    Args:
        doc_id:
            Stable policy document identifier.
        section:
            Complete leaf heading or numeric section identifier.

    Returns:
        Trimmed ``(doc_id, section)`` lookup values.

    Raises:
        TypeError:
            If either argument is not a string.
        ValueError:
            If either argument is empty or whitespace-only.
    """

    if not isinstance(
        doc_id,
        str,
    ):
        raise TypeError(
            "doc_id must be a string."
        )

    if not isinstance(
        section,
        str,
    ):
        raise TypeError(
            "section must be a string."
        )

    doc_id = doc_id.strip()
    section = section.strip()

    if not doc_id:
        raise ValueError(
            "doc_id must be a non-empty string."
        )

    if not section:
        raise ValueError(
            "section must be a non-empty string."
        )

    return (
        doc_id,
        section,
    )


@dataclass(frozen=True)
class PolicySection:
    """Represent one complete normalized policy section.

    This exact-lookup domain object preserves the full normalized
    section text produced before semantic chunking. It deliberately
    contains no vector-search score, chunk identifier, or retrieval
    ranking state.

    Attributes:
        doc_id:
            Stable policy document identifier.
        title:
            Human-readable policy title and root heading.
        section:
            Concise public section value defined as the leaf heading.
        section_path:
            Complete immutable heading hierarchy.
        section_number:
            Leading numeric section identifier, or ``None`` for an
            unnumbered document-root section.
        text:
            Complete non-empty normalized section text.
        source_format:
            Original corpus source format.
        section_order:
            Deterministic parser order within the source document.
    """

    doc_id: str
    title: str
    section: str
    section_path: tuple[str, ...]
    section_number: str | None
    text: str
    source_format: str
    section_order: int

    def __post_init__(self) -> None:
        """Validate invariants intrinsic to one exact policy section."""

        for field_name in (
            "doc_id",
            "title",
            "section",
            "text",
            "source_format",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if (
            not isinstance(
                self.section_path,
                tuple,
            )
            or not self.section_path
            or any(
                not isinstance(
                    part,
                    str,
                )
                or not part.strip()
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

        if (
            self.source_format
            not in SUPPORTED_SOURCE_FORMATS
        ):
            raise ValueError(
                "source_format must be a supported corpus format."
            )

        if (
            not isinstance(
                self.section_order,
                int,
            )
            or isinstance(
                self.section_order,
                bool,
            )
        ):
            raise ValueError(
                "section_order must be an integer."
            )

        if self.section_order < 0:
            raise ValueError(
                "section_order must be non-negative."
            )

        if (
            self.section_number is not None
            and (
                not isinstance(
                    self.section_number,
                    str,
                )
                or not self.section_number.strip()
            )
        ):
            raise ValueError(
                "section_number must be a non-empty string or None."
            )

        expected_section_number = _extract_section_number(
            self.section
        )

        if (
            self.section_number
            != expected_section_number
        ):
            raise ValueError(
                "section_number must match the numeric prefix "
                "of section."
            )


def _convert_parsed_section(
    section: ParsedSection,
) -> PolicySection:
    """Convert one normalized parsed section into exact lookup form.

    This helper performs only domain conversion. Parsing, text
    normalization, empty-section filtering, catalogue ordering, and
    caching remain responsibilities of later catalogue stages.

    Args:
        section:
            One already-normalized ParsedSection containing non-empty
            section text.

    Returns:
        An immutable PolicySection preserving parser metadata and the
        complete normalized section text.

    Raises:
        TypeError:
            If ``section`` is not a ParsedSection instance.
        ValueError:
            If the parsed section contains empty text or cannot satisfy
            the PolicySection domain contract.
    """

    if not isinstance(
        section,
        ParsedSection,
    ):
        raise TypeError(
            "section must be a ParsedSection instance."
        )

    if not section.text.strip():
        raise ValueError(
            "section text must be non-empty for exact policy lookup."
        )

    leaf_section = section.section_path[
        -1
    ]

    return PolicySection(
        doc_id=section.doc_id,
        title=section.title,
        section=leaf_section,
        section_path=section.section_path,
        section_number=_extract_section_number(
            leaf_section
        ),
        text=section.text,
        source_format=section.source_format,
        section_order=section.section_order,
    )


def _build_policy_section_catalogue(
    corpus_dir: Path,
) -> tuple[PolicySection, ...]:
    """Build the complete uncached exact-section catalogue.

    The catalogue reuses the authoritative ingestion pipeline through
    source resolution, format-specific parsing, and shared text
    normalization. Empty structural sections are omitted before
    conversion because exact policy lookup requires substantive text.

    Args:
        corpus_dir:
            Absolute root directory containing ``version.json`` and the
            ``source`` directory.

    Returns:
        Immutable PolicySection records preserving manifest, document,
        and parser section order.

    Raises:
        TypeError:
            If ``corpus_dir`` is not a pathlib.Path instance.
        ValueError:
            If ``corpus_dir`` is not absolute.
        ManifestValidationError:
            If the authoritative manifest cannot be validated.
        SourceResolutionError:
            If the corpus source tree does not match the manifest.
        MarkdownParseError:
            If a Markdown policy cannot be parsed safely.
        PdfParseError:
            If a PDF policy cannot be parsed safely.
        RuntimeError:
            If a resolved document somehow carries an unsupported
            source format.
    """

    if not isinstance(
        corpus_dir,
        Path,
    ):
        raise TypeError(
            "corpus_dir must be a pathlib.Path instance."
        )

    if not corpus_dir.is_absolute():
        raise ValueError(
            "corpus_dir must be absolute."
        )

    manifest_path = (
        corpus_dir
        / "version.json"
    )

    source_root = (
        corpus_dir
        / "source"
    )

    manifest = load_manifest(
        manifest_path
    )

    resolved_documents = resolve_manifest_sources(
        manifest,
        source_root,
    )

    catalogue: list[PolicySection] = []

    for document in resolved_documents:
        if document.source_format == "md":
            sections = parse_markdown_document(
                document
            )
        elif document.source_format == "pdf":
            sections = parse_pdf_document(
                document
            )
        else:
            raise RuntimeError(
                "Resolved corpus contains unsupported source "
                f"format: {document.source_format!r}."
            )

        for section in sections:
            normalized = normalize_section(
                section
            )

            if not normalized.text.strip():
                continue

            catalogue.append(
                _convert_parsed_section(
                    normalized
                )
            )

    return tuple(
        catalogue
    )


@lru_cache(maxsize=1)
def _get_cached_policy_section_catalogue(
    corpus_dir: Path,
    corpus_version: str,
) -> tuple[PolicySection, ...]:
    """Return the cached catalogue for one corpus identity.

    The cache key combines the resolved corpus root and validated
    top-level manifest version. Corpus version is the project's
    authoritative invalidation signal for policy-source changes.

    Args:
        corpus_dir:
            Absolute resolved policy corpus root.
        corpus_version:
            Validated top-level corpus manifest version.

    Returns:
        Immutable exact-section catalogue for this corpus identity.

    Raises:
        TypeError:
            If either cache-key value has the wrong type.
        ValueError:
            If the path is relative or the version is blank.
    """

    if not isinstance(
        corpus_dir,
        Path,
    ):
        raise TypeError(
            "corpus_dir must be a pathlib.Path instance."
        )

    if not corpus_dir.is_absolute():
        raise ValueError(
            "corpus_dir must be absolute."
        )

    if not isinstance(
        corpus_version,
        str,
    ):
        raise TypeError(
            "corpus_version must be a string."
        )

    if not corpus_version.strip():
        raise ValueError(
            "corpus_version must be a non-empty string."
        )

    return _build_policy_section_catalogue(
        corpus_dir
    )


def get_policy_section_catalogue() -> tuple[PolicySection, ...]:
    """Return the lazy catalogue for the currently configured corpus.

    Each call resolves the active corpus root and validates its
    manifest. The resolved path and top-level corpus version form the
    cache identity, so full source parsing occurs only on a cache miss.

    Returns:
        Immutable exact-section catalogue for the active corpus.
    """

    corpus_dir = resolve_corpus_dir()

    manifest = load_manifest(
        corpus_dir
        / "version.json"
    )

    return _get_cached_policy_section_catalogue(
        corpus_dir,
        manifest.version,
    )


def _match_policy_section(
    catalogue: tuple[PolicySection, ...],
    doc_id: str,
    section: str,
) -> PolicySection:
    """Match one exact policy section within an existing catalogue.

    Matching is deliberately deterministic. The stable document ID is
    matched exactly. Within that document, a complete leaf heading is
    matched case-insensitively first; if no heading matches, the lookup
    value is matched exactly against the canonical numeric section
    identifier.

    Args:
        catalogue:
            Immutable exact-section catalogue.
        doc_id:
            Validated stable policy document identifier.
        section:
            Validated complete leaf heading or numeric section
            identifier.

    Returns:
        The uniquely matched PolicySection.

    Raises:
        TypeError:
            If ``catalogue`` is not a tuple or contains non-PolicySection
            values.
        RetrievalError:
            If the document does not exist, the requested section does
            not exist within that document, or matching is ambiguous.
    """

    if not isinstance(
        catalogue,
        tuple,
    ):
        raise TypeError(
            "catalogue must be a tuple."
        )

    if any(
        not isinstance(
            item,
            PolicySection,
        )
        for item in catalogue
    ):
        raise TypeError(
            "catalogue must contain only PolicySection instances."
        )

    document_sections = tuple(
        item
        for item in catalogue
        if item.doc_id == doc_id
    )

    if not document_sections:
        raise RetrievalError(
            f"Policy document not found: {doc_id!r}."
        )

    heading_key = section.casefold()

    heading_matches = tuple(
        item
        for item in document_sections
        if item.section.casefold()
        == heading_key
    )

    if len(heading_matches) > 1:
        raise RetrievalError(
            "Ambiguous policy section heading for "
            f"document {doc_id!r}: {section!r}."
        )

    if len(heading_matches) == 1:
        return heading_matches[0]

    number_matches = tuple(
        item
        for item in document_sections
        if item.section_number
        == section
    )

    if len(number_matches) > 1:
        raise RetrievalError(
            "Ambiguous policy section number for "
            f"document {doc_id!r}: {section!r}."
        )

    if len(number_matches) == 1:
        return number_matches[0]

    raise RetrievalError(
        "Policy section not found for "
        f"document {doc_id!r}: {section!r}."
    )


def get_policy_section(
    doc_id: str,
    section: str,
) -> PolicySection:
    """Return one exact policy section from the active corpus.

    This is the public exact-section retrieval composition boundary.
    Input validation and canonicalization, active catalogue access, and
    deterministic matching remain owned by their existing lower-level
    helpers.

    Args:
        doc_id:
            Stable policy document identifier.
        section:
            Complete leaf heading or canonical numeric section
            identifier.

    Returns:
        The uniquely matched immutable PolicySection.

    Raises:
        TypeError:
            If the lookup request violates the existing public type
            contract.
        ValueError:
            If the lookup request violates the existing public value
            contract.
        RetrievalError:
            If the requested policy document or section cannot be
            resolved uniquely.
    """

    validated_doc_id, validated_section = (
        _validate_policy_section_lookup(
            doc_id,
            section,
        )
    )

    catalogue = get_policy_section_catalogue()

    return _match_policy_section(
        catalogue,
        validated_doc_id,
        validated_section,
    )


@dataclass(frozen=True)
class ValidatedRetrievalRows:
    """Represent structurally validated rows from one Chroma query.

    Chroma returns one outer list per submitted query. Retrieval submits
    exactly one query, so this contract removes that transport-specific
    outer dimension while preserving aligned raw row values for the
    later result-conversion stage.
    """

    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[dict[str, object], ...]
    distances: tuple[float, ...]


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


def _compile_chroma_where(
    filters: dict[str, str] | None,
) -> dict[str, object] | None:
    """Compile validated public filters into Chroma ``where`` syntax.

    Public retrieval filters intentionally hide Chroma's logical-filter
    representation. One equality filter maps directly to a single
    Chroma condition. Two filters require the explicit ``$and`` form
    verified against the pinned Chroma runtime.

    Args:
        filters:
            Already-validated public retrieval filters.

    Returns:
        ``None`` when no filters are supplied, one equality condition
        for a single filter, or an explicit ``$and`` expression for
        both supported filters.

    Raises:
        TypeError:
            If the public filter container or values violate the
            retrieval request contract.
        ValueError:
            If keys, values, or source format violate that contract.
    """

    _validate_retrieval_filters(
        filters
    )

    if not filters:
        return None

    conditions = [
        {
            field_name: filters[field_name],
        }
        for field_name in (
            "doc_id",
            "source_format",
        )
        if field_name in filters
    ]

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


def _query_policy_collection_raw(
    query: str,
    *,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: dict[str, str] | None = None,
) -> object:
    """Execute one validated semantic query against the active index.

    This helper owns request validation, query embedding, active-index
    access, Chroma filter compilation, and one raw Chroma query call.
    It deliberately returns the Chroma response unchanged; structural
    response validation belongs to the next retrieval checkpoint.

    Args:
        query:
            Non-empty policy retrieval query.
        k:
            Positive number of nearest records requested.
        filters:
            Optional validated public retrieval filters.

    Returns:
        Raw response returned by the pinned Chroma collection query.

    Raises:
        TypeError:
            If the query, result count, or filter request violates the
            public retrieval type contract.
        ValueError:
            If those request values violate the public retrieval value
            contract.
        RetrievalError:
            If query embedding fails, the active retrieval index cannot
            be opened, or Chroma query execution fails.
    """

    _validate_retrieval_query(
        query
    )
    _validate_retrieval_k(
        k
    )
    _validate_retrieval_filters(
        filters
    )

    try:
        query_embedding = embed_query(
            query
        )
    except EmbeddingError as exc:
        raise RetrievalError(
            "Failed to embed policy retrieval query."
        ) from exc

    collection = _get_active_policy_collection()

    where = _compile_chroma_where(
        filters
    )

    query_kwargs: dict[str, object] = {
        "query_embeddings": [
            query_embedding,
        ],
        "n_results": k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        query_kwargs["where"] = where

    try:
        return collection.query(
            **query_kwargs
        )
    except Exception as exc:
        raise RetrievalError(
            "Failed to execute policy retrieval query."
        ) from exc


def _validate_raw_retrieval_response(
    response: object,
) -> ValidatedRetrievalRows:
    """Validate and unwrap one raw Chroma retrieval response.

    The retrieval adapter submits exactly one query, so each required
    Chroma result field must contain exactly one inner list. Zero rows
    are valid. Non-empty rows must preserve the citation/provenance
    contract written during CP8 index construction.

    Extra top-level Chroma bookkeeping fields are intentionally ignored.

    Args:
        response:
            Raw object returned by ``collection.query()``.

    Returns:
        Immutable aligned retrieval rows with the one-query outer
        dimension removed.

    Raises:
        RetrievalError:
            If the Chroma response violates the frozen retrieval
            structure, row, metadata, or distance contract.
    """

    if not isinstance(
        response,
        dict,
    ):
        raise RetrievalError(
            "Chroma retrieval response must be a dictionary."
        )

    required_fields = (
        "ids",
        "documents",
        "metadatas",
        "distances",
    )

    inner_values: dict[str, list[object]] = {}

    for field_name in required_fields:
        value = response.get(
            field_name
        )

        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], list)
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid "
                f"{field_name!r} structure."
            )

        inner_values[field_name] = value[0]

    ids = inner_values["ids"]
    documents = inner_values["documents"]
    metadatas = inner_values["metadatas"]
    distances = inner_values["distances"]

    result_count = len(ids)

    for field_name, value in (
        ("documents", documents),
        ("metadatas", metadatas),
        ("distances", distances),
    ):
        if len(value) != result_count:
            raise RetrievalError(
                "Chroma retrieval response has misaligned "
                f"{field_name}."
            )

    if (
        all(
            isinstance(
                chunk_id,
                str,
            )
            for chunk_id in ids
        )
        and len(
            set(ids)
        )
        != result_count
    ):
        raise RetrievalError(
            "Chroma retrieval response contains duplicate chunk IDs."
        )

    validated_ids: list[str] = []
    validated_documents: list[str] = []
    validated_metadatas: list[dict[str, object]] = []
    validated_distances: list[float] = []

    for index in range(
        result_count
    ):
        chunk_id = ids[index]
        document = documents[index]
        metadata = metadatas[index]
        distance = distances[index]

        if (
            not isinstance(
                chunk_id,
                str,
            )
            or not chunk_id.strip()
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid chunk ID at "
                f"result {index}."
            )

        if (
            not isinstance(
                document,
                str,
            )
            or not document.strip()
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid document at "
                f"result {index}."
            )

        if not isinstance(
            metadata,
            dict,
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid metadata at "
                f"result {index}."
            )

        required_metadata_fields = (
            "doc_id",
            "title",
            "section_path",
            "snippet",
            "source_format",
        )

        missing_metadata_fields = [
            field_name
            for field_name in required_metadata_fields
            if field_name not in metadata
        ]

        if missing_metadata_fields:
            raise RetrievalError(
                "Chroma retrieval metadata is missing required "
                f"field(s) at result {index}: "
                f"{missing_metadata_fields!r}."
            )

        doc_id = metadata["doc_id"]
        title = metadata["title"]
        section_path = metadata["section_path"]
        snippet = metadata["snippet"]
        source_format = metadata["source_format"]

        for field_name, value in (
            ("doc_id", doc_id),
            ("title", title),
            ("snippet", snippet),
            ("source_format", source_format),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise RetrievalError(
                    "Chroma retrieval metadata has invalid "
                    f"{field_name!r} at result {index}."
                )

        if (
            not isinstance(
                section_path,
                list,
            )
            or not section_path
            or any(
                not isinstance(part, str)
                or not part.strip()
                for part in section_path
            )
        ):
            raise RetrievalError(
                "Chroma retrieval metadata has invalid "
                f"'section_path' at result {index}."
            )

        assert isinstance(
            title,
            str,
        )
        assert isinstance(
            snippet,
            str,
        )
        assert isinstance(
            source_format,
            str,
        )

        if title != section_path[0]:
            raise RetrievalError(
                "Chroma retrieval metadata title does not match "
                f"section_path root at result {index}."
            )

        if (
            source_format
            not in SUPPORTED_SOURCE_FORMATS
        ):
            raise RetrievalError(
                "Chroma retrieval metadata has unsupported "
                f"source_format at result {index}: "
                f"{source_format!r}."
            )

        if snippet != document:
            raise RetrievalError(
                "Chroma retrieval document and citation snippet "
                f"do not match at result {index}."
            )

        if (
            not isinstance(
                distance,
                float,
            )
            or not math.isfinite(
                distance
            )
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid distance at "
                f"result {index}."
            )

        validated_ids.append(
            chunk_id
        )
        validated_documents.append(
            document
        )
        validated_metadatas.append(
            metadata
        )
        validated_distances.append(
            distance
        )

    return ValidatedRetrievalRows(
        ids=tuple(
            validated_ids
        ),
        documents=tuple(
            validated_documents
        ),
        metadatas=tuple(
            validated_metadatas
        ),
        distances=tuple(
            validated_distances
        ),
    )


def _build_retrieval_results(
    rows: ValidatedRetrievalRows,
) -> tuple[RetrievalResult, ...]:
    """Convert validated Chroma rows into citation-ready results.

    Conversion preserves the ranking order established by Chroma.
    Structural validation belongs to the preceding R3D boundary; this
    helper maps those validated values into the stable retrieval-domain
    contract without thresholding, reranking, or score clamping.

    Args:
        rows:
            Structurally validated and aligned rows from one Chroma
            retrieval query.

    Returns:
        Immutable retrieval results in the exact input ranking order.
        Zero validated rows produce an empty tuple.

    Raises:
        TypeError:
            If ``rows`` is not a ``ValidatedRetrievalRows`` instance.
        RetrievalError:
            If a supposedly validated row cannot be converted into the
            frozen ``RetrievalResult`` domain contract.
    """

    if not isinstance(
        rows,
        ValidatedRetrievalRows,
    ):
        raise TypeError(
            "rows must be a ValidatedRetrievalRows instance."
        )

    results: list[RetrievalResult] = []

    for index in range(
        len(rows.ids)
    ):
        try:
            metadata = rows.metadatas[
                index
            ]

            section_path_value = metadata[
                "section_path"
            ]

            if not isinstance(
                section_path_value,
                list,
            ):
                raise TypeError(
                    "validated section_path must be a list."
                )

            section_path = tuple(
                section_path_value
            )

            doc_id = metadata[
                "doc_id"
            ]
            title = metadata[
                "title"
            ]
            snippet = metadata[
                "snippet"
            ]
            source_format = metadata[
                "source_format"
            ]

            distance = rows.distances[
                index
            ]

            result = RetrievalResult(
                chunk_id=rows.ids[
                    index
                ],
                doc_id=doc_id,  # type: ignore[arg-type]
                title=title,  # type: ignore[arg-type]
                section=section_path[
                    -1
                ],
                section_path=section_path,
                snippet=snippet,  # type: ignore[arg-type]
                source_format=source_format,  # type: ignore[arg-type]
                distance=distance,
                similarity=(
                    1.0
                    - distance
                ),
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RetrievalError(
                "Failed to convert validated retrieval row "
                f"{index} into RetrievalResult."
            ) from exc

        results.append(
            result
        )

    return tuple(
        results
    )


def retrieve_policy(
    query: str,
    *,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: dict[str, str] | None = None,
) -> tuple[RetrievalResult, ...]:
    """Retrieve citation-ready policy results from the active index.

    This is the public single-query retrieval composition boundary.
    Request validation, query embedding, active-index access, Chroma
    execution, response validation, and result conversion remain owned
    by their existing lower-level retrieval stages.

    Args:
        query:
            Policy retrieval query.
        k:
            Positive number of nearest policy chunks requested.
        filters:
            Optional public retrieval filters.

    Returns:
        Citation-ready retrieval results in Chroma ranking order.
        A valid query with no matching records returns an empty tuple.

    Raises:
        TypeError:
            If the retrieval request violates the existing public type
            contract.
        ValueError:
            If the retrieval request violates the existing public value
            contract.
        RetrievalError:
            If embedding, index access, Chroma execution, raw-response
            validation, or result conversion fails.
    """

    raw_response = _query_policy_collection_raw(
        query,
        k=k,
        filters=filters,
    )

    rows = _validate_raw_retrieval_response(
        raw_response
    )

    return _build_retrieval_results(
        rows
    )


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the semantic-retrieval command-line interface.

    Command-line parsing and display formatting remain CLI concerns.
    Query validation, filter semantics, embedding, vector-store access,
    ranking, and retrieval-result construction remain owned by the
    existing retrieval pipeline.

    Args:
        argv:
            Optional argument list excluding the executable/module name.
            ``None`` delegates argument acquisition to ``argparse``.

    Returns:
        ``0`` when retrieval completes successfully and ``1`` for
        expected invalid-input or retrieval-runtime failures.
    """

    parser = _build_cli_parser()

    try:
        args = parser.parse_args(
            argv
        )

        filters = _build_cli_filters(
            args.filter
        )

        results = retrieve_policy(
            args.query,
            k=args.k,
            filters=filters,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if rank > 1:
                print()

            print(
                _format_cli_retrieval_result(
                    result,
                    rank,
                )
            )

    except (
        TypeError,
        ValueError,
        RetrievalError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0




if __name__ == "__main__":
    raise SystemExit(
        main()
    )
