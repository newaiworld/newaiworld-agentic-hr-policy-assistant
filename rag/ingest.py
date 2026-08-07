"""Policy-corpus manifest loading and ingestion orchestration.

This module is the entry point for the S4 ingestion pipeline.

The pipeline is implemented incrementally:

    corpus/version.json
        -> manifest validation
        -> source-file resolution
        -> Markdown/PDF parsing
        -> text normalisation
        -> heading-aware chunking
        -> canonical chunks.json
        -> embeddings
        -> Chroma index

The current implementation loads and validates only the corpus
manifest. Importing this module performs no file-system reads and
creates no generated artefacts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Final


SUPPORTED_SOURCE_FORMATS: Final[frozenset[str]] = frozenset(
    {"md", "pdf"}
)

SOURCE_DIRECTORIES: Final[dict[str, str]] = {
    "md": "policies_md",
    "pdf": "policies_pdf",
}

MANIFEST_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"version", "created", "documents"}
)

DOCUMENT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "doc_id",
        "title",
        "format",
        "doc_version",
        "effective_date",
    }
)


class ManifestValidationError(ValueError):
    """Raised when a corpus manifest cannot be safely accepted."""


class SourceResolutionError(ValueError):
    """Raised when manifest records and source files do not reconcile."""


def _require_non_empty_string(value: Any, *, field: str) -> str:
    """Return a trimmed string or raise a contextual validation error."""

    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"{field} must be a non-empty string."
        )

    return value.strip()


def _require_iso_date(value: Any, *, field: str) -> str:
    """Return a canonical ISO date or raise a validation error."""

    text = _require_non_empty_string(value, field=field)

    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ManifestValidationError(
            f"{field} must use ISO YYYY-MM-DD format; "
            f"received {text!r}."
        ) from exc

    canonical = parsed.isoformat()

    if text != canonical:
        raise ManifestValidationError(
            f"{field} must use canonical ISO YYYY-MM-DD format; "
            f"received {text!r}."
        )

    return canonical


def _validate_exact_fields(
    value: dict[str, Any],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    """Reject missing or undocumented fields deterministically."""

    actual = frozenset(value)

    missing = sorted(required - actual)
    unexpected = sorted(actual - required)

    problems: list[str] = []

    if missing:
        problems.append("missing: " + ", ".join(missing))

    if unexpected:
        problems.append("unexpected: " + ", ".join(unexpected))

    if problems:
        raise ManifestValidationError(
            f"{context} has invalid fields ({'; '.join(problems)})."
        )


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """Validated metadata for one document listed in version.json.

    This object contains only metadata present in the manifest.
    The resolved source path is added later during source discovery.
    """

    doc_id: str
    title: str
    source_format: str
    doc_version: str
    effective_date: str

    def __post_init__(self) -> None:
        """Validate and canonicalise manifest document metadata."""

        doc_id = _require_non_empty_string(
            self.doc_id,
            field="document.doc_id",
        )
        title = _require_non_empty_string(
            self.title,
            field=f"{doc_id}.title",
        )
        source_format = _require_non_empty_string(
            self.source_format,
            field=f"{doc_id}.format",
        ).lower()
        doc_version = _require_non_empty_string(
            self.doc_version,
            field=f"{doc_id}.doc_version",
        )
        effective_date = _require_iso_date(
            self.effective_date,
            field=f"{doc_id}.effective_date",
        )

        if source_format not in SUPPORTED_SOURCE_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_SOURCE_FORMATS))
            raise ManifestValidationError(
                f"{doc_id}.format must be one of {supported}; "
                f"received {self.source_format!r}."
            )

        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "doc_version", doc_version)
        object.__setattr__(
            self,
            "effective_date",
            effective_date,
        )


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Validated metadata for one complete policy-corpus version."""

    version: str
    created: str
    documents: tuple[ManifestDocument, ...]

    def __post_init__(self) -> None:
        """Reject an unusable or internally inconsistent manifest."""

        version = _require_non_empty_string(
            self.version,
            field="manifest.version",
        )
        created = _require_iso_date(
            self.created,
            field="manifest.created",
        )

        if not isinstance(self.documents, tuple):
            raise ManifestValidationError(
                "manifest.documents must be stored as a tuple."
            )

        if not self.documents:
            raise ManifestValidationError(
                "manifest.documents must contain at least one "
                "policy document."
            )

        if not all(
            isinstance(document, ManifestDocument)
            for document in self.documents
        ):
            raise ManifestValidationError(
                "manifest.documents must contain only "
                "ManifestDocument objects."
            )

        doc_ids = [document.doc_id for document in self.documents]

        duplicate_ids = sorted(
            doc_id
            for doc_id, count in Counter(doc_ids).items()
            if count > 1
        )

        if duplicate_ids:
            raise ManifestValidationError(
                "Duplicate document IDs are not allowed: "
                + ", ".join(duplicate_ids)
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "created", created)

    @property
    def document_count(self) -> int:
        """Return the number of policy documents in the manifest."""

        return len(self.documents)

    @property
    def format_counts(self) -> dict[str, int]:
        """Return deterministic document counts by source format."""

        counts = Counter(
            document.source_format for document in self.documents
        )

        return {
            source_format: counts[source_format]
            for source_format in sorted(counts)
        }


@dataclass(frozen=True, slots=True)
class ResolvedDocument:
    """Associate validated manifest metadata with one source file."""

    manifest: ManifestDocument
    source_path: Path

    def __post_init__(self) -> None:
        """Validate the resolved-document boundary."""

        if not isinstance(self.manifest, ManifestDocument):
            raise TypeError(
                "manifest must be a ManifestDocument instance."
            )

        if not isinstance(self.source_path, Path):
            raise TypeError(
                "source_path must be a pathlib.Path instance."
            )

        if not self.source_path.is_file():
            raise SourceResolutionError(
                "Resolved source path is not a regular file: "
                f"{self.source_path}"
            )

    @property
    def doc_id(self) -> str:
        """Return the manifest document ID."""

        return self.manifest.doc_id

    @property
    def title(self) -> str:
        """Return the manifest policy title."""

        return self.manifest.title

    @property
    def source_format(self) -> str:
        """Return the declared source format."""

        return self.manifest.source_format


def _discover_supported_sources(
    source_root: Path,
) -> dict[str, tuple[Path, ...]]:
    """Discover deterministic source candidates by declared format."""

    discovered: dict[str, tuple[Path, ...]] = {}

    for source_format in sorted(SOURCE_DIRECTORIES):
        directory = source_root / SOURCE_DIRECTORIES[source_format]

        if not directory.exists():
            raise SourceResolutionError(
                f"Source directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise SourceResolutionError(
                f"Source path is not a directory: {directory}"
            )

        suffix = f".{source_format}"

        candidates = tuple(
            sorted(
                (
                    candidate
                    for candidate in directory.iterdir()
                    if candidate.is_file()
                    and candidate.suffix.lower() == suffix
                ),
                key=lambda candidate: candidate.name,
            )
        )

        discovered[source_format] = candidates

    return discovered


def resolve_manifest_sources(
    manifest: CorpusManifest,
    source_root: Path,
) -> tuple[ResolvedDocument, ...]:
    """Resolve every manifest record to exactly one source document.

    Resolution uses the frozen source directories and the filename
    prefix ``<doc_id>-``. Supported source files not claimed by the
    manifest are rejected so that corpus drift cannot pass silently.

    Args:
        manifest:
            Validated corpus manifest.
        source_root:
            Path to ``corpus/source``.

    Returns:
        Resolved documents in manifest order.

    Raises:
        TypeError:
            If an argument has the wrong Python type.
        SourceResolutionError:
            If directories are unavailable, a manifest record has
            zero or multiple matching files, or supported source
            files remain unclaimed.
    """

    if not isinstance(manifest, CorpusManifest):
        raise TypeError(
            "manifest must be a CorpusManifest instance."
        )

    if not isinstance(source_root, Path):
        raise TypeError(
            "source_root must be a pathlib.Path instance."
        )

    if not source_root.exists():
        raise SourceResolutionError(
            f"Source root does not exist: {source_root}"
        )

    if not source_root.is_dir():
        raise SourceResolutionError(
            f"Source root is not a directory: {source_root}"
        )

    discovered = _discover_supported_sources(source_root)

    resolved_documents: list[ResolvedDocument] = []
    claimed_paths: set[Path] = set()

    for document in manifest.documents:
        candidates = discovered[document.source_format]
        filename_prefix = f"{document.doc_id}-"

        matches = tuple(
            candidate
            for candidate in candidates
            if candidate.name.startswith(filename_prefix)
        )

        if not matches:
            expected_directory = (
                source_root
                / SOURCE_DIRECTORIES[document.source_format]
            )
            raise SourceResolutionError(
                "No source file found for "
                f"{document.doc_id} in {expected_directory}; "
                f"expected exactly one "
                f"{filename_prefix}*.{document.source_format} file."
            )

        if len(matches) > 1:
            rendered_matches = ", ".join(
                str(match.relative_to(source_root))
                for match in matches
            )
            raise SourceResolutionError(
                "Multiple source files found for "
                f"{document.doc_id}: {rendered_matches}"
            )

        source_path = matches[0]
        claimed_paths.add(source_path)

        resolved_documents.append(
            ResolvedDocument(
                manifest=document,
                source_path=source_path,
            )
        )

    all_supported_paths = {
        candidate
        for candidates in discovered.values()
        for candidate in candidates
    }

    unexpected_paths = sorted(
        all_supported_paths - claimed_paths,
        key=lambda candidate: candidate.as_posix(),
    )

    if unexpected_paths:
        rendered_paths = ", ".join(
            str(path.relative_to(source_root))
            for path in unexpected_paths
        )
        raise SourceResolutionError(
            "Unexpected supported source files are not declared "
            f"in the manifest: {rendered_paths}"
        )

    return tuple(resolved_documents)


def _parse_manifest_document(
    value: Any,
    *,
    position: int,
) -> ManifestDocument:
    """Validate one raw document entry from the JSON manifest."""

    context = f"manifest.documents[{position}]"

    if not isinstance(value, dict):
        raise ManifestValidationError(
            f"{context} must be a JSON object."
        )

    _validate_exact_fields(
        value,
        required=DOCUMENT_REQUIRED_FIELDS,
        context=context,
    )

    return ManifestDocument(
        doc_id=value["doc_id"],
        title=value["title"],
        source_format=value["format"],
        doc_version=value["doc_version"],
        effective_date=value["effective_date"],
    )


def parse_manifest_data(value: Any) -> CorpusManifest:
    """Convert decoded JSON data into a validated corpus manifest.

    Args:
        value:
            Python value returned by ``json.loads``.

    Returns:
        A validated, immutable ``CorpusManifest``.

    Raises:
        ManifestValidationError:
            If the decoded JSON does not follow the frozen schema.
    """

    if not isinstance(value, dict):
        raise ManifestValidationError(
            "Manifest root must be a JSON object."
        )

    _validate_exact_fields(
        value,
        required=MANIFEST_REQUIRED_FIELDS,
        context="manifest",
    )

    raw_documents = value["documents"]

    if not isinstance(raw_documents, list):
        raise ManifestValidationError(
            "manifest.documents must be a JSON array."
        )

    documents = tuple(
        _parse_manifest_document(document, position=index)
        for index, document in enumerate(raw_documents)
    )

    return CorpusManifest(
        version=value["version"],
        created=value["created"],
        documents=documents,
    )


def load_manifest(path: Path) -> CorpusManifest:
    """Read and validate a UTF-8 corpus manifest from disk.

    Args:
        path:
            Path to ``corpus/version.json`` or an equivalent test
            fixture.

    Returns:
        A validated, immutable ``CorpusManifest``.

    Raises:
        ManifestValidationError:
            If the path is missing, unreadable, not valid UTF-8,
            contains invalid JSON, or violates the manifest schema.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path instance.")

    if not path.exists():
        raise ManifestValidationError(
            f"Manifest file does not exist: {path}"
        )

    if not path.is_file():
        raise ManifestValidationError(
            f"Manifest path is not a regular file: {path}"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(
            f"Manifest is not valid UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise ManifestValidationError(
            f"Manifest could not be read: {path}: {exc}"
        ) from exc

    try:
        raw_data = json.loads(text)
    except JSONDecodeError as exc:
        raise ManifestValidationError(
            f"Manifest contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    return parse_manifest_data(raw_data)
