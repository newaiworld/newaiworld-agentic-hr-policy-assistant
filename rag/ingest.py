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
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Final

import yaml
from pypdf import PdfReader
from pypdf.errors import PdfReadError


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

MARKDOWN_METADATA_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "doc_id",
        "title",
        "document_type",
        "version",
        "effective_date",
        "owner",
        "status",
        "applies_to",
        "keywords",
    }
)

ATX_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})[ \t]+"
    r"(?P<title>.*?)[ \t]*$"
)

FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$"
)

PDF_PAGE_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[1-9][0-9]*$"
)

PDF_INLINE_HASH_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<prefix>.*?)"
    r"(?:[ \t]+|^)"
    r"#{2,6}[ \t]+"
    r"(?P<heading>\d+(?:\.\d+)*\.?[ \t]*.*)$"
)

PDF_NUMBER_ONLY_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\.[ \t]*$"
)

PDF_NUMBERED_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)"
    r"(?P<major_dot>\.)?"
    r"[ \t]+"
    r"(?P<title>\S.*)$"
)

PDF_SYNTHETIC_NOTICE_LINES: Final[frozenset[str]] = frozenset(
    {
        (
            "Synthetic policy notice: This document was created for an "
            "educational agentic AI project. It is not"
        ),
        "legal advice and does not reproduce a real employer policy.",
    }
)


class ManifestValidationError(ValueError):
    """Raised when a corpus manifest cannot be safely accepted."""


class SourceResolutionError(ValueError):
    """Raised when manifest records and source files do not reconcile."""


class MarkdownParseError(ValueError):
    """Raised when a Markdown policy cannot be parsed safely."""


class PdfParseError(ValueError):
    """Raised when a PDF policy cannot be parsed safely."""


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


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One ordered, heading-aware section from a policy document."""

    doc_id: str
    title: str
    section_path: tuple[str, ...]
    section_order: int
    text: str
    source_format: str

    def __post_init__(self) -> None:
        """Validate the parser output boundary."""

        doc_id = _require_non_empty_string(
            self.doc_id,
            field="parsed_section.doc_id",
        )
        title = _require_non_empty_string(
            self.title,
            field=f"{doc_id}.parsed_section.title",
        )
        source_format = _require_non_empty_string(
            self.source_format,
            field=f"{doc_id}.parsed_section.source_format",
        ).lower()

        if source_format not in SUPPORTED_SOURCE_FORMATS:
            raise MarkdownParseError(
                f"{doc_id}.parsed_section.source_format is unsupported: "
                f"{source_format!r}"
            )

        if not isinstance(self.section_path, tuple):
            raise TypeError(
                "section_path must be stored as a tuple."
            )

        if not self.section_path:
            raise MarkdownParseError(
                "section_path must contain at least one heading."
            )

        if not all(
            isinstance(part, str) and part.strip()
            for part in self.section_path
        ):
            raise MarkdownParseError(
                "section_path must contain only non-empty strings."
            )

        if (
            not isinstance(self.section_order, int)
            or isinstance(self.section_order, bool)
            or self.section_order < 0
        ):
            raise MarkdownParseError(
                "section_order must be a non-negative integer."
            )

        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")

        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(
            self,
            "section_path",
            tuple(part.strip() for part in self.section_path),
        )
        object.__setattr__(self, "source_format", source_format)


def _split_markdown_front_matter(
    text: str,
    *,
    source_path: Path,
) -> tuple[str, str]:
    """Split strict YAML front matter from the Markdown body."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    lines = text.splitlines(keepends=True)

    if not lines or lines[0].strip() != "---":
        raise MarkdownParseError(
            f"{source_path}: YAML front matter must begin on line 1."
        )

    closing_index: int | None = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise MarkdownParseError(
            f"{source_path}: YAML front matter has no closing delimiter."
        )

    yaml_text = "".join(lines[1:closing_index])
    body_text = "".join(lines[closing_index + 1 :])

    if not yaml_text.strip():
        raise MarkdownParseError(
            f"{source_path}: YAML front matter is empty."
        )

    if not body_text.strip():
        raise MarkdownParseError(
            f"{source_path}: Markdown body is empty."
        )

    return yaml_text, body_text


def _require_markdown_string(
    metadata: dict[str, Any],
    *,
    field: str,
    source_path: Path,
) -> str:
    """Return one validated non-empty Markdown metadata string."""

    value = metadata[field]

    if not isinstance(value, str) or not value.strip():
        raise MarkdownParseError(
            f"{source_path}: front-matter field {field!r} "
            "must be a non-empty string."
        )

    return value.strip()


def _require_markdown_string_list(
    metadata: dict[str, Any],
    *,
    field: str,
    source_path: Path,
) -> tuple[str, ...]:
    """Return one validated Markdown metadata string sequence."""

    value = metadata[field]

    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(item, str) and item.strip()
            for item in value
        )
    ):
        raise MarkdownParseError(
            f"{source_path}: front-matter field {field!r} "
            "must be a non-empty list of non-empty strings."
        )

    return tuple(item.strip() for item in value)


def _parse_and_validate_markdown_metadata(
    yaml_text: str,
    *,
    resolved: ResolvedDocument,
) -> dict[str, Any]:
    """Parse YAML and validate its exact contract against the manifest."""

    try:
        raw_metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise MarkdownParseError(
            f"{resolved.source_path}: invalid YAML front matter: {exc}"
        ) from exc

    if not isinstance(raw_metadata, dict):
        raise MarkdownParseError(
            f"{resolved.source_path}: YAML front matter must "
            "decode to a mapping."
        )

    metadata: dict[str, Any] = dict(raw_metadata)

    try:
        _validate_exact_fields(
            metadata,
            required=MARKDOWN_METADATA_REQUIRED_FIELDS,
            context=f"{resolved.source_path} front matter",
        )
    except ManifestValidationError as exc:
        raise MarkdownParseError(str(exc)) from exc

    string_fields = (
        "doc_id",
        "title",
        "document_type",
        "version",
        "effective_date",
        "owner",
        "status",
    )

    validated_strings = {
        field: _require_markdown_string(
            metadata,
            field=field,
            source_path=resolved.source_path,
        )
        for field in string_fields
    }

    applies_to = _require_markdown_string_list(
        metadata,
        field="applies_to",
        source_path=resolved.source_path,
    )
    keywords = _require_markdown_string_list(
        metadata,
        field="keywords",
        source_path=resolved.source_path,
    )

    if validated_strings["document_type"] != "policy":
        raise MarkdownParseError(
            f"{resolved.source_path}: document_type must be 'policy'."
        )

    if validated_strings["status"] != "active":
        raise MarkdownParseError(
            f"{resolved.source_path}: status must be 'active'."
        )

    expected_values = {
        "doc_id": resolved.manifest.doc_id,
        "title": resolved.manifest.title,
        "version": resolved.manifest.doc_version,
        "effective_date": resolved.manifest.effective_date,
    }

    for field, expected in expected_values.items():
        actual = validated_strings[field]

        if actual != expected:
            raise MarkdownParseError(
                f"{resolved.source_path}: front-matter {field} "
                f"does not match the manifest; "
                f"manifest={expected!r}, front_matter={actual!r}."
            )

    return {
        **validated_strings,
        "applies_to": applies_to,
        "keywords": keywords,
    }


def _strip_optional_closing_hashes(title: str) -> str:
    """Remove Markdown's optional trailing ATX heading hashes."""

    return re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()


def _parse_markdown_body_sections(
    body_text: str,
    *,
    resolved: ResolvedDocument,
) -> tuple[ParsedSection, ...]:
    """Parse ordered ATX headings while preserving Markdown content."""

    lines = body_text.splitlines()
    sections: list[ParsedSection] = []
    heading_stack: list[str] = []
    seen_paths: set[tuple[str, ...]] = set()

    current_path: tuple[str, ...] | None = None
    current_lines: list[str] = []

    fence_character: str | None = None
    fence_length = 0

    def publish_current_section() -> None:
        if current_path is None:
            return

        section_text = "\n".join(current_lines).strip("\n")

        sections.append(
            ParsedSection(
                doc_id=resolved.doc_id,
                title=resolved.title,
                section_path=current_path,
                section_order=len(sections),
                text=section_text,
                source_format=resolved.source_format,
            )
        )

    for line_number, line in enumerate(lines, start=1):
        fence_match = FENCE_PATTERN.match(line)

        if fence_match:
            marker = fence_match.group("marker")
            marker_character = marker[0]

            if fence_character is None:
                fence_character = marker_character
                fence_length = len(marker)
            elif (
                marker_character == fence_character
                and len(marker) >= fence_length
                and not fence_match.group("info").strip()
            ):
                fence_character = None
                fence_length = 0

            if current_path is None:
                if line.strip():
                    raise MarkdownParseError(
                        f"{resolved.source_path}: content appears before "
                        "the first heading."
                    )
            else:
                current_lines.append(line)

            continue

        if fence_character is not None:
            if current_path is None:
                raise MarkdownParseError(
                    f"{resolved.source_path}: fenced content appears "
                    "before the first heading."
                )

            current_lines.append(line)
            continue

        heading_match = ATX_HEADING_PATTERN.match(line)

        if heading_match:
            level = len(heading_match.group("marks"))
            heading_title = _strip_optional_closing_hashes(
                heading_match.group("title")
            )

            if not heading_title:
                raise MarkdownParseError(
                    f"{resolved.source_path}: empty heading at Markdown "
                    f"body line {line_number}."
                )

            if not sections and current_path is None and level != 1:
                raise MarkdownParseError(
                    f"{resolved.source_path}: first heading must be "
                    "level 1."
                )

            if level > len(heading_stack) + 1:
                raise MarkdownParseError(
                    f"{resolved.source_path}: heading level jumps from "
                    f"{len(heading_stack)} to {level} at Markdown body "
                    f"line {line_number}."
                )

            publish_current_section()
            current_lines = []

            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading_title)
            current_path = tuple(heading_stack)

            if current_path in seen_paths:
                raise MarkdownParseError(
                    f"{resolved.source_path}: duplicate heading path: "
                    + " > ".join(current_path)
                )

            seen_paths.add(current_path)
            continue

        if current_path is None:
            if line.strip():
                raise MarkdownParseError(
                    f"{resolved.source_path}: content appears before "
                    "the first heading."
                )
            continue

        current_lines.append(line)

    if fence_character is not None:
        raise MarkdownParseError(
            f"{resolved.source_path}: unclosed fenced code block."
        )

    publish_current_section()

    if not sections:
        raise MarkdownParseError(
            f"{resolved.source_path}: no Markdown headings were found."
        )

    level_one_sections = [
        section
        for section in sections
        if len(section.section_path) == 1
    ]

    if len(level_one_sections) != 1:
        raise MarkdownParseError(
            f"{resolved.source_path}: expected exactly one level-1 "
            f"heading; found {len(level_one_sections)}."
        )

    document_heading = level_one_sections[0].section_path[0]

    if document_heading != resolved.title:
        raise MarkdownParseError(
            f"{resolved.source_path}: level-1 heading does not match "
            f"the manifest title; manifest={resolved.title!r}, "
            f"heading={document_heading!r}."
        )

    return tuple(sections)


def parse_markdown_document(
    resolved: ResolvedDocument,
) -> tuple[ParsedSection, ...]:
    """Parse one resolved Markdown policy into ordered sections.

    The function validates exact front matter against the authoritative
    manifest and preserves Markdown body text without normalization.

    Args:
        resolved:
            A source-resolved manifest document whose format is ``md``.

    Returns:
        Ordered, immutable parsed sections.

    Raises:
        TypeError:
            If ``resolved`` has the wrong Python type.
        MarkdownParseError:
            If the source format, encoding, metadata, heading hierarchy,
            or Markdown structure is invalid.
    """

    if not isinstance(resolved, ResolvedDocument):
        raise TypeError(
            "resolved must be a ResolvedDocument instance."
        )

    if resolved.source_format != "md":
        raise MarkdownParseError(
            f"{resolved.doc_id}: parse_markdown_document requires "
            f"source_format='md'; received {resolved.source_format!r}."
        )

    try:
        text = resolved.source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MarkdownParseError(
            f"{resolved.source_path}: Markdown source is not valid UTF-8."
        ) from exc
    except OSError as exc:
        raise MarkdownParseError(
            f"{resolved.source_path}: Markdown source could not be read: "
            f"{exc}"
        ) from exc

    yaml_text, body_text = _split_markdown_front_matter(
        text,
        source_path=resolved.source_path,
    )

    _parse_and_validate_markdown_metadata(
        yaml_text,
        resolved=resolved,
    )

    return _parse_markdown_body_sections(
        body_text,
        resolved=resolved,
    )


def _extract_pdf_pages(
    resolved: ResolvedDocument,
) -> tuple[tuple[str, ...], ...]:
    """Extract non-empty page lines from one resolved PDF."""

    try:
        reader = PdfReader(resolved.source_path)
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfParseError(
            f"{resolved.source_path}: PDF could not be opened: {exc}"
        ) from exc

    if not reader.pages:
        raise PdfParseError(
            f"{resolved.source_path}: PDF contains no pages."
        )

    extracted_pages: list[tuple[str, ...]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise PdfParseError(
                f"{resolved.source_path}: text extraction failed on "
                f"page {page_number}: {exc}"
            ) from exc

        page_lines = [
            line.strip()
            for line in page_text.splitlines()
            if line.strip()
        ]

        if not page_lines:
            raise PdfParseError(
                f"{resolved.source_path}: page {page_number} "
                "contains no extractable text."
            )

        if (
            PDF_PAGE_NUMBER_PATTERN.fullmatch(page_lines[-1])
            and page_lines[-1] == str(page_number)
        ):
            page_lines.pop()

        page_lines = [
            line
            for line in page_lines
            if line not in PDF_SYNTHETIC_NOTICE_LINES
        ]

        extracted_pages.append(tuple(page_lines))

    return tuple(extracted_pages)


def _split_pdf_embedded_headings(
    lines: tuple[str, ...],
) -> tuple[str, ...]:
    """Split numbered ``##`` headings embedded in extracted text."""

    reconstructed: list[str] = []

    for line in lines:
        match = PDF_INLINE_HASH_HEADING_PATTERN.match(line)

        if match is None:
            reconstructed.append(line)
            continue

        prefix = match.group("prefix").strip()
        heading = match.group("heading").strip()

        if prefix:
            reconstructed.append(prefix)

        if heading:
            reconstructed.append(heading)

    return tuple(reconstructed)


def _join_pdf_split_headings(
    lines: tuple[str, ...],
) -> tuple[str, ...]:
    """Join headings extracted as a number line plus a title line."""

    reconstructed: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        number_only = PDF_NUMBER_ONLY_HEADING_PATTERN.match(line)

        if (
            number_only is not None
            and index + 1 < len(lines)
        ):
            next_line = lines[index + 1].strip()

            if (
                next_line
                and PDF_NUMBERED_HEADING_PATTERN.match(next_line) is None
            ):
                reconstructed.append(
                    f"{number_only.group('number')}. {next_line}"
                )
                index += 2
                continue

        reconstructed.append(line)
        index += 1

    return tuple(reconstructed)


def _remove_pdf_cover_metadata(
    lines: tuple[str, ...],
    *,
    resolved: ResolvedDocument,
) -> tuple[str, ...]:
    """Remove the extracted title and cover metadata before section 1."""

    first_heading_index: int | None = None

    for index, line in enumerate(lines):
        match = PDF_NUMBERED_HEADING_PATTERN.match(line)

        if match is None:
            continue

        number = match.group("number")
        major_dot = match.group("major_dot")

        if number == "1" and major_dot == ".":
            first_heading_index = index
            break

    if first_heading_index is None:
        raise PdfParseError(
            f"{resolved.source_path}: section 1 heading was not found."
        )

    cover_lines = lines[:first_heading_index]

    if not any(
        line == f"Document ID: {resolved.doc_id}"
        for line in cover_lines
    ):
        raise PdfParseError(
            f"{resolved.source_path}: PDF document ID does not match "
            f"the manifest entry {resolved.doc_id!r}."
        )

    if not any(
        line == f"Version: {resolved.manifest.doc_version}"
        for line in cover_lines
    ):
        raise PdfParseError(
            f"{resolved.source_path}: PDF version does not match "
            "the manifest."
        )

    if not any(
        line == (
            "Effective date: "
            f"{resolved.manifest.effective_date}"
        )
        for line in cover_lines
    ):
        raise PdfParseError(
            f"{resolved.source_path}: PDF effective date does not match "
            "the manifest."
        )

    if resolved.title not in cover_lines:
        raise PdfParseError(
            f"{resolved.source_path}: PDF title does not match "
            f"the manifest title {resolved.title!r}."
        )

    return lines[first_heading_index:]


def _classify_pdf_heading(
    line: str,
) -> tuple[int, str] | None:
    """Return a PDF heading level and canonical heading text."""

    match = PDF_NUMBERED_HEADING_PATTERN.match(line)

    if match is None:
        return None

    number = match.group("number")
    major_dot = match.group("major_dot")
    title = match.group("title").strip()

    if "." not in number:
        if major_dot != ".":
            return None

        level = 1
        canonical = f"{number}. {title}"
    else:
        level = number.count(".") + 1
        canonical = f"{number} {title}"

    return level, canonical


def _parse_pdf_sections(
    lines: tuple[str, ...],
    *,
    resolved: ResolvedDocument,
) -> tuple[ParsedSection, ...]:
    """Convert cleaned PDF lines into ordered heading-aware sections."""

    sections: list[ParsedSection] = []
    heading_stack: list[str] = []
    seen_paths: set[tuple[str, ...]] = set()

    current_path: tuple[str, ...] | None = None
    current_lines: list[str] = []

    def publish_current_section() -> None:
        if current_path is None:
            return

        sections.append(
            ParsedSection(
                doc_id=resolved.doc_id,
                title=resolved.title,
                section_path=current_path,
                section_order=len(sections),
                text="\n".join(current_lines).strip("\n"),
                source_format="pdf",
            )
        )

    for line in lines:
        classified = _classify_pdf_heading(line)

        if classified is None:
            if current_path is None:
                if line.strip():
                    raise PdfParseError(
                        f"{resolved.source_path}: content appears before "
                        "the first PDF policy heading."
                    )
                continue

            current_lines.append(line)
            continue

        level, heading_text = classified

        if level > len(heading_stack) + 1:
            raise PdfParseError(
                f"{resolved.source_path}: PDF heading hierarchy jumps "
                f"to level {level} at {heading_text!r}."
            )

        publish_current_section()
        current_lines = []

        heading_stack = heading_stack[: level - 1]
        heading_stack.append(heading_text)

        current_path = (resolved.title, *heading_stack)

        if current_path in seen_paths:
            raise PdfParseError(
                f"{resolved.source_path}: duplicate PDF heading path: "
                + " > ".join(current_path)
            )

        seen_paths.add(current_path)

    publish_current_section()

    if not sections:
        raise PdfParseError(
            f"{resolved.source_path}: no PDF policy sections were found."
        )

    major_sections = [
        section.section_path[-1].split(".", 1)[0]
        for section in sections
        if len(section.section_path) == 2
    ]

    expected_major_sections = [
        str(number) for number in range(1, 13)
    ]

    if major_sections != expected_major_sections:
        raise PdfParseError(
            f"{resolved.source_path}: expected PDF major sections "
            f"1-12 in order; found {major_sections}."
        )

    return tuple(sections)


def parse_pdf_document(
    resolved: ResolvedDocument,
) -> tuple[ParsedSection, ...]:
    """Parse one resolved PDF policy into ordered sections.

    The parser removes page-number and synthetic-notice boilerplate,
    verifies cover metadata against the manifest, reconstructs numbered
    headings, and preserves extracted policy wording for later shared
    normalization.

    Args:
        resolved:
            A source-resolved manifest document whose format is ``pdf``.

    Returns:
        Ordered, immutable parsed sections.

    Raises:
        TypeError:
            If ``resolved`` has the wrong Python type.
        PdfParseError:
            If the PDF cannot be read, metadata does not reconcile,
            text extraction fails, or headings cannot be reconstructed.
    """

    if not isinstance(resolved, ResolvedDocument):
        raise TypeError(
            "resolved must be a ResolvedDocument instance."
        )

    if resolved.source_format != "pdf":
        raise PdfParseError(
            f"{resolved.doc_id}: parse_pdf_document requires "
            f"source_format='pdf'; received {resolved.source_format!r}."
        )

    pages = _extract_pdf_pages(resolved)

    flattened_lines = tuple(
        line
        for page_lines in pages
        for line in page_lines
    )

    split_lines = _split_pdf_embedded_headings(flattened_lines)
    reconstructed_lines = _join_pdf_split_headings(split_lines)

    policy_lines = _remove_pdf_cover_metadata(
        reconstructed_lines,
        resolved=resolved,
    )

    return _parse_pdf_sections(
        policy_lines,
        resolved=resolved,
    )


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
