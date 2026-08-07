"""Tests for deterministic policy-corpus ingestion and chunking.

The file currently tests the ingestion manifest contract.
Chunking tests will be added when rag/chunk.py is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag.ingest import (
    SOURCE_DIRECTORIES,
    SUPPORTED_SOURCE_FORMATS,
    CorpusManifest,
    ManifestDocument,
    ManifestValidationError,
    MarkdownParseError,
    ParsedSection,
    PdfParseError,
    ResolvedDocument,
    _classify_pdf_heading,
    _join_pdf_split_headings,
    _parse_pdf_sections,
    _remove_pdf_cover_metadata,
    _split_pdf_embedded_headings,
    SourceResolutionError,
    load_manifest,
    normalize_text,
    parse_manifest_data,
    parse_markdown_document,
    parse_pdf_document,
    resolve_manifest_sources,
)


def make_document(
    *,
    doc_id: str = "HR-POL-004",
    title: str = "Remote and Flexible Work Policy",
    source_format: str = "md",
    doc_version: str = "1.2",
    effective_date: str = "2026-01-01",
) -> ManifestDocument:
    """Create a valid manifest document for focused tests."""

    return ManifestDocument(
        doc_id=doc_id,
        title=title,
        source_format=source_format,
        doc_version=doc_version,
        effective_date=effective_date,
    )


def make_manifest_data() -> dict[str, Any]:
    """Return a valid raw manifest dictionary."""

    return {
        "version": "1.2",
        "created": "2026-08-05",
        "documents": [
            {
                "doc_id": "HR-POL-004",
                "title": "Remote and Flexible Work Policy",
                "format": "md",
                "doc_version": "1.2",
                "effective_date": "2026-01-01",
            }
        ],
    }


def write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON for a test fixture."""

    path.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )


def test_supported_source_formats_are_frozen() -> None:
    assert SUPPORTED_SOURCE_FORMATS == frozenset({"md", "pdf"})


def test_manifest_document_normalises_text_and_format() -> None:
    document = ManifestDocument(
        doc_id=" HR-POL-004 ",
        title=" Remote and Flexible Work Policy ",
        source_format="MD",
        doc_version=" 1.2 ",
        effective_date="2026-01-01",
    )

    assert document.doc_id == "HR-POL-004"
    assert document.title == "Remote and Flexible Work Policy"
    assert document.source_format == "md"
    assert document.doc_version == "1.2"
    assert document.effective_date == "2026-01-01"


@pytest.mark.parametrize("source_format", ["html", "txt", "docx", ""])
def test_manifest_document_rejects_unsupported_formats(
    source_format: str,
) -> None:
    with pytest.raises(
        ManifestValidationError,
        match="format",
    ):
        make_document(source_format=source_format)


@pytest.mark.parametrize(
    "effective_date",
    [
        "",
        "01-01-2026",
        "2026/01/01",
        "2026-02-30",
        "2026-01-01T00:00:00",
    ],
)
def test_manifest_document_rejects_invalid_effective_dates(
    effective_date: str,
) -> None:
    with pytest.raises(
        ManifestValidationError,
        match="effective_date",
    ):
        make_document(effective_date=effective_date)


def test_manifest_rejects_duplicate_document_ids() -> None:
    first = make_document()
    duplicate = make_document(title="Duplicate Policy Title")

    with pytest.raises(
        ManifestValidationError,
        match="Duplicate document IDs",
    ):
        CorpusManifest(
            version="1.2",
            created="2026-08-05",
            documents=(first, duplicate),
        )


def test_manifest_requires_at_least_one_document() -> None:
    with pytest.raises(
        ManifestValidationError,
        match="at least one policy document",
    ):
        CorpusManifest(
            version="1.2",
            created="2026-08-05",
            documents=(),
        )


def test_manifest_summary_is_deterministic() -> None:
    markdown_document = make_document()
    pdf_document = make_document(
        doc_id="HR-POL-003",
        title="Public Holidays Policy",
        source_format="pdf",
    )

    manifest = CorpusManifest(
        version=" 1.2 ",
        created="2026-08-05",
        documents=(markdown_document, pdf_document),
    )

    assert manifest.version == "1.2"
    assert manifest.created == "2026-08-05"
    assert manifest.document_count == 2
    assert manifest.format_counts == {"md": 1, "pdf": 1}


def test_parse_manifest_data_converts_valid_json_data() -> None:
    manifest = parse_manifest_data(make_manifest_data())

    assert manifest.version == "1.2"
    assert manifest.created == "2026-08-05"
    assert manifest.document_count == 1
    assert manifest.documents[0].doc_id == "HR-POL-004"
    assert manifest.documents[0].source_format == "md"


@pytest.mark.parametrize(
    "raw_value",
    [
        [],
        "manifest",
        123,
        None,
    ],
)
def test_parse_manifest_data_rejects_non_object_root(
    raw_value: Any,
) -> None:
    with pytest.raises(
        ManifestValidationError,
        match="root must be a JSON object",
    ):
        parse_manifest_data(raw_value)


def test_parse_manifest_data_rejects_missing_top_level_field() -> None:
    data = make_manifest_data()
    del data["created"]

    with pytest.raises(
        ManifestValidationError,
        match=r"missing: created",
    ):
        parse_manifest_data(data)


def test_parse_manifest_data_rejects_unexpected_top_level_field() -> None:
    data = make_manifest_data()
    data["timestamp"] = "2026-08-05T10:00:00Z"

    with pytest.raises(
        ManifestValidationError,
        match=r"unexpected: timestamp",
    ):
        parse_manifest_data(data)


def test_parse_manifest_data_requires_document_array() -> None:
    data = make_manifest_data()
    data["documents"] = {}

    with pytest.raises(
        ManifestValidationError,
        match="must be a JSON array",
    ):
        parse_manifest_data(data)


def test_parse_manifest_data_rejects_non_object_document() -> None:
    data = make_manifest_data()
    data["documents"] = ["HR-POL-004"]

    with pytest.raises(
        ManifestValidationError,
        match=r"documents\[0\] must be a JSON object",
    ):
        parse_manifest_data(data)


def test_parse_manifest_data_rejects_missing_document_field() -> None:
    data = make_manifest_data()
    del data["documents"][0]["title"]

    with pytest.raises(
        ManifestValidationError,
        match=r"missing: title",
    ):
        parse_manifest_data(data)


def test_parse_manifest_data_rejects_unexpected_document_field() -> None:
    data = make_manifest_data()
    data["documents"][0]["source_path"] = "policy.md"

    with pytest.raises(
        ManifestValidationError,
        match=r"unexpected: source_path",
    ):
        parse_manifest_data(data)


def test_load_manifest_reads_valid_utf8_json(tmp_path: Path) -> None:
    path = tmp_path / "version.json"
    write_json(path, make_manifest_data())

    manifest = load_manifest(path)

    assert manifest.version == "1.2"
    assert manifest.document_count == 1


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(
        ManifestValidationError,
        match="does not exist",
    ):
        load_manifest(path)


def test_load_manifest_rejects_directory_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ManifestValidationError,
        match="not a regular file",
    ):
        load_manifest(tmp_path)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "version.json"
    path.write_text('{"version": ', encoding="utf-8")

    with pytest.raises(
        ManifestValidationError,
        match=r"line 1, column",
    ):
        load_manifest(path)


def test_load_manifest_rejects_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "version.json"
    path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(
        ManifestValidationError,
        match="not valid UTF-8",
    ):
        load_manifest(path)


def test_real_project_manifest_loads_successfully() -> None:
    manifest = load_manifest(Path("corpus/version.json"))

    assert manifest.version == "1.2"
    assert manifest.created == "2026-08-05"
    assert manifest.document_count == 13
    assert manifest.format_counts == {"md": 9, "pdf": 4}

    assert tuple(
        document.doc_id for document in manifest.documents
    ) == tuple(
        f"HR-POL-{number:03d}" for number in range(1, 14)
    )


def make_source_tree(
    root: Path,
    *,
    markdown_names: tuple[str, ...] = (
        "HR-POL-004-remote-and-flexible-work.md",
    ),
    pdf_names: tuple[str, ...] = (),
) -> Path:
    """Create a minimal deterministic corpus source tree."""

    source_root = root / "source"
    markdown_directory = source_root / "policies_md"
    pdf_directory = source_root / "policies_pdf"

    markdown_directory.mkdir(parents=True)
    pdf_directory.mkdir(parents=True)

    for name in markdown_names:
        (markdown_directory / name).write_text(
            "# Test policy\n",
            encoding="utf-8",
        )

    for name in pdf_names:
        (pdf_directory / name).write_bytes(b"%PDF-test")

    return source_root


def test_resolve_manifest_sources_resolves_one_matching_file(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(tmp_path)

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    resolved = resolve_manifest_sources(manifest, source_root)

    assert len(resolved) == 1
    assert resolved[0].manifest is manifest.documents[0]
    assert resolved[0].doc_id == "HR-POL-004"
    assert resolved[0].source_format == "md"
    assert resolved[0].source_path == (
        source_root
        / "policies_md"
        / "HR-POL-004-remote-and-flexible-work.md"
    )


def test_resolve_manifest_sources_preserves_manifest_order(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(
        tmp_path,
        markdown_names=(
            "HR-POL-004-remote-and-flexible-work.md",
            "HR-POL-001-employee-handbook.md",
        ),
    )

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(
            make_document(
                doc_id="HR-POL-004",
            ),
            make_document(
                doc_id="HR-POL-001",
                title="Employee Handbook",
            ),
        ),
    )

    resolved = resolve_manifest_sources(manifest, source_root)

    assert tuple(item.doc_id for item in resolved) == (
        "HR-POL-004",
        "HR-POL-001",
    )


def test_resolve_manifest_sources_rejects_missing_source_root(
    tmp_path: Path,
) -> None:
    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    with pytest.raises(
        SourceResolutionError,
        match="Source root does not exist",
    ):
        resolve_manifest_sources(
            manifest,
            tmp_path / "missing-source",
        )


def test_resolve_manifest_sources_rejects_missing_format_directory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "policies_md").mkdir(parents=True)

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    with pytest.raises(
        SourceResolutionError,
        match="Source directory does not exist",
    ):
        resolve_manifest_sources(manifest, source_root)


def test_resolve_manifest_sources_rejects_missing_document(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(
        tmp_path,
        markdown_names=(),
    )

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    with pytest.raises(
        SourceResolutionError,
        match="No source file found for HR-POL-004",
    ):
        resolve_manifest_sources(manifest, source_root)


def test_resolve_manifest_sources_rejects_duplicate_matches(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(
        tmp_path,
        markdown_names=(
            "HR-POL-004-remote-work.md",
            "HR-POL-004-flexible-work.md",
        ),
    )

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    with pytest.raises(
        SourceResolutionError,
        match="Multiple source files found for HR-POL-004",
    ):
        resolve_manifest_sources(manifest, source_root)


def test_resolve_manifest_sources_rejects_unexpected_supported_file(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(
        tmp_path,
        markdown_names=(
            "HR-POL-004-remote-and-flexible-work.md",
            "HR-POL-099-unexpected-policy.md",
        ),
    )

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    with pytest.raises(
        SourceResolutionError,
        match="HR-POL-099-unexpected-policy.md",
    ):
        resolve_manifest_sources(manifest, source_root)


def test_resolve_manifest_sources_ignores_unrelated_hidden_metadata(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(tmp_path)
    (source_root / ".DS_Store").write_bytes(b"macOS metadata")

    manifest = CorpusManifest(
        version="1.2",
        created="2026-08-05",
        documents=(make_document(),),
    )

    resolved = resolve_manifest_sources(manifest, source_root)

    assert len(resolved) == 1


def test_real_project_sources_reconcile_with_manifest() -> None:
    manifest = load_manifest(Path("corpus/version.json"))

    resolved = resolve_manifest_sources(
        manifest,
        Path("corpus/source"),
    )

    assert len(resolved) == 13
    assert tuple(item.doc_id for item in resolved) == tuple(
        f"HR-POL-{number:03d}" for number in range(1, 14)
    )
    assert sum(
        item.source_format == "md" for item in resolved
    ) == 9
    assert sum(
        item.source_format == "pdf" for item in resolved
    ) == 4


def make_markdown_resolved_document(
    tmp_path: Path,
    *,
    body: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> ResolvedDocument:
    """Create one resolved Markdown policy test fixture."""

    metadata: dict[str, Any] = {
        "doc_id": "HR-POL-004",
        "title": "Remote and Flexible Work Policy",
        "document_type": "policy",
        "version": "1.2",
        "effective_date": "2026-01-01",
        "owner": "People and Culture",
        "status": "active",
        "applies_to": ["full_time", "part_time"],
        "keywords": ["remote work", "international remote work"],
    }

    if metadata_overrides:
        metadata.update(metadata_overrides)

    if body is None:
        body = """# Remote and Flexible Work Policy

**Company:** Promote Health Analytics Pty Ltd

## 1. Purpose

This policy defines remote-work requirements.

## 4. Policy Requirements

### 4.4 International duration limit

- International remote work is limited.
- Approval is required.

## 12. Version History

| Version | Date | Change |
|---|---|---|
| 1.2 | 2026-01-01 | Updated |
"""

    source_root = make_source_tree(
        tmp_path,
        markdown_names=(),
    )
    source_path = (
        source_root
        / "policies_md"
        / "HR-POL-004-remote-and-flexible-work.md"
    )

    source_path.write_text(
        "---\n"
        + yaml.safe_dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
        )
        + "---\n\n"
        + body,
        encoding="utf-8",
    )

    return ResolvedDocument(
        manifest=make_document(),
        source_path=source_path,
    )


def test_parse_markdown_document_emits_ordered_heading_paths(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    sections = parse_markdown_document(resolved)

    assert tuple(section.section_order for section in sections) == tuple(
        range(len(sections))
    )
    assert tuple(section.section_path for section in sections) == (
        ("Remote and Flexible Work Policy",),
        ("Remote and Flexible Work Policy", "1. Purpose"),
        (
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
        ),
        (
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        ("Remote and Flexible Work Policy", "12. Version History"),
    )


def test_parse_markdown_document_preserves_markdown_content(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    sections = parse_markdown_document(resolved)

    duration_section = next(
        section
        for section in sections
        if section.section_path[-1]
        == "4.4 International duration limit"
    )
    version_section = next(
        section
        for section in sections
        if section.section_path[-1] == "12. Version History"
    )

    assert duration_section.text == (
        "- International remote work is limited.\n"
        "- Approval is required."
    )
    assert "| Version | Date | Change |" in version_section.text
    assert "|---|---|---|" in version_section.text


def test_parse_markdown_document_allows_empty_parent_sections(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    sections = parse_markdown_document(resolved)

    parent = next(
        section
        for section in sections
        if section.section_path[-1] == "4. Policy Requirements"
    )

    assert parent.text == ""


def test_parse_markdown_document_rejects_manifest_metadata_mismatch(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        metadata_overrides={"version": "9.9"},
    )

    with pytest.raises(
        MarkdownParseError,
        match="does not match the manifest",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_missing_metadata_field(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    text = resolved.source_path.read_text(encoding="utf-8")
    text = text.replace(
        "owner: People and Culture\n",
        "",
        1,
    )
    resolved.source_path.write_text(text, encoding="utf-8")

    with pytest.raises(
        MarkdownParseError,
        match="missing: owner",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_invalid_yaml(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    text = resolved.source_path.read_text(encoding="utf-8")
    text = text.replace(
        "keywords:\n",
        "keywords: [\n",
        1,
    )
    resolved.source_path.write_text(text, encoding="utf-8")

    with pytest.raises(
        MarkdownParseError,
        match="invalid YAML front matter",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_invalid_document_type(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        metadata_overrides={"document_type": "procedure"},
    )

    with pytest.raises(
        MarkdownParseError,
        match="document_type must be 'policy'",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_inactive_status(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        metadata_overrides={"status": "draft"},
    )

    with pytest.raises(
        MarkdownParseError,
        match="status must be 'active'",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_heading_level_jump(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body="""# Remote and Flexible Work Policy

### 3.1 Invalid jump

Text.
""",
    )

    with pytest.raises(
        MarkdownParseError,
        match="heading level jumps",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_duplicate_heading_path(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body="""# Remote and Flexible Work Policy

## 1. Purpose

First.

## 1. Purpose

Second.
""",
    )

    with pytest.raises(
        MarkdownParseError,
        match="duplicate heading path",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_ignores_headings_inside_fences(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body=(
            "# Remote and Flexible Work Policy\n\n"
            "## 1. Purpose\n\n"
            "```text\n"
            "### This is not a policy heading\n"
            "```\n\n"
            "Final sentence.\n"
        ),
    )

    sections = parse_markdown_document(resolved)

    assert len(sections) == 2
    assert "### This is not a policy heading" in sections[1].text
    assert sections[1].text.endswith("Final sentence.")


def test_parse_markdown_document_rejects_unclosed_fence(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body=(
            "# Remote and Flexible Work Policy\n\n"
            "## 1. Purpose\n\n"
            "```text\n"
            "Unclosed.\n"
        ),
    )

    with pytest.raises(
        MarkdownParseError,
        match="unclosed fenced code block",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_content_before_first_heading(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body="""Introductory content.

# Remote and Flexible Work Policy

## 1. Purpose

Text.
""",
    )

    with pytest.raises(
        MarkdownParseError,
        match="content appears before the first heading",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_title_heading_mismatch(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(
        tmp_path,
        body="""# Different Policy Title

## 1. Purpose

Text.
""",
    )

    with pytest.raises(
        MarkdownParseError,
        match="level-1 heading does not match",
    ):
        parse_markdown_document(resolved)


def test_parse_markdown_document_rejects_wrong_source_format(
    tmp_path: Path,
) -> None:
    source_root = make_source_tree(
        tmp_path,
        markdown_names=(),
        pdf_names=("HR-POL-003-public-holidays.pdf",),
    )

    resolved = ResolvedDocument(
        manifest=make_document(
            doc_id="HR-POL-003",
            title="Public Holidays Policy",
            source_format="pdf",
        ),
        source_path=(
            source_root
            / "policies_pdf"
            / "HR-POL-003-public-holidays.pdf"
        ),
    )

    with pytest.raises(
        MarkdownParseError,
        match="requires source_format='md'",
    ):
        parse_markdown_document(resolved)


def test_real_markdown_corpus_parses_in_manifest_order() -> None:
    manifest = load_manifest(Path("corpus/version.json"))
    resolved_documents = resolve_manifest_sources(
        manifest,
        Path("corpus/source"),
    )

    markdown_documents = tuple(
        document
        for document in resolved_documents
        if document.source_format == "md"
    )

    parsed_documents = tuple(
        parse_markdown_document(document)
        for document in markdown_documents
    )

    assert len(parsed_documents) == 9
    assert all(sections for sections in parsed_documents)

    assert tuple(
        sections[0].doc_id for sections in parsed_documents
    ) == (
        "HR-POL-001",
        "HR-POL-002",
        "HR-POL-004",
        "HR-POL-005",
        "HR-POL-007",
        "HR-POL-008",
        "HR-POL-010",
        "HR-POL-011",
        "HR-POL-013",
    )

    assert all(
        sections[0].section_path == (sections[0].title,)
        for sections in parsed_documents
    )

    assert all(
        section.source_format == "md"
        for sections in parsed_documents
        for section in sections
    )

    assert all(
        tuple(section.section_order for section in sections)
        == tuple(range(len(sections)))
        for sections in parsed_documents
    )


def make_pdf_resolved_document(
    tmp_path: Path,
) -> ResolvedDocument:
    """Create a resolved PDF metadata fixture."""

    source_root = make_source_tree(
        tmp_path,
        markdown_names=(),
        pdf_names=("HR-POL-003-public-holidays.pdf",),
    )

    return ResolvedDocument(
        manifest=make_document(
            doc_id="HR-POL-003",
            title="Public Holidays Policy",
            source_format="pdf",
        ),
        source_path=(
            source_root
            / "policies_pdf"
            / "HR-POL-003-public-holidays.pdf"
        ),
    )


def test_split_pdf_embedded_headings_preserves_prefix() -> None:
    lines = (
        "Applies to: full_time, part_time ## 1. Purpose",
        "Policy wording.",
        (
            "An exception does not guarantee approval. "
            "## 7. Responsibilities"
        ),
    )

    reconstructed = _split_pdf_embedded_headings(lines)

    assert reconstructed == (
        "Applies to: full_time, part_time",
        "1. Purpose",
        "Policy wording.",
        "An exception does not guarantee approval.",
        "7. Responsibilities",
    )


def test_join_pdf_split_headings_joins_number_and_title() -> None:
    lines = (
        "11. Related Documents",
        "Policy reference.",
        "12.",
        "Version History",
        "Version Effective date Summary",
    )

    reconstructed = _join_pdf_split_headings(lines)

    assert reconstructed == (
        "11. Related Documents",
        "Policy reference.",
        "12. Version History",
        "Version Effective date Summary",
    )


def test_classify_pdf_heading_handles_major_and_subsection() -> None:
    assert _classify_pdf_heading("3. Definitions") == (
        1,
        "3. Definitions",
    )
    assert _classify_pdf_heading("3.1 Assigned work location") == (
        2,
        "3.1 Assigned work location",
    )
    assert _classify_pdf_heading("Ordinary sentence.") is None


def test_remove_pdf_cover_metadata_validates_manifest(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "Public Holidays Policy",
        "Promote Health Analytics Pty Ltd",
        "Public Holidays Policy",
        "Document ID: HR-POL-003",
        "Company: Promote Health Analytics Pty Ltd",
        "Version: 1.2",
        "Effective date: 2026-01-01",
        "Owner: People and Culture",
        "Status: active",
        "Applies to: full_time, part_time",
        "1. Purpose",
        "Policy purpose.",
    )

    policy_lines = _remove_pdf_cover_metadata(
        lines,
        resolved=resolved,
    )

    assert policy_lines == (
        "1. Purpose",
        "Policy purpose.",
    )


def test_parse_pdf_sections_emits_deterministic_paths(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "1. Purpose",
        "Purpose text.",
        "2. Scope",
        "Scope text.",
        "3. Definitions",
        "3.1 Public holiday",
        "Definition text.",
        "4. Policy Requirements",
        "4.1 Location-based observance",
        "Requirement text.",
        "5. Procedures or Application",
        "Procedure text.",
        "6. Exceptions and Escalation",
        "Exception text.",
        "7. Responsibilities",
        "7.1 Employees",
        "Responsibility text.",
        "8. Decision Rules",
        "Decision text.",
        "9. Examples",
        "Example text.",
        "10. Frequently Asked Questions",
        "10.1 Which calendar applies?",
        "FAQ text.",
        "11. Related Documents",
        "Related text.",
        "12. Version History",
        "Version text.",
    )

    sections = _parse_pdf_sections(
        lines,
        resolved=resolved,
    )

    assert tuple(section.section_order for section in sections) == tuple(
        range(len(sections))
    )

    assert sections[0].section_path == (
        "Public Holidays Policy",
        "1. Purpose",
    )

    definition = next(
        section
        for section in sections
        if section.section_path[-1] == "3.1 Public holiday"
    )

    assert definition.section_path == (
        "Public Holidays Policy",
        "3. Definitions",
        "3.1 Public holiday",
    )
    assert definition.text == "Definition text."
    assert all(section.source_format == "pdf" for section in sections)


def test_parse_pdf_document_rejects_wrong_source_format(
    tmp_path: Path,
) -> None:
    resolved = make_markdown_resolved_document(tmp_path)

    with pytest.raises(
        PdfParseError,
        match="requires source_format='pdf'",
    ):
        parse_pdf_document(resolved)


def test_remove_pdf_cover_metadata_rejects_missing_section_one(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "Public Holidays Policy",
        "Document ID: HR-POL-003",
        "Version: 1.2",
        "Effective date: 2026-01-01",
        "2. Scope",
        "Scope text.",
    )

    with pytest.raises(
        PdfParseError,
        match="section 1 heading was not found",
    ):
        _remove_pdf_cover_metadata(
            lines,
            resolved=resolved,
        )


def test_remove_pdf_cover_metadata_rejects_document_id_mismatch(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "Public Holidays Policy",
        "Document ID: HR-POL-999",
        "Version: 1.2",
        "Effective date: 2026-01-01",
        "1. Purpose",
        "Purpose text.",
    )

    with pytest.raises(
        PdfParseError,
        match="document ID does not match",
    ):
        _remove_pdf_cover_metadata(
            lines,
            resolved=resolved,
        )


def test_remove_pdf_cover_metadata_rejects_version_mismatch(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "Public Holidays Policy",
        "Document ID: HR-POL-003",
        "Version: 9.9",
        "Effective date: 2026-01-01",
        "1. Purpose",
        "Purpose text.",
    )

    with pytest.raises(
        PdfParseError,
        match="PDF version does not match",
    ):
        _remove_pdf_cover_metadata(
            lines,
            resolved=resolved,
        )


def test_remove_pdf_cover_metadata_rejects_effective_date_mismatch(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "Public Holidays Policy",
        "Document ID: HR-POL-003",
        "Version: 1.2",
        "Effective date: 2030-01-01",
        "1. Purpose",
        "Purpose text.",
    )

    with pytest.raises(
        PdfParseError,
        match="effective date does not match",
    ):
        _remove_pdf_cover_metadata(
            lines,
            resolved=resolved,
        )


def test_parse_pdf_sections_rejects_heading_level_jump(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "1. Purpose",
        "Purpose text.",
        "1.1.1 Invalid nested heading",
        "Nested text.",
    )

    with pytest.raises(
        PdfParseError,
        match="heading hierarchy jumps",
    ):
        _parse_pdf_sections(
            lines,
            resolved=resolved,
        )


def test_parse_pdf_sections_rejects_missing_major_sections(
    tmp_path: Path,
) -> None:
    resolved = make_pdf_resolved_document(tmp_path)

    lines = (
        "1. Purpose",
        "Purpose text.",
        "2. Scope",
        "Scope text.",
    )

    with pytest.raises(
        PdfParseError,
        match="expected PDF major sections 1-12 in order",
    ):
        _parse_pdf_sections(
            lines,
            resolved=resolved,
        )


def test_real_pdf_corpus_parses_in_manifest_order() -> None:
    manifest = load_manifest(Path("corpus/version.json"))
    resolved_documents = resolve_manifest_sources(
        manifest,
        Path("corpus/source"),
    )

    pdf_documents = tuple(
        document
        for document in resolved_documents
        if document.source_format == "pdf"
    )

    parsed_documents = tuple(
        parse_pdf_document(document)
        for document in pdf_documents
    )

    assert len(parsed_documents) == 4
    assert all(sections for sections in parsed_documents)

    assert tuple(
        sections[0].doc_id for sections in parsed_documents
    ) == (
        "HR-POL-003",
        "HR-POL-006",
        "HR-POL-009",
        "HR-POL-012",
    )

    assert all(
        tuple(section.section_order for section in sections)
        == tuple(range(len(sections)))
        for sections in parsed_documents
    )

    assert all(
        section.source_format == "pdf"
        for sections in parsed_documents
        for section in sections
    )

    for sections in parsed_documents:
        major_headings = tuple(
            section.section_path[-1]
            for section in sections
            if len(section.section_path) == 2
        )

        assert tuple(
            heading.split(".", 1)[0]
            for heading in major_headings
        ) == tuple(str(number) for number in range(1, 13))

        assert any(
            section.section_path[-1].startswith("3.")
            and len(section.section_path) == 3
            for section in sections
        )

        assert any(
            section.section_path[-1].startswith("10.")
            and len(section.section_path) == 3
            for section in sections
        )



@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        3.14,
        [],
        {},
        b"policy",
    ],
)
def test_normalize_text_rejects_non_string_input(
    value: Any,
) -> None:
    """Reject unsupported input types with a clean API error."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        normalize_text(value)


def test_normalize_text_preserves_empty_string() -> None:
    """An empty parsed section remains deterministically empty."""

    assert normalize_text("") == ""


def test_normalize_text_normalizes_whitespace_only_input() -> None:
    """Whitespace-only content has no policy meaning."""

    assert normalize_text("   \n\t\n") == ""



def test_normalize_text_normalizes_line_endings() -> None:
    """Canonicalize CRLF and bare CR line endings to LF."""

    source = "First line.\r\nSecond line.\rThird line."

    assert normalize_text(source) == (
        "First line.\n"
        "Second line.\n"
        "Third line."
    )


def test_normalize_text_handles_nbsp_and_soft_hyphen() -> None:
    """Canonicalize non-breaking spaces and remove soft hyphens."""

    source = "Policy\u00a0rule: inter\u00adnational work."

    assert normalize_text(source) == (
        "Policy rule: international work."
    )


def test_normalize_text_applies_nfkc() -> None:
    """Normalize Unicode compatibility characters deterministically."""

    source = "Ｐｏｌｉｃｙ １２３"

    assert normalize_text(source) == "Policy 123"



def test_normalize_text_collapses_internal_horizontal_whitespace() -> None:
    """Collapse repeated spaces and tabs inside textual content."""

    source = (
        "Approval   is\t\t  required.\n"
        "Single spacing remains."
    )

    assert normalize_text(source) == (
        "Approval is required.\n"
        "Single spacing remains."
    )


def test_normalize_text_removes_trailing_horizontal_whitespace() -> None:
    """Remove trailing spaces and tabs without changing line content."""

    source = (
        "First line.   \n"
        "Second line.\t\n"
        "Third line."
    )

    assert normalize_text(source) == (
        "First line.\n"
        "Second line.\n"
        "Third line."
    )


def test_normalize_text_collapses_excess_blank_lines() -> None:
    """Preserve paragraph breaks while removing excessive blank lines."""

    source = (
        "\n\n"
        "First paragraph."
        "\n\n\n\n"
        "Second paragraph."
        "\n\n"
    )

    assert normalize_text(source) == (
        "First paragraph.\n\n"
        "Second paragraph."
    )



def test_normalize_text_preserves_nested_list_indentation() -> None:
    """Preserve leading indentation used by nested Markdown lists."""

    source = (
        "- First item\n"
        "  - Nested item\n"
        "    - Deeper item"
    )

    assert normalize_text(source) == source


def test_normalize_text_preserves_markdown_table_structure() -> None:
    """Preserve Markdown table delimiters and row structure."""

    source = (
        "| Rule | Outcome |\n"
        "|---|---|\n"
        "| Eligible | Yes |"
    )

    assert normalize_text(source) == source



def test_normalize_text_does_not_guess_split_word_repairs() -> None:
    """Do not perform speculative lexical repair during normalization."""

    source = (
        "Temporary is correct. "
        "T emporary remains diagnostic evidence. "
        "T eams also remains unchanged. "
        "A manager remains valid English."
    )

    assert normalize_text(source) == source



def test_normalize_text_is_idempotent() -> None:
    """Applying normalization twice must equal applying it once."""

    source = (
        "\n"
        "Ｐｏｌｉｃｙ\u00a0rule   applies.\r\n"
        "\r\n"
        "\r\n"
        "Second\u00ad paragraph.   "
        "\n"
    )

    once = normalize_text(source)
    twice = normalize_text(once)

    assert twice == once
