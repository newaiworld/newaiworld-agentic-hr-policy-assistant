"""Tests for deterministic policy-corpus ingestion and chunking.

This module covers manifest/source resolution, Markdown/PDF parsing,
normalization, exact token counting, heading-aware chunking, long-section
fallbacks, and bounded semantic overlap.

Chunking tests intentionally use deterministic synthetic policy fixtures so
boundary behavior can be verified independently of the current real corpus,
whose sections are all below the long-section threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag.chunk import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_OVERLAP_TOKENS,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_OVERLAP_TOKENS,
    TARGET_CHUNK_TOKENS,
    Chunk,
    _line_cut_positions,
    _paragraph_cut_positions,
    _select_best_boundary_cut,
    _select_overlap_start,
    _select_semantic_cut_position,
    _sentence_cut_positions,
    _split_at_token_boundary,
    _split_into_lines,
    _split_into_paragraphs,
    _split_into_sentences,
    chunk_section,
    count_tokens,
    get_tokenizer,
    split_long_section,
    CHUNK_ID_DIGEST_LENGTH,
    generate_chunk_id,
)
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
    SourceResolutionError,
    _classify_pdf_heading,
    _join_pdf_split_headings,
    _parse_pdf_sections,
    _remove_pdf_cover_metadata,
    _split_pdf_embedded_headings,
    load_manifest,
    normalize_section,
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

def make_chunk_test_section(
    text: str,
    *,
    doc_id: str = "TEST-LONG-001",
    title: str = "Synthetic Long Policy",
    section_path: tuple[str, ...] = (
        "Synthetic Long Policy",
        "1. Long Section",
    ),
    section_order: int = 0,
    source_format: str = "md",
) -> ParsedSection:
    """Build a deterministic ParsedSection for chunking tests."""

    return ParsedSection(
        doc_id=doc_id,
        title=title,
        section_path=section_path,
        section_order=section_order,
        text=text,
        source_format=source_format,
    )


def make_policy_sentence(index: int) -> str:
    """Return deterministic policy-like prose for chunk fixtures."""

    return (
        f"Policy rule {index} requires employees to obtain manager "
        "approval before changing the agreed work arrangement, "
        "maintain required security controls, record the approved "
        "location accurately, and escalate any exception to People "
        "and Culture before proceeding."
    )


def make_policy_paragraph(
    start_index: int,
    sentence_count: int,
) -> str:
    """Build one deterministic multi-sentence policy paragraph."""

    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index < 1
    ):
        raise ValueError(
            "start_index must be a positive integer."
        )

    if (
        not isinstance(sentence_count, int)
        or isinstance(sentence_count, bool)
        or sentence_count < 1
    ):
        raise ValueError(
            "sentence_count must be a positive integer."
        )

    return " ".join(
        make_policy_sentence(start_index + offset)
        for offset in range(sentence_count)
    )

BOUNDARY_LONG_TEXT = "\n\n".join(
    [
        make_policy_paragraph(1, 3),
        make_policy_paragraph(4, 3),
        make_policy_paragraph(7, 4),
    ]
)


NORMAL_LONG_TEXT = "\n\n".join(
    [
        make_policy_paragraph(1, 4),
        make_policy_paragraph(5, 4),
        make_policy_paragraph(9, 4),
        make_policy_paragraph(13, 3),
        make_policy_paragraph(16, 3),
    ]
)


STRUCTURED_LONG_TEXT = "\n\n".join(
    [
        make_policy_paragraph(1, 6),
        "\n".join(
            [
                (
                    f"- Requirement {index}: employees must confirm "
                    "manager approval, approved work location, "
                    "security controls, and escalation requirements "
                    "before proceeding."
                )
                for index in range(1, 13)
            ]
        ),
        make_policy_paragraph(20, 6),
        make_policy_paragraph(30, 6),
    ]
)


EXTREME_LONG_TEXT = "\n\n".join(
    make_policy_paragraph(index * 6 + 1, 6)
    for index in range(10)
)


GIANT_PARAGRAPH_TEXT = make_policy_paragraph(
    start_index=1,
    sentence_count=20,
)


GIANT_SENTENCE_TEXT = (
    "The policy requires "
    + " ".join(
        f"control{index}"
        for index in range(1, 650)
    )
    + "."
)


MANY_TINY_PARAGRAPHS_TEXT = "\n\n".join(
    (
        f"Rule {index} requires approval before the employee "
        "proceeds."
    )
    for index in range(1, 51)
)

def test_long_chunking_fixtures_match_expected_token_shapes() -> None:
    """Keep synthetic long-section fixtures stable and meaningful."""

    assert (
        TARGET_CHUNK_TOKENS
        < count_tokens(BOUNDARY_LONG_TEXT)
        <= MAX_CHUNK_TOKENS
    )

    assert 600 <= count_tokens(NORMAL_LONG_TEXT) <= 850

    assert 800 <= count_tokens(STRUCTURED_LONG_TEXT) <= 1200

    assert count_tokens(EXTREME_LONG_TEXT) >= 1500

    assert count_tokens(GIANT_PARAGRAPH_TEXT) > MAX_CHUNK_TOKENS

    assert count_tokens(GIANT_SENTENCE_TEXT) > MAX_CHUNK_TOKENS

    assert (
        count_tokens(MANY_TINY_PARAGRAPHS_TEXT)
        > TARGET_CHUNK_TOKENS
    )

    assert MANY_TINY_PARAGRAPHS_TEXT.count("\n\n") == 49

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

def test_split_into_paragraphs_preserves_normal_long_structure() -> None:
    """Split canonical blank-line paragraphs without changing order."""

    paragraphs = _split_into_paragraphs(
        NORMAL_LONG_TEXT
    )

    assert len(paragraphs) == 5

    assert [
        count_tokens(paragraph)
        for paragraph in paragraphs
    ] == [
        160,
        160,
        160,
        120,
        120,
    ]

    assert "\n\n".join(paragraphs) == NORMAL_LONG_TEXT


def test_split_into_paragraphs_returns_one_unit_without_blank_lines() -> None:
    """Continuous prose remains one paragraph unit."""

    text = (
        "Employees require approval before international work. "
        "Security controls remain mandatory."
    )

    assert _split_into_paragraphs(text) == (
        text,
    )


def test_split_into_paragraphs_handles_empty_text() -> None:
    """Empty or whitespace-only content produces no semantic units."""

    assert _split_into_paragraphs("") == ()
    assert _split_into_paragraphs("   ") == ()

def test_split_into_lines_preserves_structured_list_order() -> None:
    """Preserve ordered line structure in mixed list content."""

    text = "\n".join(
        [
            "- First requirement.",
            "- Second requirement.",
            "- Third requirement.",
        ]
    )

    assert _split_into_lines(text) == (
        "- First requirement.",
        "- Second requirement.",
        "- Third requirement.",
    )


def test_split_into_lines_preserves_structured_long_fixture() -> None:
    """Expose the committed structured fixture as ordered lines."""

    lines = _split_into_lines(
        STRUCTURED_LONG_TEXT
    )

    assert len(lines) == 15

    assert lines[0].startswith(
        "Policy rule 1 requires employees"
    )

    assert lines[1].startswith(
        "- Requirement 1:"
    )

    assert lines[12].startswith(
        "- Requirement 12:"
    )

    assert lines[-1].startswith(
        "Policy rule 30 requires employees"
    )


def test_split_into_lines_handles_empty_text() -> None:
    """Empty or whitespace-only content produces no line units."""

    assert _split_into_lines("") == ()
    assert _split_into_lines("   ") == ()

def test_split_into_sentences_preserves_giant_paragraph_order() -> None:
    """Expose a giant paragraph as ordered sentence units."""

    sentences = _split_into_sentences(
        GIANT_PARAGRAPH_TEXT
    )

    assert len(sentences) == 20

    assert sentences[0].startswith(
        "Policy rule 1 requires employees"
    )

    assert sentences[-1].startswith(
        "Policy rule 20 requires employees"
    )

    assert all(
        count_tokens(sentence) == 40
        for sentence in sentences
    )


def test_split_into_sentences_keeps_giant_sentence_as_one_unit() -> None:
    """A sentence without internal terminators must remain intact."""

    sentences = _split_into_sentences(
        GIANT_SENTENCE_TEXT
    )

    assert sentences == (
        GIANT_SENTENCE_TEXT,
    )

    assert (
        count_tokens(sentences[0])
        > MAX_CHUNK_TOKENS
    )


def test_split_into_sentences_handles_empty_text() -> None:
    """Empty or whitespace-only content produces no sentence units."""

    assert _split_into_sentences("") == ()
    assert _split_into_sentences("   ") == ()

def test_split_at_token_boundary_splits_giant_sentence_safely() -> None:
    """Split one oversized sentence without exceeding the hard max."""

    prefix, suffix = _split_at_token_boundary(
        GIANT_SENTENCE_TEXT,
        MAX_CHUNK_TOKENS,
    )

    assert prefix
    assert suffix

    assert count_tokens(prefix) <= MAX_CHUNK_TOKENS

    assert prefix + suffix == GIANT_SENTENCE_TEXT


def test_split_at_token_boundary_returns_whole_text_when_already_safe() -> None:
    """Text already within budget should not be changed."""

    text = "International remote work requires approval."

    prefix, suffix = _split_at_token_boundary(
        text,
        MAX_CHUNK_TOKENS,
    )

    assert prefix == text
    assert suffix == ""


def test_split_at_token_boundary_makes_forward_progress() -> None:
    """Repeated fallback splitting must consume source text."""

    remaining = GIANT_SENTENCE_TEXT

    iterations = 0

    while count_tokens(remaining) > MAX_CHUNK_TOKENS:
        prefix, suffix = _split_at_token_boundary(
            remaining,
            MAX_CHUNK_TOKENS,
        )

        assert prefix
        assert len(suffix) < len(remaining)

        remaining = suffix
        iterations += 1

        if iterations > 20:
            raise AssertionError(
                "token fallback failed to make bounded progress"
            )

    assert iterations >= 1

def test_select_best_boundary_cut_prefers_nearest_target() -> None:
    """Choose the semantic boundary closest to the 350-token target."""

    paragraphs = _split_into_paragraphs(
        NORMAL_LONG_TEXT
    )

    first_boundary = len(paragraphs[0]) + 2

    second_boundary = (
        first_boundary
        + len(paragraphs[1])
        + 2
    )

    third_boundary = (
        second_boundary
        + len(paragraphs[2])
        + 2
    )

    selected = _select_best_boundary_cut(
        NORMAL_LONG_TEXT,
        (
            first_boundary,
            second_boundary,
            third_boundary,
        ),
    )

    assert selected == second_boundary

    assert count_tokens(
        NORMAL_LONG_TEXT[:selected]
    ) == 320


def test_select_best_boundary_cut_rejects_candidate_above_hard_max() -> None:
    """A semantic boundary above 450 tokens must never be selected."""

    paragraphs = _split_into_paragraphs(
        NORMAL_LONG_TEXT
    )

    second_boundary = (
        len(paragraphs[0])
        + 2
        + len(paragraphs[1])
        + 2
    )

    third_boundary = (
        second_boundary
        + len(paragraphs[2])
        + 2
    )

    assert (
        count_tokens(
            NORMAL_LONG_TEXT[:third_boundary]
        )
        > MAX_CHUNK_TOKENS
    )

    selected = _select_best_boundary_cut(
        NORMAL_LONG_TEXT,
        (
            second_boundary,
            third_boundary,
        ),
    )

    assert selected == second_boundary


def test_select_best_boundary_cut_returns_none_without_safe_candidate() -> None:
    """Return None when every available semantic cut exceeds hard max."""

    text = GIANT_SENTENCE_TEXT

    unsafe_position = len(text) - 1

    assert (
        count_tokens(text[:unsafe_position])
        > MAX_CHUNK_TOKENS
    )

    assert _select_best_boundary_cut(
        text,
        (unsafe_position,),
    ) is None


def test_select_best_boundary_cut_rejects_non_increasing_positions() -> None:
    """Reject candidate offsets that cannot represent source order."""

    text = NORMAL_LONG_TEXT

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        _select_best_boundary_cut(
            text,
            (
                100,
                100,
            ),
        )

def test_paragraph_cut_positions_preserve_exact_source_slices() -> None:
    """Paragraph cuts must reconstruct the original fixture exactly."""

    cuts = _paragraph_cut_positions(
        NORMAL_LONG_TEXT
    )

    assert len(cuts) == 4
    assert tuple(sorted(cuts)) == cuts

    parts = []

    start = 0

    for end in cuts:
        parts.append(
            NORMAL_LONG_TEXT[start:end]
        )
        start = end

    parts.append(
        NORMAL_LONG_TEXT[start:]
    )

    assert "".join(parts) == NORMAL_LONG_TEXT


def test_line_cut_positions_preserve_structured_fixture_order() -> None:
    """Line cuts must expose the structured fixture without loss."""

    cuts = _line_cut_positions(
        STRUCTURED_LONG_TEXT
    )

    assert len(cuts) == (
        STRUCTURED_LONG_TEXT.count("\n")
    )

    assert tuple(sorted(cuts)) == cuts

    reconstructed = []
    start = 0

    for end in cuts:
        reconstructed.append(
            STRUCTURED_LONG_TEXT[start:end]
        )
        start = end

    reconstructed.append(
        STRUCTURED_LONG_TEXT[start:]
    )

    assert "".join(reconstructed) == STRUCTURED_LONG_TEXT


def test_sentence_cut_positions_preserve_giant_paragraph() -> None:
    """Sentence cuts must preserve exact source text and order."""

    cuts = _sentence_cut_positions(
        GIANT_PARAGRAPH_TEXT
    )

    assert len(cuts) == 19
    assert tuple(sorted(cuts)) == cuts

    rebuilt = []
    start = 0

    for end in cuts:
        rebuilt.append(
            GIANT_PARAGRAPH_TEXT[start:end]
        )
        start = end

    rebuilt.append(
        GIANT_PARAGRAPH_TEXT[start:]
    )

    assert "".join(rebuilt) == GIANT_PARAGRAPH_TEXT


def test_sentence_cut_positions_returns_empty_for_giant_sentence() -> None:
    """One giant sentence has no internal sentence cut."""

    assert _sentence_cut_positions(
        GIANT_SENTENCE_TEXT
    ) == ()

def test_select_semantic_cut_prefers_paragraph_boundary() -> None:
    """Use paragraph boundaries before weaker semantic boundaries."""

    cut = _select_semantic_cut_position(
        NORMAL_LONG_TEXT
    )

    prefix = NORMAL_LONG_TEXT[:cut]
    suffix = NORMAL_LONG_TEXT[cut:]

    assert prefix
    assert suffix

    assert count_tokens(prefix) == 320

    assert prefix + suffix == NORMAL_LONG_TEXT

    assert cut in _paragraph_cut_positions(
        NORMAL_LONG_TEXT
    )

def test_select_semantic_cut_falls_back_to_line_boundary() -> None:
    """Use line boundaries when no paragraph boundary is available."""

    text = "\n".join(
        make_policy_sentence(index)
        for index in range(1, 13)
    )

    assert count_tokens(text) > MAX_CHUNK_TOKENS

    assert _paragraph_cut_positions(text) == ()

    cut = _select_semantic_cut_position(text)

    prefix = text[:cut]
    suffix = text[cut:]

    assert prefix
    assert suffix

    assert prefix + suffix == text

    assert cut in _line_cut_positions(text)

    assert count_tokens(prefix) <= MAX_CHUNK_TOKENS

def test_select_semantic_cut_falls_back_to_sentence_boundary() -> None:
    """Use sentence boundaries for one oversized prose paragraph."""

    assert (
        count_tokens(GIANT_PARAGRAPH_TEXT)
        > MAX_CHUNK_TOKENS
    )

    assert _paragraph_cut_positions(
        GIANT_PARAGRAPH_TEXT
    ) == ()

    assert _line_cut_positions(
        GIANT_PARAGRAPH_TEXT
    ) == ()

    cut = _select_semantic_cut_position(
        GIANT_PARAGRAPH_TEXT
    )

    prefix = GIANT_PARAGRAPH_TEXT[:cut]
    suffix = GIANT_PARAGRAPH_TEXT[cut:]

    assert prefix
    assert suffix

    assert prefix + suffix == GIANT_PARAGRAPH_TEXT

    assert cut in _sentence_cut_positions(
        GIANT_PARAGRAPH_TEXT
    )

    assert count_tokens(prefix) <= MAX_CHUNK_TOKENS

def test_select_semantic_cut_uses_token_fallback_for_giant_sentence() -> None:
    """Use exact token offsets when no semantic boundary is available."""

    assert (
        count_tokens(GIANT_SENTENCE_TEXT)
        > MAX_CHUNK_TOKENS
    )

    assert _paragraph_cut_positions(
        GIANT_SENTENCE_TEXT
    ) == ()

    assert _line_cut_positions(
        GIANT_SENTENCE_TEXT
    ) == ()

    assert _sentence_cut_positions(
        GIANT_SENTENCE_TEXT
    ) == ()

    cut = _select_semantic_cut_position(
        GIANT_SENTENCE_TEXT
    )

    prefix = GIANT_SENTENCE_TEXT[:cut]
    suffix = GIANT_SENTENCE_TEXT[cut:]

    assert prefix
    assert suffix

    assert prefix + suffix == GIANT_SENTENCE_TEXT

    assert count_tokens(prefix) <= TARGET_CHUNK_TOKENS

    assert count_tokens(prefix) <= MAX_CHUNK_TOKENS

# ---------------------------------------------------------------------------
# Long-section overlap traversal helpers and tests
# ---------------------------------------------------------------------------


def locate_chunk_spans(
    source: str,
    chunks: tuple[Chunk, ...],
) -> tuple[tuple[int, int], ...]:
    """Locate overlapping chunks as exact source spans in output order.

    The first chunk must begin at source offset zero. Each later chunk must
    overlap the previous chunk while extending beyond the previous unique
    coverage frontier. The helper searches only for an occurrence satisfying
    those ordering constraints, which avoids silently accepting a repeated
    substring at an unrelated source position.
    """

    if not isinstance(source, str):
        raise TypeError("source must be a string.")

    if not isinstance(chunks, tuple):
        raise TypeError("chunks must be a tuple.")

    if not chunks:
        return ()

    spans: list[tuple[int, int]] = []

    first = chunks[0]

    if not source.startswith(first.text):
        raise AssertionError(
            "first chunk must begin at the start of the source"
        )

    first_end = len(first.text)

    if first_end > len(source):
        raise AssertionError(
            "first chunk extends beyond the source"
        )

    spans.append((0, first_end))

    for chunk in chunks[1:]:
        previous_start, previous_end = spans[-1]
        search_from = previous_start
        selected_start: int | None = None

        while True:
            candidate_start = source.find(
                chunk.text,
                search_from,
            )

            if candidate_start < 0:
                break

            candidate_end = (
                candidate_start
                + len(chunk.text)
            )

            if (
                candidate_start < previous_end
                and candidate_end > previous_end
                and candidate_end <= len(source)
            ):
                selected_start = candidate_start
                break

            if candidate_start >= previous_end:
                break

            search_from = candidate_start + 1

        if selected_start is None:
            raise AssertionError(
                "later chunk was not found as an overlapping "
                "forward-progress source span"
            )

        end = selected_start + len(chunk.text)

        if source[selected_start:end] != chunk.text:
            raise AssertionError(
                "located chunk does not match exact source text"
            )

        spans.append(
            (
                selected_start,
                end,
            )
        )

    return tuple(spans)


def assert_overlapping_chunk_coverage(
    section: ParsedSection,
    chunks: tuple[Chunk, ...],
) -> tuple[tuple[int, int], ...]:
    """Assert bounded overlap, provenance, coverage, and forward progress."""

    if not chunks:
        raise AssertionError(
            "non-empty long section must produce chunks"
        )

    spans = locate_chunk_spans(
        section.text,
        chunks,
    )

    assert spans[0][0] == 0
    assert spans[-1][1] == len(section.text)

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))

    for chunk in chunks:
        assert chunk.token_count == count_tokens(
            chunk.text
        )
        assert chunk.token_count <= MAX_CHUNK_TOKENS

        assert chunk.doc_id == section.doc_id
        assert chunk.title == section.title
        assert chunk.section_path == section.section_path
        assert chunk.section_order == section.section_order
        assert chunk.source_format == section.source_format

    for index in range(1, len(spans)):
        _, previous_end = spans[index - 1]
        current_start, current_end = spans[index]

        assert current_start < previous_end
        assert current_end > previous_end

        overlap_text = section.text[
            current_start:previous_end
        ]
        overlap_tokens = count_tokens(
            overlap_text
        )

        assert (
            MIN_CHUNK_OVERLAP_TOKENS
            <= overlap_tokens
            <= MAX_CHUNK_OVERLAP_TOKENS
        )

    return spans


def test_split_long_section_preserves_normal_long_coverage() -> None:
    """Split ordinary long prose with deterministic semantic overlap."""

    section = make_chunk_test_section(
        NORMAL_LONG_TEXT
    )

    first = split_long_section(section)
    second = split_long_section(section)

    assert first == second

    assert [
        chunk.token_count
        for chunk in first
    ] == [
        320,
        440,
    ]

    spans = assert_overlapping_chunk_coverage(
        section,
        first,
    )

    overlap_text = section.text[
        spans[1][0]:spans[0][1]
    ]

    assert count_tokens(overlap_text) == 40


def test_split_long_section_handles_structured_long_text() -> None:
    """Preserve mixed paragraph/list structure with bounded overlap."""

    section = make_chunk_test_section(
        STRUCTURED_LONG_TEXT
    )

    chunks = split_long_section(section)

    assert [
        chunk.token_count
        for chunk in chunks
    ] == [
        240,
        340,
        290,
        280,
    ]

    assert_overlapping_chunk_coverage(
        section,
        chunks,
    )


def test_split_long_section_recomputes_end_for_giant_paragraph() -> None:
    """Recompute chunk ends so overlap never violates the hard maximum."""

    section = make_chunk_test_section(
        GIANT_PARAGRAPH_TEXT
    )

    chunks = split_long_section(section)

    assert [
        chunk.token_count
        for chunk in chunks
    ] == [
        360,
        360,
        160,
    ]

    assert_overlapping_chunk_coverage(
        section,
        chunks,
    )


def test_split_long_section_handles_giant_sentence_with_token_fallback() -> None:
    """Use deterministic token-level overlap when no semantic boundary exists."""

    section = make_chunk_test_section(
        GIANT_SENTENCE_TEXT
    )

    first = split_long_section(section)
    second = split_long_section(section)

    assert first == second

    assert [
        chunk.token_count
        for chunk in first
    ] == [
        350,
        350,
        350,
        350,
        350,
        377,
    ]

    spans = assert_overlapping_chunk_coverage(
        section,
        first,
    )

    overlap_counts = [
        count_tokens(
            section.text[
                spans[index][0]:
                spans[index - 1][1]
            ]
        )
        for index in range(1, len(spans))
    ]

    assert all(
        MIN_CHUNK_OVERLAP_TOKENS
        <= value
        <= MAX_CHUNK_OVERLAP_TOKENS
        for value in overlap_counts
    )


def test_split_long_section_rejects_section_that_already_fits() -> None:
    """The long-section API should reject already-safe sections."""

    section = make_chunk_test_section(
        BOUNDARY_LONG_TEXT
    )

    assert (
        count_tokens(section.text)
        <= MAX_CHUNK_TOKENS
    )

    with pytest.raises(
        ValueError,
        match="does not require long-section splitting",
    ):
        split_long_section(section)

@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
        True,
    ],
)
def test_split_at_token_boundary_rejects_invalid_max_tokens(
    max_tokens: Any,
) -> None:
    """Reject invalid token budgets."""

    with pytest.raises(
        ValueError,
        match="max_tokens must be a positive integer",
    ):
        _split_at_token_boundary(
            "Policy text.",
            max_tokens,
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



@pytest.mark.parametrize(
    "value",
    [
        None,
        "section",
        123,
        {},
        [],
    ],
)
def test_normalize_section_rejects_non_parsed_section(
    value: Any,
) -> None:
    """Reject values outside the ParsedSection contract."""

    with pytest.raises(
        TypeError,
        match="section must be a ParsedSection instance",
    ):
        normalize_section(value)


def test_normalize_section_preserves_metadata_and_normalizes_text() -> None:
    """Normalize only section text while preserving citation metadata."""

    section = ParsedSection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        text=(
            "Approval   is required.\r\n"
            "Second\u00ad line."
        ),
        source_format="md",
    )

    normalized = normalize_section(section)

    assert normalized is not section

    assert normalized.doc_id == section.doc_id
    assert normalized.title == section.title
    assert normalized.section_path == section.section_path
    assert normalized.section_order == section.section_order
    assert normalized.source_format == section.source_format

    assert normalized.text == (
        "Approval is required.\n"
        "Second line."
    )



def test_normalize_section_is_idempotent() -> None:
    """Applying section normalization twice must be stable."""

    section = ParsedSection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        text=(
            "\n"
            "Approval   is required.\r\n"
            "\r\n"
            "Second\u00ad line.   "
            "\n"
        ),
        source_format="md",
    )

    once = normalize_section(section)
    twice = normalize_section(once)

    assert twice == once

def test_generate_chunk_id_is_deterministic() -> None:
    """Identical canonical chunk inputs must produce identical IDs."""

    arguments = {
        "doc_id": "HR-POL-004",
        "section_path": (
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        "chunk_index": 0,
        "text": "International remote work requires approval.",
    }

    first = generate_chunk_id(**arguments)
    second = generate_chunk_id(**arguments)

    assert first == second
    assert first.startswith("HR-POL-004__0000__")

    digest = first.rsplit("__", maxsplit=1)[-1]

    assert len(digest) == CHUNK_ID_DIGEST_LENGTH
    assert all(
        character in "0123456789abcdef"
        for character in digest
    )


def test_generate_chunk_id_changes_with_chunk_index() -> None:
    """Different chunks in one section must have different IDs."""

    common = {
        "doc_id": "HR-POL-004",
        "section_path": (
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        "text": "International remote work requires approval.",
    }

    first = generate_chunk_id(
        chunk_index=0,
        **common,
    )
    second = generate_chunk_id(
        chunk_index=1,
        **common,
    )

    assert first != second


def test_generate_chunk_id_changes_with_text() -> None:
    """A policy-content change must invalidate the prior chunk ID."""

    common = {
        "doc_id": "HR-POL-004",
        "section_path": (
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        "chunk_index": 0,
    }

    first = generate_chunk_id(
        text="International remote work requires approval.",
        **common,
    )
    second = generate_chunk_id(
        text="International remote work requires prior approval.",
        **common,
    )

    assert first != second


def test_generate_chunk_id_changes_with_document() -> None:
    """Identical text in different policy documents needs distinct IDs."""

    common = {
        "section_path": (
            "Shared Policy Heading",
            "1. Requirements",
        ),
        "chunk_index": 0,
        "text": "Employees must obtain approval.",
    }

    first = generate_chunk_id(
        doc_id="HR-POL-004",
        **common,
    )
    second = generate_chunk_id(
        doc_id="HR-POL-005",
        **common,
    )

    assert first != second


def test_generate_chunk_id_changes_with_section_path() -> None:
    """Identical text in different policy sections needs distinct IDs."""

    common = {
        "doc_id": "HR-POL-004",
        "chunk_index": 0,
        "text": "Employees must obtain approval.",
    }

    first = generate_chunk_id(
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
        ),
        **common,
    )
    second = generate_chunk_id(
        section_path=(
            "Remote and Flexible Work Policy",
            "5. Approval Process",
        ),
        **common,
    )

    assert first != second

@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "doc_id": "",
                "section_path": ("Policy", "1. Section"),
                "chunk_index": 0,
                "text": "Policy text.",
            },
            "doc_id must be a non-empty string",
        ),
        (
            {
                "doc_id": "HR-POL-001",
                "section_path": (),
                "chunk_index": 0,
                "text": "Policy text.",
            },
            "section_path must be a non-empty tuple",
        ),
        (
            {
                "doc_id": "HR-POL-001",
                "section_path": ("Policy", "1. Section"),
                "chunk_index": -1,
                "text": "Policy text.",
            },
            "chunk_index must be a non-negative integer",
        ),
        (
            {
                "doc_id": "HR-POL-001",
                "section_path": ("Policy", "1. Section"),
                "chunk_index": 0,
                "text": "",
            },
            "text must be a non-empty string",
        ),
    ],
)
def test_generate_chunk_id_rejects_invalid_identity_inputs(
    arguments: dict[str, object],
    message: str,
) -> None:
    """Reject identity values that cannot represent a valid chunk."""

    with pytest.raises(ValueError, match=message):
        generate_chunk_id(**arguments)

def test_chunk_preserves_minimal_section_provenance() -> None:
    """Represent one chunk without losing its source section."""

    chunk = Chunk(
        chunk_id="HR-POL-004__0000__0000000000000000",
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        chunk_index=0,
        text="International remote work requires approval.",
        token_count=6,
        source_format="md",
    )

    assert chunk.doc_id == "HR-POL-004"
    assert chunk.title == "Remote and Flexible Work Policy"
    assert chunk.section_path == (
        "Remote and Flexible Work Policy",
        "4. Policy Requirements",
        "4.4 International duration limit",
    )
    assert chunk.section_order == 12
    assert chunk.chunk_index == 0
    assert chunk.text == (
        "International remote work requires approval."
    )
    assert chunk.token_count == 6
    assert chunk.source_format == "md"

@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
    ],
)
def test_chunk_rejects_empty_text(text: str) -> None:
    """Empty structural sections must never become chunks."""

    with pytest.raises(
        ValueError,
        match="text must be a non-empty string",
    ):
        Chunk(
            chunk_id="HR-POL-004__0000__0000000000000000",
            doc_id="HR-POL-004",
            title="Remote and Flexible Work Policy",
            section_path=(
                "Remote and Flexible Work Policy",
                "4. Policy Requirements",
            ),
            section_order=8,
            chunk_index=0,
            text=text,
            token_count=1,
            source_format="md",
        )
def test_chunk_rejects_token_count_above_hard_maximum() -> None:
    """No materialized chunk may exceed the 450-token hard limit."""

    with pytest.raises(
        ValueError,
        match="token_count exceeds",
    ):
        Chunk(
            chunk_id="HR-POL-004__0000__0000000000000000",
            doc_id="HR-POL-004",
            title="Remote and Flexible Work Policy",
            section_path=(
                "Remote and Flexible Work Policy",
                "4. Policy Requirements",
            ),
            section_order=8,
            chunk_index=0,
            text="Policy text.",
            token_count=MAX_CHUNK_TOKENS + 1,
            source_format="md",
        )

def test_chunk_is_immutable() -> None:
    """Prevent mutation after a chunk has been materialized."""

    chunk = Chunk(
        chunk_id="HR-POL-004__0000__0000000000000000",
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        chunk_index=0,
        text="International remote work requires approval.",
        token_count=6,
        source_format="md",
    )

    with pytest.raises(AttributeError):
        chunk.text = "Changed policy text."

def test_chunk_configuration_matches_frozen_spec() -> None:
    """Keep CP4/CP5 tokenizer and chunk budgets aligned with the spec."""

    assert EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert TARGET_CHUNK_TOKENS == 350
    assert MAX_CHUNK_TOKENS == 450
    assert CHUNK_OVERLAP_TOKENS == 50


def test_chunk_overlap_stays_within_frozen_percentage_range() -> None:
    """Keep overlap within the required 10-15% of target size."""

    overlap_ratio = CHUNK_OVERLAP_TOKENS / TARGET_CHUNK_TOKENS

    assert 0.10 <= overlap_ratio <= 0.15


def test_chunk_token_limits_are_internally_consistent() -> None:
    """Reject configuration relationships that cannot support chunking."""

    assert TARGET_CHUNK_TOKENS > 0
    assert MAX_CHUNK_TOKENS > TARGET_CHUNK_TOKENS
    assert 0 < CHUNK_OVERLAP_TOKENS < TARGET_CHUNK_TOKENS

def test_chunk_section_skips_empty_structural_section() -> None:
    """Empty parent headings must produce no retrievable chunk."""

    section = ParsedSection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
        ),
        section_order=8,
        text="",
        source_format="md",
    )

    chunks = chunk_section(section)

    assert chunks == ()

def test_chunk_section_creates_one_chunk_for_short_section() -> None:
    """A non-empty section below target becomes one chunk."""

    text = "International remote work requires approval."

    section = ParsedSection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        text=text,
        source_format="md",
    )

    chunks = chunk_section(section)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.doc_id == section.doc_id
    assert chunk.title == section.title
    assert chunk.section_path == section.section_path
    assert chunk.section_order == section.section_order
    assert chunk.chunk_index == 0
    assert chunk.text == text
    assert chunk.token_count == count_tokens(text)
    assert chunk.source_format == section.source_format

def test_chunk_section_splits_section_above_hard_maximum() -> None:
    """Oversized sections dispatch to deterministic overlapping splitting."""

    section = make_chunk_test_section(
        NORMAL_LONG_TEXT
    )

    assert (
        count_tokens(section.text)
        > MAX_CHUNK_TOKENS
    )

    chunks = chunk_section(section)

    assert [
        chunk.token_count
        for chunk in chunks
    ] == [
        320,
        440,
    ]

    assert_overlapping_chunk_coverage(
        section,
        chunks,
    )
def test_chunk_section_keeps_section_within_hard_max_as_one_chunk() -> None:
    """A section above target but within hard max remains intact."""

    section = make_chunk_test_section(
        BOUNDARY_LONG_TEXT
    )

    token_count = count_tokens(section.text)

    assert token_count > TARGET_CHUNK_TOKENS
    assert token_count <= MAX_CHUNK_TOKENS

    chunks = chunk_section(section)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.chunk_index == 0
    assert chunk.text == section.text
    assert chunk.token_count == token_count

    assert chunk.doc_id == section.doc_id
    assert chunk.title == section.title
    assert chunk.section_path == section.section_path
    assert chunk.section_order == section.section_order
    assert chunk.source_format == section.source_format

def test_chunk_section_assigns_expected_deterministic_id() -> None:
    """A materialized short chunk carries its canonical generated ID."""

    text = "International remote work requires approval."

    section = ParsedSection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section_path=(
            "Remote and Flexible Work Policy",
            "4. Policy Requirements",
            "4.4 International duration limit",
        ),
        section_order=12,
        text=text,
        source_format="md",
    )

    chunk = chunk_section(section)[0]

    assert chunk.chunk_id == generate_chunk_id(
        doc_id=section.doc_id,
        section_path=section.section_path,
        chunk_index=0,
        text=text,
    )

def test_long_section_assigns_unique_deterministic_chunk_ids() -> None:
    """Each materialized chunk in one long section gets its own stable ID."""

    section = make_chunk_test_section(
        NORMAL_LONG_TEXT
    )

    first_run = chunk_section(section)
    second_run = chunk_section(section)

    first_ids = tuple(
        chunk.chunk_id
        for chunk in first_run
    )
    second_ids = tuple(
        chunk.chunk_id
        for chunk in second_run
    )

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))

    for chunk in first_run:
        assert chunk.chunk_id == generate_chunk_id(
            doc_id=chunk.doc_id,
            section_path=chunk.section_path,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )
def test_real_corpus_chunk_ids_are_unique_and_deterministic() -> None:
    """Real policy chunks have unique, reproducible canonical IDs."""

    def build_real_corpus_chunks() -> tuple[Chunk, ...]:
        """Run the verified corpus path through chunk materialization."""

        manifest = load_manifest(
            Path("corpus/version.json")
        )

        resolved_documents = resolve_manifest_sources(
            manifest,
            Path("corpus/source"),
        )

        parsed_sections: list[ParsedSection] = []

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
                raise AssertionError(
                    "Unexpected supported source format in "
                    f"real corpus: {document.source_format!r}"
                )

            parsed_sections.extend(sections)

        normalized_sections = tuple(
            normalize_section(section)
            for section in parsed_sections
        )

        return tuple(
            chunk
            for section in normalized_sections
            for chunk in chunk_section(section)
        )

    first_run = build_real_corpus_chunks()
    second_run = build_real_corpus_chunks()

    assert len(first_run) == 400
    assert len(second_run) == 400

    first_ids = tuple(
        chunk.chunk_id
        for chunk in first_run
    )

    second_ids = tuple(
        chunk.chunk_id
        for chunk in second_run
    )

    assert first_ids == second_ids

    assert len(first_ids) == len(set(first_ids))

    assert all(
        chunk.chunk_id
        for chunk in first_run
    )

    for chunk in first_run:
        assert chunk.chunk_id == generate_chunk_id(
            doc_id=chunk.doc_id,
            section_path=chunk.section_path,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )


def test_chunk_rejects_empty_chunk_id() -> None:
    """A materialized chunk must always have a persistent identity."""

    with pytest.raises(
        ValueError,
        match="chunk_id must be a non-empty string",
    ):
        Chunk(
            chunk_id="",
            doc_id="HR-POL-004",
            title="Remote and Flexible Work Policy",
            section_path=(
                "Remote and Flexible Work Policy",
                "4. Policy Requirements",
            ),
            section_order=8,
            chunk_index=0,
            text="Policy text.",
            token_count=2,
            source_format="md",
        )

def test_select_overlap_start_prefers_semantic_sentence_boundary() -> None:
    """Prefer a compliant semantic overlap before token fallback."""

    previous_text = make_policy_paragraph(
        start_index=1,
        sentence_count=9,
    )

    assert count_tokens(previous_text) == 360

    start = _select_overlap_start(
        previous_text
    )

    overlap_text = previous_text[start:]
    overlap_tokens = count_tokens(overlap_text)

    assert (
        MIN_CHUNK_OVERLAP_TOKENS
        <= overlap_tokens
        <= MAX_CHUNK_OVERLAP_TOKENS
    )

    assert overlap_tokens == 40

    assert start in _sentence_cut_positions(
        previous_text
    )


def test_select_overlap_start_prefers_line_boundary_when_available() -> None:
    """Use line structure before sentence fallback when compliant."""

    lines = [
        (
            f"- Requirement {index}: employees must confirm "
            "manager approval, approved work location, "
            "security controls, and escalation requirements "
            "before proceeding."
        )
        for index in range(1, 8)
    ]

    previous_text = "\n".join(lines)

    start = _select_overlap_start(
        previous_text
    )

    overlap_tokens = count_tokens(
        previous_text[start:]
    )

    assert (
        MIN_CHUNK_OVERLAP_TOKENS
        <= overlap_tokens
        <= MAX_CHUNK_OVERLAP_TOKENS
    )

    assert start in _line_cut_positions(
        previous_text
    )


def test_select_overlap_start_uses_token_fallback_for_giant_sentence() -> None:
    """Use tokenizer offsets when no semantic overlap exists."""

    previous_text = GIANT_SENTENCE_TEXT[:2000]

    assert _paragraph_cut_positions(
        previous_text
    ) == ()

    assert _line_cut_positions(
        previous_text
    ) == ()

    assert _sentence_cut_positions(
        previous_text
    ) == ()

    start = _select_overlap_start(
        previous_text
    )

    overlap_text = previous_text[start:]
    overlap_tokens = count_tokens(overlap_text)

    assert (
        MIN_CHUNK_OVERLAP_TOKENS
        <= overlap_tokens
        <= MAX_CHUNK_OVERLAP_TOKENS
    )


def test_select_overlap_start_is_deterministic() -> None:
    """Repeated overlap selection must return the same source offset."""

    previous_text = make_policy_paragraph(
        start_index=1,
        sentence_count=9,
    )

    first = _select_overlap_start(
        previous_text
    )

    second = _select_overlap_start(
        previous_text
    )

    assert second == first

def test_get_tokenizer_loads_expected_fast_tokenizer() -> None:
    """Load the frozen BGE tokenizer with the required fast backend."""

    get_tokenizer.cache_clear()

    tokenizer = get_tokenizer()

    assert tokenizer.is_fast is True
    assert tokenizer.name_or_path == EMBEDDING_MODEL_NAME
    assert tokenizer.model_max_length >= MAX_CHUNK_TOKENS


def test_get_tokenizer_is_cached_per_process() -> None:
    """Repeated loader calls must return the same tokenizer object."""

    get_tokenizer.cache_clear()

    first = get_tokenizer()
    second = get_tokenizer()

    assert second is first

    cache_info = get_tokenizer.cache_info()

    assert cache_info.misses == 1
    assert cache_info.hits == 1
    assert cache_info.currsize == 1


def test_get_tokenizer_context_covers_hard_chunk_limit() -> None:
    """Keep tokenizer context safely above the configured hard maximum."""

    tokenizer = get_tokenizer()

    assert tokenizer.model_max_length == 512
    assert MAX_CHUNK_TOKENS == 450
    assert tokenizer.model_max_length > MAX_CHUNK_TOKENS



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
def test_count_tokens_rejects_non_string_input(
    value: Any,
) -> None:
    """Reject values outside the exact token-counting contract."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        count_tokens(value)


def test_count_tokens_returns_zero_for_empty_text() -> None:
    """Empty content contributes no document tokens."""

    assert count_tokens("") == 0


def test_count_tokens_matches_verified_bge_sample() -> None:
    """Keep exact counting aligned with the frozen BGE tokenizer."""

    text = "International remote work requires approval."

    assert count_tokens(text) == 6



def test_count_tokens_does_not_truncate_long_input() -> None:
    """Measure the complete input even beyond model context length."""

    tokenizer = get_tokenizer()

    # "policy" is one token for this tokenizer. Repeating it 600 times
    # gives us an input that clearly exceeds the 512-token model context.
    text = " ".join(["policy"] * 600)

    direct_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    assert len(direct_ids) > tokenizer.model_max_length
    assert count_tokens(text) == len(direct_ids)
    assert count_tokens(text) > MAX_CHUNK_TOKENS
