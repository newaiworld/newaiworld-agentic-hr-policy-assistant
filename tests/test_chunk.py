"""Tests for deterministic policy-corpus ingestion and chunking.

The file currently tests the ingestion manifest contract.
Chunking tests will be added when rag/chunk.py is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.ingest import (
    SUPPORTED_SOURCE_FORMATS,
    CorpusManifest,
    ManifestDocument,
    ManifestValidationError,
    load_manifest,
    parse_manifest_data,
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
