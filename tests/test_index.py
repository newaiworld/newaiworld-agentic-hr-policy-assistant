"""Focused tests for the offline S4 index lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import rag.store as store_module

from rag.index import (
    SEMANTIC_SMOKE_CASES,
    SEMANTIC_SMOKE_N_RESULTS,
    IndexBuildInputs,
    IndexBuildState,
    IndexEmbeddingInputs,
    IndexFinalizedBuildState,
    IndexLifecycleError,
    IndexValidatedBuildState,
    build_chroma_index,
    build_index_inputs,
    build_policy_index,
    embed_index_documents,
    finalize_chroma_build_metadata,
    load_canonical_chunks,
    publish_policy_index,
    validate_canonical_chunk_records,
    validate_chroma_build,
    validate_chroma_build_semantic_smoke,
    validate_index_build_preconditions,
)

from rag.chunk import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    TARGET_CHUNK_TOKENS,
)

from rag.embed import (
    EMBEDDING_DIMENSION,
)

def test_index_lifecycle_error_is_runtime_error() -> None:
    """Lifecycle failures must cross a project-owned runtime boundary."""

    assert issubclass(
        IndexLifecycleError,
        RuntimeError,
    )


def test_load_canonical_chunks_preserves_ordered_records(
    tmp_path: Path,
) -> None:
    """Loader must preserve JSON record order and values."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_text(
        (
            "["
            '{"chunk_id":"chunk-a","text":"First."},'
            '{"chunk_id":"chunk-b","text":"Second."}'
            "]"
        ),
        encoding="utf-8",
    )

    result = load_canonical_chunks(
        path
    )

    assert result == [
        {
            "chunk_id": "chunk-a",
            "text": "First.",
        },
        {
            "chunk_id": "chunk-b",
            "text": "Second.",
        },
    ]


def test_load_canonical_chunks_rejects_non_path() -> None:
    """The artifact location must use the pathlib contract."""

    with pytest.raises(
        TypeError,
        match="path must be a pathlib.Path instance",
    ):
        load_canonical_chunks(
            "chunks.json"  # type: ignore[arg-type]
        )


def test_load_canonical_chunks_rejects_relative_path() -> None:
    """Index lifecycle paths must not depend on working directory."""

    with pytest.raises(
        ValueError,
        match="path must be absolute",
    ):
        load_canonical_chunks(
            Path("chunks.json")
        )


def test_load_canonical_chunks_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing canonical artifacts must fail explicitly."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    with pytest.raises(
        IndexLifecycleError,
        match="does not exist",
    ):
        load_canonical_chunks(
            path
        )


def test_load_canonical_chunks_rejects_directory(
    tmp_path: Path,
) -> None:
    """The canonical artifact must be a regular file."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.mkdir()

    with pytest.raises(
        IndexLifecycleError,
        match="not a regular file",
    ):
        load_canonical_chunks(
            path
        )


def test_load_canonical_chunks_rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    """Malformed byte content must not enter the lifecycle."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_bytes(
        b"\xff\xfe\xfa"
    )

    with pytest.raises(
        IndexLifecycleError,
        match="not valid UTF-8",
    ):
        load_canonical_chunks(
            path
        )


def test_load_canonical_chunks_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON must fail with the lifecycle error boundary."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_text(
        "[invalid-json]",
        encoding="utf-8",
    )

    with pytest.raises(
        IndexLifecycleError,
        match="contains invalid JSON",
    ) as exc_info:
        load_canonical_chunks(
            path
        )

    assert isinstance(
        exc_info.value.__cause__,
        Exception,
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '"text"',
        "123",
        "null",
    ],
)
def test_load_canonical_chunks_rejects_non_list_top_level(
    tmp_path: Path,
    payload: str,
) -> None:
    """Canonical chunks.json must always contain a JSON list."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_text(
        payload,
        encoding="utf-8",
    )

    with pytest.raises(
        IndexLifecycleError,
        match="top-level list",
    ):
        load_canonical_chunks(
            path
        )


def test_load_canonical_chunks_rejects_empty_list(
    tmp_path: Path,
) -> None:
    """An empty canonical artifact cannot build a policy index."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        IndexLifecycleError,
        match="at least one record",
    ):
        load_canonical_chunks(
            path
        )


def test_load_canonical_chunks_rejects_non_dictionary_record(
    tmp_path: Path,
) -> None:
    """Every top-level list member must be a record mapping."""

    path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    path.write_text(
        (
            "["
            '{"chunk_id":"chunk-a"},'
            '"invalid-record"'
            "]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        IndexLifecycleError,
        match="non-dictionary record at index 1",
    ):
        load_canonical_chunks(
            path
        )


def _valid_chunk_record(
    *,
    chunk_id: str = "HR-POL-TEST__0000__abcdef1234567890",
) -> dict[str, object]:
    """Return one minimal record matching the canonical chunk schema."""

    return {
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "doc_id": "HR-POL-TEST",
        "section_order": 1,
        "section_path": [
            "Test Policy",
            "1. Scope",
        ],
        "source_format": "md",
        "text": "Synthetic canonical policy text.",
        "title": "Test Policy",
        "token_count": 6,
    }


def test_validate_canonical_chunk_records_preserves_order() -> None:
    """Canonical validation must not reorder records."""

    first = _valid_chunk_record(
        chunk_id="chunk-a"
    )

    second = _valid_chunk_record(
        chunk_id="chunk-b"
    )

    result = (
        validate_canonical_chunk_records(
            [
                first,
                second,
            ]
        )
    )

    assert result == (
        first,
        second,
    )


def test_validate_canonical_chunk_records_rejects_wrong_schema() -> None:
    """Every record must match the exact nine-field schema."""

    record = _valid_chunk_record()

    record["unexpected"] = "value"

    with pytest.raises(
        IndexLifecycleError,
        match="unexpected schema",
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


def test_validate_canonical_chunk_records_rejects_duplicate_ids() -> None:
    """Chunk IDs must remain globally unique."""

    first = _valid_chunk_record(
        chunk_id="duplicate"
    )

    second = _valid_chunk_record(
        chunk_id="duplicate"
    )

    with pytest.raises(
        IndexLifecycleError,
        match="must be unique",
    ):
        validate_canonical_chunk_records(
            [
                first,
                second,
            ]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "chunk_id",
        "doc_id",
        "title",
        "text",
    ],
)
def test_validate_canonical_chunk_records_rejects_blank_string_fields(
    field_name: str,
) -> None:
    """Required canonical string fields must never be blank."""

    record = _valid_chunk_record()

    record[field_name] = "   "

    with pytest.raises(
        IndexLifecycleError,
        match=field_name,
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "section_order",
        "chunk_index",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
    ],
)
def test_validate_canonical_chunk_records_rejects_invalid_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    """Section and chunk indexes must be real non-negative integers."""

    record = _valid_chunk_record()

    record[field_name] = value

    with pytest.raises(
        IndexLifecycleError,
        match=field_name,
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        "6",
    ],
)
def test_validate_canonical_chunk_records_rejects_invalid_token_count(
    value: object,
) -> None:
    """Canonical token counts must be positive integers."""

    record = _valid_chunk_record()

    record["token_count"] = value

    with pytest.raises(
        IndexLifecycleError,
        match="token_count",
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        [],
        ["Policy", ""],
        "Policy",
    ],
)
def test_validate_canonical_chunk_records_rejects_invalid_section_path(
    value: object,
) -> None:
    """Section path must remain a non-empty ordered string list."""

    record = _valid_chunk_record()

    record["section_path"] = value

    with pytest.raises(
        IndexLifecycleError,
        match="section_path",
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


def test_validate_canonical_chunk_records_rejects_invalid_source_format() -> None:
    """Only canonical Markdown/PDF format labels are accepted."""

    record = _valid_chunk_record()

    record["source_format"] = "docx"

    with pytest.raises(
        IndexLifecycleError,
        match="source_format",
    ):
        validate_canonical_chunk_records(
            [
                record
            ]
        )


def test_build_index_inputs_preserves_text_order(
    tmp_path: Path,
) -> None:
    """Build inputs must align texts exactly with canonical chunk order."""

    chunks_path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    manifest_path = Path(
        "corpus/version.json"
    ).resolve()

    first = _valid_chunk_record(
        chunk_id="chunk-a"
    )

    first["text"] = "First text."

    second = _valid_chunk_record(
        chunk_id="chunk-b"
    )

    second["text"] = "Second text."

    import json

    chunks_path.write_text(
        json.dumps(
            [
                first,
                second,
            ]
        ),
        encoding="utf-8",
    )

    result = build_index_inputs(
        chunks_path,
        manifest_path,
    )

    assert result.corpus_version == "1.2"

    assert result.texts == (
        "First text.",
        "Second text.",
    )

    assert result.chunks == (
        first,
        second,
    )


def test_build_index_inputs_is_immutable(
    tmp_path: Path,
) -> None:
    """Lifecycle build inputs must not be reassigned after validation."""

    chunks_path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    manifest_path = Path(
        "corpus/version.json"
    ).resolve()

    import json

    record = _valid_chunk_record()

    chunks_path.write_text(
        json.dumps(
            [
                record
            ]
        ),
        encoding="utf-8",
    )

    result = build_index_inputs(
        chunks_path,
        manifest_path,
    )

    with pytest.raises(
        AttributeError,
    ):
        result.corpus_version = "2.0"  # type: ignore[misc]


def _build_inputs_for_embedding() -> IndexBuildInputs:
    """Return minimal valid lifecycle inputs for embedding tests."""

    first = _valid_chunk_record(
        chunk_id="chunk-a"
    )
    first["text"] = "First policy text."

    second = _valid_chunk_record(
        chunk_id="chunk-b"
    )
    second["text"] = "Second policy text."

    return IndexBuildInputs(
        corpus_version="1.2",
        chunks=(
            first,
            second,
        ),
        texts=(
            "First policy text.",
            "Second policy text.",
        ),
    )


def test_embed_index_documents_calls_embedding_layer_with_ordered_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle embedding must preserve validated text order."""

    build_inputs = (
        _build_inputs_for_embedding()
    )

    embeddings = np.zeros(
        (
            2,
            384,
        ),
        dtype=np.float32,
    )

    embed = Mock(
        return_value=embeddings
    )

    monkeypatch.setattr(
        "rag.index.embed_documents",
        embed,
    )

    result = embed_index_documents(
        build_inputs
    )

    embed.assert_called_once_with(
        build_inputs.texts
    )

    assert result.build_inputs is build_inputs
    assert result.embeddings is embeddings


def test_embed_index_documents_preserves_embedding_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding rows must remain aligned with canonical chunk count."""

    build_inputs = (
        _build_inputs_for_embedding()
    )

    embeddings = np.zeros(
        (
            2,
            384,
        ),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        "rag.index.embed_documents",
        Mock(return_value=embeddings),
    )

    result = embed_index_documents(
        build_inputs
    )

    assert result.embeddings.shape == (
        2,
        384,
    )

    assert result.embeddings.dtype == np.float32


def test_embed_index_documents_rejects_wrong_input_type() -> None:
    """Only validated lifecycle build inputs may reach embedding."""

    with pytest.raises(
        TypeError,
        match="build_inputs must be an IndexBuildInputs instance",
    ):
        embed_index_documents(
            object()  # type: ignore[arg-type]
        )


def test_embed_index_documents_rejects_cross_layer_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle must reject embeddings misaligned with chunk count."""

    build_inputs = (
        _build_inputs_for_embedding()
    )

    embeddings = np.zeros(
        (
            1,
            384,
        ),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        "rag.index.embed_documents",
        Mock(return_value=embeddings),
    )

    with pytest.raises(
        IndexLifecycleError,
        match="not aligned with canonical build inputs",
    ):
        embed_index_documents(
            build_inputs
        )


def test_index_embedding_inputs_is_immutable() -> None:
    """Embedding lifecycle container must not be reassigned."""

    build_inputs = (
        _build_inputs_for_embedding()
    )

    result = IndexEmbeddingInputs(
        build_inputs=build_inputs,
        embeddings=np.zeros(
            (
                2,
                384,
            ),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        AttributeError,
    ):
        result.build_inputs = build_inputs  # type: ignore[misc]


def _embedding_inputs_for_build() -> IndexEmbeddingInputs:
    """Return minimal embedding inputs for build-lifecycle tests."""

    build_inputs = (
        _build_inputs_for_embedding()
    )

    return IndexEmbeddingInputs(
        build_inputs=build_inputs,
        embeddings=np.zeros(
            (
                2,
                384,
            ),
            dtype=np.float32,
        ),
    )


def test_validate_index_build_preconditions_accepts_first_build(
    tmp_path: Path,
) -> None:
    """No active/build/backup state is valid for an initial build."""

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = store_module.resolve_publication_paths(
        active
    )

    result = validate_index_build_preconditions(
        paths
    )

    assert result is None


def test_validate_index_build_preconditions_accepts_existing_active(
    tmp_path: Path,
) -> None:
    """An existing active directory is valid for replacement build."""

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = store_module.resolve_publication_paths(
        active
    )

    result = validate_index_build_preconditions(
        paths
    )

    assert result is None


def test_validate_index_build_preconditions_rejects_existing_build(
    tmp_path: Path,
) -> None:
    """Stale build state must never be silently reused."""

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = store_module.resolve_publication_paths(
        active
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="build path already exists",
    ):
        validate_index_build_preconditions(
            paths
        )


def test_validate_index_build_preconditions_rejects_existing_backup(
    tmp_path: Path,
) -> None:
    """Residual backup state must block a new build."""

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = store_module.resolve_publication_paths(
        active
    )

    paths["backup"].mkdir(
        parents=True,
        exist_ok=True,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="backup path already exists",
    ):
        validate_index_build_preconditions(
            paths
        )


def test_validate_index_build_preconditions_rejects_active_file(
    tmp_path: Path,
) -> None:
    """Existing active state must be a directory."""

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    active.write_text(
        "invalid\n",
        encoding="utf-8",
    )

    paths = store_module.resolve_publication_paths(
        active
    )

    with pytest.raises(
        IndexLifecycleError,
        match="Active Chroma index path is not a directory",
    ):
        validate_index_build_preconditions(
            paths
        )


def test_build_chroma_index_composes_build_into_hidden_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lifecycle construction must target only the hidden build path."""

    embedding_inputs = (
        _embedding_inputs_for_build()
    )

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    def fake_build(
        chunks: list[dict[str, object]],
        embeddings: np.ndarray,
        chroma_dir: Path,
    ) -> None:
        assert chunks == list(
            embedding_inputs.build_inputs.chunks
        )

        assert embeddings is (
            embedding_inputs.embeddings
        )

        assert chroma_dir == (
            tmp_path
            / ".chroma_db.build"
        ).resolve()

        chroma_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    monkeypatch.setattr(
        "rag.index.build_index",
        fake_build,
    )

    result = build_chroma_index(
        embedding_inputs,
        active,
    )

    assert isinstance(
        result,
        IndexBuildState,
    )

    assert (
        result.embedding_inputs
        is embedding_inputs
    )

    assert result.publication_paths[
        "active"
    ] == active

    assert result.publication_paths[
        "build"
    ].exists()

    assert not result.publication_paths[
        "backup"
    ].exists()


def test_build_chroma_index_rejects_wrong_embedding_input_type(
    tmp_path: Path,
) -> None:
    """Only validated embedding lifecycle state may reach Chroma."""

    with pytest.raises(
        TypeError,
        match="embedding_inputs must be an IndexEmbeddingInputs instance",
    ):
        build_chroma_index(
            object(),  # type: ignore[arg-type]
            (
                tmp_path
                / "chroma_db"
            ).resolve(),
        )


def test_build_chroma_index_validates_before_build_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unsafe residual state must fail before Chroma construction."""

    embedding_inputs = (
        _embedding_inputs_for_build()
    )

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    build_path = (
        tmp_path
        / ".chroma_db.build"
    ).resolve()

    build_path.mkdir()

    build = Mock()

    monkeypatch.setattr(
        "rag.index.build_index",
        build,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="build path already exists",
    ):
        build_chroma_index(
            embedding_inputs,
            active,
        )

    build.assert_not_called()


def _build_state_for_validation(
    tmp_path: Path,
) -> IndexBuildState:
    """Return one materialized hidden build state for validation tests."""

    embedding_inputs = (
        _embedding_inputs_for_build()
    )

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    return IndexBuildState(
        embedding_inputs=embedding_inputs,
        publication_paths=paths,
    )


def test_validate_chroma_build_prepares_and_validates_persisted_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lifecycle validation must compose preparation and fresh reopen."""

    build_state = (
        _build_state_for_validation(
            tmp_path
        )
    )

    expected_records = {
        "ids": [
            "chunk-a",
            "chunk-b",
        ],
        "documents": [
            "First policy text.",
            "Second policy text.",
        ],
        "embeddings": (
            build_state
            .embedding_inputs
            .embeddings
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Test Policy",
                "section_path": [
                    "Test Policy",
                    "1. Scope",
                ],
                "source_format": "md",
                "snippet": "First policy text.",
            },
            {
                "doc_id": "HR-POL-TEST",
                "title": "Test Policy",
                "section_path": [
                    "Test Policy",
                    "1. Scope",
                ],
                "source_format": "md",
                "snippet": "Second policy text.",
            },
        ],
    }

    prepare = Mock(
        return_value=expected_records
    )

    validate = Mock()

    monkeypatch.setattr(
        "rag.index.prepare_chroma_records",
        prepare,
    )

    monkeypatch.setattr(
        "rag.index.validate_persisted_index",
        validate,
    )

    result = validate_chroma_build(
        build_state
    )

    prepare.assert_called_once()

    prepared_chunks, prepared_embeddings = (
        prepare.call_args.args
    )

    assert prepared_chunks == list(
        build_state
        .embedding_inputs
        .build_inputs
        .chunks
    )

    assert prepared_embeddings is (
        build_state
        .embedding_inputs
        .embeddings
    )

    validate.assert_called_once_with(
        build_state.publication_paths[
            "build"
        ],
        expected_records,
    )

    assert isinstance(
        result,
        IndexValidatedBuildState,
    )

    assert result.build_state is build_state
    assert result.records is expected_records


def test_validate_chroma_build_rejects_wrong_state_type() -> None:
    """Only a completed lifecycle build may enter persisted validation."""

    with pytest.raises(
        TypeError,
        match="build_state must be an IndexBuildState instance",
    ):
        validate_chroma_build(
            object()  # type: ignore[arg-type]
        )


def test_validate_chroma_build_rejects_missing_build_directory(
    tmp_path: Path,
) -> None:
    """Persisted validation must never fabricate missing build state."""

    embedding_inputs = (
        _embedding_inputs_for_build()
    )

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    state = IndexBuildState(
        embedding_inputs=embedding_inputs,
        publication_paths=paths,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="build directory does not exist",
    ):
        validate_chroma_build(
            state
        )


def test_validate_chroma_build_rejects_build_file(
    tmp_path: Path,
) -> None:
    """Persisted validation requires a real Chroma build directory."""

    embedding_inputs = (
        _embedding_inputs_for_build()
    )

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].write_text(
        "invalid\n",
        encoding="utf-8",
    )

    state = IndexBuildState(
        embedding_inputs=embedding_inputs,
        publication_paths=paths,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="build path is not a directory",
    ):
        validate_chroma_build(
            state
        )


def test_index_validated_build_state_is_immutable(
    tmp_path: Path,
) -> None:
    """Validated lifecycle state must not be reassigned."""

    build_state = (
        _build_state_for_validation(
            tmp_path
        )
    )

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": np.zeros(
            (
                1,
                384,
            ),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Test Policy",
                "section_path": [
                    "Test Policy"
                ],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    result = IndexValidatedBuildState(
        build_state=build_state,
        records=records,
    )

    with pytest.raises(
        AttributeError,
    ):
        result.build_state = build_state  # type: ignore[misc]


def _validated_state_for_semantic_smoke(
    tmp_path: Path,
) -> IndexValidatedBuildState:
    """Return minimal validated build state for semantic-smoke tests."""

    build_state = (
        _build_state_for_validation(
            tmp_path
        )
    )

    records = {
        "ids": [
            "chunk-a",
            "chunk-b",
        ],
        "documents": [
            "First policy text.",
            "Second policy text.",
        ],
        "embeddings": (
            build_state
            .embedding_inputs
            .embeddings
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Test Policy",
                "section_path": [
                    "Test Policy",
                    "1. Scope",
                ],
                "source_format": "md",
                "snippet": "First policy text.",
            },
            {
                "doc_id": "HR-POL-TEST",
                "title": "Test Policy",
                "section_path": [
                    "Test Policy",
                    "1. Scope",
                ],
                "source_format": "md",
                "snippet": "Second policy text.",
            },
        ],
    }

    return IndexValidatedBuildState(
        build_state=build_state,
        records=records,
    )


def test_semantic_smoke_contract_matches_frozen_cases() -> None:
    """Lifecycle smoke cases must remain deliberately small and fixed."""

    assert SEMANTIC_SMOKE_CASES == (
        (
            "Can I work remotely from another country?",
            "HR-POL-004",
        ),
        (
            "How much annual leave do employees receive?",
            "HR-POL-002",
        ),
    )

    assert SEMANTIC_SMOKE_N_RESULTS == 5


def test_validate_chroma_build_semantic_smoke_runs_frozen_cases_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each frozen query must use query embedding and bounded smoke."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    client = Mock()
    collection = Mock()

    query_embeddings = {
        SEMANTIC_SMOKE_CASES[0][0]: np.full(
            384,
            1.0,
            dtype=np.float32,
        ),
        SEMANTIC_SMOKE_CASES[1][0]: np.full(
            384,
            2.0,
            dtype=np.float32,
        ),
    }

    get_client = Mock(
        return_value=client
    )

    get_collection = Mock(
        return_value=collection
    )

    embed = Mock(
        side_effect=lambda query: query_embeddings[
            query
        ]
    )

    validate = Mock()

    monkeypatch.setattr(
        "rag.index.get_chroma_client",
        get_client,
    )

    monkeypatch.setattr(
        "rag.index.get_policy_collection",
        get_collection,
    )

    monkeypatch.setattr(
        "rag.index.embed_query",
        embed,
    )

    monkeypatch.setattr(
        "rag.index.validate_semantic_smoke",
        validate,
    )

    result = (
        validate_chroma_build_semantic_smoke(
            validated_state
        )
    )

    assert result is None

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    get_client.assert_called_once_with(
        build_path
    )

    get_collection.assert_called_once_with(
        client
    )

    assert embed.call_args_list == [
        (
            (
                SEMANTIC_SMOKE_CASES[
                    0
                ][0],
            ),
        ),
        (
            (
                SEMANTIC_SMOKE_CASES[
                    1
                ][0],
            ),
        ),
    ]

    assert validate.call_count == 2

    first_call = (
        validate.call_args_list[0]
    )

    assert first_call.args == (
        collection,
        query_embeddings[
            SEMANTIC_SMOKE_CASES[
                0
            ][0]
        ],
    )

    assert first_call.kwargs == {
        "expected_doc_id": "HR-POL-004",
        "n_results": 5,
    }

    second_call = (
        validate.call_args_list[1]
    )

    assert second_call.args == (
        collection,
        query_embeddings[
            SEMANTIC_SMOKE_CASES[
                1
            ][0]
        ],
    )

    assert second_call.kwargs == {
        "expected_doc_id": "HR-POL-002",
        "n_results": 5,
    }


def test_validate_chroma_build_semantic_smoke_rejects_wrong_state_type() -> None:
    """Only persisted validated state may enter semantic smoke."""

    with pytest.raises(
        TypeError,
        match=(
            "validated_state must be an "
            "IndexValidatedBuildState instance"
        ),
    ):
        validate_chroma_build_semantic_smoke(
            object()  # type: ignore[arg-type]
        )


def test_validate_chroma_build_semantic_smoke_rejects_missing_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing build state must fail before Chroma client creation."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    build_path.rmdir()

    get_client = Mock()

    monkeypatch.setattr(
        "rag.index.get_chroma_client",
        get_client,
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "build directory does not exist "
            "before semantic smoke validation"
        ),
    ):
        validate_chroma_build_semantic_smoke(
            validated_state
        )

    get_client.assert_not_called()


def test_validate_chroma_build_semantic_smoke_propagates_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed frozen smoke case must stop lifecycle validation."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    client = Mock()
    collection = Mock()

    failure = store_module.ChromaStoreError(
        "synthetic semantic smoke failure"
    )

    monkeypatch.setattr(
        "rag.index.get_chroma_client",
        Mock(return_value=client),
    )

    monkeypatch.setattr(
        "rag.index.get_policy_collection",
        Mock(return_value=collection),
    )

    monkeypatch.setattr(
        "rag.index.embed_query",
        Mock(
            return_value=np.zeros(
                384,
                dtype=np.float32,
            )
        ),
    )

    validate = Mock(
        side_effect=failure
    )

    monkeypatch.setattr(
        "rag.index.validate_semantic_smoke",
        validate,
    )

    with pytest.raises(
        store_module.ChromaStoreError,
    ) as exc_info:
        validate_chroma_build_semantic_smoke(
            validated_state
        )

    assert exc_info.value is failure

    validate.assert_called_once()


def test_finalize_chroma_build_metadata_composes_metadata_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validated build metadata must be written then checked current."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    metadata = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
        "created": "2026-08-12T07:32:57Z",
    }

    payload = (
        b'{"metadata":"synthetic"}\n'
    )

    build_metadata = Mock(
        return_value=metadata
    )

    serialize = Mock(
        return_value=payload
    )

    write = Mock()

    current = Mock(
        return_value=True
    )

    monkeypatch.setattr(
        "rag.index.build_index_metadata",
        build_metadata,
    )

    monkeypatch.setattr(
        "rag.index.serialize_index_metadata",
        serialize,
    )

    monkeypatch.setattr(
        "rag.index.write_index_metadata_atomic",
        write,
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        current,
    )

    result = finalize_chroma_build_metadata(
        validated_state
    )

    assert result is None

    build_metadata.assert_called_once_with(
        corpus_version="1.2"
    )

    serialize.assert_called_once_with(
        metadata
    )

    write.assert_called_once_with(
        build_path,
        payload,
    )

    current.assert_called_once_with(
        build_path,
        corpus_version="1.2",
        embedding_model=(
            store_module.EMBEDDING_MODEL_NAME
        ),
        embedding_dimension=(
            store_module.EMBEDDING_DIMENSION
        ),
        chunk_tokens=(
            store_module.TARGET_CHUNK_TOKENS
        ),
        chunk_overlap=(
            store_module.CHUNK_OVERLAP_TOKENS
        ),
    )

def test_finalize_chroma_build_metadata_rejects_wrong_state_type() -> None:
    """Only exhaustively validated build state may be finalized."""

    with pytest.raises(
        TypeError,
        match=(
            "validated_state must be an "
            "IndexValidatedBuildState instance"
        ),
    ):
        finalize_chroma_build_metadata(
            object()  # type: ignore[arg-type]
        )


def test_finalize_chroma_build_metadata_rejects_missing_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing build state must fail before metadata construction."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    build_path.rmdir()

    build_metadata = Mock()

    monkeypatch.setattr(
        "rag.index.build_index_metadata",
        build_metadata,
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "build directory does not exist "
            "before metadata finalization"
        ),
    ):
        finalize_chroma_build_metadata(
            validated_state
        )

    build_metadata.assert_not_called()


def test_finalize_chroma_build_metadata_rejects_existing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata must appear only at the final validated build stage."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    metadata_path = (
        build_path
        / "index_metadata.json"
    )

    metadata_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    build_metadata = Mock()

    monkeypatch.setattr(
        "rag.index.build_index_metadata",
        build_metadata,
    )

    with pytest.raises(
        IndexLifecycleError,
        match="metadata already exists before finalization",
    ):
        finalize_chroma_build_metadata(
            validated_state
        )

    build_metadata.assert_not_called()


def test_finalize_chroma_build_metadata_rejects_failed_freshness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A written metadata artifact must make the build current."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    metadata = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
        "created": "2026-08-12T07:32:57Z",
    }

    monkeypatch.setattr(
        "rag.index.build_index_metadata",
        Mock(return_value=metadata),
    )

    monkeypatch.setattr(
        "rag.index.serialize_index_metadata",
        Mock(return_value=b"{}\n"),
    )

    monkeypatch.setattr(
        "rag.index.write_index_metadata_atomic",
        Mock(),
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        Mock(return_value=False),
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "does not match the expected "
            "freshness configuration"
        ),
    ):
        finalize_chroma_build_metadata(
            validated_state
        )


def test_finalize_chroma_build_metadata_does_not_reopen_or_query_chroma(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata finalization must perform no post-smoke Chroma access."""

    validated_state = (
        _validated_state_for_semantic_smoke(
            tmp_path
        )
    )

    monkeypatch.setattr(
        "rag.index.build_index_metadata",
        Mock(
            return_value={
                "corpus_version": "1.2",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimension": 384,
                "chunk_tokens": 350,
                "chunk_overlap": 50,
                "created": "2026-08-12T07:32:57Z",
            }
        ),
    )

    monkeypatch.setattr(
        "rag.index.serialize_index_metadata",
        Mock(return_value=b"{}\n"),
    )

    monkeypatch.setattr(
        "rag.index.write_index_metadata_atomic",
        Mock(),
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        Mock(return_value=True),
    )

    get_client = Mock()
    get_collection = Mock()
    semantic_smoke = Mock()

    monkeypatch.setattr(
        "rag.index.get_chroma_client",
        get_client,
    )

    monkeypatch.setattr(
        "rag.index.get_policy_collection",
        get_collection,
    )

    monkeypatch.setattr(
        "rag.index.validate_semantic_smoke",
        semantic_smoke,
    )

    finalize_chroma_build_metadata(
        validated_state
    )

    get_client.assert_not_called()
    get_collection.assert_not_called()
    semantic_smoke.assert_not_called()


def test_build_policy_index_composes_lifecycle_in_required_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Full build phase must compose verified stages in exact order."""

    chunks_path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    build_inputs = Mock(
        spec=IndexBuildInputs
    )

    embedding_inputs = Mock(
        spec=IndexEmbeddingInputs
    )

    build_state = Mock(
        spec=IndexBuildState
    )

    validated_state = Mock(
        spec=IndexValidatedBuildState
    )

    calls: list[str] = []

    def fake_build_inputs(
        received_chunks_path: Path,
        received_manifest_path: Path,
    ) -> object:
        assert received_chunks_path == chunks_path
        assert received_manifest_path == manifest_path

        calls.append(
            "build_index_inputs"
        )

        return build_inputs

    def fake_embed(
        received: object,
    ) -> object:
        assert received is build_inputs

        calls.append(
            "embed_index_documents"
        )

        return embedding_inputs

    def fake_build_chroma(
        received_inputs: object,
        received_active: Path,
    ) -> object:
        assert received_inputs is embedding_inputs
        assert received_active == active

        calls.append(
            "build_chroma_index"
        )

        return build_state

    def fake_validate(
        received: object,
    ) -> object:
        assert received is build_state

        calls.append(
            "validate_chroma_build"
        )

        return validated_state

    def fake_semantic(
        received: object,
    ) -> None:
        assert received is validated_state

        calls.append(
            "validate_chroma_build_semantic_smoke"
        )

    def fake_finalize(
        received: object,
    ) -> None:
        assert received is validated_state

        calls.append(
            "finalize_chroma_build_metadata"
        )

    monkeypatch.setattr(
        "rag.index.build_index_inputs",
        fake_build_inputs,
    )

    monkeypatch.setattr(
        "rag.index.embed_index_documents",
        fake_embed,
    )

    monkeypatch.setattr(
        "rag.index.build_chroma_index",
        fake_build_chroma,
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build",
        fake_validate,
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build_semantic_smoke",
        fake_semantic,
    )

    monkeypatch.setattr(
        "rag.index.finalize_chroma_build_metadata",
        fake_finalize,
    )

    result = build_policy_index(
        chunks_path,
        manifest_path,
        active,
    )

    assert calls == [
        "build_index_inputs",
        "embed_index_documents",
        "build_chroma_index",
        "validate_chroma_build",
        "validate_chroma_build_semantic_smoke",
        "finalize_chroma_build_metadata",
    ]

    assert isinstance(
        result,
        IndexFinalizedBuildState,
    )

    assert result.validated_state is validated_state


def test_build_policy_index_does_not_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Build phase must stop before C8 process-boundary publication."""

    chunks_path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    build_inputs = Mock(
        spec=IndexBuildInputs
    )

    embedding_inputs = Mock(
        spec=IndexEmbeddingInputs
    )

    build_state = Mock(
        spec=IndexBuildState
    )

    validated_state = Mock(
        spec=IndexValidatedBuildState
    )

    monkeypatch.setattr(
        "rag.index.build_index_inputs",
        Mock(return_value=build_inputs),
    )

    monkeypatch.setattr(
        "rag.index.embed_index_documents",
        Mock(return_value=embedding_inputs),
    )

    monkeypatch.setattr(
        "rag.index.build_chroma_index",
        Mock(return_value=build_state),
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build",
        Mock(return_value=validated_state),
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build_semantic_smoke",
        Mock(),
    )

    monkeypatch.setattr(
        "rag.index.finalize_chroma_build_metadata",
        Mock(),
    )

    publish = Mock()

    monkeypatch.setattr(
        store_module,
        "publish_index",
        publish,
    )

    build_policy_index(
        chunks_path,
        manifest_path,
        active,
    )

    publish.assert_not_called()

def test_build_policy_index_stops_before_metadata_when_semantic_smoke_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata must never be written after failed semantic validation."""

    chunks_path = (
        tmp_path
        / "chunks.json"
    ).resolve()

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    build_inputs = Mock(
        spec=IndexBuildInputs
    )

    embedding_inputs = Mock(
        spec=IndexEmbeddingInputs
    )

    build_state = Mock(
        spec=IndexBuildState
    )

    validated_state = Mock(
        spec=IndexValidatedBuildState
    )

    monkeypatch.setattr(
        "rag.index.build_index_inputs",
        Mock(return_value=build_inputs),
    )

    monkeypatch.setattr(
        "rag.index.embed_index_documents",
        Mock(return_value=embedding_inputs),
    )

    monkeypatch.setattr(
        "rag.index.build_chroma_index",
        Mock(return_value=build_state),
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build",
        Mock(return_value=validated_state),
    )

    failure = store_module.ChromaStoreError(
        "synthetic semantic failure"
    )

    monkeypatch.setattr(
        "rag.index.validate_chroma_build_semantic_smoke",
        Mock(side_effect=failure),
    )

    finalize = Mock()

    monkeypatch.setattr(
        "rag.index.finalize_chroma_build_metadata",
        finalize,
    )

    with pytest.raises(
        store_module.ChromaStoreError,
    ) as exc_info:
        build_policy_index(
            chunks_path,
            manifest_path,
            active,
        )

    assert exc_info.value is failure

    finalize.assert_not_called()


def test_index_finalized_build_state_is_immutable() -> None:
    """Finalized build state must not be reassigned."""

    validated_state = Mock(
        spec=IndexValidatedBuildState
    )

    result = IndexFinalizedBuildState(
        validated_state=validated_state
    )

    with pytest.raises(
        AttributeError,
    ):
        result.validated_state = validated_state  # type: ignore[misc]


def _write_test_manifest(
    path: Path,
) -> None:
    """Write one minimal valid manifest using the current corpus schema."""

    path.write_text(
        """
{
  "version": "1.2",
  "created": "2026-08-05",
  "documents": [
    {
      "doc_id": "HR-POL-TEST",
      "title": "Test Policy",
      "format": "md",
      "doc_version": "1.0",
      "effective_date": "2026-01-01"
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

def test_publish_policy_index_delegates_current_build_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A current prepared build must delegate once to C8 publication."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = Mock()
    manifest.version = "1.2"

    load_manifest = Mock(
        return_value=manifest
    )

    current = Mock(
        return_value=True
    )

    publish = Mock(
        return_value=True
    )

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        load_manifest,
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        current,
    )

    monkeypatch.setattr(
        "rag.index.publish_index",
        publish,
    )

    result = publish_policy_index(
        manifest_path,
        active,
    )

    assert result is True

    load_manifest.assert_called_once_with(
        manifest_path
    )

    current.assert_called_once_with(
        paths["build"],
        corpus_version="1.2",
        embedding_model=EMBEDDING_MODEL_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
        chunk_tokens=TARGET_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    publish.assert_called_once_with(
        paths
    )


def test_publish_policy_index_rejects_stale_build_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stale metadata must block publication before any directory move."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = Mock()
    manifest.version = "1.2"

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        Mock(return_value=manifest),
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        Mock(return_value=False),
    )

    publish = Mock()

    monkeypatch.setattr(
        "rag.index.publish_index",
        publish,
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "Prepared Chroma build is not current"
        ),
    ):
        publish_policy_index(
            manifest_path,
            active,
        )

    publish.assert_not_called()


def test_publish_policy_index_rejects_missing_build_before_freshness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing prepared build must fail before freshness evaluation."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    manifest = Mock()
    manifest.version = "1.2"

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        Mock(return_value=manifest),
    )

    current = Mock()
    publish = Mock()

    monkeypatch.setattr(
        "rag.index.is_index_current",
        current,
    )

    monkeypatch.setattr(
        "rag.index.publish_index",
        publish,
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "Prepared Chroma build directory "
            "does not exist"
        ),
    ):
        publish_policy_index(
            manifest_path,
            active,
        )

    current.assert_not_called()
    publish.assert_not_called()


@pytest.mark.parametrize(
    "publication_result",
    [
        True,
        False,
    ],
)
def test_publish_policy_index_preserves_publication_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publication_result: bool,
) -> None:
    """C9 must preserve the exact C8 publication result semantics."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = Mock()
    manifest.version = "1.2"

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        Mock(return_value=manifest),
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        Mock(return_value=True),
    )

    monkeypatch.setattr(
        "rag.index.publish_index",
        Mock(
            return_value=publication_result
        ),
    )

    result = publish_policy_index(
        manifest_path,
        active,
    )

    assert result is publication_result


@pytest.mark.parametrize(
    (
        "manifest_path",
        "active_chroma_dir",
        "message",
    ),
    [
        (
            "corpus/version.json",
            Path("/tmp/chroma_db"),
            "manifest_path must be a pathlib.Path instance",
        ),
        (
            Path("corpus/version.json"),
            Path("/tmp/chroma_db"),
            "manifest_path must be absolute",
        ),
        (
            Path("/tmp/version.json"),
            "chroma_db",
            "active_chroma_dir must be a pathlib.Path instance",
        ),
        (
            Path("/tmp/version.json"),
            Path("chroma_db"),
            "active_chroma_dir must be absolute",
        ),
    ],
)
def test_publish_policy_index_rejects_invalid_paths(
    manifest_path: object,
    active_chroma_dir: object,
    message: str,
) -> None:
    """Publication-process paths must use explicit absolute Path values."""

    with pytest.raises(
        (TypeError, ValueError),
        match=message,
    ):
        publish_policy_index(
            manifest_path,  # type: ignore[arg-type]
            active_chroma_dir,  # type: ignore[arg-type]
        )


def test_publish_policy_index_does_not_access_chroma_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publication process must remain metadata/filesystem only."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = Mock()
    manifest.version = "1.2"

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        Mock(return_value=manifest),
    )

    monkeypatch.setattr(
        "rag.index.is_index_current",
        Mock(return_value=True),
    )

    monkeypatch.setattr(
        "rag.index.publish_index",
        Mock(return_value=True),
    )

    get_client = Mock()
    get_collection = Mock()
    embed_documents_call = Mock()
    embed_query_call = Mock()
    validate_persisted = Mock()
    validate_semantic = Mock()

    monkeypatch.setattr(
        "rag.index.get_chroma_client",
        get_client,
    )

    monkeypatch.setattr(
        "rag.index.get_policy_collection",
        get_collection,
    )

    monkeypatch.setattr(
        "rag.index.embed_documents",
        embed_documents_call,
    )

    monkeypatch.setattr(
        "rag.index.embed_query",
        embed_query_call,
    )

    monkeypatch.setattr(
        "rag.index.validate_persisted_index",
        validate_persisted,
    )

    monkeypatch.setattr(
        "rag.index.validate_semantic_smoke",
        validate_semantic,
    )

    publish_policy_index(
        manifest_path,
        active,
    )

    get_client.assert_not_called()
    get_collection.assert_not_called()
    embed_documents_call.assert_not_called()
    embed_query_call.assert_not_called()
    validate_persisted.assert_not_called()
    validate_semantic.assert_not_called()


def test_publish_policy_index_rejects_build_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepared build state must be a directory, never a regular file."""

    manifest_path = (
        tmp_path
        / "version.json"
    ).resolve()

    active = (
        tmp_path
        / "chroma_db"
    ).resolve()

    paths = (
        store_module.resolve_publication_paths(
            active
        )
    )

    paths["build"].write_text(
        "invalid\n",
        encoding="utf-8",
    )

    manifest = Mock()
    manifest.version = "1.2"

    monkeypatch.setattr(
        "rag.ingest.load_manifest",
        Mock(return_value=manifest),
    )

    current = Mock()
    publish = Mock()

    monkeypatch.setattr(
        "rag.index.is_index_current",
        current,
    )

    monkeypatch.setattr(
        "rag.index.publish_index",
        publish,
    )

    with pytest.raises(
        IndexLifecycleError,
        match=(
            "Prepared Chroma build path is not a directory"
        ),
    ):
        publish_policy_index(
            manifest_path,
            active,
        )

    current.assert_not_called()
    publish.assert_not_called()
