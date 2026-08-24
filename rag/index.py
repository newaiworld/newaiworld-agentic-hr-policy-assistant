"""Offline index-lifecycle orchestration for the S4 RAG pipeline.

This module composes already-verified ingestion, embedding, and Chroma
storage primitives into the offline policy-index lifecycle.

C9.2 introduces only the canonical chunks.json loading boundary.
Embedding, index construction, validation, metadata publication, and
safe directory publication are added by later C9 checkpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path


import numpy as np

from rag.chunk import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    TARGET_CHUNK_TOKENS,
)


from rag.embed import (
    EMBEDDING_DIMENSION,
    embed_documents,
    embed_query,
)


from rag.store import (
    ChromaRecords,
    PublicationPaths,
    build_index,
    build_index_metadata,
    get_chroma_client,
    get_policy_collection,
    is_index_current,
    prepare_chroma_records,
    publish_index,
    resolve_chroma_dir,
    resolve_publication_paths,
    serialize_index_metadata,
    validate_persisted_index,
    validate_semantic_smoke,
    write_index_metadata_atomic,
)

SEMANTIC_SMOKE_CASES: tuple[
    tuple[str, str],
    ...,
] = (
    (
        "Can I work remotely from another country?",
        "HR-POL-004",
    ),
    (
        "How much annual leave do employees receive?",
        "HR-POL-002",
    ),
)

SEMANTIC_SMOKE_N_RESULTS: int = 5


class IndexLifecycleError(RuntimeError):
    """Base exception for offline index-lifecycle failures."""


@dataclass(frozen=True)
class IndexBuildInputs:
    """Validated canonical inputs for one offline Chroma index build."""

    corpus_version: str
    chunks: tuple[dict[str, object], ...]
    texts: tuple[str, ...]


@dataclass(frozen=True)
class IndexEmbeddingInputs:
    """Validated build inputs aligned with document embeddings."""

    build_inputs: IndexBuildInputs
    embeddings: np.ndarray


@dataclass(frozen=True)
class IndexBuildState:
    """Prepared filesystem state for one Chroma index build."""

    embedding_inputs: IndexEmbeddingInputs
    publication_paths: PublicationPaths


@dataclass(frozen=True)
class IndexValidatedBuildState:
    """Chroma build state paired with exhaustively validated records."""

    build_state: IndexBuildState
    records: ChromaRecords

@dataclass(frozen=True)
class IndexFinalizedBuildState:
    """Completed hidden build ready for later process-boundary publication."""

    validated_state: IndexValidatedBuildState


def load_canonical_chunks(
    path: Path,
) -> list[dict[str, object]]:
    """Load the canonical processed chunk artifact from JSON.

    This function performs only artifact-level loading validation.
    It deliberately does not validate the canonical chunk-field schema;
    that build-input contract belongs to a later C9 checkpoint.

    Args:
        path:
            Absolute path to the canonical ``chunks.json`` artifact.

    Returns:
        Ordered list of chunk-record dictionaries exactly as represented
        by the JSON artifact.

    Raises:
        TypeError:
            If ``path`` is not a ``pathlib.Path``.
        ValueError:
            If ``path`` is not absolute.
        IndexLifecycleError:
            If the artifact does not exist, is not a regular file,
            cannot be read as UTF-8, contains invalid JSON, has a
            non-list top-level value, is empty, or contains a
            non-dictionary item.
    """

    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path must be a pathlib.Path instance."
        )

    if not path.is_absolute():
        raise ValueError(
            "path must be absolute."
        )

    if not path.exists():
        raise IndexLifecycleError(
            "Canonical chunks artifact does not exist: "
            f"{str(path)!r}."
        )

    if not path.is_file():
        raise IndexLifecycleError(
            "Canonical chunks artifact is not a regular file: "
            f"{str(path)!r}."
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise IndexLifecycleError(
            "Canonical chunks artifact is not valid UTF-8: "
            f"{str(path)!r}."
        ) from exc
    except OSError as exc:
        raise IndexLifecycleError(
            "Canonical chunks artifact could not be read: "
            f"{str(path)!r}."
        ) from exc

    try:
        raw_data = json.loads(
            text
        )
    except JSONDecodeError as exc:
        raise IndexLifecycleError(
            "Canonical chunks artifact contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc

    if not isinstance(
        raw_data,
        list,
    ):
        raise IndexLifecycleError(
            "Canonical chunks artifact must contain a top-level list."
        )

    if not raw_data:
        raise IndexLifecycleError(
            "Canonical chunks artifact must contain at least one record."
        )

    records: list[
        dict[str, object]
    ] = []

    for index, item in enumerate(
        raw_data
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise IndexLifecycleError(
                "Canonical chunks artifact contains a non-dictionary "
                f"record at index {index}."
            )

        records.append(
            item
        )

    return records

def validate_canonical_chunk_records(
    records: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate the frozen canonical chunk-record schema.

    Record order is preserved exactly. Validation ensures that each
    record matches the deterministic schema produced by
    ``rag.chunk.chunk_to_record`` and that chunk IDs are unique.

    Args:
        records:
            Ordered records loaded from canonical ``chunks.json``.

    Returns:
        Immutable ordered tuple containing the validated records.

    Raises:
        TypeError:
            If ``records`` is not a list.
        IndexLifecycleError:
            If the list is empty, a record has the wrong schema,
            a field has an invalid value, or chunk IDs are duplicated.
    """

    if not isinstance(
        records,
        list,
    ):
        raise TypeError(
            "records must be a list."
        )

    if not records:
        raise IndexLifecycleError(
            "Canonical chunk records must not be empty."
        )

    required_keys = {
        "chunk_id",
        "doc_id",
        "title",
        "section_path",
        "section_order",
        "chunk_index",
        "text",
        "token_count",
        "source_format",
    }

    seen_chunk_ids: set[str] = set()

    for index, record in enumerate(
        records
    ):
        if not isinstance(
            record,
            dict,
        ):
            raise IndexLifecycleError(
                "Canonical chunk record is not a dictionary at "
                f"index {index}."
            )

        if set(record) != required_keys:
            raise IndexLifecycleError(
                "Canonical chunk record has unexpected schema at "
                f"index {index}."
            )

        for field_name in (
            "chunk_id",
            "doc_id",
            "title",
            "text",
        ):
            value = record[field_name]

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise IndexLifecycleError(
                    "Canonical chunk field "
                    f"{field_name!r} must be a non-empty string "
                    f"at index {index}."
                )

        chunk_id = record["chunk_id"]

        if chunk_id in seen_chunk_ids:
            raise IndexLifecycleError(
                "Canonical chunk IDs must be unique; duplicate "
                f"{chunk_id!r} at index {index}."
            )

        seen_chunk_ids.add(
            chunk_id
        )

        section_path = record[
            "section_path"
        ]

        if (
            not isinstance(
                section_path,
                list,
            )
            or not section_path
            or any(
                not isinstance(item, str)
                or not item.strip()
                for item in section_path
            )
        ):
            raise IndexLifecycleError(
                "Canonical chunk field 'section_path' must be a "
                "non-empty list of non-empty strings at "
                f"index {index}."
            )

        for field_name in (
            "section_order",
            "chunk_index",
        ):
            value = record[field_name]

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise IndexLifecycleError(
                    "Canonical chunk field "
                    f"{field_name!r} must be a non-negative integer "
                    f"at index {index}."
                )

        token_count = record[
            "token_count"
        ]

        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise IndexLifecycleError(
                "Canonical chunk field 'token_count' must be a "
                "positive integer at "
                f"index {index}."
            )

        source_format = record[
            "source_format"
        ]

        if source_format not in {
            "md",
            "pdf",
        }:
            raise IndexLifecycleError(
                "Canonical chunk field 'source_format' must be "
                "'md' or 'pdf' at "
                f"index {index}."
            )

    return tuple(
        records
    )

def build_index_inputs(
    chunks_path: Path,
    manifest_path: Path,
) -> IndexBuildInputs:
    """Build validated immutable inputs for the offline index lifecycle.

    Canonical chunks are loaded from the committed processed artifact.
    Corpus version is obtained through the existing validated manifest
    loader rather than by reparsing ``version.json`` independently.

    Args:
        chunks_path:
            Absolute path to canonical ``chunks.json``.
        manifest_path:
            Absolute path to authoritative corpus ``version.json``.

    Returns:
        Validated corpus version, ordered canonical records, and ordered
        document texts for later embedding.

    Raises:
        TypeError:
            If either path violates an existing path contract.
        ValueError:
            If either path is not absolute.
        IndexLifecycleError:
            If canonical chunk validation fails.
        ManifestValidationError:
            If the corpus manifest is invalid.
    """

    if not isinstance(
        manifest_path,
        Path,
    ):
        raise TypeError(
            "manifest_path must be a pathlib.Path instance."
        )

    if not manifest_path.is_absolute():
        raise ValueError(
            "manifest_path must be absolute."
        )

    from rag.ingest import load_manifest

    records = load_canonical_chunks(
        chunks_path
    )

    validated_chunks = (
        validate_canonical_chunk_records(
            records
        )
    )

    manifest = load_manifest(
        manifest_path
    )

    corpus_version = (
        manifest.version.strip()
    )

    if not corpus_version:
        raise IndexLifecycleError(
            "Corpus manifest version must be non-empty."
        )

    texts = tuple(
        record["text"]
        for record in validated_chunks
    )

    return IndexBuildInputs(
        corpus_version=corpus_version,
        chunks=validated_chunks,
        texts=texts,
    )

def embed_index_documents(
    build_inputs: IndexBuildInputs,
) -> IndexEmbeddingInputs:
    """Embed validated canonical texts for one offline index build.

    Text order is inherited directly from ``IndexBuildInputs.texts``.
    The embedding layer is responsible for model loading, batching,
    normalization, shape validation, floating-point validation, and
    finite-value validation.

    Args:
        build_inputs:
            Validated immutable canonical build inputs.

    Returns:
        Build inputs paired with the aligned normalized embedding matrix.

    Raises:
        TypeError:
            If ``build_inputs`` is not an ``IndexBuildInputs`` instance.
        IndexLifecycleError:
            If the returned embedding matrix is not aligned with the
            validated chunk/text count.
        EmbeddingError:
            If the underlying embedding layer fails.
    """

    if not isinstance(
        build_inputs,
        IndexBuildInputs,
    ):
        raise TypeError(
            "build_inputs must be an IndexBuildInputs instance."
        )

    embeddings = embed_documents(
        build_inputs.texts
    )

    expected_shape = (
        len(build_inputs.chunks),
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise IndexLifecycleError(
            "Document embeddings are not aligned with canonical "
            "build inputs: "
            f"{embeddings.shape!r} != {expected_shape!r}."
        )

    return IndexEmbeddingInputs(
        build_inputs=build_inputs,
        embeddings=embeddings,
    )


def validate_index_build_preconditions(
    paths: PublicationPaths,
) -> None:
    """Validate filesystem state before constructing a new build index.

    Build preparation is conservative. Existing build or backup state is
    rejected rather than deleted or reused automatically because either
    may represent an interrupted earlier lifecycle.

    The active index may be absent for a first build or present as a
    directory for a replacement build.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Raises:
        TypeError:
            If ``paths`` is not a dictionary.
        IndexLifecycleError:
            If the mapping shape is invalid, any path is invalid, an
            existing active path is not a directory, or build/backup
            state already exists.
    """

    if not isinstance(
        paths,
        dict,
    ):
        raise TypeError(
            "paths must be a dictionary."
        )

    required_keys = {
        "active",
        "build",
        "backup",
    }

    if set(paths) != required_keys:
        raise IndexLifecycleError(
            "Index build paths must contain exactly "
            "'active', 'build', and 'backup'."
        )

    for field_name in (
        "active",
        "build",
        "backup",
    ):
        value = paths[field_name]

        if not isinstance(
            value,
            Path,
        ):
            raise IndexLifecycleError(
                "Index build path "
                f"{field_name!r} must be a pathlib.Path."
            )

        if not value.is_absolute():
            raise IndexLifecycleError(
                "Index build path "
                f"{field_name!r} must be absolute."
            )

    active = paths["active"]
    build = paths["build"]
    backup = paths["backup"]

    if active.exists() and not active.is_dir():
        raise IndexLifecycleError(
            "Active Chroma index path is not a directory: "
            f"{str(active)!r}."
        )

    if build.exists():
        raise IndexLifecycleError(
            "Chroma build path already exists: "
            f"{str(build)!r}."
        )

    if backup.exists():
        raise IndexLifecycleError(
            "Chroma backup path already exists: "
            f"{str(backup)!r}."
        )


def build_chroma_index(
    embedding_inputs: IndexEmbeddingInputs,
    active_chroma_dir: Path,
) -> IndexBuildState:
    """Construct one new Chroma index in the hidden build directory.

    This function performs only build-path preparation and Chroma
    construction. Structural validation, persisted reopen validation,
    semantic smoke, metadata publication, and active-index publication
    belong to later C9 checkpoints.

    Args:
        embedding_inputs:
            Validated canonical inputs paired with document embeddings.
        active_chroma_dir:
            Absolute final active Chroma directory. The corresponding
            hidden build and backup sibling paths are derived from it.

    Returns:
        Immutable build state containing the original embedding inputs
        and resolved publication paths.

    Raises:
        TypeError:
            If either input violates its type contract.
        ValueError:
            If ``active_chroma_dir`` is not absolute.
        IndexLifecycleError:
            If build-start filesystem state is unsafe.
        ChromaStoreError:
            If Chroma construction fails.
    """

    if not isinstance(
        embedding_inputs,
        IndexEmbeddingInputs,
    ):
        raise TypeError(
            "embedding_inputs must be an "
            "IndexEmbeddingInputs instance."
        )

    if not isinstance(
        active_chroma_dir,
        Path,
    ):
        raise TypeError(
            "active_chroma_dir must be a pathlib.Path instance."
        )

    if not active_chroma_dir.is_absolute():
        raise ValueError(
            "active_chroma_dir must be absolute."
        )

    paths = resolve_publication_paths(
        active_chroma_dir
    )

    validate_index_build_preconditions(
        paths
    )

    build_index(
        list(
            embedding_inputs.build_inputs.chunks
        ),
        embedding_inputs.embeddings,
        paths["build"],
    )

    if not paths["build"].exists():
        raise IndexLifecycleError(
            "Chroma build completed without creating the build "
            f"directory {str(paths['build'])!r}."
        )

    if not paths["build"].is_dir():
        raise IndexLifecycleError(
            "Chroma build path is not a directory after construction: "
            f"{str(paths['build'])!r}."
        )

    return IndexBuildState(
        embedding_inputs=embedding_inputs,
        publication_paths=paths,
    )


def validate_chroma_build(
    build_state: IndexBuildState,
) -> IndexValidatedBuildState:
    """Exhaustively validate one persisted hidden Chroma build.

    Canonical Chroma records are reconstructed from the already-validated
    lifecycle chunks and embeddings. The hidden build directory is then
    reopened through the existing persistence-validation path, which
    creates a fresh Chroma client and performs exhaustive ID-keyed
    integrity validation.

    Args:
        build_state:
            Completed hidden Chroma build and its aligned lifecycle
            inputs.

    Returns:
        Immutable state containing the validated build plus the prepared
        Chroma records used for exhaustive comparison.

    Raises:
        TypeError:
            If ``build_state`` is not an ``IndexBuildState``.
        IndexLifecycleError:
            If the expected hidden build directory is absent or is not a
            directory.
        ChromaStoreError:
            If Chroma record preparation or persisted exhaustive
            validation fails.
    """

    if not isinstance(
        build_state,
        IndexBuildState,
    ):
        raise TypeError(
            "build_state must be an IndexBuildState instance."
        )

    build_path = (
        build_state.publication_paths[
            "build"
        ]
    )

    if not build_path.exists():
        raise IndexLifecycleError(
            "Chroma build directory does not exist before persisted "
            f"validation: {str(build_path)!r}."
        )

    if not build_path.is_dir():
        raise IndexLifecycleError(
            "Chroma build path is not a directory before persisted "
            f"validation: {str(build_path)!r}."
        )

    embedding_inputs = (
        build_state.embedding_inputs
    )

    records = prepare_chroma_records(
        list(
            embedding_inputs
            .build_inputs
            .chunks
        ),
        embedding_inputs.embeddings,
    )

    validate_persisted_index(
        build_path,
        records,
    )

    return IndexValidatedBuildState(
        build_state=build_state,
        records=records,
    )


def validate_chroma_build_semantic_smoke(
    validated_state: IndexValidatedBuildState,
) -> None:
    """Run the frozen semantic smoke suite against one validated build.

    The hidden Chroma build is opened only after persisted exhaustive
    validation has succeeded. Each frozen query is embedded through the
    verified query-embedding path and delegated to the existing bounded
    semantic-smoke validator.

    This function is deliberately not a retrieval-quality evaluation.
    It performs no score conversion, thresholds, filtering, reranking,
    recall measurement, or retrieval abstraction.

    Args:
        validated_state:
            Persisted and exhaustively validated hidden Chroma build.

    Raises:
        TypeError:
            If ``validated_state`` is not an
            ``IndexValidatedBuildState``.
        IndexLifecycleError:
            If the hidden build directory no longer exists or is not a
            directory.
        EmbeddingError:
            If query embedding fails.
        ChromaStoreError:
            If the build collection cannot be opened or any frozen
            semantic smoke case fails.
    """

    if not isinstance(
        validated_state,
        IndexValidatedBuildState,
    ):
        raise TypeError(
            "validated_state must be an "
            "IndexValidatedBuildState instance."
        )

    build_path = (
        validated_state
        .build_state
        .publication_paths[
            "build"
        ]
    )

    if not build_path.exists():
        raise IndexLifecycleError(
            "Chroma build directory does not exist before semantic "
            f"smoke validation: {str(build_path)!r}."
        )

    if not build_path.is_dir():
        raise IndexLifecycleError(
            "Chroma build path is not a directory before semantic "
            f"smoke validation: {str(build_path)!r}."
        )

    client = get_chroma_client(
        build_path
    )

    collection = get_policy_collection(
        client
    )

    for (
        query,
        expected_doc_id,
    ) in SEMANTIC_SMOKE_CASES:
        query_embedding = embed_query(
            query
        )

        validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id=expected_doc_id,
            n_results=SEMANTIC_SMOKE_N_RESULTS,
        )


def finalize_chroma_build_metadata(
    validated_state: IndexValidatedBuildState,
) -> None:
    """Write index metadata last and verify hidden-build freshness.

    This lifecycle stage must run only after persisted exhaustive
    validation and the frozen semantic-smoke suite have succeeded.

    Metadata is written inside the hidden Chroma build directory so it
    is published together with the validated index by the later
    process-boundary directory publication phase.

    After metadata publication this function performs only a read-only
    freshness check. It does not mutate vector contents, reopen the
    collection, perform semantic retrieval, or publish the build.

    Args:
        validated_state:
            Persisted and exhaustively validated hidden Chroma build.
            The caller is responsible for invoking semantic-smoke
            validation before this metadata-finalization stage.

    Raises:
        TypeError:
            If ``validated_state`` is not an
            ``IndexValidatedBuildState``.
        IndexLifecycleError:
            If the hidden build directory is missing or invalid,
            metadata already exists before finalization, or the freshly
            written metadata does not make the build current.
        ChromaStoreError:
            If metadata construction, serialization, writing, or
            freshness reading fails through the storage layer.
    """

    if not isinstance(
        validated_state,
        IndexValidatedBuildState,
    ):
        raise TypeError(
            "validated_state must be an "
            "IndexValidatedBuildState instance."
        )

    build_state = (
        validated_state.build_state
    )

    build_path = (
        build_state.publication_paths[
            "build"
        ]
    )

    if not build_path.exists():
        raise IndexLifecycleError(
            "Chroma build directory does not exist before metadata "
            f"finalization: {str(build_path)!r}."
        )

    if not build_path.is_dir():
        raise IndexLifecycleError(
            "Chroma build path is not a directory before metadata "
            f"finalization: {str(build_path)!r}."
        )

    metadata_path = (
        build_path
        / "index_metadata.json"
    )

    if metadata_path.exists():
        raise IndexLifecycleError(
            "Chroma build metadata already exists before finalization: "
            f"{str(metadata_path)!r}."
        )

    corpus_version = (
        build_state
        .embedding_inputs
        .build_inputs
        .corpus_version
    )

    metadata = build_index_metadata(
        corpus_version=corpus_version
    )

    payload = serialize_index_metadata(
        metadata
    )

    write_index_metadata_atomic(
        build_path,
        payload,
    )

    current = is_index_current(
        build_path,
        corpus_version=corpus_version,
        embedding_model=EMBEDDING_MODEL_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
        chunk_tokens=TARGET_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    if not current:
        raise IndexLifecycleError(
            "Chroma build metadata was written but the hidden build "
            "does not match the expected freshness configuration."
        )


def build_policy_index(
    chunks_path: Path,
    manifest_path: Path,
    active_chroma_dir: Path,
) -> IndexFinalizedBuildState:
    """Execute the complete offline hidden-index build phase.

    This function composes the already-verified C9 lifecycle stages in
    their required order:

    1. load and validate canonical build inputs;
    2. embed ordered policy texts;
    3. construct the hidden Chroma build;
    4. exhaustively validate persisted Chroma contents;
    5. run the frozen semantic-smoke suite;
    6. write index metadata last and verify build freshness.

    The resulting hidden build is ready for later publication, but this
    function deliberately does not publish it. The process using this
    build must terminate before the C8 directory-publication phase is
    executed.

    Args:
        chunks_path:
            Absolute path to canonical ``chunks.json``.
        manifest_path:
            Absolute path to authoritative corpus ``version.json``.
        active_chroma_dir:
            Absolute final active Chroma directory. The hidden build and
            backup sibling paths are derived from this path.

    Returns:
        Immutable finalized hidden-build state ready for later
        process-boundary publication.

    Raises:
        TypeError:
            If any existing lower-level lifecycle contract rejects an
            input type.
        ValueError:
            If any supplied path violates an absolute-path contract.
        IndexLifecycleError:
            If canonical loading, build preflight, persisted validation,
            semantic validation, metadata finalization, or freshness
            verification fails.
        EmbeddingError:
            If document or query embedding fails.
        ChromaStoreError:
            If Chroma construction, validation, semantic smoke, or
            metadata storage fails.
    """

    build_inputs = build_index_inputs(
        chunks_path,
        manifest_path,
    )

    embedding_inputs = embed_index_documents(
        build_inputs
    )

    build_state = build_chroma_index(
        embedding_inputs,
        active_chroma_dir,
    )

    validated_state = validate_chroma_build(
        build_state
    )

    validate_chroma_build_semantic_smoke(
        validated_state
    )

    finalize_chroma_build_metadata(
        validated_state
    )

    return IndexFinalizedBuildState(
        validated_state=validated_state
    )


def publish_policy_index(
    manifest_path: Path,
    active_chroma_dir: Path,
) -> bool:
    """Publish one finalized hidden policy index in a fresh process.

    This function is the publication-process counterpart to
    ``build_policy_index``. It derives the authoritative corpus version
    from the validated manifest, verifies that the prepared hidden build
    is current under the frozen embedding/chunk configuration, and then
    delegates all filesystem publication mechanics to the proven C8
    ``publish_index`` primitive.

    This function deliberately performs no Chroma client creation,
    collection access, embedding, semantic query, persisted-integrity
    validation, or index construction. The build process must have exited
    before this function is invoked.

    Args:
        manifest_path:
            Absolute path to authoritative ``corpus/version.json``.
        active_chroma_dir:
            Absolute final active Chroma directory. The hidden build and
            backup sibling paths are derived from this path.

    Returns:
        The exact Boolean returned by ``publish_index``:
        ``True`` when publication succeeds and no backup remains;
        ``False`` when publication succeeds but obsolete backup cleanup
        fails.

    Raises:
        TypeError:
            If either path is not a ``pathlib.Path``.
        ValueError:
            If either path is not absolute.
        IndexLifecycleError:
            If the prepared hidden build is missing, is not a directory,
            or is stale under the current corpus/embedding configuration.
        ManifestValidationError:
            If the authoritative corpus manifest is invalid.
        ChromaStoreError:
            If C8 publication preconditions or filesystem publication
            fail.
    """

    if not isinstance(
        manifest_path,
        Path,
    ):
        raise TypeError(
            "manifest_path must be a pathlib.Path instance."
        )

    if not manifest_path.is_absolute():
        raise ValueError(
            "manifest_path must be absolute."
        )

    if not isinstance(
        active_chroma_dir,
        Path,
    ):
        raise TypeError(
            "active_chroma_dir must be a pathlib.Path instance."
        )

    if not active_chroma_dir.is_absolute():
        raise ValueError(
            "active_chroma_dir must be absolute."
        )

    from rag.ingest import load_manifest

    manifest = load_manifest(
        manifest_path
    )

    corpus_version = (
        manifest.version.strip()
    )

    if not corpus_version:
        raise IndexLifecycleError(
            "Corpus manifest version must be non-empty before "
            "index publication."
        )

    paths = resolve_publication_paths(
        active_chroma_dir
    )

    build_path = paths[
        "build"
    ]

    if not build_path.exists():
        raise IndexLifecycleError(
            "Prepared Chroma build directory does not exist before "
            f"publication: {str(build_path)!r}."
        )

    if not build_path.is_dir():
        raise IndexLifecycleError(
            "Prepared Chroma build path is not a directory before "
            f"publication: {str(build_path)!r}."
        )

    current = is_index_current(
        build_path,
        corpus_version=corpus_version,
        embedding_model=EMBEDDING_MODEL_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
        chunk_tokens=TARGET_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    if not current:
        raise IndexLifecycleError(
            "Prepared Chroma build is not current under the "
            "authoritative corpus and frozen embedding configuration."
        )

    return publish_index(
        paths
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the deployment index-lifecycle command parser."""

    parser = argparse.ArgumentParser(
        prog="python -m rag.index",
        description=(
            "Build or publish the generated policy Chroma index."
        ),
    )

    parser.add_argument(
        "command",
        choices=(
            "build",
            "publish",
        ),
        help=(
            "Build a validated hidden index or publish a prepared "
            "hidden index."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    """Execute one deployment index-lifecycle phase."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]

    chunks_path = (
        project_root
        / "corpus"
        / "processed"
        / "chunks.json"
    ).resolve()

    manifest_path = (
        project_root
        / "corpus"
        / "version.json"
    ).resolve()

    active_chroma_dir = resolve_chroma_dir()

    if args.command == "build":
        build_policy_index(
            chunks_path,
            manifest_path,
            active_chroma_dir,
        )
        return 0

    publish_policy_index(
        manifest_path,
        active_chroma_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
