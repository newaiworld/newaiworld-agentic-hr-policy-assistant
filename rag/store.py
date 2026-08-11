"""Chroma storage lifecycle for the S4 RAG pipeline.

This module owns the persistent-vector-store boundary for policy chunks.

C1.3 defines storage configuration, persistent Chroma client creation,
and the frozen policy collection lifecycle. Index construction,
freshness checks, and safe publication are introduced by later CP8
checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, TypedDict

import chromadb
import numpy as np
from chromadb.config import Settings

from rag.embed import EMBEDDING_DIMENSION


COLLECTION_NAME: Final[str] = "policy_chunks"
DISTANCE_METRIC: Final[str] = "cosine"
DEFAULT_CHROMA_DIR: Final[Path] = Path("chroma_db")

class ChromaRecords(TypedDict):
    """Aligned records ready for one Chroma collection.add() call."""

    ids: list[str]
    documents: list[str]
    embeddings: np.ndarray
    metadatas: list[dict[str, object]]


class ChromaStoreError(RuntimeError):
    """Base exception for Chroma storage-pipeline failures."""



def resolve_chroma_dir() -> Path:
    """Return the configured Chroma persistence directory.

    The ``CHROMA_DIR`` environment variable overrides the project
    default. User-home markers are expanded and the result is resolved
    to an absolute path so lower-level storage code does not depend on
    the caller's current path representation.

    Returns:
        Absolute path to the configured Chroma persistence directory.

    Raises:
        ChromaStoreError:
            If ``CHROMA_DIR`` is defined but contains only whitespace.
    """

    configured = os.getenv("CHROMA_DIR")

    if configured is None:
        path = DEFAULT_CHROMA_DIR
    else:
        if not configured.strip():
            raise ChromaStoreError(
                "CHROMA_DIR must not be blank when configured."
            )

        path = Path(configured.strip())

    return path.expanduser().resolve()


def get_chroma_client(path: Path) -> chromadb.api.ClientAPI:
    """Return a persistent Chroma client for one explicit directory.

    Telemetry is disabled because this project uses Chroma entirely as
    a local generated vector store. The caller must provide an absolute
    path, normally produced by ``resolve_chroma_dir()``, so persistence
    never depends implicitly on the current working directory.

    Args:
        path:
            Absolute directory used for Chroma persistence.

    Returns:
        Persistent Chroma client configured with anonymized telemetry
        disabled.

    Raises:
        TypeError:
            If ``path`` is not a ``Path`` instance.
        ValueError:
            If ``path`` is not absolute.
        ChromaStoreError:
            If Chroma cannot create the persistent client.
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path must be a pathlib.Path instance."
        )

    if not path.is_absolute():
        raise ValueError(
            "path must be absolute."
        )

    settings = Settings(
        anonymized_telemetry=False,
    )

    try:
        return chromadb.PersistentClient(
            path=path,
            settings=settings,
        )
    except Exception as exc:
        raise ChromaStoreError(
            "Failed to create persistent Chroma client for "
            f"{str(path)!r}."
        ) from exc


def _validate_collection_contract(
    collection: object,
) -> None:
    """Validate one Chroma collection against the frozen CP8 contract.

    Args:
        collection:
            Collection returned by the pinned Chroma client.

    Raises:
        ChromaStoreError:
            If the collection name, distance metric, or embedding
            function does not match the project contract.
    """

    name = getattr(collection, "name", None)

    if name != COLLECTION_NAME:
        raise ChromaStoreError(
            "Chroma collection name does not match the frozen "
            f"configuration: {name!r} != {COLLECTION_NAME!r}."
        )

    configuration = getattr(
        collection,
        "configuration",
        None,
    )

    if not isinstance(configuration, dict):
        raise ChromaStoreError(
            "Chroma collection configuration is unavailable or invalid."
        )

    hnsw = configuration.get("hnsw")

    if not isinstance(hnsw, dict):
        raise ChromaStoreError(
            "Chroma collection HNSW configuration is unavailable."
        )

    distance_metric = hnsw.get("space")

    if distance_metric != DISTANCE_METRIC:
        raise ChromaStoreError(
            "Chroma collection distance metric does not match the "
            "frozen configuration: "
            f"{distance_metric!r} != {DISTANCE_METRIC!r}."
        )

    if configuration.get("embedding_function") is not None:
        raise ChromaStoreError(
            "Chroma collection must not define an embedding function."
        )


def create_policy_collection(
    client: chromadb.api.ClientAPI,
):
    """Create and validate the frozen policy-vector collection.

    This operation is intended only for index construction. It does not
    silently reuse an existing collection.

    Args:
        client:
            Chroma client used to create the collection.

    Returns:
        Newly created collection satisfying the frozen CP8 contract.

    Raises:
        ChromaStoreError:
            If creation fails or the resulting collection violates the
            collection contract.
    """

    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": DISTANCE_METRIC,
                },
            },
            embedding_function=None,
        )
    except Exception as exc:
        raise ChromaStoreError(
            f"Failed to create Chroma collection {COLLECTION_NAME!r}."
        ) from exc

    _validate_collection_contract(collection)

    return collection


def get_policy_collection(
    client: chromadb.api.ClientAPI,
):
    """Open and validate the existing policy-vector collection.

    This operation never creates a missing collection. A missing or
    incompatible collection is a storage failure so later freshness
    logic can distinguish an absent index from a valid one.

    Args:
        client:
            Chroma client used to open the collection.

    Returns:
        Existing collection satisfying the frozen CP8 contract.

    Raises:
        ChromaStoreError:
            If the collection is missing, cannot be opened, or violates
            the frozen collection contract.
    """

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
    except Exception as exc:
        raise ChromaStoreError(
            f"Failed to open Chroma collection {COLLECTION_NAME!r}."
        ) from exc

    _validate_collection_contract(collection)

    return collection


def prepare_chroma_records(
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
) -> ChromaRecords:
    """Prepare validated canonical chunks for Chroma insertion.

    Canonical chunk ordering is preserved exactly. Each embedding row
    remains aligned with the chunk at the same index.

    Citation snippets are runtime metadata and therefore derive
    directly from the complete canonical chunk text. Canonical
    ``chunks.json`` remains unchanged.

    Args:
        chunks:
            Ordered canonical chunk records, normally produced by
            ``chunk_to_record()`` or loaded from canonical
            ``chunks.json``.
        embeddings:
            Ordered document-embedding matrix corresponding one-to-one
            with ``chunks``.

    Returns:
        Four aligned collections ready for one Chroma ``add()`` call.

    Raises:
        TypeError:
            If ``chunks`` or ``embeddings`` has the wrong top-level
            type, or a required canonical field has the wrong type.
        ChromaStoreError:
            If records are empty, IDs are duplicated, required text or
            metadata is blank, section paths are invalid, embedding
            dimensions do not match the frozen contract, or embeddings
            contain non-finite values.
    """

    if not isinstance(chunks, list):
        raise TypeError(
            "chunks must be a list of canonical chunk records."
        )

    if not chunks:
        raise ChromaStoreError(
            "chunks must contain at least one canonical record."
        )

    if not isinstance(embeddings, np.ndarray):
        raise TypeError(
            "embeddings must be a numpy.ndarray."
        )

    expected_shape = (
        len(chunks),
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ChromaStoreError(
            "Embedding matrix has an unexpected shape for Chroma "
            f"storage: {embeddings.shape!r} != {expected_shape!r}."
        )

    if not np.issubdtype(
        embeddings.dtype,
        np.floating,
    ):
        raise ChromaStoreError(
            "Embedding matrix for Chroma storage must contain "
            f"floating-point values; received dtype {embeddings.dtype!r}."
        )

    if not np.isfinite(embeddings).all():
        raise ChromaStoreError(
            "Embedding matrix for Chroma storage contains "
            "non-finite values."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, object]] = []

    seen_ids: set[str] = set()

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(
                "chunks must contain only dictionaries; "
                f"item {index} is {type(chunk).__name__}."
            )

        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        title = chunk.get("title")
        section_path = chunk.get("section_path")
        text = chunk.get("text")
        source_format = chunk.get("source_format")

        for field_name, value in (
            ("chunk_id", chunk_id),
            ("doc_id", doc_id),
            ("title", title),
            ("text", text),
            ("source_format", source_format),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"chunk {index} field {field_name!r} "
                    "must be a string."
                )

            if not value.strip():
                raise ChromaStoreError(
                    f"chunk {index} field {field_name!r} "
                    "must not be blank."
                )

        if not isinstance(section_path, list):
            raise TypeError(
                f"chunk {index} field 'section_path' must be a list."
            )

        if (
            not section_path
            or any(
                not isinstance(part, str)
                or not part.strip()
                for part in section_path
            )
        ):
            raise ChromaStoreError(
                f"chunk {index} field 'section_path' must contain "
                "only non-empty strings."
            )

        assert isinstance(chunk_id, str)
        assert isinstance(doc_id, str)
        assert isinstance(title, str)
        assert isinstance(text, str)
        assert isinstance(source_format, str)

        if chunk_id in seen_ids:
            raise ChromaStoreError(
                f"Duplicate chunk_id detected: {chunk_id!r}."
            )

        seen_ids.add(chunk_id)

        ids.append(chunk_id)
        documents.append(text)

        metadatas.append(
            {
                "doc_id": doc_id,
                "title": title,
                "section_path": list(section_path),
                "source_format": source_format,
                "snippet": text,
            }
        )

    return {
        "ids": ids,
        "documents": documents,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }


def add_chroma_records(
    collection: object,
    records: ChromaRecords,
) -> None:
    """Insert one validated Chroma record payload into a collection.

    Record preparation and validation are deliberately separate from
    persistence. This function assumes ``records`` came from
    ``prepare_chroma_records()`` and performs one collection ``add()``
    call so the small V1 corpus is written atomically at the API-call
    level.

    Args:
        collection:
            Validated Chroma collection receiving the records.
        records:
            Aligned payload returned by ``prepare_chroma_records()``.

    Raises:
        TypeError:
            If ``records`` is not a dictionary.
        ChromaStoreError:
            If the payload structure is incomplete or Chroma rejects
            the insertion.
    """

    if not isinstance(records, dict):
        raise TypeError(
            "records must be a ChromaRecords dictionary."
        )

    required_keys = {
        "ids",
        "documents",
        "embeddings",
        "metadatas",
    }

    missing_keys = required_keys.difference(records)

    if missing_keys:
        missing = ", ".join(
            sorted(missing_keys)
        )

        raise ChromaStoreError(
            "Chroma record payload is missing required keys: "
            f"{missing}."
        )

    try:
        collection.add(
            ids=records["ids"],
            documents=records["documents"],
            embeddings=records["embeddings"],
            metadatas=records["metadatas"],
        )
    except Exception as exc:
        first_id = (
            records["ids"][0]
            if records["ids"]
            else "<none>"
        )
        last_id = (
            records["ids"][-1]
            if records["ids"]
            else "<none>"
        )

        raise ChromaStoreError(
            "Failed to add Chroma records "
            f"from {first_id!r} to {last_id!r}."
        ) from exc

def build_index(
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
    chroma_dir: Path,
) -> None:
    """Build one complete Chroma policy index at an explicit path.

    The caller owns path selection. During CP8 validation this function
    is called with a temporary directory; later publication logic may
    use the same primitive with another explicitly selected build path.

    All canonical records are validated before Chroma persistence begins.
    The V1 corpus is written in one ``add()`` call because the inspected
    pinned Chroma batch limit safely exceeds the canonical corpus size.

    Args:
        chunks:
            Ordered canonical chunk records.
        embeddings:
            Ordered document embeddings corresponding one-to-one with
            ``chunks``.
        chroma_dir:
            Absolute persistence directory for this index build.

    Raises:
        TypeError:
            If an existing lower-level storage contract rejects an input
            type.
        ValueError:
            If ``chroma_dir`` violates the explicit-path contract.
        ChromaStoreError:
            If record validation, client creation, collection creation,
            insertion, or final count verification fails.
    """

    records = prepare_chroma_records(
        chunks,
        embeddings,
    )

    client = get_chroma_client(
        chroma_dir
    )

    collection = create_policy_collection(
        client
    )

    add_chroma_records(
        collection,
        records,
    )

    expected_count = len(records["ids"])
    actual_count = collection.count()

    if actual_count != expected_count:
        raise ChromaStoreError(
            "Chroma collection count does not match the prepared "
            "record count after insertion: "
            f"{actual_count} != {expected_count}."
        )


def validate_index_integrity(
    collection: object,
    records: ChromaRecords,
) -> None:
    """Exhaustively validate stored Chroma records by canonical ID.

    Validation is deliberately ID-keyed because Chroma does not
    preserve the order of IDs supplied to ``collection.get()``.

    Every prepared ID must exist exactly once in the stored collection,
    with identical document text and metadata. Stored embeddings must
    have the frozen dimension, contain only finite values, and remain
    numerically equivalent to the prepared embeddings.

    Args:
        collection:
            Existing Chroma collection containing the built index.
        records:
            Prepared canonical payload previously written to Chroma.

    Raises:
        ChromaStoreError:
            If Chroma cannot return the stored records or if any ID,
            document, metadata, or embedding differs from the prepared
            payload.
    """

    try:
        result = collection.get(
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )
    except Exception as exc:
        raise ChromaStoreError(
            "Failed to read Chroma records for integrity validation."
        ) from exc

    stored_ids = result.get("ids")
    stored_documents = result.get("documents")
    stored_metadatas = result.get("metadatas")
    stored_embeddings = result.get("embeddings")

    if not isinstance(stored_ids, list):
        raise ChromaStoreError(
            "Chroma integrity result has invalid IDs."
        )

    if not isinstance(stored_documents, list):
        raise ChromaStoreError(
            "Chroma integrity result has invalid documents."
        )

    if not isinstance(stored_metadatas, list):
        raise ChromaStoreError(
            "Chroma integrity result has invalid metadatas."
        )

    if stored_embeddings is None:
        raise ChromaStoreError(
            "Chroma integrity result has no embeddings."
        )

    stored_count = len(stored_ids)

    if len(stored_documents) != stored_count:
        raise ChromaStoreError(
            "Chroma integrity result has misaligned documents."
        )

    if len(stored_metadatas) != stored_count:
        raise ChromaStoreError(
            "Chroma integrity result has misaligned metadatas."
        )

    embedding_matrix = np.asarray(
        stored_embeddings
    )

    expected_embedding_shape = (
        stored_count,
        EMBEDDING_DIMENSION,
    )

    if embedding_matrix.shape != expected_embedding_shape:
        raise ChromaStoreError(
            "Chroma integrity result has an unexpected embedding "
            f"shape: {embedding_matrix.shape!r} != "
            f"{expected_embedding_shape!r}."
        )

    if not np.issubdtype(
        embedding_matrix.dtype,
        np.floating,
    ):
        raise ChromaStoreError(
            "Stored Chroma embeddings must contain "
            "floating-point values."
        )

    if not np.isfinite(
        embedding_matrix
    ).all():
        raise ChromaStoreError(
            "Stored Chroma embeddings contain non-finite values."
        )

    if len(set(stored_ids)) != stored_count:
        raise ChromaStoreError(
            "Chroma integrity result contains duplicate IDs."
        )

    expected_ids = records["ids"]

    if len(set(expected_ids)) != len(expected_ids):
        raise ChromaStoreError(
            "Prepared Chroma records contain duplicate IDs."
        )

    expected_id_set = set(expected_ids)
    stored_id_set = set(stored_ids)

    if stored_id_set != expected_id_set:
        missing_ids = sorted(
            expected_id_set - stored_id_set
        )
        unexpected_ids = sorted(
            stored_id_set - expected_id_set
        )

        raise ChromaStoreError(
            "Chroma integrity ID set mismatch: "
            f"missing={missing_ids!r}, "
            f"unexpected={unexpected_ids!r}."
        )

    stored_position_by_id = {
        chunk_id: index
        for index, chunk_id in enumerate(
            stored_ids
        )
    }

    expected_position_by_id = {
        chunk_id: index
        for index, chunk_id in enumerate(
            expected_ids
        )
    }

    for chunk_id in expected_ids:
        expected_index = expected_position_by_id[
            chunk_id
        ]
        stored_index = stored_position_by_id[
            chunk_id
        ]

        expected_document = records[
            "documents"
        ][expected_index]

        stored_document = stored_documents[
            stored_index
        ]

        if stored_document != expected_document:
            raise ChromaStoreError(
                "Chroma document mismatch for "
                f"{chunk_id!r}."
            )

        expected_metadata = records[
            "metadatas"
        ][expected_index]

        stored_metadata = stored_metadatas[
            stored_index
        ]

        if stored_metadata != expected_metadata:
            raise ChromaStoreError(
                "Chroma metadata mismatch for "
                f"{chunk_id!r}."
            )

        expected_embedding = records[
            "embeddings"
        ][expected_index]

        stored_embedding = embedding_matrix[
            stored_index
        ]

        if not np.allclose(
            stored_embedding,
            expected_embedding,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ChromaStoreError(
                "Chroma embedding mismatch for "
                f"{chunk_id!r}."
            )
