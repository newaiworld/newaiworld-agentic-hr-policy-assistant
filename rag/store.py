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
from typing import Final

import chromadb
from chromadb.config import Settings


COLLECTION_NAME: Final[str] = "policy_chunks"
DISTANCE_METRIC: Final[str] = "cosine"
DEFAULT_CHROMA_DIR: Final[Path] = Path("chroma_db")


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
