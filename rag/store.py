"""Chroma storage lifecycle for the S4 RAG pipeline.

This module owns the persistent-vector-store boundary for policy chunks.

C1.3 defines storage configuration, persistent Chroma client creation,
and the frozen policy collection lifecycle. Index construction,
freshness checks, and safe publication are introduced by later CP8
checkpoints.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TypedDict

import chromadb
import numpy as np
from chromadb.config import Settings

from rag.chunk import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    TARGET_CHUNK_TOKENS,
)

from rag.embed import EMBEDDING_DIMENSION


COLLECTION_NAME: Final[str] = "policy_chunks"
DISTANCE_METRIC: Final[str] = "cosine"
DEFAULT_CHROMA_DIR: Final[Path] = Path("chroma_db")

INDEX_METADATA_FILENAME: Final[str] = "index_metadata.json"


class IndexMetadata(TypedDict):
    """Configuration identity and provenance for one Chroma index."""

    corpus_version: str
    embedding_model: str
    embedding_dimension: int
    chunk_tokens: int
    chunk_overlap: int
    created: str


class PublicationPaths(TypedDict):
    """Resolved filesystem paths for one Chroma publication cycle."""

    active: Path
    build: Path
    backup: Path


class ChromaRecords(TypedDict):
    """Aligned records ready for one Chroma collection.add() call."""

    ids: list[str]
    documents: list[str]
    embeddings: np.ndarray
    metadatas: list[dict[str, object]]


class ChromaStoreError(RuntimeError):
    """Base exception for Chroma storage-pipeline failures."""


def build_index_metadata(
    corpus_version: str,
    *,
    created: datetime | None = None,
) -> IndexMetadata:
    """Build metadata describing the current index configuration.

    The five configuration identity fields are taken from the current
    corpus/configuration contract. ``created`` records when the metadata
    was generated and does not itself determine index freshness.

    Args:
        corpus_version:
            Current top-level version from ``corpus/version.json``.
        created:
            Optional timezone-aware UTC datetime. When omitted, the
            current UTC time is used.

    Returns:
        Six-field index metadata object defined by the frozen RAG
        lifecycle contract.

    Raises:
        TypeError:
            If ``corpus_version`` or ``created`` has an invalid type.
        ValueError:
            If ``corpus_version`` is blank or ``created`` is not an
            explicit UTC datetime.
    """

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

    if created is None:
        created = datetime.now(
            timezone.utc
        )
    elif not isinstance(
        created,
        datetime,
    ):
        raise TypeError(
            "created must be a datetime or None."
        )

    if (
        created.tzinfo is None
        or created.utcoffset() is None
    ):
        raise ValueError(
            "created must be timezone-aware UTC."
        )

    if created.utcoffset() != timezone.utc.utcoffset(
        created
    ):
        raise ValueError(
            "created must be timezone-aware UTC."
        )

    created_text = created.isoformat().replace(
        "+00:00",
        "Z",
    )

    return {
        "corpus_version": corpus_version.strip(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_tokens": TARGET_CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "created": created_text,
    }


def serialize_index_metadata(
    metadata: IndexMetadata,
) -> bytes:
    """Serialize index metadata as stable UTF-8 JSON bytes.

    The metadata artifact is generated state rather than a deterministic
    corpus artifact because ``created`` changes between builds. Stable
    serialization is still used so the file remains predictable,
    inspectable, and easy to test.

    Args:
        metadata:
            Valid six-field index metadata object.

    Returns:
        UTF-8 encoded JSON bytes with sorted keys, compact separators,
        preserved Unicode, and exactly one trailing newline.

    Raises:
        TypeError:
            If ``metadata`` is not a dictionary.
        ChromaStoreError:
            If the metadata schema or field types are invalid.
    """

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "metadata must be a dictionary."
        )

    required_keys = {
        "corpus_version",
        "embedding_model",
        "embedding_dimension",
        "chunk_tokens",
        "chunk_overlap",
        "created",
    }

    if set(metadata) != required_keys:
        raise ChromaStoreError(
            "Index metadata must contain exactly the frozen six fields."
        )

    string_fields = (
        "corpus_version",
        "embedding_model",
        "created",
    )

    for field_name in string_fields:
        value = metadata[field_name]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ChromaStoreError(
                "Index metadata field "
                f"{field_name!r} must be a non-empty string."
            )

    integer_fields = (
        "embedding_dimension",
        "chunk_tokens",
        "chunk_overlap",
    )

    for field_name in integer_fields:
        value = metadata[field_name]

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ChromaStoreError(
                "Index metadata field "
                f"{field_name!r} must be a positive integer."
            )

    try:
        serialized = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ChromaStoreError(
            "Failed to serialize index metadata."
        ) from exc

    return (
        serialized + "\n"
    ).encode("utf-8")

def resolve_publication_paths(
    chroma_dir: Path,
) -> PublicationPaths:
    """Resolve sibling paths used for safe Chroma publication.

    The active, build, and backup directories must remain siblings on
    the same filesystem so publication can use same-filesystem rename
    semantics.

    Args:
        chroma_dir:
            Absolute active Chroma persistence directory.

    Returns:
        Mapping containing the active path plus deterministic hidden
        build and backup sibling paths.

    Raises:
        TypeError:
            If ``chroma_dir`` is not a pathlib.Path.
        ValueError:
            If ``chroma_dir`` is not absolute or its final name is
            unusable for deterministic sibling-path construction.
    """

    if not isinstance(
        chroma_dir,
        Path,
    ):
        raise TypeError(
            "chroma_dir must be a pathlib.Path instance."
        )

    if not chroma_dir.is_absolute():
        raise ValueError(
            "chroma_dir must be absolute."
        )

    if not chroma_dir.name:
        raise ValueError(
            "chroma_dir must have a final path name."
        )

    active = chroma_dir

    build = active.with_name(
        f".{active.name}.build"
    )

    backup = active.with_name(
        f".{active.name}.backup"
    )

    if (
        build.parent != active.parent
        or backup.parent != active.parent
    ):
        raise ChromaStoreError(
            "Chroma publication paths must be sibling directories."
        )

    if len(
        {
            active,
            build,
            backup,
        }
    ) != 3:
        raise ChromaStoreError(
            "Chroma publication paths must be distinct."
        )

    return {
        "active": active,
        "build": build,
        "backup": backup,
    }

def validate_publication_preconditions(
    paths: PublicationPaths,
) -> None:
    """Validate filesystem state before Chroma directory publication.

    Validation is intentionally read-only. Unexpected publication state
    is rejected rather than cleaned or overwritten automatically.

    Valid states are:

    * first build:
      active absent, build present, backup absent

    * replacement:
      active present, build present, backup absent

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Raises:
        TypeError:
            If ``paths`` is not a dictionary.
        ChromaStoreError:
            If required paths are missing, are not directories, or an
            unexpected backup path already exists.
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
        raise ChromaStoreError(
            "Publication paths must contain exactly "
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
            raise ChromaStoreError(
                "Publication path "
                f"{field_name!r} must be a pathlib.Path."
            )

        if not value.is_absolute():
            raise ChromaStoreError(
                "Publication path "
                f"{field_name!r} must be absolute."
            )

    active = paths["active"]
    build = paths["build"]
    backup = paths["backup"]

    if backup.exists():
        raise ChromaStoreError(
            "Chroma publication backup path already exists: "
            f"{str(backup)!r}."
        )

    if not build.exists():
        raise ChromaStoreError(
            "Chroma publication build path does not exist: "
            f"{str(build)!r}."
        )

    if not build.is_dir():
        raise ChromaStoreError(
            "Chroma publication build path is not a directory: "
            f"{str(build)!r}."
        )

    if active.exists() and not active.is_dir():
        raise ChromaStoreError(
            "Chroma publication active path is not a directory: "
            f"{str(active)!r}."
        )


def publish_first_index(
    paths: PublicationPaths,
) -> None:
    """Publish a prepared Chroma build when no active index exists.

    This helper implements only the first-build branch. Publication
    preconditions are validated before any filesystem mutation occurs.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Raises:
        ChromaStoreError:
            If publication preconditions are invalid, an active index
            already exists, or the build directory cannot be promoted.
    """

    validate_publication_preconditions(
        paths
    )

    active = paths["active"]
    build = paths["build"]

    if active.exists():
        raise ChromaStoreError(
            "First-build publication requires the active "
            "Chroma directory to be absent."
        )

    try:
        os.replace(
            build,
            active,
        )
    except OSError as exc:
        raise ChromaStoreError(
            "Failed to publish first Chroma index from "
            f"{str(build)!r} to {str(active)!r}."
        ) from exc


def publish_replacement_index(
    paths: PublicationPaths,
) -> None:
    """Publish a prepared Chroma build over an existing active index.

    The existing active directory is first moved to the backup path.
    The prepared build is then promoted to the active path.

    If promotion of the build fails after the old active directory has
    already moved to backup, rollback is attempted immediately by
    restoring the backup to the active path.

    This helper does not remove the backup after successful publication
    and does not clean residual build state after failed publication.
    Those cleanup responsibilities belong to a later checkpoint.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Raises:
        ChromaStoreError:
            If publication preconditions fail, no active index exists,
            either publication move fails, or rollback fails.
    """

    validate_publication_preconditions(
        paths
    )

    active = paths["active"]
    build = paths["build"]
    backup = paths["backup"]

    if not active.exists():
        raise ChromaStoreError(
            "Replacement publication requires an existing "
            "active Chroma directory."
        )

    try:
        os.replace(
            active,
            backup,
        )
    except OSError as exc:
        raise ChromaStoreError(
            "Failed to move active Chroma index to backup "
            f"from {str(active)!r} to {str(backup)!r}."
        ) from exc

    try:
        os.replace(
            build,
            active,
        )
    except OSError as publish_exc:
        try:
            os.replace(
                backup,
                active,
            )
        except OSError as rollback_exc:
            error = ChromaStoreError(
                "Failed to publish replacement Chroma index and "
                "failed to restore the previous active index."
            )

            error.add_note(
                "Publication failure: "
                f"{publish_exc!r}"
            )
            error.add_note(
                "Rollback failure: "
                f"{rollback_exc!r}"
            )

            raise error from rollback_exc

        raise ChromaStoreError(
            "Failed to publish replacement Chroma index; "
            "the previous active index was restored."
        ) from publish_exc

def cleanup_failed_build(
    paths: PublicationPaths,
) -> None:
    """Remove residual build state after a failed publication.

    This cleanup is intended only after the previous active index has
    already been restored successfully. A residual build directory is
    therefore stale generated state and must not be reused implicitly.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Raises:
        TypeError:
            If ``paths`` is not a dictionary.
        ChromaStoreError:
            If publication paths are malformed, the build path is not a
            directory, or residual build cleanup fails.
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
        raise ChromaStoreError(
            "Publication paths must contain exactly "
            "'active', 'build', and 'backup'."
        )

    build = paths["build"]

    if not isinstance(
        build,
        Path,
    ):
        raise ChromaStoreError(
            "Publication path 'build' must be a pathlib.Path."
        )

    if not build.is_absolute():
        raise ChromaStoreError(
            "Publication path 'build' must be absolute."
        )

    if not build.exists():
        return

    if not build.is_dir():
        raise ChromaStoreError(
            "Residual Chroma build path is not a directory: "
            f"{str(build)!r}."
        )

    try:
        shutil.rmtree(
            build
        )
    except OSError as exc:
        raise ChromaStoreError(
            "Failed to remove residual Chroma build directory "
            f"{str(build)!r}."
        ) from exc


def cleanup_published_backup(
    paths: PublicationPaths,
) -> bool:
    """Best-effort removal of backup after successful publication.

    Once the replacement has been published successfully, failure to
    remove the superseded backup must not invalidate the active index.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Returns:
        ``True`` when no backup remains after cleanup.
        ``False`` when backup cleanup fails.

    Raises:
        TypeError:
            If ``paths`` is not a dictionary.
        ChromaStoreError:
            If publication paths are malformed or the backup path exists
            but is not a directory.
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
        raise ChromaStoreError(
            "Publication paths must contain exactly "
            "'active', 'build', and 'backup'."
        )

    backup = paths["backup"]

    if not isinstance(
        backup,
        Path,
    ):
        raise ChromaStoreError(
            "Publication path 'backup' must be a pathlib.Path."
        )

    if not backup.is_absolute():
        raise ChromaStoreError(
            "Publication path 'backup' must be absolute."
        )

    if not backup.exists():
        return True

    if not backup.is_dir():
        raise ChromaStoreError(
            "Published Chroma backup path is not a directory: "
            f"{str(backup)!r}."
        )

    try:
        shutil.rmtree(
            backup
        )
    except OSError:
        return False

    return True


def publish_index(
    paths: PublicationPaths,
) -> bool:
    """Publish one prepared Chroma build into the active path.

    This function composes the already-validated first-build,
    replacement, rollback, and cleanup primitives.

    It does not build or validate Chroma contents and must be called only
    after the build directory is complete and all Chroma processes using
    the active/build paths have exited.

    Args:
        paths:
            Resolved active/build/backup publication paths.

    Returns:
        ``True`` when publication succeeds and no backup remains.
        ``False`` when publication succeeds but backup cleanup fails.

    Raises:
        ChromaStoreError:
            If publication preconditions fail or publication itself
            fails. When replacement publication fails after a successful
            rollback, residual build cleanup is attempted before the
            original publication error is re-raised.
    """

    validate_publication_preconditions(
        paths
    )

    active = paths["active"]
    build = paths["build"]
    backup = paths["backup"]

    if not active.exists():
        publish_first_index(
            paths
        )

        return True

    try:
        publish_replacement_index(
            paths
        )
    except ChromaStoreError:
        rollback_succeeded = (
            active.exists()
            and not backup.exists()
            and build.exists()
        )

        if rollback_succeeded:
            cleanup_failed_build(
                paths
            )

        raise

    return cleanup_published_backup(
        paths
    )



def get_index_metadata_path(
    chroma_dir: Path,
) -> Path:
    """Return the metadata path owned by one Chroma directory.

    Args:
        chroma_dir:
            Absolute Chroma persistence directory.

    Returns:
        ``<chroma_dir>/index_metadata.json``.

    Raises:
        TypeError:
            If ``chroma_dir`` is not a pathlib.Path.
        ValueError:
            If ``chroma_dir`` is not absolute.
    """

    if not isinstance(
        chroma_dir,
        Path,
    ):
        raise TypeError(
            "chroma_dir must be a pathlib.Path instance."
        )

    if not chroma_dir.is_absolute():
        raise ValueError(
            "chroma_dir must be absolute."
        )

    return (
        chroma_dir
        / INDEX_METADATA_FILENAME
    )


def write_index_metadata_atomic(
    chroma_dir: Path,
    payload: bytes,
) -> Path:
    """Atomically publish serialized index metadata.

    The metadata file is written as a sibling temporary artifact inside
    the Chroma directory and published with ``os.replace`` only after
    the complete payload has been written.

    Args:
        chroma_dir:
            Absolute Chroma persistence directory.
        payload:
            Already-validated serialized metadata bytes.

    Returns:
        Final ``index_metadata.json`` path.

    Raises:
        TypeError:
            If ``chroma_dir`` or ``payload`` has the wrong type.
        ValueError:
            If ``chroma_dir`` is not absolute or ``payload`` is empty.
        OSError:
            If directory creation, temporary writing, or atomic
            replacement fails.
    """

    metadata_path = get_index_metadata_path(
        chroma_dir
    )

    if not isinstance(
        payload,
        bytes,
    ):
        raise TypeError(
            "payload must be bytes."
        )

    if not payload:
        raise ValueError(
            "payload must be non-empty."
        )

    chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = metadata_path.with_name(
        f".{metadata_path.name}.tmp"
    )

    try:
        temporary_path.write_bytes(
            payload
        )

        os.replace(
            temporary_path,
            metadata_path,
        )
    except Exception:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise

    return metadata_path


def read_index_metadata(
    chroma_dir: Path,
) -> IndexMetadata | None:
    """Read and validate persisted Chroma index metadata.

    A missing metadata file represents "no index" according to the
    frozen RAG startup lifecycle and therefore returns ``None``.

    Existing metadata must contain exactly the six frozen fields with
    valid types and an explicit ISO 8601 UTC ``created`` timestamp.

    This function validates persisted metadata structure only. It does
    not decide whether the metadata matches the current corpus or index
    configuration; that freshness decision belongs to a later layer.

    Args:
        chroma_dir:
            Absolute Chroma persistence directory.

    Returns:
        Validated six-field index metadata, or ``None`` when
        ``index_metadata.json`` does not exist.

    Raises:
        TypeError:
            If ``chroma_dir`` is not a pathlib.Path.
        ValueError:
            If ``chroma_dir`` is not absolute.
        ChromaStoreError:
            If an existing metadata file cannot be read, contains
            invalid JSON, or violates the frozen metadata contract.
    """

    metadata_path = get_index_metadata_path(
        chroma_dir
    )

    if not metadata_path.exists():
        return None

    try:
        text = metadata_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ChromaStoreError(
            "Failed to read Chroma index metadata from "
            f"{str(metadata_path)!r}."
        ) from exc

    try:
        data = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise ChromaStoreError(
            "Chroma index metadata contains invalid JSON."
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise ChromaStoreError(
            "Chroma index metadata must be a JSON object."
        )

    required_keys = {
        "corpus_version",
        "embedding_model",
        "embedding_dimension",
        "chunk_tokens",
        "chunk_overlap",
        "created",
    }

    if set(data) != required_keys:
        raise ChromaStoreError(
            "Chroma index metadata must contain exactly "
            "the frozen six fields."
        )

    for field_name in (
        "corpus_version",
        "embedding_model",
        "created",
    ):
        value = data[field_name]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ChromaStoreError(
                "Chroma index metadata field "
                f"{field_name!r} must be a non-empty string."
            )

    for field_name in (
        "embedding_dimension",
        "chunk_tokens",
        "chunk_overlap",
    ):
        value = data[field_name]

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ChromaStoreError(
                "Chroma index metadata field "
                f"{field_name!r} must be a positive integer."
            )

    created_text = data["created"]

    if not created_text.endswith("Z"):
        raise ChromaStoreError(
            "Chroma index metadata field 'created' must be "
            "an ISO 8601 UTC timestamp ending in 'Z'."
        )

    if "T" not in created_text:
        raise ChromaStoreError(
            "Chroma index metadata field 'created' must be "
            "an ISO 8601 UTC timestamp ending in 'Z'."
        )

    try:
        created_datetime = datetime.fromisoformat(
            created_text[:-1]
            + "+00:00"
        )
    except ValueError as exc:
        raise ChromaStoreError(
            "Chroma index metadata field 'created' must be "
            "a valid ISO 8601 UTC timestamp."
        ) from exc

    if (
        created_datetime.tzinfo is None
        or created_datetime.utcoffset()
        != timezone.utc.utcoffset(
            created_datetime
        )
    ):
        raise ChromaStoreError(
            "Chroma index metadata field 'created' must be "
            "an ISO 8601 UTC timestamp."
        )

    return {
        "corpus_version": data[
            "corpus_version"
        ],
        "embedding_model": data[
            "embedding_model"
        ],
        "embedding_dimension": data[
            "embedding_dimension"
        ],
        "chunk_tokens": data[
            "chunk_tokens"
        ],
        "chunk_overlap": data[
            "chunk_overlap"
        ],
        "created": data[
            "created"
        ],
    }


def is_index_current(
    chroma_dir: Path,
    *,
    corpus_version: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_tokens: int,
    chunk_overlap: int,
) -> bool:
    """Return whether persisted index metadata matches current config.

    Freshness is determined conservatively. Missing or unusable
    persisted metadata is treated as stale rather than allowed to
    propagate into the startup fast path.

    The ``created`` timestamp is validated by ``read_index_metadata()``
    as provenance metadata but intentionally does not participate in
    freshness equality.

    Args:
        chroma_dir:
            Absolute Chroma persistence directory.
        corpus_version:
            Current top-level version from ``corpus/version.json``.
        embedding_model:
            Current embedding model identifier.
        embedding_dimension:
            Current embedding-vector dimension.
        chunk_tokens:
            Current target chunk-token configuration.
        chunk_overlap:
            Current chunk-overlap token configuration.

    Returns:
        ``True`` only when persisted metadata is structurally valid and
        all five freshness-identity fields exactly match the expected
        current configuration. Otherwise ``False``.

    Raises:
        TypeError:
            If one of the expected configuration values has an invalid
            type.
        ValueError:
            If one of the expected configuration values is unusable.
    """

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

    if not isinstance(
        embedding_model,
        str,
    ):
        raise TypeError(
            "embedding_model must be a string."
        )

    if not embedding_model.strip():
        raise ValueError(
            "embedding_model must be a non-empty string."
        )

    for field_name, value in (
        (
            "embedding_dimension",
            embedding_dimension,
        ),
        (
            "chunk_tokens",
            chunk_tokens,
        ),
        (
            "chunk_overlap",
            chunk_overlap,
        ),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be positive."
            )

    try:
        metadata = read_index_metadata(
            chroma_dir
        )
    except ChromaStoreError:
        return False

    if metadata is None:
        return False

    return (
        metadata["corpus_version"]
        == corpus_version.strip()
        and metadata["embedding_model"]
        == embedding_model.strip()
        and metadata["embedding_dimension"]
        == embedding_dimension
        and metadata["chunk_tokens"]
        == chunk_tokens
        and metadata["chunk_overlap"]
        == chunk_overlap
    )


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


def validate_persisted_index(
    chroma_dir: Path,
    records: ChromaRecords,
) -> None:
    """Reopen and exhaustively validate one persisted Chroma index.

    A fresh persistent client is created for the supplied directory so
    validation does not rely on the client or collection objects used
    during index construction.

    Args:
        chroma_dir:
            Absolute directory containing an already-built Chroma index.
        records:
            Prepared canonical payload expected to exist in the index.

    Raises:
        TypeError:
            If ``chroma_dir`` violates the client path type contract.
        ValueError:
            If ``chroma_dir`` is not absolute.
        ChromaStoreError:
            If the persisted client or collection cannot be opened, the
            collection contract is invalid, or exhaustive integrity
            validation fails.
    """

    client = get_chroma_client(
        chroma_dir
    )

    collection = get_policy_collection(
        client
    )

    validate_index_integrity(
        collection,
        records,
    )


def validate_semantic_smoke(
    collection: object,
    query_embedding: np.ndarray,
    *,
    expected_doc_id: str,
    n_results: int = 5,
) -> None:
    """Validate basic semantic connectivity of one Chroma index.

    This is intentionally a smoke check rather than a retrieval-quality
    evaluation. It proves that a valid query embedding can reach the
    expected policy document within a small bounded result set.

    Raw Chroma cosine distances are inspected only for structural
    validity. No similarity-score conversion, thresholding, reranking,
    filtering, or retrieval abstraction is introduced here.

    Args:
        collection:
            Existing validated Chroma policy collection.
        query_embedding:
            One validated query vector with the frozen embedding
            dimension.
        expected_doc_id:
            Policy document ID expected to appear in the returned
            semantic neighborhood.
        n_results:
            Positive number of nearest records requested from Chroma.

    Raises:
        TypeError:
            If the query embedding, expected document ID, or result-count
            input has the wrong type.
        ValueError:
            If the expected document ID is blank or ``n_results`` is not
            positive.
        ChromaStoreError:
            If the query vector is invalid, Chroma query execution fails,
            the returned result structure is invalid, distances are
            non-finite, or the expected policy document is absent.
    """

    if not isinstance(
        query_embedding,
        np.ndarray,
    ):
        raise TypeError(
            "query_embedding must be a numpy.ndarray."
        )

    expected_shape = (
        EMBEDDING_DIMENSION,
    )

    if query_embedding.shape != expected_shape:
        raise ChromaStoreError(
            "Semantic-smoke query embedding has an unexpected shape: "
            f"{query_embedding.shape!r} != {expected_shape!r}."
        )

    if not np.issubdtype(
        query_embedding.dtype,
        np.floating,
    ):
        raise ChromaStoreError(
            "Semantic-smoke query embedding must contain "
            "floating-point values."
        )

    if not np.isfinite(
        query_embedding
    ).all():
        raise ChromaStoreError(
            "Semantic-smoke query embedding contains non-finite values."
        )

    if not isinstance(
        expected_doc_id,
        str,
    ):
        raise TypeError(
            "expected_doc_id must be a string."
        )

    if not expected_doc_id.strip():
        raise ValueError(
            "expected_doc_id must be a non-empty string."
        )

    if (
        not isinstance(n_results, int)
        or isinstance(n_results, bool)
    ):
        raise TypeError(
            "n_results must be an integer."
        )

    if n_results <= 0:
        raise ValueError(
            "n_results must be positive."
        )

    try:
        result = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )
    except Exception as exc:
        raise ChromaStoreError(
            "Failed to execute Chroma semantic smoke query."
        ) from exc

    ids = result.get("ids")
    documents = result.get("documents")
    metadatas = result.get("metadatas")
    distances = result.get("distances")

    for field_name, value in (
        ("ids", ids),
        ("documents", documents),
        ("metadatas", metadatas),
        ("distances", distances),
    ):
        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], list)
        ):
            raise ChromaStoreError(
                "Chroma semantic-smoke result has invalid "
                f"{field_name} structure."
            )

    returned_ids = ids[0]
    returned_documents = documents[0]
    returned_metadatas = metadatas[0]
    returned_distances = distances[0]

    result_count = len(returned_ids)

    if result_count == 0:
        raise ChromaStoreError(
            "Chroma semantic-smoke query returned no records."
        )

    if len(returned_documents) != result_count:
        raise ChromaStoreError(
            "Chroma semantic-smoke documents are misaligned."
        )

    if len(returned_metadatas) != result_count:
        raise ChromaStoreError(
            "Chroma semantic-smoke metadatas are misaligned."
        )

    if len(returned_distances) != result_count:
        raise ChromaStoreError(
            "Chroma semantic-smoke distances are misaligned."
        )

    distance_array = np.asarray(
        returned_distances,
        dtype=float,
    )

    if not np.isfinite(
        distance_array
    ).all():
        raise ChromaStoreError(
            "Chroma semantic-smoke distances contain non-finite values."
        )

    returned_doc_ids: list[str] = []

    for index, metadata in enumerate(
        returned_metadatas
    ):
        if not isinstance(metadata, dict):
            raise ChromaStoreError(
                "Chroma semantic-smoke metadata is invalid at "
                f"result {index}."
            )

        doc_id = metadata.get("doc_id")

        if not isinstance(
            doc_id,
            str,
        ) or not doc_id.strip():
            raise ChromaStoreError(
                "Chroma semantic-smoke metadata has invalid doc_id at "
                f"result {index}."
            )

        returned_doc_ids.append(
            doc_id
        )

    if expected_doc_id not in returned_doc_ids:
        raise ChromaStoreError(
            "Semantic smoke query did not return expected policy "
            f"document {expected_doc_id!r}; "
            f"returned_doc_ids={returned_doc_ids!r}."
        )
