"""Focused tests for the S4 Chroma storage lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import rag.store as store_module
from rag.store import (
    COLLECTION_NAME,
    DEFAULT_CHROMA_DIR,
    DISTANCE_METRIC,
    ChromaStoreError,
    get_chroma_client,
    resolve_chroma_dir,
)


def test_store_configuration_matches_frozen_contract() -> None:
    """Storage constants must match the approved CP8 configuration."""

    assert COLLECTION_NAME == "policy_chunks"
    assert DISTANCE_METRIC == "cosine"
    assert DEFAULT_CHROMA_DIR == Path("chroma_db")


def test_chroma_store_error_is_runtime_error() -> None:
    """Storage failures must expose a project-owned runtime exception."""

    assert issubclass(ChromaStoreError, RuntimeError)


def test_resolve_chroma_dir_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing CHROMA_DIR must resolve the project default path."""

    monkeypatch.delenv("CHROMA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = resolve_chroma_dir()

    assert result == (tmp_path / "chroma_db").resolve()
    assert result.is_absolute()


def test_resolve_chroma_dir_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CHROMA_DIR must override the project default."""

    configured = tmp_path / "custom_chroma"

    monkeypatch.setenv(
        "CHROMA_DIR",
        str(configured),
    )

    result = resolve_chroma_dir()

    assert result == configured.resolve()
    assert result.is_absolute()


def test_resolve_chroma_dir_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Surrounding whitespace must not become part of the path."""

    configured = tmp_path / "custom_chroma"

    monkeypatch.setenv(
        "CHROMA_DIR",
        f"  {configured}  ",
    )

    assert resolve_chroma_dir() == configured.resolve()


def test_resolve_chroma_dir_expands_user_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A user-home marker must expand before the path is resolved."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv(
        "CHROMA_DIR",
        "~/project_chroma",
    )

    result = resolve_chroma_dir()

    assert result == (fake_home / "project_chroma").resolve()


@pytest.mark.parametrize(
    "configured",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
    ],
)
def test_resolve_chroma_dir_rejects_blank_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    """An explicitly blank CHROMA_DIR must fail instead of using cwd."""

    monkeypatch.setenv(
        "CHROMA_DIR",
        configured,
    )

    with pytest.raises(
        ChromaStoreError,
        match="CHROMA_DIR must not be blank",
    ):
        resolve_chroma_dir()


def test_resolve_chroma_dir_does_not_create_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configuration resolution must not mutate the filesystem."""

    configured = tmp_path / "not_created"

    monkeypatch.setenv(
        "CHROMA_DIR",
        str(configured),
    )

    result = resolve_chroma_dir()

    assert result == configured.resolve()
    assert not configured.exists()


def test_get_chroma_client_uses_persistent_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Client creation must use the supplied persistence directory."""

    expected_client = Mock()
    constructor = Mock(return_value=expected_client)

    monkeypatch.setattr(
        store_module.chromadb,
        "PersistentClient",
        constructor,
    )

    result = get_chroma_client(tmp_path.resolve())

    assert result is expected_client

    constructor.assert_called_once()

    _, kwargs = constructor.call_args

    assert kwargs["path"] == tmp_path.resolve()
    assert (
        kwargs["settings"].anonymized_telemetry
        is False
    )


def test_get_chroma_client_disables_anonymized_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every project-owned Chroma client must disable telemetry."""

    constructor = Mock(return_value=Mock())

    monkeypatch.setattr(
        store_module.chromadb,
        "PersistentClient",
        constructor,
    )

    get_chroma_client(tmp_path.resolve())

    settings = constructor.call_args.kwargs["settings"]

    assert settings.anonymized_telemetry is False


@pytest.mark.parametrize(
    "value",
    [
        "chroma_db",
        None,
        1,
        object(),
    ],
)
def test_get_chroma_client_rejects_non_path(
    value: object,
) -> None:
    """The storage boundary must receive an explicit Path object."""

    with pytest.raises(
        TypeError,
        match="path must be a pathlib.Path instance",
    ):
        get_chroma_client(value)  # type: ignore[arg-type]


def test_get_chroma_client_rejects_relative_path() -> None:
    """A relative persistence path must fail before Chroma is called."""

    with pytest.raises(
        ValueError,
        match="path must be absolute",
    ):
        get_chroma_client(
            Path("chroma_db")
        )


def test_get_chroma_client_wraps_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low-level Chroma construction errors must become project errors."""

    constructor = Mock(
        side_effect=RuntimeError("database unavailable")
    )

    monkeypatch.setattr(
        store_module.chromadb,
        "PersistentClient",
        constructor,
    )

    with pytest.raises(
        ChromaStoreError,
        match="Failed to create persistent Chroma client",
    ) as exc_info:
        get_chroma_client(tmp_path.resolve())

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_create_policy_collection_uses_frozen_configuration() -> None:
    """Collection creation must freeze cosine space and no embedder."""

    client = Mock()

    collection = Mock()
    collection.name = COLLECTION_NAME
    collection.configuration = {
        "hnsw": {
            "space": DISTANCE_METRIC,
        },
        "embedding_function": None,
    }

    client.create_collection.return_value = collection

    result = store_module.create_policy_collection(client)

    assert result is collection

    client.create_collection.assert_called_once_with(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": DISTANCE_METRIC,
            },
        },
        embedding_function=None,
    )


def test_create_policy_collection_wraps_creation_failure() -> None:
    """Low-level collection-creation failures must be project errors."""

    client = Mock()

    client.create_collection.side_effect = RuntimeError(
        "collection already exists"
    )

    with pytest.raises(
        ChromaStoreError,
        match="Failed to create Chroma collection",
    ) as exc_info:
        store_module.create_policy_collection(client)

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_get_policy_collection_opens_existing_collection() -> None:
    """The read path must open rather than create the policy collection."""

    client = Mock()

    collection = Mock()
    collection.name = COLLECTION_NAME
    collection.configuration = {
        "hnsw": {
            "space": DISTANCE_METRIC,
        },
        "embedding_function": None,
    }

    client.get_collection.return_value = collection

    result = store_module.get_policy_collection(client)

    assert result is collection

    client.get_collection.assert_called_once_with(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    client.create_collection.assert_not_called()


def test_get_policy_collection_wraps_missing_collection() -> None:
    """A missing collection must fail rather than be silently created."""

    client = Mock()

    client.get_collection.side_effect = RuntimeError(
        "collection does not exist"
    )

    with pytest.raises(
        ChromaStoreError,
        match="Failed to open Chroma collection",
    ) as exc_info:
        store_module.get_policy_collection(client)

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    client.create_collection.assert_not_called()


@pytest.mark.parametrize(
    ("name", "configuration", "expected_message"),
    [
        (
            "wrong_collection",
            {
                "hnsw": {
                    "space": DISTANCE_METRIC,
                },
                "embedding_function": None,
            },
            "collection name",
        ),
        (
            COLLECTION_NAME,
            None,
            "collection configuration",
        ),
        (
            COLLECTION_NAME,
            {
                "hnsw": None,
                "embedding_function": None,
            },
            "HNSW configuration",
        ),
        (
            COLLECTION_NAME,
            {
                "hnsw": {
                    "space": "l2",
                },
                "embedding_function": None,
            },
            "distance metric",
        ),
        (
            COLLECTION_NAME,
            {
                "hnsw": {
                    "space": DISTANCE_METRIC,
                },
                "embedding_function": "unexpected",
            },
            "must not define an embedding function",
        ),
    ],
)
def test_validate_collection_contract_rejects_incompatible_collection(
    name: str,
    configuration: object,
    expected_message: str,
) -> None:
    """Any incompatible collection must fail the frozen CP8 contract."""

    collection = Mock()
    collection.name = name
    collection.configuration = configuration

    with pytest.raises(
        ChromaStoreError,
        match=expected_message,
    ):
        store_module._validate_collection_contract(
            collection
        )


def test_validate_collection_contract_accepts_frozen_configuration() -> None:
    """A correctly configured collection must pass validation."""

    collection = Mock()
    collection.name = COLLECTION_NAME
    collection.configuration = {
        "hnsw": {
            "space": DISTANCE_METRIC,
        },
        "embedding_function": None,
    }

    store_module._validate_collection_contract(
        collection
    )


def test_prepare_chroma_records_preserves_alignment() -> None:
    """Canonical records and embeddings must remain index-aligned."""


    chunks = [
        {
            "chunk_id": "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "First policy passage.",
            "token_count": 4,
            "source_format": "md",
        },
        {
            "chunk_id": "HR-POL-002__0000__bbbbbbbbbbbbbbbb",
            "doc_id": "HR-POL-002",
            "title": "Leave Policy",
            "section_path": [
                "Leave Policy",
                "Annual Leave",
            ],
            "section_order": 1,
            "chunk_index": 0,
            "text": "Second policy passage.",
            "token_count": 4,
            "source_format": "md",
        },
    ]

    embeddings = np.zeros(
        (2, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0

    records = store_module.prepare_chroma_records(
        chunks,
        embeddings,
    )

    assert records["ids"] == [
        "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
        "HR-POL-002__0000__bbbbbbbbbbbbbbbb",
    ]

    assert records["documents"] == [
        "First policy passage.",
        "Second policy passage.",
    ]

    assert records["embeddings"] is embeddings

    assert records["metadatas"] == [
        {
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "source_format": "md",
            "snippet": "First policy passage.",
        },
        {
            "doc_id": "HR-POL-002",
            "title": "Leave Policy",
            "section_path": [
                "Leave Policy",
                "Annual Leave",
            ],
            "source_format": "md",
            "snippet": "Second policy passage.",
        },
    ]


def test_prepare_chroma_records_does_not_mutate_section_path() -> None:
    """Prepared metadata must not share mutable section-path lists."""

    section_path = [
        "Remote Work Policy",
        "International Remote Work",
    ]

    chunks = [
        {
            "chunk_id": "HR-POL-004__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-004",
            "title": "Remote Work Policy",
            "section_path": section_path,
            "section_order": 1,
            "chunk_index": 0,
            "text": "Approval is required.",
            "token_count": 4,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = store_module.prepare_chroma_records(
        chunks,
        embeddings,
    )

    prepared_path = records["metadatas"][0]["section_path"]

    assert prepared_path == section_path
    assert prepared_path is not section_path


def test_prepare_chroma_records_rejects_empty_chunks() -> None:
    """An empty index payload must fail before Chroma is called."""

    embeddings = np.empty(
        (0, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="at least one canonical record",
    ):
        store_module.prepare_chroma_records(
            [],
            embeddings,
        )


def test_prepare_chroma_records_rejects_non_list_chunks() -> None:
    """Canonical storage input must use the JSON list representation."""

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError,
        match="chunks must be a list",
    ):
        store_module.prepare_chroma_records(
            (),  # type: ignore[arg-type]
            embeddings,
        )


def test_prepare_chroma_records_rejects_wrong_embedding_type() -> None:
    """Storage must reject embeddings that are not NumPy arrays."""

    chunks = [
        {
            "chunk_id": "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Policy passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    with pytest.raises(
        TypeError,
        match="embeddings must be a numpy.ndarray",
    ):
        store_module.prepare_chroma_records(
            chunks,
            [[0.0] * store_module.EMBEDDING_DIMENSION],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 383),
        (1, 385),
        (2, 384),
    ],
)
def test_prepare_chroma_records_rejects_wrong_embedding_shape(
    shape: tuple[int, int],
) -> None:
    """Embedding rows and dimensions must match the chunk payload."""


    chunks = [
        {
            "chunk_id": "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Policy passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        shape,
        dtype=np.float32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="unexpected shape",
    ):
        store_module.prepare_chroma_records(
            chunks,
            embeddings,
        )


def test_prepare_chroma_records_rejects_integer_embeddings() -> None:
    """Integer vectors must not enter the Chroma index."""

    chunks = [
        {
            "chunk_id": "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Policy passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.int32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="floating-point",
    ):
        store_module.prepare_chroma_records(
            chunks,
            embeddings,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_prepare_chroma_records_rejects_non_finite_embeddings(
    bad_value: float,
) -> None:
    """NaN and infinite vectors must fail before Chroma persistence."""


    chunks = [
        {
            "chunk_id": "HR-POL-001__0000__aaaaaaaaaaaaaaaa",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Policy passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    embeddings[0, 0] = bad_value

    with pytest.raises(
        ChromaStoreError,
        match="non-finite",
    ):
        store_module.prepare_chroma_records(
            chunks,
            embeddings,
        )


def test_prepare_chroma_records_rejects_duplicate_ids() -> None:
    """Duplicate canonical IDs must never reach Chroma."""

    chunks = [
        {
            "chunk_id": "duplicate-id",
            "doc_id": "HR-POL-001",
            "title": "Employee Handbook",
            "section_path": ["Employee Handbook"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "First.",
            "token_count": 1,
            "source_format": "md",
        },
        {
            "chunk_id": "duplicate-id",
            "doc_id": "HR-POL-002",
            "title": "Leave Policy",
            "section_path": ["Leave Policy"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Second.",
            "token_count": 1,
            "source_format": "md",
        },
    ]

    embeddings = np.zeros(
        (2, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="Duplicate chunk_id",
    ):
        store_module.prepare_chroma_records(
            chunks,
            embeddings,
        )

def test_add_chroma_records_calls_collection_add() -> None:
    """Prepared records must be passed unchanged to Chroma."""

    collection = Mock()

    embeddings = np.zeros(
        (2, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": [
            "synthetic-001",
            "synthetic-002",
        ],
        "documents": [
            "First passage.",
            "Second passage.",
        ],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": [
                    "Synthetic Policy",
                    "Section 1",
                ],
                "source_format": "md",
                "snippet": "First passage.",
            },
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": [
                    "Synthetic Policy",
                    "Section 2",
                ],
                "source_format": "md",
                "snippet": "Second passage.",
            },
        ],
    }

    store_module.add_chroma_records(
        collection,
        records,
    )

    collection.add.assert_called_once_with(
        ids=records["ids"],
        documents=records["documents"],
        embeddings=records["embeddings"],
        metadatas=records["metadatas"],
    )


def test_add_chroma_records_rejects_non_dict_records() -> None:
    """The write helper must receive a prepared record dictionary."""

    collection = Mock()

    with pytest.raises(
        TypeError,
        match="records must be a ChromaRecords dictionary",
    ):
        store_module.add_chroma_records(
            collection,
            [],  # type: ignore[arg-type]
        )

    collection.add.assert_not_called()


@pytest.mark.parametrize(
    "missing_key",
    [
        "ids",
        "documents",
        "embeddings",
        "metadatas",
    ],
)
def test_add_chroma_records_rejects_missing_payload_key(
    missing_key: str,
) -> None:
    """Incomplete prepared payloads must fail before Chroma is called."""

    collection = Mock()

    records: dict[str, object] = {
        "ids": ["synthetic-001"],
        "documents": ["Synthetic passage."],
        "embeddings": np.zeros(
            (1, store_module.EMBEDDING_DIMENSION),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": ["Synthetic Policy"],
                "source_format": "md",
                "snippet": "Synthetic passage.",
            }
        ],
    }

    del records[missing_key]

    with pytest.raises(
        ChromaStoreError,
        match="missing required keys",
    ):
        store_module.add_chroma_records(
            collection,
            records,  # type: ignore[arg-type]
        )

    collection.add.assert_not_called()


def test_add_chroma_records_wraps_chroma_failure() -> None:
    """Low-level Chroma insertion errors must become project errors."""

    collection = Mock()
    collection.add.side_effect = RuntimeError(
        "write failed"
    )

    records = {
        "ids": [
            "synthetic-001",
            "synthetic-002",
        ],
        "documents": [
            "First passage.",
            "Second passage.",
        ],
        "embeddings": np.zeros(
            (2, store_module.EMBEDDING_DIMENSION),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": ["Synthetic Policy"],
                "source_format": "md",
                "snippet": "First passage.",
            },
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": ["Synthetic Policy"],
                "source_format": "md",
                "snippet": "Second passage.",
            },
        ],
    }

    with pytest.raises(
        ChromaStoreError,
        match="synthetic-001",
    ) as exc_info:
        store_module.add_chroma_records(
            collection,
            records,
        )

    assert "synthetic-002" in str(
        exc_info.value
    )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

def test_build_index_composes_verified_storage_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Index building must compose the verified storage primitives."""

    chunks = [
        {
            "chunk_id": "synthetic-001",
            "doc_id": "HR-POL-TEST",
            "title": "Synthetic Policy",
            "section_path": ["Synthetic Policy"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Synthetic passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": ["synthetic-001"],
        "documents": ["Synthetic passage."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": ["Synthetic Policy"],
                "source_format": "md",
                "snippet": "Synthetic passage.",
            }
        ],
    }

    client = Mock()
    collection = Mock()
    collection.count.return_value = 1

    prepare = Mock(
        return_value=records
    )
    get_client = Mock(
        return_value=client
    )
    create_collection = Mock(
        return_value=collection
    )
    add_records = Mock()

    monkeypatch.setattr(
        store_module,
        "prepare_chroma_records",
        prepare,
    )
    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        get_client,
    )
    monkeypatch.setattr(
        store_module,
        "create_policy_collection",
        create_collection,
    )
    monkeypatch.setattr(
        store_module,
        "add_chroma_records",
        add_records,
    )

    chroma_dir = tmp_path.resolve()

    result = store_module.build_index(
        chunks,
        embeddings,
        chroma_dir,
    )

    assert result is None

    prepare.assert_called_once_with(
        chunks,
        embeddings,
    )

    get_client.assert_called_once_with(
        chroma_dir
    )

    create_collection.assert_called_once_with(
        client
    )

    add_records.assert_called_once_with(
        collection,
        records,
    )

    collection.count.assert_called_once_with()


def test_build_index_rejects_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A partial or inconsistent Chroma build must fail immediately."""

    chunks = [
        {
            "chunk_id": "synthetic-001",
            "doc_id": "HR-POL-TEST",
            "title": "Synthetic Policy",
            "section_path": ["Synthetic Policy"],
            "section_order": 0,
            "chunk_index": 0,
            "text": "Synthetic passage.",
            "token_count": 3,
            "source_format": "md",
        }
    ]

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": ["synthetic-001"],
        "documents": ["Synthetic passage."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Synthetic Policy",
                "section_path": ["Synthetic Policy"],
                "source_format": "md",
                "snippet": "Synthetic passage.",
            }
        ],
    }

    collection = Mock()
    collection.count.return_value = 0

    monkeypatch.setattr(
        store_module,
        "prepare_chroma_records",
        Mock(return_value=records),
    )
    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        store_module,
        "create_policy_collection",
        Mock(return_value=collection),
    )
    monkeypatch.setattr(
        store_module,
        "add_chroma_records",
        Mock(),
    )

    with pytest.raises(
        ChromaStoreError,
        match="count does not match",
    ):
        store_module.build_index(
            chunks,
            embeddings,
            tmp_path.resolve(),
        )


def test_build_index_validates_records_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid canonical input must fail before persistent state starts."""

    chunks: list[dict[str, object]] = []

    embeddings = np.empty(
        (0, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    get_client = Mock()

    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        get_client,
    )

    with pytest.raises(
        ChromaStoreError,
        match="at least one canonical record",
    ):
        store_module.build_index(
            chunks,
            embeddings,
            tmp_path.resolve(),
        )

    get_client.assert_not_called()


def test_validate_index_integrity_is_id_keyed() -> None:
    """Validation must succeed even when Chroma returns another order."""

    collection = Mock()

    expected_embeddings = np.zeros(
        (2, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    expected_embeddings[0, 0] = 1.0
    expected_embeddings[1, 1] = 1.0

    records = {
        "ids": [
            "chunk-a",
            "chunk-b",
        ],
        "documents": [
            "Document A.",
            "Document B.",
        ],
        "embeddings": expected_embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-A",
                "title": "Policy A",
                "section_path": ["Policy A"],
                "source_format": "md",
                "snippet": "Document A.",
            },
            {
                "doc_id": "HR-POL-B",
                "title": "Policy B",
                "section_path": ["Policy B"],
                "source_format": "md",
                "snippet": "Document B.",
            },
        ],
    }

    collection.get.return_value = {
        "ids": [
            "chunk-b",
            "chunk-a",
        ],
        "documents": [
            "Document B.",
            "Document A.",
        ],
        "metadatas": [
            records["metadatas"][1],
            records["metadatas"][0],
        ],
        "embeddings": np.asarray(
            [
                expected_embeddings[1],
                expected_embeddings[0],
            ],
            dtype=np.float64,
        ),
    }

    store_module.validate_index_integrity(
        collection,
        records,
    )

    collection.get.assert_called_once_with(
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )


def test_validate_index_integrity_rejects_id_set_mismatch() -> None:
    """Missing or unexpected stored IDs must fail validation."""

    collection = Mock()

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": ["expected-id"],
        "documents": ["Expected document."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Expected document.",
            }
        ],
    }

    collection.get.return_value = {
        "ids": ["unexpected-id"],
        "documents": ["Expected document."],
        "metadatas": records["metadatas"],
        "embeddings": embeddings.astype(
            np.float64
        ),
    }

    with pytest.raises(
        ChromaStoreError,
        match="ID set mismatch",
    ):
        store_module.validate_index_integrity(
            collection,
            records,
        )


def test_validate_index_integrity_rejects_document_mismatch() -> None:
    """Stored document text must exactly match canonical text."""

    collection = Mock()

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": ["chunk-a"],
        "documents": ["Expected document."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Expected document.",
            }
        ],
    }

    collection.get.return_value = {
        "ids": ["chunk-a"],
        "documents": ["Wrong document."],
        "metadatas": records["metadatas"],
        "embeddings": embeddings.astype(
            np.float64
        ),
    }

    with pytest.raises(
        ChromaStoreError,
        match="document mismatch",
    ):
        store_module.validate_index_integrity(
            collection,
            records,
        )


def test_validate_index_integrity_rejects_metadata_mismatch() -> None:
    """Stored citation metadata must exactly match prepared metadata."""

    collection = Mock()

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    collection.get.return_value = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "metadatas": [
            {
                **records["metadatas"][0],
                "title": "Wrong title",
            }
        ],
        "embeddings": embeddings.astype(
            np.float64
        ),
    }

    with pytest.raises(
        ChromaStoreError,
        match="metadata mismatch",
    ):
        store_module.validate_index_integrity(
            collection,
            records,
        )


def test_validate_index_integrity_rejects_embedding_mismatch() -> None:
    """Stored vector values must remain numerically equivalent."""

    collection = Mock()

    expected_embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    stored_embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float64,
    )
    stored_embeddings[0, 0] = 0.01

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": expected_embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    collection.get.return_value = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "metadatas": records["metadatas"],
        "embeddings": stored_embeddings,
    }

    with pytest.raises(
        ChromaStoreError,
        match="embedding mismatch",
    ):
        store_module.validate_index_integrity(
            collection,
            records,
        )


def test_validate_index_integrity_rejects_non_finite_stored_embedding() -> None:
    """NaN or infinite stored vectors must fail validation."""

    collection = Mock()

    embeddings = np.zeros(
        (1, store_module.EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    stored = embeddings.astype(
        np.float64
    )
    stored[0, 0] = np.nan

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": embeddings,
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    collection.get.return_value = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "metadatas": records["metadatas"],
        "embeddings": stored,
    }

    with pytest.raises(
        ChromaStoreError,
        match="non-finite",
    ):
        store_module.validate_index_integrity(
            collection,
            records,
        )


def test_validate_index_integrity_wraps_get_failure() -> None:
    """Chroma read failures must become project-owned storage errors."""

    collection = Mock()
    collection.get.side_effect = RuntimeError(
        "read failed"
    )

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": np.zeros(
            (1, store_module.EMBEDDING_DIMENSION),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    with pytest.raises(
        ChromaStoreError,
        match="Failed to read Chroma records",
    ) as exc_info:
        store_module.validate_index_integrity(
            collection,
            records,
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_validate_persisted_index_uses_fresh_client_and_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persistence validation must reopen through the public read path."""

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": np.zeros(
            (1, store_module.EMBEDDING_DIMENSION),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    client = Mock()
    collection = Mock()

    get_client = Mock(
        return_value=client
    )
    get_collection = Mock(
        return_value=collection
    )
    validate_integrity = Mock()

    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        get_client,
    )
    monkeypatch.setattr(
        store_module,
        "get_policy_collection",
        get_collection,
    )
    monkeypatch.setattr(
        store_module,
        "validate_index_integrity",
        validate_integrity,
    )

    chroma_dir = tmp_path.resolve()

    result = store_module.validate_persisted_index(
        chroma_dir,
        records,
    )

    assert result is None

    get_client.assert_called_once_with(
        chroma_dir
    )

    get_collection.assert_called_once_with(
        client
    )

    validate_integrity.assert_called_once_with(
        collection,
        records,
    )


def test_validate_persisted_index_does_not_create_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persistence validation must remain strictly read-only."""

    client = Mock()
    collection = Mock()

    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        store_module,
        "get_policy_collection",
        Mock(return_value=collection),
    )
    monkeypatch.setattr(
        store_module,
        "validate_index_integrity",
        Mock(),
    )

    store_module.validate_persisted_index(
        tmp_path.resolve(),
        {
            "ids": ["chunk-a"],
            "documents": ["Document."],
            "embeddings": np.zeros(
                (1, store_module.EMBEDDING_DIMENSION),
                dtype=np.float32,
            ),
            "metadatas": [
                {
                    "doc_id": "HR-POL-TEST",
                    "title": "Policy",
                    "section_path": ["Policy"],
                    "source_format": "md",
                    "snippet": "Document.",
                }
            ],
        },
    )

    client.create_collection.assert_not_called()


def test_validate_persisted_index_propagates_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing or unreadable persisted state must fail clearly."""

    failure = ChromaStoreError(
        "Failed to open Chroma collection 'policy_chunks'."
    )

    monkeypatch.setattr(
        store_module,
        "get_chroma_client",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        store_module,
        "get_policy_collection",
        Mock(side_effect=failure),
    )

    records = {
        "ids": ["chunk-a"],
        "documents": ["Document."],
        "embeddings": np.zeros(
            (1, store_module.EMBEDDING_DIMENSION),
            dtype=np.float32,
        ),
        "metadatas": [
            {
                "doc_id": "HR-POL-TEST",
                "title": "Policy",
                "section_path": ["Policy"],
                "source_format": "md",
                "snippet": "Document.",
            }
        ],
    }

    with pytest.raises(
        ChromaStoreError,
        match="Failed to open Chroma collection",
    ):
        store_module.validate_persisted_index(
            tmp_path.resolve(),
            records,
        )


def test_validate_semantic_smoke_accepts_expected_doc_within_top_k() -> None:
    """Smoke validation must not require the expected policy at rank one."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )
    query_embedding[0] = 1.0

    collection.query.return_value = {
        "ids": [[
            "chunk-a",
            "chunk-b",
        ]],
        "documents": [[
            "First result.",
            "Expected result.",
        ]],
        "metadatas": [[
            {
                "doc_id": "HR-POL-OTHER",
            },
            {
                "doc_id": "HR-POL-004",
            },
        ]],
        "distances": [[
            0.20,
            0.25,
        ]],
    }

    result = store_module.validate_semantic_smoke(
        collection,
        query_embedding,
        expected_doc_id="HR-POL-004",
        n_results=5,
    )

    assert result is None

    collection.query.assert_called_once()

    kwargs = collection.query.call_args.kwargs

    assert kwargs["n_results"] == 5
    assert kwargs["include"] == [
        "documents",
        "metadatas",
        "distances",
    ]

    assert len(
        kwargs["query_embeddings"]
    ) == 1

    assert (
        kwargs["query_embeddings"][0]
        is query_embedding
    )


def test_validate_semantic_smoke_rejects_missing_expected_doc() -> None:
    """A semantically disconnected policy domain must fail the smoke test."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    collection.query.return_value = {
        "ids": [["chunk-a"]],
        "documents": [["Other result."]],
        "metadatas": [[
            {
                "doc_id": "HR-POL-OTHER",
            }
        ]],
        "distances": [[0.25]],
    }

    with pytest.raises(
        ChromaStoreError,
        match="did not return expected policy document",
    ):
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
        )


def test_validate_semantic_smoke_rejects_wrong_query_shape() -> None:
    """Semantic smoke queries must use the frozen embedding dimension."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION - 1,
        dtype=np.float32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="unexpected shape",
    ):
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
        )

    collection.query.assert_not_called()


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_semantic_smoke_rejects_non_finite_query(
    bad_value: float,
) -> None:
    """Non-finite query embeddings must never reach Chroma."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )
    query_embedding[0] = bad_value

    with pytest.raises(
        ChromaStoreError,
        match="non-finite",
    ):
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
        )

    collection.query.assert_not_called()


@pytest.mark.parametrize(
    "n_results",
    [
        0,
        -1,
    ],
)
def test_validate_semantic_smoke_rejects_non_positive_n_results(
    n_results: int,
) -> None:
    """Semantic smoke result count must be positive."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="n_results must be positive",
    ):
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
            n_results=n_results,
        )

    collection.query.assert_not_called()


def test_validate_semantic_smoke_rejects_non_finite_distance() -> None:
    """Invalid Chroma distances must fail the smoke validation."""

    collection = Mock()

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    collection.query.return_value = {
        "ids": [["chunk-a"]],
        "documents": [["Result."]],
        "metadatas": [[
            {
                "doc_id": "HR-POL-004",
            }
        ]],
        "distances": [[np.nan]],
    }

    with pytest.raises(
        ChromaStoreError,
        match="distances contain non-finite",
    ):
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
        )


def test_validate_semantic_smoke_wraps_query_failure() -> None:
    """Low-level Chroma query failures must become storage errors."""

    collection = Mock()
    collection.query.side_effect = RuntimeError(
        "query failed"
    )

    query_embedding = np.zeros(
        store_module.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    with pytest.raises(
        ChromaStoreError,
        match="Failed to execute Chroma semantic smoke query",
    ) as exc_info:
        store_module.validate_semantic_smoke(
            collection,
            query_embedding,
            expected_doc_id="HR-POL-004",
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_build_index_metadata_matches_frozen_configuration() -> None:
    """Metadata must record the exact current corpus/index configuration."""

    from datetime import datetime, timezone

    created = datetime(
        2026,
        8,
        11,
        10,
        26,
        54,
        302304,
        tzinfo=timezone.utc,
    )

    metadata = store_module.build_index_metadata(
        "1.2",
        created=created,
    )

    assert metadata == {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
        "created": "2026-08-11T10:26:54.302304Z",
    }


def test_build_index_metadata_uses_exact_six_field_schema() -> None:
    """Generated metadata must not silently expand the frozen schema."""

    metadata = store_module.build_index_metadata(
        "1.2"
    )

    assert set(metadata) == {
        "corpus_version",
        "embedding_model",
        "embedding_dimension",
        "chunk_tokens",
        "chunk_overlap",
        "created",
    }


@pytest.mark.parametrize(
    "corpus_version",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_build_index_metadata_rejects_blank_corpus_version(
    corpus_version: str,
) -> None:
    """An unusable corpus identity must fail before metadata generation."""

    with pytest.raises(
        ValueError,
        match="corpus_version must be a non-empty string",
    ):
        store_module.build_index_metadata(
            corpus_version
        )


def test_build_index_metadata_rejects_non_string_corpus_version() -> None:
    """Corpus version must remain an explicit string identity."""

    with pytest.raises(
        TypeError,
        match="corpus_version must be a string",
    ):
        store_module.build_index_metadata(
            1.2  # type: ignore[arg-type]
        )


def test_build_index_metadata_rejects_naive_created_datetime() -> None:
    """Metadata timestamps must never silently depend on local timezone."""

    from datetime import datetime

    with pytest.raises(
        ValueError,
        match="created must be timezone-aware UTC",
    ):
        store_module.build_index_metadata(
            "1.2",
            created=datetime(
                2026,
                8,
                11,
                10,
                26,
                54,
            ),
        )


def test_build_index_metadata_rejects_non_utc_created_datetime() -> None:
    """Only explicit UTC timestamps are accepted in index metadata."""

    from datetime import (
        datetime,
        timedelta,
        timezone,
    )

    non_utc = timezone(
        timedelta(hours=10)
    )

    with pytest.raises(
        ValueError,
        match="created must be timezone-aware UTC",
    ):
        store_module.build_index_metadata(
            "1.2",
            created=datetime(
                2026,
                8,
                11,
                20,
                26,
                54,
                tzinfo=non_utc,
            ),
        )


def test_serialize_index_metadata_produces_stable_utf8_json() -> None:
    """Index metadata serialization must use the project JSON convention."""

    metadata = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
        "created": "2026-08-11T10:26:54.302304Z",
    }

    payload = store_module.serialize_index_metadata(
        metadata
    )

    assert isinstance(
        payload,
        bytes,
    )

    assert payload == (
        b'{"chunk_overlap":50,'
        b'"chunk_tokens":350,'
        b'"corpus_version":"1.2",'
        b'"created":"2026-08-11T10:26:54.302304Z",'
        b'"embedding_dimension":384,'
        b'"embedding_model":"BAAI/bge-small-en-v1.5"}\n'
    )


def test_serialize_index_metadata_has_single_trailing_newline() -> None:
    """Generated metadata must have exactly one terminating LF."""

    payload = store_module.serialize_index_metadata(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    assert b"\r\n" not in payload


def test_serialize_index_metadata_rejects_wrong_schema() -> None:
    """Missing or additional metadata fields must fail explicitly."""

    metadata = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
        "created": "2026-08-11T10:26:54Z",
        "unexpected": "value",
    }

    with pytest.raises(
        ChromaStoreError,
        match="exactly the frozen six fields",
    ):
        store_module.serialize_index_metadata(
            metadata  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("corpus_version", ""),
        ("embedding_model", " "),
        ("created", ""),
    ],
)
def test_serialize_index_metadata_rejects_invalid_string_field(
    field_name: str,
    bad_value: object,
) -> None:
    """Required text metadata must remain non-empty strings."""

    metadata = dict(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    metadata[field_name] = bad_value

    with pytest.raises(
        ChromaStoreError,
        match="must be a non-empty string",
    ):
        store_module.serialize_index_metadata(
            metadata  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("embedding_dimension", 0),
        ("embedding_dimension", True),
        ("chunk_tokens", -1),
        ("chunk_overlap", 0),
    ],
)
def test_serialize_index_metadata_rejects_invalid_integer_field(
    field_name: str,
    bad_value: object,
) -> None:
    """Configuration integers must remain strictly positive integers."""

    metadata = dict(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    metadata[field_name] = bad_value

    with pytest.raises(
        ChromaStoreError,
        match="must be a positive integer",
    ):
        store_module.serialize_index_metadata(
            metadata  # type: ignore[arg-type]
        )


def test_get_index_metadata_path_uses_chroma_directory(
    tmp_path: Path,
) -> None:
    """Metadata must live inside the owning Chroma directory."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    result = store_module.get_index_metadata_path(
        chroma_dir
    )

    assert result == (
        chroma_dir
        / "index_metadata.json"
    )


def test_get_index_metadata_path_rejects_relative_path() -> None:
    """Metadata ownership must never depend on the current directory."""

    with pytest.raises(
        ValueError,
        match="chroma_dir must be absolute",
    ):
        store_module.get_index_metadata_path(
            Path("chroma_db")
        )


def test_write_index_metadata_atomic_publishes_exact_bytes(
    tmp_path: Path,
) -> None:
    """Atomic publication must preserve serialized metadata exactly."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    payload = (
        b'{"corpus_version":"1.2"}\n'
    )

    metadata_path = (
        store_module.write_index_metadata_atomic(
            chroma_dir,
            payload,
        )
    )

    assert metadata_path == (
        chroma_dir
        / "index_metadata.json"
    )

    assert metadata_path.read_bytes() == payload

    assert not (
        chroma_dir
        / ".index_metadata.json.tmp"
    ).exists()


def test_write_index_metadata_atomic_replaces_existing_file(
    tmp_path: Path,
) -> None:
    """A previous metadata artifact must be replaced atomically."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        chroma_dir
        / "index_metadata.json"
    )

    metadata_path.write_bytes(
        b'{"old":true}\n'
    )

    payload = (
        b'{"corpus_version":"1.2"}\n'
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    assert metadata_path.read_bytes() == payload


def test_write_index_metadata_atomic_creates_chroma_directory(
    tmp_path: Path,
) -> None:
    """Metadata publication may create the owning generated directory."""

    chroma_dir = (
        tmp_path
        / "nested"
        / "chroma"
    ).resolve()

    assert not chroma_dir.exists()

    store_module.write_index_metadata_atomic(
        chroma_dir,
        b'{"corpus_version":"1.2"}\n',
    )

    assert chroma_dir.exists()

    assert (
        chroma_dir
        / "index_metadata.json"
    ).exists()


@pytest.mark.parametrize(
    ("chroma_dir", "payload", "exception", "message"),
    [
        (
            "chroma_db",
            b"{}\n",
            TypeError,
            "chroma_dir must be a pathlib.Path instance",
        ),
        (
            Path("chroma_db"),
            b"{}\n",
            ValueError,
            "chroma_dir must be absolute",
        ),
        (
            Path("/tmp/chroma"),
            "{}\n",
            TypeError,
            "payload must be bytes",
        ),
        (
            Path("/tmp/chroma"),
            b"",
            ValueError,
            "payload must be non-empty",
        ),
    ],
)
def test_write_index_metadata_atomic_rejects_invalid_inputs(
    chroma_dir: object,
    payload: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Invalid publication inputs must fail before usable state appears."""

    with pytest.raises(
        exception,
        match=message,
    ):
        store_module.write_index_metadata_atomic(
            chroma_dir,  # type: ignore[arg-type]
            payload,  # type: ignore[arg-type]
        )


def test_write_index_metadata_atomic_cleans_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed publication must not leave misleading temporary metadata."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    temporary_path = (
        chroma_dir
        / ".index_metadata.json.tmp"
    )

    metadata_path = (
        chroma_dir
        / "index_metadata.json"
    )

    def fail_replace(
        source: object,
        target: object,
    ) -> None:
        raise OSError(
            "simulated metadata replace failure"
        )

    monkeypatch.setattr(
        store_module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="simulated metadata replace failure",
    ):
        store_module.write_index_metadata_atomic(
            chroma_dir,
            b'{"corpus_version":"1.2"}\n',
        )

    assert not metadata_path.exists()
    assert not temporary_path.exists()


def test_read_index_metadata_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    """Missing metadata must represent the spec's 'no index' state."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    result = store_module.read_index_metadata(
        chroma_dir
    )

    assert result is None


def test_read_index_metadata_returns_valid_six_field_metadata(
    tmp_path: Path,
) -> None:
    """A valid persisted metadata artifact must round-trip exactly."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = store_module.build_index_metadata(
        "1.2",
        created=store_module.datetime(
            2026,
            8,
            11,
            10,
            26,
            54,
            302304,
            tzinfo=store_module.timezone.utc,
        ),
    )

    payload = store_module.serialize_index_metadata(
        metadata
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    result = store_module.read_index_metadata(
        chroma_dir
    )

    assert result == metadata


def test_read_index_metadata_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed persisted JSON must never be accepted as index state."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    store_module.write_index_metadata_atomic(
        chroma_dir,
        b"{invalid-json}\n",
    )

    with pytest.raises(
        ChromaStoreError,
        match="contains invalid JSON",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"[]\n",
        b'"metadata"\n',
        b"42\n",
        b"true\n",
        b"null\n",
    ],
)
def test_read_index_metadata_rejects_non_object_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    """Persisted metadata root must always be a JSON object."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    with pytest.raises(
        ChromaStoreError,
        match="must be a JSON object",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


def test_read_index_metadata_rejects_wrong_schema(
    tmp_path: Path,
) -> None:
    """Missing or additional fields must invalidate persisted metadata."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    payload = (
        b'{'
        b'"corpus_version":"1.2",'
        b'"embedding_model":"BAAI/bge-small-en-v1.5",'
        b'"embedding_dimension":384,'
        b'"chunk_tokens":350,'
        b'"chunk_overlap":50,'
        b'"created":"2026-08-11T10:26:54Z",'
        b'"unexpected":true'
        b'}\n'
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    with pytest.raises(
        ChromaStoreError,
        match="exactly the frozen six fields",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("corpus_version", ""),
        ("embedding_model", " "),
        ("created", ""),
    ],
)
def test_read_index_metadata_rejects_invalid_string_field(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    """Persisted text fields must remain non-empty strings."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = dict(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    metadata[field_name] = bad_value

    payload = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    with pytest.raises(
        ChromaStoreError,
        match="must be a non-empty string",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("embedding_dimension", 0),
        ("embedding_dimension", True),
        ("chunk_tokens", -1),
        ("chunk_overlap", 0),
    ],
)
def test_read_index_metadata_rejects_invalid_integer_field(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    """Persisted numeric configuration must be strictly positive ints."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = dict(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    metadata[field_name] = bad_value

    payload = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    with pytest.raises(
        ChromaStoreError,
        match="must be a positive integer",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


@pytest.mark.parametrize(
    "created",
    [
        "2026-08-11",
        "2026-08-11T10:26:54",
        "2026-08-11T10:26:54+10:00",
        "not-a-timestamp",
        "2026-99-99T99:99:99Z",
    ],
)
def test_read_index_metadata_rejects_invalid_created_timestamp(
    tmp_path: Path,
    created: str,
) -> None:
    """Persisted creation time must be a parseable explicit UTC value."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = dict(
        store_module.build_index_metadata(
            "1.2"
        )
    )

    metadata["created"] = created

    payload = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    with pytest.raises(
        ChromaStoreError,
        match="ISO 8601 UTC",
    ):
        store_module.read_index_metadata(
            chroma_dir
        )


def test_read_index_metadata_wraps_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem read failures must cross the project-owned error boundary."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata_path = (
        chroma_dir
        / "index_metadata.json"
    )

    chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    def fail_read_text(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        raise OSError(
            "simulated metadata read failure"
        )

    monkeypatch.setattr(
        Path,
        "read_text",
        fail_read_text,
    )

    with pytest.raises(
        ChromaStoreError,
        match="Failed to read Chroma index metadata",
    ) as exc_info:
        store_module.read_index_metadata(
            chroma_dir
        )

    assert isinstance(
        exc_info.value.__cause__,
        OSError,
    )

def test_is_index_current_returns_true_for_matching_metadata(
    tmp_path: Path,
) -> None:
    """Exact persisted and expected configuration must be current."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = store_module.build_index_metadata(
        "1.2",
        created=store_module.datetime(
            2026,
            8,
            11,
            10,
            26,
            54,
            tzinfo=store_module.timezone.utc,
        ),
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        store_module.serialize_index_metadata(
            metadata
        ),
    )

    assert store_module.is_index_current(
        chroma_dir,
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        chunk_tokens=350,
        chunk_overlap=50,
    )


def test_is_index_current_returns_false_when_metadata_missing(
    tmp_path: Path,
) -> None:
    """A missing metadata artifact must represent a stale index."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    assert not store_module.is_index_current(
        chroma_dir,
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        chunk_tokens=350,
        chunk_overlap=50,
    )


@pytest.mark.parametrize(
    ("field_name", "expected_value"),
    [
        ("corpus_version", "2.0"),
        (
            "embedding_model",
            "different/model",
        ),
        ("embedding_dimension", 768),
        ("chunk_tokens", 400),
        ("chunk_overlap", 40),
    ],
)
def test_is_index_current_returns_false_for_configuration_mismatch(
    tmp_path: Path,
    field_name: str,
    expected_value: object,
) -> None:
    """Any freshness-identity mismatch must invalidate the index."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = store_module.build_index_metadata(
        "1.2",
        created=store_module.datetime(
            2026,
            8,
            11,
            10,
            26,
            54,
            tzinfo=store_module.timezone.utc,
        ),
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        store_module.serialize_index_metadata(
            metadata
        ),
    )

    expected = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
    }

    expected[field_name] = expected_value

    result = store_module.is_index_current(
        chroma_dir,
        corpus_version=expected[
            "corpus_version"
        ],
        embedding_model=expected[
            "embedding_model"
        ],
        embedding_dimension=expected[
            "embedding_dimension"
        ],
        chunk_tokens=expected[
            "chunk_tokens"
        ],
        chunk_overlap=expected[
            "chunk_overlap"
        ],
    )

    assert result is False


@pytest.mark.parametrize(
    "payload",
    [
        b"{invalid-json}\n",
        b"[]\n",
        (
            b'{'
            b'"corpus_version":"1.2",'
            b'"embedding_model":"BAAI/bge-small-en-v1.5"'
            b'}\n'
        ),
    ],
)
def test_is_index_current_returns_false_for_unusable_metadata(
    tmp_path: Path,
    payload: bytes,
) -> None:
    """Malformed persisted metadata must conservatively become stale."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    store_module.write_index_metadata_atomic(
        chroma_dir,
        payload,
    )

    assert not store_module.is_index_current(
        chroma_dir,
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        chunk_tokens=350,
        chunk_overlap=50,
    )


def test_is_index_current_ignores_created_for_freshness(
    tmp_path: Path,
) -> None:
    """Creation time is provenance and must not determine freshness."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    metadata = store_module.build_index_metadata(
        "1.2",
        created=store_module.datetime(
            2025,
            1,
            1,
            0,
            0,
            0,
            tzinfo=store_module.timezone.utc,
        ),
    )

    store_module.write_index_metadata_atomic(
        chroma_dir,
        store_module.serialize_index_metadata(
            metadata
        ),
    )

    assert store_module.is_index_current(
        chroma_dir,
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        chunk_tokens=350,
        chunk_overlap=50,
    )


@pytest.mark.parametrize(
    (
        "field_name",
        "bad_value",
        "exception",
        "message",
    ),
    [
        (
            "corpus_version",
            "",
            ValueError,
            "corpus_version must be a non-empty string",
        ),
        (
            "embedding_model",
            " ",
            ValueError,
            "embedding_model must be a non-empty string",
        ),
        (
            "embedding_dimension",
            True,
            TypeError,
            "embedding_dimension must be an integer",
        ),
        (
            "embedding_dimension",
            0,
            ValueError,
            "embedding_dimension must be positive",
        ),
        (
            "chunk_tokens",
            0,
            ValueError,
            "chunk_tokens must be positive",
        ),
        (
            "chunk_overlap",
            -1,
            ValueError,
            "chunk_overlap must be positive",
        ),
    ],
)
def test_is_index_current_rejects_invalid_expected_configuration(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Programming/configuration errors must not masquerade as staleness."""

    chroma_dir = (
        tmp_path
        / "chroma"
    ).resolve()

    expected = {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
    }

    expected[field_name] = bad_value

    with pytest.raises(
        exception,
        match=message,
    ):
        store_module.is_index_current(
            chroma_dir,
            corpus_version=expected[
                "corpus_version"
            ],
            embedding_model=expected[
                "embedding_model"
            ],
            embedding_dimension=expected[
                "embedding_dimension"
            ],
            chunk_tokens=expected[
                "chunk_tokens"
            ],
            chunk_overlap=expected[
                "chunk_overlap"
            ],
        )
