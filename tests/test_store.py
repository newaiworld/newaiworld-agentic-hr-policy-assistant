"""Focused tests for the S4 Chroma storage lifecycle."""

from __future__ import annotations

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