"""Citation-ready retrieval contracts for the S4 RAG pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from rag.embed import EmbeddingError, embed_query
from rag.ingest import SUPPORTED_SOURCE_FORMATS
from rag.store import (
    ChromaStoreError,
    get_chroma_client,
    get_policy_collection,
    resolve_chroma_dir,
)


DEFAULT_RETRIEVAL_K: Final[int] = 5
ALLOWED_RETRIEVAL_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "doc_id",
        "source_format",
    }
)


class RetrievalError(RuntimeError):
    """Base exception for retrieval-runtime failures."""


@dataclass(frozen=True)
class ValidatedRetrievalRows:
    """Represent structurally validated rows from one Chroma query.

    Chroma returns one outer list per submitted query. Retrieval submits
    exactly one query, so this contract removes that transport-specific
    outer dimension while preserving aligned raw row values for the
    later result-conversion stage.
    """

    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[dict[str, object], ...]
    distances: tuple[float, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """Represent one validated citation-ready retrieval result.

    The retrieval layer preserves both the concise section value needed
    by downstream MCP/API contracts and the complete section path needed
    for provenance, debugging, and evaluation.

    ``distance`` stores the raw Chroma cosine distance. ``similarity``
    uses the project-facing orientation where higher is better:

        similarity = 1.0 - distance

    Attributes:
        chunk_id:
            Deterministic canonical chunk identifier.
        doc_id:
            Stable policy document identifier.
        title:
            Human-readable policy title.
        section:
            Public citation section, defined as the leaf heading of
            ``section_path``.
        section_path:
            Complete immutable heading hierarchy.
        snippet:
            Non-empty supporting policy text.
        source_format:
            Original corpus source format.
        distance:
            Finite raw cosine distance returned by Chroma.
        similarity:
            Finite similarity score derived as ``1.0 - distance``.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_path: tuple[str, ...]
    snippet: str
    source_format: str
    distance: float
    similarity: float

    def __post_init__(self) -> None:
        """Validate invariants intrinsic to one retrieval result."""

        for field_name in (
            "chunk_id",
            "doc_id",
            "title",
            "section",
            "snippet",
            "source_format",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if (
            not isinstance(self.section_path, tuple)
            or not self.section_path
            or any(
                not isinstance(part, str) or not part.strip()
                for part in self.section_path
            )
        ):
            raise ValueError(
                "section_path must be a non-empty tuple of "
                "non-empty strings."
            )

        if self.title != self.section_path[0]:
            raise ValueError(
                "title must match the first section_path element."
            )

        if self.section != self.section_path[-1]:
            raise ValueError(
                "section must match the final section_path element."
            )

        for field_name in (
            "distance",
            "similarity",
        ):
            value = getattr(self, field_name)

            if (
                not isinstance(value, float)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{field_name} must be a finite float."
                )

        expected_similarity = 1.0 - self.distance

        if not math.isclose(
            self.similarity,
            expected_similarity,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "similarity must equal 1.0 - distance."
            )

def _validate_retrieval_query(
    query: str,
) -> None:
    """Validate one public retrieval query.

    Retrieval owns its request boundary independently from the embedding
    layer so callers receive deterministic validation before any model
    or vector-store work begins.

    Args:
        query:
            Raw policy-search query.

    Raises:
        TypeError:
            If ``query`` is not a string.
        ValueError:
            If ``query`` is empty or contains only whitespace.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "query must be a string."
        )

    if not query.strip():
        raise ValueError(
            "query must be a non-empty string."
        )


def _validate_retrieval_k(
    k: int,
) -> None:
    """Validate one requested retrieval depth.

    Args:
        k:
            Number of nearest policy chunks requested by the caller.

    Raises:
        TypeError:
            If ``k`` is not an integer or is a Boolean.
        ValueError:
            If ``k`` is not positive.
    """

    if (
        not isinstance(
            k,
            int,
        )
        or isinstance(
            k,
            bool,
        )
    ):
        raise TypeError(
            "k must be an integer."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )


def _validate_retrieval_filters(
    filters: dict[str, str] | None,
) -> None:
    """Validate optional public metadata filters for retrieval.

    Only the retrieval fields intentionally exposed by the CP9 contract
    are accepted. Chroma-specific ``where`` construction belongs to a
    later adapter step and is deliberately not performed here.

    Args:
        filters:
            Optional mapping containing ``doc_id`` and/or
            ``source_format`` equality filters.

    Raises:
        TypeError:
            If ``filters`` is not ``None`` or a dictionary, or if a
            filter value is not a string.
        ValueError:
            If a filter key is unsupported, a filter value is blank, or
            ``source_format`` is not a supported corpus format.
    """

    if filters is None:
        return

    if not isinstance(
        filters,
        dict,
    ):
        raise TypeError(
            "filters must be a dictionary or None."
        )

    for field_name in filters:
        if not isinstance(
            field_name,
            str,
        ):
            raise TypeError(
                "retrieval filter keys must be strings."
            )

    unsupported_keys = (
        set(filters)
        - ALLOWED_RETRIEVAL_FILTERS
    )

    if unsupported_keys:
        unsupported = ", ".join(
            sorted(
                str(key)
                for key in unsupported_keys
            )
        )

        raise ValueError(
            "Unsupported retrieval filter key(s): "
            f"{unsupported}."
        )

    for field_name, value in filters.items():
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "retrieval filter "
                f"{field_name!r} must be a string."
            )

        if not value.strip():
            raise ValueError(
                "retrieval filter "
                f"{field_name!r} must be a non-empty string."
            )

    source_format = filters.get(
        "source_format"
    )

    if (
        source_format is not None
        and source_format
        not in SUPPORTED_SOURCE_FORMATS
    ):
        supported = ", ".join(
            sorted(
                SUPPORTED_SOURCE_FORMATS
            )
        )

        raise ValueError(
            "retrieval filter 'source_format' is unsupported: "
            f"{source_format!r}; expected one of {supported}."
        )


def _get_active_policy_collection() -> object:
    """Open the validated active policy collection for runtime retrieval.

    Runtime retrieval is read-only with respect to index lifecycle.
    The configured persistence directory must already exist before
    Chroma client construction so an absent published index is not
    accidentally materialized as an empty database directory.

    Freshness checking and rebuilding belong to application startup and
    deployment lifecycle logic rather than the per-query retrieval path.

    Returns:
        Existing Chroma policy collection satisfying the frozen storage
        contract.

    Raises:
        RetrievalError:
            If the configured active index is missing, is not a
            directory, or the storage layer cannot open and validate
            the existing policy collection.
    """

    try:
        chroma_dir = resolve_chroma_dir()
    except ChromaStoreError as exc:
        raise RetrievalError(
            "Failed to resolve the active Chroma index directory."
        ) from exc

    if not chroma_dir.exists():
        raise RetrievalError(
            "Active Chroma index directory does not exist: "
            f"{str(chroma_dir)!r}."
        )

    if not chroma_dir.is_dir():
        raise RetrievalError(
            "Active Chroma index path is not a directory: "
            f"{str(chroma_dir)!r}."
        )

    try:
        client = get_chroma_client(
            chroma_dir
        )

        return get_policy_collection(
            client
        )
    except ChromaStoreError as exc:
        raise RetrievalError(
            "Failed to open the active policy retrieval index."
        ) from exc


def _compile_chroma_where(
    filters: dict[str, str] | None,
) -> dict[str, object] | None:
    """Compile validated public filters into Chroma ``where`` syntax.

    Public retrieval filters intentionally hide Chroma's logical-filter
    representation. One equality filter maps directly to a single
    Chroma condition. Two filters require the explicit ``$and`` form
    verified against the pinned Chroma runtime.

    Args:
        filters:
            Already-validated public retrieval filters.

    Returns:
        ``None`` when no filters are supplied, one equality condition
        for a single filter, or an explicit ``$and`` expression for
        both supported filters.

    Raises:
        TypeError:
            If the public filter container or values violate the
            retrieval request contract.
        ValueError:
            If keys, values, or source format violate that contract.
    """

    _validate_retrieval_filters(
        filters
    )

    if not filters:
        return None

    conditions = [
        {
            field_name: filters[field_name],
        }
        for field_name in (
            "doc_id",
            "source_format",
        )
        if field_name in filters
    ]

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


def _query_policy_collection_raw(
    query: str,
    *,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: dict[str, str] | None = None,
) -> object:
    """Execute one validated semantic query against the active index.

    This helper owns request validation, query embedding, active-index
    access, Chroma filter compilation, and one raw Chroma query call.
    It deliberately returns the Chroma response unchanged; structural
    response validation belongs to the next retrieval checkpoint.

    Args:
        query:
            Non-empty policy retrieval query.
        k:
            Positive number of nearest records requested.
        filters:
            Optional validated public retrieval filters.

    Returns:
        Raw response returned by the pinned Chroma collection query.

    Raises:
        TypeError:
            If the query, result count, or filter request violates the
            public retrieval type contract.
        ValueError:
            If those request values violate the public retrieval value
            contract.
        RetrievalError:
            If query embedding fails, the active retrieval index cannot
            be opened, or Chroma query execution fails.
    """

    _validate_retrieval_query(
        query
    )
    _validate_retrieval_k(
        k
    )
    _validate_retrieval_filters(
        filters
    )

    try:
        query_embedding = embed_query(
            query
        )
    except EmbeddingError as exc:
        raise RetrievalError(
            "Failed to embed policy retrieval query."
        ) from exc

    collection = _get_active_policy_collection()

    where = _compile_chroma_where(
        filters
    )

    query_kwargs: dict[str, object] = {
        "query_embeddings": [
            query_embedding,
        ],
        "n_results": k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        query_kwargs["where"] = where

    try:
        return collection.query(
            **query_kwargs
        )
    except Exception as exc:
        raise RetrievalError(
            "Failed to execute policy retrieval query."
        ) from exc


def _validate_raw_retrieval_response(
    response: object,
) -> ValidatedRetrievalRows:
    """Validate and unwrap one raw Chroma retrieval response.

    The retrieval adapter submits exactly one query, so each required
    Chroma result field must contain exactly one inner list. Zero rows
    are valid. Non-empty rows must preserve the citation/provenance
    contract written during CP8 index construction.

    Extra top-level Chroma bookkeeping fields are intentionally ignored.

    Args:
        response:
            Raw object returned by ``collection.query()``.

    Returns:
        Immutable aligned retrieval rows with the one-query outer
        dimension removed.

    Raises:
        RetrievalError:
            If the Chroma response violates the frozen retrieval
            structure, row, metadata, or distance contract.
    """

    if not isinstance(
        response,
        dict,
    ):
        raise RetrievalError(
            "Chroma retrieval response must be a dictionary."
        )

    required_fields = (
        "ids",
        "documents",
        "metadatas",
        "distances",
    )

    inner_values: dict[str, list[object]] = {}

    for field_name in required_fields:
        value = response.get(
            field_name
        )

        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], list)
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid "
                f"{field_name!r} structure."
            )

        inner_values[field_name] = value[0]

    ids = inner_values["ids"]
    documents = inner_values["documents"]
    metadatas = inner_values["metadatas"]
    distances = inner_values["distances"]

    result_count = len(ids)

    for field_name, value in (
        ("documents", documents),
        ("metadatas", metadatas),
        ("distances", distances),
    ):
        if len(value) != result_count:
            raise RetrievalError(
                "Chroma retrieval response has misaligned "
                f"{field_name}."
            )

    if (
        all(
            isinstance(
                chunk_id,
                str,
            )
            for chunk_id in ids
        )
        and len(
            set(ids)
        )
        != result_count
    ):
        raise RetrievalError(
            "Chroma retrieval response contains duplicate chunk IDs."
        )

    validated_ids: list[str] = []
    validated_documents: list[str] = []
    validated_metadatas: list[dict[str, object]] = []
    validated_distances: list[float] = []

    for index in range(
        result_count
    ):
        chunk_id = ids[index]
        document = documents[index]
        metadata = metadatas[index]
        distance = distances[index]

        if (
            not isinstance(
                chunk_id,
                str,
            )
            or not chunk_id.strip()
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid chunk ID at "
                f"result {index}."
            )

        if (
            not isinstance(
                document,
                str,
            )
            or not document.strip()
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid document at "
                f"result {index}."
            )

        if not isinstance(
            metadata,
            dict,
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid metadata at "
                f"result {index}."
            )

        required_metadata_fields = (
            "doc_id",
            "title",
            "section_path",
            "snippet",
            "source_format",
        )

        missing_metadata_fields = [
            field_name
            for field_name in required_metadata_fields
            if field_name not in metadata
        ]

        if missing_metadata_fields:
            raise RetrievalError(
                "Chroma retrieval metadata is missing required "
                f"field(s) at result {index}: "
                f"{missing_metadata_fields!r}."
            )

        doc_id = metadata["doc_id"]
        title = metadata["title"]
        section_path = metadata["section_path"]
        snippet = metadata["snippet"]
        source_format = metadata["source_format"]

        for field_name, value in (
            ("doc_id", doc_id),
            ("title", title),
            ("snippet", snippet),
            ("source_format", source_format),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise RetrievalError(
                    "Chroma retrieval metadata has invalid "
                    f"{field_name!r} at result {index}."
                )

        if (
            not isinstance(
                section_path,
                list,
            )
            or not section_path
            or any(
                not isinstance(part, str)
                or not part.strip()
                for part in section_path
            )
        ):
            raise RetrievalError(
                "Chroma retrieval metadata has invalid "
                f"'section_path' at result {index}."
            )

        assert isinstance(
            title,
            str,
        )
        assert isinstance(
            snippet,
            str,
        )
        assert isinstance(
            source_format,
            str,
        )

        if title != section_path[0]:
            raise RetrievalError(
                "Chroma retrieval metadata title does not match "
                f"section_path root at result {index}."
            )

        if (
            source_format
            not in SUPPORTED_SOURCE_FORMATS
        ):
            raise RetrievalError(
                "Chroma retrieval metadata has unsupported "
                f"source_format at result {index}: "
                f"{source_format!r}."
            )

        if snippet != document:
            raise RetrievalError(
                "Chroma retrieval document and citation snippet "
                f"do not match at result {index}."
            )

        if (
            not isinstance(
                distance,
                float,
            )
            or not math.isfinite(
                distance
            )
        ):
            raise RetrievalError(
                "Chroma retrieval response has invalid distance at "
                f"result {index}."
            )

        validated_ids.append(
            chunk_id
        )
        validated_documents.append(
            document
        )
        validated_metadatas.append(
            metadata
        )
        validated_distances.append(
            distance
        )

    return ValidatedRetrievalRows(
        ids=tuple(
            validated_ids
        ),
        documents=tuple(
            validated_documents
        ),
        metadatas=tuple(
            validated_metadatas
        ),
        distances=tuple(
            validated_distances
        ),
    )
