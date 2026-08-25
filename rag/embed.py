"""Local embedding-model lifecycle for the S4 RAG pipeline.

This module owns loading and validating the frozen sentence-transformers
embedding model.

E1 intentionally implements model lifecycle only. Document embedding,
query embedding, Chroma persistence, and retrieval are introduced by
later checkpoints.

Importing this module must not download or load the model. Model loading
is lazy and cached per Python process.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from rag.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_SEQUENCE_LENGTH,
    EMBEDDING_MODEL_NAME,
)

if TYPE_CHECKING:
    from sentence_transformers import (
        SentenceTransformer as SentenceTransformerType,
    )

SentenceTransformer: Any = None
DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 32

QUERY_INSTRUCTION: Final[str] = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingError(RuntimeError):
    """Base exception for embedding-pipeline failures."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the configured embedding model cannot be loaded."""


class EmbeddingModelValidationError(EmbeddingError):
    """Raised when the loaded model violates the frozen embedding contract."""


def _validate_embedding_model(
    model: "SentenceTransformerType",
) -> None:
    """Validate one loaded model against the frozen S4 contract.

    Args:
        model:
            SentenceTransformer instance to validate.

    Raises:
        EmbeddingModelValidationError:
            If the embedding dimension or usable sequence length does not
            match the project contract.
    """

    try:
        dimension = model.get_embedding_dimension()
    except Exception as exc:
        raise EmbeddingModelValidationError(
            "Unable to determine the embedding model dimension."
        ) from exc

    if dimension != EMBEDDING_DIMENSION:
        raise EmbeddingModelValidationError(
            "Embedding model dimension does not match the frozen "
            f"configuration: {dimension!r} != {EMBEDDING_DIMENSION}."
        )

    max_sequence_length = getattr(model, "max_seq_length", None)

    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length < EMBEDDING_MAX_SEQUENCE_LENGTH
    ):
        raise EmbeddingModelValidationError(
            "Embedding model sequence length does not satisfy the "
            "frozen configuration: "
            f"{max_sequence_length!r} < "
            f"{EMBEDDING_MAX_SEQUENCE_LENGTH}."
        )


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformerType":
    """Return the validated embedding model, loaded once per process.

    Loading is deliberately lazy so importing ``rag.embed`` has no model
    download or model-initialisation side effects.

    Returns:
        The validated cached SentenceTransformer instance.

    Raises:
        EmbeddingModelLoadError:
            If the configured model cannot be loaded.
        EmbeddingModelValidationError:
            If the loaded model violates the project's frozen dimension
            or sequence-length contract.
    """

    global SentenceTransformer

    if SentenceTransformer is None:
        from sentence_transformers import (
            SentenceTransformer as sentence_transformer_class,
        )

        SentenceTransformer = sentence_transformer_class

    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    except Exception as exc:
        offline_mode = (
            os.getenv("HF_HUB_OFFLINE") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )

        message = (
            "Failed to load embedding model "
            f"{EMBEDDING_MODEL_NAME!r}."
        )

        if offline_mode:
            message += (
                " Hugging Face offline mode is enabled. If this model "
                "is not already cached locally, disable offline mode "
                "for the initial download."
            )

        raise EmbeddingModelLoadError(message) from exc

    _validate_embedding_model(model)

    return model


def _validate_document_texts(texts: tuple[str, ...]) -> None:
    """Validate ordered document texts before embedding.

    Args:
        texts:
            Ordered tuple of normalized policy-chunk texts.

    Raises:
        TypeError:
            If ``texts`` is not a tuple or contains non-string values.
        ValueError:
            If ``texts`` is empty or contains blank text.
    """

    if not isinstance(texts, tuple):
        raise TypeError(
            "texts must be a tuple of strings."
        )

    if not texts:
        raise ValueError(
            "texts must contain at least one document."
        )

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                "texts must contain only strings; "
                f"item {index} is {type(text).__name__}."
            )

        if not text.strip():
            raise ValueError(
                "texts must not contain blank documents; "
                f"item {index} is blank."
            )


def _validate_batch_size(batch_size: int) -> None:
    """Validate the embedding batch size.

    Args:
        batch_size:
            Number of documents encoded per model batch.

    Raises:
        ValueError:
            If ``batch_size`` is not a positive integer.
    """

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size must be a positive integer."
        )


def _validate_document_embeddings(
    embeddings: object,
    *,
    expected_rows: int,
) -> np.ndarray:
    """Validate and return one document-embedding matrix.

    Args:
        embeddings:
            Raw result returned by ``SentenceTransformer.encode``.
        expected_rows:
            Number of source documents supplied to the encoder.

    Returns:
        Validated NumPy array with shape
        ``(expected_rows, EMBEDDING_DIMENSION)``.

    Raises:
        EmbeddingError:
            If the result is not a NumPy array, has the wrong shape,
            or contains NaN or infinite values.
    """

    if not isinstance(embeddings, np.ndarray):
        raise EmbeddingError(
            "Embedding model returned an unexpected result type: "
            f"{type(embeddings).__name__}; expected numpy.ndarray."
        )

    expected_shape = (
        expected_rows,
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise EmbeddingError(
            "Embedding matrix has an unexpected shape: "
            f"{embeddings.shape!r} != {expected_shape!r}."
        )

    if not np.issubdtype(
        embeddings.dtype,
        np.floating,
    ):
        raise EmbeddingError(
            "Embedding matrix must contain floating-point values; "
            f"received dtype {embeddings.dtype!r}."
        )

    if not np.isfinite(embeddings).all():
        raise EmbeddingError(
            "Embedding matrix contains non-finite values."
        )

    return embeddings


def embed_documents(
    texts: tuple[str, ...],
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """Embed ordered policy documents into normalized dense vectors.

    Document texts are encoded exactly as supplied. No query instruction
    or query-specific prompt is applied at this stage.

    Input order is preserved by the model call and therefore defines the
    row order of the returned matrix.

    Args:
        texts:
            Ordered tuple of non-empty normalized policy-chunk texts.
        batch_size:
            Positive number of texts encoded in each model batch.

    Returns:
        NumPy array with shape ``(len(texts), EMBEDDING_DIMENSION)``.
        Each row is a normalized document embedding corresponding to the
        input text at the same index.

    Raises:
        TypeError:
            If ``texts`` or one of its members violates the input type
            contract.
        ValueError:
            If ``texts`` is empty, contains blank text, or ``batch_size``
            is invalid.
        EmbeddingError:
            If model encoding fails or returns invalid embeddings.
    """

    _validate_document_texts(texts)
    _validate_batch_size(batch_size)

    model = get_embedding_model()

    try:
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise EmbeddingError(
            "Failed to embed document texts."
        ) from exc

    return _validate_document_embeddings(
        embeddings,
        expected_rows=len(texts),
    )


def _validate_query(query: str) -> None:
    """Validate one user query before embedding.

    Args:
        query:
            Raw user search query.

    Raises:
        TypeError:
            If ``query`` is not a string.
        ValueError:
            If ``query`` is empty or contains only whitespace.
    """

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    if not query.strip():
        raise ValueError(
            "query must be a non-empty string."
        )

def embed_query(query: str) -> np.ndarray:
    """Embed one retrieval query into a normalized dense vector.

    The BGE retrieval instruction is prepended exactly once to the
    validated user query. Policy documents are embedded separately by
    ``embed_documents`` without this query-specific instruction.

    Args:
        query:
            Non-empty user search query.

    Returns:
        One normalized NumPy vector with shape
        ``(EMBEDDING_DIMENSION,)``.

    Raises:
        TypeError:
            If ``query`` is not a string.
        ValueError:
            If ``query`` is empty or contains only whitespace.
        EmbeddingError:
            If model encoding fails or returns an invalid embedding.
    """

    _validate_query(query)

    model = get_embedding_model()

    query_text = QUERY_INSTRUCTION + query

    try:
        embedding = model.encode(
            query_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise EmbeddingError(
            "Failed to embed query text."
        ) from exc

    if not isinstance(embedding, np.ndarray):
        raise EmbeddingError(
            "Query embedding model returned an unexpected result type: "
            f"{type(embedding).__name__}; expected numpy.ndarray."
        )

    expected_shape = (
        EMBEDDING_DIMENSION,
    )

    if embedding.shape != expected_shape:
        raise EmbeddingError(
            "Query embedding has an unexpected shape: "
            f"{embedding.shape!r} != {expected_shape!r}."
        )

    if not np.issubdtype(
        embedding.dtype,
        np.floating,
    ):
        raise EmbeddingError(
            "Query embedding must contain floating-point values; "
            f"received dtype {embedding.dtype!r}."
        )

    if not np.isfinite(embedding).all():
        raise EmbeddingError(
            "Query embedding contains non-finite values."
        )

    return embedding
