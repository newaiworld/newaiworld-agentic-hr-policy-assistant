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
from typing import Final

from sentence_transformers import SentenceTransformer

from rag.chunk import EMBEDDING_MODEL_NAME


EMBEDDING_DIMENSION: Final[int] = 384
EMBEDDING_MAX_SEQUENCE_LENGTH: Final[int] = 512
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


def _validate_embedding_model(model: SentenceTransformer) -> None:
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
def get_embedding_model() -> SentenceTransformer:
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

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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
