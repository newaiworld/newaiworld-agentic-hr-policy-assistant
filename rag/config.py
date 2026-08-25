"""Frozen lightweight RAG configuration.

This module intentionally contains scalar configuration only so that
application startup and index-currentness checks do not import the local
ML/tokenizer runtime.
"""

from __future__ import annotations

from typing import Final


EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: Final[int] = 384
EMBEDDING_MAX_SEQUENCE_LENGTH: Final[int] = 512

TARGET_CHUNK_TOKENS: Final[int] = 350
MAX_CHUNK_TOKENS: Final[int] = 450

CHUNK_OVERLAP_TOKENS: Final[int] = 50
MIN_CHUNK_OVERLAP_TOKENS: Final[int] = 35
MAX_CHUNK_OVERLAP_TOKENS: Final[int] = 52
