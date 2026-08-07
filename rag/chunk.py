"""Deterministic token measurement and heading-aware chunking.

This module owns the S4 token-counting and chunking boundary.

Implementation is introduced incrementally:

    normalized ParsedSection
        -> exact tokenizer integration
        -> token measurement
        -> heading-aware chunk splitting
        -> deterministic chunk IDs

CP4 implements tokenizer loading and exact token counting only.
CP5 will add chunk generation. Importing this module must not
download models, read corpus files, or create generated artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"

TARGET_CHUNK_TOKENS: Final[int] = 350
MAX_CHUNK_TOKENS: Final[int] = 450
CHUNK_OVERLAP_TOKENS: Final[int] = 50
@dataclass(frozen=True)
class Chunk:
    """Represent one text chunk from exactly one policy section.

    A chunk preserves the originating section's provenance while
    storing only information owned by the chunking stage.

    ``chunk_index`` is zero-based within the originating section.
    Deterministic chunk IDs, snippets, embeddings, and retrieval
    scores are intentionally handled by later pipeline stages.

    Attributes:
        doc_id:
            Stable policy document identifier.
        title:
            Human-readable policy title.
        section_path:
            Complete heading hierarchy for the source section.
        section_order:
            Stable zero-based order of the source section within
            its parsed document.
        chunk_index:
            Stable zero-based order of this chunk within its
            originating section.
        text:
            Non-empty normalized text contained in the chunk.
        token_count:
            Exact tokenizer count for ``text``.
        source_format:
            Source format inherited from the parsed section.
    """

    doc_id: str
    title: str
    section_path: tuple[str, ...]
    section_order: int
    chunk_index: int
    text: str
    token_count: int
    source_format: str

    def __post_init__(self) -> None:
        """Validate invariants intrinsic to a materialized chunk."""

        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError(
                "doc_id must be a non-empty string."
            )

        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError(
                "title must be a non-empty string."
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

        if (
            not isinstance(self.section_order, int)
            or isinstance(self.section_order, bool)
            or self.section_order < 0
        ):
            raise ValueError(
                "section_order must be a non-negative integer."
            )

        if (
            not isinstance(self.chunk_index, int)
            or isinstance(self.chunk_index, bool)
            or self.chunk_index < 0
        ):
            raise ValueError(
                "chunk_index must be a non-negative integer."
            )

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                "text must be a non-empty string."
            )

        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count <= 0
        ):
            raise ValueError(
                "token_count must be a positive integer."
            )

        if self.token_count > MAX_CHUNK_TOKENS:
            raise ValueError(
                "token_count exceeds the configured hard chunk "
                f"maximum: {self.token_count} > "
                f"{MAX_CHUNK_TOKENS}."
            )

        if (
            not isinstance(self.source_format, str)
            or not self.source_format.strip()
        ):
            raise ValueError(
                "source_format must be a non-empty string."
            )

class TokenizerLoadError(RuntimeError):
    """Raised when the frozen tokenizer cannot be loaded safely."""


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load and cache the frozen tokenizer for exact token counting.

    The tokenizer is loaded lazily on first use so importing
    ``rag.chunk`` does not trigger network access or model downloads.

    Returns:
        The fast tokenizer associated with
        ``BAAI/bge-small-en-v1.5``.

    Raises:
        TokenizerLoadError:
            If the tokenizer cannot be loaded, is not a fast tokenizer,
            or does not support the configured hard chunk maximum.
    """

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_NAME,
            use_fast=True,
        )
    except Exception as exc:
        raise TokenizerLoadError(
            "Could not load tokenizer for "
            f"{EMBEDDING_MODEL_NAME!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not tokenizer.is_fast:
        raise TokenizerLoadError(
            "Tokenizer must use the fast backend."
        )

    if tokenizer.model_max_length < MAX_CHUNK_TOKENS:
        raise TokenizerLoadError(
            "Tokenizer model_max_length is below the configured "
            f"hard chunk maximum: "
            f"{tokenizer.model_max_length} < {MAX_CHUNK_TOKENS}."
        )

    return tokenizer


def count_tokens(text: str) -> int:
    """Return the exact BGE tokenizer count for one text string.

    Token counting uses the same frozen tokenizer that will later be
    used to enforce chunk-size limits. Special framing tokens are not
    included because chunk budgets measure document content itself.

    Args:
        text:
            Normalized text to measure.

    Returns:
        The number of tokenizer input IDs produced for the complete
        input text.

    Raises:
        TypeError:
            If ``text`` is not a string.
        TokenizerLoadError:
            If the frozen tokenizer cannot be loaded safely.
        RuntimeError:
            If tokenization does not return a valid input-ID sequence.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    tokenizer = get_tokenizer()

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenization failed for exact token counting: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    input_ids = encoded.get("input_ids")

    if not isinstance(input_ids, list):
        raise RuntimeError(
            "Tokenizer did not return input_ids as a list."
        )

    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise RuntimeError(
            "Tokenizer returned invalid input_ids."
        )

    return len(input_ids)
