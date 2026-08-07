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

from functools import lru_cache
from typing import Final

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"

TARGET_CHUNK_TOKENS: Final[int] = 350
MAX_CHUNK_TOKENS: Final[int] = 450
CHUNK_OVERLAP_TOKENS: Final[int] = 50


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
