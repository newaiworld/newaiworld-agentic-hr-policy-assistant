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

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from rag.ingest import ParsedSection

EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"

TARGET_CHUNK_TOKENS: Final[int] = 350
MAX_CHUNK_TOKENS: Final[int] = 450
CHUNK_OVERLAP_TOKENS: Final[int] = 50
_PARAGRAPH_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(
    r"\n[ \t]*\n+"
)
_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])(?:[ \t]+|\n+)(?=[A-Z0-9\"'(])"
)
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

def _split_into_paragraphs(text: str) -> tuple[str, ...]:
    """Split normalized text at blank-line paragraph boundaries.

    The helper preserves paragraph order and returns only non-empty
    semantic units. It does not perform line, sentence, token, or
    chunk splitting.

    Args:
        text:
            Normalized section text.

    Returns:
        Paragraph strings in original source order. Empty or
        whitespace-only input returns an empty tuple.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text.strip():
        return ()

    return tuple(
        part.strip()
        for part in _PARAGRAPH_BOUNDARY_RE.split(text)
        if part.strip()
    )
def _split_into_lines(text: str) -> tuple[str, ...]:
    """Split normalized text at single-line boundaries.

    The helper preserves line order and returns only non-empty
    semantic units. It does not perform paragraph, sentence, token,
    or chunk splitting.

    Args:
        text:
            Normalized section text.

    Returns:
        Non-empty lines in original source order. Empty or
        whitespace-only input returns an empty tuple.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text.strip():
        return ()

    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )
def _split_into_sentences(text: str) -> tuple[str, ...]:
    """Split normalized prose at conservative sentence boundaries.

    The helper preserves sentence order and returns only non-empty
    semantic units. It does not perform paragraph, line, token, or
    chunk splitting.

    Args:
        text:
            Normalized section text.

    Returns:
        Sentence strings in original source order. Text containing
        no recognized sentence boundary remains one unit. Empty or
        whitespace-only input returns an empty tuple.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text.strip():
        return ()

    return tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(text)
        if sentence.strip()
    )

def _split_at_token_boundary(
    text: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Split text after at most ``max_tokens`` exact content tokens.

    The split uses fast-tokenizer character offsets so the returned
    strings are slices of the original normalized text rather than
    re-decoded token sequences.

    Args:
        text:
            Normalized text to split.
        max_tokens:
            Maximum exact tokenizer tokens allowed in the prefix.

    Returns:
        A ``(prefix, suffix)`` tuple. If ``text`` already fits within
        ``max_tokens``, the suffix is an empty string.

    Raises:
        TypeError:
            If ``text`` is not a string.
        ValueError:
            If ``max_tokens`` is not a positive integer.
        RuntimeError:
            If tokenizer offsets are invalid or no forward-progress
            split can be determined.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "max_tokens must be a positive integer."
        )

    if not text:
        return ("", "")

    tokenizer = get_tokenizer()

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenization failed for token-boundary splitting: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")

    if not isinstance(input_ids, list):
        raise RuntimeError(
            "Tokenizer did not return input_ids as a list."
        )

    if not isinstance(offsets, list):
        raise RuntimeError(
            "Tokenizer did not return offset_mapping as a list."
        )

    if len(input_ids) != len(offsets):
        raise RuntimeError(
            "Tokenizer input_ids and offset_mapping lengths differ."
        )

    if len(input_ids) <= max_tokens:
        return (text, "")

    split_offset = offsets[max_tokens - 1]

    if (
        not isinstance(split_offset, tuple)
        or len(split_offset) != 2
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            for value in split_offset
        )
    ):
        raise RuntimeError(
            "Tokenizer returned an invalid split offset."
        )

    _, split_end = split_offset

    if not 0 < split_end < len(text):
        raise RuntimeError(
            "Token-boundary split did not make forward progress."
        )

    prefix = text[:split_end]
    suffix = text[split_end:]

    if not prefix:
        raise RuntimeError(
            "Token-boundary split produced an empty prefix."
        )

    if len(suffix) >= len(text):
        raise RuntimeError(
            "Token-boundary split did not reduce remaining text."
        )

    return (prefix, suffix)

def _select_best_boundary_cut(
    text: str,
    cut_positions: tuple[int, ...],
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> int | None:
    """Choose the best valid semantic cut in exact source text.

    Candidate positions are character offsets in ``text``. A valid
    candidate must make forward progress and produce a prefix whose
    exact token count does not exceed ``max_tokens``.

    Among valid candidates, choose the prefix whose token count is
    closest to ``target_tokens``. If two candidates are equally
    close, choose the larger prefix.

    Args:
        text:
            Normalized source text being considered for splitting.
        cut_positions:
            Candidate character offsets in deterministic source order.
        target_tokens:
            Preferred exact token count for the resulting prefix.
        max_tokens:
            Absolute maximum exact token count for the prefix.

    Returns:
        The selected character offset, or ``None`` if no candidate
        satisfies the hard token maximum.

    Raises:
        TypeError:
            If ``text`` or ``cut_positions`` has an invalid type.
        ValueError:
            If token budgets or candidate offsets are invalid.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not isinstance(cut_positions, tuple):
        raise TypeError(
            "cut_positions must be a tuple."
        )

    if (
        not isinstance(target_tokens, int)
        or isinstance(target_tokens, bool)
        or target_tokens <= 0
    ):
        raise ValueError(
            "target_tokens must be a positive integer."
        )

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "max_tokens must be a positive integer."
        )

    if target_tokens > max_tokens:
        raise ValueError(
            "target_tokens must not exceed max_tokens."
        )

    if not text:
        return None

    valid_candidates: list[tuple[int, int]] = []

    previous_position = 0

    for position in cut_positions:
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
        ):
            raise ValueError(
                "cut_positions must contain integers."
            )

        if position <= 0 or position >= len(text):
            raise ValueError(
                "cut_positions must be inside the source text."
            )

        if position <= previous_position:
            raise ValueError(
                "cut_positions must be strictly increasing."
            )

        previous_position = position

        candidate_text = text[:position]
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens <= max_tokens:
            valid_candidates.append(
                (
                    position,
                    candidate_tokens,
                )
            )

    if not valid_candidates:
        return None

    selected_position, _ = min(
        valid_candidates,
        key=lambda candidate: (
            abs(candidate[1] - target_tokens),
            -candidate[1],
            -candidate[0],
        ),
    )

    return selected_position

def _paragraph_cut_positions(text: str) -> tuple[int, ...]:
    """Return exact character offsets after paragraph separators.

    Offsets refer to positions in the original normalized source
    string. The returned tuple is strictly increasing.

    Args:
        text:
            Normalized section text.

    Returns:
        Character positions immediately after recognized paragraph
        separators. Empty text or text without paragraph boundaries
        returns an empty tuple.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text:
        return ()

    return tuple(
        match.end()
        for match in _PARAGRAPH_BOUNDARY_RE.finditer(text)
        if 0 < match.end() < len(text)
    )

def _line_cut_positions(text: str) -> tuple[int, ...]:
    """Return exact character offsets after single newline boundaries.

    Args:
        text:
            Normalized section text.

    Returns:
        Strictly increasing character positions immediately after
        newline characters that occur before the end of the text.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text:
        return ()

    return tuple(
        index + 1
        for index, character in enumerate(text)
        if character == "\n"
        and index + 1 < len(text)
    )

def _sentence_cut_positions(text: str) -> tuple[int, ...]:
    """Return exact character offsets after sentence separators.

    The helper uses the same conservative sentence-boundary regex
    as ``_split_into_sentences`` and returns source-relative offsets.

    Args:
        text:
            Normalized section text.

    Returns:
        Strictly increasing character positions immediately after
        each recognized sentence separator.

    Raises:
        TypeError:
            If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if not text:
        return ()

    return tuple(
        match.end()
        for match in _SENTENCE_BOUNDARY_RE.finditer(text)
        if 0 < match.end() < len(text)
    )

def _select_semantic_cut_position(
    text: str,
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> int:
    """Choose the next deterministic cut position for oversized text.

    Semantic boundaries are tried in strict priority order:
    paragraph, line, sentence, then exact token fallback.

    The returned value is a character offset into ``text`` and always
    makes forward progress. Semantic candidates are selected using
    exact token counts and may never exceed ``max_tokens``.

    Args:
        text:
            Normalized text whose exact token count exceeds
            ``max_tokens``.
        target_tokens:
            Preferred chunk size.
        max_tokens:
            Absolute chunk-size ceiling.

    Returns:
        A character offset satisfying ``0 < offset < len(text)``.

    Raises:
        TypeError:
            If ``text`` is not a string.
        ValueError:
            If token budgets are invalid or ``text`` does not
            actually require splitting.
        RuntimeError:
            If no safe forward-progress cut can be produced.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if (
        not isinstance(target_tokens, int)
        or isinstance(target_tokens, bool)
        or target_tokens <= 0
    ):
        raise ValueError(
            "target_tokens must be a positive integer."
        )

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "max_tokens must be a positive integer."
        )

    if target_tokens > max_tokens:
        raise ValueError(
            "target_tokens must not exceed max_tokens."
        )

    if not text:
        raise ValueError(
            "text must be non-empty."
        )

    text_tokens = count_tokens(text)

    if text_tokens <= max_tokens:
        raise ValueError(
            "text does not require splitting."
        )

    boundary_getters = (
        _paragraph_cut_positions,
        _line_cut_positions,
        _sentence_cut_positions,
    )

    for get_positions in boundary_getters:
        cut_position = _select_best_boundary_cut(
            text,
            get_positions(text),
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )

        if cut_position is not None:
            return cut_position

    prefix, suffix = _split_at_token_boundary(
        text,
        target_tokens,
    )

    if not prefix or not suffix:
        raise RuntimeError(
            "token fallback did not produce two non-empty parts."
        )

    cut_position = len(prefix)

    if not 0 < cut_position < len(text):
        raise RuntimeError(
            "semantic cut selection did not make forward progress."
        )

    if prefix + suffix != text:
        raise RuntimeError(
            "token fallback did not preserve exact source text."
        )

    if count_tokens(prefix) > max_tokens:
        raise RuntimeError(
            "token fallback exceeded the hard chunk maximum."
        )

    return cut_position

def split_long_section(
    section: ParsedSection,
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> tuple[Chunk, ...]:
    """Split one normalized long section into non-overlapping chunks.

    This Step 5D implementation preserves exact source order and
    section-local provenance. It prefers semantic boundaries through
    ``_select_semantic_cut_position`` and falls back to tokenizer
    offsets when necessary.

    Overlap is intentionally excluded here and will be added in the
    next checkpoint.

    Args:
        section:
            One normalized ParsedSection.
        target_tokens:
            Preferred chunk size.
        max_tokens:
            Absolute chunk-size ceiling.

    Returns:
        Ordered immutable chunks from exactly one source section.

    Raises:
        TypeError:
            If ``section`` is not a ParsedSection.
        ValueError:
            If token budgets are invalid, the section is empty, or
            the section does not require long-section splitting.
        RuntimeError:
            If splitting fails to preserve source text, token limits,
            or forward progress.
    """

    if not isinstance(section, ParsedSection):
        raise TypeError(
            "section must be a ParsedSection instance."
        )

    if (
        not isinstance(target_tokens, int)
        or isinstance(target_tokens, bool)
        or target_tokens <= 0
    ):
        raise ValueError(
            "target_tokens must be a positive integer."
        )

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "max_tokens must be a positive integer."
        )

    if target_tokens > max_tokens:
        raise ValueError(
            "target_tokens must not exceed max_tokens."
        )

    if not section.text:
        raise ValueError(
            "section text must be non-empty."
        )

    total_tokens = count_tokens(section.text)

    if total_tokens <= max_tokens:
        raise ValueError(
            "section does not require long-section splitting."
        )

    chunks: list[Chunk] = []

    start = 0
    chunk_index = 0

    while start < len(section.text):
        remaining = section.text[start:]

        if not remaining:
            break

        remaining_tokens = count_tokens(remaining)

        if remaining_tokens <= max_tokens:
            end = len(section.text)
        else:
            relative_cut = _select_semantic_cut_position(
                remaining,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
            )

            if not 0 < relative_cut < len(remaining):
                raise RuntimeError(
                    "long-section splitter did not make "
                    "forward progress."
                )

            end = start + relative_cut

        if not start < end <= len(section.text):
            raise RuntimeError(
                "long-section splitter produced invalid "
                "source offsets."
            )

        chunk_text = section.text[start:end]

        if not chunk_text:
            raise RuntimeError(
                "long-section splitter produced empty chunk text."
            )

        token_count = count_tokens(chunk_text)

        if token_count > max_tokens:
            raise RuntimeError(
                "long-section splitter exceeded hard maximum: "
                f"{token_count} > {max_tokens}."
            )

        chunks.append(
            Chunk(
                doc_id=section.doc_id,
                title=section.title,
                section_path=section.section_path,
                section_order=section.section_order,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=token_count,
                source_format=section.source_format,
            )
        )

        start = end
        chunk_index += 1

    if not chunks:
        raise RuntimeError(
            "long-section splitter produced no chunks."
        )

    reconstructed = "".join(
        chunk.text
        for chunk in chunks
    )

    if reconstructed != section.text:
        raise RuntimeError(
            "long-section splitting did not preserve exact "
            "source text."
        )

    if [
        chunk.chunk_index
        for chunk in chunks
    ] != list(range(len(chunks))):
        raise RuntimeError(
            "long-section chunk indexes are not contiguous."
        )

    return tuple(chunks)

def chunk_section(section: ParsedSection) -> tuple[Chunk, ...]:
    """Convert one normalized section into zero or more chunks.

    Empty structural sections produce no chunks. Non-empty sections
    that fit within the configured hard maximum remain one chunk.
    Sections above the hard maximum are delegated to deterministic
    long-section splitting.

    Overlap is intentionally not applied in this checkpoint.

    Args:
        section:
            One normalized ParsedSection from the ingestion layer.

    Returns:
        An empty tuple for an empty structural section, a one-element
        tuple for text within the hard maximum, or multiple ordered
        chunks for an oversized section.

    Raises:
        TypeError:
            If ``section`` is not a ParsedSection.
        RuntimeError:
            If long-section splitting violates its verified contract.
    """

    if not isinstance(section, ParsedSection):
        raise TypeError(
            "section must be a ParsedSection instance."
        )

    token_count = count_tokens(section.text)

    if token_count == 0:
        return ()

    if token_count > MAX_CHUNK_TOKENS:
        return split_long_section(section)

    return (
        Chunk(
            doc_id=section.doc_id,
            title=section.title,
            section_path=section.section_path,
            section_order=section.section_order,
            chunk_index=0,
            text=section.text,
            token_count=token_count,
            source_format=section.source_format,
        ),
    )
