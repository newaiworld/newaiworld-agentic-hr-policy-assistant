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

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rag.ingest import ParsedSection

EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"

TARGET_CHUNK_TOKENS: Final[int] = 350
MAX_CHUNK_TOKENS: Final[int] = 450
CHUNK_OVERLAP_TOKENS: Final[int] = 50
MIN_CHUNK_OVERLAP_TOKENS: Final[int] = 35
MAX_CHUNK_OVERLAP_TOKENS: Final[int] = 52
CHUNK_ID_DIGEST_LENGTH: Final[int] = 16
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

    Deterministic chunk IDs are materialized at the chunking boundary.
    Snippets, embeddings, and retrieval scores are intentionally
    handled by later pipeline stages.

    Attributes:
        chunk_id:
            Deterministic identifier derived from stable provenance,
            chunk index, and exact normalized chunk text.
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

    chunk_id: str
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

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError(
                "chunk_id must be a non-empty string."
            )

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

def generate_chunk_id(
    *,
    doc_id: str,
    section_path: tuple[str, ...],
    chunk_index: int,
    text: str,
) -> str:
    """Return a deterministic identifier for one logical policy chunk.

    Identity is derived only from stable chunk provenance and content:
    document ID, complete section path, zero-based chunk index, and
    exact normalized chunk text.

    The canonical payload is serialized to compact UTF-8 JSON before
    hashing with SHA-256. The visible digest is truncated to
    ``CHUNK_ID_DIGEST_LENGTH`` hexadecimal characters to keep IDs
    concise while retaining ample collision resistance for this
    corpus.

    Args:
        doc_id:
            Stable policy document identifier.
        section_path:
            Complete canonical heading hierarchy for the source
            section.
        chunk_index:
            Zero-based chunk ordinal within the source section.
        text:
            Exact normalized text contained in the chunk.

    Returns:
        An identifier in the form
        ``<doc_id>__<zero-padded chunk index>__<digest>``.

    Raises:
        ValueError:
            If any identity input violates the chunk provenance
            contract.
    """

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(
            "doc_id must be a non-empty string."
        )

    if (
        not isinstance(section_path, tuple)
        or not section_path
        or any(
            not isinstance(part, str) or not part.strip()
            for part in section_path
        )
    ):
        raise ValueError(
            "section_path must be a non-empty tuple of "
            "non-empty strings."
        )

    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise ValueError(
            "chunk_index must be a non-negative integer."
        )

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "text must be a non-empty string."
        )

    payload = {
        "chunk_index": chunk_index,
        "doc_id": doc_id,
        "section_path": list(section_path),
        "text": text,
    }

    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(
        canonical_payload
    ).hexdigest()[:CHUNK_ID_DIGEST_LENGTH]

    return (
        f"{doc_id}__"
        f"{chunk_index:04d}__"
        f"{digest}"
    )
def chunk_to_record(chunk: Chunk) -> dict[str, object]:
    """Convert one materialized Chunk into its canonical JSON record.

    The returned mapping contains only deterministic persisted chunk
    data. Runtime-only values such as embeddings, retrieval scores,
    timestamps, and filesystem paths are intentionally excluded.

    ``section_path`` is converted from its immutable tuple form to a
    JSON-compatible list while preserving exact heading order.

    Args:
        chunk:
            One validated materialized Chunk.

    Returns:
        A new JSON-compatible dictionary containing the complete
        canonical persisted chunk schema.

    Raises:
        TypeError:
            If ``chunk`` is not a Chunk instance.
    """

    if not isinstance(chunk, Chunk):
        raise TypeError(
            "chunk must be a Chunk instance."
        )

    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section_path": list(chunk.section_path),
        "section_order": chunk.section_order,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "source_format": chunk.source_format,
    }

def serialize_chunks(chunks: tuple[Chunk, ...]) -> bytes:
    """Serialize ordered chunks into canonical UTF-8 JSON bytes.

    Chunk ordering is preserved exactly. Each Chunk is converted
    through ``chunk_to_record`` before JSON serialization.

    Canonical output uses sorted object keys, Unicode-preserving JSON,
    LF line endings, and exactly one trailing newline. Runtime-only
    data is excluded by ``chunk_to_record``.

    Args:
        chunks:
            Ordered tuple of validated materialized chunks.

    Returns:
        Canonical UTF-8 JSON bytes ending in exactly one LF byte.

    Raises:
        TypeError:
            If ``chunks`` is not a tuple or contains a non-Chunk item.
    """

    if not isinstance(chunks, tuple):
        raise TypeError(
            "chunks must be a tuple of Chunk instances."
        )

    if any(
        not isinstance(chunk, Chunk)
        for chunk in chunks
    ):
        raise TypeError(
            "chunks must contain only Chunk instances."
        )

    records = [
        chunk_to_record(chunk)
        for chunk in chunks
    ]

    serialized = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return (serialized + "\n").encode("utf-8")

def write_chunks_atomic(
    path: Path,
    payload: bytes,
) -> None:
    """Atomically publish canonical chunk bytes to one destination.

    The payload is first written to a temporary sibling file in the
    destination directory. Only after that write succeeds is the
    temporary file atomically moved into place with ``os.replace``.

    Existing destination files are replaced only after the complete
    new payload has been written successfully.

    Args:
        path:
            Final artifact path, such as
            ``corpus/processed/chunks.json``.
        payload:
            Already-canonical UTF-8 JSON bytes.

    Raises:
        TypeError:
            If ``path`` is not a Path or ``payload`` is not bytes.
        ValueError:
            If ``payload`` is empty.
        OSError:
            If the destination directory cannot be created, the
            temporary file cannot be written, or atomic replacement
            fails.
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path must be a pathlib.Path instance."
        )

    if not isinstance(payload, bytes):
        raise TypeError(
            "payload must be bytes."
        )

    if not payload:
        raise ValueError(
            "payload must be non-empty."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    try:
        temporary_path.write_bytes(payload)

        os.replace(
            temporary_path,
            path,
        )
    except Exception:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise

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

def _select_overlap_start(
    previous_text: str,
    *,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_overlap_tokens: int = MIN_CHUNK_OVERLAP_TOKENS,
    max_overlap_tokens: int = MAX_CHUNK_OVERLAP_TOKENS,
) -> int:
    """Choose a deterministic overlap start within previous chunk text.

    Semantic boundaries are preferred in this order:
    paragraph, line, sentence. A semantic candidate is valid only
    when the exact overlap suffix is within the configured overlap
    window. If no semantic candidate exists, an exact tokenizer
    offset fallback is used near ``overlap_tokens``.

    Args:
        previous_text:
            Exact text of the previously emitted chunk.
        overlap_tokens:
            Preferred overlap size.
        min_overlap_tokens:
            Minimum permitted overlap size.
        max_overlap_tokens:
            Maximum permitted overlap size.

    Returns:
        Character offset into ``previous_text`` satisfying
        ``0 <= offset < len(previous_text)``.

    Raises:
        TypeError:
            If ``previous_text`` is not a string.
        ValueError:
            If overlap configuration is invalid or the previous
            chunk is too small to provide the minimum overlap.
        RuntimeError:
            If tokenizer fallback cannot produce a safe overlap.
    """

    if not isinstance(previous_text, str):
        raise TypeError(
            "previous_text must be a string."
        )

    if not previous_text:
        raise ValueError(
            "previous_text must be non-empty."
        )

    values = (
        ("overlap_tokens", overlap_tokens),
        ("min_overlap_tokens", min_overlap_tokens),
        ("max_overlap_tokens", max_overlap_tokens),
    )

    for name, value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if min_overlap_tokens > overlap_tokens:
        raise ValueError(
            "min_overlap_tokens must not exceed overlap_tokens."
        )

    if overlap_tokens > max_overlap_tokens:
        raise ValueError(
            "overlap_tokens must not exceed max_overlap_tokens."
        )

    previous_tokens = count_tokens(previous_text)

    if previous_tokens < min_overlap_tokens:
        raise ValueError(
            "previous chunk is too small for the minimum overlap."
        )

    boundary_getters = (
        _paragraph_cut_positions,
        _line_cut_positions,
        _sentence_cut_positions,
    )

    for get_positions in boundary_getters:
        candidates: list[tuple[int, int]] = []

        positions = (0,) + get_positions(previous_text)

        for position in positions:
            if not 0 <= position < len(previous_text):
                continue

            overlap_text = previous_text[position:]
            token_count = count_tokens(overlap_text)

            if (
                min_overlap_tokens
                <= token_count
                <= max_overlap_tokens
            ):
                candidates.append(
                    (
                        position,
                        token_count,
                    )
                )

        if candidates:
            selected_position, _ = min(
                candidates,
                key=lambda candidate: (
                    abs(
                        candidate[1]
                        - overlap_tokens
                    ),
                    -candidate[1],
                    -candidate[0],
                ),
            )

            return selected_position

    tokenizer = get_tokenizer()

    try:
        encoded = tokenizer(
            previous_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenization failed for overlap selection: "
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

    if len(input_ids) < overlap_tokens:
        raise RuntimeError(
            "token fallback cannot provide the requested overlap."
        )

    start_token_index = len(input_ids) - overlap_tokens
    start_offset = offsets[start_token_index]

    if (
        not isinstance(start_offset, tuple)
        or len(start_offset) != 2
    ):
        raise RuntimeError(
            "Tokenizer returned an invalid overlap offset."
        )

    overlap_start, _ = start_offset

    if not isinstance(overlap_start, int):
        raise RuntimeError(
            "Tokenizer returned a non-integer overlap offset."
        )

    if not 0 <= overlap_start < len(previous_text):
        raise RuntimeError(
            "Overlap fallback produced an invalid source offset."
        )

    overlap_text = previous_text[overlap_start:]
    actual_tokens = count_tokens(overlap_text)

    if not (
        min_overlap_tokens
        <= actual_tokens
        <= max_overlap_tokens
    ):
        raise RuntimeError(
            "Token fallback produced overlap outside the "
            "configured window."
        )

    return overlap_start

def split_long_section(
    section: ParsedSection,
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> tuple[Chunk, ...]:
    """Split one normalized long section into overlapping chunks.

    The first chunk begins at the start of the section. Every later
    chunk includes a deterministic semantic overlap from the previous
    chunk while still extending the unique-content frontier.

    Semantic cut selection and token fallback are delegated to the
    verified helper functions. Every materialized chunk must remain
    within ``max_tokens``.

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
            If overlap traversal fails to preserve source coverage,
            token limits, provenance, or forward progress.
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

    covered_end = 0
    chunk_index = 0

    while covered_end < len(section.text):
        previous_covered_end = covered_end

        if not chunks:
            start = 0
        else:
            previous_chunk = chunks[-1]

            previous_text = section.text[
                previous_chunk_start:previous_covered_end
            ]

            relative_overlap_start = _select_overlap_start(
                previous_text
            )

            start = (
                previous_chunk_start
                + relative_overlap_start
            )

            if not (
                previous_chunk_start
                <= start
                < previous_covered_end
            ):
                raise RuntimeError(
                    "overlap start is outside the previous chunk."
                )

        remaining = section.text[start:]
        remaining_tokens = count_tokens(remaining)

        if remaining_tokens <= max_tokens:
            end = len(section.text)
        else:
            relative_end = _select_semantic_cut_position(
                remaining,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
            )

            if not 0 < relative_end < len(remaining):
                raise RuntimeError(
                    "overlap splitter did not produce a valid "
                    "relative end."
                )

            end = start + relative_end

        if not start < end <= len(section.text):
            raise RuntimeError(
                "overlap splitter produced invalid source offsets."
            )

        if end <= previous_covered_end:
            raise RuntimeError(
                "overlap splitter failed to extend the "
                "unique-content frontier."
            )

        chunk_text = section.text[start:end]

        if not chunk_text:
            raise RuntimeError(
                "overlap splitter produced empty chunk text."
            )

        token_count = count_tokens(chunk_text)

        if token_count > max_tokens:
            raise RuntimeError(
                "overlap splitter exceeded hard maximum: "
                f"{token_count} > {max_tokens}."
            )

        chunks.append(
            Chunk(
                chunk_id=generate_chunk_id(
                    doc_id=section.doc_id,
                    section_path=section.section_path,
                    chunk_index=chunk_index,
                    text=chunk_text,
                ),
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

        previous_chunk_start = start
        covered_end = end
        chunk_index += 1

        if chunk_index > 1000:
            raise RuntimeError(
                "overlap splitter exceeded safe iteration limit."
            )

    if not chunks:
        raise RuntimeError(
            "overlap splitter produced no chunks."
        )

    if covered_end != len(section.text):
        raise RuntimeError(
            "overlap splitter did not cover the complete source."
        )

    if [
        chunk.chunk_index
        for chunk in chunks
    ] != list(range(len(chunks))):
        raise RuntimeError(
            "overlap chunk indexes are not contiguous."
        )

    return tuple(chunks)

def chunk_section(section: ParsedSection) -> tuple[Chunk, ...]:
    """Convert one normalized section into zero or more chunks.

    Empty structural sections produce no chunks. Non-empty sections
    that fit within the configured hard maximum remain one chunk.
    Sections above the hard maximum are delegated to deterministic
    long-section splitting.

    Overlap is applied only when an oversized section is delegated
    to ``split_long_section``.

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
            chunk_id=generate_chunk_id(
                doc_id=section.doc_id,
                section_path=section.section_path,
                chunk_index=0,
                text=section.text,
            ),
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
