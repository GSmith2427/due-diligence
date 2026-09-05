"""Recursive structural text chunking.

The chunker splits text into retrieval-sized units while respecting natural
boundaries. It tries paragraph breaks first, then sentence breaks, then
character splits as a fallback. Adjacent chunks share a small overlap so
that facts split near a boundary remain recoverable.

Design choices worth surfacing:

* **Token counts are approximate.** The pipeline embeds with ``bge-m3``,
  which uses SentencePiece, while we count tokens with ``tiktoken``'s
  ``cl100k_base`` (GPT-4 family). This is a deliberate trade — tiktoken is
  fast and consistent, and the difference (typically <20%) is well within
  the chunk-size tolerance we care about. The embedder enforces its own
  hard limit; our budget is a soft target.

* **Hierarchy of separators.** Paragraph (``\\n\\n``) before single newline
  before sentence-end punctuation before words. This matches the structure
  of cleaned 10-K text where double newlines mark section boundaries.

* **Recursive packing.** Splits are joined back into chunks up to the
  target token budget. This avoids the common pathology of one chunk per
  paragraph (too small) or one chunk per section (too big).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import tiktoken

from duediligence.ingestion.models import Chunk

# Ordered from most-preferred to least-preferred split boundary.
_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", " ", "")

# We use the GPT-4 family tokenizer as a fast, deterministic proxy for the
# embedder's tokenizer. See module docstring for the trade-off.
_ENCODING_NAME: Final[str] = "cl100k_base"


@dataclass(frozen=True)
class ChunkConfig:
    """Tunables for the chunker.

    Defaults aim at high-quality retrieval over ``bge-m3``: chunks large
    enough to carry context, small enough that retrieval surfaces specific
    answers, with overlap sized to cover most boundary-straddling facts.
    """

    target_tokens: int = 600
    max_tokens: int = 800
    overlap_tokens: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("target_tokens must be in (0, max_tokens]")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be in [0, target_tokens)")


class Chunker:
    """Splits documents into overlapping, token-bounded chunks."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()
        self._encoder = tiktoken.get_encoding(_ENCODING_NAME)

    def chunk(self, text: str) -> list[Chunk]:
        """Return chunks covering the entire input text."""
        if not text.strip():
            return []

        # Step 1: recursively split into pieces under the max token budget.
        pieces = self._split(text, separator_idx=0)

        # Step 2: greedily merge adjacent pieces up to the target budget,
        # then re-cross with overlap.
        return self._pack(text, pieces)

    # ------------------------------------------------------------------ split

    def _split(self, text: str, *, separator_idx: int) -> list[str]:
        """Recursively split text on the preferred separator until each
        piece is under ``max_tokens``.
        """
        if self._count_tokens(text) <= self._config.max_tokens:
            return [text]

        # Find the first separator at or beyond ``separator_idx`` that
        # actually splits the text into multiple pieces.
        for idx in range(separator_idx, len(_SEPARATORS)):
            sep = _SEPARATORS[idx]
            parts = _split_with_separator(text, sep)
            if len(parts) > 1:
                # Recurse on parts that are still too big.
                result: list[str] = []
                for part in parts:
                    if self._count_tokens(part) <= self._config.max_tokens:
                        result.append(part)
                    else:
                        result.extend(self._split(part, separator_idx=idx + 1))
                return result

        # Hard fallback: slice by characters. Only reached for pathological
        # input with no whitespace at all.
        return _hard_char_split(text, max_chars=self._config.max_tokens * 4)

    # ------------------------------------------------------------------- pack

    def _pack(self, original_text: str, pieces: list[str]) -> list[Chunk]:
        """Merge pieces into chunks of ``target_tokens``, with overlap.

        Character offsets back to the original text are computed by scanning
        forward from the previous chunk's start. This handles cases where
        the same piece appears multiple times in the document (rare in 10-Ks
        but possible in highly-templated filings).
        """
        chunks: list[Chunk] = []
        cursor = 0  # next char index to search from in original_text
        buffer: list[str] = []
        buffer_tokens = 0

        for piece in pieces:
            piece_tokens = self._count_tokens(piece)
            if buffer and buffer_tokens + piece_tokens > self._config.target_tokens:
                chunk_text, chunk_tokens = self._flush(buffer)
                cursor = self._emit(original_text, chunk_text, chunk_tokens, cursor, chunks)
                # Seed the next buffer with a tail-overlap of the previous chunk.
                buffer, buffer_tokens = self._seed_overlap(chunk_text)
            buffer.append(piece)
            buffer_tokens += piece_tokens

        if buffer:
            chunk_text, chunk_tokens = self._flush(buffer)
            self._emit(original_text, chunk_text, chunk_tokens, cursor, chunks)

        return chunks

    def _flush(self, buffer: list[str]) -> tuple[str, int]:
        text = "".join(buffer).strip()
        return text, self._count_tokens(text)

    def _emit(
        self,
        original: str,
        chunk_text: str,
        chunk_tokens: int,
        cursor: int,
        chunks: list[Chunk],
    ) -> int:
        """Append a chunk with offsets resolved against ``original``.

        Returns the new ``cursor`` (start offset of the emitted chunk plus 1)
        so subsequent searches don't re-discover the same span.
        """
        char_start = original.find(chunk_text, cursor)
        if char_start < 0:
            # Whitespace normalisation between pieces and the original may
            # prevent an exact match. Fall back to the cursor position; the
            # resulting offsets are approximate but stable.
            char_start = cursor
        char_end = char_start + len(chunk_text)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=chunk_text,
                token_count=chunk_tokens,
                char_start=char_start,
                char_end=char_end,
            )
        )
        return max(char_start + 1, cursor)

    def _seed_overlap(self, previous_chunk_text: str) -> tuple[list[str], int]:
        """Build the starting buffer for the next chunk from the tail of
        the previous one, sized to roughly ``overlap_tokens``.
        """
        if self._config.overlap_tokens == 0:
            return [], 0
        tokens = self._encoder.encode(previous_chunk_text)
        tail_tokens = tokens[-self._config.overlap_tokens :]
        tail_text = self._encoder.decode(tail_tokens)
        return [tail_text], len(tail_tokens)

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))


# ---------------------------------------------------------------------------
# Pure helpers — kept module-level for clarity and ease of testing.
# ---------------------------------------------------------------------------


def _split_with_separator(text: str, separator: str) -> list[str]:
    """Split while keeping the separator attached to the preceding piece.

    Keeping the separator on the left preserves natural reading order when
    we re-join later: ``"foo. bar"`` split on ``". "`` yields
    ``["foo. ", "bar"]``, which joins back to the original.
    """
    if separator == "":
        # Caller will hard-split; signal "no split possible" so the recursion
        # falls through to the character fallback.
        return [text]
    parts = re.split(f"({re.escape(separator)})", text)
    # ``re.split`` with a capturing group returns separator pieces between
    # content pieces; recombine each content with its trailing separator.
    result: list[str] = []
    for i in range(0, len(parts), 2):
        content = parts[i]
        sep_piece = parts[i + 1] if i + 1 < len(parts) else ""
        combined = content + sep_piece
        if combined:
            result.append(combined)
    return result if len(result) > 1 else [text]


def _hard_char_split(text: str, *, max_chars: int) -> list[str]:
    """Last-resort splitter for input that has no natural boundaries."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
