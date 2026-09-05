"""Tests for the recursive structural chunker."""

from __future__ import annotations

import pytest

from duediligence.ingestion.chunker import ChunkConfig, Chunker


@pytest.fixture
def chunker() -> Chunker:
    # Smaller budgets exercise the splitting logic on test-sized inputs.
    return Chunker(ChunkConfig(target_tokens=50, max_tokens=80, overlap_tokens=10))


def test_short_text_produces_single_chunk(chunker: Chunker) -> None:
    text = "Apple reported strong fourth-quarter results."
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_empty_text_returns_no_chunks(chunker: Chunker) -> None:
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n   ") == []


def test_long_text_is_split_into_multiple_chunks(chunker: Chunker) -> None:
    # ~500 tokens of text — well above the 80-token max
    paragraph = "Quarterly revenue increased substantially. " * 40
    text = paragraph + "\n\n" + paragraph + "\n\n" + paragraph

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 80
        assert chunk.text.strip()


def test_chunks_have_monotonic_indices(chunker: Chunker) -> None:
    text = ("Sentence one. " * 100) + "\n\n" + ("Sentence two. " * 100)
    chunks = chunker.chunk(text)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_have_overlap_with_neighbours(chunker: Chunker) -> None:
    """Adjacent chunks should share some tail/head content."""
    text = " ".join(f"word{i:04d}" for i in range(500))
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    # The last few tokens of chunk N should appear at the start of chunk N+1
    for i in range(len(chunks) - 1):
        tail_words = chunks[i].text.split()[-5:]
        next_chunk_start = chunks[i + 1].text[:200]
        # At least one of the tail words should appear in the next chunk's head
        assert any(
            w in next_chunk_start for w in tail_words
        ), f"No overlap found between chunks {i} and {i + 1}"


def test_paragraph_boundaries_are_respected(chunker: Chunker) -> None:
    """Splits prefer paragraph breaks over arbitrary positions."""
    para_a = "Sentence A. " * 10  # fits well under max
    para_b = "Sentence B. " * 10
    text = para_a + "\n\n" + para_b

    chunks = chunker.chunk(text)

    # Each paragraph fits in a chunk on its own; we should not see content
    # from both paragraphs jammed mid-sentence.
    for chunk in chunks:
        # No chunk should start mid-sentence with lowercase
        first_word = chunk.text.lstrip().split(" ", 1)[0]
        assert first_word[:1].isupper() or first_word[:1].isdigit()


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="target_tokens"):
        ChunkConfig(target_tokens=0, max_tokens=100)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkConfig(target_tokens=100, max_tokens=200, overlap_tokens=100)


def test_char_offsets_cover_chunks(chunker: Chunker) -> None:
    """char_start and char_end should be a valid range over the source text."""
    text = (f"Sentence number {i}. " for i in range(200))
    full_text = "".join(text)

    chunks = chunker.chunk(full_text)

    for chunk in chunks:
        assert 0 <= chunk.char_start < chunk.char_end
        assert chunk.char_end <= len(full_text)
