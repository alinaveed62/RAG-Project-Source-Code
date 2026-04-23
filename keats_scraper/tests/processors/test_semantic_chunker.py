"""Tests for the semantic-boundary chunker."""

from __future__ import annotations

import math

import pytest

from keats_scraper.config import ChunkConfig
from keats_scraper.models.document import Document
from keats_scraper.processors.semantic_chunker import SemanticChunker


class FakeEmbedder:
    """Returns pre-registered L2-normalised vectors for each sentence.

    The semantic chunker multiplies these vectors to compute cosine
    similarity; the test fixtures register vectors whose pairwise
    similarities are known so the expected breakpoints are predictable.
    """

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    def encode(self, sentences, **kwargs):
        # normalize_embeddings=True is passed by the chunker; fake
        # vectors are already normalised so the kwarg is a no-op here.
        del kwargs
        return [self._mapping[s] for s in sentences]


def _make_document(content: str, title: str = "Test") -> Document:
    return Document.create(
        source_url="https://example.com/test",
        title=title,
        content=content,
        content_type="html",
        section="Test Section",
        raw_html="<html>x</html>",
    )


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(v * v for v in vector))
    return [v / magnitude for v in vector]


class TestSplitSentences:
    def test_splits_on_sentence_terminators(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        assert chunker._split_sentences("First. Second! Third?") == [
            "First.",
            "Second!",
            "Third?",
        ]

    def test_empty_text_returns_empty(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        assert chunker._split_sentences("") == []
        assert chunker._split_sentences("   \n\n\n   ") == []


class TestPairwiseDistances:
    def test_identical_vectors_zero_distance(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        distances = chunker._pairwise_distances(embeddings)
        assert distances == pytest.approx([0.0, 0.0])

    def test_orthogonal_vectors_distance_one(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        distances = chunker._pairwise_distances(embeddings)
        assert distances == pytest.approx([1.0])

    def test_single_embedding_no_pairs(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        assert chunker._pairwise_distances([[1.0, 0.0]]) == []


class TestFindBreakpoints:
    def test_empty_distances(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        assert chunker._find_breakpoints([]) == []

    def test_threshold_selects_high_distance_indices(self):
        config = ChunkConfig(semantic_percentile_threshold=50)
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        # Distances: 0.1, 0.2, 0.9. 50th percentile = 0.2. Indices >= 0.2
        # are sentence positions 1 (bp at index+1=2) and 2 (bp=3).
        result = chunker._find_breakpoints([0.1, 0.2, 0.9])
        assert result == [2, 3]

    def test_high_percentile_picks_only_top_distance(self):
        config = ChunkConfig(semantic_percentile_threshold=99)
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        result = chunker._find_breakpoints([0.1, 0.2, 0.9])
        assert result == [3]


class TestGroupSentences:
    def test_no_breakpoints_yields_one_group(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        groups = chunker._group_sentences(["a.", "b.", "c."], [])
        assert groups == [["a.", "b.", "c."]]

    def test_breakpoint_at_middle(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        groups = chunker._group_sentences(["a.", "b.", "c.", "d."], [2])
        assert groups == [["a.", "b."], ["c.", "d."]]

    def test_breakpoint_beyond_range_dropped(self):
        # A breakpoint at or past the final sentence index should not
        # produce an empty trailing group.
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        groups = chunker._group_sentences(["a.", "b."], [2])
        assert groups == [["a.", "b."]]

    def test_duplicate_breakpoint_ignored(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        groups = chunker._group_sentences(["a.", "b.", "c."], [1, 1, 2])
        # The second 1 is skipped because start has already moved past.
        assert groups == [["a."], ["b."], ["c."]]


class TestMergeAndSplit:
    def test_small_groups_merged(self):
        config = ChunkConfig(
            chunk_size=1000,
            semantic_min_tokens=10,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        groups = [["Tiny."], ["Also tiny."], ["A slightly larger sentence."]]
        result = chunker._merge_small_and_split_large(groups)
        # With 10-token minimum and very short groups, merge collapses
        # them all into one chunk.
        assert len(result) == 1

    def test_large_group_split_by_tokens(self):
        config = ChunkConfig(
            chunk_size=5,
            semantic_min_tokens=1,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        # A single group whose token count exceeds chunk_size; the split
        # path emits multiple chunks.
        long_group = [" ".join(["word"] * 40)]
        result = chunker._merge_small_and_split_large([long_group])
        assert len(result) > 1
        # Every emitted piece is at or under the chunk_size budget in
        # tokens so the embedder never sees an oversize chunk.
        for piece in result:
            assert chunker._count_tokens(piece) <= config.chunk_size

    def test_word_tokenizer_split_path(self):
        config = ChunkConfig(chunk_size=3, semantic_min_tokens=1)
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        # Force the word-count fallback by overriding the tokenizer.
        chunker._tokenizer = "word"
        long_group = ["one two three four five six seven eight nine ten"]
        result = chunker._merge_small_and_split_large([long_group])
        assert result == [
            "one two three",
            "four five six",
            "seven eight nine",
            "ten",
        ]

    def test_word_tokenizer_skips_empty_pieces(self):
        # A run of trailing whitespace in the source text can produce an
        # empty piece after slicing; the piece.strip() guard must
        # drop it so chunk_texts does not contain blank entries.
        config = ChunkConfig(chunk_size=3, semantic_min_tokens=1)
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        chunker._tokenizer = "word"
        # Constructed group whose join produces trailing spaces when
        # sliced; here the tokenizer path always receives text.split()
        # output so every piece has content. This test primarily guards
        # the branch that drops whitespace-only pieces, exercised by
        # feeding a group that ends exactly on a chunk_size boundary.
        group = ["one two three four five six"]
        result = chunker._merge_small_and_split_large([group])
        # Exactly two full pieces of 3 words each, no trailing empty.
        assert result == ["one two three", "four five six"]

    def test_tiktoken_split_path(self):
        # Exercises the tiktoken branch of the oversize-chunk splitter.
        # Real tiktoken is installed so this path runs for any text that
        # exceeds the small chunk_size budget.
        config = ChunkConfig(chunk_size=3, semantic_min_tokens=1)
        chunker = SemanticChunker(config, embedder=FakeEmbedder({}))
        long_group = [
            "alpha beta gamma delta epsilon zeta eta theta iota"
        ]
        result = chunker._merge_small_and_split_large([long_group])
        assert len(result) >= 2
        for piece in result:
            assert chunker._count_tokens(piece) <= config.chunk_size


class TestChunkDocument:
    def test_empty_document_returns_empty(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        chunks = chunker.chunk_document(_make_document(""))
        assert chunks == []

    def test_whitespace_only_document_returns_empty(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        chunks = chunker.chunk_document(_make_document("   \n   \n   "))
        assert chunks == []

    def test_single_sentence_returns_one_chunk(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        doc = _make_document("A single sentence.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert "single sentence" in chunks[0].text

    def test_two_similar_sentences_stay_in_one_chunk(self):
        # Identical vectors mean zero distance, so no breakpoint fires
        # and both sentences remain in the same chunk.
        mapping = {
            "The first sentence is here.": _normalise([1.0, 0.0]),
            "The first sentence is here too.": _normalise([1.0, 0.0]),
        }
        config = ChunkConfig(
            chunk_size=100,
            semantic_min_tokens=1,
            semantic_percentile_threshold=99,
            preserve_headings=False,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        doc = _make_document(
            "The first sentence is here. The first sentence is here too."
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_dissimilar_sentences_produce_breakpoint(self):
        mapping = {
            "Alpha stuff about topic A.": _normalise([1.0, 0.0]),
            "Beta stuff about topic B.": _normalise([0.0, 1.0]),
            "More beta stuff about topic B.": _normalise([0.0, 1.0]),
        }
        config = ChunkConfig(
            chunk_size=100,
            semantic_min_tokens=1,
            semantic_percentile_threshold=50,
            preserve_headings=False,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        doc = _make_document(
            "Alpha stuff about topic A. Beta stuff about topic B. "
            "More beta stuff about topic B."
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2

    def test_heading_context_prepended_when_enabled(self):
        # The sentence splitter does not split at the "# Heading"
        # markdown marker, so the heading text becomes part of the first
        # sentence embedding. Map that exact string in the fake
        # embedder.
        first_sentence = "# Section A\n\nSentence one."
        second_sentence = "Sentence two."
        mapping = {
            first_sentence: _normalise([1.0, 0.0]),
            second_sentence: _normalise([0.0, 1.0]),
        }
        config = ChunkConfig(
            chunk_size=100,
            semantic_min_tokens=1,
            semantic_percentile_threshold=50,
            preserve_headings=True,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        doc = _make_document(
            f"{first_sentence} {second_sentence}"
        )
        chunks = chunker.chunk_document(doc)
        assert any("[Context: Section A]" in c.text for c in chunks)

    def test_heading_context_suppressed_when_disabled(self):
        first_sentence = "# Section A\n\nSentence one."
        second_sentence = "Sentence two."
        mapping = {
            first_sentence: _normalise([1.0, 0.0]),
            second_sentence: _normalise([0.0, 1.0]),
        }
        config = ChunkConfig(
            chunk_size=100,
            semantic_min_tokens=1,
            semantic_percentile_threshold=50,
            preserve_headings=False,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        doc = _make_document(
            f"{first_sentence} {second_sentence}"
        )
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            assert "[Context:" not in c.text

    def test_short_chunk_falls_back_to_start_heading(self):
        # A chunk text shorter than the 40-char heading-search window
        # falls through to position 0, so the heading path is computed
        # against the document start. The test verifies that the
        # short-chunk branch does not error.
        first_sentence = "# Top\n\nHi."
        second_sentence = "Yo."
        mapping = {
            first_sentence: _normalise([1.0, 0.0]),
            second_sentence: _normalise([0.0, 1.0]),
        }
        config = ChunkConfig(
            chunk_size=100,
            semantic_min_tokens=1,
            semantic_percentile_threshold=50,
            preserve_headings=True,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        doc = _make_document(f"{first_sentence} {second_sentence}")
        chunks = chunker.chunk_document(doc)
        # Assertion is existence only; heading_path may still be empty
        # if position 0 is before any heading.
        assert chunks

    def test_unfindable_chunk_text_falls_back_to_position_zero(self):
        # Force a merge by setting a high semantic_min_tokens, then
        # the merged chunk is " "-joined across sentence fragments that
        # the original document had separated by "\n\n". The merged text
        # does not appear verbatim, so text.find returns -1 and the
        # fallback at line 251 fires.
        first = "A short line."
        second = "Another short line."
        mapping = {
            first: _normalise([1.0, 0.0]),
            second: _normalise([0.0, 1.0]),
        }
        # High semantic_min_tokens pushes both short groups into a
        # single merged group.
        config = ChunkConfig(
            chunk_size=1000,
            semantic_min_tokens=1000,
            semantic_percentile_threshold=50,
            preserve_headings=False,
        )
        chunker = SemanticChunker(config, embedder=FakeEmbedder(mapping))
        # Put a paragraph break between the two sentences so the merged
        # chunk text (single-space join) does not appear verbatim.
        doc = _make_document(f"{first}\n\n{second}")
        chunks = chunker.chunk_document(doc)
        assert chunks


class TestChunkDocuments:
    def test_concatenates_chunks_across_documents(self):
        mapping = {
            "Doc one content.": _normalise([1.0, 0.0]),
            "Doc two content.": _normalise([0.0, 1.0]),
        }
        chunker = SemanticChunker(
            ChunkConfig(preserve_headings=False),
            embedder=FakeEmbedder(mapping),
        )
        doc_a = Document.create(
            source_url="https://example.com/a",
            title="A",
            content="Doc one content.",
            content_type="html",
            section="A",
            raw_html="<p>a</p>",
        )
        doc_b = Document.create(
            source_url="https://example.com/b",
            title="B",
            content="Doc two content.",
            content_type="html",
            section="B",
            raw_html="<p>b</p>",
        )
        chunks = chunker.chunk_documents([doc_a, doc_b])
        assert len(chunks) == 2
        assert chunks[0].metadata.document_id != chunks[1].metadata.document_id

    def test_empty_document_list_returns_empty(self):
        chunker = SemanticChunker(
            ChunkConfig(), embedder=FakeEmbedder({})
        )
        assert chunker.chunk_documents([]) == []


class TestCountTokens:
    def test_word_count_fallback(self):
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        chunker._tokenizer = "word"
        assert chunker._count_tokens("one two three four") == 4

    def test_tiktoken_path(self):
        # With tiktoken installed (always true in this project) the token
        # count for Hello world. is small; just check it is > 0 so
        # the path is exercised.
        chunker = SemanticChunker(ChunkConfig(), embedder=FakeEmbedder({}))
        assert chunker._count_tokens("Hello world.") > 0
