"""Tests for the inline citation parser"""

from rag_pipeline.generation.citation_parser import (
    Citation,
    parse_citations,
    strip_citations,
)


class TestParseCitations:
    def test_no_markers_returns_empty(self):
        out = parse_citations("Just a plain answer.", valid_chunk_ids=["c1"])
        assert out == []

    def test_single_valid_citation(self):
        text = "Lectures are recommended [Source: c1]."
        out = parse_citations(text, valid_chunk_ids=["c1", "c2"])
        assert len(out) == 1
        assert out[0].chunk_id == "c1"
        # span_start/span_end correspond to "[Source: c1]" within the text.
        assert text[out[0].span_start : out[0].span_end] == "[Source: c1]"

    def test_multiple_valid_citations(self):
        text = "Foo [Source: a1] and bar [Source: b2] and baz [Source: a1]."
        out = parse_citations(text, valid_chunk_ids=["a1", "b2"])
        assert [c.chunk_id for c in out] == ["a1", "b2", "a1"]

    def test_drops_hallucinated_chunk_ids(self):
        text = "Real [Source: real_id] and fake [Source: invented_id]."
        out = parse_citations(text, valid_chunk_ids=["real_id"])
        assert len(out) == 1
        assert out[0].chunk_id == "real_id"

    def test_handles_complex_chunk_ids(self):
        text = "Cite [Source: chunk-2025-01_handbook.pg42:line8]."
        out = parse_citations(
            text, valid_chunk_ids=["chunk-2025-01_handbook.pg42:line8"]
        )
        assert len(out) == 1
        assert out[0].chunk_id == "chunk-2025-01_handbook.pg42:line8"

    def test_extra_whitespace_is_tolerated(self):
        text = "[Source:   c1   ]"
        out = parse_citations(text, valid_chunk_ids=["c1"])
        assert len(out) == 1
        assert out[0].chunk_id == "c1"

    def test_returns_citation_models(self):
        text = "x [Source: c1]"
        out = parse_citations(text, valid_chunk_ids=["c1"])
        assert isinstance(out[0], Citation)

    def test_valid_chunk_ids_accepts_iterable(self):
        text = "x [Source: c1]"

        def _gen():
            yield "c1"

        out = parse_citations(text, valid_chunk_ids=_gen())
        assert len(out) == 1


class TestStripCitations:
    def test_removes_markers(self):
        text = "Lectures are recommended [Source: c1]."
        assert strip_citations(text) == "Lectures are recommended ."

    def test_removes_multiple_and_collapses_whitespace(self):
        """After marker removal, adjacent spaces are collapsed so the
        rendered prose does not contain the double-space that a
        mid-sentence citation would otherwise leave behind."""
        text = "Foo [Source: a] and bar [Source: b]."
        assert strip_citations(text) == "Foo and bar ."

    def test_no_markers_returns_unchanged_stripped(self):
        assert strip_citations("plain text  ") == "plain text"

    def test_collapses_whitespace_at_marker_boundary(self):
        """A citation inside a sentence must not leave a double space."""
        assert strip_citations("hello [Source: c1] world") == "hello world"

    def test_preserves_paragraph_breaks(self):
        """Multi-paragraph answers must keep their newline structure so
        downstream consumers rendering outside a <p> block (logs, CLI,
        markdown export) still show paragraph separation."""
        text = (
            "First paragraph with a citation [Source: c1].\n"
            "\n"
            "Second paragraph [Source: c2] continues here."
        )
        assert strip_citations(text) == (
            "First paragraph with a citation .\n\nSecond paragraph continues here."
        )
