"""Tests for query processor."""

from rag_pipeline.retrieval.query_processor import QueryProcessor


class TestQueryProcessorExpansion:
    """Tests for abbreviation expansion."""

    def test_expands_single_abbreviation(self):
        qp = QueryProcessor()
        result = qp.process("What is the EC policy?")
        assert "Extenuating Circumstances" in result
        assert "EC" in result

    def test_expands_multiple_abbreviations(self):
        qp = QueryProcessor()
        result = qp.process("UG and PGT programmes")
        assert "Undergraduate" in result
        assert "Postgraduate Taught" in result

    def test_no_expansion_when_no_abbreviations(self):
        qp = QueryProcessor()
        result = qp.process("What are the lecture times?")
        assert result == "What are the lecture times?"

    def test_expands_kcl(self):
        qp = QueryProcessor()
        result = qp.process("KCL student handbook")
        assert "King's College London" in result

    def test_case_sensitive_matching(self):
        qp = QueryProcessor()
        # "ec" lowercase should NOT match "EC"
        result = qp.process("the ec value is low")
        assert "Extenuating Circumstances" not in result

    def test_word_boundary_matching(self):
        qp = QueryProcessor()
        # "RECALL" contains "EC" but should not trigger expansion
        result = qp.process("RECALL the information")
        assert "Extenuating Circumstances" not in result


class TestQueryProcessorNormalization:
    """Tests for whitespace normalization."""

    def test_strips_leading_trailing_whitespace(self):
        qp = QueryProcessor()
        result = qp.process("  hello world  ")
        assert result == "hello world"

    def test_collapses_multiple_spaces(self):
        qp = QueryProcessor()
        result = qp.process("hello    world")
        assert result == "hello world"

    def test_handles_empty_string(self):
        qp = QueryProcessor()
        result = qp.process("")
        assert result == ""

    def test_custom_abbreviations(self):
        qp = QueryProcessor(abbreviations={"AI": "Artificial Intelligence"})
        result = qp.process("What is AI?")
        assert "Artificial Intelligence" in result


class TestQueryProcessorExpansionGate:
    """Tests for the enable_expansion flag (G.3 ablation support)."""

    def test_default_enables_expansion(self):
        qp = QueryProcessor()
        assert qp.enable_expansion is True
        result = qp.process("What is the EC policy?")
        assert "Extenuating Circumstances" in result

    def test_enable_expansion_false_leaves_abbreviations_raw(self):
        qp = QueryProcessor(enable_expansion=False)
        result = qp.process("What is the EC policy?")
        assert "Extenuating Circumstances" not in result
        assert "EC" in result

    def test_enable_expansion_false_still_normalizes_whitespace(self):
        qp = QueryProcessor(enable_expansion=False)
        result = qp.process("  hello    world  ")
        assert result == "hello world"

    def test_enable_expansion_false_with_custom_abbreviations(self):
        qp = QueryProcessor(
            abbreviations={"AI": "Artificial Intelligence"},
            enable_expansion=False,
        )
        result = qp.process("What is AI?")
        assert "Artificial Intelligence" not in result
        assert result == "What is AI?"
