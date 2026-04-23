"""Tests for prompt template generation."""

from rag_pipeline.generation.prompt_templates import (
    _format_heading_suffix,
    build_rag_prompt,
)
from rag_pipeline.models import RetrievalResult


class TestBuildRagPrompt:
    """Tests for the Mistral-7B prompt builder."""

    def test_contains_inst_tags(self):
        prompt = build_rag_prompt("test?", [])
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_contains_system_prompt_default_citation_injection(self):
        """Default-on citation injection: prompt mentions the chunk_id rule."""
        prompt = build_rag_prompt("test?", [])
        assert "King's College London Informatics" in prompt
        assert "[Source: chunk_id]" in prompt

    def test_contains_system_prompt_legacy_titles(self):
        """Legacy mode (citation_injection disabled) uses title-based citation."""
        prompt = build_rag_prompt("test?", [], enable_citation_injection=False)
        assert "King's College London Informatics" in prompt
        assert "Cite the source document title" in prompt

    def test_system_prompt_mentions_handbook_and_context_only(self):
        """Both modes must restrict the LLM to the provided handbook context."""
        injected = build_rag_prompt("q?", [])
        legacy = build_rag_prompt("q?", [], enable_citation_injection=False)
        for prompt in (injected, legacy):
            assert "Student Handbook" in prompt
            assert "ONLY" in prompt

    def test_contains_question(self):
        prompt = build_rag_prompt("What is the attendance policy?", [])
        assert "What is the attendance policy?" in prompt

    def test_includes_context_chunkid_marker(self):
        contexts = [
            RetrievalResult(
                chunk_id="c1",
                text="Attend all lectures.",
                score=0.9,
                source="https://example.com",
                title="Attendance",
                section="T&A",
            )
        ]
        prompt = build_rag_prompt("attendance?", contexts)
        assert "Attend all lectures." in prompt
        assert "[Source: c1]" in prompt
        # Title and section still surfaced for human readability.
        assert "Attendance" in prompt
        assert "T&A" in prompt

    def test_includes_heading_path_when_present(self):
        """Retrieved chunks with a non-empty heading_path should have their
        handbook heading hierarchy rendered into the prompt context, giving
        the LLM more locality context than just title/section."""
        contexts = [
            RetrievalResult(
                chunk_id="c1",
                text="Attend all lectures.",
                score=0.9,
                source="https://example.com",
                title="Attendance",
                section="Teaching & Assessment",
                heading_path=["Teaching and Assessment", "Attendance"],
            )
        ]
        injected = build_rag_prompt("attendance?", contexts)
        assert "Teaching and Assessment > Attendance" in injected

        legacy = build_rag_prompt(
            "attendance?", contexts, enable_citation_injection=False
        )
        assert "Teaching and Assessment > Attendance" in legacy

    def test_includes_context_legacy_title_marker(self):
        contexts = [
            RetrievalResult(
                chunk_id="c1",
                text="Attend all lectures.",
                score=0.9,
                source="https://example.com",
                title="Attendance",
                section="T&A",
            )
        ]
        prompt = build_rag_prompt("attendance?", contexts, enable_citation_injection=False)
        assert "Attend all lectures." in prompt
        assert "[Source: Attendance (T&A)]" in prompt

    def test_multiple_contexts(self):
        contexts = [
            RetrievalResult(
                chunk_id="c1",
                text="First chunk.",
                score=0.9,
                source="",
                title="Title A",
                section="Section A",
            ),
            RetrievalResult(
                chunk_id="c2",
                text="Second chunk.",
                score=0.8,
                source="",
                title="Title B",
                section="Section B",
            ),
        ]
        prompt = build_rag_prompt("question?", contexts)
        assert "First chunk." in prompt
        assert "Second chunk." in prompt
        # Default-on citation injection uses chunk_ids.
        assert "[Source: c1]" in prompt
        assert "[Source: c2]" in prompt

    def test_empty_contexts(self):
        prompt = build_rag_prompt("question?", [])
        assert "Context:" in prompt
        assert "[INST]" in prompt

    def test_starts_with_bos_token(self):
        prompt = build_rag_prompt("test?", [])
        assert prompt.startswith("<s>")


class TestPromptSnapshot:
    """Regression snapshot for the <s>[INST] ... [/INST] wrapper.

    The frozen evaluation numbers were generated with this exact surface:
    the Mistral-7B chat format interpolated literally into Ollama's
    per-model Modelfile template. If a future contributor "cleans up" the
    wrapper (dropping <s>, rewriting as <|im_start|>, etc.) the
    evaluation chapter's numbers stop being reproducible from this code.
    Locking the bytes here forces any template edit to be deliberate."""

    def test_empty_context_prompt_exact_bytes(self):
        prompt = build_rag_prompt("What is X?", [])
        expected = (
            "<s>[INST] You are a helpful assistant for King's College London "
            "Informatics students. Answer questions using ONLY the provided "
            "context from the Student Handbook. If the context doesn't "
            "contain enough information, say so clearly. After each factual "
            "claim in your answer, append the source marker [Source: "
            "chunk_id] for the chunk(s) that grounded the claim, using the "
            "exact chunk_id shown in the context. Do not invent chunk_ids; "
            "if no context chunk supports a claim, do not make the claim."
            "\n\nContext:\n\n\nQuestion: What is X? [/INST]"
        )
        assert prompt == expected

    def test_single_context_prompt_exact_bytes(self):
        contexts = [
            RetrievalResult(
                chunk_id="c1",
                text="Attend all lectures.",
                score=0.9,
                source="https://example.com",
                title="Attendance",
                section="T&A",
                heading_path=["Teaching and Assessment", "Attendance"],
            )
        ]
        prompt = build_rag_prompt("attendance?", contexts)
        expected = (
            "<s>[INST] You are a helpful assistant for King's College London "
            "Informatics students. Answer questions using ONLY the provided "
            "context from the Student Handbook. If the context doesn't "
            "contain enough information, say so clearly. After each factual "
            "claim in your answer, append the source marker [Source: "
            "chunk_id] for the chunk(s) that grounded the claim, using the "
            "exact chunk_id shown in the context. Do not invent chunk_ids; "
            "if no context chunk supports a claim, do not make the claim."
            "\n\nContext:\n"
            "Context chunk [Source: c1] (from Attendance / T&A / Teaching "
            "and Assessment > Attendance):\n"
            "Attend all lectures."
            "\n\nQuestion: attendance? [/INST]"
        )
        assert prompt == expected


class TestFormatHeadingSuffix:
    """Direct unit tests for the heading-path formatter helper."""

    def test_empty_list_returns_empty_string(self):
        assert _format_heading_suffix([]) == ""

    def test_single_heading_formatted_with_leading_separator(self):
        assert _format_heading_suffix(["Attendance"]) == " / Attendance"

    def test_multi_heading_joined_with_angle_bracket(self):
        assert (
            _format_heading_suffix(["Teaching and Assessment", "Attendance"])
            == " / Teaching and Assessment > Attendance"
        )
