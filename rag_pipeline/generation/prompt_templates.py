"""Prompt templates for the RAG pipeline.

The <s>[INST] ... [/INST] wrapper below is the Mistral-7B-Instruct
chat format. Ollama's /api/generate endpoint interpolates the
caller's prompt into the model's own Modelfile template ({{ .Prompt }}),
so for Gemma 2 and Llama 3.2 the <s> and [INST] bytes become literal
user-message text that Ollama then wraps with the correct per-model
turn tokens (<start_of_turn>user for Gemma, <|begin_of_text|> for
Llama). Those literal bytes turned out to be harmless in practice
(Gemma2:2b was the evaluation winner with this wrapper in place),
so the wrapper is kept to keep the published evaluation numbers
reproducible from this code path.

The system prompt asks the model to cite sources by chunk identifier,
which citation_parser.py can then validate against the actually
retrieved set. Hallucinated identifiers are dropped with a warning
before the response leaves the pipeline.
"""

from __future__ import annotations

from rag_pipeline.models import RetrievalResult

_BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant for King's College London Informatics students. "
    "Answer questions using ONLY the provided context from the Student Handbook. "
    "If the context doesn't contain enough information, say so clearly. "
)

_CITATION_INSTRUCTION = (
    "After each factual claim in your answer, append the source marker "
    "[Source: chunk_id] for the chunk(s) that grounded the claim, using the "
    "exact chunk_id shown in the context. Do not invent chunk_ids; if no "
    "context chunk supports a claim, do not make the claim."
)

_TITLE_INSTRUCTION = (
    "Cite the source document title when referencing specific information."
)


def _format_heading_suffix(heading_path: list[str]) -> str:
    """Format a heading path as " / Heading > Sub" for the source label."""
    if not heading_path:
        return ""
    return " / " + " > ".join(heading_path)


def build_rag_prompt(
    question: str,
    contexts: list[RetrievalResult],
    *,
    enable_citation_injection: bool = True,
) -> str:
    """Build a chat-formatted prompt with the retrieved context.

    Args:
        question: The user's question.
        contexts: Retrieved chunks to include as context.
        enable_citation_injection: If True (default), wrap each
            context chunk with its chunk_id and instruct the model
            to cite sources via [Source: chunk_id] markers. When
            False, the older title-based citation prompt is used.

    Returns:
        The formatted prompt string. The <s>[INST] ... [/INST]
        wrapper is the Mistral-7B chat format; see the module
        docstring for how the other models handle it through Ollama.
    """
    if enable_citation_injection:
        system_prompt = f"{_BASE_SYSTEM_PROMPT}{_CITATION_INSTRUCTION}"
        context_block = "\n\n".join(
            f"Context chunk [Source: {c.chunk_id}] "
            f"(from {c.title} / {c.section}{_format_heading_suffix(c.heading_path)}):\n"
            f"{c.text}"
            for c in contexts
        )
    else:
        system_prompt = f"{_BASE_SYSTEM_PROMPT}{_TITLE_INSTRUCTION}"
        context_block = "\n\n".join(
            f"[Source: {c.title} ({c.section}{_format_heading_suffix(c.heading_path)})]\n"
            f"{c.text}"
            for c in contexts
        )

    return (
        f"<s>[INST] {system_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question} [/INST]"
    )
