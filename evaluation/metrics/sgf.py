"""Section-Grounded Faithfulness, a combined evaluation metric.

Existing faithfulness metrics such as Ragas and SelfCheckGPT check
that an answer is entailed by the retrieved context, but they do not
check that the context came from the right part of the knowledge
base. In a handbook split into ten sections, an answer grounded in
"Teaching and Assessment" when the question was about "Support and
Wellbeing" is locally coherent but globally misleading. SGF combines
the two axes:

    SGF = alpha * NLI-Faithfulness + (1 - alpha) * Section-Match

NLI-Faithfulness is the mean entailment probability (from an NLI
cross-encoder) of each answer sentence against the concatenated
retrieved context. It uses cross-encoder/nli-deberta-v3-base; the
model's output order is (contradiction, entailment, neutral).

Section-Match is the Jaccard similarity between the set of retrieved
section labels and the set of ground-truth relevant sections.

alpha is the weighting hyper-parameter. The default 0.5 weights the
two components equally.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Matches sentence boundaries on ., ! or ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Output column order for cross-encoder/nli-deberta-v3-base (per the
# model card): contradiction, entailment, neutral.
_ENTAILMENT_INDEX = 1

_cached_default_model: Any = None


def _get_default_nli_model() -> Any:
    """Load and cache the default NLI cross-encoder on first use."""
    global _cached_default_model
    if _cached_default_model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading NLI cross-encoder %s", _DEFAULT_NLI_MODEL)
        _cached_default_model = CrossEncoder(_DEFAULT_NLI_MODEL)
    return _cached_default_model


def _reset_default_nli_model() -> None:
    """Clear the cached NLI model. Intended for tests."""
    global _cached_default_model
    _cached_default_model = None


def _split_sentences(text: str) -> list[str]:
    """Split an answer into sentences on ., ! or ? boundaries."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _section_match(
    retrieved_sections: list[str], relevant_sections: list[str]
) -> float:
    """Jaccard similarity between retrieved and ground-truth section sets.

    Two empty sets count as a match (1.0): there is nothing to
    disagree about, which is the sensible behaviour for edge-case
    queries where relevant_sections is intentionally empty.
    """
    r = {s for s in retrieved_sections if s}
    g = {s for s in relevant_sections if s}
    if not r and not g:
        return 1.0
    if not r or not g:
        return 0.0
    return len(r & g) / len(r | g)


def _nli_faithfulness(
    answer_sentences: list[str],
    concatenated_context: str,
    nli_model: Any,
    batch_size: int,
) -> float:
    """Mean entailment probability across answer sentences."""
    if not answer_sentences:
        return 0.0
    pairs = [(concatenated_context, s) for s in answer_sentences]
    scores = nli_model.predict(
        pairs, apply_softmax=True, batch_size=batch_size
    )
    # scores is (N, 3); entailment is column 1.
    entailment = [float(row[_ENTAILMENT_INDEX]) for row in scores]
    return sum(entailment) / len(entailment)


def section_grounded_faithfulness(
    answer: str,
    contexts: list[str],
    retrieved_sections: list[str],
    relevant_sections: list[str],
    alpha: float = 0.5,
    nli_model: Any = None,
    batch_size: int = 16,
) -> dict[str, float]:
    """Compute the SGF metric for a single QA result.

    Args:
        answer: The generated answer text.
        contexts: Retrieved context chunks.
        retrieved_sections: Section labels of the retrieved chunks,
            in the same order as contexts.
        relevant_sections: Ground-truth section labels from the QA
            pair's relevant_sections field.
        alpha: Weight of the NLI-Faithfulness component; 0.5 by
            default.
        nli_model: A pre-loaded CrossEncoder. If None, the default
            cross-encoder/nli-deberta-v3-base is loaded lazily on the
            first call and cached.
        batch_size: Batch size for the NLI predict call. Defaults to
            16, which fits comfortably on an M2 Pro CPU.

    Returns:
        Dict with sgf, nli_faith, section_match and alpha.

    Raises:
        ValueError: If alpha is not in [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    section_match = _section_match(retrieved_sections, relevant_sections)

    answer_sentences = _split_sentences(answer)
    if not answer_sentences or not contexts:
        nli_faith = 0.0
    else:
        if nli_model is None:
            nli_model = _get_default_nli_model()
        concatenated = "\n\n".join(contexts)
        nli_faith = _nli_faithfulness(
            answer_sentences, concatenated, nli_model, batch_size=batch_size
        )

    sgf = alpha * nli_faith + (1.0 - alpha) * section_match

    return {
        "sgf": float(sgf),
        "nli_faith": float(nli_faith),
        "section_match": float(section_match),
        "alpha": float(alpha),
    }
