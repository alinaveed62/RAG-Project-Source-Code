"""Answer quality metrics: ROUGE, BERTScore, faithfulness."""

from __future__ import annotations


def compute_rouge(predicted: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predicted: Generated answer text.
        reference: Expected answer text.

    Returns:
        Dict with rouge_1, rouge_2, rouge_l F1 scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, predicted)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }


def compute_bert_score(
    predicted: str, reference: str, model_type: str = "roberta-large"
) -> dict[str, float]:
    """Compute BERTScore precision, recall, and F1.

    Args:
        predicted: Generated answer text.
        reference: Expected answer text.
        model_type: Model to use for BERTScore.

    Returns:
        Dict with bert_score_precision, bert_score_recall, bert_score_f1.
    """
    from bert_score import score as bert_score

    P, R, F1 = bert_score(
        [predicted], [reference], model_type=model_type, verbose=False
    )
    return {
        "bert_score_precision": P[0].item(),
        "bert_score_recall": R[0].item(),
        "bert_score_f1": F1[0].item(),
    }


def compute_faithfulness(answer: str, context_texts: list[str]) -> float:
    """Lexical groundedness heuristic based on per-sentence overlap.

    Splits the answer into sentences and returns the fraction whose
    content-word set (after removing a fixed English stopword list)
    overlaps the union of context-chunk content words by at least
    30%. The 30% threshold is a design constant, not a validated
    correctness cutoff.

    This is a cheap, deterministic proxy suitable as a per-query
    sanity signal and as a comparison against earlier experimental
    runs that used the same heuristic. The more rigorous correctness
    metric used in the evaluation chapter is
    evaluation.metrics.sgf.section_grounded_faithfulness, which
    combines NLI entailment with a Jaccard section-match score. This
    function is retained because it is fast, auditable, and directly
    comparable to the older runs.

    Args:
        answer: The generated answer.
        context_texts: The retrieved context chunk texts.

    Returns:
        The fraction of answer sentences with at least 30% content-
        word overlap with the retrieved context (0.0 to 1.0). An
        empty answer returns 0.0.
    """
    import re

    answer_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()
    ]
    if not answer_sentences:
        return 0.0

    context_words = set()
    for ctx in context_texts:
        context_words.update(ctx.lower().split())

    faithful_count = 0
    for sentence in answer_sentences:
        sentence_words = set(sentence.lower().split())
        # Drop common stopwords before measuring overlap.
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when", "that",
            "this", "it", "its", "they", "them", "their", "we", "our",
            "you", "your", "he", "she", "his", "her", "i", "me", "my",
        }
        content_words = sentence_words - stopwords
        context_content = context_words - stopwords

        if not content_words:
            faithful_count += 1
            continue

        overlap = len(content_words & context_content) / len(content_words)
        if overlap >= 0.3:
            faithful_count += 1

    return faithful_count / len(answer_sentences)


def evaluate_answer_quality(
    predicted: str,
    reference: str,
    context_texts: list[str] | None = None,
    compute_bert: bool = True,
) -> dict[str, float]:
    """Compute all answer quality metrics.

    Args:
        predicted: Generated answer text.
        reference: Expected/reference answer text.
        context_texts: Retrieved context chunks (for faithfulness).
        compute_bert: Whether to compute BERTScore (slow).

    Returns:
        Dict with all answer quality metrics.
    """
    results = compute_rouge(predicted, reference)

    if compute_bert:
        results.update(compute_bert_score(predicted, reference))

    if context_texts:
        results["faithfulness"] = compute_faithfulness(predicted, context_texts)

    return results
