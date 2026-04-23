"""Retrieval quality metrics: Precision@k, Recall@k, MRR, nDCG."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_section_index_cache: dict[str, dict[str, list[str]]] = {}


def _build_section_index(chunks: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for chunk in chunks:
        section = chunk.get("section", "")
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue
        index.setdefault(section, []).append(chunk_id)
    return index


def sections_to_chunk_ids(
    chunks_source: str | Path | Iterable[Mapping[str, Any]],
    relevant_sections: list[str],
) -> list[str]:
    """Expand a list of section labels to the chunk IDs belonging to those sections.

    Used by retrieval experiments so that metrics compare chunk IDs (unique per
    retrieved item) rather than section labels (duplicated across retrievals).

    Args:
        chunks_source: Path to a chunks JSONL file, or an iterable of chunk
            dicts carrying at least the keys id and section.
        relevant_sections: Section labels to expand.

    Returns:
        Flattened list of chunk IDs. De-duplicated, preserving insertion order.
    """
    if isinstance(chunks_source, (str, Path)):
        key = str(Path(chunks_source).resolve())
        index = _section_index_cache.get(key)
        if index is None:
            chunks: list[Mapping[str, Any]] = []
            with open(chunks_source, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
            index = _build_section_index(chunks)
            _section_index_cache[key] = index
    else:
        index = _build_section_index(list(chunks_source))

    seen: set[str] = set()
    ordered: list[str] = []
    for section in relevant_sections:
        for chunk_id in index.get(section, []):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _reset_section_cache() -> None:
    """Clear the cached section → chunk-id index. Intended for tests."""
    _section_index_cache.clear()


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Compute Precision@k.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Fraction of top-k retrieved items that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Compute Recall@k.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Fraction of relevant items found in top-k.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.

    Returns:
        1/rank of the first relevant result, or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Compute normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        nDCG@k score between 0 and 1.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Discounted cumulative gain.
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG, with all relevant items packed at the top. The
    # function's top-level guard already handles k <= 0 and an empty
    # relevant_ids, so ideal_count is at least 1 here and idcg is
    # always positive.
    ideal_count = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all retrieval metrics at multiple k values.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth relevant chunk IDs.
        k_values: List of k values to evaluate. Defaults to [1, 3, 5, 10].

    Returns:
        Dict mapping metric names to values.
    """
    k_values = k_values or [1, 3, 5, 10]
    results = {"mrr": mrr(retrieved_ids, relevant_ids)}

    for k in k_values:
        results[f"precision_at_{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"recall_at_{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results
