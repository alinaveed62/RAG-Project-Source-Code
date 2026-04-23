"""Generate static figures for the report from existing data.

Creates charts that do not require running any experiment: the QA
category distribution and QA difficulty distribution from
qa_pairs.json, and the KEATS section distribution derived live
from chunk_index.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.config import REPORT_FIGURES_DIR

FIGURES_DIR = REPORT_FIGURES_DIR
QA_PAIRS_PATH = Path(__file__).resolve().parent / "test_set/qa_pairs.json"
CHUNK_INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "keats_scraper/data/chunks/chunk_index.json"
)

KCL_PURPLE = "#6e1d45"


def load_section_counts(chunk_index_path: Path) -> tuple[dict[str, int], int, int]:
    """Return per-section document counts plus corpus-wide totals.

    The per-section document count is derived from the unique
    document IDs that own each section's chunks, so the number shown
    on each bar in the figure matches the "documents" label. The
    corpus-wide totals (total documents and total chunks) let the
    figure caption state the size of the current scrape.
    """
    with chunk_index_path.open(encoding="utf-8") as f:
        data = json.load(f)

    total_chunks = int(data.get("total_chunks", 0))
    by_section = data.get("chunks_by_section", {})

    # chunks_by_document is the source of truth for the total
    # document count. chunks_by_section maps section to a list of
    # chunk IDs, and each chunk ID encodes its parent document, so
    # the per-section document count is derived from the chunk-id
    # prefix.
    by_document = data.get("chunks_by_document", {})
    total_docs = len(by_document)

    section_doc_counts: dict[str, int] = {}
    for section, chunk_ids in by_section.items():
        doc_ids_in_section: set[str] = set()
        for chunk_id in chunk_ids:
            # Chunk IDs have the form <doc_id>_chunk_<n>.
            if "_chunk_" in chunk_id:
                doc_ids_in_section.add(chunk_id.split("_chunk_")[0])
        section_doc_counts[section] = len(doc_ids_in_section)

    return section_doc_counts, total_docs, total_chunks


def generate_category_distribution(
    qa_pairs: list[dict], figures_dir: Path = FIGURES_DIR
) -> None:
    """Horizontal bar chart of QA pair categories."""
    if not qa_pairs:
        return
    categories = Counter(qa["category"] for qa in qa_pairs)
    labels = sorted(categories.keys(), key=lambda k: categories[k])
    values = [categories[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=KCL_PURPLE, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of QA Pairs")
    ax.set_title(f"Test Set: Category Distribution ({sum(values)} pairs)")
    ax.set_xlim(0, max(values) + 2)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "qa_category_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_difficulty_distribution(
    qa_pairs: list[dict], figures_dir: Path = FIGURES_DIR
) -> None:
    """Bar chart of QA pair difficulty levels."""
    difficulties = Counter(qa.get("difficulty", "unknown") for qa in qa_pairs)
    order = ["easy", "medium", "hard"]
    labels = [d for d in order if d in difficulties]
    values = [difficulties[d] for d in labels]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, values, color=KCL_PURPLE, edgecolor="white", width=0.5)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title("Test Set: Difficulty Distribution")
    ax.set_ylim(0, max(values) + 5)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "qa_difficulty_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_section_distribution(
    section_counts: dict[str, int],
    total_docs: int,
    total_chunks: int,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Horizontal bar chart of KEATS section document counts."""
    if not section_counts:
        return
    labels = sorted(section_counts.keys(), key=lambda k: section_counts[k])
    values = [section_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=KCL_PURPLE, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of Documents")
    ax.set_title(
        f"KEATS Handbook: Documents per Section ({total_docs} documents / "
        f"{total_chunks} chunks)"
    )
    ax.set_xlim(0, max(values) + 5)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "section_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def main(
    qa_pairs_path: Path = QA_PAIRS_PATH,
    chunk_index_path: Path = CHUNK_INDEX_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(qa_pairs_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    generate_category_distribution(qa_pairs, figures_dir=figures_dir)
    generate_difficulty_distribution(qa_pairs, figures_dir=figures_dir)

    section_counts, total_docs, total_chunks = load_section_counts(chunk_index_path)
    generate_section_distribution(
        section_counts, total_docs, total_chunks, figures_dir=figures_dir
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
