"""Tests for generate_static_figures.py."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.generate_static_figures import (
    generate_category_distribution,
    generate_difficulty_distribution,
    generate_section_distribution,
    load_section_counts,
    main,
)


def _qa_pairs() -> list[dict]:
    return [
        {"id": "q1", "category": "attendance", "difficulty": "easy"},
        {"id": "q2", "category": "attendance", "difficulty": "medium"},
        {"id": "q3", "category": "assessment", "difficulty": "hard"},
        {"id": "q4", "category": "support"},  # missing difficulty → "unknown"
    ]


def _chunk_index(tmp_path, by_section: dict[str, list[str]] | None = None) -> Path:
    path: Path = tmp_path / "chunk_index.json"
    if by_section is None:
        by_section = {
            "Teaching & Assessment": ["doc_a_chunk_0", "doc_a_chunk_1", "doc_b_chunk_0"],
            "Programmes": ["doc_c_chunk_0"],
        }
    data = {
        "total_chunks": sum(len(v) for v in by_section.values()),
        "chunks_by_section": by_section,
        "chunks_by_document": {
            doc_id: []
            for chunk_ids in by_section.values()
            for chunk_id in chunk_ids
            if "_chunk_" in chunk_id
            for doc_id in [chunk_id.split("_chunk_")[0]]
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadSectionCounts:
    def test_counts_unique_documents_per_section(self, tmp_path):
        path = _chunk_index(tmp_path)
        section_docs, total_docs, total_chunks = load_section_counts(path)
        assert section_docs["Teaching & Assessment"] == 2  # doc_a + doc_b
        assert section_docs["Programmes"] == 1  # doc_c
        assert total_docs == 3
        assert total_chunks == 4

    def test_handles_chunk_without_parseable_suffix(self, tmp_path):
        path = _chunk_index(
            tmp_path,
            by_section={"WeirdSection": ["bogusid"]},
        )
        section_docs, total_docs, total_chunks = load_section_counts(path)
        assert section_docs["WeirdSection"] == 0
        assert total_chunks == 1

    def test_missing_total_chunks_defaults_to_zero(self, tmp_path):
        path: Path = tmp_path / "x.json"
        path.write_text(json.dumps({"chunks_by_section": {}, "chunks_by_document": {}}))
        _, total_docs, total_chunks = load_section_counts(path)
        assert total_chunks == 0
        assert total_docs == 0


class TestGenerateFigures:
    def test_category_distribution_writes_pdf(self, tmp_path):
        generate_category_distribution(_qa_pairs(), figures_dir=tmp_path)
        assert (tmp_path / "qa_category_distribution.pdf").exists()

    def test_category_distribution_empty_qa_pairs_noop(self, tmp_path):
        generate_category_distribution([], figures_dir=tmp_path)
        assert not (tmp_path / "qa_category_distribution.pdf").exists()

    def test_difficulty_distribution_writes_pdf(self, tmp_path):
        generate_difficulty_distribution(_qa_pairs(), figures_dir=tmp_path)
        assert (tmp_path / "qa_difficulty_distribution.pdf").exists()

    def test_section_distribution_writes_pdf(self, tmp_path):
        generate_section_distribution(
            {"Teaching & Assessment": 23, "Programmes": 47},
            total_docs=70,
            total_chunks=160,
            figures_dir=tmp_path,
        )
        assert (tmp_path / "section_distribution.pdf").exists()

    def test_section_distribution_empty_noop(self, tmp_path):
        generate_section_distribution(
            {}, total_docs=0, total_chunks=0, figures_dir=tmp_path
        )
        assert not (tmp_path / "section_distribution.pdf").exists()


class TestMain:
    def test_main_end_to_end(self, tmp_path):
        qa_path = tmp_path / "qa.json"
        qa_path.write_text(json.dumps(_qa_pairs()), encoding="utf-8")
        idx_path = _chunk_index(tmp_path)
        figures_dir = tmp_path / "figures"
        main(
            qa_pairs_path=qa_path,
            chunk_index_path=idx_path,
            figures_dir=figures_dir,
        )
        assert (figures_dir / "qa_category_distribution.pdf").exists()
        assert (figures_dir / "qa_difficulty_distribution.pdf").exists()
        assert (figures_dir / "section_distribution.pdf").exists()
