"""Regression tests for the manifest-aware coverage report.

The coverage report historically counted only *processed* chunks and
documents. A silent drop during scraping (for example: a discovered
KEATS book that failed to enumerate any chapters) was therefore
invisible once the run finished, because nothing on disk recorded the
original discover_resources output to compare against.

Day-4 audit finding (2026-04-19): persist the discovery manifest at
data/checkpoints/discovered_resources.jsonl and reconcile it inside
build_coverage_report. These tests pin the reconciliation
behaviour:

- When the manifest is present, the report exposes discovered_total,
  processed_total, missing_urls and missing_count.
- When the manifest is absent the report falls back to the pre-fix
  shape so legacy callers keep working.
"""

from __future__ import annotations

import json
from pathlib import Path

from keats_scraper.analyses.coverage_report import build_coverage_report, render_markdown


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _fake_chunks(n: int, url: str, section: str) -> list[dict]:
    return [
        {
            "section": section,
            "document_id": f"doc_{url}",
            "text": f"chunk {i} of {url}",
        }
        for i in range(n)
    ]


def _fake_document(url: str, section: str, rtype: str = "page") -> dict:
    return {
        "id": f"doc_{url}",
        "content": f"content of {url}",
        "metadata": {
            "source_url": url,
            "title": f"Title {url}",
            "section": section,
            "resource_type": rtype,
        },
    }


class TestManifestReconciliation:
    """Day-4 regression: discovery manifest closes the audit trail."""

    def test_manifest_absent_keeps_legacy_shape(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        _write_jsonl(chunks_path, _fake_chunks(3, "a", "Programmes"))
        _write_jsonl(docs_path, [_fake_document("a", "Programmes")])

        report = build_coverage_report(chunks_path, docs_path, None)

        assert report["total_documents"] == 1
        assert report["total_chunks"] == 3
        assert "discovered_total" not in report
        assert "missing_urls" not in report

    def test_manifest_all_processed_reports_zero_missing(
        self, tmp_path: Path
    ) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        manifest_path = tmp_path / "discovered.jsonl"

        _write_jsonl(
            chunks_path,
            _fake_chunks(2, "a", "Programmes") + _fake_chunks(2, "b", "Main"),
        )
        _write_jsonl(
            docs_path,
            [
                _fake_document("a", "Programmes"),
                _fake_document("b", "Main"),
            ],
        )
        _write_jsonl(
            manifest_path,
            [
                {
                    "url": "a",
                    "title": "A",
                    "resource_type": "page",
                    "section": "Programmes",
                },
                {
                    "url": "b",
                    "title": "B",
                    "resource_type": "page",
                    "section": "Main",
                },
            ],
        )

        report = build_coverage_report(chunks_path, docs_path, manifest_path)

        assert report["discovered_total"] == 2
        assert report["processed_total"] == 2
        assert report["missing_count"] == 0
        assert report["missing_urls"] == []

    def test_manifest_silent_drop_is_surfaced(self, tmp_path: Path) -> None:
        """Two URLs were discovered but only one reached the document index."""
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        manifest_path = tmp_path / "discovered.jsonl"

        _write_jsonl(chunks_path, _fake_chunks(2, "a", "Programmes"))
        _write_jsonl(docs_path, [_fake_document("a", "Programmes")])
        _write_jsonl(
            manifest_path,
            [
                {
                    "url": "a",
                    "title": "A",
                    "resource_type": "page",
                    "section": "Programmes",
                },
                {
                    "url": "b_silently_dropped",
                    "title": "B",
                    "resource_type": "pdf",
                    "section": "Support",
                },
            ],
        )

        report = build_coverage_report(chunks_path, docs_path, manifest_path)

        assert report["discovered_total"] == 2
        assert report["processed_total"] == 1
        assert report["missing_count"] == 1
        assert report["missing_urls"] == ["b_silently_dropped"]
        assert report["discovered_per_section"] == {
            "Programmes": 1,
            "Support": 1,
        }

    def test_markdown_names_the_missing_urls(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        manifest_path = tmp_path / "discovered.jsonl"

        _write_jsonl(chunks_path, _fake_chunks(1, "a", "Programmes"))
        _write_jsonl(docs_path, [_fake_document("a", "Programmes")])
        _write_jsonl(
            manifest_path,
            [
                {
                    "url": "a",
                    "title": "A",
                    "resource_type": "page",
                    "section": "Programmes",
                },
                {
                    "url": "https://keats/missing/42",
                    "title": "M",
                    "resource_type": "pdf",
                    "section": "Main",
                },
            ],
        )

        report = build_coverage_report(chunks_path, docs_path, manifest_path)
        md = render_markdown(report)

        assert "Missing URLs" in md
        assert "https://keats/missing/42" in md
        assert "1" in md

    def test_manifest_path_missing_on_disk_skips_reconciliation(
        self, tmp_path: Path
    ) -> None:
        """Passing a path that does not exist must not crash the build."""
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        _write_jsonl(chunks_path, _fake_chunks(1, "a", "Programmes"))
        _write_jsonl(docs_path, [_fake_document("a", "Programmes")])

        report = build_coverage_report(
            chunks_path, docs_path, tmp_path / "does_not_exist.jsonl"
        )

        assert "discovered_total" not in report


class TestBuildCoverageReportErrors:
    """Error branches in build_coverage_report."""

    def test_missing_chunks_file_raises(self, tmp_path: Path) -> None:
        docs_path = tmp_path / "docs.jsonl"
        _write_jsonl(docs_path, [])
        import pytest

        with pytest.raises(FileNotFoundError):
            build_coverage_report(tmp_path / "absent.jsonl", docs_path)

    def test_missing_documents_file_raises(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_path, [])
        import pytest

        with pytest.raises(FileNotFoundError):
            build_coverage_report(chunks_path, tmp_path / "absent.jsonl")

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path: Path) -> None:
        """_read_jsonl must silently drop blank and whitespace-only lines.

        Scraper JSONL outputs are newline-delimited and a trailing blank line
        or an editor-inserted whitespace line should not break the coverage
        report. This test pins the skip behaviour by feeding a mixed file
        containing valid JSON rows interleaved with blank and whitespace
        lines, and asserting the valid rows are still counted.
        """
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"

        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        valid_chunk = json.dumps(
            {"section": "Programmes", "document_id": "doc_a", "text": "c0"}
        )
        chunks_path.write_text(
            f"\n{valid_chunk}\n   \n{valid_chunk}\n",
            encoding="utf-8",
        )
        _write_jsonl(docs_path, [_fake_document("a", "Programmes")])

        report = build_coverage_report(chunks_path, docs_path)

        assert report["total_chunks"] == 2
        assert report["total_documents"] == 1


class TestRenderMarkdownEdgeCases:
    """Edge cases in render_markdown that the reconciliation tests
    above don't reach."""

    def test_render_without_manifest_has_section_fallback_message(
        self, tmp_path: Path
    ) -> None:
        """When no manifest is supplied and every section has chunks, the
        output contains the 'every section produced at least one chunk'
        line instead of the 'zero chunks' warning."""
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        _write_jsonl(chunks_path, _fake_chunks(2, "a", "Main"))
        _write_jsonl(docs_path, [_fake_document("a", "Main")])

        report = build_coverage_report(chunks_path, docs_path)
        md = render_markdown(report)
        assert "at least one chunk" in md

    def test_render_with_zero_missing_reconciliation_message(
        self, tmp_path: Path
    ) -> None:
        """When manifest reconciliation succeeds with zero missing URLs,
        the 'every discovered resource produced a document' line fires."""
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"
        _write_jsonl(chunks_path, _fake_chunks(1, "a", "Main"))
        _write_jsonl(docs_path, [_fake_document("a", "Main")])
        _write_jsonl(
            manifest_path,
            [{"url": "a", "title": "t", "resource_type": "page", "section": "Main"}],
        )

        report = build_coverage_report(chunks_path, docs_path, manifest_path)
        md = render_markdown(report)
        assert "every discovered resource produced a document" in md

    def test_render_flags_sections_with_zero_chunks(self, tmp_path: Path) -> None:
        """A section that has documents but produced zero chunks is
        surfaced via the 'Sections with zero chunks' bullet in the markdown
        report so a reviewer can see dropped coverage at a glance."""
        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        # Chunks exist for section 'A'; docs cover both 'A' and 'B'.
        _write_jsonl(chunks_path, _fake_chunks(1, "x", "A"))
        _write_jsonl(
            docs_path,
            [_fake_document("x", "A"), _fake_document("y", "B")],
        )

        report = build_coverage_report(chunks_path, docs_path)
        md = render_markdown(report)
        assert "Sections with zero chunks" in md
        assert "B" in md


class TestMainEntryPoint:
    """Exercises the main CLI function for the coverage report."""

    def test_main_writes_json_and_markdown(self, tmp_path: Path) -> None:
        from keats_scraper.analyses.coverage_report import main

        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"
        output_dir = tmp_path / "out"

        _write_jsonl(chunks_path, _fake_chunks(2, "a", "Main"))
        _write_jsonl(docs_path, [_fake_document("a", "Main")])
        _write_jsonl(
            manifest_path,
            [{"url": "a", "title": "t", "resource_type": "page", "section": "Main"}],
        )

        main(
            [
                "--chunks-jsonl",
                str(chunks_path),
                "--documents-jsonl",
                str(docs_path),
                "--discovered-jsonl",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
            ]
        )

        json_out = output_dir / "coverage_report.json"
        md_out = output_dir / "coverage_report.md"
        assert json_out.exists()
        assert md_out.exists()
        # Machine-readable JSON must parse and contain the known keys.
        report = json.loads(json_out.read_text())
        assert report["total_documents"] == 1
        assert report["total_chunks"] == 2
        assert report["discovered_total"] == 1
        # Markdown body is human-readable.
        assert "Handbook Coverage Report" in md_out.read_text()

    def test_main_without_manifest_falls_back(self, tmp_path: Path) -> None:
        from keats_scraper.analyses.coverage_report import main

        chunks_path = tmp_path / "chunks.jsonl"
        docs_path = tmp_path / "docs.jsonl"
        output_dir = tmp_path / "out"

        _write_jsonl(chunks_path, _fake_chunks(1, "a", "Main"))
        _write_jsonl(docs_path, [_fake_document("a", "Main")])

        main(
            [
                "--chunks-jsonl",
                str(chunks_path),
                "--documents-jsonl",
                str(docs_path),
                "--discovered-jsonl",
                str(tmp_path / "does_not_exist.jsonl"),
                "--output-dir",
                str(output_dir),
            ]
        )

        report = json.loads((output_dir / "coverage_report.json").read_text())
        assert "discovered_total" not in report
