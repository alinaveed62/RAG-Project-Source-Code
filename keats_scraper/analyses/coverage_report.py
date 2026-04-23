"""Handbook coverage analysis.

Produces both a machine-readable JSON and a human-readable Markdown report
that quantify what fraction of the discovered KEATS resources made it into
the RAG corpus, broken down by the ten handbook sections and by resource
type. The supervisor's fully automated approach to data preparation
requirement is best supported by this kind of empirical coverage evidence
alongside the pipeline itself.

Usage::

    python keats_scraper/analyses/coverage_report.py \\
        --chunks-jsonl keats_scraper/data/chunks/chunks_for_embedding.jsonl \\
        --documents-jsonl keats_scraper/data/processed/documents.jsonl \\
        --output-dir keats_scraper/data
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:  # pragma: no branch - empty JSONL would return [] via the for-loop fall-through
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_coverage_report(
    chunks_path: Path,
    documents_path: Path,
    discovered_path: Path | None = None,
) -> dict[str, Any]:
    """Compute coverage statistics from the scraper artefacts.

    Args:
        chunks_path: Path to chunks_for_embedding.jsonl produced by
            the chunker.
        documents_path: Path to documents.jsonl produced by the
            scraper, before chunking.
        discovered_path: Optional path to discovered_resources.jsonl
            produced by the scraper during resource discovery. When
            present, the report reconciles the list of URLs the
            navigator found against the URLs that reached the chunk
            index, so a silent drop is visible to a reviewer without
            re-running the scrape.

    Returns:
        Dict with keys:
            - total_documents, total_chunks
            - chunks_per_section: dict mapping section label to
              count of chunks
            - documents_per_section: dict mapping section label to
              count of distinct source documents
            - resource_types_seen: per-resource-type counts when
              present in document metadata
            - sections_with_zero_chunks: list of sections where at
              least one document exists but no chunks were produced
            - discovered_total, missing_urls (manifest-aware
              keys, only present when discovered_path is supplied
              and exists on disk)
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks_jsonl not found: {chunks_path}")
    if not documents_path.exists():
        raise FileNotFoundError(f"documents_jsonl not found: {documents_path}")

    chunks = _read_jsonl(chunks_path)
    documents = _read_jsonl(documents_path)

    chunks_per_section: Counter[str] = Counter()
    doc_ids_per_section: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        section = c.get("section", "Unknown")
        chunks_per_section[section] += 1
        doc_id = c.get("document_id") or c.get("source", "unknown")
        doc_ids_per_section[section].add(doc_id)

    documents_per_section = {s: len(ids) for s, ids in doc_ids_per_section.items()}

    resource_types: Counter[str] = Counter()
    doc_sections: Counter[str] = Counter()
    processed_urls: set[str] = set()
    for d in documents:
        metadata = d.get("metadata") or {}
        resource_types[metadata.get("resource_type", "unknown")] += 1
        doc_sections[metadata.get("section", "Unknown")] += 1
        source_url = metadata.get("source_url")
        if source_url:  # pragma: no branch - DocumentMetadata guarantees source_url is set
            processed_urls.add(source_url)

    sections_with_zero_chunks = sorted(
        {s for s in doc_sections if chunks_per_section.get(s, 0) == 0}
    )

    report: dict[str, Any] = {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "chunks_per_section": dict(sorted(chunks_per_section.items())),
        "documents_per_section": dict(sorted(documents_per_section.items())),
        "documents_per_section_from_docs": dict(sorted(doc_sections.items())),
        "resource_types_seen": dict(sorted(resource_types.items())),
        "sections_with_zero_chunks": sections_with_zero_chunks,
    }

    if discovered_path is not None and discovered_path.exists():
        manifest = _read_jsonl(discovered_path)
        discovered_urls = {row["url"] for row in manifest if row.get("url")}
        missing = sorted(discovered_urls - processed_urls)
        discovered_by_section: Counter[str] = Counter()
        for row in manifest:
            discovered_by_section[row.get("section", "Unknown")] += 1
        report["discovered_total"] = len(discovered_urls)
        report["processed_total"] = len(processed_urls)
        report["missing_urls"] = missing
        report["missing_count"] = len(missing)
        report["discovered_per_section"] = dict(
            sorted(discovered_by_section.items())
        )

    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the coverage report as human-readable Markdown."""
    lines: list[str] = []
    lines.append("# Handbook Coverage Report")
    lines.append("")
    lines.append(
        f"- Total documents scraped: **{report['total_documents']}**"
    )
    lines.append(f"- Total chunks produced: **{report['total_chunks']}**")
    if "discovered_total" in report:
        lines.append(
            f"- Resources discovered by the navigator: "
            f"**{report['discovered_total']}**"
        )
        lines.append(
            f"- Resources whose content reached the document index: "
            f"**{report['processed_total']}**"
        )
        missing = report["missing_count"]
        if missing == 0:
            lines.append(
                "- Reconciliation: **every discovered resource produced a "
                "document.**"
            )
        else:
            lines.append(
                f"- Reconciliation: **{missing}** discovered URL(s) did "
                "not reach the document index. See *Missing URLs* below."
            )
    if report["sections_with_zero_chunks"]:
        lines.append(
            f"- Sections with zero chunks: {report['sections_with_zero_chunks']}"
        )
    else:
        lines.append("- Every handbook section produced at least one chunk.")
    lines.append("")

    lines.append("## Chunks per section")
    lines.append("")
    lines.append("| Section | Chunks |")
    lines.append("|---|---:|")
    for section, n in report["chunks_per_section"].items():
        lines.append(f"| {section} | {n} |")
    lines.append("")

    lines.append("## Documents per section (from chunk metadata)")
    lines.append("")
    lines.append("| Section | Documents |")
    lines.append("|---|---:|")
    for section, n in report["documents_per_section"].items():
        lines.append(f"| {section} | {n} |")
    lines.append("")

    lines.append("## Resource types seen")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    for rtype, n in report["resource_types_seen"].items():
        lines.append(f"| {rtype} | {n} |")
    lines.append("")

    if report.get("missing_urls"):
        lines.append("## Missing URLs (discovered but not in document index)")
        lines.append("")
        for url in report["missing_urls"]:
            lines.append(f"- {url}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=Path("keats_scraper/data/chunks/chunks_for_embedding.jsonl"),
    )
    parser.add_argument(
        "--documents-jsonl",
        type=Path,
        default=Path("keats_scraper/data/processed/documents.jsonl"),
    )
    parser.add_argument(
        "--discovered-jsonl",
        type=Path,
        default=Path(
            "keats_scraper/data/checkpoints/discovered_resources.jsonl"
        ),
        help=(
            "Path to the scraper's discovery manifest. If the file is absent "
            "the report falls back to document-only metrics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("keats_scraper/data"),
    )
    ns = parser.parse_args(argv)

    report = build_coverage_report(
        ns.chunks_jsonl, ns.documents_jsonl, ns.discovered_jsonl
    )

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = ns.output_dir / "coverage_report.json"
    md_path = ns.output_dir / "coverage_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("Wrote coverage report to %s and %s", json_path, md_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
