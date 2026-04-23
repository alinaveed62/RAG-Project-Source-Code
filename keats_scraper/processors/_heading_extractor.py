"""Shared heading-hierarchy extractor for chunkers.

Chunker (recursive) and SemanticChunker both need to look at the
Markdown headings that precede a given character position in the document
so each emitted Chunk can record its heading path. The logic was
originally a private method on Chunker; keeping two copies in parallel
would drift. Factoring the logic into a module-level function lets both
chunkers share a single implementation while Chunker._extract_heading_at_position
stays a thin delegator so the existing tests (which call it as a method)
continue to work unchanged.
"""

from __future__ import annotations

import re

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def extract_heading_path(text: str, position: int) -> list[str]:
    """Return the Markdown heading hierarchy at position in text.

    Scans every heading that appears before position and folds
    lower-level headings away when a higher-level heading arrives. The
    result is the path from the document's top-most heading to the most
    specific heading that still applies at position.

    Args:
        text: The full document text (Markdown).
        position: A character offset into text. Headings after the
            offset are ignored.

    Returns:
        Ordered list of heading titles from most general to most specific.
        Empty if the position sits before any heading or the document has
        no Markdown headings.
    """
    # Collect (level, title) for every heading preceding position.
    headings: list[tuple[int, str]] = []
    for match in _HEADING_PATTERN.finditer(text[:position]):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((level, title))

    # Fold deeper headings away when a new heading at or above the level
    # appears. hierarchy ends up with one title per active level.
    hierarchy: dict[int, str] = {}
    for level, title in headings:
        hierarchy[level] = title
        for deeper in list(hierarchy):
            if deeper > level:
                del hierarchy[deeper]
    return [hierarchy[lvl] for lvl in sorted(hierarchy)]
