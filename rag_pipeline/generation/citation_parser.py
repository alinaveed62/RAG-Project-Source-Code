"""Parse and validate inline citation markers in generated answers.

The prompt template wraps each retrieved chunk as:

    Context chunk [Source: {chunk_id}]:
    {chunk_text}

The model is instructed to append [Source: chunk_id] after each
grounded claim. This module extracts those markers from the generated
answer and validates them against the set of chunk IDs the model was
actually shown. Any citation that references an unknown chunk is
dropped with a logged warning.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Matches [Source: chunk_id] with flexible whitespace. The chunk id
# alphabet (letters, digits, underscore, hyphen, dot, colon) is a
# superset of the scraper's current doc_<md5>_chunk_<N> format, so
# every real chunk id is matched. If the chunk id scheme ever needs
# to accept new characters (for example / or whitespace) widen this
# class and update the test that pins the accepted alphabet.
_CITATION_RE = re.compile(r"\[Source:\s*([A-Za-z0-9_\-.:]+)\s*\]")


class Citation(BaseModel):
    """A single validated citation extracted from an LLM answer."""

    chunk_id: str = Field(..., description="The cited chunk identifier.")
    span_start: int = Field(..., ge=0, description="Inclusive start offset in the answer text.")
    span_end: int = Field(..., ge=0, description="Exclusive end offset in the answer text.")


def parse_citations(
    answer_text: str,
    valid_chunk_ids: Iterable[str],
) -> list[Citation]:
    """Extract [Source: chunk_id] markers from answer_text.

    Citations whose chunk IDs are not in valid_chunk_ids are dropped
    with a warning, so hallucinated identifiers cannot leak into the UI.

    Args:
        answer_text: The model-generated answer, possibly carrying
            [Source: chunk_id] markers.
        valid_chunk_ids: The set of chunk IDs the model was shown as
            context for this answer.

    Returns:
        Validated Citation objects in order of appearance. Each one
        records the character offsets of its marker inside answer_text,
        so downstream rendering can replace the marker with a UI chip.
    """
    valid_set: set[str] = set(valid_chunk_ids)
    out: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        chunk_id = match.group(1)
        if chunk_id not in valid_set:
            logger.warning(
                "Dropping hallucinated citation %r (not in retrieved set)",
                chunk_id,
            )
            continue
        out.append(
            Citation(
                chunk_id=chunk_id,
                span_start=match.start(),
                span_end=match.end(),
            )
        )
    return out


def strip_citations(answer_text: str) -> str:
    """Remove all [Source: ...] markers for plain text rendering.

    Only collapses horizontal whitespace left behind by the removal,
    so "text [Source: c1] more" becomes "text more" and not
    "text  more". Paragraph breaks (newlines) are preserved, so a
    multi-paragraph answer still reads as paragraphs when rendered
    outside an HTML block.
    """
    stripped = _CITATION_RE.sub("", answer_text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in stripped.splitlines()]
    return "\n".join(lines).strip("\n")
