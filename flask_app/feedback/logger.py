"""Feedback logging to JSONL."""

import json
from datetime import UTC, datetime
from pathlib import Path


def log_feedback(data: dict, feedback_dir: Path) -> None:
    """Append a feedback entry to feedback.jsonl.

    Args:
        data: Dict with question, answer_rating, source_rating, comments.
        feedback_dir: Directory to store feedback.jsonl.
    """
    feedback_dir = Path(feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        **data,
    }

    feedback_path = feedback_dir / "feedback.jsonl"
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
