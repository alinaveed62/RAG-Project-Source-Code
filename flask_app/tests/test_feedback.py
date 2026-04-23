"""Tests for feedback logger."""

import json

from flask_app.feedback.logger import log_feedback


class TestLogFeedback:
    def test_creates_file(self, tmp_path):
        log_feedback({"question": "test?", "answer_rating": 5}, tmp_path)
        assert (tmp_path / "feedback.jsonl").exists()

    def test_appends_to_file(self, tmp_path):
        log_feedback({"question": "q1"}, tmp_path)
        log_feedback({"question": "q2"}, tmp_path)

        lines = (tmp_path / "feedback.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_includes_timestamp(self, tmp_path):
        log_feedback({"question": "test?"}, tmp_path)
        entry = json.loads((tmp_path / "feedback.jsonl").read_text().strip())
        assert "timestamp" in entry

    def test_preserves_data_fields(self, tmp_path):
        log_feedback(
            {"question": "test?", "answer_rating": 4, "comments": "good"},
            tmp_path,
        )
        entry = json.loads((tmp_path / "feedback.jsonl").read_text().strip())
        assert entry["question"] == "test?"
        assert entry["answer_rating"] == 4
        assert entry["comments"] == "good"

    def test_creates_directory_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b"
        log_feedback({"question": "test?"}, nested)
        assert (nested / "feedback.jsonl").exists()

    def test_valid_jsonl(self, tmp_path):
        for i in range(5):
            log_feedback({"question": f"q{i}"}, tmp_path)

        with open(tmp_path / "feedback.jsonl") as f:
            for line in f:
                data = json.loads(line)
                assert "question" in data
                assert "timestamp" in data
