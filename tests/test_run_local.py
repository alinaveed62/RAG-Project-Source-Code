"""Tests for the one-command local startup script.

Covers two behaviours. First, check_ollama must catch the full
requests.RequestException hierarchy, so a slow Ollama that raises
Timeout or ConnectTimeout is reported as "not running" rather than
producing a traceback. Second, main must default host to 127.0.0.1
(loopback) so the Flask app is not exposed on the LAN by default; the
--host flag is the explicit opt-in for other interfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import run_local  # noqa: E402, path inserted above


class TestCheckOllama:
    """Verify that check_ollama handles every requests exception type."""

    def test_returns_true_on_200(self, monkeypatch):
        response = MagicMock()
        response.status_code = 200
        monkeypatch.setattr(run_local.requests, "get", lambda *a, **k: response)

        assert run_local.check_ollama("http://localhost:11434") is True

    def test_returns_false_on_non_200(self, monkeypatch):
        response = MagicMock()
        response.status_code = 503
        monkeypatch.setattr(run_local.requests, "get", lambda *a, **k: response)

        assert run_local.check_ollama("http://localhost:11434") is False

    @pytest.mark.parametrize(
        "exc",
        [
            requests.ConnectionError("refused"),
            requests.Timeout("slow"),
            requests.ConnectTimeout("handshake"),
            requests.ReadTimeout("stalled"),
            requests.RequestException("generic"),
        ],
    )
    def test_returns_false_on_any_request_exception(self, monkeypatch, exc):
        def _raise(*_a, **_k):
            raise exc

        monkeypatch.setattr(run_local.requests, "get", _raise)
        assert run_local.check_ollama("http://localhost:11434") is False


class TestHostFlag:
    """The default host is 127.0.0.1; passing --host 0.0.0.0 opts in to LAN."""

    def _run_main(self, argv: list[str], tmp_path: Path | None = None):
        """Invoke run_local.main with the given argv, intercepting the
        side-effects that would otherwise require Ollama, a FAISS index and a
        running Flask process."""
        dummy_pipeline = MagicMock()
        dummy_pipeline.build_index = MagicMock()
        dummy_pipeline.setup = MagicMock()

        dummy_app = MagicMock()

        # Pydantic v2 rejects a MagicMock when the field is typed as Path,
        # so build real paths under tmp_path to let RAGConfig construct.
        base = tmp_path or Path("/tmp/run_local_test")
        base.mkdir(parents=True, exist_ok=True)
        chunks = base / "chunks.jsonl"
        chunks.write_text("")
        index = base / "index"
        index.mkdir(exist_ok=True)
        (index / "index.faiss").write_bytes(b"")  # make the index look built

        with patch.object(run_local, "check_ollama", return_value=True), patch.object(
            run_local, "RAGPipeline", return_value=dummy_pipeline
        ), patch.object(run_local, "create_app", return_value=dummy_app), patch.object(
            run_local, "CHUNKS_PATH", chunks
        ), patch.object(run_local, "INDEX_DIR", index), patch.object(sys, "argv", argv):
            run_local.main()

        return dummy_app

    def test_default_host_is_loopback(self, tmp_path):
        app = self._run_main(["run_local.py"], tmp_path=tmp_path)
        app.run.assert_called_once()
        kwargs = app.run.call_args.kwargs
        assert kwargs["host"] == "127.0.0.1"

    def test_explicit_host_passed_through(self, tmp_path):
        app = self._run_main(["run_local.py", "--host", "0.0.0.0"], tmp_path=tmp_path)
        kwargs = app.run.call_args.kwargs
        assert kwargs["host"] == "0.0.0.0"

    def test_port_flag_passed_through(self, tmp_path):
        app = self._run_main(["run_local.py", "--port", "5500"], tmp_path=tmp_path)
        kwargs = app.run.call_args.kwargs
        assert kwargs["port"] == 5500


class TestPreflightExits:
    """``main`` must exit with a non-zero status and an actionable log line
    when a prerequisite (Ollama, chunks file) is missing, so a user running
    the script for the first time sees a clear hint instead of a stack trace
    deep inside RAGPipeline.setup()."""

    def test_main_exits_when_ollama_is_not_running(self, monkeypatch, caplog):
        monkeypatch.setattr(run_local, "check_ollama", lambda *a, **k: False)
        monkeypatch.setattr(sys, "argv", ["run_local.py"])
        with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
            run_local.main()
        assert exc_info.value.code == 1
        assert "Ollama is not running" in caplog.text

    def test_main_exits_when_chunks_file_is_missing(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(run_local, "check_ollama", lambda *a, **k: True)
        monkeypatch.setattr(
            run_local, "CHUNKS_PATH", tmp_path / "does_not_exist.jsonl"
        )
        monkeypatch.setattr(sys, "argv", ["run_local.py"])
        with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
            run_local.main()
        assert exc_info.value.code == 1
        assert "Chunks file not found" in caplog.text


class TestBuildIndexFlag:
    """``--build-index`` forces ``pipeline.build_index()`` to run even when
    an existing ``index.faiss`` is present, so the user can refresh the
    corpus after a new scrape without deleting files by hand."""

    def test_build_index_flag_invokes_build_even_when_index_exists(
        self, tmp_path, monkeypatch
    ):
        dummy_pipeline = MagicMock()
        dummy_app = MagicMock()

        chunks = tmp_path / "chunks.jsonl"
        chunks.write_text("")
        index = tmp_path / "index"
        index.mkdir()
        (index / "index.faiss").write_bytes(b"")

        monkeypatch.setattr(run_local, "check_ollama", lambda *a, **k: True)
        monkeypatch.setattr(run_local, "RAGPipeline", lambda *a, **k: dummy_pipeline)
        monkeypatch.setattr(run_local, "create_app", lambda *a, **k: dummy_app)
        monkeypatch.setattr(run_local, "CHUNKS_PATH", chunks)
        monkeypatch.setattr(run_local, "INDEX_DIR", index)
        monkeypatch.setattr(sys, "argv", ["run_local.py", "--build-index"])

        run_local.main()

        dummy_pipeline.build_index.assert_called_once()
