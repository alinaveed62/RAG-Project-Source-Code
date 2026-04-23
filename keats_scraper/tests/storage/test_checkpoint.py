"""Tests for CheckpointManager."""

import json
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from keats_scraper.storage.checkpoint import CheckpointManager, ScrapingProgress
from keats_scraper.utils.exceptions import CheckpointError


class TestScrapingProgress:
    """Tests for ScrapingProgress dataclass."""

    def test_new_sets_timestamps(self):
        """Test new() sets started_at and last_updated."""
        progress = ScrapingProgress.new()
        assert progress.started_at is not None
        assert progress.last_updated is not None

    def test_new_timestamps_are_iso_format(self):
        """Test timestamps are ISO format strings."""
        progress = ScrapingProgress.new()
        # Should be parseable as ISO datetime
        datetime.fromisoformat(progress.started_at)
        datetime.fromisoformat(progress.last_updated)

    def test_new_initializes_empty_lists(self):
        """Test new() initializes empty URL lists."""
        progress = ScrapingProgress.new()
        assert progress.processed_urls == []
        assert progress.failed_urls == []

    def test_new_sets_total_resources(self):
        """Test new() sets total_resources."""
        progress = ScrapingProgress.new(total_resources=100)
        assert progress.total_resources == 100

    def test_new_default_total_resources(self):
        """Test new() defaults total_resources to 0."""
        progress = ScrapingProgress.new()
        assert progress.total_resources == 0

    def test_new_documents_saved_zero(self):
        """Test new() sets documents_saved to 0."""
        progress = ScrapingProgress.new()
        assert progress.documents_saved == 0

    def test_new_current_section_empty(self):
        """Test new() sets current_section to empty string."""
        progress = ScrapingProgress.new()
        assert progress.current_section == ""


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test checkpoint directory is created."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_init_sets_checkpoint_file(self, tmp_path):
        """Test checkpoint file path is set."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)
        assert manager.checkpoint_file == checkpoint_dir / "progress.json"

    def test_init_progress_is_none(self, tmp_path):
        """Test _progress is initially None."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)
        assert manager._progress is None


class TestLoad:
    """Tests for load method."""

    def test_load_no_file_returns_none(self, tmp_path):
        """Test None returned when no checkpoint exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        result = manager.load()
        assert result is None

    def test_load_success(self, tmp_path):
        """Test successful checkpoint loading."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "progress.json"

        # Create checkpoint file
        data = {
            "started_at": "2024-01-01T10:00:00",
            "last_updated": "2024-01-01T11:00:00",
            "total_resources": 50,
            "processed_urls": ["http://url1", "http://url2"],
            "failed_urls": ["http://url3"],
            "current_section": "Section 1",
            "documents_saved": 2,
        }
        checkpoint_file.write_text(json.dumps(data))

        manager = CheckpointManager(checkpoint_dir)
        result = manager.load()

        assert result is not None
        assert result.total_resources == 50
        assert len(result.processed_urls) == 2
        assert len(result.failed_urls) == 1

    def test_load_sets_internal_progress(self, tmp_path):
        """Test load sets _progress."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "progress.json"

        data = ScrapingProgress.new(10).__dict__
        data["started_at"] = str(data["started_at"])
        data["last_updated"] = str(data["last_updated"])
        checkpoint_file.write_text(json.dumps(data))

        manager = CheckpointManager(checkpoint_dir)
        manager.load()

        assert manager._progress is not None

    def test_load_invalid_json_returns_none(self, tmp_path):
        """Test None returned for invalid JSON."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "progress.json"
        checkpoint_file.write_text("invalid json {{{")

        manager = CheckpointManager(checkpoint_dir)
        result = manager.load()

        assert result is None


class TestSave:
    """Tests for save method."""

    def test_save_creates_file(self, tmp_path):
        """Test checkpoint file is created."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)
        manager.save(progress)

        assert manager.checkpoint_file.exists()

    def test_save_writes_json(self, tmp_path):
        """Test checkpoint file contains valid JSON."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)
        manager.save(progress)

        content = manager.checkpoint_file.read_text()
        data = json.loads(content)
        assert data["total_resources"] == 10

    def test_save_updates_last_updated(self, tmp_path):
        """Test last_updated is updated on save."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)
        old_updated = progress.last_updated

        # Wait a tiny bit
        import time
        time.sleep(0.01)

        manager.save(progress)

        # The progress object should have updated timestamp
        assert progress.last_updated != old_updated

    def test_save_sets_internal_progress(self, tmp_path):
        """Test save sets _progress."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)
        manager.save(progress)

        assert manager._progress is progress

    def test_save_cleans_up_temp_file_on_write_failure(self, tmp_path, mocker):
        """When write_text fails after the tmp file has been created,
        the save path must remove the tmp file and raise CheckpointError so
        the working tree never holds a half-written .json.tmp."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)

        tmp_path_ref = manager.checkpoint_file.with_suffix(".json.tmp")
        # Force tmp_path to exist (so the cleanup branch runs) and force the
        # replace step to fail so the except clause is entered.

        def failing_replace(src, dst):
            raise OSError("simulated replace failure")

        mocker.patch("keats_scraper.storage.checkpoint.os.replace", side_effect=failing_replace)

        with pytest.raises(CheckpointError):
            manager.save(progress)

        # The tmp file should not survive the error.
        assert not tmp_path_ref.exists()

    def test_save_raises_on_error(self, tmp_path, mocker):
        """Test CheckpointError on save failure."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(10)

        # Mock file write to fail
        mocker.patch.object(Path, "write_text", side_effect=IOError("Disk full"))

        with pytest.raises(CheckpointError):
            manager.save(progress)


class TestStartNew:
    """Tests for start_new method."""

    def test_start_new_creates_progress(self, tmp_path):
        """Test new progress is created."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = manager.start_new(total_resources=25)

        assert isinstance(progress, ScrapingProgress)
        assert progress.total_resources == 25

    def test_start_new_saves_immediately(self, tmp_path):
        """Test progress is saved after creation."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(total_resources=25)

        assert manager.checkpoint_file.exists()

    def test_start_new_returns_progress(self, tmp_path):
        """Test start_new returns the progress object."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        result = manager.start_new(total_resources=50)

        assert result is not None
        assert result.total_resources == 50


class TestMarkProcessed:
    """Tests for mark_processed method."""

    def test_mark_processed_adds_url(self, tmp_path):
        """Test URL is added to processed list."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/page1")

        assert "http://example.com/page1" in manager._progress.processed_urls

    def test_mark_processed_increments_count(self, tmp_path):
        """Test documents_saved is incremented."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/page1")

        assert manager._progress.documents_saved == 1

    def test_mark_processed_no_duplicates(self, tmp_path):
        """Test same URL isn't added twice."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/page1")
        manager.mark_processed("http://example.com/page1")

        assert manager._progress.processed_urls.count("http://example.com/page1") == 1
        assert manager._progress.documents_saved == 1

    def test_mark_processed_creates_progress_if_none(self, tmp_path):
        """Test progress is created if not exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.mark_processed("http://example.com/page1")

        assert manager._progress is not None
        assert "http://example.com/page1" in manager._progress.processed_urls

    def test_mark_processed_saves(self, tmp_path):
        """Test save is called after marking."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/page1")

        # Reload and verify
        manager2 = CheckpointManager(tmp_path / "checkpoints")
        loaded = manager2.load()
        assert "http://example.com/page1" in loaded.processed_urls

    def test_mark_processed_container_does_not_bump_documents_saved(self, tmp_path):
        """increment_documents=False marks the URL processed without
        touching the document counter. This is the path used for Moodle
        books and folders so their child chapters/files are not double-
        counted toward documents_saved."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/book/chapter1")
        manager.mark_processed(
            "http://example.com/book", increment_documents=False
        )

        assert "http://example.com/book" in manager._progress.processed_urls
        assert "http://example.com/book/chapter1" in manager._progress.processed_urls
        assert manager._progress.documents_saved == 1

    def test_mark_processed_container_argument_is_keyword_only(self, tmp_path):
        """increment_documents is keyword-only so it cannot be set
        positionally by accident. Calling it positionally must raise
        TypeError to prevent silent stat corruption."""
        import pytest

        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        with pytest.raises(TypeError):
            manager.mark_processed("http://example.com/book", False)  # type: ignore[misc]


class TestMarkFailed:
    """Tests for mark_failed method."""

    def test_mark_failed_adds_url(self, tmp_path):
        """Test URL is added to failed list."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_failed("http://example.com/page1")

        assert "http://example.com/page1" in manager._progress.failed_urls

    def test_mark_failed_no_duplicates(self, tmp_path):
        """Test same URL isn't added twice."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_failed("http://example.com/page1")
        manager.mark_failed("http://example.com/page1")

        assert manager._progress.failed_urls.count("http://example.com/page1") == 1

    def test_mark_failed_creates_progress_if_none(self, tmp_path):
        """Test progress is created if not exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.mark_failed("http://example.com/page1")

        assert manager._progress is not None
        assert "http://example.com/page1" in manager._progress.failed_urls

    def test_mark_failed_saves(self, tmp_path):
        """Test save is called after marking."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_failed("http://example.com/page1")

        # Reload and verify
        manager2 = CheckpointManager(tmp_path / "checkpoints")
        loaded = manager2.load()
        assert "http://example.com/page1" in loaded.failed_urls


class TestIsProcessed:
    """Tests for is_processed method."""

    def test_is_processed_true(self, tmp_path):
        """Test True for processed URL."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.mark_processed("http://example.com/page1")

        assert manager.is_processed("http://example.com/page1") is True

    def test_is_processed_false(self, tmp_path):
        """Test False for unprocessed URL."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)

        assert manager.is_processed("http://example.com/page1") is False

    def test_is_processed_no_progress(self, tmp_path):
        """Test False when no progress exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        assert manager.is_processed("http://example.com/page1") is False


class TestUpdateSection:
    """Tests for update_section method."""

    def test_update_section_sets_value(self, tmp_path):
        """Test section is updated."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.update_section("New Section")

        assert manager._progress.current_section == "New Section"

    def test_update_section_saves(self, tmp_path):
        """Test save is called after update."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(5)
        manager.update_section("New Section")

        # Reload and verify
        manager2 = CheckpointManager(tmp_path / "checkpoints")
        loaded = manager2.load()
        assert loaded.current_section == "New Section"

    def test_update_section_creates_progress_if_none(self, tmp_path):
        """Test progress is created if not exists."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.update_section("New Section")

        assert manager._progress is not None


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_no_session(self, tmp_path):
        """Test 'no session' status when no progress."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        stats = manager.get_stats()

        assert stats["status"] == "no session"

    def test_get_stats_with_progress(self, tmp_path):
        """Test all stats are returned."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(10)
        manager.mark_processed("http://url1")
        manager.mark_processed("http://url2")
        manager.mark_failed("http://url3")

        stats = manager.get_stats()

        assert "started_at" in stats
        assert "last_updated" in stats
        assert stats["total_resources"] == 10
        assert stats["processed"] == 2
        assert stats["failed"] == 1
        assert stats["documents_saved"] == 2

    def test_get_stats_remaining_calculation(self, tmp_path):
        """Test remaining calculation is correct."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(10)
        manager.mark_processed("http://url1")
        manager.mark_processed("http://url2")
        manager.mark_failed("http://url3")

        stats = manager.get_stats()

        # remaining = total - processed - failed = 10 - 2 - 1 = 7
        assert stats["remaining"] == 7


class TestClear:
    """Tests for clear method."""

    def test_clear_deletes_file(self, tmp_path):
        """Test checkpoint file is deleted."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(10)
        assert manager.checkpoint_file.exists()

        manager.clear()

        assert not manager.checkpoint_file.exists()

    def test_clear_resets_progress(self, tmp_path):
        """Test _progress is set to None."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(10)
        assert manager._progress is not None

        manager.clear()

        assert manager._progress is None

    def test_clear_no_file(self, tmp_path):
        """Test no error when file doesn't exist."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        # Should not raise
        manager.clear()

        assert manager._progress is None


class TestAtomicSave:
    """save() must be atomic: write to a temp file and os.replace."""

    def test_save_writes_via_tempfile(self, tmp_path):
        """The write should go to a .json.tmp first, then atomic rename."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(total_resources=5)

        manager.save(progress)

        # The final file exists, the tmp file does not (cleaned up by rename).
        assert manager.checkpoint_file.exists()
        assert not manager.checkpoint_file.with_suffix(".json.tmp").exists()

    def test_save_is_atomic_under_simulated_crash(self, tmp_path):
        """If the temp-file write raises, the original checkpoint is preserved
        and the orphaned tmp file is cleaned up."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        original = ScrapingProgress.new(total_resources=10)
        original.processed_urls = ["url-a", "url-b"]
        manager.save(original)
        original_bytes = manager.checkpoint_file.read_bytes()

        new_progress = ScrapingProgress.new(total_resources=10)
        new_progress.processed_urls = ["url-a", "url-b", "url-c"]

        # Patch Path.write_text to fail on the .tmp write.
        original_write_text = Path.write_text

        def raising_write_text(self, *args, **kwargs):
            if self.suffix == ".tmp":
                raise OSError("disk full simulated")
            return original_write_text(self, *args, **kwargs)

        with patch.object(Path, "write_text", raising_write_text):
            with pytest.raises(CheckpointError):
                manager.save(new_progress)

        # Original checkpoint is intact.
        assert manager.checkpoint_file.exists()
        assert manager.checkpoint_file.read_bytes() == original_bytes
        # Temp file has been cleaned up.
        assert not manager.checkpoint_file.with_suffix(".json.tmp").exists()

    def test_save_holds_lock(self, tmp_path):
        """save() acquires the manager's lock; concurrent saves do not corrupt."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        progress = ScrapingProgress.new(total_resources=400)
        manager.save(progress)
        assert isinstance(manager._lock, type(threading.Lock()))


class TestConcurrentMarkProcessed:
    """Concurrent mark_processed calls must go through the lock."""

    def test_concurrent_mark_processed_no_lost_updates(self, tmp_path):
        """8 threads x 50 marks each must produce 400 unique processed URLs."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(total_resources=400)

        urls_per_thread = 50
        n_threads = 8

        def worker(tid: int) -> None:
            for i in range(urls_per_thread):
                manager.mark_processed(f"https://example.test/t{tid}/{i}")

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 400 URLs are present and unique.
        processed = manager._progress.processed_urls
        assert len(processed) == n_threads * urls_per_thread
        assert len(set(processed)) == n_threads * urls_per_thread

    def test_concurrent_mark_processed_shared_urls_no_duplicates(self, tmp_path):
        """When the check-and-append runs outside the lock, two
        threads that hit the same URL at the same time can both see
        "url not in processed_urls" and both append it, producing
        duplicates and inflating the documents_saved counter.

        All threads mark the same small set of URLs (rather than
        unique URLs per thread) so the TOCTOU race is observable.
        """
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(total_resources=10)

        shared_urls = [f"https://example.test/shared/{i}" for i in range(10)]

        def worker() -> None:
            for url in shared_urls:
                manager.mark_processed(url)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        processed = manager._progress.processed_urls
        # Each URL appears exactly once, regardless of contention.
        assert sorted(processed) == sorted(shared_urls)
        assert manager._progress.documents_saved == len(shared_urls)

    def test_concurrent_mark_failed_shared_urls_no_duplicates(self, tmp_path):
        """Same regression for the parallel mark_failed path."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        manager.start_new(total_resources=5)

        shared_urls = [f"https://example.test/fail/{i}" for i in range(5)]

        def worker() -> None:
            for url in shared_urls:
                manager.mark_failed(url)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        failed = manager._progress.failed_urls
        assert sorted(failed) == sorted(shared_urls)
