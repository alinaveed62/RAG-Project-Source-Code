"""Checkpoint management for resumable scraping."""

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keats_scraper.utils.exceptions import CheckpointError
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


@dataclass
class ScrapingProgress:
    """Tracks scraping progress for resumption."""

    started_at: str
    last_updated: str
    total_resources: int
    processed_urls: list
    failed_urls: list
    current_section: str
    documents_saved: int

    @classmethod
    def new(cls, total_resources: int = 0) -> "ScrapingProgress":
        """Create new progress tracker."""
        now = datetime.now(UTC).isoformat()
        return cls(
            started_at=now,
            last_updated=now,
            total_resources=total_resources,
            processed_urls=[],
            failed_urls=[],
            current_section="",
            documents_saved=0,
        )


class CheckpointManager:
    """Manages scraping checkpoints for resumption."""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / "progress.json"
        self._progress: ScrapingProgress | None = None
        self._lock = threading.Lock()

    def load(self) -> ScrapingProgress | None:
        """
        Load existing checkpoint.

        Returns:
            ScrapingProgress or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            logger.debug("No checkpoint found")
            return None

        try:
            data = json.loads(self.checkpoint_file.read_text())
            self._progress = ScrapingProgress(**data)
            logger.info(
                f"Loaded checkpoint: {len(self._progress.processed_urls)} URLs processed"
            )
            return self._progress

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def save(self, progress: ScrapingProgress) -> None:
        """
        Save checkpoint atomically.

        Writes to a temporary file in the same directory, then renames
        atomically via os.replace. This guarantees that a reader
        always sees either the previous complete checkpoint or the new
        complete checkpoint, even if the process crashes mid-write.

        Args:
            progress: Current scraping progress
        """
        with self._lock:
            self._write_to_disk(progress)

    def _write_to_disk(self, progress: ScrapingProgress) -> None:
        """Atomic write helper. Caller must already hold self._lock.

        Splitting this out lets mark_processed and mark_failed
        perform their check-append-write sequence inside a single lock
        acquisition, closing the TOCTOU window without recursive locking.
        """
        tmp_path = self.checkpoint_file.with_suffix(".json.tmp")
        try:
            progress.last_updated = datetime.now(UTC).isoformat()
            self._progress = progress

            data = asdict(progress)
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, self.checkpoint_file)
            logger.debug(f"Checkpoint saved: {len(progress.processed_urls)} URLs")

        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover
                # Only reachable if unlink itself fails during cleanup
                # (e.g. the filesystem became read-only between write and
                # cleanup). We swallow so the original CheckpointError is
                # what surfaces to the caller.
                pass
            logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointError(f"Checkpoint save failed: {e}")

    def start_new(self, total_resources: int) -> ScrapingProgress:
        """
        Start a new scraping session.

        Args:
            total_resources: Total number of resources to process

        Returns:
            New ScrapingProgress
        """
        progress = ScrapingProgress.new(total_resources)
        self.save(progress)
        return progress

    def mark_processed(self, url: str, *, increment_documents: bool = True) -> None:
        """
        Mark a URL as successfully processed.

        The check-and-append must run under the same lock as save() to
        avoid a TOCTOU race: two workers can both observe url not in
        processed_urls and both append the same URL, producing duplicates
        that inflate documents_saved. save() re-acquires the lock,
        which is re-entrant in effect because the work done inside is short
        and we use a plain Lock consistently at one level of nesting --
        so mirror the disk write here (not via self.save) to keep the
        critical section flat and avoid deadlock.

        Args:
            url: Processed URL
            increment_documents: When True (default) the documents_saved
                counter is bumped because this URL produced a document. Pass
                False for container resources such as Moodle books and
                folders, whose child chapters/files are already counted
                individually via their own mark_processed calls. Marking
                the container as processed is still useful so a later
                --resume run can skip re-discovery of its children.
        """
        with self._lock:
            if self._progress is None:
                self._progress = ScrapingProgress.new()

            if url in self._progress.processed_urls:
                return

            self._progress.processed_urls.append(url)
            if increment_documents:
                self._progress.documents_saved += 1
            self._write_to_disk(self._progress)

    def mark_failed(self, url: str) -> None:
        """
        Mark a URL as failed.

        Same lock discipline as mark_processed.

        Args:
            url: Failed URL
        """
        with self._lock:
            if self._progress is None:
                self._progress = ScrapingProgress.new()

            if url in self._progress.failed_urls:
                return

            self._progress.failed_urls.append(url)
            self._write_to_disk(self._progress)

    def is_processed(self, url: str) -> bool:
        """
        Check if URL was already processed.

        Args:
            url: URL to check

        Returns:
            True if already processed
        """
        if self._progress is None:
            return False
        return url in self._progress.processed_urls

    def update_section(self, section: str) -> None:
        """
        Update current section being processed.

        Args:
            section: Section name
        """
        if self._progress is None:
            self._progress = ScrapingProgress.new()

        self._progress.current_section = section
        self.save(self._progress)

    def get_stats(self) -> dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary of stats
        """
        if self._progress is None:
            return {"status": "no session"}

        return {
            "started_at": self._progress.started_at,
            "last_updated": self._progress.last_updated,
            "total_resources": self._progress.total_resources,
            "processed": len(self._progress.processed_urls),
            "failed": len(self._progress.failed_urls),
            "remaining": self._progress.total_resources
            - len(self._progress.processed_urls)
            - len(self._progress.failed_urls),
            "documents_saved": self._progress.documents_saved,
        }

    def clear(self) -> None:
        """Clear checkpoint data."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._progress = None
        logger.info("Checkpoint cleared")
