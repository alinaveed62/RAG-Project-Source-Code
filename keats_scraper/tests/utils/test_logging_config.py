"""Tests for logging configuration."""

import logging

import pytest

from keats_scraper.utils.logging_config import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging(name="test_logger_1")
        assert isinstance(logger, logging.Logger)

    def test_sets_correct_name(self):
        """Test that logger has correct name."""
        logger = setup_logging(name="test_logger_2")
        assert logger.name == "test_logger_2"

    def test_default_name(self):
        """Test default logger name is keats_scraper."""
        logger = setup_logging()
        assert logger.name == "keats_scraper"

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_sets_log_level(self, level):
        """Test various log levels are set correctly."""
        logger = setup_logging(level=level, name=f"test_level_{level}")
        assert logger.level == getattr(logging, level)

    def test_invalid_log_level_defaults_to_info(self):
        """Test invalid log level defaults to INFO."""
        logger = setup_logging(level="INVALID", name="test_invalid_level")
        assert logger.level == logging.INFO

    def test_lowercase_log_level(self):
        """Test lowercase log level is handled."""
        logger = setup_logging(level="debug", name="test_lowercase")
        assert logger.level == logging.DEBUG

    def test_console_handler_added(self):
        """Test console handler is added to logger."""
        logger = setup_logging(name="test_console_handler")

        # Find StreamHandler
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_file_handler_added_when_log_file_specified(self, tmp_path):
        """Test file handler is added when log_file is specified."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file, name="test_file_handler")

        # Find FileHandler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

        # Cleanup
        for handler in file_handlers:
            handler.close()

    def test_no_file_handler_without_log_file(self):
        """Test no file handler when log_file is not specified."""
        logger = setup_logging(log_file=None, name="test_no_file_handler")

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_clears_existing_handlers(self):
        """Test existing handlers are cleared on subsequent calls."""
        logger = setup_logging(name="test_clear_handlers")
        initial_count = len(logger.handlers)

        # Call again
        logger = setup_logging(name="test_clear_handlers")

        # Should have same number of handlers, not doubled
        assert len(logger.handlers) == initial_count

    def test_writes_to_log_file(self, tmp_path):
        """Test that log messages are written to file."""
        log_file = tmp_path / "test_write.log"
        logger = setup_logging(level="DEBUG", log_file=log_file, name="test_file_write")

        test_message = "Test log message for file"
        logger.info(test_message)

        # Close handlers to flush
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

        # Check file contents
        content = log_file.read_text()
        assert test_message in content

    def test_console_format_includes_time(self):
        """Test console format includes time."""
        logger = setup_logging(name="test_console_format")

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]

        assert len(stream_handlers) > 0
        formatter = stream_handlers[0].formatter
        assert formatter is not None
        assert "asctime" in formatter._fmt

    def test_file_format_includes_name(self, tmp_path):
        """Test file format includes logger name."""
        log_file = tmp_path / "test_format.log"
        logger = setup_logging(log_file=log_file, name="test_file_format")

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

        assert len(file_handlers) > 0
        formatter = file_handlers[0].formatter
        assert formatter is not None
        assert "name" in formatter._fmt

        # Cleanup
        for handler in file_handlers:
            handler.close()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Test get_logger returns a Logger instance named after the argument."""
        logger = get_logger("test_get_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger"

    def test_returns_existing_logger(self):
        """Test get_logger returns existing logger if already created."""
        # Create logger first
        setup_logging(name="test_existing")

        # Get it
        logger = get_logger("test_existing")
        assert logger.name == "test_existing"

    def test_default_name(self):
        """Test default name is keats_scraper."""
        logger = get_logger()
        assert logger.name == "keats_scraper"

    def test_custom_name(self):
        """Test custom logger name."""
        logger = get_logger("custom_logger_name")
        assert logger.name == "custom_logger_name"

    def test_returns_unconfigured_logger_if_not_setup(self):
        """Test returns logger even if not previously configured."""
        # This should not raise
        logger = get_logger("never_configured_logger")
        assert isinstance(logger, logging.Logger)
