"""Tests for CLI commands in main.py."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner
from main import (
    clear,
    cli,
    login,
    logout,
    process,
    scrape,
    setup_environment,
    status,
    validate,
)


class TestSetupEnvironment:
    """Tests for setup_environment function."""

    @patch("main.get_logger")
    @patch("main.setup_logging")
    @patch("main.config")
    def test_setup_environment_ensures_directories(
        self, mock_config, mock_setup_logging, mock_get_logger
    ):
        """Test ensure_directories is called."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = setup_environment()

        mock_config.ensure_directories.assert_called_once()

    @patch("main.get_logger")
    @patch("main.setup_logging")
    @patch("main.config")
    def test_setup_environment_sets_up_logging(
        self, mock_config, mock_setup_logging, mock_get_logger
    ):
        """Test logging is set up."""
        mock_config.log_level = "DEBUG"
        mock_config.log_file = Path("/tmp/test.log")
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        setup_environment()

        mock_setup_logging.assert_called_once_with(
            level="DEBUG", log_file=Path("/tmp/test.log")
        )

    @patch("main.get_logger")
    @patch("main.setup_logging")
    @patch("main.config")
    def test_setup_environment_returns_logger(
        self, mock_config, mock_setup_logging, mock_get_logger
    ):
        """Test logger is returned."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = setup_environment()

        assert result is mock_logger


class TestCLI:
    """Tests for CLI group."""

    def test_cli_group_exists(self):
        """Test CLI group is defined."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "KEATS Student Handbook Scraper" in result.output

    def test_cli_version_option(self):
        """Test version option works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestLoginCommand:
    """Tests for login command."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_login_with_valid_session(
        self, mock_config, mock_sso_class, mock_setup_env
    ):
        """Test login with existing valid session."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso

        # Mock valid cached session
        mock_sso.session_manager.load_cookies.return_value = [{"name": "session"}]
        mock_session = Mock()
        mock_sso.session_manager.create_session_with_cookies.return_value = mock_session
        mock_sso.session_manager.validate_session.return_value = True
        mock_config.auth.session_check_url = "https://keats.kcl.ac.uk/my/"

        runner = CliRunner()
        result = runner.invoke(login)

        assert result.exit_code == 0
        assert "still valid" in result.output

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_login_force_flag(self, mock_config, mock_sso_class, mock_setup_env):
        """Test --force flag triggers new login."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        runner = CliRunner()
        result = runner.invoke(login, ["--force"])

        mock_sso.get_valid_session.assert_called_once_with(force_login=True)

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_login_success(self, mock_config, mock_sso_class, mock_setup_env):
        """Test successful login output."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.session_manager.load_cookies.return_value = None
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        runner = CliRunner()
        result = runner.invoke(login)

        assert result.exit_code == 0
        assert "successful" in result.output

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_login_failure(self, mock_config, mock_sso_class, mock_setup_env):
        """Test login failure exits with code 1."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.session_manager.load_cookies.return_value = None
        mock_sso.get_valid_session.side_effect = Exception("Auth failed")

        runner = CliRunner()
        result = runner.invoke(login)

        assert result.exit_code == 1
        assert "failed" in result.output


class TestLogoutCommand:
    """Tests for logout command."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_logout_clears_session(
        self, mock_config, mock_sso_class, mock_setup_env
    ):
        """Test logout clears session."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso

        runner = CliRunner()
        result = runner.invoke(logout)

        mock_sso.logout.assert_called_once()
        assert result.exit_code == 0
        assert "cleared" in result.output


class TestScrapeCommand:
    """Tests for scrape command."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_auth_required(
        self,
        mock_config,
        mock_exporter,
        mock_pdf,
        mock_page,
        mock_nav,
        mock_limiter,
        mock_normalizer,
        mock_cleaner,
        mock_checkpoint,
        mock_sso_class,
        mock_setup_env,
    ):
        """Unexpected Exception during get_valid_session must still
        exit with code 1 and the friendly CLI message, and must call
        logger.exception so the traceback is captured in scraper.log."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.side_effect = RuntimeError(
            "ChromeDriver missing"
        )

        with patch("main.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(scrape)

            assert result.exit_code == 1
            assert "Authentication required" in result.output
            mock_logger.exception.assert_called_once_with(
                "Unexpected auth error in 'scrape' CLI"
            )
            mock_logger.warning.assert_not_called()

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_typed_auth_error_logs_warning(
        self,
        mock_config,
        mock_exporter,
        mock_pdf,
        mock_page,
        mock_nav,
        mock_limiter,
        mock_normalizer,
        mock_cleaner,
        mock_checkpoint,
        mock_sso_class,
        mock_setup_env,
    ):
        """Typed AuthenticationError / SessionExpiredError paths go
        through the friendly warning branch, not the full traceback branch.
        This separates "cookies went stale" from "ChromeDriver exploded"
        in scraper.log."""
        from keats_scraper.utils.exceptions import AuthenticationError

        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.side_effect = AuthenticationError(
            "Session expired"
        )

        with patch("main.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(scrape)

            assert result.exit_code == 1
            assert "Authentication required" in result.output
            mock_logger.warning.assert_called_once()
            args, _ = mock_logger.warning.call_args
            assert "Expected auth failure" in args[0]
            mock_logger.exception.assert_not_called()

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_discovers_resources(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape discovers resources."""
        # Setup mocks
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 0}

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = []

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_nav.discover_resources.assert_called_once()

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_resume_flag(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test --resume flag loads checkpoint."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_progress = Mock()
        mock_progress.processed_urls = ["url1", "url2"]
        mock_checkpoint.load.return_value = mock_progress
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 2, "failed": 0}

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = []

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape, ["--resume"])

        mock_checkpoint.load.assert_called_once()
        assert "Resuming from checkpoint" in result.output


class TestProcessCommand:
    """Tests for process command."""

    @patch("main.setup_environment")
    @patch("main.JSONLExporter")
    @patch("main.Chunker")
    @patch("main.config")
    def test_process_no_documents(
        self, mock_config, mock_chunker_class, mock_exporter_class, mock_setup_env
    ):
        """Test process with no documents."""
        mock_config.processed_dir = Path("/nonexistent")

        runner = CliRunner()
        result = runner.invoke(process)

        assert result.exit_code == 1
        assert "No documents found" in result.output

    @patch("main.setup_environment")
    @patch("main.JSONLExporter")
    @patch("main.Chunker")
    @patch("main.config")
    def test_process_chunks_documents(
        self, mock_config, mock_chunker_class, mock_exporter_class, mock_setup_env, tmp_path
    ):
        """Test process chunks documents."""
        # Create mock document file
        doc_file = tmp_path / "documents.jsonl"
        doc_file.write_text('{"id":"doc1","content":"test","metadata":{}}\n')

        mock_config.processed_dir = tmp_path
        mock_config.chunks_dir = tmp_path / "chunks"
        mock_config.chunk = Mock()

        # Mock document loading
        mock_doc = Mock()
        mock_exporter_class.load_documents.return_value = iter([mock_doc])

        # Mock chunker
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunk = Mock()
        mock_chunker.chunk_documents.return_value = [mock_chunk]

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_chunks.return_value = Path("/tmp/chunks.jsonl")
        mock_exporter.export_embedding_format.return_value = Path("/tmp/embed.jsonl")
        mock_exporter.create_index.return_value = Path("/tmp/index.json")

        runner = CliRunner()
        result = runner.invoke(process)

        assert result.exit_code == 0
        mock_chunker.chunk_documents.assert_called_once()


class TestStatusCommand:
    """Tests for status command."""

    @patch("main.setup_environment")
    @patch("main.CheckpointManager")
    @patch("main.config")
    def test_status_no_session(
        self, mock_config, mock_checkpoint_class, mock_setup_env
    ):
        """Test status with no scraping session."""
        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.get_stats.return_value = {"status": "no session"}
        mock_config.data_dir = Path("/tmp")

        runner = CliRunner()
        result = runner.invoke(status)

        assert result.exit_code == 0
        assert "No scraping session found" in result.output

    @patch("main.setup_environment")
    @patch("main.CheckpointManager")
    @patch("main.config")
    def test_status_shows_stats(
        self, mock_config, mock_checkpoint_class, mock_setup_env
    ):
        """Test status shows scraping stats."""
        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.get_stats.return_value = {
            "started_at": "2024-01-01T10:00:00",
            "last_updated": "2024-01-01T11:00:00",
            "total_resources": 100,
            "processed": 50,
            "failed": 5,
            "remaining": 45,
            "documents_saved": 48,
        }
        mock_config.data_dir = Path("/tmp")

        runner = CliRunner()
        result = runner.invoke(status)

        assert result.exit_code == 0
        assert "Processed" in result.output


class TestClearCommand:
    """Tests for clear command."""

    @patch("main.setup_environment")
    @patch("main.CheckpointManager")
    @patch("main.config")
    def test_clear_with_confirmation(
        self, mock_config, mock_checkpoint_class, mock_setup_env, tmp_path
    ):
        """Test clear with user confirmation."""
        # Use paths that don't exist so shutil.rmtree doesn't find them
        mock_config.raw_dir = tmp_path / "nonexistent_raw"
        mock_config.processed_dir = tmp_path / "nonexistent_processed"
        mock_config.chunks_dir = tmp_path / "nonexistent_chunks"
        mock_config.data_dir = tmp_path

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint

        runner = CliRunner()
        result = runner.invoke(clear, input="y\n")

        assert result.exit_code == 0
        assert "cleared" in result.output
        mock_checkpoint.clear.assert_called_once()

    @patch("main.setup_environment")
    @patch("main.CheckpointManager")
    @patch("main.config")
    def test_clear_cancelled(
        self, mock_config, mock_checkpoint_class, mock_setup_env, tmp_path
    ):
        """Test clear cancelled by user."""
        mock_config.data_dir = tmp_path

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint

        runner = CliRunner()
        result = runner.invoke(clear, input="n\n")

        # Should not clear if user says no
        mock_checkpoint.clear.assert_not_called()


class TestAllCommand:
    """Tests for all command (uses ctx.invoke to call scrape + process)."""

    @patch("main.process")
    @patch("main.scrape")
    @patch("main.setup_environment")
    def test_all_runs_scrape_and_process(
        self, mock_setup_env, mock_scrape, mock_process
    ):
        """Test all runs both scrape and process via ctx.invoke."""
        runner = CliRunner()
        result = runner.invoke(cli, ["all"])

        mock_scrape.assert_called_once()
        mock_process.assert_called_once()

    @patch("main.process")
    @patch("main.scrape")
    @patch("main.setup_environment")
    def test_all_scrape_failure_propagates(self, mock_setup_env, mock_scrape, mock_process):
        """Test all propagates scrape failure."""
        mock_scrape.side_effect = SystemExit(1)

        runner = CliRunner()
        result = runner.invoke(cli, ["all"])

        assert result.exit_code != 0


class TestScrapeResourceDiscoveryFailure:
    """Tests for resource discovery failure."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_resource_discovery_fails(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape fails when resource discovery raises exception."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.side_effect = Exception("Network error")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 1
        assert "Failed to discover resources" in result.output


class TestScrapeResourceProcessing:
    """Tests for resource processing loop."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_processes_page_resources(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape processes page resources correctly."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 1, "failed": 0}

        # Create a page resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/page"
        mock_resource.title = "Test Page"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        # Mock page scraper to return a document
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_doc = Mock()
        mock_doc.raw_html = None
        mock_doc.content = "Test content"
        mock_page.scrape_page.return_value = mock_doc

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_page.scrape_page.assert_called_once()
        mock_checkpoint.mark_processed.assert_called()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_processes_pdf_resources(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape processes PDF resources correctly."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 1, "failed": 0}

        # Create a PDF resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/file.pdf"
        mock_resource.title = "Test PDF"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "pdf"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        # Mock PDF handler to return a document
        mock_pdf = Mock()
        mock_pdf_class.return_value = mock_pdf
        mock_doc = Mock()
        mock_doc.raw_html = None
        mock_doc.content = "PDF content"
        mock_pdf.process_pdf.return_value = mock_doc

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized PDF content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_pdf.process_pdf.assert_called_once()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_processes_book_resources(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape processes book resources and their chapters."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 2, "failed": 0}

        # Create a book resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/book"
        mock_resource.title = "Test Book"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "book"

        # Create book chapters
        mock_chapter = Mock()
        mock_chapter.url = "https://example.com/book/chapter1"
        mock_chapter.title = "Chapter 1"
        mock_chapter.section = "Section 1"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_book_chapters.return_value = [mock_chapter]

        # Mock page scraper for chapters
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_chapter_doc = Mock()
        mock_chapter_doc.raw_html = None
        mock_chapter_doc.content = "Chapter content"
        mock_page.scrape_page.return_value = mock_chapter_doc

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_nav.discover_book_chapters.assert_called_once()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_processes_folder_resources(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape processes folder resources and their contents."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 2, "failed": 0}

        # Create a folder resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/folder"
        mock_resource.title = "Test Folder"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "folder"

        # Create folder file - PDF type
        mock_file = Mock()
        mock_file.url = "https://example.com/folder/file.pdf"
        mock_file.title = "Folder File"
        mock_file.section = "Section 1"
        mock_file.resource_type = "pdf"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_folder_contents.return_value = [mock_file]

        # Mock PDF handler for folder files
        mock_pdf = Mock()
        mock_pdf_class.return_value = mock_pdf
        mock_file_doc = Mock()
        mock_file_doc.raw_html = None
        mock_file_doc.content = "File content"
        mock_pdf.process_pdf.return_value = mock_file_doc

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_nav.discover_folder_contents.assert_called_once()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_folder_with_non_pdf_file(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape processes folder with non-PDF files using page scraper."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 1, "failed": 0}

        # Create a folder resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/folder"
        mock_resource.title = "Test Folder"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "folder"

        # Create folder file - page type (not PDF)
        mock_file = Mock()
        mock_file.url = "https://example.com/folder/page"
        mock_file.title = "Folder Page"
        mock_file.section = "Section 1"
        mock_file.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_folder_contents.return_value = [mock_file]

        # Mock page scraper for folder pages
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_file_doc = Mock()
        mock_file_doc.raw_html = None
        mock_file_doc.content = "Page content"
        mock_page.scrape_page.return_value = mock_file_doc

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_page.scrape_page.assert_called()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_book_marks_parent_processed_without_bumping_documents_saved(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Successfully scraped books mark the parent book URL processed
        with increment_documents=False so --resume skips re-discovery
        but documents_saved remains an accurate count of chapters."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 2, "failed": 0}

        mock_resource = Mock()
        mock_resource.url = "https://example.com/book"
        mock_resource.title = "Test Book"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "book"

        mock_chapter = Mock()
        mock_chapter.url = "https://example.com/book/chapter1"
        mock_chapter.title = "Chapter 1"
        mock_chapter.section = "Section 1"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_book_chapters.return_value = [mock_chapter]

        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_chapter_doc = Mock()
        mock_chapter_doc.raw_html = None
        mock_chapter_doc.content = "Chapter content"
        mock_page.scrape_page.return_value = mock_chapter_doc

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_processed.assert_any_call("https://example.com/book/chapter1")
        mock_checkpoint.mark_processed.assert_any_call(
            "https://example.com/book", increment_documents=False
        )

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_book_with_no_chapters_marks_parent_failed(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """A book whose discovery yields no chapters must be marked failed
        so the operator sees it in the failed_urls list rather than lost."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        mock_resource = Mock()
        mock_resource.url = "https://example.com/empty-book"
        mock_resource.title = "Empty Book"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "book"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_book_chapters.return_value = []

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/empty-book")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_folder_marks_parent_processed_without_bumping_documents_saved(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Successfully scraped folders mark the parent folder URL processed
        with increment_documents=False for the same reason as books."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 2, "failed": 0}

        mock_resource = Mock()
        mock_resource.url = "https://example.com/folder"
        mock_resource.title = "Test Folder"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "folder"

        mock_file = Mock()
        mock_file.url = "https://example.com/folder/file.pdf"
        mock_file.title = "Folder File"
        mock_file.section = "Section 1"
        mock_file.resource_type = "pdf"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_folder_contents.return_value = [mock_file]

        mock_pdf = Mock()
        mock_pdf_class.return_value = mock_pdf
        mock_file_doc = Mock()
        mock_file_doc.raw_html = None
        mock_file_doc.content = "File content"
        mock_pdf.process_pdf.return_value = mock_file_doc

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_processed.assert_any_call("https://example.com/folder/file.pdf")
        mock_checkpoint.mark_processed.assert_any_call(
            "https://example.com/folder", increment_documents=False
        )

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_folder_with_no_files_marks_parent_failed(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """A folder whose discovery yields no files must be marked failed."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        mock_resource = Mock()
        mock_resource.url = "https://example.com/empty-folder"
        mock_resource.title = "Empty Folder"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "folder"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]
        mock_nav.discover_folder_contents.return_value = []

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/empty-folder")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_cleans_raw_html(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape cleans raw HTML content."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 1, "failed": 0}

        # Create a page resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/page"
        mock_resource.title = "Test Page"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        # Mock page scraper to return a document WITH raw_html
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_doc = Mock()
        mock_doc.raw_html = "<html><body>Test</body></html>"
        mock_doc.content = ""
        mock_page.scrape_page.return_value = mock_doc

        # Mock cleaner
        mock_cleaner = Mock()
        mock_cleaner_class.return_value = mock_cleaner
        mock_cleaner.clean.return_value = "Cleaned content"

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = "Normalized content"

        # Mock exporter
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_cleaner.clean.assert_called_once_with("<html><body>Test</body></html>")
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_marks_failed_on_null_doc(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape marks resource as failed when no document returned."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        # Create a page resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/page"
        mock_resource.title = "Test Page"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        # Mock page scraper to return None
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.return_value = None

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_checkpoint.mark_failed.assert_called_with("https://example.com/page")
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_handles_exception_during_processing(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape handles exception during resource processing."""
        mock_logger = Mock()
        mock_setup_env.return_value = mock_logger

        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        # Create a page resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/page"
        mock_resource.title = "Test Page"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        # Mock page scraper to raise exception
        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.side_effect = Exception("Network error")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        mock_checkpoint.mark_failed.assert_called()
        assert result.exit_code == 0

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_scrape_skips_already_processed(
        self,
        mock_config,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        mock_setup_env,
    ):
        """Test scrape skips already processed resources."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_session = Mock()
        mock_sso.get_valid_session.return_value = mock_session

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        # All resources are already processed
        mock_checkpoint.is_processed.return_value = True
        mock_checkpoint.get_stats.return_value = {"processed": 1, "failed": 0}

        # Create a page resource
        mock_resource = Mock()
        mock_resource.url = "https://example.com/page"
        mock_resource.title = "Test Page"
        mock_resource.section = "Section 1"
        mock_resource.resource_type = "page"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [mock_resource]

        mock_page = Mock()
        mock_page_class.return_value = mock_page

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        # Should not call scrape_page since resource is already processed
        mock_page.scrape_page.assert_not_called()
        assert result.exit_code == 0


class TestScrapeResourceTypeBranches:
    """Covers the dispatch branches in scrape for resource types that
    the earlier happy-path tests don't exercise: forum, glossary, url
    (external, skipped), plus the 'empty content after cleaning' paths for
    pages, book chapters and folder files.
    """

    def _patches(self, func):
        # Apply the same decorator stack the other tests use, in one place.
        for decorator in reversed([
            patch("main.setup_environment"),
            patch("main.SSOHandler"),
            patch("main.CheckpointManager"),
            patch("main.HTMLCleaner"),
            patch("main.TextNormalizer"),
            patch("main.RateLimiter"),
            patch("main.CourseNavigator"),
            patch("main.PageScraper"),
            patch("main.PDFHandler"),
            patch("main.JSONLExporter"),
            patch("main.ContentValidator"),
            patch("main.config"),
        ]):
            func = decorator(func)
        return func

    def _mk_resource(self, url, title, section, resource_type):
        r = Mock()
        r.url = url
        r.title = title
        r.section = section
        r.resource_type = resource_type
        return r

    def _run_scrape(
        self,
        mock_config,
        mock_validator_class,
        mock_exporter_class,
        mock_pdf_class,
        mock_page_class,
        mock_nav_class,
        mock_limiter_class,
        mock_normalizer_class,
        mock_cleaner_class,
        mock_checkpoint_class,
        mock_sso_class,
        resources,
        page_doc=None,
        pdf_doc=None,
        normalize_returns="normalized content",
    ):
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 0}

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = resources
        mock_nav.discover_book_chapters.return_value = []
        mock_nav.discover_folder_contents.return_value = []

        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.return_value = page_doc

        mock_pdf = Mock()
        mock_pdf_class.return_value = mock_pdf
        mock_pdf.process_pdf.return_value = pdf_doc

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = normalize_returns

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        return runner.invoke(scrape), mock_checkpoint, mock_page, mock_pdf, mock_nav

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_forum_resource_routes_to_scrape_page(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        doc = Mock(raw_html=None, content="forum thread")
        result, _, mock_page, _, _ = self._run_scrape(
            mock_config, mock_validator_class, mock_exporter_class,
            mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
            mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
            mock_sso_class,
            resources=[self._mk_resource("https://example.com/forum", "F", "S", "forum")],
            page_doc=doc,
        )
        assert result.exit_code == 0
        mock_page.scrape_page.assert_called_once()

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_unknown_resource_type_falls_through_to_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Resource types not matched by any earlier elif (e.g. quiz
        or assign) must fall through the dispatch chain and be marked
        failed by the catch-all at the end. Exercises the False-branch of
        elif resource.resource_type == "folder": along with the
        resource_type not in ("book", "folder") fallback."""
        result, mock_checkpoint, mock_page, mock_pdf, _ = self._run_scrape(
            mock_config, mock_validator_class, mock_exporter_class,
            mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
            mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
            mock_sso_class,
            resources=[self._mk_resource("https://example.com/quiz", "Q", "S", "quiz")],
        )
        assert result.exit_code == 0
        mock_page.scrape_page.assert_not_called()
        mock_pdf.process_pdf.assert_not_called()
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/quiz")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_glossary_resource_routes_to_scrape_page(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        doc = Mock(raw_html=None, content="glossary entry")
        result, _, mock_page, _, _ = self._run_scrape(
            mock_config, mock_validator_class, mock_exporter_class,
            mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
            mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
            mock_sso_class,
            resources=[self._mk_resource("https://example.com/g", "G", "S", "glossary")],
            page_doc=doc,
        )
        assert result.exit_code == 0
        mock_page.scrape_page.assert_called_once()

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_url_resource_is_skipped_and_marked_processed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        result, mock_checkpoint, mock_page, mock_pdf, _ = self._run_scrape(
            mock_config, mock_validator_class, mock_exporter_class,
            mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
            mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
            mock_sso_class,
            resources=[self._mk_resource("https://external.example/x", "External", "S", "url")],
        )
        assert result.exit_code == 0
        mock_page.scrape_page.assert_not_called()
        mock_pdf.process_pdf.assert_not_called()
        mock_checkpoint.mark_processed.assert_called_with(
            "https://external.example/x", increment_documents=False
        )

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_page_with_empty_content_after_cleaning_marks_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Empty normalised content on the main resource path exercises the
        clean_document returns-None branch where the resource URL is
        marked failed rather than appended to the documents list."""
        doc = Mock(raw_html="<p></p>", content="")
        result, mock_checkpoint, _, _, _ = self._run_scrape(
            mock_config, mock_validator_class, mock_exporter_class,
            mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
            mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
            mock_sso_class,
            resources=[self._mk_resource("https://example.com/p", "P", "S", "page")],
            page_doc=doc,
            normalize_returns="",
        )
        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_called_with("https://example.com/p")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_book_chapter_with_empty_content_marks_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Covers the book-chapter branch where discover_book_chapters
        returns chapters and cleaning yields empty content, so the chapter
        URL is marked failed."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 2}

        chapter = Mock()
        chapter.url = "https://example.com/book/ch1"
        chapter.title = "Ch1"
        chapter.resource_type = "book_chapter"
        chapter.section = "S"
        chapter_doc = Mock(raw_html="<p></p>", content="")

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [
            self._mk_resource("https://example.com/book", "Book", "S", "book")
        ]
        mock_nav.discover_book_chapters.return_value = [chapter]

        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.return_value = chapter_doc

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = ""

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/book/ch1")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_book_chapter_scrape_returns_none_marks_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Covers the book-chapter branch where scrape_page returns
        None (network or HTTP failure) so the chapter URL is marked
        failed without being appended to the documents list."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        chapter = Mock()
        chapter.url = "https://example.com/book/ch2"
        chapter.title = "Ch2"
        chapter.resource_type = "book_chapter"
        chapter.section = "S"

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [
            self._mk_resource("https://example.com/book", "Book", "S", "book")
        ]
        mock_nav.discover_book_chapters.return_value = [chapter]

        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.return_value = None

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = ""

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/book/ch2")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_folder_file_with_empty_content_marks_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Covers the folder branch where a PDF file yields empty content
        after cleaning, so the file URL is marked failed while the parent
        folder marking still runs."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        file_info = Mock()
        file_info.url = "https://example.com/folder/a.pdf"
        file_info.title = "A"
        file_info.resource_type = "pdf"
        file_doc = Mock(raw_html=None, content="")

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [
            self._mk_resource("https://example.com/folder", "Folder", "S", "folder")
        ]
        mock_nav.discover_folder_contents.return_value = [file_info]

        mock_pdf = Mock()
        mock_pdf_class.return_value = mock_pdf
        mock_pdf.process_pdf.return_value = file_doc

        mock_normalizer = Mock()
        mock_normalizer_class.return_value = mock_normalizer
        mock_normalizer.normalize.return_value = ""

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/folder/a.pdf")

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.CheckpointManager")
    @patch("main.HTMLCleaner")
    @patch("main.TextNormalizer")
    @patch("main.RateLimiter")
    @patch("main.CourseNavigator")
    @patch("main.PageScraper")
    @patch("main.PDFHandler")
    @patch("main.JSONLExporter")
    @patch("main.ContentValidator")
    @patch("main.config")
    def test_folder_non_pdf_file_with_none_doc_marks_failed(
        self, mock_config, mock_validator_class, mock_exporter_class,
        mock_pdf_class, mock_page_class, mock_nav_class, mock_limiter_class,
        mock_normalizer_class, mock_cleaner_class, mock_checkpoint_class,
        mock_sso_class, mock_setup_env,
    ):
        """Folder branch with a non-PDF file where scrape_page returns None
        (exercises the inner else of the file_doc check)."""
        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.get_valid_session.return_value = Mock()

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint
        mock_checkpoint.load.return_value = None
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.get_stats.return_value = {"processed": 0, "failed": 1}

        file_info = Mock()
        file_info.url = "https://example.com/folder/a.html"
        file_info.title = "A"
        file_info.resource_type = "resource"  # triggers else branch (scrape_page)

        mock_nav = Mock()
        mock_nav_class.return_value = mock_nav
        mock_nav.discover_resources.return_value = [
            self._mk_resource("https://example.com/folder", "Folder", "S", "folder")
        ]
        mock_nav.discover_folder_contents.return_value = [file_info]

        mock_page = Mock()
        mock_page_class.return_value = mock_page
        mock_page.scrape_page.return_value = None

        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_documents.return_value = Path("/tmp/documents.jsonl")

        mock_config.data_dir = Path("/tmp")
        mock_config.processed_dir = Path("/tmp/processed")

        runner = CliRunner()
        result = runner.invoke(scrape)

        assert result.exit_code == 0
        mock_checkpoint.mark_failed.assert_any_call("https://example.com/folder/a.html")


class TestLoginSessionExpiredError:
    """Covers the typed SessionExpiredError branch on the login
    command when the existing cached session is already invalid (lines
    109-110 of main.py)."""

    @patch("main.setup_environment")
    @patch("main.SSOHandler")
    @patch("main.config")
    def test_login_falls_through_when_cached_session_is_expired(
        self, mock_config, mock_sso_class, mock_setup_env,
    ):
        from keats_scraper.utils.exceptions import SessionExpiredError

        mock_sso = Mock()
        mock_sso_class.return_value = mock_sso
        mock_sso.session_manager.load_cookies.return_value = [{"name": "old"}]
        mock_sso.session_manager.create_session_with_cookies.return_value = Mock()
        mock_sso.session_manager.validate_session.side_effect = SessionExpiredError(
            "expired"
        )
        mock_sso.get_valid_session.return_value = Mock()
        mock_config.auth.session_check_url = "https://keats.kcl.ac.uk/my/"

        runner = CliRunner()
        result = runner.invoke(login)

        # login falls through to a fresh-login path and reports success.
        assert result.exit_code == 0
        assert "successful" in result.output


class TestClearWithExistingDirectories:
    """Tests for clear command with existing directories."""

    @patch("main.setup_environment")
    @patch("main.CheckpointManager")
    @patch("main.config")
    def test_clear_deletes_existing_directories(
        self, mock_config, mock_checkpoint_class, mock_setup_env, tmp_path
    ):
        """Test clear deletes existing directories."""
        # Create actual directories
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        chunks_dir = tmp_path / "chunks"
        raw_dir.mkdir()
        processed_dir.mkdir()
        chunks_dir.mkdir()

        # Create a file in one directory to verify deletion
        (raw_dir / "test.txt").write_text("test")

        mock_config.raw_dir = raw_dir
        mock_config.processed_dir = processed_dir
        mock_config.chunks_dir = chunks_dir
        mock_config.data_dir = tmp_path

        mock_checkpoint = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint

        runner = CliRunner()
        result = runner.invoke(clear, input="y\n")

        assert result.exit_code == 0
        assert "cleared" in result.output
        # Directories should be recreated (empty)
        assert raw_dir.exists()
        assert processed_dir.exists()
        assert chunks_dir.exists()
        # But test file should be gone
        assert not (raw_dir / "test.txt").exists()


class TestAllProcessFailure:
    """Tests for all command when process fails."""

    @patch("main.process")
    @patch("main.scrape")
    @patch("main.setup_environment")
    def test_all_process_failure_propagates(self, mock_setup_env, mock_scrape, mock_process):
        """Test all propagates process failure."""
        mock_process.side_effect = SystemExit(1)

        runner = CliRunner()
        result = runner.invoke(cli, ["all"])

        assert result.exit_code != 0


class TestValidateCommand:
    """Tests for the validate CLI subcommand."""

    @patch("main.setup_environment")
    @patch("main.config")
    def test_validate_exits_non_zero_when_documents_missing(
        self, mock_config, mock_setup_env, tmp_path
    ):
        """No documents.jsonl on disk -> non-zero exit with a clear message.
        Guards against running the validator on an empty data directory."""
        mock_config.processed_dir = tmp_path
        mock_config.log_level = "INFO"
        mock_config.log_file = tmp_path / "scraper.log"

        runner = CliRunner()
        result = runner.invoke(validate)

        assert result.exit_code == 1
        assert "No documents found" in result.output

    @patch("main.setup_environment")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_validate_exits_non_zero_on_empty_jsonl(
        self, mock_config, mock_exporter_class, mock_setup_env, tmp_path
    ):
        """documents.jsonl present but empty -> non-zero exit. This is
        distinct from the 'file missing' case so operators can tell which
        problem they are dealing with."""
        documents_file = tmp_path / "documents.jsonl"
        documents_file.touch()
        mock_config.processed_dir = tmp_path
        mock_config.log_level = "INFO"
        mock_config.log_file = tmp_path / "scraper.log"

        mock_exporter_class.load_documents.return_value = iter([])

        runner = CliRunner()
        result = runner.invoke(validate)

        assert result.exit_code == 1
        assert "No documents to validate" in result.output

    @patch("main.setup_environment")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_validate_exits_non_zero_on_read_error(
        self, mock_config, mock_exporter_class, mock_setup_env, tmp_path
    ):
        """Malformed JSONL -> non-zero exit. load_documents raising any
        exception must not leave the CLI in a success state."""
        documents_file = tmp_path / "documents.jsonl"
        documents_file.write_text("not valid json\n")
        mock_config.processed_dir = tmp_path
        mock_config.log_level = "INFO"
        mock_config.log_file = tmp_path / "scraper.log"

        mock_exporter_class.load_documents.side_effect = ValueError("boom")

        runner = CliRunner()
        result = runner.invoke(validate)

        assert result.exit_code == 1
        assert "Failed to read" in result.output

    @patch("main.setup_environment")
    @patch("main.ContentValidator")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_validate_prints_quality_report(
        self,
        mock_config,
        mock_exporter_class,
        mock_validator_class,
        mock_setup_env,
        tmp_path,
    ):
        """Happy path: documents present, validator called, summary/section
        tables printed. Asserts the command wires the existing
        generate_quality_report to the Rich renderer."""
        documents_file = tmp_path / "documents.jsonl"
        documents_file.write_text('{"id": "a"}\n')
        mock_config.processed_dir = tmp_path
        mock_config.log_level = "INFO"
        mock_config.log_file = tmp_path / "scraper.log"

        fake_doc = Mock()
        mock_exporter_class.load_documents.return_value = iter([fake_doc])

        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.generate_quality_report.return_value = {
            "total_documents": 1,
            "valid_documents": 1,
            "invalid_documents": 0,
            "empty_documents": 0,
            "total_words": 120,
            "avg_words_per_doc": 120.0,
            "min_words": 120,
            "max_words": 120,
            "sections_covered": ["Main"],
            "section_count": 1,
            "documents_with_issues": {},
        }

        runner = CliRunner()
        result = runner.invoke(validate)

        assert result.exit_code == 0
        mock_validator.generate_quality_report.assert_called_once_with([fake_doc])
        assert "Content Quality Report" in result.output
        assert "Main" in result.output
        assert "No document-level issues reported" in result.output

    @patch("main.setup_environment")
    @patch("main.ContentValidator")
    @patch("main.JSONLExporter")
    @patch("main.config")
    def test_validate_prints_issues_table_when_issues_exist(
        self,
        mock_config,
        mock_exporter_class,
        mock_validator_class,
        mock_setup_env,
        tmp_path,
    ):
        """When documents_with_issues is non-empty the command emits a
        dedicated issues table listing the flagged documents."""
        documents_file = tmp_path / "documents.jsonl"
        documents_file.write_text('{"id": "a"}\n')
        mock_config.processed_dir = tmp_path
        mock_config.log_level = "INFO"
        mock_config.log_file = tmp_path / "scraper.log"

        fake_doc = Mock()
        mock_exporter_class.load_documents.return_value = iter([fake_doc])

        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.generate_quality_report.return_value = {
            "total_documents": 1,
            "valid_documents": 0,
            "invalid_documents": 1,
            "empty_documents": 1,
            "total_words": 0,
            "avg_words_per_doc": 0.0,
            "min_words": 0,
            "max_words": 0,
            "sections_covered": [],
            "section_count": 0,
            "documents_with_issues": {
                "docid-123": {"title": "", "issues": ["Empty content"]}
            },
        }

        runner = CliRunner()
        result = runner.invoke(validate)

        assert result.exit_code == 0
        assert "Documents with issues" in result.output
        assert "docid-123" in result.output
        assert "(untitled)" in result.output
        assert "Empty content" in result.output


class TestIntegration:
    """Integration tests for CLI."""

    def test_help_for_all_commands(self):
        """Test help is available for all commands."""
        runner = CliRunner()

        commands = [
            "login",
            "logout",
            "scrape",
            "process",
            "all",
            "status",
            "validate",
            "clear",
        ]
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"Help failed for {cmd}"

    def test_command_discovery(self):
        """Test all commands are discoverable."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "login" in result.output
        assert "logout" in result.output
        assert "scrape" in result.output
        assert "process" in result.output
        assert "all" in result.output
        assert "status" in result.output
        assert "validate" in result.output
        assert "clear" in result.output
