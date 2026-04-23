"""Tests for PageScraper."""

from unittest.mock import Mock, patch

import pytest
import requests

from keats_scraper.models.document import Document
from keats_scraper.scraper.page_scraper import PageScraper
from keats_scraper.scraper.rate_limiter import RateLimiter
from keats_scraper.utils.exceptions import ContentExtractionError


class TestPageScraperInit:
    """Tests for PageScraper initialization."""

    def test_init_sets_session(self):
        """Test session is set correctly."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        scraper = PageScraper(mock_session, mock_limiter)
        assert scraper.session is mock_session

    def test_init_sets_rate_limiter(self):
        """Test rate_limiter is set correctly."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        scraper = PageScraper(mock_session, mock_limiter)
        assert scraper.rate_limiter is mock_limiter

    def test_content_selectors_defined(self):
        """Test CONTENT_SELECTORS class attribute exists."""
        assert hasattr(PageScraper, "CONTENT_SELECTORS")
        assert len(PageScraper.CONTENT_SELECTORS) > 0

    def test_remove_selectors_defined(self):
        """Test REMOVE_SELECTORS class attribute exists."""
        assert hasattr(PageScraper, "REMOVE_SELECTORS")
        assert len(PageScraper.REMOVE_SELECTORS) > 0


class TestFetchPage:
    """Tests for fetch_page method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_fetch_calls_rate_limiter(self, scraper):
        """Test rate limiter wait is called."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        scraper.fetch_page("https://example.com")
        scraper.rate_limiter.wait.assert_called_once()

    def test_fetch_returns_html_and_status(self, scraper):
        """Test fetch returns tuple of html and status."""
        mock_response = Mock()
        mock_response.text = "<html><body>Content</body></html>"
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        html, status = scraper.fetch_page("https://example.com")

        assert html == "<html><body>Content</body></html>"
        assert status == 200

    def test_fetch_uses_timeout(self, scraper):
        """Test fetch uses 30 second timeout."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        scraper.fetch_page("https://example.com/page")

        scraper.session.get.assert_called_once_with(
            "https://example.com/page", timeout=30
        )

    def test_fetch_raises_for_http_error(self, scraper):
        """Test raises for status errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        scraper.session.get.return_value = mock_response

        with pytest.raises(ContentExtractionError) as exc_info:
            scraper.fetch_page("https://example.com/notfound")

        assert "Failed to fetch page" in str(exc_info.value)

    def test_fetch_raises_for_connection_error(self, scraper):
        """Test raises for connection errors."""
        scraper.session.get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(ContentExtractionError):
            scraper.fetch_page("https://example.com")

    def test_fetch_raises_for_timeout(self, scraper):
        """Test raises for timeout errors."""
        scraper.session.get.side_effect = requests.Timeout("Read timed out")

        with pytest.raises(ContentExtractionError):
            scraper.fetch_page("https://example.com")


class TestExtractContent:
    """Tests for extract_content method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_extract_title_from_h1(self, scraper):
        """Test title extracted from h1 element."""
        html = """
        <html>
        <head><title>Meta Title</title></head>
        <body>
            <h1>Main Title</h1>
            <div id="region-main">Content here</div>
        </body>
        </html>
        """
        title, _ = scraper.extract_content(html, "https://example.com")
        assert title == "Main Title"

    def test_extract_title_from_title_fallback(self, scraper):
        """Test title extracted from title element when no h1."""
        html = """
        <html>
        <head><title>Page Title</title></head>
        <body>
            <div id="region-main">Content here</div>
        </body>
        </html>
        """
        title, _ = scraper.extract_content(html, "https://example.com")
        assert title == "Page Title"

    def test_extract_empty_title(self, scraper):
        """Test empty title when no title elements."""
        html = """
        <html>
        <body>
            <div id="region-main">Content here</div>
        </body>
        </html>
        """
        title, _ = scraper.extract_content(html, "https://example.com")
        assert title == ""

    def test_extract_content_from_region_main(self, scraper):
        """Test content extracted from #region-main."""
        html = """
        <html>
        <body>
            <nav>Navigation</nav>
            <div id="region-main">
                <p>Main content here</p>
            </div>
            <footer>Footer</footer>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Main content here" in content
        assert "region-main" in content

    def test_extract_content_from_course_content(self, scraper):
        """Test content extracted from .course-content."""
        html = """
        <html>
        <body>
            <div class="course-content">
                <p>Course content here</p>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Course content here" in content

    def test_extract_removes_nav(self, scraper):
        """Test nav elements are removed."""
        html = """
        <html>
        <body>
            <nav>Navigation Links</nav>
            <div id="region-main">Main Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Navigation Links" not in content

    def test_extract_removes_footer(self, scraper):
        """Test footer elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">Main Content</div>
            <footer>Footer Content</footer>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Footer Content" not in content

    def test_extract_removes_scripts(self, scraper):
        """Test script elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                Main Content
                <script>alert('hello');</script>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "alert" not in content
        assert "<script>" not in content

    def test_extract_removes_styles(self, scraper):
        """Test style elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                Main Content
                <style>.class { color: red; }</style>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "color: red" not in content
        assert "<style>" not in content

    def test_extract_removes_breadcrumb(self, scraper):
        """Test breadcrumb elements are removed."""
        html = """
        <html>
        <body>
            <div class="breadcrumb">Home > Course > Page</div>
            <div id="region-main">Main Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Home > Course > Page" not in content

    def test_extract_removes_sr_only(self, scraper):
        """Test screen-reader-only elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                <span class="sr-only">Skip to content</span>
                Main Content
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Skip to content" not in content

    def test_extract_fallback_to_body(self, scraper):
        """Test fallback to body when no content selectors match."""
        html = """
        <html>
        <body>
            <div class="custom-content">Body Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Body Content" in content

    def test_extract_content_returns_string(self, scraper):
        """Test content returned as string."""
        html = """
        <html>
        <body>
            <div id="region-main">Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert isinstance(content, str)


class TestScrapePage:
    """Tests for scrape_page method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_scrape_returns_document(self, scraper):
        """Test successful scrape returns Document."""
        mock_response = Mock()
        mock_response.text = """
        <html>
        <body>
            <h1>Test Page Title</h1>
            <div id="region-main"><p>Page content</p></div>
        </body>
        </html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/page", section="Test Section")

        assert isinstance(result, Document)
        assert result.metadata.title == "Test Page Title"
        assert result.metadata.section == "Test Section"
        assert result.metadata.content_type == "page"

    def test_scrape_stores_raw_html(self, scraper):
        """Test raw HTML is stored in document."""
        mock_response = Mock()
        mock_response.text = """
        <html>
        <body>
            <h1>Title</h1>
            <div id="region-main"><p>Content here</p></div>
        </body>
        </html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/page")

        assert result.raw_html is not None
        assert "Content here" in result.raw_html

    def test_scrape_sets_untitled_for_no_title(self, scraper):
        """Test 'Untitled Page' used when no title found."""
        mock_response = Mock()
        mock_response.text = """
        <html>
        <body>
            <div id="region-main"><p>Content only</p></div>
        </body>
        </html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/page")

        assert result.metadata.title == "Untitled Page"

    def test_scrape_returns_none_for_non_200(self, scraper):
        """Test None returned for non-200 status."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        mock_response.status_code = 404
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/notfound")

        assert result is None

    def test_scrape_returns_none_on_fetch_error(self, scraper):
        """Test None returned when fetch fails."""
        scraper.session.get.side_effect = requests.ConnectionError("Failed")

        result = scraper.scrape_page("https://example.com/error")

        assert result is None

    def test_scrape_returns_none_on_unexpected_error(self, scraper):
        """Test None returned for unexpected errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Title</h1></body></html>"
        scraper.session.get.return_value = mock_response

        # Mock extract_content to raise unexpected error
        with patch.object(scraper, "extract_content", side_effect=ValueError("Unexpected")):
            result = scraper.scrape_page("https://example.com/page")

        assert result is None

    def test_scrape_sets_source_url(self, scraper):
        """Test source_url is set in metadata."""
        mock_response = Mock()
        mock_response.text = """
        <html><body>
            <h1>Title</h1>
            <div id="region-main">Content</div>
        </body></html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/mypage")

        assert result.metadata.source_url == "https://example.com/mypage"

    def test_scrape_default_section_empty(self, scraper):
        """Test section defaults to empty string."""
        mock_response = Mock()
        mock_response.text = """
        <html><body>
            <h1>Title</h1>
            <div id="region-main">Content</div>
        </body></html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/page")

        assert result.metadata.section == ""

    def test_scrape_content_initially_empty(self, scraper):
        """Test content is empty (to be populated after cleaning)."""
        mock_response = Mock()
        mock_response.text = """
        <html><body>
            <h1>Title</h1>
            <div id="region-main">Some content</div>
        </body></html>
        """
        mock_response.status_code = 200
        scraper.session.get.return_value = mock_response

        result = scraper.scrape_page("https://example.com/page")

        # Content is empty, raw_html has the content
        assert result.content == ""
        assert result.raw_html is not None


class TestContentSelectors:
    """Tests for content selector priority."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_region_main_priority(self, scraper):
        """Test #region-main has highest priority."""
        html = """
        <html>
        <body>
            <div id="region-main">Priority Content</div>
            <div class="course-content">Secondary Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Priority Content" in content
        # course-content should not be the main container
        assert "region-main" in content

    def test_page_content_fallback(self, scraper):
        """Test #page-content is used as fallback."""
        html = """
        <html>
        <body>
            <div id="page-content">Page Content Area</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Page Content Area" in content

    def test_role_main_fallback(self, scraper):
        """Test div[role='main'] is used as fallback."""
        html = """
        <html>
        <body>
            <div role="main">Accessible Main Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Accessible Main Content" in content


class TestRemoveSelectors:
    """Tests for element removal."""

    @pytest.fixture
    def scraper(self):
        """Create scraper with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_removes_navbar(self, scraper):
        """Test .navbar elements are removed."""
        html = """
        <html>
        <body>
            <div class="navbar">Top Navigation</div>
            <div id="region-main">Main Content</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Top Navigation" not in content

    def test_preserves_block_content(self, scraper):
        """Test .block elements are preserved (they can contain real content in Moodle)."""
        html = """
        <html>
        <body>
            <div id="region-main">
                <div class="block">Block Content</div>
                <p>Main Content</p>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        # .block is no longer removed because it can contain actual content
        # in Moodle book chapters, which was causing 154/161 empty documents
        assert "Block Content" in content
        assert "Main Content" in content

    def test_removes_activity_navigation(self, scraper):
        """Test .activity-navigation elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                <div class="activity-navigation">Prev | Next</div>
                <p>Main Content</p>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Prev | Next" not in content

    def test_removes_visually_hidden(self, scraper):
        """Test .visually-hidden elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                <span class="visually-hidden">Hidden text</span>
                <p>Visible Content</p>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Hidden text" not in content

    def test_removes_noscript(self, scraper):
        """Test noscript elements are removed."""
        html = """
        <html>
        <body>
            <div id="region-main">
                <noscript>JavaScript required</noscript>
                <p>Main Content</p>
            </div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "JavaScript required" not in content

    def test_removes_page_footer(self, scraper):
        """Test #page-footer is removed."""
        html = """
        <html>
        <body>
            <div id="region-main">Main Content</div>
            <div id="page-footer">Copyright Info</div>
        </body>
        </html>
        """
        _, content = scraper.extract_content(html, "https://example.com")
        assert "Copyright Info" not in content
