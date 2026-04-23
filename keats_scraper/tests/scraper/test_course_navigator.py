"""Tests for CourseNavigator."""

from unittest.mock import Mock

import pytest
import requests
from bs4 import BeautifulSoup

from keats_scraper.config import KEATSConfig, ScraperConfig
from keats_scraper.models.document import ResourceInfo
from keats_scraper.scraper.course_navigator import CourseNavigator
from keats_scraper.scraper.rate_limiter import RateLimiter
from keats_scraper.utils.exceptions import ContentExtractionError


class TestCourseNavigatorInit:
    """Tests for CourseNavigator initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.base_url = "https://keats.kcl.ac.uk"
        config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=12345"
        return config

    def test_init_sets_session(self, mock_config):
        """Test session is set correctly."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        navigator = CourseNavigator(mock_session, mock_config, mock_limiter)
        assert navigator.session is mock_session

    def test_init_sets_config(self, mock_config):
        """Test config is set correctly."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        navigator = CourseNavigator(mock_session, mock_config, mock_limiter)
        assert navigator.config is mock_config

    def test_init_sets_rate_limiter(self, mock_config):
        """Test rate_limiter is set correctly."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        navigator = CourseNavigator(mock_session, mock_config, mock_limiter)
        assert navigator.rate_limiter is mock_limiter

    def test_init_sets_base_url(self, mock_config):
        """Test base_url is set from config."""
        mock_session = Mock(spec=requests.Session)
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        navigator = CourseNavigator(mock_session, mock_config, mock_limiter)
        assert navigator.base_url == "https://keats.kcl.ac.uk"

    def test_resource_patterns_defined(self):
        """Test RESOURCE_PATTERNS class attribute exists."""
        assert hasattr(CourseNavigator, "RESOURCE_PATTERNS")
        assert "page" in CourseNavigator.RESOURCE_PATTERNS
        assert "resource" in CourseNavigator.RESOURCE_PATTERNS
        assert "folder" in CourseNavigator.RESOURCE_PATTERNS
        assert "book" in CourseNavigator.RESOURCE_PATTERNS


class TestFetchCoursePage:
    """Tests for fetch_course_page method."""

    @pytest.fixture
    def navigator(self):
        """Create navigator with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=123"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_fetch_calls_rate_limiter(self, navigator):
        """Test rate limiter wait is called."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        navigator.fetch_course_page()
        navigator.rate_limiter.wait.assert_called_once()

    def test_fetch_returns_html(self, navigator):
        """Test fetch returns HTML content."""
        mock_response = Mock()
        mock_response.text = "<html><body>Course content</body></html>"
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        result = navigator.fetch_course_page()
        assert result == "<html><body>Course content</body></html>"

    def test_fetch_uses_course_url(self, navigator):
        """Test fetch uses config course URL."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        navigator.fetch_course_page()

        navigator.session.get.assert_called_once_with(
            navigator.config.keats.course_url, timeout=30
        )

    def test_fetch_raises_on_login_redirect(self, navigator):
        """Test raises when redirected to login."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.url = "https://keats.kcl.ac.uk/login/index.php"
        navigator.session.get.return_value = mock_response

        with pytest.raises(ContentExtractionError) as exc_info:
            navigator.fetch_course_page()

        assert "Session expired" in str(exc_info.value)

    def test_fetch_raises_on_http_error(self, navigator):
        """Test raises for HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        navigator.session.get.return_value = mock_response

        with pytest.raises(ContentExtractionError):
            navigator.fetch_course_page()

    def test_fetch_raises_on_connection_error(self, navigator):
        """Test raises for connection errors."""
        navigator.session.get.side_effect = requests.ConnectionError("Failed")

        with pytest.raises(ContentExtractionError):
            navigator.fetch_course_page()


class TestIdentifyResourceType:
    """Tests for _identify_resource_type method."""

    @pytest.fixture
    def navigator(self):
        """Create navigator with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        return CourseNavigator(mock_session, mock_config, mock_limiter)

    @pytest.mark.parametrize("url,expected", [
        ("https://keats.kcl.ac.uk/mod/page/view.php?id=123", "page"),
        ("https://keats.kcl.ac.uk/mod/resource/view.php?id=456", "resource"),
        ("https://keats.kcl.ac.uk/mod/folder/view.php?id=789", "folder"),
        ("https://keats.kcl.ac.uk/mod/book/view.php?id=111", "book"),
        ("https://keats.kcl.ac.uk/mod/url/view.php?id=222", "url"),
        ("https://keats.kcl.ac.uk/mod/label/view.php?id=333", "label"),
    ])
    def test_identify_known_types(self, navigator, url, expected):
        """Test identification of known resource types."""
        result = navigator._identify_resource_type(url)
        assert result == expected

    def test_identify_unknown_type(self, navigator):
        """Test unknown type for unrecognized URL."""
        url = "https://keats.kcl.ac.uk/mod/quiz/view.php?id=123"
        result = navigator._identify_resource_type(url)
        assert result == "unknown"

    def test_identify_non_moodle_url(self, navigator):
        """Test unknown type for non-Moodle URL."""
        url = "https://example.com/document.pdf"
        result = navigator._identify_resource_type(url)
        assert result == "unknown"


class TestDiscoverResources:
    """Tests for discover_resources method."""

    @pytest.fixture
    def navigator(self):
        """Create navigator with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=123"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_discover_returns_list(self, navigator):
        """Test discover_resources returns list."""
        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        result = navigator.discover_resources()
        assert isinstance(result, list)

    def test_discover_finds_page_resources(self, navigator):
        """Test discovery of page resources."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Test Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=123">
                        <span class="instancename">Test Page</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 1
        assert resources[0].resource_type == "page"
        assert resources[0].title == "Test Page"
        # Main page resources get "Main" as section name
        assert resources[0].section == "Main"

    def test_discover_returns_resource_info(self, navigator):
        """Test returns ResourceInfo objects."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=1">
                        <span class="instancename">Page 1</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 1
        assert isinstance(resources[0], ResourceInfo)

    def test_discover_sets_section_index(self, navigator):
        """Test section_index is set correctly for main page resources."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section 0</h3>
            </div>
            <div class="section main">
                <h3 class="sectionname">Section 1</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=1">
                        <span class="instancename">Page</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        # Main page resources get section_index 0
        assert resources[0].section_index == 0

    def test_discover_removes_duplicates(self, navigator):
        """Test duplicate URLs are removed."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=123">
                        <span class="instancename">Page</span>
                    </a>
                </div>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=123">
                        <span class="instancename">Same Page</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 1

    def test_discover_skips_unknown_types(self, navigator):
        """Test unknown resource types are skipped."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/quiz/view.php?id=123">
                        <span class="instancename">Quiz</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 0

    def test_discover_skips_external_urls(self, navigator):
        """Test external URLs are skipped."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://example.com/external">
                        <span class="instancename">External Link</span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 0

    def test_discover_skips_empty_href(self, navigator):
        """Test empty hrefs are skipped."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="">Empty</a>
                </div>
                <div class="activity">
                    <a href="#">Hash</a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert len(resources) == 0

    def test_discover_removes_accesshide(self, navigator):
        """Test accesshide spans are removed from title."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=1">
                        <span class="instancename">
                            Page Title
                            <span class="accesshide">File</span>
                        </span>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert resources[0].title == "Page Title"
        assert "File" not in resources[0].title

    def test_discover_uses_default_title(self, navigator):
        """Test default title when none found."""
        html = """
        <html>
        <body>
            <div class="section main">
                <h3 class="sectionname">Section</h3>
                <div class="activity">
                    <a href="https://keats.kcl.ac.uk/mod/page/view.php?id=1">
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_response.url = "https://keats.kcl.ac.uk/course"
        navigator.session.get.return_value = mock_response

        resources = navigator.discover_resources()

        assert resources[0].title == "Untitled page"


class TestDiscoverBookChapters:
    """Tests for discover_book_chapters method."""

    @pytest.fixture
    def navigator(self):
        """Create navigator with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_discover_calls_rate_limiter(self, navigator):
        """Test rate limiter wait is called."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        navigator.session.get.return_value = mock_response

        navigator.discover_book_chapters("https://keats.kcl.ac.uk/mod/book/view.php?id=1")
        navigator.rate_limiter.wait.assert_called_once()

    def test_discover_book_chapters_success(self, navigator):
        """Test successful chapter discovery."""
        html = """
        <html>
        <body>
            <div class="book_toc">
                <a href="/mod/book/view.php?id=1&chapterid=1">Chapter 1</a>
                <a href="/mod/book/view.php?id=1&chapterid=2">Chapter 2</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )

        assert len(chapters) == 2
        assert chapters[0].title == "Chapter 1"
        assert chapters[1].title == "Chapter 2"
        assert chapters[0].resource_type == "book_chapter"

    def test_discover_book_toc_selector(self, navigator):
        """Test #book-toc selector."""
        html = """
        <html>
        <body>
            <div id="book-toc">
                <a href="/mod/book/view.php?id=1&chapterid=1">TOC Chapter</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )

        assert len(chapters) == 1
        assert chapters[0].title == "TOC Chapter"

    def test_discover_book_returns_empty_on_error(self, navigator):
        """Test empty list on request error."""
        navigator.session.get.side_effect = requests.ConnectionError("Failed")

        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )

        assert chapters == []

    def test_discover_book_makes_absolute_urls(self, navigator):
        """Test relative URLs are made absolute."""
        html = """
        <html>
        <body>
            <div class="book_toc">
                <a href="/mod/book/view.php?id=1&chapterid=5">Chapter</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )

        assert chapters[0].url.startswith("https://keats.kcl.ac.uk")

    def test_discover_book_skips_empty_href(self, navigator):
        """Test links with empty href are skipped."""
        html = """
        <html>
        <body>
            <div class="book_toc">
                <a href="">Empty Link</a>
                <a>No Href</a>
                <a href="/mod/book/view.php?id=1&chapterid=5">Valid Chapter</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )

        # Only the valid chapter should be found
        assert len(chapters) == 1
        assert chapters[0].title == "Valid Chapter"


class TestDiscoverFolderContents:
    """Tests for discover_folder_contents method."""

    @pytest.fixture
    def navigator(self):
        """Create navigator with mocked dependencies."""
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()

        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_discover_calls_rate_limiter(self, navigator):
        """Test rate limiter wait is called."""
        mock_response = Mock()
        mock_response.text = "<html></html>"
        navigator.session.get.return_value = mock_response

        navigator.discover_folder_contents("https://keats.kcl.ac.uk/mod/folder/view.php?id=1")
        navigator.rate_limiter.wait.assert_called_once()

    def test_discover_folder_success(self, navigator):
        """Test successful file discovery."""
        html = """
        <html>
        <body>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/file1.docx">Document.docx</a>
            </div>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/file2.pdf">Guide.pdf</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert len(files) == 2
        assert files[0].title == "Document.docx"
        assert files[1].title == "Guide.pdf"

    def test_discover_folder_identifies_pdf(self, navigator):
        """Test PDF files are identified."""
        html = """
        <html>
        <body>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/handbook.pdf">Handbook</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert files[0].resource_type == "pdf"

    def test_discover_folder_pdf_in_title(self, navigator):
        """Test PDF identified from title."""
        html = """
        <html>
        <body>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/file">Report.PDF</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert files[0].resource_type == "pdf"

    def test_discover_folder_non_pdf_resource(self, navigator):
        """Test non-PDF files are typed as resource."""
        html = """
        <html>
        <body>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/document.docx">Document</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert files[0].resource_type == "resource"

    def test_discover_folder_returns_empty_on_error(self, navigator):
        """Test empty list on request error."""
        navigator.session.get.side_effect = requests.HTTPError("404")

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert files == []

    def test_discover_folder_content_selector(self, navigator):
        """Test .folder-content selector."""
        html = """
        <html>
        <body>
            <div class="folder-content">
                <a href="/pluginfile.php/123/file.txt">Text File</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert len(files) == 1
        assert files[0].title == "Text File"

    def test_discover_folder_skips_empty_href(self, navigator):
        """Test empty hrefs are skipped."""
        html = """
        <html>
        <body>
            <div class="fp-filename-icon">
                <a href="">Empty</a>
            </div>
            <div class="fp-filename-icon">
                <a href="/pluginfile.php/123/real.pdf">Real File</a>
            </div>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.text = html
        navigator.session.get.return_value = mock_response

        files = navigator.discover_folder_contents(
            "https://keats.kcl.ac.uk/mod/folder/view.php?id=1"
        )

        assert len(files) == 1
        assert files[0].title == "Real File"


class TestDiscoverSectionLinks:
    """Tests for _discover_section_links which handles grid/onetopic
    Moodle course formats where each section lives at its own
    section.php URL.
    """

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_courseindex_structure_produces_section_links(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Programmes</div>
            <a href='/course/section.php?id=10'>Programmes section</a>
          </div>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Main</div>
            <a href='/course/section.php?id=20'>Main section</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = navigator._discover_section_links(soup)
        names = [link["name"] for link in links]
        assert "Programmes" in names
        assert "Main" in names
        assert all(link["url"].startswith("https://keats.kcl.ac.uk") for link in links)

    def test_courseindex_deduplicates_by_section_id(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Main</div>
            <a href='/course/section.php?id=10'>Main A</a>
          </div>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Main dup</div>
            <a href='/course/section.php?id=10'>Main B</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = navigator._discover_section_links(soup)
        assert len(links) == 1

    def test_courseindex_missing_name_uses_link_text(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-section'>
            <a href='/course/section.php?id=77'>Induction</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = navigator._discover_section_links(soup)
        assert links and links[0]["name"] == "Induction"

    def test_fallback_finds_section_php_anchors(self, navigator):
        """When no courseindex-section wrappers exist, the fallback picks
        up any section.php anchor."""
        html = """
        <html><body>
          <a href='/course/section.php?id=30'>Section Three</a>
          <a href='/course/section.php?id=40'>Section Four</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = navigator._discover_section_links(soup)
        assert len(links) == 2
        assert links[0]["name"] == "Section Three"

    def test_fallback_deduplicates_and_handles_empty_text(self, navigator):
        html = """
        <html><body>
          <a href='/course/section.php?id=50'>First</a>
          <a href='/course/section.php?id=50'>Dup</a>
          <a href='/course/section.php?id=60'></a>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        links = navigator._discover_section_links(soup)
        assert len(links) == 2
        # Second entry should have the auto-generated name since its text is empty
        names = [link["name"] for link in links]
        assert any(n.startswith("Section ") for n in names)


class TestExtractResourcesFromCourseindex:
    """Tests for _extract_resources_from_courseindex which picks up
    Moodle's sidebar navigation when the course is rendered with the
    courseindex block enabled.
    """

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_extracts_page_resources_with_section_name(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Teaching</div>
            <div class='courseindex-item'>
              <a href='/mod/page/view.php?id=11'>Attendance Policy</a>
            </div>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert len(resources) == 1
        assert resources[0].resource_type == "page"
        assert resources[0].title == "Attendance Policy"
        assert resources[0].section == "Teaching"

    def test_skips_labels_and_unknown_types(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-item'>
            <a href='/mod/label/view.php?id=1'>Label</a>
          </div>
          <div class='courseindex-item'>
            <a href='/mod/unknown/view.php?id=2'>Unknown</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert resources == []

    def test_skips_non_keats_urls(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-item'>
            <a href='https://other.edu/mod/page/view.php?id=1'>External</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert resources == []

    def test_deduplicates_same_url(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-item'>
            <a href='/mod/page/view.php?id=5'>Page A</a>
          </div>
          <div class='courseindex-item'>
            <a href='/mod/page/view.php?id=5'>Page B (dup)</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert len(resources) == 1

    def test_default_section_when_no_parent_section(self, navigator):
        """When a courseindex-item has no courseindex-section ancestor,
        the resource gets the default 'Course Content' section."""
        html = """
        <html><body>
          <div class='courseindex-item'>
            <a href='/mod/page/view.php?id=9'>Orphan Page</a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert resources and resources[0].section == "Course Content"

    def test_skips_empty_href_and_empty_title(self, navigator):
        html = """
        <html><body>
          <div class='courseindex-item'>
            <a href=''>empty href</a>
          </div>
          <div class='courseindex-item'>
            <a href='/mod/page/view.php?id=11'></a>
          </div>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_courseindex(soup)
        assert resources == []


class TestDiscoverResourcesSectionWalk:
    """Covers the discover_resources path that walks each discovered
    section URL with a GET."""

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_discovers_resources_from_section_pages(self, navigator):
        course_html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Programmes</div>
            <a href='/course/section.php?id=10'>Programmes</a>
          </div>
        </body></html>
        """
        section_html = """
        <html><body>
          <div class='activity'>
            <a class='aalink' href='/mod/page/view.php?id=100'>
              <span class='instancename'>Module Spec</span>
            </a>
          </div>
        </body></html>
        """
        course_response = Mock()
        course_response.text = course_html
        course_response.url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        course_response.raise_for_status = Mock()

        section_response = Mock()
        section_response.text = section_html
        section_response.raise_for_status = Mock()

        navigator.session.get.side_effect = [course_response, section_response]
        resources = navigator.discover_resources()
        urls = [r.url for r in resources]
        assert any("/mod/page/view.php?id=100" in u for u in urls)

    def test_section_fetch_failure_is_logged_not_raised(self, navigator):
        """A RequestException on a section GET is caught and logged; the
        top-level discovery still returns the partial list."""
        course_html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Main</div>
            <a href='/course/section.php?id=10'>Main</a>
          </div>
        </body></html>
        """
        course_response = Mock()
        course_response.text = course_html
        course_response.url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        course_response.raise_for_status = Mock()

        navigator.session.get.side_effect = [
            course_response,
            requests.ConnectionError("section fetch failed"),
        ]
        # Must not raise
        resources = navigator.discover_resources()
        assert isinstance(resources, list)


class TestDiscoverBookChaptersEmptyHref:
    """Pin the href == '' and empty-title skip branches of
    discover_book_chapters that existing tests don't reach."""

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_book_skips_anchor_with_empty_title(self, navigator):
        html = """
        <html><body>
          <div class='book_toc'>
            <a href='/mod/book/view.php?id=1&chapterid=5'></a>
            <a href='/mod/book/view.php?id=1&chapterid=6'>Chapter 6</a>
          </div>
        </body></html>
        """
        response = Mock()
        response.text = html
        response.raise_for_status = Mock()
        navigator.session.get.return_value = response
        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )
        titles = [c.title for c in chapters]
        assert titles == ["Chapter 6"]

    def test_book_deduplicates_duplicate_chapter_urls(self, navigator):
        """Two identical chapter URLs in the TOC (same id, same chapterid)
        must produce exactly one chapter -- the second hit skips via
        seen_urls."""
        html = """
        <html><body>
          <div class='book_toc'>
            <a href='/mod/book/view.php?id=1&chapterid=5'>Ch A</a>
            <a href='/mod/book/view.php?id=1&chapterid=5'>Ch A dup</a>
          </div>
        </body></html>
        """
        response = Mock()
        response.text = html
        response.raise_for_status = Mock()
        navigator.session.get.return_value = response
        chapters = navigator.discover_book_chapters(
            "https://keats.kcl.ac.uk/mod/book/view.php?id=1"
        )
        assert len(chapters) == 1


class TestExtractResourcesFromSoupLabel:
    """Labels match the activity CSS selectors but must be skipped at the
    resource-type check inside _extract_resources_from_soup because
    Moodle labels carry no content URL."""

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_label_resource_is_skipped(self, navigator):
        html = """
        <html><body>
          <li class='activity label'>
            <a class='aalink' href='/mod/label/view.php?id=99'>
              <span class='instancename'>A label</span>
            </a>
          </li>
          <li class='activity page'>
            <a class='aalink' href='/mod/page/view.php?id=100'>
              <span class='instancename'>A page</span>
            </a>
          </li>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        resources = navigator._extract_resources_from_soup(soup, "Main", 0)
        # Labels skipped, only the page survives.
        assert len(resources) == 1
        assert resources[0].resource_type == "page"


class TestDiscoverResourcesLogsCourseindexCount:
    """Pin the log line that fires inside discover_resources when the
    courseindex sidebar yielded resources. Regressions here would silently
    disable the coverage analyser's sidebar-vs-content diagnostic."""

    @pytest.fixture
    def navigator(self):
        mock_session = Mock(spec=requests.Session)
        mock_config = Mock(spec=ScraperConfig)
        mock_config.keats = Mock(spec=KEATSConfig)
        mock_config.keats.base_url = "https://keats.kcl.ac.uk"
        mock_config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return CourseNavigator(mock_session, mock_config, mock_limiter)

    def test_courseindex_resources_are_included(self, navigator):
        course_html = """
        <html><body>
          <div class='courseindex-section'>
            <div class='courseindex-section-title'>Programmes</div>
            <div class='courseindex-item'>
              <a href='/mod/page/view.php?id=900'>Intro</a>
            </div>
          </div>
        </body></html>
        """
        course_response = Mock()
        course_response.text = course_html
        course_response.url = "https://keats.kcl.ac.uk/course/view.php?id=1"
        course_response.raise_for_status = Mock()
        navigator.session.get.return_value = course_response

        resources = navigator.discover_resources()
        urls = [r.url for r in resources]
        assert any("/mod/page/view.php?id=900" in u for u in urls)
