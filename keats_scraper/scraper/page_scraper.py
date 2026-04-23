"""Page content scraper for KEATS Moodle pages."""


import requests
from bs4 import BeautifulSoup

from keats_scraper.models.document import Document
from keats_scraper.scraper.rate_limiter import RateLimiter
from keats_scraper.utils.exceptions import ContentExtractionError, RateLimitError
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class PageScraper:
    """Scrapes content from KEATS Moodle pages."""

    # CSS selectors for Moodle content areas
    CONTENT_SELECTORS = [
        # Book-specific selectors (more specific, check first)
        ".book_content",
        ".book-content",
        "#book_content",
        "#book-content",
        ".chapter-content",
        ".chapter",
        # General Moodle selectors
        "#region-main",
        ".course-content",
        "#page-content",
        "div[role='main']",
        ".generalbox",
    ]

    # Elements to remove. .block is intentionally excluded: Moodle wraps
    # book-chapter content in .block-named containers, so removing them
    # strips the actual handbook text we are trying to scrape.
    REMOVE_SELECTORS = [
        "nav",
        ".navbar",
        "footer",
        ".footer",
        "#page-footer",
        ".breadcrumb",
        ".activity-navigation",
        "script",
        "style",
        "noscript",
        ".sr-only",
        ".visually-hidden",
        # Book-specific navigation elements
        ".block_book_toc",
        ".book_toc",
        "#book-toc",
        ".book_toc_numbered",
        ".book-navigation",
        ".navbuttons",
    ]

    def __init__(self, session: requests.Session, rate_limiter: RateLimiter):
        """
        Initialize page scraper.

        Args:
            session: Authenticated requests session
            rate_limiter: Rate limiter instance
        """
        self.session = session
        self.rate_limiter = rate_limiter

    def fetch_page(self, url: str) -> tuple[str, int]:
        """
        Fetch page HTML with rate limiting.

        Args:
            url: Page URL to fetch

        Returns:
            Tuple of (html_content, status_code)

        Raises:
            ContentExtractionError: If fetch fails (including exhausted
                rate-limit retries, which are re-raised with the
                original RateLimitError chained).
        """
        self.rate_limiter.wait()

        def do_get() -> requests.Response:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response

        try:
            response = self.rate_limiter.retry_on_rate_limit(do_get)
            return response.text, response.status_code

        except (requests.RequestException, RateLimitError) as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise ContentExtractionError(f"Failed to fetch page: {e}") from e

    def extract_content(self, html: str, url: str) -> tuple[str, str]:
        """
        Extract main content from Moodle page HTML.

        Args:
            html: Raw HTML content
            url: Source URL (for logging)

        Returns:
            Tuple of (title, main_content_html)
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = ""
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Remove unwanted elements
        for selector in self.REMOVE_SELECTORS:
            for elem in soup.select(selector):
                elem.decompose()

        # Find main content - validate each candidate has meaningful text
        content_elem = None
        for selector in self.CONTENT_SELECTORS:
            candidate = soup.select_one(selector)
            if candidate and len(candidate.get_text(strip=True)) > 20:
                content_elem = candidate
                break

        if not content_elem:
            logger.warning(f"No meaningful content found for {url}")
            content_elem = soup.body if soup.body else soup

        return title, str(content_elem)

    def scrape_page(
        self,
        url: str,
        section: str = "",
    ) -> Document | None:
        """
        Scrape a single KEATS page.

        Args:
            url: Page URL
            section: Handbook section name

        Returns:
            Document or None if extraction fails
        """
        logger.info(f"Scraping page: {url}")

        try:
            html, status = self.fetch_page(url)

            if status != 200:
                logger.warning(f"Non-200 status ({status}) for {url}")
                return None

            title, content_html = self.extract_content(html, url)

            if not title:
                title = "Untitled Page"

            # Create document with raw HTML (cleaning happens later)
            document = Document.create(
                source_url=url,
                title=title,
                content="",  # Will be populated after cleaning
                content_type="page",
                section=section,
                raw_html=content_html,
            )

            logger.info(f"Extracted: '{title}' from {url}")
            return document

        except ContentExtractionError:
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None
