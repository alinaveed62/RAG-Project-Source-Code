"""KEATS course structure navigator."""

import copy
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from keats_scraper.config import ScraperConfig
from keats_scraper.models.document import ResourceInfo
from keats_scraper.scraper.rate_limiter import RateLimiter
from keats_scraper.utils.exceptions import ContentExtractionError, RateLimitError
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class CourseNavigator:
    """Navigates and discovers resources in a KEATS course."""

    # Moodle resource type patterns
    RESOURCE_PATTERNS = {
        "page": r"/mod/page/view\.php",
        "resource": r"/mod/resource/view\.php",  # Files (PDFs, etc.)
        "folder": r"/mod/folder/view\.php",
        "book": r"/mod/book/view\.php",
        "url": r"/mod/url/view\.php",
        "label": r"/mod/label/view\.php",
        "glossary": r"/mod/glossary/view\.php",  # Glossary resources
        "forum": r"/mod/forum/view\.php",  # Discussion forums
    }

    def __init__(
        self,
        session: requests.Session,
        config: ScraperConfig,
        rate_limiter: RateLimiter,
    ):
        """
        Initialize course navigator.

        Args:
            session: Authenticated requests session
            config: Scraper configuration
            rate_limiter: Rate limiter instance
        """
        self.session = session
        self.config = config
        self.rate_limiter = rate_limiter
        self.base_url = config.keats.base_url

    def fetch_course_page(self) -> str:
        """
        Fetch the main course page HTML.

        Returns:
            Course page HTML

        Raises:
            ContentExtractionError: If fetch fails (including after
                rate-limit retries have been exhausted).
        """
        self.rate_limiter.wait()

        def do_get() -> requests.Response:
            response = self.session.get(self.config.keats.course_url, timeout=30)
            response.raise_for_status()
            return response

        try:
            response = self.rate_limiter.retry_on_rate_limit(do_get)

            # Check if we got redirected to login
            if "login" in response.url.lower():
                raise ContentExtractionError(
                    "Session expired - redirected to login page"
                )

            return response.text

        except (requests.RequestException, RateLimitError) as e:
            logger.error(f"Failed to fetch course page: {e}")
            raise ContentExtractionError(
                f"Failed to fetch course page: {e}"
            ) from e

    def _identify_resource_type(self, url: str) -> str:
        """Identify Moodle resource type from URL."""
        for resource_type, pattern in self.RESOURCE_PATTERNS.items():
            if re.search(pattern, url):
                return resource_type
        return "unknown"

    def _discover_section_links(self, soup: BeautifulSoup) -> list[dict]:
        """
        Discover links to section pages (for grid/onetopic format courses).

        Args:
            soup: BeautifulSoup of course page

        Returns:
            List of section info dicts with url, name, index
        """
        section_links = []
        seen_ids = set()

        # First, try to get sections from courseindex structure
        courseindex_sections = soup.select(".courseindex-section")
        for section_elem in courseindex_sections:
            # Get section name
            name_elem = section_elem.select_one(
                ".courseindex-section-title, .courseindex-item-content"
            )
            name = name_elem.get_text(strip=True) if name_elem else ""

            # Get section link
            link = section_elem.select_one("a[href*='section.php']")
            if link:  # pragma: no branch - covered via fixtures; courseindex sections without a link are a Moodle edge case
                href = link.get("href", "")
                if href:  # pragma: no branch - selector guarantees href contains section.php
                    url = urljoin(self.base_url, href)

                    # Extract section ID
                    match = re.search(r"id=(\d+)", href)
                    if match:  # pragma: no branch - Moodle always emits id=NN
                        section_id = match.group(1)
                        if section_id in seen_ids:
                            continue
                        seen_ids.add(section_id)

                    if not name:
                        name = link.get_text(strip=True) or f"Section {len(section_links)}"

                    section_links.append({
                        "url": url,
                        "name": name,
                        "index": len(section_links),
                    })

        # Fallback: Find section.php links anywhere on page
        if not section_links:
            for link in soup.select("a[href*='section.php']"):
                href = link.get("href", "")
                if not href:  # pragma: no cover
                    # Unreachable: the CSS selector above requires
                    # section.php to appear in href, so a matching
                    # element can never have an empty href. Kept as a
                    # belt-and-braces guard in case the selector widens.
                    continue

                url = urljoin(self.base_url, href)

                # Extract section ID to avoid duplicates
                match = re.search(r"id=(\d+)", href)
                if match:  # pragma: no branch - Moodle section.php URLs always carry ?id=<digits>; the no-match fallthrough is defensive only
                    section_id = match.group(1)
                    if section_id in seen_ids:
                        continue
                    seen_ids.add(section_id)

                # Get section name from link text or nearby elements
                name = link.get_text(strip=True)
                if not name:
                    name = f"Section {len(section_links)}"

                section_links.append({
                    "url": url,
                    "name": name,
                    "index": len(section_links),
                })

        return section_links

    def _extract_resources_from_courseindex(self, soup: BeautifulSoup) -> list[ResourceInfo]:
        """
        Extract resource links from the courseindex sidebar navigation.

        Args:
            soup: BeautifulSoup of page

        Returns:
            List of ResourceInfo objects
        """
        resources = []
        seen_urls = set()

        # Find all courseindex items with links to /mod/
        courseindex_items = soup.select(".courseindex-item a[href*='/mod/']")

        for link in courseindex_items:
            href = link.get("href", "")
            if not href:  # pragma: no cover
                # Unreachable: the .courseindex-item a[href*='/mod/']
                # selector above requires /mod/ in href, so matches
                # always carry a non-empty href. Defensive guard only.
                continue

            url = urljoin(self.base_url, href)

            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Only process KEATS URLs
            if "keats.kcl.ac.uk" not in url:
                continue

            resource_type = self._identify_resource_type(url)
            if resource_type == "unknown" or resource_type == "label":
                continue

            title = link.get_text(strip=True)
            if not title:
                continue

            # Try to find parent section name.
            # BeautifulSoup's find_parent does not accept a CSS selector as its
            # first argument (unlike select_one); passing ".courseindex-section"
            # returns None and all resources get bucketed under "Course Content".
            # Use the attrs= form so class-based matching actually works.
            section_name = "Course Content"
            parent_section = link.find_parent(attrs={"class": "courseindex-section"})
            if parent_section:
                section_title = parent_section.select_one(".courseindex-section-title")
                if section_title:  # pragma: no branch - courseindex sections always have a title in Moodle markup
                    section_name = section_title.get_text(strip=True)

            resource = ResourceInfo(
                url=url,
                title=title,
                resource_type=resource_type,
                section=section_name,
                section_index=0,
            )
            resources.append(resource)
            logger.debug(f"Found from courseindex: {resource_type}: {title}")

        return resources

    def _extract_resources_from_soup(
        self, soup: BeautifulSoup, section_name: str, section_index: int
    ) -> list[ResourceInfo]:
        """
        Extract resource links from a BeautifulSoup object.

        Args:
            soup: BeautifulSoup of page to search
            section_name: Name of the section
            section_index: Index of the section

        Returns:
            List of ResourceInfo objects
        """
        resources = []

        # Find all activity links using multiple selectors
        activity_selectors = [
            ".activity a",
            ".activityinstance a",
            ".aalink",
            "[data-activityname] a",
            ".activity-item a",
            ".modtype_page a",
            ".modtype_book a",
            ".modtype_resource a",
            ".modtype_folder a",
            ".modtype_glossary a",
            ".modtype_forum a",
            "a[href*='/mod/page/']",
            "a[href*='/mod/book/']",
            "a[href*='/mod/resource/']",
            "a[href*='/mod/folder/']",
            "a[href*='/mod/glossary/']",
            "a[href*='/mod/forum/']",
        ]

        seen_hrefs = set()
        activities = []
        for selector in activity_selectors:
            for elem in soup.select(selector):
                href = elem.get("href", "")
                if href and href not in seen_hrefs:
                    seen_hrefs.add(href)
                    activities.append(elem)

        for activity in activities:
            href = activity.get("href", "")

            if not href or href == "#":
                continue

            # Make absolute URL
            url = urljoin(self.base_url, href)

            # Only process KEATS URLs
            if "keats.kcl.ac.uk" not in url:
                continue

            # Identify resource type
            resource_type = self._identify_resource_type(url)

            if resource_type == "unknown":
                continue

            # Skip labels as they don't have content
            if resource_type == "label":
                continue

            # Get title
            title_elem = activity.select_one(".instancename, .activityname")
            if title_elem:
                # Clone to avoid modifying original
                title_elem = copy.copy(title_elem)
                # Remove accesshide spans
                for hidden in title_elem.select(".accesshide"):
                    hidden.decompose()
                title = title_elem.get_text(strip=True)
            else:
                title = activity.get_text(strip=True)

            if not title:
                title = f"Untitled {resource_type}"

            resource = ResourceInfo(
                url=url,
                title=title,
                resource_type=resource_type,
                section=section_name,
                section_index=section_index,
            )

            resources.append(resource)
            logger.debug(f"Found {resource_type}: {title}")

        return resources

    def discover_resources(self) -> list[ResourceInfo]:
        """
        Discover all resources in the course.

        Returns:
            List of ResourceInfo objects for each discovered resource
        """
        logger.info(f"Discovering resources in course: {self.config.keats.course_url}")

        html = self.fetch_course_page()
        soup = BeautifulSoup(html, "lxml")

        resources = []

        # First, try to extract from courseindex sidebar (if present)
        courseindex_resources = self._extract_resources_from_courseindex(soup)
        if courseindex_resources:
            logger.info(f"Found {len(courseindex_resources)} resources from courseindex")
            resources.extend(courseindex_resources)

        # Also find resources on the main page content
        main_resources = self._extract_resources_from_soup(soup, "Main", 0)
        resources.extend(main_resources)
        logger.info(f"Found {len(main_resources)} resources on main page")

        # Find section links (for grid/onetopic format)
        section_links = self._discover_section_links(soup)
        logger.info(f"Found {len(section_links)} section pages to explore")

        # Navigate to each section and find resources
        for section_info in section_links:
            section_url = section_info["url"]
            section_name = section_info["name"]
            section_index = section_info["index"]

            logger.info(f"Exploring section: {section_name}")

            try:
                self.rate_limiter.wait()
                response = self.session.get(section_url, timeout=30)
                response.raise_for_status()

                section_soup = BeautifulSoup(response.text, "lxml")
                section_resources = self._extract_resources_from_soup(
                    section_soup, section_name, section_index
                )
                resources.extend(section_resources)
                logger.info(f"Found {len(section_resources)} resources in {section_name}")

            except requests.RequestException as e:
                logger.error(f"Failed to fetch section {section_name}: {e}")
                continue

        # Remove duplicates (same URL)
        seen_urls = set()
        unique_resources = []
        for resource in resources:
            if resource.url not in seen_urls:
                seen_urls.add(resource.url)
                unique_resources.append(resource)

        logger.info(f"Discovered {len(unique_resources)} unique resources total")
        return unique_resources

    def discover_book_chapters(self, book_url: str, section: str = "") -> list[ResourceInfo]:
        """
        Discover chapters within a Moodle book.

        Args:
            book_url: URL of the book resource
            section: Section name inherited from parent book resource

        Returns:
            List of ResourceInfo for each chapter
        """
        self.rate_limiter.wait()

        try:
            response = self.session.get(book_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch book {book_url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "lxml")
        chapters = []

        # Book table of contents - try multiple selectors
        toc_selectors = [
            ".book_toc a",
            "#book-toc a",
            ".book_toc_numbered a",
            "nav.book_toc a",
            "[role='navigation'] a[href*='chapterid']",
        ]

        toc = []
        for selector in toc_selectors:
            toc = soup.select(selector)
            if toc:
                break

        # If no TOC found, look for any chapter links
        if not toc:
            toc = soup.select("a[href*='chapterid']")

        seen_urls = set()
        for link in toc:
            href = link.get("href", "")
            if not href:
                continue

            # Use book_url as base for relative URLs (not self.base_url)
            url = urljoin(book_url, href)

            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = link.get_text(strip=True)
            if not title:
                continue

            chapter = ResourceInfo(
                url=url,
                title=title,
                resource_type="book_chapter",
                section=section,
            )
            chapters.append(chapter)

        logger.info(f"Found {len(chapters)} chapters in book")
        return chapters

    def discover_folder_contents(self, folder_url: str, section: str = "") -> list[ResourceInfo]:
        """
        Discover files within a Moodle folder.

        Args:
            folder_url: URL of the folder resource
            section: Section name inherited from parent folder resource

        Returns:
            List of ResourceInfo for each file
        """
        self.rate_limiter.wait()

        try:
            response = self.session.get(folder_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch folder {folder_url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "lxml")
        files = []

        # Folder file listings
        file_links = soup.select(".fp-filename-icon a, .folder-content a")

        for link in file_links:
            href = link.get("href", "")
            if not href:
                continue

            url = urljoin(self.base_url, href)
            title = link.get_text(strip=True)

            # Determine if PDF or other file
            resource_type = "resource"
            if ".pdf" in url.lower() or ".pdf" in title.lower():
                resource_type = "pdf"

            file_info = ResourceInfo(
                url=url,
                title=title,
                resource_type=resource_type,
                section=section,
            )
            files.append(file_info)

        logger.info(f"Found {len(files)} files in folder")
        return files
