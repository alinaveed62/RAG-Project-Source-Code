"""SSO authentication handler for KEATS using Selenium."""

import time

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from keats_scraper.auth.session_manager import SessionManager
from keats_scraper.config import ScraperConfig
from keats_scraper.utils.exceptions import AuthenticationError, SessionExpiredError
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class SSOHandler:
    """Handles SSO authentication to KEATS via browser automation."""

    def __init__(self, config: ScraperConfig):
        """
        Initialize SSO handler.

        Args:
            config: Scraper configuration
        """
        self.config = config
        self.session_manager = SessionManager(
            cookie_file=config.auth.cookie_file,
            encryption_key=config.auth.encryption_key,
        )
        self._driver: webdriver.Chrome | None = None

    def _create_driver(self) -> webdriver.Chrome:
        """
        Create and configure Chrome WebDriver.

        Always runs in visible (non-headless) mode because KCL SSO
        requires manual 2FA in a real browser window; a headless option
        would be dead surface area.

        Returns:
            Configured Chrome WebDriver
        """
        options = Options()

        # Common options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        # Prevent detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Additional anti-detection
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        return driver

    def login_interactive(self) -> list[dict]:
        """
        Perform interactive login with manual 2FA.

        Opens browser for user to complete SSO login manually,
        waits for successful authentication, then extracts cookies.

        Returns:
            List of session cookies

        Raises:
            AuthenticationError: If login fails or times out
        """
        logger.info("Starting interactive SSO login...")
        logger.info("A browser window will open. Please complete the login process.")
        logger.info(f"Timeout: {self.config.auth.login_timeout} seconds")

        driver = self._create_driver()
        self._driver = driver

        try:
            # Navigate to KEATS login
            driver.get(self.config.keats.login_url)
            logger.info("Navigated to KEATS login page")

            # Wait for user to complete login
            # Success is indicated by URL change to the KEATS dashboard/course page
            success_indicators = [
                "keats.kcl.ac.uk/my/",  # Dashboard
                "keats.kcl.ac.uk/course/",  # Course page
            ]

            start_time = time.time()
            logged_in = False

            print("\n" + "=" * 60)
            print("MANUAL LOGIN REQUIRED")
            print("=" * 60)
            print("1. Complete the SSO login in the browser window")
            print("2. Complete 2FA when prompted")
            print("3. Wait until you see the KEATS dashboard")
            print("=" * 60 + "\n")

            while time.time() - start_time < self.config.auth.login_timeout:
                current_url = driver.current_url

                # Check if we've reached a success page
                if any(indicator in current_url for indicator in success_indicators):
                    logged_in = True
                    logger.info("Login successful - detected KEATS dashboard")
                    break

                time.sleep(1)

            if not logged_in:
                raise AuthenticationError(
                    f"Login timed out after {self.config.auth.login_timeout} seconds. "
                    "Please try again and complete login faster."
                )

            # Extract cookies
            cookies = driver.get_cookies()
            logger.info(f"Extracted {len(cookies)} cookies from browser")

            # Save cookies for future use
            self.session_manager.save_cookies(cookies)

            return cookies

        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise AuthenticationError(f"SSO login failed: {e}")

        finally:
            if driver:
                driver.quit()
                self._driver = None

    def get_valid_session(self, force_login: bool = False) -> requests.Session:
        """
        Get a valid authenticated session.

        Attempts to use cached cookies first, falls back to interactive login.

        Args:
            force_login: Force new login even if cookies exist

        Returns:
            Authenticated requests.Session
        """

        if not force_login:
            # Try to use cached cookies
            cookies = self.session_manager.load_cookies()

            if cookies:
                session = self.session_manager.create_session_with_cookies(cookies)

                # Validate the session. SessionExpiredError is the
                # explicit "redirected to login" signal; catching it here
                # and falling through to the fresh-login branch gives us
                # one retry path regardless of whether the failure was a
                # typed expiry or a generic False (network / non-200).
                try:
                    if self.session_manager.validate_session(
                        session, self.config.auth.session_check_url
                    ):
                        logger.info("Using cached session - still valid")
                        return session
                    logger.warning("Cached session expired")
                except SessionExpiredError:
                    logger.warning("Cached session expired (redirected to login)")

        # Need fresh login
        logger.info("Fresh login required")
        cookies = self.login_interactive()

        session = self.session_manager.create_session_with_cookies(cookies)
        return session

    def logout(self) -> None:
        """Clear saved session."""
        self.session_manager.clear_cookies()
        logger.info("Logged out - session cleared")
