"""Session and cookie management for KEATS authentication."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests
from cryptography.fernet import Fernet

from keats_scraper.utils.exceptions import SessionExpiredError
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class SessionManager:
    """Manages session cookies for KEATS access."""

    def __init__(self, cookie_file: Path, encryption_key: str | None = None):
        """
        Initialize session manager.

        Args:
            cookie_file: Path to store encrypted cookies
            encryption_key: Fernet encryption key (optional)
        """
        self.cookie_file = cookie_file
        self.encryption_key = encryption_key
        self._fernet = None

        if encryption_key:
            try:
                self._fernet = Fernet(encryption_key.encode())
            except Exception as e:
                logger.warning(f"Invalid encryption key, cookies will be stored unencrypted: {e}")

    def save_cookies(self, cookies: list[dict]) -> None:
        """
        Save cookies to file.

        Args:
            cookies: List of cookie dictionaries from Selenium
        """
        # Convert cookies to serializable format
        cookie_data = {
            "cookies": cookies,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        data = json.dumps(cookie_data).encode()

        if self._fernet:
            data = self._fernet.encrypt(data)

        self.cookie_file.write_bytes(data)
        # Restrict the on-disk cookie payload to owner-only read/write even
        # when Fernet encryption is in place -- defence in depth. Failures
        # on filesystems that do not honour POSIX modes (e.g. FAT/NTFS via
        # WSL) are logged and swallowed because they are not security
        # regressions on their native platform.
        try:
            os.chmod(self.cookie_file, 0o600)
        except OSError as e:
            logger.debug(
                "chmod 0o600 on %s failed (non-POSIX filesystem?): %s",
                self.cookie_file,
                e,
            )
        logger.info(f"Saved {len(cookies)} cookies to {self.cookie_file}")

    def load_cookies(self) -> list[dict] | None:
        """
        Load cookies from file.

        Returns:
            List of cookie dictionaries or None if not found
        """
        if not self.cookie_file.exists():
            logger.debug("No saved cookies found")
            return None

        try:
            data = self.cookie_file.read_bytes()

            if self._fernet:
                data = self._fernet.decrypt(data)

            cookie_data = json.loads(data.decode())
            cookies = cookie_data.get("cookies", [])

            logger.info(f"Loaded {len(cookies)} cookies from file")
            return cookies

        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            return None

    def clear_cookies(self) -> None:
        """Delete saved cookies."""
        if self.cookie_file.exists():
            self.cookie_file.unlink()
            logger.info("Cleared saved cookies")

    def apply_to_session(self, session: requests.Session, cookies: list[dict]) -> None:
        """
        Apply cookies to a requests session.

        Args:
            session: requests.Session to apply cookies to
            cookies: List of cookie dictionaries
        """
        for cookie in cookies:
            session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain", ""),
                path=cookie.get("path", "/"),
            )

    def create_session_with_cookies(self, cookies: list[dict]) -> requests.Session:
        """
        Create a new requests session with loaded cookies.

        Args:
            cookies: List of cookie dictionaries

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Set common headers
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        self.apply_to_session(session, cookies)
        return session

    def validate_session(
        self, session: requests.Session, check_url: str
    ) -> bool:
        """
        Check if session is still valid.

        Args:
            session: requests.Session to validate
            check_url: URL to check (should require auth)

        Returns:
            True if session is valid. Non-redirect failures (network
            errors, unexpected non-200 responses) are reported as
            False so callers can decide whether to retry.

        Raises:
            SessionExpiredError: When the server redirects to the login
                page on any of the HTTP redirect codes (301, 302, 303,
                307, 308). This is the typed expiry signal so callers
                can distinguish "session is gone" from "network hiccup".
        """
        try:
            response = session.get(check_url, allow_redirects=False, timeout=10)

            # If we get redirected to login, session is invalid. All five
            # redirect codes count: 301/302/303 are the classic forms,
            # 307/308 preserve the original method and are what Moodle
            # can emit when an auth middleware bounces POSTs back to the
            # login form.
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location", "")
                if "login" in location.lower():
                    raise SessionExpiredError(
                        "Session expired - redirected to login"
                    )

            # 200 means authenticated
            return response.status_code == 200

        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return False
