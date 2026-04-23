"""Tests for SSOHandler."""

from unittest.mock import Mock, patch

import pytest
import requests

from keats_scraper.auth.session_manager import SessionManager
from keats_scraper.auth.sso_handler import SSOHandler
from keats_scraper.config import AuthConfig, KEATSConfig, ScraperConfig
from keats_scraper.utils.exceptions import AuthenticationError, SessionExpiredError


class TestSSOHandlerInit:
    """Tests for SSOHandler initialization."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.base_url = "https://keats.kcl.ac.uk"
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.keats.course_url = "https://keats.kcl.ac.uk/course/view.php?id=123"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        config.auth.login_timeout = 180
        config.auth.session_check_url = "https://keats.kcl.ac.uk/my/"
        return config

    def test_init_sets_config(self, mock_config):
        """Test config is set correctly."""
        handler = SSOHandler(mock_config)
        assert handler.config is mock_config

    def test_init_creates_session_manager(self, mock_config):
        """Test SessionManager is created."""
        handler = SSOHandler(mock_config)
        assert handler.session_manager is not None
        assert isinstance(handler.session_manager, SessionManager)

    def test_init_driver_is_none(self, mock_config):
        """Test driver is initially None."""
        handler = SSOHandler(mock_config)
        assert handler._driver is None

    def test_init_session_manager_uses_config(self, mock_config):
        """Test SessionManager uses config values."""
        mock_config.auth.cookie_file = "/custom/path/cookies.dat"
        mock_config.auth.encryption_key = "test_key"

        with patch("keats_scraper.auth.sso_handler.SessionManager") as mock_sm:
            handler = SSOHandler(mock_config)
            mock_sm.assert_called_once_with(
                cookie_file="/custom/path/cookies.dat",
                encryption_key="test_key",
            )


class TestCreateDriver:
    """Tests for _create_driver method."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create handler with mocked config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        config.auth.login_timeout = 180
        return SSOHandler(config)

    @patch("keats_scraper.auth.sso_handler.ChromeDriverManager")
    @patch("keats_scraper.auth.sso_handler.webdriver.Chrome")
    @patch("keats_scraper.auth.sso_handler.Service")
    @patch("keats_scraper.auth.sso_handler.Options")
    def test_create_driver_returns_chrome(
        self, mock_options, mock_service, mock_chrome, mock_manager, handler
    ):
        """Test _create_driver returns Chrome WebDriver."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        result = handler._create_driver()

        assert result is mock_driver

    @patch("keats_scraper.auth.sso_handler.ChromeDriverManager")
    @patch("keats_scraper.auth.sso_handler.webdriver.Chrome")
    @patch("keats_scraper.auth.sso_handler.Service")
    @patch("keats_scraper.auth.sso_handler.Options")
    def test_create_driver_adds_stability_options(
        self, mock_options, mock_service, mock_chrome, mock_manager, handler
    ):
        """Test stability options are added."""
        mock_options_instance = Mock()
        mock_options.return_value = mock_options_instance
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        handler._create_driver()

        # Check for common stability arguments
        calls = [str(c) for c in mock_options_instance.add_argument.call_args_list]
        call_str = " ".join(calls)
        assert "--no-sandbox" in call_str
        assert "--disable-dev-shm-usage" in call_str

    @patch("keats_scraper.auth.sso_handler.ChromeDriverManager")
    @patch("keats_scraper.auth.sso_handler.webdriver.Chrome")
    @patch("keats_scraper.auth.sso_handler.Service")
    @patch("keats_scraper.auth.sso_handler.Options")
    def test_create_driver_uses_chrome_driver_manager(
        self, mock_options, mock_service, mock_chrome, mock_manager, handler
    ):
        """Test ChromeDriverManager is used."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        handler._create_driver()

        mock_manager.assert_called_once()
        mock_manager.return_value.install.assert_called_once()

    @patch("keats_scraper.auth.sso_handler.ChromeDriverManager")
    @patch("keats_scraper.auth.sso_handler.webdriver.Chrome")
    @patch("keats_scraper.auth.sso_handler.Service")
    @patch("keats_scraper.auth.sso_handler.Options")
    def test_create_driver_executes_anti_detection_script(
        self, mock_options, mock_service, mock_chrome, mock_manager, handler
    ):
        """Test anti-detection script is executed."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        handler._create_driver()

        mock_driver.execute_script.assert_called_once()
        script = mock_driver.execute_script.call_args[0][0]
        assert "webdriver" in script


class TestLoginInteractive:
    """Tests for login_interactive method."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create handler with mocked config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        config.auth.login_timeout = 2  # Short timeout for tests
        return SSOHandler(config)

    @patch.object(SSOHandler, "_create_driver")
    def test_login_success(self, mock_create_driver, handler):
        """Test successful login flow."""
        mock_driver = Mock()
        # Simulate URL changing to success page
        mock_driver.current_url = "https://keats.kcl.ac.uk/my/"
        mock_driver.get_cookies.return_value = [
            {"name": "session", "value": "abc123"}
        ]
        mock_create_driver.return_value = mock_driver

        with patch.object(handler.session_manager, "save_cookies"):
            cookies = handler.login_interactive()

        assert len(cookies) == 1
        assert cookies[0]["name"] == "session"
        mock_driver.quit.assert_called_once()

    @patch.object(SSOHandler, "_create_driver")
    def test_login_navigates_to_login_url(self, mock_create_driver, handler):
        """Test driver navigates to login URL."""
        mock_driver = Mock()
        mock_driver.current_url = "https://keats.kcl.ac.uk/my/"
        mock_driver.get_cookies.return_value = []
        mock_create_driver.return_value = mock_driver

        with patch.object(handler.session_manager, "save_cookies"):
            handler.login_interactive()

        mock_driver.get.assert_called_once_with(handler.config.keats.login_url)

    @patch.object(SSOHandler, "_create_driver")
    @patch("time.time")
    @patch("time.sleep")
    def test_login_timeout_raises_error(
        self, mock_sleep, mock_time, mock_create_driver, handler
    ):
        """Test timeout raises AuthenticationError."""
        mock_driver = Mock()
        # URL never changes to success
        mock_driver.current_url = "https://login.kcl.ac.uk/sso"
        mock_create_driver.return_value = mock_driver

        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 1, 2, 3]  # Start at 0, then pass timeout

        with pytest.raises(AuthenticationError) as exc_info:
            handler.login_interactive()

        assert "timed out" in str(exc_info.value)
        mock_driver.quit.assert_called_once()

    @patch.object(SSOHandler, "_create_driver")
    def test_login_saves_cookies(self, mock_create_driver, handler):
        """Test cookies are saved after login."""
        mock_driver = Mock()
        mock_driver.current_url = "https://keats.kcl.ac.uk/course/view.php?id=123"
        mock_driver.get_cookies.return_value = [
            {"name": "session", "value": "xyz789"}
        ]
        mock_create_driver.return_value = mock_driver

        with patch.object(handler.session_manager, "save_cookies") as mock_save:
            handler.login_interactive()
            mock_save.assert_called_once()
            saved_cookies = mock_save.call_args[0][0]
            assert saved_cookies[0]["value"] == "xyz789"

    @patch.object(SSOHandler, "_create_driver")
    def test_login_quits_driver_on_error(self, mock_create_driver, handler):
        """Test driver is quit even on error."""
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("Browser error")
        mock_create_driver.return_value = mock_driver

        with pytest.raises(AuthenticationError):
            handler.login_interactive()

        mock_driver.quit.assert_called_once()

    @patch.object(SSOHandler, "_create_driver")
    def test_login_sets_driver_to_none_after(self, mock_create_driver, handler):
        """Test _driver is set to None after login."""
        mock_driver = Mock()
        mock_driver.current_url = "https://keats.kcl.ac.uk/my/"
        mock_driver.get_cookies.return_value = []
        mock_create_driver.return_value = mock_driver

        with patch.object(handler.session_manager, "save_cookies"):
            handler.login_interactive()

        assert handler._driver is None

    @patch.object(SSOHandler, "_create_driver")
    def test_login_detects_course_page(self, mock_create_driver, handler):
        """Test course page URL is detected as success."""
        mock_driver = Mock()
        mock_driver.current_url = "https://keats.kcl.ac.uk/course/view.php?id=456"
        mock_driver.get_cookies.return_value = []
        mock_create_driver.return_value = mock_driver

        with patch.object(handler.session_manager, "save_cookies"):
            # Should not raise - course page is success indicator
            handler.login_interactive()


class TestGetValidSession:
    """Tests for get_valid_session method."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create handler with mocked config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        config.auth.login_timeout = 180
        config.auth.session_check_url = "https://keats.kcl.ac.uk/my/"
        return SSOHandler(config)

    def test_get_session_uses_cached_cookies(self, handler):
        """Test cached cookies are used when valid."""
        cached_cookies = [{"name": "session", "value": "cached123"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=cached_cookies):
            mock_session = Mock()
            with patch.object(
                handler.session_manager, "create_session_with_cookies", return_value=mock_session
            ):
                with patch.object(
                    handler.session_manager, "validate_session", return_value=True
                ):
                    result = handler.get_valid_session()

        assert result is mock_session

    def test_get_session_validates_cached(self, handler):
        """Test cached session is validated."""
        cached_cookies = [{"name": "session", "value": "cached123"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=cached_cookies):
            mock_session = Mock()
            with patch.object(
                handler.session_manager, "create_session_with_cookies", return_value=mock_session
            ):
                with patch.object(
                    handler.session_manager, "validate_session", return_value=True
                ) as mock_validate:
                    handler.get_valid_session()

        mock_validate.assert_called_once_with(mock_session, handler.config.auth.session_check_url)

    def test_get_session_fresh_login_on_session_expired_error(self, handler):
        """When validate_session raises SessionExpiredError, fall through to a
        fresh login (this is the explicit typed-expiry path, distinct from a
        plain False return)."""
        cached_cookies = [{"name": "session", "value": "expired123"}]
        fresh_cookies = [{"name": "session", "value": "fresh456"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=cached_cookies):
            mock_cached_session = Mock()
            mock_fresh_session = Mock()

            with patch.object(
                handler.session_manager,
                "create_session_with_cookies",
                side_effect=[mock_cached_session, mock_fresh_session],
            ):
                with patch.object(
                    handler.session_manager,
                    "validate_session",
                    side_effect=SessionExpiredError("expired"),
                ):
                    with patch.object(
                        handler, "login_interactive", return_value=fresh_cookies
                    ):
                        result = handler.get_valid_session()

        assert result is mock_fresh_session

    def test_get_session_fresh_login_on_expired(self, handler):
        """Test fresh login when cached session expired."""
        cached_cookies = [{"name": "session", "value": "expired123"}]
        fresh_cookies = [{"name": "session", "value": "fresh456"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=cached_cookies):
            mock_cached_session = Mock()
            mock_fresh_session = Mock()

            with patch.object(
                handler.session_manager,
                "create_session_with_cookies",
                side_effect=[mock_cached_session, mock_fresh_session],
            ):
                with patch.object(
                    handler.session_manager, "validate_session", return_value=False
                ):
                    with patch.object(
                        handler, "login_interactive", return_value=fresh_cookies
                    ):
                        result = handler.get_valid_session()

        assert result is mock_fresh_session

    def test_get_session_no_cached_cookies(self, handler):
        """Test fresh login when no cached cookies."""
        fresh_cookies = [{"name": "session", "value": "fresh123"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=None):
            mock_session = Mock()
            with patch.object(
                handler.session_manager, "create_session_with_cookies", return_value=mock_session
            ):
                with patch.object(handler, "login_interactive", return_value=fresh_cookies):
                    result = handler.get_valid_session()

        assert result is mock_session

    def test_get_session_force_login(self, handler):
        """Test force_login bypasses cached cookies."""
        fresh_cookies = [{"name": "session", "value": "fresh123"}]

        with patch.object(handler.session_manager, "load_cookies") as mock_load:
            mock_session = Mock()
            with patch.object(
                handler.session_manager, "create_session_with_cookies", return_value=mock_session
            ):
                with patch.object(handler, "login_interactive", return_value=fresh_cookies):
                    handler.get_valid_session(force_login=True)

        # load_cookies should not be called when force_login=True
        mock_load.assert_not_called()

    def test_get_session_returns_requests_session(self, handler):
        """Test returns requests.Session type."""
        fresh_cookies = [{"name": "session", "value": "fresh123"}]

        with patch.object(handler.session_manager, "load_cookies", return_value=None):
            mock_session = Mock(spec=requests.Session)
            with patch.object(
                handler.session_manager, "create_session_with_cookies", return_value=mock_session
            ):
                with patch.object(handler, "login_interactive", return_value=fresh_cookies):
                    result = handler.get_valid_session()

        assert result is mock_session


class TestLogout:
    """Tests for logout method."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create handler with mocked config."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        return SSOHandler(config)

    def test_logout_clears_cookies(self, handler):
        """Test logout clears saved cookies."""
        with patch.object(handler.session_manager, "clear_cookies") as mock_clear:
            handler.logout()
            mock_clear.assert_called_once()


class TestIntegration:
    """Integration tests for SSOHandler."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create handler with real SessionManager."""
        config = Mock(spec=ScraperConfig)
        config.keats = Mock(spec=KEATSConfig)
        config.keats.login_url = "https://keats.kcl.ac.uk/login"
        config.auth = Mock(spec=AuthConfig)
        config.auth.cookie_file = tmp_path / "cookies.dat"
        config.auth.encryption_key = None
        config.auth.login_timeout = 180
        config.auth.session_check_url = "https://keats.kcl.ac.uk/my/"
        return SSOHandler(config)

    def test_full_login_logout_cycle(self, handler):
        """Test complete login and logout cycle."""
        # Mock the driver and login
        with patch.object(SSOHandler, "_create_driver") as mock_create:
            mock_driver = Mock()
            mock_driver.current_url = "https://keats.kcl.ac.uk/my/"
            mock_driver.get_cookies.return_value = [
                {"name": "MoodleSession", "value": "test123"}
            ]
            mock_create.return_value = mock_driver

            # Login
            cookies = handler.login_interactive()
            assert len(cookies) == 1

            # Cookies should be saved
            assert handler.config.auth.cookie_file.exists()

            # Logout
            handler.logout()

            # Cookies should be cleared
            assert not handler.config.auth.cookie_file.exists()
