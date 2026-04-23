"""Tests for SessionManager."""

import json
import sys
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
import requests
from cryptography.fernet import Fernet

from keats_scraper.auth.session_manager import SessionManager
from keats_scraper.utils.exceptions import SessionExpiredError


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_sets_cookie_file(self, tmp_path):
        """Test cookie_file is set correctly."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        assert manager.cookie_file == cookie_file

    def test_init_no_encryption_key(self, tmp_path):
        """Test init without encryption key."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        assert manager.encryption_key is None
        assert manager._fernet is None

    def test_init_with_valid_encryption_key(self, tmp_path):
        """Test init with valid Fernet encryption key."""
        cookie_file = tmp_path / "cookies.dat"
        key = Fernet.generate_key().decode()
        manager = SessionManager(cookie_file, encryption_key=key)
        assert manager.encryption_key == key
        assert manager._fernet is not None

    def test_init_with_invalid_encryption_key(self, tmp_path):
        """Test init with invalid encryption key falls back gracefully."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file, encryption_key="invalid_key")
        assert manager.encryption_key == "invalid_key"
        assert manager._fernet is None


class TestSaveCookies:
    """Tests for save_cookies method."""

    @pytest.fixture
    def sample_cookies(self):
        """Sample cookie list."""
        return [
            {"name": "session_id", "value": "abc123", "domain": ".example.com", "path": "/"},
            {"name": "user_token", "value": "xyz789", "domain": ".example.com", "path": "/app"},
        ]

    def test_save_creates_file(self, tmp_path, sample_cookies):
        """Test cookie file is created."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        manager.save_cookies(sample_cookies)
        assert cookie_file.exists()

    def test_save_without_encryption(self, tmp_path, sample_cookies):
        """Test cookies are saved as JSON without encryption."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        manager.save_cookies(sample_cookies)

        data = json.loads(cookie_file.read_text())
        assert "cookies" in data
        assert len(data["cookies"]) == 2
        assert data["cookies"][0]["name"] == "session_id"

    def test_save_with_encryption(self, tmp_path, sample_cookies):
        """Test cookies are encrypted when key provided."""
        cookie_file = tmp_path / "cookies.dat"
        key = Fernet.generate_key().decode()
        manager = SessionManager(cookie_file, encryption_key=key)
        manager.save_cookies(sample_cookies)

        # Raw content should not be valid JSON (encrypted)
        raw_data = cookie_file.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw_data.decode())

    def test_save_includes_timestamp(self, tmp_path, sample_cookies):
        """Test saved data includes timestamp."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        before = datetime.now(UTC)
        manager.save_cookies(sample_cookies)
        after = datetime.now(UTC)

        data = json.loads(cookie_file.read_text())
        assert "saved_at" in data
        saved_at = datetime.fromisoformat(data["saved_at"])
        assert before <= saved_at <= after

    def test_save_empty_list(self, tmp_path):
        """Test saving empty cookie list."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        manager.save_cookies([])

        data = json.loads(cookie_file.read_text())
        assert data["cookies"] == []

    def test_save_overwrites_existing(self, tmp_path, sample_cookies):
        """Test saving overwrites existing cookies."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        # Save first set
        manager.save_cookies(sample_cookies)

        # Save new set
        new_cookies = [{"name": "new_cookie", "value": "new_value"}]
        manager.save_cookies(new_cookies)

        data = json.loads(cookie_file.read_text())
        assert len(data["cookies"]) == 1
        assert data["cookies"][0]["name"] == "new_cookie"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
    def test_save_sets_owner_only_permissions(self, tmp_path, sample_cookies):
        """Cookie file must be 0o600 on POSIX so other local users cannot
        read the (possibly encrypted) session payload."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        manager.save_cookies(sample_cookies)

        mode = cookie_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_save_swallows_chmod_failure(self, tmp_path, sample_cookies, mocker):
        """If the underlying filesystem refuses the chmod (FAT/NTFS under
        WSL, network mounts), save_cookies must still succeed rather
        than crash the scraper on a platform concern."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mocker.patch(
            "keats_scraper.auth.session_manager.os.chmod",
            side_effect=OSError("Operation not supported"),
        )

        manager.save_cookies(sample_cookies)

        assert cookie_file.exists()
        data = json.loads(cookie_file.read_text())
        assert len(data["cookies"]) == 2


class TestLoadCookies:
    """Tests for load_cookies method."""

    @pytest.fixture
    def sample_cookies(self):
        """Sample cookie list."""
        return [
            {"name": "session_id", "value": "abc123", "domain": ".example.com"},
            {"name": "user_token", "value": "xyz789", "domain": ".example.com"},
        ]

    def test_load_no_file_returns_none(self, tmp_path):
        """Test None returned when no cookie file exists."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)
        result = manager.load_cookies()
        assert result is None

    def test_load_success(self, tmp_path, sample_cookies):
        """Test successful cookie loading."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        # Save first
        manager.save_cookies(sample_cookies)

        # Then load
        loaded = manager.load_cookies()
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["name"] == "session_id"

    def test_load_with_encryption(self, tmp_path, sample_cookies):
        """Test loading encrypted cookies."""
        cookie_file = tmp_path / "cookies.dat"
        key = Fernet.generate_key().decode()
        manager = SessionManager(cookie_file, encryption_key=key)

        # Save encrypted
        manager.save_cookies(sample_cookies)

        # Load with same key
        loaded = manager.load_cookies()
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_wrong_encryption_key_returns_none(self, tmp_path, sample_cookies):
        """Test None returned when decryption fails."""
        cookie_file = tmp_path / "cookies.dat"
        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()

        # Save with key1
        manager1 = SessionManager(cookie_file, encryption_key=key1)
        manager1.save_cookies(sample_cookies)

        # Try to load with key2
        manager2 = SessionManager(cookie_file, encryption_key=key2)
        result = manager2.load_cookies()
        assert result is None

    def test_load_corrupted_file_returns_none(self, tmp_path):
        """Test None returned for corrupted file."""
        cookie_file = tmp_path / "cookies.dat"
        cookie_file.write_text("not valid json {{{{")

        manager = SessionManager(cookie_file)
        result = manager.load_cookies()
        assert result is None

    def test_load_empty_cookies_key(self, tmp_path):
        """Test loading when cookies key is empty."""
        cookie_file = tmp_path / "cookies.dat"
        data = {"saved_at": datetime.now(UTC).isoformat()}
        cookie_file.write_text(json.dumps(data))

        manager = SessionManager(cookie_file)
        loaded = manager.load_cookies()
        assert loaded == []


class TestClearCookies:
    """Tests for clear_cookies method."""

    def test_clear_deletes_file(self, tmp_path):
        """Test cookie file is deleted."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        # Create file
        manager.save_cookies([{"name": "test", "value": "123"}])
        assert cookie_file.exists()

        # Clear
        manager.clear_cookies()
        assert not cookie_file.exists()

    def test_clear_no_file_no_error(self, tmp_path):
        """Test no error when clearing non-existent file."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        # Should not raise
        manager.clear_cookies()


class TestApplyToSession:
    """Tests for apply_to_session method."""

    def test_apply_single_cookie(self, tmp_path):
        """Test applying a single cookie."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        cookies = [{"name": "session_id", "value": "abc123", "domain": ".example.com", "path": "/"}]

        manager.apply_to_session(session, cookies)

        assert "session_id" in session.cookies.keys()

    def test_apply_multiple_cookies(self, tmp_path):
        """Test applying multiple cookies."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        cookies = [
            {"name": "cookie1", "value": "value1"},
            {"name": "cookie2", "value": "value2"},
            {"name": "cookie3", "value": "value3"},
        ]

        manager.apply_to_session(session, cookies)

        assert len(session.cookies) == 3

    def test_apply_cookie_values(self, tmp_path):
        """Test cookie values are correct."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        cookies = [{"name": "test_cookie", "value": "test_value_123"}]

        manager.apply_to_session(session, cookies)

        assert session.cookies.get("test_cookie") == "test_value_123"

    def test_apply_empty_cookies(self, tmp_path):
        """Test applying empty cookie list."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        manager.apply_to_session(session, [])

        assert len(session.cookies) == 0

    def test_apply_missing_domain(self, tmp_path):
        """Test cookie without domain uses empty string."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        cookies = [{"name": "no_domain", "value": "value"}]

        # Should not raise
        manager.apply_to_session(session, cookies)
        assert "no_domain" in session.cookies.keys()

    def test_apply_missing_path(self, tmp_path):
        """Test cookie without path uses default."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = requests.Session()
        cookies = [{"name": "no_path", "value": "value", "domain": ".example.com"}]

        # Should not raise
        manager.apply_to_session(session, cookies)


class TestCreateSessionWithCookies:
    """Tests for create_session_with_cookies method."""

    def test_returns_session(self, tmp_path):
        """Test returns a configured requests.Session with no cookies and a User-Agent header."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = manager.create_session_with_cookies([])
        assert isinstance(session, requests.Session)
        assert len(session.cookies) == 0
        assert "User-Agent" in session.headers

    def test_session_has_user_agent(self, tmp_path):
        """Test session has User-Agent header."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = manager.create_session_with_cookies([])
        assert "User-Agent" in session.headers
        assert "Mozilla" in session.headers["User-Agent"]

    def test_session_has_accept_header(self, tmp_path):
        """Test session has Accept header."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = manager.create_session_with_cookies([])
        assert "Accept" in session.headers
        assert "text/html" in session.headers["Accept"]

    def test_session_has_accept_language(self, tmp_path):
        """Test session has Accept-Language header."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        session = manager.create_session_with_cookies([])
        assert "Accept-Language" in session.headers
        assert "en-US" in session.headers["Accept-Language"]

    def test_session_has_cookies(self, tmp_path):
        """Test session has cookies applied."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        cookies = [
            {"name": "test1", "value": "val1"},
            {"name": "test2", "value": "val2"},
        ]
        session = manager.create_session_with_cookies(cookies)

        assert len(session.cookies) == 2


class TestValidateSession:
    """Tests for validate_session method."""

    def test_valid_session_200(self, tmp_path):
        """Test True returned for 200 status."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        result = manager.validate_session(mock_session, "https://example.com/check")
        assert result is True

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    def test_redirect_to_login_raises_session_expired(self, tmp_path, status_code):
        """All five redirect codes (301/302/303/307/308) to a login URL
        raise SessionExpiredError so callers can distinguish expiry
        from transient failures."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = {"Location": "https://example.com/login"}
        mock_session.get.return_value = mock_response

        with pytest.raises(SessionExpiredError):
            manager.validate_session(mock_session, "https://example.com/check")

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    def test_redirect_not_to_login_invalid(self, tmp_path, status_code):
        """Test False returned for redirect not to login (still invalid for 200 check)."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = {"Location": "https://example.com/other"}
        mock_session.get.return_value = mock_response

        result = manager.validate_session(mock_session, "https://example.com/check")
        # Still False because we expect 200 for valid
        assert result is False

    def test_redirect_login_case_insensitive(self, tmp_path):
        """Test login detection is case insensitive."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {"Location": "https://example.com/LOGIN"}
        mock_session.get.return_value = mock_response

        with pytest.raises(SessionExpiredError):
            manager.validate_session(mock_session, "https://example.com/check")

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500])
    def test_error_status_invalid(self, tmp_path, status_code):
        """Test False returned for error status codes."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_session.get.return_value = mock_response

        result = manager.validate_session(mock_session, "https://example.com/check")
        assert result is False

    def test_exception_returns_false(self, tmp_path):
        """Test False returned when request raises exception."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_session.get.side_effect = requests.RequestException("Network error")

        result = manager.validate_session(mock_session, "https://example.com/check")
        assert result is False

    def test_timeout_exception_returns_false(self, tmp_path):
        """Test False returned on timeout."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_session.get.side_effect = requests.Timeout("Timeout")

        result = manager.validate_session(mock_session, "https://example.com/check")
        assert result is False

    def test_validate_uses_correct_params(self, tmp_path):
        """Test validate uses correct request parameters."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        manager.validate_session(mock_session, "https://example.com/check")

        mock_session.get.assert_called_once_with(
            "https://example.com/check",
            allow_redirects=False,
            timeout=10,
        )

    def test_redirect_no_location_header(self, tmp_path):
        """Test redirect without Location header is handled."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        mock_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {}  # No Location header
        mock_session.get.return_value = mock_response

        result = manager.validate_session(mock_session, "https://example.com/check")
        # Should not crash, and still returns False (not 200)
        assert result is False


class TestIntegration:
    """Integration tests for SessionManager."""

    def test_full_cookie_lifecycle(self, tmp_path):
        """Test complete save -> load -> apply -> clear cycle."""
        cookie_file = tmp_path / "cookies.dat"
        manager = SessionManager(cookie_file)

        # Original cookies
        original_cookies = [
            {"name": "session", "value": "mysession", "domain": ".test.com", "path": "/"},
            {"name": "user", "value": "testuser", "domain": ".test.com", "path": "/"},
        ]

        # Save
        manager.save_cookies(original_cookies)
        assert cookie_file.exists()

        # Load
        loaded = manager.load_cookies()
        assert len(loaded) == 2

        # Apply to session
        session = manager.create_session_with_cookies(loaded)
        assert len(session.cookies) == 2

        # Clear
        manager.clear_cookies()
        assert not cookie_file.exists()
        assert manager.load_cookies() is None

    def test_encrypted_lifecycle(self, tmp_path):
        """Test complete lifecycle with encryption."""
        cookie_file = tmp_path / "cookies.dat"
        key = Fernet.generate_key().decode()
        manager = SessionManager(cookie_file, encryption_key=key)

        cookies = [{"name": "secure", "value": "secret123"}]

        # Save encrypted
        manager.save_cookies(cookies)

        # Verify file is encrypted (not readable as JSON)
        raw = cookie_file.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw.decode())

        # Load with same manager
        loaded = manager.load_cookies()
        assert loaded[0]["value"] == "secret123"
