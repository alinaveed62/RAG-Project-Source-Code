"""Shared fixtures for KEATS scraper tests.

Most test modules define their own ``mock_config`` / ``sample_document`` /
``sample_chunks`` / ``sample_cookies`` fixtures inline (with class-scoped
state) and ``mock_sleep``/``mock_time`` are injected by ``@patch`` decorators
at the test-method level, so the only shared fixtures here are the two that
are genuinely consumed from multiple modules without shadowing.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from keats_scraper.config import RateLimitConfig


@pytest.fixture
def rate_limit_config():
    """Create RateLimitConfig for fast testing (no delays)."""
    return RateLimitConfig(
        requests_per_minute=10000,
        min_delay_seconds=0,
        max_delay_seconds=0.001,
        max_retries=3,
        backoff_factor=2.0,
    )


@pytest.fixture
def sample_moodle_page_html():
    """Sample Moodle page content HTML."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome Page - KEATS</title>
        <script>alert('test');</script>
        <style>.hidden { display: none; }</style>
    </head>
    <body>
        <nav class="navbar">Navigation content to remove</nav>
        <div id="page">
            <div id="region-main">
                <h1>Welcome to the Informatics Handbook</h1>
                <div class="content">
                    <p>This handbook contains important information for students.</p>
                    <h2>Key Information</h2>
                    <p>Please read all sections carefully.</p>
                    <table>
                        <tr><th>Topic</th><th>Page</th></tr>
                        <tr><td>Attendance</td><td>5</td></tr>
                        <tr><td>Assessment</td><td>10</td></tr>
                    </table>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
            </div>
        </div>
        <footer>Footer content</footer>
        <span class="accesshide">Screen reader text</span>
    </body>
    </html>
    """
