"""Tests for evaluation/config.py: central report paths."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def reset_config_module():
    """Reload evaluation.config after each test so env-var changes don't leak."""
    import evaluation.config as cfg

    yield cfg
    importlib.reload(cfg)


class TestReportDirDefaults:
    def test_default_report_dir_under_project_root(self, reset_config_module, monkeypatch):
        monkeypatch.delenv("EVAL_REPORT_DIR", raising=False)
        cfg = importlib.reload(reset_config_module)
        assert cfg.REPORT_DIR.name == "MuhammadNaveed_23015182_JeroenKeppens_BSPR_2025_26"
        assert cfg.REPORT_DIR.parent == Path(__file__).resolve().parent.parent.parent

    def test_figures_and_tables_subpaths(self, reset_config_module, monkeypatch):
        monkeypatch.delenv("EVAL_REPORT_DIR", raising=False)
        cfg = importlib.reload(reset_config_module)
        assert cfg.REPORT_FIGURES_DIR == cfg.REPORT_DIR / "Figures"
        assert cfg.REPORT_TABLES_DIR == cfg.REPORT_DIR / "tables"


class TestEnvOverride:
    def test_env_var_override_sets_report_dir(self, reset_config_module, monkeypatch, tmp_path):
        monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path / "custom"))
        cfg = importlib.reload(reset_config_module)
        assert cfg.REPORT_DIR == tmp_path / "custom"
        assert cfg.REPORT_FIGURES_DIR == tmp_path / "custom" / "Figures"
        assert cfg.REPORT_TABLES_DIR == tmp_path / "custom" / "tables"

    def test_empty_env_var_falls_back_to_default(
        self, reset_config_module, monkeypatch
    ):
        monkeypatch.setenv("EVAL_REPORT_DIR", "")
        cfg = importlib.reload(reset_config_module)
        assert cfg.REPORT_DIR.name == "MuhammadNaveed_23015182_JeroenKeppens_BSPR_2025_26"
