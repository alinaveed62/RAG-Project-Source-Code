"""Central output paths for evaluation artefacts.

Several experiment scripts write figures and LaTeX tables into the
report's Figures and tables folders. Centralising the report
directory name here avoids hard-coding it in each module and lets
the EVAL_REPORT_DIR environment variable redirect the output for CI
or a reorganised checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REPORT_DIR_NAME = "MuhammadNaveed_23015182_JeroenKeppens_BSPR_2025_26"


def _resolve_report_dir() -> Path:
    override = os.environ.get("EVAL_REPORT_DIR")
    if override:
        return Path(override)
    return _PROJECT_ROOT / _DEFAULT_REPORT_DIR_NAME


REPORT_DIR: Path = _resolve_report_dir()
REPORT_FIGURES_DIR: Path = REPORT_DIR / "Figures"
REPORT_TABLES_DIR: Path = REPORT_DIR / "tables"
