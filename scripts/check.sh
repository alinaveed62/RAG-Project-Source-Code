#!/usr/bin/env bash
# Static analysis gate: ruff + mypy. Run before committing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[check] ruff check"
ruff check rag_pipeline keats_scraper evaluation flask_app

echo "[check] mypy"
mypy rag_pipeline keats_scraper evaluation flask_app

echo "[check] ok"
