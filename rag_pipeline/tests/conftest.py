"""Shared fixtures for RAG pipeline tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

SAMPLE_CHUNKS = [
    {
        "id": "doc_abc123_chunk_0",
        "text": "Students must attend all scheduled lectures and tutorials.",
        "source": "https://keats.kcl.ac.uk/mod/page/view.php?id=1",
        "title": "Attendance Policy",
        "section": "Teaching & Assessment",
        "heading_path": ["Teaching and Assessment", "Attendance"],
    },
    {
        "id": "doc_abc123_chunk_1",
        "text": "Extenuating circumstances claims must be submitted within 7 days.",
        "source": "https://keats.kcl.ac.uk/mod/page/view.php?id=2",
        "title": "EC Policy",
        "section": "Teaching & Assessment",
        "heading_path": ["Teaching and Assessment", "Extenuating Circumstances"],
    },
    {
        "id": "doc_def456_chunk_0",
        "text": "The BSc Computer Science programme consists of 360 credits.",
        "source": "https://keats.kcl.ac.uk/mod/page/view.php?id=3",
        "title": "BSc Computer Science",
        "section": "Programmes",
        "heading_path": ["Programmes", "BSc Computer Science"],
    },
    {
        "id": "doc_ghi789_chunk_0",
        "text": "Student representatives are elected each academic year.",
        "source": "https://keats.kcl.ac.uk/mod/page/view.php?id=4",
        "title": "Student Voice",
        "section": "Student Voice",
        "heading_path": ["Student Voice", "Representation"],
    },
    {
        "id": "doc_jkl012_chunk_0",
        "text": "The Department of Informatics is part of the Faculty of NMES.",
        "source": "https://keats.kcl.ac.uk/mod/page/view.php?id=5",
        "title": "Department Info",
        "section": "Faculty & Dept",
        "heading_path": [],
    },
]


@pytest.fixture
def sample_chunks():
    """Return sample chunk dicts."""
    return SAMPLE_CHUNKS.copy()


@pytest.fixture
def chunks_jsonl_path(sample_chunks):
    """Write sample chunks to a temp JSONL file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for chunk in sample_chunks:
            f.write(json.dumps(chunk) + "\n")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_embeddings():
    """Return normalized random embeddings (5 x 384)."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms
