"""Pin the rag_pipeline public API against silent drift.

rag_pipeline/__init__.py exports a curated __all__ so consumers
(flask_app, run_local.py, evaluation/) can depend on a stable
surface. These tests fail if a future refactor deletes a public name or
accidentally leaks a new one without updating the contract.
"""

import rag_pipeline

_EXPECTED_PUBLIC_API = {
    "Citation",
    "DEFAULT_EMBEDDING_MODEL",
    "GENERATION_ERROR_ANSWER",
    "LOW_CONFIDENCE_ANSWER",
    "RAGConfig",
    "RAGPipeline",
    "RAGResponse",
    "RetrievalResult",
    "parse_citations",
    "strip_citations",
}


class TestPublicAPISurface:
    def test_all_names_resolve_on_package(self):
        for name in rag_pipeline.__all__:
            assert hasattr(rag_pipeline, name), (
                f"rag_pipeline.__all__ lists {name!r} but the attribute is "
                "missing from the package."
            )
            assert getattr(rag_pipeline, name) is not None, (
                f"rag_pipeline.{name} resolved to None; the re-export chain "
                "is broken."
            )

    def test_all_matches_expected_set(self):
        actual = set(rag_pipeline.__all__)
        assert actual == _EXPECTED_PUBLIC_API, (
            "rag_pipeline public API has drifted. "
            f"Missing: {_EXPECTED_PUBLIC_API - actual}; "
            f"unexpectedly added: {actual - _EXPECTED_PUBLIC_API}."
        )
