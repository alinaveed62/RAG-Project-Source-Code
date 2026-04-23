"""Tests for the FAISS index builder."""

import numpy as np
import pytest

from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder


class TestFAISSIndexBuilderBuild:
    """Tests for building FAISS indices."""

    def test_build_index_creates_index(self, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        index = builder.build_index(sample_embeddings, sample_chunks)
        assert index.ntotal == 5

    def test_build_index_stores_metadata(self, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        builder.build_index(sample_embeddings, sample_chunks)
        assert len(builder.metadata) == 5
        assert builder.metadata[0]["id"] == "doc_abc123_chunk_0"

    def test_build_index_correct_dimension(self, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        index = builder.build_index(sample_embeddings, sample_chunks)
        assert index.d == 384

    def test_search_returns_results(self, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        index = builder.build_index(sample_embeddings, sample_chunks)

        query = sample_embeddings[0:1]
        scores, indices = index.search(query, 3)
        assert indices.shape == (1, 3)
        assert indices[0][0] == 0  # Closest match should be itself


class TestFAISSIndexBuilderDimGuard:
    """Tests for the optional expected_dim sanity check and ndim guard."""

    def test_default_no_guard(self, sample_embeddings, sample_chunks):
        """Omitting expected_dim keeps legacy (unchecked) behaviour."""
        builder = FAISSIndexBuilder()
        assert builder.expected_dim is None
        builder.build_index(sample_embeddings, sample_chunks)
        assert builder.index.ntotal == 5

    def test_matching_expected_dim_passes(self, sample_embeddings, sample_chunks):
        """expected_dim equal to the actual dim does not raise."""
        builder = FAISSIndexBuilder(expected_dim=384)
        builder.build_index(sample_embeddings, sample_chunks)
        assert builder.index.ntotal == 5

    def test_mismatched_expected_dim_raises(
        self, sample_embeddings, sample_chunks
    ):
        """expected_dim different from actual dim raises ValueError with a
        helpful message naming both dimensions."""
        builder = FAISSIndexBuilder(expected_dim=768)
        with pytest.raises(
            ValueError,
            match=r"Embedding dim mismatch: got 384, expected 768",
        ):
            builder.build_index(sample_embeddings, sample_chunks)

    def test_one_dimensional_embeddings_raise(self, sample_chunks):
        """A 1-D embeddings array must fail loudly instead of raising a
        cryptic IndexError when the guard reads shape[1]."""
        flat = np.zeros(384, dtype=np.float32)
        builder = FAISSIndexBuilder(expected_dim=384)
        with pytest.raises(
            ValueError,
            match=r"Embeddings must be a 2-D array",
        ):
            builder.build_index(flat, sample_chunks[:1])

    def test_metadata_row_count_mismatch_raises(
        self, sample_embeddings, sample_chunks
    ):
        """len(metadata) must equal embeddings.shape[0]. Without this guard
        the mismatch would surface as an IndexError at query time inside
        FAISSRetriever.retrieve rather than at build time, making the
        root cause (a corrupted build pipeline) hard to diagnose."""
        builder = FAISSIndexBuilder()
        truncated_metadata = sample_chunks[:3]  # 5 embeddings, 3 metadata
        with pytest.raises(
            ValueError,
            match=r"metadata length 3 does not match embeddings rows 5",
        ):
            builder.build_index(sample_embeddings, truncated_metadata)


class TestFAISSIndexBuilderSaveLoad:
    """Tests for saving and loading FAISS indices."""

    def test_save_creates_files(self, tmp_path, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        builder.build_index(sample_embeddings, sample_chunks)
        builder.save(tmp_path / "index")

        assert (tmp_path / "index" / "index.faiss").exists()
        assert (tmp_path / "index" / "metadata.json").exists()

    def test_save_without_build_raises(self, tmp_path):
        builder = FAISSIndexBuilder()
        with pytest.raises(ValueError, match="No index built"):
            builder.save(tmp_path / "index")

    def test_load_restores_index(self, tmp_path, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        builder.build_index(sample_embeddings, sample_chunks)
        builder.save(tmp_path / "index")

        new_builder = FAISSIndexBuilder()
        index, metadata = new_builder.load(tmp_path / "index")
        assert index.ntotal == 5
        assert len(metadata) == 5

    def test_roundtrip_search_consistency(self, tmp_path, sample_embeddings, sample_chunks):
        builder = FAISSIndexBuilder()
        builder.build_index(sample_embeddings, sample_chunks)

        query = sample_embeddings[2:3]
        scores_before, indices_before = builder.index.search(query, 3)

        builder.save(tmp_path / "index")
        new_builder = FAISSIndexBuilder()
        index, _ = new_builder.load(tmp_path / "index")
        scores_after, indices_after = index.search(query, 3)

        np.testing.assert_array_equal(indices_before, indices_after)
        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_load_detects_out_of_sync_metadata(
        self, tmp_path, sample_embeddings, sample_chunks
    ):
        """save writes index.faiss and metadata.json together but
        load reads them independently. If the two files disagree on
        row count (because of a partial re-encode, a manual edit, or
        a disk-full mid-write), queries would silently return the
        wrong chunks or raise IndexError deep inside FAISSRetriever.
        load must fail loudly at the mismatch instead.
        """
        import json

        builder = FAISSIndexBuilder()
        builder.build_index(sample_embeddings, sample_chunks)
        builder.save(tmp_path / "index")

        metadata_path = tmp_path / "index" / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata[:-1], f, ensure_ascii=False)  # drop one entry

        new_builder = FAISSIndexBuilder()
        with pytest.raises(
            ValueError,
            match=r"index and metadata are out of sync",
        ):
            new_builder.load(tmp_path / "index")
