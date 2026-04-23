"""Tests for the chunk encoder."""

from unittest.mock import MagicMock, patch

import numpy as np

from rag_pipeline.embeddings.encoder import ChunkEncoder


class TestChunkEncoderLoadChunks:
    """Tests for loading chunks from JSONL."""

    def test_load_chunks_returns_all_chunks(self, chunks_jsonl_path):
        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer"):
            encoder = ChunkEncoder()
        chunks = encoder.load_chunks(chunks_jsonl_path)
        assert len(chunks) == 5

    def test_load_chunks_preserves_fields(self, chunks_jsonl_path):
        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer"):
            encoder = ChunkEncoder()
        chunks = encoder.load_chunks(chunks_jsonl_path)
        assert chunks[0]["id"] == "doc_abc123_chunk_0"
        assert chunks[0]["title"] == "Attendance Policy"
        assert chunks[0]["section"] == "Teaching & Assessment"

    def test_load_chunks_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer"):
            encoder = ChunkEncoder()
        assert encoder.load_chunks(path) == []

    def test_load_chunks_skips_blank_lines(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            '{"id": "a", "text": "hello", "source": "", "title": "", "section": ""}\n'
            "\n"
            '{"id": "b", "text": "world", "source": "", "title": "", "section": ""}\n'
        )
        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer"):
            encoder = ChunkEncoder()
        chunks = encoder.load_chunks(path)
        assert len(chunks) == 2


class TestChunkEncoderEncode:
    """Tests for encoding chunks and queries."""

    def test_encode_chunks_shape(self, sample_chunks):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)

        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer", return_value=mock_model):
            encoder = ChunkEncoder()

        result = encoder.encode_chunks(sample_chunks)
        assert result.shape == (5, 384)
        mock_model.encode.assert_called_once()

    def test_encode_chunks_passes_texts(self, sample_chunks):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)

        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer", return_value=mock_model):
            encoder = ChunkEncoder()

        encoder.encode_chunks(sample_chunks)
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert len(texts) == 5
        assert texts[0] == "Students must attend all scheduled lectures and tutorials."

    def test_encode_query_returns_1d(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        with patch("rag_pipeline.embeddings.encoder.SentenceTransformer", return_value=mock_model):
            encoder = ChunkEncoder()

        result = encoder.encode_query("test query")
        assert result.shape == (384,)
