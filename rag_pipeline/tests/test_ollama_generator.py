"""Tests for the Ollama-based generator."""

from unittest.mock import patch

import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation.ollama_generator import OllamaGenerator


@pytest.fixture
def ollama_config(tmp_path):
    return RAGConfig(
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        inference_backend="ollama",
        ollama_model="mistral",
        ollama_base_url="http://localhost:11434",
    )


class TestOllamaGeneratorInit:
    """Tests for OllamaGenerator initialisation."""

    def test_init_stores_config(self, ollama_config):
        gen = OllamaGenerator(ollama_config)
        assert gen.config is ollama_config
        assert gen._client is None

    def test_generate_before_load_raises(self, ollama_config):
        gen = OllamaGenerator(ollama_config)
        with pytest.raises(RuntimeError, match="not initialised"):
            gen.generate("test prompt")


class TestOllamaGeneratorLoadModel:
    """Tests for model loading / Ollama connection."""

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_creates_client(self, MockClient, ollama_config):
        mock_client = MockClient.return_value
        mock_client.show.return_value = {"modelfile": "..."}

        gen = OllamaGenerator(ollama_config)
        gen.load_model()

        MockClient.assert_called_once_with(host="http://localhost:11434")
        mock_client.show.assert_called_once_with("mistral")
        assert gen._client is mock_client

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_missing_model_raises(self, MockClient, ollama_config):
        import ollama as ollama_lib

        mock_client = MockClient.return_value
        mock_client.show.side_effect = ollama_lib.ResponseError("not found")

        gen = OllamaGenerator(ollama_config)
        with pytest.raises(RuntimeError, match="not available"):
            gen.load_model()

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_connection_error_raises(self, MockClient, ollama_config):
        mock_client = MockClient.return_value
        mock_client.show.side_effect = ConnectionError("refused")

        gen = OllamaGenerator(ollama_config)
        with pytest.raises(RuntimeError, match="Cannot connect"):
            gen.load_model()


class TestOllamaGeneratorGenerate:
    """Tests for answer generation."""

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_generate_returns_stripped_text(self, MockClient, ollama_config):
        mock_client = MockClient.return_value
        mock_client.show.return_value = {}
        mock_client.generate.return_value = {"response": "  The answer is 42.  "}

        gen = OllamaGenerator(ollama_config)
        gen.load_model()
        result = gen.generate("[INST] test prompt [/INST]")

        assert result == "The answer is 42."
        mock_client.generate.assert_called_once()

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_generate_passes_config_options(self, MockClient, ollama_config):
        mock_client = MockClient.return_value
        mock_client.show.return_value = {}
        mock_client.generate.return_value = {"response": "answer"}

        gen = OllamaGenerator(ollama_config)
        gen.load_model()
        gen.generate("prompt")

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["model"] == "mistral"
        assert call_kwargs["options"]["temperature"] == ollama_config.temperature
        assert call_kwargs["options"]["top_p"] == ollama_config.top_p
        assert call_kwargs["options"]["num_predict"] == ollama_config.max_new_tokens

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_generate_empty_response(self, MockClient, ollama_config):
        mock_client = MockClient.return_value
        mock_client.show.return_value = {}
        mock_client.generate.return_value = {"response": ""}

        gen = OllamaGenerator(ollama_config)
        gen.load_model()
        result = gen.generate("prompt")

        assert result == ""

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_generate_missing_response_key_raises(self, MockClient, ollama_config):
        """Defend against future ollama-python API changes that rename the
        response key: a missing "response" key is re-raised as a clear
        RuntimeError with the observed keys, not swallowed."""
        mock_client = MockClient.return_value
        mock_client.show.return_value = {}
        mock_client.generate.return_value = {"not_response": "oops"}

        gen = OllamaGenerator(ollama_config)
        gen.load_model()
        with pytest.raises(
            RuntimeError, match="Unexpected Ollama response structure"
        ) as exc_info:
            gen.generate("prompt")
        # Chain preserves the underlying KeyError for debuggers.
        assert isinstance(exc_info.value.__cause__, KeyError)

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_generate_wraps_response_error_as_runtime_error(
        self, MockClient, ollama_config
    ):
        """Ollama-side errors (model unloaded mid-request, 5xx) reach the
        pipeline as a single RuntimeError type with the original
        ollama.ResponseError preserved in __cause__."""
        import ollama as ollama_lib

        mock_client = MockClient.return_value
        mock_client.show.return_value = {}
        mock_client.generate.side_effect = ollama_lib.ResponseError("boom")

        gen = OllamaGenerator(ollama_config)
        gen.load_model()
        with pytest.raises(
            RuntimeError, match="Ollama generation error"
        ) as exc_info:
            gen.generate("prompt")
        assert isinstance(exc_info.value.__cause__, ollama_lib.ResponseError)

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_os_error_raises(self, MockClient, ollama_config):
        """Socket-layer failures (e.g. DNS resolution failing) are narrowed
        into the same 'Cannot connect' RuntimeError as ConnectionError."""
        mock_client = MockClient.return_value
        mock_client.show.side_effect = OSError("socket gone")

        gen = OllamaGenerator(ollama_config)
        with pytest.raises(RuntimeError, match="Cannot connect"):
            gen.load_model()

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_timeout_error_raises(self, MockClient, ollama_config):
        """Ollama-server-hung timeouts surface as 'Cannot connect' rather
        than propagating raw TimeoutError across the Flask boundary."""
        mock_client = MockClient.return_value
        mock_client.show.side_effect = TimeoutError("slow")

        gen = OllamaGenerator(ollama_config)
        with pytest.raises(RuntimeError, match="Cannot connect"):
            gen.load_model()

    @patch("rag_pipeline.generation.ollama_generator.ollama.Client")
    def test_load_model_unexpected_exception_propagates(
        self, MockClient, ollama_config
    ):
        """Programmer errors (TypeError, ValueError) must NOT be masked as
        connection failures. Pins the narrow except tuple is not a
        blanket catch-all."""
        mock_client = MockClient.return_value
        mock_client.show.side_effect = ValueError("bad config")

        gen = OllamaGenerator(ollama_config)
        with pytest.raises(ValueError, match="bad config"):
            gen.load_model()
