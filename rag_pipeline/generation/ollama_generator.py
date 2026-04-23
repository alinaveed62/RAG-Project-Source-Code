"""Ollama-based answer generation for local inference on Apple Silicon or CPU."""

import logging

import ollama

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Generates answers using a locally running Ollama model.

    Ollama handles quantisation and Metal (MPS) acceleration on
    Apple Silicon automatically, so this backend works on both
    Apple Silicon and plain CPU without any extra wiring.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._client: ollama.Client | None = None

    def load_model(self) -> None:
        """Connect to Ollama and verify the configured model is available.

        Raises:
            RuntimeError: If the configured model has not been pulled on
                the Ollama server (re-raised from ollama.ResponseError)
                or the server cannot be reached (re-raised from
                ConnectionError, OSError or TimeoutError). Programming
                errors like TypeError are left to propagate so they are
                not misdiagnosed as connection failures.
        """
        self._client = ollama.Client(host=self.config.ollama_base_url)

        # Ask Ollama to describe the model; this returns quickly if
        # the model is pulled and throws otherwise.
        try:
            self._client.show(self.config.ollama_model)
            logger.info(
                "Ollama model ready: %s at %s",
                self.config.ollama_model,
                self.config.ollama_base_url,
            )
        except ollama.ResponseError as e:
            raise RuntimeError(
                f"Ollama model '{self.config.ollama_model}' not available. "
                f"Run: ollama pull {self.config.ollama_model}"
            ) from e
        except (ConnectionError, OSError, TimeoutError) as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.ollama_base_url}. "
                "Is Ollama running? Start with: ollama serve"
            ) from e

    def generate(self, prompt: str) -> str:
        """Generate an answer for the given prompt.

        Args:
            prompt: The full prompt string. For non-Mistral models,
                the Mistral [INST] wrapper is literal user text; Ollama
                applies each model's Modelfile template server-side.

        Returns:
            The generated answer text with leading and trailing
            whitespace stripped.

        Raises:
            RuntimeError: If the client was not initialised (load_model
                not called), the Ollama HTTP call fails, or the response
                payload is missing the expected "response" key. Callers
                therefore only need to catch one exception type.
        """
        if self._client is None:
            raise RuntimeError("Client not initialised. Call load_model() first.")

        try:
            response = self._client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_new_tokens,
                },
            )
        except ollama.ResponseError as e:
            raise RuntimeError(
                f"Ollama generation error for model "
                f"{self.config.ollama_model!r}: {e}"
            ) from e

        try:
            return response["response"].strip()
        except KeyError as e:
            raise RuntimeError(
                "Unexpected Ollama response structure: expected 'response' "
                f"key, got keys {sorted(response)}"
            ) from e
