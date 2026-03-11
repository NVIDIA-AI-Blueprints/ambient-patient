# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA NIM Embedder

A reusable embedding provider that calls any NVIDIA NIM embedding endpoint
via the OpenAI-compatible API.  Implements neo4j-agent-memory's BaseEmbedder
interface so it can be dropped straight into MemoryClient as an embedder override,
but it is framework-agnostic and can be used by any agent in this repo.

Default model: nvidia/nv-embedcode-7b-v1
  - 7B Mistral-based, optimised for code + text retrieval
  - 4096-dimensional output vectors
  - Available at https://integrate.api.nvidia.com/v1

Usage:
    from utils.nvidia_embedder import NVIDIAEmbedder

    embedder = NVIDIAEmbedder()                           # uses defaults + NVIDIA_API_KEY env var
    embedder = NVIDIAEmbedder(model="nvidia/nv-embedqa-e5-v5", dimensions=1024)
"""

import os
import logging

logger = logging.getLogger(__name__)


class NVIDIAEmbedder:
    """
    Async embedding provider backed by NVIDIA NIM (OpenAI-compatible endpoint).

    Implements neo4j-agent-memory's BaseEmbedder interface:
      - dimensions  (property)
      - embed(text) -> list[float]
      - embed_batch(texts) -> list[list[float]]

    The openai client is initialised lazily on first use so the object is cheap
    to construct even if the event loop is not yet running.
    """

    DEFAULT_MODEL = "nvidia/nv-embedcode-7b-v1"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_DIMENSIONS = 4096

    # Known dimension overrides for other popular NVIDIA NIM embedding models
    _MODEL_DIMENSIONS: dict[str, int] = {
        "nvidia/nv-embedcode-7b-v1": 4096,
        "nvidia/nv-embedqa-mistral-7b-v2": 4096,
        "nvidia/nv-embedqa-e5-v5": 1024,
        "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 32,
        input_type: str = "passage",
    ):
        """
        Args:
            model:      NVIDIA NIM embedding model name.
            base_url:   NIM endpoint base URL (OpenAI-compatible).
            api_key:    NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
            dimensions: Expected output dimension. Auto-detected from known
                        models if not provided.
            batch_size: Max texts per API call.
            input_type: NVIDIA NIM input type hint. Must be ``"passage"`` when
                        embedding documents for storage, or ``"query"`` when
                        embedding a search query.  Many NVIDIA NIM models
                        (including nv-embedcode-7b-v1) require this field and
                        will return HTTP 400 if it is absent.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY") or ""
        self._batch_size = batch_size
        self._input_type = input_type
        self._dimensions = (
            dimensions
            or self._MODEL_DIMENSIONS.get(model, self.DEFAULT_DIMENSIONS)
        )
        self._client = None  # lazy-initialised inside async context

    # ── BaseEmbedder interface ──────────────────────────────────────────────

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ensure_client(self):
        """Lazily create the AsyncOpenAI client (safe to call inside a coroutine)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for NVIDIAEmbedder. "
                    "Install it with: pip install openai"
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    # ── Public async API ───────────────────────────────────────────────────

    async def embed(self, text: str, input_type: str | None = None) -> list[float]:
        """Return the embedding vector for a single text string.

        Args:
            text:       The text to embed.
            input_type: Override the instance-level ``input_type``.  Pass
                        ``"query"`` when embedding a search query.
        """
        client = self._ensure_client()
        _itype = input_type if input_type is not None else self._input_type
        try:
            response = await client.embeddings.create(
                input=text,
                model=self._model,
                extra_body={"input_type": _itype},
            )
            return response.data[0].embedding
        except Exception as exc:
            self._raise_embedding_error(f"NVIDIA NIM embed() failed: {exc}", exc)

    async def embed_batch(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        """Return embedding vectors for a list of texts, batched.

        Args:
            texts:      The texts to embed.
            input_type: Override the instance-level ``input_type``.  Pass
                        ``"query"`` when embedding search queries.
        """
        if not texts:
            return []
        client = self._ensure_client()
        _itype = input_type if input_type is not None else self._input_type
        all_embeddings: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                response = await client.embeddings.create(
                    input=batch,
                    model=self._model,
                    extra_body={"input_type": _itype},
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([d.embedding for d in sorted_data])
            return all_embeddings
        except Exception as exc:
            self._raise_embedding_error(f"NVIDIA NIM embed_batch() failed: {exc}", exc)

    # ── Helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _raise_embedding_error(msg: str, cause: Exception) -> None:
        try:
            from neo4j_agent_memory.core.exceptions import EmbeddingError
            raise EmbeddingError(msg) from cause
        except ImportError:
            raise RuntimeError(msg) from cause

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NVIDIAEmbedder(model={self._model!r}, "
            f"base_url={self._base_url!r}, "
            f"dimensions={self._dimensions}, "
            f"input_type={self._input_type!r})"
        )
