# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neo4j Memory Middleware

LangChain AgentMiddleware that persists every user/assistant message pair to
Neo4j short-term memory after each model call.

Embedding backend: NVIDIAEmbedder
  model  : nvidia/nv-embedcode-7b-v1
  base_url: https://integrate.api.nvidia.com/v1
  dims   : 4096

Usage::

    from graph_definitions.mmse_construct.mmse_memory import Neo4jMemoryMiddleware
    from langchain.agents import create_agent

    middleware = Neo4jMemoryMiddleware(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="your-password",
    )
    agent = create_agent(model="...", middleware=[middleware], tools=[...])
"""

import logging
import traceback
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from pydantic import SecretStr

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    ExtractionConfig,
    ExtractorType,
    MemoryClient,
    MemorySettings,
    Neo4jConfig,
    ResolutionConfig,
    ResolverStrategy,
)
from .nvidia_embedder import NVIDIAEmbedder

logger = logging.getLogger(__name__)


class Neo4jMemoryMiddleware(AgentMiddleware):
    """
    LangChain AgentMiddleware that persists conversation messages to Neo4j.

    After every model call the ``after_model`` hook extracts the most recent
    human/AI message pair from the agent state and writes both messages to
    Neo4j short-term memory via neo4j-agent-memory.  The operation is
    fire-and-forget so it never blocks the agent.

    Session identity is derived automatically from the LangGraph ``thread_id``
    held in the runtime config.  All Neo4j and embedding configuration is
    supplied at construction time through ``__init__``.

    Usage::

        middleware = Neo4jMemoryMiddleware(
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="your-password",
        )
        agent = create_agent(model="...", middleware=[middleware], tools=[...])
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_password: str = "neo4j",
        neo4j_username: str = "neo4j",
        nvidia_api_key: str | None = None,
        llm_model: str = "meta/llama-3.3-70b-instruct",
        llm_base_url: str = "https://integrate.api.nvidia.com/v1",
        extract_entities: bool = False,
        section: str | None = None,
    ) -> None:
        """
        Args:
            neo4j_uri: Bolt URI of the Neo4j instance,
                e.g. ``"bolt://host.docker.internal:7687"``.
            neo4j_password: Neo4j password.
            neo4j_username: Neo4j username (default: ``"neo4j"``).
            nvidia_api_key: NVIDIA API key for the embedding model.
            llm_model: LLM used for entity extraction.
            llm_base_url: Base URL for the NVIDIA LLM API.
            extract_entities: When ``True`` the user message is also passed
                through the LLM entity extractor and relevant entities are
                stored in long-term memory.
            section: Optional metadata tag attached to every stored message
                (e.g. ``"orientation_place"``).
        """
        self._extract_entities = extract_entities
        self._section = section

        settings = MemorySettings(
            neo4j=Neo4jConfig(
                uri=neo4j_uri,
                username=neo4j_username,
                password=SecretStr(neo4j_password),
            ),
            # Embedding provider is overridden by NVIDIAEmbedder below;
            # the config here only informs Neo4j schema dimension setup.
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                model=NVIDIAEmbedder.DEFAULT_MODEL,
                dimensions=NVIDIAEmbedder.DEFAULT_DIMENSIONS,
            ),
            extraction=ExtractionConfig(
                extractor_type=ExtractorType.LLM,
                llm_model=llm_model,
                entity_types=["PERSON", "LOCATION", "ORGANIZATION"],
            ),
            resolution=ResolutionConfig(
                strategy=ResolverStrategy.COMPOSITE,
                fuzzy_threshold=0.85,
                semantic_threshold=0.80,
            ),
        )

        self._embedder = NVIDIAEmbedder(api_key=nvidia_api_key)
        self._settings = settings
        self._client: MemoryClient | None = None
        # Connection is deferred to the first async call; see _ensure_connected().

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _ensure_connected(self) -> bool:
        """Lazily connect on first use; returns True when the client is ready."""
        if self._client is not None and self._client.is_connected:
            return True
        try:
            client = MemoryClient(self._settings, embedder=self._embedder)
            await client.connect()
            self._client = client
            logger.info("Neo4jMemoryMiddleware: connected to Neo4j.")
            return True
        except Exception:
            logger.warning(
                "Neo4jMemoryMiddleware: Neo4j connection failed — "
                "message storage will be skipped.\n%s",
                traceback.format_exc(),
            )
            return False

    # ── AgentMiddleware hook ───────────────────────────────────────────────

    async def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> "dict[str, Any] | None":
        """
        Called after every model response.

        Extracts the most recent human/AI message pair from *state* and
        persists both to Neo4j short-term memory.
        """
        if not await self._ensure_connected():
            return None

        session_id = self._session_id(runtime)
        messages = state.get("messages", [])
        user_text = self._last_of(messages, HumanMessage)
        ai_text = self._last_of(messages, AIMessage)

        if not user_text or not ai_text:
            return None

        meta = {"section": self._section} if self._section else {}
        try:
            await self._client.short_term.add_message(
                session_id, "user", user_text,
                extract_entities=self._extract_entities,
                metadata=meta,
            )
            await self._client.short_term.add_message(
                session_id, "assistant", ai_text,
                metadata=meta,
            )
            logger.debug("[neo4j] stored message pair session=%s", session_id)
        except Exception:
            logger.warning("[neo4j] store failed:\n%s", traceback.format_exc())
        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _session_id(runtime: Runtime) -> str:
        """Return the LangGraph thread_id from runtime config, or ``"default"``."""
        try:
            return runtime.config["configurable"]["thread_id"]
        except (AttributeError, KeyError, TypeError):
            return "default"

    @staticmethod
    def _last_of(messages: list, msg_type: type) -> str:
        """Return the text content of the last message of *msg_type*."""
        for msg in reversed(messages):
            if isinstance(msg, msg_type):
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Disconnect from Neo4j."""
        if self._client:
            await self._client.close()
