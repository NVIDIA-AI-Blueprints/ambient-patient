# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MMSE Agentic Memory Module

Wraps neo4j-agent-memory (https://github.com/neo4j-labs/agent-memory) so that
async memory operations can be called safely from synchronous LangGraph nodes.

Three memory types are used:
  - Short-term : full conversation history per session (keyed by LangGraph thread_id)
  - Long-term  : patient entities extracted from MMSE orientation answers
  - Reasoning  : scoring traces logged after each MMSEScoringAgent call

Embedding backend: utils.nvidia_embedder.NVIDIAEmbedder
  model  : nvidia/nv-embedcode-7b-v1
  base_url: https://integrate.api.nvidia.com/v1
  dims   : 4096
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
import traceback
from contextlib import nullcontext

from neo4j.exceptions import ServiceUnavailable as _Neo4jServiceUnavailable
from pydantic import SecretStr

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    ExtractionConfig,
    ExtractorType,
    LLMConfig,
    LLMProvider,
    MemoryClient,
    MemorySettings,
    Neo4jConfig,
    ResolutionConfig,
    ResolverStrategy,
)
from .nvidia_embedder import NVIDIAEmbedder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sync-friendly memory manager
# ─────────────────────────────────────────────────────────────────────────────

class MMSEMemoryManager:
    """
    Sync-friendly wrapper around MemoryClient for use inside LangGraph nodes.

    LangGraph nodes are synchronous, but neo4j-agent-memory is fully async.
    This class runs a dedicated asyncio event loop in a daemon thread and
    submits coroutines to it via run_coroutine_threadsafe, so sync callers
    get a clean, blocking interface with a configurable timeout.

    Write operations (store_message_pair, extract_and_store_entities,
    record_scoring_trace) are fire-and-forget — they do NOT block the node.

    Read operations (get_session_context) block for up to `read_timeout`
    seconds (default 10 s) and return "" on timeout/error.

    Usage:
        memory = MMSEMemoryManager.from_env()
        # in a LangGraph node:
        memory.store_message_pair(thread_id, user_text, ai_text, section)
        memory.extract_and_store_entities(thread_id, user_text, section)
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_password: str ="neo4j",
        neo4j_username: str = "neo4j",
        nvidia_api_key: str | None = None,
        llm_model: str = "meta/llama-3.3-70b-instruct",
        llm_base_url: str = "https://integrate.api.nvidia.com/v1",
        read_timeout: float = 10.0,
        connect_timeout: float = 60.0,
        langfuse_client=None,
    ):
        self._read_timeout = read_timeout
        self._connect_timeout = connect_timeout
        self._langfuse = langfuse_client

        # ── Build MemorySettings ───────────────────────────────────────────
        self._settings = MemorySettings(
            neo4j=Neo4jConfig(
                uri=neo4j_uri,
                username=neo4j_username,
                password=SecretStr(neo4j_password),
            ),
            # Embedding config is only used for schema dimension setup;
            # the actual embedder is overridden with NVIDIAEmbedder below.
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,   # placeholder
                model=NVIDIAEmbedder.DEFAULT_MODEL,
                dimensions=NVIDIAEmbedder.DEFAULT_DIMENSIONS,
            ),
            extraction=ExtractionConfig(
                # Use the NVIDIA LLM for entity extraction from patient answers.
                # Only PERSON / LOCATION / ORGANIZATION are relevant for MMSE.
                extractor_type=ExtractorType.LLM,
                llm_model=llm_model,
                entity_types=["PERSON", "LOCATION", "ORGANIZATION"],
            ),
            # Fuzzy + semantic resolver so patient name/place variants merge
            resolution=ResolutionConfig(
                strategy=ResolverStrategy.COMPOSITE,
                fuzzy_threshold=0.85,
                semantic_threshold=0.80,
            ),
        )

        # Custom NVIDIA embedder
        self._embedder = NVIDIAEmbedder(
            api_key=nvidia_api_key or os.getenv("NVIDIA_API_KEY"),
        )

        # ── Dedicated async loop in a daemon thread ────────────────────────
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mmse-memory-loop"
        )
        self._thread.start()

        # Connect to Neo4j (blocks until done or timeout)
        self._client: MemoryClient | None = None
        self._connect()

    # ── Class factory ──────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "MMSEMemoryManager | None":
        """
        Construct from environment variables. Returns None (memory disabled)
        if NEO4J_URI or NEO4J_PASSWORD is not set.

        Required env vars:
          NEO4J_URI       e.g. bolt://host.docker.internal:7687
          NEO4J_PASSWORD  Neo4j password

        Optional env vars:
          NEO4J_USERNAME      (default: neo4j)
          NVIDIA_API_KEY      (used for embeddings)
          AGENT_LLM_MODEL     (used for entity extraction)
          AGENT_LLM_BASE_URL  (base URL for NVIDIA LLM)
        """
        neo4j_uri = os.getenv("NEO4J_URI", "").strip()
        neo4j_password = os.getenv("NEO4J_PASSWORD", "").strip()

        if not neo4j_uri or not neo4j_password:
            logger.info(
                "NEO4J_URI or NEO4J_PASSWORD not set — agentic memory disabled."
            )
            return None

        try:
            return cls(
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
                nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                llm_model=os.getenv("AGENT_LLM_MODEL", "meta/llama-3.3-70b-instruct"),
                llm_base_url=os.getenv("AGENT_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            )
        except Exception as exc:
            logger.warning(f"MMSEMemoryManager init failed — memory disabled: {exc}", exc_info=True)
            return None

    @classmethod
    def from_env_with_langfuse(cls, langfuse_client) -> "MMSEMemoryManager | None":
        """
        Like from_env() but attaches an existing Langfuse client so that
        memory operations appear as child spans inside the current session
        trace in the Langfuse UI.
        """
        neo4j_uri = os.getenv("NEO4J_URI", "").strip()
        neo4j_password = os.getenv("NEO4J_PASSWORD", "").strip()

        if not neo4j_uri or not neo4j_password:
            logger.info(
                "NEO4J_URI or NEO4J_PASSWORD not set — agentic memory disabled."
            )
            return None

        try:
            return cls(
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
                nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                llm_model=os.getenv("AGENT_LLM_MODEL", "meta/llama-3.3-70b-instruct"),
                llm_base_url=os.getenv("AGENT_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                langfuse_client=langfuse_client,
            )
        except Exception as exc:
            logger.warning(f"MMSEMemoryManager init failed — memory disabled: {exc}", exc_info=True)
            return None

    # ── Internal async loop management ─────────────────────────────────────

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float | None = None) -> object:
        """Submit coroutine to background loop; block until done or timeout."""
        timeout = timeout if timeout is not None else self._read_timeout
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"[neo4j] Memory operation timed out after {timeout}s")
            return None
        except Exception as exc:
            logger.warning(
                f"[neo4j] Memory operation error: {exc}\n{traceback.format_exc()}"
            )
            return None

    def _lf_span(self, name: str, input_data: dict):
        """Return a Langfuse span context manager if a client is configured, else a no-op."""
        if self._langfuse is None:
            return nullcontext()
        try:
            return self._langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
            )
        except Exception:
            return nullcontext()

    def _fire(self, coro) -> None:
        """Submit coroutine and do NOT wait — fire-and-forget."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _async_reconnect(self) -> None:
        """Close the current MemoryClient and open a fresh one.

        Called from *inside* the background event loop when the driver reports
        a defunct (dead SSH-tunnel) connection.  After this method returns,
        self._client is either a live new client or None.
        """
        try:
            if self._client:
                await self._client.close()
        except Exception:
            pass
        self._client = None
        try:
            client = MemoryClient(self._settings, embedder=self._embedder)
            await client.connect()
            self._client = client
            logger.info("MMSEMemoryManager: reconnected to Neo4j successfully.")
        except Exception as exc:
            logger.warning(f"[neo4j] Reconnect failed: {exc}\n{traceback.format_exc()}")

    # ── Neo4j connection ───────────────────────────────────────────────────

    def _connect(self):
        async def _do_connect():
            try:
                client = MemoryClient(self._settings, embedder=self._embedder)
                await client.connect()
                return client
            except Exception as exc:
                logger.warning(
                    f"[neo4j] MemoryClient.connect() raised: {exc}\n{traceback.format_exc()}"
                )
                raise

        with self._lf_span("neo4j-memory-connect", {"uri": str(self._settings.neo4j.uri)}) as obs:
            self._client = self._run_async(_do_connect(), timeout=self._connect_timeout)
            if self._client:
                logger.info("MMSEMemoryManager: connected to Neo4j successfully.")
                if obs is not None:
                    try:
                        obs.update(output={"status": "connected"})
                    except Exception:
                        pass
            else:
                logger.warning(
                    "MMSEMemoryManager: Neo4j connection failed — memory ops are no-ops. "
                    "Check NEO4J_URI, NEO4J_PASSWORD, and that the Neo4j bolt port is reachable "
                    "from the container (e.g. bolt://host.docker.internal:7687)."
                )
                if obs is not None:
                    try:
                        obs.update(output={"status": "failed"})
                    except Exception:
                        pass

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    def store_message_pair(
        self,
        session_id: str,
        user_text: str,
        ai_text: str,
        section: str | None = None,
        extract_entities: bool = False,
    ) -> None:
        """
        Persist a user + assistant message pair to short-term memory.
        Fire-and-forget — does not block the calling node.

        Args:
            extract_entities: When True, the user-message ``add_message`` call
                also triggers LLM entity extraction for storage in long-term
                memory.  Pass ``True`` for entity-rich sections such as
                ``orientation_place``, ``orientation_time``, and ``intake``.
                Do **not** call ``extract_and_store_entities`` separately —
                that would attempt to insert the same user message twice and
                trigger a Neo4j ConstraintError.
        """
        if not self.is_connected:
            logger.warning(
                f"[neo4j] store_message_pair SKIPPED (not connected) session={session_id} section={section}"
            )
            return

        meta = {"section": section} if section else {}

        async def _store():
            for attempt in range(1, 3):  # attempt 1 = normal, attempt 2 = after reconnect
                try:
                    await self._client.short_term.add_message(
                        session_id, "user", user_text,
                        extract_entities=extract_entities,
                        metadata=meta,
                    )
                    await self._client.short_term.add_message(
                        session_id, "assistant", ai_text, metadata=meta
                    )
                    logger.info(f"[neo4j] store_message_pair OK session={session_id} section={section}")
                    return
                except _Neo4jServiceUnavailable as exc:
                    if attempt == 1:
                        logger.warning(f"[neo4j] store_message_pair: defunct connection, reconnecting... ({exc})")
                        await self._async_reconnect()
                        if self._client is None:
                            logger.warning(f"[neo4j] store_message_pair FAILED (reconnect failed) session={session_id}")
                            return
                    else:
                        logger.warning(
                            f"[neo4j] store_message_pair FAILED session={session_id} section={section}: {exc}\n"
                            + traceback.format_exc()
                        )
                except Exception as exc:
                    logger.warning(
                        f"[neo4j] store_message_pair FAILED session={session_id} section={section}: {exc}\n"
                        + traceback.format_exc()
                    )
                    return

        self._fire(_store())

    def extract_and_store_entities(
        self,
        session_id: str,
        text: str,
        section: str,
    ) -> None:
        """
        .. deprecated::
            Entity extraction is now handled inline by ``store_message_pair``
            via ``extract_entities=True``.  Calling this method separately
            would attempt to insert the same user message a second time and
            trigger a Neo4j ``ConstraintError``.

            This method is intentionally a no-op.  Pass
            ``extract_entities=True`` to ``store_message_pair`` instead.
        """
        logger.debug(
            f"[neo4j] extract_and_store_entities is a no-op (use store_message_pair(extract_entities=True)) "
            f"session={session_id} section={section}"
        )

    def record_scoring_trace(
        self,
        session_id: str,
        score_payload: dict,
    ) -> None:
        """
        Log an MMSE scoring invocation as a reasoning trace in Neo4j.
        Fire-and-forget.
        """
        if not self.is_connected:
            logger.warning(
                f"[neo4j] record_scoring_trace SKIPPED (not connected) session={session_id}"
            )
            return

        total = score_payload.get("total_score", "?")
        max_av = score_payload.get("max_available", "?")
        interp = score_payload.get("interpretation", "")

        async def _record():
            try:
                trace = await self._client.reasoning.start_trace(
                    session_id=session_id,
                    task="MMSE scoring",
                )
                await self._client.reasoning.add_step(
                    trace.id,
                    thought="Scored full MMSE conversation transcript",
                    action="score_transcript",
                    observation=(
                        f"total_score={total}/{max_av}; interpretation={interp}"
                    ),
                )
                await self._client.reasoning.complete_trace(
                    trace.id,
                    outcome=f"score={total}/{max_av} {interp}",
                    success=True,
                )
                logger.info(f"[neo4j] record_scoring_trace OK session={session_id} score={total}/{max_av}")
            except Exception as exc:
                logger.warning(
                    f"[neo4j] record_scoring_trace FAILED session={session_id}: {exc}\n"
                    + traceback.format_exc()
                )

        self._fire(_record())

    def get_session_context(self, session_id: str, query: str = "") -> str:
        """
        Retrieve combined short-term + long-term context for `session_id`.
        Blocks for up to `read_timeout` seconds; returns "" on failure.
        Suitable for injecting into an LLM system prompt.
        """
        if not self.is_connected:
            logger.warning(
                f"[neo4j] get_session_context SKIPPED (not connected) session={session_id}"
            )
            return ""

        coro = self._client.get_context(
            query or "MMSE patient session",
            session_id=session_id,
            include_reasoning=False,
        )
        with self._lf_span(
            "neo4j-get-context",
            {"session_id": session_id, "query": query or "MMSE patient session"},
        ) as obs:
            result = self._run_async(coro)
            if obs is not None:
                try:
                    obs.update(output={
                        "context_length": len(result) if result else 0,
                        "found": bool(result),
                    })
                except Exception:
                    pass
        return result or ""

    def close(self) -> None:
        """Disconnect from Neo4j and stop the background loop."""
        if self._client:
            self._run_async(self._client.close(), timeout=10.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
