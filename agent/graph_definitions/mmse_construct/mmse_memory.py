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
import re
import threading
import traceback
from contextlib import nullcontext

from neo4j.exceptions import (
    ConstraintError as _Neo4jConstraintError,
    ServiceUnavailable as _Neo4jServiceUnavailable,
)
from neo4j_agent_memory.core.exceptions import ConnectionError as _MemoryConnectionError
from pydantic import SecretStr

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    LLMConfig,
    LLMProvider,
    MemoryClient,
    MemorySettings,
    Neo4jConfig,
    ResolutionConfig,
    ResolverStrategy,
)
from utils.nvidia_embedder import NVIDIAEmbedder, NVIDIALLMExtractor

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
        neo4j_username: str = "password",
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

        # Custom NVIDIA LLM extractor (entity extraction uses MEMORY_LLM_MODEL via NVIDIA API,
        # not OpenAI — avoids OPENAI_API_KEY requirement)
        self._extractor = NVIDIALLMExtractor(
            model=llm_model,
            api_key=nvidia_api_key or os.getenv("NVIDIA_API_KEY"),
        )

        # ── Dedicated async loop in a daemon thread ────────────────────────
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mmse-memory-loop"
        )
        self._thread.start()

        # Connect to Neo4j (blocks until done or timeout)
        self._client = MemoryClient(
                            self._settings,
                            embedder=self._embedder,
                            extractor=self._extractor,
                        ) 
        self._reconnect_task: asyncio.Task | None = None  # background retry loop
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

    async def _with_reconnect(self, coro_factory):
        """
        Run ``coro_factory()`` (a callable that returns a coroutine) and, if
        Neo4j raises ServiceUnavailable (defunct SSH-tunnel connection), attempt
        one reconnect and retry.  All paths in the memory manager should use
        this so that reconnect works both for blocking reads and fire-and-forget
        writes.
        """
        for attempt in range(1, 3):
            try:
                return await coro_factory()
            except (_Neo4jServiceUnavailable, _MemoryConnectionError) as exc:
                if attempt == 1:
                    logger.warning(
                        f"[neo4j] Connection error on attempt {attempt} — "
                        f"reconnecting... ({type(exc).__name__}: {exc})"
                    )
                    await self._async_reconnect()
                    if self._client is None:
                        logger.warning("[neo4j] Reconnect failed — giving up.")
                        return None
                else:
                    logger.warning(
                        f"[neo4j] Connection error after reconnect — giving up: "
                        f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()
                    )
                    return None
        return None

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
            client = MemoryClient(
                self._settings,
                embedder=self._embedder,
                extractor=self._extractor,
            )
            # Hard timeout so a hung SSH tunnel can't block the event loop indefinitely.
            await asyncio.wait_for(client.connect(), timeout=15.0)
            self._client = client
            logger.info("MMSEMemoryManager: reconnected to Neo4j successfully.")
            # Cancel the background retry loop — we're live again.
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
                self._reconnect_task = None
        except asyncio.TimeoutError:
            logger.warning("[neo4j] Reconnect timed out (15 s) — memory marked disconnected.")
            self._schedule_reconnect_loop()
        except Exception as exc:
            logger.warning(f"[neo4j] Reconnect failed: {exc}")
            self._schedule_reconnect_loop()

    def _schedule_reconnect_loop(self) -> None:
        """Start _reconnect_loop() as a background task if not already running."""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = self._loop.create_task(self._reconnect_loop())
            logger.info("[neo4j] Background reconnect loop started (retrying every 5 s).")

    async def _reconnect_loop(
        self,
        delay: float = 5.0,
        max_attempts: int = 120,
    ) -> None:
        """
        Keep attempting _async_reconnect() every `delay` seconds until
        the connection is restored or `max_attempts` is exhausted.
        Runs fully inside the background async loop — does not block any
        LangGraph node.
        """
        for attempt in range(1, max_attempts + 1):
            await asyncio.sleep(delay)
            if self._client is not None:
                return  # already reconnected elsewhere
            logger.info(f"[neo4j] Background reconnect attempt {attempt}/{max_attempts}...")
            await self._async_reconnect()
            if self._client is not None:
                return  # _async_reconnect already logged success
        logger.warning(
            f"[neo4j] Background reconnect gave up after {max_attempts} attempts "
            f"({max_attempts * delay / 60:.1f} min). Memory will remain disabled "
            "until the container is restarted."
        )

    # ── Neo4j connection ───────────────────────────────────────────────────

    def _connect(self):
        async def _do_connect():
            try:
                client = MemoryClient(
                    self._settings,
                    embedder=self._embedder,
                    extractor=self._extractor,
                )
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
        reasoning_dict: dict | None = None,
    ) -> None:
        """
        Persist a user + assistant message pair to short-term memory.

        Creates the following Neo4j chain per turn:
          (:Conversation)-[:HAS_MESSAGE]->(:Message{role:"user"})
                                         -[:NEXT]->(:Message{role:"assistant"})

        If reasoning_dict is provided (from the LLM's embedded JSON block), a
        structured :ReasoningTrace is created and linked to the user :Message:
          (:Message{role:"user"})-[:TRIGGERED]->(:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)

        Entities extracted by gliner-pii are attached only to the user message
        (the source of PII) to avoid duplicate nodes.

        Fire-and-forget — does not block the calling node.
        """
        if not self.is_connected:
            logger.warning(
                f"[neo4j] store_message_pair SKIPPED (not connected) session={session_id} section={section}"
            )
            return

        meta = {"section": section} if section else {}
        section_label = (
            "".join(word.capitalize() for word in re.sub(r'[^A-Za-z0-9]+', '_', section).split('_'))
            if section else None
        )

        def extract_entities_gliner(text, section):
            import requests, json as _json
            GLINER_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
            api_key = getattr(self._embedder, "_api_key", None) or os.getenv("NVIDIA_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "nvidia/gliner-pii",
                "messages": [{"role": "user", "content": text}],
                "threshold": 0.3,
            }
            try:
                response = requests.post(GLINER_NIM_URL, json=payload, headers=headers, timeout=15)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return _json.loads(content).get("entities", [])
            except Exception as exc:
                logger.error(f"[gliner-nim] Entity extraction failed: {exc}")
                return []

        async def _store():
            async def _do_store():
                user_msg = None
                # ── 1. Persist user message ────────────────────────────────────────
                try:
                    user_msg = await self._client.short_term.add_message(
                        session_id, "user", user_text,
                        extract_entities=False,  # gliner handled below
                        metadata=meta,
                    )
                except _Neo4jConstraintError:
                    logger.debug("[neo4j] user Message already exists — idempotent skip")

                # ── 2. Persist assistant message (no internal entity extraction) ──
                try:
                    await self._client.short_term.add_message(
                        session_id, "assistant", ai_text,
                        extract_entities=False,  # avoid double extraction (library default is True)
                        metadata=meta,
                    )
                except _Neo4jConstraintError:
                    logger.debug("[neo4j] assistant Message already exists — idempotent skip")

                # ── 3. Tag both messages with MMSE section label ───────────────────
                if section_label:
                    try:
                        cypher = (
                            f"MATCH (c:Conversation {{id: $sid}})-[:HAS_MESSAGE]->(m:Message) "
                            f"WHERE m.metadata.section = $sec "
                            f"SET m:{section_label} "
                            f"RETURN count(m) AS n"
                        )
                        rows = await self._client.graph.execute_write(
                            cypher, {"sid": session_id, "sec": section}
                        )
                        n = rows[0]["n"] if rows else 0
                        logger.debug(f"[neo4j] section label '{section_label}' applied to {n} Message node(s)")
                    except Exception as label_exc:
                        logger.warning(f"[neo4j] section label write failed (non-fatal): {label_exc}")

                # ── 4. Extract PII entities from the USER message only ────────────
                # The assistant message is generated text and never contains patient
                # PII — scanning it adds noise and duplicates entities.
                user_entities = await asyncio.get_event_loop().run_in_executor(
                    None, extract_entities_gliner, user_text, section
                )
                for entity in user_entities:
                    try:
                        self._fire(
                            self._client.long_term.add_entity(
                                name=entity.get("text", ""),
                                entity_type=entity.get("label", "ENTITY"),
                                subtype=entity.get("subtype", None),
                                description=entity.get("description", None),
                                metadata={"context": user_text},
                            )
                        )
                        logger.info(f"[neo4j] Stored entity (user): {entity}")
                    except Exception as exc:
                        logger.error(f"[neo4j] Failed to store entity (user): {entity} — {exc}")

                # ── 5. Write reasoning trace linked to the user message ───────────
                # If the LLM embedded a structured JSON block we use its richer
                # fields; otherwise fall back to a generic observation string.
                if section:
                    try:
                        user_msg_id = user_msg.id if user_msg is not None else None

                        if reasoning_dict is not None:
                            # Structured path: use fields from the LLM's JSON block
                            q  = reasoning_dict.get("question_asked") or f"MMSE {section} question"
                            pa = reasoning_dict.get("patient_answer_raw") or user_text[:200]
                            sc = reasoning_dict.get("score_this_turn")
                            tone = reasoning_dict.get("patient_tone", "neutral")
                            why  = reasoning_dict.get("reasoning", "")
                            complete = reasoning_dict.get("is_complete", False)
                            running  = reasoning_dict.get("score_running", 0)

                            task_label = f"MMSE {section} — Q: {q[:120]}"
                            thought    = (
                                f"Question: {q}\n"
                                f"Patient answer: {pa}\n"
                                f"Score this turn: {sc}  |  Running: {running}  |  "
                                f"Complete: {complete}"
                            )
                            observation = (
                                f"tone={tone}  score={sc}  running={running}  "
                                f"complete={complete}\n{why}"
                            )
                            outcome = (
                                f"section={section} score={sc}/{running} "
                                f"complete={complete}"
                            )
                        else:
                            # Generic path: no JSON from LLM
                            task_label  = f"MMSE {section} turn"
                            thought     = f"Section: {section} — processing patient response"
                            observation = (
                                f"patient: {user_text[:500]}\n"
                                f"agent: {ai_text[:500]}"
                            )
                            outcome = f"section={section} response_len={len(ai_text)}"

                        trace = await self._client.reasoning.start_trace(
                            session_id=session_id,
                            task=task_label,
                            triggered_by_message_id=user_msg_id,
                        )
                        await self._client.reasoning.add_step(
                            trace.id,
                            thought=thought,
                            action="ask_and_respond",
                            observation=observation,
                        )
                        await self._client.reasoning.complete_trace(
                            trace.id,
                            outcome=outcome,
                            success=True,
                        )
                        logger.info(
                            f"[neo4j] reasoning trace OK session={session_id} section={section} "
                            f"structured={reasoning_dict is not None}"
                        )
                    except Exception as r_exc:
                        logger.warning(f"[neo4j] reasoning trace failed (non-fatal): {r_exc}")

                logger.info(f"[neo4j] store_message_pair OK session={session_id} section={section}")

            try:
                await self._with_reconnect(_do_store)
            except Exception as exc:
                logger.warning(
                    f"[neo4j] store_message_pair FAILED session={session_id} section={section}: {exc}\n"
                    + traceback.format_exc()
                )

        self._fire(_store())

    def extract_and_store_entities(
        self,
        session_id: str,
        text: str,
        section: str,
    ) -> None:
        """
        Extract entities from text using gliner NIM and store them in Neo4j.
        """
        import requests, json as _json
        GLINER_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key = getattr(self._embedder, "_api_key", None) or os.getenv("NVIDIA_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/gliner-pii",
            "messages": [{"role": "user", "content": text}],
            "threshold": 0.3,
        }
        try:
            response = requests.post(GLINER_NIM_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            entities = _json.loads(content).get("entities", [])
        except Exception as exc:
            logger.error(f"[gliner-nim] Entity extraction failed: {exc}")
            entities = []

        # Store entities in Neo4j
        if not self.is_connected:
            logger.warning("[neo4j] Memory not connected, cannot store entities.")
            return

        for entity in entities:
            # entity: {"type": "PERSON", "text": "John Doe", ...}
            try:
                self._fire(
                    self._client.long_term.add_entity(
                        name=entity.get("text", ""),
                        entity_type=entity.get("label", "ENTITY"),
                        subtype=entity.get("subtype", None),
                        description=entity.get("description", None),
                        metadata={"context": text},
                    )
                )
                logger.info(f"[neo4j] Stored entity: {entity}")
            except Exception as exc:
                logger.error(f"[neo4j] Failed to store entity: {entity} — {exc}")

    def record_turn_reasoning(
        self,
        session_id: str,
        section: str,
        user_text: str,
        ai_text: str,
    ) -> None:
        """
        Write a reasoning trace step for one LLM turn (question + response).
        Fire-and-forget — does not block the calling node.
        """
        if not self.is_connected:
            return

        async def _do_record():
            trace = await self._client.reasoning.start_trace(
                session_id=session_id,
                task=f"MMSE {section} turn",
            )
            await self._client.reasoning.add_step(
                trace.id,
                thought=f"Section: {section} — processing patient response",
                action="ask_and_respond",
                observation=f"patient: {user_text[:500]}\nagent: {ai_text[:500]}",
            )
            await self._client.reasoning.complete_trace(
                trace.id,
                outcome=f"section={section} response_len={len(ai_text)}",
                success=True,
            )
            logger.info(f"[neo4j] record_turn_reasoning OK session={session_id} section={section}")

        async def _record():
            try:
                await self._with_reconnect(_do_record)
            except Exception as exc:
                logger.warning(f"[neo4j] record_turn_reasoning FAILED session={session_id}: {exc}")

        self._fire(_record())

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
            async def _do_record():
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

            try:
                await self._with_reconnect(_do_record)
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

        async def coro():
            if self._client is None:
                return None
            return await self._client.get_context(
                query or "MMSE patient session",
                session_id=session_id,
                include_reasoning=False,
            )
        with self._lf_span(
            "neo4j-get-context",
            {"session_id": session_id, "query": query or "MMSE patient session"},
        ) as obs:
            result = self._run_async(self._with_reconnect(coro))
            if obs is not None:
                try:
                    obs.update(output={
                        "context_length": len(result) if result else 0,
                        "found": bool(result),
                    })
                except Exception:
                    pass
        return result or ""

    def clear_session(self, session_id: str) -> None:
        """
        Drop the entire graph (all nodes and relationships) on every new session
        start so each assessment begins with a clean slate.
        Fire-and-forget, safe to call at session start.
        """
        if not self.is_connected:
            logger.warning(f"[neo4j] clear_session SKIPPED (not connected) session={session_id}")
            return

        async def _do_clear():
            if self._client is None:
                return
            # Delete only the Conversation + Message nodes for THIS session.
            # Entity nodes are long-term knowledge and kept across sessions.
            # We also delete any ReasoningTrace/ReasoningStep nodes that were
            # triggered by messages in this conversation.
            cypher = (
                "MATCH (c:Conversation {id: $sid}) "
                "OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m) "
                "OPTIONAL MATCH (m)-[:TRIGGERED]->(rt)-[:HAS_STEP]->(rs) "
                "DETACH DELETE rs, rt, m, c"
            )
            await self._client.graph.execute_write(cypher, {"sid": session_id})
            logger.info(f"[neo4j] Conversation/Messages/Reasoning cleared for session={session_id}")

        async def _clear():
            try:
                await self._with_reconnect(_do_clear)
            except Exception as exc:
                logger.warning(f"[neo4j] clear_session FAILED session={session_id}: {exc}")

        self._fire(_clear())

    def close(self) -> None:
        """Disconnect from Neo4j and stop the background loop."""
        if self._client:
            self._run_async(self._client.close(), timeout=10.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
