# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# stdlib
import argparse
import os
import random
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from urllib.parse import urlparse

# third-party
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langfuse import Langfuse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import gradio as gr
import uvicorn

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except ImportError:
    LangfuseCallbackHandler = None

# local
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent))
from utils.ui import launch_demo_ui
from utils.logging import get_logging
from graph_definitions.utils_graph import add_messages_with_reset
from graph_definitions.mmse_scoring_agent import MMSEScoringAgent, build_scoring_runnable
from graph_definitions.mmse_memory import MMSEMemoryManager

save_graph_to_png = False
env_var_file = "/env_vars/vars.env"


def load_env_variables():
    """Load environment variables from vars.env file and return configuration"""
    load_dotenv(env_var_file, override=True)

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    assert nvidia_api_key, "NVIDIA_API_KEY environment variable is not set."

    llm_model = os.getenv("AGENT_LLM_MODEL")
    assert llm_model, "AGENT_LLM_MODEL environment variable is not set."

    base_url = os.getenv("AGENT_LLM_BASE_URL")
    assert base_url, "AGENT_LLM_BASE_URL environment variable is not set."

    # MEMORY_LLM_MODEL controls the LLM used for Neo4j entity extraction.
    # Defaults to AGENT_LLM_MODEL so a single variable covers the simple case.
    memory_llm_model = os.getenv("MEMORY_LLM_MODEL") or llm_model

    nemo_guardrails_config_path = os.getenv("NEMO_GUARDRAILS_CONFIG_PATH")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")
    langfuse_prompt_label = os.getenv("LANGFUSE_PROMPT_LABEL", "production")
    langfuse_scoring_prompt_name = os.getenv("LANGFUSE_SCORING_PROMPT_NAME", "mmse-agent-scoring")
    langfuse_scoring_prompt_label = os.getenv("LANGFUSE_SCORING_PROMPT_LABEL", langfuse_prompt_label)
    langfuse_trace_enabled = os.getenv("LANGFUSE_TRACE_ENABLED", "true").lower() in ["1", "true", "yes"]

    # Neo4j agentic memory
    neo4j_uri = os.getenv("NEO4J_URI", "").strip()
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    neo4j_password = os.getenv("NEO4J_PASSWORD", "").strip()

    return {
        'nvidia_api_key': nvidia_api_key,
        'llm_model': llm_model,
        'base_url': base_url,
        'nemo_guardrails_config_path': nemo_guardrails_config_path,
        'langfuse_public_key': langfuse_public_key,
        'langfuse_secret_key': langfuse_secret_key,
        'langfuse_base_url': langfuse_base_url,
        'langfuse_prompt_label': langfuse_prompt_label,
        'langfuse_scoring_prompt_name': langfuse_scoring_prompt_name,
        'langfuse_scoring_prompt_label': langfuse_scoring_prompt_label,
        'langfuse_trace_enabled': langfuse_trace_enabled,
        'neo4j_uri': neo4j_uri,
        'neo4j_username': neo4j_username,
        'neo4j_password': neo4j_password,
        'memory_llm_model': memory_llm_model,
    }


logger = get_logging(__name__)


# ── MMSE section constants ───────────────────────────────────────────────────
SECTION_ORDER = [
    "intake",              # greeting + consent → transitions into MMSE
    "orientation_time",
    "orientation_place",
    "immediate_recall",
    "attention",
    "delayed_recall",
    "language",
    "complete",
]

# Maps section key → Langfuse prompt name
SECTION_PROMPT_MAP: dict[str, str] = {
    "intake":            "mmse-intake",              # new prompt — greet + consent
    "orientation_time":  "mmse-orientation-time",
    "orientation_place": "mmse-orientation-place",
    "immediate_recall":  "mmse-immediate-recall",
    "attention":         "mmse-attention-calculation",
    "delayed_recall":    "mmse-delayed-recall",
    "language":          "patient-intake-inital",    # reuse existing prompt
}

# Signals the LLM uses to indicate a section is finished
SECTION_COMPLETE_SIGNALS = [
    # These are matched only against the TAIL of the LLM response (last 200 chars)
    # so opening phrases like "Let's begin with time orientation…" never trigger them.
    "thank you, let's continue",
    "thank you, lets continue",
    "thank you, let us continue",
    "thank you, that completes our session",
    "that completes our session",
    "section is now complete",
    # Intake-specific: LLM may confirm readiness with these closing phrases
    "great, let's begin",
    "great, lets begin",
    "let's begin",
    "lets begin",
    "let's continue",
    "lets continue",
    "let us continue",
    "let's get started",
    "lets get started",
]

# Recall word pool — 3 words drawn randomly per session
RECALL_WORD_POOL: list[list[str]] = [
    ["APPLE",  "TABLE",   "PENNY"],
    ["ORANGE", "CHAIR",   "DOLLAR"],
    ["BALL",   "WINDOW",  "RIVER"],
    ["FLOWER", "DOCTOR",  "SUNSET"],
    ["MIRROR", "GARDEN",  "BRIDGE"],
]
# Sections where entity extraction is useful (rich PII / orientation data)
_ENTITY_SECTIONS = {"orientation_place", "orientation_time", "intake"}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages_with_reset]
    mmse_score: dict
    mmse_scoring_history: list[dict]
    recall_words: list[str]    # generated on first immediate_recall turn
    current_section: str       # active MMSE section key (default: orientation_time)


class SectionAssistant:
    """LangGraph node for a single MMSE section.

    Handles:
    - Dynamic prompt injection of recall_words (for immediate/delayed recall sections)
    - Detecting the LLM's section-completion signal and advancing current_section
    - Generating recall_words on the first immediate_recall turn
    - Langfuse tracing
    """

    def __init__(self, section_name: str, prompt_text: str, llm, langfuse_client=None, langfuse_handler=None, guardrails=None, memory_manager=None):
        self.section_name = section_name
        self.prompt_text = prompt_text
        self.llm = llm
        self.guardrails = guardrails
        self.langfuse_client = langfuse_client
        self.langfuse_handler = langfuse_handler
        self.memory_manager = memory_manager  # MMSEMemoryManager | None

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_runnable(self, resolved_prompt: str):
        # Escape literal { } so ChatPromptTemplate doesn't mistake JSON
        # examples in the Langfuse prompt for template variables.
        # By this point all {word_N} substitutions have already been done.
        safe_prompt = resolved_prompt.replace("{", "{{").replace("}", "}}")
        template = ChatPromptTemplate.from_messages([
            ("system", safe_prompt),
            ("placeholder", "{messages}"),
        ])
        llm_chain = (self.guardrails | self.llm) if self.guardrails else self.llm
        return template | llm_chain

    def _resolve_prompt(self, state: State) -> str:
        """Substitute {word_1}/{word_2}/{word_3} if present in prompt text."""
        if "{word_1}" not in self.prompt_text:
            return self.prompt_text
        words = state.get("recall_words") or ["APPLE", "TABLE", "PENNY"]
        return self.prompt_text.format(
            word_1=words[0] if len(words) > 0 else "APPLE",
            word_2=words[1] if len(words) > 1 else "TABLE",
            word_3=words[2] if len(words) > 2 else "PENNY",
        )

    def _extract_last_user_message(self, state: State) -> str:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "user":
                return str(msg[1])
            if hasattr(msg, "type") and msg.type == "human":
                return str(getattr(msg, "content", ""))
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _to_content_str(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    chunks.append(str(item["text"]))
                elif item:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)

    # ── reasoning-JSON splitter ───────────────────────────────────────────

    _REASONING_MARKER = "<<<REASONING>>>"

    def _split_reasoning(self, raw_text: str) -> tuple[str, dict | None]:
        """Split LLM output into (spoken_text, reasoning_dict).

        Prompts that embed a <<<REASONING>>> marker produce two parts:
          PART 1 — the text spoken aloud to the patient
          PART 2 — a private JSON object for the graph memory system

        Returns (spoken_text, reasoning_dict).  reasoning_dict is None when
        the marker is absent (e.g. sections that do not use the new format).
        """
        import json as _json
        if self._REASONING_MARKER not in raw_text:
            return raw_text, None
        parts = raw_text.split(self._REASONING_MARKER, 1)
        spoken = parts[0].strip()
        json_part = parts[1].strip() if len(parts) > 1 else ""
        reasoning_dict = None
        if json_part:
            # The model sometimes wraps JSON in a markdown code block — strip it.
            json_clean = json_part.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            try:
                reasoning_dict = _json.loads(json_clean)
            except Exception as ex:
                logger.warning(f"[reasoning] JSON parse failed for section '{self.section_name}': {ex} — raw: {json_clean[:300]}")
        return spoken, reasoning_dict

    def _is_section_complete(self, text: str) -> bool:
        # The completion signal must be at the END of the response, not just
        # anywhere in it. This prevents false positives from opening phrases
        # like "...just do your best. Let's begin. To start, what is today's date?"
        # where "let's begin" is a mid-sentence opener, not a section-end marker.
        # Strategy: strip trailing punctuation/whitespace, then check endswith.
        # NOTE: always check against the SPOKEN part (before <<<REASONING>>>).
        normalised = text.strip().rstrip(".!?\"'").strip().lower()
        return any(normalised.endswith(sig) for sig in SECTION_COMPLETE_SIGNALS)

    def _next_section(self) -> str:
        try:
            idx = SECTION_ORDER.index(self.section_name)
            return SECTION_ORDER[idx + 1] if idx + 1 < len(SECTION_ORDER) else "complete"
        except ValueError:
            return "complete"

    # ── main call ──────────────────────────────────────────────────────────

    def __call__(self, state: State, config: RunnableConfig):
        state_updates: dict = {}

        # Reset section tracking when returning to intake (e.g. after "start over")
        if self.section_name == "intake":
            thread_id_early = config.get("configurable", {}).get("thread_id", "unknown") if isinstance(config, dict) else "unknown"
            current_sec = state.get("current_section")
            if current_sec not in (None, "", "intake"):
                # Explicit mid-session restart → reset state and wipe graph
                state_updates["current_section"] = "intake"
                state_updates["recall_words"] = []
                state = {**state, "current_section": "intake", "recall_words": []}
                logger.info("Session reset detected — clearing current_section and recall_words")
                if self.memory_manager is not None:
                    self.memory_manager.clear_session(thread_id_early)
                    logger.info(f"[memory] Graph cleared on restart for session={thread_id_early}")
            elif current_sec in (None, ""):
                # First message of a brand-new session → wipe any leftover graph data
                if self.memory_manager is not None:
                    self.memory_manager.clear_session(thread_id_early)
                    logger.info(f"[memory] Graph cleared for new session={thread_id_early}")


        # Generate recall words at the start of the immediate_recall section
        if self.section_name == "immediate_recall" and not state.get("recall_words"):
            words = random.choice(RECALL_WORD_POOL)
            state_updates["recall_words"] = words
            state = {**state, "recall_words": words}
            logger.info(f"Generated recall words for session: {words}")

        resolved_prompt = self._resolve_prompt(state)
        input_message = self._extract_last_user_message(state)
        thread_id = config.get("configurable", {}).get("thread_id", "unknown") if isinstance(config, dict) else "unknown"

        # ── Inject memory context so the agent doesn't repeat questions ───────
        if self.memory_manager is not None and self.memory_manager.is_connected:
            mem_query = f"MMSE {self.section_name}: what questions were already asked and answered?"
            mem_context = self.memory_manager.get_session_context(thread_id, query=mem_query)
            if mem_context:
                separator = "─" * 45
                resolved_prompt = (
                    f"{resolved_prompt}\n\n─── PREVIOUS SESSION MEMORY {separator}\n"
                    f"{mem_context}\n"
                    f"─── END MEMORY {separator}\n"
                    "IMPORTANT: Do NOT ask again about anything already answered above. "
                    "Reference prior answers naturally if relevant."
                )
                logger.info(
                    f"[memory] Context injected: section={self.section_name} "
                    f"session={thread_id} length={len(mem_context)} chars"
                )
            else:
                logger.info(
                    f"[memory] No prior context: section={self.section_name} session={thread_id}"
                )

        runnable = self._build_runnable(resolved_prompt)

        # Build invoke config with Langfuse CallbackHandler + metadata for this section
        invoke_config = dict(config) if isinstance(config, dict) else {}
        if self.langfuse_handler is not None:
            # LangGraph injects a CallbackManager object (not a list) into config["callbacks"].
            # We must not call .append() on it — extract its handlers into a new list instead.
            existing = invoke_config.get("callbacks")
            if isinstance(existing, list):
                invoke_config["callbacks"] = existing + [self.langfuse_handler]
            elif hasattr(existing, "handlers"):
                invoke_config["callbacks"] = list(existing.handlers) + [self.langfuse_handler]
            else:
                invoke_config["callbacks"] = [self.langfuse_handler]
            invoke_config["metadata"] = {
                **invoke_config.get("metadata", {}),
                "langfuse_session_id": thread_id,
                "langfuse_tags": ["mmse", self.section_name],
            }

        root_obs_ctx = nullcontext()
        generation_ctx = nullcontext()
        if self.langfuse_client is not None:
            root_obs_ctx = self.langfuse_client.start_as_current_observation(
                as_type="span",
                name=f"mmse-{self.section_name}",
                input={"thread_id": thread_id, "input_message": input_message, "section": self.section_name},
            )
            generation_ctx = self.langfuse_client.start_as_current_observation(
                as_type="generation",
                name=f"mmse-llm-{self.section_name}",
                model=os.getenv("AGENT_LLM_MODEL", "unknown"),
                input=input_message,
            )

        with root_obs_ctx as root_observation:
            with generation_ctx as generation_observation:
                max_retries = 3
                for attempt in range(max_retries):
                    result = runnable.invoke(state, invoke_config)
                    if (
                        not result.content
                        or isinstance(result.content, list)
                        and not result.content[0].get("text")
                    ):
                        logger.warning(f"Empty LLM output for section '{self.section_name}', attempt {attempt + 1}/{max_retries}. Retrying.")
                        state = {**state, "messages": state["messages"] + [("user", "Respond with a real output.")]}
                        if generation_observation is not None:
                            generation_observation.update(output="empty_output_retry")
                    else:
                        break
                else:
                    logger.error(f"LLM returned empty output for section '{self.section_name}' after {max_retries} attempts.")

                output_text = self._to_content_str(result.content)
                if generation_observation is not None:
                    generation_observation.update(output=output_text)
                if root_observation is not None:
                    root_observation.update(output={"response_preview": output_text[:500]})

        # ── Split spoken text from embedded reasoning JSON ─────────────────────
        spoken_text, reasoning_dict = self._split_reasoning(output_text)
        logger.debug(
            f"[reasoning] section={self.section_name} has_json={reasoning_dict is not None} "
            f"spoken_len={len(spoken_text)}"
        )
        # Overwrite output_text (and the result message content) with the spoken
        # part only — so TTS, section-complete detection, and LangGraph history
        # all see clean patient-facing text, never the raw JSON block.
        if reasoning_dict is not None:
            output_text = spoken_text
            try:
                result.content = spoken_text
            except AttributeError:
                pass  # immutable — output_text is already correct

        if self.langfuse_client is not None:
            # Flush in a background thread — do NOT block the graph turn on network I/O
            # to the Langfuse server (SSH-tunnelled, can add seconds of latency).
            _lf = self.langfuse_client
            _sec = self.section_name
            def _flush_langfuse():
                try:
                    _lf.flush()
                except Exception as ex:
                    logger.warning(f"Langfuse flush failed after {_sec}: {ex}")
            threading.Thread(target=_flush_langfuse, daemon=True).start()

        # Advance to next section when LLM signals completion
        if self._is_section_complete(output_text):
            next_sec = self._next_section()
            state_updates["current_section"] = next_sec
            logger.info(f"Section '{self.section_name}' complete → advancing to '{next_sec}'")
        # Persist to agentic memory (fire-and-forget background thread).
        # store_message_pair(extract_entities=True) handles entity extraction in
        # a single add_message call.  Never call extract_and_store_entities
        # separately — that inserts the same user message twice and triggers a
        # Neo4j ConstraintError.
        if self.memory_manager is not None:
            _mm = self.memory_manager
            _tid = thread_id
            _utext = input_message
            _atext = output_text
            _sec = self.section_name
            _lf = self.langfuse_client
            _extract = _sec in _ENTITY_SECTIONS
            _reasoning_dict = reasoning_dict  # None for sections without embedded JSON
            def _write_memory():
                # Create a Langfuse span for this memory write so it shows in the UI.
                # start_as_current_observation can be called from any thread; it will
                # create a top-level observation if there is no active trace context.
                lf_ctx = nullcontext()
                if _lf is not None:
                    try:
                        lf_ctx = _lf.start_as_current_observation(
                            as_type="span",
                            name="neo4j-memory-write",
                            input={
                                "session_id": _tid,
                                "section": _sec,
                                "extract_entities": _extract,
                                "user_text_preview": _utext[:300],
                                "ai_text_preview": _atext[:300],
                            },
                        )
                    except Exception:
                        lf_ctx = nullcontext()
                with lf_ctx as obs:
                    try:
                        _mm.store_message_pair(
                            session_id=_tid,
                            user_text=_utext,
                            ai_text=_atext,
                            section=_sec,
                            extract_entities=_extract,
                            reasoning_dict=_reasoning_dict,
                        )  # reasoning_dict is None for sections without embedded JSON
                        if obs is not None:
                            try:
                                obs.update(output={"status": "ok", "extract_entities": _extract})
                            except Exception:
                                pass
                    except Exception as exc:
                        logger.warning(
                            f"[neo4j] Memory write failed for section '{_sec}': {exc}",
                            exc_info=True,
                        )
                        if obs is not None:
                            try:
                                obs.update(output={"status": "error", "error": str(exc)})
                            except Exception:
                                pass
                if _lf is not None:
                    try:
                        _lf.flush()
                    except Exception:
                        pass
            threading.Thread(target=_write_memory, daemon=True).start()
        return {"messages": result, **state_updates}


def get_langfuse_tracing_client(env_config: dict):
    langfuse_host = _normalize_langfuse_base_url(env_config.get('langfuse_base_url'))

    required_langfuse_values = [
        env_config.get('langfuse_public_key'),
        env_config.get('langfuse_secret_key'),
        langfuse_host,
    ]

    if not env_config.get('langfuse_trace_enabled', True):
        logger.info("Langfuse tracing disabled by LANGFUSE_TRACE_ENABLED.")
        return None, None

    if not all(required_langfuse_values):
        logger.info("Langfuse tracing config not fully set; tracing disabled for MMSE chat.")
        return None, None

    try:
        os.environ["LANGFUSE_PUBLIC_KEY"] = env_config['langfuse_public_key']
        os.environ["LANGFUSE_SECRET_KEY"] = env_config['langfuse_secret_key']
        os.environ["LANGFUSE_HOST"] = langfuse_host
        from langfuse import get_client
        langfuse_client = get_client()

        langfuse_handler = None
        if LangfuseCallbackHandler is not None:
            langfuse_handler = LangfuseCallbackHandler()
            logger.info("Langfuse CallbackHandler initialized for LangChain tracing.")
        else:
            logger.warning("langfuse.langchain not available — LangChain callback tracing disabled.")

        logger.info(f"Langfuse tracing enabled for MMSE chat (host={langfuse_host}).")
        return langfuse_client, langfuse_handler
    except Exception as ex:
        logger.error(f"Error initializing Langfuse tracing client: {ex}")
        logger.warning("Continuing without Langfuse tracing.")
        return None, None


def _normalize_langfuse_base_url(raw_base_url: str | None) -> str | None:
    if raw_base_url is None:
        return None

    cleaned = str(raw_base_url).strip().strip('"').strip("'")
    cleaned = os.path.expandvars(cleaned)
    if cleaned == "":
        return None

    if "${" in cleaned and "}" in cleaned:
        raise ValueError(
            "LANGFUSE_BASE_URL contains unresolved environment placeholders. "
            "Set a concrete value like http://<host>:3000 inside vars.env for the app container."
        )

    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"LANGFUSE_BASE_URL is invalid: '{cleaned}'. "
            )
        return cleaned

    normalized = f"http://{cleaned}"
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(
            f"LANGFUSE_BASE_URL is invalid after normalization: '{normalized}'. "
        )
    logger.warning(
        f"LANGFUSE_BASE_URL missing protocol; auto-normalizing from '{cleaned}' to '{normalized}'."
    )
    return normalized


def _normalize_prompt_name(raw_prompt_name: str | None) -> str:
    if raw_prompt_name is None:
        return ""

    cleaned = str(raw_prompt_name).strip().strip('"').strip("'")
    cleaned = os.path.expandvars(cleaned)
    if cleaned == "":
        return ""

    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        path_parts = [part for part in parsed.path.split("/") if part]
        if "prompts" in path_parts:
            prompts_idx = path_parts.index("prompts")
            if prompts_idx + 1 < len(path_parts):
                prompt_name = path_parts[prompts_idx + 1]
                prompt_name = prompt_name.split(":", 1)[0].strip()
                return prompt_name

    return cleaned


def _prompt_name_candidates(prompt_name: str) -> list[str]:
    if not prompt_name:
        return []

    candidates = [prompt_name]

    if "initial" in prompt_name:
        candidates.append(prompt_name.replace("initial", "inital"))
    elif "inital" in prompt_name:
        candidates.append(prompt_name.replace("inital", "initial"))

    return list(dict.fromkeys(candidates))


def _prompt_label_candidates(prompt_label: str | None) -> list[str | None]:
    base_label = str(prompt_label).strip() if prompt_label is not None else ""
    candidates: list[str | None] = []

    if base_label != "":
        candidates.append(base_label)

    for fallback_label in ["latest", "production", None]:
        if fallback_label not in candidates:
            candidates.append(fallback_label)

    return candidates


def _load_local_mmse_chat_prompt() -> str:
    local_prompt_path = SCRIPT_DIR / "system_prompts" / "mmse_system_prompt.txt"
    try:
        with open(local_prompt_path, "r") as f:
            prompt = f.read().strip()
        if prompt:
            logger.info(f"Loaded MMSE chat prompt from local file {local_prompt_path}")
            return prompt
    except Exception as ex:
        logger.warning(f"Failed to read local MMSE chat prompt at {local_prompt_path}: {ex}")

    return (
        "You are a specialized assistant that administers a Mini-Mental State Examination (MMSE)-style "
        "conversational cognitive screening. Ask one question at a time in a clear, respectful tone and "
        "do not provide diagnosis."
    )


def _default_mmse_scoring_prompt() -> str:
    return (
        "You are an MMSE scoring assistant. Given a conversation transcript, return only valid JSON with keys: "
        "total_score (number or null), domain_scores (object), clinical_interpretation (string), confidence "
        "(one of low/medium/high), missing_information (array of strings). Do not include markdown."
    )


def _load_prompt_from_langfuse(env_config: dict, prompt_name: str, prompt_label: str, prompt_purpose: str) -> str:
    langfuse_host = _normalize_langfuse_base_url(env_config.get('langfuse_base_url'))
    normalized_prompt_name = _normalize_prompt_name(prompt_name)
    prompt_candidates = _prompt_name_candidates(normalized_prompt_name)
    label_candidates = _prompt_label_candidates(prompt_label)

    required_langfuse_values = [
        env_config.get('langfuse_public_key'),
        env_config.get('langfuse_secret_key'),
        langfuse_host,
        normalized_prompt_name,
    ]

    if not all(required_langfuse_values):
        raise ValueError(
            f"Langfuse config incomplete for {prompt_purpose}. "
            "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL and the prompt name env var."
        )

    langfuse_client = Langfuse(
        public_key=env_config['langfuse_public_key'],
        secret_key=env_config['langfuse_secret_key'],
        host=langfuse_host,
        timeout=5,
    )

    last_not_found_error = None
    attempted_pairs = []

    for candidate_name in prompt_candidates:
        for label_candidate in label_candidates:
            attempted_pairs.append((candidate_name, label_candidate))
            try:
                if label_candidate is None:
                    prompt_client = langfuse_client.get_prompt(candidate_name)
                else:
                    prompt_client = langfuse_client.get_prompt(
                        candidate_name,
                        label=label_candidate,
                    )
            except Exception as ex:
                if "Prompt not found" in str(ex):
                    logger.warning(
                        f"Prompt not found for {prompt_purpose}: name={candidate_name} label={label_candidate}."
                    )
                    last_not_found_error = ex
                    continue
                raise

            if hasattr(prompt_client, "compile") and callable(prompt_client.compile):
                compiled_prompt = prompt_client.compile()
                if isinstance(compiled_prompt, str) and compiled_prompt.strip():
                    logger.info(f"Loaded {prompt_purpose} from Langfuse name={candidate_name} label={label_candidate}")
                    return compiled_prompt
                # Chat-type prompt: compile() returns a list of message dicts
                if isinstance(compiled_prompt, list):
                    for msg in compiled_prompt:
                        role = ""
                        content = ""
                        if isinstance(msg, dict):
                            role = str(msg.get("role", "")).lower()
                            content = str(msg.get("content", "")).strip()
                        elif hasattr(msg, "role") and hasattr(msg, "content"):
                            role = str(getattr(msg, "role", "")).lower()
                            content = str(getattr(msg, "content", "")).strip()
                        if role == "system" and content:
                            logger.info(f"Loaded {prompt_purpose} (chat→system) from Langfuse name={candidate_name} label={label_candidate}")
                            return content
                    # No system message found — take first non-empty content
                    for msg in compiled_prompt:
                        content = ""
                        if isinstance(msg, dict):
                            content = str(msg.get("content", "")).strip()
                        elif hasattr(msg, "content"):
                            content = str(getattr(msg, "content", "")).strip()
                        if content:
                            logger.info(f"Loaded {prompt_purpose} (chat→first) from Langfuse name={candidate_name} label={label_candidate}")
                            return content

            for attr_name in ["prompt", "content", "text"]:
                prompt_text = getattr(prompt_client, attr_name, None)
                if isinstance(prompt_text, str) and prompt_text.strip():
                    logger.info(f"Loaded {prompt_purpose} from Langfuse name={candidate_name} label={label_candidate}")
                    return prompt_text

    if last_not_found_error is not None:
        raise ValueError(
            f"Prompt not found for {prompt_purpose}. Tried name/label pairs={attempted_pairs}. "
            f"Original prompt value='{prompt_name}'."
        ) from last_not_found_error

    raise ValueError(
        f"Langfuse returned empty prompt content for {prompt_purpose}. "
        f"Tried name/label pairs={attempted_pairs}."
    )



def load_mmse_scoring_prompt(env_config: dict) -> str:
    try:
        return _load_prompt_from_langfuse(
            env_config=env_config,
            prompt_name=env_config.get('langfuse_scoring_prompt_name', ''),
            prompt_label=env_config.get('langfuse_scoring_prompt_label', env_config.get('langfuse_prompt_label', 'production')),
            prompt_purpose="MMSE scoring prompt",
        )
    except Exception as ex:
        logger.warning(f"Falling back to default MMSE scoring prompt due to Langfuse fetch failure: {ex}")
        return _default_mmse_scoring_prompt()


_SECTION_FALLBACK_PROMPTS: dict[str, str] = {
    "intake": (
        "You are a warm, professional clinical intake assistant. "
        "Greet the patient, introduce yourself briefly, and explain that you will be asking a series of "
        "routine questions as part of a standard check-up. "
        "Ask if they are ready to begin. Once they confirm readiness, say exactly: "
        "'Thank you, let's continue.' and stop. Do not start the questions yet."
    ),
    "orientation_time": (
        "You are a clinical intake assistant administering the MMSE Time Orientation section. "
        "Ask these 5 questions one at a time: today's date, day of the week, month, year, season. "
        "When all 5 are answered, say: 'Thank you, let's continue.' and stop."
    ),
    "orientation_place": (
        "You are a clinical intake assistant administering the MMSE Place Orientation section. "
        "Ask these 5 questions one at a time: country, state/region, city, facility name, floor/room. "
        "When all 5 are answered, say: 'Thank you, let's continue.' and stop."
    ),
    "immediate_recall": (
        "You are a clinical intake assistant administering the MMSE Immediate Recall section. "
        "Say these three words one at a time: {word_1}... {word_2}... {word_3}. "
        "Ask the patient to repeat them. Then say: 'Thank you, let's continue.' and stop."
    ),
    "attention": (
        "You are a clinical intake assistant administering the MMSE Attention section. "
        "Ask the patient to subtract 7 from 100 and keep going for 5 steps (93, 86, 79, 72, 65). "
        "When 5 steps are done, say: 'Thank you, let's continue.' and stop."
    ),
    "delayed_recall": (
        "You are a clinical intake assistant administering the MMSE Delayed Recall section. "
        "Ask: 'Earlier I asked you to remember three words. Can you tell me what they were?' "
        "The target words are: {word_1}, {word_2}, {word_3}. "
        "After the patient responds, say: 'Thank you, let's continue.' and stop."
    ),
    "language": (
        "You are a clinical intake assistant administering the MMSE Language section. "
        "Administer in order: (1) name a pencil and a watch, (2) repeat 'No ifs, ands, or buts', "
        "(3) follow a 3-step command. When done, say: 'Thank you, that completes our session.' and stop."
    ),
}


def load_section_prompt(env_config: dict, section_name: str) -> str:
    """Load a per-section MMSE prompt.

    Priority order:
      1. Local file at system_prompts/mmse_<section_name>_prompt.txt
         (allows rapid iteration without touching Langfuse)
      2. Langfuse (keyed by SECTION_PROMPT_MAP)
      3. Built-in fallback string from _SECTION_FALLBACK_PROMPTS
    """
    # ── 1. Try local file first ────────────────────────────────────────────
    local_path = SCRIPT_DIR / "system_prompts" / f"mmse_{section_name}_prompt.txt"
    if local_path.exists():
        try:
            prompt = local_path.read_text(encoding="utf-8").strip()
            if prompt:
                logger.info(f"Loaded MMSE {section_name} prompt from local file {local_path}")
                return prompt
        except Exception as ex:
            logger.warning(f"Failed to read local prompt {local_path}: {ex}")

    # ── 2. Try Langfuse ────────────────────────────────────────────────────
    langfuse_name = SECTION_PROMPT_MAP.get(section_name, "")
    prompt_label = env_config.get('langfuse_prompt_label', 'production')
    try:
        return _load_prompt_from_langfuse(
            env_config=env_config,
            prompt_name=langfuse_name,
            prompt_label=prompt_label,
            prompt_purpose=f"MMSE {section_name} prompt",
        )
    except Exception as ex:
        # ── 3. Built-in fallback ───────────────────────────────────────────
        fallback = _SECTION_FALLBACK_PROMPTS.get(section_name, "You are an MMSE clinical assistant.")
        logger.warning(
            f"Falling back to built-in prompt for section '{section_name}' "
            f"(Langfuse name='{langfuse_name}'): {ex}"
        )
        return fallback


def route_after_section(section_name: str):
    """
    Returns a routing function for use as a conditional edge after a section node.

    - If the section just advanced to a new section, chain to the next section node
      immediately so it asks its first question in the same turn.
    - If the exam just completed (current_section == "complete"), run final scoring.
    - If the section is still in progress (no advance), skip scoring and end the turn
      immediately to avoid an extra LLM call that would cause the UI to hang.
    """
    def _route(state: State) -> str:
        current = state.get("current_section") or "intake"
        if current == "complete":
            # Exam just finished → run final scoring before END
            return "mmse_scoring_agent"
        if current != section_name:
            # Section just advanced → run the new section node immediately
            return f"{current}_node"
        # Section still in progress → skip scoring this turn, go straight to END
        return END
    return _route


def route_to_section(state: State) -> str:
    """Conditional router: dispatches each user turn to the active MMSE section node."""
    messages = state.get("messages") or []
    # Detect a reset: messages were cleared to a single 'Hi' by add_messages_with_reset
    if len(messages) <= 1:
        return "intake_node"

    section = state.get("current_section") or "intake"
    if section == "complete":
        return "mmse_scoring_agent"
    return f"{section}_node"


def create_mmse_graph():
    """Create and return a fresh instance of the MMSE graph.

    Graph flow (per user turn):
        START
          └─ route_to_section ───► <section>_node ─► (mid-section)  END   (1 LLM call)
                                                  └► (section done) <next_section>_node ─► ... ─► END
                                                  └► (exam done)    mmse_scoring_agent ─► END

    Scoring only fires at section transitions and exam completion — NOT on every
    mid-section turn — to avoid the extra LLM call that would hang the UI.
    """
    env_config = load_env_variables()

    assistant_llm = ChatNVIDIA(
        model=env_config['llm_model'],
        base_url=env_config['base_url']
    )

    # Optional guardrails (applied to all section nodes)
    guardrails = None
    if env_config['nemo_guardrails_config_path'] not in (None, ""):
        try:
            rails_config_path = os.path.join("/app", env_config['nemo_guardrails_config_path'])
            config = RailsConfig.from_path(rails_config_path)
            guardrails = RunnableRails(config=config, passthrough=True)
            logger.warning(f"Guardrails loaded from {env_config['nemo_guardrails_config_path']}")
        except Exception as ex:
            logger.error(f"Guardrails load failed: {ex} — continuing without guardrails.")
    else:
        logger.warning("Standing up the MMSE agent without NeMo Guardrails.")

    # Load all section prompts from Langfuse (with fallback)
    section_prompts = {
        section: load_section_prompt(env_config, section)
        for section in SECTION_ORDER
        if section != "complete"
    }

    scoring_prompt = load_mmse_scoring_prompt(env_config)
    langfuse_client, langfuse_handler = get_langfuse_tracing_client(env_config)
    mmse_scoring_runnable = build_scoring_runnable(assistant_llm, scoring_prompt)

    # ── Agentic memory (Neo4j + NVIDIA embeddings) ────────────────────────
    # Initialised from env vars; gracefully disabled if NEO4J_URI / NEO4J_PASSWORD
    # are not set. All memory writes are fire-and-forget, so latency is unaffected.
    memory_manager = None
    if env_config.get('neo4j_uri') and env_config.get('neo4j_password'):
        try:
            memory_manager = MMSEMemoryManager(
                neo4j_uri=env_config['neo4j_uri'],
                neo4j_password=env_config['neo4j_password'],
                neo4j_username=env_config.get('neo4j_username', 'neo4j'),
                nvidia_api_key=env_config.get('nvidia_api_key'),
                llm_model=env_config.get('memory_llm_model', 'meta/llama-3.3-70b-instruct'),
                llm_base_url=env_config.get('base_url', 'https://integrate.api.nvidia.com/v1'),
                langfuse_client=langfuse_client,
            )
            logger.info(
                f"Agentic memory: entity extraction LLM = {env_config.get('memory_llm_model')}, "
                f"conversation LLM = {env_config.get('llm_model')}"
            )
            logger.info("Agentic memory enabled (Neo4j + NV-EmbedCode-7B-v1).")
        except Exception as exc:
            logger.warning(f"Agentic memory init failed — continuing without memory: {exc}")
            memory_manager = None
    else:
        logger.info("Agentic memory disabled (NEO4J_URI or NEO4J_PASSWORD not set).")

    # Build the graph
    builder = StateGraph(State)

    # Add one node per MMSE section
    section_node_names = []
    for section in SECTION_ORDER:
        if section == "complete":
            continue
        node_name = f"{section}_node"
        section_node_names.append(node_name)
        builder.add_node(
            node_name,
            SectionAssistant(
                section_name=section,
                prompt_text=section_prompts[section],
                llm=assistant_llm,
                langfuse_client=langfuse_client,
                langfuse_handler=langfuse_handler,
                #guardrails=guardrails,
                guardrails=None,
                memory_manager=memory_manager,
            ),
        )

    builder.add_node("mmse_scoring_agent", MMSEScoringAgent(mmse_scoring_runnable, langfuse_client=langfuse_client, langfuse_handler=langfuse_handler, memory_manager=memory_manager))

    # Conditional routing from START based on current_section
    route_map = {name: name for name in section_node_names}
    route_map["mmse_scoring_agent"] = "mmse_scoring_agent"
    route_map[END] = END  # allow mid-section nodes to route directly to END (skip scoring)
    builder.add_conditional_edges(START, route_to_section, route_map)

    # After each section node: if section just advanced, chain immediately to the
    # next section node (so it asks its first question in the same turn).
    # Otherwise go to scoring.
    for section in SECTION_ORDER:
        if section == "complete":
            continue
        node_name = f"{section}_node"
        builder.add_conditional_edges(node_name, route_after_section(section), route_map)

    builder.add_edge("mmse_scoring_agent", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("MMSE graph built successfully with per-section nodes: %s", section_node_names)
    return graph


mmse_graph = create_mmse_graph()

if save_graph_to_png:
    try:
        save_image_path = "/graph_images/appgraph_mmse_agent.png"
        with open(save_image_path, "wb") as png:
            png.write(mmse_graph.get_graph(xray=True).draw_mermaid_png())
        logger.info(f"Graph PNG saved to {save_image_path}")
    except Exception as ex:
        logger.error(f"Error saving graph to PNG: {ex}")

demo = launch_demo_ui(mmse_graph)

app = FastAPI()


@app.get("/")
def app_main():
    return {"message": "This is your main app"}


@app.get("/mmse-score")
def get_mmse_score(thread_id: str = Query(..., description="LangGraph thread_id used for the MMSE chat session.")):
    thread_config = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": 10,
        }
    }

    try:
        graph_state_snapshot = mmse_graph.get_state(thread_config)
        values = getattr(graph_state_snapshot, "values", None)
        if not isinstance(values, dict):
            raise HTTPException(status_code=404, detail="No MMSE score found for this thread_id.")

        latest_score = values.get("mmse_score")
        score_history = values.get("mmse_scoring_history", [])

        if latest_score is None:
            raise HTTPException(status_code=404, detail="No MMSE score found for this thread_id.")

        return {
            "thread_id": thread_id,
            "mmse_score": latest_score,
            "mmse_scoring_history": score_history,
        }
    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"Failed to retrieve MMSE score for thread_id={thread_id}: {ex}")
        raise HTTPException(status_code=500, detail="Unable to retrieve MMSE score.")


app = gr.mount_gradio_app(app, demo, path="/mmse-agent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861,
                        help="Port for the Gradio UI / FastAPI server.")
    parser.add_argument("--debug", action="store_true",
                        help="Start debugpy and wait for a VS Code remote-attach on --debug-port.")
    parser.add_argument("--debug-port", type=int, default=5678,
                        help="Port debugpy listens on (default: 5678).")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", args.debug_port))
        logger.info(f"debugpy listening on 0.0.0.0:{args.debug_port} — attach from VS Code now …")
        debugpy.wait_for_client()
        logger.info("VS Code debugger attached.")

    uvicorn.run(
        "graph_mmse_agent:app",
        host="0.0.0.0",
        port=args.port,
        # reload forks a child process which breaks debugpy; disable when debugging
        reload=not args.debug,
        reload_includes=["*.env", "*.txt", "*.co", "*.yml", "*.py"],
        reload_excludes=["**/__pycache__/**", "**/graph_images/**", "**/*.png"],
        reload_dirs=[
            "/app/graph_definitions",
            "/app/nmgr-config-store",
            "/app/utils",
            "/env_vars",
        ],
    )