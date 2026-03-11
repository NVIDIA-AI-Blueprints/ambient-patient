# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone MMSE Scoring Agent module.

Usage:
    from graph_definitions.mmse_scoring_agent import MMSEScoringAgent, build_scoring_runnable

This module can be used independently or wired into any LangGraph graph as a node.
"""

import json
import re
import datetime
import logging
import threading
from typing import Annotated

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default fallback scoring prompt (used when Langfuse is unavailable)
# ---------------------------------------------------------------------------

DEFAULT_SCORING_PROMPT = """\
You are a specialist MMSE scoring assistant.

Given a full conversation transcript between a clinical assistant and a patient,
extract and score all completed MMSE sections.

Return ONLY valid JSON (no markdown, no prose) in exactly this structure:

{
  "exam": "MMSE",
  "date": "<ISO date of exam, or null if unknown>",
  "scores": {
    "orientation_time":        <0-5 or null>,
    "orientation_place":       <0-5 or null>,
    "immediate_recall":        <0-3 or null>,
    "attention_calculation":   <0-5 or null>,
    "short_term_memory":       <0-3 or null>,
    "naming":                  <0-2 or null>,
    "repetition":              <0-1 or null>,
    "three_step_command":      <0-3 or null>,
    "read_and_follow":         "TODO",
    "write_sentence":          "TODO",
    "copy_drawing":            "TODO"
  },
  "total_score":     <integer, sum of non-null non-TODO scores>,
  "max_available":   <integer, max possible given completed sections>,
  "interpretation":  "<Normal|Mild|Moderate|Severe> cognitive impairment",
  "confidence":      "<low|medium|high>",
  "notes":           "<notable observations: hesitation, distress, refusals, unusual responses>",
  "missing_sections": ["<list of section names not yet administered>"]
}

Scoring thresholds (out of max 30, adjust proportionally if exam is partial):
  24-30: Normal cognition
  18-23: Mild cognitive impairment
  10-17: Moderate cognitive impairment
  Below 10: Severe cognitive impairment

IMPORTANT:
- Do NOT diagnose. You are a scoring tool only.
- If a section was refused, score 0 and note it.
- If a section was not reached yet, set its score to null and add it to missing_sections.
- Return only the JSON object. No other text.

Conversation transcript:
{conversation_transcript}
"""


# ---------------------------------------------------------------------------
# MMSEScoringAgent
# ---------------------------------------------------------------------------

class MMSEScoringAgent:
    """
    LangGraph-compatible node that scores an MMSE conversation transcript.

    Can be instantiated standalone or wired into a StateGraph as a node:

        builder.add_node("mmse_scoring_agent", MMSEScoringAgent(runnable))

    Input state keys used:
        messages (list)              — full conversation history
        section_scores (dict, opt.)  — per-section scores collected during exam nodes

    Output state keys written:
        mmse_score (dict)            — latest full score payload
        mmse_scoring_history (list)  — all score payloads produced during the session
    """

    def __init__(self, runnable: Runnable, langfuse_client=None, langfuse_handler=None, memory_manager=None):
        self.runnable = runnable
        self.langfuse_client = langfuse_client
        self.langfuse_handler = langfuse_handler
        self.memory_manager = memory_manager  # MMSEMemoryManager | None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _messages_to_transcript(self, state: dict) -> str:
        transcript_lines = []
        for msg in state.get("messages", []):
            role = "unknown"
            content = ""

            if isinstance(msg, tuple) and len(msg) >= 2:
                role = str(msg[0])
                content = str(msg[1])
            elif isinstance(msg, dict):
                role = str(msg.get("role", "unknown"))
                content = str(msg.get("content", ""))
            elif hasattr(msg, "type"):
                role = str(getattr(msg, "type", "unknown"))
                content = str(getattr(msg, "content", ""))
            else:
                content = str(msg)

            transcript_lines.append(f"{role}: {content}")

        return "\n".join(transcript_lines)

    def _safe_parse_score_json(self, raw_text: str) -> dict:
        if not raw_text:
            return {
                "total_score": None,
                "domain_scores": {},
                "clinical_interpretation": "Scoring output was empty.",
                "confidence": "low",
            }

        # Try direct parse
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # Extract first JSON object if wrapped in prose/markdown
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw_text)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass

        return {
            "total_score": None,
            "domain_scores": {},
            "clinical_interpretation": "Unable to parse scoring response as JSON.",
            "confidence": "low",
            "raw_response": raw_text,
        }

    def _get_thread_id(self, config: RunnableConfig) -> str:
        if isinstance(config, dict):
            return config.get("configurable", {}).get("thread_id", "unknown")
        return "unknown"

    def _write_langfuse_span(self, thread_id: str, transcript: str, score_payload: dict):
        if self.langfuse_client is None:
            return
        try:
            with self.langfuse_client.start_as_current_observation(
                as_type="span",
                name="mmse-scoring-result",
                input={
                    "thread_id": thread_id,
                    "transcript_preview": transcript[:1000],
                },
                output=score_payload,
            ):
                pass
            # Flush in a background thread — don't block the scoring step on network I/O
            _lf = self.langfuse_client
            threading.Thread(target=lambda: _lf.flush(), daemon=True).start()
        except Exception as ex:
            logger.warning(f"Langfuse write failed for MMSE score payload: {ex}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_transcript(self, transcript: str, config: RunnableConfig = None, thread_id: str = "unknown") -> dict:
        """
        Score a raw transcript string directly.
        Returns the parsed score dict.
        """
        invoke_config = dict(config) if isinstance(config, dict) else {}
        if self.langfuse_handler is not None:
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
                "langfuse_tags": ["mmse", "scoring"],
            }

        score_result = self.runnable.invoke({"conversation_transcript": transcript}, invoke_config)

        score_text = ""
        if hasattr(score_result, "content"):
            if isinstance(score_result.content, str):
                score_text = score_result.content
            elif isinstance(score_result.content, list):
                chunks = []
                for item in score_result.content:
                    if isinstance(item, dict) and item.get("text"):
                        chunks.append(str(item.get("text")))
                    elif item:
                        chunks.append(str(item))
                score_text = "\n".join(chunks)
            else:
                score_text = str(score_result.content)
        else:
            score_text = str(score_result)

        return self._safe_parse_score_json(score_text)

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        """LangGraph node entry point."""
        transcript = self._messages_to_transcript(state)
        thread_id = self._get_thread_id(config)

        parsed_score = self.score_transcript(transcript, config, thread_id=thread_id)

        # Merge in any per-section scores collected by upstream question nodes
        section_scores = state.get("section_scores", {})
        if section_scores:
            parsed_score.setdefault("scores", {}).update(section_scores)

        score_payload = {
            **parsed_score,
            "thread_id": thread_id,
            "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        self._write_langfuse_span(thread_id, transcript, score_payload)

        # Record scoring as a reasoning trace in Neo4j agentic memory
        if self.memory_manager is not None:
            try:
                self.memory_manager.record_scoring_trace(thread_id, score_payload)
            except Exception as exc:
                logger.debug(f"Memory scoring trace skipped: {exc}")

        history = list(state.get("mmse_scoring_history", []))
        history.append(score_payload)

        return {
            "mmse_score": score_payload,
            "mmse_scoring_history": history,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_scoring_runnable(llm, scoring_prompt: str = None):
    """
    Build and return the LangChain runnable for MMSE scoring.

    Args:
        llm:             Any LangChain chat model (ChatNVIDIA, ChatOpenAI, etc.)
        scoring_prompt:  Override prompt text. Uses DEFAULT_SCORING_PROMPT if None.

    Returns:
        A LangChain Runnable ready to be passed to MMSEScoringAgent().
    """
    prompt_text = scoring_prompt or DEFAULT_SCORING_PROMPT
    # Escape all literal curly braces so LangChain doesn't treat JSON fields as
    # template variables, then restore the one real placeholder.
    safe_prompt = (
        prompt_text
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("{{conversation_transcript}}", "{conversation_transcript}")
    )
    scoring_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", safe_prompt),
            ("human", "{conversation_transcript}"),
        ]
    )
    return scoring_prompt_template | llm
