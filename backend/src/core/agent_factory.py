"""
Agent factory — Microsoft Agent Framework (``agent-framework``).

See ``docs/agent_patterns.py`` for tool and session patterns.
"""

import inspect
import os
from typing import Dict, Optional

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

DEFAULT_AGENT_CHAT_MODEL = "gpt-5.4-nano"

_shared_client: Optional[OpenAIChatClient] = None
_ranker_agent: Optional[Agent] = None


def get_chat_client() -> OpenAIChatClient:
    """Return a process-wide OpenAI chat client."""
    global _shared_client
    if _shared_client is None:
        _shared_client = _build_client()
    return _shared_client


def _build_client() -> OpenAIChatClient:
    model_id = os.getenv("OPENAI_CHAT_MODEL_ID", DEFAULT_AGENT_CHAT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    _params = inspect.signature(OpenAIChatClient.__init__).parameters
    kwargs: dict = {}
    if "model" in _params:
        kwargs["model"] = model_id
    elif "model_id" in _params:
        kwargs["model_id"] = model_id
    else:
        kwargs["model"] = model_id
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    try:
        return OpenAIChatClient(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize OpenAI chat client: {e}. "
            "Set OPENAI_API_KEY and optionally OPENAI_CHAT_MODEL_ID / OPENAI_BASE_URL."
        ) from e


def _create_ranker_agent() -> Agent:
    client = get_chat_client()
    instructions = """You are the Ranker agent in an ML model recommendation system.

Given a pool of Hugging Face model candidates already retrieved by embedding
similarity, you run a staged, evidence-weighted decision:

1. HARD FILTER: drop candidates that violate the user's EXPLICIT requirements
   (pipeline/task type, modality, language, hardware, license).
2. SCORE survivors in two separate evaluations (each returns 0.0–1.0 per model):
   a. TASK/DOMAIN MATCH — explicit training or specialization for the requested task.
   b. OBJECTIVE EVIDENCE — benchmarks, evaluation results, training details, datasets,
      architecture, documented capabilities (tool use, function calling). Ignore marketing hype.
   Community signal (downloads/likes) is computed deterministically in Python — do not score it.
3. EXPLAIN each top pick with 2-3 readable sentences grounded in model_card content.
   Prefer citing objective facts over marketing language.

Composite score (computed in Python, not by you):
  0.40 * task_match + 0.50 * objective_evidence + 0.10 * community_signal

Rules:
- Use ONLY metadata and model_card text in the prompt. Never invent benchmarks or capabilities.
- Penalize unsubstantiated marketing claims; reward verifiable evidence.
- Return ONLY valid JSON in the requested schema."""
    return Agent(client=client, name="Ranker", instructions=instructions)


def create_agents() -> Dict[str, Agent]:
    """Return shared LLM agents used by the orchestrator tools."""
    global _ranker_agent
    if _ranker_agent is None:
        _ranker_agent = _create_ranker_agent()
    return {"ranker": _ranker_agent}


def reset_agents() -> None:
    """Clear cached client and agents (for tests)."""
    global _shared_client, _ranker_agent
    _shared_client = None
    _ranker_agent = None
