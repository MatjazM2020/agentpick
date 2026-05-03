"""
Agent factory for creating and initializing agents.

Centralizes agent instantiation with consistent configuration across the system.
"""

import inspect
import os
from typing import Optional, Dict

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Requirements Analyst + Synthesizer use OpenAI Chat Completions via agent-framework.
DEFAULT_AGENT_CHAT_MODEL = "gpt-5.4-nano"


def _get_client() -> OpenAIChatClient:
    """
    OpenAI Chat Completions client for agent LLM calls.

    Environment (agent-framework / OpenAIChatClient):
    - OPENAI_API_KEY: required for successful API calls (unless provided elsewhere)
    - OPENAI_CHAT_MODEL_ID: optional; defaults to DEFAULT_AGENT_CHAT_MODEL (passed as model / model_id per SDK)
    - OPENAI_BASE_URL: optional; OpenAI-compatible API base URL
    """
    model_id = os.getenv("OPENAI_CHAT_MODEL_ID", DEFAULT_AGENT_CHAT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    # PyPI-stable agent-framework uses ``model``; some releases/docs use ``model_id``.
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


def _create_requirements_analyst_agent() -> Agent:
    """
    Create Requirements Analyst agent.
    
    Responsible for:
    - Parsing natural language queries
    - Extracting structured constraints and preferences
    - Returning JSON with task type, constraints, preferences
    """
    client = _get_client()
    
    instructions = """You are a Requirements Analyst agent for a machine learning model recommendation system.

Your job: Transform user queries into structured JSON with:
1. task_type: The ML task (e.g., summarization, qa, code_generation, translation, classification, etc.)
2. constraints: dict with latency, memory, license, hardware constraints
3. preferences: dict with user preferences (speed vs accuracy, model size, etc.)

Rules:
- Return ONLY valid JSON (no markdown, no explanations)
- Be thorough in extracting all mentioned constraints
- If a constraint is not mentioned, omit it (don't guess)
- Set confidence based on how clear the requirements are
"""
    
    agent = Agent(
        client=client,
        name="RequirementsAnalyst",
        instructions=instructions,
    )
    
    return agent


def _create_synthesizer_agent() -> Agent:
    """
    Create Synthesizer agent.
    
    Responsible for:
    - Generating human-readable explanations for recommendations
    - Converting scores and metadata into clear reasoning
    - Suggesting follow-up questions for refinement
    
    CRITICAL: No hallucination - explanations must be grounded in provided data.
    """
    client = _get_client()
    
    instructions = """You are a Synthesizer agent for explaining model recommendations.

Your job: Generate clear, factual explanations for why specific models are recommended.

Rules:
- Base all explanations on provided metadata and scores
- Never invent features or capabilities
- Explain trade-offs between models
- Generate 2-3 follow-up clarifying questions if the user might refine
- Return ONLY valid JSON (no markdown, no explanations)
- Keep explanations concise (2-3 sentences per model)

Grounded explanation structure:
{
  "recommendations": [
    {
      "model_id": "string",
      "score": float,
      "why": "explanation based on score breakdown and metadata",
      "pros": ["list of strengths from metadata"],
      "cons": ["list of limitations"]
    }
  ],
  "follow_up_questions": ["question1", "question2"]
}
"""
    
    agent = Agent(
        client=client,
        name="Synthesizer",
        instructions=instructions,
    )
    
    return agent


class AgentFactory:
    """Factory for creating and managing agents."""
    
    _agents: Optional[Dict[str, Agent]] = None
    
    @classmethod
    def create_all(cls) -> Dict[str, Agent]:
        """
        Create all agents needed for the recommendation pipeline.
        
        Caches agents for reuse.
        
        Returns:
            Dict mapping agent names to Agent objects:
            {
                "requirements_analyst": Agent,
                "synthesizer": Agent
            }
        """
        if cls._agents is None:
            cls._agents = {
                "requirements_analyst": _create_requirements_analyst_agent(),
                "synthesizer": _create_synthesizer_agent(),
            }
        return cls._agents
    
    @classmethod
    def reset(cls) -> None:
        """Reset cached agents (useful for testing)."""
        cls._agents = None


def create_agents() -> Dict[str, Agent]:
    """
    Create all agents needed for the recommendation pipeline.
    
    Public interface to AgentFactory.
    
    Returns:
        Dict mapping agent names to Agent objects
    """
    return AgentFactory.create_all()
