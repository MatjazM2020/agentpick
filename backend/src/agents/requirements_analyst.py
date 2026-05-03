"""
Requirements Analyst agent.

Transforms a user query into structured constraints and preferences.
Uses an LLM agent to interpret intent and extract task type, constraints, and preferences.

Responsibilities:
- Parse natural language query
- Extract task_type (e.g., summarization, qa, code_generation)
- Extract constraints (latency, memory, license, hardware)
- Extract preferences (speed vs accuracy, model size, etc.)
- Return JSON with strict validation
- Retry on parse failures (up to 3 attempts)
- Fall back to safe defaults on final failure
"""

import json
import logging
from typing import Optional
from agent_framework import Agent
from pydantic import BaseModel, Field, ValidationError

from src.core.agent_output import agent_run_text
from src.core.state import RecommendationState


logger = logging.getLogger(__name__)


class RequirementsOutput(BaseModel):
    """Validated output from requirements extraction."""
    task_type: str = Field(description="Inferred task type")
    constraints: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    confidence: float = Field(default=1.0)


# Safe defaults when all retries fail
_SAFE_DEFAULTS = {
    "task_type": "general",
    "constraints": {},
    "preferences": {
        "speed_vs_accuracy": "balanced",
        "model_size": "medium"
    }
}


async def run(
    state: RecommendationState,
    agent: Agent,
    max_retries: int = 3
) -> RecommendationState:
    """
    Extract structured requirements from user query.
    
    Args:
        state: RecommendationState with user_query populated
        agent: RequirementsAnalyst agent configured with instructions
        max_retries: Number of retries on JSON parse failure
        
    Returns:
        Updated state with task_type, constraints, preferences populated
    """
    
    query = state.user_query
    logger.info(f"[RequirementsAnalyst] Processing query: {query}")
    
    prompt = f"""Analyze this user query and extract structured requirements.

Query: "{query}"

Return ONLY a valid JSON object (no markdown, no explanations) with this structure:
{{
  "task_type": "string (e.g., summarization, qa, code_generation, translation, etc.)",
  "constraints": {{
    "max_latency_ms": "number (optional)",
    "max_memory_gb": "number (optional)",
    "hardware": "string (optional, e.g., cpu_only, gpu)",
    "license": "string (optional, e.g., apache-2, mit)"
  }},
  "preferences": {{
    "speed_vs_accuracy": "string (speed, balanced, or accuracy)",
    "model_size": "string (tiny, small, medium, large)",
    "popularity": "boolean (optional)"
  }},
  "confidence": "number (0.0 to 1.0)"
}}

Rules:
1. Return ONLY valid JSON. No markdown, no text outside JSON.
2. Omit optional fields if not mentioned in the query.
3. For task_type, make best inference or use "general" if unclear.
4. Set confidence to 1.0 if all fields clear, lower if guessing.
5. Be strict: only extract explicitly mentioned or clearly implied requirements.
"""
    
    raw_output = None
    parse_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[RequirementsAnalyst] Attempt {attempt}/{max_retries}")
            
            # Run agent
            run_result = await agent.run(prompt)
            raw_output = agent_run_text(run_result)
            logger.debug(f"[RequirementsAnalyst] Raw LLM output: {raw_output}")
            
            # Try to parse JSON
            output_dict = json.loads(raw_output)
            
            # Validate with Pydantic
            validated = RequirementsOutput(**output_dict)
            
            # Success - update state
            state.task_type = validated.task_type
            state.constraints = validated.constraints
            state.preferences = validated.preferences
            state.requirements_extracted = True
            
            logger.info(
                f"[RequirementsAnalyst] ✓ Success. "
                f"task_type={validated.task_type}, "
                f"confidence={validated.confidence}"
            )
            
            state.agent_logs.append(
                f"RequirementsAnalyst: Extracted task_type='{validated.task_type}' "
                f"(confidence={validated.confidence})"
            )
            
            return state
            
        except (json.JSONDecodeError, ValidationError) as e:
            parse_error = e
            logger.warning(
                f"[RequirementsAnalyst] Attempt {attempt} failed: {type(e).__name__}: {str(e)}"
            )
            
            if attempt == max_retries:
                logger.error(
                    f"[RequirementsAnalyst] All {max_retries} retries exhausted. "
                    f"Using safe defaults."
                )
                if raw_output:
                    logger.error(f"[RequirementsAnalyst] Final raw output: {raw_output}")
                state.agent_logs.append(
                    f"RequirementsAnalyst: Failed to parse after {max_retries} retries. "
                    f"Error: {parse_error}. Using safe defaults."
                )
    
    # Fallback to safe defaults
    state.task_type = _SAFE_DEFAULTS["task_type"]
    state.constraints = _SAFE_DEFAULTS["constraints"]
    state.preferences = _SAFE_DEFAULTS["preferences"]
    state.requirements_extracted = False  # Mark as unsuccessful extraction
    
    logger.warning(
        f"[RequirementsAnalyst] Fallback: task_type={state.task_type}, "
        f"constraints={state.constraints}, preferences={state.preferences}"
    )
    
    return state
