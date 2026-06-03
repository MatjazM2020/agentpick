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


class PopularityIntent(BaseModel):
    """How (and whether) the query depends on popularity metrics, for DB routing."""
    mode: str = Field(
        default="none",
        description="'none' (semantic only), 'popularity_only', or 'hybrid'",
    )
    sort_by: Optional[str] = Field(
        default=None, description="'downloads', 'likes', or null"
    )
    min_downloads: Optional[int] = Field(default=None)
    min_likes: Optional[int] = Field(default=None)


class RequirementsOutput(BaseModel):
    """Validated output from requirements extraction."""
    task_type: str = Field(description="Inferred task type")
    constraints: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    popularity: PopularityIntent = Field(default_factory=PopularityIntent)
    confidence: float = Field(default=1.0)


# Safe defaults when all retries fail
_SAFE_DEFAULTS = {
    "task_type": "general",
    "constraints": {},
    "preferences": {
        "speed_vs_accuracy": "balanced",
        "model_size": "medium"
    },
    "popularity": {"mode": "none"},
}


async def run(
    state: RecommendationState,
    agent: Agent,
    max_retries: int = 3
) -> RecommendationState:
    """
    Extract structured requirements (task_type, constraints, preferences) from
    the user query, retrying on parse failure and using safe defaults on final
    failure. Returns the updated state.
    """
    query = state.natural_language_context_for_requirements() or state.user_query
    logger.info(f"[RequirementsAnalyst] Processing context ({len(query)} chars)")
    
    prompt = f"""Analyze this user query and extract structured requirements.

User messages / conversation (oldest first, use all turns):
\"\"\"{query}\"\"\"

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
  "popularity": {{
    "mode": "none | popularity_only | hybrid",
    "sort_by": "downloads | likes | null",
    "min_downloads": "number or null",
    "min_likes": "number or null"
  }},
  "confidence": "number (0.0 to 1.0)"
}}

Rules:
1. Return ONLY valid JSON. No markdown, no text outside JSON.
2. Omit optional fields if not mentioned in the query.
3. For task_type, make best inference or use "general" if unclear.
4. Set confidence to 1.0 if all fields clear, lower if guessing.
5. Be strict: only extract explicitly mentioned or clearly implied requirements.
6. popularity.mode rules (decide how the request depends on popularity metrics):
   - "popularity_only": the request is primarily about popularity with no real semantic task,
     e.g. "most popular models", "top/trending/best-known models", "models with the most downloads",
     "rank/sort models by downloads or likes". Set sort_by to "downloads" (default) or "likes".
   - "hybrid": the request needs BOTH semantic relevance AND popularity,
     e.g. "best coding models with lots of downloads", "popular summarization models".
     Set sort_by when a metric is named.
   - "none": no popularity signal (default).
   - Thresholds: ">100k downloads" -> min_downloads=100000; "at least 1,000 likes" -> min_likes=1000.
     Leave thresholds null when none are stated. A stated threshold implies popularity_only or hybrid.
"""
    
    raw_output = None
    parse_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[RequirementsAnalyst] Attempt {attempt}/{max_retries}")

            run_result = await agent.run(prompt)
            raw_output = agent_run_text(run_result)
            logger.debug(f"[RequirementsAnalyst] Raw LLM output: {raw_output}")

            output_dict = json.loads(raw_output)
            validated = RequirementsOutput(**output_dict)

            state.task_type = validated.task_type
            state.constraints = validated.constraints
            state.preferences = validated.preferences
            state.popularity = validated.popularity.model_dump()
            state.requirements_confidence = validated.confidence
            state.requirements_extracted = True
            
            logger.info(
                f"[RequirementsAnalyst] ✓ Success. "
                f"task_type={validated.task_type}, "
                f"popularity_mode={state.popularity.get('mode')}, "
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
    state.popularity = dict(_SAFE_DEFAULTS["popularity"])
    state.requirements_confidence = 0.25
    state.requirements_extracted = False  # Mark as unsuccessful extraction
    
    logger.warning(
        f"[RequirementsAnalyst] Fallback: task_type={state.task_type}, "
        f"constraints={state.constraints}, preferences={state.preferences}"
    )
    
    return state
