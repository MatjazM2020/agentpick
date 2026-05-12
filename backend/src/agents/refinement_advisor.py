"""
Refinement Advisor agent.

When the query is too broad, produces an assistant-facing message with
questions and suggested details only for requirement slots that are still
missing (see query_specificity).

Does not retrieve from Qdrant and must not invent model names.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agent_framework import Agent
from pydantic import BaseModel, Field, ValidationError

from src.core.agent_output import agent_run_text
from src.core.query_specificity import SLOT_DESCRIPTIONS, fallback_refinement_text, missing_requirement_slots
from src.core.state import RecommendationState


logger = logging.getLogger(__name__)


class RefinementAdvisorOutput(BaseModel):
    """Validated JSON from the Refinement Advisor."""

    intro: str = Field(description="Short paragraph explaining the query is broad")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Questions only for missing information; do not repeat known facts",
    )
    suggested_user_inputs: list[str] = Field(
        default_factory=list,
        description="Concrete examples of what the user could type next",
    )


async def run(
    state: RecommendationState,
    agent: Agent,
    max_retries: int = 2,
) -> RecommendationState:
    """
    Fill state.refinement_assistant_text and state.follow_up_questions from the LLM.

    On repeated parse failure, uses query_specificity.fallback_refinement_text.
    """
    slots = missing_requirement_slots(state)
    slot_lines = "\n".join(f"- {s}: {SLOT_DESCRIPTIONS.get(s, s)}" for s in slots)

    nl_context = state.natural_language_context_for_requirements()

    payload = {
        "task_type": state.task_type,
        "constraints": state.constraints,
        "preferences": state.preferences,
        "requirements_confidence": state.requirements_confidence,
    }

    prompt = f"""The user's model-recommendation query is underspecified. We will NOT search a catalog yet.

User messages so far (oldest first; use all of this when phrasing questions):
\"\"\"{nl_context}\"\"\"

Latest turn (same as last user block above if single message): "{state.user_query}"

Already extracted (do not ask again for anything clearly covered here):
{json.dumps(payload, indent=2)}

Information still missing (you MUST only ask about these areas, not about fields already present):
{slot_lines if slot_lines else "(none — still keep the message short and ask how they will use models in practice.)"}

Return ONLY valid JSON (no markdown) with this shape:
{{
  "intro": "1-2 sentences explaining we need a bit more detail before recommending.",
  "follow_up_questions": ["question only for missing slots", "..."],
  "suggested_user_inputs": ["example phrase the user could send next", "..."]
}}

Rules:
1. follow_up_questions must address ONLY missing areas listed above.
2. Do not name or recommend specific Hugging Face or other models (we have not retrieved yet).
3. Keep follow_up_questions to at most 4 items.
4. suggested_user_inputs: 2-4 short example snippets the user could paste.
"""

    raw_output: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[RefinementAdvisor] Attempt {attempt}/{max_retries}")
            run_result = await agent.run(prompt)
            raw_output = agent_run_text(run_result)
            json_str = raw_output.strip()
            if "```" in raw_output:
                start = raw_output.find("```json")
                if start != -1:
                    start = start + 7
                else:
                    start = raw_output.find("```") + 3
                end = raw_output.find("```", start)
                if end != -1:
                    json_str = raw_output[start:end].strip()
            data = json.loads(json_str)
            out = RefinementAdvisorOutput(**data)
            parts = [out.intro.strip(), ""]
            if out.follow_up_questions:
                parts.append("Questions:")
                for q in out.follow_up_questions:
                    parts.append(f"- {q.strip()}")
                parts.append("")
            if out.suggested_user_inputs:
                parts.append("You could reply with something like:")
                for s in out.suggested_user_inputs:
                    parts.append(f"- {s.strip()}")
            text = "\n".join(p for p in parts if p is not None).strip()
            state.refinement_assistant_text = text
            state.follow_up_questions = [q.strip() for q in out.follow_up_questions if q.strip()]
            state.stopped_for_query_refinement = True
            state.agent_logs.append(
                f"RefinementAdvisor: produced interactive refinement ({len(state.follow_up_questions)} questions)"
            )
            return state
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"[RefinementAdvisor] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                break
        except Exception as e:
            logger.warning(f"[RefinementAdvisor] Attempt {attempt} LLM error: {e}")
            if attempt == max_retries:
                break

    fb = fallback_refinement_text(slots)
    state.refinement_assistant_text = fb
    state.follow_up_questions = []
    state.stopped_for_query_refinement = True
    state.agent_logs.append("RefinementAdvisor: used fallback refinement text")
    return state
