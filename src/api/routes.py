"""
API routes for the recommendation system.

Endpoints:
- POST /recommend - Generate recommendations for a query
- POST /recommend/refine - Refine previous recommendations
- POST /recommend/debug - Get full internal state
- GET /health - Health check
"""

import logging
import asyncio
from typing import Optional
from flask import Blueprint, request, jsonify
from pydantic import BaseModel, Field, ValidationError

from src.services.recommendation_pipeline import run_recommendation
from src.core.state import RecommendationState


logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# === Pydantic Models for Validation ===

class RecommendRequest(BaseModel):
    """Validation model for /recommend endpoint."""
    query: str = Field(..., min_length=1, max_length=500)


class RefineRequest(BaseModel):
    """Validation model for /recommend/refine endpoint."""
    query: str = Field(..., min_length=1, max_length=500)
    state: dict = Field(..., description="Previous RecommendationState as dict")


class DebugRequest(BaseModel):
    """Validation model for /recommend/debug endpoint."""
    state: dict = Field(..., description="RecommendationState as dict")


# === Helper Functions ===

def _state_to_json(state: RecommendationState) -> dict:
    """Convert RecommendationState to JSON-serializable dict."""
    return {
        "user_query": state.user_query,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in state.messages
        ],
        "task_type": state.task_type,
        "constraints": state.constraints,
        "preferences": state.preferences,
        "retrieved_models": state.retrieved_models,
        "scored_models": [
            {
                "model_id": m.model_id,
                "score": m.score,
                "score_breakdown": m.score_breakdown,
                "metadata": m.metadata
            }
            for m in state.scored_models
        ],
        "final_recommendations": [
            {
                "model_id": m.model_id,
                "score": m.score,
                "score_breakdown": m.score_breakdown,
                "metadata": m.metadata
            }
            for m in state.final_recommendations
        ],
        "explanations": state.explanations,
        "follow_up_questions": state.follow_up_questions,
        "iteration": state.iteration,
        "requirements_extracted": state.requirements_extracted,
        "retrieval_complete": state.retrieval_complete,
        "evaluation_complete": state.evaluation_complete,
        "agent_logs": state.agent_logs
    }


def _dict_to_state(data: dict) -> RecommendationState:
    """Reconstruct RecommendationState from JSON dict."""
    try:
        return RecommendationState.parse_obj(data)
    except ValidationError as e:
        logger.error(f"State deserialization failed: {e}")
        raise ValueError(f"Invalid state format: {str(e)}")


# === Endpoints ===

@api_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Returns:
        {
            "status": "ok",
            "service": "recommendation-api"
        }
    """
    return jsonify({
        "status": "ok",
        "service": "recommendation-api"
    }), 200


@api_bp.route("/recommend", methods=["POST"])
def recommend():
    """
    Generate recommendations for a new query.
    
    Request:
        {
            "query": "I need a small model for classification on CPU"
        }
    
    Response:
        {
            "status": "success",
            "recommendations": [
                {
                    "model_id": "model-1",
                    "score": 0.95,
                    "score_breakdown": {...},
                    "metadata": {...}
                }
            ],
            "metadata": {
                "num_candidates": 10,
                "execution_time": 2.34,
                "iteration": 1
            },
            "state": {...}
        }
    
    Error Response:
        {
            "status": "error",
            "error": "error_code",
            "message": "error description"
        }
    """
    
    # Validate request
    try:
        payload = request.get_json()
        if not payload:
            raise ValueError("Request body is empty")
        
        req = RecommendRequest(**payload)
        query = req.query.strip()
    except (ValueError, ValidationError, TypeError) as e:
        logger.warning(f"Invalid request: {e}")
        return jsonify({
            "status": "error",
            "error": "invalid_request",
            "message": f"Invalid request format: {str(e)}"
        }), 400
    
    # Run recommendation pipeline
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state = loop.run_until_complete(run_recommendation(query))
        loop.close()
        
        logger.info(
            f"[API] Recommendation generated. Query: {query[:50]}... "
            f"Recommendations: {len(state.final_recommendations)}"
        )
        
        return jsonify({
            "status": "success",
            "recommendations": [
                {
                    "model_id": rec.model_id,
                    "score": rec.score,
                    "score_breakdown": rec.score_breakdown,
                    "metadata": rec.metadata
                }
                for rec in state.final_recommendations
            ],
            "metadata": {
                "num_candidates": len(state.scored_models),
                "execution_time": None,  # Could add timing
                "iteration": state.iteration,
                "scored_models_count": len(state.scored_models)
            },
            "state": _state_to_json(state)
        }), 200
    
    except Exception as e:
        logger.error(f"[API] Recommendation pipeline failed: {e}")
        return jsonify({
            "status": "error",
            "error": "pipeline_error",
            "message": str(e)
        }), 500


@api_bp.route("/recommend/refine", methods=["POST"])
def refine():
    """
    Refine previous recommendations with a follow-up query.
    
    Request:
        {
            "query": "make it faster",
            "state": {...}  # Previous state from /recommend response
        }
    
    Response:
        {
            "status": "success",
            "recommendations": [...],
            "metadata": {...},
            "state": {...}
        }
    
    Error Response:
        {
            "status": "error",
            "error": "error_code",
            "message": "error description"
        }
    """
    
    # Validate request
    try:
        payload = request.get_json()
        if not payload:
            raise ValueError("Request body is empty")
        
        req = RefineRequest(**payload)
        query = req.query.strip()
        prev_state = _dict_to_state(req.state)
    except (ValueError, ValidationError, TypeError) as e:
        logger.warning(f"Invalid refine request: {e}")
        return jsonify({
            "status": "error",
            "error": "invalid_request",
            "message": f"Invalid request format: {str(e)}"
        }), 400
    
    # Run refinement
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state = loop.run_until_complete(run_recommendation(query, state=prev_state))
        loop.close()
        
        logger.info(
            f"[API] Refinement completed. Query: {query[:50]}... "
            f"Iteration: {state.iteration}"
        )
        
        return jsonify({
            "status": "success",
            "recommendations": [
                {
                    "model_id": rec.model_id,
                    "score": rec.score,
                    "score_breakdown": rec.score_breakdown,
                    "metadata": rec.metadata
                }
                for rec in state.final_recommendations
            ],
            "metadata": {
                "num_candidates": len(state.scored_models),
                "iteration": state.iteration,
                "scored_models_count": len(state.scored_models)
            },
            "state": _state_to_json(state)
        }), 200
    
    except Exception as e:
        logger.error(f"[API] Refinement pipeline failed: {e}")
        return jsonify({
            "status": "error",
            "error": "pipeline_error",
            "message": str(e)
        }), 500


@api_bp.route("/recommend/debug", methods=["POST"])
def debug():
    """
    Return full internal state for debugging.
    
    Request:
        {
            "state": {...}
        }
    
    Response:
        {
            "status": "success",
            "debug": {
                "requirements": {...},
                "retrieved_models": [...],
                "scored_models": [...],
                "final_recommendations": [...],
                "agent_logs": [...],
                "state": {...}
            }
        }
    """
    
    try:
        payload = request.get_json()
        if not payload:
            raise ValueError("Request body is empty")
        
        req = DebugRequest(**payload)
        state = _dict_to_state(req.state)
    except (ValueError, ValidationError, TypeError) as e:
        logger.warning(f"Invalid debug request: {e}")
        return jsonify({
            "status": "error",
            "error": "invalid_request",
            "message": f"Invalid request format: {str(e)}"
        }), 400
    
    return jsonify({
        "status": "success",
        "debug": {
            "user_query": state.user_query,
            "iteration": state.iteration,
            "requirements": {
                "task_type": state.task_type,
                "constraints": state.constraints,
                "preferences": state.preferences,
                "extracted": state.requirements_extracted
            },
            "retrieval": {
                "retrieved_models_count": len(state.retrieved_models),
                "complete": state.retrieval_complete
            },
            "evaluation": {
                "scored_models_count": len(state.scored_models),
                "complete": state.evaluation_complete,
                "top_score": state.scored_models[0].score if state.scored_models else None
            },
            "final": {
                "recommendations_count": len(state.final_recommendations),
                "follow_up_questions": state.follow_up_questions,
                "explanations_count": len(state.explanations)
            },
            "agent_logs": state.agent_logs,
            "state": _state_to_json(state)
        }
    }), 200
