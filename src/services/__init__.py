"""
Services package.

Contains service layer implementations for business logic.
"""

from src.services.recommendation_pipeline import run_recommendation

__all__ = [
    "run_recommendation",
]
