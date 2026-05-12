"""Utility functions for retries, logging, and helpers."""

import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def retry(fn: Callable, retries: int = 3, delay: float = 2) -> Any:
    """Retry wrapper for API calls with exponential backoff.
    
    Args:
        fn: Callable to retry
        retries: Number of retries
        delay: Initial delay in seconds (exponential backoff)
        
    Returns:
        Result of fn()
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            wait = delay * (2 ** i)
            logger.warning(f"Retry {i+1}/{retries} after {wait}s: {e}")
            time.sleep(wait)
