"""
Entry point for FastAPI backend.

Starts the uvicorn server for the OpenAI-compatible recommendation API.
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path (contains ``app`` and ``src``)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.main import app


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "production") == "development"
    )
