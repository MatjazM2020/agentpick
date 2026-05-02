#!/usr/bin/env python3
"""
Entry point for the recommendation API (FastAPI / OpenAI-compatible).

Sets Python path so ``app`` and ``src`` resolve, then starts uvicorn.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("ENV", "production") == "development"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )
