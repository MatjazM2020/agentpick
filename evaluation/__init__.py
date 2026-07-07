"""Evaluation of the AgentPick recommendation system against a gold-standard dataset.

Lives outside ``backend/`` but evaluates its code (``src.agent``,
``src.catalog``), so importing this package puts the backend directory on
``sys.path``. Run from the repository root with the backend virtualenv:

    backend/.venv/bin/python -m evaluation.run --systems agent
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = str(_REPO_ROOT / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Load the repo-root .env (OPENAI_API_KEY, OPENAI_BASE_URL, etc.) the same way
# the backend app does on startup, so the evaluation "just works" without the
# caller having to export the variables manually.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")
