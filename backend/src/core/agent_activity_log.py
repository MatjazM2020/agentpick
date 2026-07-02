"""
Minimal file log for agent tool calls and orchestrator loop activity.

Writes to ``logs/agentpick.log`` under the backend directory by default
(override with ``AGENT_ACTIVITY_LOG``). Intended for thesis demos — shows
that the tool loop runs and which tools were invoked.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER = logging.getLogger("agentpick.activity")
_CONFIGURED = False

_DEFAULT_LOG = Path(__file__).resolve().parents[2] / "logs" / "agentpick.log"


def configure_activity_log() -> Path:
    """Attach a file handler once; return the log file path."""
    global _CONFIGURED
    if _CONFIGURED:
        return _log_path()

    raw = (os.getenv("AGENT_ACTIVITY_LOG") or "").strip()
    log_file = Path(raw) if raw else _DEFAULT_LOG
    if not log_file.is_absolute():
        log_file = Path(__file__).resolve().parents[2] / log_file

    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    _CONFIGURED = True
    return log_file


def _log_path() -> Path:
    raw = (os.getenv("AGENT_ACTIVITY_LOG") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
    return _DEFAULT_LOG


def log_activity(message: str) -> None:
    configure_activity_log()
    _LOGGER.info(message)
