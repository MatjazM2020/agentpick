"""
Activity log — one line per agent request and tool call.

Lines go to a file (``AGENT_ACTIVITY_LOG`` env override, ``/app/logs`` in
Docker, ``backend/logs`` locally) and are mirrored to stdout through the
root logging configuration.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger("agentpick.activity")
_configured = False

# Per-request context — safe for concurrent asyncio tasks.
_CTX: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "_agentpick_ctx", default=None
)


@dataclass
class RequestContext:
    """Per-request state: id prefix for log lines, tool counter, elapsed time."""

    request_id: str
    tool_count: int = 0
    _start: float = field(default_factory=time.monotonic, repr=False)
    _token: object = field(default=None, repr=False)

    def attach(self) -> None:
        self._token = _CTX.set(self)

    def detach(self) -> None:
        if self._token is not None:
            try:
                _CTX.reset(self._token)
            except ValueError:
                # Token was created in a different asyncio context (e.g. parent task
                # attached, StreamingResponse body ran in a child task).
                pass
            finally:
                self._token = None

    def next_tool(self, name: str) -> str:
        self.tool_count += 1
        return f"tool#{self.tool_count} {name}"

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000


def current_context() -> Optional[RequestContext]:
    return _CTX.get(None)


def _log_file() -> Path:
    """Resolve log file path: env override → Docker mount → local backend/logs."""
    raw = (os.getenv("AGENT_ACTIVITY_LOG") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
    # Inside the Docker image WORKDIR is /app and compose mounts ./backend/logs → /app/logs
    if Path("/app/logs").is_dir():
        return Path("/app/logs/agentpick.log")
    return Path(__file__).resolve().parents[2] / "logs" / "agentpick.log"


def configure_activity_log() -> Path:
    """Attach the file handler (once) and return the resolved log file path."""
    global _configured
    path = _log_file()
    if _configured:
        return path
    _configured = True
    _LOGGER.setLevel(logging.INFO)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        _LOGGER.addHandler(handler)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "activity log file unavailable (%s): %s", path, exc
        )
    return path


def log_activity(message: str) -> None:
    """Write one line, prefixed with the current request id when inside a request."""
    configure_activity_log()
    ctx = current_context()
    prefix = f"[{ctx.request_id}] " if ctx else ""
    _LOGGER.info("%s%s", prefix, message)


def log_request_start(request_id: str, query: str, streaming: bool) -> None:
    log_activity(
        f"REQUEST START | id={request_id} | stream={streaming} | query={query[:120]!r}"
    )


def log_request_end(elapsed_ms: float, tool_count: int, status: str = "ok") -> None:
    log_activity(
        f"REQUEST END   | status={status} | tools_called={tool_count} | elapsed={elapsed_ms:.0f}ms"
    )


def log_tool_call(
    label: str,
    args: dict,
    result_count: Optional[int],
    elapsed_ms: float,
    error: Optional[str] = None,
) -> None:
    args_str = " | ".join(
        f"{k}={str(v)[:60]!r}" for k, v in args.items() if v is not None
    )
    if error:
        log_activity(f"{label} | {args_str} | ERROR={error[:80]} | {elapsed_ms:.0f}ms")
    else:
        count_str = f"results={result_count}" if result_count is not None else ""
        parts = [label, args_str, count_str, f"{elapsed_ms:.0f}ms"]
        log_activity(" | ".join(p for p in parts if p))
