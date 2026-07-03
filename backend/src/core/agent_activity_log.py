"""
Activity log — structured log for every agent request and tool call.

Default file path:
  - Docker:  /app/logs/agentpick.log  (set explicitly via AGENT_ACTIVITY_LOG in compose)
  - Local:   backend/logs/agentpick.log

Each line is appended and flushed immediately so bind-mounted volumes update on the host
without waiting for process exit or handler buffers.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger("agentpick.activity")
_FILE_LOCK = threading.Lock()
_LOG_PATH: Optional[Path] = None

# Per-request context — safe for concurrent asyncio tasks.
_CTX: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "_agentpick_ctx", default=None
)


@dataclass
class RequestContext:
    """Lightweight per-request state shared by all log calls within one turn."""

    request_id: str
    query_snippet: str
    _tool_count: int = field(default=0, repr=False)
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
        self._tool_count += 1
        return f"tool#{self._tool_count} {name}"

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000


def current_context() -> Optional[RequestContext]:
    return _CTX.get(None)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _default_log_path() -> Path:
    """Resolve log file path: env override → Docker mount → local backend/logs."""
    raw = (os.getenv("AGENT_ACTIVITY_LOG") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
    # Inside the Docker image WORKDIR is /app and compose mounts ./backend/logs → /app/logs
    if Path("/app/logs").is_dir():
        return Path("/app/logs/agentpick.log")
    return Path(__file__).resolve().parents[2] / "logs" / "agentpick.log"


def log_path() -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = _default_log_path()
    return _LOG_PATH


# ---------------------------------------------------------------------------
# File write (append + flush — reliable on Docker bind mounts)
# ---------------------------------------------------------------------------

def _format_line(message: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return f"{ts} | {message}\n"


def _append_to_file(line: str) -> None:
    """Append one line and flush immediately. Never raises."""
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _FILE_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "activity log write failed (%s): %s", path, exc
        )


def configure_activity_log() -> Path:
    """
    Ensure log directory exists and return the resolved log file path.

    File writes go through ``_append_to_file`` (append + fsync). The logger
    mirrors to stdout only via propagation — no FileHandler, no duplicate lines.
    """
    path = log_path()
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not create activity log directory %s: %s", path.parent, exc
        )
    return path


# ---------------------------------------------------------------------------
# Public log helpers
# ---------------------------------------------------------------------------

def _prefix() -> str:
    ctx = current_context()
    return f"[{ctx.request_id}] " if ctx else ""


def log_activity(message: str) -> None:
    """Write one structured line to file (flushed) and stdout. Never raises."""
    full = f"{_prefix()}{message}"
    try:
        configure_activity_log()
        _append_to_file(_format_line(full))
        _LOGGER.info("%s", full)
    except Exception:
        pass


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


def log_llm_call(turn: int, input_tokens: Optional[int] = None) -> None:
    tok_str = f" | ~{input_tokens} input tokens" if input_tokens else ""
    log_activity(f"LLM CALL      | loop_turn={turn}{tok_str}")
