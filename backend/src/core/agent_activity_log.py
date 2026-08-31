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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

_LOGGER = logging.getLogger("agentpick.activity")
_configured = False

# Per-request context — safe for concurrent asyncio tasks.
_CTX: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "_agentpick_ctx", default=None
)
_DIALOGUE_CTX: contextvars.ContextVar[Optional["DialogueContext"]] = contextvars.ContextVar(
    "_agentpick_dialogue_ctx", default=None
)


@dataclass(frozen=True)
class DialogueContext:
    """Optional multi-turn dialogue metadata for eval/API requests."""

    dialogue_id: str
    user_turn: int
    user_turns_total: int
    question_id: str = ""


@dataclass
class RequestContext:
    """Per-request state: id prefix for log lines, tool counter, elapsed time."""

    request_id: str
    tool_count: int = 0
    llm_loop_turn: int = 0
    history_messages: int = 0
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

    def next_llm_loop_turn(self) -> int:
        self.llm_loop_turn += 1
        return self.llm_loop_turn

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000


def current_context() -> Optional[RequestContext]:
    return _CTX.get(None)


def current_dialogue() -> Optional[DialogueContext]:
    return _DIALOGUE_CTX.get(None)


@contextmanager
def dialogue_turn(
    dialogue_id: str,
    user_turn: int,
    user_turns_total: int,
    question_id: str = "",
) -> Iterator[None]:
    """Attach multi-turn dialogue metadata for the next agent request(s)."""
    ctx = DialogueContext(
        dialogue_id=dialogue_id,
        user_turn=user_turn,
        user_turns_total=user_turns_total,
        question_id=question_id,
    )
    token = _DIALOGUE_CTX.set(ctx)
    try:
        yield
    finally:
        _DIALOGUE_CTX.reset(token)


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


def log_request_start(
    request_id: str,
    query: str,
    streaming: bool,
    *,
    history_messages: int = 0,
    system: str = "agent",
) -> None:
    ctx = current_context()
    if ctx is not None:
        ctx.history_messages = history_messages
    parts = [
        "REQUEST START",
        f"id={request_id}",
        f"system={system}",
        f"stream={streaming}",
    ]
    dialogue = current_dialogue()
    if dialogue is not None:
        parts.append(f"dialogue={dialogue.dialogue_id}")
        parts.append(f"user_turn={dialogue.user_turn}/{dialogue.user_turns_total}")
        if dialogue.question_id:
            parts.append(f"q={dialogue.question_id}")
    if history_messages:
        parts.append(f"history_messages={history_messages}")
    parts.append(f"query={query[:120]!r}")
    log_activity(" | ".join(parts))


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


def log_llm_loop_turn(turn: int, input_tokens: Optional[int] = None, elapsed_ms: Optional[float] = None) -> None:
    """Log one LLM call inside the agent's tool-calling loop."""
    parts = [f"LLM CALL | agent_loop_turn={turn}"]
    if input_tokens:
        parts.append(f"~{input_tokens} input tokens")
    if elapsed_ms is not None:
        parts.append(f"{elapsed_ms:.0f}ms")
    log_activity(" | ".join(parts))
