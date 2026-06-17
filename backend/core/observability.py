"""Structured-log + request-context plumbing.

LangSmith (when enabled) covers the LLM-call angle: prompts, responses,
tool calls, token usage, span tree. That side is sovereign and rich.

This module covers the OTHER angle — HTTP-request lifecycle, caught
exceptions, deterministic-node decisions, system events — and ties
every log line to the request that produced it so the two views are
correlatable even when LangSmith is off:

  * `RequestIDMiddleware` mints a UUID per HTTP request (honors a
    client-supplied `X-Request-ID` header if present) and echoes it on
    the response. The id lives in a contextvar for the duration of the
    request, so it propagates across `await` boundaries automatically
    — including through every LangGraph node.
  * `session_id_var` / `book_id_var` are set by the chat handler the
    moment those values come into scope (two lines, no plumbing).
  * `ContextVarFilter` attaches whatever contextvars are populated to
    every `LogRecord` — existing `logger.info(...)` call sites need NO
    changes; they get request_id / session_id / book_id for free.
  * `JsonFormatter` emits one JSON object per line (stdlib only, no
    third-party log libraries). Default output is JSON to stdout; a
    human-readable mode is available for the rare case it's wanted.

Design choice: stdlib only. structlog and python-json-logger are
solid libraries, but they buy you very little in a single-process app
and cost you a dependency that has to be maintained alongside the
rest. Stdlib `logging` + a 30-line Formatter does the same job.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# ── Request-scoped context ────────────────────────────────────────────────


request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
book_id_var: ContextVar[str | None] = ContextVar("book_id", default=None)


# ── Middleware: assign + propagate + echo request_id ──────────────────────


_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Per-request UUID (or echoed client header), in a contextvar and
    on the response, plus a paired request_start / request_end log."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(_REQUEST_ID_HEADER)
        # Trust client-supplied IDs but require a sane shape — anything
        # longer or weirder we discard and mint our own. Cheap defense
        # against an attacker stuffing arbitrary strings into our logs.
        if incoming and 8 <= len(incoming) <= 64 and incoming.isascii():
            rid = incoming
        else:
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        logger = logging.getLogger("sagespace.http")
        start = time.perf_counter()
        logger.info(
            "request_start",
            extra={"method": request.method, "path": request.url.path},
        )
        status = 500
        try:
            response: Response = await call_next(request)
            # Echoing the id on the response is what makes user bug
            # reports actionable ("the failing request was abc123 →
            # grep the logs"). BaseHTTPMiddleware lets us mutate
            # headers on the response object before it is sent.
            response.headers[_REQUEST_ID_HEADER] = rid
            status = response.status_code
            return response
        finally:
            took_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "request_end",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "took_ms": took_ms,
                },
            )
            request_id_var.reset(token)


# ── Filter: pull contextvars onto every LogRecord ─────────────────────────


class ContextVarFilter(logging.Filter):
    """Attach request_id / session_id / book_id from contextvars to
    each LogRecord. A filter (not a Formatter) so that EVERY logger in
    the process — including third-party ones — gets the same fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        record.book_id = book_id_var.get()
        return True


# ── Formatters ────────────────────────────────────────────────────────────


# Default LogRecord attributes set by the logging library itself; the
# JSON formatter inspects `record.__dict__` for "extras" by skipping
# everything in this set. Keeps the JSON shape stable.
_LOG_RECORD_BUILTINS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime",
    "request_id", "session_id", "book_id",  # explicitly handled below
    "taskName",  # py3.12+
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record. Stable field order, ISO
    timestamps, exceptions serialized as a single string field."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")
                    + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Context fields — only emitted when populated.
        for k in ("request_id", "session_id", "book_id"):
            v = getattr(record, k, None)
            if v:
                payload[k] = v
        # Extras (anything the caller passed via `extra={...}` that
        # isn't a stdlib LogRecord attribute) — preserved verbatim.
        for k, v in record.__dict__.items():
            if k not in _LOG_RECORD_BUILTINS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Public entry: wire everything ────────────────────────────────────────


def configure_logging(level: str | int | None = None) -> None:
    """Replace the root handler with a single stdout handler emitting
    JSON, scoped to `level`. Idempotent — safe to call from app
    startup AND from worker subprocesses.

    The level defaults to LOG_LEVEL env var, else INFO. Pass an
    explicit level only from tests or from a config object.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    # Drop any pre-existing handlers so this function is idempotent
    # (uvicorn / pytest both pre-configure root, and we want our
    # formatter on top).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextVarFilter())
    root.addHandler(handler)
    root.setLevel(level)
