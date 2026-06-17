"""Structured-log + request-context plumbing.

Three surfaces under test:

  * `RequestIDMiddleware` — mints / echoes / propagates the request id;
  * `ContextVarFilter` + `JsonFormatter` — every log line is one JSON
    object carrying whatever contextvars are populated;
  * the chat handler sets session_id / book_id contextvars so logs
    emitted DEEP in LangGraph nodes are still attributable to the turn.
"""
from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.observability import (
    ContextVarFilter,
    JsonFormatter,
    RequestIDMiddleware,
    book_id_var,
    request_id_var,
    session_id_var,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _capture_one(record_factory) -> dict:
    """Format ONE log record through JsonFormatter (+ filter) and
    return it parsed."""
    fmt = JsonFormatter()
    filt = ContextVarFilter()
    record = record_factory()
    filt.filter(record)
    return json.loads(fmt.format(record))


def _make_record(msg: str = "hello", extra: dict | None = None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in (extra or {}).items():
        setattr(record, k, v)
    return record


# ── RequestIDMiddleware ──────────────────────────────────────────────────


@pytest.fixture
def app_with_middleware():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    captured: dict = {}

    @app.get("/ping")
    def ping():
        # Capture the contextvar inside the request scope.
        captured["request_id"] = request_id_var.get()
        return {"ok": True}

    return TestClient(app), captured


def test_middleware_mints_request_id_when_client_omits(app_with_middleware):
    client, captured = app_with_middleware
    res = client.get("/ping")
    assert res.status_code == 200
    rid_header = res.headers.get("X-Request-ID")
    assert rid_header and len(rid_header) == 32
    assert captured["request_id"] == rid_header


def test_middleware_honors_client_supplied_request_id(app_with_middleware):
    client, captured = app_with_middleware
    client_id = "client-supplied-12345"
    res = client.get("/ping", headers={"X-Request-ID": client_id})
    assert res.headers["X-Request-ID"] == client_id
    assert captured["request_id"] == client_id


def test_middleware_rejects_pathological_client_id(app_with_middleware):
    """Too-long / non-ASCII client ids are discarded and replaced.
    Cheap defense against an attacker stuffing junk into our logs."""
    client, _ = app_with_middleware
    bogus = "x" * 1000
    res = client.get("/ping", headers={"X-Request-ID": bogus})
    assert res.headers["X-Request-ID"] != bogus
    assert len(res.headers["X-Request-ID"]) == 32


def test_request_id_clears_after_response(app_with_middleware):
    """Outside any request the contextvar is back to None — no leakage
    between concurrent requests in the same event loop."""
    client, _ = app_with_middleware
    client.get("/ping")
    assert request_id_var.get() is None


# ── JsonFormatter ─────────────────────────────────────────────────────────


def test_formatter_emits_required_fields():
    payload = _capture_one(lambda: _make_record("startup ok"))
    assert payload["msg"] == "startup ok"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert "time" in payload


def test_formatter_omits_absent_context_fields():
    """When no contextvars are set, request_id / session_id / book_id
    are NOT in the JSON — they would be misleading nulls otherwise."""
    payload = _capture_one(lambda: _make_record())
    assert "request_id" not in payload
    assert "session_id" not in payload
    assert "book_id" not in payload


def test_formatter_includes_populated_context():
    request_id_var.set("rid-abc")
    session_id_var.set("sid-xyz")
    book_id_var.set("bid-123")
    try:
        payload = _capture_one(lambda: _make_record())
    finally:
        request_id_var.set(None)
        session_id_var.set(None)
        book_id_var.set(None)
    assert payload["request_id"] == "rid-abc"
    assert payload["session_id"] == "sid-xyz"
    assert payload["book_id"] == "bid-123"


def test_formatter_passes_through_extras():
    """Anything caller-supplied via `extra={...}` reaches the JSON
    verbatim (e.g. the request_start / request_end logger uses this
    for method / path / took_ms / status)."""
    record = _make_record("request_end")
    record.method = "POST"
    record.path = "/api/chat"
    record.took_ms = 12.3
    record.status = 200
    payload = _capture_one(lambda: record)
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/chat"
    assert payload["took_ms"] == 12.3
    assert payload["status"] == 200


def test_formatter_serializes_exception():
    try:
        raise ValueError("simulated outage")
    except ValueError:
        import sys
        record = _make_record("boom")
        record.exc_info = sys.exc_info()
    payload = _capture_one(lambda: record)
    assert "ValueError: simulated outage" in payload["exc"]


# ── ContextVarFilter is a process-wide filter, not record-specific ────────


def test_filter_returns_true_always():
    """A filter that returned False would swallow log records — the
    contract here is 'enrich, never drop'."""
    f = ContextVarFilter()
    assert f.filter(_make_record()) is True


# ── End-to-end: middleware + filter + formatter ───────────────────────────


def test_logs_in_handler_carry_request_id(monkeypatch):
    """A logger call inside a request handler emits JSON containing the
    request id minted by the middleware. End-to-end proof of the
    contextvar plumbing."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextVarFilter())
    test_logger = logging.getLogger("sagespace.test.e2e")
    test_logger.handlers = [handler]
    test_logger.propagate = False
    test_logger.setLevel(logging.INFO)

    @app.get("/work")
    def work():
        test_logger.info("inside_handler")
        return {"ok": True}

    client = TestClient(app)
    rid = "deterministic-request-id"
    client.get("/work", headers={"X-Request-ID": rid})

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    inside = next(p for p in lines if p["msg"] == "inside_handler")
    assert inside["request_id"] == rid
