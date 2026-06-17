"""Tests for core/cover.py — the OpenRouter Gemini cover generator.

The real API is mocked: every test injects a fake httpx.Client whose
.post() returns a canned response object. We cover:

  - happy path: image bytes are decoded and written to disk
  - missing API key: returns None, no file written
  - empty title: returns None, no file written
  - HTTP non-200: returns None, no file written
  - response shape with no images (Gemini text-only refusal): returns None
  - malformed image_url: returns None
  - delete_cover removes the file and returns True; returns False if absent
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from core import cover


# Minimal valid PNG (1x1 transparent). Just needs to round-trip through
# base64 decode; we don't validate the PNG itself.
_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
)
_PIXEL_B64 = base64.b64encode(_PIXEL_PNG).decode("ascii")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | str):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError("not json")


class _FakeClient:
    """Stand-in for httpx.Client. Captures the last request for assertions."""
    def __init__(self, response: _FakeResponse | Exception):
        self._response = response
        self.last_url: str | None = None
        self.last_body: dict | None = None
        self.last_headers: dict | None = None

    def post(self, url, json=None, headers=None):
        self.last_url = url
        self.last_body = json
        self.last_headers = headers
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    def close(self):
        pass


@pytest.fixture
def covers_dir(tmp_path, monkeypatch):
    """Redirect the module's COVERS_DIR into a tmp path so tests don't
    pollute the real uploads/covers/ folder."""
    target = tmp_path / "covers"
    monkeypatch.setattr(cover, "COVERS_DIR", target)
    return target


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """Default to a present key. Tests that need it missing will delenv."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _ok_response(b64: str = _PIXEL_B64) -> _FakeResponse:
    return _FakeResponse(200, {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "images": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }],
            }
        }],
        "usage": {"cost": 0.0388},
    })


# ── Happy path ─────────────────────────────────────────────────────────────

def test_generate_cover_writes_png(covers_dir):
    client = _FakeClient(_ok_response())
    path = cover.generate_cover("book_abc", "Some Book", client=client)
    assert path == covers_dir / "book_abc.png"
    assert path.read_bytes() == _PIXEL_PNG

    # Request shape is the contract we validated in the probes; lock it.
    body = client.last_body
    assert body["model"] == "google/gemini-2.5-flash-image"
    assert body["modalities"] == ["image", "text"]
    assert body["image_config"] == {"aspect_ratio": "2:3"}
    assert "Some Book" in body["messages"][0]["content"]


# ── Soft-failure paths ─────────────────────────────────────────────────────

def test_missing_api_key_returns_none(covers_dir, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = cover.generate_cover("book_abc", "Title", client=_FakeClient(_ok_response()))
    assert path is None
    assert not (covers_dir / "book_abc.png").exists()


def test_empty_title_returns_none(covers_dir):
    path = cover.generate_cover("book_abc", "   ", client=_FakeClient(_ok_response()))
    assert path is None
    assert not (covers_dir / "book_abc.png").exists()


def test_http_error_returns_none(covers_dir):
    client = _FakeClient(httpx.ConnectError("boom"))
    path = cover.generate_cover("book_abc", "Title", client=client)
    assert path is None
    assert not (covers_dir / "book_abc.png").exists()


def test_non_200_returns_none(covers_dir):
    client = _FakeClient(_FakeResponse(500, "internal error"))
    path = cover.generate_cover("book_abc", "Title", client=client)
    assert path is None
    assert not (covers_dir / "book_abc.png").exists()


def test_text_only_response_returns_none(covers_dir):
    """Gemini occasionally returns a text-only message with no images
    (e.g. content filter / model declined). Should soft-fail."""
    response = _FakeResponse(200, {
        "choices": [{
            "message": {"role": "assistant", "content": "Sorry, I can't.", "images": []}
        }],
    })
    path = cover.generate_cover("book_abc", "Title", client=_FakeClient(response))
    assert path is None
    assert not (covers_dir / "book_abc.png").exists()


def test_non_data_uri_returns_none(covers_dir):
    """If the image_url is an https URL instead of data:..., we skip
    rather than make a second fetch."""
    response = _FakeResponse(200, {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "images": [{"type": "image_url",
                            "image_url": {"url": "https://cdn.example/x.png"}}],
            }
        }],
    })
    path = cover.generate_cover("book_abc", "Title", client=_FakeClient(response))
    assert path is None


def test_malformed_response_returns_none(covers_dir):
    """Defensive: response missing the expected keys must not crash."""
    response = _FakeResponse(200, {"unexpected": "shape"})
    path = cover.generate_cover("book_abc", "Title", client=_FakeClient(response))
    assert path is None


# ── get_cover_path / delete_cover ──────────────────────────────────────────

def test_get_cover_path_returns_existing(covers_dir):
    covers_dir.mkdir()
    target = covers_dir / "book_abc.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert cover.get_cover_path("book_abc") == target


def test_get_cover_path_returns_none_when_missing(covers_dir):
    assert cover.get_cover_path("nope") is None


def test_delete_cover_removes_file(covers_dir):
    covers_dir.mkdir()
    target = covers_dir / "book_abc.png"
    target.write_bytes(b"x")
    assert cover.delete_cover("book_abc") is True
    assert not target.exists()


def test_delete_cover_returns_false_when_absent(covers_dir):
    assert cover.delete_cover("book_abc") is False
