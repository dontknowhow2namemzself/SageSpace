"""Tests for the memory-notes API (api/memory.py) via TestClient.

The "What I remember" panel: list, edit (text + type), delete, with 404/400s.
"""
import pytest
from fastapi.testclient import TestClient

import core.database as db_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    from main import app
    db_module.init_db()
    return TestClient(app)


def _seed(text="用户在研读斯多葛", type="interest"):
    return db_module.add_memory_note(text, type=type)


def test_list_memory_notes_empty(client):
    resp = client.get("/api/memory-notes")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_memory_notes(client):
    _seed("note A", type="fact")
    _seed("note B", type="interest")
    resp = client.get("/api/memory-notes")
    assert resp.status_code == 200
    assert {n["text"] for n in resp.json()} == {"note A", "note B"}


def test_edit_memory_note_text(client):
    nid = _seed("before")
    resp = client.patch(f"/api/memory-notes/{nid}", json={"text": "after"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "after"
    assert db_module.get_memory_note(nid)["text"] == "after"


def test_edit_memory_note_type(client):
    nid = _seed("x", type="fact")
    resp = client.patch(f"/api/memory-notes/{nid}", json={"text": "x", "type": "interest"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "interest"


def test_edit_missing_note_404(client):
    resp = client.patch("/api/memory-notes/ghost", json={"text": "x"})
    assert resp.status_code == 404


def test_edit_empty_text_400(client):
    nid = _seed("keep")
    resp = client.patch(f"/api/memory-notes/{nid}", json={"text": "   "})
    assert resp.status_code == 400
    assert db_module.get_memory_note(nid)["text"] == "keep"  # unchanged


def test_delete_memory_note(client):
    nid = _seed("bye")
    resp = client.delete(f"/api/memory-notes/{nid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == nid
    assert db_module.get_memory_note(nid) is None


def test_delete_missing_note_404(client):
    assert client.delete("/api/memory-notes/ghost").status_code == 404
