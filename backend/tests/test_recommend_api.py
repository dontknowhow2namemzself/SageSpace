"""Tests for the recommendations API (api/recommend.py) via TestClient.

recommend() itself is stubbed (no LLM / no network); we test the endpoints:
lazy compute vs cached, status transitions, 换一批 retire-then-recompute, 404s,
and the stats surface.
"""
import pytest
from fastapi.testclient import TestClient

import core.database as db_module
import api.recommend as rec_api


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    from main import app
    db_module.init_db()
    return TestClient(app)


def _seed(status="suggested", title="Seeded"):
    return db_module.insert_recommendation(
        title=title, author="A", blurb="b", reason="r",
        which_interest="X", status=status,
    )


def _fake_recompute(title="Fresh"):
    def _fake():
        rid = db_module.insert_recommendation(
            title=title, author="A", blurb="b", reason="r",
            which_interest="X", status="suggested",
        )
        return [db_module.get_recommendation(rid)]
    return _fake


def _no_recompute():
    raise AssertionError("recommend() should not be called here")


def test_get_returns_cached_suggested_without_recompute(client, monkeypatch):
    _seed(title="Already Here")
    monkeypatch.setattr(rec_api, "recommend", _no_recompute)
    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    assert [r["title"] for r in resp.json()] == ["Already Here"]


def test_get_lazy_computes_when_empty(client, monkeypatch):
    monkeypatch.setattr(rec_api, "recommend", _fake_recompute("Computed"))
    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["title"] for r in body] == ["Computed"]
    assert body[0]["status"] == "suggested"


def test_add_flips_status(client):
    rid = _seed()
    resp = client.post(f"/api/recommendations/{rid}/add")
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    assert db_module.get_recommendation(rid)["status"] == "added"


def test_dismiss_flips_status(client):
    rid = _seed()
    resp = client.post(f"/api/recommendations/{rid}/dismiss")
    assert resp.status_code == 200
    assert db_module.get_recommendation(rid)["status"] == "dismissed"


def test_add_unknown_id_is_404(client):
    assert client.post("/api/recommendations/ghost/add").status_code == 404


def test_dismiss_unknown_id_is_404(client):
    assert client.post("/api/recommendations/ghost/dismiss").status_code == 404


def test_refresh_retires_current_then_recomputes(client, monkeypatch):
    old = _seed(title="Old")
    monkeypatch.setattr(rec_api, "recommend", _fake_recompute("New"))
    resp = client.post("/api/recommendations/refresh")
    assert resp.status_code == 200
    assert [r["title"] for r in resp.json()] == ["New"]
    # The retired suggestion is preserved as 'seen' (eval denominator intact).
    assert db_module.get_recommendation(old)["status"] == "seen"


def test_stats_groups_by_status(client):
    _seed(status="added", title="a")
    _seed(status="added", title="b")
    _seed(status="dismissed", title="c")
    resp = client.get("/api/recommendations/stats")
    assert resp.status_code == 200
    assert resp.json() == {"added": 2, "dismissed": 1}


# ── Want-to-read list (saved / unsave) ───────────────────────────────────────


def test_saved_returns_added_only(client):
    _seed(status="added", title="Added One")
    _seed(status="suggested", title="Just Suggested")
    _seed(status="added", title="Added Two")
    resp = client.get("/api/recommendations/saved")
    assert resp.status_code == 200
    assert {r["title"] for r in resp.json()} == {"Added One", "Added Two"}


def test_unsave_moves_added_to_seen(client):
    rid = _seed(status="added", title="Saved")
    resp = client.post(f"/api/recommendations/{rid}/unsave")
    assert resp.status_code == 200
    assert resp.json()["status"] == "seen"
    assert db_module.get_recommendation(rid)["status"] == "seen"
    # ...and it drops off the saved list
    saved = client.get("/api/recommendations/saved").json()
    assert all(r["id"] != rid for r in saved)


def test_unsave_unknown_id_404(client):
    assert client.post("/api/recommendations/ghost/unsave").status_code == 404
