import pytest
from fastapi.testclient import TestClient
import core.database as db_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    # Patch upload/export dirs to tmp
    import api.ingest as ingest_mod
    import api.export as export_mod
    monkeypatch.setattr(ingest_mod, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(export_mod, "EXPORT_DIR", tmp_path / "exports")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "exports").mkdir()

    from main import app
    db_module.init_db()
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_books_empty(client):
    resp = client.get("/api/books")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_book_not_found(client):
    resp = client.get("/api/books/nonexistent")
    assert resp.status_code == 404


def test_get_book_content_not_found(client):
    resp = client.get("/api/books/nonexistent/content")
    assert resp.status_code == 404


def test_delete_book_not_found(client):
    resp = client.delete("/api/books/nonexistent")
    assert resp.status_code == 404


def test_create_session_book_not_found(client):
    resp = client.post("/api/chat/session", params={"book_id": "nonexistent"})
    assert resp.status_code == 404


def test_progress_book_not_found(client):
    resp = client.get("/api/progress/nonexistent", params={"session_id": "x"})
    assert resp.status_code == 404


def test_books_crud_roundtrip(client):
    # Create via DB directly (no real file needed)
    book_id = db_module.create_book("Test Book", "Author", "/tmp/fake.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=50, total_chapters=3)

    resp = client.get("/api/books")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["title"] == "Test Book"

    resp = client.get(f"/api/books/{book_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == book_id

    resp = client.delete(f"/api/books/{book_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == book_id

    resp = client.get("/api/books")
    assert resp.json() == []


def test_get_book_content_returns_inline_file(client, tmp_path):
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4 test")

    book_id = db_module.create_book("PDF Book", "Author", str(sample))
    db_module.update_book_status(book_id, "ready", total_chunks=5, total_chapters=1)

    resp = client.get(f"/api/books/{book_id}/content")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")
    assert "inline" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF-1.4")
