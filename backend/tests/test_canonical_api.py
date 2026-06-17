"""Tests for /api canonical browse + citation endpoints + the
resolve_citation helper itself.

Strategy:
  - Real SQLite (tempfile DB) seeded with one v2 book and one v1 book.
  - Chroma is stubbed via core.raptor.get_vectorstore so no embeddings or
    persistence is required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import core.database as database
from core.canonical import db as canonical_db
from core.canonical.citations import resolve_citation
from core.canonical.ids import make_block_id, make_section_id
from core.canonical.models import Block, Section


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(database, "DB_PATH", Path(tmp.name))
    database.init_db()
    yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


def _make_v2_book() -> str:
    book_id = "bk_v2"
    from core.database import get_conn
    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, upload_date, raptor_status, file_path, ingest_version) "
        "VALUES (?, 'V2 Book', '2026-01-01T00:00:00Z', 'ready', '/tmp/x.pdf', 1)",
        (book_id,),
    )
    conn.commit()
    conn.close()

    sec0 = Section(
        section_id=make_section_id(book_id, 0), book_id=book_id, order_idx=0,
        label="Chapter 1", level=1, source="outline",
    )
    sec1 = Section(
        section_id=make_section_id(book_id, 1), book_id=book_id, order_idx=1,
        label="Chapter 2", level=1, source="outline",
    )
    blocks = [
        Block(
            block_id=make_block_id(book_id, i),
            book_id=book_id, order_idx=i, kind="paragraph",
            text=f"Paragraph {i} content here, sentence body.",
            book_offset_start=i * 50, book_offset_end=i * 50 + 40,
            locator_type="pdf",
            locator={"page": i // 2 + 1, "bbox": [10, 20, 100, 30]},
            section_id=(sec0.section_id if i < 3 else sec1.section_id),
        )
        for i in range(5)
    ]
    canonical_db.replace_canonical_book(book_id, [sec0, sec1], blocks, report={"final_blocks": 5})
    # Pre-seed a raptor_node_blocks row for the RAPTOR resolution test
    canonical_db.replace_raptor_node_blocks(book_id, [
        ("raptor_l1_c000", blocks[0].block_id),
        ("raptor_l1_c000", blocks[1].block_id),
    ])
    return book_id


def _make_v1_book() -> str:
    book_id = "bk_v1"
    from core.database import get_conn
    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, upload_date, raptor_status, file_path, ingest_version) "
        "VALUES (?, 'V1 Book', '2026-01-01T00:00:00Z', 'ready', '/tmp/y.pdf', 1)",
        (book_id,),
    )
    conn.commit()
    conn.close()
    return book_id


class _FakeVS:
    """Minimal stand-in for langchain_chroma.Chroma exposing .get(where=...)
    over a static metadata table. Indexed by chunk_id.
    """
    def __init__(self, rows: dict[str, dict]):
        # rows: chunk_id -> {"metadata": {...}, "document": "..."}
        self._rows = rows

    def get(self, where=None):
        if not where:
            return {"metadatas": [], "documents": []}
        cid = where.get("chunk_id")
        if cid in self._rows:
            r = self._rows[cid]
            return {"metadatas": [r["metadata"]], "documents": [r["document"]]}
        return {"metadatas": [], "documents": []}


@pytest.fixture
def client(temp_db, monkeypatch):
    # We deliberately do NOT import backend/main.py here: it triggers the
    # known evaluate.py SyntaxError (docs/ARCHITECTURE.md §6 P0, today
    # untouchable). Compose a minimal FastAPI app with only the canonical
    # router so this test stays scoped to what this commit owns.
    from fastapi import FastAPI
    from api import canonical as canonical_api

    app = FastAPI()
    app.include_router(canonical_api.router, prefix="/api")
    return TestClient(app)


# ── Browse API ──────────────────────────────────────────────────────────────


def test_sections_endpoint_returns_canonical_rows(client):
    v2 = _make_v2_book()
    r = client.get(f"/api/books/{v2}/sections")
    assert r.status_code == 200
    body = r.json()
    assert [s["label"] for s in body["sections"]] == ["Chapter 1", "Chapter 2"]
    assert all(s["source"] == "outline" for s in body["sections"])


def test_sections_endpoint_409_for_v1_book(client):
    v1 = _make_v1_book()
    r = client.get(f"/api/books/{v1}/sections")
    assert r.status_code == 409
    assert "legacy ingest_version=1" in r.json()["detail"]


def test_sections_endpoint_404_for_unknown_book(client):
    r = client.get("/api/books/nope/sections")
    assert r.status_code == 404


def test_blocks_endpoint_basic_paging_and_filters(client):
    v2 = _make_v2_book()
    # No filters: all 5 blocks
    r = client.get(f"/api/books/{v2}/blocks")
    assert r.status_code == 200
    body = r.json()
    assert [b["order_idx"] for b in body["blocks"]] == [0, 1, 2, 3, 4]
    assert body["next_cursor"] is None  # length < limit

    # Pagination cursor
    r = client.get(f"/api/books/{v2}/blocks?limit=3")
    body = r.json()
    assert [b["order_idx"] for b in body["blocks"]] == [0, 1, 2]
    assert body["next_cursor"] == 2
    r2 = client.get(f"/api/books/{v2}/blocks?limit=3&after=2")
    body2 = r2.json()
    assert [b["order_idx"] for b in body2["blocks"]] == [3, 4]

    # Filter by section_id
    sec1_id = next(s["section_id"] for s in
                   client.get(f"/api/books/{v2}/sections").json()["sections"]
                   if s["label"] == "Chapter 2")
    r = client.get(f"/api/books/{v2}/blocks?section_id={sec1_id}")
    assert [b["order_idx"] for b in r.json()["blocks"]] == [3, 4]


def test_single_block_endpoint(client):
    v2 = _make_v2_book()
    bid = client.get(f"/api/books/{v2}/blocks").json()["blocks"][0]["block_id"]
    r = client.get(f"/api/books/{v2}/blocks/{bid}")
    assert r.status_code == 200
    assert r.json()["block_id"] == bid
    # Unknown block_id
    r404 = client.get(f"/api/books/{v2}/blocks/blk_doesnotexist")
    assert r404.status_code == 404


def test_ingestion_report_endpoint(client):
    v2 = _make_v2_book()
    r = client.get(f"/api/books/{v2}/ingestion-report")
    assert r.status_code == 200
    assert r.json()["report"]["final_blocks"] == 5


# ── Citation resolution (unit) ──────────────────────────────────────────────


def test_resolve_raw_chunk_citation(temp_db):
    v2 = _make_v2_book()
    blocks = canonical_db.get_blocks(v2)
    bids = [b["block_id"] for b in blocks[:3]]  # first three are in Chapter 1
    primary = bids[1]
    vs = _FakeVS({
        "chk_xyz": {
            "metadata": {
                "chunk_id": "chk_xyz",
                "raptor_level": 0,
                "block_ids": ",".join(bids),
                "primary_block_id": primary,
            },
            "document": "the chunk text that was retrieved",
        }
    })
    cit = resolve_citation(v2, "chk_xyz", vs)
    assert cit is not None
    assert cit["anchor"]["primary_block_id"] == primary
    assert cit["anchor"]["block_ids"] == bids
    assert cit["section_label"] == "Chapter 1"
    assert cit["evidence"]["retrieved_from"]["layer"] == "raw"
    assert cit["evidence"]["snippet"].startswith("the chunk text")
    # Full evidence text (what the minimal popup renders) is the chunk's
    # own text for raw hits — untruncated.
    assert cit["evidence"]["text"] == "the chunk text that was retrieved"
    assert cit["source_locator"]["page"] >= 1


def test_resolve_raptor_node_citation_uses_reverse_index(temp_db):
    """A RAPTOR summary node has no block_ids in Chroma metadata. The
    resolver must fall back to raptor_node_blocks to find coverage. The
    first id (lexical order from the reverse index) becomes the primary
    jump target.
    """
    v2 = _make_v2_book()
    blocks = canonical_db.get_blocks(v2)
    expected_set = {blocks[0]["block_id"], blocks[1]["block_id"]}

    vs = _FakeVS({
        "raptor_l1_c000": {
            "metadata": {
                "chunk_id": "raptor_l1_c000",
                "raptor_level": 1,
                # NOTE: no block_ids field. Reverse index must kick in.
            },
            "document": "summary text",
        }
    })
    cit = resolve_citation(v2, "raptor_l1_c000", vs)
    assert cit is not None
    assert set(cit["anchor"]["block_ids"]) == expected_set
    # primary must be one of the covered blocks (not asserting which - the
    # reverse index returns lexical order, which is fine for our jump UX)
    assert cit["anchor"]["primary_block_id"] in expected_set
    assert cit["evidence"]["retrieved_from"]["layer"] == "raptor"
    assert cit["evidence"]["retrieved_from"]["raptor_level"] == 1
    # Snippet + full text for RAPTOR hits are the node's OWN summary
    # text (2026-06-10): summaries are first-class citation targets and
    # the popup labels them as AI-generated, so showing the summary the
    # fact was grounded in is the honest behavior.
    assert cit["evidence"]["snippet"] == "summary text"
    assert cit["evidence"]["text"] == "summary text"


def test_resolve_returns_none_for_unknown_chunk(temp_db):
    v2 = _make_v2_book()
    vs = _FakeVS({})
    assert resolve_citation(v2, "missing_id", vs) is None
    assert resolve_citation(v2, "", vs) is None


def test_resolve_returns_none_for_legacy_chunk_without_block_ids(temp_db):
    v2 = _make_v2_book()
    vs = _FakeVS({
        "legacy_chunk_42": {
            "metadata": {
                "chunk_id": "legacy_chunk_42",
                "raptor_level": 0,
                "chapter": 3,
                "page": 12,
                # No block_ids, no primary_block_id - looks like v1 chunk
            },
            "document": "...",
        }
    })
    assert resolve_citation(v2, "legacy_chunk_42", vs) is None


# ── Citation endpoint plumbing ──────────────────────────────────────────────


def test_citations_endpoint_404_for_unresolvable(client, monkeypatch):
    v2 = _make_v2_book()
    # Patch the vectorstore factory used by the API to a deterministic stub.
    import api.canonical as api_canonical
    monkeypatch.setattr(
        api_canonical, "get_vectorstore",
        lambda *_args, **_kw: _FakeVS({}),
        raising=False,
    )
    # Need to also patch on core.raptor since api/canonical imports it inside the handler
    import core.raptor as raptor_mod
    monkeypatch.setattr(raptor_mod, "get_vectorstore", lambda *_a, **_kw: _FakeVS({}))

    r = client.get(f"/api/books/{v2}/citations/whatever")
    assert r.status_code == 404


def test_citations_endpoint_200_path(client, monkeypatch):
    v2 = _make_v2_book()
    blocks = canonical_db.get_blocks(v2)
    bids = [b["block_id"] for b in blocks[:2]]
    primary = bids[0]
    fake_vs = _FakeVS({
        "chk_real": {
            "metadata": {
                "chunk_id": "chk_real",
                "raptor_level": 0,
                "block_ids": ",".join(bids),
                "primary_block_id": primary,
            },
            "document": "evidence text here",
        }
    })
    import core.raptor as raptor_mod
    monkeypatch.setattr(raptor_mod, "get_vectorstore", lambda *_a, **_kw: fake_vs)

    r = client.get(f"/api/books/{v2}/citations/chk_real")
    assert r.status_code == 200
    body = r.json()
    assert body["anchor"]["primary_block_id"] == primary
    assert body["anchor"]["block_ids"] == bids
    assert body["section_label"] == "Chapter 1"
