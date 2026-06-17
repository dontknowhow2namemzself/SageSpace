import pytest
from fastapi.testclient import TestClient
import core.database as db_module
from core import database as db
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


@pytest.fixture
def book_and_session():
    book_id = db.create_book("Debug Book", "Author", "/tmp/d.pdf")
    db.update_book_status(book_id, "ready", total_chunks=5, total_chapters=1)
    session_id = db.create_session(book_id)
    db.record_retrieved_chunks(session_id, book_id, ["chunk_0001", "chunk_0002"])
    event_id = db.create_retrieval_event(
        session_id=session_id, book_id=book_id,
        query_text="test query",
        multi_query_variants_json='["v1","v2"]',
        hyde_hypothesis="假设",
        raw_hits_count=2, new_raw_hits_count=2, summary_hits_count=0,
    )
    db.add_event_chunks(event_id, [
        {"chunk_id": "chunk_0001", "raptor_level": 0, "chapter": 1, "page": 1,
         "rank": 1, "origin": "multi_query", "is_new_lighting": 1,
         "preview_text": "片段1"},
        {"chunk_id": "chunk_0002", "raptor_level": 0, "chapter": 1, "page": 2,
         "rank": 2, "origin": "hyde", "is_new_lighting": 1,
         "preview_text": "片段2"},
    ])
    return book_id, session_id, event_id


def test_chunk_map_book_not_found(client):
    resp = client.get("/api/debug/books/nonexistent/chunk-map",
                      params={"session_id": "x"})
    assert resp.status_code == 404


def test_retrieval_events_empty(client, book_and_session):
    book_id, session_id, _ = book_and_session
    new_session = db.create_session(book_id)
    resp = client.get(f"/api/debug/sessions/{new_session}/retrieval-events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_retrieval_events_returns_events(client, book_and_session):
    _, session_id, _ = book_and_session
    resp = client.get(f"/api/debug/sessions/{session_id}/retrieval-events")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 1
    assert events[0]["query_text"] == "test query"


def test_retrieval_event_detail(client, book_and_session):
    _, _, event_id = book_and_session
    db.attach_event_answer_attribution(event_id, {
        "retrieval_event_ids": [event_id],
        "chunk_ids": ["chunk_0001", "chunk_0002"],
        "raptor_ids": ["raptor_l1_0001"],
        "facts": [
            {
                "fact_id": "f1",
                "text": "测试 fact",
                "chunk_ids": ["chunk_0001"],
                "retrieval_event_ids": [event_id],
            }
        ],
    })
    resp = client.get(f"/api/debug/retrieval-events/{event_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == event_id
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["origin"] == "multi_query"
    assert data["answer_attribution"]["facts"][0]["fact_id"] == "f1"
    assert data["answer_attribution"]["raptor_ids"] == ["raptor_l1_0001"]


# ── PR8: Reading Map groups by canonical section ─────────────────────────


def _seed_canonical_book_with_chunks(monkeypatch):
    """Build a v2 book + seed Chroma metadata so get_chunk_map produces
    sectioned groups. Uses the same _FakeVS pattern as other tests --
    chunk-map only needs vs.get(where={raptor_level: 0}), so we patch
    that minimally.

    Two sections (front-matter + chapter) each with two chunks. The
    user-reported bug was that the front-matter slot showed as
    "Chapter 3" -- here we assert the canonical section_label flows
    through correctly.
    """
    from core.canonical import db as canonical_db
    from core.canonical.ids import make_section_id, make_block_id
    from core.canonical.models import Section, Block

    book_id = db.create_book("Alice", "Carroll", "/tmp/a.epub")
    db.update_book_status(book_id, "ready", total_chunks=4, total_chapters=2)

    sec_cover = Section(
        section_id=make_section_id(book_id, 0), book_id=book_id,
        order_idx=0, label="Cover", level=1, source="toc",
        kind="cover", printed_number=None,
    )
    sec_ch1 = Section(
        section_id=make_section_id(book_id, 1), book_id=book_id,
        order_idx=1, label="CHAPTER I. Down the Rabbit-Hole",
        level=1, source="toc", kind="chapter", printed_number=1,
    )
    blocks = [
        Block(
            block_id=make_block_id(book_id, i), book_id=book_id, order_idx=i,
            kind="paragraph", text=f"text {i}",
            book_offset_start=i * 10, book_offset_end=i * 10 + 8,
            locator_type="epub", locator={"spine_idx": 0},
            section_id=(sec_cover.section_id if i < 2 else sec_ch1.section_id),
        )
        for i in range(4)
    ]
    canonical_db.replace_canonical_book(book_id, [sec_cover, sec_ch1], blocks, report={})

    # Patch get_vectorstore so the chunk-map endpoint reads our fixture.
    chunks_meta = [
        {"chunk_id": "chk_cover_a", "raptor_level": 0,
         "section_id": sec_cover.section_id, "page": 0, "chapter": 1},
        {"chunk_id": "chk_cover_b", "raptor_level": 0,
         "section_id": sec_cover.section_id, "page": 0, "chapter": 1},
        {"chunk_id": "chk_ch1_a", "raptor_level": 0,
         "section_id": sec_ch1.section_id, "page": 1, "chapter": 2},
        {"chunk_id": "chk_ch1_b", "raptor_level": 0,
         "section_id": sec_ch1.section_id, "page": 2, "chapter": 2},
    ]
    chunk_docs = ["cover text A", "cover text B", "ch1 text A", "ch1 text B"]

    class _FakeVS:
        def get(self, where=None):
            # chunk-map endpoint calls vs.get(where={"raptor_level": 0}).
            # The fixture only contains level-0 chunks so we ignore the
            # filter and return the full set.
            return {"metadatas": chunks_meta, "documents": chunk_docs}

    import api.debug as debug_mod
    monkeypatch.setattr(debug_mod, "get_vectorstore", lambda _bid: _FakeVS())
    return book_id, sec_cover, sec_ch1


def test_chunk_map_groups_carry_section_label_and_kind(client, monkeypatch):
    """User-reported PR8 bug: Reading Map showed 'Chapter 3' for a
    front-matter slot because grouping was done by section.order_idx + 1.
    After PR8, each group carries the canonical section_label / kind /
    printed_number from the canonical sections table."""
    book_id, sec_cover, sec_ch1 = _seed_canonical_book_with_chunks(monkeypatch)
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_chunks"] == 4

    groups = body["chapters"]
    # 2 sections with chunks -> 2 groups, sorted by order_idx
    assert len(groups) == 2
    assert groups[0]["section_id"] == sec_cover.section_id
    assert groups[0]["section_label"] == "Cover"
    assert groups[0]["kind"] == "cover"
    assert groups[0]["printed_number"] is None
    assert groups[1]["section_id"] == sec_ch1.section_id
    assert groups[1]["section_label"] == "CHAPTER I. Down the Rabbit-Hole"
    assert groups[1]["kind"] == "chapter"
    assert groups[1]["printed_number"] == 1


def test_chunk_map_groups_sorted_by_section_order(client, monkeypatch):
    """Groups must come back in canonical reading order so the
    Reading Map renders top-to-bottom matching the book."""
    book_id, sec_cover, sec_ch1 = _seed_canonical_book_with_chunks(monkeypatch)
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map")
    groups = resp.json()["chapters"]
    assert [g["order_idx"] for g in groups] == [0, 1]


def test_chunk_map_legacy_chapter_field_still_present(client, monkeypatch):
    """Pre-PR8 frontend builds read group.chapter as an int. We keep
    that field (== section.order_idx + 1) so older deploys do not
    break before they pull the typescript update."""
    book_id, _, _ = _seed_canonical_book_with_chunks(monkeypatch)
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map")
    groups = resp.json()["chapters"]
    assert groups[0]["chapter"] == 1   # order_idx=0 -> 0+1
    assert groups[1]["chapter"] == 2   # order_idx=1 -> 1+1


# ── Lighting: lit = cited by per-fact attribution ────────────────────────


def _flat_chunks(body: dict) -> dict[str, dict]:
    return {c["chunk_id"]: c for g in body["chapters"] for c in g["chunks"]}


def _make_event(session_id: str, book_id: str) -> str:
    return db.create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="q",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=2, new_raw_hits_count=2, summary_hits_count=0,
    )


def test_chunk_map_lights_only_fact_cited_chunks(client, monkeypatch):
    """The map's lit semantics: a chunk lights up when an answer's
    per-fact attribution cites it (= the user could READ it through
    the answer), not when retrieval merely fetched it. Here the turn
    retrieved chk_cover_a and chk_ch1_a, but the answer's only fact
    cites chk_ch1_a."""
    book_id, _, _ = _seed_canonical_book_with_chunks(monkeypatch)
    session_id = db.create_session(book_id)
    db.record_retrieved_chunks(session_id, book_id, ["chk_cover_a", "chk_ch1_a"])
    event_id = _make_event(session_id, book_id)
    db.attach_event_answer_attribution(event_id, {
        "retrieval_event_ids": [event_id],
        "chunk_ids": ["chk_cover_a", "chk_ch1_a"],  # turn-level union
        "raptor_ids": [],
        "facts": [
            {"fact_id": "f1", "text": "cited fact",
             "chunk_ids": ["chk_ch1_a"], "retrieval_event_ids": [event_id]},
        ],
    })
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map",
                      params={"session_id": session_id})
    chunks = _flat_chunks(resp.json())
    assert chunks["chk_ch1_a"]["is_lit"] is True
    assert chunks["chk_ch1_a"]["first_lit_event_id"] == event_id
    assert chunks["chk_cover_a"]["is_lit"] is False  # retrieved, never cited
    assert chunks["chk_cover_b"]["is_lit"] is False


def test_chunk_map_never_lights_from_raptor_citations(client, monkeypatch):
    """Facts may cite RAPTOR summary nodes (popup shows them, labeled),
    but the Reading Map lights BOOK TEXT only — a summary citation must
    not light anything, and must not flip the legacy fallback either."""
    book_id, _, _ = _seed_canonical_book_with_chunks(monkeypatch)
    session_id = db.create_session(book_id)
    event_id = _make_event(session_id, book_id)
    db.attach_event_answer_attribution(event_id, {
        "retrieval_event_ids": [event_id],
        "chunk_ids": ["chk_cover_a", "chk_ch1_a"],
        "raptor_ids": ["raptor_l1_sec_x"],
        "facts": [
            {"fact_id": "f1", "text": "summary-grounded fact",
             "chunk_ids": ["raptor_l1_sec_x"],
             "retrieval_event_ids": [event_id]},
            {"fact_id": "f2", "text": "book-grounded fact",
             "chunk_ids": ["chk_ch1_a"],
             "retrieval_event_ids": [event_id]},
        ],
    })
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map",
                      params={"session_id": session_id})
    chunks = _flat_chunks(resp.json())
    assert chunks["chk_ch1_a"]["is_lit"] is True       # raw citation lights
    assert chunks["chk_cover_a"]["is_lit"] is False    # union not used
    assert chunks["chk_cover_b"]["is_lit"] is False


def test_chunk_map_lighting_falls_back_to_turn_level_for_legacy_events(client, monkeypatch):
    """Events attached before per-fact attribution existed have no
    facts key. They fall back to the turn-level chunk_ids union so
    old sessions don't go dark."""
    book_id, _, _ = _seed_canonical_book_with_chunks(monkeypatch)
    session_id = db.create_session(book_id)
    event_id = _make_event(session_id, book_id)
    db.attach_event_answer_attribution(event_id, {
        "retrieval_event_ids": [event_id],
        "chunk_ids": ["chk_cover_b"],
        "raptor_ids": [],
    })
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map",
                      params={"session_id": session_id})
    chunks = _flat_chunks(resp.json())
    assert chunks["chk_cover_b"]["is_lit"] is True
    assert chunks["chk_cover_b"]["first_lit_event_id"] == event_id


def test_chunk_map_event_without_attribution_lights_nothing(client, monkeypatch):
    """A retrieval event whose turn never produced an attributed answer
    (e.g. abandoned at a clarify interrupt) must not light anything."""
    book_id, _, _ = _seed_canonical_book_with_chunks(monkeypatch)
    session_id = db.create_session(book_id)
    db.record_retrieved_chunks(session_id, book_id, ["chk_ch1_b"])
    _make_event(session_id, book_id)
    resp = client.get(f"/api/debug/books/{book_id}/chunk-map",
                      params={"session_id": session_id})
    chunks = _flat_chunks(resp.json())
    assert all(not c["is_lit"] for c in chunks.values())
