import pytest
from pathlib import Path
import core.database as db_module
from core.database import (
    init_db, create_book, get_book, list_books, delete_book,
    update_book_status, create_session, get_session,
    record_retrieved_chunks, get_retrieved_chunk_ids,
    get_all_retrieved_chunk_ids_for_book,
    create_retrieval_event, add_event_chunks,
    get_retrieval_events, get_event_chunks,
    add_memory_note,
)


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    init_db()


def test_create_and_get_book():
    book_id = create_book("Test Book", "Test Author", "/tmp/test.pdf")
    book = get_book(book_id)
    assert book["title"] == "Test Book"
    assert book["author"] == "Test Author"
    assert book["raptor_status"] == "pending"


def test_get_book_nonexistent():
    assert get_book("nonexistent-id") is None


def test_list_books():
    create_book("Book A", None, "/tmp/a.pdf")
    create_book("Book B", "Author B", "/tmp/b.pdf")
    books = list_books()
    assert len(books) == 2


def test_update_book_status():
    book_id = create_book("Status Book", None, "/tmp/s.pdf")
    update_book_status(book_id, "ready", total_chunks=150, total_chapters=10)
    book = get_book(book_id)
    assert book["raptor_status"] == "ready"
    assert book["total_chunks"] == 150
    assert book["total_chapters"] == 10


def test_delete_book():
    book_id = create_book("To Delete", None, "/tmp/del.pdf")
    delete_book(book_id)
    assert get_book(book_id) is None


def test_delete_book_cascades_all_referencing_tables():
    """delete_book must wipe every table that has a book_id FK. If you
    add a new table later, add a line to this test AND to delete_book().

    ONE deliberate exception: memory_notes.source_book_id is provenance for a
    user-level fact, so delete_book NULLs it instead of cascading (design §A) --
    asserted at the end (row survives, provenance cleared).
    """
    from core.canonical import db as canonical_db
    from core.canonical.ids import make_block_id, make_section_id
    from core.canonical.models import Block, Section

    book_id = create_book("Full Footprint", None, "/tmp/x.pdf")
    session_id = create_session(book_id)
    record_retrieved_chunks(session_id, book_id, ["chk_1", "chk_2"])
    db_module.record_cited_chunks(session_id, book_id, ["chk_1"])
    eid = create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="q",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
    )
    add_event_chunks(eid, [{
        "chunk_id": "chk_1", "raptor_level": 0, "chapter": 1, "page": 1,
        "rank": 1, "origin": "hyde", "is_new_lighting": 1, "preview_text": "x",
    }])
    # Canonical footprint
    sec = Section(section_id=make_section_id(book_id, 0), book_id=book_id,
                  order_idx=0, label="L", level=1, source="inferred")
    blk = Block(block_id=make_block_id(book_id, 0), book_id=book_id,
                order_idx=0, kind="paragraph", text="t",
                book_offset_start=0, book_offset_end=1,
                locator_type="pdf", locator={"page": 1},
                section_id=sec.section_id)
    canonical_db.replace_canonical_book(book_id, [sec], [blk], report={"ok": True})
    canonical_db.replace_raptor_node_blocks(book_id, [("rap_l1_c000", blk.block_id)])
    # Memory note tied to this book's session (the non-cascaded exception).
    add_memory_note("用户希望被称为小王", type="fact",
                    source_book_id=book_id, source_locator=session_id)

    # Sanity: every dependent table has a row for this book
    import sqlite3
    conn = sqlite3.connect(str(db_module.DB_PATH))
    def _count(table, where_col="book_id"):
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where_col}=?", (book_id,)
        ).fetchone()[0]
    assert _count("books", "id") == 1
    assert _count("sessions") == 1
    assert _count("retrieved_chunks") == 2
    assert _count("cited_chunks") == 1
    assert _count("retrieval_events") == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_event_chunks WHERE event_id=?", (eid,)
    ).fetchone()[0] == 1
    assert _count("sections") == 1
    assert _count("blocks") == 1
    assert _count("ingestion_reports") == 1
    assert _count("raptor_node_blocks") == 1
    assert _count("memory_notes", "source_book_id") == 1
    conn.close()

    delete_book(book_id)

    # Everything book-scoped is gone
    conn = sqlite3.connect(str(db_module.DB_PATH))
    assert _count("books", "id") == 0
    assert _count("sessions") == 0
    assert _count("retrieved_chunks") == 0
    assert _count("cited_chunks") == 0
    assert _count("retrieval_events") == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_event_chunks WHERE event_id=?", (eid,)
    ).fetchone()[0] == 0
    assert _count("sections") == 0
    assert _count("blocks") == 0
    assert _count("ingestion_reports") == 0
    assert _count("raptor_node_blocks") == 0
    # Deliberate exception: the note SURVIVES, its provenance is NULLed.
    assert conn.execute("SELECT COUNT(*) FROM memory_notes").fetchone()[0] == 1
    assert _count("memory_notes", "source_book_id") == 0
    conn.close()


def test_session_lifecycle():
    book_id = create_book("Session Book", None, "/tmp/sess.pdf")
    session_id = create_session(book_id)
    session = get_session(session_id)
    assert session["book_id"] == book_id
    assert session["total_tokens_in"] == 0


def test_progress_tracking_deduplicates():
    book_id = create_book("Progress Book", None, "/tmp/p.pdf")
    session_id = create_session(book_id)
    record_retrieved_chunks(session_id, book_id, ["chunk_0001", "chunk_0002"])
    record_retrieved_chunks(session_id, book_id, ["chunk_0002", "chunk_0003"])  # dup
    ids = get_retrieved_chunk_ids(session_id)
    assert set(ids) == {"chunk_0001", "chunk_0002", "chunk_0003"}


def test_all_retrieved_chunks_for_book():
    book_id = create_book("Multi-session", None, "/tmp/m.pdf")
    s1 = create_session(book_id)
    s2 = create_session(book_id)
    record_retrieved_chunks(s1, book_id, ["chunk_0001", "chunk_0002"])
    record_retrieved_chunks(s2, book_id, ["chunk_0003"])
    all_ids = get_all_retrieved_chunk_ids_for_book(book_id)
    assert set(all_ids) == {"chunk_0001", "chunk_0002", "chunk_0003"}


def test_create_retrieval_event():
    book_id = create_book("Event Book", None, "/tmp/e.pdf")
    session_id = create_session(book_id)
    event_id = create_retrieval_event(
        session_id=session_id,
        book_id=book_id,
        query_text="测试 query",
        multi_query_variants_json='["v1","v2","v3"]',
        hyde_hypothesis="假设原文",
        raw_hits_count=5,
        new_raw_hits_count=3,
        summary_hits_count=1,
    )
    assert isinstance(event_id, str) and len(event_id) > 0
    events = get_retrieval_events(session_id)
    assert len(events) == 1
    assert events[0]["query_text"] == "测试 query"
    assert events[0]["raw_hits_count"] == 5


def test_add_and_get_event_chunks():
    book_id = create_book("Chunk Book", None, "/tmp/c.pdf")
    session_id = create_session(book_id)
    event_id = create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="q",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=2, new_raw_hits_count=1, summary_hits_count=0,
    )
    add_event_chunks(event_id, [
        {"chunk_id": "chunk_0001", "raptor_level": 0, "chapter": 1, "page": 3,
         "rank": 1, "origin": "multi_query", "is_new_lighting": 1,
         "preview_text": "原文片段..."},
        {"chunk_id": "chunk_0002", "raptor_level": 0, "chapter": 1, "page": 5,
         "rank": 2, "origin": "hyde", "is_new_lighting": 0,
         "preview_text": "另一片段..."},
    ])
    chunks = get_event_chunks(event_id)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "chunk_0001"
    assert chunks[0]["is_new_lighting"] == 1


def test_get_retrieval_events():
    book_id = create_book("Events Book", None, "/tmp/ev.pdf")
    session_id = create_session(book_id)
    create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="first",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=3, new_raw_hits_count=2, summary_hits_count=0,
    )
    create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="second",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=4, new_raw_hits_count=1, summary_hits_count=1,
    )
    events = get_retrieval_events(session_id)
    assert len(events) == 2
    assert events[0]["query_text"] == "first"
