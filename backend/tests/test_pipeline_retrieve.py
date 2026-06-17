"""Tests for the Retrieve pipeline node.

Covers the two entry points that produce a RetrievalResult:

  * run_retrieval         - search / book_overview path
  * chapter_summary_retrieval - chapter_summary path with the PR6 fast
                                lookup and the legacy similarity fallback

Both write retrieval_event + retrieval_event_chunks rows, so we run
against an isolated SQLite DB per test. The Chroma side is mocked
with a _FakeVS that supports .get(where=...) and .similarity_search.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

import core.database as db_module
from core.canonical import db as canonical_db
from core.canonical.ids import make_block_id, make_section_id
from core.canonical.models import Block, Section
from core.pipeline.retrieve import chapter_summary_retrieval, run_retrieval


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()


class _FakeVS:
    """Minimum Chroma surface for these tests."""
    def __init__(self, rows=None, hits=None):
        self._rows = rows or {}
        self._hits = hits or []
        self.last_similarity_filter = None
        self.last_similarity_query = None

    def get(self, where=None):
        where = where or {}
        if "chunk_id" in where:
            cid = where["chunk_id"]
            if cid in self._rows:
                r = self._rows[cid]
                return {"metadatas": [r["metadata"]], "documents": [r["document"]]}
            return {"metadatas": [], "documents": []}
        if where.get("raptor_level") == 0:
            metas = [r["metadata"] for r in self._rows.values()
                     if int(r["metadata"].get("raptor_level", 0)) == 0]
            return {"metadatas": metas}
        return {"metadatas": [], "documents": []}

    def similarity_search(self, query, k=4, filter=None):
        self.last_similarity_query = query
        self.last_similarity_filter = filter
        return list(self._hits)


def _make_book(book_id: str = "bk_t"):
    bid = db_module.create_book("Test Book", "Author", "/tmp/t.pdf")
    db_module.update_book_status(bid, "ready", total_chunks=10, total_chapters=2)
    sid = db_module.create_session(bid)
    return bid, sid


# ── run_retrieval ──────────────────────────────────────────────────────────


def test_run_retrieval_writes_retrieval_event_and_chunks():
    book_id, session_id = _make_book()
    doc = Document(
        page_content="The Cheshire Cat appeared on a branch.",
        metadata={
            "chunk_id": "chk_a", "section_id": "s_ch6", "section_label": "CHAPTER VI",
            "chapter": 6, "page": 65, "raptor_level": 0, "retrieval_origin": "hyde",
        },
    )
    vs = _FakeVS(rows={"chk_a": {
        "metadata": {"chunk_id": "chk_a", "raptor_level": 0},
        "document": doc.page_content,
    }})
    with patch("core.pipeline.retrieve.retrieve_combined", return_value=[doc]):
        result = run_retrieval(
            query="who is the cheshire cat",
            book_id=book_id, session_id=session_id, vectorstore=vs,
        )

    assert result.event_id, "must persist a retrieval_event row"
    events = db_module.get_retrieval_events(session_id)
    assert len(events) == 1
    chunks = db_module.get_event_chunks(events[0]["id"])
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chk_a"


def test_run_retrieval_emits_sse_payload_with_chapter_clusters_and_sources():
    book_id, session_id = _make_book()
    doc = Document(
        page_content="text",
        metadata={
            "chunk_id": "chk_b", "section_id": "s_ch2",
            "chapter": 2, "page": 1, "raptor_level": 0, "retrieval_origin": "multi_query",
        },
    )
    vs = _FakeVS(rows={"chk_b": {
        "metadata": {"chunk_id": "chk_b", "chapter": 2, "raptor_level": 0},
        "document": doc.page_content,
    }})
    with patch("core.pipeline.retrieve.retrieve_combined", return_value=[doc]):
        result = run_retrieval("q", book_id, session_id, vs)

    assert result.sse_payload, "must produce an SSE retrieval_update frame"
    payload = json.loads(result.sse_payload)
    assert payload["type"] == "retrieval_update"
    assert payload["newly_lit_count"] == 1
    assert "chk_b" in payload["newly_lit_chunk_ids"]
    assert payload["chapter_clusters"] == [{"chapter": 2, "total": 1, "lit": 1}]
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["chunk_id"] == "chk_b"


def test_run_retrieval_lights_only_synthesis_context_not_full_hitset():
    """Reading Map lit-set records only the docs that feed the synthesizer
    (top MAX_SYNTH_DOCS), not every chunk the agent surfaced -- else one
    over-fetching agentic question lights a huge slice of the book."""
    from core.pipeline.retrieve import MAX_SYNTH_DOCS

    book_id, session_id = _make_book()
    n = MAX_SYNTH_DOCS + 4
    docs = [
        Document(
            page_content=f"chunk {i}",
            metadata={"chunk_id": f"chk_{i:02d}", "section_id": "s1",
                      "chapter": 1, "page": i, "raptor_level": 0},
        )
        for i in range(n)
    ]
    rows = {
        f"chk_{i:02d}": {
            "metadata": {"chunk_id": f"chk_{i:02d}", "chapter": 1, "raptor_level": 0},
            "document": f"chunk {i}",
        }
        for i in range(n)
    }
    vs = _FakeVS(rows=rows)
    with patch("core.pipeline.retrieve.retrieve_combined", return_value=docs):
        run_retrieval("q", book_id, session_id, vs)

    lit = set(db_module.get_retrieved_chunk_ids(session_id))
    assert len(lit) == MAX_SYNTH_DOCS                              # capped to synth context
    assert lit == {f"chk_{i:02d}" for i in range(MAX_SYNTH_DOCS)}  # the TOP ones, in order
    assert f"chk_{MAX_SYNTH_DOCS:02d}" not in lit                  # over-fetched tail stays dark


def test_run_retrieval_sources_get_legacy_shape_when_citation_unresolvable():
    """A chunk with no block_ids in Chroma metadata cannot be resolved
    into a Citation payload -- the source ref degrades to the legacy
    {label, chapter, page, text, chunk_id} shape with NO citation_id /
    block_ids leaking through as null."""
    book_id, session_id = _make_book()
    doc = Document(
        page_content="orphan chunk text",
        metadata={
            "chunk_id": "chk_orphan", "chapter": 3, "page": 7,
            "raptor_level": 0, "retrieval_origin": "hyde",
        },
    )
    vs = _FakeVS(rows={"chk_orphan": {
        "metadata": {"chunk_id": "chk_orphan", "raptor_level": 0},
        "document": doc.page_content,
    }})
    with patch("core.pipeline.retrieve.retrieve_combined", return_value=[doc]):
        result = run_retrieval("q", book_id, session_id, vs)

    payload = json.loads(result.sse_payload)
    src = payload["sources"][0]
    assert src["label"] == "Chapter 3 · Page 7"
    assert "citation_id" not in src
    assert "block_ids" not in src


# ── chapter_summary_retrieval ─────────────────────────────────────────────


def _seed_v2_book_with_chapter(book_id: str, chapter_section_id: str, printed: int):
    """Insert a v2 book with one chapter section so chapter_summary
    can resolve it via (kind='chapter', printed_number=printed)."""
    sec = Section(
        section_id=chapter_section_id, book_id=book_id, order_idx=0,
        label=f"Chapter {printed}", level=1, source="outline",
        kind="chapter", printed_number=printed,
    )
    blocks = [
        Block(
            block_id=make_block_id(book_id, 0), book_id=book_id, order_idx=0,
            kind="paragraph", text="content",
            book_offset_start=0, book_offset_end=7,
            locator_type="pdf", locator={"page": 1},
            section_id=chapter_section_id,
        )
    ]
    canonical_db.replace_canonical_book(book_id, [sec], blocks, report={})


def test_chapter_summary_retrieval_uses_pr6_level_1_fast_path():
    """When the per-section level-1 node exists, chapter_summary_retrieval
    pulls it via .get with NO similarity_search."""
    book_id = db_module.create_book("Test", "A", "/tmp/x")
    db_module.update_book_status(book_id, "ready", total_chunks=5, total_chapters=1)
    session_id = db_module.create_session(book_id)
    section_id = make_section_id(book_id, 0)
    _seed_v2_book_with_chapter(book_id, section_id, printed=1)

    node_id = f"raptor_l1_{section_id}"
    vs = _FakeVS(rows={
        node_id: {
            "metadata": {
                "chunk_id": node_id, "raptor_level": 1,
                "section_id": section_id, "section_label": "Chapter 1",
            },
            "document": "Pre-built summary of Chapter 1.",
        },
    })

    result = chapter_summary_retrieval(
        book_id=book_id, printed_number=1,
        session_id=session_id, vectorstore=vs,
    )

    assert result.docs and "Pre-built summary" in result.docs[0]["text"]
    assert vs.last_similarity_query is None, "fast path must skip similarity_search"
    # And the retrieval_event was still written so Reading Map / Debug
    # timelines pick up the turn.
    events = db_module.get_retrieval_events(session_id)
    assert len(events) == 1
    assert "chapter_summary" in events[0]["query_text"]


def test_chapter_summary_retrieval_falls_back_to_similarity_when_no_level_1():
    """Pre-PR6 books have no per-section level-1 node. Fall through to
    a filtered level-0 similarity_search."""
    book_id = db_module.create_book("Test", "A", "/tmp/x")
    db_module.update_book_status(book_id, "ready", total_chunks=5, total_chapters=1)
    session_id = db_module.create_session(book_id)
    section_id = make_section_id(book_id, 0)
    _seed_v2_book_with_chapter(book_id, section_id, printed=1)

    # vectorstore HAS NO raptor_l1_<section_id> row.
    vs = _FakeVS(
        rows={},
        hits=[Document(
            page_content="legacy level-0 chunk text",
            metadata={"chunk_id": "chk_legacy", "raptor_level": 0,
                      "section_id": section_id, "page": 1, "chapter": 1},
        )],
    )

    result = chapter_summary_retrieval(
        book_id=book_id, printed_number=1,
        session_id=session_id, vectorstore=vs,
    )

    assert result.docs and "legacy" in result.docs[0]["text"]
    # Similarity search ran with the section_id filter
    assert vs.last_similarity_filter is not None
    flat = json.dumps(vs.last_similarity_filter)
    assert section_id in flat


def test_chapter_summary_retrieval_empty_when_no_matching_section():
    """If no section maps to (kind=chapter, printed_number=N) AND there
    is no Nth body-matter slot, return an empty RetrievalResult (no
    event written, no false 'all sections' fallback)."""
    book_id, session_id = _make_book()
    vs = _FakeVS()
    result = chapter_summary_retrieval(
        book_id=book_id, printed_number=99,
        session_id=session_id, vectorstore=vs,
    )
    assert result.docs == []
    assert result.event_id is None
    events = db_module.get_retrieval_events(session_id)
    assert events == []


# ── PR7: chapter_summary surfaces level-0 chunks alongside level-1 ────────


def test_chapter_summary_retrieval_returns_level_1_and_level_0_chunks():
    """Pre-PR7 the function returned ONLY the level-1 RAPTOR node, so
    every <fact> in the resulting answer had an empty data-chunk-ids
    attribute and citation chips all pointed to the single summary
    node. PR7 surfaces the section's level-0 chunks too so attribution
    has multiple block-level anchors."""
    book_id = db_module.create_book("T", "A", "/tmp/x")
    db_module.update_book_status(book_id, "ready", total_chunks=10, total_chapters=1)
    session_id = db_module.create_session(book_id)
    section_id = make_section_id(book_id, 0)
    _seed_v2_book_with_chapter(book_id, section_id, printed=1)

    node_id = f"raptor_l1_{section_id}"
    # Vectorstore: one level-1 summary node + four level-0 chunks in the
    # same section.
    rows = {
        node_id: {
            "metadata": {
                "chunk_id": node_id, "raptor_level": 1,
                "section_id": section_id, "section_label": "Chapter 1",
            },
            "document": "Pre-built summary text.",
        },
    }
    for i in range(4):
        cid = f"chk_{i:02d}"
        rows[cid] = {
            "metadata": {
                "chunk_id": cid, "raptor_level": 0,
                "section_id": section_id, "page": i + 1, "chapter": 1,
            },
            "document": f"level-0 chunk text {i}",
        }

    class _SectionAwareVS(_FakeVS):
        """Extends the base FakeVS .get to honor the level-0 + section_id
        AND filter that chapter_summary_retrieval issues."""
        def get(self, where=None):
            where = where or {}
            if "$and" in where:
                clauses = where["$and"]
                want_level = None
                want_section = None
                for c in clauses:
                    if "raptor_level" in c:
                        want_level = c["raptor_level"]["$eq"]
                    if "section_id" in c:
                        want_section = c["section_id"]["$eq"]
                metas, docs = [], []
                for r in self._rows.values():
                    md = r["metadata"]
                    if want_level is not None and md.get("raptor_level") != want_level:
                        continue
                    if want_section is not None and md.get("section_id") != want_section:
                        continue
                    metas.append(md)
                    docs.append(r["document"])
                return {"metadatas": metas, "documents": docs}
            return super().get(where=where)

    vs = _SectionAwareVS(rows=rows)
    result = chapter_summary_retrieval(
        book_id=book_id, printed_number=1,
        session_id=session_id, vectorstore=vs,
    )

    # Both the level-1 summary AND the level-0 chunks are in docs
    levels = [d["raptor_level"] for d in result.docs]
    assert 1 in levels, "level-1 summary must be present"
    assert 0 in levels, "level-0 chunks must be surfaced too"
    assert sum(1 for L in levels if L == 0) >= 1, "at least one level-0 chunk"
    # Level-1 leads so the synthesizer prompt sees it first.
    assert result.docs[0]["raptor_level"] == 1


# ── lit-set counts raw chunks only (digested-% denominator is level-0) ─────


def test_run_retrieval_lit_set_excludes_summary_nodes():
    """RAPTOR summary nodes among the synthesis context must NOT be
    recorded in retrieved_chunks: books.total_chunks counts only raw
    level-0 chunks, so recording raptor ids inflates digested_pct
    past 100% (the Alice 101.7% bug)."""
    book_id, session_id = _make_book()
    raw = Document(
        page_content="raw chunk",
        metadata={"chunk_id": "chk_a", "section_id": "s1",
                  "chapter": 1, "page": 1, "raptor_level": 0},
    )
    summary = Document(
        page_content="summary node",
        metadata={"chunk_id": "raptor_l1_s1", "section_id": "s1",
                  "section_label": "Chapter 1", "raptor_level": 1},
    )
    vs = _FakeVS(rows={"chk_a": {
        "metadata": {"chunk_id": "chk_a", "chapter": 1, "raptor_level": 0},
        "document": raw.page_content,
    }})
    with patch("core.pipeline.retrieve.retrieve_combined",
               return_value=[summary, raw]):
        run_retrieval("q", book_id, session_id, vs)

    lit = set(db_module.get_retrieved_chunk_ids(session_id))
    assert lit == {"chk_a"}, "only raw level-0 chunks may light the map"


def test_chapter_summary_retrieval_lit_set_excludes_summary_nodes():
    """Same invariant on the chapter_summary path: the level-1 node it
    surfaces must not enter retrieved_chunks."""
    book_id = db_module.create_book("Test", "A", "/tmp/x")
    db_module.update_book_status(book_id, "ready", total_chunks=5, total_chapters=1)
    session_id = db_module.create_session(book_id)
    section_id = make_section_id(book_id, 0)
    _seed_v2_book_with_chapter(book_id, section_id, printed=1)

    node_id = f"raptor_l1_{section_id}"
    vs = _FakeVS(rows={
        node_id: {
            "metadata": {
                "chunk_id": node_id, "raptor_level": 1,
                "section_id": section_id, "section_label": "Chapter 1",
            },
            "document": "Pre-built summary of Chapter 1.",
        },
    })
    result = chapter_summary_retrieval(
        book_id=book_id, printed_number=1,
        session_id=session_id, vectorstore=vs,
    )

    assert result.docs, "sanity: the summary doc is surfaced"
    lit = set(db_module.get_retrieved_chunk_ids(session_id))
    assert node_id not in lit
    assert all(cid.startswith("chk_") for cid in lit)
