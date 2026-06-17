"""Tests for the v2 ingest wiring (canonical → chunker → RAPTOR coverage).

These tests use the canonical_ingest path directly and stub out the
RAPTOR build (which calls OpenAI). The point is to verify that:

  * v1 books continue to flow through the legacy parser path untouched.
  * v2 books end up with sections + blocks in SQLite AND chunks with
    block_ids in chunk metadata AND raptor_node_blocks rows populated.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.database as database
from core.canonical import db as canonical_db
from core.canonical.chunker import chunk_blocks, decode_block_ids


@pytest.fixture
def temp_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(database, "DB_PATH", Path(tmp.name))
    database.init_db()
    yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


def _seed_book(book_id: str, file_path: str) -> None:
    from core.database import get_conn
    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, upload_date, raptor_status, file_path, ingest_version) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (book_id, "T", "2026-01-01T00:00:00Z", "ready", file_path),
    )
    conn.commit()
    conn.close()


def test_raptor_node_blocks_table_replace_and_read(temp_db):
    _seed_book("bk_x", "/tmp/x.epub")
    # canonical_db.replace_canonical_book requires real sections/blocks for FK,
    # but raptor_node_blocks rows don't need to satisfy FK on block_id since
    # SQLite only enforces FK when PRAGMA foreign_keys is on (default off).
    canonical_db.replace_raptor_node_blocks(
        "bk_x",
        [("rap_l1_c000", "blk_aaa"), ("rap_l1_c000", "blk_bbb"),
         ("rap_l1_c001", "blk_ccc")],
    )
    assert canonical_db.get_node_block_ids("bk_x", "rap_l1_c000") == ["blk_aaa", "blk_bbb"]
    assert canonical_db.get_node_block_ids("bk_x", "rap_l1_c001") == ["blk_ccc"]
    assert canonical_db.get_node_block_ids("bk_x", "missing") == []

    # Replace must wipe prior rows
    canonical_db.replace_raptor_node_blocks("bk_x", [("rap_l1_c000", "blk_aaa")])
    assert canonical_db.get_node_block_ids("bk_x", "rap_l1_c001") == []
    assert canonical_db.get_node_block_ids("bk_x", "rap_l1_c000") == ["blk_aaa"]

    # Duplicates in input are de-duplicated by PK
    canonical_db.replace_raptor_node_blocks(
        "bk_x",
        [("rap", "b1"), ("rap", "b1"), ("rap", "b2")],
    )
    assert canonical_db.get_node_block_ids("bk_x", "rap") == ["b1", "b2"]


def test_build_raptor_index_propagates_block_coverage(monkeypatch, temp_db):
    """Drive build_raptor_index end-to-end with mocked LLM/embeddings/Chroma,
    feeding it canonical chunks. Verify the register_block_links callback
    receives the union of children's block_ids at each summary level.
    """
    from core import raptor as raptor_mod
    from langchain_core.documents import Document

    # ── Mock OpenAI / Chroma surface so no network calls happen ──────────
    mock_emb = MagicMock()
    # Embeddings: produce deterministic small vectors so KMeans is stable
    def _embed_documents(texts):
        import numpy as np
        # Map each text to a fixed-ish 16-d vector by hashing
        out = []
        for t in texts:
            h = abs(hash(t)) % 1000
            v = np.array([(h >> b) & 1 for b in range(16)], dtype=float)
            out.append(v.tolist())
        return out
    mock_emb.embed_documents.side_effect = _embed_documents
    monkeypatch.setattr(raptor_mod, "_get_embeddings", lambda: mock_emb)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="summary text")
    monkeypatch.setattr(raptor_mod, "_get_llm", lambda: mock_llm)

    class _FakeVectorstore:
        def __init__(self, *a, **kw):
            self.added: list[Document] = []
        def add_documents(self, docs):
            self.added.extend(docs)
    monkeypatch.setattr(raptor_mod, "Chroma", _FakeVectorstore)

    # ── Build canonical chunks via the real chunker on synthetic blocks ──
    blocks = [
        {
            "block_id": f"blk_{i:03d}",
            "section_id": "s_A" if i < 10 else "s_B",
            "text": f"sentence number {i:03d} alpha beta gamma delta " * 4,
            "locator": {"page": 1},
            "locator_type": "pdf",
        }
        for i in range(20)
    ]
    chunks = chunk_blocks("bk_t", blocks, chunk_size=300, chunk_overlap=30)
    assert len(chunks) >= 6, "need enough chunks for RAPTOR clustering"

    # ── Run build_raptor_index, capturing block_link calls ───────────────
    received: list[tuple[str, set[str]]] = []
    def _register(node_id, covers):
        received.append((node_id, set(covers)))

    raptor_mod.build_raptor_index(
        chunks, book_id="bk_t", register_block_links=_register
    )

    # At least one summary level must have produced registrations
    assert received, "expected at least one summary node registration"
    # Every recorded coverage must be a subset of the union of source block_ids
    all_source_block_ids = {
        bid for ch in chunks for bid in decode_block_ids(ch.metadata["block_ids"])
    }
    for node_id, covered in received:
        assert covered, f"node {node_id} has empty coverage"
        assert covered <= all_source_block_ids, (
            f"node {node_id} covers unknown blocks: {covered - all_source_block_ids}"
        )


def test_build_raptor_index_legacy_path_emits_no_block_links(monkeypatch, temp_db):
    """v1 chunks have no block_ids metadata. The callback must never be
    invoked, so passing it in is harmless for legacy books.
    """
    from core import raptor as raptor_mod
    from langchain_core.documents import Document

    monkeypatch.setattr(raptor_mod, "_get_embeddings", lambda: MagicMock(
        embed_documents=MagicMock(return_value=[[0.0] * 8 for _ in range(20)])
    ))
    monkeypatch.setattr(raptor_mod, "_get_llm", lambda: MagicMock(
        invoke=MagicMock(return_value=MagicMock(content="s"))
    ))

    class _FakeVS:
        def __init__(self, *a, **kw): self.added = []
        def add_documents(self, docs): self.added.extend(docs)
    monkeypatch.setattr(raptor_mod, "Chroma", _FakeVS)

    # Legacy-style chunks: chapter/page only, no block_ids
    legacy_chunks = [
        Document(
            page_content=f"old content {i} " * 30,
            metadata={
                "chunk_id": f"chunk_{i:04d}",
                "chapter": (i // 5) + 1,
                "page": i + 1,
                "raptor_level": 0,
                "source": "legacy.pdf",
            },
        )
        for i in range(20)
    ]
    received: list = []
    raptor_mod.build_raptor_index(
        legacy_chunks, "bk_legacy", register_block_links=lambda *a: received.append(a)
    )
    assert received == [], "legacy v1 chunks must never trigger block-link callbacks"


# ── PR6: structural level-1 (one summary per body-matter section) ─────────


def _make_mocked_raptor(monkeypatch):
    """Patch raptor module's external dependencies (OpenAI + Chroma) so
    tests can call build_raptor_index without network. Returns the
    fake vectorstore singleton so tests can inspect what got added."""
    from core import raptor as raptor_mod
    from unittest.mock import MagicMock

    # Embeddings: produce one deterministic vector per input text so
    # downstream KMeans gets the array size it expects. Vary the vectors
    # slightly so KMeans does not collapse to one cluster.
    mock_emb = MagicMock()
    mock_emb.embed_documents.side_effect = lambda texts: [
        [float(i % 7), float(i % 5), float(i % 3), 1.0] for i in range(len(texts))
    ]
    monkeypatch.setattr(raptor_mod, "_get_embeddings", lambda: mock_emb)
    monkeypatch.setattr(raptor_mod, "_get_llm", lambda: MagicMock(
        invoke=MagicMock(return_value=MagicMock(content="section summary text"))
    ))

    added: list = []

    class _FakeVS:
        def __init__(self, *a, **kw):
            pass
        def add_documents(self, docs):
            added.extend(docs)

    monkeypatch.setattr(raptor_mod, "Chroma", _FakeVS)
    return raptor_mod, added


def test_pr6_level_1_is_one_node_per_body_matter_section(monkeypatch, temp_db):
    """When sections are passed in, level 1 must consist of EXACTLY one
    summary per chapter/prologue/epilogue/appendix section -- not a KMeans
    cluster grab-bag. Front-matter sections must produce NO level-1 node
    (the user does not ask "summarize the copyright page")."""
    raptor_mod, added = _make_mocked_raptor(monkeypatch)
    blocks = [
        {
            "block_id": f"blk_{i:03d}",
            # 2 front-matter blocks, then 3 sections of body content
            "section_id": (
                "s_cover" if i < 2
                else "s_ch1" if i < 8
                else "s_ch2" if i < 14
                else "s_ch3"
            ),
            "text": f"some content number {i} alpha beta gamma " * 4,
            "locator": {"page": 1},
            "locator_type": "pdf",
        }
        for i in range(20)
    ]
    chunks = chunk_blocks("bk_p6", blocks, chunk_size=300, chunk_overlap=30)
    sections = [
        {"section_id": "s_cover", "label": "Cover", "kind": "cover",
         "printed_number": None, "order_idx": 0},
        {"section_id": "s_ch1", "label": "CHAPTER I", "kind": "chapter",
         "printed_number": 1, "order_idx": 1},
        {"section_id": "s_ch2", "label": "CHAPTER II", "kind": "chapter",
         "printed_number": 2, "order_idx": 2},
        {"section_id": "s_ch3", "label": "CHAPTER III", "kind": "chapter",
         "printed_number": 3, "order_idx": 3},
    ]
    raptor_mod.build_raptor_index(chunks, "bk_p6", sections=sections)

    level_1 = [d for d in added if d.metadata.get("raptor_level") == 1]
    # 3 body-matter sections -> 3 level-1 nodes. Cover -> none.
    assert len(level_1) == 3, [d.metadata for d in level_1]
    section_ids = {d.metadata["section_id"] for d in level_1}
    assert section_ids == {"s_ch1", "s_ch2", "s_ch3"}
    # Cover must NOT have a level-1 node
    assert all(d.metadata["section_id"] != "s_cover" for d in level_1)


def test_pr6_level_1_node_ids_are_deterministic_from_section_id(monkeypatch, temp_db):
    """The level-1 node_id must be `raptor_l1_<section_id>` so
    get_chapter_summary can do a direct .get() with no similarity
    search. This is the PR6 fast path contract."""
    raptor_mod, added = _make_mocked_raptor(monkeypatch)
    blocks = [
        {
            "block_id": f"blk_{i:03d}", "section_id": "s_ch1",
            "text": f"text {i} " * 30, "locator": {"page": 1}, "locator_type": "pdf",
        }
        for i in range(6)
    ]
    chunks = chunk_blocks("bk_id", blocks, chunk_size=300, chunk_overlap=30)
    sections = [
        {"section_id": "s_ch1", "label": "Ch I", "kind": "chapter",
         "printed_number": 1, "order_idx": 0},
    ]
    raptor_mod.build_raptor_index(chunks, "bk_id", sections=sections)

    level_1 = [d for d in added if d.metadata.get("raptor_level") == 1]
    assert len(level_1) == 1
    assert level_1[0].metadata["chunk_id"] == "raptor_l1_s_ch1"


def test_pr6_per_section_register_block_links_uses_chunks_block_ids(monkeypatch, temp_db):
    """Each per-section level-1 node must register the union of its
    member chunks' block_ids. raptor_node_blocks needs this so summary
    hits can resolve to canonical blocks."""
    raptor_mod, _ = _make_mocked_raptor(monkeypatch)
    blocks = [
        {
            "block_id": f"blk_{i:03d}", "section_id": "s_ch1",
            "text": f"text {i} " * 30, "locator": {"page": 1}, "locator_type": "pdf",
        }
        for i in range(6)
    ]
    chunks = chunk_blocks("bk_cov", blocks, chunk_size=300, chunk_overlap=30)
    sections = [
        {"section_id": "s_ch1", "label": "Ch I", "kind": "chapter",
         "printed_number": 1, "order_idx": 0},
    ]

    received: list = []
    raptor_mod.build_raptor_index(
        chunks, "bk_cov", sections=sections,
        register_block_links=lambda node_id, covers: received.append((node_id, set(covers))),
    )

    # The level-1 node should appear in the registration list
    by_node = {nid: covers for nid, covers in received}
    assert "raptor_l1_s_ch1" in by_node
    # Its coverage must be subset of the source block_ids
    all_blocks = {b["block_id"] for b in blocks}
    assert by_node["raptor_l1_s_ch1"] <= all_blocks
    assert by_node["raptor_l1_s_ch1"], "coverage must not be empty"


def test_pr6_sections_with_no_chunks_are_skipped(monkeypatch, temp_db):
    """A body-matter section that the chunker never produced any chunks
    for (e.g. an empty appendix) must NOT trigger an LLM call OR a
    level-1 node. Otherwise we waste tokens summarizing nothing."""
    raptor_mod, added = _make_mocked_raptor(monkeypatch)
    blocks = [
        {
            "block_id": f"blk_{i:03d}", "section_id": "s_ch1",
            "text": f"text {i} " * 30, "locator": {"page": 1}, "locator_type": "pdf",
        }
        for i in range(6)
    ]
    chunks = chunk_blocks("bk_empty", blocks, chunk_size=300, chunk_overlap=30)
    sections = [
        {"section_id": "s_ch1", "label": "Ch I", "kind": "chapter",
         "printed_number": 1, "order_idx": 0},
        {"section_id": "s_empty_app", "label": "Appendix", "kind": "appendix",
         "printed_number": None, "order_idx": 1},
    ]
    raptor_mod.build_raptor_index(chunks, "bk_empty", sections=sections)

    level_1 = [d for d in added if d.metadata.get("raptor_level") == 1]
    assert len(level_1) == 1
    assert level_1[0].metadata["section_id"] == "s_ch1"


def test_pr6_no_sections_falls_back_to_legacy_kmeans(monkeypatch, temp_db):
    """When `sections` is None, build_raptor_index must use the legacy
    KMeans-at-every-level path. This preserves backward compatibility
    for callers (or older books) that have not adopted the structural
    level-1 contract."""
    raptor_mod, added = _make_mocked_raptor(monkeypatch)
    # Enough chunks to trigger the KMeans branch (> 4)
    blocks = [
        {
            "block_id": f"blk_{i:03d}", "section_id": "s_any",
            "text": f"alpha beta gamma {i} " * 8,
            "locator": {"page": 1}, "locator_type": "pdf",
        }
        for i in range(30)
    ]
    chunks = chunk_blocks("bk_legacy_path", blocks, chunk_size=300, chunk_overlap=30)
    raptor_mod.build_raptor_index(chunks, "bk_legacy_path", sections=None)

    level_1 = [d for d in added if d.metadata.get("raptor_level") == 1]
    # Legacy KMeans path produces nodes named raptor_l1_c000 / c001 / ...
    # (NOT raptor_l1_<section_id>). Some must exist (KMeans ran).
    assert level_1, "legacy path must still produce level-1 KMeans clusters"
    assert all(d.metadata["chunk_id"].startswith("raptor_l1_c") for d in level_1)
