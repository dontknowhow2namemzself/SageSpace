"""Read-only data access for the SageSpace inspector.

Talks straight to SQLite + ChromaDB on disk. No HTTP, no embeddings, no
writes. SQLite is opened with `mode=ro`; ChromaDB has no read-only mode,
so we simply never call any write methods on it.

╔══════════════════════════════════════════════════════════════════════╗
║  SCHEMA MIRROR — keep in sync with backend                           ║
║                                                                      ║
║  Column / metadata names below are duplicated from:                  ║
║    backend/core/database.py          (SQLite table columns)          ║
║    backend/core/canonical/db.py      (canonical read APIs)           ║
║    backend/core/canonical/chunker.py (Chroma level-0 chunk metadata) ║
║    backend/core/raptor.py            (Chroma level-1+ node metadata) ║
║                                                                      ║
║  Renaming any of those WITHOUT also editing this file produces       ║
║  silent empty results, not loud errors — this viewer does NOT        ║
║  import backend code, it hardcodes the schema. Edit it in the same   ║
║  commit as any backend schema rename.                                ║
╚══════════════════════════════════════════════════════════════════════╝

Resolution order for the data files:
  1. Env vars `SAGESPACE_DB_PATH` / `SAGESPACE_CHROMA_PATH` if set.
  2. The sibling `../backend/{sagespace.db,chroma_db}` (the inspect tool
     lives next to the backend it reads).
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import chromadb
import streamlit as st


_INSPECT_DIR = Path(__file__).resolve().parent
_CANDIDATE_BACKEND_DIRS = [
    _INSPECT_DIR.parent / "backend",
]


def _resolve(env_key: str, filename: str) -> Path:
    override = os.environ.get(env_key)
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{env_key}={override!r} does not exist")
        return p
    for d in _CANDIDATE_BACKEND_DIRS:
        candidate = d / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Set {env_key} or place the file under "
        + " or ".join(str(d) for d in _CANDIDATE_BACKEND_DIRS)
    )


def sqlite_path() -> Path:
    return _resolve("SAGESPACE_DB_PATH", "sagespace.db")


def chroma_dir() -> Path:
    return _resolve("SAGESPACE_CHROMA_PATH", "chroma_db")


def _connect_ro() -> sqlite3.Connection:
    uri = f"file:{sqlite_path()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def _chroma_client(path_str: str) -> chromadb.api.client.Client:
    return chromadb.PersistentClient(path=path_str)


def _chroma_collection(book_id: str):
    client = _chroma_client(str(chroma_dir()))
    return client.get_collection(f"book_{book_id}")


# ── Books / sections / blocks ───────────────────────────────────────────────


def _section_by_id(book_id: str, section_id: str) -> dict | None:
    with _connect_ro() as conn:
        row = conn.execute(
            "SELECT section_id, order_idx, kind, printed_number, label, "
            "parent_section_id, level FROM sections "
            "WHERE book_id=? AND section_id=?",
            (book_id, section_id),
        ).fetchone()
    return dict(row) if row else None


@st.cache_data(show_spinner=False)
def list_books() -> list[dict]:
    with _connect_ro() as conn:
        rows = conn.execute(
            "SELECT id, title, author, total_chunks, total_chapters, "
            "raptor_status FROM books ORDER BY upload_date DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(show_spinner=False)
def load_sections(book_id: str) -> list[dict]:
    with _connect_ro() as conn:
        rows = conn.execute(
            "SELECT section_id, order_idx, kind, printed_number, label, "
            "parent_section_id, level FROM sections WHERE book_id=? "
            "ORDER BY order_idx",
            (book_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(show_spinner=False)
def load_blocks(book_id: str, section_id: str) -> list[dict]:
    with _connect_ro() as conn:
        rows = conn.execute(
            "SELECT block_id, order_idx, kind, text, locator_json, "
            "book_offset_start, book_offset_end FROM blocks "
            "WHERE book_id=? AND section_id=? ORDER BY order_idx",
            (book_id, section_id),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["locator"] = json.loads(d.pop("locator_json"))
        except Exception:
            d["locator"] = {}
        out.append(d)
    return out


# ── Chunks (Chroma, level-0 only) ───────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _all_l0_chunks(book_id: str) -> list[dict]:
    """Every level-0 chunk for `book_id` as plain dicts. Cached per book.

    Filtering by section_id happens client-side in `load_chunks` so the
    per-book Chroma round-trip is only paid once per session.
    """
    try:
        col = _chroma_collection(book_id)
    except Exception:
        return []
    data = col.get(where={"raptor_level": 0})
    out: list[dict] = []
    for meta, doc in zip(data.get("metadatas") or [], data.get("documents") or []):
        raw_block_ids = meta.get("block_ids") or ""
        block_ids = [b for b in raw_block_ids.split(",") if b]
        out.append({
            "chunk_id": meta.get("chunk_id", ""),
            "primary_block_id": meta.get("primary_block_id") or "",
            "block_ids": block_ids,
            "section_id": meta.get("section_id") or "",
            "page": meta.get("page", 0),
            "char_length": len(doc or ""),
            "text": doc or "",
        })
    return out


def load_chunks(book_id: str, section_id: str) -> list[dict]:
    """Chunks for `section_id`, in reading order.

    Reading order = ascending `order_idx` of the chunk's first block.
    `chunk_id` is a sha1 hash of the generation index, so sorting on it
    would be effectively random — see chunker._chunk_id.
    """
    chunks = [c for c in _all_l0_chunks(book_id) if c["section_id"] == section_id]
    order_by_block: dict[str, int] = {
        b["block_id"]: b["order_idx"] for b in load_blocks(book_id, section_id)
    }
    LAST = 10**9

    def sort_key(c: dict) -> int:
        first = (c["block_ids"] or [""])[0]
        return order_by_block.get(first, LAST)

    chunks.sort(key=sort_key)
    return chunks


# ── RAPTOR tree ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _summary_nodes(book_id: str) -> dict[int, list[dict]]:
    """All RAPTOR summary nodes (level >= 1) grouped by level."""
    try:
        col = _chroma_collection(book_id)
    except Exception:
        return {}
    out: dict[int, list[dict]] = defaultdict(list)
    # The tree builder caps at level 3, but probe a couple extra to stay
    # robust if that changes.
    for level in range(1, 6):
        data = col.get(where={"raptor_level": level})
        ids = data.get("ids") or []
        if not ids:
            continue
        for meta, doc in zip(data.get("metadatas") or [], data.get("documents") or []):
            out[level].append({
                "node_id": meta.get("chunk_id", ""),
                "level": level,
                "section_id": meta.get("section_id") or "",
                "section_label": meta.get("section_label") or "",
                "cluster_size": meta.get("cluster_size"),
                "text": doc or "",
            })
    return dict(out)


@st.cache_data(show_spinner=False)
def _node_block_index(book_id: str) -> dict[str, list[str]]:
    """node_id -> sorted list of canonical block_ids it covers."""
    with _connect_ro() as conn:
        rows = conn.execute(
            "SELECT node_id, block_id FROM raptor_node_blocks WHERE book_id=?",
            (book_id,),
        ).fetchall()
    grouped: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        grouped[r["node_id"]].add(r["block_id"])
    return {nid: sorted(blocks) for nid, blocks in grouped.items()}


def load_raptor_tree(book_id: str) -> dict:
    """Build a nested L_top -> L1 -> L0 tree for the inspector.

    Parent/child linkage between top-level and L1 is derived by block
    coverage: an L1 node is the child of the highest-level node whose
    block set fully contains the L1's blocks. Sections (and hence L1
    coverages) are disjoint, so this gives a clean partition.

    Returns:
        {
          "has_top_level": bool,    # True when L2+ exists
          "roots":   list[node],    # top-level nodes (or L1 if no L2)
          "orphans": list[node],    # L1 with no L2 parent (rare)
        }

    Each node carries: node_id, level, text, block_ids, block_count,
    cluster_size, section_id (L1 only), section_label (L1 only).
    L1 nodes also carry `leaf_chunks` (the L0 chunks under that section).
    Top-level nodes carry `children` (their L1 descendants).
    """
    nodes_by_level = _summary_nodes(book_id)
    node_blocks = _node_block_index(book_id)

    if not nodes_by_level:
        return {"has_top_level": False, "roots": [], "orphans": []}

    for nodes in nodes_by_level.values():
        for n in nodes:
            blocks = node_blocks.get(n["node_id"], [])
            n["block_ids"] = blocks
            n["block_count"] = len(blocks)

    chunks_by_section: dict[str, list[dict]] = defaultdict(list)
    for c in _all_l0_chunks(book_id):
        chunks_by_section[c["section_id"]].append(c)
    l1_nodes = nodes_by_level.get(1, [])
    for n in l1_nodes:
        n["leaf_chunks"] = chunks_by_section.get(n["section_id"], [])

    max_level = max(nodes_by_level)
    if max_level < 2:
        return {
            "has_top_level": False,
            "roots": sorted(l1_nodes, key=lambda n: n.get("section_label", "")),
            "orphans": [],
        }

    top = nodes_by_level[max_level]
    used: set[str] = set()
    for parent in top:
        pblocks = set(parent["block_ids"])
        children = []
        for child in l1_nodes:
            cblocks = set(child["block_ids"])
            if cblocks and cblocks <= pblocks:
                children.append(child)
                used.add(child["node_id"])
        parent["children"] = sorted(
            children, key=lambda c: c.get("section_label", "")
        )

    orphans = [n for n in l1_nodes if n["node_id"] not in used]
    return {
        "has_top_level": True,
        "roots": sorted(top, key=lambda n: n.get("node_id", "")),
        "orphans": orphans,
    }


# ── ID lookup (search by chunk_id / raptor node_id) ─────────────────────────


def _normalize_id(raw: str) -> str | None:
    """Apply the inspector's id-input conventions.

    - Trim whitespace.
    - Pass `raptor_l…` ids through verbatim.
    - Allow a bare hex tail for level-0 chunks: prepend `chk_` if missing.
    Returns None for empty input.
    """
    qid = (raw or "").strip()
    if not qid:
        return None
    if qid.startswith("raptor_l"):
        return qid
    if qid.startswith("chk_"):
        return qid
    return f"chk_{qid}"


def _blocks_for_node(book_id: str, node_id: str) -> list[str]:
    """Block ids covered by a RAPTOR summary node (level >= 1)."""
    with _connect_ro() as conn:
        rows = conn.execute(
            "SELECT block_id FROM raptor_node_blocks "
            "WHERE book_id=? AND node_id=? ORDER BY block_id",
            (book_id, node_id),
        ).fetchall()
    return [r["block_id"] for r in rows]


def _sections_for_blocks(book_id: str, block_ids: list[str]) -> list[dict]:
    """Distinct sections that own the given blocks, in reading order.

    Used to describe what an L2+ cluster spans, since those nodes don't
    carry a single section_id.
    """
    if not block_ids:
        return []
    # SQLite has a default limit of 999 host parameters; chunk the IN clause.
    out: dict[str, dict] = {}
    with _connect_ro() as conn:
        for i in range(0, len(block_ids), 500):
            window = block_ids[i:i + 500]
            placeholders = ",".join("?" * len(window))
            rows = conn.execute(
                "SELECT DISTINCT s.section_id, s.order_idx, s.kind, "
                "s.printed_number, s.label FROM blocks b "
                "JOIN sections s "
                "  ON s.book_id = b.book_id AND s.section_id = b.section_id "
                f"WHERE b.book_id=? AND b.block_id IN ({placeholders})",
                (book_id, *window),
            ).fetchall()
            for r in rows:
                d = dict(r)
                out.setdefault(d["section_id"], d)
    return sorted(out.values(), key=lambda s: s.get("order_idx") or 0)


@st.cache_data(show_spinner=False)
def lookup_by_id(book_id: str, raw_id: str) -> dict | None:
    """Resolve a normalized id to a display record, or None if not found.

    Works for any node level — the level is read from the Chroma metadata
    rather than parsed from the id, so a typo like `raptor_l9_xyz` simply
    misses instead of crashing.
    """
    qid = _normalize_id(raw_id)
    if not qid:
        return None
    try:
        col = _chroma_collection(book_id)
    except Exception:
        return None
    data = col.get(where={"chunk_id": qid})
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    if not metas:
        return None
    meta = metas[0]
    text = docs[0] if docs else ""
    try:
        level = int(meta.get("raptor_level", 0))
    except (TypeError, ValueError):
        level = 0

    record: dict = {
        "id": qid,
        "level": level,
        "text": text,
        "section": None,
        "sections": [],
        "block_ids": [],
        "primary_block_id": None,
        "page": None,
        "cluster_size": meta.get("cluster_size"),
    }

    if level == 0:
        raw_blocks = meta.get("block_ids") or ""
        record["block_ids"] = [b for b in raw_blocks.split(",") if b]
        record["primary_block_id"] = meta.get("primary_block_id") or None
        # Chunker stores `page` as legacy_page: PDF page number, or 0 for
        # EPUB. Surface only when meaningful.
        page = meta.get("page")
        if isinstance(page, (int, float)) and page > 0:
            record["page"] = int(page)
        sid = meta.get("section_id") or None
        if sid:
            record["section"] = _section_by_id(book_id, sid)
    else:
        # raptor_node_blocks is the canonical source for L>=1 coverage.
        record["block_ids"] = _blocks_for_node(book_id, qid)
        sid = meta.get("section_id") or None
        if sid:
            # L1 nodes carry section_id directly; trust it.
            record["section"] = _section_by_id(book_id, sid)
        else:
            # L2+: span multiple sections, derived from block coverage.
            record["sections"] = _sections_for_blocks(
                book_id, record["block_ids"]
            )
    return record
