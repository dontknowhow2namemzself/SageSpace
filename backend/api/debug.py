import json
from fastapi import APIRouter, HTTPException
from core import database as db
from core.canonical import db as canonical_db
from core.raptor import get_vectorstore

router = APIRouter()


@router.get("/debug/books/{book_id}/chunk-map")
def get_chunk_map(book_id: str, session_id: str = ""):
    """Return the per-section chunk grid the Reading Map renders.

    Groups are formed by canonical section_id (post-PR4). Each group
    carries the section's printed label, kind, and printed_number so
    the frontend renders "CHAPTER VI. Pig and Pepper" instead of
    "Chapter 7" -- the latter came from the legacy chapter int mirror
    (= section.order_idx + 1) which silently included front-matter
    slots and confused users (see ARCHITECTURE.md §P2).

    Sort order is section.order_idx (book order). Sections that exist
    in the canonical layer but have no level-0 chunks do NOT appear in
    the grid. A "chapter": int field is still populated on each group
    for backwards-compat with older frontend builds; it equals
    section.order_idx + 1 unchanged.
    """
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    if not book["raptor_status"].startswith("ready"):
        raise HTTPException(status_code=400, detail="Book index not ready")

    try:
        vs = get_vectorstore(book_id)
        vs_data = vs.get(where={"raptor_level": 0})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vectorstore error: {exc}")

    metadatas = vs_data.get("metadatas") or []
    documents = vs_data.get("documents") or []

    lit_ids: set[str] = set()
    first_lit_event: dict[str, str] = {}

    if session_id:
        # "Lit" = cited by an answer's per-fact attribution, NOT merely
        # retrieved. The map shows what the user actually read through
        # the sage's answers (per-fact data-chunk-ids); retrieval hits
        # the synthesizer never cited stay dark. The turn-level
        # chunk_ids fallback applies ONLY to legacy events recorded
        # before per-fact attribution existed (no `facts` key at all):
        # a modern turn whose facts the mapper attributed to nothing
        # stays dark on purpose. Events are chronological (ORDER BY
        # created_at), so setdefault keeps the FIRST citing event per
        # chunk.
        for event in db.get_retrieval_events(session_id):
            attribution = json.loads(event.get("answer_attribution_json") or "null")
            if not attribution:
                continue
            facts = attribution.get("facts")
            cited = {
                cid
                for fact in (facts or [])
                for cid in (fact.get("chunk_ids") or [])
            }
            if not cited and facts is None:
                cited = set(attribution.get("chunk_ids") or [])
            for cid in cited:
                # Raw-only contract: facts may cite RAPTOR summary nodes
                # (the popup shows them, labeled), but the Reading Map
                # lights BOOK TEXT only — a summary citation must not
                # light anything.
                if cid.startswith("raptor_"):
                    continue
                lit_ids.add(cid)
                first_lit_event.setdefault(cid, event["id"])

    # Pull canonical sections once and index by section_id for O(1)
    # lookup during the grouping loop.
    sections_by_id: dict[str, dict] = {
        s["section_id"]: s for s in canonical_db.get_sections(book_id)
    }
    by_section: dict[str, list] = {}

    for meta, doc_text in zip(metadatas, documents):
        section_id = meta.get("section_id") or ""
        cid = meta.get("chunk_id", "")
        chunk_entry = {
            "chunk_id":           cid,
            "page":               meta.get("page", 0),
            "char_length":        len(doc_text),
            "is_lit":             cid in lit_ids,
            "first_lit_event_id": first_lit_event.get(cid),
            "preview_text":       doc_text[:200] if doc_text else "",
        }
        by_section.setdefault(section_id, []).append(chunk_entry)

    # Build groups sorted by canonical order_idx. Orphan chunks
    # (section_id missing from sections table) fall to the end so
    # they are visible for debug rather than silently dropped.
    groups: list[dict] = []
    for section_id, chunks in by_section.items():
        section = sections_by_id.get(section_id)
        if section is None:
            groups.append({
                "section_id":     section_id or "",
                "section_label":  "Unsorted",
                "kind":           "other",
                "printed_number": None,
                "order_idx":      10**9,  # sort last
                "chapter":        0,
                "chunks":         chunks,
            })
            continue
        order_idx = section.get("order_idx", 0) or 0
        groups.append({
            "section_id":     section_id,
            "section_label":  section.get("label") or "",
            "kind":           section.get("kind") or "other",
            "printed_number": section.get("printed_number"),
            "order_idx":      order_idx,
            "chapter":        order_idx + 1,  # legacy mirror, kept for compat
            "chunks":         chunks,
        })
    groups.sort(key=lambda g: g["order_idx"])

    return {
        "total_chunks": len(metadatas),
        "chapters": groups,
    }


@router.get("/debug/sessions/{session_id}/retrieval-events")
def get_session_retrieval_events(session_id: str):
    return db.get_retrieval_events(session_id)


@router.get("/debug/retrieval-events/{event_id}")
def get_retrieval_event_detail(event_id: str):
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE id=?", (event_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    event = dict(row)
    chunks = db.get_event_chunks(event_id)

    try:
        variants = json.loads(event.get("multi_query_variants_json") or "[]")
    except Exception:
        variants = []

    try:
        answer_attribution = json.loads(event.get("answer_attribution_json") or "null")
    except Exception:
        answer_attribution = None

    return {
        "event_id":              event["id"],
        "query_text":            event["query_text"],
        "created_at":            event["created_at"],
        "raw_hits_count":        event["raw_hits_count"],
        "new_raw_hits_count":    event["new_raw_hits_count"],
        "summary_hits_count":    event["summary_hits_count"],
        "multi_query_variants":  variants,
        "hyde_hypothesis":       event.get("hyde_hypothesis") or "",
        "chunks":                chunks,
        "answer_attribution":    answer_attribution,
        "faithfulness_score":    event.get("faithfulness_score"),
        "faithfulness_status":   event.get("faithfulness_status") or "pending",
        "faithfulness_reasoning": event.get("faithfulness_reasoning") or "",
    }


# ── Full chunk lookup (for "Show full chunk" in DetailPanel) ─────────────
_FULL_CHUNK_MAX_CHARS = 20_000


@router.get("/debug/books/{book_id}/chunks/{chunk_id}/full")
def get_chunk_full(book_id: str, chunk_id: str):
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    try:
        vs = get_vectorstore(book_id)
        data = vs.get(where={"chunk_id": chunk_id})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vectorstore error: {exc}")

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        raise HTTPException(status_code=404, detail="Chunk not found")

    text = documents[0] or ""
    truncated = len(text) > _FULL_CHUNK_MAX_CHARS
    if truncated:
        text = text[:_FULL_CHUNK_MAX_CHARS]

    meta = metadatas[0] if metadatas else {}
    # Look up the canonical section so the DetailPanel can render the
    # printed section label instead of the legacy chapter int.
    section_id = meta.get("section_id") or ""
    section_label = ""
    if section_id:
        for s in canonical_db.get_sections(book_id):
            if s["section_id"] == section_id:
                section_label = s.get("label") or ""
                break
    return {
        "chunk_id":      chunk_id,
        "chapter":       meta.get("chapter", 0),  # legacy mirror, kept for compat
        "page":          meta.get("page", 0),
        "raptor_level":  meta.get("raptor_level", 0),
        "char_length":   len(documents[0] or ""),
        "full_text":     text,
        "truncated":     truncated,
        "section_id":    section_id,
        "section_label": section_label,
    }
