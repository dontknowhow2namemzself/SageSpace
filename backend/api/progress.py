from fastapi import APIRouter, HTTPException
from core import database as db

router = APIRouter()


@router.get("/progress/{book_id}")
def get_progress(book_id: str, session_id: str):
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    total_chunks = book.get("total_chunks") or 1
    # Reader-facing progress (2026-06-10): chunks this session's answers
    # actually CITED (per-fact attribution), matching the Reading Map
    # and the shelf %. The internal "fed to the synthesizer" ledger
    # (retrieved_chunks) is no longer surfaced to the user.
    cited = db.get_cited_chunk_ids(session_id)
    digested_pct = min(round(len(cited) / total_chunks * 100, 1), 100.0)

    # chapter_clusters: aggregate from ChromaDB, counting lit/total per chapter
    from core.raptor import get_vectorstore
    try:
        vs = get_vectorstore(book_id)
        vs_data = vs.get(where={"raptor_level": 0})
        metadatas = vs_data.get("metadatas") or []
    except Exception:
        metadatas = []

    lit_ids = set(cited)
    chapter_totals: dict[int, int] = {}
    chapter_lit_counts: dict[int, int] = {}
    for meta in metadatas:
        ch = meta.get("chapter", 0)
        cid = meta.get("chunk_id", "")
        chapter_totals[ch] = chapter_totals.get(ch, 0) + 1
        if cid in lit_ids:
            chapter_lit_counts[ch] = chapter_lit_counts.get(ch, 0) + 1

    chapter_clusters = [
        {"chapter": ch, "total": chapter_totals[ch], "lit": chapter_lit_counts.get(ch, 0)}
        for ch in sorted(chapter_totals)
    ]

    # last_retrieval: the most recent retrieval event for this session
    events = db.get_retrieval_events(session_id)
    last_retrieval = None
    if events:
        last = events[-1]
        last_retrieval = {
            "event_id": last["id"],
            "query_text": last["query_text"],
            "newly_lit_count": last["new_raw_hits_count"],
        }

    return {
        "digested_pct": digested_pct,
        "cited_chunks": len(cited),
        "total_chunks": total_chunks,
        "token_stats": {
            "tokens_in": session.get("total_tokens_in", 0),
            "tokens_out": session.get("total_tokens_out", 0),
            "cost_usd": round(session.get("total_cost_usd", 0.0), 4),
        },
        "chapter_clusters": chapter_clusters,
        "last_retrieval": last_retrieval,
    }
