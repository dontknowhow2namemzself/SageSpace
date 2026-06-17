"""Retrieve node (pipeline node 2).

Pure functions that turn an intent into a RetrievalResult. Two entry
points cover the three retrieval-bearing intents:

  * run_retrieval(query, ...)
      Used for kind=search and kind=book_overview. Runs the
      multi-query + HyDE combined retriever, persists a
      retrieval_event row, computes the chapter_clusters payload,
      and resolves citations for the top hits.

  * chapter_summary_retrieval(book_id, printed_number, ...)
      Used for kind=chapter_summary. Looks up the per-section
      level-1 RAPTOR node added in PR6 (deterministic chunk_id
      `raptor_l1_<section_id>`). When that node exists we skip
      similarity_search entirely. Falls back to a filtered
      similarity_search at level 0 for older books that have not
      been re-ingested under PR6.

Both functions:
  * take explicit (book_id, session_id, vectorstore) args -- no
    ContextVar, no module-level globals,
  * write the same retrieval_event row + retrieved_chunks rows the
    pre-PR5 code wrote, so Reading Map / Debug timelines stay
    coherent across the two intents,
  * return a RetrievalResult shaped for the synthesizer (docs +
    sources) AND for the SSE router (sse_payload).
"""
from __future__ import annotations

import json
import logging
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core import database as db
from core.canonical import db as canonical_db
from core.canonical.chunker import decode_block_ids
from core.canonical.citations import resolve_citation
from core.pipeline.types import RetrievalResult, RetrievalSource
from core.retriever import retrieve_combined


logger = logging.getLogger(__name__)


# MAX_SYNTH_DOCS caps how many retrieved docs feed the synthesizer on the
# search / book_overview path (assemble_retrieval_result). The fixed pipeline
# used 8, but the agentic ReAct + fan-out path routinely gathers far more (a
# "final three chapters" question fetches 3 chapters = 24 docs), so an 8-doc
# cap starved the synthesizer of most of the evidence. Raised to 16 and
# section-balanced upstream (nodes._balance_by_subquestion) so multi-section
# evidence survives the cut instead of the first section monopolizing it.
# (24 was tried first: it lifted completeness but cost faithfulness as the
# extra low-relevance docs invited ungrounded claims; 16 is the trade point.)
# MAX_CONTEXT_DOCS stays the chapter_summary path's internal fetch budget
# (_gather_section_docs). MAX_SOURCE_REFS feeds the UI citation card;
# _SOURCE_REF_TEXT_PREVIEW caps the source_refs snippet length.
MAX_SYNTH_DOCS = 16
MAX_CONTEXT_DOCS = 8
MAX_SOURCE_REFS = 4
_SOURCE_REF_TEXT_PREVIEW = 400


# ─────────────────────────────────────────────────────────────────────────────
# Search / book_overview path
# ─────────────────────────────────────────────────────────────────────────────


def run_retrieval(
    query: str,
    book_id: str,
    session_id: str,
    vectorstore: Chroma,
) -> RetrievalResult:
    """Run multi-query + HyDE retrieval and produce a RetrievalResult.

    Side effects (always run on success):
      * insert one retrieval_events row
      * insert N retrieval_event_chunks rows (one per hit)
      * update retrieved_chunks (session-level lit set)

    Returns a RetrievalResult whose sse_payload is the JSON-encoded
    retrieval_update frame the caller emits to the UI.
    """
    if vectorstore is None:
        return RetrievalResult(docs=[], sources=[], event_id=None, sse_payload=None)

    docs = retrieve_combined(query, vectorstore)
    return assemble_retrieval_result(
        book_id=book_id,
        session_id=session_id,
        query_text=query,
        docs=docs,
        vectorstore=vectorstore,
    )


def assemble_retrieval_result(
    *,
    book_id: str,
    session_id: str,
    query_text: str,
    docs: list[Document],
    vectorstore: Chroma,
) -> RetrievalResult:
    """Persist one turn's retrieval + build its RetrievalResult from an
    already-gathered list of Documents.

    Extracted from run_retrieval so the PR2 bounded-ReAct retrieve node
    (whose `docs` are the union of multiple tools' hits) produces the
    IDENTICAL side effects + payload a single semantic pass does:
      * one retrieval_events row + N retrieval_event_chunks rows,
      * retrieved_chunks lit-set update,
      * canonical citations + UI source_refs,
      * the retrieval_update SSE frame (chapter_clusters dot-grid).

    `query_text` labels the event (the turn's question). Docs should be
    ordered best-first; the top MAX_SYNTH_DOCS feed the synthesizer.
    """
    if not docs:
        return RetrievalResult(docs=[], sources=[], event_id=None, sse_payload=None)

    raw_docs = [d for d in docs if d.metadata.get("raptor_level", 0) == 0]
    summary_docs = [d for d in docs if d.metadata.get("raptor_level", 0) > 0]

    # The Reading Map / reading-progress lit-set records ONLY the docs that
    # actually feed the synthesizer (the top MAX_SYNTH_DOCS), not every chunk
    # the agent surfaced. The agentic ReAct + fan-out retrieve over-fetches
    # (whole chapters via get_chapter, parallel branches), so recording the
    # full hit-set would light a large slice of the book on a single question.
    # context_docs is what reaches the answer, so it is what we light.
    context_docs = docs[:MAX_SYNTH_DOCS]

    # Per-session lit-set diff for Reading Map's "new lighting" signal.
    lit_raw_ids = [
        d.metadata["chunk_id"] for d in context_docs
        if d.metadata.get("raptor_level", 0) == 0 and d.metadata.get("chunk_id")
    ]
    already_lit = set(db.get_retrieved_chunk_ids(session_id))
    newly_lit = list(dict.fromkeys(cid for cid in lit_raw_ids if cid not in already_lit))

    # Record the synthesis-context RAW chunks so the lit-set + progress
    # reflect what grounded the answer, not everything search touched.
    # Summary nodes are excluded: the digested-% denominator
    # (books.total_chunks) counts level-0 chunks only, so recording
    # raptor ids would push progress past 100%. (Debug timeline below
    # still logs the FULL hit set via raw_docs/summary_docs.)
    db.record_retrieved_chunks(session_id, book_id, lit_raw_ids)

    # retrieval_events + retrieval_event_chunks for the Debug timeline.
    book = db.get_book(book_id) or {}
    total_chunks = book.get("total_chunks") or 1
    event_id = db.create_retrieval_event(
        session_id=session_id,
        book_id=book_id,
        query_text=query_text,
        multi_query_variants_json="[]",
        hyde_hypothesis="",
        raw_hits_count=len(raw_docs),
        new_raw_hits_count=len(newly_lit),
        summary_hits_count=len(summary_docs),
    )
    _persist_event_chunks(event_id, docs, newly_lit)

    # Compute the chapter_clusters dot-grid Reading Map renders.
    chapter_clusters = _compute_chapter_clusters(vectorstore, session_id)

    # Resolve canonical citations for the top hits we plan to surface
    # (context_docs computed above — the synthesizer input + lit-set).
    citations_by_chunk = _resolve_citations(book_id, context_docs, vectorstore)

    # Build source_refs (UI citation chips, top MAX_SOURCE_REFS).
    source_refs = _build_source_refs(
        context_docs[:MAX_SOURCE_REFS], citations_by_chunk
    )

    # Pack the synthesizer input docs as dicts (so synthesize doesn't
    # depend on langchain Document types).
    synth_docs = [
        {
            "text": d.page_content,
            "chunk_id": d.metadata.get("chunk_id", ""),
            "section_id": d.metadata.get("section_id", ""),
            # Level-0 chunks carry `chapter_label` (the correct section title),
            # NOT `section_label`. Falling through to chapter_label keeps the
            # synthesizer's CONTEXT header on the real title ("CHAPTER VI. Pig
            # and Pepper") instead of the wrong numeric fallback "Chapter N"
            # (the numeric `chapter` is the section's position index, which
            # counts front-matter — see chunker; Fix B re-ingest fixes that).
            "section_label": d.metadata.get("section_label")
            or d.metadata.get("chapter_label", ""),
            "chapter": d.metadata.get("chapter", 0),
            "page": d.metadata.get("page", 0),
            "raptor_level": d.metadata.get("raptor_level", 0),
        }
        for d in context_docs
    ]

    all_lit_ids = set(db.get_retrieved_chunk_ids(session_id))
    sse_payload = json.dumps(
        {
            "type": "retrieval_update",
            "session_lit_chunks": len(all_lit_ids),
            "total_chunks": total_chunks,
            "newly_lit_count": len(newly_lit),
            "newly_lit_chunk_ids": newly_lit,
            "chapter_clusters": chapter_clusters,
            "sources": [_source_to_dict(s) for s in source_refs],
        },
        ensure_ascii=False,
    )

    return RetrievalResult(
        docs=synth_docs,
        sources=source_refs,
        event_id=event_id,
        sse_payload=sse_payload,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chapter-summary path
# ─────────────────────────────────────────────────────────────────────────────


def chapter_summary_retrieval(
    book_id: str,
    printed_number: int,
    session_id: str,
    vectorstore: Chroma,
) -> RetrievalResult:
    """Resolve "Chapter N" to the section the author labeled Chapter N,
    then return the section's level-1 RAPTOR summary AND its level-0
    chunks together.

    Why both: pre-PR7, this function returned ONLY the level-1 node
    so the synthesizer always had a clean chapter summary as input.
    Side effect: every <fact> tag in the answer ended up with the same
    single raptor_l1_<section_id> as its only attributable source, so
    citation chips on a chapter_summary turn all pointed to the same
    block. PR7 fixes that by also surfacing the section's level-0
    chunks -- each one anchored to its own primary_block_id -- giving
    the answer attribution multiple block-level provenance options.

    Falls back to a filtered level-0 similarity_search for books that
    have not been re-ingested under PR6 (no per-section level-1 node).
    """
    if vectorstore is None or printed_number is None or printed_number <= 0:
        return RetrievalResult(docs=[], sources=[], event_id=None, sse_payload=None)

    section = _resolve_chapter_section(book_id, printed_number)
    if section is None:
        return RetrievalResult(docs=[], sources=[], event_id=None, sse_payload=None)

    section_id = section["section_id"]
    section_label = section.get("label") or f"Chapter {printed_number}"

    docs = _gather_section_docs(vectorstore, section, printed_number)
    if not docs:
        return RetrievalResult(docs=[], sources=[], event_id=None, sse_payload=None)

    # Persist retrieval_event so chapter_summary turns light up the
    # Reading Map like search turns do. This is one of the gaps PR5
    # closes (chapter_summary was silently skipping observability).
    book = db.get_book(book_id) or {}
    total_chunks = book.get("total_chunks") or 1
    event_id = db.create_retrieval_event(
        session_id=session_id,
        book_id=book_id,
        query_text=f"chapter_summary(printed_number={printed_number})",
        multi_query_variants_json="[]",
        hyde_hypothesis="",
        raw_hits_count=sum(1 for d in docs if d.metadata.get("raptor_level", 0) == 0),
        new_raw_hits_count=0,  # filled in below
        summary_hits_count=sum(1 for d in docs if d.metadata.get("raptor_level", 0) > 0),
    )
    raw_chunk_ids = [
        d.metadata["chunk_id"] for d in docs
        if d.metadata.get("raptor_level", 0) == 0 and d.metadata.get("chunk_id")
    ]
    already_lit = set(db.get_retrieved_chunk_ids(session_id))
    newly_lit = list(dict.fromkeys(cid for cid in raw_chunk_ids if cid not in already_lit))
    # Raw chunks only — summary nodes are not part of the digested-%
    # denominator (see persist_retrieval above).
    db.record_retrieved_chunks(session_id, book_id, raw_chunk_ids)
    _persist_event_chunks(event_id, docs, newly_lit)

    # Citation resolution for the surfaced docs.
    citations_by_chunk = _resolve_citations(book_id, docs, vectorstore)
    source_refs = _build_source_refs(docs[:MAX_SOURCE_REFS], citations_by_chunk)

    synth_docs = [
        {
            "text": d.page_content,
            "chunk_id": d.metadata.get("chunk_id", ""),
            "section_id": section_id,
            "section_label": section_label,
            "chapter": printed_number,
            "page": d.metadata.get("page", 0),
            "raptor_level": d.metadata.get("raptor_level", 0),
        }
        for d in docs
    ]

    all_lit_ids = set(db.get_retrieved_chunk_ids(session_id))
    chapter_clusters = _compute_chapter_clusters(vectorstore, session_id)
    sse_payload = json.dumps(
        {
            "type": "retrieval_update",
            "session_lit_chunks": len(all_lit_ids),
            "total_chunks": total_chunks,
            "newly_lit_count": len(newly_lit),
            "newly_lit_chunk_ids": newly_lit,
            "chapter_clusters": chapter_clusters,
            "sources": [_source_to_dict(s) for s in source_refs],
        },
        ensure_ascii=False,
    )

    return RetrievalResult(
        docs=synth_docs,
        sources=source_refs,
        event_id=event_id,
        sse_payload=sse_payload,
    )


def fetch_chapter_docs(
    book_id: str, printed_number: int, vectorstore: Chroma,
    query: str | None = None,
) -> list[Document]:
    """Pure chapter-section doc fetch -- the PR2 `get_chapter` agent tool.

    Resolves "Chapter N" to its section and returns that section's level-1
    summary + level-0 chunks (or a similarity fallback for pre-PR6 books),
    the SAME gather chapter_summary_retrieval uses -- but with NO DB side
    effects (no retrieval_event / no SSE). The ReAct retrieve node persists
    exactly one event for the whole turn, so its tools must be side-effect
    free. Returns [] when the book/chapter can't be resolved.

    `query` (the agent's sub-question) selects the most relevant level-0
    chunks instead of an even reading-order spread -- see
    _fetch_level_0_chunks_for_section.
    """
    if vectorstore is None or not printed_number or printed_number <= 0:
        return []
    section = _resolve_chapter_section(book_id, printed_number)
    if section is None:
        return []
    return _gather_section_docs(vectorstore, section, printed_number, query=query)


def _gather_section_docs(
    vectorstore: Chroma, section: dict, printed_number: int,
    query: str | None = None,
) -> list[Document]:
    """Level-1 summary + level-0 chunks for a resolved section (summary
    first so the synthesizer sees the chapter overview before raw
    passages), falling back to a filtered similarity_search for pre-PR6
    books. Shared by chapter_summary_retrieval (no query -> reading-order
    spread) and fetch_chapter_docs (agent sub-question -> relevance)."""
    section_id = section["section_id"]
    section_label = section.get("label") or f"Chapter {printed_number}"

    level_1_docs = _fetch_level_1_node(vectorstore, section_id)
    # Also pull level-0 chunks so citation attribution has block-level
    # anchors. Cap so the combined input stays under MAX_CONTEXT_DOCS.
    level_0_docs = _fetch_level_0_chunks_for_section(
        vectorstore, section_id,
        limit=MAX_CONTEXT_DOCS - len(level_1_docs),
        query=query,
    )
    if level_1_docs or level_0_docs:
        docs = level_1_docs + level_0_docs
    else:
        docs = _similarity_fallback(vectorstore, section_id, printed_number)

    # Ensure section/chapter labels are present for downstream context
    # formatting (raw Chroma metadata may lack them on some books).
    for d in docs:
        d.metadata.setdefault("section_id", section_id)
        d.metadata.setdefault("section_label", section_label)
        d.metadata.setdefault("chapter", printed_number)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _persist_event_chunks(event_id: str, docs: Iterable[Document], newly_lit: list[str]) -> None:
    newly_lit_set = set(newly_lit)
    rows = []
    for rank, doc in enumerate(docs, start=1):
        rows.append(
            {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "raptor_level": doc.metadata.get("raptor_level", 0),
                "chapter": doc.metadata.get("chapter", 0),
                "page": doc.metadata.get("page", 0),
                "rank": rank,
                # Prefer the ReAct tool origin (semantic/keyword/chapter/
                # neighbor/both); fall back to the multi_query/hyde origin
                # the legacy single-pass retriever stamps.
                "origin": doc.metadata.get("origin")
                or doc.metadata.get("retrieval_origin", "unknown"),
                "is_new_lighting": 1 if doc.metadata.get("chunk_id") in newly_lit_set else 0,
                # Was [:200], which truncated level-0 chunks (chunk_size=800) to a
                # quarter — that starved the sage-eval precision/faithfulness LLM
                # judges, which read this field and marked even gold chunks
                # "not relevant" because the supporting text fell past char 200.
                # 2000 covers a full level-0 chunk + most RAPTOR summaries.
                "preview_text": doc.page_content[:2000],
            }
        )
    db.add_event_chunks(event_id, rows)


def _compute_chapter_clusters(vectorstore: Chroma, session_id: str) -> list[dict]:
    """Build the dot-grid Reading Map payload by counting level-0 chunks
    per chapter int and comparing against the session's lit set."""
    all_lit_ids = set(db.get_retrieved_chunk_ids(session_id))
    chapter_totals: dict[int, int] = {}
    chapter_lit: dict[int, int] = {}
    if not hasattr(vectorstore, "get"):
        return []
    vs_docs = vectorstore.get(where={"raptor_level": 0})
    for meta in (vs_docs.get("metadatas") or []):
        ch = meta.get("chapter", 0)
        cid = meta.get("chunk_id", "")
        chapter_totals[ch] = chapter_totals.get(ch, 0) + 1
        if cid in all_lit_ids:
            chapter_lit[ch] = chapter_lit.get(ch, 0) + 1
    return [
        {"chapter": ch, "total": chapter_totals[ch], "lit": chapter_lit.get(ch, 0)}
        for ch in sorted(chapter_totals)
    ]


def _resolve_citations(
    book_id: str, docs: list[Document], vectorstore: Chroma
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "")
        if not cid:
            continue
        citation = resolve_citation(book_id, cid, vectorstore)
        if citation is not None:
            out[cid] = citation
    return out


def _build_source_refs(
    docs: list[Document], citations_by_chunk: dict[str, dict]
) -> list[RetrievalSource]:
    out: list[RetrievalSource] = []
    for doc in docs:
        ch = doc.metadata.get("chapter", 0)
        page = doc.metadata.get("page", 0)
        level = doc.metadata.get("raptor_level", 0)
        cid = doc.metadata.get("chunk_id", "")
        citation = citations_by_chunk.get(cid)

        if citation is not None:
            section_label = citation.get("section_label") or f"Chapter {ch}"
            primary_page = citation.get("source_locator", {}).get("page", page)
            label = (
                f"{section_label} · Page {primary_page}"
                if level == 0
                else f"{section_label} · Summary"
            )
            out.append(
                RetrievalSource(
                    label=label,
                    chunk_id=cid,
                    text=doc.page_content[:_SOURCE_REF_TEXT_PREVIEW],
                    chapter=ch if isinstance(ch, int) else 0,
                    page=primary_page if isinstance(primary_page, int) else 0,
                    citation_id=cid,
                    section_id=citation.get("section_id") or "",
                    section_label=section_label,
                    primary_block_id=citation["anchor"]["primary_block_id"],
                    block_ids=citation["anchor"]["block_ids"],
                    retrieved_layer=citation["evidence"]["retrieved_from"]["layer"],
                )
            )
        else:
            label = (
                f"Chapter {ch} · Page {page}"
                if level == 0
                else f"Chapter {ch} · Summary"
            )
            out.append(
                RetrievalSource(
                    label=label,
                    chunk_id=cid,
                    text=doc.page_content[:_SOURCE_REF_TEXT_PREVIEW],
                    chapter=ch if isinstance(ch, int) else 0,
                    page=page if isinstance(page, int) else 0,
                )
            )
    return out


def _source_to_dict(src: RetrievalSource) -> dict:
    """RetrievalSource -> dict that matches the JSON shape pre-PR5 callers
    consumed. Drops null canonical fields so legacy consumers do not see
    citation_id=None / section_id=None / etc. -- matches the original
    "key absent when not present" contract."""
    base = {
        "label": src.label,
        "chunk_id": src.chunk_id,
        "text": src.text,
        "chapter": src.chapter,
        "page": src.page,
    }
    if src.citation_id is not None:
        base["citation_id"] = src.citation_id
        base["section_id"] = src.section_id or ""
        base["section_label"] = src.section_label or ""
        base["primary_block_id"] = src.primary_block_id or ""
        base["block_ids"] = list(src.block_ids or [])
        base["retrieved_layer"] = src.retrieved_layer or ""
    return base


def _resolve_chapter_section(book_id: str, printed_number: int) -> dict | None:
    """PR4 lookup: pick the section whose author labeled it Chapter N.
    See ARCHITECTURE.md §P2 / tools.get_chapter_summary for the
    fallback ladder rationale."""
    if not book_id:
        return None
    sections = canonical_db.get_sections(book_id)
    chapters = [s for s in sections if s.get("kind") == "chapter"]
    target = next(
        (s for s in chapters if s.get("printed_number") == printed_number),
        None,
    )
    if target is not None:
        return target
    # Secondary: Nth body-matter section
    body_matter = [
        s for s in sections
        if s.get("kind") in ("chapter", "prologue", "epilogue")
    ]
    if 0 < printed_number <= len(body_matter):
        return body_matter[printed_number - 1]
    return None


def _fetch_level_1_node(vectorstore: Chroma, section_id: str) -> list[Document]:
    """PR6 fast path: pull the deterministic raptor_l1_<section_id>
    node directly. Returns [] when the node is not present (legacy
    book or section had no level-1 produced)."""
    try:
        hit = vectorstore.get(where={"chunk_id": f"raptor_l1_{section_id}"})
    except Exception:
        return []
    metadatas = (hit or {}).get("metadatas") or []
    documents = (hit or {}).get("documents") or []
    if not metadatas or not documents:
        return []
    return [Document(page_content=documents[0], metadata=metadatas[0])]


def _fetch_level_0_chunks_for_section(
    vectorstore: Chroma, section_id: str, limit: int = 7, query: str | None = None
) -> list[Document]:
    """Pull level-0 chunks that belong to `section_id`, at most `limit`.

    A chapter is usually LARGER than `limit`, so which chunks survive the
    cut matters. The original implementation took the first `limit` in
    Chroma's arbitrary return order (~insertion ~reading order), which
    silently dropped everything past the chapter opening — sage-eval traced
    two chronic completeness misses (the Dormouse's story, the pig
    transformation) to chunks sitting at positions 12-21 of their chapters.

    Selection strategy:
      * `query` given (the ReAct get_chapter tool threads the agent's
        sub-question through): the `limit` most RELEVANT chunks via a
        section-filtered similarity_search.
      * no `query` (chapter_summary's generic "what happens in ch N"):
        an even READING-ORDER spread across the whole section, so the
        beginning, middle and end are all represented.
    """
    if limit <= 0:
        return []

    section_filter = {
        "$and": [
            {"raptor_level": {"$eq": 0}},
            {"section_id": {"$eq": section_id}},
        ]
    }

    if query:
        try:
            return vectorstore.similarity_search(query, k=limit, filter=section_filter)
        except Exception:
            return []

    try:
        hit = vectorstore.get(where=section_filter)
    except Exception:
        return []
    metadatas = (hit or {}).get("metadatas") or []
    documents = (hit or {}).get("documents") or []
    if not metadatas or not documents:
        return []
    docs = [
        Document(page_content=doc_text or "", metadata=meta or {})
        for meta, doc_text in zip(metadatas, documents)
    ]
    return _stride_sample(_sort_by_reading_order(docs, section_id), limit)


def _sort_by_reading_order(docs: list[Document], section_id: str) -> list[Document]:
    """Order chunks by their first block's order_idx (the chunker emits
    block_ids in reading order, so a chunk's first block marks its place
    in the chapter). Falls back to the incoming order when canonical
    block data is unavailable (legacy books)."""
    book_id = next(
        (d.metadata.get("book_id") for d in docs if d.metadata.get("book_id")), None
    )
    if not book_id:
        return docs
    try:
        blocks = canonical_db.get_blocks(book_id, section_id=section_id)
    except Exception:
        return docs
    order = {b["block_id"]: b["order_idx"] for b in blocks}
    if not order:
        return docs

    def reading_pos(d: Document) -> int:
        ids = decode_block_ids(d.metadata.get("block_ids"))
        idxs = [order[b] for b in ids if b in order]
        return min(idxs) if idxs else 1 << 30

    return sorted(docs, key=reading_pos)


def _stride_sample(docs: list[Document], limit: int) -> list[Document]:
    """Even spread of `limit` docs across the list, always keeping the
    first and last — so a chapter's middle and ending survive the cut
    instead of only its opening."""
    n = len(docs)
    if n <= limit:
        return docs
    step = (n - 1) / (limit - 1)
    idxs = sorted({round(i * step) for i in range(limit)})
    return [docs[i] for i in idxs]


def _similarity_fallback(
    vectorstore: Chroma, section_id: str, printed_number: int
) -> list[Document]:
    """Legacy / pre-PR6 path: filtered similarity_search at level 0.
    Returns up to 6 chunks for the section, which is enough context
    for the synthesizer to produce a chapter-level answer."""
    section_filter = {
        "$and": [
            {"raptor_level": {"$eq": 0}},
            {"section_id": {"$eq": section_id}},
        ]
    }
    try:
        return vectorstore.similarity_search(
            f"Chapter {printed_number}", k=6, filter=section_filter
        )
    except Exception:
        return []
