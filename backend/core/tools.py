"""Deterministic tool implementations for the chat pipeline.

After PR5 (pipeline refactor) these are plain Python functions taking
explicit (book_id, session_id, ...) args -- no ContextVar plumbing, no
@tool LangChain wrapper, no ReAct agent above them. The pipeline's
dispatch in api/chat.py calls them directly:

  * compute_reading_progress -> dict (synthesizer renders the prose)
  * run_export               -> dict (synthesizer renders the prose)

Search / chapter_summary retrieval logic was moved to
core/pipeline/retrieve.py in the same PR. This module now houses only
the two intents that have no LLM step (progress + export).
"""
from __future__ import annotations

import json
from pathlib import Path

from core import database as db
from core.paths import DATA_DIR


# ── reading_progress ──────────────────────────────────────────────────────


def compute_reading_progress(book_id: str, session_id: str) -> dict:
    """Snapshot per-session reading progress metrics.

    Returns a dict consumed by the synthesizer (which weaves these
    numbers into prose). Returns a defaulted-empty dict when the book
    or session cannot be found -- the synthesizer downgrades to a
    polite "no progress data available" reply.
    """
    book = db.get_book(book_id)
    session = db.get_session(session_id)
    if not book or not session:
        return {
            "digested_pct": "0.0%",
            "cited_chunk_count": 0,
            "total_chunks": 0,
            "available": False,
        }

    total_chunks = book.get("total_chunks") or 1
    # Reader-facing progress = chunks CITED by this session's answers
    # (per-fact attribution), matching the Insight panel / shelf %.
    cited = db.get_cited_chunk_ids(session_id)
    digested_pct = round(len(cited) / total_chunks * 100, 1)

    # Time-based metrics (session minutes, "minutes to finish" forecast)
    # were removed 2026-06-10: they measured wall-clock time since the
    # session row was created, so an idle tab inflated them and the
    # forecast read as nonsense.
    return {
        "digested_pct": f"{digested_pct}%",
        "cited_chunk_count": len(cited),
        "total_chunks": total_chunks,
        "available": True,
    }


# ── export_notes ──────────────────────────────────────────────────────────


def run_export(book_id: str, session_id: str, format: str = "markdown") -> dict:
    """Compile the session conversation into a markdown or PDF file.

    Returns a dict {format, path, available} that the synthesizer renders
    as a confirmation. `available=False` indicates we had nothing to
    write (no session or empty history).
    """
    session = db.get_session(session_id)
    if not session:
        return {"format": format, "path": "", "available": False, "reason": "session_not_found"}

    history = json.loads(session.get("conversation_json", "[]"))
    if not history:
        return {"format": format, "path": "", "available": False, "reason": "empty_history"}

    book = db.get_book(book_id)
    book_title = book["title"] if book else "Unknown Book"

    export_dir = DATA_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c for c in book_title if c.isalnum() or c in "_ -")[:20]
    filename = f"session_{session_id[:8]}_{safe_title}"

    if format == "pdf":
        path = _export_pdf(history, book_title, export_dir, filename)
    else:
        path = _export_markdown(history, book_title, export_dir, filename)
    return {"format": format, "path": path, "available": True}


def _export_markdown(history: list, book_title: str, export_dir: Path, filename: str) -> str:
    lines = [f"# 《{book_title}》Conversation Notes\n"]
    for msg in history:
        role = "**YOU**" if msg["role"] == "user" else "**SAGE**"
        lines.append(f"{role}: {msg['content']}\n")
    path = export_dir / f"{filename}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"/exports/{filename}.md"


def _export_pdf(history: list, book_title: str, export_dir: Path, filename: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    path = export_dir / f"{filename}.pdf"
    doc_template = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"《{book_title}》Conversation Notes", styles["Title"]),
        Spacer(1, 20),
    ]
    for msg in history:
        prefix = "YOU: " if msg["role"] == "user" else "SAGE: "
        text = msg["content"][:500].replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(f"<b>{prefix}</b>{text}", styles["Normal"]))
        story.append(Spacer(1, 8))
    doc_template.build(story)
    return f"/exports/{filename}.pdf"
