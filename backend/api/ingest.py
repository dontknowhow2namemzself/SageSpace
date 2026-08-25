import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form, Request

from core import cover as cover_gen
from core import database as db
from core.canonical import db as canonical_db
from core.canonical.chunker import chunk_blocks
from core.canonical.ingest import ingest_canonical
from core.paths import DATA_DIR
from core.raptor import build_raptor_index
from core.ratelimit import INGEST_RATE_LIMIT, limiter


router = APIRouter()
UPLOAD_DIR = DATA_DIR / "uploads"


_MAX_LABEL_CHARS = 200
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_label(text: str | None, max_len: int = _MAX_LABEL_CHARS) -> str:
    """Normalize a user-supplied label (title / author) for safe storage.

    Strip surrounding whitespace, drop ASCII control chars (no legitimate
    use in book metadata; a common injection vector for log forgery and
    XSS payloads that begin with newlines or NULs), then cap length.
    Returns an empty string when nothing usable remains — callers decide
    whether to reject or fall back to a default.
    """
    cleaned = _CONTROL_CHARS_RE.sub("", (text or "").strip())
    return cleaned[:max_len]


@router.post("/ingest")
@limiter.limit(INGEST_RATE_LIMIT)
async def ingest_book(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=""),
    author: Optional[str] = Form(default=""),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".epub"):
        raise HTTPException(status_code=400, detail="Only PDF and ePub files are supported")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must not exceed 50 MB")

    # Title / author / filename-stem fallback all run through the same
    # sanitizer: the upload filename is also user-controlled, so the
    # fallback must not bypass the cleaning.
    # Starlette percent-encodes control bytes in upload filenames before
    # we see them (NUL → %00, etc.), so the fallback decodes first to
    # close that bypass — otherwise %00%00.epub would survive as a title.
    raw_stem = unquote(Path(file.filename or "").stem)
    file_title = (
        _sanitize_label(title)
        or _sanitize_label(raw_stem)
        or "Untitled"
    )
    sanitized_author = _sanitize_label(author) or None
    book_id = db.create_book(file_title, sanitized_author, "")

    save_path = UPLOAD_DIR / f"{book_id}{suffix}"
    save_path.write_bytes(content)

    conn = db.get_conn()
    conn.execute("UPDATE books SET file_path=? WHERE id=?", (str(save_path), book_id))
    conn.commit()
    conn.close()

    background_tasks.add_task(_build_index, book_id, str(save_path))
    return {"book_id": book_id, "status": "building"}


def _build_index(book_id: str, file_path: str) -> None:
    """Background task entry. Runs the canonical ingest pipeline:

      1. ingest_canonical -> sections, blocks, ingestion_report; bumps
         books.ingest_version to 2.
      2. derive chunks from blocks via the canonical chunker, with metadata
         carrying block_ids / primary_block_id / section_id.
      3. build the RAPTOR tree, registering node->block links so summary
         hits can resolve back to canonical blocks.
    """
    try:
        db.update_book_status(book_id, "building")

        ingest_canonical(book_id, file_path)

        blocks = canonical_db.get_blocks(book_id)
        sections = canonical_db.get_sections(book_id)
        chunks = chunk_blocks(
            book_id,
            blocks,
            section_label_by_id={s["section_id"]: s["label"] for s in sections},
            section_order_by_id={s["section_id"]: s["order_idx"] for s in sections},
        )

        node_block_pairs: list[tuple[str, str]] = []

        def _register(node_id: str, covers_block_ids: set[str]) -> None:
            for bid in covers_block_ids:
                node_block_pairs.append((node_id, bid))

        build_raptor_index(
            chunks,
            book_id,
            sections=sections,
            register_block_links=_register,
        )
        canonical_db.replace_raptor_node_blocks(book_id, node_block_pairs)

        db.update_book_status(book_id, "ready", len(chunks), len(sections))

        # Cover generation is best-effort and runs after the book is
        # already "ready" — the user can chat immediately; the cover
        # materializes ~30 s later on the next /api/books poll. Never
        # blocks or affects the ready status.
        book = db.get_book(book_id)
        if book:
            cover_gen.generate_cover(book_id, book.get("title") or "")
    except Exception as exc:
        db.update_book_status(book_id, f"error: {str(exc)[:100]}")
