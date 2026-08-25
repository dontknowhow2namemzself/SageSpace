import mimetypes
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from core import cover as cover_gen
from core import database as db

router = APIRouter()


def _cover_url_for(book_id: str) -> str | None:
    """Resolve the cover endpoint URL if the file exists, else None.
    The path is relative; the frontend prefixes its NEXT_PUBLIC_API_URL."""
    return f"/api/books/{book_id}/cover" if cover_gen.get_cover_path(book_id) else None


@router.get("/books")
def list_books():
    books = db.list_books()
    result = []
    for book in books:
        total = book.get("total_chunks") or 1
        # Reader-facing progress: chunks CITED by answers across all
        # sessions (deduped) — not merely fetched by retrieval.
        cited = db.get_all_cited_chunk_ids_for_book(book["id"])
        digested_pct = min(round(len(cited) / total * 100, 1), 100.0)
        result.append({
            **book,
            "digested_pct": digested_pct,
            "cover_url": _cover_url_for(book["id"]),
        })
    return result


@router.get("/books/{book_id}")
def get_book(book_id: str):
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    total = book.get("total_chunks") or 1
    cited = db.get_all_cited_chunk_ids_for_book(book_id)
    return {
        **book,
        "digested_pct": min(round(len(cited) / total * 100, 1), 100.0),
        "cover_url": _cover_url_for(book_id),
    }


@router.get("/books/{book_id}/cover")
def get_book_cover(book_id: str):
    """Serve the generated cover PNG. 404 if the book has no cover —
    the frontend then falls back to the icon placeholder."""
    path = cover_gen.get_cover_path(book_id)
    if not path:
        raise HTTPException(status_code=404, detail="No cover for this book")
    return FileResponse(
        path=str(path),
        media_type="image/png",
        # Aggressive caching — covers are immutable per book_id; on
        # delete the file is removed so a stale cache cannot leak.
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/books/{book_id}/content")
def get_book_content(book_id: str):
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    file_path = book.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Book file is unavailable")

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Book file not found on disk")

    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.delete("/books/{book_id}")
def delete_book(book_id: str):
    import os
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if book.get("file_path") and os.path.exists(book["file_path"]):
        try:
            os.remove(book["file_path"])
        except OSError:
            pass

    cover_gen.delete_cover(book_id)

    try:
        import chromadb
        from core.raptor import CHROMA_DIR
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(f"book_{book_id}")
    except Exception:
        pass

    # GC LangGraph checkpoints for this book's threads BEFORE the session
    # rows are deleted (design §8 Q1: GC by ownership). Best-effort.
    from core.graph.build import gc_checkpoints_for_book
    gc_checkpoints_for_book(book_id)

    db.delete_book(book_id)
    return {"deleted": book_id}
