"""Top-level canonical ingest.

ingest_canonical(book_id, file_path)
    Pure ingest of the canonical layer (sections + blocks + ingestion report).
    Does NOT touch Chroma, chunks, RAPTOR, or any chat/SSE paths. After this
    call the book's ingest_version is 2 and the canonical browse APIs (added
    in a later step) will serve it; the chunk + RAPTOR rebuild that wires
    block_ids into Chroma metadata lives in phase 2 of the refactor plan.

CLI usage:
    python -m core.canonical.ingest <book_id>
    python -m core.canonical.ingest --file /abs/path.pdf  # for ad-hoc files

The CLI is intentionally minimal: it is meant for manual smoke runs and
re-ingests during the migration window, not for production traffic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core import database as db
from core.canonical import db as canonical_db


def ingest_canonical(book_id: str, file_path: str) -> dict:
    """Normalize `file_path` and persist its canonical layer for `book_id`.
    Returns the ingestion report. Raises on unknown format or read error.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        sections, blocks, report = _ingest_pdf(book_id, file_path)
    elif suffix == ".epub":
        sections, blocks, report = _ingest_epub(book_id, file_path)
    else:
        raise ValueError(f"Unsupported source format: {suffix!r}")

    canonical_db.replace_canonical_book(
        book_id=book_id,
        sections=sections,
        blocks=blocks,
        report=report,
    )
    return report


def _ingest_pdf(book_id: str, file_path: str):
    import fitz  # PyMuPDF - imported lazily so EPUB-only paths don't pay for it
    from core.canonical.normalize_pdf import normalize

    doc = fitz.open(file_path)
    try:
        return normalize(doc, book_id=book_id)
    finally:
        doc.close()


def _ingest_epub(book_id: str, file_path: str):
    from ebooklib import epub
    from core.canonical.normalize_epub import normalize

    book = epub.read_epub(file_path, options={"ignore_ncx": True})
    return normalize(book, book_id=book_id)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m core.canonical.ingest",
        description="Run canonical ingest for one book (no Chroma writes).",
    )
    parser.add_argument(
        "book_id",
        nargs="?",
        help="Existing books.id; file_path is read from the DB row.",
    )
    parser.add_argument(
        "--file",
        help="Override file path. Useful when ingesting a standalone PDF/EPUB "
        "that is not yet registered in the books table.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the ingestion report as JSON instead of a human summary.",
    )
    args = parser.parse_args(argv)

    if not args.book_id and not args.file:
        parser.error("either book_id or --file is required")

    db.init_db()  # idempotent; safe even if main app already ran it

    if args.book_id:
        book = db.get_book(args.book_id)
        if book is None:
            print(f"error: no book with id {args.book_id!r}", file=sys.stderr)
            return 2
        file_path = args.file or book.get("file_path")
        if not file_path:
            print(
                f"error: book {args.book_id!r} has no file_path and --file not given",
                file=sys.stderr,
            )
            return 2
        book_id = args.book_id
    else:
        # Ad-hoc mode: synthesize a book row keyed on file path so we have a
        # stable book_id without forcing the user to insert manually.
        from uuid import uuid5, NAMESPACE_URL

        file_path = args.file
        book_id = f"adhoc_{uuid5(NAMESPACE_URL, file_path).hex[:12]}"
        if db.get_book(book_id) is None:
            from core.database import get_conn
            conn = get_conn()
            try:
                conn.execute(
                    "INSERT INTO books (id, title, upload_date, raptor_status, "
                    "file_path, ingest_version) VALUES (?, ?, ?, ?, ?, 1)",
                    (
                        book_id,
                        Path(file_path).stem,
                        "1970-01-01T00:00:00Z",
                        "skipped",
                        file_path,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    report = ingest_canonical(book_id, file_path)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    # Human summary
    print(f"book_id        : {book_id}")
    print(f"file           : {file_path}")
    print(f"locator_type   : {report.get('locator_type')}")
    print(f"final_blocks   : {report.get('final_blocks')}")
    print(f"sections_total : {report.get('sections_total')}")
    print(f"section_source : {report.get('section_source')}")
    if report.get("locator_type") == "pdf":
        print(f"pages_total    : {report.get('pages_total')}")
        print(f"  dropped headers/footers : {report.get('dropped_headers_or_footers')}")
        print(f"  dropped page numbers    : {report.get('dropped_page_numbers')}")
        print(f"  cross-page merges       : {report.get('merged_page_breaks')}")
        print(f"  dehyphenations          : {report.get('dehyphenated_words')}")
    else:
        print(f"spine_items    : {report.get('spine_items')}")
        print(f"  dropped nav subtrees    : {report.get('dropped_nav_subtrees')}")
        print(f"  dropped empty blocks    : {report.get('dropped_empty_blocks')}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
