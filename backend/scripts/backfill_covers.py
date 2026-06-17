"""Generate covers for books that don't have one yet.

Books ingested before the cover-generation feature landed will show a
fallback icon on the shelf. This script walks the books table, finds
each book without a cover file on disk, and calls the same code path
the ingest pipeline uses.

Usage (from sagespace/backend, with the venv active):

    python scripts/backfill_covers.py            # dry-run: list missing
    python scripts/backfill_covers.py --apply    # actually generate

The --apply mode charges roughly $0.039 per book (current OpenRouter
pricing for google/gemini-2.5-flash-image). The script prints a total-
cost estimate up front and asks for confirmation.

To regenerate a specific book's cover (e.g. you don't like the first
output), pass --book-id:

    python scripts/backfill_covers.py --apply --book-id <id>

This deletes the existing PNG first, then re-runs generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly from scripts/ without installing
# the backend as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cover  # noqa: E402
from core import database as db  # noqa: E402

PER_COVER_COST_USD = 0.039


def _list_missing(force_book_id: str | None) -> list[tuple[str, str]]:
    """Return [(book_id, title), ...] for books that need a cover.

    If force_book_id is set, returns just that book (and the script will
    delete any existing cover first so the call regenerates)."""
    books = db.list_books()
    if force_book_id is not None:
        match = next((b for b in books if b["id"] == force_book_id), None)
        if not match:
            sys.exit(f"book_id not found: {force_book_id}")
        return [(match["id"], match.get("title") or "")]
    return [
        (b["id"], b.get("title") or "")
        for b in books
        if cover.get_cover_path(b["id"]) is None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually generate covers (default is dry-run).",
    )
    parser.add_argument(
        "--book-id", default=None,
        help="Operate on a single book (regenerates its cover even if one exists).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive cost confirmation.",
    )
    args = parser.parse_args()

    targets = _list_missing(args.book_id)
    if not targets:
        print("All books already have covers — nothing to do.")
        return

    print(f"{len(targets)} book(s) need a cover:")
    for book_id, title in targets:
        print(f"  {book_id}  {title}")
    est_cost = len(targets) * PER_COVER_COST_USD
    print(f"\nEstimated cost: ${est_cost:.3f} USD ({len(targets)} × ${PER_COVER_COST_USD})")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to actually generate.")
        return

    if not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    print()
    successes = 0
    for book_id, title in targets:
        # For regenerate-mode, blow away the existing PNG first so the
        # generator does not skip due to an in-place file.
        if args.book_id is not None:
            cover.delete_cover(book_id)
        print(f"[{book_id}] {title!r}...", flush=True)
        path = cover.generate_cover(book_id, title)
        if path:
            print(f"  -> {path.name} ({path.stat().st_size // 1024} KB)")
            successes += 1
        else:
            print("  -> failed (see logs for details)")

    print(f"\nDone: {successes}/{len(targets)} covers generated.")
    print(f"Actual cost: ~${successes * PER_COVER_COST_USD:.3f}")


if __name__ == "__main__":
    main()
