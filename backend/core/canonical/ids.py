"""Stable identifier generation for canonical blocks / sections.

Design rule: IDs MUST depend only on (book_id, order_idx). They MUST NOT
depend on block text, normalization output, page numbers, or chapter labels
- those evolve as the normalizer improves, but block_id must stay stable so
that already-stored citations keep resolving across re-ingests.

Format: 'blk_<8 hex chars>' / 'sec_<8 hex chars>' / 'rap_<level>_<8 hex chars>'.
Hex is short enough to be eyeballable in logs and long enough to avoid
collisions within a single book (~4e9 namespace per book).
"""
from __future__ import annotations

import hashlib


def _short_hex(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:8]


def make_block_id(book_id: str, order_idx: int) -> str:
    return f"blk_{_short_hex(book_id, str(order_idx))}"


def make_section_id(book_id: str, order_idx: int) -> str:
    return f"sec_{_short_hex(book_id, 'section', str(order_idx))}"
