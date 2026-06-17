"""Canonical text layer (ingest_version=2).

The book → section → block hierarchy is the source of truth for source
browsing and citation grounding. Chunks (in Chroma) and RAPTOR summary
nodes will reference blocks via stable block_ids, never the other way
around.

See docs/ARCHITECTURE.md §canonical-refactor for the design rationale.

Public surface:
- models.Block, models.Section
- ids.make_block_id, ids.make_section_id
- db.upsert_canonical_book / get_blocks / get_sections / ...
- normalize_pdf.normalize / normalize_epub.normalize  (added in later steps)
- ingest.ingest_canonical                              (added in later step)
"""
