"""Generate a duotone wood-engraving cover image for a book via the
OpenRouter Gemini multimodal API.

The cover style is locked to a "gilt-stamped on dark leather binding"
aesthetic: warm amber lines and figures on a deep walnut brown ground,
in the tradition of Thomas Bewick white-line wood engravings and
Folio Society deluxe editions. The prompt was settled across four
probe iterations (see scripts/probe_gemini_image_v4.py for the final
exploration that this STYLE_LOCK is copied from).

Cost: ~$0.039 per cover at current OpenRouter pricing (1290 image tokens).
Latency: ~30 seconds per cover.

Soft-failure design: every error returns None and is logged. Never
raises out of the call site. Cover generation is decorative — a book
without a cover still works.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MODEL = "google/gemini-2.5-flash-image"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
COVERS_DIR = Path(__file__).parent.parent / "uploads" / "covers"
DEFAULT_TIMEOUT = 120.0

# Style lock — INVERTED duotone: dark walnut ground, amber lines.
# Copy of probe_gemini_image_v4.py's STYLE_LOCK; this is the prompt the
# user signed off on.
STYLE_LOCK = (
    "Style: a two-color hand-cut book cover plate in the tradition of "
    "Thomas Bewick WHITE-LINE wood engravings, gilt-stamped Victorian leather "
    "book bindings, and 19th-century relief prints where the carved marks "
    "appear as light lines on a dark ground. "
    "STRICTLY DUOTONE: deep walnut brown (RGB roughly #2a190d) as the BACKGROUND "
    "/ ground / paper, and warm amber (RGB roughly #d49457) as the LINES, "
    "marks, highlights, and figure. The dark walnut covers the whole plate; "
    "every visible line, shape, and detail is in glowing amber, as if carved "
    "into a dark plate or gilt-stamped onto dark leather. NO OTHER COLORS. "
    "No black, no white, no green, no blue, no red. "
    "Mark-making: bold confident carved amber lines on the dark ground, "
    "cross-hatching and parallel-line shading in amber, stippling for soft "
    "tonal areas. Flat values — no airbrush gradients, no photographic softness. "
    "High contrast. The amber should glow against the deep brown like firelight "
    "or gilt. "
    "Texture: subtle paper grain or ink-bleed irregularities as if printed by "
    "hand on dark aged book stock. "
    "Composition: a single iconic motif, centered, with generous dark negative "
    "space above and below the motif for future typography. Optional thin "
    "engraved amber border line around the edge of the plate. "
    "ABSOLUTELY NO text, letters, glyphs, runes, numerals, or signatures. "
    "NO human faces, NO recognizable people, NO photo-realism, NO painterly "
    "oil styles, NO modern digital illustration look. "
    "NO light-background-with-dark-lines (the inverse) — the background MUST "
    "be deep walnut brown."
)


def _build_prompt(title: str) -> str:
    """Build the per-book prompt. The motif is left to the model; only
    the title drives content, so the user does not need to hand-author
    a brief per book."""
    return (
        f"A book cover plate for the book titled '{title}'. "
        f"Choose a single iconic visual motif that evokes the spirit of "
        f"this title — let it emerge naturally from the words alone.\n\n"
        f"{STYLE_LOCK}"
    )


def get_cover_path(book_id: str) -> Optional[Path]:
    """Return the on-disk PNG path if a cover exists for this book."""
    path = COVERS_DIR / f"{book_id}.png"
    return path if path.is_file() else None


def generate_cover(
    book_id: str,
    title: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    client: Optional[httpx.Client] = None,
) -> Optional[Path]:
    """Generate and save a 2:3 portrait duotone cover for the book.

    Returns the saved PNG path on success, or None on any failure.
    Never raises. The `client` parameter exists for test injection.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("cover: OPENROUTER_API_KEY missing; skipping %s", book_id)
        return None

    if not title.strip():
        logger.warning("cover: empty title for %s; skipping", book_id)
        return None

    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COVERS_DIR / f"{book_id}.png"

    body = {
        "model": MODEL,
        "modalities": ["image", "text"],
        "messages": [{"role": "user", "content": _build_prompt(title)}],
        # OpenRouter's image_config extension; verified working in probe v2/v3/v4.
        "image_config": {"aspect_ratio": "2:3"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "SageSpace cover",
    }

    owned_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        try:
            resp = http.post(ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("cover: HTTP error for %s: %s", book_id, exc)
            return None

        if resp.status_code != 200:
            logger.warning(
                "cover: HTTP %s for %s: %s",
                resp.status_code, book_id, resp.text[:200],
            )
            return None

        try:
            payload = resp.json()
            images = payload["choices"][0]["message"].get("images") or []
            if not images:
                # Gemini sometimes returns text-only with no image. Treat as soft fail.
                logger.warning("cover: no image in response for %s", book_id)
                return None
            url = images[0]["image_url"]["url"]
            if not isinstance(url, str) or not url.startswith("data:image"):
                logger.warning("cover: unexpected image_url for %s", book_id)
                return None
            b64 = url.split(",", 1)[1]
            out_path.write_bytes(base64.b64decode(b64))
            logger.info(
                "cover: saved %s (%d bytes)",
                out_path.name, out_path.stat().st_size,
            )
            return out_path
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("cover: parse error for %s: %s", book_id, exc)
            return None
    finally:
        if owned_client:
            http.close()


def delete_cover(book_id: str) -> bool:
    """Remove the cover file for a book. Returns True if a file was
    removed, False if no file existed. Never raises."""
    path = COVERS_DIR / f"{book_id}.png"
    if path.is_file():
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("cover: failed to delete %s: %s", path, exc)
    return False
