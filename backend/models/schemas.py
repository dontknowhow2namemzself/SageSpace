from pydantic import BaseModel
from typing import Optional, List


class BookCreate(BaseModel):
    title: str
    author: Optional[str] = None


class BookResponse(BaseModel):
    id: str
    title: str
    author: Optional[str]
    total_chunks: Optional[int]
    total_chapters: Optional[int]
    upload_date: str
    raptor_status: str  # pending | building | ready | error:...
    digested_pct: float = 0.0
    cover_url: Optional[str] = None  # /api/books/{id}/cover when file exists


class ChatRequest(BaseModel):
    book_id: str
    session_id: str
    message: str


class ResumeRequest(BaseModel):
    """Resume a turn paused at a clarify interrupt (PR3, design §7.4).

    The turn is identified by session_id alone (thread_id = session_id; only
    one interrupt is pending per turn). `answer` is the user's reply to the
    clarifying question; an empty string means "proceed without clarifying"
    (the same safe default applied on expiry).
    """
    session_id: str
    answer: str = ""


class ExportRequest(BaseModel):
    session_id: str
    format: str  # "pdf" | "markdown"


class RecommendationResponse(BaseModel):
    """One row of the home "For you" block (memory-system-design.md §B)."""
    id: str
    title: str
    author: Optional[str] = None
    blurb: Optional[str] = None
    reason: Optional[str] = None
    which_interest: Optional[str] = None
    status: str
    created_at: str


class MemoryNoteResponse(BaseModel):
    """One captured user fact/interest, shown in the home "What I remember"
    panel (memory-system-design.md §A)."""
    id: str
    text: str
    type: str
    source_book_id: Optional[str] = None
    source_locator: Optional[str] = None
    created_at: str


class MemoryNoteUpdate(BaseModel):
    """Edit payload for a memory note (text required; type optional)."""
    text: str
    type: Optional[str] = None
