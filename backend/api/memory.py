"""Memory-notes API (memory-system-design.md §A).

The notes the fast lane captures are normally invisible "fuel" for
recommendations, written silently. But because they DO shape what the app
suggests, the user gets an honest, always-available way to see and correct
them (the home "What I remember" panel):

  GET    /memory-notes        -> all captured notes, newest first
  PATCH  /memory-notes/{id}    -> edit a note's text (and optionally its type)
  DELETE /memory-notes/{id}    -> forget a note

Writes stay silent (no panel needed to capture); this surface is read/correct
only -- keeping the "low-interruption, quietly delightful" philosophy intact.
"""
from fastapi import APIRouter, HTTPException

from core import database as db
from models.schemas import MemoryNoteResponse, MemoryNoteUpdate


router = APIRouter()


@router.get("/memory-notes", response_model=list[MemoryNoteResponse])
def list_memory_notes():
    return db.list_memory_notes()


@router.patch("/memory-notes/{note_id}", response_model=MemoryNoteResponse)
def update_memory_note(note_id: str, payload: MemoryNoteUpdate):
    if not db.update_memory_note(note_id, payload.text, payload.type):
        # missing id OR empty text -> 404 / 400 respectively
        if db.get_memory_note(note_id) is None:
            raise HTTPException(status_code=404, detail="Memory note not found")
        raise HTTPException(status_code=400, detail="Note text cannot be empty")
    return db.get_memory_note(note_id)


@router.delete("/memory-notes/{note_id}")
def delete_memory_note(note_id: str):
    if not db.delete_memory_note(note_id):
        raise HTTPException(status_code=404, detail="Memory note not found")
    return {"deleted": note_id}
