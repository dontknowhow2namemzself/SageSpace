"""Recommendations API (memory-system-design.md §B).

  GET  /recommendations          -> current 'suggested' rows (lazy-compute if none)
  GET  /recommendations/saved        -> the "Want to read" list (status=added)
  POST /recommendations/refresh      -> Shuffle: current 'suggested' -> 'seen', recompute
  POST /recommendations/{id}/add     -> Want-to-read: status=added
  POST /recommendations/{id}/dismiss -> Dismiss: status=dismissed
  POST /recommendations/{id}/unsave  -> remove from Want-to-read: status=seen
  GET  /recommendations/stats        -> GROUP BY status (the eval surface)

All three actions are status transitions; rows are never deleted, so the eval
denominator (add-rate = added/total) stays intact. recompute is lazy (here) or
explicit (Shuffle) -- no cron, since this runs on localhost.
"""
import logging

from fastapi import APIRouter, HTTPException

from core import database as db
from core.recommend import recommend
from models.schemas import RecommendationResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/recommendations", response_model=list[RecommendationResponse])
def get_recommendations():
    """Current on-screen suggestions. Lazy compute: when nothing is pending,
    generate a fresh batch (cold start returns []). Cached otherwise."""
    rows = db.list_recommendations(status="suggested")
    if not rows:
        rows = recommend()
    return rows


@router.get("/recommendations/saved", response_model=list[RecommendationResponse])
def saved_recommendations():
    """The "Want to read" list = every rec the user added (status='added'),
    newest first. Shown as the home collapsible list."""
    return db.list_recommendations(status="added")


@router.post("/recommendations/refresh", response_model=list[RecommendationResponse])
def refresh_recommendations():
    """Shuffle (NEUTRAL): retire the current suggestions to 'seen' -- preserving
    the eval denominator -- then compute a fresh batch. recommend() excludes
    every already-recommended title (any status), so the just-shown ones won't
    return."""
    for row in db.list_recommendations(status="suggested"):
        db.set_recommendation_status(row["id"], "seen")
    return recommend()


@router.post("/recommendations/{rec_id}/add")
def add_recommendation(rec_id: str):
    """Want-to-read (POSITIVE)."""
    if not db.set_recommendation_status(rec_id, "added"):
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"id": rec_id, "status": "added"}


@router.post("/recommendations/{rec_id}/dismiss")
def dismiss_recommendation(rec_id: str):
    """Dismiss (NEGATIVE)."""
    if not db.set_recommendation_status(rec_id, "dismissed"):
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"id": rec_id, "status": "dismissed"}


@router.post("/recommendations/{rec_id}/unsave")
def unsave_recommendation(rec_id: str):
    """Remove from the Want-to-read list. Transitions 'added' -> 'seen'
    (neutral: off the list, not a negative signal like dismiss)."""
    if not db.set_recommendation_status(rec_id, "seen"):
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"id": rec_id, "status": "seen"}


@router.get("/recommendations/stats")
def get_recommendation_stats():
    """The MVP eval: counts GROUP BY status (add-rate = added / total)."""
    return db.recommendation_stats()
