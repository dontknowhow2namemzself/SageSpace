from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from models.schemas import ExportRequest
from core import database as db
from core.tools import run_export

router = APIRouter()
EXPORT_DIR = Path(__file__).parent.parent / "exports"


@router.post("/export")
def export_notes(req: ExportRequest):
    session = db.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.format not in ("pdf", "markdown"):
        raise HTTPException(status_code=400, detail="Format must be 'pdf' or 'markdown'")

    result = run_export(
        book_id=session["book_id"], session_id=req.session_id, format=req.format,
    )
    if not result.get("available"):
        raise HTTPException(status_code=500, detail="Export failed")

    full_path = Path(__file__).parent.parent / result["path"].lstrip("/")
    if not full_path.exists():
        raise HTTPException(status_code=500, detail="Export failed")

    media_type = "application/pdf" if req.format == "pdf" else "text/markdown"
    return FileResponse(str(full_path), filename=full_path.name, media_type=media_type)
