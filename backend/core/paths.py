"""Single source of truth for application state written to disk.

The main SQLite DB (and the LangGraph checkpoint DB derived from it), the
Chroma vector index, uploaded book files, generated covers, and exported
notes all resolve under ``DATA_DIR``.

Default is the backend source directory, preserving the historical
local-dev layout (backend/sagespace.db, backend/chroma_db/, ...).
Production points SAGESPACE_DATA_DIR at a mounted volume (e.g. /data)
so all state survives container replacement with a single mount.
"""
import os
from pathlib import Path


_BACKEND_DIR = Path(__file__).parent.parent
_RUNTIME_SUBDIRECTORIES = ("chroma_db", "uploads", "uploads/covers", "exports")


def resolve_data_dir(configured: str | os.PathLike[str] | None) -> Path:
    """Resolve a configured data root, preserving the local-dev default."""
    return Path(configured or _BACKEND_DIR).expanduser().resolve()


DATA_DIR = resolve_data_dir(os.getenv("SAGESPACE_DATA_DIR"))


def ensure_data_directories(data_dir: Path = DATA_DIR) -> tuple[Path, ...]:
    """Create and validate the writable directory layout at app startup.

    Errors intentionally propagate: starting without a usable persistent data
    root is safer than appearing healthy and failing on the first write.
    """
    if data_dir.exists() and not data_dir.is_dir():
        raise NotADirectoryError(f"SageSpace data path is not a directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    directories = tuple(data_dir / relative for relative in _RUNTIME_SUBDIRECTORIES)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories
