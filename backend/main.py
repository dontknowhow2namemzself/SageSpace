from dotenv import load_dotenv
load_dotenv()

import logging
import os

# Privacy-first defaults must be set before importing the route modules, which
# load ChromaDB/ONNX Runtime transitively. Operators can still opt in by
# explicitly setting either variable before process startup.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

# Replace the default handler with a JSON formatter + request-context
# filter, configured BEFORE any other module logs anything so even
# import-time messages get the structured shape.
from core.observability import RequestIDMiddleware, configure_logging
configure_logging()

# Silence ChromaDB's posthog telemetry error spam. chromadb 0.5.20 tries to
# send analytics through a posthog whose capture() signature it mismatches,
# logging "Failed to send telemetry event ...: capture() takes 1 positional
# argument but 3 were given" on every Chroma op. anonymized_telemetry=False
# does NOT suppress it in this version, so we quiet the module logger directly.
# Retrieval is entirely unaffected — this is analytics noise only.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api import books, ingest, chat, progress, export, debug, canonical, recommend, memory
from core.config import cors_origins_from_env
from core.database import init_db
from core.paths import ensure_data_directories
from core.ratelimit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_directories()
    init_db()
    yield


app = FastAPI(title="SageSpace API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestIDMiddleware)

# Comma-separated allowlist; defaults cover local dev. In production the
# frontend is served same-origin behind nginx, so CORS never fires — the
# env override exists for any split-origin setup (e.g. Vercel frontend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_from_env(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(books.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(progress.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(debug.router, prefix="/api")
app.include_router(canonical.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(memory.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
