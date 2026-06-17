from dotenv import load_dotenv
load_dotenv()

import logging

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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api import books, ingest, chat, progress, export, debug, canonical, recommend, memory
from core.database import init_db

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SageSpace API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
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
