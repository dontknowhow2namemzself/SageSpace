"""Shared slowapi rate limiter.

Lives in its own module (not main.py) so route modules can import it
for @limiter.limit decorators without a circular import through main.

Disabled by default: local dev and the offline test suite run without
limits. Production sets RATE_LIMIT_ENABLED=1 (see the future compose.yaml).
Limits key on the client IP — behind nginx this is only meaningful when
uvicorn runs with --proxy-headers so X-Forwarded-For is honored.
"""
import os

from limits import parse
from slowapi import Limiter
from slowapi.util import get_remote_address


def env_flag(name: str, default: bool = False) -> bool:
    """Read a strict boolean environment flag and fail on typos."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off)")


def rate_limit_from_env(name: str, default: str) -> str:
    """Read and validate a SlowAPI limit such as ``20/minute``."""
    value = (os.getenv(name) or default).strip()
    if not value:
        value = default
    try:
        parse(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} is not a valid rate limit: {value!r}") from exc
    return value


CHAT_RATE_LIMIT = rate_limit_from_env("CHAT_RATE_LIMIT", "20/minute")
INGEST_RATE_LIMIT = rate_limit_from_env("INGEST_RATE_LIMIT", "10/hour")

limiter = Limiter(
    key_func=get_remote_address,
    enabled=env_flag("RATE_LIMIT_ENABLED"),
)
