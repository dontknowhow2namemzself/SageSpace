"""Lightweight environment parsing shared by application startup and tests."""
import os
from urllib.parse import urlsplit


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://localhost:3001,"
    "http://127.0.0.1:3000,http://127.0.0.1:3001"
)


def parse_cors_origins(value: str) -> list[str]:
    """Return unique exact HTTP(S) origins, rejecting unsafe ambiguity."""
    origins: list[str] = []
    for item in value.split(","):
        origin = item.strip()
        if not origin:
            continue
        try:
            parsed = urlsplit(origin)
            parsed.port  # Validate malformed/non-numeric ports.
        except ValueError as exc:
            raise RuntimeError(f"CORS_ORIGINS contains an invalid origin: {origin!r}") from exc
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "CORS_ORIGINS entries must be exact http(s) origins without "
                f"credentials, paths, queries, fragments, or wildcards: {origin!r}"
            )
        if origin not in origins:
            origins.append(origin)
    return origins


def cors_origins_from_env() -> list[str]:
    """Return the configured allowlist or the local-development defaults."""
    return parse_cors_origins(os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS))
