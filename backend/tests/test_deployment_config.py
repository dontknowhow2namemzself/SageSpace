"""Offline tests for deployment-facing environment configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.config import cors_origins_from_env, parse_cors_origins
from core.paths import ensure_data_directories, resolve_data_dir
from core.ratelimit import env_flag, rate_limit_from_env


def test_resolve_data_dir_preserves_default_and_resolves_custom_path(tmp_path):
    assert resolve_data_dir(None) == Path(__file__).parents[1].resolve()
    assert resolve_data_dir(tmp_path / "data") == (tmp_path / "data").resolve()


def test_ensure_data_directories_creates_complete_layout(tmp_path):
    data_dir = tmp_path / "nested" / "data"

    created = ensure_data_directories(data_dir)

    assert data_dir.is_dir()
    assert set(created) == {
        data_dir / "chroma_db",
        data_dir / "uploads",
        data_dir / "uploads" / "covers",
        data_dir / "exports",
    }
    assert all(path.is_dir() for path in created)


def test_ensure_data_directories_rejects_file_as_data_root(tmp_path):
    data_file = tmp_path / "not-a-directory"
    data_file.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        ensure_data_directories(data_file)


def test_parse_cors_origins_trims_and_discards_empty_entries():
    assert parse_cors_origins(
        " https://one.example, ,https://two.example,,https://one.example "
    ) == ["https://one.example", "https://two.example"]


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "https://example.com/path",
        "https://user:password@example.com",
        "ftp://example.com",
        "https://example.com:invalid",
    ],
)
def test_parse_cors_origins_rejects_non_origins_and_wildcard(value):
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        parse_cors_origins(value)


def test_cors_origins_from_env_uses_custom_allowlist(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS", "https://one.example, https://two.example"
    )
    assert cors_origins_from_env() == [
        "https://one.example",
        "https://two.example",
    ]


def test_cors_origins_from_env_defaults_to_local_development(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert cors_origins_from_env() == [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_env_flag_accepts_explicit_true_values(monkeypatch, value):
    monkeypatch.setenv("DEPLOYMENT_TEST_FLAG", value)
    assert env_flag("DEPLOYMENT_TEST_FLAG") is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_env_flag_accepts_explicit_false_values(monkeypatch, value):
    monkeypatch.setenv("DEPLOYMENT_TEST_FLAG", value)
    assert env_flag("DEPLOYMENT_TEST_FLAG", default=True) is False


def test_env_flag_rejects_typo(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_TEST_FLAG", "treu")
    with pytest.raises(RuntimeError, match="must be a boolean"):
        env_flag("DEPLOYMENT_TEST_FLAG")


def test_rate_limit_from_env_accepts_custom_value(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_TEST_LIMIT", "7/minute")
    assert rate_limit_from_env("DEPLOYMENT_TEST_LIMIT", "20/minute") == "7/minute"


def test_rate_limit_from_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_TEST_LIMIT", "often-ish")
    with pytest.raises(RuntimeError, match="not a valid rate limit"):
        rate_limit_from_env("DEPLOYMENT_TEST_LIMIT", "20/minute")


def test_custom_rate_limits_are_registered_on_routes(tmp_path):
    probe = """
import json
from fastapi.testclient import TestClient
from api import chat, ingest
from core.ratelimit import limiter
from main import app

with TestClient(app) as client:
    enabled_statuses = [
        client.post(
            "/api/ingest",
            files={"file": ("bad.txt", b"x", "text/plain")},
        ).status_code
        for _ in range(2)
    ]
limiter.reset()
limiter.enabled = False
with TestClient(app) as client:
    disabled_statuses = [
        client.post(
            "/api/ingest",
            files={"file": ("bad.txt", b"x", "text/plain")},
        ).status_code
        for _ in range(2)
    ]

print(json.dumps({
    "enabled": limiter.enabled,
    "enabled_statuses": enabled_statuses,
    "disabled_statuses": disabled_statuses,
    "routes": {
        name: [str(rule.limit) for rule in rules]
        for name, rules in limiter._route_limits.items()
    },
}))
"""
    env = os.environ.copy()
    env.update({
        "RATE_LIMIT_ENABLED": "1",
        "CHAT_RATE_LIMIT": "7/minute",
        "INGEST_RATE_LIMIT": "1/hour",
        "SAGESPACE_DATA_DIR": str(tmp_path / "data"),
    })
    backend_dir = Path(__file__).parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(backend_dir), env.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.splitlines()[-1])

    # The probe disables the limiter after first proving the enabled path.
    assert payload["enabled"] is False
    assert payload["enabled_statuses"] == [400, 429]
    assert payload["disabled_statuses"] == [400, 400]
    assert payload["routes"]["api.chat.chat"] == ["7 per 1 minute"]
    assert payload["routes"]["api.chat.chat_resume"] == ["7 per 1 minute"]
    assert payload["routes"]["api.ingest.ingest_book"] == ["1 per 1 hour"]
