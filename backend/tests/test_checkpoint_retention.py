"""Per-thread checkpoint retention.

`prune_thread_checkpoints(session_id, keep_last=N)` is called from
finalize_node after every completed turn so a long-running session
cannot grow `sagespace_checkpoints.db` indefinitely. These tests cover
the prune mechanics (keep newest N, isolate other threads, also clean
the `writes` table) and the safety guarantees (no-op when DB is
absent, swallow all errors)."""
from __future__ import annotations

import sqlite3
import uuid

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import core.database as db_module
import core.graph.build as build_module


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def saver_and_conn(tmp_path, monkeypatch):
    """A real SqliteSaver bound to a temp DB that checkpoint_db_path()
    returns. get_chat_graph() is stubbed to return a graph whose
    checkpointer is this saver."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "app.db")
    cp_path = tmp_path / "sagespace_checkpoints.db"
    conn = sqlite3.connect(str(cp_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    class _StubGraph:
        checkpointer = saver

    monkeypatch.setattr(build_module, "get_chat_graph", lambda: _StubGraph())
    return saver, conn


def _seed(conn: sqlite3.Connection, thread_id: str, n: int):
    """Insert n synthetic checkpoint + writes rows for a thread.
    checkpoint_id is built so DESC order = newest-first, matching how
    LangGraph itself orders (uuid6 / time-prefixed)."""
    for i in range(n):
        cp_id = f"{i:010d}-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            " type, checkpoint, metadata) VALUES (?, '', ?, NULL, ?, ?, ?)",
            (thread_id, cp_id, "msgpack", b"", b"{}"),
        )
        conn.execute(
            "INSERT INTO writes "
            "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
            " channel, type, value) VALUES (?, '', ?, ?, ?, ?, ?, ?)",
            (thread_id, cp_id, "t", 0, "channel", "msgpack", b""),
        )
    conn.commit()


def _count(conn, table: str, thread_id: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]


# ── Mechanics ──────────────────────────────────────────────────────────────


def test_prune_keeps_only_the_most_recent_n(saver_and_conn):
    """Long thread → trimmed to the LAST N checkpoints by checkpoint_id
    order (most recent), with writes table cleaned in lockstep."""
    _, conn = saver_and_conn
    _seed(conn, "thread-A", n=50)

    from core.graph.build import prune_thread_checkpoints
    deleted = prune_thread_checkpoints("thread-A", keep_last=10)

    assert deleted == 40
    assert _count(conn, "checkpoints", "thread-A") == 10
    assert _count(conn, "writes", "thread-A") == 10

    # The 10 survivors are the LATEST (largest checkpoint_id values).
    kept = [
        r[0] for r in conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? "
            "ORDER BY checkpoint_id DESC",
            ("thread-A",),
        )
    ]
    assert kept[0].startswith("0000000049-")  # newest
    assert kept[-1].startswith("0000000040-")  # 10th-newest


def test_prune_short_thread_is_a_no_op(saver_and_conn):
    """A session that hasn't outgrown the window stays untouched."""
    _, conn = saver_and_conn
    _seed(conn, "thread-B", n=5)

    from core.graph.build import prune_thread_checkpoints
    assert prune_thread_checkpoints("thread-B", keep_last=30) == 0
    assert _count(conn, "checkpoints", "thread-B") == 5
    assert _count(conn, "writes", "thread-B") == 5


def test_prune_isolates_other_threads(saver_and_conn):
    """Pruning thread X must not touch thread Y — the retention cap is
    per-session, not per-DB."""
    _, conn = saver_and_conn
    _seed(conn, "thread-X", n=50)
    _seed(conn, "thread-Y", n=50)

    from core.graph.build import prune_thread_checkpoints
    prune_thread_checkpoints("thread-X", keep_last=10)

    assert _count(conn, "checkpoints", "thread-X") == 10
    assert _count(conn, "checkpoints", "thread-Y") == 50
    assert _count(conn, "writes", "thread-X") == 10
    assert _count(conn, "writes", "thread-Y") == 50


def test_prune_default_keep_last_is_30(saver_and_conn):
    _, conn = saver_and_conn
    _seed(conn, "thread-C", n=100)

    from core.graph.build import prune_thread_checkpoints
    assert prune_thread_checkpoints("thread-C") == 70
    assert _count(conn, "checkpoints", "thread-C") == 30


# ── Safety: no-op / never raise ────────────────────────────────────────────


def test_prune_no_op_when_checkpoint_db_missing(tmp_path, monkeypatch):
    """Fresh installation, no chat ever happened: checkpoint DB does
    not exist yet → return 0, never touch a missing file."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "app.db")
    assert not (tmp_path / "sagespace_checkpoints.db").exists()

    from core.graph.build import prune_thread_checkpoints
    assert prune_thread_checkpoints("any-session") == 0


def test_prune_swallows_saver_errors(tmp_path, monkeypatch):
    """If anything inside the saver layer explodes (corrupted DB,
    locked file, …), the prune logs and returns 0 — finalize_node's
    caller must never see the exception."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "app.db")
    (tmp_path / "sagespace_checkpoints.db").touch()  # make path exist

    class _BoomSaver:
        @property
        def conn(self):
            raise RuntimeError("simulated corruption")

    class _StubGraph:
        checkpointer = _BoomSaver()

    monkeypatch.setattr(build_module, "get_chat_graph", lambda: _StubGraph())

    from core.graph.build import prune_thread_checkpoints
    assert prune_thread_checkpoints("any-session") == 0


def test_prune_keep_last_zero_is_a_no_op(saver_and_conn):
    """Defensive: keep_last <= 0 must not delete EVERYTHING — return
    early instead. (Wired callers always pass a positive default, but
    a misconfigured override should not nuke history.)"""
    _, conn = saver_and_conn
    _seed(conn, "thread-D", n=10)

    from core.graph.build import prune_thread_checkpoints
    assert prune_thread_checkpoints("thread-D", keep_last=0) == 0
    assert _count(conn, "checkpoints", "thread-D") == 10
