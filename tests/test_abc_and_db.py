"""ABC conformance + db.py schema/catch-up tests."""

from __future__ import annotations

import sqlite3
import time

import pytest


def test_implements_real_memory_provider_abc(memohood):
    """MemoHoodMemoryProvider must satisfy the REAL agent.memory_provider.MemoryProvider
    ABC (v0.18.0) -- instantiates without TypeError, and isinstance holds
    against the actual imported class (not a reimplementation)."""
    from agent.memory_provider import MemoryProvider as RealMemoryProvider

    instance = memohood.provider.MemoHoodMemoryProvider()
    assert isinstance(instance, RealMemoryProvider)
    assert instance.name == "memohood"
    assert instance.is_available() is True


def test_all_abstract_methods_present(memohood):
    from agent.memory_provider import MemoryProvider as RealMemoryProvider

    abstract_names = RealMemoryProvider.__abstractmethods__
    instance = memohood.provider.MemoHoodMemoryProvider()
    for attr_name in abstract_names:
        if attr_name == "name":
            assert isinstance(instance.name, str) and instance.name  # abstract property, not a method
            continue
        assert callable(getattr(instance, attr_name, None)), f"missing abstract method {attr_name}"


def test_initialize_creates_all_tables(memohood):
    hermes_home = memohood._hermes_home_for_test
    conn = memohood.db.get_connection(hermes_home=str(hermes_home))
    try:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            ).fetchall()
        }
        for expected in (
            "captures", "captures_fts", "messages_fts", "signals",
            "session_tags", "session_links", "spend", "_meta",
        ):
            assert expected in tables, f"missing table {expected}"
    finally:
        conn.close()
    assert (hermes_home / "memory.db").exists()


def test_init_schema_idempotent(memohood):
    hermes_home = str(memohood._hermes_home_for_test)
    conn1 = memohood.db.get_connection(hermes_home=hermes_home)
    conn1.close()
    # Re-opening + re-initializing schema must not raise or duplicate anything.
    conn2 = memohood.db.get_connection(hermes_home=hermes_home)
    try:
        memohood.db.init_schema(conn2)  # explicit second call, same connection
        n = conn2.execute("SELECT COUNT(*) AS n FROM _meta WHERE key='schema_version'").fetchone()["n"]
        assert n == 1
    finally:
        conn2.close()


def _make_fake_state_db(path, rows):
    """rows: list of (id, session_id, role, content, timestamp, active)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
        "content TEXT, timestamp REAL, active INTEGER)"
    )
    conn.executemany(
        "INSERT INTO messages(id, session_id, role, content, timestamp, active) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_catch_up_from_state_idempotent_and_resumable(memohood):
    hermes_home = memohood._hermes_home_for_test
    state_db_path = hermes_home / "state.db"
    now = time.time()
    _make_fake_state_db(state_db_path, [
        (1, "s1", "user", "привет, помоги мне с договором", now, 1),
        (2, "s1", "assistant", "конечно, помогу", now, 1),
        (3, "s1", "user", "", now, 1),  # empty content -> skipped but watermark advances
        (4, "s1", "user", "неактивное сообщение", now, 0),  # inactive -> not indexed
    ])
    conn = memohood.db.get_connection(hermes_home=str(hermes_home))
    try:
        stats1 = memohood.db.catch_up_from_state(conn, str(hermes_home))
        assert stats1["state_db_found"] is True
        assert stats1["indexed"] == 2  # rows 1 and 2 (row 3 empty, row 4 inactive)
        assert stats1["skipped_empty"] == 1
        # Row 4 has active=0 -- the SQL WHERE clause excludes it from the
        # SELECT entirely, so it is never "seen" and cannot advance the
        # watermark; the watermark tracks the highest ACTIVE row id only
        # (verified against db.py's actual query, not the docstring alone).
        assert stats1["watermark_after"] == 3

        # Second call: nothing new -> idempotent, no re-indexing.
        stats2 = memohood.db.catch_up_from_state(conn, str(hermes_home))
        assert stats2["indexed"] == 0
        assert stats2["watermark_before"] == 3
        assert stats2["watermark_after"] == 3

        rows = conn.execute("SELECT COUNT(*) AS n FROM messages_fts").fetchall()
        assert rows[0]["n"] == 2
    finally:
        conn.close()


def test_catch_up_from_state_no_state_db_is_not_an_error(memohood):
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    try:
        stats = memohood.db.catch_up_from_state(conn, hermes_home)
        assert stats["state_db_found"] is False
        assert stats["indexed"] == 0
    finally:
        conn.close()
