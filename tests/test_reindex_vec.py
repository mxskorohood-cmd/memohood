"""B11 regression: ``db.swap_vec_table`` must not use ``ALTER TABLE ...
RENAME`` on a sqlite-vec ``vec0`` virtual table.

sqlite-vec's ``vec0`` module does not implement ``xRename`` (SQLite's
virtual-table rename callback), so a bare ``ALTER TABLE captures_vec_v2
RENAME TO captures_vec`` leaves the renamed table in a broken state --
queries against it raise or misbehave. ``swap_vec_table`` now instead
recreates the live table under its permanent name and copies rows across
from the shadow table via ``INSERT ... SELECT`` inside one transaction (see
``db.py``'s docstring for the function).

These tests use the REAL ``sqlite_vec`` extension (installed in this
environment, 0.1.9) rather than mocking it -- the whole point of B11 is a
real vec0 quirk that a mock would hide.
"""

from __future__ import annotations

import sqlite3

import pytest


def _serialize(memohood, vec):
    """Pack a float vector exactly the way ``_engine/embed.py``'s
    ``reembed_captures_shadow`` does before inserting into a vec0 table."""
    return memohood._engine.embed.serialize_vector(vec)


def _insert_capture_row(conn, *, capture_id: str, content: str, session_id: str, created_at: float) -> None:
    conn.execute(
        """
        INSERT INTO captures (id, content, kind, session_id, tags, created_at, updated_at, valid_from, pinned)
        VALUES (?, ?, 'fact', ?, '', ?, ?, ?, 0)
        """,
        (capture_id, content, session_id, created_at, created_at, created_at),
    )
    conn.commit()


def test_swap_vec_table_promotes_shadow_and_is_queryable(memohood):
    """Core B11 regression. Pre-fix (``ALTER TABLE ... RENAME``), the live
    table is left corrupt after the swap and the KNN query below either
    raises or returns garbage; post-fix it must return exactly the shadow
    rows, and the shadow table must be gone."""
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    try:
        if not memohood.db.ensure_vec_table(conn, dims=3, shadow=False):
            pytest.skip("sqlite-vec extension not available in this environment")
        memohood.db.ensure_vec_table(conn, dims=3, shadow=True)

        live = memohood.db.vec_table_name(shadow=False)
        shadow = memohood.db.vec_table_name(shadow=True)

        # A stale row that only exists in the OLD live table -- swap must
        # fully replace live's contents with shadow's, not merge them.
        conn.execute(
            f"INSERT INTO {live}(capture_id, embedding) VALUES (?, ?)",
            ("stale-live-only", _serialize(memohood, [9.0, 9.0, 9.0])),
        )
        conn.commit()

        shadow_vecs = {
            "cap-1": [1.0, 0.0, 0.0],
            "cap-2": [0.0, 1.0, 0.0],
            "cap-3": [0.0, 0.0, 1.0],
        }
        for cid, vec in shadow_vecs.items():
            conn.execute(
                f"INSERT INTO {shadow}(capture_id, embedding) VALUES (?, ?)",
                (cid, _serialize(memohood, vec)),
            )
        conn.commit()

        memohood.db.swap_vec_table(conn, 3)

        # shadow table must be gone
        assert memohood.db.vec_table_exists(conn, shadow=True) is False

        # live table must still exist and be a genuinely working vec0 table
        assert memohood.db.vec_table_exists(conn, shadow=False) is True
        query_vec = _serialize(memohood, [1.0, 0.0, 0.0])
        rows = conn.execute(
            f"SELECT capture_id FROM {live} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_vec, 3),
        ).fetchall()
        ids = {r["capture_id"] for r in rows}
        assert ids == {"cap-1", "cap-2", "cap-3"}
        assert "stale-live-only" not in ids
    finally:
        conn.close()


def test_swap_vec_table_without_shadow_raises_dberror(memohood):
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    try:
        if not memohood.db.ensure_vec_table(conn, dims=3, shadow=False):
            pytest.skip("sqlite-vec extension not available in this environment")
        assert memohood.db.vec_table_exists(conn, shadow=True) is False

        with pytest.raises(memohood.db.DbError):
            memohood.db.swap_vec_table(conn, 3)
    finally:
        conn.close()


def test_reembed_captures_shadow_end_to_end_drops_stale_shadow(memohood, monkeypatch):
    """End-to-end (no network -- embedder monkeypatched to a fixed-vector
    fake): drives ``ensure_vec_table`` -> batched INSERTs -> the fixed
    ``swap_vec_table`` -> ``embed_signature`` update -> 'idle' state.

    Also covers the OTHER embed.py B11 fix: a shadow table left behind by a
    previous FAILED reindex (containing a row for a capture that no longer
    exists) must be dropped before the new reindex repopulates it, not
    reused via ``ensure_vec_table``'s ``CREATE ... IF NOT EXISTS`` -- which
    would otherwise leak that stale row into the freshly-promoted live
    table.
    """
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    try:
        if not memohood.db.ensure_vec_table(conn, dims=3, shadow=False):
            pytest.skip("sqlite-vec extension not available in this environment")

        # Simulate a previous FAILED reindex: a shadow table already exists,
        # holding a row for a capture that is no longer live.
        memohood.db.ensure_vec_table(conn, dims=3, shadow=True)
        shadow = memohood.db.vec_table_name(shadow=True)
        conn.execute(
            f"INSERT INTO {shadow}(capture_id, embedding) VALUES (?, ?)",
            ("stale-from-failed-run", _serialize(memohood, [5.0, 5.0, 5.0])),
        )
        conn.commit()

        now = memohood.db.now()
        _insert_capture_row(conn, capture_id="cap-1", content="hello world", session_id="s1", created_at=now)
        _insert_capture_row(conn, capture_id="cap-2", content="goodbye world", session_id="s1", created_at=now)

        def _fake_embed_texts(texts, cfg):
            return [[1.0, 0.0, 0.0] for _ in texts]

        monkeypatch.setattr(memohood._engine.embed, "embed_texts", _fake_embed_texts)

        new_cfg = {"embedder": {"provider": "fake", "model": "fake-model", "dims": 3}}
        result = memohood._engine.embed.reembed_captures_shadow(conn, new_cfg)

        assert result["status"] == "done"
        assert result["captures_embedded"] == 2
        assert result["vector_index_ready"] is True

        # shadow gone, live has exactly the two real captures -- no leak of
        # the stale row from the simulated failed run.
        assert memohood.db.vec_table_exists(conn, shadow=True) is False
        live = memohood.db.vec_table_name(shadow=False)
        rows = conn.execute(f"SELECT capture_id FROM {live}").fetchall()
        ids = {r["capture_id"] for r in rows}
        assert ids == {"cap-1", "cap-2"}
        assert "stale-from-failed-run" not in ids

        state = conn.execute("SELECT value FROM _meta WHERE key='migration_state'").fetchone()
        assert state["value"] == "idle"

        sigs = {r["embed_signature"] for r in conn.execute("SELECT embed_signature FROM captures").fetchall()}
        assert sigs == {"fake|fake-model|3"}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# B11 field case: resilience to an ALREADY-CORRUPT vec0 table.
#
# Users who ever ran the OLD (rename-based) reindex have a live ``captures_vec``
# whose internal vec0 shadow tables are out of sync -- a plain ``DROP TABLE``
# on it fails, so the NEW reindex (the recovery mechanism itself) could not run
# for them until ``safe_drop_vec_table`` learned to heal it.
# ---------------------------------------------------------------------------


def _corrupt_live_by_dropping_internal(conn):
    """Simulate the pre-B11 corruption cheaply and deterministically: knock
    out one of the live table's internal vec0 shadow tables. This reproduces
    the field breakage (plain ``DROP TABLE captures_vec`` then raises) without
    depending on the exact ALTER-RENAME internals."""
    conn.execute("DROP TABLE captures_vec_rowids")
    conn.commit()


def test_safe_drop_vec_table_recovers_corrupt_live(memohood):
    """A bare ``DROP TABLE`` on the corrupt live raises; ``safe_drop_vec_table``
    does not, removes the table, and leaves no orphaned internal shadows.

    The bare-DROP assertion runs on its own throwaway connection: a failed
    vec0 drop can poison the connection it ran on, and we must prove
    ``safe_drop_vec_table`` succeeds on a FRESH connection over the same
    on-disk corruption (exactly the operator's situation)."""
    hermes_home = str(memohood._hermes_home_for_test)
    db = memohood.db

    conn = db.get_connection(hermes_home=hermes_home)
    try:
        if not db.ensure_vec_table(conn, dims=3, shadow=False):
            pytest.skip("sqlite-vec extension not available in this environment")
        conn.execute(
            f"INSERT INTO {db.vec_table_name(shadow=False)}(capture_id, embedding) VALUES (?, ?)",
            ("cap-1", _serialize(memohood, [1.0, 0.0, 0.0])),
        )
        conn.commit()
        _corrupt_live_by_dropping_internal(conn)
    finally:
        conn.close()

    # The field breakage: a plain DROP now fails.
    conn = db.get_connection(hermes_home=hermes_home)
    try:
        with pytest.raises(sqlite3.Error):
            conn.execute("DROP TABLE captures_vec")
    finally:
        conn.close()

    # safe_drop_vec_table heals it, on a fresh connection.
    conn = db.get_connection(hermes_home=hermes_home)
    try:
        with conn:
            db.safe_drop_vec_table(conn, db.vec_table_name(shadow=False))
        assert db.vec_table_exists(conn, shadow=False) is False
        leftover = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'captures_vec%'"
        ).fetchall()
        assert leftover == []
    finally:
        conn.close()


def test_safe_drop_vec_table_is_idempotent_noop(memohood):
    """Calling it when nothing exists is a harmless no-op (never raises)."""
    hermes_home = str(memohood._hermes_home_for_test)
    db = memohood.db
    conn = db.get_connection(hermes_home=hermes_home)
    try:
        if not db.load_sqlite_vec(conn):
            pytest.skip("sqlite-vec extension not available in this environment")
        with conn:
            db.safe_drop_vec_table(conn, db.vec_table_name(shadow=False))
        assert db.vec_table_exists(conn, shadow=False) is False
    finally:
        conn.close()


def test_swap_vec_table_over_corrupt_live_recovers(memohood):
    """The full swap must succeed even when the OLD live is corrupt (the
    operator's real reindex): the corrupt live is dropped via
    ``safe_drop_vec_table``, the healthy shadow is promoted, and a KNN query
    over the new live works. The still-live shadow must survive the corrupt
    live's drop long enough to be copied from."""
    hermes_home = str(memohood._hermes_home_for_test)
    db = memohood.db
    conn = db.get_connection(hermes_home=hermes_home)
    try:
        if not db.ensure_vec_table(conn, dims=3, shadow=False):
            pytest.skip("sqlite-vec extension not available in this environment")

        # Healthy shadow holding the intended post-reindex rows.
        db.ensure_vec_table(conn, dims=3, shadow=True)
        shadow = db.vec_table_name(shadow=True)
        for cid, vec in {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}.items():
            conn.execute(
                f"INSERT INTO {shadow}(capture_id, embedding) VALUES (?, ?)",
                (cid, _serialize(memohood, vec)),
            )
        conn.commit()

        # Now corrupt the live (as if a pre-B11 reindex had run).
        _corrupt_live_by_dropping_internal(conn)

        db.swap_vec_table(conn, 3)

        assert db.vec_table_exists(conn, shadow=True) is False
        live = db.vec_table_name(shadow=False)
        rows = conn.execute(
            f"SELECT capture_id FROM {live} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_serialize(memohood, [1.0, 0.0, 0.0]), 2),
        ).fetchall()
        assert {r["capture_id"] for r in rows} == {"a", "b"}
    finally:
        conn.close()
