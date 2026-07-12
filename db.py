"""SQLite schema and connection management for memohood's dialogue memory.

Single database file: ``<hermes_home>/memory.db`` (WAL mode), living beside
hermes-core's own ``state.db`` in the SAME directory — DESIGN_v1.md: "State
under ``get_hermes_home()`` -> ``memory.db`` beside ``state.db``". This
module NEVER derives its own path from ``hermes_constants.get_hermes_home()``
by default the way a plain script might — every public function that needs
the db path takes an explicit ``hermes_home``/``db_path`` argument, because
``MemoryProvider.initialize(session_id, **kwargs)`` is handed ``hermes_home``
explicitly and per this project's non-negotiables that value (not a
hardcoded ``~/.hermes``) is what MUST be used, for correct behavior across
profiles/tests. ``get_hermes_home()`` is only used as a last-resort default
for ad-hoc/CLI callers -- ``provider.py`` is expected to ALWAYS pass the
``initialize()``-supplied ``hermes_home`` explicitly rather than relying on
this module's default.

Schema (verbatim from DESIGN_v1.md's "memory.db schema" section):

  * ``captures`` — the fact store. ``id`` is a TEXT primary key (capture ids
    are content-derived/UUID strings minted by ``capture.py`` -- NOT an
    autoincrement integer) so a capture's id is stable across a SUPERSEDE
    rewrite and can be embedded directly as ``captures_vec``'s primary key
    without an extra join table.
  * ``captures_fts`` — FTS5 over ``content``/``content_stem`` (RU-stemmed
    leg), keyed by the UNINDEXED ``capture_id`` column. This is NOT a true
    FTS5 external-content table (no ``content='captures'`` clause) because
    ``content_stem`` has no counterpart column in ``captures`` -- callers
    must keep the two in sync manually (insert into both in the same
    transaction; on invalidate, ``DELETE FROM captures_fts WHERE capture_id
    = ?``).
  * ``captures_vec`` — sqlite-vec ``vec0`` table, created LAZILY (dims come
    from the configured embedder, unknown until ``memory.memohood.embedder.dims``
    is read) via :func:`ensure_vec_table`. memohood has no "collections"
    concept, so there is exactly ONE global vec table (``captures_vec``).
    Its primary key column is ``capture_id TEXT PRIMARY KEY`` (sqlite-vec's
    vec0 supports an explicit non-rowid TEXT primary key, matching
    ``captures.id``'s own TEXT type).
  * ``messages_fts`` — FTS5 catch-up index over ``state.db`` messages
    (RU-stemmed), populated ONLY by :func:`catch_up_from_state` below. Not
    joined to any local ``messages`` table -- memohood does not copy hermes-core's
    message rows, only indexes their (message_id, session_id, role, content,
    timestamp) tuple redundantly inside the FTS row itself, so a recall hit
    can be displayed without a second read of ``state.db``.
  * ``signals`` / ``session_tags`` / ``session_links`` / ``spend`` / ``_meta``
    — as specified; see DDL below for exact columns.

Defensiveness contract:
  * Schema/DDL and connection setup raise :class:`DbError` with context on
    failure.
  * ``ensure_vec_table`` never raises for a missing/unloadable sqlite-vec
    extension -- only for a genuinely corrupt schema state after the
    extension loaded successfully. Its return value (bool) is the "is
    vector search ready" signal.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._engine import stem as stem_mod

logger = logging.getLogger("memohood.db")

# Bump this and add a migration step to `_MIGRATIONS` when the schema changes.
SCHEMA_VERSION = 1

DB_BUSY_TIMEOUT_MS = 5000
CONNECT_TIMEOUT_S = 5.0

# hermes_state.py's own sentinel for JSON-encoded (multimodal) message
# content -- see hermes_state.py's SessionDB._encode_content/_decode_content
# (verified 2026-07-06, hermes-agent v0.18.0, commit 09693cd3). Mirrored here
# (not imported -- catch_up_from_state opens state.db directly via a raw,
# read-only sqlite3 connection rather than going through
# ``hermes_state.SessionDB``, to avoid depending on that class's write-side
# locking/WAL-checkpoint machinery for what is a pure polling read) so a
# multimodal message (content = a list of {"type": "text"/"image_url", ...}
# parts) can still be decoded into indexable text instead of being skipped
# or indexed as a raw, unreadable JSON blob.
_CONTENT_JSON_PREFIX = "\x00json:"

# Batch size for one catch_up_from_state() pass over state.db's messages
# table. Chosen so a single pass is fast enough to run synchronously from
# initialize() without noticeably delaying agent startup on a large history,
# while still making visible incremental progress if interrupted --
# catch_up_from_state loops internally until fully caught up OR
# max_batches is exhausted, advancing (and persisting) the watermark after
# EVERY batch, never only at the end.
CATCH_UP_BATCH_SIZE = 500
CATCH_UP_DEFAULT_MAX_BATCHES = 200  # caps one initialize() call at ~100K messages


class DbError(RuntimeError):
    """Raised for memohood memory-database failures the caller must not ignore."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_memory_db_path(hermes_home: Optional[str | Path] = None) -> Path:
    """Return ``<hermes_home>/memory.db``.

    ``hermes_home`` should be the exact string/Path handed to
    ``MemoryProvider.initialize(session_id, **kwargs)`` via
    ``kwargs["hermes_home"]``. Falls back to
    ``hermes_constants.get_hermes_home()`` only for ad-hoc/CLI callers that
    genuinely have no session context -- ``provider.py``'s real lifecycle
    methods must always pass ``hermes_home`` explicitly, never rely on this
    fallback.
    """
    if hermes_home is not None:
        return Path(hermes_home) / "memory.db"
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "memory.db"


def get_state_db_path(hermes_home: Optional[str | Path] = None) -> Path:
    """Return ``<hermes_home>/state.db`` -- hermes-core's own session/message
    database, read-only source of truth for :func:`catch_up_from_state`."""
    if hermes_home is not None:
        return Path(hermes_home) / "state.db"
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state.db"


# ---------------------------------------------------------------------------
# Schema DDL — verbatim from DESIGN_v1.md's "memory.db schema" section
# ---------------------------------------------------------------------------

DDL_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS captures (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        notability TEXT NOT NULL DEFAULT 'medium',
        source TEXT NOT NULL DEFAULT 'EXTRACTED',
        pinned INTEGER NOT NULL DEFAULT 0,
        supersedes TEXT NOT NULL DEFAULT '',
        history TEXT NOT NULL DEFAULT '',
        session_id TEXT,
        message_id INTEGER,
        tags TEXT NOT NULL DEFAULT '',
        last_seen_at REAL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        valid_from REAL NOT NULL,
        invalidated_at REAL,
        embed_signature TEXT
    )
    """,
    # Not a true FTS5 external-content table -- see module docstring.
    # collection-free, single global corpus: no collection_id column.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts USING fts5(
        content,
        content_stem,
        capture_id UNINDEXED,
        tokenize='unicode61'
    )
    """,
    # Self-contained FTS5 index over state.db's messages table -- no local
    # `messages` table exists in memory.db at all (see module docstring).
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        content_stem,
        message_id UNINDEXED,
        session_id UNINDEXED,
        role UNINDEXED,
        timestamp UNINDEXED,
        tokenize='unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY,
        session_id TEXT,
        signal_type TEXT NOT NULL,
        score REAL,
        content TEXT,
        message_id INTEGER,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_tags (
        session_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        source TEXT,
        created_at REAL NOT NULL,
        PRIMARY KEY (session_id, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_links (
        id INTEGER PRIMARY KEY,
        from_session_id TEXT NOT NULL,
        to_session_id TEXT NOT NULL,
        relationship TEXT,
        label TEXT,
        weight REAL,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS spend (
        id INTEGER PRIMARY KEY,
        ts REAL NOT NULL,
        provider TEXT NOT NULL,
        op TEXT NOT NULL,
        units REAL,
        est_usd REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS _meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
]

# Additive, non-spec indexes for the query patterns capture.py/provider.py
# (next round) need -- these do not change any spec'd column/table shape.
INDEX_STATEMENTS: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_captures_session ON captures(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_captures_kind ON captures(kind)",
    "CREATE INDEX IF NOT EXISTS idx_captures_pinned ON captures(pinned)",
    "CREATE INDEX IF NOT EXISTS idx_captures_invalidated ON captures(invalidated_at)",
    "CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_links_from ON session_links(from_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_links_to ON session_links(to_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_spend_ts ON spend(ts)",
]


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply the project's mandatory PRAGMAs and row factory.

    WAL + busy_timeout=5000 + synchronous=NORMAL per DESIGN_v1.md's exact
    spec (readers do not block on a writer holding the WAL; writers wait up
    to 5s instead of raising "database is locked" immediately).
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row


def connect(db_path: Optional[Path] = None, *, hermes_home: Optional[str | Path] = None,
            read_only: bool = False) -> sqlite3.Connection:
    """Open a raw connection to ``memory.db``.

    Pass EITHER an explicit ``db_path`` OR ``hermes_home`` (resolved via
    :func:`get_memory_db_path`) -- if neither is given, falls back to
    ``hermes_constants.get_hermes_home()`` (ad-hoc/CLI callers only; see
    :func:`get_memory_db_path`'s docstring).

    ``read_only=True`` opens via a ``file:...?mode=ro`` URI (the same
    ``hermes_state.SessionDB(read_only=True)`` pattern documented in
    the plugin loader's API contract).

    Raises :class:`DbError` on any failure to open/configure — never lets a
    raw ``sqlite3.Error`` escape this module.
    """
    path = db_path or get_memory_db_path(hermes_home)

    if read_only:
        if not path.exists():
            raise DbError(
                f"memory.db not found at {path} (read-only open requested before "
                f"any write has created it)"
            )
        uri = f"file:{urllib.parse.quote(path.as_posix())}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=CONNECT_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise DbError(f"failed to open memory.db read-only at {path}: {exc}") from exc
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DbError(f"failed to create memory.db dir {path.parent}: {exc}") from exc
        try:
            conn = sqlite3.connect(str(path), timeout=CONNECT_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise DbError(f"failed to open memory.db at {path}: {exc}") from exc

    try:
        _configure_connection(conn)
    except sqlite3.Error as exc:
        conn.close()
        raise DbError(f"failed to configure memory.db connection: {exc}") from exc

    # Best-effort: try to load sqlite-vec on EVERY new connection -- it is a
    # runtime-loaded extension, not a persisted file-format feature, so a
    # later connection touching an EXISTING captures_vec table needs its own
    # load, not just the connection that created the table.
    load_sqlite_vec(conn)
    return conn


def get_connection(*, hermes_home: Optional[str | Path] = None, read_only: bool = False,
                    ensure_schema: bool = True, db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Convenience wrapper: :func:`connect` + (optionally) :func:`init_schema`.

    ``ensure_schema`` is ignored (forced off) when ``read_only=True`` -- a
    read-only connection cannot CREATE TABLE.
    """
    conn = connect(db_path=db_path, hermes_home=hermes_home, read_only=read_only)
    if ensure_schema and not read_only:
        init_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema init / migrations
# ---------------------------------------------------------------------------


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes/FTS if missing, then run any pending
    migrations. Idempotent -- safe to call on every plugin/provider
    ``initialize()``.
    """
    try:
        with conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)
            for stmt in INDEX_STATEMENTS:
                conn.execute(stmt)
        _ensure_meta_defaults(conn)
        migrate(conn)
    except sqlite3.Error as exc:
        raise DbError(f"failed to initialize memohood memory schema: {exc}") from exc


def _ensure_meta_defaults(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT INTO _meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
    watermark_row = conn.execute("SELECT value FROM _meta WHERE key = 'last_indexed_message_id'").fetchone()
    if watermark_row is None:
        with conn:
            conn.execute(
                "INSERT INTO _meta(key, value) VALUES ('last_indexed_message_id', '0')",
            )


def get_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


# Migration steps: (target_version, list_of_statements_or_callables).
# Callables receive the connection and run inside the same transaction.
# Empty for v1 -- this is the initial schema. Add entries here (never
# rewrite DDL_STATEMENTS/history) as the schema evolves.
_MIGRATIONS: List[tuple] = []


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations in ``_MIGRATIONS``, in order, updating
    ``_meta.schema_version`` after each. Never raises for "nothing to do".
    """
    current = get_schema_version(conn)
    for target_version, steps in _MIGRATIONS:
        if target_version <= current:
            continue
        try:
            with conn:
                for step in steps:
                    if callable(step):
                        step(conn)
                    else:
                        conn.execute(step)
                conn.execute(
                    "INSERT INTO _meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(target_version),),
                )
            current = target_version
        except sqlite3.Error as exc:
            raise DbError(f"migration to schema version {target_version} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# captures_vec (sqlite-vec) — one global table, optional dependency
# ---------------------------------------------------------------------------

_VEC_TABLE_LIVE = "captures_vec"
_VEC_TABLE_SHADOW = "captures_vec_v2"


def vec_table_name(*, shadow: bool = False) -> str:
    """Return ``"captures_vec"`` (or ``"captures_vec_v2"`` for the shadow-
    table embedding migration). memohood has exactly one global vec table,
    not one per collection, so there is nothing to interpolate and
    therefore no SQL-injection-via-identifier surface here at all."""
    return _VEC_TABLE_SHADOW if shadow else _VEC_TABLE_LIVE


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into *conn*. Returns False (never
    raises) if the ``sqlite_vec`` Python package is not installed or the
    native extension fails to load for any reason -- callers must treat
    that as "vector search unavailable, use FTS-only", not a fatal error.
    """
    try:
        import sqlite_vec  # optional dependency; heavy/native import kept local
    except ImportError:
        logger.warning("sqlite-vec package not installed; vector search disabled (FTS-only)")
        return False

    try:
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except (sqlite3.Error, AttributeError, OSError) as exc:
        logger.warning("sqlite-vec extension failed to load: %s", exc)
        return False
    return True


def ensure_vec_table(conn: sqlite3.Connection, dims: int, *, shadow: bool = False) -> bool:
    """Create the global ``captures_vec`` (or ``captures_vec_v2`` shadow)
    ``vec0`` virtual table if missing, keyed by ``capture_id TEXT PRIMARY
    KEY`` (sqlite-vec's vec0 supports an explicit non-rowid TEXT primary key
    for exactly this case).

    Returns True iff the table exists and is ready for use after this call.
    Returns False (never raises) when sqlite-vec cannot be loaded -- the
    caller falls back to FTS-only search. Raises :class:`DbError` only for a
    genuinely corrupt schema state (extension loaded fine, but CREATE
    VIRTUAL TABLE itself failed for a reason other than "already exists").
    """
    if not isinstance(dims, int) or dims <= 0:
        raise DbError(f"invalid dims for captures_vec table: {dims!r}")
    table = vec_table_name(shadow=shadow)

    if not load_sqlite_vec(conn):
        return False

    try:
        with conn:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                f"capture_id TEXT PRIMARY KEY, embedding FLOAT[{dims}])"
            )
    except sqlite3.Error as exc:
        raise DbError(f"failed to create vec table {table}: {exc}") from exc
    return True


def vec_table_exists(conn: sqlite3.Connection, *, shadow: bool = False) -> bool:
    table = vec_table_name(shadow=shadow)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def swap_vec_table(conn: sqlite3.Connection) -> None:
    """Atomically promote the shadow table (``captures_vec_v2``) to live
    (``captures_vec``), for the embedding-migration flow in
    ``_engine/embed.py``'s ``reembed_captures_shadow``. Caller is
    responsible for having fully re-embedded into the shadow table BEFORE
    calling this -- this function only does the rename, inside one
    transaction so readers never observe a half-swapped state.
    """
    live = vec_table_name(shadow=False)
    shadow = vec_table_name(shadow=True)
    if not vec_table_exists(conn, shadow=True):
        raise DbError(f"cannot swap: shadow table {shadow} does not exist")
    try:
        with conn:
            if vec_table_exists(conn, shadow=False):
                conn.execute(f"DROP TABLE {live}")
            conn.execute(f"ALTER TABLE {shadow} RENAME TO {live}")
    except sqlite3.Error as exc:
        raise DbError(f"failed to swap captures_vec tables: {exc}") from exc


# ---------------------------------------------------------------------------
# Spend ledger CRUD (backs `_engine/ledger.py`)
# ---------------------------------------------------------------------------


def now() -> float:
    """Unix timestamp helper — single spot so every writer uses the same
    clock source (makes it trivial to monkeypatch in tests)."""
    return time.time()


def record_spend(conn: sqlite3.Connection, *, provider: str, op: str, units: Optional[float] = None,
                  est_usd: Optional[float] = None) -> int:
    """Append one row to memohood's external-spend ledger. Cloudflare/Cohere/
    Gemini calls made by THIS plugin are invisible to hermes' own
    token/cost accounting, so this ledger is the only place their dollar cost is
    recorded and capped."""
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO spend(ts, provider, op, units, est_usd) VALUES (?, ?, ?, ?, ?)",
                (now(), provider, op, units, est_usd),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DbError(f"failed to record spend ({provider}/{op}): {exc}") from exc


def monthly_spend(conn: sqlite3.Connection, provider: str, *, since_ts: Optional[float] = None) -> float:
    """Sum ``est_usd`` for a provider since ``since_ts`` (defaults to 30
    days ago) -- used to enforce ``memory.memohood.monthly_ceiling_usd.<provider>``
    before starting a costly job."""
    cutoff = since_ts if since_ts is not None else now() - 30 * 24 * 3600
    row = conn.execute(
        "SELECT COALESCE(SUM(est_usd), 0) AS total FROM spend WHERE provider = ? AND ts >= ?",
        (provider, cutoff),
    ).fetchone()
    return float(row["total"]) if row is not None else 0.0


# ---------------------------------------------------------------------------
# catch_up_from_state — incremental, watermarked backfill of state.db
# messages into messages_fts (DESIGN_v1.md's non-negotiable: "источник
# истины -- state.db самого hermes, плагин его только читает... водяной
# знак (last_indexed_message_id): прервалась -- продолжится с того же
# места, без дыр и дублей")
# ---------------------------------------------------------------------------


def _decode_message_content(raw: Any) -> str:
    """Reverse hermes_state.SessionDB._encode_content's sentinel scheme
    (``"\\x00json:" + json.dumps(...)``) for a single message's ``content``
    column, and flatten it to plain indexable text.

    Mirrors (does not import) hermes_state.py's private encode/decode pair
    (verified 2026-07-06, hermes-agent v0.18.0) -- catch_up_from_state opens
    state.db via a bare read-only sqlite3 connection rather than
    ``hermes_state.SessionDB`` (see module docstring), so there is no
    ``SessionDB`` instance around to call the real ``_decode_content`` on.

    A scalar (plain string) content is returned unchanged. A JSON-encoded
    multimodal content (list of ``{"type": "text"/"image_url", ...}``
    parts, or a dict) has its ``"text"``-typed parts joined with newlines;
    non-text parts (e.g. images) contribute nothing indexable and are
    silently skipped. Never raises -- any decode failure falls back to the
    raw string as-is (still better than dropping the message).
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return str(raw)
    if not raw.startswith(_CONTENT_JSON_PREFIX):
        return raw

    try:
        parsed = json.loads(raw[len(_CONTENT_JSON_PREFIX):])
    except (json.JSONDecodeError, TypeError):
        return raw

    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        parts = []
        for part in parsed:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    if isinstance(parsed, dict):
        text = parsed.get("text")
        return str(text) if text else ""
    return raw


def _open_state_db_readonly(state_db_path: Path) -> sqlite3.Connection:
    """Open ``state.db`` via a read-only ``file:...?mode=ro`` URI. Raises
    :class:`DbError` if the file does not exist or cannot be opened --
    callers (``catch_up_from_state``) treat "no state.db yet" (a genuinely
    fresh hermes install with no history at all) as a normal, non-fatal
    condition and catch this themselves."""
    if not state_db_path.exists():
        raise DbError(f"state.db not found at {state_db_path}")
    uri = f"file:{urllib.parse.quote(state_db_path.as_posix())}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=CONNECT_TIMEOUT_S)
    except sqlite3.Error as exc:
        raise DbError(f"failed to open state.db read-only at {state_db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def _get_watermark(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'last_indexed_message_id'").fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _set_watermark(conn: sqlite3.Connection, message_id: int) -> None:
    """Write the watermark. Caller is expected to already be inside a
    ``with conn:`` transaction block (see ``catch_up_from_state``'s loop) so
    the watermark update commits atomically together with the
    ``messages_fts`` inserts for the same batch — this function does NOT
    open its own transaction, to avoid a redundant nested commit."""
    conn.execute(
        "INSERT INTO _meta(key, value) VALUES ('last_indexed_message_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(message_id),),
    )


def catch_up_from_state(
    conn: sqlite3.Connection,
    hermes_home: Optional[str | Path] = None,
    *,
    state_db_path: Optional[Path] = None,
    batch_size: int = CATCH_UP_BATCH_SIZE,
    max_batches: int = CATCH_UP_DEFAULT_MAX_BATCHES,
) -> Dict[str, Any]:
    """Incrementally index ``state.db``'s ``messages`` table into
    ``messages_fts`` (RU-stemmed), resuming from
    ``_meta.last_indexed_message_id``.

    *conn* is an already-open, already-schema'd connection to THIS plugin's
    ``memory.db`` (the write target). *state_db_path* (or, if omitted,
    ``<hermes_home>/state.db``) is hermes-core's own database, opened
    READ-ONLY here and never written to -- memohood's memory-provider design
    non-negotiable is that state.db is the sole source of truth and this
    plugin only ever reads it (DESIGN_v1.md: "плагин его только читает,
    поэтому потерять историю невозможно в принципе").

    Idempotent and resumable: processes messages in ``id`` order, strictly
    greater than the current watermark, ``batch_size`` at a time, advancing
    (and persisting) the watermark after EVERY batch -- an interruption
    (crash, process kill) mid-run loses at most one partial batch's worth of
    indexing progress, never re-indexes already-caught-up messages, and
    never leaves a gap. Stops after ``max_batches`` batches in one call
    (default caps a single call at ~100K messages) so a very large first
    backfill can't block ``initialize()`` indefinitely -- the NEXT call
    (e.g. the following turn's ``initialize()``, or a future scheduled
    call) picks up exactly where this one left off, per the watermark.

    Only rows with ``active = 1`` (hermes-core's soft-delete flag) are
    indexed -- a rewound/superseded message should not surface in recall.
    Messages whose decoded content is empty/whitespace-only (e.g. a
    tool-call-only assistant turn with no text) are skipped but still
    advance the watermark (their id is still "seen").

    Returns a stats dict: ``{"indexed": int, "skipped_empty": int,
    "batches": int, "watermark_before": int, "watermark_after": int,
    "state_db_found": bool}``. If ``state.db`` does not exist yet (a
    genuinely fresh hermes install with zero history), returns immediately
    with ``state_db_found=False`` and all counters at 0 -- this is a normal
    condition, not an error, and callers must not treat it as one.
    """
    resolved_state_db_path = state_db_path or get_state_db_path(hermes_home)
    watermark_before = _get_watermark(conn)
    stats: Dict[str, Any] = {
        "indexed": 0,
        "skipped_empty": 0,
        "batches": 0,
        "watermark_before": watermark_before,
        "watermark_after": watermark_before,
        "state_db_found": False,
    }

    try:
        state_conn = _open_state_db_readonly(resolved_state_db_path)
    except DbError:
        logger.info(
            "catch_up_from_state: no state.db at %s yet (fresh hermes install); nothing to index",
            resolved_state_db_path,
        )
        return stats

    stats["state_db_found"] = True
    watermark = watermark_before
    try:
        for _ in range(max_batches):
            rows = state_conn.execute(
                """
                SELECT id, session_id, role, content, timestamp
                  FROM messages
                 WHERE id > ? AND active = 1
                 ORDER BY id
                 LIMIT ?
                """,
                (watermark, batch_size),
            ).fetchall()
            if not rows:
                break

            to_insert = []
            for row in rows:
                text = _decode_message_content(row["content"]).strip()
                if text:
                    to_insert.append(
                        (text, stem_mod.stem_ru(text), row["id"], row["session_id"], row["role"], row["timestamp"])
                    )
                else:
                    stats["skipped_empty"] += 1

            watermark = rows[-1]["id"]
            with conn:
                if to_insert:
                    conn.executemany(
                        "INSERT INTO messages_fts(content, content_stem, message_id, session_id, role, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        to_insert,
                    )
                _set_watermark(conn, watermark)

            stats["indexed"] += len(to_insert)
            stats["batches"] += 1

            if len(rows) < batch_size:
                break  # fully caught up
    finally:
        state_conn.close()

    stats["watermark_after"] = watermark
    return stats
