"""Nightly consolidation for memohood's captures corpus (DESIGN_v1.md
"Consolidation (consolidate.py) -- via hermes cron (nightly), Gemini
flash-lite").

Intended to be invoked once a day from a ``hermes cron`` job (not wired up
by this module itself -- ``cli.py``/the operator's own cron config calls
:func:`run_nightly`). Four independent stages, each caught separately so
one stage's failure never blocks the others:

  1. :func:`run_decay` -- Ebbinghaus decay: ``confidence * exp(-age_days /
     halflife)`` per kind (``memory.memohood.decay.halflife_days``, defaults
     event=7, preference/decision/correction=90,
     fact/persona/instruction/summary=365). ``pinned``
     captures are skipped entirely (decay-exempt). Age is measured from
     ``last_seen_at`` (reinforced on every
     recall hit by ``provider.py``'s prefetch, not just at write time), so
     an old-but-frequently-recalled capture survives longer than an old,
     never-revisited one. A capture whose decayed confidence falls below
     ``memory.memohood.decay.floor`` (default 0.05) is ARCHIVED, not deleted:
     ``invalidated_at`` is set (the same column ``_engine/retrieve.py``
     already excludes on) and ``tags`` gains an ``archived_decay`` marker
     so a future audit can tell "archived by decay" apart from "superseded
     by a newer fact" -- both are excluded from retrieval identically via
     ``invalidated_at IS NOT NULL``, but the tag preserves *why*.
  2. :func:`run_dedup` -- merges near-duplicate ACTIVE captures. Reuses
     each capture's ALREADY-STORED vector (a vec0 self-KNN: "find captures
     near capture X's own embedding") rather than re-embedding anything,
     so this stage costs zero additional Cloudflare calls. Falls back to
     an exact-normalized-text dedup pass (cheaper, lower recall) when the
     vec0 table is unavailable.
  3. :func:`run_rollup` -- day -> week -> month summarization via
     ``extract_llm.summarize()`` (one Gemini call per time-bucket that has
     accumulated enough captures). Each summary is written as a NEW
     capture with ``kind='summary'`` and a ``consolidation_summary`` tag
     (read by ``capture.py``'s :func:`capture._is_echo_of_summary`
     anti-loop guard) plus a
     ``rollup_level:<day|week|month>`` tag (so the NEXT level up can find
     and re-roll ONLY already-rolled-up summaries, not raw captures). The
     source captures are tagged ``rolled_up`` (NOT invalidated -- they stay
     individually recallable; only excluded from being rolled up again).
  4. :func:`rebuild_fts` -- a cheap maintenance safety net: rebuilds
     ``captures_fts`` from the live ``captures`` table, in case any future
     code path ever writes/updates a capture's content without keeping the
     FTS shadow column in sync.

:func:`run_nightly` runs all four in order and returns a combined stats
dict; every stage's failure is caught and recorded as ``{"error": True}``
for that stage rather than aborting the whole run.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import db
from . import extract_llm
from ._engine import stem as stem_mod

logger = logging.getLogger("memohood.consolidate")

_DEFAULT_HALFLIFE_DAYS: Dict[str, float] = {
    "event": 7,
    "preference": 90,
    "decision": 90,
    "correction": 90,
    "fact": 365,
    "persona": 365,
    "instruction": 365,
    "summary": 365,
}
_DEFAULT_FLOOR = 0.05

_ROLLUP_MIN_CAPTURES = 5
_ROLLUP_MIN_AGE_DAYS = {"day": 1.0, "week": 7.0, "month": 30.0}


# ---------------------------------------------------------------------------
# Stage 1 — Ebbinghaus decay (pinned exempt)
# ---------------------------------------------------------------------------


def _halflife_for(kind: str, cfg: Optional[Dict[str, Any]]) -> float:
    table = ((cfg or {}).get("decay") or {}).get("halflife_days") or {}
    return float(table.get(kind, _DEFAULT_HALFLIFE_DAYS.get(kind, 365)))


def compute_decay_confidence(
    base_confidence: float,
    kind: str,
    last_seen_at: Optional[float],
    *,
    now: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """Pure function: ``confidence * exp(-age_days / halflife)``, clamped
    to ``[0, 1]``. ``last_seen_at is None`` (never recalled since capture)
    is treated as "seen right now" -- age is 0, no decay yet -- rather than
    exploding to an enormous age from a 1970 epoch default.
    """
    now = now if now is not None else db.now()
    if last_seen_at is None:
        last_seen_at = now
    age_days = max(0.0, (now - float(last_seen_at)) / 86400.0)
    halflife = _halflife_for(kind, cfg)
    if halflife <= 0:
        return max(0.0, min(1.0, float(base_confidence)))
    decayed = float(base_confidence) * math.exp(-age_days / halflife)
    return max(0.0, min(1.0, decayed))


def run_decay(conn: sqlite3.Connection, cfg: Dict[str, Any]) -> Dict[str, int]:
    floor = float(((cfg or {}).get("decay") or {}).get("floor", _DEFAULT_FLOOR))
    now = db.now()
    rows = conn.execute(
        "SELECT id, kind, confidence, last_seen_at, tags FROM captures "
        "WHERE invalidated_at IS NULL AND pinned = 0"
    ).fetchall()

    decayed_count = 0
    archived_count = 0
    for r in rows:
        new_conf = compute_decay_confidence(r["confidence"], r["kind"], r["last_seen_at"], now=now, cfg=cfg)
        if new_conf < floor:
            tags = r["tags"] or ""
            new_tags = f"{tags};archived_decay" if tags else "archived_decay"
            try:
                with conn:
                    conn.execute(
                        "UPDATE captures SET confidence = ?, invalidated_at = ?, tags = ?, updated_at = ? WHERE id = ?",
                        (new_conf, now, new_tags, now, r["id"]),
                    )
                archived_count += 1
            except sqlite3.Error:
                logger.warning("consolidate.run_decay: failed to archive %s", r["id"], exc_info=True)
        else:
            try:
                with conn:
                    conn.execute(
                        "UPDATE captures SET confidence = ?, updated_at = ? WHERE id = ?",
                        (new_conf, now, r["id"]),
                    )
                decayed_count += 1
            except sqlite3.Error:
                logger.warning("consolidate.run_decay: failed to update confidence for %s", r["id"], exc_info=True)

    return {"decayed": decayed_count, "archived": archived_count}


# ---------------------------------------------------------------------------
# Stage 2 — dedup (reuses stored vectors; zero extra embed calls)
# ---------------------------------------------------------------------------


def _self_knn(conn: sqlite3.Connection, capture_id: str, *, k: int = 6) -> List[Dict[str, Any]]:
    if not db.vec_table_exists(conn):
        return []
    vec_table = db.vec_table_name()
    try:
        rows = conn.execute(
            f"SELECT capture_id, distance FROM {vec_table} "
            f"WHERE embedding MATCH (SELECT embedding FROM {vec_table} WHERE capture_id = ?) AND k = ? "
            f"ORDER BY distance",
            (capture_id, k),
        ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        dist = float(r["distance"])
        cosine = max(-1.0, min(1.0, 1.0 - (dist ** 2) / 2.0))
        out.append({"id": r["capture_id"], "cosine": cosine})
    return out


def _run_dedup_vec(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id FROM captures WHERE invalidated_at IS NULL ORDER BY created_at"
    ).fetchall()
    seen: "set[str]" = set()
    merged = 0
    now = db.now()
    for r in rows:
        cid = r["id"]
        if cid in seen:
            continue
        for cand in _self_knn(conn, cid, k=6):
            other = cand["id"]
            if other == cid or other in seen:
                continue
            if cand["cosine"] >= 0.95:
                try:
                    row = conn.execute(
                        "SELECT tags, invalidated_at FROM captures WHERE id = ?", (other,)
                    ).fetchone()
                    if row is None or row["invalidated_at"] is not None:
                        continue
                    tags = row["tags"] or ""
                    new_tags = f"{tags};dedup_merged" if tags else "dedup_merged"
                    with conn:
                        conn.execute(
                            "UPDATE captures SET invalidated_at = ?, tags = ? WHERE id = ?",
                            (now, new_tags, other),
                        )
                    seen.add(other)
                    merged += 1
                except sqlite3.Error:
                    logger.debug("consolidate.run_dedup: failed to merge %s into %s", other, cid, exc_info=True)
    return merged


def _run_dedup_fallback(conn: sqlite3.Connection) -> int:
    """Reduced-recall fallback when the vec leg is unavailable: merges
    captures whose whitespace/case-normalized content is byte-identical.
    Cheap (single pass, no pairwise embedding comparison) but will miss
    paraphrased duplicates -- acceptable v1 tradeoff when sqlite-vec isn't
    installed/configured."""
    rows = conn.execute(
        "SELECT id, content FROM captures WHERE invalidated_at IS NULL ORDER BY created_at"
    ).fetchall()
    seen_keys: Dict[str, str] = {}
    merged = 0
    now = db.now()
    for r in rows:
        key = " ".join((r["content"] or "").split()).lower()
        if not key:
            continue
        if key in seen_keys:
            try:
                with conn:
                    conn.execute(
                        "UPDATE captures SET invalidated_at = ?, tags = tags || ';dedup_merged' WHERE id = ?",
                        (now, r["id"]),
                    )
                merged += 1
            except sqlite3.Error:
                logger.debug("consolidate.run_dedup: fallback merge failed for %s", r["id"], exc_info=True)
        else:
            seen_keys[key] = r["id"]
    return merged


def run_dedup(conn: sqlite3.Connection, cfg: Dict[str, Any]) -> Dict[str, int]:
    if db.vec_table_exists(conn):
        merged = _run_dedup_vec(conn)
    else:
        merged = _run_dedup_fallback(conn)
    return {"merged": merged}


# ---------------------------------------------------------------------------
# Stage 3 — day -> week -> month rollup
# ---------------------------------------------------------------------------


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _week_key(ts: float) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _month_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _insert_summary_capture(
    conn: sqlite3.Connection, summary_text: str, *, level: str, session_id: str, now: float
) -> str:
    new_id = uuid.uuid4().hex
    tags = f"consolidation_summary;rollup_level:{level}"
    with conn:
        conn.execute(
            """
            INSERT INTO captures(
                id, content, kind, confidence, notability, source, pinned,
                supersedes, history, session_id, message_id, tags, last_seen_at,
                created_at, updated_at, valid_from, invalidated_at, embed_signature
            ) VALUES (?, ?, 'summary', 1.0, 'low', 'EXTRACTED', 0, '', '', ?, NULL, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (new_id, summary_text, session_id or "", tags, now, now, now, now),
        )
        conn.execute(
            "INSERT INTO captures_fts(content, content_stem, capture_id) VALUES (?, ?, ?)",
            (summary_text, stem_mod.stem_ru(summary_text), new_id),
        )
    return new_id


def _mark_rolled_up(conn: sqlite3.Connection, ids: List[str]) -> None:
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with conn:
        conn.execute(
            f"UPDATE captures SET tags = CASE WHEN tags IS NULL OR tags = '' THEN 'rolled_up' "
            f"ELSE tags || ';rolled_up' END WHERE id IN ({placeholders})",
            ids,
        )


def _rollup_level(
    conn: sqlite3.Connection,
    cfg: Dict[str, Any],
    *,
    level: str,
    source_sql: str,
    key_fn,
) -> int:
    """Shared rollup pass for one level (day/week/month). *source_sql*
    selects the candidate rows (id, content, session_id, created_at) for
    this level -- callers supply a level-appropriate WHERE clause (raw
    captures for "day", ``rollup_level:day`` summaries for "week", etc.).
    """
    now = db.now()
    cutoff = now - _ROLLUP_MIN_AGE_DAYS[level] * 86400.0
    try:
        rows = conn.execute(source_sql, (cutoff,)).fetchall()
    except sqlite3.Error:
        logger.warning("consolidate.run_rollup: query failed for level=%s", level, exc_info=True)
        return 0

    buckets: Dict[Any, List[sqlite3.Row]] = {}
    for r in rows:
        key = (r["session_id"] or "", key_fn(r["created_at"]))
        buckets.setdefault(key, []).append(r)

    created = 0
    for (session_id, _period), items in buckets.items():
        if len(items) < _ROLLUP_MIN_CAPTURES:
            continue
        texts = [it["content"] for it in items if it["content"]]
        try:
            summary_text = extract_llm.summarize(texts, level=level, model=extract_llm.resolve_model(cfg), conn=conn)
        except Exception:  # noqa: BLE001 - one bucket's LLM failure must not block the rest
            logger.warning("consolidate.run_rollup: summarize() raised for level=%s", level, exc_info=True)
            summary_text = None
        if not summary_text:
            continue
        try:
            _insert_summary_capture(conn, summary_text, level=level, session_id=session_id, now=now)
            _mark_rolled_up(conn, [it["id"] for it in items])
            created += 1
        except sqlite3.Error:
            logger.error("consolidate.run_rollup: failed to write summary for level=%s", level, exc_info=True)

    return created


def run_rollup(conn: sqlite3.Connection, cfg: Dict[str, Any]) -> Dict[str, int]:
    if not ((cfg or {}).get("consolidate") or {}).get("enabled", True):
        return {"day": 0, "week": 0, "month": 0}

    day_created = _rollup_level(
        conn, cfg, level="day", key_fn=_day_key,
        source_sql=(
            "SELECT id, content, session_id, created_at FROM captures "
            "WHERE invalidated_at IS NULL AND kind != 'summary' AND created_at <= ? "
            "AND (tags IS NULL OR tags NOT LIKE '%rolled_up%')"
        ),
    )
    week_created = _rollup_level(
        conn, cfg, level="week", key_fn=_week_key,
        source_sql=(
            "SELECT id, content, session_id, created_at FROM captures "
            "WHERE invalidated_at IS NULL AND kind = 'summary' AND tags LIKE '%rollup_level:day%' "
            "AND created_at <= ? AND (tags IS NULL OR tags NOT LIKE '%rolled_up%')"
        ),
    )
    month_created = _rollup_level(
        conn, cfg, level="month", key_fn=_month_key,
        source_sql=(
            "SELECT id, content, session_id, created_at FROM captures "
            "WHERE invalidated_at IS NULL AND kind = 'summary' AND tags LIKE '%rollup_level:week%' "
            "AND created_at <= ? AND (tags IS NULL OR tags NOT LIKE '%rolled_up%')"
        ),
    )
    return {"day": day_created, "week": week_created, "month": month_created}


# ---------------------------------------------------------------------------
# Stage 4 — FTS rebuild (maintenance safety net)
# ---------------------------------------------------------------------------


def rebuild_fts(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute("SELECT id, content FROM captures WHERE invalidated_at IS NULL").fetchall()
    with conn:
        conn.execute("DELETE FROM captures_fts")
        conn.executemany(
            "INSERT INTO captures_fts(content, content_stem, capture_id) VALUES (?, ?, ?)",
            [(r["content"], stem_mod.stem_ru(r["content"]), r["id"]) for r in rows],
        )
    return {"rebuilt": len(rows)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_nightly(conn: sqlite3.Connection, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run all four consolidation stages in order. Each stage's exception
    is caught and recorded as ``{"error": True}`` for that stage rather
    than aborting the run -- a failing rollup (network down) must not
    prevent decay/dedup/FTS-rebuild from still happening.
    """
    cfg = cfg or {}
    result: Dict[str, Any] = {"started_at": db.now()}
    stages = (
        ("decay", lambda: run_decay(conn, cfg)),
        ("dedup", lambda: run_dedup(conn, cfg)),
        ("rollup", lambda: run_rollup(conn, cfg)),
        ("fts_rebuild", lambda: rebuild_fts(conn)),
    )
    for stage_name, fn in stages:
        try:
            result[stage_name] = fn()
        except Exception:  # noqa: BLE001 - one stage's failure must not abort the nightly job
            logger.error("consolidate.run_nightly: stage %s failed", stage_name, exc_info=True)
            result[stage_name] = {"error": True}
    result["finished_at"] = db.now()
    return result
