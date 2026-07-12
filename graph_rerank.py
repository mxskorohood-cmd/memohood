"""Graph-rerank -- a post-retrieval step that exploits ``session_links``
(DESIGN_v1.md line 107: "Out of v1: ... graph-rerank via session_links
(v1.1)" -- this is that v1.1 piece, built standalone).

``db.py``'s schema has carried ``session_links(from_session_id,
to_session_id, relationship, label, weight, created_at)`` since v1 (see
``db.py``'s ``DDL_STATEMENTS``), but nothing in this plugin has ever read
it -- ``_engine/retrieve.hybrid_search()`` only ranks by FTS/vector/rerank
signal, with zero awareness that two sessions might be conversationally
related. This module is the missing read side ("Граф-реранк по связям":
"буст результатов из связанных сессий (x1.5/x1.3/
x1.15 по близости) и добор 1-hop соседей, которых лексический поиск не
нашёл").

Call this AFTER retrieval (i.e. after ``_engine.retrieve.hybrid_search()``
has produced its ranked ``captures`` list), never before -- it needs the
retrieval scores to know which sessions are "top hits" in the first place.
Two independent effects, both keyed off the same set of "anchor" sessions
(the sessions the top-N *already-ranked* results came from):

1. **BOOST** (reorders existing results, adds nothing): a result whose OWN
   session is graph-linked (1-hop, either direction -- ``session_links`` is
   treated as an undirected "these two sessions are related" edge, not a
   one-way relevance signal) to an anchor session has its ``score``
   multiplied by a closeness-tiered factor. Closeness is read from the
   link's ``weight`` column, bucketed into ``len(cfg.boost)`` tiers via
   ``cfg.graph_rerank.weight_tiers`` (default two thresholds -> three
   tiers, matching the three default boost multipliers 1.5/1.3/1.15). A
   ``NULL``/missing ``weight`` defaults to ``1.0`` (top tier) -- an
   explicit link row with no recorded weight is read as maximally
   confident, mirroring the original schema's own
   ``weight REAL DEFAULT 1.0`` convention (see this project's memory
   architecture reference; this project's own
   ``db.py`` DDL leaves the column nullable with no SQL-level default, so
   this module supplies the same default in Python instead).

2. **1-HOP EXPANSION** (adds NEW candidates -- actual recall, not just
   reordering): a small, capped number of captures from sessions directly
   linked to an anchor session, that lexical/vector search did NOT already
   surface, are pulled in as additional candidates. Capped globally (across
   all neighbor sessions combined, not per-session) by
   ``cfg.graph_rerank.max_neighbors``. Each added candidate is scored as
   ``anchor_score * tier_boost * a fixed discount`` (see
   ``DEFAULT_NEIGHBOR_SCORE_DISCOUNT``) -- it was never confirmed by
   lexical/vector search, so it must not be able to outrank an actual
   retrieval hit boosted at the same closeness tier, only fill in below it.

Wiring (NOT done by this module -- see this feature's build task: "Build
ONLY your own new module file(s) ... RETURN a precise wiring spec"). The
integrator is expected to call this from ``provider.py``'s
``_compute_prefetch_text``, right after ``hybrid_search`` and before the
``messages`` leg is merged in::

    from . import graph_rerank as graph_rerank_mod
    ...
    captures = retrieve_mod.hybrid_search(conn, normalized, k, self._cfg)
    captures = graph_rerank_mod.graph_rerank(captures, db=conn, cfg=self._cfg)

Degradation contract (matches every other module in this plugin --
non-negotiable per this feature's build task): :func:`graph_rerank` NEVER
raises. It is a pure no-op (returns the exact ``results`` list object,
untouched) whenever: ``cfg.graph_rerank.enabled`` is falsy; ``results`` is
empty; ``db`` is ``None``; the ``session_links`` table has zero rows; or
literally any exception occurs while reading the graph. A graph read must
never be able to break ``prefetch()``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("memohood.graph_rerank")

# ---------------------------------------------------------------------------
# Internal defaults.
#
# Deliberately NOT sourced from ``config.py``'s ``DEFAULTS`` dict -- this
# module's build task forbids editing config.py, and (more importantly) this
# way the function stays fully self-contained and safe to call even if the
# integrator never adds a ``graph_rerank`` section to ``DEFAULTS`` at all:
# every key below has its own fallback, read one at a time via `_cfg_get`,
# so a caller that only sets e.g. ``graph_rerank.enabled: false`` in
# config.yaml still gets sane boost/max_neighbors/tier values for free.
# ---------------------------------------------------------------------------

DEFAULT_ENABLED = True
DEFAULT_BOOST: Tuple[float, float, float] = (1.5, 1.3, 1.15)
DEFAULT_MAX_NEIGHBORS = 3
# How many of the (assumed best-first) input `results` count as "top hits"
# whose sessions become BFS anchors. Not called out by name in this
# feature's build task, but needed internally; exposed as a config knob
# anyway so a future tune-up doesn't require touching this module.
DEFAULT_TOP_N_ANCHORS = 3
# Two thresholds carving link `weight` into 3 closeness tiers:
#   weight >= tiers[0]  -> boost[0]  ("strong"/closest)
#   weight >= tiers[1]  -> boost[1]  ("medium")
#   weight <  tiers[1]  -> boost[2]  ("weak"/farthest)
DEFAULT_WEIGHT_TIERS: Tuple[float, float] = (0.66, 0.33)
DEFAULT_WEIGHT_IF_NULL = 1.0
DEFAULT_NEIGHBOR_SCORE_DISCOUNT = 0.5


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    section = (cfg or {}).get("graph_rerank")
    if not isinstance(section, dict):
        return default
    value = section.get(key, default)
    return default if value is None else value


def _bool_cfg(cfg: Optional[Dict[str, Any]], key: str, default: bool) -> bool:
    return bool(_cfg_get(cfg, key, default))


def _float_list_cfg(cfg: Optional[Dict[str, Any]], key: str, default: Sequence[float]) -> List[float]:
    raw = _cfg_get(cfg, key, list(default))
    if not isinstance(raw, (list, tuple)) or not raw:
        return list(default)
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return list(default)


def _int_cfg(cfg: Optional[Dict[str, Any]], key: str, default: int) -> int:
    raw = _cfg_get(cfg, key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _tier_index(weight: Any, tiers: Sequence[float]) -> int:
    """Map a link weight to a 0-indexed closeness tier (0 = strongest,
    matching ``boost[0]``). A missing/unparseable weight is treated as
    :data:`DEFAULT_WEIGHT_IF_NULL` (top tier) -- see module docstring."""
    w = _safe_float(weight, DEFAULT_WEIGHT_IF_NULL)
    for i, threshold in enumerate(tiers):
        if w >= threshold:
            return i
    return len(tiers)  # weakest tier (index past the last threshold)


def _boost_for_tier(boost: Sequence[float], tier: int) -> float:
    if not boost:
        return 1.0
    idx = min(tier, len(boost) - 1)
    return boost[idx]


# ---------------------------------------------------------------------------
# session_links reads (pure functions of a connection -- no cfg needed here)
# ---------------------------------------------------------------------------


def _session_links_has_rows(db: sqlite3.Connection) -> bool:
    row = db.execute("SELECT 1 FROM session_links LIMIT 1").fetchone()
    return row is not None


def _anchor_sessions(results: List[Dict[str, Any]], top_n: int) -> Dict[str, float]:
    """Return ``{session_id: best_score}`` for the top *top_n* entries of
    *results* that carry a non-empty ``session_id``. A session appearing
    more than once in the top slice keeps the highest score seen."""
    anchors: Dict[str, float] = {}
    for r in results[:top_n]:
        sid = r.get("session_id")
        if not sid:
            continue
        score = _safe_float(r.get("score"), 0.0)
        if sid not in anchors or score > anchors[sid]:
            anchors[sid] = score
    return anchors


def _linked_neighbors(db: sqlite3.Connection, anchor_sessions: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Return ``{neighbor_session_id: (best_link_weight, anchor_score)}``
    for every session directly (1-hop) linked to any session in
    *anchor_sessions*. ``session_links`` is treated as an UNDIRECTED edge
    (its ``from``/``to`` columns record which side happened to write the
    row, not a one-way relevance relationship -- this matches the
    "BFS over a graph" framing this feature is modeled on ("iva
    memory_search: BFS по vault-graph"). Excludes links between
    two anchor sessions (an anchor is already top-ranked; there is nothing
    to attach a neighbor-only boost/expansion to). When a session is
    reachable from more than one anchor (or via more than one link row),
    the STRONGEST (highest-weight) link wins.
    """
    if not anchor_sessions:
        return {}
    ids = list(anchor_sessions.keys())
    placeholders = ",".join("?" for _ in ids)
    rows = db.execute(
        f"""
        SELECT from_session_id, to_session_id, weight
          FROM session_links
         WHERE from_session_id IN ({placeholders}) OR to_session_id IN ({placeholders})
        """,
        ids + ids,
    ).fetchall()

    neighbors: Dict[str, Tuple[float, float]] = {}
    for row in rows:
        a, b = row["from_session_id"], row["to_session_id"]
        w = _safe_float(row["weight"], DEFAULT_WEIGHT_IF_NULL)
        for anchor_side, other_side in ((a, b), (b, a)):
            if anchor_side not in anchor_sessions:
                continue
            if other_side == anchor_side or other_side in anchor_sessions:
                continue  # self-link, or a link between two anchors -- skip
            anchor_score = anchor_sessions[anchor_side]
            existing = neighbors.get(other_side)
            if existing is None or w > existing[0]:
                neighbors[other_side] = (w, anchor_score)
    return neighbors


def _fetch_neighbor_captures(
    db: sqlite3.Connection,
    neighbors: Dict[str, Tuple[float, float]],
    *,
    exclude_ids: Set[str],
    max_neighbors: int,
    boost: Sequence[float],
    tiers: Sequence[float],
) -> List[Dict[str, Any]]:
    """Fetch up to *max_neighbors* NEW (not in *exclude_ids*) active
    captures from sessions in *neighbors*, closest sessions (highest link
    weight) first. The cap is GLOBAL across all neighbor sessions combined,
    per this feature's spec ("capped by cfg graph_rerank.max_neighbors"),
    not per-session. Returns dicts in the SAME shape
    ``_engine.retrieve.hybrid_search`` produces, plus a few private
    (underscore-prefixed) provenance keys, so the caller can merge+sort
    this list together with the original retrieval results without any
    special-casing.
    """
    if max_neighbors <= 0 or not neighbors:
        return []

    added: List[Dict[str, Any]] = []
    ordered_sessions = sorted(neighbors.items(), key=lambda kv: kv[1][0], reverse=True)

    for session_id, (weight, anchor_score) in ordered_sessions:
        if len(added) >= max_neighbors:
            break
        remaining = max_neighbors - len(added)
        tier = _tier_index(weight, tiers)
        factor = _boost_for_tier(boost, tier)
        neighbor_score = anchor_score * factor * DEFAULT_NEIGHBOR_SCORE_DISCOUNT

        # Over-fetch a little so rows already present in `exclude_ids` don't
        # eat into the (small) neighbor budget for nothing.
        rows = db.execute(
            """
            SELECT id, content, kind, confidence, notability, pinned, session_id, tags
              FROM captures
             WHERE session_id = ? AND invalidated_at IS NULL
             ORDER BY pinned DESC, created_at DESC
             LIMIT ?
            """,
            (session_id, remaining + len(exclude_ids) + 5),
        ).fetchall()

        for row in rows:
            if len(added) >= max_neighbors:
                break
            if row["id"] in exclude_ids:
                continue
            added.append({
                "capture_id": row["id"],
                "text": row["content"],
                "score": neighbor_score,
                "source": "graph",
                "rrf_score": None,
                "rerank_score": None,
                "mode": "graph",
                "degraded": False,
                "degraded_reason": None,
                "kind": row["kind"],
                "confidence": row["confidence"],
                "notability": row["notability"],
                "pinned": row["pinned"],
                "session_id": row["session_id"],
                "tags": row["tags"],
                "_graph_added": True,
                "_graph_link_weight": weight,
                "_graph_boost_factor": factor,
            })
            exclude_ids.add(row["id"])  # never add the same capture twice

    return added


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def graph_rerank(
    results: List[Dict[str, Any]],
    *,
    db: Optional[sqlite3.Connection],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """BOOST + 1-hop-EXPAND *results* (the list ``_engine.retrieve.
    hybrid_search`` returns) using memohood's ``session_links`` graph.

    ``db`` is an open ``sqlite3.Connection`` to ``memory.db`` -- the same
    connection object ``provider.py``'s ``self._conn``/``_engine.retrieve``'s
    ``conn`` parameter are, just named ``db`` here per this feature's
    signature spec (no new connection is opened; this function never calls
    ``db.get_connection``/``db.connect`` itself).

    Returns a NEW list, re-sorted by (possibly boosted) ``score`` descending
    -- input dicts are never mutated in place. On any of the degrade
    conditions in the module docstring, returns *results* itself (the exact
    same list object, untouched) -- a pure no-op.
    """
    if not results:
        return results
    if not _bool_cfg(cfg, "enabled", DEFAULT_ENABLED):
        return results
    if db is None:
        return results

    try:
        return _graph_rerank_impl(results, db=db, cfg=cfg)
    except Exception:  # noqa: BLE001 - a graph read must never break prefetch
        logger.warning("graph_rerank: unexpected error; returning results unchanged", exc_info=True)
        return results


def _graph_rerank_impl(
    results: List[Dict[str, Any]],
    *,
    db: sqlite3.Connection,
    cfg: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:
        if not _session_links_has_rows(db):
            return results
    except sqlite3.Error:
        logger.debug("graph_rerank: session_links unreadable (missing table?); treating as empty", exc_info=True)
        return results

    top_n = _int_cfg(cfg, "top_n_anchors", DEFAULT_TOP_N_ANCHORS)
    max_neighbors = _int_cfg(cfg, "max_neighbors", DEFAULT_MAX_NEIGHBORS)
    boost = _float_list_cfg(cfg, "boost", DEFAULT_BOOST)
    tiers = _float_list_cfg(cfg, "weight_tiers", DEFAULT_WEIGHT_TIERS)

    anchors = _anchor_sessions(results, top_n)
    if not anchors:
        return results

    neighbors = _linked_neighbors(db, anchors)
    if not neighbors:
        return results

    # --- 1) BOOST: existing results whose session is linked to an anchor ---
    boosted: List[Dict[str, Any]] = []
    for r in results:
        sid = r.get("session_id")
        link = neighbors.get(sid) if sid else None
        if link is None:
            boosted.append(r)
            continue
        weight, _anchor_score = link
        factor = _boost_for_tier(boost, _tier_index(weight, tiers))
        item = dict(r)
        item["score"] = _safe_float(r.get("score"), 0.0) * factor
        item["_graph_boosted"] = True
        item["_graph_boost_factor"] = factor
        item["_graph_link_weight"] = weight
        boosted.append(item)

    # --- 2) 1-HOP EXPANSION: new captures from linked sessions -------------
    exclude_ids = {r.get("capture_id") for r in results if r.get("capture_id")}
    try:
        added = _fetch_neighbor_captures(
            db, neighbors,
            exclude_ids=exclude_ids, max_neighbors=max_neighbors,
            boost=boost, tiers=tiers,
        )
    except sqlite3.Error:
        logger.debug("graph_rerank: neighbor capture fetch failed; skipping expansion", exc_info=True)
        added = []

    merged = boosted + added
    merged.sort(key=lambda c: _safe_float(c.get("score"), 0.0), reverse=True)
    return merged
