"""Hybrid retrieval for memohood: FTS5(BM25, RU-stemmed) + vec0 KNN over
``captures``, fused via Reciprocal Rank Fusion, then blended with an
optional reranker — plus a lighter FTS-only leg over ``messages_fts`` (the
state.db message backfill index) for prefetch's "captures + messages" recall
per DESIGN_v1.md.

Schema conventions:

  * ``captures``/``captures_fts`` -- one global corpus; captures already
    carry all their own metadata columns, so there is no per-collection join.
  * ``capture_id`` identifies a record; ``content``/``content_stem`` hold
    its raw and stemmed text.
  * ``captures_vec`` is one global vec0 table (``db.vec_table_name()`` takes
    no id argument).
  * ``invalidated_at`` marks a record superseded/archived, alongside the
    bi-temporal ``valid_from`` column (DESIGN_v1.md's schema).
  * :func:`fts_search_messages` is an FTS-only leg (no vector table exists
    for messages per DESIGN_v1.md's schema) over ``messages_fts``, for the
    catch-up-indexed state.db conversation history. Kept separate from
    :func:`hybrid_search` (rather than unifying into one call) because the
    two corpora have different shapes (messages have no rerank-worthy
    dedicated vector leg, captures do) — ``capture.py``/``provider.py``
    call both and merge/label by source for ``prefetch()``.

Pipeline for :func:`hybrid_search`:

1. **FTS leg**: query hardened/split into RU-stemmed alpha terms (matched
   against ``captures_fts.content_stem``) and "coded" terms — anything with
   a digit, hyphen, or dot, e.g. ``gpt-4``, ``2026.4.10`` — matched against
   the RAW ``captures_fts.content`` column instead. This is necessary, not
   just stylistic: ``stem.stem_ru``'s tokenizer regex (``[^\\W\\d_]+``)
   deliberately excludes digits, so a purely-numeric/coded token produces
   **zero** stems and would silently vanish from ``content_stem`` —
   searching it against the raw ``content`` column is the only way to
   still find it ("закалка FTS-запросов").
2. **Vector leg**: embed the query (same embedder as configured), KNN over
   ``captures_vec`` via sqlite-vec's ``vec0`` MATCH/``k`` syntax. Skipped
   entirely (not an error) if sqlite-vec/the vec table is unavailable, or
   captures are mid-embedding-migration (see ``embed.reembed_captures_shadow``).
3. **RRF fuse**: k=60, top-rank bonus (+0.05 rank #1 of either leg, +0.02
   ranks #2-3), each over-fetched at ``k*3``.
4. **Rerank + positional blend**: :func:`rerank.rerank` is called on the
   top of the fused list; when it returns ``mode == "cohere"``, blend the
   (min-max-normalized) RRF score with the reranker's ``relevance_score``
   using qmd's positional weights (rank 1-3: 75/25, 4-10: 60/40, 11+:
   40/60) keyed off each candidate's ORIGINAL RRF rank — so the reranker
   can refine ordering but cannot "dissolve" an exact top-RRF hit.
   When ``mode == "rrf-only"``, the
   RRF order is kept as-is (no blend).
5. **Invalidated-capture exclusion**: this module re-checks
   ``captures.invalidated_at IS NULL`` when hydrating candidates, as
   defense in depth against any future code path that supersedes/expires a
   capture without cleaning the FTS/vec indexes.

Every candidate dict returned by :func:`hybrid_search` carries: ``capture_id,
text, score, source ("fts"|"vector"|"both"), rrf_score, rerank_score (or
None), mode ("cohere"|"rrf-only"), degraded, degraded_reason, kind,
confidence, notability, pinned, session_id, tags``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .. import db
from . import embed as embed_mod
from . import rerank as rerank_mod
from . import stem as stem_mod

logger = logging.getLogger("memohood.retrieve")

RRF_K = 60
OVERFETCH_MULT = 3
TOP_RANK_BONUS_1 = 0.05
TOP_RANK_BONUS_2_3 = 0.02
RERANK_INPUT_MAX = 20  # candidates handed to the (paid) reranker per query

# qmd positional blend weights: (max_rank_inclusive, weight_rrf, weight_rerank)
_BLEND_BUCKETS: List[Tuple[int, float, float]] = [
    (3, 0.75, 0.25),
    (10, 0.60, 0.40),
    (10**9, 0.40, 0.60),
]


# ---------------------------------------------------------------------------
# Query hardening
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")
_CODED_SHAPE_RE = re.compile(r"[\d\-.]")  # contains a digit, hyphen, or dot
_STRIP_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _escape_fts_string(s: str) -> str:
    """Escape a literal for use inside a double-quoted FTS5 MATCH string
    (FTS5's escape convention: a literal ``"`` is written as ``""``)."""
    return s.replace('"', '""')


def _build_match_expression(raw_query: str, *, stem_col: str, raw_col: str) -> Optional[str]:
    """Split *raw_query* into RU-stemmed alpha terms (searched against
    *stem_col*) and coded/numeric terms (searched against *raw_col*), then
    build an FTS5 MATCH expression OR-ing all of them together (OR, not AND
    — BM25 ranking rewards more term overlap, and OR keeps recall high for
    multi-word natural-language questions).

    Returns None if the query has no usable terms at all (e.g. only
    stopword-shaped punctuation) — caller must treat that as "no FTS leg".
    """
    parts: List[str] = []
    seen_stems: set = set()
    seen_coded: set = set()

    for raw_tok in _TOKEN_RE.findall(raw_query):
        tok = _STRIP_EDGE_PUNCT_RE.sub("", raw_tok)
        if not tok:
            continue
        if _CODED_SHAPE_RE.search(tok):
            key = tok.lower()
            if key and key not in seen_coded:
                seen_coded.add(key)
                parts.append(f'{raw_col}:"{_escape_fts_string(key)}"')
        else:
            stemmed = stem_mod.stem_ru(tok)
            for s in stemmed.split():
                if s and s not in seen_stems:
                    seen_stems.add(s)
                    parts.append(f'{stem_col}:"{_escape_fts_string(s)}"')

    if not parts:
        return None
    return " OR ".join(parts)


# ---------------------------------------------------------------------------
# FTS leg — captures
# ---------------------------------------------------------------------------


def _fts_search_captures(conn: sqlite3.Connection, query: str, limit: int) -> List[str]:
    """Return capture ids (TEXT primary keys — see DESIGN_v1.md's
    ``captures(id TEXT PK, ...)``) in BM25-rank order (best first). Empty
    list on no usable query terms or any FTS syntax/runtime error — logged,
    never raised, so the vector leg can still carry the search."""
    match_expr = _build_match_expression(query, stem_col="content_stem", raw_col="content")
    if not match_expr:
        return []
    try:
        rows = conn.execute(
            "SELECT capture_id FROM captures_fts WHERE captures_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_expr, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("retrieve: FTS query failed (match=%r); FTS leg empty for this query", match_expr, exc_info=True)
        return []
    return [r["capture_id"] for r in rows]


# ---------------------------------------------------------------------------
# FTS leg — messages (state.db catch-up index; no vector leg per schema)
# ---------------------------------------------------------------------------


def fts_search_messages(conn: sqlite3.Connection, query: str, limit: int) -> List[Dict[str, Any]]:
    """Return up to *limit* messages matching *query* via BM25 over
    ``messages_fts`` (RU-stemmed), best-first. Never raises — empty list on
    no usable query terms or any FTS error.

    Each result dict carries: ``message_id, session_id, role, content,
    timestamp``. There is no vector/rerank leg for messages (DESIGN_v1.md's
    schema has no ``messages_vec`` table) — this is intentionally FTS-only;
    ``capture.py``/``provider.py`` merge this with :func:`hybrid_search`'s
    captures results for the combined recall context.
    """
    match_expr = _build_match_expression(query, stem_col="content_stem", raw_col="content")
    if not match_expr:
        return []
    try:
        rows = conn.execute(
            """
            SELECT message_id, session_id, role, content, timestamp
              FROM messages_fts
             WHERE messages_fts MATCH ?
             ORDER BY rank LIMIT ?
            """,
            (match_expr, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("retrieve: messages FTS query failed (match=%r)", match_expr, exc_info=True)
        return []
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Vector leg — captures
# ---------------------------------------------------------------------------


def _vec_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff the global ``captures_vec`` table exists AND the
    sqlite-vec extension is loaded (or loads successfully) on *conn*.

    Checks via a cheap ``SELECT vec_version()`` probe first rather than
    unconditionally calling ``db.load_sqlite_vec`` on every query — loading
    a SQLite extension twice on the same already-loaded connection is not
    guaranteed to be a harmless no-op, and this connection is very likely
    reused across many retrieval calls (hot path).
    """
    if not db.vec_table_exists(conn):
        return False
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.Error:
        return db.load_sqlite_vec(conn)


def _vector_search(conn: sqlite3.Connection, query: str, limit: int, cfg: Dict[str, Any]) -> List[str]:
    """Return capture ids in nearest-first order via vec0 KNN. Empty list
    (never raises) if the vec leg is unavailable, captures are
    mid-migration, or query embedding fails for any reason — all are
    "degrade to FTS-only for this call" conditions, not errors."""
    migration_state = cfg.get("migration_state") or "idle"
    if migration_state not in ("idle", None):
        logger.info("retrieve: captures mid-migration (state=%r); skipping vector leg", migration_state)
        return []

    if not _vec_ready(conn):
        return []

    try:
        vectors = embed_mod.embed_texts([query], cfg, is_query=True)
    except embed_mod.EmbedError as exc:
        logger.warning("retrieve: query embedding failed (%s); vector leg empty for this query", exc)
        return []
    if not vectors:
        return []

    vec_table = db.vec_table_name()
    packed = embed_mod.serialize_vector(vectors[0])
    try:
        rows = conn.execute(
            f"SELECT capture_id FROM {vec_table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (packed, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("retrieve: vec0 KNN query failed; vector leg empty for this query", exc_info=True)
        return []
    return [r["capture_id"] for r in rows]


# ---------------------------------------------------------------------------
# RRF fuse
# ---------------------------------------------------------------------------


def rrf_fuse(fts_ids: List[str], vector_ids: List[str], *, k: int = RRF_K) -> "Dict[str, Dict[str, Any]]":
    """Reciprocal Rank Fusion of two rank-ordered id lists, plus qmd's
    top-rank bonus. Returns ``{capture_id: {"score": float, "source": str}}``
    — a pure function, independently unit-testable (no DB/network).

    ``source`` is ``"fts"``, ``"vector"``, or ``"both"``.
    """
    scores: Dict[str, float] = defaultdict(float)
    sources: Dict[str, set] = defaultdict(set)

    for rank, cid in enumerate(fts_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
        if rank == 1:
            scores[cid] += TOP_RANK_BONUS_1
        elif rank in (2, 3):
            scores[cid] += TOP_RANK_BONUS_2_3
        sources[cid].add("fts")

    for rank, cid in enumerate(vector_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
        if rank == 1:
            scores[cid] += TOP_RANK_BONUS_1
        elif rank in (2, 3):
            scores[cid] += TOP_RANK_BONUS_2_3
        sources[cid].add("vector")

    out: Dict[str, Dict[str, Any]] = {}
    for cid, score in scores.items():
        src = sources[cid]
        source_label = "both" if len(src) > 1 else next(iter(src))
        out[cid] = {"score": score, "source": source_label}
    return out


# ---------------------------------------------------------------------------
# Hydration (drop invalidated captures)
# ---------------------------------------------------------------------------


def _hydrate(conn: sqlite3.Connection, capture_ids_in_order: List[str]) -> Dict[str, Dict[str, Any]]:
    if not capture_ids_in_order:
        return {}
    placeholders = ",".join("?" for _ in capture_ids_in_order)
    rows = conn.execute(
        f"""
        SELECT id AS capture_id, content, kind, confidence, notability, pinned,
               session_id, tags, invalidated_at
        FROM captures
        WHERE id IN ({placeholders})
        """,
        capture_ids_in_order,
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["invalidated_at"] is not None:
            continue  # defense in depth (see module docstring)
        out[r["capture_id"]] = dict(r)
    return out


# ---------------------------------------------------------------------------
# Positional blend
# ---------------------------------------------------------------------------


def _blend_weights(rrf_rank: int) -> Tuple[float, float]:
    for max_rank, w_rrf, w_rerank in _BLEND_BUCKETS:
        if rrf_rank <= max_rank:
            return w_rrf, w_rerank
    return _BLEND_BUCKETS[-1][1], _BLEND_BUCKETS[-1][2]  # unreachable in practice


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Public entry point — captures hybrid search
# ---------------------------------------------------------------------------


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    k: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return up to *k* fused, (optionally reranked+blended) capture
    candidates for *query*. Never raises for ordinary degradation (empty
    query, no candidates, reranker unavailable, vector leg unavailable) —
    returns ``[]`` only when there is truly nothing to show.

    *cfg* is the effective ``memory.memohood`` config (needs ``embedder`` for
    query embedding, ``rerank`` for the reranker step, and
    ``migration_state``).
    """
    query = (query or "").strip()
    if not query or k <= 0:
        return []

    overfetch = max(k * OVERFETCH_MULT, k)
    fts_ids = _fts_search_captures(conn, query, overfetch)
    vector_ids = _vector_search(conn, query, overfetch, cfg)

    fused = rrf_fuse(fts_ids, vector_ids)
    if not fused:
        return []

    fused_order = sorted(fused.keys(), key=lambda cid: fused[cid]["score"], reverse=True)

    hydrated = _hydrate(conn, fused_order)
    # Re-filter the order to only ids that survived hydration (dropped
    # invalidated ones), preserving RRF rank among the rest.
    fused_order = [cid for cid in fused_order if cid in hydrated]
    if not fused_order:
        return []

    migration_state = cfg.get("migration_state") or "idle"
    degraded = migration_state not in ("idle", None)
    degraded_reason = "migrating" if degraded else None

    rerank_input_ids = fused_order[:RERANK_INPUT_MAX]
    rerank_candidates = [
        {
            "capture_id": cid,
            "text": hydrated[cid]["content"],
            "rrf_score": fused[cid]["score"],
            "source": fused[cid]["source"],
            "_rrf_rank": rank,  # 1-indexed rank BEFORE reranking; private to this module
        }
        for rank, cid in enumerate(rerank_input_ids, start=1)
    ]

    ranked, mode = rerank_mod.rerank(query, rerank_candidates, cfg, conn=conn)

    if mode == "cohere":
        rrf_scores = [c["rrf_score"] for c in ranked]
        norm_rrf = dict(zip((c["capture_id"] for c in ranked), _min_max_normalize(rrf_scores)))
        for c in ranked:
            w_rrf, w_rerank = _blend_weights(c["_rrf_rank"])
            c["score"] = w_rrf * norm_rrf[c["capture_id"]] + w_rerank * c.get("rerank_score", 0.0)
        ranked.sort(key=lambda c: c["score"], reverse=True)
    else:
        # rrf-only: keep RRF order as-is; expose the RRF score itself as the
        # gating "score" (the caller applies its own rrf_threshold).
        for c in ranked:
            c["score"] = c["rrf_score"]

    # Candidates beyond RERANK_INPUT_MAX (if k*3 overfetch produced more than
    # we sent to the reranker) are appended in their original RRF order,
    # scored purely by RRF — they were never going to outrank the reranked
    # head, but the caller may still want up to k results.
    tail_ids = fused_order[RERANK_INPUT_MAX:]
    for cid in tail_ids:
        ranked.append({
            "capture_id": cid,
            "text": hydrated[cid]["content"],
            "rrf_score": fused[cid]["score"],
            "score": fused[cid]["score"],
            "source": fused[cid]["source"],
        })

    results: List[Dict[str, Any]] = []
    for c in ranked[:k]:
        cid = c["capture_id"]
        meta = hydrated[cid]
        results.append({
            "capture_id": cid,
            "text": meta["content"],
            "score": c["score"],
            "source": c.get("source", "fts"),
            "rrf_score": c.get("rrf_score"),
            "rerank_score": c.get("rerank_score"),
            "mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "kind": meta["kind"],
            "confidence": meta["confidence"],
            "notability": meta["notability"],
            "pinned": meta["pinned"],
            "session_id": meta["session_id"],
            "tags": meta["tags"],
        })
    return results
