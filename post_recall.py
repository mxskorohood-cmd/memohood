"""PostRecall diversity pass for memohood: Maximal Marginal Relevance (MMR) +
optional near-duplicate collapse over the ranked capture list
:func:`_engine.retrieve.hybrid_search` already produced.

DESIGN_v1.md line 107 explicitly scoped this OUT of v1 ("PostRecall
MMR/cluster (chronological+rerank is enough for v1)"); this project's
internal design notes describe the target shape as "PostRecall (bge-small
rerank + dedup + MMR)". This module is the "dedup + MMR" half (reranking is already done by
``_engine/rerank.py`` upstream of this step) -- v1's hybrid_search only
removed EXACT duplicates (same ``capture_id`` cannot appear twice); this
module additionally removes NEAR-duplicates (paraphrases of the same fact,
cosine-close in embedding space) and, more importantly, actively prefers a
DIVERSE top-k over a top-k that is "the same fact said five times" -- five
near-identical captures can each individually have a high relevance score
(they are, after all, all relevant) while collectively wasting the whole
injected memory budget on one fact and starving every other fact out of the
context window.

Pipeline position (caller's responsibility -- this module does not wire
itself into ``provider.py``, see the module-level "WIRING" note below):

    captures = retrieve.hybrid_search(conn, query, k, cfg)   # retrieval + rerank
    captures = post_recall.attach_vectors(conn, captures)    # (optional) fetch embeddings
    captures = post_recall.diversify(captures, cfg=cfg)      # <-- THIS MODULE
    # ... format captures into the <memory-context> text ...

Public API
----------
``diversify(results, *, cfg, query=None) -> list[dict]``
    Pure, in-memory, no I/O. Reorders/trims ``results`` (never invents new
    items, never mutates the input list or its dicts) to maximize a
    relevance/diversity tradeoff. See its docstring for the exact
    degrade-to-passthrough conditions -- by design this function can NEVER
    raise; on any missing dependency, invalid config, or invalid/missing
    per-item data, it returns ``results`` completely unchanged.

``attach_vectors(conn, results, *, id_key="capture_id", vector_key="vector") -> list[dict]``
    Optional convenience helper: best-effort batch fetch of each result's
    stored embedding from the ``captures_vec`` sqlite-vec table (by
    ``capture_id``), attached in place under ``vector_key``. Exists because
    ``hybrid_search``'s result dicts do not carry the capture's embedding
    (only ``capture_id``/``text``/scores/metadata) -- ``diversify()`` needs
    a vector per candidate for its cosine-similarity terms, so something
    has to fetch them. Never raises; degrades to "leave vector_key absent"
    on ANY failure (no sqlite-vec, no captures_vec table, DB error,
    malformed blob) -- callers do not need to check its return value for
    success, ``diversify()``'s own missing-vector guard handles the rest.

Config (``memory.memohood.post_recall.*`` -- see WIRING note for the exact
defaults to add to ``config.py``'s ``DEFAULTS``; this module applies the
SAME defaults itself when a key is absent, so it behaves correctly even
before ``config.py`` is updated):

    post_recall:
      mmr:
        enabled: true      # master switch; false => passthrough
        lambda: 0.7         # 1.0 = pure relevance order; 0.0 = pure diversity
        score_key: score    # which field on each result dict is "relevance"
        vector_key: vector  # which field on each result dict is the embedding
      cluster:
        enabled: true       # collapse near-duplicate clusters before MMR
        threshold: 0.93     # cosine >= this => same cluster => collapse

Algorithm
---------
1. (optional) Near-duplicate collapse: greedy single-linkage-to-
   representative clustering in relevance-descending order -- the highest-
   relevance item in a cluster survives as its representative; every other
   item within ``cluster.threshold`` cosine similarity of an already-kept
   representative is dropped. This never keeps a LOWER-relevance duplicate
   over a higher-relevance one.
2. MMR selection over whatever survived step 1: repeatedly pick the
   candidate maximizing
   ``lambda * relevance_norm(c) - (1 - lambda) * max_similarity(c, selected)``
   until every surviving candidate has been placed. ``relevance_norm`` is a
   min-max normalization of the ORIGINAL relevance scores purely so the two
   terms of the MMR objective live on a comparable 0..1 scale for the
   internal argmax comparison -- the scores on the OUTPUT dicts are the
   caller's original, untouched values (see "Preserve the original
   relevance signal" in the task brief this module was built against: MMR
   only reorders/trims, it never invents or overwrites a score).

WIRING (for the integrator -- this module deliberately does not import or
edit ``provider.py``):

    In ``MemoHoodMemoryProvider._compute_prefetch_text`` (provider.py), right
    after::

        captures = retrieve_mod.hybrid_search(conn, normalized, k, self._cfg)

    and BEFORE ``self._reinforce(conn, captures)`` / the ``lines.append``
    formatting block, add::

        from . import post_recall  # top-level import, alongside retrieve_mod

        captures = post_recall.attach_vectors(conn, captures)
        captures = post_recall.diversify(captures, cfg=self._cfg, query=normalized)

    Do NOT run this over ``messages`` (the ``fts_search_messages`` leg) --
    there is no ``messages_vec`` table (DESIGN_v1.md's schema), so every
    message candidate would lack a vector and ``diversify()`` would just
    degrade to a no-op for that list anyway; skipping the call there avoids
    the wasted work.

    Add to ``config.py``'s ``DEFAULTS`` dict::

        "post_recall": {
            "mmr": {"enabled": True, "lambda": 0.7, "score_key": "score", "vector_key": "vector"},
            "cluster": {"enabled": True, "threshold": 0.93},
        },
"""

from __future__ import annotations

import logging
import math
import struct
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("memohood.post_recall")

# ---------------------------------------------------------------------------
# Defaults (mirrored into config.py's DEFAULTS by the integrator -- see the
# module docstring's WIRING note; applied here too so this module is
# correct standalone even before that edit lands).
# ---------------------------------------------------------------------------

DEFAULT_MMR_ENABLED = True
DEFAULT_LAMBDA = 0.7
DEFAULT_SCORE_KEY = "score"
DEFAULT_VECTOR_KEY = "vector"
DEFAULT_CLUSTER_ENABLED = True
DEFAULT_CLUSTER_THRESHOLD = 0.93


# ---------------------------------------------------------------------------
# Small numeric helpers (no numpy dependency -- candidate lists here are at
# most a few dozen items, plain Python is plenty fast enough).
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 (never
    raises) if either vector has zero norm -- a zero vector has no
    direction, so "similarity" is undefined; treating it as dissimilar is
    the safe choice (it will neither wrongly suppress nor wrongly privilege
    any candidate)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _min_max_normalize(values: List[float]) -> List[float]:
    """Scale *values* into [0, 1], preserving order. All-equal input (or
    empty input) maps to a constant 1.0 for every value -- a degenerate but
    safe stand-in (mirrors ``_engine/retrieve.py``'s own
    ``_min_max_normalize``; duplicated here rather than imported since that
    one is a private, non-reusable helper of a different module)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _extract_vectors(results: List[Dict[str, Any]], vector_key: str) -> Optional[List[List[float]]]:
    """Return one vector per result, in order, or ``None`` if ANY result is
    missing a usable vector under *vector_key* (not a list/tuple, empty,
    non-finite, or a dimension mismatch against the others) -- the
    all-or-nothing contract :func:`diversify` relies on for its degrade
    guard."""
    vectors: List[List[float]] = []
    dim: Optional[int] = None
    for r in results:
        if not isinstance(r, dict):
            return None
        v = r.get(vector_key)
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            return None
        try:
            fv = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(x) for x in fv):
            return None
        if dim is None:
            dim = len(fv)
        elif len(fv) != dim:
            return None
        vectors.append(fv)
    return vectors


def _extract_scores(results: List[Dict[str, Any]], score_key: str) -> Optional[List[float]]:
    """Return one relevance score per result, in order, or ``None`` if ANY
    result is missing a usable numeric score under *score_key*."""
    scores: List[float] = []
    for r in results:
        if not isinstance(r, dict) or score_key not in r:
            return None
        v = r.get(score_key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        scores.append(fv)
    return scores


# ---------------------------------------------------------------------------
# Near-duplicate collapse
# ---------------------------------------------------------------------------


def _collapse_near_duplicates(
    indices: List[int],
    vectors: List[List[float]],
    relevance_norm: List[float],
    threshold: float,
) -> List[int]:
    """Greedy single-linkage-to-representative clustering: process
    *indices* in relevance-descending order; an index joins the first
    already-kept representative it is within *threshold* cosine similarity
    of (dropped), otherwise it becomes a new representative (kept). Always
    keeps the HIGHEST-relevance member of a duplicate cluster, never a
    lower one, because processing order is relevance-descending."""
    order = sorted(indices, key=lambda i: relevance_norm[i], reverse=True)
    kept: List[int] = []
    for i in order:
        is_dup = False
        for j in kept:
            if _cosine(vectors[i], vectors[j]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------


def _mmr_select(
    indices: List[int],
    relevance_norm: List[float],
    vectors: List[List[float]],
    lam: float,
) -> List[int]:
    """Iteratively pick, from *indices*, the index maximizing
    ``lam * relevance_norm[i] - (1 - lam) * max_similarity_to_already_picked``,
    until all of *indices* have been placed. Deterministic tie-break:
    *indices* is seeded in relevance-descending order and only a STRICTLY
    greater score replaces the running best, so among exact ties the
    earlier (higher-relevance, or first-listed) candidate wins -- this also
    makes ``lam == 1.0`` reduce to exactly the relevance-descending order
    (the similarity term is multiplied by zero and drops out entirely)."""
    remaining = sorted(indices, key=lambda i: relevance_norm[i], reverse=True)
    selected: List[int] = []
    while remaining:
        best_idx = None
        best_val = None
        for i in remaining:
            if selected:
                max_sim = max(_cosine(vectors[i], vectors[j]) for j in selected)
            else:
                max_sim = 0.0
            val = lam * relevance_norm[i] - (1.0 - lam) * max_sim
            if best_val is None or val > best_val:
                best_val = val
                best_idx = i
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def diversify(
    results: List[Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]],
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Reorder (and, if clustering collapses near-duplicates, trim)
    *results* to maximize relevance while surfacing a diverse set of facts,
    instead of five rephrasings of the single most-relevant one.

    *results* is the ranked list ``_engine.retrieve.hybrid_search`` (or an
    equivalent recall step) returns: each item is a dict carrying at least
    a relevance score (``cfg``-configurable key, default ``"score"``) and,
    for this function to do anything at all, an embedding vector
    (``cfg``-configurable key, default ``"vector"`` -- see
    :func:`attach_vectors`). *query* is accepted for API symmetry / future
    use (e.g. a query-anchored relevance mode) but is NOT used by the
    current implementation -- honoring "MMR preserves the original
    relevance signal, it does not invent scores" means relevance always
    comes from the candidates' own already-computed score, never from a
    fresh query-embedding comparison computed here.

    Never raises. Returns ``results`` UNCHANGED (same order, same objects)
    when:
      - *results* has 0 or 1 items (nothing to diversify);
      - ``cfg`` disables ``post_recall.mmr.enabled``;
      - ``cfg["post_recall"]["mmr"]["lambda"]`` is not a valid number;
      - ANY candidate lacks a usable vector under the configured
        ``vector_key`` (missing, wrong shape, non-finite, or a dimension
        mismatch against its peers);
      - ANY candidate lacks a usable numeric score under the configured
        ``score_key``;
      - any unexpected error occurs during computation (defense in depth --
        this function must never be the reason a turn's prefetch fails).
    """
    try:
        return _diversify_impl(results, cfg=cfg, query=query)
    except Exception:  # noqa: BLE001 - post_recall must never fail prefetch
        logger.warning("post_recall.diversify: unexpected error; passthrough", exc_info=True)
        return results


def _diversify_impl(
    results: List[Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]],
    query: Optional[str],
) -> List[Dict[str, Any]]:
    if not results or len(results) <= 1:
        return results

    post_cfg = (cfg or {}).get("post_recall") or {}
    if not isinstance(post_cfg, dict):
        return results
    mmr_cfg = post_cfg.get("mmr") or {}
    cluster_cfg = post_cfg.get("cluster") or {}
    if not isinstance(mmr_cfg, dict) or not isinstance(cluster_cfg, dict):
        return results

    if not mmr_cfg.get("enabled", DEFAULT_MMR_ENABLED):
        return results

    lam = mmr_cfg.get("lambda", DEFAULT_LAMBDA)
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        return results
    lam = float(lam)
    if not math.isfinite(lam):
        return results
    lam = max(0.0, min(1.0, lam))

    score_key = mmr_cfg.get("score_key", DEFAULT_SCORE_KEY)
    vector_key = mmr_cfg.get("vector_key", DEFAULT_VECTOR_KEY)
    if not isinstance(score_key, str) or not isinstance(vector_key, str):
        return results

    vectors = _extract_vectors(results, vector_key)
    if vectors is None:
        return results  # degrade: some candidate lacks a usable vector

    scores = _extract_scores(results, score_key)
    if scores is None:
        return results  # degrade: some candidate lacks a usable score

    relevance_norm = _min_max_normalize(scores)
    indices = list(range(len(results)))

    cluster_enabled = cluster_cfg.get("enabled", DEFAULT_CLUSTER_ENABLED)
    threshold = cluster_cfg.get("threshold", DEFAULT_CLUSTER_THRESHOLD)
    if cluster_enabled and threshold is not None and not isinstance(threshold, bool) and isinstance(threshold, (int, float)):
        threshold = float(threshold)
        if math.isfinite(threshold):
            indices = _collapse_near_duplicates(indices, vectors, relevance_norm, threshold)

    order = _mmr_select(indices, relevance_norm, vectors, lam)
    return [results[i] for i in order]


# ---------------------------------------------------------------------------
# Optional helper: fetch embeddings for capture_ids from captures_vec
# ---------------------------------------------------------------------------


def _deserialize_vector(blob: bytes) -> Optional[List[float]]:
    """Unpack a sqlite-vec ``FLOAT[n]`` blob (a raw little-endian float32
    array, no header -- the same layout ``db.serialize_vector`` writes,
    whether it used ``sqlite_vec.serialize_float32`` or the plain
    ``struct.pack`` fallback) back into a list of Python floats. Returns
    ``None`` (never raises) on a malformed/truncated blob."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) % 4 != 0 or len(blob) == 0:
        return None
    n = len(blob) // 4
    try:
        return list(struct.unpack(f"<{n}f", blob))
    except struct.error:
        return None


def attach_vectors(
    conn: Any,
    results: List[Dict[str, Any]],
    *,
    id_key: str = "capture_id",
    vector_key: str = "vector",
) -> List[Dict[str, Any]]:
    """Best-effort: batch-fetch each result's stored embedding from the
    global ``captures_vec`` sqlite-vec table (keyed by ``capture_id``) and
    attach it under *vector_key*, mutating the dicts in *results* in place
    (and returning *results* for convenient chaining). This does not
    "invent" anything -- it reads back a vector the capture pipeline
    already computed and stored at capture time.

    Never raises. On ANY failure -- ``conn`` is ``None``, ``db`` cannot be
    imported, sqlite-vec is not installed/loadable, the ``captures_vec``
    table does not exist, or a query/row error -- this is a silent no-op:
    whichever (or however many) results already lacked *vector_key* simply
    keep lacking it, and :func:`diversify`'s own all-or-nothing vector
    guard takes it from there (passthrough, not a crash).
    """
    if conn is None or not results:
        return results

    try:
        from . import db as db_mod  # local import: keep this module import-light/optional
    except Exception:  # noqa: BLE001
        logger.debug("post_recall.attach_vectors: could not import db module", exc_info=True)
        return results

    try:
        if not db_mod.vec_table_exists(conn):
            return results
        try:
            conn.execute("SELECT vec_version()")
        except Exception:  # noqa: BLE001
            if not db_mod.load_sqlite_vec(conn):
                return results

        ids = [r.get(id_key) for r in results if isinstance(r, dict) and r.get(id_key)]
        if not ids:
            return results

        table = db_mod.vec_table_name()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT capture_id, embedding FROM {table} WHERE capture_id IN ({placeholders})",
            ids,
        ).fetchall()

        vec_by_id: Dict[str, List[float]] = {}
        for row in rows:
            cid = row["capture_id"]
            vec = _deserialize_vector(row["embedding"])
            if vec is not None:
                vec_by_id[cid] = vec

        for r in results:
            if not isinstance(r, dict):
                continue
            cid = r.get(id_key)
            if cid in vec_by_id:
                r[vector_key] = vec_by_id[cid]
    except Exception:  # noqa: BLE001 - attaching vectors is best-effort, never worth failing prefetch over
        logger.debug("post_recall.attach_vectors: failed to fetch vectors", exc_info=True)

    return results
