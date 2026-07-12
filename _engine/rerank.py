"""Cohere reranking for memohood, with a mandatory RRF-only fallback.

This module works purely on generic ``{"text": ...}`` candidate dicts
handed to it by ``retrieve.py`` — it has no capture/table-specific logic.

Interface:

    def rerank(query: str, candidates: list[dict], cfg: dict) -> tuple[list[dict], str]
    # (ranked, mode) mode in {'cohere', 'rrf-only'}

This module's contract is: **never raise for an unavailable/failing
reranker** — Cohere's free trial is 10 req/min, which a live bot will
exceed easily, and "reranker degraded to RRF-only" is a normal, expected,
visibly-flagged operating mode, not an error path.

Every failure mode (no API key, disabled in config, network failure after
retries, 429/5xx exhausted, malformed response) degrades to
``(candidates, "rrf-only")`` — the candidates list is returned UNCHANGED
(already RRF-ordered by retrieve.py) so the caller always has something to
gate/generate from.

Ledger integration: pass ``conn`` to record actual Cohere spend and to
refuse (degrade, not crash) once memohood's own monthly ceiling is reached
(``memory.memohood.monthly_ceiling_usd.cohere``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from . import embed as embed_mod
from . import ledger

logger = logging.getLogger("memohood.rerank")

DEFAULT_MODEL = "rerank-v3.5"
DEFAULT_TIMEOUT_S = 15.0
MAX_RETRIES = 3
# Cohere charges per "search unit" = 1 query x up to 100 documents.
DOCS_PER_SEARCH_UNIT = 100
# Cap on chars sent per document — keeps request size/cost bounded and avoids
# sending pathologically huge chunks to a paid API by accident.
MAX_DOC_CHARS = 6000

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class RerankError(RuntimeError):
    """Raised internally for a failed Cohere call. Never escapes
    :func:`rerank` — always caught there and turned into an ``'rrf-only'``
    degradation."""


def _cohere_rerank(query: str, documents: List[str], *, model: str, api_key: str) -> List[Dict[str, Any]]:
    """Call Cohere's rerank endpoint. Returns the raw ``results`` list
    (``[{"index": int, "relevance_score": float}, ...]``), NOT necessarily
    sorted the same as the input (Cohere sorts by relevance_score desc).

    Raises :class:`RerankError` on any failure — including retries
    exhausted on 429/5xx, which is the expected/common case for Cohere's
    trial rate limit (10 req/min) and must be treated as "reranker
    unavailable right now", not a crash.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "query": query,
        "documents": [d[:MAX_DOC_CHARS] for d in documents],
        "top_n": len(documents),
    }
    # Reuse embed.py's browser-UA + timeout + exp-backoff HTTP helper rather
    # than duplicating that logic — both modules make simple JSON POSTs with
    # identical retry semantics (429/5xx, capped exponential backoff).
    resp = embed_mod._request_with_backoff(
        "POST", COHERE_RERANK_URL, headers=headers, json_body=body,
        timeout=DEFAULT_TIMEOUT_S, max_retries=MAX_RETRIES,
    )
    if resp.status_code != 200:
        raise RerankError(f"Cohere rerank call failed: HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RerankError(f"Cohere rerank response was not JSON: {exc}") from exc
    results = payload.get("results")
    if not isinstance(results, list):
        raise RerankError(f"Cohere rerank response missing results[]: {str(payload)[:500]}")
    return results


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    conn: Any = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Rerank *candidates* (each a dict with at least a ``"text"`` key —
    the fields produced by ``retrieve.hybrid_search``'s RRF fusion step)
    against *query* via Cohere.

    Returns ``(ranked, mode)``: on success, ``ranked`` is *candidates*
    reordered with a ``"rerank_score"`` key added to each (Cohere's
    ``relevance_score``, 0..1) and ``mode == "cohere"``. On ANY degradation
    (disabled, no key, network/API failure, empty input), returns the
    original *candidates* list unchanged and ``mode == "rrf-only"``.

    *cfg* accepts either a full config with a nested ``rerank`` dict (as
    read from ``memory.memohood`` in config.yaml) or a flat rerank-config dict —
    mirrors ``embed.embedding_signature``'s defensive shape handling.
    """
    if not candidates:
        return candidates, "rrf-only"

    rerank_cfg = cfg.get("rerank", cfg) if isinstance(cfg.get("rerank"), dict) else cfg
    if not rerank_cfg.get("enabled", True):
        return candidates, "rrf-only"

    provider = (rerank_cfg.get("provider") or "cohere").lower()
    if provider != "cohere":
        logger.warning("rerank: unsupported provider %r configured; degrading to rrf-only", provider)
        return candidates, "rrf-only"

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        logger.info("rerank: COHERE_API_KEY not set; degrading to rrf-only")
        return candidates, "rrf-only"

    if conn is not None:
        try:
            ledger.ensure_within_ceiling(conn, "cohere", cfg)
        except ledger.LedgerError as exc:
            logger.info("rerank: %s; degrading to rrf-only", exc)
            return candidates, "rrf-only"

    model = rerank_cfg.get("model") or DEFAULT_MODEL
    documents = [c.get("text") or "" for c in candidates]

    try:
        results = _cohere_rerank(query, documents, model=model, api_key=api_key)
    except RerankError as exc:
        logger.warning("rerank: Cohere call failed (%s); degrading to rrf-only", exc)
        return candidates, "rrf-only"
    except Exception:  # noqa: BLE001 - any unexpected shape/network error must degrade, not crash retrieval
        logger.warning("rerank: unexpected error calling Cohere; degrading to rrf-only", exc_info=True)
        return candidates, "rrf-only"

    if conn is not None:
        search_units = max(1, -(-len(candidates) // DOCS_PER_SEARCH_UNIT))  # ceil division
        try:
            ledger.record_call(conn, provider="cohere", op="rerank", units=search_units)
        except Exception:  # noqa: BLE001 - ledger bookkeeping must never break a successful rerank result
            logger.error("rerank: failed to record ledger spend", exc_info=True)

    ranked: List[Dict[str, Any]] = []
    seen_indices = set()
    for r in sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True):
        idx = r.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates) or idx in seen_indices:
            continue
        seen_indices.add(idx)
        item = dict(candidates[idx])
        item["rerank_score"] = float(r.get("relevance_score", 0.0))
        ranked.append(item)

    # Defensive: if Cohere's response omitted some indices (malformed/partial
    # response), append the missing candidates at the end in their original
    # order rather than silently dropping them.
    if len(ranked) < len(candidates):
        for i, c in enumerate(candidates):
            if i not in seen_indices:
                item = dict(c)
                item.setdefault("rerank_score", 0.0)
                ranked.append(item)

    return ranked, "cohere"
