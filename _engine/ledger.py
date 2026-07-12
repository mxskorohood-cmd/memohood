"""MemoHood external-spend ledger.

Captures are embedded through Cloudflare — an external dependency sitting
on memory's hot path — so memory gets its own spend ledger and monthly
ceiling, the same way any external-API-backed feature in this project does.

Cloudflare (embed) and Cohere (rerank) calls happen entirely outside
hermes-core's own token/cost-tracking hooks (they are plain HTTP calls this
plugin makes, not LLM API calls hermes tracks) — so this module is the ONLY
place their dollar cost is recorded and capped for the MEMORY side.
memohood's ``spend`` table lives in ``memory.db``, and ``db.py`` (with the
``record_spend``/``monthly_spend`` functions this module wraps) lives at the
plugin root, one level up from this ``_engine/`` package.

Public surface (used by ``_engine/embed.py``, ``_engine/rerank.py``, and
``capture.py``):

    record_call(conn, provider, op, units=None, est_usd=None) -> int
    estimate_cost_usd(provider, op, units) -> float
    check_monthly_ceiling(conn, provider, memohood_cfg) -> (within_ceiling, spent_so_far, ceiling)
    ensure_within_ceiling(conn, provider, memohood_cfg)  # raises LedgerError if already over

Pricing is a best-effort estimate — these constants should be reviewed
periodically against the providers' current published pricing, not treated
as billing-accurate.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Tuple

from .. import db

logger = logging.getLogger("memohood.ledger")


class LedgerError(RuntimeError):
    """Raised by :func:`ensure_within_ceiling` when a provider's monthly
    spend ceiling has already been reached — caller must not start the job."""


# (provider, op) -> $ per billing unit. Unit meaning is documented at each
# call site (embed.py passes "thousands of approx tokens" for cloudflare
# embed calls; rerank.py passes "number of rerank search-units" for cohere
# rerank calls).
#
# Cloudflare Workers AI @cf/baai/bge-m3: priced per-token, on the order of
# ~$0.012 / 1M input tokens as of Cloudflare's public Workers AI pricing
# page — expressed here as $ per 1K tokens.
# Cohere rerank-v3.5: priced per "search unit" (roughly: 1 query x up to 100
# documents), historically ~$2.00 / 1000 search units.
_PRICING_USD: dict = {
    ("cloudflare", "embed"): 0.000012,   # $ per 1,000 approx-tokens
    ("cohere", "rerank"): 0.002,         # $ per rerank search-unit
    ("gemini", "extract"): 0.0,          # flash-lite generous free tier; see extract_llm.py
}

_DEFAULT_FALLBACK_PRICE = 0.0  # unknown (provider, op): estimate as free rather than guess wildly


def estimate_cost_usd(provider: str, op: str, units: float) -> float:
    """Best-effort dollar estimate for *units* of (*provider*, *op*).

    Returns 0.0 for an unrecognized (provider, op) pair rather than raising
    — callers should not let a pricing-table gap block an otherwise-valid
    operation, but SHOULD log/surface that the estimate is unknown (spend is
    still recorded with ``est_usd=0.0`` and remains visible in the ledger for
    manual review).
    """
    price = _PRICING_USD.get((provider, op))
    if price is None:
        logger.warning("ledger: no pricing entry for (%s, %s); estimating $0.00", provider, op)
        price = _DEFAULT_FALLBACK_PRICE
    return max(0.0, float(units)) * price


def record_call(
    conn: sqlite3.Connection,
    *,
    provider: str,
    op: str,
    units: Optional[float] = None,
    est_usd: Optional[float] = None,
) -> int:
    """Append one row to the ``spend`` table. Thin, single-call-site wrapper
    around :func:`db.record_spend` so every module (embed.py/rerank.py/
    extract_llm.py) records spend the same way. If ``est_usd`` is not
    supplied, it is computed via :func:`estimate_cost_usd` (requires
    ``units``).
    """
    if est_usd is None and units is not None:
        est_usd = estimate_cost_usd(provider, op, units)
    return db.record_spend(conn, provider=provider, op=op, units=units, est_usd=est_usd)


def check_monthly_ceiling(
    conn: sqlite3.Connection, provider: str, memohood_cfg: dict
) -> Tuple[bool, float, float]:
    """Return ``(within_ceiling, spent_so_far_usd, ceiling_usd)`` for
    *provider* over the trailing 30 days, per
    ``memory.memohood.monthly_ceiling_usd.<provider>`` (DESIGN_v1.md config
    block).

    A provider with no configured ceiling (or a ceiling of 0/None, meaning
    "not tracked") is always reported ``within_ceiling=True`` — callers that
    want a hard "0 = forbidden" semantics should check ``ceiling_usd`` itself
    rather than relying on this function to enforce it.
    """
    ceiling = (memohood_cfg.get("monthly_ceiling_usd") or {}).get(provider)
    spent = db.monthly_spend(conn, provider)
    if ceiling is None:
        return True, spent, float("inf")
    ceiling = float(ceiling)
    return (spent < ceiling), spent, ceiling


def ensure_within_ceiling(conn: sqlite3.Connection, provider: str, memohood_cfg: dict) -> None:
    """Raise :class:`LedgerError` if *provider*'s trailing-30-day spend has
    already reached/exceeded its configured monthly ceiling. Callers should
    call this BEFORE spending more, not after — it does not itself prevent
    the in-flight call that triggered it from completing.
    """
    within, spent, ceiling = check_monthly_ceiling(conn, provider, memohood_cfg)
    if not within:
        raise LedgerError(
            f"{provider} monthly spend ceiling reached: ${spent:.4f} spent >= ${ceiling:.4f} ceiling "
            f"(memory.memohood.monthly_ceiling_usd.{provider}); job refused"
        )
