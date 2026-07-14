"""Two-stage fact capture for memohood (DESIGN_v1.md "Capture (capture.py) --
two-stage, accepted design").

Pipeline per conversation-turn side (user OR assistant text), matching
DESIGN_v1.md's six numbered steps verbatim:

  1. Free keyword-signal scoring (RU+EN, :func:`compute_signals`) --
     ``score >= capture_threshold`` is a DEFINITE KEEP (no LLM call);
     ``score <= 0`` (no signal matched at all) is a DEFINITE DROP (no LLM
     call, this is the "явный шум пропускается" case); anything in between
     is the BORDERLINE band -> exactly ONE ``extract_llm.extract()`` call.
  2. Borderline band -> ``extract_llm.extract()`` (already built,
     ``extract_llm.py``) decides is_memorable/kind/notability/source_type/
     pinned via one Gemini flash-lite call.
  3. Injection-sanitize: the turn text going IN to the extractor is fenced
     by ``extract_llm.extract()`` itself (via ``_engine.security.
     fence_untrusted``); the fact coming OUT (whatever text we are about to
     store) is scrubbed here via :func:`_scrub_secrets` before it is ever
     written to ``captures``/embedded/FTS-indexed.
  4. Supersede: a three-tier classifier (modeled on gbrain's
     ``facts/classify.ts``) against existing non-invalidated
     captures found "similar" to the new content:
       - cosine >= 0.95 (or, if the embedder is unavailable, a
         :func:`_fts_dup_candidates` word-overlap fallback >= 0.95) ->
         ``duplicate``, no LLM call, no new row (bump the old row's
         ``last_seen_at`` -- Ebbinghaus "reinforce on access").
       - cosine < 0.92 -> ``independent``, no LLM call, insert normally.
       - 0.92 <= cosine < 0.95 -> ONE ``extract_llm.judge()`` call decides
         duplicate/supersede/independent. On ``supersede``: the OLD
         capture gets ``invalidated_at`` set (soft-invalidate, excluded
         from retrieval by ``_engine/retrieve.py``'s hydration filter) and
         the NEW row's ``history`` field gets a dated line carrying the
         old capture's content -- "текущая правда наверху, история
         сохранена" (DESIGN_v1.md).
  5. Pinned tier: a capture is ``pinned=1`` (Ebbinghaus decay-exempt, see
     ``consolidate.py``) if EITHER the free-signal pass matched an
     identity/safety/medical/explicit-"remember forever" trigger
     (:func:`compute_signals`'s ``pinned`` flag) OR the borderline-band LLM
     call said so.
  6. Embed the capture (Cloudflare, via ``_engine/embed.py``) for the
     vector leg; write FTS(RU-stem, via ``_engine/stem.py``) + vec (via
     ``_engine/embed.py``'s :func:`serialize_vector` +
     ``db.ensure_vec_table``). If the embedder is unavailable (no
     credentials, network down, monthly ceiling reached) the capture is
     still written FTS-only -- matches this project's project-wide
     "degrade to FTS-only, never block a capture on the embedder" contract.

Public entry points consumed by ``provider.py``:

    process_turn(conn, user_content, assistant_content, *, session_id, cfg)
        Called from ``sync_turn()``'s background thread -- runs both
        sides independently, never letting one side's failure block the
        other.
    extract_and_store(conn, text, *, side, session_id, cfg)
        The single-side pipeline above (steps 1-6). Also used directly by
        ``provider.py``'s ``on_pre_compress()`` rescue pass.
    manual_capture(conn, content, *, kind, notability, pinned, session_id, cfg)
        Explicit user-requested capture (the ``memohood_capture`` tool,
        ``on_memory_write``'s mirror, ``on_delegation``'s observation) --
        skips the gate/LLM classification (already decided memorable by
        the caller) but still runs supersede + sanitize + embed (steps
        3-6).
    compute_signals(text, *, side)
        The free keyword-signal scorer (step 1), exposed standalone so
        ``provider.py``'s ``on_pre_compress()`` can pre-screen messages
        without paying for a full ``extract_and_store()`` call on every
        message about to be compressed.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import db
from . import extract_llm
from . import query_norm
from ._engine import embed as embed_mod
from ._engine import ledger as ledger_mod
from ._engine import retrieve as retrieve_mod
from ._engine import security
from ._engine import stem as stem_mod

logger = logging.getLogger("memohood.capture")

# ---------------------------------------------------------------------------
# Step 1 — free keyword-signal scoring (RU + EN)
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, weight, kind, pinned_hint). Weights are summed
# across ALL matching patterns (a message can trip more than one signal);
# `capture_threshold` (default 4.0, memory.memohood.capture_threshold) is the
# score at/above which a match is a DEFINITE KEEP with no LLM call at all.
_USER_SIGNALS: List[Tuple["re.Pattern[str]", float, str, bool]] = [
    # correction
    (re.compile(r"\bне так\b|\bне так,|\bисправ|\bэто неверно\b", re.IGNORECASE), 3.0, "correction", False),
    (re.compile(r"\bwrong\b|\bactually,?\s+(?:it'?s|no)\b|\bcorrection\b|\bthat'?s not right\b", re.IGNORECASE), 3.0, "correction", False),
    # decision
    (re.compile(r"реши(?:ли|м)\b|договорились\b|будем использовать\b", re.IGNORECASE), 3.0, "decision", False),
    (re.compile(r"\blet'?s go with\b|\bwe(?:'ll| will) use\b|\bdecided to\b|\bwe agreed\b", re.IGNORECASE), 3.0, "decision", False),
    # preference
    (re.compile(r"предпочита|никогда не\s+\S+|всегда\s+\S+", re.IGNORECASE), 2.0, "preference", False),
    (re.compile(r"\bi prefer\b|\bi always\b|\bi never\b|\bmy favorite\b", re.IGNORECASE), 2.0, "preference", False),
    # explicit remember (pinned only if "forever"/"навсегда" present — see
    # _PINNED_TRIGGER_RE below, checked separately from the base weight)
    (re.compile(r"запомни\b|важно запомнить\b", re.IGNORECASE), 4.0, "fact", False),
    (re.compile(r"\bremember (?:this|that)\b|\bkeep in mind\b", re.IGNORECASE), 4.0, "fact", False),
    # url / path (worth keeping, low weight on their own)
    (re.compile(r"https?://\S+"), 1.0, "fact", False),
    (re.compile(r"[A-Za-z]:[\\/][\w\\/.\-]+|(?<![\w.])~?/[\w./\-]+\.\w{1,5}\b"), 1.0, "fact", False),
]

# Identity/safety/medical — ALWAYS treated as a pinned (decay-exempt)
# trigger regardless of which band (definite-keep vs borderline) the
# message otherwise falls into.
_PINNED_TRIGGER_RE = re.compile(
    r"запомни навсегда|это важно навсегда|"
    r"меня зовут|мой день рождения|я родил(?:ся|ась)|"
    r"аллерги|диагноз|группа крови|непереносимост|"
    r"\bremember forever\b|\bmy name is\b|\bmy birthday\b|\bi'?m allergic\b|\bblood type\b",
    re.IGNORECASE,
)

_ASSISTANT_SIGNALS: List[Tuple["re.Pattern[str]", float, str, bool]] = [
    (re.compile(r"^\s*remember:|^\s*важно:|ключевой вывод|root cause|итог:|вывод:", re.IGNORECASE | re.MULTILINE), 3.0, "fact", False),
]


def compute_signals(text: str, *, side: str = "user") -> Dict[str, Any]:
    """Free (no LLM) keyword-signal score for *text*.

    Returns ``{"score": float, "kind": str, "pinned": bool, "matched":
    [pattern-name, ...]}``. ``kind`` is the kind of the highest-weight
    matching pattern (``"fact"`` if nothing matched -- ``score`` is what
    the caller actually gates on, not ``kind`` alone). Never raises;
    empty/falsy *text* returns a zero score.
    """
    text = text or ""
    if not text.strip():
        return {"score": 0.0, "kind": "fact", "pinned": False, "matched": []}

    patterns = _USER_SIGNALS if side != "assistant" else _ASSISTANT_SIGNALS
    score = 0.0
    best_kind = "fact"
    best_weight = -1.0
    matched: List[str] = []
    for pattern, weight, kind, _unused in patterns:
        if pattern.search(text):
            score += weight
            matched.append(pattern.pattern[:40])
            if weight > best_weight:
                best_weight = weight
                best_kind = kind

    pinned = bool(_PINNED_TRIGGER_RE.search(text))
    if pinned:
        best_kind = "persona" if best_kind == "fact" else best_kind
        score = max(score, 4.0)  # identity/safety/medical is always at least a definite-keep signal

    return {"score": score, "kind": best_kind, "pinned": pinned, "matched": matched}


_NOTABILITY_BY_KIND = {
    "event": "high",
    "persona": "high",
    "correction": "medium",
    "decision": "medium",
    "preference": "medium",
    "instruction": "medium",
    "fact": "medium",
    "summary": "low",
}


def _default_notability(kind: str) -> str:
    return _NOTABILITY_BY_KIND.get(kind, "low")


# ---------------------------------------------------------------------------
# Step 3 — sanitize the fact OUT before it is ever stored/embedded
# ---------------------------------------------------------------------------


def _scrub_secrets(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Redact any secret-shaped substring found by
    ``_engine.security.scan_secrets`` in *text*. Returns ``(clean_text,
    findings)`` -- ``findings`` is the raw finding list (already
    redacted-excerpt, never the real secret value) for logging by callers.
    Never raises; empty/falsy *text* returns ``("", [])``.
    """
    text = text or ""
    if not text:
        return "", []
    findings = security.scan_secrets(text)
    if not findings:
        return text, []
    logger.warning("capture: redacting %d secret-shaped finding(s) before storing a capture", len(findings))
    redacted = text
    for f in sorted(findings, key=lambda x: x["start"], reverse=True):
        redacted = redacted[: f["start"]] + "[REDACTED]" + redacted[f["end"] :]
    return redacted, findings


# ---------------------------------------------------------------------------
# Anti-loop: never re-extract from consolidate.py's own rollup summaries
# ---------------------------------------------------------------------------


def _is_echo_of_summary(conn: sqlite3.Connection, text: str) -> bool:
    """True if *text* is (near-)identical to an active ``kind='summary'``
    capture tagged ``consolidation_summary`` -- i.e. the turn we are about
    to extract from is very likely a recalled summary being echoed back
    into the conversation (e.g. via a ``<memory-context>`` recall block
    the user or assistant then repeats), not a genuinely new fact. Never
    raises; degrades to ``False`` (extract normally) on any DB error.
    """
    text = (text or "").strip()
    if len(text) < 20:
        return False
    try:
        rows = conn.execute(
            "SELECT content FROM captures WHERE kind = 'summary' AND invalidated_at IS NULL "
            "AND tags LIKE '%consolidation_summary%'"
        ).fetchall()
    except sqlite3.Error:
        return False

    for r in rows:
        c = (r["content"] or "").strip()
        if not c:
            continue
        if c in text or text in c:
            return True
        a = set(query_norm.meaningful_terms(c))
        b = set(query_norm.meaningful_terms(text))
        if len(a) >= 5 and b:
            overlap = len(a & b) / len(a)
            if overlap >= 0.8:
                return True
    return False


# ---------------------------------------------------------------------------
# Step 4 — supersede: near-duplicate candidate discovery
# ---------------------------------------------------------------------------

_DUP_COSINE_HI = 0.95
_DUP_COSINE_LO = 0.92


def _filter_active(conn: sqlite3.Connection, ids: List[str]) -> "set[str]":
    if not ids:
        return set()
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id FROM captures WHERE id IN ({placeholders}) AND invalidated_at IS NULL", ids,
    ).fetchall()
    return {r["id"] for r in rows}


def _hydrate_content(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, str]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT id, content FROM captures WHERE id IN ({placeholders})", ids).fetchall()
    return {r["id"]: r["content"] for r in rows}


def _nearest_captures(
    conn: sqlite3.Connection, content: str, cfg: Dict[str, Any], *, k: int = 5
) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    """Embed *content* and find its nearest existing (active) captures via
    vec0 KNN. Returns ``(candidates, new_vector)`` where each candidate is
    ``{"id", "content", "cosine"}`` best-first, and ``new_vector`` is the
    raw embedding of *content* (so the caller can reuse it when writing the
    capture, rather than re-embedding).

    Returns ``([], None)`` if the embedder/vec table is unavailable, the
    monthly Cloudflare ceiling has been reached, or embedding fails for any
    reason -- callers must fall back to :func:`_fts_dup_candidates`. Never
    raises.

    Cosine similarity is APPROXIMATED from vec0's Euclidean (L2) KNN
    distance under the assumption that BGE-M3 embeddings are (approximately)
    unit-normalized: for unit vectors, ``cosine = 1 - distance**2 / 2``.
    This avoids a second round-trip to read back stored vectors just to
    compute an exact dot product -- acceptable v1 approximation, clamped to
    ``[-1, 1]``.
    """
    embedder_cfg = cfg.get("embedder") or {}
    dims = embedder_cfg.get("dims")
    if not isinstance(dims, int) or dims <= 0:
        return [], None

    try:
        ledger_mod.ensure_within_ceiling(conn, "cloudflare", cfg)
    except ledger_mod.LedgerError as exc:
        logger.info("capture: %s; falling back to FTS-based dup detection", exc)
        return [], None

    if not db.ensure_vec_table(conn, dims):
        return [], None

    try:
        vectors = embed_mod.embed_texts([content], cfg)
    except embed_mod.EmbedError as exc:
        logger.info("capture: embedding failed (%s); falling back to FTS-based dup detection", exc)
        return [], None
    if not vectors:
        return [], None
    new_vec = vectors[0]

    approx_units = max(1, round(len(content) / 4))  # ~4 chars/token, standard rough heuristic
    try:
        ledger_mod.record_call(conn, provider="cloudflare", op="embed", units=approx_units)
    except Exception:  # noqa: BLE001 - ledger bookkeeping must never break a successful embed
        logger.error("capture: failed to record ledger spend", exc_info=True)

    vec_table = db.vec_table_name()
    packed = embed_mod.serialize_vector(new_vec)
    try:
        rows = conn.execute(
            f"SELECT capture_id, distance FROM {vec_table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (packed, k),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("capture: vec0 KNN query failed during supersede check", exc_info=True)
        return [], new_vec

    active_ids = _filter_active(conn, [r["capture_id"] for r in rows])
    content_by_id = _hydrate_content(conn, list(active_ids))
    candidates: List[Dict[str, Any]] = []
    for r in rows:
        cid = r["capture_id"]
        if cid not in active_ids:
            continue
        dist = float(r["distance"])
        cosine = max(-1.0, min(1.0, 1.0 - (dist ** 2) / 2.0))
        candidates.append({"id": cid, "content": content_by_id.get(cid, ""), "cosine": cosine})
    return candidates, new_vec


def _fts_dup_candidates(conn: sqlite3.Connection, content: str, *, limit: int = 5) -> List[Dict[str, Any]]:
    """Fallback near-duplicate candidate discovery when the embedder/vec
    leg is unavailable: FTS5 lookup (reusing ``_engine.retrieve``'s own
    query-hardening helper) + a word-level Jaccard overlap as a rough
    "cosine" stand-in. Never raises; empty list on no usable query terms
    or any FTS error.
    """
    match_expr = retrieve_mod._build_match_expression(content, stem_col="content_stem", raw_col="content")
    if not match_expr:
        return []
    try:
        rows = conn.execute(
            "SELECT capture_id, content FROM captures_fts WHERE captures_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_expr, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("capture: FTS dup-candidate query failed", exc_info=True)
        return []

    ids = [r["capture_id"] for r in rows]
    active_ids = _filter_active(conn, ids)
    new_terms = set(query_norm.meaningful_terms(content))
    candidates: List[Dict[str, Any]] = []
    for r in rows:
        cid = r["capture_id"]
        if cid not in active_ids:
            continue
        cand_terms = set(query_norm.meaningful_terms(r["content"] or ""))
        union = new_terms | cand_terms
        jaccard = (len(new_terms & cand_terms) / len(union)) if union else 0.0
        candidates.append({"id": cid, "content": r["content"], "cosine": jaccard})
    candidates.sort(key=lambda c: c["cosine"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Storage — supersede decision + insert/touch + FTS/vec write
# ---------------------------------------------------------------------------


def _get_capture(conn: sqlite3.Connection, capture_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()


def _touch_last_seen(conn: sqlite3.Connection, capture_id: str, now: float) -> None:
    try:
        with conn:
            conn.execute("UPDATE captures SET last_seen_at = ? WHERE id = ?", (now, capture_id))
    except sqlite3.Error:
        logger.debug("capture: failed to touch last_seen_at for %s", capture_id, exc_info=True)


def _store_capture(
    conn: sqlite3.Connection,
    content: str,
    *,
    kind: str,
    notability: str,
    source: str,
    pinned: bool,
    session_id: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the step-4 supersede classifier against *content*, then either
    touch an existing duplicate or insert a new row (independent, or
    superseding an old one) with FTS + (best-effort) vec indexing.

    Returns ``{"action": "duplicate"|"supersede"|"independent",
    "capture_id": str, "supersedes": str|None}``.
    """
    candidates, new_vec = _nearest_captures(conn, content, cfg, k=5)
    if not candidates:
        candidates = _fts_dup_candidates(conn, content, limit=5)

    action = "independent"
    supersedes_id: Optional[str] = None
    top = candidates[0] if candidates else None
    if top is not None:
        sim = top["cosine"]
        if sim >= _DUP_COSINE_HI:
            action, supersedes_id = "duplicate", top["id"]
        elif sim >= _DUP_COSINE_LO:
            judged = extract_llm.judge(content, candidates[:3], model=extract_llm.resolve_model(cfg), conn=conn)
            action = judged.get("action") or "independent"
            supersedes_id = judged.get("supersedes_id")
            if action not in ("duplicate", "supersede"):
                action, supersedes_id = "independent", None

    now = db.now()

    if action == "duplicate" and supersedes_id:
        _touch_last_seen(conn, supersedes_id, now)
        return {"action": "duplicate", "capture_id": supersedes_id, "supersedes": None}

    history = ""
    supersedes_field = ""
    if action == "supersede" and supersedes_id:
        old = _get_capture(conn, supersedes_id)
        if old is not None:
            supersedes_field = supersedes_id
            old_history = old["history"] or ""
            date_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
            new_line = f"[{date_str}] {old['content']}"
            history = f"{old_history}\n{new_line}" if old_history else new_line
            try:
                with conn:
                    conn.execute("UPDATE captures SET invalidated_at = ? WHERE id = ?", (now, supersedes_id))
            except sqlite3.Error:
                logger.warning("capture: failed to invalidate superseded capture %s", supersedes_id, exc_info=True)
                action = "independent"
                supersedes_field = ""
                history = ""
        else:
            action = "independent"

    new_id = uuid.uuid4().hex
    embed_signature = embed_mod.embedding_signature(cfg) if new_vec is not None else None
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO captures(
                    id, content, kind, confidence, notability, source, pinned,
                    supersedes, history, session_id, message_id, tags, last_seen_at,
                    created_at, updated_at, valid_from, invalidated_at, embed_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    new_id, content, kind, 1.0, notability, source, int(bool(pinned)),
                    supersedes_field, history, session_id or "", None, "", now,
                    now, now, now, embed_signature,
                ),
            )
            conn.execute(
                "INSERT INTO captures_fts(content, content_stem, capture_id) VALUES (?, ?, ?)",
                (content, stem_mod.stem_ru(content), new_id),
            )
    except sqlite3.Error as exc:
        raise db.DbError(f"failed to insert capture {new_id}: {exc}") from exc

    if new_vec is not None:
        dims = (cfg.get("embedder") or {}).get("dims")
        try:
            if isinstance(dims, int) and dims > 0 and db.ensure_vec_table(conn, dims):
                vec_table = db.vec_table_name()
                with conn:
                    conn.execute(
                        f"INSERT OR REPLACE INTO {vec_table}(capture_id, embedding) VALUES (?, ?)",
                        (new_id, embed_mod.serialize_vector(new_vec)),
                    )
        except Exception:  # noqa: BLE001 - a failed vec write must not lose the FTS-indexed capture already committed
            logger.warning("capture: failed to write vector for %s (capture stays FTS-only)", new_id, exc_info=True)

    return {"action": action, "capture_id": new_id, "supersedes": supersedes_field or None}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_and_store(
    conn: sqlite3.Connection,
    text: str,
    *,
    side: str = "user",
    session_id: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run the full two-stage capture pipeline (steps 1-6) on one side
    (``"user"`` or ``"assistant"``) of a conversation turn.

    Returns the :func:`_store_capture` result dict, or ``None`` if the turn
    was dropped (empty, definite-noise per the free-signal score, the
    borderline-band LLM said not memorable, an echo of a consolidation
    summary, or the content was entirely redacted as secret-shaped). Never
    raises -- callers (``provider.py``'s ``sync_turn``/``on_pre_compress``)
    should still wrap this in their own try/except as defense in depth,
    but every internal failure here degrades to "no capture", not a crash.
    """
    cfg = cfg or {}
    text = (text or "").strip()
    if not text:
        return None

    try:
        if _is_echo_of_summary(conn, text):
            return None
    except Exception:  # noqa: BLE001
        logger.debug("capture: anti-loop echo check failed; proceeding", exc_info=True)

    threshold = float(cfg.get("capture_threshold", 4.0))
    sig = compute_signals(text, side=side)
    score = sig["score"]

    if score <= 0:
        return None  # definite drop — free, no LLM

    if score >= threshold:
        kind = sig["kind"] or "fact"
        notability = _default_notability(kind)
        source_type = "EXTRACTED"
        pinned = sig["pinned"]
    else:
        result = extract_llm.extract(text, model=extract_llm.resolve_model(cfg), conn=conn)
        if result is None or not result.get("is_memorable"):
            return None
        kind = result["kind"]
        notability = result["notability"]
        source_type = result["source_type"]
        pinned = bool(result["pinned"]) or sig["pinned"]

    content, _findings = _scrub_secrets(text)
    content = content.strip()
    if not content:
        return None

    try:
        return _store_capture(
            conn, content, kind=kind, notability=notability, source=source_type,
            pinned=pinned, session_id=session_id, cfg=cfg,
        )
    except db.DbError:
        logger.error("capture: failed to store capture", exc_info=True)
        return None


def process_turn(
    conn: sqlite3.Connection,
    user_content: str,
    assistant_content: str,
    *,
    session_id: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run :func:`extract_and_store` independently on both sides of one
    completed turn (``provider.py``'s ``sync_turn()`` background-thread
    entry point). Each side's failure is caught and logged without
    affecting the other. Returns ``{"user": result_or_None, "assistant":
    result_or_None}``.
    """
    cfg = cfg or {}
    results: Dict[str, Optional[Dict[str, Any]]] = {"user": None, "assistant": None}
    try:
        results["user"] = extract_and_store(conn, user_content, side="user", session_id=session_id, cfg=cfg)
    except Exception:  # noqa: BLE001 - one side's crash must not block the other
        logger.error("capture.process_turn: user-side extraction failed", exc_info=True)
    try:
        results["assistant"] = extract_and_store(conn, assistant_content, side="assistant", session_id=session_id, cfg=cfg)
    except Exception:  # noqa: BLE001
        logger.error("capture.process_turn: assistant-side extraction failed", exc_info=True)
    return results


def manual_capture(
    conn: sqlite3.Connection,
    content: str,
    *,
    kind: str = "fact",
    notability: str = "high",
    pinned: bool = False,
    session_id: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Explicit, caller-decided capture (the ``memohood_capture`` tool,
    ``on_memory_write``'s built-in-memory mirror, ``on_delegation``'s
    parent-side observation) -- skips the gate/LLM-classification stage
    (the caller has already decided this is memorable) but still runs
    steps 3-6 (sanitize, supersede, embed+FTS+vec).

    Raises :class:`ValueError` if *content* is empty/whitespace-only, or if
    it is entirely redacted as secret-shaped (nothing left to store) --
    unlike :func:`extract_and_store`'s "return None" degradation, a manual
    capture request should tell its caller (a tool handler) WHY it failed
    rather than silently doing nothing.
    """
    cfg = cfg or {}
    content = (content or "").strip()
    if not content:
        raise ValueError("пустое содержимое")

    if kind not in extract_llm._VALID_KINDS:
        kind = "fact"
    if notability not in extract_llm._VALID_NOTABILITY:
        notability = "high"

    clean, _findings = _scrub_secrets(content)
    clean = clean.strip()
    if not clean:
        raise ValueError("всё содержимое распознано как секрет и было бы полностью вымарано")

    pinned = bool(pinned) or bool(_PINNED_TRIGGER_RE.search(clean))

    return _store_capture(
        conn, clean, kind=kind, notability=notability, source="EXTRACTED",
        pinned=pinned, session_id=session_id, cfg=cfg,
    )
