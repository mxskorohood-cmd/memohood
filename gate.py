"""Recall gate for memohood: the step that runs BEFORE retrieval to decide
whether a given user turn even warrants a memory lookup.

DESIGN_v1.md's ``prefetch()`` line names this step explicitly: "gate (v1 =
pass-through; model2vec optional later) -> query_norm -> hybrid search" --
and its "Out of v1" line lists "model2vec gate training" as deferred. This
module IS that later gate, built as a self-contained, config-toggleable
addition: nothing about ``provider.py``'s current pass-through behavior
changes until an operator opts in via ``memory.memohood.gate.backend:
model2vec`` in config.yaml (see the module docstring's "Wiring" section for
exactly how a future edit of ``provider.py`` would call this).

Public entry point::

    should_recall(query: str, *, cfg: dict) -> tuple[bool, float, str]
    #                                            ^^^^  ^^^^^  ^^^^^^^
    #                                            pass?  score  human reason

``cfg`` is the effective ``memory.memohood`` config dict (same shape
``config.get_memohood_config()``/``get_memohood_config_readonly()`` return, or just
its ``gate`` sub-dict -- both accepted, mirroring ``embed.py``/
``rerank.py``'s own defensive "nested-or-flat" config handling).

Two backends, selected by ``cfg["gate"]["backend"]``:

* ``"pass"`` (DEFAULT -- matches ``config.DEFAULTS["gate"]["backend"]``
  today): always returns ``(True, 1.0, ...)``. This is the ONLY backend
  active until a user explicitly opts in; ships zero behavior change.
* ``"model2vec"``: a cheap LOCAL static-embedding classifier (lazy-imports
  the optional ``model2vec`` package -- see "Wiring" for the
  ``pip_dependencies`` addition this requires in ``plugin.yaml``, which this
  module does NOT edit). Embeds the query and scores it against two small
  built-in seed sets (:data:`POSITIVE_SEEDS` / :data:`NEGATIVE_SEEDS`).
  Passes (recalls) unless the query looks CLEARLY like a no-recall
  utterance -- greeting/ack/filler with no reference to the past and no
  question shape. See :func:`_decide` for the exact bias-toward-pass rule.

Degrade contract (project hard constraint -- "never crash prefetch", "on
any error ... fall through to a no-op"): if the ``model2vec`` package is
missing, the configured model fails to load, or ANYTHING else goes wrong
while scoring, this module logs once (WARNING, then DEBUG for repeats --
see :func:`_log_degrade_once`) and returns exactly what backend ``"pass"``
would have returned. ``should_recall`` never raises.

Wiring (for whoever edits provider.py -- NOT done by this module per the
"build ONLY your own new module file(s)" constraint):
``provider.py``'s ``_compute_prefetch_text`` currently has::

    gate_backend = (self._cfg.get("gate") or {}).get("backend") or "pass"
    if gate_backend != "pass":
        logger.debug("memohood: unknown gate.backend=%r; defaulting to pass-through", gate_backend)

    normalized = self._normalize_query(query)

Replace those lines with::

    from . import gate as gate_mod  # add to the module-level import block

    recall_ok, gate_score, gate_reason = gate_mod.should_recall(query, cfg=self._cfg)
    if not recall_ok:
        logger.debug("memohood.prefetch: gate skipped recall (score=%.3f, %s)", gate_score, gate_reason)
        return ""

    normalized = self._normalize_query(query)

Because both ``prefetch()`` and ``queue_prefetch()``'s background thread
call ``_compute_prefetch_text`` for their actual work, wiring the gate at
that single call site covers both entry points. No other file needs to
change for the gate itself; ``config.py`` DEFAULTS and ``plugin.yaml``
additions are listed separately in this round's wiring spec (not applied
here, per the "do NOT edit config.py/plugin.yaml" constraint).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import query_norm

logger = logging.getLogger("memohood.gate")

# ---------------------------------------------------------------------------
# Defaults (mirrors config.DEFAULTS["gate"] plus the new model2vec-only keys
# this module reads -- config.py is NOT edited here; see module docstring's
# "Wiring" section for the DEFAULTS patch the integrator applies separately).
# ---------------------------------------------------------------------------

DEFAULT_BACKEND = "pass"
DEFAULT_MODEL2VEC_MODEL = "minishlab/potion-base-8M"  # tiny (~tens of MB)
DEFAULT_MARGIN = 0.05
DEFAULT_THRESHOLD = 0.5
DEFAULT_MEANINGFUL_TERMS_FLOOR = 3

_KNOWN_BACKENDS = frozenset({"pass", "model2vec"})

# ---------------------------------------------------------------------------
# Built-in seed sets (small, on purpose -- a few dozen short RU/EN phrases
# each). NEGATIVE = greetings/acks/fillers with no reference to the past and
# no question shape. POSITIVE = questions, references to prior conversation,
# or possessives ("my", "мой") -- the shape of an utterance that actually
# needs recalled context to answer well.
# ---------------------------------------------------------------------------

NEGATIVE_SEEDS: Tuple[str, ...] = (
    "ок",
    "окей",
    "хорошо",
    "ладно",
    "спасибо",
    "спасибо большое",
    "благодарю",
    "продолжай",
    "давай дальше",
    "го",
    "поехали",
    "да",
    "ага",
    "угу",
    "понял",
    "понятно",
    "принято",
    "привет",
    "здравствуй",
    "пока",
    "до встречи",
    "hi",
    "hello",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "sure",
    "got it",
    "sounds good",
    "go on",
    "go ahead",
    "continue",
    "keep going",
    "cool",
    "great",
    "nice",
    "bye",
    "goodbye",
    "see you",
)

POSITIVE_SEEDS: Tuple[str, ...] = (
    "помнишь",
    "ты помнишь",
    "а ты помнишь",
    "напомни",
    "напомни мне",
    "мы решили",
    "мы договорились",
    "мы обсуждали",
    "что мы решили насчёт",
    "как мы решили",
    "я предпочитаю",
    "я говорил тебе",
    "я тебе говорил",
    "как я говорил ранее",
    "в прошлый раз",
    "в прошлом разговоре",
    "ранее мы",
    "мой",
    "моя",
    "моё",
    "мои",
    "какой у меня",
    "remember",
    "do you remember",
    "you remember",
    "we discussed",
    "we decided",
    "we agreed",
    "as we discussed",
    "earlier",
    "last time",
    "previously",
    "as I mentioned",
    "as I said before",
    "you told me",
    "my",
    "mine",
    "what did we decide about",
    "what do I prefer",
    "remind me",
)


# ---------------------------------------------------------------------------
# Defensive config coercion -- gate.py must never raise on a malformed
# config.yaml value (a user hand-editing e.g. `margin: "high"` must degrade,
# not crash prefetch).
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gate_section(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept either a full ``memory.memohood`` config (with a nested ``gate``
    dict) or the ``gate`` sub-dict itself -- mirrors ``embed.py``/
    ``rerank.py``'s own "nested-or-flat" defensive config handling."""
    cfg = cfg or {}
    if not isinstance(cfg, dict):
        return {}
    nested = cfg.get("gate")
    return nested if isinstance(nested, dict) else cfg


# ---------------------------------------------------------------------------
# Pure math helpers -- independently unit-testable, no model/network
# (mirrors _engine/retrieve.py's rrf_fuse/`_blend_weights` pattern of
# isolating the pure decision math from the I/O that feeds it).
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _decide(pos_sim: float, neg_sim: float, *, margin: float, threshold: float) -> Tuple[bool, float, str]:
    """The bias-toward-pass decision rule, isolated from embedding I/O.

    Rule (spec): pass if ``pos_sim >= neg_sim - margin``. If that fails --
    the query leans toward the negative (no-recall) seeds -- only actually
    skip if ``neg_sim >= threshold``, i.e. the classifier is CONFIDENT this
    looks like a greeting/ack/filler. A low-confidence negative lean (both
    similarities low/noisy, ``neg_sim`` under threshold) still passes: "a
    false skip is worse than a false recall" (task spec) means uncertainty
    must resolve toward recalling, not skipping.
    """
    score = pos_sim - neg_sim
    if pos_sim >= neg_sim - margin:
        return True, score, (
            f"model2vec: pos_sim={pos_sim:.3f} >= neg_sim={neg_sim:.3f} - margin={margin:.3f} -> recall"
        )
    if neg_sim < threshold:
        return True, score, (
            f"model2vec: pos_sim={pos_sim:.3f} < neg_sim={neg_sim:.3f} - margin={margin:.3f}, "
            f"but neg_sim below threshold={threshold:.3f} (low-confidence skip) -> recall (bias toward pass)"
        )
    return False, score, (
        f"model2vec: pos_sim={pos_sim:.3f} < neg_sim={neg_sim:.3f} - margin={margin:.3f} "
        f"and neg_sim >= threshold={threshold:.3f} (confident no-recall utterance) -> skip"
    )


# ---------------------------------------------------------------------------
# model2vec embedder -- lazy import, process-cached, monkeypatchable.
#
# ``_load_embedder`` is a separate, named module-level function (not inlined
# into the caller) SPECIFICALLY so tests can monkeypatch it to return a fake
# embedder object without ever importing the real ``model2vec`` package or
# downloading/loading a real model (project hard constraint: tests never hit
# the network or require a real model download).
# ---------------------------------------------------------------------------

_EMBEDDER_CACHE: Dict[str, Any] = {}
_SEED_EMBED_CACHE: Dict[str, Dict[str, List[List[float]]]] = {}
_degrade_logged = False  # module-level "log once" flag (see _log_degrade_once)


def _load_embedder(model_name: str) -> Any:
    """Lazy-load (and process-cache by model name) a model2vec
    ``StaticModel``. Raises whatever the underlying import/load raises --
    ``ImportError`` if the optional ``model2vec`` package isn't installed,
    or any error ``StaticModel.from_pretrained`` raises for a bad/
    unreachable/corrupt model -- callers (:func:`_should_recall_model2vec`)
    catch and degrade to pass-through.
    """
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    from model2vec import StaticModel  # heavy/optional import, kept local (project convention)

    model = StaticModel.from_pretrained(model_name)
    _EMBEDDER_CACHE[model_name] = model
    return model


def _embed(embedder: Any, texts: Sequence[str]) -> List[List[float]]:
    """Encode *texts*, normalizing the result to plain ``list[list[float]]``
    regardless of whether *embedder*.encode() returns a numpy array, a list
    of numpy arrays, or a list of lists."""
    vectors = embedder.encode(list(texts))
    return [[float(x) for x in vec] for vec in vectors]


def _seed_embeddings(embedder: Any, model_name: str) -> Dict[str, List[List[float]]]:
    """Embed (and process-cache, per model name) :data:`POSITIVE_SEEDS`/
    :data:`NEGATIVE_SEEDS` once -- these never change at runtime, so there is
    no reason to re-embed them on every ``should_recall`` call."""
    cached = _SEED_EMBED_CACHE.get(model_name)
    if cached is not None:
        return cached
    seeds = {
        "positive": _embed(embedder, POSITIVE_SEEDS),
        "negative": _embed(embedder, NEGATIVE_SEEDS),
    }
    _SEED_EMBED_CACHE[model_name] = seeds
    return seeds


def _log_degrade_once(exc: BaseException) -> None:
    """Log the model2vec degrade at WARNING the first time it happens in
    this process, DEBUG on every subsequent occurrence -- avoids spamming
    the log on every single prefetch call for a persistently-missing
    dependency/model, per the task's "log once" requirement."""
    global _degrade_logged
    if not _degrade_logged:
        logger.warning(
            "gate: model2vec unavailable (%s); degrading to pass-through for the rest of this process",
            exc,
            exc_info=True,
        )
        _degrade_logged = True
    else:
        logger.debug("gate: model2vec unavailable (%s); degrading to pass-through", exc)


# ---------------------------------------------------------------------------
# model2vec backend
# ---------------------------------------------------------------------------


def _should_recall_model2vec(query: str, *, gate_cfg: Dict[str, Any]) -> Tuple[bool, float, str]:
    floor = _safe_int(gate_cfg.get("meaningful_terms_floor"), DEFAULT_MEANINGFUL_TERMS_FLOOR)
    terms = query_norm.meaningful_terms(query)
    if len(terms) >= floor:
        # A query with this many real content terms is almost certainly
        # worth searching for -- skip the (comparatively expensive) embed
        # call entirely and just recall. This is the "meaningful-terms
        # floor" the task spec asks for: a safety net biasing long/
        # substantive queries toward pass regardless of what the tiny
        # classifier would have said.
        return True, 1.0, (
            f"model2vec: meaningful-terms floor met ({len(terms)} >= {floor}); recall without scoring"
        )

    model_name = gate_cfg.get("model2vec_model") or DEFAULT_MODEL2VEC_MODEL
    try:
        embedder = _load_embedder(model_name)
        seeds = _seed_embeddings(embedder, model_name)
        query_vec = _embed(embedder, [query])[0]
    except Exception as exc:  # noqa: BLE001 - any model2vec failure degrades to pass-through, never crashes
        _log_degrade_once(exc)
        return True, 1.0, f"model2vec unavailable ({exc}); degraded to pass-through"

    pos_sim = max((_cosine(query_vec, v) for v in seeds["positive"]), default=0.0)
    neg_sim = max((_cosine(query_vec, v) for v in seeds["negative"]), default=0.0)

    margin = _safe_float(gate_cfg.get("margin"), DEFAULT_MARGIN)
    threshold = _safe_float(gate_cfg.get("threshold"), DEFAULT_THRESHOLD)
    return _decide(pos_sim, neg_sim, margin=margin, threshold=threshold)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def should_recall(query: str, *, cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, str]:
    """Decide whether *query* warrants a memory lookup, BEFORE retrieval
    runs. Never raises -- any internal failure degrades to the same result
    backend ``"pass"`` would give.

    Returns ``(recall_ok, score, reason)``:

    * ``recall_ok``: ``True`` -> caller should proceed with retrieval,
      ``False`` -> caller should skip retrieval for this turn (return "" /
      no memory context).
    * ``score``: a float, higher = more clearly recall-worthy. For backend
      ``"pass"`` (and any degrade path) this is a fixed ``1.0`` placeholder,
      not a real similarity gap -- callers should treat it as informational
      only, never threshold on it themselves.
    * ``reason``: a short human-readable string for logging/debugging.

    *cfg* is the effective ``memory.memohood`` config (or just its ``gate``
    sub-dict) -- see :func:`_gate_section`. A missing/falsy *cfg* is treated
    as "no gate config at all", which resolves to the default backend
    (``"pass"``).
    """
    try:
        gate_cfg = _gate_section(cfg)
        backend = str(gate_cfg.get("backend") or DEFAULT_BACKEND).strip().lower()

        query = (query or "").strip()
        if not query:
            return True, 0.0, "gate: empty query; nothing to gate, defaulting to pass"

        if backend == "pass":
            return True, 1.0, "gate: backend=pass; always recall"

        if backend == "model2vec":
            return _should_recall_model2vec(query, gate_cfg=gate_cfg)

        if backend not in _KNOWN_BACKENDS:
            logger.debug("gate: unknown gate.backend=%r; defaulting to pass-through", backend)
        return True, 1.0, f"gate: unknown backend {backend!r}; defaulting to pass-through"
    except Exception:  # noqa: BLE001 - the gate must NEVER block recall due to an internal bug
        logger.warning("gate: unexpected error in should_recall; degrading to pass-through", exc_info=True)
        return True, 1.0, "gate: unexpected internal error; degraded to pass-through"
