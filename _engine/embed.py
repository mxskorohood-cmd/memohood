"""Embedding generation for memohood's ``captures_vec`` table.

Interface:

    def embed_texts(texts: list[str], cfg: dict) -> list[list[float]]
    def embedding_signature(cfg: dict) -> str  # "provider|model|dims"

``cfg`` here is the effective embedder config
(``memory.memohood.embedder`` from config.yaml, e.g.
``{"provider": "cloudflare", "model": "@cf/baai/bge-m3", "dims": 1024}``).
memohood captures have no per-item chunk/overlap knobs, so
``embedding_signature`` is just a ``"provider|model|dims"`` fingerprint --
a captures corpus re-embeds wholesale on provider/model/dims change only.

Two providers:

  * ``cloudflare`` (default): Cloudflare Workers AI ``@cf/baai/bge-m3`` via
    ``https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}``,
    ``Authorization: Bearer {CLOUDFLARE_API_TOKEN}`` — both read from the
    process environment (``$HERMES_HOME/.env``, already loaded by
    hermes-core at startup).
  * ``openai``/``openai-compat``: any OpenAI-embeddings-API-shaped endpoint
    at a configurable ``embedder.base_url`` — a pluggable escape hatch, so
    adding any future provider is three config lines, not new code.

Every request: browser-like ``User-Agent`` (reusing ``security.DEFAULT_USER_AGENT``),
a timeout, and exponential backoff on 429/5xx — matching this project's
"every external HTTP call" non-negotiable.

``embed_texts`` validates its own output before returning: vector count
must match input count, every vector's length must equal
``cfg['dims']``, and every component must be finite (``math.isfinite``) —
malformed vectors must never reach vec0. It does NOT record ledger spend
itself — the caller (``capture.py``) calls ``ledger.record_call`` after a
successful ``embed_texts`` call, since it knows the actual billed unit
count.

The shadow-table migration helper at the bottom (``reembed_captures_shadow``)
re-embeds the whole ``captures``/``captures_vec`` pair into a shadow table
and atomically promotes it live, for use after an embedder config change.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .. import db
from .security import DEFAULT_USER_AGENT

logger = logging.getLogger("memohood.embed")

DEFAULT_TIMEOUT_S = 30.0
MAX_RETRIES = 4
BATCH_SIZE = 96
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class EmbedError(RuntimeError):
    """Raised for any embedding failure: missing credentials, network
    failure after retries, a malformed provider response, or a validation
    failure (dimension mismatch / non-finite values)."""


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def embedding_signature(cfg: Dict[str, Any]) -> str:
    """``"provider|model|dims"`` fingerprint of an embedder config.

    Accepts either a full config with a nested ``embedder`` dict (as read
    from ``memory.memohood`` in config.yaml) or a flat embedder dict —
    defensive against either shape being passed in.
    """
    embedder = cfg.get("embedder", cfg) or {}
    provider = embedder.get("provider", "")
    model = embedder.get("model", "")
    dims = embedder.get("dims", "")
    return f"{provider}|{model}|{dims}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_vectors(vectors: List[Any], expected_dims: int, expected_count: int) -> None:
    if len(vectors) != expected_count:
        raise EmbedError(f"embedder returned {len(vectors)} vectors for {expected_count} inputs")
    for i, vec in enumerate(vectors):
        if not isinstance(vec, (list, tuple)) or len(vec) != expected_dims:
            got = len(vec) if isinstance(vec, (list, tuple)) else "n/a"
            raise EmbedError(f"vector {i} has dim {got}, expected {expected_dims}")
        for x in vec:
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                raise EmbedError(f"vector {i} contains a non-finite value ({x!r})")


# ---------------------------------------------------------------------------
# HTTP with backoff
# ---------------------------------------------------------------------------


def _request_with_backoff(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[dict] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
):
    import requests  # heavy/optional import kept local

    req_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}
    attempt = 0
    while True:
        try:
            resp = requests.request(method, url, headers=req_headers, json=json_body, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise EmbedError(f"request to {url} failed after {attempt} retries: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
            attempt += 1
            continue

        if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            time.sleep(min(2 ** attempt, 30))
            attempt += 1
            continue

        return resp


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _embed_cloudflare(texts: List[str], model: str) -> List[List[float]]:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise EmbedError("CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN not set in environment")

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = _request_with_backoff("POST", url, headers=headers, json_body={"text": batch})
        if resp.status_code != 200:
            raise EmbedError(f"Cloudflare embed call failed: HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise EmbedError(f"Cloudflare embed response was not JSON: {exc}") from exc
        if payload.get("success") is False:
            raise EmbedError(f"Cloudflare embed call reported failure: {payload.get('errors')}")
        data = (payload.get("result") or {}).get("data")
        if data is None:
            raise EmbedError(f"Cloudflare embed response missing result.data: {str(payload)[:500]}")
        all_vectors.extend(data)
    return all_vectors


def _embed_openai_compat(
    texts: List[str], *, base_url: str, model: str, api_key: Optional[str]
) -> List[List[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = _request_with_backoff("POST", url, headers=headers, json_body={"input": batch, "model": model})
        if resp.status_code != 200:
            raise EmbedError(f"OpenAI-compat embed call failed: HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise EmbedError(f"OpenAI-compat embed response was not JSON: {exc}") from exc
        items = payload.get("data")
        if not isinstance(items, list):
            raise EmbedError(f"OpenAI-compat embed response missing data[]: {str(payload)[:500]}")
        try:
            ordered = sorted(items, key=lambda it: it.get("index", 0))
            all_vectors.extend(it["embedding"] for it in ordered)
        except (KeyError, TypeError) as exc:
            raise EmbedError(f"OpenAI-compat embed response malformed: {exc}") from exc
    return all_vectors


# ---------------------------------------------------------------------------
# Local provider (fastembed / ONNX Runtime — NO PyTorch). Opt-in alternative
# to Cloudflare: downloads a quantized ONNX model once, then runs on CPU, so
# the query text and snippets never leave the machine. ``fastembed`` is an
# OPTIONAL dependency, imported lazily and only when embedder.provider=local,
# so a default (cloud) install never pulls it in. Default model is
# multilingual-e5-large: 1024 dims (drop-in for the Cloudflare bge-m3 schema),
# multilingual (RU+EN), MIT-licensed.
# ---------------------------------------------------------------------------

# Needs fastembed>=0.6: since 0.6 fastembed pools e5 with MEAN (the way e5 was
# trained), which is the behaviour this default is validated against. fastembed
# <=0.5.1 used CLS pooling = a DIFFERENT vector space — indexes must not be mixed
# across that boundary (re-embed on any such upgrade). The installers pin
# ``fastembed>=0.6`` for exactly this reason.
DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-large"

# One TextEmbedding instance per model id, cached for the process lifetime:
# the first call loads (and, on first run ever, downloads) the ONNX weights;
# every call after that is in-memory.
_LOCAL_MODELS: Dict[str, Any] = {}


def _get_local_model(model: str):
    inst = _LOCAL_MODELS.get(model)
    if inst is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbedError(
                "local embedder needs the optional 'fastembed' package "
                "(pip install \"fastembed>=0.6\") — only required when "
                "embedder.provider is 'local'."
            ) from exc
        try:
            inst = TextEmbedding(model_name=model)
        except Exception as exc:  # noqa: BLE001 - surface load/download failure as EmbedError
            raise EmbedError(f"could not load local embed model {model!r}: {exc}") from exc
        _LOCAL_MODELS[model] = inst
    return inst


def _apply_e5_prefix(texts: List[str], model: str, is_query: bool) -> List[str]:
    """e5 models are trained with ``query:`` / ``passage:`` prefixes and lose
    accuracy without them; models without ``e5`` in the id are left as-is."""
    if "e5" in model.lower():
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]
    return list(texts)


def _embed_local(texts: List[str], model: str, *, is_query: bool = False) -> List[List[float]]:
    inst = _get_local_model(model)
    prepared = _apply_e5_prefix(texts, model, is_query)
    try:
        # fastembed yields numpy arrays; return plain Python float lists so
        # _validate_vectors (list/tuple of finite numbers) and serialize_vector
        # both keep working unchanged.
        return [[float(x) for x in vec] for vec in inst.embed(prepared)]
    except EmbedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EmbedError(f"local embedder ({model}) failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: List[str], cfg: Dict[str, Any], *, is_query: bool = False) -> List[List[float]]:
    """Embed *texts*, returning one vector per input, in the same order.

    ``is_query`` only matters for the local e5 provider (query vs passage
    prefix); cloud providers ignore it. Raises :class:`EmbedError` on any
    failure — missing credentials, network failure after retries, malformed
    provider response, or output validation failure (see module docstring).
    Never records spend itself.
    """
    if not texts:
        return []

    embedder = cfg.get("embedder", cfg) or {}
    provider = (embedder.get("provider") or "cloudflare").lower()
    model = embedder.get("model") or "@cf/baai/bge-m3"
    dims = embedder.get("dims")
    if not isinstance(dims, int) or dims <= 0:
        raise EmbedError(f"cfg['embedder']['dims'] must be a positive int, got {dims!r}")

    if provider == "cloudflare":
        vectors = _embed_cloudflare(texts, model)
    elif provider in ("local", "fastembed"):
        vectors = _embed_local(texts, embedder.get("model") or DEFAULT_LOCAL_MODEL, is_query=is_query)
    elif provider in ("openai", "openai-compat", "openai_compat"):
        base_url = embedder.get("base_url")
        if not base_url:
            raise EmbedError("openai-compat embedder requires embedder.base_url in config")
        api_key_env = embedder.get("api_key_env", "OPENAI_API_KEY")
        api_key = embedder.get("api_key") or os.environ.get(api_key_env)
        vectors = _embed_openai_compat(texts, base_url=base_url, model=model, api_key=api_key)
    else:
        raise EmbedError(f"unknown embedder provider: {provider!r}")

    _validate_vectors(vectors, dims, len(texts))
    return vectors


# ---------------------------------------------------------------------------
# vec0 serialization
# ---------------------------------------------------------------------------


def serialize_vector(vec: Sequence[float]) -> bytes:
    """Pack a float vector into the byte layout sqlite-vec's ``vec0`` tables
    expect. Public (not underscore-prefixed) because ``capture.py`` needs
    the exact same packing when inserting into the LIVE ``captures_vec``
    table, not just the shadow-table migration path below."""
    try:
        import sqlite_vec

        return sqlite_vec.serialize_float32(list(vec))
    except (ImportError, AttributeError):
        import struct

        return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Shadow-table re-embed / migration
# (adapted for memohood's single global captures corpus — no per-collection rows)
# ---------------------------------------------------------------------------


def _set_migration_state(conn, state: str) -> None:
    try:
        with conn:
            conn.execute(
                "INSERT INTO _meta(key, value) VALUES ('migration_state', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (state,),
            )
    except Exception:  # noqa: BLE001 - status flag write must not mask the real error
        logger.error("failed to set migration_state=%s", state, exc_info=True)


def reembed_captures_shadow(
    conn,
    new_cfg: Dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    """Re-embed every live (non-invalidated) capture into the shadow vec
    table (``captures_vec_v2``) using *new_cfg*'s embedder, then atomically
    promote it live and record the migration state in ``_meta``.

    Sequence: mark 'migrating' -> create+fill shadow table -> atomic
    ``db.swap_vec_table`` -> update embed_signature on every live capture ->
    mark 'idle'. On ANY failure, marks 'failed' (never left stuck at
    'migrating') and re-raises.

    ``_engine/retrieve.py`` is expected to treat a non-'idle' migration
    state as "serve FTS-only or block with a visible degraded-mode status".

    If sqlite-vec is unavailable, the vector side is skipped (logged) but
    the embedder config + embed_signature are still updated — FTS-only
    search keeps working, matching this project's general vec0-optional
    degradation contract.
    """
    embedder = new_cfg.get("embedder", {}) or {}
    dims = embedder.get("dims")
    if not isinstance(dims, int) or dims <= 0:
        raise EmbedError(f"new_cfg['embedder']['dims'] must be a positive int, got {dims!r}")

    _set_migration_state(conn, "migrating")
    try:
        vec_ready = db.ensure_vec_table(conn, dims, shadow=True)
        captures = conn.execute(
            "SELECT id, content FROM captures WHERE invalidated_at IS NULL ORDER BY id",
        ).fetchall()

        embedded_count = 0
        if vec_ready:
            shadow_table = db.vec_table_name(shadow=True)
            for start in range(0, len(captures), batch_size):
                batch = captures[start : start + batch_size]
                texts = [row["content"] for row in batch]
                vectors = embed_texts(texts, new_cfg)
                with conn:
                    for row, vec in zip(batch, vectors):
                        conn.execute(
                            f"INSERT OR REPLACE INTO {shadow_table}(capture_id, embedding) VALUES (?, ?)",
                            (row["id"], serialize_vector(vec)),
                        )
                embedded_count += len(batch)
            db.swap_vec_table(conn)
        else:
            logger.warning(
                "sqlite-vec unavailable; captures migration updates embedder config/"
                "embed_signature only, vector index stays disabled (FTS-only)"
            )

        new_signature = embedding_signature(new_cfg)
        with conn:
            conn.execute(
                "UPDATE captures SET embed_signature = ? WHERE invalidated_at IS NULL",
                (new_signature,),
            )
        _set_migration_state(conn, "idle")
        return {"status": "done", "captures_embedded": embedded_count, "vector_index_ready": vec_ready}
    except Exception:
        _set_migration_state(conn, "failed")
        raise
