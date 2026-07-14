"""Config access for memohood's ``memory.memohood.*`` config tree (DESIGN_v1.md
"Config (config.yaml memory.*)").

Follows this project's own ``config.py`` convention (per the plugin loader's
API contract: "no ctx.config -- every plugin that needs config imports
hermes_cli.config directly"), with one addition:
``MemoryProvider.save_config(values, hermes_home)`` is handed an explicit
*hermes_home* that may not be the "currently active" profile
``hermes_cli.config`` resolves to (it is whatever profile ``hermes memory
setup`` happens to be running against) -- so :func:`save_memohood_config_at`
writes directly to ``<hermes_home>/config.yaml`` instead of going through
``hermes_cli.config``'s active-profile resolution, mirroring
``plugins/memory/holographic/__init__.py``'s own ``save_config()``.

Heavy/host imports (``hermes_cli.config``, ``yaml``) are kept inside
functions per the project convention (per the plugin loader's API contract)
so this module has no load-time cost and no hard dependency on hermes internals --
useful for unit-testing config.py in isolation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# memory.memohood.* defaults -- verbatim from DESIGN_v1.md's "Config" section
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    # v1.1: gate.py's model2vec pre-retrieval "should we even bother recalling?"
    # classifier. backend stays "pass" -- OFF/opt-in only, identical to v1's
    # pass-through stub -- until an operator explicitly sets
    # memory.memohood.gate.backend: model2vec in config.yaml. The other keys are
    # gate.py's own tunables (each defended by gate.py's own DEFAULT_* constant
    # too, so this dict is documentation/discoverability, not load-bearing).
    "gate": {
        "backend": "pass",
        "threshold": 0.5,
        "margin": 0.05,
        "model2vec_model": "minishlab/potion-base-8M",
        "meaningful_terms_floor": 3,
    },
    "model": {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "embedder": {"provider": "cloudflare", "model": "@cf/baai/bge-m3", "dims": 1024},
    "rerank": {"provider": "cohere", "enabled": True},
    "auto_capture": True,
    "capture_threshold": 4.0,
    "monthly_ceiling_usd": {"cloudflare": 5, "cohere": 5, "gemini": 5},
    # Not in DESIGN_v1.md's literal config.yaml sample but required by
    # consolidate.py's decay pass and capture.py's pinned-trigger checks --
    # mirrors gbrain's half-life table.
    "decay": {
        "floor": 0.05,
        "halflife_days": {
            "event": 7,
            "preference": 90,
            "decision": 90,
            "correction": 90,
            "fact": 365,
            "persona": 365,
            "instruction": 365,
            "summary": 365,
        },
    },
    "recall": {"k": 8, "messages_k": 4},
    "consolidate": {"enabled": True},
    # v1.1: post_recall.py's MMR + near-duplicate-collapse diversity pass,
    # run after retrieval+rerank(+graph_rerank), before formatting the
    # <memory-context> block. mmr.enabled defaults TRUE per this round's
    # task spec (unlike gate, this ships ON by default).
    "post_recall": {
        "mmr": {"enabled": True, "lambda": 0.7, "score_key": "score", "vector_key": "vector"},
        "cluster": {"enabled": True, "threshold": 0.93},
    },
    # v1.1: graph_rerank.py's session_links BOOST + 1-hop EXPANSION step,
    # run right after hybrid_search. enabled defaults TRUE per this round's
    # task spec.
    "graph_rerank": {
        "enabled": True,
        "boost": [1.5, 1.3, 1.15],
        "max_neighbors": 3,
        "top_n_anchors": 3,
        "weight_tiers": [0.66, 0.33],
    },
}


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: *base* with *override* merged in, recursively.

    Only dict values are merged recursively; any other type in *override*
    (including lists) replaces the base value outright. Never mutates
    either input.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _memohood_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    memory_section = cfg_get(cfg, "memory", default={}) or {}
    if not isinstance(memory_section, dict):
        # A user could clobber `memory:` with a scalar by hand-editing
        # config.yaml -- don't let that crash every memory.memohood.* read site.
        return {}
    memohood_section = memory_section.get("memohood", {})
    return memohood_section if isinstance(memohood_section, dict) else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_memohood_config() -> Dict[str, Any]:
    """Return the effective ``memory.memohood.*`` config: DEFAULTS deep-merged
    with the user's ``config.yaml``. Safe to mutate -- a fresh dict every
    call."""
    from hermes_cli.config import load_config

    cfg = load_config()
    return _deep_merge(DEFAULTS, _memohood_section(cfg))


def get_memohood_config_readonly() -> Dict[str, Any]:
    """Fast-path variant of :func:`get_memohood_config` for hot, read-only call
    sites (``provider.py``'s ``prefetch()``/``sync_turn()``, ``tools.py``'s
    handlers). Uses ``load_config_readonly()`` to skip hermes' own
    defensive deepcopy of the whole config; ``_deep_merge`` here still
    deepcopies DEFAULTS/override into a fresh result, so the dict returned
    to the caller is always safe to read AND safe to mutate.
    """
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    return _deep_merge(DEFAULTS, _memohood_section(cfg))


def save_memohood_config(patch: Dict[str, Any]) -> None:
    """Deep-merge *patch* into the persisted ``memory.memohood`` config section
    of the CURRENTLY ACTIVE profile and save via
    ``hermes_cli.config.save_config``. For writing to an explicit
    (possibly non-active) ``hermes_home`` -- as ``MemoryProvider.
    save_config()`` requires -- use :func:`save_memohood_config_at` instead.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    memory_section = cfg.get("memory", {})
    if not isinstance(memory_section, dict):
        memory_section = {}
    existing_memohood = memory_section.get("memohood", {})
    if not isinstance(existing_memohood, dict):
        existing_memohood = {}
    memory_section["memohood"] = _deep_merge(existing_memohood, patch)
    memory_section.setdefault("provider", "memohood")
    cfg["memory"] = memory_section
    save_config(cfg)


def set_memohood_value(dotted_key: str, value: Any) -> None:
    """Set a single ``memory.memohood.<dotted_key>`` config value on the active
    profile. Thin pass-through to ``hermes_cli.config.set_config_value``
    (dotted-path writer; routes API-key-shaped values to ``.env``
    automatically, honors managed-scope locks)."""
    from hermes_cli.config import set_config_value

    key = dotted_key if dotted_key.startswith("memory.memohood.") else f"memory.memohood.{dotted_key}"
    set_config_value(key, value)


_SECRET_SETUP_KEYS = frozenset(
    {"GEMINI_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "COHERE_API_KEY"}
)


def save_memohood_config_at(values: Dict[str, Any], hermes_home: str) -> None:
    """Write *values* (the non-secret fields ``hermes memory setup``
    collected, per ``MemoryProvider.save_config()``'s contract -- secrets
    go to ``.env`` and are never passed here) to
    ``<hermes_home>/config.yaml``'s ``memory.memohood`` section directly.

    Unlike :func:`save_memohood_config`, this does NOT go through
    ``hermes_cli.config``'s active-profile resolution -- it honors
    ``save_config()``'s literal contract that *hermes_home* is the profile
    to write to, mirroring ``holographic``'s own ``save_config()``.

    ``values`` keys may be dotted (``"embedder.dims"``, matching
    :meth:`MemoHoodMemoryProvider.get_config_schema`'s own key naming) and are
    expanded into nested dicts before merging. Any of the secret-shaped
    keys in :data:`_SECRET_SETUP_KEYS` accidentally present in *values* are
    dropped rather than written to config.yaml -- defense in depth in case
    a future setup-wizard caller passes the whole collected form instead of
    just the non-secret subset.
    """
    import yaml

    config_path = Path(hermes_home) / "config.yaml"
    existing: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8-sig") as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    memory_section = existing.get("memory", {})
    if not isinstance(memory_section, dict):
        memory_section = {}
    existing_memohood = memory_section.get("memohood", {})
    if not isinstance(existing_memohood, dict):
        existing_memohood = {}

    nested: Dict[str, Any] = {}
    for k, v in (values or {}).items():
        if k in _SECRET_SETUP_KEYS:
            continue
        parts = k.split(".")
        cursor = nested
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = v

    memory_section["memohood"] = _deep_merge(existing_memohood, nested)
    memory_section.setdefault("provider", "memohood")
    existing["memory"] = memory_section

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
