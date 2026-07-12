"""Shared pytest fixtures for memohood's test suite.

Isolation pattern (DESIGN_v1.md "Tests" section, "hermes_plugins namespace
load like tests/plugins/test_disk_cleanup_plugin.py", generalized for a
multi-module plugin): monkeypatch ``HERMES_HOME`` to a per-test tmp dir, strip credential-shaped
env vars so mocked-network tests are hermetic by default, and load the
plugin's modules via a synthetic ``hermes_plugins_test_<uuid>.memohood``
namespace package so relative imports (``from . import db``, ``from
._engine import retrieve``, etc.) resolve exactly as they would under the
real memory-provider loader (``plugins/memory/__init__.py``'s
``_load_provider_from_dir``).

A fresh copy of the WHOLE package is imported under a brand-new synthetic
top-level name every call (:func:`load_plugin_package`), so every test gets
fully isolated module-level state (``_engine.stem``'s lazy PyStemmer
singleton, etc.) with zero cross-test leakage risk.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

# `--import-mode=importlib` (pytest.ini) deliberately does NOT add the tests
# directory to sys.path -- add it explicitly in case a future test file wants
# `from _helpers import ...` without needing tests/__init__.py.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Captured at conftest IMPORT time -- before any test's monkeypatch has run --
# so this always reflects the REAL ambient hermes installation, never a
# test's isolated tmp HERMES_HOME. Used only for (a) the real-HERMES_HOME
# untouched guard below and (b) reading real API keys for @pytest.mark.integration.
try:
    from hermes_constants import get_hermes_home as _real_get_hermes_home

    REAL_HERMES_HOME = _real_get_hermes_home()
except Exception:
    REAL_HERMES_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
REAL_ENV_PATH = REAL_HERMES_HOME / ".env"

# Credential-shaped env vars this plugin reads directly from os.environ
# (embed.py/rerank.py/extract_llm.py) -- stripped by default in every test via
# the `memohood` fixture so a developer's real keys can never leak into a "no
# credentials configured" assertion; `real_api_env` restores the real values
# explicitly, opt-in, for @pytest.mark.integration tests only.
_CREDENTIAL_ENV_VARS = (
    "GEMINI_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "COHERE_API_KEY",
)


def load_plugin_package() -> Any:
    """Import a FRESH copy of the entire memohood plugin package (__init__
    + every submodule reachable from it, including the ``_engine`` vendored
    subpackage) under a unique synthetic package name, and return the
    __init__ module object.

    All submodules become available as attributes of the returned object
    (``pkg.db``, ``pkg.capture``, ``pkg.provider``, ``pkg._engine.retrieve``,
    ...) -- a plain consequence of Python's import system always setting an
    imported submodule as an attribute of its parent package.
    """
    suffix = uuid.uuid4().hex[:10]
    pkg_name = f"hermes_plugins_test_{suffix}"
    full_pkg = f"{pkg_name}.memohood"

    ns = types.ModuleType(pkg_name)
    ns.__path__ = []
    sys.modules[pkg_name] = ns

    spec = importlib.util.spec_from_file_location(
        full_pkg,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full_pkg
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules[full_pkg] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def memohood(tmp_path, monkeypatch):
    """The single fixture almost every test uses: isolates HERMES_HOME to a
    fresh tmp dir, strips credential env vars (hermetic by default), and
    returns a freshly-imported copy of the whole memohood plugin package.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    pkg = load_plugin_package()
    pkg._hermes_home_for_test = hermes_home
    return pkg


class _ProviderCollector:
    """Minimal stand-in for `plugins/memory/__init__.py`'s own
    ``_ProviderCollector`` -- the REAL fake ctx the memory-provider loader
    hands to a provider's ``register(ctx)``. Only ``register_memory_provider``
    does anything; every other method is a documented no-op (verified against
    ``plugins/memory/__init__.py``, 2026-07-06) -- CLI registration for memory
    providers happens through the separate ``register_cli``/``memohood_command``
    module-level functions in ``cli.py``, never through ``ctx``.
    """

    def __init__(self) -> None:
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass


@pytest.fixture()
def provider_collector():
    return _ProviderCollector()


def real_env_keys() -> Dict[str, str]:
    """Read GEMINI_API_KEY/CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN/
    COHERE_API_KEY from the REAL ``~/.hermes/.env`` (never hardcoded, never
    committed). Returns an empty dict (never raises) if python-dotenv or the
    file is unavailable -- @pytest.mark.integration tests must treat missing
    keys as "skip", not "fail".
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    if not REAL_ENV_PATH.exists():
        return {}
    try:
        values = dotenv_values(REAL_ENV_PATH)
    except Exception:
        return {}
    return {k: v for k, v in values.items() if k in _CREDENTIAL_ENV_VARS and v}


@pytest.fixture()
def real_api_env(monkeypatch):
    """For @pytest.mark.integration tests only: force whichever of
    GEMINI_API_KEY/CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN/COHERE_API_KEY
    are present in the REAL ~/.hermes/.env into os.environ, overriding
    whatever the ambient shell/process environment already has (a stale
    ambient copy of a credential can cause a live 401 that looks like a
    client bug).
    """
    keys = real_env_keys()
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    return keys


# ---------------------------------------------------------------------------
# Session-wide guard: the REAL HERMES_HOME must never be touched by this
# suite -- every test isolates HERMES_HOME via the `memohood` fixture above. This
# guard is narrow (memory.db + config.yaml mtime) because a full-tree rglob
# would false-positive on unrelated __pycache__ churn from importing
# hermes_cli/hermes_constants during collection.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _guard_real_hermes_home():
    memory_db = REAL_HERMES_HOME / "memory.db"
    config_path = REAL_HERMES_HOME / "config.yaml"

    memory_db_existed_before = memory_db.exists()
    memory_db_mtime_before = memory_db.stat().st_mtime if memory_db_existed_before else None
    config_mtime_before = config_path.stat().st_mtime if config_path.exists() else None

    yield

    memory_db_existed_after = memory_db.exists()
    assert memory_db_existed_after == memory_db_existed_before, (
        f"REAL HERMES_HOME/memory.db ({memory_db}) existence changed during the test run -- "
        "some code path escaped the per-test HERMES_HOME isolation (the `memohood` fixture)."
    )
    if memory_db_existed_before:
        memory_db_mtime_after = memory_db.stat().st_mtime
        assert memory_db_mtime_after == memory_db_mtime_before, (
            f"REAL HERMES_HOME/memory.db ({memory_db}) was modified during the test run."
        )

    config_mtime_after = config_path.stat().st_mtime if config_path.exists() else None
    assert config_mtime_after == config_mtime_before, (
        f"REAL HERMES_HOME/config.yaml ({config_path}) was modified during the test run -- "
        "a test must have written against the real config instead of the isolated tmp HERMES_HOME."
    )
