"""Integration test against the REAL memory-provider loader
(plugins/memory/__init__.py), not our own synthetic package copy.

This is the test that would have caught the register_cli/memohood_command
discrepancy documented in cli.py's module docstring: the loader's
``discover_plugin_cli_commands()`` scans for module-level ``register_cli``/
``memohood_command`` names, NOT ``ctx.register_cli_command`` (a general-plugin-only
convention). ``plugins.memory.load_memory_provider("memohood")`` is also the exact
call `hermes memory setup`/agent startup makes to activate the provider.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_memohood_at(hermes_home: Path) -> Path:
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    dest = plugins_dir / "memohood"
    shutil.copytree(
        PLUGIN_DIR, dest,
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc", "DESIGN_v1.md"),
    )
    return dest


def _write_config_yaml(hermes_home: Path) -> None:
    (hermes_home / "config.yaml").write_text(
        "memory:\n  provider: memohood\n", encoding="utf-8",
    )


@pytest.fixture()
def installed_memohood(memohood, monkeypatch):
    """Depends on the `memohood` fixture only for its HERMES_HOME isolation +
    credential stripping -- deliberately does NOT use its synthetic package
    copy, since this test exercises the REAL loader's own import path."""
    hermes_home = memohood._hermes_home_for_test
    _install_memohood_at(hermes_home)
    _write_config_yaml(hermes_home)
    # Force a fresh read of config.yaml (hermes_cli.config caches on
    # (mtime_ns, size) keyed by path, so a fresh HERMES_HOME/path is already
    # enough, but be explicit that no stale cache should be at play).
    return hermes_home


def test_load_memory_provider_finds_memohood(installed_memohood):
    import plugins.memory as memory_mod

    provider = memory_mod.load_memory_provider("memohood")
    assert provider is not None, "plugins.memory.load_memory_provider('memohood') returned None"
    assert type(provider).__name__ == "MemoHoodMemoryProvider"
    assert provider.name == "memohood"

    from agent.memory_provider import MemoryProvider as RealMemoryProvider
    assert isinstance(provider, RealMemoryProvider)


def test_discover_plugin_cli_commands_finds_register_cli_and_memohood_command(installed_memohood):
    import plugins.memory as memory_mod

    results = memory_mod.discover_plugin_cli_commands()
    assert len(results) == 1, f"expected exactly one CLI command entry, got {results}"
    entry = results[0]
    assert entry["name"] == "memohood"
    assert entry["setup_fn"] is not None
    assert entry["setup_fn"].__name__ == "register_cli"
    assert entry["handler_fn"] is not None
    assert entry["handler_fn"].__name__ == "memohood_command"


def test_discover_memory_providers_lists_memohood(installed_memohood):
    import plugins.memory as memory_mod

    providers = memory_mod.discover_memory_providers()
    names = [p[0] for p in providers]
    assert "memohood" in names
