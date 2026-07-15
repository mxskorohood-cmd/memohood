"""Tests for the read-only key-visibility surface added to setup_wizard.py
(``env_file_path`` / ``key_status`` / ``relevant_keys`` / ``format_keys_block``)
and its wiring into ``memohood_stats``.

Why it exists: after `hermes memohood setup` a human should be able to run
``hermes memohood stats`` and SEE which keys are configured, where the .env
lives, and whether the running process has actually picked them up (systemd
bug B12), without hand-opening the file — and never see a full secret.

Isolation: the ``memohood`` fixture points HERMES_HOME at a fresh tmp dir and
strips credential env vars, so ``env_file_path()`` is ``<tmp>/.hermes/.env`` and
every key starts absent unless a test sets it. No network anywhere.
"""

from __future__ import annotations

import copy
import importlib


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _sw(memohood):
    # setup_wizard is a deferred import inside tools.py (not loaded at package
    # import), so it isn't auto-set as a package attribute — fetch it directly
    # under the synthetic package name, same as systemd_env below.
    return importlib.import_module(f"{memohood.__name__}.setup_wizard")


# ===========================================================================
# env_file_path / key_status
# ===========================================================================


class TestEnvFilePathAndKeyStatus:
    def test_env_file_path_is_hermes_home_dotenv(self, memohood):
        assert _sw(memohood).env_file_path() == memohood._hermes_home_for_test / ".env"

    def test_key_only_in_file_is_the_b12_case(self, memohood):
        _sw(memohood).env_file_path().write_text(
            "CLOUDFLARE_API_TOKEN=cfat_FILEONLY_TAIL\n", encoding="utf-8"
        )
        st = _sw(memohood).key_status(["CLOUDFLARE_API_TOKEN"])["CLOUDFLARE_API_TOKEN"]
        assert st["in_file"] is True
        assert st["in_process"] is False
        assert st["mask"] == "cfat…"

    def test_key_in_process_is_seen(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_LIVE_TAIL")
        st = _sw(memohood).key_status(["GEMINI_API_KEY"])["GEMINI_API_KEY"]
        assert st["in_process"] is True
        assert st["mask"] == "AIza…"

    def test_key_nowhere_is_empty(self, memohood):
        st = _sw(memohood).key_status(["COHERE_API_KEY"])["COHERE_API_KEY"]
        assert st == {"in_file": False, "in_process": False, "mask": ""}

    def test_commented_line_does_not_count(self, memohood):
        _sw(memohood).env_file_path().write_text(
            "# COHERE_API_KEY=commented\n", encoding="utf-8"
        )
        assert _sw(memohood).key_status(["COHERE_API_KEY"])["COHERE_API_KEY"]["in_file"] is False

    def test_mask_never_leaks_full_secret(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_SUPERSECRETTAIL_1234")
        st = _sw(memohood).key_status(["GEMINI_API_KEY"])["GEMINI_API_KEY"]
        assert "SUPERSECRETTAIL" not in st["mask"]


# ===========================================================================
# relevant_keys
# ===========================================================================


class TestRelevantKeys:
    def _vars(self, items):
        return [i["env_var"] for i in items]

    def test_defaults_cloudflare_cohere_gemini(self, memohood):
        vars_ = self._vars(_sw(memohood).relevant_keys(_cfg(memohood)))
        assert "CLOUDFLARE_ACCOUNT_ID" in vars_ and "CLOUDFLARE_API_TOKEN" in vars_
        assert "COHERE_API_KEY" in vars_ and "GEMINI_API_KEY" in vars_

    def test_cloudflare_embedder_keys_required(self, memohood):
        items = _sw(memohood).relevant_keys(_cfg(memohood))
        for i in items:
            if i["env_var"].startswith("CLOUDFLARE_"):
                assert i["required"] is True
            if i["env_var"] in ("COHERE_API_KEY", "GEMINI_API_KEY"):
                assert i["required"] is False

    def test_local_embedder_drops_cloudflare_keys(self, memohood):
        cfg = _cfg(memohood)
        cfg["embedder"]["provider"] = "local"
        vars_ = self._vars(_sw(memohood).relevant_keys(cfg))
        assert "CLOUDFLARE_ACCOUNT_ID" not in vars_
        assert "COHERE_API_KEY" in vars_ and "GEMINI_API_KEY" in vars_

    def test_rerank_disabled_drops_cohere(self, memohood):
        cfg = _cfg(memohood)
        cfg["rerank"]["enabled"] = False
        vars_ = self._vars(_sw(memohood).relevant_keys(cfg))
        assert "COHERE_API_KEY" not in vars_


# ===========================================================================
# format_keys_block
# ===========================================================================


class TestFormatKeysBlock:
    def test_shows_path_and_marks(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_LIVE")
        block = _sw(memohood).format_keys_block(_cfg(memohood))
        assert str(_sw(memohood).env_file_path()) in block
        assert "GEMINI_API_KEY — ✓ настроен (AIza…)" in block
        assert "CLOUDFLARE_ACCOUNT_ID — ✗ нет" in block

    def test_stale_key_warns_with_restart_hint(self, memohood, monkeypatch):
        systemd_env = importlib.import_module(f"{memohood.__name__}.systemd_env")
        monkeypatch.setattr(systemd_env, "detect_systemd_gateway", lambda **kw: False)
        _sw(memohood).env_file_path().write_text(
            "CLOUDFLARE_API_TOKEN=cfat_ONLYFILE\n", encoding="utf-8"
        )
        block = _sw(memohood).format_keys_block(_cfg(memohood))
        assert "⚠ есть в .env" in block
        assert "перезапустите hermes" in block.lower()

    def test_never_prints_full_secret(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_DO_NOT_LEAK_TAIL")
        block = _sw(memohood).format_keys_block(_cfg(memohood))
        assert "DO_NOT_LEAK_TAIL" not in block


# ===========================================================================
# Wiring: memohood_stats includes the "Ключи" block
# ===========================================================================


class TestStatsIncludesKeys:
    def test_stats_appends_keys_block(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_STATSLIVE")
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        try:
            out = memohood.tools.memohood_stats({}, conn=conn, cfg=_cfg(memohood), session_id="s1")
        finally:
            conn.close()
        assert "Ключи (.env:" in out
        assert "GEMINI_API_KEY — ✓ настроен (AIza…)" in out

    def test_stats_keys_block_never_leaks_secret(self, memohood, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_STATS_SECRETTAIL")
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        try:
            out = memohood.tools.memohood_stats({}, conn=conn, cfg=_cfg(memohood), session_id="s1")
        finally:
            conn.close()
        assert "STATS_SECRETTAIL" not in out
