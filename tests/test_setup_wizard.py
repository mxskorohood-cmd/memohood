"""Tests for ``setup_wizard.py`` (``hermes memohood setup``).

Same isolation pattern as the rest of the suite (see ``conftest.py``): the
``memohood`` fixture gives a fresh package copy with HERMES_HOME monkeypatched
to a tmp dir and all credential env vars stripped. No test here ever makes
a live HTTP call -- the three ``check_*`` functions are monkeypatched on
the wizard module (they are looked up via module globals at call time, by
design), and the "skipped" flows assert they were NOT called at all.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pytest


def _wizard(memohood):
    """Import the setup_wizard submodule of THIS test's fresh package copy."""
    return importlib.import_module(f"{memohood.__name__}.setup_wizard")


def _fail_if_called(*args, **kwargs):  # pragma: no cover - failure path only
    raise AssertionError("live check must not be called when the step is skipped")


def _feed(answers):
    """Build an input_fn that returns scripted answers one by one."""
    it = iter(answers)
    return lambda prompt: next(it)


# ---------------------------------------------------------------------------
# upsert_env_var
# ---------------------------------------------------------------------------


class TestUpsertEnvVar:
    def test_adds_new_variable_creating_the_file(self, memohood, tmp_path):
        sw = _wizard(memohood)
        env = tmp_path / "sub" / ".env"  # parent dir doesn't exist yet either
        action = sw.upsert_env_var(env, "GEMINI_API_KEY", "AIzaFakeValue123")
        assert action == "added"
        assert env.read_text(encoding="utf-8") == "GEMINI_API_KEY=AIzaFakeValue123\n"

    def test_appends_to_existing_file_without_touching_other_lines(self, memohood, tmp_path):
        sw = _wizard(memohood)
        env = tmp_path / ".env"
        env.write_text("TELEGRAM_TOKEN=abc\n", encoding="utf-8")
        action = sw.upsert_env_var(env, "COHERE_API_KEY", "co-fake")
        assert action == "added"
        assert env.read_text(encoding="utf-8") == "TELEGRAM_TOKEN=abc\nCOHERE_API_KEY=co-fake\n"

    def test_replaces_existing_line_in_place(self, memohood, tmp_path):
        sw = _wizard(memohood)
        env = tmp_path / ".env"
        env.write_text("A=1\nGEMINI_API_KEY=old-value\nB=2\n", encoding="utf-8")
        action = sw.upsert_env_var(env, "GEMINI_API_KEY", "new-value")
        assert action == "replaced"
        lines = env.read_text(encoding="utf-8").splitlines()
        assert lines == ["A=1", "GEMINI_API_KEY=new-value", "B=2"]
        assert "old-value" not in env.read_text(encoding="utf-8")

    def test_uncomments_a_commented_line_with_the_new_value(self, memohood, tmp_path):
        sw = _wizard(memohood)
        env = tmp_path / ".env"
        env.write_text("A=1\n# COHERE_API_KEY=stale\nB=2\n", encoding="utf-8")
        action = sw.upsert_env_var(env, "COHERE_API_KEY", "fresh")
        assert action == "uncommented"
        lines = env.read_text(encoding="utf-8").splitlines()
        assert lines == ["A=1", "COHERE_API_KEY=fresh", "B=2"]

    def test_active_line_wins_over_commented_one(self, memohood, tmp_path):
        """If both `# KEY=` and `KEY=` exist, only the active line is
        rewritten; the comment stays put (it may be a human's note)."""
        sw = _wizard(memohood)
        env = tmp_path / ".env"
        env.write_text("# GEMINI_API_KEY=note\nGEMINI_API_KEY=old\n", encoding="utf-8")
        action = sw.upsert_env_var(env, "GEMINI_API_KEY", "new")
        assert action == "replaced"
        lines = env.read_text(encoding="utf-8").splitlines()
        assert lines == ["# GEMINI_API_KEY=note", "GEMINI_API_KEY=new"]

    def test_utf8_content_survives_a_roundtrip(self, memohood, tmp_path):
        sw = _wizard(memohood)
        env = tmp_path / ".env"
        env.write_text(
            "# Ключи облаков -- не коммитить\nCOHERE_API_KEY=старое-значение\n",
            encoding="utf-8",
        )
        sw.upsert_env_var(env, "COHERE_API_KEY", "новое-значение")
        text = env.read_text(encoding="utf-8")
        assert "# Ключи облаков -- не коммитить" in text
        assert "COHERE_API_KEY=новое-значение" in text
        assert "старое-значение" not in text


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    @pytest.mark.parametrize(
        "value",
        [
            "0123456789abcdef0123456789abcdef",
            "0123456789ABCDEF0123456789ABCDEF",  # uppercase hex accepted
            "  0123456789abcdef0123456789abcdef  ",  # surrounding whitespace stripped
        ],
    )
    def test_cf_account_id_valid(self, memohood, value):
        assert _wizard(memohood).validate_cf_account_id(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "zzzz56789abcdef0123456789abcdefz",  # non-hex chars
            "0123456789abcdef0123456789abcde",  # 31 chars
            "0123456789abcdef0123456789abcdef0",  # 33 chars
            "0123456789abcdef 123456789abcdef",  # inner space
        ],
    )
    def test_cf_account_id_garbage(self, memohood, value):
        assert _wizard(memohood).validate_cf_account_id(value) is False

    @pytest.mark.parametrize("value", ["sk-abc123", "x", "co-FAKE-0000"])
    def test_api_token_valid(self, memohood, value):
        assert _wizard(memohood).validate_api_token(value) is True

    @pytest.mark.parametrize("value", ["", "   ", "has space", "tab\tchar"])
    def test_api_token_garbage(self, memohood, value):
        assert _wizard(memohood).validate_api_token(value) is False

    @pytest.mark.parametrize("value", ["AIza" + "x" * 35, "AIzaSyFakeFakeFakeFakeFake"])
    def test_gemini_key_valid(self, memohood, value):
        assert _wizard(memohood).validate_gemini_key(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "sk-not-a-gemini-key-000000000000",  # wrong prefix
            "AIza",  # too short
            "AIza key with spaces 000000000000",  # whitespace
        ],
    )
    def test_gemini_key_garbage(self, memohood, value):
        assert _wizard(memohood).validate_gemini_key(value) is False


# ---------------------------------------------------------------------------
# mask_key
# ---------------------------------------------------------------------------


class TestMaskKey:
    @pytest.mark.parametrize(
        "value",
        ["", "a", "abcd", "abcde", "AIzaSyFakeFakeFakeFakeFake", "x" * 200],
    )
    def test_never_returns_the_full_key(self, memohood, value):
        masked = _wizard(memohood).mask_key(value)
        assert masked != value
        if value:
            assert value not in masked

    def test_shows_first_four_chars_and_ellipsis(self, memohood):
        assert _wizard(memohood).mask_key("AIzaSyFake") == "AIza…"

    def test_short_values_collapse_to_bare_ellipsis(self, memohood):
        """Anything <= 4 chars must not leak a single char -- showing 4 of 4
        would be the whole secret."""
        assert _wizard(memohood).mask_key("abcd") == "…"


# ---------------------------------------------------------------------------
# Full wizard flow (mocked input, mocked live checks, no network)
# ---------------------------------------------------------------------------


class TestWizardFlow:
    def test_all_steps_skipped_env_untouched(self, memohood, monkeypatch, capsys):
        """Three Enters (CF account id / Cohere key / Gemini key) skip every
        service step; .env must not even be created, and no live check runs."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        for name in ("check_cloudflare", "check_cohere", "check_gemini"):
            monkeypatch.setattr(sw, name, _fail_if_called)

        sw.run_wizard(hermes_home=str(home), input_fn=_feed(["", "", ""]))

        assert not (home / ".env").exists()
        out = capsys.readouterr().out
        assert "не тронут" in out
        assert "пропущено" in out.lower()

    def test_all_keys_entered_env_contains_everything(self, memohood, monkeypatch, capsys):
        """Full happy path: every key entered, every live check accepted
        (Enter = да) and mocked green -> .env holds all four vars, each
        check ran exactly once, and no full key ever hit the console."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test

        cf_account = "0123456789abcdef0123456789abcdef"
        cf_token = "cf-token-FAKE-000000"
        cohere_key = "co-FAKE-key-000000"
        gemini_key = "AIzaFAKE" + "0" * 31

        calls = []
        monkeypatch.setattr(sw, "check_cloudflare", lambda a, t: (calls.append("cf"), (True, "ok"))[1])
        monkeypatch.setattr(sw, "check_cohere", lambda k: (calls.append("cohere"), (True, "ok"))[1])
        monkeypatch.setattr(sw, "check_gemini", lambda k: (calls.append("gemini"), (True, "ok"))[1])

        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    cf_account, cf_token, "",  # шаг 1: id, токен, Enter = проверить
                    cohere_key, "",            # шаг 2: ключ, Enter = проверить
                    gemini_key, "",            # шаг 3: ключ, Enter = проверить
                ]
            ),
        )

        env_text = (home / ".env").read_text(encoding="utf-8")
        assert f"CLOUDFLARE_ACCOUNT_ID={cf_account}" in env_text
        assert f"CLOUDFLARE_API_TOKEN={cf_token}" in env_text
        assert f"COHERE_API_KEY={cohere_key}" in env_text
        assert f"GEMINI_API_KEY={gemini_key}" in env_text
        assert calls == ["cf", "cohere", "gemini"]

        # Секреты никогда не печатаются целиком -- только маска.
        out = capsys.readouterr().out
        for secret in (cf_token, cohere_key, gemini_key):
            assert secret not in out
        assert "AIza…" in out  # маска Gemini-ключа присутствует

    def test_failed_check_and_decline_writes_nothing(self, memohood, monkeypatch, capsys):
        """Cohere key entered, live check fails, user answers 'n' to 'save
        anyway' -> nothing is written; other steps skipped."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", _fail_if_called)
        monkeypatch.setattr(sw, "check_cohere", lambda k: (False, "HTTP 401: unauthorized"))

        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "",                 # шаг 1: пропуск Cloudflare
                    "co-FAKE-bad-key",  # шаг 2: ключ Cohere
                    "",                 # Enter = проверить
                    "n",                # проверка упала -> не сохранять
                    "",                 # шаг 3: пропуск Gemini
                ]
            ),
        )

        assert not (home / ".env").exists()
        out = capsys.readouterr().out
        assert "Проверка не прошла" in out
        assert "проверка не прошла" in out  # строка статуса в итогах

    def test_ctrl_c_is_graceful(self, memohood, capsys):
        """KeyboardInterrupt mid-flow prints the 'come back later' hint
        instead of a traceback, and .env stays untouched."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test

        def boom(prompt):
            raise KeyboardInterrupt

        sw.run_wizard(hermes_home=str(home), input_fn=boom)  # must not raise

        assert not (home / ".env").exists()
        out = capsys.readouterr().out
        assert "hermes memohood setup" in out

    def test_invalid_then_valid_account_id_reasks(self, memohood, monkeypatch, capsys):
        """Garbage account id is re-asked (not accepted, not fatal); a valid
        one on the second try proceeds to the token prompt."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", lambda a, t: (True, "ok"))
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", _fail_if_called)

        cf_account = "0123456789abcdef0123456789abcdef"
        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "not-a-real-id",  # мусор -> переспросить
                    cf_account,       # валидный id
                    "cf-token-FAKE",  # токен
                    "",               # Enter = проверить
                    "",               # шаг 2: пропуск
                    "",               # шаг 3: пропуск
                ]
            ),
        )

        assert "32 символа" in capsys.readouterr().out
        env_text = (home / ".env").read_text(encoding="utf-8")
        assert f"CLOUDFLARE_ACCOUNT_ID={cf_account}" in env_text


# ---------------------------------------------------------------------------
# Dependencies + CLI wiring
# ---------------------------------------------------------------------------


class TestDependenciesAndCli:
    def test_check_dependencies_covers_the_plugin_yaml_list(self, memohood):
        sw = _wizard(memohood)
        results = {pip_name: found for _imp, pip_name, found in sw.check_dependencies()}
        assert set(results) == {"sqlite-vec", "PyStemmer", "ftfy", "requests"}
        # requests точно стоит в venv -- им пользуется сам плагин (и тесты).
        assert results["requests"] is True

    def test_cli_setup_subcommand_parses_and_dispatches(self, memohood, monkeypatch):
        """`hermes memohood setup` parses through register_cli's tree and
        memohood_command routes it into run_wizard with the resolved HERMES_HOME
        (the tmp one, thanks to the memohood fixture)."""
        sw = _wizard(memohood)
        called = {}
        monkeypatch.setattr(sw, "run_wizard", lambda hermes_home=None, **kw: called.setdefault("home", hermes_home))

        parser = argparse.ArgumentParser(prog="hermes memohood")
        memohood.cli.register_cli(parser)
        args = parser.parse_args(["setup"])
        assert args.memohood_subcommand == "setup"

        memohood.cli.memohood_command(args)
        assert Path(called["home"]) == memohood._hermes_home_for_test
