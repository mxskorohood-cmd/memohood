"""Tests for ``setup_wizard.py`` (``hermes memohood setup``).

Same isolation pattern as the rest of the suite (see ``conftest.py``): the
``memohood`` fixture gives a fresh package copy with HERMES_HOME monkeypatched
to a tmp dir and all credential env vars stripped. No test here ever makes
a live HTTP call -- the three ``check_*`` functions are monkeypatched on
the wizard module (they are looked up via module globals at call time, by
design), and the "skipped" flows assert they were NOT called at all.

B12: ``_run`` now also calls ``systemd_env.wire_gateway_env(home)`` right
before printing its final "Что дальше" hint. The autouse
``_default_no_systemd`` fixture below defaults every test in this module to
the safe "not systemd" no-op, so the suite never shells out to a real
``systemctl`` -- even a Linux CI box that happens to have the binary on
PATH but no user systemd session would otherwise pay a real (if harmless)
subprocess call on every single test that runs `run_wizard` to completion.
``TestB12SystemdGatewayHint`` overrides this per-test to exercise both
branches of the hint.
"""

from __future__ import annotations

import argparse
import importlib
import json
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


@pytest.fixture(autouse=True)
def _default_no_systemd(memohood, monkeypatch):
    """B12 safety net (see module docstring): every test in this file gets a
    mocked, guaranteed-no-op ``systemd_env.wire_gateway_env`` by default, so
    ``run_wizard`` reaching its final "Что дальше" hint never shells out to
    a real ``systemctl`` -- regardless of what happens to be on the test
    machine's PATH. ``TestB12SystemdGatewayHint`` overrides this per-test."""
    sw = _wizard(memohood)
    monkeypatch.setattr(
        sw.systemd_env,
        "wire_gateway_env",
        lambda home, **kwargs: (False, "не под systemd (test default)"),
    )


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

    @pytest.mark.parametrize(
        "value",
        ["AIza" + "x" * 35, "AIzaSyFakeFakeFakeFakeFake", "AQ." + "b" * 50],
    )
    def test_gemini_key_valid(self, memohood, value):
        # No provider-prefix requirement -- the legacy AIza... shape, the
        # newer AQ.... shape, and anything else plausible are all accepted;
        # the live check (not this format gate) is the real validator.
        assert _wizard(memohood).validate_gemini_key(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
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
        # B6: model discovery is a SEPARATE live call from check_gemini -- keep
        # this test network-free by making it come back empty (graceful
        # fallback to the default model, no extra input_fn call needed).
        monkeypatch.setattr(sw, "discover_gemini_models", lambda key: [])

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
# B4: incremental .env writes -- an EOFError/KeyboardInterrupt on a LATER
# step must never lose an EARLIER step's already-confirmed keys.
# ---------------------------------------------------------------------------


class TestB4IncrementalWrites:
    def test_eof_on_later_step_keeps_earlier_steps_keys(self, memohood, monkeypatch, capsys):
        """Cloudflare (step 1) completes and is confirmed; step 2 (Cohere)
        then hits EOFError (closed stdin) -- the wizard must exit calmly AND
        .env must still contain the step-1 keys instead of losing them."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", lambda a, t: (True, "ok"))
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", _fail_if_called)

        cf_account = "0123456789abcdef0123456789abcdef"
        cf_token = "cf-token-FAKE-000000"
        answers = iter([cf_account, cf_token, ""])  # step 1 only

        def input_fn(prompt):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        sw.run_wizard(hermes_home=str(home), input_fn=input_fn)  # must not raise

        env_text = (home / ".env").read_text(encoding="utf-8")
        assert f"CLOUDFLARE_ACCOUNT_ID={cf_account}" in env_text
        assert f"CLOUDFLARE_API_TOKEN={cf_token}" in env_text
        out = capsys.readouterr().out
        assert "hermes memohood setup" in out  # calm come-back-later message


# ---------------------------------------------------------------------------
# B8: live-check confirmation defaults. Gate 1 (offer a check) stays
# Enter = да; gate 2 (keep a key that just FAILED its check) now defaults
# to Enter = НЕ сохранять.
# ---------------------------------------------------------------------------


class TestB8LiveCheckDefaults:
    def test_failed_check_enter_declines_by_default(self, memohood, monkeypatch, capsys):
        """After a FAILED live check, pressing Enter on 'Сохранить?' must now
        mean 'не сохранять' -- the old code defaulted to saving the broken
        key, which is the bug this fixes."""
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
                    "",                 # Enter на "Сохранить всё равно?" -> НЕ сохранять (новый дефолт)
                    "",                 # шаг 3: пропуск Gemini
                ]
            ),
        )

        assert not (home / ".env").exists()
        out = capsys.readouterr().out
        assert "Проверка не прошла" in out
        assert "проверка не прошла" in out  # строка статуса в итогах

    def test_declining_the_check_warns_but_still_saves(self, memohood, monkeypatch, capsys):
        """Explicitly declining the live check (typing 'n' on gate 1) must
        print an explicit warning that the key goes in unverified, but the
        key is still saved -- gate 1's Enter-default is unchanged, only
        gate 2 flipped."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", _fail_if_called)
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)  # must NOT be called -- check declined

        cohere_key = "co-FAKE-unverified-key"
        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "",          # шаг 1: пропуск Cloudflare
                    cohere_key,  # шаг 2: ключ Cohere
                    "n",         # явный отказ от проверки
                    "",          # шаг 3: пропуск Gemini
                ]
            ),
        )

        env_text = (home / ".env").read_text(encoding="utf-8")
        assert f"COHERE_API_KEY={cohere_key}" in env_text
        out = capsys.readouterr().out
        assert "без проверки" in out
        assert "молча не заработает" in out


# ---------------------------------------------------------------------------
# B10: resume checkpoint (.memohood-setup.partial.json) -- step-completion
# flags only, never key values; lets a later run skip already-done steps.
# ---------------------------------------------------------------------------


class TestB10ResumeCheckpoint:
    def test_partial_file_created_after_a_step_holds_only_flags(self, memohood, monkeypatch):
        """After Cloudflare completes, the checkpoint file must exist, list
        "cloudflare" in steps_done, and contain NO key material at all --
        secrets stay in .env exclusively."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", lambda a, t: (True, "ok"))

        cf_account = "0123456789abcdef0123456789abcdef"
        cf_token = "cf-token-FAKE-000000"
        answers = iter([cf_account, cf_token, ""])

        def input_fn(prompt):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        sw.run_wizard(hermes_home=str(home), input_fn=input_fn)

        partial = home / ".memohood-setup.partial.json"
        assert partial.exists()
        raw = partial.read_text(encoding="utf-8")
        assert cf_account not in raw
        assert cf_token not in raw

        data = json.loads(raw)
        assert data == {"steps_done": ["cloudflare"]}

    def test_existing_partial_offers_resume_and_skips_done_step(self, memohood, monkeypatch, capsys):
        """A second run that finds an existing checkpoint offers to resume;
        answering Enter (= да) must skip re-asking Cloudflare entirely --
        its live check is never invoked again."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        partial = home / ".memohood-setup.partial.json"
        partial.write_text('{"steps_done": ["cloudflare"]}', encoding="utf-8")

        monkeypatch.setattr(sw, "check_cloudflare", _fail_if_called)  # must NOT be re-asked
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", _fail_if_called)

        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "",  # Enter = да, возобновить и пропустить готовые шаги
                    "",  # шаг 2: пропуск Cohere
                    "",  # шаг 3: пропуск Gemini
                ]
            ),
        )

        out = capsys.readouterr().out
        assert "Уже готово" in out
        assert "Cloudflare" in out

    def test_full_clean_run_deletes_partial_file(self, memohood, monkeypatch):
        """Once all three key steps + dependencies finish in a single clean
        run, the checkpoint file must be gone."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw, "check_cloudflare", lambda a, t: (True, "ok"))
        monkeypatch.setattr(sw, "check_cohere", lambda k: (True, "ok"))
        monkeypatch.setattr(sw, "check_gemini", lambda k: (True, "ok"))
        monkeypatch.setattr(sw, "discover_gemini_models", lambda key: [])

        cf_account = "0123456789abcdef0123456789abcdef"
        gemini_key = "AIzaFAKE" + "0" * 31
        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    cf_account, "cf-token-FAKE", "",  # шаг 1
                    "co-FAKE-key", "",                # шаг 2
                    gemini_key, "",                   # шаг 3
                ]
            ),
        )

        assert not (home / ".memohood-setup.partial.json").exists()


# ---------------------------------------------------------------------------
# B6: Gemini model auto-discovery (ListModels) after the key step.
# ---------------------------------------------------------------------------


class TestB6GeminiModelDiscovery:
    def test_discover_gemini_models_filters_and_strips_prefix(self, memohood, monkeypatch):
        """ListModels response is parsed -- only generateContent-capable
        models are kept, and the 'models/' name prefix is stripped."""
        sw = _wizard(memohood)

        class _FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash-lite",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-2.5-pro",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        {
                            "name": "models/embedding-001",
                            "supportedGenerationMethods": ["embedContent"],  # no generateContent -> filtered
                        },
                    ]
                }

        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured["url"] = url
            return _FakeResp()

        monkeypatch.setattr("requests.get", fake_get)

        models = sw.discover_gemini_models("fake-gemini-key")

        assert models == ["gemini-2.5-flash-lite", "gemini-2.5-pro"]
        assert "fake-gemini-key" in captured["url"]

    def test_discover_gemini_models_never_raises_on_network_failure(self, memohood, monkeypatch):
        """Graceful fallback: any network/parsing failure yields []."""
        sw = _wizard(memohood)

        def fake_get(url, headers=None, timeout=None):
            raise OSError("no network in this test")

        monkeypatch.setattr("requests.get", fake_get)

        assert sw.discover_gemini_models("fake-gemini-key") == []

    def test_wizard_offers_only_generatecontent_models_and_saves_choice(self, memohood, monkeypatch, capsys):
        """End-to-end: after the Gemini key is entered and confirmed, the
        wizard lists only generateContent-capable models; picking a
        NON-default one persists it to config.yaml's
        memory.memohood.model.model (never to .env)."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test

        monkeypatch.setattr(sw, "check_cloudflare", _fail_if_called)
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", lambda k: (True, "ok"))
        monkeypatch.setattr(
            sw,
            "discover_gemini_models",
            lambda key: ["gemini-2.5-flash-lite", "gemini-2.5-pro"],
        )

        gemini_key = "AIzaFAKE" + "0" * 31
        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "",          # шаг 1: пропуск Cloudflare
                    "",          # шаг 2: пропуск Cohere
                    gemini_key,  # шаг 3: ключ Gemini
                    "",          # Enter = проверить
                    "2",         # выбор модели: gemini-2.5-pro (не дефолт)
                ]
            ),
        )

        out = capsys.readouterr().out
        assert "gemini-2.5-flash-lite" in out
        assert "gemini-2.5-pro" in out

        config_path = home / "config.yaml"
        assert config_path.exists()

        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["memory"]["memohood"]["model"]["model"] == "gemini-2.5-pro"

        # Секрет не попадает в config.yaml.
        assert gemini_key not in config_path.read_text(encoding="utf-8")

    def test_wizard_keeps_default_model_when_discovery_empty(self, memohood, monkeypatch, capsys):
        """Graceful fallback end-to-end: ListModels comes back empty ->
        no prompt, no config.yaml write, default model stays in effect."""
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test

        monkeypatch.setattr(sw, "check_cloudflare", _fail_if_called)
        monkeypatch.setattr(sw, "check_cohere", _fail_if_called)
        monkeypatch.setattr(sw, "check_gemini", lambda k: (True, "ok"))
        monkeypatch.setattr(sw, "discover_gemini_models", lambda key: [])

        gemini_key = "AIzaFAKE" + "0" * 31
        sw.run_wizard(
            hermes_home=str(home),
            input_fn=_feed(
                [
                    "",          # шаг 1: пропуск Cloudflare
                    "",          # шаг 2: пропуск Cohere
                    gemini_key,  # шаг 3: ключ Gemini
                    "",          # Enter = проверить
                ]
            ),
        )

        out = capsys.readouterr().out
        assert "оставляю модель по умолчанию" in out
        assert not (home / "config.yaml").exists()


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


# ---------------------------------------------------------------------------
# B12: the final "Что дальше" hint branches on whether wire_gateway_env
# found (or wired) a systemd-hosted gateway. These three tests override the
# module's `_default_no_systemd` autouse fixture per-test to exercise both
# branches, still never touching a real systemctl.
# ---------------------------------------------------------------------------


class TestB12SystemdGatewayHint:
    def test_systemd_wired_prints_gateway_restart_hint(self, memohood, monkeypatch, capsys):
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw.systemd_env, "wire_gateway_env", lambda h, **kw: (True, "wired"))

        sw.run_wizard(hermes_home=str(home), input_fn=_feed(["", "", ""]))

        out = capsys.readouterr().out
        assert "hermes gateway restart" in out
        assert "Перезапустите hermes, чтобы он подхватил ключи из .env." not in out

    def test_non_systemd_keeps_original_restart_hint(self, memohood, monkeypatch, capsys):
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        monkeypatch.setattr(sw.systemd_env, "wire_gateway_env", lambda h, **kw: (False, "not systemd"))

        sw.run_wizard(hermes_home=str(home), input_fn=_feed(["", "", ""]))

        out = capsys.readouterr().out
        assert "Перезапустите hermes, чтобы он подхватил ключи из .env." in out
        assert "hermes gateway restart" not in out

    def test_wire_gateway_env_called_with_resolved_hermes_home(self, memohood, monkeypatch):
        sw = _wizard(memohood)
        home = memohood._hermes_home_for_test
        captured = {}

        def fake_wire(h, **kw):
            captured["home"] = h
            return (False, "not systemd")

        monkeypatch.setattr(sw.systemd_env, "wire_gateway_env", fake_wire)

        sw.run_wizard(hermes_home=str(home), input_fn=_feed(["", "", ""]))

        assert captured["home"] == str(home)
