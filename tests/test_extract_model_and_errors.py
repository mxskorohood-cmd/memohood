"""extract_llm.py: B2/B9 (``memory.memohood.model.model`` actually threads
through to the Gemini call body via :func:`extract_llm.resolve_model` and
its three call-sites in capture.py/consolidate.py), B3 (``DEFAULT_MODEL``
stays in lockstep with config.py's ``DEFAULTS`` and provider.py's setup
schema), and B5 (:class:`extract_llm.ExtractError` carries a structured
``status_code`` so a config-shaped failure (401/403/404) can be logged
differently from a transient network/rate-limit one, without changing the
degrade-to-safe-default contract)."""

from __future__ import annotations

import copy
import logging
import uuid

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


class _FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# B2/B9 -- resolve_model() itself
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_returns_configured_value(self, memohood):
        assert memohood.extract_llm.resolve_model({"model": {"model": "custom-x"}}) == "custom-x"

    def test_falls_back_to_default_on_empty_or_none_cfg(self, memohood):
        assert memohood.extract_llm.resolve_model({}) == memohood.extract_llm.DEFAULT_MODEL
        assert memohood.extract_llm.resolve_model(None) == memohood.extract_llm.DEFAULT_MODEL

    def test_falls_back_to_default_when_model_key_malformed(self, memohood):
        # "model" present but not a dict, or a dict missing "model" -- never
        # crash, always degrade to DEFAULT_MODEL.
        assert memohood.extract_llm.resolve_model({"model": "not-a-dict"}) == memohood.extract_llm.DEFAULT_MODEL
        assert memohood.extract_llm.resolve_model({"model": {}}) == memohood.extract_llm.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# B2/B9 -- the three call-sites (capture.extract_and_store, capture._store_capture's
# judge() call, consolidate._rollup_level's summarize() call) actually pass
# resolve_model(cfg) through to _call_gemini's `model` kwarg.
# ---------------------------------------------------------------------------


class TestModelThreadedFromCallSites:
    def test_extract_and_store_passes_configured_model(self, memohood, monkeypatch):
        seen_models = []

        def fake_call_gemini(system_prompt, user_content, *, model=None, timeout=None, max_retries=None, conn=None):
            seen_models.append(model)
            return {
                "is_memorable": True, "kind": "fact",
                "notability": "low", "source_type": "EXTRACTED", "pinned": False,
            }

        monkeypatch.setattr(memohood.extract_llm, "_call_gemini", fake_call_gemini)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        cfg["model"]["model"] = "custom-extract"
        # "я предпочитаю" scores 2.0 (preference pattern) -- below the 4.0
        # default capture_threshold -> borderline band -> extract_llm.extract().
        memohood.capture.extract_and_store(
            conn, "я предпочитаю тёмную тему интерфейса", side="user", session_id="s1", cfg=cfg,
        )
        assert seen_models == ["custom-extract"]
        conn.close()

    def test_extract_and_store_uses_default_model_on_empty_cfg(self, memohood, monkeypatch):
        seen_models = []

        def fake_call_gemini(system_prompt, user_content, *, model=None, timeout=None, max_retries=None, conn=None):
            seen_models.append(model)
            return {
                "is_memorable": False, "kind": "fact",
                "notability": "low", "source_type": "INFERRED", "pinned": False,
            }

        monkeypatch.setattr(memohood.extract_llm, "_call_gemini", fake_call_gemini)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        memohood.capture.extract_and_store(
            conn, "я предпочитаю тёмную тему интерфейса", side="user", session_id="s1", cfg={},
        )
        assert seen_models == [memohood.extract_llm.DEFAULT_MODEL]
        conn.close()

    def test_judge_call_site_passes_configured_model(self, memohood, monkeypatch):
        seen_models = []

        def fake_call_gemini(system_prompt, user_content, *, model=None, timeout=None, max_retries=None, conn=None):
            seen_models.append(model)
            return {"action": "independent", "supersedes_id": None, "reasoning": "ok"}

        monkeypatch.setattr(memohood.extract_llm, "_call_gemini", fake_call_gemini)

        def fake_nearest(conn_, content, cfg_, *, k=5):
            return [{"id": "old-id", "content": "старый факт", "cosine": 0.93}], None

        monkeypatch.setattr(memohood.capture, "_nearest_captures", fake_nearest)

        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        cfg["model"]["model"] = "custom-judge"
        memohood.capture._store_capture(
            conn, "новый независимый факт", kind="fact", notability="medium", source="EXTRACTED",
            pinned=False, session_id="s1", cfg=cfg,
        )
        assert seen_models == ["custom-judge"]
        conn.close()

    def test_rollup_summarize_call_site_passes_configured_model(self, memohood, monkeypatch):
        seen_models = []

        def fake_call_gemini(system_prompt, user_content, *, model=None, timeout=None, max_retries=None, conn=None):
            seen_models.append(model)
            return {"summary": "итог"}

        monkeypatch.setattr(memohood.extract_llm, "_call_gemini", fake_call_gemini)

        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        cfg["model"]["model"] = "custom-summarize"
        now = memohood.db.now()
        old = now - 2 * 86400.0  # older than the 1-day "day" rollup cutoff
        for i in range(6):  # >= _ROLLUP_MIN_CAPTURES (5)
            conn.execute(
                """
                INSERT INTO captures(
                    id, content, kind, confidence, notability, source, pinned,
                    supersedes, history, session_id, message_id, tags, last_seen_at,
                    created_at, updated_at, valid_from, invalidated_at, embed_signature
                ) VALUES (?, ?, 'fact', 1.0, 'medium', 'EXTRACTED', 0, '', '', 's1', NULL, '', ?, ?, ?, ?, NULL, NULL)
                """,
                (uuid.uuid4().hex, f"факт номер {i}", old, old, old, old),
            )
        conn.commit()

        result = memohood.consolidate.run_rollup(conn, cfg)
        assert result["day"] == 1
        assert seen_models == ["custom-summarize"]
        conn.close()


# ---------------------------------------------------------------------------
# B3 -- DEFAULT_MODEL stays in lockstep with config.py DEFAULTS and
# provider.py's setup schema, and is the expected bumped value.
# ---------------------------------------------------------------------------


class TestDefaultModelInvariant:
    def test_default_model_is_the_expected_bumped_value(self, memohood):
        assert memohood.extract_llm.DEFAULT_MODEL == "gemini-3.1-flash-lite"

    def test_default_model_matches_config_defaults(self, memohood):
        assert memohood.extract_llm.DEFAULT_MODEL == memohood.config.DEFAULTS["model"]["model"]

    def test_default_model_matches_provider_setup_schema(self, memohood):
        provider = memohood.provider.MemoHoodMemoryProvider()
        schema = provider.get_config_schema()
        entry = next(e for e in schema if e["key"] == "model.model")
        assert entry["default"] == memohood.extract_llm.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# B5 -- ExtractError.status_code: config-shaped (401/403/404, no retry) vs
# network-shaped (None, after retries exhausted) failures are distinguishable.
# ---------------------------------------------------------------------------


class TestStructuredExtractError:
    @pytest.mark.parametrize("status", [403, 404])
    def test_call_gemini_raises_with_status_code_and_does_not_retry(self, memohood, monkeypatch, status):
        import requests

        calls = []

        def fake_post(*a, **kw):
            calls.append(1)
            return _FakeResponse(status_code=status, text="denied")

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(requests, "post", fake_post)

        with pytest.raises(memohood.extract_llm.ExtractError) as exc_info:
            memohood.extract_llm._call_gemini("sys", "user")
        assert exc_info.value.status_code == status
        assert len(calls) == 1, "401/403/404 are not in _RETRYABLE_STATUS -- must not be retried"

    def test_network_failure_raises_with_status_code_none(self, memohood, monkeypatch):
        import requests

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("boom")),
        )
        monkeypatch.setattr(memohood.extract_llm.time, "sleep", lambda *a, **kw: None)

        with pytest.raises(memohood.extract_llm.ExtractError) as exc_info:
            memohood.extract_llm._call_gemini("sys", "user", max_retries=1)
        assert exc_info.value.status_code is None

    def test_extract_degrades_to_none_on_404_and_logs_config_hint(self, memohood, monkeypatch, caplog):
        import requests

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(status_code=404, text="not found"))

        with caplog.at_level(logging.WARNING, logger="memohood.extract_llm"):
            result = memohood.extract_llm.extract("любой текст диалога", conn=None)

        assert result is None
        assert "GEMINI_API_KEY" in caplog.text
        assert "memory.memohood.model.model" in caplog.text

    def test_extract_degrades_to_none_on_network_error_without_config_hint(self, memohood, monkeypatch, caplog):
        import requests

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("boom")),
        )
        monkeypatch.setattr(memohood.extract_llm.time, "sleep", lambda *a, **kw: None)

        with caplog.at_level(logging.INFO, logger="memohood.extract_llm"):
            result = memohood.extract_llm.extract("любой текст диалога", conn=None)

        assert result is None
        # A plain network failure must degrade via the ordinary info-level
        # log, not the 401/403/404 config-hint warning.
        assert not any(record.levelno == logging.WARNING for record in caplog.records)

    def test_existing_400_adversarial_case_still_degrades_to_none(self, memohood, monkeypatch):
        """Guards test_adversarial.py::test_extract_degrades_on_http_error's
        contract: HTTP 400 is neither retried nor treated as a
        401/403/404 config hint, but still degrades to None like before."""
        import requests

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(status_code=400, text="bad request"))
        result = memohood.extract_llm.extract("любой текст диалога", conn=None)
        assert result is None
