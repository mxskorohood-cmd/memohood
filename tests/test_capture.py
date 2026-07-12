"""capture.py: two-stage signal scoring, extraction gate, supersede tiers,
pinned tier, secret scrubbing, process_turn isolation."""

from __future__ import annotations

import copy

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("extract_llm.extract must NOT be called for this input")


def _count_active_captures(conn):
    return conn.execute("SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL").fetchone()["n"]


class TestSignalScoring:
    def test_definite_keep_no_llm_call(self, memohood, monkeypatch):
        monkeypatch.setattr(memohood.extract_llm, "extract", _fail_if_called)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        result = memohood.capture.extract_and_store(
            conn, "Запомни, что мой email work@example.com", side="user",
            session_id="s1", cfg=_cfg(memohood),
        )
        assert result is not None
        assert result["capture_id"]
        assert _count_active_captures(conn) == 1
        conn.close()

    def test_definite_drop_no_llm_call(self, memohood, monkeypatch):
        monkeypatch.setattr(memohood.extract_llm, "extract", _fail_if_called)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        result = memohood.capture.extract_and_store(
            conn, "привет как дела сегодня", side="user", session_id="s1", cfg=_cfg(memohood),
        )
        assert result is None
        assert _count_active_captures(conn) == 0
        conn.close()

    def test_borderline_band_calls_extract_exactly_once(self, memohood, monkeypatch):
        calls = []

        def fake_extract(text, *, conn=None):
            calls.append(text)
            return {
                "is_memorable": True, "kind": "preference",
                "notability": "medium", "source_type": "EXTRACTED", "pinned": False,
            }

        monkeypatch.setattr(memohood.extract_llm, "extract", fake_extract)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        # "я предпочитаю" matches only the preference pattern (weight 2.0),
        # which is below capture_threshold=4.0 -> borderline band.
        sig = memohood.capture.compute_signals("я предпочитаю тёмную тему интерфейса", side="user")
        assert 0 < sig["score"] < 4.0
        result = memohood.capture.extract_and_store(
            conn, "я предпочитаю тёмную тему интерфейса", side="user",
            session_id="s1", cfg=_cfg(memohood),
        )
        assert len(calls) == 1
        assert result is not None
        conn.close()

    def test_borderline_not_memorable_drops(self, memohood, monkeypatch):
        def fake_extract(text, *, conn=None):
            return {
                "is_memorable": False, "kind": "fact",
                "notability": "low", "source_type": "INFERRED", "pinned": False,
            }

        monkeypatch.setattr(memohood.extract_llm, "extract", fake_extract)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        result = memohood.capture.extract_and_store(
            conn, "я предпочитаю тёмную тему интерфейса", side="user",
            session_id="s1", cfg=_cfg(memohood),
        )
        assert result is None
        conn.close()

    def test_pinned_trigger_marks_pinned(self, memohood, monkeypatch):
        monkeypatch.setattr(memohood.extract_llm, "extract", _fail_if_called)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        result = memohood.capture.extract_and_store(
            conn, "меня зовут Максим", side="user", session_id="s1", cfg=_cfg(memohood),
        )
        assert result is not None
        row = conn.execute("SELECT * FROM captures WHERE id=?", (result["capture_id"],)).fetchone()
        assert row["pinned"] == 1
        conn.close()

    def test_empty_text_returns_none(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        assert memohood.capture.extract_and_store(conn, "   ", side="user", cfg=_cfg(memohood)) is None
        assert memohood.capture.extract_and_store(conn, "", side="user", cfg=_cfg(memohood)) is None
        conn.close()


class TestSecretScrubbing:
    def test_secret_is_redacted_before_storage(self, memohood, monkeypatch):
        monkeypatch.setattr(memohood.extract_llm, "extract", _fail_if_called)
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        fake_key = "sk-proj-" + ("a" * 40)
        result = memohood.capture.extract_and_store(
            conn, f"запомни, мой ключ {fake_key}", side="user", session_id="s1", cfg=_cfg(memohood),
        )
        assert result is not None
        row = conn.execute("SELECT content FROM captures WHERE id=?", (result["capture_id"],)).fetchone()
        assert fake_key not in row["content"]
        assert "[REDACTED]" in row["content"]
        conn.close()


class TestSupersede:
    def test_duplicate_via_fts_fallback_no_new_row(self, memohood):
        """Without embedder credentials, capture.py falls back to FTS/Jaccard
        dup detection. Storing the exact same content twice via
        manual_capture (which skips the gate, exercising steps 3-6 directly)
        should be recognized as a duplicate: no new row, old row's
        last_seen_at bumped."""
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        text = "мы решили использовать PostgreSQL для основной базы данных проекта"
        first = memohood.capture.manual_capture(conn, text, kind="decision", session_id="s1", cfg=_cfg(memohood))
        second = memohood.capture.manual_capture(conn, text, kind="decision", session_id="s1", cfg=_cfg(memohood))
        assert second["action"] == "duplicate"
        assert second["capture_id"] == first["capture_id"]
        assert _count_active_captures(conn) == 1
        conn.close()

    def test_supersede_judge_path_invalidates_old_and_writes_history(self, memohood, monkeypatch):
        """Directly exercise the 0.92-0.95 ambiguous band by monkeypatching
        _nearest_captures to return a fixed mid-band candidate, and
        extract_llm.judge to say 'supersede'."""
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        old = memohood.capture.manual_capture(
            conn, "мы используем MySQL", kind="decision", notability="medium",
            pinned=False, session_id="s1", cfg=_cfg(memohood),
        )
        old_id = old["capture_id"]

        def fake_nearest(conn_, content, cfg, *, k=5):
            return [{"id": old_id, "content": "мы используем MySQL", "cosine": 0.93}], None

        def fake_judge(new_content, candidates, *, conn=None):
            return {"action": "supersede", "supersedes_id": old_id, "reasoning": "updated decision"}

        monkeypatch.setattr(memohood.capture, "_nearest_captures", fake_nearest)
        monkeypatch.setattr(memohood.extract_llm, "judge", fake_judge)

        new = memohood.capture._store_capture(
            conn, "мы используем PostgreSQL", kind="decision", notability="medium",
            source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
        )
        assert new["action"] == "supersede"
        assert new["supersedes"] == old_id

        old_row = conn.execute("SELECT * FROM captures WHERE id=?", (old_id,)).fetchone()
        assert old_row["invalidated_at"] is not None

        new_row = conn.execute("SELECT * FROM captures WHERE id=?", (new["capture_id"],)).fetchone()
        assert "мы используем MySQL" in new_row["history"]
        assert _count_active_captures(conn) == 1  # only the new row is active
        conn.close()

    def test_independent_below_cosine_floor(self, memohood, monkeypatch):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))

        def fake_nearest(conn_, content, cfg, *, k=5):
            return [{"id": "unrelated-id", "content": "нечто совсем другое", "cosine": 0.10}], None

        def fail_judge(*a, **kw):
            raise AssertionError("judge() must not be called below the 0.92 floor")

        monkeypatch.setattr(memohood.capture, "_nearest_captures", fake_nearest)
        monkeypatch.setattr(memohood.extract_llm, "judge", fail_judge)

        result = memohood.capture._store_capture(
            conn, "новый независимый факт", kind="fact", notability="medium",
            source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
        )
        assert result["action"] == "independent"
        conn.close()


class TestManualCapture:
    def test_empty_content_raises(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        with pytest.raises(ValueError):
            memohood.capture.manual_capture(conn, "   ", cfg=_cfg(memohood))
        conn.close()

    def test_content_that_is_only_a_secret_is_stored_redacted_not_rejected(self, memohood):
        """_scrub_secrets() REPLACES a matched span with the literal
        "[REDACTED]" (not an empty string) -- so content consisting solely
        of a secret is never actually "entirely blank" after redaction, and
        manual_capture's ValueError-on-empty path is unreachable via this
        route (only a whitespace-only *input* triggers it). Documents the
        actual behavior: the secret itself never reaches storage, but a
        placeholder capture is still written rather than the call raising."""
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        fake_key = "sk-ant-" + ("b" * 30)
        result = memohood.capture.manual_capture(conn, fake_key, cfg=_cfg(memohood))
        row = conn.execute("SELECT content FROM captures WHERE id=?", (result["capture_id"],)).fetchone()
        assert fake_key not in row["content"]
        assert "[REDACTED]" in row["content"]
        conn.close()


class TestProcessTurnIsolation:
    def test_one_side_failure_does_not_block_the_other(self, memohood, monkeypatch):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        real_extract_and_store = memohood.capture.extract_and_store

        def flaky(conn_, text, *, side="user", session_id="", cfg=None):
            if side == "user":
                raise RuntimeError("boom")
            return real_extract_and_store(conn_, text, side=side, session_id=session_id, cfg=cfg)

        monkeypatch.setattr(memohood.capture, "extract_and_store", flaky)
        # capture.py's single assistant-side signal pattern is weight 3.0,
        # below capture_threshold=4.0 on its own -- "важно: обязательно
        # завершить проект" only scores 3.0 (borderline band), so mock the
        # LLM gate rather than crafting a multi-pattern-match string.
        monkeypatch.setattr(
            memohood.extract_llm, "extract",
            lambda text, *, conn=None: {
                "is_memorable": True, "kind": "fact", "notability": "medium",
                "source_type": "EXTRACTED", "pinned": False,
            },
        )
        results = memohood.capture.process_turn(
            conn, "любой текст пользователя", "важно: обязательно завершить проект вовремя",
            session_id="s1", cfg=_cfg(memohood),
        )
        assert results["user"] is None
        assert results["assistant"] is not None
        conn.close()
