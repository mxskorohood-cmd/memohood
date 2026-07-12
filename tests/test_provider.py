"""MemoHoodMemoryProvider lifecycle: initialize, prefetch (+recall text +
reinforce), child/delegated session hard no-op, sync_turn (background
thread), on_pre_compress rescue."""

from __future__ import annotations

import time

import pytest


def _make_provider(memohood, *, agent_context=None, parent_session_id=""):
    p = memohood.provider.MemoHoodMemoryProvider()
    kwargs = {"hermes_home": str(memohood._hermes_home_for_test), "platform": "cli"}
    if agent_context is not None:
        kwargs["agent_context"] = agent_context
    if parent_session_id:
        kwargs["parent_session_id"] = parent_session_id
    p.initialize("s1", **kwargs)
    return p


class TestInitialize:
    def test_initialize_opens_db_and_runs_catch_up(self, memohood):
        p = _make_provider(memohood)
        assert p._conn is not None
        assert (memohood._hermes_home_for_test / "memory.db").exists()
        p.shutdown()

    def test_missing_hermes_home_does_not_crash(self, memohood):
        p = memohood.provider.MemoHoodMemoryProvider()
        p.initialize("s1", platform="cli")  # no hermes_home kwarg at all
        assert p._conn is None
        # Every method must degrade gracefully, never raise.
        assert p.prefetch("что угодно") == ""
        assert p.system_prompt_block() == ""
        assert p.get_tool_schemas() == []


class TestChildSessionHardNoOp:
    @pytest.mark.parametrize("agent_context", ["subagent", "cron", "flush"])
    def test_agent_context_child_is_hard_noop(self, memohood, agent_context):
        p = _make_provider(memohood, agent_context=agent_context)
        assert p._is_child is True
        assert p.prefetch("что угодно") == ""
        assert p.system_prompt_block() == ""
        assert p.get_tool_schemas() == []
        p.shutdown()

    def test_parent_session_id_alone_marks_child(self, memohood):
        p = _make_provider(memohood, parent_session_id="parent-123")
        assert p._is_child is True
        assert p.prefetch("что угодно") == ""
        p.shutdown()

    def test_primary_context_is_not_child(self, memohood):
        p = _make_provider(memohood, agent_context="primary")
        assert p._is_child is False
        p.shutdown()


class TestPrefetchRecall:
    def test_prefetch_returns_recall_text_and_reinforces(self, memohood):
        p = _make_provider(memohood)
        result = memohood.capture.manual_capture(
            p._conn, "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s1", cfg=p._cfg,
        )
        cid = result["capture_id"]
        before = p._conn.execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]

        time.sleep(0.01)
        text = p.prefetch("расскажи про договора с поставщиком", session_id="s1")
        assert "договор" in text.lower()

        after = p._conn.execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]
        assert after >= before
        p.shutdown()

    def test_prefetch_empty_query_returns_empty(self, memohood):
        p = _make_provider(memohood)
        assert p.prefetch("") == ""
        assert p.prefetch("   ") == ""
        p.shutdown()

    def test_prefetch_no_hits_returns_empty(self, memohood):
        p = _make_provider(memohood)
        assert p.prefetch("совершенно несвязанный запрос без совпадений") == ""
        p.shutdown()


class TestSyncTurn:
    def test_sync_turn_is_background_and_persists(self, memohood):
        p = _make_provider(memohood)
        p.sync_turn(
            "запомни, что мой любимый редактор - VS Code",
            "хорошо, запомнил",
            session_id="s1",
        )
        p._join_background(timeout=5.0)
        row = p._conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] >= 1
        p.shutdown()

    def test_sync_turn_noop_for_child_session(self, memohood):
        p = _make_provider(memohood, agent_context="subagent")
        p.sync_turn("запомни это навсегда", "ок", session_id="s1")
        p._join_background(timeout=2.0)
        # initialize() still opens a connection for a child context (only
        # catch_up_from_state is skipped) -- sync_turn's own is_child guard
        # is what must prevent any capture from being written.
        assert p._conn is not None
        row = p._conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 0
        p.shutdown()

    def test_sync_turn_respects_auto_capture_false(self, memohood):
        p = _make_provider(memohood)
        p._cfg["auto_capture"] = False
        p.sync_turn("запомни, что мой любимый редактор - VS Code", "ок", session_id="s1")
        p._join_background(timeout=2.0)
        row = p._conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 0
        p.shutdown()


class TestOnPreCompress:
    def test_rescues_definite_keep_insight_without_llm_call(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        monkeypatch.setattr(
            memohood.extract_llm, "extract",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no LLM call during compression")),
        )
        messages = [
            {"role": "user", "content": "просто болтовня ни о чём"},
            {"role": "user", "content": "Запомни навсегда: имя проекта - Феникс"},
        ]
        summary_note = p.on_pre_compress(messages)
        assert "Феникс" in summary_note or "rescued" in summary_note.lower()
        row = p._conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 1
        p.shutdown()

    def test_empty_messages_returns_empty_string(self, memohood):
        p = _make_provider(memohood)
        assert p.on_pre_compress([]) == ""
        p.shutdown()


class TestToolSchemas:
    def test_get_tool_schemas_returns_all_when_primary(self, memohood):
        p = _make_provider(memohood)
        schemas = p.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert {"memohood_search", "memohood_fetch", "memohood_recall", "memohood_stats", "memohood_capture"} <= names
        p.shutdown()

    def test_handle_tool_call_dispatches(self, memohood):
        p = _make_provider(memohood)
        result = p.handle_tool_call("memohood_capture", {"content": "тестовый факт для инструмента"}, session_id="s1")
        assert "Сохранено" in result or "id=" in result
        p.shutdown()
