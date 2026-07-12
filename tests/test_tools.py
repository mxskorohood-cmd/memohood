"""tools.py: dispatch(), memohood_capture/memohood_stats/memohood_search/memohood_fetch/memohood_recall."""

from __future__ import annotations

import copy


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def test_dispatch_unknown_tool_returns_error_string_not_raise(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    result = memohood.tools.dispatch("not_a_real_tool", {}, conn=conn, cfg=_cfg(memohood), session_id="s1")
    assert "неизвестный инструмент" in result.lower()
    conn.close()


def test_memohood_capture_then_memohood_stats(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    cfg = _cfg(memohood)
    result = memohood.tools.memohood_capture({"content": "тестовый факт", "kind": "fact"}, conn=conn, cfg=cfg, session_id="s1")
    assert "id=" in result or "Сохранено" in result

    stats = memohood.tools.memohood_stats({}, conn=conn, cfg=cfg, session_id="s1")
    assert "Активных записей: 1" in stats
    conn.close()


def test_memohood_capture_missing_content(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    result = memohood.tools.memohood_capture({}, conn=conn, cfg=_cfg(memohood), session_id="s1")
    assert "обязателен" in result
    conn.close()


def test_memohood_search_and_fetch_roundtrip(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    cfg = _cfg(memohood)
    cap = memohood.capture.manual_capture(
        conn, "Проект использует SQLite для хранения памяти", kind="fact",
        notability="high", pinned=False, session_id="s1", cfg=cfg,
    )
    found = memohood.tools.memohood_search({"query": "SQLite хранения"}, conn=conn, cfg=cfg, session_id="s1")
    assert cap["capture_id"] in found

    fetched = memohood.tools.memohood_fetch({"capture_id": cap["capture_id"]}, conn=conn, cfg=cfg, session_id="s1")
    assert "SQLite" in fetched
    conn.close()


def test_memohood_fetch_missing_id(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    result = memohood.tools.memohood_fetch({"capture_id": "does-not-exist"}, conn=conn, cfg=_cfg(memohood), session_id="s1")
    assert "не найдена" in result
    conn.close()


def test_all_tool_schemas_have_required_fields(memohood):
    for schema in memohood.tools.ALL_TOOL_SCHEMAS:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
