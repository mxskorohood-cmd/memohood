"""Tests for ``graph_rerank.py`` -- the post-retrieval session_links BOOST +
1-hop EXPANSION step. Uses the `memohood` fixture (tests/conftest.py) only for
its isolated-HERMES_HOME real sqlite ``memory.db`` (via ``memohood.db.get_connection``,
which creates the real schema including ``session_links``/``captures``) --
``graph_rerank.py`` itself has zero relative imports (pure stdlib:
``logging``/``sqlite3``/``typing``), so it is loaded directly by file path
here rather than through conftest's synthetic-package loader (that machinery
exists only to make THIS plugin's OWN relative imports resolve, which
graph_rerank.py doesn't use).

No network, no model download, no mocking needed anywhere in this file --
graph_rerank.py never touches an embedder/LLM/reranker at all.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def graph_rerank_mod():
    """A fresh import of graph_rerank.py by file path (no package context
    needed -- see module docstring)."""
    spec = importlib.util.spec_from_file_location("graph_rerank_under_test", PLUGIN_DIR / "graph_rerank.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Seeding helpers -- raw SQL against the real schema (db.py's DDL), so these
# tests exercise the exact table shape graph_rerank.py reads, independent of
# capture.py's own (network-adjacent) two-stage capture pipeline.
# ---------------------------------------------------------------------------


def _insert_capture(
    conn: sqlite3.Connection, *, capture_id: str, content: str, session_id: str,
    kind: str = "fact", pinned: int = 0, created_at: Optional[float] = None,
) -> None:
    ts = created_at if created_at is not None else time.time()
    conn.execute(
        """
        INSERT INTO captures (id, content, kind, session_id, tags, created_at, updated_at, valid_from, pinned)
        VALUES (?, ?, ?, ?, '', ?, ?, ?, ?)
        """,
        (capture_id, content, kind, session_id, ts, ts, ts, pinned),
    )
    conn.commit()


def _insert_link(
    conn: sqlite3.Connection, *, from_sid: str, to_sid: str,
    weight: Optional[float] = None, created_at: Optional[float] = None,
) -> None:
    ts = created_at if created_at is not None else time.time()
    conn.execute(
        "INSERT INTO session_links (from_session_id, to_session_id, relationship, label, weight, created_at) "
        "VALUES (?, ?, 'related', '', ?, ?)",
        (from_sid, to_sid, weight, ts),
    )
    conn.commit()


def _mk_result(capture_id: str, session_id: str, score: float, **overrides: Any) -> Dict[str, Any]:
    """A dict in the exact shape `_engine.retrieve.hybrid_search` produces."""
    base = {
        "capture_id": capture_id,
        "text": f"text of {capture_id}",
        "score": score,
        "source": "fts",
        "rrf_score": score,
        "rerank_score": None,
        "mode": "rrf-only",
        "degraded": False,
        "degraded_reason": None,
        "kind": "fact",
        "confidence": 1.0,
        "notability": "medium",
        "pinned": 0,
        "session_id": session_id,
        "tags": "",
    }
    base.update(overrides)
    return base


def _open_conn(memohood) -> sqlite3.Connection:
    return memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))


# ---------------------------------------------------------------------------
# 1) BOOST
# ---------------------------------------------------------------------------


class TestBoost:
    def test_linked_session_result_boosted_above_unlinked(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        # s-linked is directly linked to s-top (the top hit's session);
        # s-unlinked has no session_links row at all.
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.9)

        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-linked", "s-linked", 5.0),
            _mk_result("cap-unlinked", "s-unlinked", 6.0),
        ]
        # Before boost: unlinked (6.0) already outranks linked (5.0).
        assert results[2]["score"] > results[1]["score"]

        # top_n_anchors=1: only "cap-top" (the single actual top hit) is the
        # anchor here -- with the default 3 and only 3 results seeded, every
        # item would otherwise count as its own anchor and never see itself
        # as a "neighbor" of another anchor.
        cfg = {"graph_rerank": {"top_n_anchors": 1}}
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg=cfg)
        by_id = {r["capture_id"]: r for r in out}

        # weight 0.9 sits in the top closeness tier (default tiers 0.66/0.33)
        # -> boost[0] == 1.5 -> 5.0 * 1.5 == 7.5, now above the unlinked 6.0.
        assert by_id["cap-linked"]["score"] == pytest.approx(7.5)
        assert by_id["cap-unlinked"]["score"] == pytest.approx(6.0)  # untouched
        assert by_id["cap-linked"]["score"] > by_id["cap-unlinked"]["score"]

        order = [r["capture_id"] for r in out]
        assert order.index("cap-linked") < order.index("cap-unlinked")
        conn.close()

    def test_weaker_link_gets_smaller_boost(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-weak", weight=0.1)  # weakest tier

        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-weak", "s-weak", 4.0),
        ]
        cfg = {"graph_rerank": {"top_n_anchors": 1}}
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg=cfg)
        by_id = {r["capture_id"]: r for r in out}
        # weight 0.1 < both default tiers -> weakest boost (1.15) -> 4.0*1.15==4.6
        assert by_id["cap-weak"]["score"] == pytest.approx(4.0 * 1.15)
        conn.close()

    def test_custom_boost_and_weight_tiers_from_cfg(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.5)

        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-linked", "s-linked", 2.0),
        ]
        cfg = {"graph_rerank": {"boost": [2.0, 1.5, 1.1], "weight_tiers": [0.9, 0.4], "top_n_anchors": 1}}
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg=cfg)
        by_id = {r["capture_id"]: r for r in out}
        # weight 0.5 is < tiers[0]=0.9 but >= tiers[1]=0.4 -> tier 1 -> boost 1.5
        assert by_id["cap-linked"]["score"] == pytest.approx(2.0 * 1.5)
        conn.close()


# ---------------------------------------------------------------------------
# 2) 1-HOP EXPANSION
# ---------------------------------------------------------------------------


class TestExpansion:
    def test_1hop_neighbor_capture_added_and_capped_by_max_neighbors(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-neighbor", weight=0.9)
        for i in range(5):
            _insert_capture(
                conn, capture_id=f"neighbor-cap-{i}", content=f"neighbor fact {i}",
                session_id="s-neighbor", created_at=1000.0 + i,
            )

        results = [_mk_result("cap-top", "s-top", 10.0)]
        out = graph_rerank_mod.graph_rerank(
            results, db=conn, cfg={"graph_rerank": {"max_neighbors": 2}},
        )

        added = [r for r in out if r.get("_graph_added")]
        assert len(added) == 2  # capped, even though 5 neighbor captures exist
        assert {r["session_id"] for r in added} == {"s-neighbor"}
        assert all(r["capture_id"].startswith("neighbor-cap-") for r in added)
        assert len(out) == len(results) + 2

        # None of the added captures were already in the original results.
        original_ids = {r["capture_id"] for r in results}
        assert not (original_ids & {r["capture_id"] for r in added})
        conn.close()

    def test_neighbor_expansion_excludes_captures_already_in_results(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-neighbor", weight=0.9)
        _insert_capture(conn, capture_id="already-found", content="x", session_id="s-neighbor")
        _insert_capture(conn, capture_id="new-neighbor", content="y", session_id="s-neighbor")

        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("already-found", "s-neighbor", 3.0),  # lexical search already found this one
        ]
        out = graph_rerank_mod.graph_rerank(
            results, db=conn, cfg={"graph_rerank": {"max_neighbors": 3, "top_n_anchors": 1}},
        )
        added_ids = {r["capture_id"] for r in out if r.get("_graph_added")}
        assert added_ids == {"new-neighbor"}
        conn.close()

    def test_max_neighbors_zero_disables_expansion_only(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.9)
        _insert_capture(conn, capture_id="neighbor-cap", content="z", session_id="s-linked")

        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-linked", "s-linked", 5.0),
        ]
        out = graph_rerank_mod.graph_rerank(
            results, db=conn, cfg={"graph_rerank": {"max_neighbors": 0, "top_n_anchors": 1}},
        )
        # BOOST still applies...
        by_id = {r["capture_id"]: r for r in out}
        assert by_id["cap-linked"]["score"] == pytest.approx(5.0 * 1.5)
        # ...but no new candidate is added.
        assert len(out) == len(results)
        conn.close()


# ---------------------------------------------------------------------------
# 3) Degrade paths -- must ALWAYS return `results` unchanged (pure no-op)
# ---------------------------------------------------------------------------


class TestDegrade:
    def test_empty_session_links_is_passthrough(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)  # schema created, but zero session_links rows
        results = [_mk_result("cap-1", "s1", 1.0)]
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg={})
        assert out is results
        conn.close()

    def test_disabled_is_passthrough_even_with_links_present(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.9)
        results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-linked", "s-linked", 5.0),
        ]
        out = graph_rerank_mod.graph_rerank(
            results, db=conn, cfg={"graph_rerank": {"enabled": False}},
        )
        assert out is results
        conn.close()

    def test_db_none_is_passthrough(self, graph_rerank_mod):
        results = [_mk_result("cap-1", "s1", 1.0)]
        out = graph_rerank_mod.graph_rerank(results, db=None, cfg={})
        assert out is results

    def test_empty_results_is_passthrough(self, graph_rerank_mod):
        results: list = []
        out = graph_rerank_mod.graph_rerank(results, db=None, cfg={})
        assert out is results

    def test_missing_schema_degrades_to_noop_never_raises(self, graph_rerank_mod):
        # A bare connection with NO tables at all (session_links doesn't
        # exist) -- graph_rerank must catch the resulting sqlite3.Error
        # internally and degrade, never raise out of prefetch.
        conn = sqlite3.connect(":memory:")
        results = [_mk_result("cap-1", "s1", 1.0)]
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg={})
        assert out is results
        conn.close()

    def test_cfg_none_uses_defaults_without_raising(self, memohood, graph_rerank_mod):
        conn = _open_conn(memohood)  # empty session_links -> passthrough regardless
        results = [_mk_result("cap-1", "s1", 1.0)]
        out = graph_rerank_mod.graph_rerank(results, db=conn, cfg=None)
        assert out is results
        conn.close()
