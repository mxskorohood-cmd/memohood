"""Integration tests for memohood v1.1's prefetch pipeline WIRING (not module
internals -- those are covered by test_gate.py/test_graph_rerank.py/
test_post_recall.py in isolation).

Proves the pipeline order ``MemoHoodMemoryProvider._compute_prefetch_text`` now
runs (DESIGN_v1.md's stub line replaced this round):

    gate (pre-retrieval) -> hybrid_search -> graph_rerank -> post_recall
    (attach_vectors + MMR/diversify) -> reinforce/format

Strategy: rather than trying to hand-predict real FTS/BM25/RRF numbers
(fragile), :func:`_engine.retrieve.hybrid_search` -- the one call in this
chain that would otherwise touch the network embedder/reranker -- is
monkeypatched to return a small, fully controlled, hand-crafted candidate
list (same dict shape ``hybrid_search`` itself documents, mirroring
``test_graph_rerank.py``'s own ``_mk_result`` helper). Everything AFTER that
point (``graph_rerank_mod.graph_rerank``, ``post_recall.attach_vectors``,
``post_recall.diversify``, ``_reinforce``, formatting) is the REAL code,
run against a REAL temp ``memory.db`` (the ``memohood`` fixture's isolated
per-test HERMES_HOME) with real ``session_links``/``captures``/
``captures_vec`` rows -- this is what actually proves the WIRING (this
task's job), as opposed to re-testing retrieval or the modules' own
internals (already covered elsewhere).

No network calls anywhere in this file: credential env vars are stripped by
the ``memohood`` fixture by default (COHERE_API_KEY absent -> rerank already
degrades to rrf-only on its own), and the one call that WOULD need
Cloudflare credentials (``hybrid_search``) is monkeypatched away entirely.
"""

from __future__ import annotations

import copy
import struct
import time
from typing import Any, Dict, Optional

import pytest


def _make_provider(memohood, *, session_id="s-top"):
    p = memohood.provider.MemoHoodMemoryProvider()
    p.initialize(session_id, hermes_home=str(memohood._hermes_home_for_test), platform="cli")
    return p


def _mk_result(capture_id: str, session_id: str, score: float, **overrides: Any) -> Dict[str, Any]:
    """A dict in the exact shape ``_engine.retrieve.hybrid_search`` produces
    (mirrors ``test_graph_rerank.py``'s own helper of the same name)."""
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


def _insert_link(conn, *, from_sid: str, to_sid: str, weight: Optional[float] = None) -> None:
    conn.execute(
        "INSERT INTO session_links (from_session_id, to_session_id, relationship, label, weight, created_at) "
        "VALUES (?, ?, 'related', '', ?, ?)",
        (from_sid, to_sid, weight, time.time()),
    )
    conn.commit()


def _insert_capture_row(conn, *, capture_id: str, content: str, session_id: str, created_at: float) -> None:
    conn.execute(
        """
        INSERT INTO captures (id, content, kind, session_id, tags, created_at, updated_at, valid_from, pinned)
        VALUES (?, ?, 'fact', ?, '', ?, ?, ?, 0)
        """,
        (capture_id, content, session_id, created_at, created_at, created_at),
    )
    conn.commit()


def _seed_vector(conn, table: str, capture_id: str, vec) -> None:
    blob = struct.pack(f"<{len(vec)}f", *vec)
    conn.execute(f"INSERT OR REPLACE INTO {table}(capture_id, embedding) VALUES (?, ?)", (capture_id, blob))
    conn.commit()


def _stub_hybrid_search(fake_results):
    def _fake(conn, query, k, cfg):
        return [dict(r) for r in fake_results]

    return _fake


# ---------------------------------------------------------------------------
# 1) gate.backend=pass (the DEFAULT) -> the pipeline behaves exactly as
#    before v1.1 -- no behavior change until an operator opts in.
# ---------------------------------------------------------------------------


class TestGatePassBackendBaseline:
    def test_default_cfg_prefetch_recalls_and_reinforces_as_before(self, memohood):
        p = _make_provider(memohood)
        assert p._cfg["gate"]["backend"] == "pass"  # DEFAULTS confirm the gate is off
        # New v1.1 sections are ON by default (per this round's task spec)
        # but must be harmless no-ops here: no session_links exist and there
        # is only one capture, so graph_rerank/post_recall both degrade to
        # passthrough and the end-to-end behavior matches pre-v1.1 exactly.
        assert p._cfg["graph_rerank"]["enabled"] is True
        assert p._cfg["post_recall"]["mmr"]["enabled"] is True

        result = memohood.capture.manual_capture(
            p._conn, "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s-top", cfg=p._cfg,
        )
        cid = result["capture_id"]
        before = p._conn.execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]
        time.sleep(0.01)

        text = p.prefetch("расскажи про договор с поставщиком", session_id="s-top")
        assert "договор" in text.lower()

        after = p._conn.execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]
        assert after >= before
        p.shutdown()

    def test_pass_backend_still_returns_empty_on_no_hits(self, memohood):
        p = _make_provider(memohood)
        assert p.prefetch("совершенно несвязанный запрос без совпадений") == ""
        p.shutdown()


# ---------------------------------------------------------------------------
# 2) gate sits strictly BEFORE retrieval: a skip decision must short-circuit
#    before hybrid_search (and therefore graph_rerank/post_recall too) ever
#    runs.
# ---------------------------------------------------------------------------


class TestGateSkipsBeforeRetrieval:
    def test_gate_skip_short_circuits_before_hybrid_search_runs(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        memohood.capture.manual_capture(
            p._conn, "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s-top", cfg=p._cfg,
        )
        p._cfg["gate"]["backend"] = "model2vec"

        monkeypatch.setattr(
            memohood.gate, "should_recall", lambda query, *, cfg=None: (False, 0.0, "test: forced skip")
        )

        def _boom(*a, **kw):
            raise AssertionError("hybrid_search must not run when the gate skips recall")

        monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", _boom)

        text = p.prefetch("расскажи про договор с поставщиком", session_id="s-top")
        assert text == ""
        p.shutdown()

    def test_gate_pass_decision_still_lets_retrieval_run(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        calls = []

        def _spy_hybrid_search(conn, query, k, cfg):
            calls.append(query)
            return []

        monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", _spy_hybrid_search)
        p.prefetch("а мы точно решили использовать HERMES_HOME для конфига?", session_id="s-top")
        assert len(calls) == 1
        p.shutdown()


# ---------------------------------------------------------------------------
# 3) graph_rerank + post_recall (MMR/cluster) actually run inside prefetch()
#    and reorder/trim exactly as their own unit tests prove they do in
#    isolation -- this is the core "wiring is correct" proof.
# ---------------------------------------------------------------------------


class TestGraphRerankAndPostRecallWiring:
    def test_boost_expansion_and_collapse_all_fire_inside_prefetch(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        conn = p._conn

        vec_ready = memohood.db.ensure_vec_table(conn, dims=2)
        if not vec_ready:
            pytest.skip("sqlite-vec extension not available in this environment")

        # --- graph: s-top (the top hit's session) is linked to s-linked ---
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.9)

        # A real capture row in s-linked that lexical/vector search did NOT
        # surface -- graph_rerank's 1-hop expansion should pull this in.
        _insert_capture_row(
            conn, capture_id="cap-a-hidden", content="text of cap-a-hidden",
            session_id="s-linked", created_at=time.time(),
        )

        # --- retrieval (mocked): cap-b (unlinked, 6.0) outranks cap-a
        # (linked, 5.0) BEFORE graph_rerank runs.
        fake_results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-b", "s-unlinked", 6.0),
            _mk_result("cap-a", "s-linked", 5.0),
        ]
        monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", _stub_hybrid_search(fake_results))

        # --- vectors: cap-a and cap-a-hidden are near-duplicates (same
        # session, colinear vectors) -- post_recall's cluster-collapse
        # should drop cap-a-hidden and keep cap-a (higher/tied relevance,
        # first in processing order).
        table = memohood.db.vec_table_name()
        _seed_vector(conn, table, "cap-top", [1.0, 0.0])
        _seed_vector(conn, table, "cap-a", [0.0, 1.0])
        _seed_vector(conn, table, "cap-a-hidden", [0.0, 0.99])  # colinear with cap-a -> cosine 1.0
        _seed_vector(conn, table, "cap-b", [1.0, 1.0])

        # top_n_anchors=1: only cap-top's own session anchors the graph
        # lookup -- otherwise cap-a's own session would itself count as an
        # anchor and the s-top<->s-linked link would be skipped as an
        # anchor-to-anchor edge (see test_graph_rerank.py's identical note).
        p._cfg["graph_rerank"]["top_n_anchors"] = 1

        calls: Dict[str, Any] = {}
        real_graph_rerank = memohood.graph_rerank.graph_rerank
        real_diversify = memohood.post_recall.diversify

        def _spy_graph_rerank(results, *, db, cfg=None):
            calls["graph_rerank_in"] = [dict(r) for r in results]
            out = real_graph_rerank(results, db=db, cfg=cfg)
            calls["graph_rerank_out"] = [dict(r) for r in out]
            return out

        def _spy_diversify(results, *, cfg, query=None):
            calls["diversify_in"] = [dict(r) for r in results]
            out = real_diversify(results, cfg=cfg, query=query)
            calls["diversify_out"] = [dict(r) for r in out]
            return out

        monkeypatch.setattr(memohood.graph_rerank, "graph_rerank", _spy_graph_rerank)
        monkeypatch.setattr(memohood.post_recall, "diversify", _spy_diversify)

        text = p.prefetch("проверка графа и MMR в prefetch", session_id="s-top")

        # --- graph_rerank actually ran, in the documented input order ---
        assert [r["capture_id"] for r in calls["graph_rerank_in"]] == ["cap-top", "cap-b", "cap-a"]

        # --- BOOST reordered cap-a (now 7.5) above cap-b (6.0, untouched) ---
        out_ids = [r["capture_id"] for r in calls["graph_rerank_out"]]
        by_id = {r["capture_id"]: r for r in calls["graph_rerank_out"]}
        assert by_id["cap-a"]["score"] == pytest.approx(7.5)
        assert by_id["cap-b"]["score"] == pytest.approx(6.0)
        assert out_ids.index("cap-a") < out_ids.index("cap-b")

        # --- EXPANSION added the 1-hop neighbor capture ---
        assert "cap-a-hidden" in out_ids
        assert by_id["cap-a-hidden"].get("_graph_added") is True

        # --- post_recall received vectors attached (attach_vectors ran) ---
        assert all("vector" in r for r in calls["diversify_in"])

        # --- CLUSTER-COLLAPSE dropped the near-duplicate expansion capture ---
        diversify_out_ids = [r["capture_id"] for r in calls["diversify_out"]]
        assert "cap-a-hidden" not in diversify_out_ids
        assert diversify_out_ids == ["cap-top", "cap-a", "cap-b"]

        # --- the final formatted text reflects the post-pipeline list: the
        # collapsed duplicate's distinctive content is gone, the survivors
        # appear in the diversified order.
        assert "text of cap-a-hidden" not in text
        assert text.index("text of cap-top") < text.index("text of cap-a") < text.index("text of cap-b")
        p.shutdown()


# ---------------------------------------------------------------------------
# 4) Each new feature is independently disableable via config and the
#    pipeline keeps working (no crash, sensible degrade to "as before").
# ---------------------------------------------------------------------------


class TestFeatureTogglesStillWork:
    def test_graph_rerank_disabled_keeps_pre_boost_order(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        conn = p._conn
        _insert_link(conn, from_sid="s-top", to_sid="s-linked", weight=0.9)

        fake_results = [
            _mk_result("cap-top", "s-top", 10.0),
            _mk_result("cap-b", "s-unlinked", 6.0),
            _mk_result("cap-a", "s-linked", 5.0),
        ]
        monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", _stub_hybrid_search(fake_results))

        p._cfg["graph_rerank"]["enabled"] = False
        p._cfg["graph_rerank"]["top_n_anchors"] = 1
        p._cfg["post_recall"]["mmr"]["enabled"] = False  # isolate: only graph_rerank's toggle under test here

        text = p.prefetch("что угодно", session_id="s-top")
        # Without graph_rerank, cap-b (unlinked, 6.0) still outranks cap-a
        # (linked, 5.0) -- the pre-boost order survives untouched, proving
        # the disabled step is a genuine no-op, not just "boost happens to
        # not matter here".
        assert text.index("text of cap-b") < text.index("text of cap-a")
        p.shutdown()

    def test_post_recall_disabled_keeps_near_duplicates_uncollapsed(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        conn = p._conn
        vec_ready = memohood.db.ensure_vec_table(conn, dims=2)
        if not vec_ready:
            pytest.skip("sqlite-vec extension not available in this environment")

        fake_results = [
            _mk_result("cap-1", "s1", 0.9),
            _mk_result("cap-2", "s1", 0.8),  # near-duplicate of cap-1
        ]
        monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", _stub_hybrid_search(fake_results))

        table = memohood.db.vec_table_name()
        _seed_vector(conn, table, "cap-1", [1.0, 0.0])
        _seed_vector(conn, table, "cap-2", [0.99, 0.0])  # cosine ~1.0 -- would collapse if MMR were on

        p._cfg["graph_rerank"]["enabled"] = False  # isolate: only post_recall's toggle under test here
        p._cfg["post_recall"]["mmr"]["enabled"] = False

        text = p.prefetch("что угодно", session_id="s-top")
        assert "text of cap-1" in text
        assert "text of cap-2" in text  # NOT collapsed -- MMR/cluster never ran
        p.shutdown()

    def test_gate_model2vec_misconfigured_degrades_to_pass_pipeline_still_works(self, memohood):
        """An operator opts into gate.backend=model2vec without the optional
        model2vec package installed (this round's hard constraint: never pip
        install it) -- gate.py's own degrade contract falls back to
        pass-through, and the rest of the pipeline is unaffected."""
        p = _make_provider(memohood)
        memohood.capture.manual_capture(
            p._conn, "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s-top", cfg=p._cfg,
        )
        p._cfg["gate"]["backend"] = "model2vec"
        p._cfg["gate"]["meaningful_terms_floor"] = 99  # force the scoring path, not the floor shortcut

        text = p.prefetch("расскажи про договор с поставщиком", session_id="s-top")
        assert "договор" in text.lower()
        p.shutdown()
