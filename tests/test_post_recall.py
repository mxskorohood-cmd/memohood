"""Tests for ``post_recall.py`` -- the MMR/near-duplicate-collapse pass that
runs after retrieval+rerank, before formatting the ``<memory-context>``
block (see that module's own docstring for the full pipeline-position
rationale).

This module is NOT wired into ``provider.py`` (by design -- see the HARD
CONSTRAINTS this was built under), so it is never imported by
``__init__.py``'s import chain and therefore never becomes an attribute of
the ``memohood`` fixture's loaded package automatically. ``_load_post_recall``
below imports it explicitly as a submodule of the SAME synthetic package
``conftest.py``'s ``memohood`` fixture already registered in ``sys.modules`` --
this resolves ``post_recall.py``'s own ``from . import db`` exactly as it
would under the real memory-provider loader, with zero changes to
conftest.py.

No network, no real model/API calls anywhere in this file -- ``diversify``
is a pure function over hand-built vectors/scores, and the ``attach_vectors``
tests use a local sqlite-vec table (if the optional ``sqlite-vec`` package
is unavailable in the environment, those specific tests skip rather than
fail -- everything else in this file has zero dependency on sqlite-vec).
"""

from __future__ import annotations

import importlib
import struct

import pytest


def _load_post_recall(memohood_pkg):
    return importlib.import_module(f"{memohood_pkg.__name__}.post_recall")


def _cfg(mmr=None, cluster=None):
    return {
        "post_recall": {
            "mmr": {} if mmr is None else mmr,
            "cluster": {} if cluster is None else cluster,
        }
    }


# ---------------------------------------------------------------------------
# Degrade / passthrough conditions
# ---------------------------------------------------------------------------


def test_empty_list_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = []
    out = post_recall.diversify(results, cfg=_cfg())
    assert out is results


def test_single_item_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [{"capture_id": "c1", "score": 0.5, "vector": [1.0, 0.0]}]
    out = post_recall.diversify(results, cfg=_cfg())
    assert out is results


def test_mmr_disabled_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.5, "vector": [0.0, 1.0]},
    ]
    out = post_recall.diversify(results, cfg=_cfg(mmr={"enabled": False}))
    assert out is results


def test_missing_post_recall_config_section_still_runs_with_defaults(memohood):
    """cfg with no post_recall key at all -- mmr.enabled defaults to True,
    so this must NOT be a passthrough (it must actually diversify)."""
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.5, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.9, "vector": [0.0, 1.0]},
    ]
    out = post_recall.diversify(results, cfg={})
    # Defaults enabled -> a real (potentially reordering) computation ran,
    # not a bare passthrough; with these orthogonal vectors the highest
    # relevance item must still end up first regardless of lambda.
    assert out[0]["capture_id"] == "c2"


def test_missing_vector_on_one_candidate_is_global_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.5},  # no vector at all
    ]
    out = post_recall.diversify(results, cfg=_cfg())
    assert out is results


def test_invalid_vector_shape_is_global_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.5, "vector": [1.0, 0.0, 0.0]},  # dim mismatch
    ]
    out = post_recall.diversify(results, cfg=_cfg())
    assert out is results


def test_missing_score_on_one_candidate_is_global_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "vector": [0.0, 1.0]},  # no score at all
    ]
    out = post_recall.diversify(results, cfg=_cfg())
    assert out is results


def test_invalid_lambda_type_is_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.5, "vector": [0.0, 1.0]},
    ]
    out = post_recall.diversify(results, cfg=_cfg(mmr={"lambda": "oops"}))
    assert out is results


def test_non_dict_post_recall_section_is_passthrough(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.5, "vector": [0.0, 1.0]},
    ]
    out = post_recall.diversify(results, cfg={"post_recall": "not-a-dict"})
    assert out is results


def test_cfg_none_does_not_raise_and_uses_defaults(memohood):
    post_recall = _load_post_recall(memohood)
    results = [
        {"capture_id": "c1", "score": 0.5, "vector": [1.0, 0.0]},
        {"capture_id": "c2", "score": 0.9, "vector": [0.0, 1.0]},
    ]
    out = post_recall.diversify(results, cfg=None)
    assert out[0]["capture_id"] == "c2"


def test_diversify_never_mutates_input_dicts_or_list(memohood):
    post_recall = _load_post_recall(memohood)
    c1 = {"capture_id": "c1", "score": 0.9, "vector": [1.0, 0.0]}
    c2 = {"capture_id": "c2", "score": 0.5, "vector": [0.0, 1.0]}
    results = [c1, c2]
    original_keys = [set(c1.keys()), set(c2.keys())]

    out = post_recall.diversify(results, cfg=_cfg())

    assert results == [c1, c2]  # input list order untouched
    assert set(c1.keys()) == original_keys[0]
    assert set(c2.keys()) == original_keys[1]
    # every item in the output is one of the ORIGINAL dict objects (no copies,
    # no new items invented)
    for item in out:
        assert item is c1 or item is c2


# ---------------------------------------------------------------------------
# Core MMR behavior
# ---------------------------------------------------------------------------


def _five_candidates():
    """Three near-identical "dark mode" rephrasings (high relevance) plus two
    genuinely distinct facts (lower relevance) -- the "five rephrasings of
    one fact" scenario from the task brief."""
    return [
        {"capture_id": "c1", "text": "User prefers dark mode", "score": 0.95, "vector": [1.0, 0.0, 0.0]},
        {"capture_id": "c2", "text": "User likes the dark theme", "score": 0.93, "vector": [0.99, 0.02, 0.0]},
        {"capture_id": "c3", "text": "User wants a dark UI", "score": 0.91, "vector": [0.98, 0.03, 0.0]},
        {"capture_id": "c4", "text": "User is in UTC+3", "score": 0.60, "vector": [0.0, 1.0, 0.0]},
        {"capture_id": "c5", "text": "Project name is Fable", "score": 0.55, "vector": [0.0, 0.0, 1.0]},
    ]


def test_lambda_one_equals_pure_relevance_order(memohood):
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()
    # Shuffle input order so a passing test can't be an accident of
    # "input happened to already be sorted".
    shuffled = [candidates[4], candidates[2], candidates[0], candidates[3], candidates[1]]

    out = post_recall.diversify(
        shuffled, cfg=_cfg(mmr={"lambda": 1.0}, cluster={"enabled": False}),
    )

    assert [r["capture_id"] for r in out] == ["c1", "c2", "c3", "c4", "c5"]


def test_lambda_zero_still_starts_from_highest_relevance(memohood):
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()

    out = post_recall.diversify(
        candidates, cfg=_cfg(mmr={"lambda": 0.0}, cluster={"enabled": False}),
    )

    assert out[0]["capture_id"] == "c1"


def test_mmr_promotes_diverse_facts_over_near_duplicate_rephrasings(memohood):
    """The central claim of this module: with clustering OFF (so this is
    testing pure MMR reordering, not dedup collapse), the two genuinely
    distinct facts (c4, c5) must surface ABOVE the near-duplicate "dark
    mode" rephrasings (c2, c3) even though c2/c3 carry a higher raw
    relevance score than c4/c5 -- diversity is winning over redundant
    relevance, which is the entire point of MMR."""
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()

    out = post_recall.diversify(
        candidates, cfg=_cfg(mmr={"lambda": 0.5}, cluster={"enabled": False}),
    )

    ids = [r["capture_id"] for r in out]
    assert len(ids) == 5  # no clustering -> nothing dropped, only reordered
    assert ids[0] == "c1"  # single strongest signal still leads
    pos = {cid: i for i, cid in enumerate(ids)}
    assert pos["c4"] < pos["c2"]
    assert pos["c4"] < pos["c3"]
    assert pos["c5"] < pos["c2"]
    assert pos["c5"] < pos["c3"]


def test_near_duplicates_collapse_to_single_representative(memohood):
    """With clustering ON, the three near-identical "dark mode" rephrasings
    must collapse into ONE representative (the highest-scoring one, c1) --
    dropping c2/c3 entirely rather than merely demoting them."""
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()

    out = post_recall.diversify(
        candidates,
        cfg=_cfg(mmr={"lambda": 0.7}, cluster={"enabled": True, "threshold": 0.95}),
    )

    ids = [r["capture_id"] for r in out]
    assert ids == ["c1", "c4", "c5"]


def test_cluster_disabled_keeps_all_candidates(memohood):
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()

    out = post_recall.diversify(
        candidates, cfg=_cfg(mmr={"lambda": 0.7}, cluster={"enabled": False}),
    )

    assert len(out) == 5


def test_cluster_threshold_none_skips_clustering(memohood):
    post_recall = _load_post_recall(memohood)
    candidates = _five_candidates()

    out = post_recall.diversify(
        candidates, cfg=_cfg(mmr={"lambda": 0.7}, cluster={"enabled": True, "threshold": None}),
    )

    assert len(out) == 5  # no threshold configured -> collapse step is a no-op


# ---------------------------------------------------------------------------
# attach_vectors -- best-effort DB helper
# ---------------------------------------------------------------------------


def test_attach_vectors_conn_none_is_noop(memohood):
    post_recall = _load_post_recall(memohood)
    results = [{"capture_id": "cap-1", "score": 0.9}]
    out = post_recall.attach_vectors(None, results)
    assert out is results
    assert "vector" not in out[0]


def test_attach_vectors_empty_results_is_noop(memohood):
    post_recall = _load_post_recall(memohood)
    out = post_recall.attach_vectors(object(), [])
    assert out == []


def test_attach_vectors_no_vec_table_is_noop(memohood):
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    post_recall = _load_post_recall(memohood)
    try:
        results = [{"capture_id": "cap-1", "score": 0.9}]
        out = post_recall.attach_vectors(conn, results)
        assert "vector" not in out[0]
    finally:
        conn.close()


def test_attach_vectors_reads_stored_embeddings(memohood):
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    post_recall = _load_post_recall(memohood)
    try:
        vec_ready = memohood.db.ensure_vec_table(conn, dims=3)
        if not vec_ready:
            pytest.skip("sqlite-vec extension not available in this environment")

        vecs = {"cap-1": [1.0, 0.0, 0.0], "cap-2": [0.0, 1.0, 0.0]}
        table = memohood.db.vec_table_name()
        for cid, v in vecs.items():
            blob = struct.pack(f"<{len(v)}f", *v)
            conn.execute(
                f"INSERT OR REPLACE INTO {table}(capture_id, embedding) VALUES (?, ?)", (cid, blob),
            )
        conn.commit()

        results = [
            {"capture_id": "cap-1", "text": "a", "score": 0.9},
            {"capture_id": "cap-2", "text": "b", "score": 0.8},
            {"capture_id": "cap-missing", "text": "c", "score": 0.7},
        ]
        out = post_recall.attach_vectors(conn, results)

        assert out is results
        assert results[0]["vector"] == pytest.approx([1.0, 0.0, 0.0])
        assert results[1]["vector"] == pytest.approx([0.0, 1.0, 0.0])
        assert "vector" not in results[2]
    finally:
        conn.close()


def test_attach_vectors_then_diversify_end_to_end(memohood):
    """attach_vectors + diversify chained together, exactly as the wiring
    spec instructs the integrator to call them, over real (if available)
    sqlite-vec-backed vectors."""
    hermes_home = str(memohood._hermes_home_for_test)
    conn = memohood.db.get_connection(hermes_home=hermes_home)
    post_recall = _load_post_recall(memohood)
    try:
        vec_ready = memohood.db.ensure_vec_table(conn, dims=3)
        if not vec_ready:
            pytest.skip("sqlite-vec extension not available in this environment")

        vecs = {
            "c1": [1.0, 0.0, 0.0],
            "c2": [0.99, 0.02, 0.0],
            "c4": [0.0, 1.0, 0.0],
        }
        table = memohood.db.vec_table_name()
        for cid, v in vecs.items():
            blob = struct.pack(f"<{len(v)}f", *v)
            conn.execute(
                f"INSERT OR REPLACE INTO {table}(capture_id, embedding) VALUES (?, ?)", (cid, blob),
            )
        conn.commit()

        results = [
            {"capture_id": "c1", "score": 0.95},
            {"capture_id": "c2", "score": 0.93},
            {"capture_id": "c4", "score": 0.60},
        ]
        results = post_recall.attach_vectors(conn, results)
        out = post_recall.diversify(
            results, cfg=_cfg(cluster={"enabled": True, "threshold": 0.95}),
        )
        assert [r["capture_id"] for r in out] == ["c1", "c4"]
    finally:
        conn.close()
