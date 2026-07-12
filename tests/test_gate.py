"""gate.should_recall: the pre-retrieval recall gate.

Covers: (1) backend="pass" always recalls, whatever the query looks like;
(2) backend="model2vec" with a FAKE embedder (never the real package/model)
correctly skips an obvious greeting and passes a real memory-referencing
question; (3) a model2vec import/load failure degrades to pass-through,
logging the degrade only once; (4) the pure `_decide` margin/threshold rule
is honored at its boundaries, independent of any embedder.

Note on module loading: ``gate.py`` is a brand-new module not yet imported
by ``__init__.py``'s chain (this round's wiring spec asks the integrator to
add ``from . import gate`` to ``provider.py`` separately -- see gate.py's
own module docstring). The ``gate_mod`` fixture below imports it directly as
a submodule of the already-loaded synthetic test package (conftest.py's
``memohood`` fixture), so its relative import (``from . import query_norm``)
resolves exactly as it will once wired in, with zero changes to conftest.py
or any existing test.
"""

from __future__ import annotations

import copy
import importlib
import logging

import pytest


@pytest.fixture()
def gate_mod(memohood):
    return importlib.import_module(f"{memohood.__name__}.gate")


def _cfg(memohood, **gate_overrides):
    cfg = copy.deepcopy(memohood.config.DEFAULTS)
    cfg["gate"].update(gate_overrides)
    return cfg


class _FakeEmbedder:
    """Deterministic stand-in for a model2vec ``StaticModel``: looks each
    input string up in a fixed table and returns its vector, falling back to
    a neutral default for anything unrecognized. No model2vec import, no
    network, no real model -- exactly what the project's test hard
    constraint requires.
    """

    def __init__(self, table, default):
        self.table = table
        self.default = default
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [self.table.get(t, self.default) for t in texts]


def _make_fake_embedder(gate_mod, *, greeting_query=None, memory_query=None):
    """Build a _FakeEmbedder whose table maps every real POSITIVE_SEEDS/
    NEGATIVE_SEEDS entry to a clean [1,0]/[0,1] vector, plus (optionally) a
    couple of test-controlled query strings pinned to one side or the
    other -- so should_recall's full pipeline (seed embedding + query
    embedding + cosine + _decide) runs for real, with only the embedder
    itself faked out.
    """
    table = {}
    for s in gate_mod.POSITIVE_SEEDS:
        table[s] = [1.0, 0.0]
    for s in gate_mod.NEGATIVE_SEEDS:
        table[s] = [0.0, 1.0]
    if greeting_query is not None:
        table[greeting_query] = [0.0, 1.0]
    if memory_query is not None:
        table[memory_query] = [1.0, 0.0]
    return _FakeEmbedder(table, default=[0.2, 0.2])


class TestPassBackendAlwaysTrue:
    def test_greeting_like_query_still_passes(self, memohood, gate_mod):
        ok, score, reason = gate_mod.should_recall("ок, спасибо", cfg=_cfg(memohood))
        assert ok is True
        assert "pass" in reason

    def test_real_question_passes(self, memohood, gate_mod):
        ok, score, reason = gate_mod.should_recall(
            "а мы точно решили использовать HERMES_HOME для конфига?", cfg=_cfg(memohood)
        )
        assert ok is True

    def test_empty_query_passes(self, memohood, gate_mod):
        ok, score, reason = gate_mod.should_recall("", cfg=_cfg(memohood))
        assert ok is True
        ok2, _, _ = gate_mod.should_recall("   ", cfg=_cfg(memohood))
        assert ok2 is True

    def test_is_the_default_backend(self, memohood, gate_mod):
        # No explicit gate.backend override -- DEFAULTS already say "pass".
        assert memohood.config.DEFAULTS["gate"]["backend"] == "pass"
        ok, _, reason = gate_mod.should_recall("продолжай", cfg=copy.deepcopy(memohood.config.DEFAULTS))
        assert ok is True
        assert "pass" in reason

    def test_missing_cfg_defaults_to_pass(self, gate_mod):
        ok, _, reason = gate_mod.should_recall("привет", cfg=None)
        assert ok is True

    def test_unknown_backend_degrades_to_pass(self, memohood, gate_mod):
        ok, _, reason = gate_mod.should_recall("что угодно", cfg=_cfg(memohood, backend="not-a-real-backend"))
        assert ok is True
        assert "unknown backend" in reason


class TestModel2VecBackendWithFakeEmbedder:
    def test_greeting_scores_skip(self, memohood, gate_mod, monkeypatch):
        greeting = "ладно, го дальше"
        fake = _make_fake_embedder(gate_mod, greeting_query=greeting)
        monkeypatch.setattr(gate_mod, "_load_embedder", lambda name: fake)

        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)  # force scoring path
        ok, score, reason = gate_mod.should_recall(greeting, cfg=cfg)

        assert ok is False
        assert "skip" in reason
        assert score < 0

    def test_real_memory_question_scores_pass(self, memohood, gate_mod, monkeypatch):
        question = "напомни, мы решили использовать Cloudflare для эмбеддингов?"
        fake = _make_fake_embedder(gate_mod, memory_query=question)
        monkeypatch.setattr(gate_mod, "_load_embedder", lambda name: fake)

        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)  # force scoring path
        ok, score, reason = gate_mod.should_recall(question, cfg=cfg)

        assert ok is True
        assert score > 0

    def test_seeds_embedded_once_and_cached(self, memohood, gate_mod, monkeypatch):
        greeting = "спасибо, хватит"
        fake = _make_fake_embedder(gate_mod, greeting_query=greeting)
        monkeypatch.setattr(gate_mod, "_load_embedder", lambda name: fake)
        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)

        gate_mod.should_recall(greeting, cfg=cfg)
        gate_mod.should_recall(greeting, cfg=cfg)

        # Seeds are embedded in exactly two encode() calls total (one for
        # POSITIVE_SEEDS, one for NEGATIVE_SEEDS -- cached by model name
        # after that); the query itself is embedded once per should_recall
        # call, always as its own single-item batch.
        seed_batches = [c for c in fake.calls if len(c) > 1]
        assert len(seed_batches) == 2

    def test_meaningful_terms_floor_forces_pass_without_scoring(self, memohood, gate_mod, monkeypatch):
        # A long, content-rich query should recall via the floor shortcut
        # alone -- the embedder must never even be asked to load.
        def _boom(name):
            raise AssertionError("embedder should not be loaded when the floor is met")

        monkeypatch.setattr(gate_mod, "_load_embedder", _boom)
        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=3)
        query = "напомни какой стек мы выбрали для HERMES_HOME и config.yaml проекта"

        ok, score, reason = gate_mod.should_recall(query, cfg=cfg)
        assert ok is True
        assert "floor" in reason


class TestModel2VecImportFailureDegradesToPass:
    def test_missing_package_degrades_and_never_raises(self, memohood, gate_mod, monkeypatch):
        def _raise_import_error(name):
            raise ImportError("no module named model2vec")

        monkeypatch.setattr(gate_mod, "_load_embedder", _raise_import_error)
        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)

        ok, score, reason = gate_mod.should_recall("ок", cfg=cfg)
        assert ok is True
        assert "degraded" in reason or "unavailable" in reason

    def test_degrade_logged_once_not_spammed(self, memohood, gate_mod, monkeypatch, caplog):
        def _raise_import_error(name):
            raise ImportError("no module named model2vec")

        monkeypatch.setattr(gate_mod, "_load_embedder", _raise_import_error)
        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)

        with caplog.at_level(logging.DEBUG, logger="memohood.gate"):
            for _ in range(3):
                ok, _, _ = gate_mod.should_recall("ок", cfg=cfg)
                assert ok is True

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "degrad" in r.message.lower()]
        assert len(warnings) == 1
        assert len(debugs) == 2

    def test_model_load_runtime_error_also_degrades(self, memohood, gate_mod, monkeypatch):
        def _raise_runtime_error(name):
            raise RuntimeError("corrupt model weights")

        monkeypatch.setattr(gate_mod, "_load_embedder", _raise_runtime_error)
        cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99)

        ok, _, reason = gate_mod.should_recall("привет", cfg=cfg)
        assert ok is True


class TestThresholdAndMarginHonored:
    """Pure unit tests of `_decide` -- no embedder involved at all."""

    def test_within_margin_passes(self, gate_mod):
        # pos_sim (0.50) is within margin (0.05) of neg_sim (0.53) -> pass.
        ok, score, reason = gate_mod._decide(0.50, 0.53, margin=0.05, threshold=0.5)
        assert ok is True
        assert ">=" in reason

    def test_beyond_margin_and_above_threshold_skips(self, gate_mod):
        # pos_sim (0.10) is well below neg_sim (0.80) - margin (0.05); and
        # neg_sim (0.80) is a confident hit above threshold (0.5) -> skip.
        ok, score, reason = gate_mod._decide(0.10, 0.80, margin=0.05, threshold=0.5)
        assert ok is False
        assert score < 0

    def test_beyond_margin_but_below_threshold_still_passes(self, gate_mod):
        # Same gap as above (0.10 vs 0.30, margin 0.05 -> fails the margin
        # check) but neg_sim (0.30) is BELOW threshold (0.5) -- too
        # low-confidence a "negative" match to justify skipping -> bias to
        # pass wins.
        ok, score, reason = gate_mod._decide(0.10, 0.30, margin=0.05, threshold=0.5)
        assert ok is True
        assert "threshold" in reason

    def test_widening_margin_flips_a_skip_into_a_pass(self, gate_mod):
        pos_sim, neg_sim = 0.40, 0.50
        skip_ok, _, _ = gate_mod._decide(pos_sim, neg_sim, margin=0.05, threshold=0.0)
        assert skip_ok is False
        pass_ok, _, _ = gate_mod._decide(pos_sim, neg_sim, margin=0.20, threshold=0.0)
        assert pass_ok is True

    def test_raising_threshold_flips_a_skip_into_a_pass(self, gate_mod):
        pos_sim, neg_sim = 0.10, 0.60
        skip_ok, _, _ = gate_mod._decide(pos_sim, neg_sim, margin=0.05, threshold=0.5)
        assert skip_ok is False
        pass_ok, _, _ = gate_mod._decide(pos_sim, neg_sim, margin=0.05, threshold=0.9)
        assert pass_ok is True

    def test_margin_and_threshold_read_from_cfg(self, memohood, gate_mod, monkeypatch):
        # End-to-end (fake embedder) check that should_recall actually reads
        # gate.margin/gate.threshold from cfg rather than only the hardcoded
        # defaults.
        # A query vector NOT perfectly aligned with the negative-seed axis
        # (neg_sim ~0.92, not a maxed-out 1.0) so a high-but-sub-1.0
        # threshold can actually straddle it and flip the decision. The
        # query string must NOT be a literal entry of either seed list --
        # the fake embedder's table is a single dict keyed by text, so a
        # query that collides with a seed's own text would overwrite (or be
        # overwritten by) that seed's vector too.
        query = "проверка эвристики гейта"
        assert query not in gate_mod.POSITIVE_SEEDS and query not in gate_mod.NEGATIVE_SEEDS
        table = {s: [1.0, 0.0] for s in gate_mod.POSITIVE_SEEDS}
        table.update({s: [0.0, 1.0] for s in gate_mod.NEGATIVE_SEEDS})
        table[query] = [0.3, 0.7]
        fake = _FakeEmbedder(table, default=[0.2, 0.2])
        monkeypatch.setattr(gate_mod, "_load_embedder", lambda name: fake)

        strict_cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99, margin=0.0, threshold=0.0)
        ok_strict, _, _ = gate_mod.should_recall(query, cfg=strict_cfg)
        assert ok_strict is False

        lenient_cfg = _cfg(memohood, backend="model2vec", meaningful_terms_floor=99, margin=0.0, threshold=0.99)
        ok_lenient, _, _ = gate_mod.should_recall(query, cfg=lenient_cfg)
        assert ok_lenient is True
