"""consolidate.py: Ebbinghaus decay (pinned exempt), dedup fallback, rollup."""

from __future__ import annotations

import copy
import time

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _insert_capture(conn, memohood, *, kind, pinned, last_seen_age_days, confidence=1.0):
    now = memohood.db.now()
    last_seen_at = now - last_seen_age_days * 86400.0
    import uuid
    cid = uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO captures(
            id, content, kind, confidence, notability, source, pinned,
            supersedes, history, session_id, message_id, tags, last_seen_at,
            created_at, updated_at, valid_from, invalidated_at, embed_signature
        ) VALUES (?, 'contenido', ?, ?, 'medium', 'EXTRACTED', ?, '', '', 's1', NULL, '', ?, ?, ?, ?, NULL, NULL)
        """,
        (cid, kind, confidence, int(pinned), last_seen_at, now, now, now),
    )
    conn.commit()
    return cid


class TestDecay:
    def test_pinned_survives_400_days_ordinary_archived(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        pinned_id = _insert_capture(conn, memohood, kind="persona", pinned=True, last_seen_age_days=400)
        # preference halflife=90d; 400 days unseen decays confidence*exp(-400/90)
        # = exp(-4.44) ~= 0.0118, well below the 0.05 floor -> archived.
        ordinary_id = _insert_capture(conn, memohood, kind="preference", pinned=False, last_seen_age_days=400)

        result = memohood.consolidate.run_decay(conn, cfg)
        assert result["archived"] >= 1

        pinned_row = conn.execute("SELECT * FROM captures WHERE id=?", (pinned_id,)).fetchone()
        ordinary_row = conn.execute("SELECT * FROM captures WHERE id=?", (ordinary_id,)).fetchone()

        assert pinned_row["invalidated_at"] is None, "pinned capture must never be archived by decay"
        assert pinned_row["confidence"] == 1.0, "pinned capture's confidence must not decay at all"
        assert ordinary_row["invalidated_at"] is not None, "ordinary stale capture should be archived"
        assert "archived_decay" in (ordinary_row["tags"] or "")
        conn.close()

    def test_recently_seen_ordinary_capture_not_archived(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cid = _insert_capture(conn, memohood, kind="fact", pinned=False, last_seen_age_days=1)
        result = memohood.consolidate.run_decay(conn, _cfg(memohood))
        row = conn.execute("SELECT * FROM captures WHERE id=?", (cid,)).fetchone()
        assert row["invalidated_at"] is None
        assert result["decayed"] >= 1
        conn.close()

    def test_compute_decay_confidence_pure_function(self, memohood):
        # The formula is confidence * exp(-age_days / halflife) (verbatim
        # from DESIGN_v1.md) -- despite the config key's name, this is an
        # e-folding time constant, not a true half-life (which would need
        # exp(-age * ln(2) / halflife) to reach exactly 0.5 at age=halflife).
        # At age == halflife, confidence decays to 1/e =~ 0.368, not 0.5.
        now = 1_000_000.0
        last_seen = now - 90 * 86400.0
        conf = memohood.consolidate.compute_decay_confidence(
            1.0, "preference", last_seen, now=now, cfg=_cfg(memohood),
        )
        import math
        assert conf == pytest.approx(math.exp(-1), rel=1e-6)

    def test_never_seen_treated_as_seen_now(self, memohood):
        conf = memohood.consolidate.compute_decay_confidence(1.0, "fact", None, cfg=_cfg(memohood))
        assert conf == pytest.approx(1.0, abs=1e-6)


class TestDedupFallback:
    def test_exact_duplicate_content_merged_without_vec(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        # No sqlite-vec table exists yet in this fresh db -> fallback path.
        assert memohood.db.vec_table_exists(conn) is False
        import uuid
        now = memohood.db.now()
        for i in range(2):
            conn.execute(
                """
                INSERT INTO captures(
                    id, content, kind, confidence, notability, source, pinned,
                    supersedes, history, session_id, message_id, tags, last_seen_at,
                    created_at, updated_at, valid_from, invalidated_at, embed_signature
                ) VALUES (?, 'одинаковый факт', 'fact', 1.0, 'medium', 'EXTRACTED', 0, '', '', 's1', NULL, '', ?, ?, ?, ?, NULL, NULL)
                """,
                (uuid.uuid4().hex, now, now, now, now),
            )
        conn.commit()
        result = memohood.consolidate.run_dedup(conn, _cfg(memohood))
        assert result["merged"] == 1
        active = conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()["n"]
        assert active == 1
        conn.close()


class TestRollup:
    def test_rollup_disabled_via_config(self, memohood):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        cfg["consolidate"]["enabled"] = False
        result = memohood.consolidate.run_rollup(conn, cfg)
        assert result == {"day": 0, "week": 0, "month": 0}
        conn.close()

    def test_rollup_creates_summary_capture(self, memohood, monkeypatch):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
        cfg = _cfg(memohood)
        now = memohood.db.now()
        old = now - 2 * 86400.0  # older than the 1-day "day" rollup cutoff
        import uuid
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

        monkeypatch.setattr(memohood.extract_llm, "summarize", lambda texts, *, level="day", conn=None: "Итоговое резюме дня.")
        result = memohood.consolidate.run_rollup(conn, cfg)
        assert result["day"] == 1

        summary_row = conn.execute(
            "SELECT * FROM captures WHERE kind='summary' AND tags LIKE '%consolidation_summary%'"
        ).fetchone()
        assert summary_row is not None
        assert summary_row["content"] == "Итоговое резюме дня."

        rolled = conn.execute(
            "SELECT COUNT(*) AS n FROM captures WHERE kind='fact' AND tags LIKE '%rolled_up%'"
        ).fetchone()["n"]
        assert rolled == 6
        conn.close()


class TestRunNightly:
    def test_run_nightly_stage_isolation(self, memohood, monkeypatch):
        conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))

        def boom(*a, **kw):
            raise RuntimeError("rollup exploded")

        monkeypatch.setattr(memohood.consolidate, "run_rollup", boom)
        result = memohood.consolidate.run_nightly(conn, _cfg(memohood))
        assert result["rollup"] == {"error": True}
        assert "decay" in result and result["decay"] != {"error": True}
        assert "fts_rebuild" in result and result["fts_rebuild"] != {"error": True}
        conn.close()
