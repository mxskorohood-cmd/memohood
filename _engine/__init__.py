"""Hybrid-search engine for the memohood memory provider.

This package implements the retrieval stack behind memohood's captures:
FTS5 (BM25, RU-stemmed) plus vector search fused via Reciprocal Rank
Fusion, with an optional Cohere rerank pass on top. It is a standalone,
dependency-free package that memohood's ``db.py``/``capture.py``/
``provider.py`` import via
``from ._engine import stem, security, embed, rerank, retrieve, ledger``.

Schema conventions used throughout this package:

  * ``captures``/``captures_fts`` -- the single global corpus of memory
    records (no per-collection split; each row already carries its own
    metadata columns).
  * ``capture_id`` identifies a record; ``content``/``content_stem`` hold
    its raw and stemmed text.
  * ``captures_vec`` is one global vec0 table for the whole memory corpus.
  * ``invalidated_at`` marks a record superseded/archived, alongside the
    bi-temporal ``valid_from`` column (see DESIGN_v1.md's schema).
"""

from __future__ import annotations
