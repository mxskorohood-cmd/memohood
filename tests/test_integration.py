"""Live-network integration probes. Auto-skipped when the corresponding real
key is absent from ~/.hermes/.env (never hardcoded, never committed)."""

from __future__ import annotations

import copy

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


@pytest.mark.integration
def test_real_cloudflare_embed_one_call(memohood, real_api_env):
    keys = real_api_env  # fixture returns the dict of real keys it injected
    if "CLOUDFLARE_ACCOUNT_ID" not in keys or "CLOUDFLARE_API_TOKEN" not in keys:
        pytest.skip("CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN not present in the real ~/.hermes/.env")

    cfg = _cfg(memohood)
    vectors = memohood._engine.embed.embed_texts(["hermes memohood integration probe"], cfg)
    assert len(vectors) == 1
    assert len(vectors[0]) == cfg["embedder"]["dims"]
    assert all(isinstance(x, (int, float)) for x in vectors[0])
    import math
    assert all(math.isfinite(x) for x in vectors[0])


@pytest.mark.integration
def test_real_gemini_extract_one_call(memohood, real_api_env):
    keys = real_api_env
    if "GEMINI_API_KEY" not in keys:
        pytest.skip(
            "GEMINI_API_KEY not present in the real ~/.hermes/.env (commented out as of this build) "
            "-- cannot make a real extraction call"
        )
    result = memohood.extract_llm.extract(
        "Мы решили всегда использовать PostgreSQL для новых проектов.", conn=None,
    )
    assert result is not None
    assert result["is_memorable"] in (True, False)
    assert result["kind"] in memohood.extract_llm._VALID_KINDS
