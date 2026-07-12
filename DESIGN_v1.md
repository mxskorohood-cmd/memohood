# memohood v1 — Build Spec (dialogue memory provider, ships as **MemoHood**)

A hermes **MemoryProvider** plugin: `~/.hermes/plugins/memory/memohood/` (name: `memohood`), activated by `memory.provider: memohood`. Auto-recall of past dialogue + auto-captured facts. NOT a general plugin — uses the memory-provider loader (which DOES support pip_dependencies lazy-install, unlike general plugins).

READ FIRST: the project's internal memory-design notes and decisions record (memory design, donor mechanisms, borrowed mechanisms, gap-closure; decisions accepted 2026-07-06) and the plugin loader's API contract — both private planning docs, not part of this repo. MemoryProvider ABC lives in your local hermes-agent installation (`<HERMES_HOME>/hermes-agent/agent/memory_provider.py`) — READ IT, implement against the REAL ABC (v0.18.0). Existing bundled providers in `.../plugins/memory/{holographic,mem0,honcho}/` are reference implementations — read holographic (local SQLite+FTS5) especially.

**`_engine/` hybrid-search module**: `retrieve.py`, `embed.py`, `rerank.py`, `stem.py`, `security.py` (fence_untrusted, scan_secrets), `ledger.py` implement FTS5+vector+RRF+rerank search, the spend ledger, and security guards as a self-contained package under `plugins/memory/memohood/_engine/`, with zero import dependency on any other plugin.

## Non-negotiables
- Injection tag is `<memory-context>` (hermes StreamingContextScrubber keyed to it) — emit via hermes's `build_memory_context_block()`, never a custom tag. kwarg is `hermes_home` (NOT memohood_home).
- prefetch NEVER mutates persisted history — recall wraps only the current API user message (hermes handles the wrap when prefetch returns text).
- sync_turn MUST be non-blocking (daemon thread).
- User strings RU simple; code EN. State under `get_hermes_home()` → `memory.db` beside `state.db`.
- External HTTP (Cloudflare embed, Cohere rerank, Gemini extract): browser UA, timeout, backoff, log to spend ledger, honor monthly ceiling.
- Never modify core files. Never run hermes CLI.

## Files → this repo's root  (installs to ~/.hermes/plugins/memory/memohood/)
```
plugin.yaml          # name: memohood, kind: exclusive (memory provider), pip_dependencies: [sqlite-vec, PyStemmer, ftfy, requests], author: Maxim Vasko, hooks list
__init__.py          # register(ctx): ctx.register_memory_provider(MemoHoodMemoryProvider())
provider.py          # MemoHoodMemoryProvider(MemoryProvider) — full ABC impl
db.py                # memory.db schema; WAL, busy_timeout, synchronous=NORMAL; catch_up_from_state (watermark)
capture.py           # two-stage fact extraction: keyword signals (free) + Gemini borderline; supersede 3-tier; pinned tier
consolidate.py       # nightly rollup (day→week→month), Ebbinghaus decay per-kind (pinned exempt), anti-loop flag
extract_llm.py       # Gemini gemini-2.5-flash-lite via OpenAI-compat REST (generativelanguage.../v1beta/openai/), GEMINI_API_KEY, browser UA
query_norm.py        # _meaningful_terms: strip RU/EN stopwords/pronouns/question-words, keep CamelCase/UPPER_SNAKE/digits/paths; RU-aware
tools.py             # memohood_search, memohood_fetch, memohood_recall(recall_memory), memohood_stats, memohood_capture(manual)
cli.py               # hermes memohood status|stats|reindex|seed
_engine/             # hybrid-search engine: retrieve.py, embed.py, rerank.py, stem.py, security.py, ledger.py
README.md / README.en.md / GUIDE.md / SKILL/ / LICENSE
tests/
```

## memory.db schema
```sql
captures(id TEXT PK, content TEXT, kind TEXT,           -- persona|event|preference|decision|correction|fact|instruction
  confidence REAL, notability TEXT,                      -- high|medium|low (gbrain triage)
  source TEXT,                                            -- EXTRACTED|INFERRED|AMBIGUOUS (iva)
  pinned INTEGER DEFAULT 0,                               -- 1 = identity/safety/medical: exempt from decay
  supersedes TEXT DEFAULT '', history TEXT DEFAULT '',    -- dated history on supersede (iva ADD/SUPERSEDE/NOOP)
  session_id TEXT, message_id INTEGER, tags TEXT,
  last_seen_at REAL, created_at REAL, updated_at REAL,
  valid_from REAL, invalidated_at REAL,                   -- bi-temporal (Graphiti)
  embed_signature TEXT)
captures_fts USING fts5(content, content_stem, capture_id UNINDEXED, tokenize='unicode61')   -- RU-stemmed leg
captures_vec USING vec0(...)                              -- Cloudflare BGE-M3 1024-dim
messages_fts USING fts5(...)                              -- catch_up index over state.db messages (RU-stemmed)
signals(id INTEGER PK, session_id, signal_type, score, content, message_id, created_at)
session_tags(session_id, tag, source, created_at, PK(session_id,tag))
session_links(from_session_id, to_session_id, relationship, label, weight, created_at)
spend(...)                                                -- external-call spend ledger
_meta(key PK, value)                                      -- schema_version, last_indexed_message_id
```

## Provider ABC methods (implement against REAL agent/memory_provider.py)
- `name`→"memohood"; `is_available()`→True (no network); `initialize(session_id, **kw)`: hermes_home=kw["hermes_home"], open memory.db, `catch_up_from_state()` (index state.db messages incrementally by watermark, RU-stem into messages_fts).
- `system_prompt_block()`: static RU/EN text describing persistent memory + tools (NO recalled facts, just mechanism).
- `prefetch(query, *, session_id="")`: gate (v1 = pass-through; model2vec optional later) → query_norm → hybrid search via _engine.retrieve over captures + messages (FTS RU-stem + Cloudflare vec + RRF + optional Cohere rerank) → dedup/format → return text (hermes wraps in `<memory-context>`). **Skip prefetch for delegated/child sessions** (check session kind — no memory bleed into retrieval-focused sub-agents).
- `queue_prefetch(query)`: pre-warm next turn.
- `sync_turn(user_content, assistant_content, *, session_id="", messages=None)`: daemon thread → capture.extract_and_store (two-stage) → signals/tags/links. If interrupted/error, don't write partial.
- `on_pre_compress(messages)`: rescue high-score insights into captures before compaction.
- `on_session_end`, `on_session_switch(new_id, reset)`, `on_memory_write(action,target,content)` (bridge builtin MEMORY.md writes), `on_delegation(task,result,child_session_id)`.
- `get_config_schema()`/`save_config()` (for `hermes memory setup`), `backup_paths()`→[memory.db], `shutdown()`.

## Capture (capture.py) — two-stage, accepted design
1. Free keyword signals (curated lists, RU+EN): correction (не так/wrong/actually), decision (решили/go with), preference (предпочитаю/always/never), remember (запомни), url, path. Assistant-side: remember:/важно:/ключевой вывод/root cause.
2. Borderline band (score between definite-keep/definite-drop) → ONE Gemini flash-lite call: {is_memorable, kind, notability, source_type(EXTRACTED/INFERRED), pinned?}. Clear keep/drop skip the LLM.
3. Injection-sanitize the turn text IN and the extracted fact OUT (security.scan/fence).
4. Supersede: new fact vs existing candidates (recall) → cosine≥0.95 dup (no LLM) → else Gemini judge duplicate|supersede|independent → fallback cosine≥0.92. On supersede: new value on top, old → dated `history` line, set invalidated_at on old.
5. pinned kinds (identity/safety/medical/explicit "запомни навсегда") → pinned=1, decay-exempt.
6. Embed the capture (Cloudflare) for the vector leg; write FTS (RU-stem) + vec.

## Consolidation (consolidate.py) — via hermes cron (nightly), Gemini flash-lite
- Rollup day→week→month summaries as capture kind=summary (anti-loop flag: never re-extract from summary output).
- Ebbinghaus decay: `confidence × exp(-age_days/halflife)`, halflife per kind (event≈7, preference/decision≈90, fact/belief≈365); **pinned exempt**; reinforce on access (last_seen_at). Drop below floor to archive (not delete).
- Dedup captures; rebuild FTS; health/stats.

## Config (config.yaml memory.*)
```yaml
memory:
  provider: memohood
  memohood:
    gate: {backend: pass}                 # pass-through v1; model2vec later
    model: {provider: gemini, model: gemini-2.5-flash-lite}   # extraction/consolidation
    embedder: {provider: cloudflare, model: "@cf/baai/bge-m3", dims: 1024}   # reuse kb keys
    rerank: {provider: cohere, enabled: true}
    auto_capture: true
    capture_threshold: 4.0
    monthly_ceiling_usd: {cloudflare: 5, cohere: 5, gemini: 5}
```

## Tests (live venv, pytest, HERMES_HOME→tmp, hermes_plugins namespace load like disk-cleanup)
- ABC conformance: MemoHoodMemoryProvider satisfies the real MemoryProvider (all abstract methods present).
- initialize creates memory.db with all tables; catch_up watermark works; re-run is idempotent.
- capture: explicit signal → capture written (no LLM, mock); borderline → Gemini call (mocked) → capture; noise → skipped.
- supersede: cosine dup path (no LLM), Gemini judge path (mock), history line appended, old invalidated.
- pinned fact not decayed by consolidate; ordinary fact decayed.
- prefetch returns text; child/delegated session → prefetch skipped (no memory bleed).
- injection sanitize in+out.
- RU stemming recall: capture "договор" found by query "договора".
- Integration (skip if no keys): 1 real Cloudflare embed + 1 Gemini extraction call, ~2 calls total.
- Run with the venv python + import-mode=importlib, pytest.ini inside tests/ (lesson learned the hard way).

## Out of v1: model2vec gate training, PostRecall MMR/cluster (chronological+rerank is enough for v1), graph-rerank via session_links (v1.1), Obsidian export of memory.
