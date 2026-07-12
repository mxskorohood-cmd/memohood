"""``MemoHoodMemoryProvider`` -- hermes ``MemoryProvider`` implementation for
memohood's dialogue memory (DESIGN_v1.md "Provider ABC methods (implement
against REAL agent/memory_provider.py)").

Every abstract method of ``agent.memory_provider.MemoryProvider`` (v0.18.0)
is implemented; every documented optional hook is overridden too, per
DESIGN_v1.md's method list.

Threading contract (non-negotiable: "sync_turn MUST be non-blocking"):
``self._conn`` (opened once in :meth:`initialize`) is used ONLY from the
thread that calls the provider's synchronous methods (``prefetch``,
``on_pre_compress``, ``handle_tool_call``, etc. -- always the agent's own
turn-processing thread, never a thread this provider spawns itself).
Every background thread this provider starts (``queue_prefetch``,
``sync_turn``, ``on_memory_write``, ``on_delegation``) opens its OWN fresh
``sqlite3.Connection`` via ``db.get_connection()`` and closes it when done
-- Python's ``sqlite3.Connection`` is not safe to share across threads
without ``check_same_thread=False`` (which this project deliberately does
not set, preferring one connection per thread over disabling sqlite3's own
safety check).

Child/delegated-session handling (DESIGN_v1.md: "Skip prefetch for
delegated/child sessions ... no memory bleed into KB sub-agents"): hermes-
core already skips ``MemoryProvider.initialize()`` entirely for delegated
subagents via ``delegate_task``'s ``skip_memory=True`` (confirmed by code
inspection of ``tools/delegate_tool.py``) -- so in the common case this
provider is simply never instantiated for a child session at all. This
class still checks ``kwargs["agent_context"]``/``kwargs["parent_session_id"]``
at ``initialize()`` time and sets ``self._is_child`` accordingly, as
defense in depth for any future caller path that initializes a provider for
a non-primary context without ``skip_memory``. When ``self._is_child`` is
set, every recall/capture/tool-schema method is a hard no-op.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from . import capture as capture_mod
from . import config as memohood_config
from . import db
from . import gate as gate_mod
from . import graph_rerank as graph_rerank_mod
from . import post_recall
from . import query_norm
from ._engine import retrieve as retrieve_mod

logger = logging.getLogger("memohood.provider")

_CHILD_AGENT_CONTEXTS = frozenset({"subagent", "cron", "flush"})

_SYSTEM_PROMPT_BLOCK = (
    "# Память (memohood)\n"
    "У тебя есть постоянная память диалогов: перед каждым ходом релевантные факты и решения "
    "из прошлых разговоров автоматически подмешиваются в контекст (блок <memory-context>, если "
    "он есть) -- специально вызывать recall не нужно. Явно попросить что-то запомнить навсегда "
    "можно фразой «запомни, что ...» -- такие факты не забываются со временем.\n\n"
    "# Memory (memohood)\n"
    "You have persistent dialogue memory: relevant facts/decisions from past conversations are "
    "automatically recalled before each turn (see the <memory-context> block, if present) -- no "
    "explicit recall call needed. Say \"remember that ...\" to pin a fact permanently."
)


class MemoHoodMemoryProvider(MemoryProvider):
    """Dialogue-memory provider: auto-recall + auto-capture, hybrid
    FTS5(RU-stem)+vector(Cloudflare BGE-M3)+RRF+Cohere-rerank search over
    ``captures`` and a catch-up FTS index of ``state.db`` messages."""

    def __init__(self) -> None:
        self._hermes_home: Optional[str] = None
        self._session_id: str = ""
        self._conn: Optional[Any] = None
        self._cfg: Dict[str, Any] = {}
        self._is_child = False
        self._agent_context = "primary"

        self._prefetch_cache: Dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        self._bg_threads: List[threading.Thread] = []
        self._bg_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "memohood"

    def is_available(self) -> bool:
        # Local-only check, no network: sqlite3/stdlib is always available.
        # Optional deps (sqlite-vec/PyStemmer/requests) degrade gracefully
        # inside db.py/_engine -- memohood works FTS-only without them.
        return True

    # -- lifecycle ------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._hermes_home = kwargs.get("hermes_home")
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context") or "primary"
        parent_session_id = kwargs.get("parent_session_id") or ""
        self._is_child = bool(parent_session_id) or self._agent_context in _CHILD_AGENT_CONTEXTS

        if not self._hermes_home:
            logger.error("memohood.initialize: no hermes_home in kwargs; provider will not persist correctly")
            return

        try:
            self._cfg = memohood_config.get_memohood_config_readonly()
        except Exception:
            logger.warning("memohood.initialize: failed to load memory.memohood config; using defaults", exc_info=True)
            self._cfg = dict(memohood_config.DEFAULTS)

        try:
            self._conn = db.get_connection(hermes_home=self._hermes_home)
        except db.DbError:
            logger.error("memohood.initialize: failed to open memory.db", exc_info=True)
            self._conn = None
            return

        if not self._is_child:
            try:
                stats = db.catch_up_from_state(self._conn, self._hermes_home)
                if stats.get("indexed"):
                    logger.info(
                        "memohood: catch_up_from_state indexed %d message(s) in %d batch(es)",
                        stats["indexed"], stats["batches"],
                    )
            except Exception:  # noqa: BLE001 - a backfill hiccup must not fail agent startup
                logger.warning("memohood.initialize: catch_up_from_state failed", exc_info=True)

    def system_prompt_block(self) -> str:
        if self._is_child or self._conn is None:
            return ""
        return _SYSTEM_PROMPT_BLOCK

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._is_child or self._conn is None:
            return []
        from . import tools as memohood_tools

        return memohood_tools.ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if self._conn is None:
            return "memohood: memory.db не инициализирована."
        from . import tools as memohood_tools

        session_id = kwargs.get("session_id") or self._session_id
        return memohood_tools.dispatch(tool_name, args, conn=self._conn, cfg=self._cfg, session_id=session_id)

    # -- recall -----------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        terms = query_norm.meaningful_terms(query)
        return " ".join(terms) if terms else query

    def _reinforce(self, conn: Any, captures: List[Dict[str, Any]]) -> None:
        ids = [c["capture_id"] for c in captures if c.get("capture_id")]
        if not ids:
            return
        try:
            now = db.now()
            placeholders = ",".join("?" for _ in ids)
            with conn:
                conn.execute(f"UPDATE captures SET last_seen_at = ? WHERE id IN ({placeholders})", [now, *ids])
        except Exception:  # noqa: BLE001 - reinforcement is best-effort, never worth failing prefetch over
            logger.debug("memohood: failed to reinforce last_seen_at", exc_info=True)

    def _compute_prefetch_text(self, conn: Any, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""

        recall_ok, gate_score, gate_reason = gate_mod.should_recall(query, cfg=self._cfg)
        if not recall_ok:
            logger.debug("memohood.prefetch: gate skipped recall (score=%.3f, %s)", gate_score, gate_reason)
            return ""

        normalized = self._normalize_query(query)
        recall_cfg = self._cfg.get("recall") or {}
        k = int(recall_cfg.get("k", 8))
        messages_k = int(recall_cfg.get("messages_k", 4))

        try:
            captures = retrieve_mod.hybrid_search(conn, normalized, k, self._cfg)
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: hybrid_search failed", exc_info=True)
            captures = []

        # v1.1: session_links BOOST + 1-hop EXPANSION. graph_rerank() already
        # degrades internally and never raises; this try/except is defense in
        # depth only, matching every other call site in this method.
        try:
            captures = graph_rerank_mod.graph_rerank(captures, db=conn, cfg=self._cfg)
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: graph_rerank failed", exc_info=True)

        # v1.1: MMR + near-duplicate collapse diversity pass. Both functions
        # are pure/never-raise by contract -- no try/except needed here (see
        # post_recall.py's own module docstring "WIRING" note).
        captures = post_recall.attach_vectors(conn, captures)
        captures = post_recall.diversify(captures, cfg=self._cfg, query=normalized)

        try:
            messages = retrieve_mod.fts_search_messages(conn, normalized, messages_k)
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: fts_search_messages failed", exc_info=True)
            messages = []

        if not captures and not messages:
            return ""

        self._reinforce(conn, captures)

        lines: List[str] = []
        if captures:
            lines.append("Воспоминания (факты/решения из прошлых разговоров):")
            for c in captures:
                tag = "[закреплено] " if c.get("pinned") else ""
                lines.append(f"- {tag}[{c.get('kind')}] {c.get('text')}")
        if messages:
            if lines:
                lines.append("")
            lines.append("Похожие места в истории диалога:")
            for m in messages:
                role = m.get("role") or "?"
                content = (m.get("content") or "")[:300]
                lines.append(f"- ({role}) {content}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._is_child or self._conn is None:
            return ""
        key = session_id or self._session_id or "-"
        with self._prefetch_lock:
            cached = self._prefetch_cache.pop(key, None)
        if cached is not None:
            return cached
        try:
            return self._compute_prefetch_text(self._conn, query)
        except Exception:  # noqa: BLE001 - prefetch must never fail a turn
            logger.warning("memohood.prefetch: unexpected error", exc_info=True)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_child or self._hermes_home is None:
            return
        key = session_id or self._session_id or "-"
        hermes_home = self._hermes_home

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                return
            try:
                text = self._compute_prefetch_text(conn, query)
            except Exception:  # noqa: BLE001
                logger.debug("memohood.queue_prefetch: background compute failed", exc_info=True)
                return
            finally:
                conn.close()
            with self._prefetch_lock:
                self._prefetch_cache[key] = text

        t = threading.Thread(target=_run, daemon=True, name="memohood-prefetch")
        t.start()
        self._track_thread(t)

    # -- capture ------------------------------------------------------------------

    def _track_thread(self, t: threading.Thread) -> None:
        with self._bg_lock:
            self._bg_threads = [x for x in self._bg_threads if x.is_alive()]
            self._bg_threads.append(t)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._is_child or self._hermes_home is None:
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return
        if not self._cfg.get("auto_capture", True):
            return

        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = session_id or self._session_id

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                logger.warning("memohood.sync_turn: failed to open memory.db in background thread", exc_info=True)
                return
            try:
                capture_mod.process_turn(conn, user_content, assistant_content, session_id=sid, cfg=cfg)
            except Exception:  # noqa: BLE001 - a capture failure must never surface to the conversation loop
                logger.error("memohood.sync_turn: capture.process_turn failed", exc_info=True)
            finally:
                conn.close()

        t = threading.Thread(target=_run, daemon=True, name="memohood-sync-turn")
        t.start()
        self._track_thread(t)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._is_child or self._conn is None or not messages:
            return ""
        rescued: List[str] = []
        try:
            threshold = float(self._cfg.get("capture_threshold", 4.0))
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                side = "assistant" if msg.get("role") == "assistant" else "user"
                sig = capture_mod.compute_signals(content, side=side)
                if sig["score"] < threshold:
                    continue  # only rescue DEFINITE-KEEP insights here -- no LLM calls during compression
                result = capture_mod.extract_and_store(
                    self._conn, content, side=side, session_id=self._session_id, cfg=self._cfg,
                )
                if result and result.get("capture_id"):
                    rescued.append(content[:200])
        except Exception:  # noqa: BLE001
            logger.warning("memohood.on_pre_compress: rescue pass failed", exc_info=True)

        if not rescued:
            return ""
        bullets = "\n".join(f"- {r}" for r in rescued)
        return f"memohood rescued {len(rescued)} durable fact(s) into memory before this context was compressed:\n{bullets}"

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._join_background(timeout=5.0)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id
        if reset:
            with self._prefetch_lock:
                self._prefetch_cache.clear()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._is_child or self._hermes_home is None or action != "add" or not content:
            return
        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = self._session_id
        kind = "persona" if target == "user" else "instruction"

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                return
            try:
                capture_mod.manual_capture(
                    conn, content, kind=kind, notability="medium", pinned=False, session_id=sid, cfg=cfg,
                )
            except Exception:  # noqa: BLE001
                logger.debug("memohood.on_memory_write: mirror failed", exc_info=True)
            finally:
                conn.close()

        t = threading.Thread(target=_run, daemon=True, name="memohood-memwrite")
        t.start()
        self._track_thread(t)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> None:
        if self._is_child or self._hermes_home is None or not task:
            return
        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = self._session_id
        content = f"Делегирование ({child_session_id or '?'}): {task[:300]} -> {(result or '')[:300]}"

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                return
            try:
                capture_mod.manual_capture(
                    conn, content, kind="event", notability="low", pinned=False, session_id=sid, cfg=cfg,
                )
            except Exception:  # noqa: BLE001
                logger.debug("memohood.on_delegation: observation capture failed", exc_info=True)
            finally:
                conn.close()

        t = threading.Thread(target=_run, daemon=True, name="memohood-delegation")
        t.start()
        self._track_thread(t)

    # -- config / setup ---------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "gate.backend", "description": "Бэкенд гейта recall (pass = всегда вспоминать, model2vec = локальный классификатор)", "default": "pass", "choices": ["pass", "model2vec"]},
            {"key": "gate.threshold", "description": "gate: мин. похожесть на негативные примеры, чтобы реально пропустить recall", "default": 0.5},
            {"key": "gate.margin", "description": "gate: допустимое отставание позитивной похожести от негативной", "default": 0.05},
            {"key": "gate.model2vec_model", "description": "gate: модель model2vec (лениво устанавливается)", "default": "minishlab/potion-base-8M"},
            {"key": "gate.meaningful_terms_floor", "description": "gate: мин. число значимых слов в запросе, чтобы не гейтить вовсе", "default": 3},
            {"key": "post_recall.mmr.enabled", "description": "Включить MMR-разнообразие после ретрива", "default": True, "choices": [True, False]},
            {"key": "post_recall.mmr.lambda", "description": "post_recall: баланс релевантность/разнообразие (1.0 = только релевантность)", "default": 0.7},
            {"key": "post_recall.cluster.enabled", "description": "post_recall: схлопывать почти-дубликаты перед MMR", "default": True, "choices": [True, False]},
            {"key": "post_recall.cluster.threshold", "description": "post_recall: порог косинусной похожести для схлопывания дубликатов", "default": 0.93},
            {"key": "graph_rerank.enabled", "description": "Буст+добор воспоминаний по графу связанных сессий (session_links)", "default": True, "choices": [True, False]},
            {"key": "graph_rerank.max_neighbors", "description": "graph_rerank: макс. число новых воспоминаний из связанных сессий за раз", "default": 3},
            {"key": "model.provider", "description": "Провайдер LLM для извлечения/консолидации", "default": "gemini"},
            {"key": "model.model", "description": "Модель для извлечения/консолидации", "default": "gemini-2.5-flash-lite"},
            {"key": "embedder.provider", "description": "Провайдер эмбеддингов для captures", "default": "cloudflare"},
            {"key": "embedder.model", "description": "Модель эмбеддингов", "default": "@cf/baai/bge-m3"},
            {"key": "embedder.dims", "description": "Размерность эмбеддинга", "default": 1024},
            {"key": "rerank.enabled", "description": "Включить реранк Cohere", "default": True, "choices": [True, False]},
            {"key": "auto_capture", "description": "Авто-извлечение фактов из каждого хода", "default": True, "choices": [True, False]},
            {"key": "capture_threshold", "description": "Порог бесплатного сигнального скоринга для гарантированного захвата", "default": 4.0},
            {
                "key": "GEMINI_API_KEY", "description": "Ключ Gemini (извлечение/консолидация)", "secret": True,
                "env_var": "GEMINI_API_KEY", "url": "https://aistudio.google.com/apikey",
            },
            {"key": "CLOUDFLARE_ACCOUNT_ID", "description": "Cloudflare account id (эмбеддинги)", "secret": True, "env_var": "CLOUDFLARE_ACCOUNT_ID"},
            {"key": "CLOUDFLARE_API_TOKEN", "description": "Cloudflare Workers AI токен (эмбеддинги)", "secret": True, "env_var": "CLOUDFLARE_API_TOKEN"},
            {
                "key": "COHERE_API_KEY", "description": "Ключ Cohere (реранк)", "secret": True,
                "env_var": "COHERE_API_KEY", "url": "https://dashboard.cohere.com/api-keys",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            memohood_config.save_memohood_config_at(values, hermes_home)
        except Exception:  # noqa: BLE001 - setup-wizard writer must degrade, not crash `hermes memory setup`
            logger.error("memohood.save_config: failed to persist config", exc_info=True)

    def backup_paths(self) -> List[str]:
        # memory.db lives INSIDE HERMES_HOME (beside state.db) -- `hermes
        # backup` already walks HERMES_HOME on its own, so there is nothing
        # external to declare (see MemoryProvider.backup_paths' own
        # docstring: paths this provider stores OUTSIDE HERMES_HOME).
        return []

    def shutdown(self) -> None:
        self._join_background(timeout=3.0)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("memohood.shutdown: error closing memory.db", exc_info=True)
            self._conn = None

    def _join_background(self, *, timeout: float) -> None:
        with self._bg_lock:
            threads = list(self._bg_threads)
        for t in threads:
            if t.is_alive():
                t.join(timeout=timeout)
