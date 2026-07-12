"""``hermes memohood status|stats|reindex|seed|consolidate|setup`` -- CLI entry
point (DESIGN_v1.md "cli.py: hermes memohood status|stats|reindex|seed").

IMPORTANT (found by code inspection of the REAL loader, 2026-07-06 --
the plugin loader's ``register_cli_command`` convention is for
GENERAL plugins only): memory-provider CLI commands are NOT wired up via
``ctx.register_cli_command()`` at all. ``plugins/memory/__init__.py``'s own
``discover_plugin_cli_commands()`` (see also ``AGENTS.md``'s "Memory-provider
plugins" section) scans the ACTIVE provider's ``cli.py`` for two specific
module-level names:

  * ``register_cli(subparser)`` -- builds the argparse subcommand tree
    (used directly as ``setup_fn``).
  * ``memohood_command(args)`` -- the dispatch handler (looked up as
    ``f"{provider_name}_command"``, mirroring ``honcho_command`` in
    ``plugins/memory/honcho/cli.py``).

The provider is registered separately via ``__init__.py``'s
``register(ctx): ctx.register_memory_provider(...)`` (that path goes through
a DIFFERENT, minimal ``_ProviderCollector`` context whose own
``register_cli_command`` is a documented no-op -- CLI registration for
memory providers happens ONLY through the module-level functions below).
``register(ctx)`` in this file therefore no longer calls
``ctx.register_cli_command`` -- it never did anything for a memory provider.

``status``/``stats`` are aliases for the same report (``tools.memohood_stats``).
``reindex`` re-embeds every live capture into the shadow vec table (after an
``embedder`` config change) via ``_engine/embed.py``'s
``reembed_captures_shadow`` -- CLI-only (no tool/slash equivalent): a
potentially-costly re-embed is something an operator triggers explicitly
from a terminal, not something the model can call as a tool. ``seed`` runs
the (cheap, local) ``messages_fts`` catch-up
backfill and reports its progress; the LLM-based historical-fact-extraction
half of DESIGN_v1.md's "seed" concept (extracting captures from OLD history,
not just indexing it for FTS recall) is NOT implemented in this round --
this command says so honestly rather than silently doing nothing.
``consolidate`` runs ``consolidate.run_nightly()`` (decay/dedup/rollup/FTS-
rebuild) on demand -- DESIGN_v1.md describes this as a nightly ``hermes
cron`` job, but nothing in this plugin registers that cron job itself (out
of this round's scope); an operator wanting automatic nightly consolidation
must add a ``hermes cron`` entry that runs ``hermes memohood consolidate``
themselves. This subcommand is what makes that possible at all -- without
it there was no operator-triggerable path to ``consolidate.py`` whatsoever.
``setup`` runs the interactive onboarding wizard (``setup_wizard.py``):
collects the Cloudflare/Cohere/Gemini keys step by step, live-checks them
(one request each), and upserts them into ``HERMES_HOME/.env`` -- imported
lazily inside its dispatch branch so the other subcommands pay zero extra
import weight for it.
"""

from __future__ import annotations

from . import config as memohood_config
from . import consolidate as memohood_consolidate
from . import db
from . import tools as memohood_tools
from ._engine import embed as embed_mod


def register_cli(subparser) -> None:
    """Build the ``hermes memohood`` argparse subcommand tree.

    Called by ``plugins/memory/__init__.py``'s ``discover_plugin_cli_commands()``
    as ``setup_fn`` -- see this module's docstring. Kept as the module-level
    name the real loader scans for (NOT ``ctx.register_cli_command``).
    """
    sub = subparser.add_subparsers(dest="memohood_subcommand", required=True)

    sub.add_parser("status", help="Статус памяти: captures, watermark индексации истории, расходы")
    sub.add_parser("stats", help="То же, что status (алиас)")
    sub.add_parser("reindex", help="Пере-эмбеддинг всех captures в теневую таблицу (после смены embedder в конфиге)")

    p_seed = sub.add_parser(
        "seed",
        help="Догнать индекс истории диалогов (messages_fts) до текущего состояния state.db",
    )
    p_seed.add_argument(
        "--dry-run", action="store_true",
        help="Ничего не индексировать -- только показать текущий watermark",
    )

    sub.add_parser(
        "consolidate",
        help="Запустить ночную консолидацию (decay/dedup/rollup/FTS-rebuild) вручную",
    )

    sub.add_parser(
        "setup",
        help="Пошаговый мастер настройки ключей (Cloudflare/Cohere/Gemini) с записью в HERMES_HOME/.env",
    )


# Backward-compat alias -- some callers/tests may still look for the old
# private name; `register_cli` above is the one the real loader scans for.
_setup = register_cli


def _print_status(hermes_home: str) -> None:
    conn = db.get_connection(hermes_home=hermes_home)
    try:
        cfg = memohood_config.get_memohood_config_readonly()
        print(memohood_tools.memohood_stats({}, conn=conn, cfg=cfg, session_id=None))
    finally:
        conn.close()


def memohood_command(args) -> None:
    """Dispatch handler -- looked up by name (``f"{provider_name}_command"``)
    by ``discover_plugin_cli_commands()``. See this module's docstring."""
    from hermes_constants import get_hermes_home

    hermes_home = str(get_hermes_home())
    sub = getattr(args, "memohood_subcommand", None)

    if sub in ("status", "stats"):
        _print_status(hermes_home)
        return

    if sub == "setup":
        # Imported lazily: the wizard (and its live-check plumbing) is only
        # needed for this one subcommand -- keeps `hermes memohood status` etc.
        # free of any extra import weight (contract: "keep heavy imports
        # inside functions").
        from . import setup_wizard

        setup_wizard.run_wizard(hermes_home=hermes_home)
        return

    if sub == "reindex":
        conn = db.get_connection(hermes_home=hermes_home)
        try:
            cfg = memohood_config.get_memohood_config_readonly()
            print(f"Переиндексация captures начата (embed_signature: {embed_mod.embedding_signature(cfg)})...")
            result = embed_mod.reembed_captures_shadow(conn, cfg)
            print(
                f"Переиндексация завершена: {result.get('captures_embedded', 0)} записей переэмбеддено, "
                f"векторный индекс готов: {result.get('vector_index_ready')}."
            )
        except embed_mod.EmbedError as exc:
            print(f"Ошибка переиндексации: {exc}")
        finally:
            conn.close()
        return

    if sub == "seed":
        conn = db.get_connection(hermes_home=hermes_home)
        try:
            if args.dry_run:
                row = conn.execute("SELECT value FROM _meta WHERE key = 'last_indexed_message_id'").fetchone()
                print(f"Текущий watermark индексации истории: {row['value'] if row else '0'} (--dry-run: индексация не запущена).")
                return
            stats = db.catch_up_from_state(conn, hermes_home)
            print(
                f"Индекс истории (messages_fts): watermark {stats['watermark_before']} -> {stats['watermark_after']}, "
                f"проиндексировано {stats['indexed']} сообщений за {stats['batches']} батч(ей) "
                f"(пропущено пустых: {stats['skipped_empty']})."
            )
            print(
                "Извлечение ФАКТОВ (captures) из старой истории через LLM пока не реализовано отдельным "
                "проходом -- capture.process_turn работает только для НОВЫХ ходов. messages_fts уже "
                "наполнен и доступен для recall (memohood_recall/prefetch)."
            )
        finally:
            conn.close()
        return

    if sub == "consolidate":
        conn = db.get_connection(hermes_home=hermes_home)
        try:
            cfg = memohood_config.get_memohood_config_readonly()
            print("Запуск ночной консолидации (decay/dedup/rollup/FTS-rebuild)...")
            result = memohood_consolidate.run_nightly(conn, cfg)
            decay = result.get("decay", {})
            dedup = result.get("dedup", {})
            rollup = result.get("rollup", {})
            fts = result.get("fts_rebuild", {})
            print(
                f"decay: обновлено={decay.get('decayed', 0)}, архивировано={decay.get('archived', 0)}"
                if not decay.get("error") else "decay: ОШИБКА"
            )
            print(
                f"dedup: объединено={dedup.get('merged', 0)}" if not dedup.get("error") else "dedup: ОШИБКА"
            )
            print(
                f"rollup: day={rollup.get('day', 0)} week={rollup.get('week', 0)} month={rollup.get('month', 0)}"
                if not rollup.get("error") else "rollup: ОШИБКА"
            )
            print(
                f"fts_rebuild: перестроено={fts.get('rebuilt', 0)}" if not fts.get("error") else "fts_rebuild: ОШИБКА"
            )
        finally:
            conn.close()
        return

    print(f"Неизвестная подкоманда: {sub}")


# Backward-compat alias -- see `memohood_command` above (the name the real loader
# scans for via f"{provider_name}_command").
_handle = memohood_command


def register(ctx) -> None:
    """Plugin entry point for `register_memory_provider` (called from
    `__init__.py`'s own `register(ctx)`). Deliberately does NOT call
    `ctx.register_cli_command` -- that path is a no-op for memory providers
    (see this module's docstring); `register_cli`/`memohood_command` above are
    the real CLI wiring, picked up by `plugins/memory/__init__.py`'s
    `discover_plugin_cli_commands()` independently of this function.
    Kept only so a generic caller that treats memohood as an ordinary plugin
    (e.g. a future general-plugin-style loader) still finds an inert,
    harmless `register(ctx)` here instead of an AttributeError.
    """
