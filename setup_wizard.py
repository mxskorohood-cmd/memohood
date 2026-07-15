"""``hermes memohood setup`` -- interactive onboarding wizard (plain input()/print).

Walks the operator through the three cloud keys memohood can use (Cloudflare
Workers AI embeddings, Cohere rerank, Gemini extraction/consolidation),
checks the local python dependencies, and writes the keys into
``HERMES_HOME/.env`` -- the same file hermes-core loads into the process
environment at startup (see ``_engine/embed.py``'s module docstring), which
is exactly where ``embed.py``/``rerank.py``/``extract_llm.py`` read them
back from via ``os.environ``.

Design constraints (mirroring the rest of this plugin):

* Every step is skippable with a plain Enter. memohood degrades gracefully
  without any key (FTS-only search / rrf-only ordering / signal-only
  capture -- see the per-service fallbacks in embed/rerank/extract_llm),
  so the wizard never insists, and each skip message says HONESTLY what
  the user loses.
* Live checks are exactly ONE http request each: browser ``User-Agent``
  (reused from ``_engine/security.py`` -- Cloudflare rejects bare
  python-requests UAs), 15s timeout, NO retries. An onboarding wizard must
  never turn into a retry storm.
* Secrets are never echoed back whole -- only :func:`mask_key`'s
  "first 4 chars + ellipsis" form ever reaches the console, and live-check
  error texts are scrubbed of the secret values before printing.
* The pure helpers (:func:`validate_cf_account_id`,
  :func:`validate_api_token`, :func:`validate_gemini_key`,
  :func:`mask_key`, :func:`upsert_env_var`, :func:`check_dependencies`)
  are module-level and side-effect-free so ``tests/test_setup_wizard.py``
  can unit-test them directly; the interactive flow takes an injectable
  ``input_fn`` for the same reason.
* ``hermes_home`` is an explicit argument (same philosophy as ``db.py``:
  never derive our own path unless asked to), with the standard
  ``hermes_constants.get_hermes_home()`` fallback for standalone use.
* Ctrl+C / closed stdin anywhere in the flow prints a calm "come back
  later" note instead of a traceback -- and never loses an already-
  confirmed EARLIER step's keys: each step's keys are upserted into .env
  the moment that step completes, not batched until the very end (B4). A
  ``.memohood-setup.partial.json`` checkpoint (step-completion flags ONLY,
  never key values) next to .env lets a later run resume and skip
  already-done steps (B10); it is deleted once every step has run clean.
* After a Gemini key is entered and kept, the wizard offers to pick the
  extraction/consolidation model from Gemini's live ListModels response
  (falling back silently to the compiled-in default on any failure) and
  persists a non-default choice to ``config.yaml``'s
  ``memory.memohood.model.model`` via ``config.save_memohood_config_at``
  -- never to .env, which stays secrets-only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Endpoint/model constants are imported from the modules that actually make
# the production calls, so a wizard live-check always tests the very same
# endpoint the plugin will use afterwards (no drift).
from ._engine.rerank import COHERE_RERANK_URL, DEFAULT_MODEL as COHERE_MODEL
from ._engine.security import DEFAULT_USER_AGENT
from .extract_llm import DEFAULT_MODEL as GEMINI_MODEL, GEMINI_OPENAI_COMPAT_URL

# Same-package config writer -- module-level import (not deferred) because
# config.py itself has zero load-time cost (its own heavy imports, `yaml`/
# `hermes_cli.config`, are local to its functions); mirrors provider.py's and
# cli.py's own `from . import config as memohood_config` precedent. Used by
# B6 to persist the ListModels-discovered model choice to config.yaml's
# ``memory.memohood.model.model`` -- never to .env, secrets stay separate.
from . import config as memohood_config

# Same rationale as memohood_config above -- systemd_env.py is pure stdlib
# (shutil/subprocess/pathlib/typing only, see its own module docstring), so
# importing it at module level costs nothing. Used at the end of `_run`
# (B12) to idempotently point a systemd-hosted gateway unit at
# HERMES_HOME/.env, so `hermes gateway restart` actually picks up the keys
# this wizard just wrote (no-op, never raises, when hermes isn't running
# under systemd -- see systemd_env.detect_systemd_gateway's own guard).
from . import systemd_env

# ``embed.py`` has no module-level model constant (the model comes from
# config's ``embedder.model``, default ``@cf/baai/bge-m3`` -- config.py
# DEFAULTS); the wizard checks the default since that is what a fresh
# install will use.
CF_EMBED_MODEL = "@cf/baai/bge-m3"

LIVE_CHECK_TIMEOUT_S = 15.0

# import-name -> (pip-name, что даёт) -- mirrors plugin.yaml's pip_dependencies.
_DEPENDENCIES: Tuple[Tuple[str, str], ...] = (
    ("sqlite_vec", "sqlite-vec"),
    ("Stemmer", "PyStemmer"),
    ("ftfy", "ftfy"),
    ("requests", "requests"),
)

_ACTION_RU = {
    "added": "добавлено",
    "replaced": "заменено",
    "uncommented": "раскомментировано",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly in tests/test_setup_wizard.py)
# ---------------------------------------------------------------------------

_CF_ACCOUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def validate_cf_account_id(value: str) -> bool:
    """Cloudflare account id: exactly 32 hex chars (case-insensitive)."""
    return bool(_CF_ACCOUNT_ID_RE.match((value or "").strip().lower()))


def validate_api_token(value: str) -> bool:
    """Generic API token/key: non-empty, no whitespace anywhere."""
    v = (value or "").strip()
    return bool(v) and not any(ch.isspace() for ch in v)


def validate_gemini_key(value: str) -> bool:
    """Gemini API key: non-empty, no whitespace, plausible length. Google now
    issues keys in more than one shape (legacy ``AIza…`` and newer ``AQ.…``),
    so we don't gate on the prefix -- the live check is the real validator."""
    v = (value or "").strip()
    return bool(v) and 8 <= len(v) <= 200 and not any(ch.isspace() for ch in v)


def mask_key(value: str) -> str:
    """Return a safe-to-print form of a secret: first 4 chars + ellipsis.

    NEVER returns the full value; anything 4 chars or shorter collapses to
    a bare ellipsis (showing "most of" a tiny secret is not masking).
    """
    v = value or ""
    if len(v) <= 4:
        return "…"
    return v[:4] + "…"


def upsert_env_var(path: "str | Path", key: str, value: str) -> str:
    """Insert or update ``KEY=value`` in the ``.env`` file at *path* (UTF-8).

    Rules (in priority order):
      1. an active ``KEY=...`` line exists -> replace it in place;
      2. a commented ``# KEY=...`` line exists -> uncomment it with the new
         value (in place, preserving its position);
      3. otherwise -> append at the end.

    Creates the file (and parent dirs) if missing. Returns which action was
    taken: ``"replaced"`` | ``"uncommented"`` | ``"added"`` -- callers use
    it only for the human-readable log line.
    """
    p = Path(path)
    lines: List[str] = []
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()

    active_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    commented_re = re.compile(rf"^\s*#\s*{re.escape(key)}\s*=")
    new_line = f"{key}={value}"
    action = "added"

    for i, line in enumerate(lines):
        if active_re.match(line):
            lines[i] = new_line
            action = "replaced"
            break
    else:
        for i, line in enumerate(lines):
            if commented_re.match(line):
                lines[i] = new_line
                action = "uncommented"
                break
        else:
            lines.append(new_line)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return action


# ---------------------------------------------------------------------------
# Resume checkpoint (B10): step-completion flags ONLY, next to .env.
# NEVER holds key values -- secrets live exclusively in .env (see module
# docstring / the project's secrets rule); this file just remembers WHICH of
# the three key-collecting steps already ran, so a wizard interrupted by
# EOFError/KeyboardInterrupt (B4) doesn't re-ask questions the operator
# already answered on a previous run.
# ---------------------------------------------------------------------------

PARTIAL_SETUP_FILENAME = ".memohood-setup.partial.json"

# "dependencies" is deliberately NOT in here: it has no prompts and is cheap
# enough to re-run fresh every time -- this checkpoint exists to avoid
# re-ASKING already-answered questions, not to skip a free instant check.
RESUMABLE_STEPS: Tuple[str, ...] = ("cloudflare", "cohere", "gemini")

# Human-readable labels for the "уже готово: ..." resume message only.
_STEP_LABELS: Dict[str, str] = {
    "cloudflare": "Cloudflare",
    "cohere": "Cohere",
    "gemini": "Gemini",
}


def load_setup_checkpoint(partial_path: "str | Path") -> List[str]:
    """Read the ``steps_done`` list from the resume checkpoint at
    *partial_path*. Never raises -- a missing or corrupt checkpoint just
    means "nothing to resume", exactly like a fresh install."""
    p = Path(partial_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken checkpoint must read as "start over", not crash setup
        return []
    steps = data.get("steps_done") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, str)]


def save_setup_checkpoint(partial_path: "str | Path", steps_done: List[str]) -> None:
    """Persist ONLY step-completion flags to *partial_path* -- never key
    values."""
    p = Path(partial_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"steps_done": list(steps_done)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_setup_checkpoint(partial_path: "str | Path") -> None:
    """Remove the resume checkpoint once every resumable step has completed
    in a single run. Best-effort: a failure to delete never fails setup."""
    try:
        Path(partial_path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - cleanup must never crash a finished setup
        pass


def check_dependencies() -> List[Tuple[str, str, bool]]:
    """Return ``(import_name, pip_name, importable)`` for each optional dep.

    Uses ``importlib.util.find_spec`` so nothing heavy/native is actually
    imported just to answer "is it installed?".
    """
    import importlib.util

    out: List[Tuple[str, str, bool]] = []
    for import_name, pip_name in _DEPENDENCIES:
        try:
            found = importlib.util.find_spec(import_name) is not None
        except Exception:  # noqa: BLE001 - a broken package must read as "missing", not crash setup
            found = False
        out.append((import_name, pip_name, found))
    return out


def _mask_secrets(text: str, secrets: List[str]) -> str:
    """Replace every occurrence of each secret in *text* with its mask, so
    an HTTP error body / exception repr can never leak a key to the console."""
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, mask_key(s))
    return out


# ---------------------------------------------------------------------------
# Live checks: ONE request each, browser UA, 15s timeout, no retries.
# ---------------------------------------------------------------------------


def _live_headers(bearer: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }


def check_cloudflare(account_id: str, api_token: str) -> Tuple[bool, str]:
    """One embedding request against Workers AI. Returns (ok, human message)."""
    import requests  # heavy/optional import kept local (plugin convention)

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CF_EMBED_MODEL}"
    secrets = [account_id, api_token]
    try:
        resp = requests.post(
            url, headers=_live_headers(api_token), json={"text": ["привет"]},
            timeout=LIVE_CHECK_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return False, _mask_secrets(f"сеть/соединение: {exc}", secrets)
    if resp.status_code != 200:
        return False, _mask_secrets(f"HTTP {resp.status_code}: {resp.text[:200]}", secrets)
    try:
        data = resp.json()
    except ValueError:
        return False, "ответ не в формате JSON"
    if isinstance(data, dict) and data.get("success") is False:
        return False, _mask_secrets(f"Cloudflare ответил success=false: {str(data.get('errors'))[:200]}", secrets)
    return True, "эмбеддинг получен, ключи рабочие"


def check_cohere(api_key: str) -> Tuple[bool, str]:
    """One rerank request with two tiny documents. Returns (ok, human message)."""
    import requests  # heavy/optional import kept local (plugin convention)

    body = {
        "model": COHERE_MODEL,
        "query": "столица Франции",
        "documents": ["Париж -- столица Франции.", "Лондон -- столица Великобритании."],
        "top_n": 1,
    }
    try:
        resp = requests.post(
            COHERE_RERANK_URL, headers=_live_headers(api_key), json=body,
            timeout=LIVE_CHECK_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return False, _mask_secrets(f"сеть/соединение: {exc}", [api_key])
    if resp.status_code != 200:
        return False, _mask_secrets(f"HTTP {resp.status_code}: {resp.text[:200]}", [api_key])
    try:
        data = resp.json()
    except ValueError:
        return False, "ответ не в формате JSON"
    if not isinstance(data, dict) or "results" not in data:
        return False, "неожиданный формат ответа rerank"
    return True, "реранк отработал, ключ рабочий"


def check_gemini(api_key: str) -> Tuple[bool, str]:
    """One tiny chat request via the OpenAI-compatible REST endpoint --
    the exact endpoint/auth style ``extract_llm.py`` uses in production."""
    import requests  # heavy/optional import kept local (plugin convention)

    body = {
        "model": GEMINI_MODEL,
        "messages": [{"role": "user", "content": "Ответь одним словом: привет"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            GEMINI_OPENAI_COMPAT_URL, headers=_live_headers(api_key), json=body,
            timeout=LIVE_CHECK_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return False, _mask_secrets(f"сеть/соединение: {exc}", [api_key])
    if resp.status_code != 200:
        return False, _mask_secrets(f"HTTP {resp.status_code}: {resp.text[:200]}", [api_key])
    try:
        data = resp.json()
    except ValueError:
        return False, "ответ не в формате JSON"
    if not isinstance(data, dict) or "choices" not in data:
        return False, "неожиданный формат ответа модели"
    return True, "модель ответила, ключ рабочий"


def discover_gemini_models(api_key: str) -> List[str]:
    """List Gemini models that support ``generateContent``, names stripped
    of their ``models/`` prefix (B6).

    Same request shape as ``plugins/hermes-setup/registry.py``'s
    ``_live_check_gemini_key`` (``GET .../v1beta/models?key=...``, browser
    UA, one attempt, no retries) but this one actually parses the body,
    which that function doesn't need to do. Never raises: any network
    error, non-200 status, or malformed JSON just yields an empty list, and
    :func:`_step_gemini_model` falls back to the compiled-in default model
    on an empty result -- B6's "грациозный фолбэк".
    """
    import requests  # heavy/optional import kept local (plugin convention)

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = requests.get(
            url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=LIVE_CHECK_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:  # noqa: BLE001 - discovery is best-effort, never fatal to setup
        return []

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []

    out: List[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = str(m.get("name") or "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------


def _ask_valid(input_fn, prompt: str, validate, invalid_msg: str) -> str:
    """Ask until *validate* passes or the user presses Enter ("" = skip)."""
    while True:
        value = (input_fn(prompt) or "").strip()
        if not value:
            return ""
        if validate(value):
            return value
        print(f"{invalid_msg} Попробуйте ещё раз (Enter = пропустить).")


def _yes(input_fn, prompt: str) -> bool:
    """Enter = да; anything starting with n/н = нет."""
    answer = (input_fn(prompt) or "").strip().lower()
    return not answer.startswith(("n", "н"))


def _yes_default_no(input_fn, prompt: str) -> bool:
    """Opposite default of :func:`_yes`: Enter/empty = НЕТ; only an explicit
    "да"/"y"/"д" counts as yes (B8). Used exactly where an Enter-happy user
    must never silently opt into a risky action -- keeping a key that just
    failed its live check."""
    answer = (input_fn(prompt) or "").strip().lower()
    return answer.startswith(("y", "д"))


def _confirm_after_check(run_check, input_fn) -> bool:
    """Offer a live check. Returns True if the entered values should be kept
    (check skipped, check passed, or check failed but the user explicitly
    insists on saving anyway).

    The two gates have DELIBERATELY different defaults (B8): gate 1 (offer a
    check at all) defaults to Enter = да -- checking is free and safe, so an
    Enter-happy user should get it. Gate 2 (keep a key that just FAILED its
    check) defaults to Enter = НЕТ (:func:`_yes_default_no`) -- an Enter-
    happy user must never silently save a key already known to be broken.
    """
    if not _yes(input_fn, "Проверить живым запросом? (Enter = да / n = нет): "):
        print("Хорошо, проверять не будем -- просто сохраним.")
        print("Ключ сохранится без проверки -- если он битый, память молча не заработает.")
        return True
    ok, msg = run_check()
    if ok:
        print(f"Проверка прошла: {msg}.")
        return True
    print(f"Проверка не прошла: {msg}.")
    return _yes_default_no(
        input_fn,
        "Сохранить ключ несмотря на ошибку? (Enter = не сохранять / «да» = сохранить): ",
    )


def _write_step_values(env_path: Path, values: Dict[str, str]) -> None:
    """Immediately persist one step's collected keys to .env and print the
    same per-key confirmation line the old end-of-run loop used to print in
    bulk (B4). Writing right after each step -- instead of batching every
    step's keys until the very end of ``_run`` -- is what stops an
    EOFError/KeyboardInterrupt on a LATER step from losing an EARLIER step's
    already-confirmed keys."""
    for key, value in values.items():
        action = upsert_env_var(env_path, key, value)
        print(f"  {key} = {mask_key(value)} -- {_ACTION_RU.get(action, action)} в {env_path}")


_CF_SKIP_MSG = "Пропущено: память будет искать только по словам (FTS), без поиска по смыслу."
_COHERE_SKIP_MSG = "Пропущено: поиск останется гибридным, но сортировка будет чуть грубее."
_GEMINI_SKIP_MSG = (
    "Пропущено: спорные факты не будут запоминаться, только очевидные; "
    "ночная консолидация -- только механическая часть."
)


def _step_cloudflare(input_fn) -> Tuple[Dict[str, str], str]:
    print()
    print("Шаг 1 из 4 -- Cloudflare Workers AI (эмбеддинги BGE-M3).")
    print("Это облачный переводчик текста в числа-смыслы: с ним память находит")
    print("записи по смыслу, а не только по точному совпадению слов.")
    print("Бесплатного лимита Cloudflare хватает с большим запасом.")
    print("Где взять: dash.cloudflare.com -> Workers AI (account id и API-токен).")

    account_id = _ask_valid(
        input_fn,
        "Cloudflare ACCOUNT_ID (Enter = пропустить): ",
        validate_cf_account_id,
        "Не похоже на account id: нужно ровно 32 символа из 0-9 и a-f.",
    )
    if not account_id:
        print(_CF_SKIP_MSG)
        return {}, "пропущено"

    api_token = _ask_valid(
        input_fn,
        "Cloudflare API_TOKEN (Enter = пропустить): ",
        validate_api_token,
        "Токен не может быть пустым или содержать пробелы.",
    )
    if not api_token:
        print(_CF_SKIP_MSG)
        return {}, "пропущено"

    if not _confirm_after_check(lambda: check_cloudflare(account_id, api_token), input_fn):
        print(_CF_SKIP_MSG)
        return {}, "пропущено (проверка не прошла)"
    return (
        {"CLOUDFLARE_ACCOUNT_ID": account_id, "CLOUDFLARE_API_TOKEN": api_token},
        f"настроено ({mask_key(api_token)})",
    )


def _step_single_key(
    input_fn,
    *,
    header_lines: Tuple[str, ...],
    env_var: str,
    prompt: str,
    validate,
    invalid_msg: str,
    check_name: str,
    skip_msg: str,
) -> Tuple[Dict[str, str], str]:
    """Shared shape of the Cohere/Gemini steps: one key, one live check.

    The check function is looked up in module globals by *check_name* at
    call time so tests can monkeypatch ``check_cohere``/``check_gemini`` on
    this module and the wizard picks the patched version up.
    """
    print()
    for line in header_lines:
        print(line)
    key = _ask_valid(input_fn, prompt, validate, invalid_msg)
    if not key:
        print(skip_msg)
        return {}, "пропущено"
    check_fn = globals()[check_name]
    if not _confirm_after_check(lambda: check_fn(key), input_fn):
        print(skip_msg)
        return {}, "пропущено (проверка не прошла)"
    return {env_var: key}, f"настроено ({mask_key(key)})"


def _step_cohere(input_fn) -> Tuple[Dict[str, str], str]:
    return _step_single_key(
        input_fn,
        header_lines=(
            "Шаг 2 из 4 -- Cohere (реранкер).",
            "Это строгий редактор: пересортировывает найденное так, что самое",
            "нужное оказывается сверху. Бесплатный ключ: dashboard.cohere.com/api-keys",
        ),
        env_var="COHERE_API_KEY",
        prompt="Cohere API_KEY (Enter = пропустить): ",
        validate=validate_api_token,
        invalid_msg="Ключ не может быть пустым или содержать пробелы.",
        check_name="check_cohere",
        skip_msg=_COHERE_SKIP_MSG,
    )


def _step_gemini(input_fn) -> Tuple[Dict[str, str], str]:
    return _step_single_key(
        input_fn,
        header_lines=(
            "Шаг 3 из 4 -- Gemini (LLM для фактов).",
            "Это младший редактор: решает судьбу спорных фактов и делает ночную",
            "уборку памяти. Модель flash-lite -- копеечная.",
            "Ключ: aistudio.google.com/apikey (бывает вида AIza… или AQ.…).",
        ),
        env_var="GEMINI_API_KEY",
        prompt="GEMINI_API_KEY (Enter = пропустить): ",
        validate=validate_gemini_key,
        invalid_msg="Ключ Gemini не должен содержать пробелов и быть разумной длины.",
        check_name="check_gemini",
        skip_msg=_GEMINI_SKIP_MSG,
    )


def _step_gemini_model(input_fn, api_key: str) -> str:
    """After a Gemini key has been entered and kept, offer to pick the
    extraction/consolidation model from Gemini's live ListModels response
    instead of blindly trusting the compiled-in default (B6). Returns the
    model id to use. Never prompts at all when discovery comes back empty --
    nothing to choose from means nothing to ask, just the graceful fallback.
    """
    print()
    print("Проверяю список доступных моделей Gemini (ListModels)...")
    models = discover_gemini_models(api_key)
    if not models:
        print(f"Не удалось получить список моделей -- оставляю модель по умолчанию ({GEMINI_MODEL}).")
        return GEMINI_MODEL

    default = GEMINI_MODEL if GEMINI_MODEL in models else models[0]
    print("Доступные модели (поддерживают generateContent):")
    for i, name in enumerate(models, start=1):
        marker = " -- по умолчанию" if name == default else ""
        print(f"  {i}. {name}{marker}")

    answer = (input_fn(f"Номер модели (Enter = {default}): ") or "").strip()
    if not answer:
        return default
    if not answer.isdigit() or not (1 <= int(answer) <= len(models)):
        print("Не похоже на номер из списка -- использую модель по умолчанию.")
        return default
    return models[int(answer) - 1]


def _step_dependencies() -> Tuple[List[str], str]:
    print()
    print("Шаг 4 из 4 -- проверка зависимостей (python-библиотек).")
    missing: List[str] = []
    for _import_name, pip_name, found in check_dependencies():
        print(f"  {pip_name:<12} {'есть' if found else 'НЕТ'}")
        if not found:
            missing.append(pip_name)
    if missing:
        print("Не хватает библиотек. Команда для установки в venv hermes:")
        print(f'  "{sys.executable}" -m pip install {" ".join(missing)}')
        print("Запускать её вручную не обязательно: hermes доустановит зависимости")
        print("сам при первом запуске памяти (pip_dependencies в plugin.yaml).")
        return missing, "не хватает: " + ", ".join(missing)
    print("Все зависимости на месте.")
    return [], "все на месте"


def _resolve_hermes_home(hermes_home: Optional[str]) -> str:
    if hermes_home:
        return str(hermes_home)
    try:
        from hermes_constants import get_hermes_home  # heavy import kept local

        return str(get_hermes_home())
    except Exception:  # noqa: BLE001 - standalone/dev run outside a hermes install
        return str(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


# ---------------------------------------------------------------------------
# Key visibility (read side) — lets `hermes memohood stats` SHOW which keys
# are configured, where the .env lives, and whether the RUNNING process has
# actually picked them up (systemd bug B12), without printing a full secret
# or making a network call. The write side is upsert_env_var/_write_step_values.
# ---------------------------------------------------------------------------

# Same line grammar upsert_env_var respects: optional indent, optional ``#``
# comment prefix, KEY, ``=``, value. Commented/empty-valued lines don't count.
_ENV_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<hash>#+\s*)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<rest>.*)$"
)


def env_file_path(hermes_home: Optional[str] = None) -> Path:
    """``<HERMES_HOME>/.env`` — the single file memohood's keys live in. Surfaced
    so stats/onboarding can SHOW the path instead of making a human hunt for it."""
    return Path(_resolve_hermes_home(hermes_home)) / ".env"


def _read_env_file_values(env_path: Path) -> Dict[str, str]:
    """``{KEY: value}`` for every ACTIVE (uncommented, non-empty) line in
    *env_path*. Reads once; never raises (missing/unreadable file -> ``{}``)."""
    result: Dict[str, str] = {}
    try:
        if not env_path.exists():
            return result
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        m = _ENV_LINE_RE.match(line)
        if not m or m.group("hash"):
            continue
        val = m.group("rest").strip()
        if val:
            result[m.group("key")] = val
    return result


def key_status(env_vars: List[str], *, hermes_home: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Read-only status of each var: ``{VAR: {"in_file","in_process","mask"}}``.

    ``in_process`` reflects what the RUNNING hermes sees (``os.environ``); a key
    present in the file but NOT here means the gateway started before it was
    added and needs a restart (bug B12). ``mask`` is :func:`mask_key`'s
    first-4-chars form (process value preferred, else file), never the full
    secret. Never raises."""
    file_vals = _read_env_file_values(env_file_path(hermes_home))
    out: Dict[str, Dict[str, Any]] = {}
    for var in env_vars:
        file_val = file_vals.get(var)
        proc_raw = os.environ.get(var)
        proc_val = proc_raw if (proc_raw and proc_raw.strip()) else None
        shown = proc_val or file_val
        out[var] = {
            "in_file": bool(file_val),
            "in_process": bool(proc_val),
            "mask": mask_key(shown) if shown else "",
        }
    return out


def relevant_keys(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered catalog of the ``.env`` keys memohood uses, from effective config:
    embedder keys (cloudflare -> account+token, local -> none), optional cohere
    rerank, optional gemini extraction. Each item ``{"env_var","role","required"}``."""
    cfg = cfg or {}
    embedder = ((cfg.get("embedder") or {}).get("provider")) or "cloudflare"
    rerank = cfg.get("rerank") or {}
    model = cfg.get("model") or {}
    items: List[Dict[str, Any]] = []
    if embedder == "cloudflare":
        items.append({"env_var": "CLOUDFLARE_ACCOUNT_ID", "role": "поиск по смыслу (Cloudflare)", "required": True})
        items.append({"env_var": "CLOUDFLARE_API_TOKEN", "role": "поиск по смыслу (Cloudflare)", "required": True})
    elif embedder in ("openai", "openai-compat"):
        items.append({"env_var": "OPENAI_API_KEY", "role": "поиск по смыслу (OpenAI-совм.)", "required": True})
    # local embedder: no keys needed.
    if rerank.get("enabled", True) and rerank.get("provider", "cohere") == "cohere":
        items.append({"env_var": "COHERE_API_KEY", "role": "сортировка (Cohere)", "required": False})
    if (model.get("provider", "gemini")) == "gemini":
        items.append({"env_var": "GEMINI_API_KEY", "role": "извлечение фактов (Gemini)", "required": False})
    return items


def _restart_hint(hermes_home: Optional[str] = None) -> str:
    """Read-only, systemd-aware restart phrasing for the stats "Ключи" block
    (never wires anything — that's the wizard's job). Never raises."""
    try:
        under = systemd_env.detect_systemd_gateway()
    except Exception:  # noqa: BLE001 - detection must never crash a status readout
        under = False
    if under:
        return "перезапустите gateway: hermes gateway restart (systemd EnvironmentFile настроен)."
    return "перезапустите hermes, чтобы подхватить ключи из .env."


def format_keys_block(cfg: Dict[str, Any], *, hermes_home: Optional[str] = None) -> str:
    """Render the read-only "Ключи" section for `hermes memohood stats`: the
    ``.env`` path + one line per relevant key — ``✓ настроен`` / ``✗ нет`` / ``⚠``
    when it is in the file but the running process hasn't picked it up. Never
    prints a full secret; never makes a network call."""
    catalog = relevant_keys(cfg)
    statuses = key_status([c["env_var"] for c in catalog], hermes_home=hermes_home)
    lines = [f"Ключи (.env: {env_file_path(hermes_home)}):"]
    stale = False
    for c in catalog:
        var = c["env_var"]
        st = statuses.get(var, {})
        role = f"[{c['role']}{'' if c['required'] else ', опц.'}]"
        if st.get("in_process"):
            lines.append(f"  {var} — ✓ настроен ({st['mask']}) {role}")
        elif st.get("in_file"):
            stale = True
            lines.append(f"  {var} — ⚠ есть в .env ({st['mask']}), но процесс не видит {role}")
        else:
            lines.append(f"  {var} — ✗ нет {role}")
    if stale:
        lines.append(f"  ⚠ Часть ключей записана, но не подхвачена процессом: {_restart_hint(hermes_home)}")
    return "\n".join(lines)


def _run(hermes_home: Optional[str], input_fn) -> None:
    home = _resolve_hermes_home(hermes_home)
    env_path = Path(home) / ".env"
    partial_path = Path(home) / PARTIAL_SETUP_FILENAME

    print("Настройка памяти MemoHood.")
    print()
    print("Мастер пройдёт 4 шага: Cloudflare (поиск по смыслу), Cohere (сортировка")
    print("результатов), Gemini (извлечение фактов) и проверка зависимостей.")
    print("Каждый шаг можно пропустить -- просто нажмите Enter: память работает и")
    print(f"без ключей, просто скромнее. Ключи будут записаны в {env_path};")
    print("в консоли они никогда не показываются целиком.")

    # B10: resume checkpoint. Holds ONLY step-completion flags (never key
    # values) -- see load_setup_checkpoint's docstring.
    steps_done: List[str] = load_setup_checkpoint(partial_path)
    if steps_done:
        print()
        labels = ", ".join(_STEP_LABELS.get(s, s) for s in steps_done)
        print(f"Найден незавершённый предыдущий запуск. Уже готово: {labels}.")
        if not _yes(input_fn, "Продолжить и пропустить готовые шаги? (Enter = да / n = начать заново): "):
            steps_done = []

    to_write: Dict[str, str] = {}
    summary: List[Tuple[str, str]] = []

    try:
        if "cloudflare" in steps_done:
            summary.append(("Cloudflare (эмбеддинги)", "уже настроено ранее -- пропущено при возобновлении"))
        else:
            values, status = _step_cloudflare(input_fn)
            to_write.update(values)
            summary.append(("Cloudflare (эмбеддинги)", status))
            if values:  # B4: write THIS step's keys immediately, don't batch until the end
                print()
                _write_step_values(env_path, values)
            steps_done.append("cloudflare")
            save_setup_checkpoint(partial_path, steps_done)

        if "cohere" in steps_done:
            summary.append(("Cohere (реранк)", "уже настроено ранее -- пропущено при возобновлении"))
        else:
            values, status = _step_cohere(input_fn)
            to_write.update(values)
            summary.append(("Cohere (реранк)", status))
            if values:
                print()
                _write_step_values(env_path, values)
            steps_done.append("cohere")
            save_setup_checkpoint(partial_path, steps_done)

        if "gemini" in steps_done:
            summary.append(("Gemini (факты)", "уже настроено ранее -- пропущено при возобновлении"))
        else:
            values, status = _step_gemini(input_fn)
            to_write.update(values)
            summary.append(("Gemini (факты)", status))
            if values:
                print()
                _write_step_values(env_path, values)
            # Mark "gemini" done (key already safely in .env) BEFORE the
            # optional model-choice prompt below, so an EOFError/
            # KeyboardInterrupt during model selection still lets a resumed
            # run skip re-asking the key it already has.
            steps_done.append("gemini")
            save_setup_checkpoint(partial_path, steps_done)

            if values:  # B6: only worth discovering a model if we HAVE a key
                chosen_model = _step_gemini_model(input_fn, values["GEMINI_API_KEY"])
                if chosen_model != GEMINI_MODEL:
                    try:
                        memohood_config.save_memohood_config_at({"model.model": chosen_model}, home)
                        summary.append(("Модель Gemini", f"выбрана {chosen_model}"))
                    except Exception:  # noqa: BLE001 - config write must degrade, not crash setup
                        summary.append(("Модель Gemini", f"не удалось сохранить выбор, останется {GEMINI_MODEL}"))
                else:
                    summary.append(("Модель Gemini", f"по умолчанию ({GEMINI_MODEL})"))

        _missing, deps_status = _step_dependencies()
        summary.append(("Зависимости", deps_status))

        if all(step in steps_done for step in RESUMABLE_STEPS):
            clear_setup_checkpoint(partial_path)
    except (EOFError, KeyboardInterrupt):
        # Belt-and-suspenders (B4): every step above already writes its OWN
        # keys to .env the moment it completes, so `to_write` here normally
        # already mirrors what's on disk -- but re-flushing it before we
        # propagate is cheap and idempotent (upsert_env_var just replaces
        # the same line), so a future refactor that reintroduces batching
        # can't silently regress into "Ctrl+D loses already-confirmed keys".
        if to_write:
            for key, value in to_write.items():
                upsert_env_var(env_path, key, value)
        raise

    print()
    if not to_write:
        if env_path.exists():
            print(f"Новых ключей в этом запуске не введено -- {env_path} оставлен как есть.")
        else:
            print(f"Ни одного ключа не введено -- {env_path} не тронут.")

    print()
    print("Итоги:")
    for name, status in summary:
        print(f"  {name}: {status}")

    # B12: idempotently point a systemd-hosted gateway unit at
    # HERMES_HOME/.env (no-op off systemd -- see systemd_env's own guard) so
    # the hint below can honestly tell a systemd user that a plain restart
    # is enough, instead of the CLI-session phrasing that doesn't apply to
    # them. Never raises -- a write/reload failure just degrades to the
    # generic hint below, same as "not systemd at all".
    systemd_wired, _systemd_msg = systemd_env.wire_gateway_env(home)

    print()
    print("Что дальше:")
    if systemd_wired:
        print(
            "  1. Перезапустите gateway: hermes gateway restart -- ключи из .env "
            "подхватятся (systemd EnvironmentFile настроен)."
        )
    else:
        print("  1. Перезапустите hermes, чтобы он подхватил ключи из .env.")
    print("  2. Спросите бота: «что ты обо мне помнишь?»")


def run_wizard(hermes_home: Optional[str] = None, *, input_fn: Callable[[str], str] = input) -> None:
    """Entry point for ``hermes memohood setup`` (see ``cli.py``).

    *input_fn* is injectable purely for tests; production callers pass
    nothing and get the builtin ``input``. Ctrl+C / EOF anywhere in the
    flow exits calmly instead of dumping a traceback.
    """
    try:
        _run(hermes_home, input_fn)
    except (KeyboardInterrupt, EOFError):
        print()
        print("Настройка прервана. Ничего не сломалось -- продолжить можно в любой момент:")
        print("  hermes memohood setup")
