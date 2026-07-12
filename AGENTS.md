# AGENTS.md — установка плагина MemoHood

> Этот файл — инструкция по **установке** плагина MemoHood в hermes. Применяй её
> **только** когда пользователь явно просит установить/поставить MemoHood (обычно
> прислав архив плагина в чат). Во всех остальных случаях игнорируй этот файл —
> он не про использование памяти (для этого есть скилл `memohood`), а только про
> разовую установку.

## Что это

MemoHood — провайдер памяти диалогов для hermes (`kind: exclusive`). Устанавливается
копированием в `$HERMES_HOME/plugins/memohood/` и включается как
`memory.provider: memohood` в `config.yaml`. Внутри плагина уже есть готовый
идемпотентный установщик (`install.sh` для Linux/macOS, `install.ps1` для Windows),
который делает всю работу — тебе не нужно копировать файлы, ставить пакеты или
править конфиг вручную.

## Способ A — установка из присланного архива (ты, агент, ставишь сам)

Нужен доступ к terminal-инструменту и к venv, где стоит hermes.

1. **Возьми путь к присланному архиву.** Когда пользователь прислал файл в чат,
   ядро уже сохранило его на диск и подставило путь в контекст — используй этот
   локальный путь.
2. **Распакуй во временную папку** и найди внутри каталог, где лежит `plugin.yaml`
   (плагин может быть как в корне архива, так и в подпапке):
   - Linux/macOS: `mkdir -p /tmp/memohood-inst && unzip -o "<путь-к-архиву>" -d /tmp/memohood-inst`
   - Windows: `Expand-Archive -Force "<путь-к-архиву>" "$env:TEMP\memohood-inst"`
3. **Проверь, что это действительно MemoHood:** в каталоге есть `plugin.yaml` со
   строкой `name: memohood` и файл `install.sh`/`install.ps1`. Если нет —
   **остановись** и скажи пользователю, что архив не похож на MemoHood.
4. **Спроси режим векторного поиска** (или поставь по умолчанию облачный):
   - **облачный** (по умолчанию): вектор через Cloudflare; ключи можно добавить
     позже, без них память работает в keyword-режиме;
   - **локальный офлайн** (`--local`/`-Local`): скачает модель
     `intfloat/multilingual-e5-large` (~2.2 ГБ, один раз), работает без ключей и
     без Cloudflare.
5. **Запусти штатный установщик** из каталога плагина:
   - Linux/macOS: `bash install.sh` (или `bash install.sh --local`)
   - Windows: `powershell -ExecutionPolicy Bypass -File install.ps1` (или `... -Local`)

   Он сам найдёт venv hermes, поставит зависимости (`sqlite-vec`, `PyStemmer`,
   `ftfy`, `requests`, `model2vec`; при `--local` ещё `fastembed>=0.6`), скопирует
   плагин в `$HERMES_HOME/plugins/memohood/` и пропишет `memory.provider: memohood`
   в `config.yaml`.
6. **Прочитай вывод установщика.** Если он написал, что не смог пропатчить
   `config.yaml` (строки про ручную правку) — впиши их сам:
   ```yaml
   memory:
     provider: memohood
   ```
7. **Сообщи пользователю итог:**
   - **перезапусти hermes** (`hermes gateway restart` или ребут процесса) — провайдер
     памяти подхватится только после рестарта;
   - проверка: `hermes memohood status`;
   - ключи для семантики и извлечения фактов (Cloudflare / Cohere / Gemini) — по
     желанию, через `hermes memohood setup` (или `/setup` у новичка).

## Способ B — через git (человек ставит вручную, без чата)

- `hermes plugins install <owner/repo>`, затем `hermes gateway restart`; **или**
- `git clone <repo> && cd <repo> && bash install.sh` (можно с `--local`), затем
  перезапуск hermes.

Публичный репозиторий MemoHood — см. `README.md` / `README.en.md`.

## Безопасность

Установка **выполняет код** из архива в окружении hermes. Ставь MemoHood только из
источника, которому доверяешь (свой архив или официальный репозиторий). Не запускай
установку из случайно присланного архива без подтверждения владельца.

## Справочные факты (для сверки при установке)

| | значение |
|---|---|
| name / kind | `memohood` / `exclusive` (провайдер памяти) |
| каталог установки | `$HERMES_HOME/plugins/memohood/` |
| включение в config | `memory.provider: memohood` |
| при `--local` | `memory.memohood.embedder: {provider: local, model: intfloat/multilingual-e5-large, dims: 1024}` |
| зависимости | `sqlite-vec PyStemmer ftfy requests model2vec` (+ `fastembed>=0.6` при `--local`) |
| ключи (опц.) | `GEMINI_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `COHERE_API_KEY` |
| настройка ключей | `hermes memohood setup` |
| проверка | `hermes memohood status` |
