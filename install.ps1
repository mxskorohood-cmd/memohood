#Requires -Version 5.1
<#
.SYNOPSIS
    One-command installer for MemoHood (dialogue-memory provider plugin).

.DESCRIPTION
    Unlike general hermes plugins (per the plugin loader's API contract: no
    pip_dependencies support), memory-provider plugins DO get lazy pip-install
    from the host loader on first activation (hermes_cli/memory_setup.py) --
    this script installs the same packages up front so the very first
    `hermes memohood status` after restart doesn't have to hit the network
    mid-session, copies the plugin into place, and switches
    `memory.provider: memohood` in config.yaml for you -- no manual copy,
    no manual config.yaml edit.

    Package list == plugin.yaml's pip_dependencies (all MIT/BSD/Apache, no
    torch): sqlite-vec PyStemmer ftfy requests model2vec

.PARAMETER PythonPath
    Explicit override -- full path to the hermes venv's python.exe, if
    auto-detection below can't find it (e.g. non-standard install).

.PARAMETER Local
    ALSO install the local embedder (fastembed, ONNX, no PyTorch) and
    pre-download multilingual-e5-large (~2.2 GB), then point
    memory.memohood.embedder at it -- so the vector leg of recall works with
    no CLOUDFLARE_* keys.

.PARAMETER NoConfig
    Skip the config.yaml auto-patch step (deps + copy only; you flip
    memory.provider yourself).

.NOTES
    Idempotent: safe to re-run any time -- re-copies over an existing
    install, re-runs pip install --upgrade, and re-patching config.yaml is a
    no-op if memory.provider is already memohood.
#>

[CmdletBinding()]
param(
    [string]$PythonPath,
    [switch]$Local,
    [switch]$NoConfig
)

$ErrorActionPreference = "Stop"

# Python on Windows decides its stdout encoding from the ANSI codepage, not
# the console's actual codepage -- Cyrillic prints mangle even though
# [Console]::OutputEncoding is already UTF-8. Force it everywhere this
# script invokes python.
$env:PYTHONIOENCODING = "utf-8"

function Resolve-HermesVenvPython {
    param([string]$Override)

    if ($Override) {
        if (Test-Path $Override) { return (Resolve-Path $Override).Path }
        throw "Указанный -PythonPath не существует: $Override"
    }

    if ($env:HERMES_VENV_PYTHON -and (Test-Path $env:HERMES_VENV_PYTHON)) {
        return (Resolve-Path $env:HERMES_VENV_PYTHON).Path
    }

    # `hermes` on PATH is normally a shim/exe inside the venv's Scripts/ dir --
    # its sibling python.exe is exactly the interpreter every plugin runs under.
    $hermesCmd = Get-Command hermes -ErrorAction SilentlyContinue
    if ($hermesCmd) {
        $scriptsDir = Split-Path -Parent $hermesCmd.Source
        $candidate = Join-Path $scriptsDir "python.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    # Fall back to the conventional HERMES_HOME/hermes-agent/venv layout.
    $hermesHome = $env:HERMES_HOME
    if (-not $hermesHome) { $hermesHome = Join-Path $env:LOCALAPPDATA "hermes" }
    $candidate = Join-Path $hermesHome "hermes-agent\venv\Scripts\python.exe"
    if (Test-Path $candidate) { return $candidate }

    throw (
        "Не удалось найти python интерпретатор hermes-agent venv автоматически. " +
        "Укажите его явно: .\install.ps1 -PythonPath 'C:\path\to\hermes-agent\venv\Scripts\python.exe' " +
        "или задайте переменную окружения HERMES_VENV_PYTHON."
    )
}

$python = Resolve-HermesVenvPython -Override $PythonPath
Write-Host "MemoHood: устанавливаю зависимости в $python" -ForegroundColor Cyan

$packages = @("sqlite-vec", "PyStemmer", "ftfy", "requests", "model2vec")
& $python -m pip install --upgrade @packages
if ($LASTEXITCODE -ne 0) {
    Write-Host "MemoHood: установка зависимостей не удалась (см. вывод pip выше)." -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- Resolve HERMES_HOME the same way hermes itself would ------------------
$hermesHomeResolved = $null
try {
    $hermesHomeResolved = & $python -c "from hermes_constants import get_hermes_home; print(get_hermes_home())"
    if ($LASTEXITCODE -ne 0) { $hermesHomeResolved = $null }
} catch {
    $hermesHomeResolved = $null
}
if (-not $hermesHomeResolved) {
    if ($env:HERMES_HOME) {
        $hermesHomeResolved = $env:HERMES_HOME
    } else {
        $hermesHomeResolved = Join-Path $env:LOCALAPPDATA "hermes"
    }
    Write-Host "MemoHood: не удалось спросить hermes_constants.get_hermes_home(), использую $hermesHomeResolved" -ForegroundColor Yellow
}

$targetDir = Join-Path $hermesHomeResolved "plugins\memohood"

# --- Copy the plugin folder into place (idempotent: overwrites) ------------
Write-Host ""
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
$sourceReal = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\')
$targetReal = (Resolve-Path $targetDir).Path.TrimEnd('\')
if ($sourceReal -ieq $targetReal) {
    Write-Host "MemoHood: уже стоит в $targetDir (запущено из целевой папки) -- копирование пропущено" -ForegroundColor Cyan
} else {
    Write-Host "MemoHood: копирую плагин в $targetDir" -ForegroundColor Cyan
    Copy-Item -Path (Join-Path $PSScriptRoot "*") -Destination $targetDir -Recurse -Force
    Get-ChildItem -Path $targetDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Optional: local offline embedder (fastembed + e5-large) ---------------
if ($Local) {
    Write-Host ""
    Write-Host "MemoHood: ставлю локальный эмбеддер (fastembed - ONNX Runtime, без PyTorch)..." -ForegroundColor Cyan
    # fastembed>=0.6: e5-large обучена под mean pooling; fastembed<=0.5.1 применял CLS -
    # другой вектор-спейс. Порог фиксирует протестированное поведение для новых баз.
    & $python -m pip install --upgrade "fastembed>=0.6"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MemoHood: установка fastembed не удалась (см. вывод pip выше)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "MemoHood: скачиваю модель intfloat/multilingual-e5-large (~2.2 ГБ, один раз)..." -ForegroundColor Cyan
    & $python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='intfloat/multilingual-e5-large'); print('  локальная модель готова к работе')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MemoHood: не удалось скачать локальную модель (см. вывод выше)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- Patch config.yaml: memory.provider: memohood (+ embedder if -Local) ---
$configStatus = "skipped"
if ($NoConfig) {
    Write-Host ""
    Write-Host "MemoHood: -NoConfig -- config.yaml не трогаю. Впишите вручную:" -ForegroundColor Yellow
    Write-Host "  memory:"
    Write-Host "    provider: memohood"
    if ($Local) {
        Write-Host "    memohood:"
        Write-Host "      embedder: {provider: local, model: intfloat/multilingual-e5-large, dims: 1024}"
    }
} else {
    Write-Host ""
    Write-Host "MemoHood: включаю как провайдера памяти в config.yaml..." -ForegroundColor Cyan

    $patchScript = @'
import sys

LOCAL = len(sys.argv) > 1 and sys.argv[1] == "1"
EMBEDDER_PATCH = (
    {"provider": "local", "model": "intfloat/multilingual-e5-large", "dims": 1024}
    if LOCAL else None
)


def _apply(memory_cfg):
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
    memory_cfg["provider"] = "memohood"
    if EMBEDDER_PATCH is not None:
        memohood_cfg = memory_cfg.get("memohood")
        if not isinstance(memohood_cfg, dict):
            memohood_cfg = {}
        memohood_cfg["embedder"] = EMBEDDER_PATCH
        memory_cfg["memohood"] = memohood_cfg
    return memory_cfg


def _print_manual():
    print("  Впишите вручную в config.yaml:")
    print("    memory:")
    print("      provider: memohood")
    if EMBEDDER_PATCH is not None:
        print("      memohood:")
        print("        embedder: {provider: local, model: intfloat/multilingual-e5-large, dims: 1024}")


def _tier1():
    # Real hermes_cli.config API -- the same load/mutate/save call pattern
    # other hermes onboarding tooling uses for this exact operation.
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["memory"] = _apply(cfg.get("memory"))
    save_config(cfg)


def _tier2():
    # Fallback: patch config.yaml directly (no hermes_cli.config dependency),
    # mirroring memohood's own config.save_memohood_config_at(). Always backs
    # up the original file first.
    import yaml
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))
    backup_path = config_path.with_name(config_path.name + ".bak")
    backup_path.write_bytes(config_path.read_bytes())
    with open(config_path, encoding="utf-8-sig") as f:
        existing = yaml.safe_load(f) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing["memory"] = _apply(existing.get("memory"))
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
    print("MemoHood: config.yaml пропатчен напрямую (бэкап: %s)" % backup_path)


_status = "ok"
try:
    _tier1()
    extra = " + embedder: local" if LOCAL else ""
    print("MemoHood: memory.provider -> memohood (через hermes_cli.config)" + extra)
except SystemExit:
    # save_config()/set_config_value() can call sys.exit(1) for a
    # managed-scope lock -- never let that kill the installer.
    print("MemoHood: hermes_cli.config отказался писать конфиг (managed scope?) -- пробую напрямую...")
    try:
        _tier2()
    except Exception as exc2:
        print("MemoHood: не удалось автопатчить config.yaml (%s)." % exc2)
        _print_manual()
        _status = "manual"
except Exception as exc:
    print("MemoHood: hermes_cli.config недоступен (%s) -- пробую напрямую..." % exc)
    try:
        _tier2()
    except Exception as exc2:
        print("MemoHood: не удалось автопатчить config.yaml (%s)." % exc2)
        _print_manual()
        _status = "manual"

print("MEMOHOOD_CONFIG_STATUS=" + _status)
'@

    $tmpPatchFile = Join-Path $env:TEMP "memohood_patch_config_$PID.py"
    $patchScript | Out-File -FilePath $tmpPatchFile -Encoding utf8 -Force

    $localFlag = "0"
    if ($Local) { $localFlag = "1" }

    $patchOutput = & $python $tmpPatchFile $localFlag
    Remove-Item -Path $tmpPatchFile -Force -ErrorAction SilentlyContinue

    foreach ($line in $patchOutput) {
        if ($line -notmatch '^MEMOHOOD_CONFIG_STATUS=') {
            Write-Host $line
        }
    }
    if ($patchOutput -match "MEMOHOOD_CONFIG_STATUS=manual") {
        $configStatus = "manual"
    } else {
        $configStatus = "ok"
    }
}

Write-Host ""
switch ($configStatus) {
    "ok" {
        Write-Host "MemoHood включён как провайдер памяти. Работает сразу (keyword-режим, без единого ключа)." -ForegroundColor Green
    }
    "manual" {
        Write-Host "MemoHood скопирован, зависимости стоят -- но config.yaml пришлось патчить вручную (см. строки выше)." -ForegroundColor Yellow
    }
    "skipped" {
        Write-Host "MemoHood скопирован, зависимости стоят; config.yaml не тронут (-NoConfig) -- включите провайдера вручную (см. строки выше)." -ForegroundColor Yellow
    }
}
Write-Host "Ключи для семантического поиска и извлечения спорных фактов: hermes memohood setup"
if ($Local) {
    Write-Host "Офлайн-вектор уже настроен (-Local, без Cloudflare)."
} else {
    Write-Host "Офлайн-вектор без Cloudflare: переустановите с флагом -Local"
}
Write-Host "Перезапустите hermes, затем проверьте: hermes memohood status"
