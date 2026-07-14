"""Wire the systemd user ``hermes-gateway.service`` unit to read
``HERMES_HOME/.env`` via a drop-in override (B12).

## The bug this fixes

When hermes runs as a systemd **user** unit (``hermes-gateway.service``),
``hermes gateway restart`` restarts exactly that unit -- and systemd builds
a unit's process environment BEFORE the process starts, from whatever the
unit file itself declares (``Environment=``/``EnvironmentFile=``). The
bundled unit declares neither, so it never reads ``~/.hermes/.env`` on its
own: keys ``setup_wizard.py`` just wrote via ``upsert_env_var`` sit in the
file, ``hermes gateway restart`` runs, and the gateway process still starts
with the OLD environment -- the new keys never reach it.

``.env``'s own format (flat ``KEY=value`` lines, one per line -- see
``setup_wizard.upsert_env_var``) is already exactly what systemd's
``EnvironmentFile=`` directive expects; nothing about *how* ``.env`` is
written needs to change, only *whether the unit is told to read it*.

## The fix: an auto drop-in, never touching the hermes-owned unit file

Editing ``hermes-gateway.service`` itself is off the table (plugins never
patch hermes -- and a future hermes upgrade could silently overwrite that
file anyway). systemd's own override mechanism -- a "drop-in" ``.conf``
snippet in ``<unit>.d/`` -- exists for exactly this: anything dropped into
``hermes-gateway.service.d/*.conf`` is merged into the unit at
``daemon-reload`` time without modifying the original unit file at all.
This module writes ONE such drop-in
(``hermes-gateway.service.d/10-memohood-env.conf``) containing a single
line::

    [Service]
    EnvironmentFile=-<hermes_home>/.env

The leading ``-`` is systemd's own "this file is optional" marker -- a
fresh install with no ``.env`` yet must not make the unit fail to start.

## Where this gets called from

* ``install.sh`` calls this module's ``__main__`` block once, right after
  the plugin is copied into place -- so a systemd-hosted gateway is wired
  up on first install, before any key has even been entered.
* ``setup_wizard.py`` calls :func:`wire_gateway_env` again at the end of
  ``hermes memohood setup`` (idempotent -- re-running just rewrites the
  same drop-in) so the wizard's own "what's next" hint can honestly tell a
  systemd user to just restart the gateway, instead of the CLI-session
  phrasing that doesn't apply to them.

## Safety / testability

* Everything that shells out is funneled through an injectable *runner*
  (defaults to ``subprocess.run``), and the drop-in's parent directory is
  an injectable *config_home* (defaults to ``~/.config``) -- exactly so
  ``tests/test_systemd_env.py`` can exercise every branch with a fake
  runner and a tmp dir, never touching a real systemd instance.
* :func:`detect_systemd_gateway` is the single guard both entry points
  rely on: no ``systemctl`` on PATH (any non-Linux dev machine, most CI),
  or the unit simply isn't installed/loaded -- either way this whole
  module becomes a documented no-op, never an error. ``install.ps1``
  (Windows) never even calls this module -- there is no systemd there at
  all; ``install.sh`` users not running hermes under systemd get the same
  safe no-op via this detection instead.
* Pure standard library only (``shutil``/``subprocess``/``pathlib``/
  ``typing``) -- no import of this plugin's own (heavier) modules, so
  ``install.sh`` can invoke this file directly as a standalone script.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Tuple

# The unit hermes itself installs/owns; a drop-in override directory hangs
# off this exact name (systemd convention: "<unit>.d/*.conf").
GATEWAY_UNIT = "hermes-gateway.service"

# Numeric prefix is systemd drop-in convention (lexical load order, not that
# it matters here with only one file) -- "memohood" makes the owner obvious
# to anyone reading `ls .../hermes-gateway.service.d/` by hand.
DROP_IN_FILENAME = "10-memohood-env.conf"

RunnerFn = Callable[..., "subprocess.CompletedProcess[str]"]


def detect_systemd_gateway(*, runner: RunnerFn = subprocess.run) -> bool:
    """True if the hermes gateway is running as a systemd **user** unit.

    Checked with ``systemctl --user cat hermes-gateway.service`` -- ``cat``
    prints the resolved unit (following drop-ins) and exits non-zero if the
    unit is unknown to systemd, which confirms BOTH "systemd is available"
    and "this specific unit is known to it" in one cheap, read-only call.

    Never raises: no ``systemctl`` binary on PATH (checked first, via
    ``shutil.which`` -- the common case on Windows/macOS dev machines,
    where even attempting the subprocess call would raise
    ``FileNotFoundError``), a *runner* that raises for any other reason, or
    a non-zero exit all read as plain ``False`` -- "not a systemd gateway"
    is the safe default everywhere :func:`wire_gateway_env` guards on this.
    """
    if shutil.which("systemctl") is None:
        return False
    try:
        result = runner(
            ["systemctl", "--user", "cat", GATEWAY_UNIT],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - detection must never crash the caller
        return False
    return getattr(result, "returncode", 1) == 0


def wire_gateway_env(
    hermes_home: "str | Path",
    *,
    config_home: "str | Path | None" = None,
    runner: RunnerFn = subprocess.run,
) -> Tuple[bool, str]:
    """Idempotently point the systemd user gateway unit at ``.env``.

    Writes ``{config_home or ~/.config}/systemd/user/hermes-gateway.service.d/
    10-memohood-env.conf`` with a single
    ``EnvironmentFile=-<hermes_home>/.env`` line (module docstring has the
    full rationale), then runs ``systemctl --user daemon-reload`` so the
    change takes effect on the NEXT ``hermes gateway restart`` -- this
    function does not itself restart the gateway (an in-flight
    ``hermes memohood setup`` session may still be running under the very
    process a restart would kill).

    *hermes_home* is written into the drop-in exactly as given -- a plain
    string substitution (only a trailing slash is stripped so the written
    path never doubles up before ``/.env``), no attempt to compress it to
    systemd's ``%h`` specifier. systemd resolves ``EnvironmentFile=``
    values on the target Linux host, and this project's own
    ``hermes_constants.get_hermes_home()`` already returns the absolute
    path systemd needs -- writing it verbatim keeps the drop-in correct
    even for a ``HERMES_HOME`` override that lives outside the unit's own
    home directory.

    Never touches ``hermes-gateway.service`` itself -- only ever writes
    inside its ``.d/`` override directory, so a future hermes upgrade that
    replaces the unit file can't collide with this.

    Guard: if :func:`detect_systemd_gateway` is False (no systemd, no
    ``systemctl``, or the unit isn't installed), this is a pure no-op --
    nothing is written, nothing is reloaded -- and returns
    ``(False, ...)``. Idempotent otherwise: the drop-in is (re)written with
    deterministic content on every call, so re-running (e.g. a second
    ``hermes memohood setup`` pass, or a re-run of ``install.sh``) never
    duplicates or corrupts it.

    Returns ``(wired, message)``. ``wired`` is True as soon as the drop-in
    file itself is written -- a failed ``daemon-reload`` is reported in the
    message (systemd will still pick the drop-in up on its next reload or
    restart of anything) but doesn't flip ``wired`` back to False, since
    the filesystem change already happened. A failure to even WRITE the
    drop-in (e.g. an unwritable ``config_home``) degrades to
    ``(False, ...)`` instead of raising -- this must never crash
    ``install.sh``'s ``|| true``-guarded call or the wizard's own
    finalization step.
    """
    if not detect_systemd_gateway(runner=runner):
        return False, "hermes gateway не запущен как systemd user unit -- drop-in не нужен, пропускаю"

    home_str = str(hermes_home).rstrip("/\\")
    conf_home = Path(config_home) if config_home is not None else Path.home() / ".config"
    drop_in_dir = conf_home / "systemd" / "user" / f"{GATEWAY_UNIT}.d"
    drop_in_path = drop_in_dir / DROP_IN_FILENAME

    try:
        drop_in_dir.mkdir(parents=True, exist_ok=True)
        drop_in_path.write_text(f"[Service]\nEnvironmentFile=-{home_str}/.env\n", encoding="utf-8")
    except OSError as exc:
        return False, f"не удалось записать systemd drop-in ({drop_in_path}): {exc}"

    try:
        reload_result = runner(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        reload_ok = getattr(reload_result, "returncode", 1) == 0
    except Exception:  # noqa: BLE001 - a failed reload must degrade, not crash install/setup
        reload_ok = False

    if reload_ok:
        return True, f"gateway настроен читать {home_str}/.env (systemd drop-in: {drop_in_path})"
    return True, (
        f"drop-in записан ({drop_in_path}), но systemctl --user daemon-reload не отработал -- "
        "выполните его вручную, затем hermes gateway restart"
    )


def _main(argv: "list[str] | None" = None) -> None:
    """``python systemd_env.py <hermes_home>`` -- install.sh's call site:
    ``"$PYTHON" "$TARGET_DIR/systemd_env.py" "$HERMES_HOME_RESOLVED"``.

    Split out from the bare ``if __name__ == "__main__":`` block purely so
    ``tests/test_systemd_env.py`` can call this directly (argv injectable,
    defaulting to ``sys.argv[1:]``) instead of needing a real subprocess or
    ``runpy`` to exercise it. Prints only the human message -- install.sh's
    call site is ``|| true``-guarded and folds this straight into its
    normal install log.
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    home = args[0] if args else str(Path.home() / ".hermes")
    _wired, msg = wire_gateway_env(home)
    print(msg)


if __name__ == "__main__":
    _main()
