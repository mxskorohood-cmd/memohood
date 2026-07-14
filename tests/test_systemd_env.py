"""Tests for ``systemd_env.py`` (B12: wire the systemd user gateway unit to
read ``HERMES_HOME/.env`` via a drop-in override).

Same isolation pattern as the rest of the suite (see ``conftest.py``): the
``memohood`` fixture gives a fresh package copy so this test's monkeypatches
of ``shutil.which``/the injected *runner* can never leak into another test.
No test here ever calls a real ``systemctl`` or writes outside a tmp
``config_home`` -- the fake *runner* below stands in for
``subprocess.run`` entirely, and ``shutil.which`` is monkeypatched on the
module's own ``shutil`` import so "systemctl is on PATH" is fully
controlled per test.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _systemd_env(memohood):
    """Import the systemd_env submodule of THIS test's fresh package copy."""
    return importlib.import_module(f"{memohood.__name__}.systemd_env")


def _fail_if_called(*args, **kwargs):  # pragma: no cover - failure path only
    raise AssertionError("runner must not be called when the guard should short-circuit")


def _make_runner(*, cat_ok: bool = True, reload_ok: bool = True, calls: "list | None" = None):
    """Fake ``subprocess.run`` stand-in: succeeds/fails ``systemctl --user
    cat`` and ``systemctl --user daemon-reload`` independently, and
    (optionally) records every invocation for assertions. Never touches a
    real process."""

    def runner(cmd, **kwargs):
        if calls is not None:
            calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "cat"]:
            return SimpleNamespace(returncode=0 if cat_ok else 1, stdout="", stderr="")
        if cmd[:3] == ["systemctl", "--user", "daemon-reload"]:
            return SimpleNamespace(returncode=0 if reload_ok else 1, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    return runner


_DROP_IN_REL = ("systemd", "user", "hermes-gateway.service.d", "10-memohood-env.conf")


def _drop_in_path(config_home):
    p = config_home
    for part in _DROP_IN_REL:
        p = p / part
    return p


# ---------------------------------------------------------------------------
# detect_systemd_gateway
# ---------------------------------------------------------------------------


class TestDetectSystemdGateway:
    def test_true_when_systemctl_present_and_unit_known(self, memohood, monkeypatch):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        runner = _make_runner(cat_ok=True)
        assert sw.detect_systemd_gateway(runner=runner) is True

    def test_false_when_unit_not_known_to_systemd(self, memohood, monkeypatch):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        runner = _make_runner(cat_ok=False)
        assert sw.detect_systemd_gateway(runner=runner) is False

    def test_false_when_systemctl_missing_from_path(self, memohood, monkeypatch):
        """No systemctl on PATH (Windows/macOS dev machines, most CI) must
        short-circuit BEFORE the runner is ever invoked -- calling the real
        subprocess.run with a nonexistent binary would raise."""
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: None)
        assert sw.detect_systemd_gateway(runner=_fail_if_called) is False

    def test_false_when_runner_raises(self, memohood, monkeypatch):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")

        def boom(cmd, **kwargs):
            raise OSError("dbus not available")

        assert sw.detect_systemd_gateway(runner=boom) is False

    def test_calls_systemctl_cat_with_exact_unit_name(self, memohood, monkeypatch):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        calls: list = []
        sw.detect_systemd_gateway(runner=_make_runner(calls=calls))
        assert calls == [["systemctl", "--user", "cat", "hermes-gateway.service"]]


# ---------------------------------------------------------------------------
# wire_gateway_env -- systemd present: writes drop-in + reloads
# ---------------------------------------------------------------------------


class TestWireGatewayEnvSystemd:
    def test_writes_drop_in_with_environmentfile_and_reloads(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        calls: list = []
        runner = _make_runner(calls=calls)
        config_home = tmp_path / "config"

        wired, msg = sw.wire_gateway_env("/home/user/.hermes", config_home=config_home, runner=runner)

        assert wired is True
        drop_in = _drop_in_path(config_home)
        assert drop_in.exists()
        content = drop_in.read_text(encoding="utf-8")
        assert "[Service]" in content
        assert "EnvironmentFile=-/home/user/.hermes/.env" in content
        assert ["systemctl", "--user", "daemon-reload"] in [c[:3] for c in calls]
        assert str(drop_in) in msg

    def test_leading_dash_makes_missing_env_optional(self, memohood, monkeypatch, tmp_path):
        """The '-' right after '=' is systemd's own "this file is optional"
        marker -- a fresh install with no .env yet must not fail unit start."""
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        sw.wire_gateway_env("/home/user/.hermes", config_home=config_home, runner=_make_runner())

        content = _drop_in_path(config_home).read_text(encoding="utf-8")
        line = next(l for l in content.splitlines() if l.startswith("EnvironmentFile="))
        assert line == "EnvironmentFile=-/home/user/.hermes/.env"

    def test_trailing_slash_in_hermes_home_is_normalized(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        sw.wire_gateway_env("/home/user/.hermes/", config_home=config_home, runner=_make_runner())

        content = _drop_in_path(config_home).read_text(encoding="utf-8")
        assert "EnvironmentFile=-/home/user/.hermes/.env" in content
        assert "//.env" not in content

    def test_reload_failure_still_reports_wired_true_with_warning(self, memohood, monkeypatch, tmp_path):
        """A failed daemon-reload doesn't undo the filesystem change -- the
        drop-in is already there and systemd will pick it up on its next
        reload/restart of anything -- so `wired` stays True, but the message
        must say so the caller can surface a manual-reload hint."""
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        wired, msg = sw.wire_gateway_env(
            "/home/user/.hermes", config_home=config_home, runner=_make_runner(reload_ok=False)
        )

        assert wired is True
        assert "daemon-reload" in msg
        assert _drop_in_path(config_home).exists()

    def test_default_config_home_is_dot_config_under_home(self, memohood, monkeypatch, tmp_path):
        """When config_home is omitted, the drop-in goes under ~/.config --
        verified by faking Path.home() rather than touching the real one."""
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        fake_home = tmp_path / "fakehome"
        monkeypatch.setattr(sw.Path, "home", lambda: fake_home)

        wired, _msg = sw.wire_gateway_env("/home/user/.hermes", runner=_make_runner())

        assert wired is True
        assert _drop_in_path(fake_home / ".config").exists()


# ---------------------------------------------------------------------------
# wire_gateway_env -- idempotency (B12 explicit requirement)
# ---------------------------------------------------------------------------


class TestWireGatewayEnvIdempotent:
    def test_second_call_does_not_duplicate_or_fail(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        wired1, _ = sw.wire_gateway_env("/home/user/.hermes", config_home=config_home, runner=_make_runner())
        wired2, _ = sw.wire_gateway_env("/home/user/.hermes", config_home=config_home, runner=_make_runner())

        assert wired1 is True
        assert wired2 is True
        content = _drop_in_path(config_home).read_text(encoding="utf-8")
        assert content.count("EnvironmentFile=") == 1
        assert content.count("[Service]") == 1

    def test_rerun_with_different_home_replaces_not_appends(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        sw.wire_gateway_env("/home/user/.hermes-old", config_home=config_home, runner=_make_runner())
        sw.wire_gateway_env("/home/user/.hermes-new", config_home=config_home, runner=_make_runner())

        content = _drop_in_path(config_home).read_text(encoding="utf-8")
        assert content.count("EnvironmentFile=") == 1
        assert "/.hermes-new/.env" in content
        assert "/.hermes-old/.env" not in content


# ---------------------------------------------------------------------------
# wire_gateway_env -- not-systemd guard: pure no-op
# ---------------------------------------------------------------------------


class TestWireGatewayEnvNotSystemd:
    def test_noop_when_systemctl_missing(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: None)
        config_home = tmp_path / "config"

        wired, msg = sw.wire_gateway_env(
            "/home/user/.hermes", config_home=config_home, runner=_fail_if_called
        )

        assert wired is False
        assert not config_home.exists()
        assert isinstance(msg, str) and msg  # human message present, never empty

    def test_noop_when_unit_not_installed(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"

        wired, _msg = sw.wire_gateway_env(
            "/home/user/.hermes", config_home=config_home, runner=_make_runner(cat_ok=False)
        )

        assert wired is False
        assert not config_home.exists()


# ---------------------------------------------------------------------------
# wire_gateway_env -- write failure degrades instead of raising
# ---------------------------------------------------------------------------


class TestWireGatewayEnvWriteFailure:
    def test_unwritable_drop_in_dir_degrades_to_false(self, memohood, monkeypatch, tmp_path):
        sw = _systemd_env(memohood)
        monkeypatch.setattr(sw.shutil, "which", lambda name: "/usr/bin/systemctl")
        config_home = tmp_path / "config"
        # A normal, working runner: the guard's own `systemctl --user cat`
        # check must SUCCEED first (that's not what's under test here) --
        # only the write_text() call below is made to fail. daemon-reload
        # must never be reached once the write raises, which `calls`
        # confirms below.
        calls: list = []
        runner = _make_runner(calls=calls)

        def boom_write_text(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(sw.Path, "write_text", boom_write_text)

        wired, msg = sw.wire_gateway_env("/home/user/.hermes", config_home=config_home, runner=runner)

        assert wired is False
        assert "не удалось записать" in msg
        assert calls == [["systemctl", "--user", "cat", "hermes-gateway.service"]]  # no daemon-reload attempted


# ---------------------------------------------------------------------------
# __main__ entry point (install.sh calls: python systemd_env.py <hermes_home>)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_uses_first_argv_as_hermes_home(self, memohood, monkeypatch, capsys):
        sw = _systemd_env(memohood)
        captured = {}

        def fake_wire(home, **kwargs):
            captured["home"] = home
            return (True, "ok from fake")

        monkeypatch.setattr(sw, "wire_gateway_env", fake_wire)
        sw._main(["/home/user/.hermes"])

        assert captured["home"] == "/home/user/.hermes"
        assert "ok from fake" in capsys.readouterr().out

    def test_main_without_argv_falls_back_to_home_dot_hermes(self, memohood, monkeypatch, capsys):
        sw = _systemd_env(memohood)
        captured = {}

        def fake_wire(home, **kwargs):
            captured["home"] = home
            return (False, "not systemd")

        monkeypatch.setattr(sw, "wire_gateway_env", fake_wire)
        fake_home = sw.Path("/fake/home")
        monkeypatch.setattr(sw.Path, "home", lambda: fake_home)

        sw._main([])

        assert captured["home"] == str(fake_home / ".hermes")
        assert "not systemd" in capsys.readouterr().out
