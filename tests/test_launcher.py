"""Tests for gettowork.launcher: how players start the game (Steam, double-click).

No real windows here (tests/test_gui_app.py and tests/test_e2e.py open
those): ``run_gui`` is replaced by a recorder, and the Windows console
checks run against a fake ``ctypes.windll`` so they work on every OS.
"""

from __future__ import annotations

import ctypes
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import gettowork
from gettowork import launcher
from gettowork.gui import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    return home


class FakeRunGui:
    """Stands in for gui.app.run_gui: records each call and what GETTOWORK_HOME was during it."""

    def __init__(self, code: int = 0, error: BaseException | None = None) -> None:
        self.code = code
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []
        self.homes: list[str | None] = []

    def __call__(self, argv=None, **kwargs):
        self.calls.append((argv, kwargs))
        home = os.environ.get("GETTOWORK_HOME")
        self.homes.append(home)
        if home:
            Path(home).mkdir(parents=True, exist_ok=True)
            Path(home, "proof.txt").write_text("the self-test was here", encoding="utf-8")
        if self.error is not None:
            raise self.error
        return self.code


@pytest.fixture
def fake_gui(monkeypatch):
    fake = FakeRunGui()
    monkeypatch.setattr(app, "run_gui", fake)
    return fake


# ---------------------------------------------------------------------------
# gui_main: the windowed game's entry point
# ---------------------------------------------------------------------------


def test_gui_main_opens_the_window_with_the_given_options(fake_gui):
    fake_gui.code = 7
    assert launcher.gui_main(["--mock", "--no-jev"]) == 7
    assert fake_gui.calls == [(["--mock", "--no-jev"], {})]


def test_gui_main_with_no_options_plays_normally(fake_gui):
    assert launcher.gui_main([]) == 0
    assert fake_gui.calls == [([], {})]


def test_gui_main_defaults_to_the_process_arguments(fake_gui, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["GetToWork", "--target", "3"])
    assert launcher.gui_main() == 0
    assert fake_gui.calls == [(["--target", "3"], {})]


def test_gui_main_does_not_change_the_callers_list(fake_gui):
    argv = ["--gui-selftest", "--mock"]
    launcher.gui_main(argv)
    assert argv == ["--gui-selftest", "--mock"]


def test_the_selftest_flag_turns_on_the_selftest(fake_gui):
    fake_gui.code = 2
    assert launcher.gui_main(["--gui-selftest"]) == 2
    assert fake_gui.calls == [([], {"selftest": True})]


def test_the_selftest_flag_can_sit_anywhere(fake_gui):
    launcher.gui_main(["--mock", launcher.SELFTEST_FLAG, "--no-jev"])
    assert fake_gui.calls == [(["--mock", "--no-jev"], {"selftest": True})]


def test_the_selftest_keeps_a_chosen_home_folder(fake_gui, isolated_home):
    launcher.gui_main(["--gui-selftest"])
    assert fake_gui.homes == [str(isolated_home)]
    assert (isolated_home / "proof.txt").exists()  # not a throwaway: the caller chose it


def test_the_selftest_never_touches_the_players_own_settings(fake_gui, monkeypatch):
    monkeypatch.delenv("GETTOWORK_HOME")
    launcher.gui_main(["--gui-selftest"])
    [home] = fake_gui.homes
    assert home and Path(home).name.startswith("gettowork-selftest-")
    assert not Path(home).exists()  # tidied away afterwards...
    assert "GETTOWORK_HOME" not in os.environ  # ...and the environment is as it was


def test_the_throwaway_home_is_tidied_even_if_the_window_crashes(monkeypatch):
    monkeypatch.delenv("GETTOWORK_HOME")
    fake = FakeRunGui(error=RuntimeError("kaboom"))
    monkeypatch.setattr(app, "run_gui", fake)
    with pytest.raises(RuntimeError):
        launcher.gui_main(["--gui-selftest"])
    assert not Path(fake.homes[0]).exists()
    assert "GETTOWORK_HOME" not in os.environ


def test_a_normal_game_uses_the_players_settings_folder(fake_gui, monkeypatch):
    monkeypatch.delenv("GETTOWORK_HOME")
    launcher.gui_main(["--mock"])
    assert fake_gui.homes == [None]  # the default config folder, as always


def test_python_dash_m_gettowork_launcher_opens_the_window(fake_gui, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--mock"])
    monkeypatch.delitem(sys.modules, "gettowork.launcher")  # run it fresh, as the __main__ module
    fake_gui.code = 5
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("gettowork.launcher", run_name="__main__")
    assert exc.value.code == 5
    assert fake_gui.calls == [(["--mock"], {})]


def test_importing_the_launcher_or_the_cli_never_loads_tk_or_the_window():
    code = ("import sys, gettowork.launcher, gettowork.cli; "
            "loaded = [m for m in ('tkinter', '_tkinter', 'gettowork.gui.app') if m in sys.modules]; "
            "print(loaded); sys.exit(1 if loaded else 0)")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# ---------------------------------------------------------------------------
# The console program, double-clicked on Windows
# ---------------------------------------------------------------------------


class FakeTTY:
    def __init__(self, tty: bool = True) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class FakeKernel32:
    def __init__(self, count: int = 1, error: BaseException | None = None) -> None:
        self.count = count
        self.error = error
        self.calls: list[int] = []

    def GetConsoleProcessList(self, buffer, size):  # noqa: N802 - the Windows API's own name
        self.calls.append(size)
        assert len(buffer) >= size  # never ask Windows to write past the end of the list
        if self.error is not None:
            raise self.error
        return self.count


class FakeWindll:
    def __init__(self, kernel32: FakeKernel32) -> None:
        self.kernel32 = kernel32


@pytest.fixture
def windows_exe(monkeypatch):
    """Pretend to be the built gettowork.exe in a Windows console; returns the fake kernel32."""
    kernel32 = FakeKernel32()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "stdin", FakeTTY())
    monkeypatch.setattr(ctypes, "windll", FakeWindll(kernel32), raising=False)
    return kernel32


def test_a_double_clicked_console_exe_is_recognised(windows_exe):
    assert launcher.console_closes_on_exit() is True
    assert windows_exe.calls == [2]


def test_a_console_shared_with_a_shell_stays_open(windows_exe):
    windows_exe.count = 2  # cmd.exe / PowerShell is attached too
    assert launcher.console_closes_on_exit() is False


def test_no_console_at_all_is_not_a_closing_window(windows_exe):
    windows_exe.count = 0  # GetConsoleProcessList fails (returns 0) without a console
    assert launcher.console_closes_on_exit() is False


def test_running_from_source_never_waits(windows_exe, monkeypatch):
    monkeypatch.delattr(sys, "frozen")
    assert launcher.console_closes_on_exit() is False
    assert windows_exe.calls == []


@pytest.mark.parametrize("stdin", [FakeTTY(False), None])
def test_piped_or_missing_input_never_waits(windows_exe, monkeypatch, stdin):
    monkeypatch.setattr(sys, "stdin", stdin)
    assert launcher.console_closes_on_exit() is False
    assert windows_exe.calls == []


def test_a_broken_windows_api_never_waits(windows_exe):
    windows_exe.error = OSError("no such function")
    assert launcher.console_closes_on_exit() is False


def test_missing_windll_never_waits(windows_exe, monkeypatch):
    monkeypatch.delattr(ctypes, "windll")
    assert launcher.console_closes_on_exit() is False


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_only_windows_closes_console_windows_like_that(windows_exe, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    assert launcher.console_closes_on_exit() is False
    assert windows_exe.calls == []


def test_the_test_run_itself_is_never_a_double_clicked_console():
    assert launcher.console_closes_on_exit() is False


def test_wait_before_closing_asks_for_enter():
    prompts = []
    launcher.wait_before_closing(lambda prompt: prompts.append(prompt) or "")
    assert len(prompts) == 1 and launcher.CLOSE_PROMPT in prompts[0]
    assert launcher.CLOSE_PROMPT == "Press Enter to close this window"


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt(), OSError("console gone"),
                                   UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")])
def test_wait_before_closing_never_raises(error):
    def closed(prompt):
        raise error

    launcher.wait_before_closing(closed)  # no exception: the program just ends


def test_wait_before_closing_uses_input_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr("builtins.input", lambda prompt="": seen.append(prompt) or "")
    launcher.wait_before_closing()
    assert seen and launcher.CLOSE_PROMPT in seen[0]


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_has_both_entry_points():
    project = _pyproject()["project"]
    assert project["scripts"]["gettowork"] == "gettowork.cli:main"  # the terminal version stays
    assert project["gui-scripts"]["gettowork-gui"] == "gettowork.launcher:gui_main"
    for target in (project["scripts"]["gettowork"], project["gui-scripts"]["gettowork-gui"]):
        module, _, name = target.partition(":")
        __import__(module)
        assert callable(getattr(sys.modules[module], name)), target


def test_the_version_lives_in_one_place():
    data = _pyproject()
    assert "version" not in data["project"] and "version" in data["project"]["dynamic"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "gettowork.__version__"}
    assert gettowork.__version__ == "0.2.0"


def test_the_window_icon_ships_in_the_package():
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["gettowork"]
    assert "assets/*" in patterns
    assert app._icon_path().parent == ROOT / "src" / "gettowork" / "assets"


# ---------------------------------------------------------------------------
# The built game gives the programs it starts the system's own libraries (Linux)
# ---------------------------------------------------------------------------


def test_the_players_library_path_is_put_back():
    env = {"LD_LIBRARY_PATH": "/game/_internal:/usr/lib/steam", "LD_LIBRARY_PATH_ORIG": "/usr/lib/steam", "HOME": "/h"}
    assert launcher.restore_system_library_path(env, frozen=True, platform="linux") is True
    assert env == {"LD_LIBRARY_PATH": "/usr/lib/steam", "HOME": "/h"}
    assert launcher.restore_system_library_path(env, frozen=True, platform="linux") is False  # already done


def test_a_library_path_the_bootloader_made_up_is_removed(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", "/game/_internal", raising=False)
    env = {"LD_LIBRARY_PATH": "/game/_internal"}
    assert launcher.restore_system_library_path(env, frozen=True, platform="linux") is True
    assert env == {}


def test_a_library_path_of_the_players_own_is_kept(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", "/game/_internal", raising=False)
    env = {"LD_LIBRARY_PATH": "/opt/my-libs"}
    assert launcher.restore_system_library_path(env, frozen=True, platform="linux") is False
    assert env == {"LD_LIBRARY_PATH": "/opt/my-libs"}


@pytest.mark.parametrize("frozen, platform", [(False, "linux"), (True, "win32"), (True, "darwin"), (True, "cygwin")])
def test_the_library_path_is_only_touched_in_a_built_game_on_linux(frozen, platform):
    env = {"LD_LIBRARY_PATH": "/game/_internal", "LD_LIBRARY_PATH_ORIG": ""}
    assert launcher.restore_system_library_path(env, frozen=frozen, platform=platform) is False
    assert env == {"LD_LIBRARY_PATH": "/game/_internal", "LD_LIBRARY_PATH_ORIG": ""}


def test_running_from_source_never_touches_the_real_environment(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert launcher.restore_system_library_path() is False
    assert os.environ["LD_LIBRARY_PATH"] == "/somewhere"


def test_restoring_the_library_path_never_raises():
    class Broken(dict):
        def get(self, *args):
            raise RuntimeError("boom")

    assert launcher.restore_system_library_path(Broken(LD_LIBRARY_PATH="x"), frozen=True, platform="linux") is False
