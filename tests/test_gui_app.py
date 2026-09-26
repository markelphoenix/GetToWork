"""Tests for gettowork.gui.app: the game's own window.

The first half needs no Tk at all (helpers, the self-test player, what
happens when the window can't open). The second half opens real Tk windows
and drives small scripted "games" through them; it is skipped where Tk or a
display is missing (on Linux CI it runs under ``xvfb-run``).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from gettowork.config import Settings
from gettowork.gui import app
from gettowork.gui.app import (
    EXIT_HINT,
    FONT_CANDIDATES,
    PALETTE_16,
    SECRET_HINT,
    SelftestPlayer,
    blend,
    glyph_fallbacks,
    pick_font_family,
    plain_prompt,
    resolve_color,
    short_label,
    style_colors,
    wants_fullscreen,
    write_crash_report,
)
from gettowork.gui.terminal import Style
from gettowork.ui import UserQuit


@pytest.fixture(autouse=True)
def collect_tk_garbage_here():
    """Free leftover Tk objects on the main thread (Tcl aborts if another thread frees them)."""
    gc.collect()
    yield
    gc.collect()


REAL_SHOW_ERROR_DIALOG = app.show_error_dialog


@pytest.fixture(autouse=True)
def no_real_dialogs(monkeypatch):
    """A failed start shows a native message box: the tests record it instead of popping one up."""
    shown: list = []
    monkeypatch.setattr(app, "show_error_dialog", lambda title, message, **kw: shown.append((title, message)) or True)
    return shown


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    for var in ("GETTOWORK_MODELS_DIR", "TYPESAFE_API_KEY", app.SELFTEST_OUT_ENV, app.SELFTEST_TIMEOUT_ENV,
                app.FULLSCREEN_ENV, "SteamDeck", "SteamGamepadUI", "GAMESCOPE_WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    return home


# ---------------------------------------------------------------------------
# Helpers (no Tk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt, expected", [
    ("[bold]How do you plan to get to work?[/bold] > ", "How do you plan to get to work?"),
    ("[bold]Your pick[/bold] [dim](1)[/dim] > ", "Your pick (1)"),
    ("[bold]Save it? \\[y/N][/bold] > ", "Save it? [y/N]"),
    ("Paste your key: ", "Paste your key"),
    ("[dim]Press Enter to continue[/dim] > ", "Press Enter to continue"),
    ("Go -> ", "Go ->"),
    ("[broken markup", "[broken markup"),
    ("", ""),
])
def test_plain_prompt(prompt, expected):
    assert plain_prompt(prompt) == expected


def test_short_labels_fit_on_a_button():
    assert short_label("Yes") == "Yes"
    assert short_label("Try again (handy if the internet hiccupped or you just started Ollama)") == "Try again"
    assert short_label("Stop for now - anything already downloaded is kept for next time") == "Stop for now"
    assert short_label("[bold]Bold[/bold] label") == "Bold label"
    long = short_label("x" * 100)
    assert len(long) == 42 and long.endswith("…")
    assert short_label("") == ""


def test_button_labels_never_show_two_buttons_alike():
    """Two options whose short labels read the same ("Yes" / "Yes") get their fuller labels instead."""
    options = [("yes", "Yes, play"), ("no", "Pick a different model"),
               ("jev", "Yes - and this time let me turn on Jev, the optional AI referee")]
    assert short_label(options[2][1]) == "Yes"  # (what the old welcome-back label shrank to)
    labels = app.button_labels([("a", "Yes - start the game now please, it is time"),
                                ("b", "Yes - but switch Jev on first, the optional referee")])
    assert labels == ["Yes - start the game now please, it is time",
                      "Yes - but switch Jev on first, the optional referee"]
    assert app.button_labels([("x", "Same"), ("y", "Same")]) == ["Same (x)", "Same (y)"]
    old_menu = app.button_labels(options)  # "Yes" next to "Yes, play": the Jev option shows its fuller label
    assert old_menu[:2] == ["Yes, play", "Pick a different model"]
    assert old_menu[2].startswith("Yes - and this time let me turn on Jev")
    assert app.button_labels([("k", "")]) == ["k"]
    assert app.labels_look_alike("Yes", "Yes, play") and app.labels_look_alike("Quit", "quit")
    assert not app.labels_look_alike("Yes, play", "Play with Jev on this time")
    assert not app.labels_look_alike("Try again", "Try another model")


def _choose_menus_in_the_source():
    """Every ``ui.choose(prompt, [(key, label), ...])`` with literal options in the game's code, as
    (where, [(key, label)]) - f-string labels with their placeholders filled in as "X"."""
    import ast

    def text(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
        return None

    src = Path(app.__file__).resolve().parents[1]
    constants = {}
    menus = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):  # module-level option lists (e.g. WELCOME_BACK_JEV_OPTIONS)
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)) and \
                    len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                constants[node.targets[0].id] = node.value
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "choose" and len(node.args) >= 2):
                continue
            arg = node.args[1]
            if isinstance(arg, ast.Call) and arg.args:  # list(SOME_OPTIONS)
                arg = arg.args[0]
            if isinstance(arg, ast.Name):
                arg = constants.get(arg.id, arg)
            if not isinstance(arg, (ast.List, ast.Tuple)):
                continue
            options = []
            for item in arg.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) == 2 and text(item.elts[0]) and text(item.elts[1]):
                    options.append((text(item.elts[0]), text(item.elts[1])))
            if len(options) >= 2:
                menus.append((f"{path.name}:{node.lineno}", options))
    return menus


def test_every_menu_in_the_game_shows_distinct_button_labels():
    """The game window shows each option's short label on a button: within one menu, no two may read alike
    (the welcome-back menu once showed "Yes, play" next to a bare "Yes" that started Jev's setup)."""
    menus = _choose_menus_in_the_source()
    assert len(menus) >= 10  # (the scan really finds the game's menus)
    assert any(dict(options).get("jev") for _where, options in menus)
    for where, options in menus:
        labels = [short_label(label) for _key, label in options]
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                assert not app.labels_look_alike(a, b), (where, labels)
    # (The check catches the old welcome-back menu.)
    old = ["Yes, play", "Pick a different model", short_label("Yes - and this time let me turn on Jev, the optional "
                                                              "AI referee")]
    assert app.labels_look_alike(old[0], old[2])


def test_pick_font_family_prefers_the_list_order_case_insensitively():
    assert pick_font_family(["courier", "dejavu sans mono", "Menlo"]) == "Menlo"
    assert pick_font_family(["DejaVu Sans Mono", "Liberation Mono"]) == "DejaVu Sans Mono"
    assert pick_font_family(["helvetica", "times"]) is None
    assert FONT_CANDIDATES[0] == "Cascadia Mono"


@pytest.mark.parametrize("env, expected", [
    ({}, False),
    ({"SteamDeck": "1"}, True),
    ({"SteamGamepadUI": "1"}, True),
    ({"GAMESCOPE_WAYLAND_DISPLAY": "gamescope-0"}, True),
    ({"SteamDeck": "1", app.FULLSCREEN_ENV: "0"}, False),
    ({app.FULLSCREEN_ENV: "yes"}, True),
    ({"SteamDeck": "0"}, False),
])
def test_full_screen_on_a_steam_deck(env, expected):
    assert wants_fullscreen(env) is expected


def test_steam_deck_detection_and_its_keyboard():
    assert app.on_steam_deck({"SteamDeck": "1"}) and app.on_steam_deck({"SteamGamepadUI": "1"})
    assert not app.on_steam_deck({}) and not app.on_steam_deck({"GAMESCOPE_WAYLAND_DISPLAY": "x"})
    opened = []
    assert app.open_steam_keyboard(lambda url: opened.append(url) or True) is True
    assert opened == ["steam://open/keyboard"]
    assert app.open_steam_keyboard(lambda url: (_ for _ in ()).throw(OSError())) is False


def test_colours_resolve_to_hex():
    assert resolve_color(None, "#123456") == "#123456"
    assert resolve_color(1, "#000000") == PALETTE_16[1]
    assert resolve_color(196, "#000000") == "#ff0000"
    assert resolve_color("#ABCDEF", "#000000") == "#abcdef"
    assert resolve_color(999, "#000000") == "#000000"
    assert blend("#000000", "#ffffff", 0.5) == "#808080"


def test_style_colors_apply_reverse_dim_and_links():
    assert style_colors(Style()) == (app.FOREGROUND, None)
    assert style_colors(Style(fg=2, bg=4)) == (PALETTE_16[2], PALETTE_16[4])
    assert style_colors(Style(reverse=True)) == (app.BACKGROUND, app.FOREGROUND)
    fg, _bg = style_colors(Style(dim=True))
    assert fg != app.FOREGROUND and fg == blend(app.FOREGROUND, app.BACKGROUND, 0.45)
    assert style_colors(Style(link="https://x.example"))[0] == app.LINK
    assert style_colors(Style(link="https://x.example", fg=1))[0] == PALETTE_16[1]


def test_glyph_fallbacks_swap_only_what_the_font_lacks():
    everything = glyph_fallbacks(lambda text: 10 * len(text))  # a font with every glyph
    assert everything == {}
    no_boxes = glyph_fallbacks(lambda text: 60 if "\u2500" <= text[0] <= "\u259f" or text[0] == "\u280b" else 10)
    assert "╭──┬─╮".translate(no_boxes) == "+--+-+"
    assert "│ ┃ ║".translate(no_boxes) == "| | |"
    assert "━━╸╺".translate(no_boxes) == "----"
    assert "█▓░".translate(no_boxes) == "###"
    assert "⠋⠙".translate(no_boxes) == "**"
    assert "✓ → …".translate(no_boxes) == "✓ → …"  # the font has those
    no_punctuation = glyph_fallbacks(lambda text: 10 if text[0].isascii() else 30)
    assert "✓ → … — •".translate(no_punctuation) == "v > . - *"
    assert all(len(v) == 1 for v in no_boxes.values())  # one cell each: tables stay aligned


def test_one_missing_probe_swaps_its_whole_group():
    table = glyph_fallbacks(lambda text: 60 if text == "╭" else 10)
    assert "─╭┃".translate(table) == "-+|"
    assert "█".translate(table) == "█"


def test_forced_fallbacks_swap_everything():
    table = glyph_fallbacks(lambda text: 10, force=True)
    assert "╭─╮ █ ⠋ ✓ →".translate(table) == "+-+ # * v >"


@pytest.mark.parametrize("cluster, width, expected", [
    ("a", 1, None), ("\u2500", 1, None), ("\u00e9", 1, None), ("\u2605", 1, None), ("\u2713", 1, None),
    ("\u26a0", 1, None), ("\u4e2d", 2, None),  # plain symbols and CJK are drawn as they are
    ("\U0001f9e0", 2, "+ "), ("\u26a1", 2, "! "), ("\u2615", 2, "* "), ("\U0001f600", 2, ": "),
    ("\U0001f98a", 2, "* "), ("\u26a0\ufe0f", 2, "\u26a0 "),
    ("\U0001f468\u200d\U0001f469\u200d\U0001f467", 2, "* "),  # one family, one stand-in
    ("\U0001d400", 1, "*"), ("\U0001f9e0", 0, ""), ("", 1, None),
])
def test_emoji_get_stand_ins_exactly_as_wide(cluster, width, expected):
    assert app.emoji_stand_in(cluster, width) == expected
    if expected is not None:
        assert len(expected) == width and all(ord(ch) <= 0xFFFF for ch in expected)


def test_emoji_safe_text_for_labels_and_buttons():
    assert app.emoji_safe_text("Plain text - with a dash") == "Plain text - with a dash"
    assert app.emoji_safe_text("\U0001f9e0 Smartest at a playable pace") == "+ Smartest at a playable pace"
    assert app.emoji_safe_text("Go \u26a1 go \U0001f98a!") == "Go ! go *!"
    assert app.emoji_safe_text("") == ""


def test_core_x11_fonts_are_recognised():
    assert app.uses_core_x11_fonts("x11", ["fixed", "courier", "helvetica"]) is True
    assert app.uses_core_x11_fonts("x11", ["DejaVu Sans Mono", "fixed"]) is False
    assert app.uses_core_x11_fonts("win32", ["courier"]) is False
    assert app.uses_core_x11_fonts("x11", []) is False


def test_glyph_fallbacks_survive_a_broken_measure():
    def broken(text):
        raise RuntimeError

    assert glyph_fallbacks(broken) == {}


def test_clean_argv_drops_macos_process_serial_numbers():
    assert app._clean_argv(["-psn_0_12345", "--mock"]) == (["--mock"], False)
    assert app._clean_argv(["--gui-selftest", "--no-jev"]) == (["--no-jev"], True)


def test_transcripts_go_to_documents_when_there_is_such_a_folder(tmp_path, monkeypatch, isolated_home):
    monkeypatch.setattr(app.Path, "home", classmethod(lambda cls: tmp_path))
    assert app.default_export_dir() == isolated_home / "transcripts"
    (tmp_path / "Documents").mkdir()
    assert app.default_export_dir() == tmp_path / "Documents" / "Get To Work"
    assert app._with_export_dir(["--mock"]) == ["--mock", "--export-dir", str(tmp_path / "Documents" / "Get To Work")]
    assert app._with_export_dir(["--export-dir", "x"]) == ["--export-dir", "x"]
    assert app._with_export_dir(["--export-dir=x"]) == ["--export-dir=x"]


def test_importing_the_gui_package_never_imports_tk():
    code = "import sys, gettowork.gui, gettowork.gui.app; print('tkinter' in sys.modules)"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# The self-test player
# ---------------------------------------------------------------------------


def test_selftest_player_answers_by_what_is_asked():
    player = SelftestPlayer()
    assert player.answer("[dim]Press Enter when you're ready to set off[/dim] > ") == ""
    assert player.answer("[dim]Press Enter to continue[/dim] > ", choices=[("", "Continue")]) == ""
    assert player.answer("[bold]How do you plan to get to work?[/bold] > ") == SelftestPlayer.PLANS[0]
    assert player.answer("[bold]What do you do?[/bold] > ") == SelftestPlayer.PLANS[1]
    assert player.answer("[bold]Save a transcript? \\[y/N][/bold] > ", choices=[("y", "Yes"), ("n", "No")]) == "n"
    assert player.answer("[bold]Play again? \\[Y/n][/bold] > ") == "n"
    assert player.answer("[bold]What would you like to do?[/bold] (retry) > ",
                         choices=[("retry", "Try again"), ("quit", "Stop")]) == ""
    assert player.answer("Paste your key: ", secret=True) == ""
    assert player.answer("Something new?") == "n"
    assert player.plans_given == 2 and len(player.answers) == 9


def test_selftest_player_keeps_inventing_plans_and_eventually_gives_up():
    player = SelftestPlayer(max_answers=10)
    plans = [player.answer("What do you do?") for _ in range(10)]
    assert plans[7] == plans[0]  # cycles through its plans
    assert player.answer("What do you do?") is None


# ---------------------------------------------------------------------------
# Crash reports, settings, and a window that can't open
# ---------------------------------------------------------------------------


def test_write_crash_report(isolated_home):
    try:
        raise RuntimeError("kaboom")
    except RuntimeError as exc:
        path = write_crash_report("testing", exc)
    assert path == isolated_home / "logs" / "gui-crash.txt"
    text = path.read_text(encoding="utf-8")
    assert "testing" in text and "RuntimeError: kaboom" in text and "Python" in text


def test_write_crash_report_never_raises(monkeypatch):
    from gettowork import crashlog

    monkeypatch.setattr(crashlog, "crash_log_path", lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")))
    assert write_crash_report("x", RuntimeError()) is None


def test_font_size_is_saved_in_settings_and_read_back(isolated_home):
    assert app.load_font_size() is None
    assert app.save_font_size(15) is True
    assert Settings.load().extra[app.FONT_SIZE_SETTING] == 15
    assert app.load_font_size() == 15


def test_saving_the_font_size_keeps_the_other_settings(isolated_home):
    settings = Settings(model_key="qwen3-4b", jev_enabled=False)
    settings.save()
    app.save_font_size(14)
    loaded = Settings.load()
    assert loaded.model_key == "qwen3-4b" and loaded.jev_enabled is False and loaded.extra["gui_font_size"] == 14


def test_an_unreadable_settings_file_is_not_overwritten(isolated_home):
    path = isolated_home / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert app.save_font_size(14) is False
    assert path.read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("value", [3, 99, "14", True, None])
def test_odd_saved_font_sizes_are_ignored(isolated_home, value):
    Settings(extra={app.FONT_SIZE_SETTING: value}).save()
    assert app.load_font_size() is None


def _no_tk():
    raise ImportError("No module named '_tkinter'")


def test_no_window_but_a_terminal_plays_in_the_terminal(isolated_home, monkeypatch):
    monkeypatch.setattr(app, "_import_tk", _no_tk)
    monkeypatch.setattr(app, "terminal_attached", lambda: True)
    calls = []
    code = app.run_gui(["--mock"], game_main=lambda argv, ui: calls.append((argv, ui)) or 7)
    assert code == 7 and calls == [(["--mock", "--export-dir", str(app.default_export_dir())], None)]
    assert "No module named '_tkinter'" in (isolated_home / "logs" / "gui-crash.txt").read_text(encoding="utf-8")


def test_no_window_and_no_terminal_exits_with_an_error(isolated_home, monkeypatch, capsys, no_real_dialogs):
    monkeypatch.setattr(app, "_import_tk", _no_tk)
    monkeypatch.setattr(app, "terminal_attached", lambda: False)
    assert app.run_gui([], game_main=lambda argv, ui: pytest.fail("must not play")) == 1
    assert "couldn't open" in capsys.readouterr().err
    crash = isolated_home / "logs" / "gui-crash.txt"
    assert crash.exists()
    # Started by a double-click or from Steam there's no console: a message box says what happened.
    [(title, message)] = no_real_dialogs
    assert title == "Get To Work"
    assert "couldn't open its window (ImportError: No module named '_tkinter')" in message
    assert "Verify integrity of game files" in message and str(crash) in message


def test_the_selftest_never_falls_back_to_the_terminal(monkeypatch, no_real_dialogs):
    monkeypatch.setattr(app, "_import_tk", _no_tk)
    monkeypatch.setattr(app, "terminal_attached", lambda: True)
    assert app.run_gui([], selftest=True, game_main=lambda argv, ui: pytest.fail("must not play")) == 1
    assert no_real_dialogs == []  # (and never pops up a message box in CI)


def test_a_terminal_player_gets_no_message_box(isolated_home, monkeypatch, no_real_dialogs):
    monkeypatch.setattr(app, "_import_tk", _no_tk)
    monkeypatch.setattr(app, "terminal_attached", lambda: True)
    assert app.run_gui([], game_main=lambda argv, ui: 0) == 0
    assert no_real_dialogs == []


class _Ran:
    def __init__(self, code=0):
        self.calls: list = []
        self.code = code

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if isinstance(self.code, BaseException):
            raise self.code
        return subprocess.CompletedProcess(args, self.code)


def test_error_dialog_on_windows_macos_and_linux(monkeypatch):
    ran = _Ran()
    assert REAL_SHOW_ERROR_DIALOG("Get To Work", "It broke.", system="win32", runner=ran)
    assert ran.calls == [["MessageBoxW", "Get To Work", "It broke."]]
    ran = _Ran()
    assert REAL_SHOW_ERROR_DIALOG("Get To Work", 'Say "hi" \\ bye', system="darwin", runner=ran)
    [call] = ran.calls
    assert call[0] == "osascript" and call[-2:] == ["Get To Work", 'Say "hi" \\ bye']  # text as arguments, never code
    assert all('"hi"' not in part for part in call[:-2])
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name in ("kdialog", "xmessage") else None)
    ran = _Ran()
    assert REAL_SHOW_ERROR_DIALOG("Get To Work", "It broke.", system="linux", runner=ran, env={"DISPLAY": ":0"})
    assert ran.calls == [["/usr/bin/kdialog", "--title", "Get To Work", "--error", "It broke."]]


def test_error_dialog_needs_a_display_on_linux_and_never_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    ran = _Ran()
    assert not REAL_SHOW_ERROR_DIALOG("t", "m", system="linux", runner=ran, env={})
    assert ran.calls == []
    assert not REAL_SHOW_ERROR_DIALOG("t", "m", system="linux", runner=_Ran(OSError("gone")), env={"DISPLAY": ":0"})
    assert not REAL_SHOW_ERROR_DIALOG("t", "m", system="darwin", runner=_Ran(OSError("no osascript")))
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert not REAL_SHOW_ERROR_DIALOG("t", "m", system="linux", runner=_Ran(), env={"WAYLAND_DISPLAY": "w"})


def test_macos_refuses_to_open_the_window_off_the_main_thread(isolated_home, monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "terminal_attached", lambda: False)
    monkeypatch.setattr(app, "_import_tk", lambda: pytest.fail("Tk must not be touched"))
    result = []
    worker = threading.Thread(target=lambda: result.append(app.run_gui([], game_main=lambda a, u: 0)))
    worker.start()
    worker.join(10)
    assert result == [1]
    assert "main thread" in (isolated_home / "logs" / "gui-crash.txt").read_text(encoding="utf-8")


def test_the_terminal_fallback_runs_the_real_cli_by_default(monkeypatch):
    monkeypatch.setattr(app, "_import_tk", _no_tk)
    monkeypatch.setattr(app, "terminal_attached", lambda: True)
    seen = []
    from gettowork import cli

    monkeypatch.setattr(cli, "main", lambda argv=None, **kw: seen.append((argv, kw)) or 0)
    assert app.run_gui(["--mock", "--export-dir", "here"]) == 0
    assert seen == [(["--mock", "--export-dir", "here"], {})]


# ---------------------------------------------------------------------------
# Real windows
# ---------------------------------------------------------------------------


def _display_available() -> Optional[str]:
    """None if Tk windows can open here, else the reason they can't."""
    try:
        import tkinter
    except ImportError:
        return "this Python has no tkinter"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return "no display (run under xvfb-run)"
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        return f"Tk can't open a window: {exc}"
    root.destroy()
    return None


@pytest.fixture
def display():
    pytest.importorskip("tkinter")
    reason = _display_available()
    if reason:
        pytest.skip(reason)


class Harness:
    """A GameWindow running a scripted game; this test thread pumps Tk events."""

    def __init__(self, game: Callable[[list, Any], int], argv=(), **kwargs) -> None:
        import tkinter

        self.root = tkinter.Tk()
        kwargs.setdefault("font_size", 11)
        kwargs.setdefault("env", {})
        kwargs.setdefault("remember_font_size", False)
        self.window = app.GameWindow(self.root, list(argv), game_main=game, **kwargs)
        self.window.start()
        self.code: Optional[int] = None

    def pump(self, until: Callable[[], bool] = lambda: False, timeout: float = 10.0, what: str = "") -> None:
        deadline = time.monotonic() + timeout
        while not until():
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {what or 'the window'}")
            self.root.update()
            time.sleep(0.005)
        self.root.update()

    def settle(self, seconds: float = 0.15) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.root.update()
            time.sleep(0.005)

    def wait_prompt(self, text: str = "", timeout: float = 10.0):
        self.pump(lambda: self.window.prompt is not None and text in self.window.prompt.text, timeout,
                  f"a prompt containing {text!r}")
        return self.window.prompt

    def transcript(self) -> str:
        return self.window.buffer.text()

    def widget_text(self) -> str:
        return self.window.text.get("1.0", "end-1c")

    def close(self) -> int:
        self.window.request_close()
        self.pump(lambda: self.window.closed, 15, "the window to close")
        self.code = self.window.finish()
        return self.code

    def cleanup(self) -> None:
        if not self.window._finished:
            self.window.bridge.close()
            self.window.finish()
        self.root = None


@pytest.fixture
def harness(display):
    made = []

    def make(game, argv=(), **kwargs) -> Harness:
        h = Harness(game, argv, **kwargs)
        made.append(h)
        return h

    yield make
    for h in made:
        h.cleanup()
        assert h.window.tk_errors == []


def test_the_window_shows_the_game_and_takes_answers(harness):
    got = {}

    def game(argv, ui):
        ui.console.print("[bold magenta]GET TO WORK[/] - a farcical race")
        got["name"] = ui.ask("What's your name?")
        ui.say(f"Hello, {got['name']}!")
        got["can_hide"] = ui.can_hide_input()
        return 0

    h = harness(game)
    assert h.root.title() == "Get To Work"
    prompt = h.wait_prompt("name")
    assert h.window.prompt_label.cget("text") == "What's your name?"
    assert h.window.entry.cget("show") == ""
    assert "GET TO WORK - a farcical race" in h.widget_text()
    h.window.type_answer("Ada")
    h.pump(lambda: "Hello, Ada!" in h.transcript(), what="the reply")
    assert got == {"name": "Ada", "can_hide": True}
    assert "What's your name? > Ada" in h.transcript()  # the question and answer stay in the transcript
    assert h.window.prompt is None and h.window.prompt_label.cget("text") in ("", EXIT_HINT)
    assert prompt.id >= 1


def test_the_widget_mirrors_the_buffer_exactly(harness):
    def game(argv, ui):
        for i in range(300):
            ui.console.print(f"line {i} [red]red[/] [link=https://example.com/{i % 3}]link[/link]")
        with ui.status("thinking"):
            time.sleep(0.2)
        ui.say("done")
        return 0

    h = harness(game, scrollback=120)
    h.pump(lambda: h.window._game_done, what="the game to finish")
    lines = h.widget_text().split("\n")
    assert lines == [h.window.buffer.line_text(i) for i in range(len(h.window.buffer))]
    assert len(lines) <= 120 and "line 299 red link" in lines
    assert "thinking" not in h.widget_text()


def test_emoji_never_reach_tk_on_linux_and_the_columns_stay_put(harness):
    from rich.panel import Panel
    from rich.table import Table

    def game(argv, ui):
        table = Table(title="Models")
        table.add_column("Model")
        table.add_column("Badge")
        table.add_row("gpt-oss 20B", "\U0001f9e0 Smartest at a playable pace")
        table.add_row("Qwen3 1.7B", "\u26a1 Fastest comfortable fit")
        ui.console.print(table)
        ui.console.print(Panel("The goose \U0001f9a2 honks, waves \U0001f44b\U0001f3fd and says \u26a0\ufe0f hello"))
        ui.choose("Pick one", [("1", "\U0001f9e0 Smartest"), ("2", "\u26a1 Fastest")], default="1")
        return 0

    h = harness(game)
    h.wait_prompt("Pick one")
    text = h.widget_text()
    if not h.window._x11:  # Windows / macOS: Tk draws emoji itself
        assert "\U0001f9e0" in text
        return
    assert all(ord(ch) <= 0xFFFF for ch in text) and "\u26a1" not in text and "\ufe0f" not in text
    assert "+  Smartest at a playable pace" in text and "!  Fastest comfortable fit" in text  # 2 cells each
    assert [label for _key, label in h.window.choice_buttons] == ["+ Smartest", "! Fastest"]
    # Each table and panel line is exactly as many characters as rich laid out cells.
    for index in range(len(h.window.buffer)):
        line = h.window.text.get(f"{index + 1}.0", f"{index + 1}.end")
        cells = sum(len(chunk) for chunk, _style in h.window.buffer.line(index, substitute=lambda c, w: "#" * w))
        assert len(line) == cells, line


def test_a_long_question_never_hides_the_newest_lines(harness):
    def game(argv, ui):
        for i in range(120):
            ui.say(f"line {i}")
        folder = "/home/player/Documents/" + "a-rather-long-folder-name/" * 8
        ui.confirm(f"Save a transcript of this game in {folder}?")
        return 0

    h = harness(game)
    h.wait_prompt("Save a transcript")
    h.settle(0.4)
    assert int(h.window.prompt_label.winfo_height()) > int(h.window.prompt_label.winfo_reqheight() / 3)
    assert h.window.text.yview()[1] >= 0.999  # the question (the last line) is in view...
    last = h.window.text.index("end-1c linestart")
    assert h.window.text.dlineinfo(last) is not None  # ...really drawn, not under the input area
    h.window.press_choice("n")
    h.pump(lambda: h.window._game_done)


def test_the_transcript_stays_where_the_player_scrolled(harness):
    def game(argv, ui):
        for i in range(200):
            ui.say(f"line {i}")
        ui.ask("Anything?")
        return 0

    h = harness(game)
    h.wait_prompt("Anything")
    h.settle(0.2)
    h.window.text.yview_moveto(0.0)  # the player scrolls up to re-read something
    h.settle(0.2)
    h.root.geometry(f"{h.root.winfo_width()}x{h.root.winfo_height() - 40}")  # ...and the window changes size
    h.settle(0.4)
    assert h.window.text.yview()[0] == 0.0


def test_styles_become_tags(harness):
    def game(argv, ui):
        ui.console.print("[red]alarm[/] [on blue]sky[/] [underline]under[/] [strike]gone[/] [bold]loud[/]")
        return 0

    h = harness(game)
    h.pump(lambda: "alarm" in h.widget_text())
    text = h.window.text

    def tag_option(word: str, option: str) -> str:
        start = text.search(word, "1.0")
        names = [t for t in text.tag_names(start) if t.startswith("style")]
        assert names, word
        return str(text.tag_cget(names[0], option))

    assert tag_option("alarm", "foreground") == PALETTE_16[1]
    assert tag_option("sky", "background") == PALETTE_16[4]
    assert tag_option("under", "underline") in ("1", "true")
    assert tag_option("gone", "overstrike") in ("1", "true")
    if h.window._bold_ok:  # a bold font as wide as the regular one: used
        assert str(h.window.font_bold) == tag_option("loud", "font")
    else:  # otherwise bold text is drawn brighter, and columns stay aligned
        assert tag_option("loud", "foreground") == blend(app.FOREGROUND, "#ffffff", 0.4)


def test_links_are_clickable(harness):
    opened = []

    def game(argv, ui):
        ui.console.print("Read the [link=https://typesafe.ai/docs]docs[/link] first.")
        ui.ask("Ready?")
        return 0

    h = harness(game, opener=lambda url: opened.append(url) or True)
    h.wait_prompt("Ready")
    text = h.window.text
    start = text.search("docs", "1.0")
    link_tags = [t for t in text.tag_names(start) if t.startswith("link")]
    assert link_tags and h.window.link_url(link_tags[0]) == "https://typesafe.ai/docs"
    h.window.open_link("https://typesafe.ai/docs")
    h.window.open_link("file:///etc/passwd")
    h.pump(lambda: opened, what="the browser")
    h.settle(0.1)
    assert opened == ["https://typesafe.ai/docs"]


def test_menus_get_buttons(harness):
    result = {}

    def game(argv, ui):
        result["menu"] = ui.choose("What would you like to do?", [
            ("retry", "Try again (handy if the internet hiccupped)"),
            ("mock", "Play now with the pretend model (offline, nothing to download)"),
            ("quit", "Stop for now - anything already downloaded is kept for next time"),
        ], default="retry")
        result["confirm"] = ui.confirm("Play again?", default=True)
        return 0

    h = harness(game)
    h.wait_prompt("What would you like")
    assert h.window.choice_buttons == [("retry", "Try again"), ("mock", "Play now with the pretend model"),
                                       ("quit", "Stop for now")]
    assert all(b.winfo_ismapped() for b in h.window._choice_buttons)
    assert h.window.press_choice("mock")
    h.wait_prompt("Play again")
    assert h.window.choice_buttons == [("y", "Yes"), ("n", "No")]
    assert h.window.press_choice("n")
    h.pump(lambda: h.window._game_done, what="the game to finish")
    assert result == {"menu": "mock", "confirm": False}
    assert "What would you like to do? (retry) > mock" in h.transcript()
    assert h.window.choice_buttons == [("close", "Close")]  # the game is over


def test_many_buttons_wrap_onto_several_rows(harness):
    options = [(str(i), f"Option number {i} with a longish label") for i in range(1, 9)]

    def game(argv, ui):
        return int(ui.choose("Pick one", options))

    h = harness(game)
    h.wait_prompt("Pick one")
    h.settle(0.2)
    rows = {b.winfo_manager() and str(b.pack_info()["in"]) for b in h.window._choice_buttons}
    assert len(rows) >= 2
    for button in h.window._choice_buttons:
        assert button.winfo_x() + button.winfo_width() <= h.root.winfo_width()
    h.window.press_choice("8")
    h.pump(lambda: h.window._game_done)
    assert h.close() == 8


def test_pauses_get_a_continue_button(harness):
    def game(argv, ui):
        ui.say("A long lesson...")
        ui.pause()
        ui.say("after the pause")
        return 0

    h = harness(game)
    h.wait_prompt("Press Enter")
    assert h.window.choice_buttons == [("", "Continue")]
    h.window.press_choice("")
    h.pump(lambda: "after the pause" in h.transcript())


def test_buttons_are_disabled_once_pressed(harness):
    def game(argv, ui):
        ui.confirm("Sure?")
        time.sleep(0.3)  # the game takes a moment before hiding the buttons
        return 0

    h = harness(game)
    h.wait_prompt("Sure?")
    buttons = list(h.window._choice_buttons)
    h.window.press_choice("y")
    assert all(b.instate(["disabled"]) for b in buttons)
    assert h.window.press_choice("n") is True  # a second press does nothing...
    h.pump(lambda: h.window._game_done)
    assert "Sure? [y/N] > y" in h.transcript()  # ...only the first answer counted


def test_secret_answers_are_masked_and_never_shown(harness):
    got = {}

    def game(argv, ui):
        got["hidden"] = ui.can_hide_input()
        got["key"] = ui.secret("Paste your TypeSafe API key")
        got["plan"] = ui.ask("Plan?")
        return 0

    h = harness(game)
    h.wait_prompt("API key")
    assert h.window.entry.cget("show") == "•"
    assert SECRET_HINT in h.window.prompt_label.cget("text")
    h.window.type_answer("tsk-super-secret")
    h.wait_prompt("Plan?")
    assert h.window.entry.cget("show") == ""
    assert got["key"] == "tsk-super-secret" and got["hidden"] is True
    assert "tsk-super-secret" not in h.transcript() and "tsk-super-secret" not in h.widget_text()
    assert "Paste your TypeSafe API key: (hidden)" in h.transcript()
    h.window._recall(-1)  # history never offers the key
    assert h.window.entry.get() == ""
    h.window.type_answer("walk")


KEY = "tsk_live_9f8e7d6c5b4a3210SECRET"


def test_a_key_pasted_at_an_ordinary_question_is_hidden_and_kept_out_of_history(harness):
    answers = []

    def game(argv, ui):
        answers.append(ui.ask("What next? (paste)"))
        answers.append(ui.ask("Plan?"))
        answers.append(ui.ask("Plan?"))
        return 0

    h = harness(game)
    h.wait_prompt("What next?")
    h.window.type_answer(KEY)  # the wrong box: an ordinary, unmasked question
    h.wait_prompt("Plan?")
    h.window.type_answer(f"I read {KEY} aloud to the bus driver")
    h.wait_prompt("Plan?")
    assert answers == [KEY, f"I read {KEY} aloud to the bus driver"]  # the game still gets what was typed
    assert KEY not in h.transcript() and KEY not in h.widget_text()
    assert "What next? (paste) > (hidden)" in h.transcript()
    h.window._recall(-1)
    assert h.window.entry.get() == ""  # never offered back with Up
    h.window.type_answer("walk")
    h.pump(lambda: h.window._game_done)
    assert h.window._history == ["walk"]


def test_a_key_given_at_the_key_question_stays_hidden_wherever_it_is_pasted_again(harness):
    def game(argv, ui):
        ui.secret("Paste your key")
        ui.ask("Plan?")
        return 0

    h = harness(game)
    h.wait_prompt("Paste your key")
    h.window.type_answer("short-key-1")  # not key-shaped at all - but known to be one
    h.wait_prompt("Plan?")
    assert h.window.is_secret_answer("short-key-1") and not h.window.is_secret_answer("walk to work")
    h.window.type_answer("short-key-1")
    h.pump(lambda: h.window._game_done)
    assert "short-key-1" not in h.transcript() and h.window._history == []


def test_model_names_and_web_addresses_are_not_mistaken_for_keys():
    from gettowork.ui import looks_like_secret

    for text in ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "bartowski/Qwen2.5-7B-Instruct-GGUF",
                 "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF", "I ride my 2 bikes to work at 9",
                 "supercalifragilisticexpialidocious", "123e4567-e89b-12d3-a456-426614174000", ""):
        assert not looks_like_secret(text), text
    for text in (KEY, "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", f"Bearer {KEY}", f"TYPESAFE_API_KEY={KEY}"):
        assert looks_like_secret(text), text


@pytest.mark.parametrize("question", ["How do you plan to get to work?", "What do you do?"])
def test_a_long_answer_wraps_with_its_question_inside_the_window(harness, question):
    plan = ("I ride my grandmother's tandem bicycle down the high street, ringing the bell at every corner "
            "and handing out warm croissants to anyone who waves back at me")

    def game(argv, ui):
        ui.ask(question)
        return 0

    h = harness(game)
    h.wait_prompt(question)
    h.window.type_answer(plan)
    h.pump(lambda: h.window._game_done)
    columns = h.window.console_size[0]
    lines = h.transcript().splitlines()
    assert all(len(line) <= columns for line in lines), [line for line in lines if len(line) > columns]
    joined = " ".join(line.strip() for line in lines)
    assert f"{question} > {plan}" in joined  # every word, in order, after the question (asked once)
    assert h.transcript().count(question) == 1
    h.window.text.update_idletasks()
    assert h.window.text.xview() == (0.0, 1.0)  # nothing runs off to the right


def test_a_short_answer_still_sits_after_its_question(harness):
    def game(argv, ui):
        ui.ask("Plan?")
        return 0

    h = harness(game)
    h.wait_prompt("Plan?")
    h.window.type_answer("walk")
    h.pump(lambda: h.window._game_done)
    assert "Plan? > walk" in h.transcript()


def test_enter_with_no_question_waiting_keeps_the_text(harness):
    release = threading.Event()

    def game(argv, ui):
        release.wait(5)
        return 0 if ui.ask("Now?") == "typed early" else 1

    h = harness(game)
    h.settle(0.1)
    h.window.entry.insert(0, "typed early")
    h.window._submit()
    assert h.window.entry.get() == "typed early"
    release.set()
    h.wait_prompt("Now?")
    h.window._submit()
    h.pump(lambda: h.window._game_done)
    assert h.close() == 0


def test_answer_history_with_up_and_down(harness):
    def game(argv, ui):
        for _ in range(3):
            ui.ask("Plan?")
        return 0

    h = harness(game)
    for plan in ("walk", "bike"):
        h.wait_prompt("Plan?")
        h.window.type_answer(plan)
        h.pump(lambda: h.window.prompt is None or h.window.prompt.text.count("Plan") == 1)
    h.wait_prompt("Plan?")
    h.window._recall(-1)
    assert h.window.entry.get() == "bike"
    h.window._recall(-1)
    assert h.window.entry.get() == "walk"
    h.window._recall(+1)
    h.window._recall(+1)
    assert h.window.entry.get() == ""


def test_typing_in_the_transcript_goes_to_the_input_bar(harness):
    def game(argv, ui):
        ui.ask("Plan?")
        return 0

    h = harness(game)
    h.wait_prompt("Plan?")
    event = type("E", (), {"state": 0, "keysym": "a", "char": "a"})()
    assert h.window._on_transcript_key(event) == "break"
    assert h.window.entry.get() == "a"
    ctrl = type("E", (), {"state": 0x4, "keysym": "c", "char": "\x03"})()
    assert h.window._on_transcript_key(ctrl) is None  # Ctrl+C stays a copy
    back = type("E", (), {"state": 0, "keysym": "BackSpace", "char": "\x08"})()
    assert h.window._on_transcript_key(back) == "break" and h.window.entry.get() == ""
    h.window._submit()


def test_ctrl_c_after_selecting_transcript_text_copies_it(harness):
    """Pressing Ctrl+C sends Control_L first - without the Control bit set yet - then "c". The bare Control_L
    must not move the focus to the input bar (the "c" would follow it there and copy nothing); nor may F11
    or Escape be swallowed on the way to the window's own shortcuts."""
    def game(argv, ui):
        ui.say("COPY-ME-PLEASE is the link")
        ui.ask("Plan?")
        return 0

    h = harness(game)
    h.wait_prompt("Plan?")
    moved = []
    real_focus = h.window.entry.focus_set
    h.window.entry.focus_set = lambda: moved.append(True)
    for keysym in ("Control_L", "Control_R", "Shift_L", "Meta_L", "Alt_L", "Super_L", "Caps_Lock", "F11",
                   "Escape", "ISO_Level3_Shift"):
        event = type("E", (), {"state": 0, "keysym": keysym, "char": ""})()
        assert h.window._on_transcript_key(event) is None, keysym
    assert moved == [] and h.window.entry.get() == ""
    h.window.entry.focus_set = real_focus
    # Copy from the transcript, without the keyboard too: its right-click menu.
    start = h.window.text.search("COPY-ME-PLEASE", "1.0")
    assert start
    h.window.text.tag_add("sel", start, f"{start}+14c")
    h.root.clipboard_clear()
    h.root.clipboard_append("CLIPBOARD-UNCHANGED")
    assert h.window._transcript_menu.entrycget(0, "label") == "Copy"
    h.window._transcript_menu.invoke(0)
    assert h.root.clipboard_get() == "COPY-ME-PLEASE"
    h.window._transcript_menu.invoke(1)  # Select all
    assert "Plan?" in h.window.text.get("sel.first", "sel.last")
    h.window.text.tag_remove("sel", "1.0", "end")
    assert h.window._copy_transcript_selection() is False
    h.window._submit()


def test_the_end_of_the_game_waits_for_enter(harness):
    def game(argv, ui):
        ui.say("bye for now")
        return 3

    h = harness(game)
    h.pump(lambda: h.window._game_done, what="the game to finish")
    assert EXIT_HINT in h.transcript() and h.window.prompt_label.cget("text") == EXIT_HINT
    assert h.window.choice_buttons == [("close", "Close")]
    assert not h.window.closed
    h.window._submit()  # Enter
    h.pump(lambda: h.window.closed)
    assert h.window.finish() == 3


def test_closing_mid_question_ends_the_game_politely(harness):
    events = []

    def game(argv, ui):
        try:
            ui.ask("How do you plan to get to work?")
        except UserQuit:
            ui.say("No problem - see you next time!")
            events.append("goodbye")
            return 130
        return 0

    h = harness(game)
    h.wait_prompt("plan")
    started = time.monotonic()
    code = h.close()
    assert code == 0  # closing the window is a normal way to stop playing
    assert time.monotonic() - started < 5
    assert events == ["goodbye"]
    assert not h.window._worker.is_alive()
    assert "see you next time" in h.transcript()


def test_closing_gives_a_stuck_game_only_a_few_seconds(harness):
    stop = threading.Event()

    def game(argv, ui):
        ui.say("downloading forever...")
        stop.wait(30)  # ignores the window closing
        return 0

    h = harness(game, close_wait_s=0.5)
    h.pump(lambda: "downloading" in h.transcript())
    started = time.monotonic()
    assert h.close() == 0
    assert time.monotonic() - started < 3
    stop.set()


def test_a_crashing_game_says_sorry_and_keeps_the_details(harness, isolated_home):
    def game(argv, ui):
        raise RuntimeError("a bug in the game")

    h = harness(game)
    h.pump(lambda: h.window._game_done)
    assert "Oops - the game stopped unexpectedly" in h.transcript()
    assert "a bug in the game" in (isolated_home / "logs" / "gui-crash.txt").read_text(encoding="utf-8")
    assert h.close() == 1


def test_system_exit_in_the_game_becomes_its_exit_code(harness):
    h = harness(lambda argv, ui: sys.exit(4))
    h.pump(lambda: h.window._game_done)
    assert h.close() == 4


def test_prints_and_errors_from_anywhere_land_in_the_window(harness):
    original = sys.stdout

    def game(argv, ui):
        print("a stray print")
        sys.stderr.write("a stray warning\n")
        return 0

    h = harness(game)
    h.pump(lambda: h.window._game_done)
    assert "a stray print" in h.transcript() and "a stray warning" in h.transcript()
    h.close()
    assert sys.stdout is original


def test_rich_progress_and_spinners_render_their_final_state(harness):
    def game(argv, ui):
        with ui.download_progress("Downloading model.gguf", 1000) as advance:
            for _ in range(10):
                advance(100)
                time.sleep(0.03)
        with ui.status("Warming up..."):
            time.sleep(0.3)
        ui.success("ready")
        return 0

    h = harness(game)
    h.pump(lambda: h.window._game_done, what="the game to finish")
    lines = [line for line in h.transcript().split("\n") if line.strip()]
    assert lines[0].startswith("Downloading model.gguf") and "1.0/1.0 kB" in lines[0]
    assert lines[1] == "OK ready"
    assert "Warming up" not in h.widget_text()


def test_the_console_matches_the_window_size(harness):
    widths = []

    def game(argv, ui):
        widths.append(ui.console.width)
        ui.ask("resize me")
        widths.append(ui.console.width)
        return 0

    h = harness(game)
    h.wait_prompt("resize me")
    columns, rows = h.window.console_size
    # The harness pins the font (11 pt), so the window can't shrink it to fit: on a small
    # desktop (a CI runner's 1024x768) 100x32 characters at that size don't fit the screen,
    # and the window fits the screen instead. (The real game shrinks the font: see
    # test_the_desktop_window_still_fits_100_columns_on_a_small_screen.)
    need_w, need_h = h.window._window_size_for(app.MIN_COLUMNS, app.MIN_ROWS)
    room_w, room_h = int(h.root.winfo_screenwidth() * 0.96), int(h.root.winfo_screenheight() * 0.88)
    if need_w <= room_w and need_h <= room_h:
        assert columns >= 100 and rows >= 32
    else:
        assert columns >= 40 and rows >= 10
    assert widths[0] == columns and h.window.bridge.columns == columns
    h.root.geometry("640x480")
    h.pump(lambda: h.window.console_size[0] < columns, what="the console to narrow")
    assert h.window.console.width == h.window.console_size[0] == h.window.bridge.columns
    char_w = h.window.font.measure("0")
    assert h.window.console_size[0] * char_w <= h.window.text.winfo_width()
    h.window.type_answer("")
    h.pump(lambda: h.window._game_done)
    assert widths[1] == h.window.console_size[0]


def test_zoom_changes_the_font_and_the_console_width(harness, isolated_home):
    def game(argv, ui):
        ui.ask("zoom?")
        return 0

    h = harness(game, remember_font_size=True)
    h.wait_prompt("zoom?")
    h.settle(0.2)
    size, (columns, rows) = h.window.font_size, h.window.console_size
    h.window.zoom(+2)
    h.settle(0.2)
    assert h.window.font_size == size + 2
    # Bigger letters, less text: fewer columns or rows (a bitmap font can keep its width for
    # a couple of point sizes and grow only in height), and the text still fits the window.
    new_columns, new_rows = h.window.console_size
    assert (new_columns, new_rows) != (columns, rows) and new_columns <= columns and new_rows <= rows
    assert new_columns * h.window.font.measure("0") <= h.window.text.winfo_width()
    h.window.zoom(-1)
    assert h.window.font_size == size + 1
    for key in ("<Control-equal>", "<Control-minus>", "<Control-0>"):
        assert h.root.bind(key)  # the keyboard shortcuts are wired up
    h.window.type_answer("")
    h.pump(lambda: h.window._game_done)
    h.close()
    assert Settings.load().extra[app.FONT_SIZE_SETTING] == size + 1  # remembered for next time


def test_zoom_reset_and_limits(harness):
    h = harness(lambda argv, ui: 0, font_size=10)
    h.window.zoom(0)
    assert h.window.font_size == app.DEFAULT_FONT_SIZE
    for _ in range(40):
        h.window.zoom(+1)
    assert h.window.font_size == app.MAX_FONT_SIZE
    for _ in range(40):
        h.window.zoom(-1)
    assert h.window.font_size == app.MIN_FONT_SIZE


def test_a_remembered_font_size_is_used(display, isolated_home):
    import tkinter

    app.save_font_size(14)
    root = tkinter.Tk()
    window = app.GameWindow(root, [], game_main=lambda argv, ui: 0, env={})
    try:
        assert window.font_size == 14
    finally:
        window.finish()


def test_the_default_window_fits_a_steam_deck_screen(display, monkeypatch):
    import tkinter

    root = tkinter.Tk()
    root.winfo_screenwidth = lambda: 1280  # type: ignore[method-assign]
    root.winfo_screenheight = lambda: 800  # type: ignore[method-assign]
    window = app.GameWindow(root, [], game_main=lambda argv, ui: 0, env={}, remember_font_size=False)
    try:
        root.update()
        width, height = (int(v) for v in root.geometry().split("+")[0].split("x"))
        assert width <= 1280 and height <= 800
        columns, rows = window.console_size
        assert columns >= 100 and rows >= 32
    finally:
        window.finish()


def test_steam_deck_starts_full_screen_with_bigger_text_and_buttons(display, monkeypatch):
    import tkinter

    if sys.platform == "darwin":  # a real full-screen switch animates into its own Space: skip the show
        monkeypatch.setattr(app.GameWindow, "set_fullscreen",
                            lambda self, on: setattr(self, "fullscreen", bool(on)))
    root = tkinter.Tk()
    window = app.GameWindow(root, [], game_main=lambda argv, ui: 0, env={"SteamDeck": "1"},
                            remember_font_size=False)
    try:
        assert window.fullscreen is True
        padding = str(window.ttk.Style(root).lookup("Choice.TButton", "padding"))
        assert "18" in padding and "12" in padding
        window.set_fullscreen(False)
        assert window.fullscreen is False
    finally:
        window.finish()


def test_steam_deck_keeps_its_big_font_on_its_own_1280x800_screen(display, monkeypatch):
    """Full screen fills the screen and keeps the 16-point font while 80x24 fit - it never
    shrinks to the desktop size just to show 100x32 characters."""
    import tkinter

    if sys.platform == "darwin":
        monkeypatch.setattr(app.GameWindow, "set_fullscreen",
                            lambda self, on: setattr(self, "fullscreen", bool(on)))
    root = tkinter.Tk()
    root.winfo_screenwidth = lambda: 1280  # the Deck's built-in screen, whatever display the tests have
    root.winfo_screenheight = lambda: 800
    window = app.GameWindow(root, [], game_main=lambda argv, ui: 0, env={"SteamDeck": "1"},
                            remember_font_size=False)
    try:
        assert window.fullscreen is True
        columns, rows = window._size
        fits = window._window_size_for(app.FULLSCREEN_MIN_COLUMNS, app.FULLSCREEN_MIN_ROWS)
        assert fits[0] <= 1280 and fits[1] <= 800
        if window.font_size == app.FULLSCREEN_FONT_SIZE:
            assert columns >= app.FULLSCREEN_MIN_COLUMNS and rows >= app.FULLSCREEN_MIN_ROWS
        else:
            # Font sizes are in points, so their pixel size depends on the display's DPI (Windows'
            # 96 dpi draws bigger letters than a virtual X screen). The big font may shrink - but
            # only as far as needed: one size up, 80x24 characters must no longer fit.
            assert window.font_size < app.FULLSCREEN_FONT_SIZE
            window._set_font_size(window.font_size + 1)
            bigger = window._window_size_for(app.FULLSCREEN_MIN_COLUMNS, app.FULLSCREEN_MIN_ROWS)
            assert bigger[0] > 1280 or bigger[1] > 800
    finally:
        window.finish()


def test_the_desktop_window_still_fits_100_columns_on_a_small_screen(display):
    import tkinter

    root = tkinter.Tk()
    root.winfo_screenwidth = lambda: 1280
    root.winfo_screenheight = lambda: 800
    window = app.GameWindow(root, [], game_main=lambda argv, ui: 0, env={}, remember_font_size=False)
    try:
        assert window.fullscreen is False
        assert window._size[0] >= app.MIN_COLUMNS or window.font_size <= app.MIN_FONT_SIZE + 1
    finally:
        window.finish()


def test_report_a_problem_opens_the_games_steam_discussions(harness):
    """Steam's overlay can't open over this window (outside a Deck's Game Mode): the window has
    its own way to the place the first-launch notice tells players to report things."""
    from gettowork import notices

    opened = []
    h = harness(lambda argv, ui: 0, opener=lambda url: opened.append(url) or True)
    assert h.window.report_button.cget("text") == notices.REPORT_BUTTON == app.REPORT_BUTTON
    assert notices.REPORT_BUTTON in notices.REPORT_HOW
    h.window.report_button.invoke()
    h.pump(lambda: opened, what="the browser")
    assert opened == [notices.report_url()]
    # The link is always in the transcript too, clickable (a browser may never appear).
    assert f"Opening {notices.report_url()} to report a problem" in " ".join(h.transcript().split())
    assert "Couldn't open a browser" not in h.transcript()


def test_report_a_problem_shows_the_link_when_no_browser_opens(harness):
    from gettowork import notices

    tried = []
    h = harness(lambda argv, ui: ui.ask("Plan?") and 0, opener=lambda url: tried.append(url) and False)
    h.wait_prompt("Plan?")
    h.window.report_button.invoke()
    h.pump(lambda: "Couldn't open a browser" in h.transcript(), what="the fallback line")
    text = " ".join(h.transcript().split())
    assert notices.report_url() in text and "open that link on any device" in text
    assert "copy the link above" in text
    h.window._submit()


def test_on_a_steam_deck_report_a_problem_goes_through_steams_own_browser(harness):
    """In Game Mode there's no desktop browser: steam://openurl/ shows the page in Steam's browser, over the game."""
    from gettowork import notices

    opened = []
    h = harness(lambda argv, ui: ui.ask("Plan?") and 0, env={"SteamDeck": "1", app.FULLSCREEN_ENV: "0"},
                opener=lambda url: opened.append(url) or True)
    h.wait_prompt("Plan?")
    h.window.report_button.invoke()
    h.pump(lambda: opened, what="the Steam request")
    assert opened == [f"steam://openurl/{notices.report_url()}"]
    assert "the Steam button's menu > Discussions" in " ".join(h.transcript().split())
    assert app.open_through_steam("javascript:alert(1)", lambda url: True) is False
    h.window._submit()


def test_on_a_steam_deck_the_question_and_input_bar_sit_above_steams_keyboard(display, monkeypatch):
    """Steam's on-screen keyboard covers the lower part of the screen (the game isn't resized): the
    question and the text being typed must be at the top, and the newest lines stay above the keyboard."""
    import tkinter

    if sys.platform == "darwin":
        monkeypatch.setattr(app.GameWindow, "set_fullscreen",
                            lambda self, on: setattr(self, "fullscreen", bool(on)))
    opened = []
    root = tkinter.Tk()
    root.winfo_screenwidth = lambda: 1280
    root.winfo_screenheight = lambda: 800
    release = threading.Event()

    def game(argv, ui):
        for n in range(60):
            ui.say(f"line {n}")
        ui.ask("What do you do?")
        return 0

    window = app.GameWindow(root, [], game_main=game, env={"SteamDeck": "1", app.FULLSCREEN_ENV: "0"},
                            remember_font_size=False, opener=lambda url: opened.append(url) or True)
    try:
        root.geometry("1280x800")
        window.start()
        deadline = time.monotonic() + 10
        while window.prompt is None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.005)
        assert window.prompt is not None
        root.update()
        height = root.winfo_height()
        assert window.input_on_top
        for widget in (window.prompt_label, window.entry, window.keyboard_button):
            top = widget.winfo_rooty() - root.winfo_rooty()
            assert top + widget.winfo_height() < height * 0.35, (widget, top)
        assert window.text.winfo_rooty() > window.entry.winfo_rooty()  # the transcript is below
        window.keyboard_button.invoke()
        root.update()
        # The transcript now ends above the keyboard's share of the screen, showing the newest line.
        bottom = window.text.winfo_rooty() - root.winfo_rooty() + window.text.winfo_height()
        assert bottom <= height * (1 - app.KEYBOARD_SHARE) + 4
        assert window.text.yview()[1] == 1.0
        window.type_answer("jump")
        root.update()
        assert not window.keyboard_space.winfo_ismapped()  # the room comes back once the answer is sent
        release.set()
    finally:
        window.bridge.close()
        window.finish()


def test_the_desktop_keeps_the_input_bar_at_the_bottom(harness):
    opened = []
    h = harness(lambda argv, ui: ui.ask("Plan?") and 0, opener=lambda url: opened.append(url) or True)
    h.wait_prompt("Plan?")
    assert not h.window.input_on_top
    assert h.window.entry.winfo_rooty() > h.window.text.winfo_rooty()
    h.window.show_keyboard()  # (no room is made on a desktop)
    assert not h.window.keyboard_space.winfo_ismapped()
    h.root.update()
    assert opened == []  # no steam:// request (without Steam it would just open a web browser)
    h.window.type_answer("walk")


def test_steam_deck_gets_a_keyboard_button(harness):
    opened = []
    h = harness(lambda argv, ui: 0, env={"SteamDeck": "1", app.FULLSCREEN_ENV: "0"},
                opener=lambda url: opened.append(url) or True)
    assert h.window.keyboard_button is not None
    h.window.keyboard_button.invoke()
    h.pump(lambda: opened, what="Steam's keyboard")
    assert opened == ["steam://open/keyboard"]
    assert harness(lambda argv, ui: 0).window.keyboard_button is None  # not on a desktop


def test_tk_callback_errors_are_logged_not_fatal(harness, isolated_home):
    h = harness(lambda argv, ui: 0)
    try:
        raise ValueError("glitch")
    except ValueError as exc:
        h.root.report_callback_exception(ValueError, exc, exc.__traceback__)
    assert "glitch" in h.window.tk_errors[0]
    assert (isolated_home / "logs" / "gui-crash.txt").exists()
    h.window.tk_errors.clear()


def test_the_real_game_quits_cleanly_when_the_window_closes_mid_prompt(harness):
    h = harness(app._default_game, ["--mock", "--no-jev"])
    h.wait_prompt("", timeout=30)  # the first question (a pause before setting off)
    assert h.close() == 0
    assert not h.window._worker.is_alive()
    assert "see you next time" in h.transcript()


def test_closing_the_window_at_the_jev_question_ends_the_game_with_no_more_model_calls(harness, monkeypatch):
    """Closing the window is not "skip Jev": nothing more may happen behind the hidden window
    (the opening story would be a whole model generation, and could race the engine's shutdown)."""
    from gettowork.backends.mock import MockBackend

    calls: list = []
    real_chat = MockBackend.chat

    def counting_chat(self, *args, **kwargs):
        calls.append(1)
        return real_chat(self, *args, **kwargs)

    monkeypatch.setattr(MockBackend, "chat", counting_chat)
    h = harness(app._default_game, ["--mock"])
    for _ in range(20):  # through the first-launch notice and the pretend-model setup
        prompt = h.wait_prompt("", timeout=30)
        if "Enable Jev" in prompt.text:
            break
        h.window.type_answer("")
        h.pump(lambda: h.window.prompt is None or h.window.prompt.id != prompt.id, what="the next question")
    else:
        raise AssertionError("never reached the Jev question")
    before = len(calls)
    assert h.close() == 0
    assert not h.window._worker.is_alive()
    assert len(calls) == before  # no opening story after the window closed
    text = h.transcript()
    assert "skipping Jev" not in text and "Good morning" not in text
    assert "see you next time" in text


def test_a_lone_surrogate_from_a_model_never_freezes_the_window(harness):
    """A model's JSON can carry "\\ud83e" (half an emoji): Tk on macOS/Linux refuses it. The
    question after it must still appear, with no errors piling up."""
    import json as _json

    got = {}

    def game(argv, ui):
        ui.narrate(_json.loads('"The goose honks \\ud83e and leaves"'))
        got["answer"] = ui.ask("What do you do?")
        return 0

    h = harness(game)
    h.wait_prompt("What do you do?")
    h.window.type_answer("wave")
    h.pump(lambda: h.window._game_done, what="the game to finish")
    assert got == {"answer": "wave"}
    assert "The goose honks " in h.widget_text() and " and leaves" in h.widget_text()
    assert "\ud83e" not in h.transcript() and "The goose honks \ufffd and leaves" in h.transcript()


def test_a_line_tk_refuses_is_redrawn_and_the_question_still_appears(harness, isolated_home):
    """Belt and braces: whatever makes a redraw fail, the prompt, buttons and the end of the game
    are still handled, the transcript is drawn again, and the crash report is written once."""
    def game(argv, ui):
        ui.say("first line")
        ui.say("the troublesome line")
        return 0 if ui.confirm("Carry on?") else 1

    h = harness(game)
    real_line_args = h.window._line_args
    failures = {"left": 1}

    def flaky_line_args(index):
        if failures["left"] and "troublesome" in h.window.buffer.line_text(index):
            failures["left"] -= 1
            raise RuntimeError("Tk refused this line")
        return real_line_args(index)

    h.window._line_args = flaky_line_args
    h.wait_prompt("Carry on?")
    assert "the troublesome line" in h.widget_text()  # redrawn after the failure
    assert [key for key, _label in h.window.choice_buttons] == ["y", "n"]
    h.window.press_choice("y")
    h.pump(lambda: h.window._game_done, what="the game to finish")
    assert len(h.window.tk_errors) == 1
    for _ in range(50):  # the same error again and again: the crash report isn't rewritten
        h.window._report_tk_error(RuntimeError, RuntimeError("Tk refused this line"), None)
    report = isolated_home / "logs" / "gui-crash.txt"
    assert report.is_file()
    stamp = report.stat().st_mtime_ns
    h.window._report_tk_error(RuntimeError, RuntimeError("Tk refused this line"), None)
    assert report.stat().st_mtime_ns == stamp
    h.window.tk_errors.clear()  # (expected here: the harness checks for unexpected ones)


def test_pasting_after_clicking_the_transcript_still_reaches_the_input_bar(harness):
    """Tk gives the read-only transcript the focus on a click (say, to bring the window forward
    after copying a key in the browser): Ctrl+V must still paste into the input bar."""
    got = {}

    def game(argv, ui):
        got["key"] = ui.secret("Paste your Jev API key and press Enter")
        return 0

    h = harness(game)
    h.wait_prompt("Paste your Jev API key")
    h.root.clipboard_clear()
    h.root.clipboard_append("jev-secret-key-123")
    h.window.text.focus_force()
    h.settle()
    h.window.text.event_generate("<Control-v>")
    h.settle()
    assert h.window.entry.get() == "jev-secret-key-123"
    assert h.window.entry.cget("show") == "\u2022"  # still masked
    h.window.text.event_generate("<Return>")
    h.pump(lambda: h.window._game_done, what="the game to finish")
    assert got == {"key": "jev-secret-key-123"}
    assert "jev-secret-key-123" not in h.transcript()


def test_ctrl_shift_v_and_right_click_paste_in_the_input_bar(harness):
    def game(argv, ui):
        ui.ask("Anything?")
        return 0

    h = harness(game)
    h.wait_prompt("Anything?")
    h.root.clipboard_clear()
    h.root.clipboard_append("pasted")
    h.window.entry.focus_force()
    h.settle()
    h.window.entry.event_generate("<Control-V>")
    h.settle()
    assert h.window.entry.get() == "pasted"
    menu = h.window._entry_menu
    assert [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)] == ["Paste", "Copy", "Cut"]
    h.window.entry.delete(0, "end")
    menu.invoke(0)
    assert h.window.entry.get() == "pasted"
    h.window.type_answer("")


def test_a_click_on_the_transcript_hands_the_focus_back_to_the_input_bar(harness):
    def game(argv, ui):
        ui.say("some text to click on")
        ui.ask("Anything?")
        return 0

    h = harness(game)
    h.wait_prompt("Anything?")
    h.window.text.focus_force()
    h.settle()
    h.window._on_transcript_click()
    h.settle()
    assert h.root.focus_get() is h.window.entry
    h.window.type_answer("")


def test_closing_mid_spinner_gives_the_process_its_own_streams_back(harness, capsys):
    """A window closed while the game waits under a spinner (the model thinking, a download):
    rich's live display must not keep stdout/stderr swapped - later messages (the self-test's
    verdict, an error) would vanish."""
    import io as _io

    original_out, original_err = sys.stdout, sys.stderr
    seen = {}

    def game(argv, ui):
        with ui.status("thinking very hard"):
            seen["stdout"], seen["stderr"] = sys.stdout, sys.stderr
            time.sleep(3)  # still waiting when the window gives up on it
        with ui.download_progress("model.gguf", 100) as advance:
            advance(10)
        return 0

    h = harness(game, close_wait_s=0.3)
    h.pump(lambda: "stdout" in seen, what="the spinner")
    assert seen["stdout"] is h.window.bridge.stream  # the spinner didn't swap in its own proxies
    assert isinstance(seen["stderr"], app._TeeStream)
    h.close()
    assert sys.stdout is original_out and sys.stderr is original_err
    print("still heard after the window closed")
    assert "still heard after the window closed" in capsys.readouterr().out
    assert not isinstance(sys.stdout, _io.StringIO) or sys.stdout is original_out


# ---------------------------------------------------------------------------
# The self-test (what CI runs on every build)
# ---------------------------------------------------------------------------


def test_selftest_plays_the_real_game_to_the_end(display, tmp_path, monkeypatch):
    out = tmp_path / "selftest.txt"
    monkeypatch.setenv(app.SELFTEST_OUT_ENV, str(out))
    assert app.run_gui(["--mock", "--no-jev"], selftest=True) == 0
    transcript = out.read_text(encoding="utf-8")
    assert "YOU GOT TO WORK" in transcript
    assert "How do you plan to get to work? > " in transcript


def test_selftest_defaults_to_the_pretend_model(display, tmp_path, monkeypatch):
    seen = []

    def game(argv, ui):
        seen.append(argv)
        ui.console.print("YOU GOT TO WORK!")
        return 0

    assert app.run_gui(["--gui-selftest"], game_main=game) == 0
    assert seen == [["--mock", "--no-jev", "--export-dir", str(app.default_export_dir())]]


def test_selftest_fails_when_the_game_never_gets_to_work(display, capsys):
    assert app.run_gui([], selftest=True, game_main=lambda argv, ui: 0) == 1
    assert "failed" in capsys.readouterr().err


def test_selftest_fails_when_the_game_errors(display):
    def game(argv, ui):
        ui.console.print("YOU GOT TO WORK!")
        return 1

    assert app.run_gui([], selftest=True, game_main=game) == 1


def test_selftest_times_out_with_code_2(display, tmp_path, monkeypatch):
    import tkinter

    stop = threading.Event()
    out = tmp_path / "timeout.txt"
    monkeypatch.setenv(app.SELFTEST_OUT_ENV, str(out))

    def game(argv, ui):
        ui.say("stuck")
        stop.wait(30)
        return 0

    root = tkinter.Tk()
    window = app.GameWindow(root, [], game_main=game, selftest=True, selftest_timeout_s=1, close_wait_s=0.3,
                            env=dict(os.environ))
    started = time.monotonic()
    assert window.run() == 2
    assert time.monotonic() - started < 5
    assert "stuck" in out.read_text(encoding="utf-8")
    stop.set()


def test_a_finished_window_keeps_no_tk_object(harness):
    """After the window is done, the GameWindow (kept alive by the game thread and its UI) must hold no widget,
    font or interpreter: Python's cyclic garbage collector may free it on any thread, and Tcl aborts the
    whole program ("Tcl_AsyncDelete: async handler deleted by the wrong thread") when that isn't this one."""
    import tkinter

    def game(argv, ui):
        ui.ask("Name?")
        return 0

    h = harness(game)
    h.window.extra_widget_added_later = [tkinter.Frame(h.root)]  # (a future attribute nobody listed)
    h.wait_prompt("Name")
    before = {name for name, value in vars(h.window).items() if app.holds_tk_object(value)}
    assert {"keyboard_space", "_transcript_frame", "text", "root"} <= before
    h.close()
    held = {name: type(value).__name__ for name, value in vars(h.window).items() if app.holds_tk_object(value)}
    assert held == {}
    assert h.window.keyboard_space is None and h.window._transcript_frame is None
    assert h.window.extra_widget_added_later == []


def test_holds_tk_object_spots_widgets_in_containers_and_bound_methods(display):
    import tkinter
    import tkinter.font

    root = tkinter.Tk()
    try:
        frame = tkinter.Frame(root)
        assert app.holds_tk_object(frame) and app.holds_tk_object([1, {"x": frame}])
        assert app.holds_tk_object(frame.focus_set) and app.holds_tk_object(root.tk)
        assert app.holds_tk_object(tkinter.font.Font(root=root, family="Courier", size=10))
        assert app.holds_tk_object(tkinter.StringVar(root))
        assert not app.holds_tk_object([1, "two", {"three": 3}]) and not app.holds_tk_object(tkinter)
    finally:
        root.destroy()
        del root, frame
        gc.collect()


def test_selftest_gives_up_on_a_question_it_cannot_answer(display):
    import tkinter

    def game(argv, ui):
        while True:
            ui.choose("Pick", [("a", "A"), ("b", "B")])  # no default: "" is never accepted

    root = tkinter.Tk()
    window = app.GameWindow(root, [], game_main=game, selftest=True, selftest_timeout_s=30, close_wait_s=2,
                            env={})
    window.selftest.max_answers = 5
    assert window.run() == 1
    assert "gave up" in window.selftest_note
