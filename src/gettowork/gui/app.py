"""The game's own window: what players see when they launch Get To Work from Steam
or double-click it.

The game itself doesn't change at all. ``cli.main`` - the same code the
terminal version runs - plays in a *worker thread*, printing through a rich
Console into a :class:`~gettowork.gui.bridge.GuiBridge`. This module owns the
Tk window on the *main thread* (macOS insists on that) and, every 30 ms,
collects what the game printed, feeds it to a
:class:`~gettowork.gui.terminal.TerminalBuffer` and redraws the lines that
changed. Questions appear in an input bar at the bottom; menus and yes/no
questions also get big buttons, so a mouse, a touch screen or a Steam Deck
works as well as a keyboard.

Closing the window ends the game politely: the waiting question gets
``EOFError`` (just like the end of piped input), the game says goodbye and
stops the local model's engine, and :func:`run_gui` returns so the
interpreter's normal shutdown (``atexit``) runs.

Tkinter is imported only inside the functions that need it, so importing
this module works on a Python built without Tk (the terminal version never
needs it).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .bridge import CHOICES, FINISHED, OUTPUT, PROMPT, GuiBridge, Prompt, clean_text
from .terminal import DEFAULT_MAX_LINES, DEFAULT_STYLE, Style, TerminalBuffer, xterm_color

__all__ = [
    "run_gui",
    "GameWindow",
    "SelftestPlayer",
    "WINDOW_TITLE",
    "EXIT_HINT",
    "FONT_CANDIDATES",
    "pick_font_family",
    "glyph_fallbacks",
    "emoji_stand_in",
    "emoji_safe_text",
    "uses_core_x11_fonts",
    "plain_prompt",
    "short_label",
    "style_colors",
    "wants_fullscreen",
    "FULLSCREEN_MIN_COLUMNS",
    "FULLSCREEN_MIN_ROWS",
    "on_steam_deck",
    "button_labels",
    "labels_look_alike",
    "open_steam_keyboard",
    "default_export_dir",
    "write_crash_report",
]

GameMain = Callable[[List[str], Any], int]  # (argv, ui) -> exit code, like cli.main(argv, ui=ui)

WINDOW_TITLE = "Get To Work"
EXIT_HINT = "Press Enter or close the window to exit"
REPORT_BUTTON = "Report a problem"  # (the same words notices.REPORT_HOW tells players to look for)
SECRET_HINT = "(what you type stays hidden)"
HIDDEN_ANSWER = "(hidden)"  # echoed in place of an API key, however it was typed

# Monospace fonts in order of preference: Windows, macOS, then Linux / Steam Deck.
FONT_CANDIDATES = (
    "Cascadia Mono", "Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono",
    "Noto Sans Mono", "Liberation Mono", "Courier New",
)
DEFAULT_FONT_SIZE = 12  # points
FULLSCREEN_FONT_SIZE = 16  # Steam Deck / Big Picture: read from arm's length
MIN_FONT_SIZE, MAX_FONT_SIZE = 7, 36
FONT_SIZE_SETTING = "gui_font_size"  # remembered in Settings.extra

MIN_COLUMNS, MIN_ROWS = 100, 32  # the default window shows at least this much text
# Full screen (Steam Deck, Big Picture) sizes the text to the screen instead: the big font
# stays as long as this much fits (the game is checked at 80 columns).
FULLSCREEN_MIN_COLUMNS, FULLSCREEN_MIN_ROWS = 80, 24
MIN_CONSOLE_COLUMNS = 20
POLL_MS = 30  # how often the window collects what the game printed
CLOSE_WAIT_S = 8.0  # after the window closes, how long the game gets to say goodbye
SELFTEST_TIMEOUT_S = 120.0
SELFTEST_TIMEOUT_ENV = "GETTOWORK_SELFTEST_TIMEOUT"
SELFTEST_OUT_ENV = "GETTOWORK_SELFTEST_OUT"
SELFTEST_SUCCESS_TEXT = "YOU GOT TO WORK"
SELFTEST_ANSWER_DELAY_MS = 20
FULLSCREEN_ENV = "GETTOWORK_FULLSCREEN"
KEYBOARD_SHARE = 0.45  # how much of a Steam Deck's screen its on-screen keyboard covers (from the bottom)
CRASH_FILE = "gui-crash.txt"
MAX_CRASH_REPORTS = 5  # different window errors written to the crash report per session (the rest are only kept in memory)

EXIT_OK, EXIT_ERROR, EXIT_TIMEOUT = 0, 1, 2
_EDGE_PIXELS = 2  # never draw the last column right against the edge
_SPARE_PIXELS = 6  # the default window gets these few extra pixels (covers _EDGE_PIXELS)

# -- the dark theme -----------------------------------------------------------------------

BACKGROUND = "#16181d"
FOREGROUND = "#d7dae0"
PANEL = "#1e2129"  # the input area under the transcript
ENTRY_BACKGROUND = "#0f1115"
BORDER = "#343a46"
ACCENT = "#d7a8f0"  # the prompt text (the game's banner is magenta)
SELECTION = "#3e4451"
LINK = "#61afef"
BUTTON = "#2c3340"
BUTTON_ACTIVE = "#3d4657"
BUTTON_PRESSED = "#4b5569"
BUTTON_TEXT = "#eef0f4"
DISABLED_TEXT = "#6b7280"

# The 16 standard terminal colours, tuned to read well on the dark background.
PALETTE_16 = (
    "#3b4048", "#e06c75", "#98c379", "#e5c07b", "#61afef", "#c678dd", "#56b6c2", "#c8ccd4",
    "#7f848e", "#ff7b86", "#b5e890", "#ffd68a", "#7cc4ff", "#de95f0", "#6fd6e3", "#ffffff",
)


# ---------------------------------------------------------------------------
# Small helpers (no Tk needed, so they're easy to test)
# ---------------------------------------------------------------------------


def resolve_color(color: Any, default: str) -> str:
    """A TerminalBuffer colour (None, palette index or "#rrggbb") as "#rrggbb"."""
    if color is None:
        return default
    if isinstance(color, int):
        if 0 <= color < 16:
            return PALETTE_16[color]
        if 16 <= color <= 255:
            return xterm_color(color)
        return default
    if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
        return color.lower()
    return default


def blend(color: str, other: str, amount: float) -> str:
    """Mix two "#rrggbb" colours (``amount`` of ``other``): how "dim" text is drawn."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(x + (y - x) * amount) for x, y in zip(a, b)]
    return "#" + "".join(f"{v:02x}" for v in mixed)


def style_colors(style: Style) -> Tuple[str, Optional[str]]:
    """(foreground, background or None) for a style: reverse and dim applied, links coloured."""
    fg_default = LINK if style.link and style.fg is None else FOREGROUND
    fg = resolve_color(style.fg, fg_default)
    bg = resolve_color(style.bg, BACKGROUND) if style.bg is not None else None
    if style.reverse:
        fg, bg = (bg or BACKGROUND), fg
    if style.dim:
        fg = blend(fg, bg or BACKGROUND, 0.45)
    return fg, bg


def plain_prompt(prompt: str) -> str:
    """A prompt as the game wrote it ("[bold]Your pick[/bold] > ") as plain text ("Your pick")."""
    text = str(prompt or "")
    try:
        from rich.text import Text

        text = Text.from_markup(text).plain
    except Exception:
        pass
    text = " ".join(text.split())  # one line
    while text.endswith((">", ":")) and not text.endswith(("->", "=>")):
        text = text[:-1].rstrip()
    return text


def short_label(label: str, limit: int = 42) -> str:
    """A menu option's label, short enough for a button."""
    text = plain_prompt(label) if label else ""
    if len(text) > limit:
        for separator in (" (", " - ", ": ", ", "):
            head = text.split(separator, 1)[0]
            if 3 <= len(head) <= limit:
                return head
        text = text[:limit - 1].rstrip() + "…"
    return text


# Keys that aren't typing (see GameWindow._on_transcript_key): modifiers on their own, lock keys,
# function keys, Escape and friends.
_NOT_TYPING_KEYSYM_RE = re.compile(
    r"^(?:Shift|Control|Alt|Meta|Super|Hyper)_[LR]$|^Win_[LR]$|^F\d{1,2}$|^XF86"
    r"|^(?:Caps_Lock|Shift_Lock|Num_Lock|Scroll_Lock|ISO_Level3_Shift|ISO_Level5_Shift|Mode_switch|Multi_key"
    r"|Escape|Print|Pause|Break|Menu|App|Insert|Cancel)$"
)


def holds_tk_object(value: Any, depth: int = 2) -> bool:
    """Is `value` a Tk object (widget, font, image, variable, the interpreter itself), a widget's bound
    method, or a list/tuple/dict/set holding one? (Only these keep the Tcl interpreter alive.)"""
    tkinter = sys.modules.get("tkinter")
    if tkinter is None:
        return False
    font = sys.modules.get("tkinter.font")
    kinds: tuple = (tkinter.Misc, tkinter.Image, tkinter.Variable) + ((font.Font,) if font is not None else ())
    if isinstance(value, kinds) or type(value).__name__ == "tkapp":
        return True
    owner = getattr(value, "__self__", None)
    if owner is not None and owner is not value and isinstance(owner, kinds):
        return True
    if depth > 0 and isinstance(value, (list, tuple, set, dict)):
        items = [*value.keys(), *value.values()] if isinstance(value, dict) else list(value)
        return any(holds_tk_object(item, depth - 1) for item in items[:10_000])
    return False


def labels_look_alike(a: str, b: str) -> bool:
    """Would two buttons be mistaken for each other? The same words - or one's words start the other's
    ("Yes" next to "Yes, play")."""
    wa, wb = re.findall(r"[\w']+", a.casefold()), re.findall(r"[\w']+", b.casefold())
    if not wa or not wb:
        return a.strip().casefold() == b.strip().casefold()
    shorter, longer = sorted((wa, wb), key=len)
    return longer[:len(shorter)] == shorter


def button_labels(options: Sequence[Tuple[str, str]], limit: int = 42) -> list[str]:
    """The button texts for one menu: each option's :func:`short_label` - never two alike.

    Where two short labels would be mistaken for each other ("Yes" next to
    "Yes, play"), those options get their full label (shortened to fit), and
    failing that their key as well - so a button never hides which choice it
    is.
    """
    labels = [short_label(label, limit) or key or "Continue" for key, label in options]

    def clashes() -> set[int]:
        return {i for i in range(len(labels)) for j in range(len(labels))
                if i != j and labels_look_alike(labels[i], labels[j])}

    for index in clashes():
        text = plain_prompt(options[index][1]) or options[index][0]
        labels[index] = text if len(text) <= limit + 20 else text[:limit + 19].rstrip() + "…"
    for index in clashes():
        labels[index] = f"{labels[index]} ({options[index][0]})"
    return labels


# One-cell look-alikes for characters a font may not have. Tk builds without
# font fallback (no Xft on some Linux systems) draw a missing character as
# "\u2500" - six cells that wreck every panel - so any of these the chosen font
# lacks is swapped for its look-alike when drawn. Always exactly one character,
# so tables stay aligned.
_PUNCTUATION_LOOKALIKES = {
    "\u2014": "-", "\u2013": "-", "\u2192": ">", "\u2190": "<", "\u2248": "~", "\u00d7": "x",
    "\u2026": ".", "\u00b7": ".", "\u2022": "*", "\u2713": "v", "\u2714": "v", "\u2605": "*",
    "\u2265": ">", "\u2264": "<", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u25cf": "*", "\u25b8": ">", "\u25b6": ">", "\u2139": "i", "\u2717": "x", "\u2718": "x",
    "\ufffd": "?",
}


def _box_lookalike(ch: str) -> str:
    """"─" -> "-", "│" -> "|", corners and junctions -> "+", blocks -> "#"."""
    import unicodedata

    name = unicodedata.name(ch, "")
    code = ord(ch)
    if 0x2580 <= code <= 0x259F:
        return "#"
    if 0x2800 <= code <= 0x28FF:
        return "*"  # braille: rich's spinner frames
    horizontal = "HORIZONTAL" in name or name.endswith((" LEFT", " RIGHT"))
    vertical = "VERTICAL" in name or name.endswith((" UP", " DOWN"))
    if horizontal and not vertical and " AND " not in name:
        return "-"
    if vertical and not horizontal and " AND " not in name:
        return "|"
    if "DIAGONAL" in name:
        return "/" if "UPPER RIGHT TO LOWER LEFT" in name else ("\\" if "UPPER LEFT" in name else "X")
    return "+"


# Groups of characters that fonts have (or lack) together, with the ones rich
# uses most as probes: if any probe is missing, the whole group is swapped.
_GLYPH_GROUPS = (
    ("─│╭╮╰╯━┃┏┓┗┛┡┩╇┳┻╸╺", range(0x2500, 0x2580)),  # box drawing
    ("█▀▄▌▐░▒▓", range(0x2580, 0x25A0)),  # block elements
    ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", range(0x2800, 0x2900)),  # braille (spinner frames)
)


def glyph_fallbacks(measure: Callable[[str], int], *, force: bool = False) -> Dict[int, str]:
    """A ``str.translate`` table for the characters a font can't draw in one cell.

    ``measure(text)`` is the font's width in pixels (``tkinter.font.Font.measure``).
    A monospace font draws every character it has at the width of "0".
    ``force=True`` swaps them all (for Tk builds that can't borrow glyphs from
    other fonts, where even a "present" character may be drawn blank).
    """
    try:
        cell = measure("0")
    except Exception:
        return {}

    def missing(ch: str) -> bool:
        if force:
            return True
        try:
            return measure(ch) != cell
        except Exception:
            return True

    table: Dict[int, str] = {}
    for probes, codes in _GLYPH_GROUPS:
        if any(missing(ch) for ch in probes):
            table.update({code: _box_lookalike(chr(code)) for code in codes})
    for ch, plain in _PUNCTUATION_LOOKALIKES.items():
        if missing(ch):
            table[ord(ch)] = plain
    return table


# Emoji in the X11 (Linux) window. Tk there can't draw colour emoji: with the
# colour emoji font most Linux systems have (SteamOS included), a libXft older
# than 2.3.5 (Ubuntu 22.04's, for one) even ends the program with an X error
# ("BadLength"), and Tk builds without Xft draw them as "�". So on X11 the
# window draws a plain stand-in instead, exactly as many cells wide as rich laid
# the emoji out, so the columns stay put. (The model's story may contain emoji;
# the game's own menus use one or two.)
_EMOJI_LOOKALIKES = {
    "\U0001f9e0": "+", "⚡": "!", "✅": "v", "✔": "v", "❌": "x", "❎": "x",
    "⭐": "*", "\U0001f31f": "*", "❓": "?", "❗": "!", "\U0001f600": ":", "\U0001f642": ":",
}
_EMOJI_MODIFIERS = frozenset("️︎‍⃣")  # emoji presentation, zero-width joiner, keycap


def emoji_stand_in(cluster: str, width: int) -> Optional[str]:
    """A plain stand-in for an emoji the X11 window must not draw, ``width`` characters long; None keeps ``cluster``.

    ``cluster`` is one character as rich measured it (an emoji may be several
    code points, e.g. a zero-width-joiner family), ``width`` its cells.
    """
    if not cluster or (len(cluster) == 1 and ord(cluster) < 0x2300):
        return None  # the common case: letters, punctuation, arrows
    first = cluster[0]
    risky = any(ord(ch) > 0xFFFF or ch in _EMOJI_MODIFIERS for ch in cluster)
    if not risky and len(cluster) == 1 and width >= 2 and 0x2300 <= ord(first) <= 0x2BFF:
        risky = True  # a symbol shown as an emoji (U+26A1 "high voltage" and friends)
    if not risky:
        return None
    base = next((_EMOJI_LOOKALIKES[ch] for ch in cluster if ch in _EMOJI_LOOKALIKES), None)
    if base is None:
        # "⚠️" (a plain symbol asked to look like an emoji) keeps its plain symbol.
        plain_symbol = ord(first) <= 0xFFFF and first not in _EMOJI_MODIFIERS and not (
            0x2300 <= ord(first) <= 0x2BFF and cluster == first)
        base = first if plain_symbol else "*"
    width = max(0, int(width))
    return (base + " " * width)[:width]


def emoji_safe_text(text: str) -> str:
    """``text`` with its emoji swapped for plain stand-ins (for labels and buttons on X11)."""
    if not text or text.isascii():
        return text
    from .terminal import _clusters

    out = []
    for cluster, width in _clusters(text):
        replacement = emoji_stand_in(cluster, width)
        out.append(cluster if replacement is None else replacement[:1])  # (a label keeps no columns: no padding)
    return "".join(out)


def uses_core_x11_fonts(windowing_system: str, families: Iterable[str]) -> bool:
    """Is this a Tk built without Xft (old-style X11 fonts, no fallback between fonts)?

    Such builds list only a handful of all-lowercase font families ("fixed",
    "courier", "helvetica"...) and draw characters their font lacks as
    "\\u2500". Modern Tk builds (Windows, macOS, Linux with Xft) list real
    font names such as "DejaVu Sans Mono".
    """
    names = [name for name in families if name]
    return windowing_system == "x11" and bool(names) and all(name == name.lower() for name in names)


def pick_font_family(available: Iterable[str], candidates: Sequence[str] = FONT_CANDIDATES) -> Optional[str]:
    """The first of ``candidates`` that is installed (case-insensitive), or None."""
    installed = {name.lower(): name for name in available}
    for name in candidates:
        if name.lower() in installed:
            return installed[name.lower()]
    return None


def wants_fullscreen(env: Mapping[str, str]) -> bool:
    """Start full screen? Yes on a Steam Deck / in Steam's Big Picture or Game Mode.

    ``GETTOWORK_FULLSCREEN=1`` / ``=0`` decides it for everyone else.
    """
    forced = env.get(FULLSCREEN_ENV, "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if forced in ("0", "false", "no", "off"):
        return False
    return (env.get("SteamDeck") == "1" or env.get("SteamGamepadUI") == "1"
            or bool(env.get("GAMESCOPE_WAYLAND_DISPLAY")))


def on_steam_deck(env: Mapping[str, str]) -> bool:
    """Running in Steam's controller-first mode (Steam Deck, Big Picture), where there may be no keyboard?"""
    return env.get("SteamDeck") == "1" or env.get("SteamGamepadUI") == "1"


STEAM_KEYBOARD_URL = "steam://open/keyboard"  # asks a running Steam client for its on-screen keyboard


def osc8_link(url: str) -> str:
    """`url` as a clickable link in the transcript (an OSC 8 hyperlink, which the window's terminal view knows)."""
    return f"\x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\"


def open_through_steam(url: str, opener: Optional[Callable[[str], Any]] = None) -> bool:
    """Ask the running Steam client to show a web link in its own browser (over the game, on a Steam Deck in
    Game Mode). Only http(s) links. True if the request was sent."""
    from .bridge import is_openable_url

    if not is_openable_url(url):
        return False
    if opener is None:
        import webbrowser

        opener = webbrowser.open
    try:
        return bool(opener(f"steam://openurl/{url}"))
    except Exception:
        return False


def open_steam_keyboard(opener: Optional[Callable[[str], Any]] = None) -> bool:
    """Show Steam's on-screen keyboard (STEAM + X does the same). True if the request was sent."""
    if opener is None:
        import webbrowser

        opener = webbrowser.open
    try:
        return bool(opener(STEAM_KEYBOARD_URL))
    except Exception:
        return False


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def terminal_attached() -> bool:
    """Was the game started from a terminal (so it can fall back to playing there)?"""
    try:
        return bool(sys.stdin and sys.stdout and sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def crash_log_path() -> Path:
    from ..crashlog import crash_log_path as _path

    return _path(CRASH_FILE)


def write_crash_report(context: str, exc: Optional[BaseException] = None) -> Optional[Path]:
    """Write what went wrong to ``<config dir>/logs/gui-crash.txt``; returns the path (never raises).

    Handy for a bug report when there was no window to show the error in.
    (Errors inside the game itself go to ``logs/crash.txt`` - see
    :mod:`gettowork.crashlog`.)
    """
    try:
        from ..crashlog import write_crash_report as _write

        return _write(context, exc, file_name=CRASH_FILE)
    except Exception:
        return None


VERIFY_HINT = ("Playing on Steam? Right-click Get To Work in your Library > Properties > Installed Files > "
               "Verify integrity of game files, then start it again.")


def window_failed_message(reason: str, path: Optional[Path]) -> str:
    """What the player reads when the game window can't open."""
    lines = [f"Get To Work couldn't open its window ({reason}).", "", VERIFY_HINT]
    if path:
        lines += ["", f"The details are saved in {path} - handy for a bug report."]
    return "\n".join(lines)


def show_error_dialog(title: str, message: str, *, system: Optional[str] = None,
                      runner: Optional[Callable[..., Any]] = None,
                      env: Optional[Mapping[str, str]] = None) -> bool:
    """Show a message box without Tk (which just failed); True if one was shown. Never raises.

    Windows: the system's own ``MessageBoxW``. macOS: ``osascript``'s alert
    (the text is passed as an argument, never inside the script). Linux:
    ``zenity``, ``kdialog`` or ``xmessage``, whichever is installed, when
    there's a display to show it on. A double-clicked or Steam-started game
    has no console, so without this a failed start would look like nothing
    happened at all.
    """
    import shutil
    import subprocess

    system = system or sys.platform
    env = os.environ if env is None else env
    run = runner or subprocess.run
    try:
        if system.startswith("win"):
            if runner is not None:
                return bool(runner(["MessageBoxW", title, message]))
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10 | 0x10000 | 0x40000)  # error, foreground, topmost
            return True
        if system == "darwin":
            script = ["-e", "on run argv", "-e",
                      "display alert (item 1 of argv) message (item 2 of argv) as critical", "-e", "end run"]
            run(["osascript", *script, title, message], check=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
            return False  # nowhere to show a window
        for program, args in (("zenity", ["--error", "--no-markup", "--title", title, "--text", message]),
                              ("kdialog", ["--title", title, "--error", message]),
                              ("xmessage", ["-center", f"{title}\n\n{message}"])):
            found = shutil.which(program)
            if found:
                result = run([found, *args], check=False, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if getattr(result, "returncode", 0) in (0, 1):  # (1: closed with the window's X)
                    return True
        return False
    except Exception:
        return False


def _say_to_stderr(text: str) -> None:
    stream = sys.stderr or sys.__stderr__
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def load_font_size() -> Optional[int]:
    """The font size the player chose last time (Ctrl+= / Ctrl+-), if any."""
    try:
        from ..config import Settings

        value = Settings.load().extra.get(FONT_SIZE_SETTING)
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and MIN_FONT_SIZE <= value <= MAX_FONT_SIZE:
        return value
    return None


def save_font_size(size: int) -> bool:
    """Remember the font size in the settings file. Never raises; True if saved.

    Done once, when the window closes (the game has finished saving its own
    settings by then). A settings file that exists but can't be read is left
    alone rather than overwritten.
    """
    try:
        import json

        from ..config import Settings

        settings = Settings.load()
        path = settings.path
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return False
        if settings.extra.get(FONT_SIZE_SETTING) == size:
            return True
        settings.extra[FONT_SIZE_SETTING] = int(size)
        settings.save()
        return True
    except Exception:
        return False


class SelftestPlayer:
    """Answers the game's questions during ``--gui-selftest`` (used by CI).

    Presses Enter at "Press Enter" pauses and menus (their default), types a
    plan whenever the game asks for one, and answers "n" to every yes/no
    question (no reasoning review, no transcript, no second game). Answers
    are chosen by *what* is asked rather than by position, so a new pause or
    notice doesn't derail the test. Gives up (returns None) after
    ``max_answers`` answers, so a question it can't handle fails the test
    instead of looping forever.
    """

    PLANS = (
        "I ride my bicycle very fast, ringing the bell the whole way",
        "I bribe the geese with a basket of warm bread rolls",
        "I build a ramp out of cereal boxes and jump clean over the problem",
        "I recite the office safety manual so loudly that the obstacle gives up",
        "I disguise myself as the manager and stroll straight past",
        "I sing the company anthem while riding the escalator backwards",
        "I tip my hat, bow politely and dance past with great confidence",
    )

    def __init__(self, max_answers: int = 120) -> None:
        self.max_answers = max_answers
        self.answers: List[Tuple[str, str]] = []  # (prompt, answer), for the log
        self._plans_given = 0

    @property
    def plans_given(self) -> int:
        return self._plans_given

    def answer(self, prompt: str, *, secret: bool = False,
               choices: Sequence[Tuple[str, str]] = ()) -> Optional[str]:
        if len(self.answers) >= self.max_answers:
            return None
        text = plain_prompt(prompt).lower()
        keys = tuple(key for key, _label in choices)
        if secret:
            reply = ""
        elif keys == ("y", "n") or "[y/n]" in text:
            reply = "n"
        elif keys == ("",) or "press enter" in text:
            reply = ""
        elif "plan" in text or "what do you do" in text:
            reply = self.PLANS[self._plans_given % len(self.PLANS)]
            self._plans_given += 1
        elif choices:
            reply = ""  # a menu: its default
        else:
            reply = "n"
        self.answers.append((text, reply))
        return reply


def _import_tk() -> Tuple[Any, Any, Any]:
    import tkinter
    from tkinter import font as tkfont
    from tkinter import ttk

    return tkinter, tkfont, ttk


def _prepare_windows_process() -> None:
    """Windows: sharp text on high-DPI screens, and our own taskbar button/icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GetToWork.Game")  # type: ignore[attr-defined]
    except Exception:
        pass


def _icon_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "icon.png"


def _default_game(argv: List[str], ui: Any) -> int:
    from .. import cli  # imported here: loading the game can take a moment, and the window is already up

    return cli.main(argv, ui=ui)


class _TeeStream:
    """sys.stderr while the window is open: into the window, and to the old stderr if there was one."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary, self._secondary = primary, secondary
        self.encoding = "utf-8"

    def write(self, text: str) -> int:
        for stream in (self._primary, self._secondary):
            if stream is None:
                continue
            try:
                stream.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        if self._secondary is not None:
            try:
                self._secondary.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


class GameWindow:
    """The Tk window plus the worker thread that plays the game.

    ``run()`` shows it and returns the exit code when it closes. Tests call
    ``start()``, pump Tk events themselves, and ``finish()``.
    """

    def __init__(
        self,
        root: Any,
        argv: Sequence[str] = (),
        *,
        game_main: Optional[GameMain] = None,
        selftest: bool = False,
        selftest_timeout_s: Optional[float] = None,
        opener: Optional[Callable[[str], Any]] = None,
        font_size: Optional[int] = None,
        remember_font_size: bool = True,
        env: Optional[Mapping[str, str]] = None,
        poll_ms: int = POLL_MS,
        close_wait_s: float = CLOSE_WAIT_S,
        scrollback: int = DEFAULT_MAX_LINES,
    ) -> None:
        from rich.console import Console

        from ..ui import UI

        self.tk, self.tkfont, self.ttk = _import_tk()
        self.root = root
        self.argv = list(argv)
        self._game_main: GameMain = game_main or _default_game
        self._keyboard_opener = opener  # (tests pass a fake browser; normally the real one)
        self._env = os.environ if env is None else env
        self._poll_ms = max(5, int(poll_ms))
        self._close_wait_s = float(close_wait_s)
        self.fullscreen = wants_fullscreen(self._env)

        self.bridge = GuiBridge(columns=MIN_COLUMNS, rows=MIN_ROWS, opener=opener)
        self.buffer = TerminalBuffer(rows=MIN_ROWS, max_lines=scrollback)
        self.tk_errors: List[str] = []  # errors inside Tk callbacks (logged, never fatal)
        self._reported_errors: set = set()  # the kinds of Tk error already written to the crash report

        # State (touched only on the Tk thread, except where noted).
        self._prompt: Optional[Prompt] = None
        self._choices: Tuple[Tuple[str, str], ...] = ()
        self._choice_buttons: List[Any] = []
        self._choice_rows: List[Any] = []
        self._history: List[str] = []
        self._history_pos = 0
        self._secret_digests: set = set()  # SHA-256 of answers given at key questions (never the keys)
        self._style_tags: Dict[Style, str] = {}
        self._link_tags: Dict[str, str] = {}
        self._rendered = 1  # buffer lines shown in the Text widget (it starts with one empty line)
        self._seen_trimmed = 0
        self._follow = True  # keep scrolled to the bottom
        self._at_bottom = True  # is the transcript showing its last line? (kept up to date by _on_yscroll)
        self._size = (MIN_COLUMNS, MIN_ROWS)
        self._resize_job: Optional[str] = None
        self._layout_job: Optional[str] = None
        self._layout_width = 0
        self._zoomed = False  # did the player change the font size (so it's worth remembering)?
        self._poll_job: Optional[str] = None
        self._worker: Optional[threading.Thread] = None
        self._game_code: Optional[int] = None  # set when the game finished
        self._game_done = False
        self._closing = False
        self._closed_early = False  # the player closed the window before the game ended
        self._close_deadline = 0.0
        self._quit_requested = False
        self._finished = False
        self._signal_close = False  # set by a signal handler, acted on by _poll
        self._saved_streams: Optional[Tuple[Any, Any]] = None
        self._saved_signals: List[Tuple[int, Any]] = []
        self._started = False

        self.selftest = SelftestPlayer() if selftest else None
        self.selftest_result: Optional[int] = None
        self.selftest_note = ""
        if selftest_timeout_s is None:
            try:
                selftest_timeout_s = float(self._env.get(SELFTEST_TIMEOUT_ENV, "") or SELFTEST_TIMEOUT_S)
            except ValueError:
                selftest_timeout_s = SELFTEST_TIMEOUT_S
        self._selftest_timeout_s = max(1.0, selftest_timeout_s)
        self._remember_font_size = remember_font_size and not selftest

        root.report_callback_exception = self._report_tk_error
        root.title(WINDOW_TITLE)
        root.configure(background=BACKGROUND)
        self._saved_font_size = None if font_size is not None else load_font_size()
        self._build_fonts(font_size or self._saved_font_size or
                          (FULLSCREEN_FONT_SIZE if self.fullscreen else DEFAULT_FONT_SIZE))
        self._build_styles()
        self._build_widgets()
        self._bind_keys()
        self._set_icon()
        self._place_window(fit_font=font_size is None and self._saved_font_size is None)

        cols, rows = self._size
        self.console = Console(
            file=self.bridge.stream, force_terminal=True, force_interactive=True,
            color_system="truecolor", width=cols, height=rows, legacy_windows=False, soft_wrap=False,
        )
        self.ui = UI(
            console=self.console,
            input_fn=self._ask_line,
            secret_fn=self._ask_secret,
            open_url_fn=self.bridge.open_url,
            choices_fn=self.bridge.show_choices,
            hides_input=True,
            window=True,  # no command line here: hints talk about the window, not options to type
            pauses=True,  # a person is reading this window: let long stretches of text be read
        )

    # -- building the window -------------------------------------------------------------------

    def _build_fonts(self, size: int) -> None:
        tkfont = self.tkfont
        self.font_size = min(max(int(size), MIN_FONT_SIZE), MAX_FONT_SIZE)
        family = None
        try:
            self._x11 = str(self.root.tk.call("tk", "windowingsystem")) == "x11"
        except Exception:
            self._x11 = sys.platform.startswith("linux")
        try:
            families = tkfont.families(self.root)
            family = pick_font_family(families)
            self._core_fonts = uses_core_x11_fonts("x11" if self._x11 else "", families)
        except Exception:
            self._core_fonts = False
        if family:
            self.font = tkfont.Font(self.root, family=family, size=self.font_size)
        else:
            self.font = tkfont.nametofont("TkFixedFont", root=self.root).copy()
            self.font.configure(size=self.font_size)
        self.font_family = self.font.actual("family")
        self.font_bold = self.font.copy()
        self.font_bold.configure(weight="bold")
        self.font_italic = self.font.copy()
        self.font_italic.configure(slant="italic")
        self.font_bold_italic = self.font.copy()
        self.font_bold_italic.configure(weight="bold", slant="italic")
        self.ui_font = tkfont.nametofont("TkDefaultFont", root=self.root).copy()
        self.ui_font.configure(size=self.font_size)
        self.ui_font_bold = self.ui_font.copy()
        self.ui_font_bold.configure(weight="bold")
        self._fonts = {
            (False, False): self.font, (True, False): self.font_bold,
            (False, True): self.font_italic, (True, True): self.font_bold_italic,
        }
        self._check_font_variants()

    def _check_font_variants(self) -> None:
        """Use only font variants as wide as the regular one, so columns stay aligned.

        Good monospace fonts draw bold and italic at the same width; some
        fallback fonts don't. Then bold is shown as brighter text instead.
        """
        self._glyph_fallback = glyph_fallbacks(self.font.measure, force=self._core_fonts)
        cell = self.font.measure("0")
        variants = {(True, False): self.font_bold, (False, True): self.font_italic,
                    (True, True): self.font_bold_italic}
        self._fonts = {(False, False): self.font}
        for key, font in variants.items():
            self._fonts[key] = font if font.measure("0") == cell else self.font
        self._bold_ok = self._fonts[(True, False)] is not self.font

    def _build_styles(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")  # the one built-in theme that takes custom colours everywhere
        except Exception:
            pass
        pad_x, pad_y = (18, 12) if self.fullscreen else (14, 8)
        style.configure(
            "Choice.TButton", font=self.ui_font, padding=(pad_x, pad_y), background=BUTTON,
            foreground=BUTTON_TEXT, bordercolor=BORDER, lightcolor=BUTTON, darkcolor=BUTTON,
            focuscolor=ACCENT, relief="flat",
        )
        style.map(
            "Choice.TButton",
            background=[("disabled", PANEL), ("pressed", BUTTON_PRESSED), ("active", BUTTON_ACTIVE)],
            foreground=[("disabled", DISABLED_TEXT)],
            lightcolor=[("pressed", BUTTON_PRESSED), ("active", BUTTON_ACTIVE)],
            darkcolor=[("pressed", BUTTON_PRESSED), ("active", BUTTON_ACTIVE)],
        )
        style.configure(
            "Game.Vertical.TScrollbar", background=BUTTON, troughcolor=BACKGROUND, bordercolor=BACKGROUND,
            arrowcolor=FOREGROUND, lightcolor=BUTTON, darkcolor=BUTTON, gripcount=0,
        )
        style.map("Game.Vertical.TScrollbar", background=[("active", BUTTON_ACTIVE)])

    def _build_widgets(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root
        outer = tk.Frame(root, background=BACKGROUND)
        outer.pack(fill="both", expand=True)

        # The input area (packed first, so it never gets squeezed out): at the bottom, like a
        # terminal - but at the top on a Steam Deck, whose on-screen keyboard covers the lower
        # part of the screen (the question and what's being typed must stay in sight).
        self.input_on_top = on_steam_deck(self._env)
        panel = tk.Frame(outer, background=PANEL, padx=12, pady=8)
        panel.pack(side="top" if self.input_on_top else "bottom", fill="x")
        self.input_area = panel
        # While Steam's keyboard is up, this keeps the newest transcript lines above it.
        self.keyboard_space = tk.Frame(outer, background=BACKGROUND, height=1)
        self._keyboard_space_shown = False
        self.prompt_label = tk.Label(
            panel, text="", font=self.ui_font_bold, foreground=ACCENT, background=PANEL,
            anchor="w", justify="left",
        )
        self.prompt_label.pack(side="top", fill="x")
        self.choice_frame = tk.Frame(panel, background=PANEL)
        self.choice_frame.pack(side="top", fill="x", pady=(4, 2))
        entry_row = tk.Frame(panel, background=PANEL)
        entry_row.pack(side="top", fill="x", pady=(4, 0))
        self.send_button = ttk.Button(entry_row, text="Enter", style="Choice.TButton",
                                      command=self._submit, takefocus=False)
        self.send_button.pack(side="right", padx=(8, 0))
        self.keyboard_button = None
        if on_steam_deck(self._env):  # no physical keyboard, most likely: one tap for Steam's own
            self.keyboard_button = ttk.Button(entry_row, text="Keyboard", style="Choice.TButton",
                                              command=self.show_keyboard, takefocus=False)
            self.keyboard_button.pack(side="right", padx=(8, 0))
        # Reporting something the AI shouldn't have written: Steam's overlay can't open over
        # this window (outside a Steam Deck's Game Mode), so the game has its own way there.
        self.report_button = ttk.Button(entry_row, text=REPORT_BUTTON, style="Choice.TButton",
                                        command=self.report_problem, takefocus=False)
        self.report_button.pack(side="right", padx=(8, 0))
        self.entry = tk.Entry(
            entry_row, font=self.font, background=ENTRY_BACKGROUND, foreground=FOREGROUND,
            insertbackground=FOREGROUND, disabledbackground=PANEL, relief="flat",
            highlightthickness=2, highlightcolor=ACCENT, highlightbackground=BORDER,
            selectbackground=SELECTION, selectforeground=FOREGROUND,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)

        # The transcript.
        body = tk.Frame(outer, background=BACKGROUND)
        body.pack(side="top", fill="both", expand=True)
        self._transcript_frame = body
        self.scrollbar = ttk.Scrollbar(body, orient="vertical", style="Game.Vertical.TScrollbar")
        self.scrollbar.pack(side="right", fill="y")
        self.text = tk.Text(
            body, font=self.font, background=BACKGROUND, foreground=FOREGROUND, wrap="none",
            padx=10, pady=8, borderwidth=0, highlightthickness=0, relief="flat", undo=False,
            insertwidth=0, selectbackground=SELECTION, inactiveselectbackground=SELECTION,
            selectforeground=FOREGROUND, cursor="xterm", takefocus=0, exportselection=True,
            yscrollcommand=self._on_yscroll,
        )
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.configure(command=self.text.yview)
        self.text.configure(state="disabled")

        self.text.bind("<Configure>", self._on_resize)
        self.choice_frame.bind("<Configure>", self._on_choice_frame_resize)
        self.text.bind("<Key>", self._on_transcript_key)
        # Tk moves the keyboard focus to the (read-only) transcript when it's clicked - to
        # bring the window forward after copying a key in the browser, say. A paste there
        # would vanish, so it goes to the input bar; and a plain click hands focus back.
        self.text.bind("<<Paste>>", self._paste_into_entry)
        self.text.bind("<ButtonRelease-1>", self._on_transcript_click, add="+")
        # Right-click the input bar: Paste (and friends), like any text box.
        self._entry_menu = tk.Menu(root, tearoff=0)
        for label, event in (("Paste", "<<Paste>>"), ("Copy", "<<Copy>>"), ("Cut", "<<Cut>>")):
            self._entry_menu.add_command(label=label, command=lambda e=event: self.entry.event_generate(e))
        for sequence in ("<Button-3>",) + (("<Button-2>", "<Control-Button-1>") if sys.platform == "darwin" else ()):
            self.entry.bind(sequence, self._show_entry_menu)
        # ...and the transcript: Copy and Select all, so copying a link or a file path never needs the keyboard.
        self._transcript_menu = tk.Menu(root, tearoff=0)
        self._transcript_menu.add_command(label="Copy", command=self._copy_transcript_selection)
        self._transcript_menu.add_command(label="Select all", command=self._select_all_transcript)
        for sequence in ("<Button-3>",) + (("<Button-2>", "<Control-Button-1>") if sys.platform == "darwin" else ()):
            self.text.bind(sequence, self._show_transcript_menu)
        root.protocol("WM_DELETE_WINDOW", self.request_close)
        if sys.platform == "darwin":
            try:
                root.createcommand("::tk::mac::Quit", self.request_close)  # the app menu's Quit / Cmd+Q
            except Exception:
                pass

    def _bind_keys(self) -> None:
        bindings = {
            "<Return>": lambda _e: self._submit(),
            "<KP_Enter>": lambda _e: self._submit(),
            "<Up>": lambda _e: self._recall(-1),
            "<Down>": lambda _e: self._recall(+1),
            "<Prior>": lambda _e: self._scroll_pages(-1),
            "<Next>": lambda _e: self._scroll_pages(+1),
        }
        for sequence, handler in bindings.items():
            self.entry.bind(sequence, handler)
        # Ctrl+Shift+V pastes too (what many terminals use), so habit never loses a paste.
        try:
            self.entry.bind("<Control-V>", lambda _e: self._paste_into_entry())
        except Exception:
            pass
        zoom = {
            "equal": +1, "plus": +1, "KP_Add": +1, "minus": -1, "underscore": -1, "KP_Subtract": -1,
            "0": 0, "KP_0": 0, "KP_Insert": 0,
        }
        modifiers = ["Control"] + (["Command"] if sys.platform == "darwin" else [])
        for modifier in modifiers:
            for key, step in zoom.items():
                try:
                    self.root.bind(f"<{modifier}-{key}>", lambda _e, s=step: self._zoom_key(s))
                except Exception:
                    pass  # a key name this Tk doesn't know
        self.root.bind("<F11>", lambda _e: self.set_fullscreen(not self.fullscreen))
        self.root.bind("<Escape>", lambda _e: self.set_fullscreen(False) if self.fullscreen else None)

    def _set_icon(self) -> None:
        path = _icon_path()
        if not path.is_file():
            return
        try:
            self._icon = self.tk.PhotoImage(master=self.root, file=str(path))
            self.root.iconphoto(True, self._icon)
        except Exception:
            pass

    def _metrics(self) -> Tuple[int, int]:
        """(character width, line height) of the transcript font, in pixels."""
        return max(1, self.font.measure("0")), max(1, self.font.metrics("linespace"))

    def _text_insets(self) -> Tuple[int, int]:
        """Pixels around the text inside the transcript widget (horizontal, vertical)."""
        text = self.text
        border = int(text.cget("borderwidth")) + int(text.cget("highlightthickness"))
        return 2 * (border + int(text.cget("padx"))), 2 * (border + int(text.cget("pady")))

    def _window_size_for(self, columns: int, rows: int) -> Tuple[int, int]:
        self.root.update_idletasks()
        char_w, line_h = self._metrics()
        inset_w, inset_h = self._text_insets()
        width = columns * char_w + inset_w + self.scrollbar.winfo_reqwidth() + _SPARE_PIXELS
        height = rows * line_h + inset_h + self.input_area.winfo_reqheight() + _SPARE_PIXELS
        return width, height

    def _place_window(self, *, fit_font: bool) -> None:
        """A default size that shows at least 100x32 characters and fits the screen (1280x800 and up).

        Full screen (a Steam Deck) fills the screen and keeps its bigger font
        as long as 80x24 characters fit - on the Deck's 1280x800 screen the
        16-point font stays, rather than shrinking to fit 100x32 at desktop size.
        """
        root = self.root
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        room_w, room_h = (screen_w, screen_h) if self.fullscreen else (int(screen_w * 0.96), int(screen_h * 0.88))
        columns, rows = ((FULLSCREEN_MIN_COLUMNS, FULLSCREEN_MIN_ROWS) if self.fullscreen
                         else (MIN_COLUMNS, MIN_ROWS))
        # Reserve room for one row of buttons, so the window doesn't jump when a menu appears.
        self.choice_frame.configure(height=self.send_button.winfo_reqheight() + 6)
        self.choice_frame.pack_propagate(False)
        width, height = self._window_size_for(columns, rows)
        while fit_font and (width > room_w or height > room_h) and self.font_size > MIN_FONT_SIZE + 1:
            self._set_font_size(self.font_size - 1)
            self.choice_frame.configure(height=self.send_button.winfo_reqheight() + 6)
            width, height = self._window_size_for(columns, rows)
        if self.fullscreen:
            width, height = room_w, room_h  # the whole screen: as much text as fits at this size
        width, height = min(width, room_w), min(height, room_h)
        root.geometry(f"{width}x{height}")
        root.minsize(min(480, width), min(320, height))
        if self.fullscreen:
            self.set_fullscreen(True)
        char_w, line_h = self._metrics()
        inset_w, inset_h = self._text_insets()
        text_w = width - self.scrollbar.winfo_reqwidth() - inset_w
        text_h = height - self.input_area.winfo_reqheight() - inset_h
        self._apply_size(*self._fit(text_w, text_h))

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self) -> None:
        """Start the game (in its worker thread) and the window's polling loop."""
        if self._started:
            return
        self._started = True
        self._saved_streams = (sys.stdout, sys.stderr)
        # Stray print()s (argparse's --help, a library's note) land in the window, like in a terminal.
        sys.stdout = self.bridge.stream
        sys.stderr = _TeeStream(self.bridge.stream, sys.stderr)
        self._install_signal_handlers()
        self._worker = threading.Thread(target=self._worker_main, name="gettowork-game", daemon=True)
        self._worker.start()
        self._poll_job = self.root.after(self._poll_ms, self._poll)
        if self.selftest is not None:
            self.root.after(int(self._selftest_timeout_s * 1000), self._selftest_timeout)
            self.root.after(50, self._selftest_exercise_window)
        self.entry.focus_set()

    def run(self) -> int:
        """Show the window until it closes; returns the exit code."""
        self.start()
        try:
            self.root.mainloop()
        finally:
            code = self.finish()
        return code

    def finish(self) -> int:
        """Tidy up after the window closed (idempotent); returns the exit code."""
        if self._finished:
            return self.exit_code
        self._finished = True
        if self._worker is not None and self._worker.is_alive():
            self.bridge.close()
            self._worker.join(timeout=max(0.0, self._close_deadline - time.monotonic()) if self._closing
                              else self._close_wait_s)
        for job in (self._poll_job, self._resize_job, self._layout_job):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
        try:
            self._drain()  # the game's last words (the selftest transcript wants them)
        except Exception:
            pass
        self.bridge.detach()  # anything printed from now on has nowhere to go
        if self._remember_font_size and self._zoomed and self.font_size != self._saved_font_size:
            save_font_size(self.font_size)
        self._restore_process_state()
        self._release_tk()
        if self.selftest is not None:
            self._write_selftest_output()
        return self.exit_code

    def _release_tk(self) -> None:
        """Destroy the window and let go of every Tk object - here, on the Tk thread.

        Tk objects must be freed by the thread that made them: if Python's
        garbage collector happened to free them on the game thread (which may
        still be finishing), Tcl would abort the whole program. So nothing that
        outlives the window - the game thread holds on to this object - keeps
        a reference to a widget, a font or the Tk interpreter.
        """
        import gc

        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            del self.root.report_callback_exception  # (it points back at this object)
        except AttributeError:
            pass
        for name in ("text", "entry", "prompt_label", "choice_frame", "send_button", "keyboard_button",
                     "report_button", "scrollbar", "input_area", "_entry_menu", "_transcript_menu", "keyboard_space",
                     "_transcript_frame",
                     "font", "font_bold", "font_italic", "font_bold_italic", "ui_font", "ui_font_bold", "_icon"):
            if hasattr(self, name):
                setattr(self, name, None)
        self._fonts, self._choice_buttons, self._choice_rows = {}, [], []
        # Whatever else holds a Tk object - a widget added later, a list of them, a widget's bound method -
        # is let go of too, so a forgotten name can't bring the "freed on the wrong thread" abort back.
        for name, value in list(vars(self).items()):
            if name != "root" and holds_tk_object(value):
                setattr(self, name, type(value)() if isinstance(value, (list, tuple, dict, set)) else None)
        self.root = None
        gc.collect()  # any other Tk garbage from this session goes now, on this thread

    @property
    def exit_code(self) -> int:
        if self.selftest is not None:
            return EXIT_ERROR if self.selftest_result is None else self.selftest_result
        if self._closed_early:
            return EXIT_OK  # closing the window is a normal way to stop playing
        return EXIT_OK if self._game_code is None else self._game_code

    @property
    def closed(self) -> bool:
        return self._quit_requested

    def _install_signal_handlers(self) -> None:
        """Ctrl+C in the launching terminal, or "stop" from Steam: close the window politely."""
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum: int, frame: object) -> None:
            self._signal_close = True  # acted on by _poll (on the Tk side, between events)

        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._saved_signals.append((sig, signal.signal(sig, handler)))
            except (OSError, ValueError):
                pass

    def _restore_process_state(self) -> None:
        if self._saved_streams is not None:
            # Whatever they are now - the window's streams, or a live display's proxies left
            # behind by a game thread still busy when the window closed - put back the
            # originals, so later messages (a self-test verdict, an error) are seen.
            out, err = self._saved_streams
            if sys.stdout is not out:
                sys.stdout = out
            if sys.stderr is not err:
                sys.stderr = err
            self._saved_streams = None
        for sig, previous in self._saved_signals:
            try:
                signal.signal(sig, previous)
            except (OSError, ValueError, TypeError):
                pass
        self._saved_signals = []

    def _report_tk_error(self, exc_type: type, exc: BaseException, tb: Any) -> None:
        """An error inside a Tk callback: log it and carry on (a glitch must not end the game).

        The crash report is written for the first of each kind of error (and
        at most a few in all): an error that repeats on every poll must not
        rewrite the file 30 times a second.
        """
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        self.tk_errors.append(details)
        key = f"{exc_type.__name__}: {exc}"
        if key in self._reported_errors or len(self._reported_errors) >= MAX_CRASH_REPORTS:
            return
        self._reported_errors.add(key)
        write_crash_report("an error in the game window (the game carried on)", exc)

    # -- the game (worker thread) -------------------------------------------------------------

    def _worker_main(self) -> None:
        code = EXIT_ERROR
        try:
            code = self._game_main(list(self.argv), self.ui)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (EXIT_OK if exc.code is None else EXIT_ERROR)
        except BaseException as exc:  # a bug: say so kindly, and keep the details for a bug report
            path = write_crash_report("the game stopped unexpectedly", exc)
            try:
                where = f" (details saved in {path})" if path else ""
                self.ui.error(f"Oops - the game stopped unexpectedly{where}. Sorry about that!")
            except Exception:
                pass
            code = EXIT_ERROR
        finally:
            self.bridge.finish(code if isinstance(code, int) else EXIT_ERROR)

    def _ask_line(self, prompt: str) -> str:
        """UI's input function (runs on the game thread): show the question, wait for the answer.

        The answer is echoed after the question, like a terminal - except one
        that looks like an API key or token (pasted into the wrong box, say),
        which shows as "(hidden)", as at the key question itself.
        """
        from ..ui import safe_text

        self.console.print(prompt, end="", highlight=False)  # like Console.input: the prompt stays in the transcript
        answer = self.bridge.request_line(prompt)
        if self.is_secret_answer(answer):
            self.console.print(HIDDEN_ANSWER, style="dim", markup=False, highlight=False)
        else:
            self._echo_answer(prompt, safe_text(answer))
        return answer

    def _echo_answer(self, prompt: str, answer: str) -> None:
        """Print the answer after its question so the two wrap as one line of text.

        rich doesn't know the answer starts after the question, so it would
        wrap it as if from the left edge and the end would run past the
        window's right edge. A long answer therefore rewrites the question's
        (last) line together with the answer, and rich wraps them as one.
        """
        from rich.cells import cell_len
        from rich.control import Control, ControlType
        from rich.text import Text

        width = self.console.width
        try:
            tail = self.console.render_str(prompt, highlight=False).split("\n")[-1]
        except Exception:
            tail = None
        if tail is None or cell_len(tail.plain) + cell_len(answer) <= width:
            self.console.print(answer, markup=False, highlight=False, emoji=False)  # fits after the question
            return
        if cell_len(tail.plain) >= width:  # (a question as wide as the window: the answer starts a new line)
            self.console.print()
            self.console.print(answer, markup=False, highlight=False, emoji=False)
            return
        self.console.control(Control((ControlType.CARRIAGE_RETURN,), (ControlType.ERASE_IN_LINE, 2)))
        self.console.print(tail + Text(answer), highlight=False, emoji=False)

    def is_secret_answer(self, answer: str) -> bool:
        """Is this typed answer (or a word in it) an API key - one given at a key question, or key-shaped?"""
        from ..ui import looks_like_secret

        text = str(answer or "").strip()
        if not text:
            return False
        if self._secret_digests and any(_digest(part) in self._secret_digests for part in [text, *text.split()]):
            return True
        return looks_like_secret(text)

    def _ask_secret(self, prompt: str) -> str:
        """UI's secret function (API keys): the typed text is masked and never echoed."""
        from ..ui import safe_text

        self.console.print(safe_text(prompt), end="", markup=False, highlight=False, emoji=False)
        answer = self.bridge.request_line(prompt, secret=True)
        self.console.print(HIDDEN_ANSWER if answer else "", style="dim", markup=False, highlight=False)
        return answer

    # -- polling (Tk thread) --------------------------------------------------------------------

    def _poll(self) -> None:
        self._poll_job = None
        try:
            self._drain()
            if self._signal_close:
                self._signal_close = False
                self.request_close()
        finally:
            if not self._finished and not self._quit_requested:
                self._poll_job = self.root.after(self._poll_ms, self._poll)

    def _drain(self) -> None:
        """Handle everything the game sent, then redraw the lines that changed."""
        for kind, payload in self.bridge.poll():
            if kind == OUTPUT:
                self.buffer.feed(payload)
            elif kind == PROMPT:
                self._safe_sync()  # show the question's text before the prompt appears
                self._on_prompt(payload)
            elif kind == CHOICES:
                self._on_choices(payload)
            elif kind == FINISHED:
                self._safe_sync()
                self._on_finished(payload)
        self._safe_sync()

    def _safe_sync(self) -> None:
        """:meth:`_sync_view`, but a line Tk won't draw never stops the game.

        The question, its buttons and the end of the game are still handled,
        and the transcript is redrawn from scratch (so the same failure isn't
        retried on every poll).
        """
        try:
            self._sync_view()
        except Exception as exc:
            self._report_tk_error(type(exc), exc, exc.__traceback__)
            self._redraw_view()

    def _redraw_view(self) -> None:
        """Draw the whole transcript again, line by line; a line Tk refuses is drawn as plain ASCII."""
        buffer, text = self.buffer, self.text
        buffer.take_dirty()
        self._seen_trimmed = buffer.trimmed_total
        total = len(buffer)
        try:
            text.configure(state="normal")
            text.delete("1.0", "end")
            for index in range(total):
                if index:
                    text.insert("end-1c", "\n")
                try:
                    args = self._line_args(index)
                    if args:
                        text.insert("end-1c", *args)
                except Exception:
                    plain = buffer.line_text(index).encode("ascii", "replace").decode("ascii")
                    text.insert("end-1c", plain)
        except Exception:
            pass
        finally:
            self._rendered = total  # whatever happened, never retry the same lines on every poll
            try:
                text.configure(state="disabled")
                text.yview_moveto(1.0)
            except Exception:
                pass

    def _sync_view(self) -> None:
        """Mirror the buffer's changed lines into the Text widget."""
        buffer, text = self.buffer, self.text
        drop = buffer.trimmed_total - self._seen_trimmed
        dirty = buffer.take_dirty()
        total = len(buffer)
        if not drop and not dirty and self._rendered == total:
            return
        self._seen_trimmed = buffer.trimmed_total
        at_bottom = self._follow or self._at_bottom or text.yview()[1] >= 0.999
        text.configure(state="normal")
        try:
            if drop:
                if drop >= self._rendered:
                    text.delete("1.0", "end")
                    self._rendered = 0
                else:
                    text.delete("1.0", f"{drop + 1}.0")
                    self._rendered -= drop
            for index in dirty:
                if index >= self._rendered:
                    break
                line = index + 1
                text.delete(f"{line}.0", f"{line}.end")
                args = self._line_args(index)
                if args:
                    text.insert(f"{line}.0", *args)
            if total > self._rendered:
                args = []
                for index in range(self._rendered, total):
                    if index > 0:  # (the widget's first line needs no newline before it)
                        args.extend(("\n", ()))
                    args.extend(self._line_args(index))
                if args:
                    text.insert("end-1c", *args)
                self._rendered = total
        finally:
            text.configure(state="disabled")
        if at_bottom:
            text.yview_moveto(1.0)
        self._follow = False

    def _line_args(self, index: int) -> List[Any]:
        args: List[Any] = []
        fallback = self._glyph_fallback
        for chunk, style in self.buffer.line(index, substitute=emoji_stand_in if self._x11 else None):
            chunk = clean_text(chunk)  # (a lone surrogate would make Tk refuse the whole line)
            args.append(chunk.translate(fallback) if fallback else chunk)
            args.append(self._tags_for(style))
        return args

    def _tags_for(self, style: Style) -> Tuple[str, ...]:
        if style == DEFAULT_STYLE:
            return ()
        tag = self._style_tags.get(style)
        if tag is None:
            tag = f"style{len(self._style_tags)}"
            fg, bg = style_colors(style)
            if style.bold and not self._bold_ok:
                fg = blend(fg, "#ffffff", 0.4)  # no same-width bold font: brighter instead
            options: Dict[str, Any] = {"font": self._fonts[(style.bold, style.italic)]}
            if fg != FOREGROUND:
                options["foreground"] = fg
            if bg is not None:
                options["background"] = bg
            if style.underline or style.link:
                options["underline"] = True
            if style.strike:
                options["overstrike"] = True
            self.text.tag_configure(tag, **options)
            self.text.tag_raise("sel")  # a selection stays visible over coloured text
            self._style_tags[style] = tag
        if style.link:
            return (tag, self._link_tag(style.link))
        return (tag,)

    def _link_tag(self, url: str) -> str:
        tag = self._link_tags.get(url)
        if tag is None:
            tag = f"link{len(self._link_tags)}"
            self._link_tags[url] = tag
            self.text.tag_bind(tag, "<Button-1>", lambda _e, u=url: self.open_link(u))
            self.text.tag_bind(tag, "<Enter>", lambda _e: self.text.configure(cursor="hand2"))
            self.text.tag_bind(tag, "<Leave>", lambda _e: self.text.configure(cursor="xterm"))
        return tag

    def link_url(self, tag: str) -> Optional[str]:
        """The URL behind a link tag (for tests and tools)."""
        for url, name in self._link_tags.items():
            if name == tag:
                return url
        return None

    def open_link(self, url: str) -> None:
        """A click on a link in the transcript: open it in the browser (off the Tk thread)."""
        threading.Thread(target=self.bridge.open_url, args=(url,), name="gettowork-open-link",
                         daemon=True).start()

    # -- questions and answers ---------------------------------------------------------------

    def _on_prompt(self, prompt: Prompt) -> None:
        self._prompt = prompt
        text = self._label_text(plain_prompt(prompt.text)) or "Your answer"
        if prompt.secret:
            text = f"{text}  {SECRET_HINT}"
        self.prompt_label.configure(text=text)
        self.entry.configure(show="•" if prompt.secret else "", state="normal")
        self.send_button.state(["!disabled"])
        for button in self._choice_buttons:
            button.state(["!disabled"])
        self._history_pos = len(self._history)
        if not self._closing:
            self.entry.focus_set()
        if self.selftest is not None:
            self.root.after(SELFTEST_ANSWER_DELAY_MS, lambda pid=prompt.id: self._selftest_answer(pid))

    def _on_choices(self, options: Sequence[Tuple[str, str]]) -> None:
        self._choices = tuple(options)
        self._set_buttons([(key, self._label_text(text) or key or "Continue")
                           for (key, _label), text in zip(options, button_labels(options))])

    def _label_text(self, text: str) -> str:
        """Text for the prompt label or a button (emoji swapped for stand-ins on X11, see emoji_stand_in)."""
        return emoji_safe_text(text) if self._x11 else text

    def _set_buttons(self, buttons: Sequence[Tuple[str, str]]) -> None:
        for widget in self._choice_buttons + self._choice_rows:
            widget.destroy()
        self._choice_buttons, self._choice_rows = [], []
        for key, label in buttons:
            button = self.ttk.Button(self.choice_frame, text=label, style="Choice.TButton", takefocus=False,
                                     command=lambda k=key: self._submit(k))
            button.choice_key = key  # type: ignore[attr-defined]
            self._choice_buttons.append(button)
        # With buttons, the frame grows to fit their rows; without, it keeps one row's height.
        self.choice_frame.pack_propagate(bool(self._choice_buttons))
        self._layout_choices()

    @property
    def choice_buttons(self) -> List[Tuple[str, str]]:
        """The buttons on show, as (key, label) pairs."""
        return [(b.choice_key, b.cget("text")) for b in self._choice_buttons]  # type: ignore[attr-defined]

    def press_choice(self, key: str) -> bool:
        """Press the button for ``key`` (as a click would). False if there's none."""
        for button in self._choice_buttons:
            if button.choice_key == key:  # type: ignore[attr-defined]
                button.invoke()
                return True
        return False

    def _on_choice_frame_resize(self, event: Any) -> None:
        if event.width != self._layout_width:  # (height changes come from our own layout)
            self._schedule_layout()

    def _schedule_layout(self) -> None:
        if self._layout_job is None:
            self._layout_job = self.root.after(30, self._layout_choices)

    def _layout_choices(self) -> None:
        """Flow the buttons into rows that fit the window's width."""
        self._layout_job = None
        for row in self._choice_rows:
            row.destroy()
        self._choice_rows = []
        if not self._choice_buttons:
            return
        available = max(self.choice_frame.winfo_width(), self.root.winfo_width() - 40, 200)
        self._layout_width = self.choice_frame.winfo_width()
        row, used = None, 0
        for button in self._choice_buttons:
            width = button.winfo_reqwidth() + 8
            if row is None or (used + width > available and used > 0):
                row = self.tk.Frame(self.choice_frame, background=PANEL)
                row.pack(side="top", fill="x")
                self._choice_rows.append(row)
                used = 0
            button.pack(in_=row, side="left", padx=(0, 8), pady=3)
            button.lift(row)
            used += width

    def _submit(self, value: Optional[str] = None) -> str:
        """Enter, the Enter button or a choice button: answer the question on show."""
        if self._game_done:
            self.request_close()
            return "break"
        prompt = self._prompt
        if prompt is None:
            return "break"  # nothing asked yet: keep what's typed for the next question
        text = self.entry.get() if value is None else value
        if not self.bridge.submit(text, prompt.id):
            return "break"
        self.entry.delete(0, "end")
        if prompt.secret and text.strip():
            self._secret_digests.add(_digest(text.strip()))  # so it's hidden if pasted again elsewhere
        elif value is None and text.strip() and not self.is_secret_answer(text):
            if not self._history or self._history[-1] != text:
                self._history.append(text)
        self._history_pos = len(self._history)
        self._show_keyboard_space(False)  # (sending the answer is when Steam's keyboard goes away)
        self._prompt = None
        self.prompt_label.configure(text="")
        self.entry.configure(show="")
        for button in self._choice_buttons:
            button.state(["disabled"])  # no double answers while the game moves on
        self._follow = True
        self.text.yview_moveto(1.0)
        return "break"

    def type_answer(self, text: str) -> None:
        """Type ``text`` into the input bar and press Enter (tests and the self-test use this)."""
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self._submit()

    @property
    def prompt(self) -> Optional[Prompt]:
        """The question on show, if any."""
        return self._prompt

    def _recall(self, step: int) -> str:
        """Up / Down in the input bar: earlier answers, like a terminal's history."""
        if not self._history or (self._prompt is not None and self._prompt.secret):
            return "break"
        self._history_pos = min(max(0, self._history_pos + step), len(self._history))
        self.entry.delete(0, "end")
        if self._history_pos < len(self._history):
            self.entry.insert(0, self._history[self._history_pos])
        return "break"

    def _scroll_pages(self, pages: int) -> str:
        self.text.yview_scroll(pages, "pages")
        return "break"

    def report_problem(self) -> None:
        """The Report a problem button: say where reports go, then open it (off the Tk thread).

        The link is always written into the transcript first: a browser may
        never appear (a Steam Deck in Game Mode has no desktop browser, and
        Linux reports success as soon as ``xdg-open`` starts). On a Steam
        Deck or in Big Picture the link goes through Steam
        (``steam://openurl/...``), whose own browser shows over the game.
        """
        from ..notices import report_url

        url = report_url()
        through_steam = on_steam_deck(self._env)
        where = " - or the Steam button's menu > Discussions" if through_steam else ""
        self._note(f"Opening {osc8_link(url)} to report a problem. If no browser appears, open that link on any "
                   f"device{where}.")
        opener = self._keyboard_opener

        def work() -> None:
            ok = open_through_steam(url, opener) if through_steam else self.bridge.open_url(url)
            if not ok:
                self.bridge.post_note("\r\n\x1b[33mCouldn't open a browser automatically - copy the link above "
                                      "(right-click > Copy).\x1b[0m\r\n")

        threading.Thread(target=work, name="gettowork-report", daemon=True).start()

    def _note(self, text: str) -> None:
        """A dim line of the window's own in the transcript (Tk thread)."""
        try:
            _row, col = self.buffer.cursor
            self.buffer.feed(("\r\n" if col else "") + f"\x1b[2m{text}\x1b[0m\r\n")
            self._safe_sync()
        except Exception:
            pass

    def show_keyboard(self) -> None:
        """The Keyboard button (Steam Deck): focus the input bar and ask Steam for its on-screen keyboard.

        Steam draws its keyboard over the lower part of the screen without
        resizing the game, so (with the input area already at the top) the
        transcript gives up that space until the answer is sent: the newest
        lines - the obstacle being answered - stay in sight above the keyboard.
        """
        self.entry.focus_set()
        self._show_keyboard_space(True)
        threading.Thread(target=open_steam_keyboard, args=(self._keyboard_opener,), name="gettowork-keyboard",
                         daemon=True).start()

    def _show_keyboard_space(self, on: bool) -> None:
        if not self.input_on_top or on == self._keyboard_space_shown:
            return
        self._keyboard_space_shown = on
        if on:
            height = int(self.root.winfo_height() * KEYBOARD_SHARE)
            self.keyboard_space.configure(height=max(1, height))
            self.keyboard_space.pack(side="bottom", fill="x", before=self._transcript_frame)
        else:
            self.keyboard_space.pack_forget()
        self._follow = True
        self.root.after_idle(self._keep_last_line_in_view)

    def _paste_into_entry(self, _event: Any = None) -> str:
        """Paste into the input bar, wherever the keyboard focus was (the transcript can't take text)."""
        self.entry.focus_set()
        try:
            self.entry.event_generate("<<Paste>>")
        except Exception:
            pass
        return "break"

    def _on_transcript_click(self, _event: Any = None) -> None:
        """A click on the transcript that selected nothing: typing (and pasting) goes to the input bar again."""
        try:
            if not self.text.tag_ranges("sel") and str(self.entry.cget("state")) == "normal":
                self.entry.focus_set()
        except Exception:
            pass

    def _show_entry_menu(self, event: Any) -> str:
        try:
            self.entry.focus_set()
            self._entry_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self._entry_menu.grab_release()
            except Exception:
                pass
        return "break"

    def _on_transcript_key(self, event: Any) -> Optional[str]:
        """Typing while the transcript has focus goes to the input bar.

        Keys that aren't typing are left alone: a modifier pressed on its own
        (the Ctrl of Ctrl+C arrives first, without the Control bit set - so
        moving the focus then would send the C to the input bar and copy
        nothing), function keys such as F11, and Escape, which the window's
        own shortcuts handle.
        """
        keysym = str(getattr(event, "keysym", "") or "")
        if _NOT_TYPING_KEYSYM_RE.match(keysym):
            return None
        control = bool(event.state & 0x4) or (sys.platform == "darwin" and bool(event.state & 0x8))
        if control and keysym in ("v", "V"):
            return self._paste_into_entry()  # (the transcript is read-only: a paste there would vanish)
        if control or keysym in ("Prior", "Next", "Up", "Down", "Left", "Right", "Home", "End"):
            return None  # copy, select all, scrolling...: the transcript's own bindings
        self.entry.focus_set()
        if keysym in ("Return", "KP_Enter"):
            return self._submit()
        if keysym == "BackSpace":
            with contextlib.suppress(Exception):
                at = int(self.entry.index("insert"))
                if at > 0:
                    self.entry.delete(at - 1)
            return "break"
        if event.char and event.char.isprintable():
            self.entry.insert("insert", event.char)
        return "break"

    def _copy_transcript_selection(self) -> bool:
        """Put the transcript's selected text on the clipboard (its right-click Copy). True if there was some."""
        try:
            if not self.text.tag_ranges("sel"):
                return False
            selected = self.text.get("sel.first", "sel.last")
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
            return True
        except Exception:
            return False

    def _select_all_transcript(self) -> None:
        with contextlib.suppress(Exception):
            self.text.tag_add("sel", "1.0", "end-1c")

    def _show_transcript_menu(self, event: Any) -> str:
        try:
            self._transcript_menu.entryconfigure(0, state="normal" if self.text.tag_ranges("sel") else "disabled")
            self._transcript_menu.tk_popup(event.x_root, event.y_root)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._transcript_menu.grab_release()
        return "break"

    def _on_finished(self, code: Any) -> None:
        self._game_code = code if isinstance(code, int) else EXIT_ERROR
        self._game_done = True
        self._prompt = None
        if self._closing:
            return  # the window is already on its way out
        if self.selftest is not None:
            self._selftest_verdict()
            self.request_close()
            return
        row, col = self.buffer.cursor
        self.buffer.feed(("\r\n" if col else "") + f"\n\x1b[2m{EXIT_HINT}\x1b[0m\n")
        self._sync_view()
        self.prompt_label.configure(text=EXIT_HINT)
        self.entry.configure(show="", state="normal")
        self._set_buttons([("close", "Close")])
        self.entry.focus_set()

    # -- closing -------------------------------------------------------------------------------

    def request_close(self) -> None:
        """The window's close button (or Enter at the end): let the game say goodbye, then close."""
        if self._closing or self._finished:
            return
        self._closing = True
        if not self._game_done:
            self._closed_early = True
        self.bridge.close()  # a waiting question raises EOFError: the game's normal goodbye path
        self._close_deadline = time.monotonic() + self._close_wait_s
        if self._worker is None or not self._worker.is_alive():
            self._quit()
            return
        try:
            self.root.withdraw()  # gone at once for the player; the game tidies up out of sight
        except Exception:
            pass
        self._wait_for_worker()

    def _wait_for_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive() or time.monotonic() >= self._close_deadline:
            self._quit()
            return
        self.root.after(50, self._wait_for_worker)

    def _quit(self) -> None:
        self._quit_requested = True
        try:
            self.root.quit()
        except Exception:
            pass

    # -- size, zoom, full screen -------------------------------------------------------------

    def _on_yscroll(self, first: Any, last: Any) -> None:
        """The transcript's view moved (scrolling, new text, a resize): move the scrollbar, note where we are."""
        self.scrollbar.set(first, last)
        try:
            self._at_bottom = float(last) >= 0.999
        except (TypeError, ValueError):
            pass

    def _keep_last_line_in_view(self) -> None:
        if self._at_bottom and self.text is not None:
            self.text.yview_moveto(1.0)

    def _on_resize(self, _event: Any = None) -> None:
        # The transcript got shorter or taller - often because a long question or a row of
        # buttons made the input area grow. Tk keeps the *top* line in place, which would
        # hide the newest lines (the question itself!) under the input area. If the
        # player was reading the latest lines, keep them in view.
        if self._at_bottom:
            self.text.yview_moveto(1.0)
            self.root.after_idle(self._keep_last_line_in_view)
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(60, self._update_size)

    def _update_size(self) -> None:
        """Keep the game's console exactly as wide as the transcript (in characters)."""
        self._resize_job = None
        width, height = self.text.winfo_width(), self.text.winfo_height()
        if width < 50 or height < 20:
            return  # not laid out yet
        inset_w, inset_h = self._text_insets()
        self._apply_size(*self._fit(width - inset_w, height - inset_h))
        self.prompt_label.configure(wraplength=max(200, self.root.winfo_width() - 40))
        self._schedule_layout()

    def _fit(self, text_width: int, text_height: int) -> Tuple[int, int]:
        """How many (columns, rows) of text fit in this many pixels (keeping a pixel or two spare)."""
        char_w, line_h = self._metrics()
        columns = (text_width - _EDGE_PIXELS) // char_w
        rows = text_height // line_h
        return max(MIN_CONSOLE_COLUMNS, columns), max(5, rows)

    def _apply_size(self, columns: int, rows: int) -> None:
        self._size = (columns, rows)
        self.bridge.set_size(columns, rows)
        self.buffer.rows = rows
        console = getattr(self, "console", None)
        if console is not None:
            # Read by rich on the game thread at its next print (an int swap is atomic).
            console.width = columns
            console.height = rows

    @property
    def console_size(self) -> Tuple[int, int]:
        """(columns, rows) the game's console currently has."""
        return self._size

    def _zoom_key(self, step: int) -> str:
        self.zoom(step)
        return "break"

    def zoom(self, step: int) -> None:
        """Ctrl+= bigger, Ctrl+- smaller, Ctrl+0 back to the default size."""
        default = FULLSCREEN_FONT_SIZE if self.fullscreen else DEFAULT_FONT_SIZE
        size = default if step == 0 else self.font_size + step
        self._zoomed = True
        self._set_font_size(size)
        self.root.after_idle(self._update_size)

    def _set_font_size(self, size: int) -> None:
        size = min(max(int(size), MIN_FONT_SIZE), MAX_FONT_SIZE)
        self.font_size = size
        for font in (self.font, self.font_bold, self.font_italic, self.font_bold_italic,
                     self.ui_font, self.ui_font_bold):
            font.configure(size=size)
        self._check_font_variants()
        for style, tag in self._style_tags.items():  # re-pick each style's font variant
            self.text.tag_configure(tag, font=self._fonts[(style.bold, style.italic)])

    def set_fullscreen(self, on: bool) -> None:
        """F11 toggles full screen (on by default on a Steam Deck); Escape leaves it."""
        self.fullscreen = bool(on)
        try:
            self.root.attributes("-fullscreen", self.fullscreen)
        except Exception:
            pass

    # -- self-test (CI) ---------------------------------------------------------------------------

    def _selftest_exercise_window(self) -> None:
        """Poke the window's own features once, so the self-test covers them too."""
        before = self.font_size
        self.zoom(+1)
        self.zoom(-1)
        if self.font_size != before:
            self.selftest_note = "zooming in and out didn't return to the same font size"

    def _selftest_answer(self, prompt_id: int) -> None:
        if self.selftest is None or self._prompt is None or self._prompt.id != prompt_id or self._closing:
            return
        answer = self.selftest.answer(self._prompt.text, secret=self._prompt.secret, choices=self._choices)
        if answer is None:
            self.selftest_result = EXIT_ERROR
            self.selftest_note = f"gave up after {self.selftest.max_answers} answers (the game kept asking)"
            self.request_close()
            return
        self.type_answer(answer)

    def _selftest_verdict(self) -> None:
        if self.selftest_result is not None:
            return
        transcript = self.buffer.text()
        if self._game_code != EXIT_OK:
            self.selftest_result = EXIT_ERROR
            self.selftest_note = f"the game ended with exit code {self._game_code}"
        elif SELFTEST_SUCCESS_TEXT not in transcript:
            self.selftest_result = EXIT_ERROR
            self.selftest_note = f"the transcript never said {SELFTEST_SUCCESS_TEXT!r}"
        elif self.selftest_note:
            self.selftest_result = EXIT_ERROR
        else:
            self.selftest_result = EXIT_OK

    def _selftest_timeout(self) -> None:
        if self.selftest_result is None and not self._game_done:
            self.selftest_result = EXIT_TIMEOUT
            self.selftest_note = f"timed out after {self._selftest_timeout_s:g} s"
            self.request_close()

    def _write_selftest_output(self) -> None:
        transcript = self.buffer.text()
        target = self._env.get(SELFTEST_OUT_ENV)
        if target:
            try:
                Path(target).write_text(transcript + "\n", encoding="utf-8")
            except OSError as exc:
                _say_to_stderr(f"GUI self-test: couldn't write the transcript to {target}: {exc}")
        verdict = {EXIT_OK: "passed", EXIT_TIMEOUT: "timed out"}.get(self.exit_code, "failed")
        answered = len(self.selftest.answers) if self.selftest is not None else 0
        note = f" - {self.selftest_note}" if self.selftest_note and self.exit_code != EXIT_OK else ""
        _say_to_stderr(f"GUI self-test {verdict} ({answered} answers){note}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def default_export_dir() -> Path:
    """Where the window's game offers to save transcripts: Documents/Get To Work, if there's a Documents folder.

    (The terminal version saves in the current folder. A game started from
    Steam or Finder has no sensible current folder - on macOS it is "/".)
    """
    documents = Path.home() / "Documents"
    if documents.is_dir():
        return documents / "Get To Work"
    from ..config import config_dir

    return config_dir() / "transcripts"


def _with_export_dir(args: List[str]) -> List[str]:
    """Add ``--export-dir <default_export_dir()>`` unless the player chose a folder."""
    if any(arg == "--export-dir" or arg.startswith("--export-dir=") for arg in args):
        return args
    try:
        return [*args, "--export-dir", str(default_export_dir())]
    except Exception:  # no home folder at all: keep the game's own default
        return args


def _clean_argv(argv: Sequence[str]) -> Tuple[List[str], bool]:
    """Drop what the OS adds when launching an app (macOS "-psn_..."); spot --gui-selftest."""
    cleaned, selftest = [], False
    for arg in argv:
        if arg == "--gui-selftest":
            selftest = True
        elif not arg.startswith("-psn_"):
            cleaned.append(arg)
    return cleaned, selftest


def run_gui(argv: Optional[List[str]] = None, *, selftest: bool = False,
            game_main: Optional[GameMain] = None) -> int:
    """Open the game's window and play; returns the exit code.

    ``argv`` are the game's usual command-line options (default: the process's
    own); transcripts are offered in :func:`default_export_dir` unless
    ``--export-dir`` is given. ``selftest=True`` plays the pretend model with scripted answers and
    returns 0 if the game reached work, 1 if not, 2 on a timeout (see
    :class:`SelftestPlayer`). ``game_main(argv, ui)`` replaces ``cli.main``
    (for tests).

    If the window can't open at all (no display, a Python without Tk), the
    reason goes to ``<config dir>/logs/gui-crash.txt`` and - when the game was
    started from a terminal - it plays there instead.
    """
    args, flagged = _clean_argv(sys.argv[1:] if argv is None else argv)
    selftest = selftest or flagged
    if selftest and not args:
        args = ["--mock", "--no-jev"]
    args = _with_export_dir(args)
    root = None
    try:
        if sys.platform == "darwin" and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("on macOS the game window must be opened from the main thread")
        tk, _tkfont, _ttk = _import_tk()
        _prepare_windows_process()
        root = tk.Tk(className="GetToWork")
        window = GameWindow(root, args, game_main=game_main, selftest=selftest)
    except Exception as exc:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        return _window_failed(exc, args, selftest=selftest, game_main=game_main)
    try:
        return window.run()
    except Exception as exc:  # the window broke while open: keep the details, exit gently
        path = write_crash_report("the game window stopped unexpectedly", exc)
        _say_to_stderr(f"The game window stopped unexpectedly.{f' Details: {path}' if path else ''}")
        return EXIT_ERROR


def _window_failed(exc: BaseException, args: List[str], *, selftest: bool,
                   game_main: Optional[GameMain]) -> int:
    path = write_crash_report("the game window couldn't open", exc)
    reason = f"{type(exc).__name__}: {exc}".strip()
    where = f" Details: {path}" if path else ""
    if not selftest and terminal_attached():
        _say_to_stderr(f"The game window couldn't open ({reason}), so let's play right here in the terminal.{where}")
        if game_main is not None:
            return game_main(args, None)
        from .. import cli

        return cli.main(args)
    _say_to_stderr(f"The game window couldn't open ({reason}).{where}")
    if not selftest:
        # Started by a double-click or from Steam, there's no console to read that line in.
        show_error_dialog(WINDOW_TITLE, window_failed_message(reason, path))
    return EXIT_ERROR
