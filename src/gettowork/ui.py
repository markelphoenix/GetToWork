"""Terminal UI helpers built on `rich`.

All user interaction goes through the `UI` class so the game can be driven by
scripted input in tests: pass `input_fn` / `secret_fn` / `open_url_fn` and a
`Console(record=True)` or `Console(file=io.StringIO())`. The game's own window
(`gui/app.py`) plugs in the same way, plus `choices_fn` (menu buttons),
`hides_input` (it masks secrets itself) and `window` (no command line there, so
hints are worded for the window).
"""

from __future__ import annotations

import codecs
import getpass
import re
import sys
import time
import webbrowser
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.spinner import Spinner
from rich.table import Column, Table
from rich.text import Text

InputFn = Callable[[str], str]
ChoicesFn = Callable[[list[tuple[str, str]]], None]  # shows a menu's (key, label) options as buttons

# Terminal control sequences: ANSI "CSI" (cursor moves, clearing lines...),
# "OSC" (window titles, hyperlinks, clipboard writes...) and other ESC codes,
# then any remaining control character except tab and newline.
#
# Callers sometimes run `rich.markup.escape` on untrusted text *before* it
# gets here, which puts backslashes in front of any "[tag]" (and doubles the
# backslashes already there). safe_text therefore *never* removes a backslash:
# taking even one would unbalance escape()'s pairs, expose the markup again,
# and rich would crash on a stray "[/]" (or silently restyle text). A string
# terminator "ESC \" loses only its ESC; the harmless backslash stays.
_FINAL_BYTE = r"[@-\[\]-~]"  # a CSI final byte - any except the backslash
_ESCAPE_SEQUENCE_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*" + _FINAL_BYTE  # CSI: ESC [ ... final byte
    + r"|\x9b[0-?]*[ -/]*" + _FINAL_BYTE  # the same, as a single C1 character
    + r"|(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x9c)?"  # OSC ... BEL / ST (an "ESC \\" end: see below)
    r"|\x1b[P^_X][^\x1b]*"  # DCS / PM / APC / SOS strings
    r"|\x1b[ -/]*[0-\[\]-~]?"  # any other ESC sequence (never taking a backslash)
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe_text(text: Any) -> str:
    """Remove terminal control sequences from text we didn't write ourselves.

    ``rich.markup.escape`` stops a model's ``[bold]`` from being read as a
    style, but it lets raw ESC codes through - and those can retitle the
    window, move the cursor to draw over earlier lines, add hidden links or
    (in some terminals) write to the clipboard. Model output, Jev replies,
    engine logs and error messages all pass through here before printing.
    Tabs and newlines are kept; everything else invisible is dropped.
    """
    cleaned = str(text if text is not None else "").replace("\r\n", "\n")
    cleaned = _ESCAPE_SEQUENCE_RE.sub("", cleaned)
    return _CONTROL_CHARS_RE.sub("", cleaned)


# One unbroken run of letters and digits, mixing both, at least this long: what an API key
# or access token has and a plan, a model name or a file name doesn't (their words are short).
_SECRET_RUN = 16
_SECRET_MIN_LENGTH = 20
_SECRET_RUN_RE = re.compile(r"[A-Za-z0-9]{%d,}" % _SECRET_RUN)


def looks_like_secret(text: Any) -> bool:
    """Does this answer look like an API key or token pasted into the wrong box?

    True when a word in it (no ``/`` - so never a web address or a Hugging
    Face model name) has 20+ characters including a run of 16+ letters and
    digits mixing both, e.g. ``tsk_live_9f8e7d6c5b4a3210abcd`` (also inside
    ``Bearer ...`` or a plan). Plans, model names
    (``Qwen3-4B-Instruct-2507-Q4_K_M.gguf``) and ordinary words don't: their
    letter/digit runs are short.
    """
    for token in str(text or "").split():
        if len(token) < _SECRET_MIN_LENGTH or "/" in token or "\\" in token:
            continue
        if any(re.search(r"[A-Za-z]", run) and re.search(r"[0-9]", run) for run in _SECRET_RUN_RE.findall(token)):
            return True
    return False


STEAM_ENV_VARIABLES = ("SteamAppId", "SteamGameId", "SteamClientLaunch")  # set by Steam for the games it starts


def launched_from_steam(env: Optional[Any] = None) -> bool:
    """Did Steam start this game? (Then its Launch Options are how a player adds a command-line option.)"""
    import os

    env = os.environ if env is None else env
    return any(str(env.get(name) or "").strip() for name in STEAM_ENV_VARIABLES)


def option_hint(ui: Any, option: str, *, env: Optional[Any] = None, markup: bool = True) -> Optional[str]:
    """How this player can start the game with a command-line `option` - or None if they can't.

    In a terminal: "start the game with gettowork --think". In the game's
    window started by Steam: its Launch Options. In the window otherwise (a
    double-clicked test build) there's no command line at all, so no hint.
    """
    if not getattr(ui, "in_window", False):
        from .config import command_name

        command = f"{command_name()} {option}"
        return f"start the game with [bold]{command}[/bold]" if markup else f"start the game with {command}"
    if launched_from_steam(env):
        return f"add {option} to its Launch Options in Steam (right-click Get To Work > Properties > General)"
    return None


def plain(text: Any) -> str:
    """Untrusted text made safe to embed in rich markup: control codes removed
    *first*, then any "[tag]" escaped so it prints literally."""
    return escape(safe_text(text))


class _Elapsed:
    """A spinner label that adds the seconds waited so far ("... 23s").

    rich re-renders it on every spinner frame, so the counter ticks by
    itself - long waits never look frozen.
    """

    def __init__(self, text: str, clock: Callable[[], float], show_after_s: float) -> None:
        self._text = text
        self._clock = clock
        self._start = clock()
        self._show_after_s = show_after_s

    def update(self, text: str) -> None:
        """Change the label (the seconds keep counting from the start)."""
        self._text = safe_text(text)

    def __rich__(self) -> Text:
        seconds = int(self._clock() - self._start)
        label = Text.from_markup(self._text)
        if seconds >= self._show_after_s:
            label.append(f"  {seconds}s", style="dim")
        return label

# Plain stand-ins for the few fancy characters the game prints, used when the
# terminal's code page can't show them (old Windows consoles, output redirected
# to a file in a legacy encoding...). Anything else unprintable becomes "?".
_PLAIN_CHARS = {
    "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2190": "<-", "\u2248": "~", "\u00d7": "x",
    "\u2026": "...", "\u00b7": "-", "\u2022": "*", "\u2713": "OK", "\u2605": "*", "\u26a1": "!",
    "\U0001f9e0": "+", "\u26a0": "!", "\u2265": ">=", "\u2264": "<=", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\ufe0f": "",
}


def _plain_replace(exc: UnicodeError) -> tuple[str, int]:
    if not isinstance(exc, UnicodeEncodeError):
        raise exc
    text = exc.object[exc.start:exc.end]
    return "".join(_PLAIN_CHARS.get(ch, "?") for ch in text), exc.end


codecs.register_error("gettowork-plain", _plain_replace)


def make_stream_safe(stream: object) -> None:
    """Never crash on a character the terminal's encoding can't show.

    UTF-8 terminals are left alone. On others (e.g. a Windows console using
    code page 437, or output piped into a cp1252 file) unprintable characters
    are swapped for plain ASCII look-alikes instead of raising UnicodeEncodeError.
    """
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    if encoding.startswith("utf"):
        return
    try:
        stream.reconfigure(errors="gettowork-plain")  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        pass


def make_input_safe(stream: object) -> None:
    """Never crash on typed or piped text the input encoding can't decode.

    In Git Bash (mintty) on Windows, or with input piped from a file, the
    terminal may send UTF-8 while Python expects the local code page. The
    usual "surrogateescape" handling then smuggles broken characters into the
    game, which crash later (e.g. when saving the transcript). Undecodable
    bytes become the standard replacement character instead.
    """
    try:
        stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        pass


class UserQuit(Exception):
    """Raised when the player presses Ctrl+C / Ctrl+D at a prompt."""


class WindowClosed(UserQuit):
    """The player closed the game's window while a question was waiting.

    A :class:`UserQuit`, so everything that stops politely on Ctrl+C stops on
    it too - but code that treats Ctrl+C as "skip this step and carry on"
    (the optional Jev setup, the review) must let it through: nobody is
    watching any more, so nothing more should happen (above all, no more
    model calls behind a hidden window).
    """


class UserChoseQuit(UserQuit):
    """The player *typed* "quit" (or "exit") at a yes/no question.

    A deliberate choice, not an interruption: the game still ends the way a
    typed "quit" does elsewhere (its quit ending and the review), and the
    program exits with 0, not the Ctrl+C code. Code that only cares that the
    player wants out can keep catching :class:`UserQuit`.
    """


class UI:
    def __init__(
        self,
        console: Optional[Console] = None,
        input_fn: Optional[InputFn] = None,
        secret_fn: Optional[InputFn] = None,
        open_url_fn: Optional[Callable[[str], bool]] = None,
        *,
        pauses: Optional[bool] = None,
        choices_fn: Optional[ChoicesFn] = None,
        hides_input: bool = False,
        window: bool = False,
    ) -> None:
        self.console = console or Console()
        # True in the game's own window (gui/app.py), where the player has no command
        # line: hints then talk about the window (buttons, Steam's launch options)
        # instead of options typed after a command.
        self.in_window = bool(window)
        self._input = input_fn or (lambda prompt: self.console.input(prompt))
        self._secret = secret_fn or getpass.getpass
        self._custom_secret = secret_fn is not None
        # The game's own window (gui/app.py) masks secret input itself.
        self._hides_input = bool(hides_input)
        # The window also shows menu options as buttons: told about each menu
        # before its question, and told [] once it's answered. (None in a terminal.)
        self._choices_fn = choices_fn
        self._open_url = open_url_fn or webbrowser.open
        # Pauses ("Press Enter to continue") only make sense when a person is
        # reading a real terminal and typing into it: piped or redirected runs
        # (and scripted tests, unless they ask for pauses) never stop.
        self._interactive = pauses if pauses is not None else (_isatty(sys.stdin) and self.console.is_terminal)
        self._clock = time.monotonic

    # -- output -------------------------------------------------------------

    # Every text method below runs its text through `safe_text`, so a stray
    # terminal control code (from a model, Jev, a log or an error message)
    # can never reach the terminal, whichever method prints it.

    def say(self, text: str = "", style: Optional[str] = None) -> None:
        self.console.print(safe_text(text), style=style, highlight=False)

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(safe_text(text)))

    def heading(self, text: str) -> None:
        self.console.rule(f"[bold]{safe_text(text)}[/bold]")

    def narrate(self, text: str, title: Optional[str] = None) -> None:
        """Story text from the game master (the local LLM)."""
        self.console.print(Panel(safe_text(text).strip(), title=title, border_style="magenta", padding=(1, 2)))

    def teach(self, title: str, body: str) -> None:
        """An edutainment aside explaining a concept. `body` is Markdown."""
        self.console.print(Panel(Markdown(safe_text(body)), title=f"Learn: {title}", border_style="cyan", padding=(0, 1)))

    def info(self, text: str) -> None:
        self.console.print(f"[cyan]i[/cyan] {safe_text(text)}", highlight=False)

    def success(self, text: str) -> None:
        self.console.print(f"[green]OK[/green] {safe_text(text)}", highlight=False)

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]![/yellow] {safe_text(text)}", highlight=False)

    def error(self, text: str) -> None:
        self.console.print(f"[red]x[/red] {safe_text(text)}", highlight=False)

    def pause(self, prompt: str = "Press Enter to continue") -> None:
        """Wait for Enter, so a long stretch of text can be read before it scrolls away.

        Only in a real, interactive terminal (or when created with
        ``pauses=True``): piped or redirected runs never stop here.
        """
        if not self._interactive:
            return
        with self._offering([("", "Continue")]):  # a button in the game's window (Steam Deck, touch)
            self.ask(f"[dim]{prompt}[/dim]", default="")

    def table(self, title: Optional[str], columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
        t = Table(title=title, show_lines=False)
        for c in columns:
            t.add_column(c)
        for r in rows:
            t.add_row(*[safe_text(x) for x in r])
        self.console.print(t)

    def json(self, data: object, title: Optional[str] = None) -> None:
        import json as _json

        from rich.syntax import Syntax

        text = _json.dumps(data, indent=2, ensure_ascii=False, default=str)
        # json.dumps escapes C0 controls (ESC becomes \u001b) but leaves the C1
        # range (U+0080-U+009F, which some terminals obey) as-is: escape those too.
        text = re.sub(r"[\x7f-\x9f]", lambda m: f"\\u{ord(m.group()):04x}", text)
        self.console.print(Panel(Syntax(text, "json", word_wrap=True), title=title, border_style="blue"))

    @contextmanager
    def status(self, text: str, *, show_elapsed_after_s: float = 3.0) -> Iterator[Callable[[str], None]]:
        """Spinner while waiting (e.g. for the LLM or Jev).

        After a few seconds the label also counts the seconds waited, so a
        slow model visibly *is* working rather than looking frozen. Yields
        ``update(text)``, which changes the label mid-wait (e.g. "asking again...").
        """
        label = _Elapsed(safe_text(text), self._clock, show_elapsed_after_s)
        if not self.in_window:
            with self.console.status(label):
                yield label.update
            return
        # In the game's window, stdout/stderr are left alone: rich's live display would
        # otherwise swap them for its own proxies, and a window closed mid-spinner (the
        # game thread still waiting on the model) would leave them swapped for good.
        spinner = Spinner("dots", text=label, style="status.spinner")
        with Live(spinner, console=self.console, refresh_per_second=12.5, transient=True,
                  redirect_stdout=False, redirect_stderr=False):
            yield label.update

    @contextmanager
    def download_progress(self, description: str, total_bytes: Optional[int]) -> Iterator[Callable[[int], None]]:
        """Yields `advance(n_bytes)`; renders a download bar."""
        # Long file names are shortened with "..." rather than wrapping the numbers
        # onto a second line in narrow (80-column) terminals.
        progress = Progress(
            TextColumn("{task.description}", table_column=Column(no_wrap=True, overflow="ellipsis", ratio=4)),
            BarColumn(bar_width=None, table_column=Column(ratio=1, min_width=8)),
            DownloadColumn(table_column=Column(no_wrap=True)),
            TransferSpeedColumn(table_column=Column(no_wrap=True)),
            TimeRemainingColumn(table_column=Column(no_wrap=True)),
            console=self.console,
            expand=True,
            # (the game's window keeps stdout/stderr as they are - see status())
            redirect_stdout=not self.in_window,
            redirect_stderr=not self.in_window,
        )
        with progress:
            task = progress.add_task(description, total=total_bytes)
            yield lambda n: progress.advance(task, n)

    # -- input --------------------------------------------------------------

    def ask(self, prompt: str, default: Optional[str] = None) -> str:
        suffix = f" [dim]({escape(default)})[/dim]" if default else ""
        while True:
            try:
                raw = self._input(f"[bold]{prompt}[/bold]{suffix} > ")
            except (EOFError, KeyboardInterrupt) as exc:
                raise self._quit_for(exc) from exc
            except UnicodeDecodeError:
                self.warn("I couldn't read that text (an unusual character?) - please type it again.")
                continue
            raw = safe_text(raw or "").strip()
            return raw if raw else (default or "")

    def confirm(self, prompt: str, default: bool = False) -> bool:
        hint = "Y/n" if default else "y/N"
        with self._offering([("y", "Yes"), ("n", "No")]):
            while True:
                # The backslash stops rich reading "[y/N]" as a (bogus) style tag and hiding it.
                ans = self.ask(f"{prompt} \\[{hint}]").lower()
                if not ans:
                    return default
                if ans in ("y", "yes"):
                    return True
                if ans in ("n", "no") or ans in _CONFIRM_NO_WORDS:
                    return False
                if ans in _CONFIRM_QUIT_WORDS:
                    raise UserChoseQuit()  # a UserQuit, so every caller already handles it politely
                self.warn("Please answer y or n (or quit).")

    def choose(self, prompt: str, options: Sequence[tuple[str, str]], default: Optional[str] = None, *,
               aliases: Optional[dict[str, str]] = None,
               accept: Optional[Callable[[str], bool]] = None) -> str:
        """Menu. `options` is [(key, label)]; returns the chosen key.

        The player may type the key or the 1-based number of the option.
        `aliases` maps other words to a key (e.g. {"back": "no"}), so the
        escape words the game teaches ("back", "quit"...) work here too.
        An answer that matches no option but that ``accept(answer)`` approves
        is returned as typed (e.g. an API key pasted straight into the menu).
        """
        keys = [k for k, _ in options]
        extra = {str(k).lower(): v for k, v in (aliases or {}).items() if v in keys}
        with self._offering(list(options)):
            while True:
                for i, (k, label) in enumerate(options, 1):
                    marker = " [dim](default)[/dim]" if k == default else ""
                    self.console.print(f"  [bold cyan]{i}[/bold cyan]) [bold]{k}[/bold] - {label}{marker}", highlight=False)
                ans = self.ask(prompt, default=default)
                picked = _match_option(ans, keys)
                if picked is None:
                    picked = extra.get(ans.strip().lower())
                if picked is not None:
                    return picked
                if accept is not None and ans and accept(ans):
                    return ans
                self.warn("Pick one of the options above (number or name).")

    @contextmanager
    def _offering(self, options: list[tuple[str, str]]) -> Iterator[None]:
        """Tell the game's window (if any) which buttons to show while a question waits.

        ``choices_fn(options)`` before asking, ``choices_fn([])`` once it's
        answered (or abandoned). A window problem never breaks the question.
        """
        self._offer(options)
        try:
            yield
        finally:
            self._offer([])

    def _offer(self, options: list[tuple[str, str]]) -> None:
        if self._choices_fn is None:
            return
        try:
            self._choices_fn(options)
        except Exception:
            pass

    def _quit_for(self, exc: BaseException) -> UserQuit:
        """What an interrupted question raises: :class:`WindowClosed` when the game's window
        closed under it (the window's input ends with EOFError), else :class:`UserQuit`."""
        return WindowClosed() if self.in_window and isinstance(exc, EOFError) else UserQuit()

    def can_hide_input(self) -> bool:
        """Can :meth:`secret` really keep typed text off the screen?

        ``getpass`` needs a real terminal. In an IDE's "Run" console, with
        piped input, or in an old Git Bash (mintty) window it silently falls
        back to *showing* what is typed - so callers check this first and
        never promise "hidden" when it isn't.
        """
        if self._hides_input or self._custom_secret:
            return True
        return _isatty(sys.stdin)

    def secret(self, prompt: str) -> str:
        """Hidden input (API keys). Nothing is echoed (check :meth:`can_hide_input` first)."""
        try:
            return safe_text(self._secret(f"{prompt}: ") or "").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise self._quit_for(exc) from exc
        except UnicodeDecodeError:
            self.warn("I couldn't read that text (an unusual character?) - let's go back a step.")
            return ""

    def open_url(self, url: str) -> None:
        """Try to open a browser; always print the URL too, in case it fails."""
        self.info(f"Opening [link={url}]{url}[/link]")
        try:
            ok = self._open_url(url)
        except Exception:
            ok = False
        if not ok:
            self.warn(f"Couldn't open a browser automatically. Copy this link instead: {url}")


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


MAX_MENU_NUMBER_DIGITS = 6  # a menu number longer than this can't be an option (and int() has a digit limit)

# At a yes/no question, the "back out" words the game teaches elsewhere mean "no"...
_CONFIRM_NO_WORDS = frozenset({"nope", "nah", "back", "cancel", "skip"})
# ...and "quit" means quit, just like it does at the model menu and in the game.
_CONFIRM_QUIT_WORDS = frozenset({"quit", "exit"})

# Short answers that mean the same as a menu key, when that key is on offer.
_ANSWER_ALIASES = {"y": "yes", "yeah": "yes", "yep": "yes", "sure": "yes", "n": "no", "nope": "no"}


def _match_option(answer: str, keys: Sequence[str]) -> Optional[str]:
    """Which menu key an answer means: the key itself (any case), its number,
    "y"/"n" for yes/no menus, or an unambiguous start of a key ("b" -> "back").
    None if it matches nothing, or more than one key."""
    text = (answer or "").strip()
    if text in keys:
        return text
    # isdecimal, not isdigit: "²" (a key of its own on AZERTY keyboards) is a
    # "digit" to Python but int() can't read it.
    # (A length limit too: int() refuses strings of over 4,300 digits, and a
    # pasted wall of digits must never crash a menu.)
    if text.isdecimal() and len(text) <= MAX_MENU_NUMBER_DIGITS and 1 <= int(text) <= len(keys):
        return keys[int(text) - 1]
    lowered = {k.lower(): k for k in keys}
    low = text.lower()
    if low in lowered:
        return lowered[low]
    alias = _ANSWER_ALIASES.get(low)
    if alias in lowered:
        return lowered[alias]
    if low:
        starts = [k for k in keys if k.lower().startswith(low)]
        if len(starts) == 1:
            return starts[0]
    return None
