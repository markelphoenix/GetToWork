"""Thread-safe plumbing between the game (a worker thread) and the window (Tk).

Tk is not thread-safe: only the thread that created the window may touch it.
The game, meanwhile, is ordinary blocking code - it prints, then waits for an
answer. :class:`GuiBridge` lets the two live side by side without ever
sharing a widget:

* the game writes to :attr:`GuiBridge.stream` (a file-like object that rich
  prints to) and asks questions with :meth:`GuiBridge.request_line`, which
  *blocks the game thread* until the player answers;
* the window calls :meth:`GuiBridge.poll` every few milliseconds (Tk's
  ``after()``), gets everything the game sent since the last time, and
  answers with :meth:`GuiBridge.submit`.

When the player closes the window, :meth:`GuiBridge.close` wakes a waiting
question with ``EOFError`` - exactly what ``input()`` raises at the end of
piped input - so the game's usual "goodbye and clean up" path runs. Its
goodbye still reaches the transcript; once the window is gone for good,
:meth:`GuiBridge.detach` makes later output vanish instead of piling up.

Nothing here imports Tk, so it is tested with plain threads.
"""

from __future__ import annotations

import io
import queue
import re
import threading
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

__all__ = [
    "GuiBridge", "BridgeStream", "Prompt", "is_openable_url", "clean_text",
    "OUTPUT", "PROMPT", "CHOICES", "FINISHED", "OPENABLE_SCHEMES",
]

# The kinds of event the game side sends to the window (see GuiBridge.poll).
OUTPUT = "output"  # payload: text to feed to the terminal view
PROMPT = "prompt"  # payload: a Prompt the player should answer
CHOICES = "choices"  # payload: tuple of (key, label) buttons to show (empty = hide them)
FINISHED = "finished"  # payload: the game's exit code

# Links the window may open in a browser. (Anything else - file:, javascript:...
# - is refused, whatever put it in the transcript.)
OPENABLE_SCHEMES = ("http", "https")

Event = Tuple[str, Any]

# Lone UTF-16 surrogates (a model's JSON can carry an escape like "\ud83e" cut from the
# middle of an emoji). They aren't text: Tk on macOS and Linux refuses them outright.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def clean_text(text: str) -> str:
    """``text`` with any lone surrogate replaced by U+FFFD (the "unknown character" mark)."""
    return text if text.isascii() else _SURROGATE_RE.sub("\ufffd", text)


@dataclass(frozen=True)
class Prompt:
    """A question the game is waiting on."""

    id: int  # answers name the question they answer, so a late click can't answer the next one
    text: str  # as the game wrote it (it may contain rich markup)
    secret: bool = False  # mask what is typed (API keys)


class BridgeStream(io.TextIOBase):
    """The file the game's rich Console writes to. Every write becomes an OUTPUT event.

    It claims to be a terminal (``isatty()`` is True) and speaks UTF-8, so rich
    sends colours, box-drawing characters and live redraws, which the window's
    terminal view understands.
    """

    def __init__(self, bridge: "GuiBridge") -> None:
        super().__init__()
        self._bridge = bridge

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    @property
    def errors(self) -> str:  # type: ignore[override]
        return "strict"

    def write(self, text: str) -> int:  # type: ignore[override]
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        if text:
            self._bridge._put_output(clean_text(text))  # (what the window can't draw never reaches it)
        return len(text)

    def flush(self) -> None:
        pass  # every write is already on its way

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        pass  # the window owns the stream's lifetime; a stray close() must not break later prints

    @property
    def closed(self) -> bool:  # type: ignore[override]
        return False


class GuiBridge:
    """Carries output, questions, menu buttons and answers between the game and the window.

    Game thread: :attr:`stream`, :meth:`request_line`, :meth:`show_choices`,
    :meth:`open_url`, :meth:`finish`.
    Window thread: :meth:`poll`, :meth:`submit`, :meth:`close`, :meth:`detach`, :meth:`set_size`.
    """

    def __init__(self, *, columns: int = 100, rows: int = 32,
                 opener: Optional[Callable[[str], Any]] = None) -> None:
        self._events: "queue.SimpleQueue[Event]" = queue.SimpleQueue()
        self._cond = threading.Condition()
        self._request_lock = threading.Lock()  # one question at a time, even if two threads ask
        self._pending: Optional[Prompt] = None
        self._answer: Optional[str] = None
        self._next_id = 0
        self._closed = False
        self._detached = False
        self._opener = opener or webbrowser.open
        self.columns = int(columns)
        self.rows = int(rows)
        self.stream = BridgeStream(self)

    # -- state ------------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True once the window has closed (every later question raises EOFError)."""
        return self._closed

    @property
    def pending(self) -> Optional[Prompt]:
        """The question the game is waiting on right now, if any."""
        with self._cond:
            return self._pending

    # -- game side ----------------------------------------------------------------------------

    def _put_output(self, text: str) -> None:
        if not self._detached:  # (after detach() nobody will ever read it)
            self._events.put((OUTPUT, text))

    def request_line(self, prompt: str, *, secret: bool = False) -> str:
        """Ask the player a question and wait for the answer (blocks this thread).

        Returns what the player typed (or the key of the button they pressed).
        Raises ``EOFError`` if the window is closed - before or while waiting.
        """
        with self._request_lock:
            with self._cond:
                if self._closed:
                    raise EOFError("the game window was closed")
                self._next_id += 1
                request = Prompt(self._next_id, str(prompt), bool(secret))
                self._pending, self._answer = request, None
                self._events.put((PROMPT, request))
                try:
                    while self._answer is None and not self._closed:
                        self._cond.wait(0.5)  # (a timeout, so a lost wake-up can never hang the game)
                    answer = self._answer
                finally:
                    self._pending, self._answer = None, None
            if answer is None:
                raise EOFError("the game window was closed")
            return answer

    def show_choices(self, options: Sequence[Tuple[str, str]]) -> None:
        """Show a row of buttons for a menu ([] hides them). Called by ``UI.choose`` / ``UI.confirm``."""
        cleaned = tuple((str(key), str(label)) for key, label in (options or ()))
        if not self._closed:
            self._events.put((CHOICES, cleaned))

    def open_url(self, url: str) -> bool:
        """Open a web link in the player's browser. True if a browser was started.

        Only http(s) links are opened. Safe to call from any thread.
        """
        if not is_openable_url(url):
            return False
        try:
            return bool(self._opener(url))
        except Exception:
            return False

    def post_note(self, text: str) -> None:
        """Add a line of the window's own to the transcript (any thread; dropped once the window is gone)."""
        self._put_output(text)

    def finish(self, code: int) -> None:
        """The game is over (``code`` is its exit code)."""
        self._events.put((FINISHED, code))

    # -- window side --------------------------------------------------------------------------

    def poll(self, max_events: int = 10_000) -> List[Event]:
        """Everything the game sent since the last call, oldest first (never blocks).

        Consecutive OUTPUT events are joined into one, so a burst of prints
        costs a single feed.
        """
        events: List[Event] = []
        for _ in range(max_events):
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == OUTPUT and events and events[-1][0] == OUTPUT:
                events[-1] = (OUTPUT, events[-1][1] + payload)
            else:
                events.append((kind, payload))
        return events

    def submit(self, text: str, prompt_id: Optional[int] = None) -> bool:
        """Answer the question the game is waiting on.

        ``prompt_id`` (from :class:`Prompt`) makes sure the answer goes to the
        question the player saw; a stale one is refused. Returns False when no
        question is waiting (the answer is then *not* kept for later: an extra
        Enter must never skip the next question).
        """
        with self._cond:
            pending = self._pending
            if self._closed or pending is None or self._answer is not None:
                return False
            if prompt_id is not None and prompt_id != pending.id:
                return False
            self._answer = "" if text is None else str(text)
            self._cond.notify_all()
            return True

    def close(self) -> None:
        """The window is closing: wake any waiting question (it raises EOFError), refuse new ones.

        Output keeps arriving (the game's goodbye belongs in the transcript)
        until :meth:`detach`.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def detach(self) -> None:
        """The window is gone for good: close, and quietly drop everything the game prints from now on."""
        self.close()
        self._detached = True

    def set_size(self, columns: int, rows: int) -> None:
        """The terminal view's size in characters changed (the window records it here)."""
        self.columns = max(1, int(columns))
        self.rows = max(1, int(rows))


def is_openable_url(url: object) -> bool:
    """True for a well-formed http(s) link (the only kind the window opens)."""
    if not isinstance(url, str) or not url or len(url) > 4096 or not url.isprintable():
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme.lower() in OPENABLE_SCHEMES and bool(parts.netloc)
