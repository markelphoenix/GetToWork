"""A tiny terminal emulator: turns what `rich` prints into styled lines of text.

The game prints everything through `rich`, which speaks the language of real
terminals: plain text mixed with *escape sequences* - "ESC [ 1 m" means
"bold from here", "ESC [ 1 A" means "move the cursor up one line", and so on.
The game's own window (``gui/app.py``) has no terminal to interpret them, so
this module plays the part of one. It is pure Python (no Tk), which keeps it
easy to test and to read.

How it works, in one paragraph: text arrives in chunks through :meth:`feed`.
A small *state machine* walks through it; ordinary characters are written into
a grid of cells (one list of characters and one list of styles per line) at
the cursor, and escape sequences move the cursor, erase parts of lines or
change the current :class:`Style`. Every line that changes is remembered as
*dirty*, so the window only redraws those - a progress bar redrawn ten times
a second costs one line, not the whole transcript.

What is understood (everything `rich` emits, and a little more):

* printable text, including wide characters (CJK, emoji) that take two cells
  and zero-width ones (accents, joiners) that attach to the previous cell;
* ``\\n`` (treated as a new line *and* a carriage return, like a terminal in
  its normal "cooked" mode), ``\\r``, ``\\b``, ``\\t``;
* CSI sequences: cursor movement ``A B C D E F G H f d``, erasing ``J K X``,
  styles ``m`` (bold, dim, italic, underline, reverse, strike-through, the 16
  standard colours, the 256-colour palette and 24-bit "truecolor");
* OSC 8 hyperlinks (the text in between carries the URL);
* everything else - private modes such as "hide the cursor", window titles,
  unknown sequences - is dropped safely, even when split across chunks.

Rich's live displays (spinners, progress bars) redraw themselves by moving the
cursor up and erasing lines; that works here exactly as in a terminal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "Color",
    "Style",
    "DEFAULT_STYLE",
    "Run",
    "TerminalBuffer",
    "xterm_color",
    "DEFAULT_MAX_LINES",
]

# A colour is None (the window's default), an int (0-15: the 16 standard
# colours, 16-255: the xterm 256-colour palette) or a "#rrggbb" string.
Color = Union[int, str, None]

DEFAULT_MAX_LINES = 10_000
TAB_WIDTH = 8
MAX_CSI_LENGTH = 64  # real sequences are a few characters long
MAX_OSC_LENGTH = 8192  # a hyperlink with a very long URL still fits
MAX_URL_LENGTH = 4096
MAX_COLUMN = 2000  # cursor moves stop here (a real terminal stops at its right edge)


@dataclass(frozen=True)
class Style:
    """How a cell looks. Frozen (and so hashable): the window keys its Tk tags on it."""

    fg: Color = None
    bg: Color = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    reverse: bool = False
    strike: bool = False
    link: Optional[str] = None  # the URL of an OSC 8 hyperlink


DEFAULT_STYLE = Style()

Run = Tuple[str, Style]  # a stretch of text in one style

# ---------------------------------------------------------------------------
# Character widths
# ---------------------------------------------------------------------------

# Characters that are always exactly one cell wide: ASCII, Latin, general
# punctuation, arrows and the box-drawing / block characters rich draws its
# panels, tables and bars with. Text made only of these takes the fast path.
_NARROW_RUN_RE = re.compile(
    "[\x20-\x7e\xa0-\xac\xae-˿‐-‧‰-⁞←-⇿─-▟]+"
)

try:  # rich >= 14 measures whole grapheme clusters ("⚠️", "👨‍👩‍👧"): match it exactly.
    from rich.cells import split_graphemes as _rich_split_graphemes
except ImportError:  # pragma: no cover - older rich
    _rich_split_graphemes = None

try:
    from rich.cells import get_character_cell_size as _rich_char_width
except ImportError:  # pragma: no cover - very old rich
    _rich_char_width = None


def _char_width(ch: str) -> int:
    """How many cells one character takes: 0 (combining/joiner), 1 or 2."""
    if _rich_char_width is not None:
        return _rich_char_width(ch)
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _clusters(text: str) -> Iterable[Tuple[str, int]]:
    """Split printable text into (cluster, width) pairs, the way rich measures it."""
    if _rich_split_graphemes is not None:
        spans, _total = _rich_split_graphemes(text)
        for start, end, width in spans:
            yield text[start:end], width
        return
    previous_joiner = False
    for ch in text:  # pragma: no cover - older rich
        width = 0 if previous_joiner else _char_width(ch)
        previous_joiner = ch == "‍"
        yield ch, width


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


def xterm_color(index: int) -> str:
    """The standard xterm colour for a 256-palette index 16-255, as "#rrggbb".

    16-231 are a 6x6x6 colour cube, 232-255 a grey ramp. (0-15 are the
    theme's own 16 colours: the window picks those.)
    """
    if 16 <= index <= 231:
        index -= 16
        levels = (0, 95, 135, 175, 215, 255)
        r, g, b = levels[index // 36], levels[(index // 6) % 6], levels[index % 6]
        return f"#{r:02x}{g:02x}{b:02x}"
    if 232 <= index <= 255:
        grey = 8 + (index - 232) * 10
        return f"#{grey:02x}{grey:02x}{grey:02x}"
    raise ValueError(f"not an extended palette index: {index}")


# ---------------------------------------------------------------------------
# The parser's states
# ---------------------------------------------------------------------------

_GROUND = 0  # ordinary text
_ESCAPE = 1  # just saw ESC
_ESCAPE_INTERMEDIATE = 2  # ESC followed by " " .. "/" (e.g. a charset choice "ESC ( B")
_CSI = 3  # ESC [ ... waiting for the final byte
_CSI_IGNORE = 4  # an over-long or malformed CSI: skip to its final byte
_OSC = 5  # ESC ] ... waiting for BEL or ST
_OSC_ESC = 6  # saw ESC inside an OSC (the start of ST, "ESC \\")
_STRING = 7  # DCS / SOS / PM / APC: ignored up to ST
_STRING_ESC = 8

# Everything that isn't plain printable text: C0 controls, DEL and the C1 range.
_SPECIAL_RE = re.compile("[\x00-\x1f\x7f-\x9f]")


class TerminalBuffer:
    """A screen model fed with terminal output: lines of styled cells plus a cursor.

    ``feed(text)`` interprets the text; ``lines`` gives every line as a list of
    ``(text, Style)`` runs; ``take_dirty()`` returns (and forgets) which lines
    changed since the last call, so a view can redraw only those.

    Scrollback is bounded: when there are more than ``max_lines`` lines, the
    oldest ones are dropped (a few hundred at a time, so a view isn't asked to
    delete one line per line printed). ``trimmed_total`` counts every line
    ever dropped, so a view can drop the same number from its top; indices in
    ``dirty`` always refer to the lines as they are *now*.

    ``rows`` is the height of the "screen" - the last ``rows`` lines - which
    absolute cursor positioning (``ESC [ row ; col H``) and "clear the screen"
    refer to. Relative moves (cursor up) may reach any line still in the
    buffer: rich erases exactly the lines it drew, even if the window has been
    made shorter in the meantime.
    """

    def __init__(self, *, max_lines: int = DEFAULT_MAX_LINES, rows: int = 24) -> None:
        self.max_lines = max(2, int(max_lines))
        self.rows = max(1, int(rows))
        # One list of characters and one of styles per line. A wide character
        # sits in its first cell; the second holds "" (a continuation).
        self._chars: List[List[str]] = [[]]
        self._styles: List[List[Style]] = [[]]
        self._row = 0
        self._col = 0
        self._style = DEFAULT_STYLE
        self._saved: Optional[Tuple[int, int, Style]] = None
        self._dirty: set[int] = {0}
        self.trimmed_total = 0
        # Parser state, kept between feed() calls so split sequences work.
        self._state = _GROUND
        self._seq: List[str] = []
        self._seq_len = 0

    # -- reading ------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chars)

    @property
    def cursor(self) -> Tuple[int, int]:
        """(line index, column) where the next character will be written."""
        return self._row, self._col

    @property
    def style(self) -> Style:
        """The style new text is written in."""
        return self._style

    @property
    def dirty(self) -> set[int]:
        """Indices of the lines changed since the last :meth:`take_dirty` (a copy)."""
        return set(self._dirty)

    def take_dirty(self) -> List[int]:
        """The changed lines' indices, in order - and start tracking afresh."""
        dirty = sorted(i for i in self._dirty if 0 <= i < len(self._chars))
        self._dirty = set()
        return dirty

    def line(self, index: int, *, substitute: Optional[Callable[[str, int], Optional[str]]] = None) -> List[Run]:
        """One line as ``(text, Style)`` runs (consecutive cells of one style joined).

        ``substitute(cluster, width)``, if given, is asked about every
        non-ASCII character (grapheme cluster) with the number of cells it
        takes; a string it returns is drawn instead (it should be exactly
        ``width`` characters, so the columns stay where rich put them), None
        keeps the character. The window uses it for characters its toolkit
        can't draw safely.
        """
        chars, styles = self._chars[index], self._styles[index]
        runs: List[Run] = []
        current: Optional[Style] = None
        parts: List[str] = []
        count = len(chars)
        for position, (ch, style) in enumerate(zip(chars, styles)):
            if not ch:  # the second half of a wide character
                continue
            if substitute is not None and not ch.isascii():
                width = 1
                while position + width < count and not chars[position + width]:
                    width += 1
                try:
                    replacement = substitute(ch, width)
                except Exception:
                    replacement = None
                if replacement is not None:
                    ch = replacement
                    if not ch:
                        continue
            if style is current or style == current:
                parts.append(ch)
                continue
            if parts:
                runs.append(("".join(parts), current))  # type: ignore[arg-type]
            current, parts = style, [ch]
        if parts:
            runs.append(("".join(parts), current))  # type: ignore[arg-type]
        return runs

    @property
    def lines(self) -> List[List[Run]]:
        """Every line as ``(text, Style)`` runs."""
        return [self.line(i) for i in range(len(self._chars))]

    def line_text(self, index: int) -> str:
        """One line's plain text (trailing spaces kept)."""
        return "".join(self._chars[index])

    def plain_lines(self) -> List[str]:
        """Every line's plain text, trailing spaces removed."""
        return ["".join(chars).rstrip() for chars in self._chars]

    def text(self) -> str:
        """The whole transcript as plain text (no trailing blank lines)."""
        return "\n".join(self.plain_lines()).rstrip("\n")

    # -- feeding ------------------------------------------------------------------------

    def feed(self, data: str) -> None:
        """Interpret a chunk of terminal output. Never raises on odd input."""
        if not data:
            return
        i, n = 0, len(data)
        while i < n:
            if self._state == _GROUND:
                match = _SPECIAL_RE.search(data, i)
                end = match.start() if match else n
                if end > i:
                    self._write(data[i:end])
                if match is None:
                    return
                self._control(data[end])
                i = end + 1
                continue
            self._escape_char(data[i])
            i += 1

    def reset(self) -> None:
        """Forget everything: one empty line, cursor home, default style.

        Counted as trimming every old line, so a view clears itself too.
        """
        self.trimmed_total += len(self._chars)
        self._chars, self._styles = [[]], [[]]
        self._row = self._col = 0
        self._style = DEFAULT_STYLE
        self._saved = None
        self._dirty = {0}
        self._state = _GROUND
        self._seq, self._seq_len = [], 0

    # -- control characters ---------------------------------------------------------------

    def _control(self, ch: str) -> None:
        if ch == "\x1b":
            self._begin(_ESCAPE)
        elif ch in "\n\x0b\x0c":
            self._newline()
        elif ch == "\r":
            self._col = 0
        elif ch == "\b":
            self._col = max(0, self._col - 1)
        elif ch == "\t":
            self._col = min((self._col // TAB_WIDTH + 1) * TAB_WIDTH, MAX_COLUMN)
        # Everything else (BEL, NUL, SO/SI, DEL, the C1 range...) is ignored.

    def _begin(self, state: int) -> None:
        self._state = state
        self._seq = []
        self._seq_len = 0

    def _escape_char(self, ch: str) -> None:
        """One character while inside an escape sequence."""
        state = self._state
        if state == _ESCAPE:
            self._after_escape(ch)
        elif state in (_CSI, _CSI_IGNORE):
            self._csi_char(ch)
        elif state == _OSC:
            if ch == "\x07" or ch == "\x9c":
                self._osc_dispatch()
                self._state = _GROUND
            elif ch == "\x1b":
                self._state = _OSC_ESC
            elif ch in "\x18\x1a":  # CAN / SUB cancel a sequence
                self._state = _GROUND
            else:
                self._seq_len += 1
                if self._seq_len <= MAX_OSC_LENGTH:
                    self._seq.append(ch)
        elif state == _OSC_ESC:
            self._osc_dispatch()
            self._state = _GROUND
            if ch != "\\":  # not ST after all: the ESC starts a new sequence
                self._begin(_ESCAPE)
                self._after_escape(ch)
        elif state == _STRING:
            if ch == "\x1b":
                self._state = _STRING_ESC
            elif ch in "\x07\x9c\x18\x1a":
                self._state = _GROUND
        elif state == _STRING_ESC:
            self._state = _GROUND
            if ch != "\\":
                self._begin(_ESCAPE)
                self._after_escape(ch)
        elif state == _ESCAPE_INTERMEDIATE:
            if "\x30" <= ch <= "\x7e":
                self._state = _GROUND  # the charset (or similar) choice is complete: ignored
            elif ch == "\x1b":
                self._begin(_ESCAPE)
            elif not ("\x20" <= ch <= "\x2f"):
                self._state = _GROUND
                if ch < "\x20":
                    self._control(ch)
        else:  # pragma: no cover - defensive
            self._state = _GROUND

    def _after_escape(self, ch: str) -> None:
        if ch == "[":
            self._begin(_CSI)
        elif ch == "]":
            self._begin(_OSC)
        elif ch in "PX^_":
            self._begin(_STRING)
        elif "\x20" <= ch <= "\x2f":
            self._state = _ESCAPE_INTERMEDIATE
        elif ch == "\x1b":
            self._begin(_ESCAPE)
        else:
            self._state = _GROUND
            if ch == "7":
                self._saved = (self._row, self._col, self._style)
            elif ch == "8":
                self._restore_cursor(with_style=True)
            elif ch == "D":  # index: down one line
                self._newline(carriage_return=False)
            elif ch == "E":  # next line
                self._newline()
            elif ch == "M":  # reverse index: up one line
                self._row = max(0, self._row - 1)
            elif ch == "c":  # full reset: default style (the transcript itself is kept)
                self._style = DEFAULT_STYLE
            elif ch < "\x20":
                self._control(ch)
            # Anything else ("ESC =", "ESC >", a stray "ESC \\"...) is ignored.

    # -- CSI --------------------------------------------------------------------------------

    def _csi_char(self, ch: str) -> None:
        if "\x40" <= ch <= "\x7e":  # the final byte
            if self._state == _CSI:
                self._csi_dispatch("".join(self._seq), ch)
            self._state = _GROUND
            return
        if ch == "\x1b":
            self._begin(_ESCAPE)
            return
        if ch in "\x18\x1a":
            self._state = _GROUND
            return
        if ch < "\x20" or "\x7f" <= ch <= "\x9f":
            self._control(ch)  # terminals obey control characters even mid-sequence
            return
        if not ("\x20" <= ch <= "\x3f"):
            self._state = _CSI_IGNORE  # not a parameter/intermediate byte: malformed
            return
        self._seq.append(ch)
        self._seq_len += 1
        if self._seq_len > MAX_CSI_LENGTH:
            self._state = _CSI_IGNORE

    def _csi_dispatch(self, body: str, final: str) -> None:
        if body[:1] in ("?", ">", "<", "="):
            return  # private modes (hide/show cursor, bracketed paste...): nothing to do here
        if any("\x20" <= c <= "\x2f" for c in body):
            return  # sequences with intermediate bytes (cursor shape...): ignored
        if final == "m":
            self._sgr(body)
            return
        params = _parse_params(body)

        def arg(index: int = 0, default: int = 1) -> int:
            value = params[index] if index < len(params) else None
            return default if value is None or (value == 0 and default == 1) else value

        if final == "A":
            self._row = max(0, self._row - arg())
        elif final == "B":
            self._move_down(arg())
        elif final in ("C", "a"):
            self._col = min(self._col + arg(), MAX_COLUMN)
        elif final == "D":
            self._col = max(0, self._col - arg())
        elif final == "E":
            self._move_down(arg())
            self._col = 0
        elif final == "F":
            self._row = max(0, self._row - arg())
            self._col = 0
        elif final in ("G", "`"):
            self._col = min(arg() - 1, MAX_COLUMN)
        elif final in ("H", "f"):
            self._goto_screen(arg(0), arg(1))
        elif final == "d":
            self._goto_screen(arg(0), self._col + 1)
        elif final == "K":
            self._erase_line(arg(0, 0))
        elif final == "J":
            self._erase_screen(arg(0, 0))
        elif final == "X":
            self._erase_chars(arg())
        elif final == "s":
            self._saved = (self._row, self._col, self._style)
        elif final == "u":
            self._restore_cursor(with_style=False)
        # Scrolling regions, insert/delete, device reports...: not needed, ignored.

    # -- SGR (styles) ----------------------------------------------------------------------

    def _sgr(self, body: str) -> None:
        groups = body.split(";") if body else [""]
        style = self._style
        i = 0
        while i < len(groups):
            group = groups[i]
            i += 1
            sub = group.split(":")
            try:
                code = int(sub[0]) if sub[0] else 0
            except ValueError:
                return  # garbage: ignore the rest of the sequence
            if code == 0:
                style = Style(link=style.link)  # a hyperlink outlives a style reset
            elif code == 1:
                style = replace(style, bold=True)
            elif code == 2:
                style = replace(style, dim=True)
            elif code == 3:
                style = replace(style, italic=True)
            elif code == 4:
                style = replace(style, underline=not (len(sub) > 1 and sub[1] == "0"))
            elif code == 7:
                style = replace(style, reverse=True)
            elif code == 9:
                style = replace(style, strike=True)
            elif code == 21:
                style = replace(style, underline=True)  # double underline: close enough
            elif code == 22:
                style = replace(style, bold=False, dim=False)
            elif code == 23:
                style = replace(style, italic=False)
            elif code == 24:
                style = replace(style, underline=False)
            elif code == 27:
                style = replace(style, reverse=False)
            elif code == 29:
                style = replace(style, strike=False)
            elif 30 <= code <= 37:
                style = replace(style, fg=code - 30)
            elif code == 39:
                style = replace(style, fg=None)
            elif 40 <= code <= 47:
                style = replace(style, bg=code - 40)
            elif code == 49:
                style = replace(style, bg=None)
            elif 90 <= code <= 97:
                style = replace(style, fg=code - 90 + 8)
            elif 100 <= code <= 107:
                style = replace(style, bg=code - 100 + 8)
            elif code in (38, 48, 58):
                if len(sub) > 1:  # colon form: 38:5:n or 38:2:[colourspace:]r:g:b
                    color = _extended_color(sub[1:], colon=True)
                else:  # semicolon form: 38;5;n or 38;2;r;g;b
                    color, used = _extended_color_from(groups, i)
                    i += used
                if color is not _INVALID and code != 58:  # 58 = underline colour: ignored
                    style = replace(style, fg=color) if code == 38 else replace(style, bg=color)
            # 5/6 blink, 8 conceal, 53 overline...: ignored
        self._style = style

    # -- OSC -------------------------------------------------------------------------------

    def _osc_dispatch(self) -> None:
        if self._seq_len > MAX_OSC_LENGTH:
            return  # too long to be anything we understand
        data = "".join(self._seq)
        command, _, rest = data.partition(";")
        if command != "8":
            return  # window titles, clipboard, colours...: ignored
        _params, _, url = rest.partition(";")
        url = url.strip()
        if url and len(url) <= MAX_URL_LENGTH and url.isprintable():
            self._style = replace(self._style, link=url)
        else:
            self._style = replace(self._style, link=None)

    # -- writing -----------------------------------------------------------------------------

    def _write(self, text: str) -> None:
        """Put printable text at the cursor, overwriting what is there."""
        row = self._row
        chars, styles = self._chars[row], self._styles[row]
        col = self._col
        style = self._style
        if col > len(chars):  # the cursor was moved past the end of the line: pad with spaces
            pad = col - len(chars)
            chars.extend(" " * pad)
            styles.extend([DEFAULT_STYLE] * pad)
        if _NARROW_RUN_RE.fullmatch(text):
            end = col + len(text)
            _break_wide(chars, col)
            _break_wide(chars, end)
            chars[col:end] = text
            styles[col:end] = [style] * len(text)
            self._col = end
        else:
            for cluster, width in _clusters(text):
                col = self._put(chars, styles, col, cluster, width, style)
            self._col = col
        self._dirty.add(row)

    def _put(self, chars: List[str], styles: List[Style], col: int, cluster: str, width: int,
             style: Style) -> int:
        """Write one character cluster at ``col``; returns the next column."""
        if width <= 0:  # an accent / joiner: belongs to the previous character
            target = col - 1
            while target >= 0 and target < len(chars) and chars[target] == "":
                target -= 1
            if 0 <= target < len(chars):
                chars[target] += cluster
            return col
        width = min(width, 2)
        end = col + width
        _break_wide(chars, col)
        _break_wide(chars, end)
        cells = [cluster] + [""] * (width - 1)
        chars[col:end] = cells
        styles[col:end] = [style] * width
        return end

    def _newline(self, *, carriage_return: bool = True) -> None:
        self._row += 1
        if carriage_return:
            self._col = 0
        self._ensure_row()

    def _move_down(self, count: int) -> None:
        # A terminal stops at the bottom of its screen; lines below the last one
        # exist there (blank), so moving onto them makes them real here.
        bottom = max(len(self._chars) - 1, self._screen_top() + self.rows - 1)
        self._row = min(self._row + count, bottom)
        self._ensure_row()

    def _ensure_row(self) -> None:
        while self._row >= len(self._chars):
            self._chars.append([])
            self._styles.append([])
            self._dirty.add(len(self._chars) - 1)
        if len(self._chars) > self.max_lines:
            self._trim()

    def _trim(self) -> None:
        """Drop the oldest lines, keeping at most ``max_lines`` (a chunk at a time)."""
        chunk = max(1, self.max_lines // 50)
        keep = max(1, self.max_lines - chunk + 1)
        drop = len(self._chars) - keep
        if drop <= 0:
            return
        del self._chars[:drop]
        del self._styles[:drop]
        self._row = max(0, self._row - drop)
        self._dirty = {i - drop for i in self._dirty if i >= drop}
        if self._saved is not None:
            row, col, style = self._saved
            self._saved = (max(0, row - drop), col, style)
        self.trimmed_total += drop

    def _screen_top(self) -> int:
        return max(0, max(len(self._chars), self._row + 1) - self.rows)

    def _goto_screen(self, row: int, col: int) -> None:
        row = min(max(1, row), self.rows)
        self._row = self._screen_top() + row - 1
        self._col = min(max(0, col - 1), MAX_COLUMN)
        self._ensure_row()

    def _restore_cursor(self, *, with_style: bool) -> None:
        """ESC 8 restores the position and style saved by ESC 7; CSI u only the position."""
        if self._saved is None:
            self._col = 0
            return
        row, col, style = self._saved
        self._row, self._col = min(row, len(self._chars) - 1), col
        if with_style:
            self._style = style

    # -- erasing -----------------------------------------------------------------------------

    def _erase_line(self, mode: int) -> None:
        row = self._row
        chars, styles = self._chars[row], self._styles[row]
        col = self._col
        if mode == 0:  # cursor to end of line
            if col < len(chars):
                _break_wide(chars, col)
                del chars[col:]
                del styles[col:]
        elif mode == 1:  # start of line to cursor (inclusive)
            end = min(col + 1, len(chars))
            _break_wide(chars, end)
            chars[:end] = " " * end
            styles[:end] = [DEFAULT_STYLE] * end
        elif mode == 2:  # the whole line
            chars.clear()
            styles.clear()
        else:
            return
        self._dirty.add(row)

    def _erase_chars(self, count: int) -> None:
        chars, styles = self._chars[self._row], self._styles[self._row]
        start = self._col
        end = min(start + count, len(chars))
        if start >= end:
            return
        _break_wide(chars, start)
        _break_wide(chars, end)
        chars[start:end] = " " * (end - start)
        styles[start:end] = [DEFAULT_STYLE] * (end - start)
        self._dirty.add(self._row)

    def _erase_screen(self, mode: int) -> None:
        top = self._screen_top()
        if mode == 0:  # cursor to end of screen
            self._erase_line(0)
            for row in range(self._row + 1, len(self._chars)):
                self._clear_row(row)
        elif mode == 1:  # start of screen to cursor
            for row in range(top, self._row):
                self._clear_row(row)
            self._erase_line(1)
        elif mode == 2:
            # "Clear the screen". Like many modern terminals, push what is on
            # the screen up into the scrollback instead of destroying it: the
            # transcript is worth keeping. The cursor keeps its place on the
            # (now empty) screen.
            last_used = max((r for r in range(top, len(self._chars)) if self._chars[r]), default=None)
            if last_used is None:
                return
            offset = self._row - top
            new_top = last_used + 1
            while len(self._chars) < new_top + self.rows:
                self._chars.append([])
                self._styles.append([])
                self._dirty.add(len(self._chars) - 1)
            for row in range(new_top, len(self._chars)):
                self._clear_row(row)
            self._row = new_top + min(offset, self.rows - 1)
            if len(self._chars) > self.max_lines:
                self._trim()
        # mode 3 ("erase the scrollback") is ignored: the transcript is kept.

    def _clear_row(self, row: int) -> None:
        if self._chars[row]:
            self._chars[row].clear()
            self._styles[row].clear()
            self._dirty.add(row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INVALID = object()


def _break_wide(chars: List[str], col: int) -> None:
    """Before overwriting from ``col``: don't leave half of a wide character behind.

    If ``col`` is the second half of a wide character, its first half becomes
    a space (the terminal rule). Called for both edges of a write.
    """
    if 0 < col < len(chars) and chars[col] == "":
        chars[col - 1] = " "
        chars[col] = " "


def _parse_params(body: str) -> List[Optional[int]]:
    """"12;;3" -> [12, None, 3]. Sub-parameters (after ":") are ignored here."""
    params: List[Optional[int]] = []
    for part in body.split(";") if body else []:
        part = part.split(":", 1)[0]
        try:
            params.append(min(int(part), 100_000) if part else None)
        except ValueError:
            params.append(None)
    return params


def _byte(text: str) -> Optional[int]:
    try:
        value = int(text)
    except ValueError:
        return None
    return value if 0 <= value <= 255 else None


def _extended_color(parts: Sequence[str], *, colon: bool) -> object:
    """The colon form's parts after "38": ["5", n] or ["2", (colourspace,) r, g, b]."""
    if not parts:
        return _INVALID
    if parts[0] == "5" and len(parts) >= 2:
        index = _byte(parts[1])
        return _INVALID if index is None else index
    if parts[0] == "2":
        rgb = list(parts[1:])
        if colon and len(rgb) >= 4:
            rgb = rgb[1:4]  # 38:2:<colourspace>:r:g:b
        values = [_byte(v) for v in rgb[:3]]
        if len(values) == 3 and all(v is not None for v in values):
            r, g, b = values
            return f"#{r:02x}{g:02x}{b:02x}"
    return _INVALID


def _extended_color_from(groups: Sequence[str], start: int) -> Tuple[object, int]:
    """The semicolon form: parse from groups[start:]; returns (colour, groups used)."""
    if start >= len(groups):
        return _INVALID, 0
    kind = groups[start]
    if kind == "5":
        if start + 1 < len(groups):
            index = _byte(groups[start + 1])
            return (_INVALID if index is None else index), 2
        return _INVALID, 1
    if kind == "2":
        values = [_byte(v) for v in groups[start + 1:start + 4]]
        if len(values) == 3 and all(v is not None for v in values):
            r, g, b = values
            return f"#{r:02x}{g:02x}{b:02x}", 4
        return _INVALID, min(4, len(groups) - start)
    return _INVALID, 1
