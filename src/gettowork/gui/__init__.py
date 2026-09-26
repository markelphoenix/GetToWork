"""The game's own window, for players who launch Get To Work from Steam or by double-clicking.

Three small pieces, each readable on its own:

* :mod:`~gettowork.gui.terminal` - :class:`TerminalBuffer`, a pure-Python
  terminal model that turns what `rich` prints (colours, panels, progress
  bars that redraw themselves) into styled lines of text;
* :mod:`~gettowork.gui.bridge` - :class:`GuiBridge`, the thread-safe
  plumbing between the game (a worker thread) and the window;
* :mod:`~gettowork.gui.app` - :func:`run_gui`, the Tk window itself.

The game code doesn't know which front end it runs in: it talks to a
:class:`gettowork.ui.UI` either way. Importing this package never imports
Tk (only :func:`run_gui` does), so it is safe on a Python built without it.
"""

from __future__ import annotations

from .app import run_gui
from .bridge import GuiBridge
from .terminal import DEFAULT_STYLE, Style, TerminalBuffer

__all__ = ["run_gui", "GuiBridge", "TerminalBuffer", "Style", "DEFAULT_STYLE"]
