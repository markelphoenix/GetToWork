"""Starting the game the way players do: from Steam, or with a double-click.

A built copy of Get To Work has two programs side by side:

* **GetToWork** (windowed) - what Steam launches and what you double-click.
  It runs :func:`gui_main`, which opens the game's own window
  (:mod:`gettowork.gui.app`) and plays the normal game inside it: the
  guided setup (hardware check, picking and downloading a model, optional
  Jev), then the game itself. ``pip install`` gives the same thing as the
  ``gettowork-gui`` command, and ``python -m gettowork.launcher`` runs it
  from a source checkout.
* **gettowork-cli** (console) - the terminal version (:func:`gettowork.cli.main`)
  for developers and tinkerers (``gettowork`` when installed with pip).

Double-clicking the *console* program on Windows opens a console window
just for it, and Windows closes that window the moment the program ends -
taking the goodbye (or an error message) with it before anyone can read it.
:func:`console_closes_on_exit` spots exactly that case, and ``cli.main``
then asks for Enter before closing (:func:`wait_before_closing`).

Nothing here imports Tk or the game itself at import time, so this module
is cheap to import from ``cli.py``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterator, Optional

__all__ = [
    "gui_main",
    "console_closes_on_exit",
    "wait_before_closing",
    "restore_system_library_path",
    "SELFTEST_FLAG",
    "CLOSE_PROMPT",
]

SELFTEST_FLAG = "--gui-selftest"  # CI: play a scripted game in the window, exit 0 if it reached work
CLOSE_PROMPT = "Press Enter to close this window"
HOME_ENV = "GETTOWORK_HOME"  # where settings, models and logs live (see config.py)


# ---------------------------------------------------------------------------
# The windowed game
# ---------------------------------------------------------------------------


def gui_main(argv: Optional[list[str]] = None) -> int:
    """Open the game's window and play; returns the exit code.

    The entry point of the windowed executable (Steam, double-click) and of
    the ``gettowork-gui`` command. ``argv`` are the game's usual options
    (default: the process's own), plus ``--gui-selftest`` for CI: a scripted
    game with the pretend model, exit code 0 if it got to work (see
    :func:`gettowork.gui.app.run_gui`).

    The self-test plays in a throwaway settings folder (unless
    ``GETTOWORK_HOME`` is set), so running it never changes - or depends
    on - the player's own saved choices.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    selftest = SELFTEST_FLAG in args
    args = [arg for arg in args if arg != SELFTEST_FLAG]

    from .gui import app  # imported here: it's only needed once we know we're opening the window

    if not selftest:
        return app.run_gui(args)
    with _throwaway_home():
        return app.run_gui(args, selftest=True)


@contextlib.contextmanager
def _throwaway_home() -> Iterator[None]:
    """Point GETTOWORK_HOME at a temporary folder for a while (unless it is already set)."""
    if os.environ.get(HOME_ENV):
        yield
        return
    folder = tempfile.mkdtemp(prefix="gettowork-selftest-")
    os.environ[HOME_ENV] = folder
    try:
        yield
    finally:
        os.environ.pop(HOME_ENV, None)
        shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Programs the built game starts (Linux)
# ---------------------------------------------------------------------------

# The library search path variable, per system (AIX calls it LIBPATH).
_LIBRARY_PATH_VARS = ("LD_LIBRARY_PATH", "LIBPATH")


def restore_system_library_path(environ: Optional[Any] = None, *, frozen: Optional[bool] = None,
                                platform: Optional[str] = None) -> bool:
    """Built game on Linux: programs the game starts get the system's own libraries back.

    The built game's start-up program (PyInstaller's bootloader) points
    ``LD_LIBRARY_PATH`` at the libraries bundled inside the game (its
    ``_internal`` folder) and keeps the player's original value in
    ``LD_LIBRARY_PATH_ORIG``. The game itself needs that only while it starts
    (the system's loader reads the variable once, when a program starts), but
    every program the game launches inherits it: the llama.cpp engine - and
    through it the graphics driver - the web browser opened for a link, and
    hardware checks such as ``nvidia-smi`` would then load the game's copies
    of system libraries instead of the system's own, which can stop a
    graphics driver or the browser from starting. So this puts the original
    value back (or removes the variable if there was none), exactly as
    PyInstaller's documentation recommends.

    Only in a built game (``sys.frozen``), never on Windows or macOS (which
    don't use this variable for it). ``environ``, ``frozen`` and ``platform``
    are for tests. Never raises; True if something changed.
    """
    try:
        env = os.environ if environ is None else environ
        if not (getattr(sys, "frozen", False) if frozen is None else frozen):
            return False
        if (platform or sys.platform).startswith(("win", "darwin", "cygwin")):
            return False
        changed = False
        for name in _LIBRARY_PATH_VARS:
            original = env.get(f"{name}_ORIG")
            if original is not None:
                if env.get(name) != original:
                    env[name] = original
                    changed = True
                del env[f"{name}_ORIG"]
            elif name in env and _points_into_bundle(env[name]):
                del env[name]  # the bootloader made it up from nothing
                changed = True
        return changed
    except Exception:
        return False


def _points_into_bundle(value: str) -> bool:
    """Does a library path list name the built game's own library folder (``sys._MEIPASS``)?"""
    bundle = getattr(sys, "_MEIPASS", None)
    if not bundle:
        return False  # not a PyInstaller build: leave it alone
    folders = [os.path.normpath(part) for part in value.split(os.pathsep) if part]
    return os.path.normpath(str(bundle)) in folders


# ---------------------------------------------------------------------------
# The console version, double-clicked on Windows
# ---------------------------------------------------------------------------


def console_closes_on_exit() -> bool:
    """Will this console window vanish the moment the game exits?

    True only for the built game's console program (``gettowork-cli.exe``)
    double-clicked in Windows: Windows then opens a console window just for
    it, so it is the *only* program attached to that console. Started from
    Command Prompt, PowerShell or Windows Terminal, the shell is attached
    too and the window stays open afterwards. Never true when running from
    source, in tests, or with piped input. Never raises.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    if not _isatty(sys.stdin):
        return False  # piped or redirected input: nobody is there to press Enter
    try:
        import ctypes

        # GetConsoleProcessList fills in the IDs of the programs attached to
        # our console and returns how many there are (a two-slot list is
        # plenty: we only need to know whether it's more than one).
        processes = (ctypes.c_ulong * 2)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(processes, 2)  # type: ignore[attr-defined]
    except Exception:
        return False
    return count == 1


def wait_before_closing(input_fn: Optional[Callable[[str], Any]] = None) -> None:
    """Keep the console window open until the player presses Enter (never raises).

    ``input_fn`` replaces the built-in :func:`input` (for tests).
    """
    try:
        (input_fn or input)(f"\n{CLOSE_PROMPT}...")
    except (EOFError, KeyboardInterrupt, OSError, ValueError):  # input already closed, or Ctrl+C: just go
        pass


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


if __name__ == "__main__":
    raise SystemExit(gui_main())
