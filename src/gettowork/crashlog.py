"""Writing down what went wrong, for a bug report.

When something unexpected breaks, the player sees a short, friendly line -
and the technical details (the Python traceback) go into a small text file
in the game's ``logs`` folder, where a player can find it and attach it to a
bug report even after the window has closed:

* ``logs/crash.txt`` - the game itself hit an unexpected error
  (``cli.main`` catches it, says sorry and writes this);
* ``logs/gui-crash.txt`` - the game's window couldn't open, or broke
  (see :mod:`gettowork.gui.app`).

Each file holds the most recent problem only. Never raises.
"""

from __future__ import annotations

import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["CRASH_FILE", "crash_log_path", "write_crash_report"]

CRASH_FILE = "crash.txt"


def crash_log_path(file_name: str = CRASH_FILE) -> Path:
    """``<config dir>/logs/<file_name>``."""
    from .config import config_dir

    return config_dir() / "logs" / file_name


def write_crash_report(context: str, exc: Optional[BaseException] = None, *,
                       file_name: str = CRASH_FILE) -> Optional[Path]:
    """Write `context` plus the traceback of `exc` (or of the exception being handled).

    Returns the file's path, or None if it couldn't be written. Never raises.
    """
    try:
        from . import __version__

        if exc is not None:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            details = traceback.format_exc()
        path = crash_log_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"Get To Work {__version__}: {context}\n"
            f"When: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Python {platform.python_version()} on {platform.platform()}"
            f" ({'built game' if getattr(sys, 'frozen', False) else 'from source'})\n\n{details}\n",
            encoding="utf-8",
        )
        return path
    except Exception:
        return None
