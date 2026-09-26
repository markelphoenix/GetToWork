"""Lets you run the game with ``python -m gettowork`` (the terminal version).

The game's own window - what Steam and a double-click open - is
``python -m gettowork.launcher`` (or the ``gettowork-gui`` command).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
