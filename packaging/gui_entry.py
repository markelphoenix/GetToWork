"""Entry point of the windowed program, ``GetToWork`` - what Steam and a double-click start.

PyInstaller turns this file into ``GetToWork`` (``GetToWork.exe`` on
Windows, ``Get To Work.app`` on macOS), which opens the game's own window
(:func:`gettowork.launcher.gui_main`). ``pip install`` users get the same
thing as the ``gettowork-gui`` command. The build recipe is
``packaging/gettowork.spec``.
"""

import multiprocessing
import os
import sys


def _give_windowless_process_somewhere_to_print() -> None:
    """A windowed program on Windows has no console, so ``sys.stdout``/``sys.stderr`` are None.

    Anything that writes to them directly (a library's warning, a progress
    bar) would then crash the game. Pointing them at the "null device" just
    throws that text away instead - the game shows everything important in
    its window anyway.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115 - lives as long as the game


def main() -> int:
    # Harmless today (the game only uses threads), but required on Windows if a
    # frozen build ever starts a worker process: without it, each worker would
    # relaunch the whole game.
    multiprocessing.freeze_support()
    _give_windowless_process_somewhere_to_print()

    from gettowork.launcher import gui_main, restore_system_library_path

    # Linux: the engine, the browser (for links) and hardware checks must load
    # the system's libraries, not the copies bundled inside the game.
    restore_system_library_path()
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
