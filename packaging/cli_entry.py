"""Entry point of the console program, ``gettowork-cli`` - the terminal version of the game.

PyInstaller turns this file into ``gettowork-cli`` (``gettowork-cli.exe`` on
Windows), which sits next to the windowed ``GetToWork`` in every build. It
runs :func:`gettowork.cli.main`, exactly like the ``gettowork`` command that
``pip install`` gives you. The build recipe is ``packaging/gettowork.spec``.
"""

import multiprocessing
import sys


def main() -> int:
    # Harmless today (the game only uses threads), but required on Windows if a
    # frozen build ever starts a worker process: without it, each worker would
    # relaunch the whole game.
    multiprocessing.freeze_support()

    from gettowork.cli import main as game_main
    from gettowork.launcher import restore_system_library_path

    # Linux: the engine, the browser (for links) and hardware checks must load
    # the system's libraries, not the copies bundled inside the game.
    restore_system_library_path()
    return game_main()


if __name__ == "__main__":
    sys.exit(main())
