"""Entry point used by PyInstaller to build the standalone `gettowork` executable.

(`pip install` users get the same thing via the `gettowork` console script.)
"""

import multiprocessing
import sys

from gettowork.cli import main

if __name__ == "__main__":
    # Harmless today (the game only uses threads), but required on Windows if a
    # frozen build ever starts a worker process: without it, each worker would
    # relaunch the whole game.
    multiprocessing.freeze_support()
    sys.exit(main())
