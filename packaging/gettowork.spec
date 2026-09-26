# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the double-click / Steam build of Get To Work.

Build it from the repository root (after ``pip install . pyinstaller pillow``)::

    pyinstaller packaging/gettowork.spec --noconfirm

One analysis of the game's code, two programs that share it, in one folder:

* ``GetToWork``      - windowed: the game's own window. This is what Steam
  launches and what players double-click (``packaging/gui_entry.py``).
* ``gettowork-cli``  - console: the terminal version (``packaging/cli_entry.py``).

Both live in ``dist/GetToWork/`` next to ``_internal/`` (Python and the
libraries). On macOS the same files are also wrapped into
``dist/Get To Work.app``, with ``GetToWork`` as the app's main program and
``gettowork-cli`` beside it in ``Contents/MacOS``.

``packaging/assemble.py`` then adds the llama.cpp engine, ``distribution.json``,
the license texts and a README, and packs the archive players download.
See docs/DISTRIBUTION.md.

(The console program is not called plain ``gettowork``: Windows and macOS
ignore upper/lower case in file names, so ``gettowork.exe`` and
``GetToWork.exe`` would be the same file.)
"""

import importlib.util
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# SPECPATH (this file's folder), workpath and the build classes (Analysis, EXE,
# ...) are provided by PyInstaller when it runs this file.
ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
PACKAGE = SRC / "gettowork"
ICON_PNG = PACKAGE / "assets" / "icon.png"
GUI_ENTRY = ROOT / "packaging" / "gui_entry.py"
CLI_ENTRY = ROOT / "packaging" / "cli_entry.py"

APP_NAME = "Get To Work"
GUI_NAME = "GetToWork"  # also the name of the output folder
CLI_NAME = "gettowork-cli"
BUNDLE_ID = "com.markelphoenix.gettowork"
# The oldest macOS the app starts on. The bundled llama.cpp engine is built by
# its project with CMAKE_OSX_DEPLOYMENT_TARGET=13.3, so on older systems the
# engine can't run - better that macOS says so up front than a confusing
# error halfway through setup.
MACOS_MINIMUM = "13.3"

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"


def app_version() -> str:
    """The game's version, read from src/gettowork/__init__.py (the one place it is set)."""
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise SystemExit("Couldn't find __version__ in src/gettowork/__init__.py")
    return match.group(1)


def platform_icon(work: Path):
    """The icon in the format this OS wants (.ico / .icns), made from icon.png with Pillow.

    Linux programs carry no icon of their own (the game window sets it from
    icon.png), so there it is None. Without Pillow the build still works,
    just with PyInstaller's default icon.
    """
    if not (IS_WINDOWS or IS_MAC):
        return None
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: Pillow isn't installed, so the build uses the default icon (pip install pillow).")
        return None
    work.mkdir(parents=True, exist_ok=True)
    picture = Image.open(ICON_PNG).convert("RGBA")
    if IS_WINDOWS:
        out = work / "GetToWork.ico"
        picture.save(out, format="ICO", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    else:
        out = work / "GetToWork.icns"
        # Pixel art: grow it with "nearest neighbour" so the big pixels stay crisp;
        # Pillow then shrinks it to every size an .icns holds.
        big = picture.resize((1024, 1024), Image.Resampling.NEAREST)
        big.save(out, format="ICNS")
    return str(out)


VERSION = app_version()
ICON = platform_icon(Path(workpath))

# --- What goes in -----------------------------------------------------------
# Modules loaded by name at run time are invisible to PyInstaller's import
# scan, so they are listed explicitly:
#   * rich loads some of its modules on demand (e.g. per-OS console code);
#   * truststore picks its code for the operating system at run time;
#   * huggingface_hub imports its own parts lazily (``from huggingface_hub
#     import HfApi`` only loads hf_api.py when first used);
#   * the whole game, so nothing it imports lazily (the GUI, the backends)
#     can be missed.
hiddenimports = (
    collect_submodules("rich")
    + collect_submodules("truststore")
    + collect_submodules("huggingface_hub")
    + collect_submodules("gettowork")
    + ["tkinter", "tkinter.font", "tkinter.ttk"]
)

datas = [(str(ICON_PNG), "gettowork/assets")]  # the window icon (gettowork/gui/app.py looks for it here)
if importlib.util.find_spec("certifi") is not None:
    datas += collect_data_files("certifi")  # the fallback certificate bundle (see gettowork/tls.py)


def tcl_tk_libraries() -> list:
    """Tcl/Tk libraries that live inside the Python installation itself (Linux).

    Some Pythons (python-build-standalone, which uv installs) keep
    libtcl/libtk in their own ``lib`` folder, where PyInstaller doesn't look
    for them; without them the game window can't open. Pythons that use the
    system's Tcl/Tk (like GitHub's) have none there, and this adds nothing.
    """
    if IS_WINDOWS or IS_MAC:
        return []
    lib = Path(sys.base_prefix) / "lib"
    found = sorted(lib.glob("libtcl8*.so*")) + sorted(lib.glob("libtk8*.so*"))
    return [(str(path), ".") for path in found if path.is_file()]


binaries = tcl_tk_libraries()

# Things that must never end up in the game, even if the build machine has them.
excludes = [
    "tests", "pytest", "_pytest",  # the test suite and its tools
    "PIL",  # Pillow is only used above, to make the icon
    "llama_cpp",  # the optional llama-cpp-python backend: the build ships llama-server instead
    "torch", "tensorflow", "jax", "keras", "IPython", "matplotlib",  # optional extras of huggingface_hub
    # Line editing for Python's interactive prompt. The stdlib pulls it in (site,
    # code, rlcompleter), and on many Linux Pythons (Ubuntu's, GitHub's) it links
    # GNU readline, which is GPL-3.0 - it would drag libreadline and libtinfo into
    # the game. The game never needs it: the window reads its own input, and the
    # terminal version's input() works without it.
    "readline",
]

a = Analysis(
    [str(GUI_ENTRY), str(CLI_ENTRY)],
    pathex=[str(SRC)],  # build the checked-out code, even if another copy is installed
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,  # keep docstrings and asserts: the game is meant to be read and learned from
)
pyz = PYZ(a.pure)

# The analysis found both entry scripts; each program runs only its own (plus
# PyInstaller's start-up scripts, which come first in a.scripts).
gui_scripts = [entry for entry in a.scripts if entry[0] != CLI_ENTRY.stem]
cli_scripts = [entry for entry in a.scripts if entry[0] != GUI_ENTRY.stem]

# --- What comes out ----------------------------------------------------------
gui_exe = EXE(
    pyz,
    gui_scripts,
    [],
    exclude_binaries=True,  # one-folder build: the libraries go next to it (COLLECT below)
    name=GUI_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed programs upset antivirus scanners
    console=False,  # a window of its own, no terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
cli_exe = EXE(
    pyz,
    cli_scripts,
    [],
    exclude_binaries=True,
    name=CLI_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # the terminal version
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
coll = COLLECT(
    gui_exe,  # first: on macOS the first program becomes the app's main one
    cli_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=GUI_NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,  # sharp text on Retina screens
            # A normal app with a Dock icon, a menu bar and keyboard focus. PyInstaller would
            # otherwise mark it "background only" (no windows in front, no typing): the app
            # inherits console=True from gettowork-cli, the last program in COLLECT.
            "LSBackgroundOnly": False,
            "LSMinimumSystemVersion": MACOS_MINIMUM,
            "LSApplicationCategoryType": "public.app-category.educational-games",
            "NSHumanReadableCopyright": "Get To Work by markelphoenix",
        },
    )
