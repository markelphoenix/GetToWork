#!/usr/bin/env python3
"""Finish a PyInstaller build of the game and pack the archive players download (used by CI).

    python packaging/assemble.py --dist dist --engine-dir build/engine --os linux --out out
    python packaging/assemble.py --dist dist --engine-dir build/engine --os macos --arch arm64 --out out

``pyinstaller packaging/gettowork.spec`` makes the programs (``dist/GetToWork/``,
and ``dist/Get To Work.app`` on macOS). This script adds everything else a
player needs, in the place the game looks for it (see docs/DISTRIBUTION.md):

======================  ===============================  ==========================================
what                    Windows / Linux                  macOS
======================  ===============================  ==========================================
llama.cpp engine        ``GetToWork/engine/``            ``Get To Work.app/Contents/Resources/engine/``
distribution.json       ``GetToWork/``                   ``.../Contents/Resources/``
THIRD_PARTY_LICENSES    ``GetToWork/``                   ``.../Contents/Resources/``
README.txt              ``GetToWork/``                   next to the app
======================  ===============================  ==========================================

The files are added to the build folder itself, so it can be tested right
away (``packaging/smoke_test.sh``, ``GetToWork --gui-selftest``). Then it
packs one archive into ``--out``:

* Windows: ``.zip`` of the ``GetToWork`` folder;
* macOS: ``.zip`` made with ``ditto`` (Apple's tool, which keeps the app's
  symbolic links, permissions and signature intact) of a ``GetToWork``
  folder holding the app and the README. The app is moved there first:
  ``dist/macos-package/GetToWork/Get To Work.app``;
* Linux: ``.tar.gz`` of the ``GetToWork`` folder, so the programs stay
  executable (a zip would lose that).

On macOS the app is signed again ("ad-hoc", no Apple account needed) after
the files are added, because adding files to a signed app breaks its seal.

The last lines printed are ``KEY=value`` pairs (``ARCHIVE``, ``APP_DIR``,
``GUI``, ``CLI``, ``ENGINE``); in GitHub Actions they are also written to
``$GITHUB_OUTPUT`` (as lower-case step outputs). Exit code 0 = success, 1 = failure. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
APP_FOLDER = "GetToWork"  # the PyInstaller output folder (and the top folder in every archive)
MAC_APP = "Get To Work.app"
MAC_STAGE = "macos-package"  # dist/macos-package/GetToWork/ holds the finished app and its README
GUI_NAME = "GetToWork"
CLI_NAME = "gettowork-cli"
ENGINE_DIR = "engine"
DISTRIBUTION_FILE = "distribution.json"
LICENSES_FILE = "THIRD_PARTY_LICENSES.txt"
README_FILE = "README.txt"
INSTALL_MARKER = "install.json"  # the note next to every engine build (see gettowork.runtime_install)
DEFAULT_ARCH = {"windows": "x64", "linux": "x64", "macos": "arm64"}

README_TEXT = {
    "windows": (
        "Get To Work {version}: extract the whole zip first (right-click it > Extract All - the game can't start "
        "from inside the zip), then double-click GetToWork.exe to play - the game walks you through the rest "
        "(checking your computer, picking and downloading an AI model, optional Jev setup).\n"
        "This build isn't code-signed. Downloaded outside Steam, Windows SmartScreen may say \"Windows protected "
        "your PC\": click More info, then Run anyway. On Windows 11 with Smart App Control on (from Steam too), "
        "Windows may block it outright (\"Smart App Control blocked an app\", or Steam error 0x11C7) - the "
        "game's README on GitHub says what to do.\n"
        "gettowork-cli.exe is the terminal version. The licenses of the software inside the game are in "
        "THIRD_PARTY_LICENSES.txt.\n"
    ),
    "macos": (
        "Get To Work {version}: double-click \"Get To Work.app\" to play - the game walks you through the rest "
        "(picking and downloading an AI model, optional Jev setup). On Steam, just press Play and leave the app "
        "where Steam put it; downloaded from GitHub, drag it into Applications first if you like.\n"
        "Downloaded outside Steam? This build isn't signed by Apple: the first time, right-click the app and "
        "choose Open (on macOS 15 or later: try to open it, then System Settings > Privacy & Security > Open "
        "Anyway), or run this in Terminal: xattr -dr com.apple.quarantine \"Get To Work.app\"\n"
        "The terminal version is \"Get To Work.app/Contents/MacOS/gettowork-cli\". The licenses of the "
        "software inside are in \"Get To Work.app/Contents/Resources/THIRD_PARTY_LICENSES.txt\".\n"
    ),
    "linux": (
        "Get To Work {version}: double-click GetToWork (or run ./GetToWork) to play - the game walks you "
        "through the rest (checking your computer, picking and downloading an AI model, optional Jev setup).\n"
        "Unpack the .tar.gz as it is (for example: tar xzf GetToWork-*.tar.gz) - it keeps the programs "
        "executable, so no chmod +x is needed.\n"
        "./gettowork-cli is the terminal version. The licenses of the software inside the game are in "
        "THIRD_PARTY_LICENSES.txt.\n"
    ),
}


class AssembleError(Exception):
    """Something needed for the build is missing or wrong (the message says what)."""


@dataclass(frozen=True)
class Layout:
    """Where things go in one OS's build."""

    app_dir: Path  # the folder (or macOS .app) that players get
    resources: Path  # where engine/, distribution.json and the licenses go
    readme_dir: Path  # where README.txt goes
    gui: Path  # the windowed program
    cli: Path  # the terminal program
    archive_root: Path  # the folder whose contents (itself included) go into the archive


def layout_for(dist: Path, os_key: str) -> Layout:
    """The build's layout for ``os_key`` ("windows", "linux" or "macos") under PyInstaller's ``dist``."""
    if os_key == "macos":
        # The app is moved into a "GetToWork" folder next to its README, which becomes the archive's
        # top folder. (dist/GetToWork itself is PyInstaller's plain folder build, so it lives one level down.)
        stage = dist / MAC_STAGE / APP_FOLDER
        app = stage / MAC_APP
        return Layout(app, app / "Contents" / "Resources", stage,
                      app / "Contents" / "MacOS" / GUI_NAME, app / "Contents" / "MacOS" / CLI_NAME, stage)
    folder = dist / APP_FOLDER
    ext = ".exe" if os_key == "windows" else ""
    return Layout(folder, folder, folder, folder / f"{GUI_NAME}{ext}", folder / f"{CLI_NAME}{ext}", folder)


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


def app_version() -> str:
    """The game's version, from src/gettowork/__init__.py (the one place it is set)."""
    text = (ROOT / "src" / "gettowork" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise AssembleError("couldn't find __version__ in src/gettowork/__init__.py")
    return match.group(1)


def engine_builds(engine_dir: Path) -> list[tuple[Path, dict]]:
    """(folder, install.json) for each engine build made by fetch_engine.py; checks each one."""
    if not engine_dir.is_dir():
        raise AssembleError(f"the engine folder {engine_dir} doesn't exist (run packaging/fetch_engine.py first)")
    builds = []
    for folder in sorted(p for p in engine_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        try:
            marker = json.loads((folder / INSTALL_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # not an engine build
        if not isinstance(marker, dict):
            continue
        exe = str(marker.get("exe") or "")
        if not exe or ".." in PurePosixPath(exe).parts or not (folder / Path(*PurePosixPath(exe).parts)).is_file():
            raise AssembleError(f"{folder.name}/{INSTALL_MARKER} names an engine program that isn't there: {exe!r}")
        builds.append((folder, marker))
    if not builds:
        raise AssembleError(f"no engine builds (folders with {INSTALL_MARKER}) in {engine_dir}")
    return builds


def engine_tag(builds: Sequence[tuple[Path, dict]]) -> str:
    """The llama.cpp release all the engine builds come from (they must agree)."""
    tags = sorted({str(marker.get("tag") or "") for _folder, marker in builds})
    if len(tags) != 1 or not tags[0]:
        raise AssembleError(f"the engine builds should all come from one llama.cpp release, found: {tags}")
    return tags[0]


def copy_engine(builds: Sequence[tuple[Path, dict]], target: Path, *, os_key: str) -> None:
    """Copy the engine builds into ``target`` (replacing an older copy), keeping links and permissions."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for folder, marker in builds:
        dest = target / folder.name
        shutil.copytree(folder, dest, symlinks=True)
        if os_key != "windows":  # make sure llama-server can run (copytree keeps modes; this is belt and braces)
            exe = dest / Path(*PurePosixPath(str(marker["exe"])).parts)
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def move_mac_app(built: Path, target: Path) -> None:
    """Move PyInstaller's fresh ``Get To Work.app`` to ``target`` (replacing an older one there).

    If there is no fresh app (assemble.py already ran once), the one at
    ``target`` is used as it is.
    """
    if not built.is_dir():
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(built, target)


def distribution_info(tag: str, version: str, built_from: Optional[str]) -> dict:
    """The contents of distribution.json (read by gettowork.distribution)."""
    return {
        "schema": 1,
        "channel": "release",
        "engine_downloads": False,  # the engine ships inside the game: never download programs at run time
        "engine_dir": ENGINE_DIR,
        "llama_cpp_tag": tag,
        "app_version": version,
        "built_from": built_from or None,
    }


def readme_text(os_key: str, version: str) -> str:
    return README_TEXT[os_key].format(version=version)


def archive_name(os_key: str, arch: str, version: str) -> str:
    ext = ".tar.gz" if os_key == "linux" else ".zip"
    return f"{APP_FOLDER}-{version}-{os_key}-{arch}{ext}"


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------


def zip_folder(folder: Path, out: Path) -> None:
    """Zip ``folder`` (itself included as the top folder), keeping Unix permissions and symbolic links.

    Python's :mod:`zipfile` normally drops both. Here each entry's Unix mode
    goes into the "external attributes" field (what ``unzip`` and macOS
    Archive Utility read), and a symbolic link is stored as a link rather
    than a copy of what it points to.
    """
    base = folder.parent
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [folder, *sorted(folder.rglob("*"))]:
            name = path.relative_to(base).as_posix()
            mode = path.lstat().st_mode
            if path.is_symlink():
                info = zipfile.ZipInfo(name)
                info.create_system = 3  # Unix, so the mode below is honoured
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, os.readlink(path))
            elif path.is_dir():
                info = zipfile.ZipInfo(name + "/")
                info.create_system = 3
                info.external_attr = ((stat.S_IFDIR | (mode & 0o777)) << 16) | 0x10  # 0x10: MS-DOS "directory"
                archive.writestr(info, b"")
            else:
                archive.write(path, name)  # (records the file's mode itself)


def tar_folder(folder: Path, out: Path) -> None:
    """``.tar.gz`` of ``folder`` (itself included), keeping permissions and links, without our user names."""

    def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(out, "w:gz", compresslevel=9) as archive:
        archive.add(folder, arcname=folder.name, filter=anonymous)


def ditto_zip(folder: Path, out: Path, runner: Callable[..., object]) -> None:
    """macOS: Apple's ``ditto`` makes the zip (keeps the app's links, permissions and signature)."""
    runner(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(folder), str(out)], check=True)


# The first four bytes of a Mach-O program or library (32/64-bit, either byte order, or "fat").
_MACH_O_MAGICS = {bytes.fromhex(m) for m in ("feedface", "feedfacf", "cefaedfe", "cffaedfe", "cafebabe", "bebafeca")}


def _is_mach_o(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) in _MACH_O_MAGICS
    except OSError:
        return False


def sign_mac_app(app: Path, runner: Callable[..., object]) -> None:
    """Re-sign the app ad-hoc after adding files (adding files breaks the signature's seal).

    Inside out, as Apple recommends: first every program and library of the
    engine (it sits in ``Contents/Resources``, where ``codesign --deep``
    doesn't sign code, only seals it as a file), then the app itself.
    """
    engine = app / "Contents" / "Resources" / ENGINE_DIR
    if engine.is_dir():
        for path in sorted(engine.rglob("*")):
            if path.is_file() and not path.is_symlink() and _is_mach_o(path):
                runner(["codesign", "--force", "--sign", "-", str(path)], check=True)
    runner(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    runner(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def assemble(dist: Path, engine_dir: Path, os_key: str, arch: str, out_dir: Path, *,
             licenses: Optional[Path] = None, version: Optional[str] = None,
             built_from: Optional[str] = None, runner: Callable[..., object] = subprocess.run,
             host: str = sys.platform, say: Callable[[str], None] = print) -> tuple[Layout, Path]:
    """Add the engine, distribution.json, licenses and README to the build, then pack it.

    Returns ``(layout, archive path)``; raises :class:`AssembleError` if
    something is missing. ``runner`` (like ``subprocess.run``) and ``host``
    (like ``sys.platform``) can be swapped out for tests.
    """
    layout = layout_for(dist, os_key)
    if os_key == "macos":
        move_mac_app(dist / MAC_APP, layout.app_dir)
    for needed in (layout.app_dir, layout.gui, layout.cli):
        if not needed.exists():
            raise AssembleError(f"{needed} is missing - build the game first: pyinstaller packaging/gettowork.spec")
    licenses = licenses if licenses is not None else Path(LICENSES_FILE)
    if not licenses.is_file() or licenses.stat().st_size == 0:
        raise AssembleError(f"{licenses} is missing or empty - run packaging/collect_licenses.py first")
    version = version or app_version()
    builds = engine_builds(engine_dir)
    tag = engine_tag(builds)

    say(f"Adding the llama.cpp {tag} engine ({', '.join(str(m.get('variant')) for _f, m in builds)}) "
        f"to {layout.resources / ENGINE_DIR}")
    copy_engine(builds, layout.resources / ENGINE_DIR, os_key=os_key)
    info = distribution_info(tag, version, built_from)
    (layout.resources / DISTRIBUTION_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(licenses, layout.resources / LICENSES_FILE)
    layout.readme_dir.mkdir(parents=True, exist_ok=True)
    (layout.readme_dir / README_FILE).write_text(readme_text(os_key, version), encoding="utf-8", newline="\n")

    if os_key == "macos":
        if host == "darwin" and shutil.which("codesign"):
            say("Signing the app again (ad-hoc) now that files were added")
            sign_mac_app(layout.app_dir, runner)
        else:
            say("Not on a Mac, so the app isn't re-signed (fine for tests, not for players).")

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / archive_name(os_key, arch, version)
    if archive.exists():
        archive.unlink()
    say(f"Packing {archive.name}")
    if os_key == "linux":
        tar_folder(layout.archive_root, archive)
    elif os_key == "macos" and host == "darwin" and shutil.which("ditto"):
        ditto_zip(layout.archive_root, archive, runner)
    else:
        zip_folder(layout.archive_root, archive)
    say(f"Done: {archive} ({archive.stat().st_size / 1e6:,.1f} MB)")
    return layout, archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assemble.py",
        description="Add the engine, distribution.json, licenses and README to a PyInstaller build, then pack it.",
    )
    parser.add_argument("--dist", required=True, help="PyInstaller's output folder (holds GetToWork/ or the .app)")
    parser.add_argument("--engine-dir", required=True, help="the engine builds made by packaging/fetch_engine.py")
    parser.add_argument("--os", required=True, choices=sorted(README_TEXT), help="the OS the build is for")
    parser.add_argument("--arch", choices=("x64", "arm64"), help="the processor (default: x64, arm64 for macOS)")
    parser.add_argument("--out", required=True, help="the folder to put the archive in")
    parser.add_argument("--licenses", default=LICENSES_FILE,
                        help=f"THIRD_PARTY_LICENSES.txt from collect_licenses.py (default: ./{LICENSES_FILE})")
    return parser


def write_outputs(values: dict[str, str]) -> None:
    """Print ``KEY=value`` lines, and append them to $GITHUB_OUTPUT in GitHub Actions (lower-case keys)."""
    for key, value in values.items():
        print(f"{key}={value}")
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            for key, value in values.items():
                fh.write(f"{key.lower()}={value}\n")


def main(argv: Optional[Sequence[str]] = None, *, runner: Callable[..., object] = subprocess.run,
         host: str = sys.platform) -> int:
    args = build_parser().parse_args(argv)
    arch = args.arch or DEFAULT_ARCH[args.os]
    try:
        layout, archive = assemble(Path(args.dist), Path(args.engine_dir), args.os, arch, Path(args.out),
                                   licenses=Path(args.licenses), built_from=os.environ.get("GITHUB_SHA"),
                                   runner=runner, host=host)
    except (AssembleError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # Forward slashes work in every shell GitHub Actions uses, Git Bash on Windows included.
    write_outputs({"ARCHIVE": archive.as_posix(), "APP_DIR": layout.app_dir.as_posix(),
                   "GUI": layout.gui.as_posix(), "CLI": layout.cli.as_posix(),
                   "ENGINE": (layout.resources / ENGINE_DIR).as_posix()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
