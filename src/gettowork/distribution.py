"""Which copy of the game is this? (a developer checkout, or a built game from Steam / GitHub)

A built game - the folder players double-click or launch from Steam - carries
a small ``distribution.json`` next to its executable (inside
``Contents/Resources`` on a Mac). It says, for example::

    {"schema": 1, "channel": "release", "engine_downloads": false,
     "engine_dir": "engine", "llama_cpp_tag": "b7000", "app_version": "0.2.0",
     "built_from": "<git sha>"}

The most important line is ``engine_downloads``. A built game ships the
llama.cpp engine *inside* the game (in the ``engine`` folder), so it never
downloads programs while you play - only the AI model itself is downloaded,
during the guided setup. A developer checkout has no ``distribution.json``;
there the game keeps downloading the engine from GitHub as before.

:func:`load` answers "which copy is this?" once and remembers the answer. It
never raises: a missing or damaged file just means "developer copy" - except
in a built game (``sys.frozen``), which then still uses the engine folder next
to it and never downloads programs.

Two environment variables help with testing:

* ``GETTOWORK_ENGINE_DIR`` - an extra folder of engine builds, checked first
  (several folders can be separated with ``os.pathsep``).
* ``GETTOWORK_ALLOW_ENGINE_DOWNLOAD`` - ``1`` lets the game download the
  engine even in a built copy (``0`` forbids it, even in a developer copy).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import __version__

__all__ = [
    "Distribution",
    "DISTRIBUTION_FILE",
    "find_distribution_file",
    "load",
]

DISTRIBUTION_FILE = "distribution.json"
ENV_DISTRIBUTION = "GETTOWORK_DISTRIBUTION"
ENV_ENGINE_DIR = "GETTOWORK_ENGINE_DIR"
ENV_ALLOW_DOWNLOAD = "GETTOWORK_ALLOW_ENGINE_DOWNLOAD"
DEFAULT_ENGINE_DIR = "engine"  # used when distribution.json doesn't name one

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Distribution:
    """What kind of copy of the game is running, and where its built-in engine lives."""

    channel: str = "dev"  # "dev" (no distribution.json found), "release", ...
    engine_downloads: bool = True  # may the game download llama.cpp at runtime?
    engine_dirs: tuple[Path, ...] = ()  # folders to scan for bundled engine builds
    llama_cpp_tag: Optional[str] = None  # the llama.cpp release the built-in engine comes from
    app_version: str = __version__
    root: Optional[Path] = None  # folder holding distribution.json, if any
    built_from: Optional[str] = None  # the git commit the build was made from, if known
    notes: tuple[str, ...] = ()  # anything odd noticed while reading distribution.json

    @property
    def bundled(self) -> bool:
        """True when at least one engine folder exists (the engine ships with the game)."""
        for folder in self.engine_dirs:
            try:
                if folder.is_dir():
                    return True
            except OSError:
                continue
        return False


def _absolute(raw: str) -> Optional[Path]:
    """A path from an environment variable, made absolute (None if it can't be: e.g. ``~nobody/x``).

    Relative engine paths break on Linux and macOS: the engine runs with its
    own folder as the working directory, where a relative path means
    something else.
    """
    try:
        return Path(os.path.abspath(Path(raw).expanduser()))
    except (RuntimeError, OSError, ValueError) as exc:  # expanduser: "Could not determine home directory"
        _log.warning("Ignoring the path %r (%s).", raw, exc)
        return None


def _candidates() -> list[Path]:
    """Where a ``distribution.json`` may be, in the order they're checked."""
    found: list[Path] = []
    override = os.environ.get(ENV_DISTRIBUTION, "").strip()
    path = _absolute(override) if override else None
    if path is not None:
        found.append(path)
    exe = getattr(sys, "executable", "") or ""
    if exe:  # (empty in some embedded Pythons: never fall back to the current folder)
        here = Path(exe).parent
        found.append(here / DISTRIBUTION_FILE)  # Windows / Linux: next to GetToWork(.exe)
        found.append(here.parent / "Resources" / DISTRIBUTION_FILE)  # macOS: Get To Work.app/Contents/Resources
    bundle = getattr(sys, "_MEIPASS", None)  # PyInstaller's folder of bundled files (only in a built game)
    if bundle:
        found.append(Path(bundle) / DISTRIBUTION_FILE)
    return found


def find_distribution_file() -> Optional[Path]:
    """The first ``distribution.json`` that exists (see :func:`_candidates`), or None."""
    for path in _candidates():
        try:
            if path.is_file():
                return path
        except OSError:  # e.g. a folder we aren't allowed to look into
            continue
    return None


def _flag(name: str) -> Optional[bool]:
    """An on/off environment variable: True, False, or None when unset / unclear."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _env_engine_dirs() -> tuple[Path, ...]:
    raw = os.environ.get(ENV_ENGINE_DIR, "")
    dirs = (_absolute(p.strip()) for p in raw.split(os.pathsep) if p.strip())
    return tuple(d for d in dirs if d is not None)  # (a bad entry is skipped, never the rest)


def _read(path: Path) -> Distribution:
    """Build a Distribution from one ``distribution.json`` (raises ValueError/OSError if unreadable)."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("it isn't a JSON object")
    root = path.parent
    notes: list[str] = []

    channel = data.get("channel")
    channel = channel.strip() if isinstance(channel, str) and channel.strip() else "release"

    downloads = data.get("engine_downloads")
    if not isinstance(downloads, bool):
        if downloads is not None:
            notes.append("engine_downloads should be true or false; using the default.")
        downloads = channel == "dev"  # a built game ships its engine: no downloads unless it says so

    wanted = data.get("engine_dir", DEFAULT_ENGINE_DIR)
    names = wanted if isinstance(wanted, list) else [wanted]
    dirs: list[Path] = []
    for name in names:
        if isinstance(name, str) and name.strip():
            dirs.append(root / name.strip())  # (an absolute path stays absolute)
        else:
            notes.append("engine_dir should be a folder name; using the default.")
    if not dirs:
        dirs.append(root / DEFAULT_ENGINE_DIR)

    tag = data.get("llama_cpp_tag")
    version = data.get("app_version")
    built_from = data.get("built_from")
    return Distribution(
        channel=channel,
        engine_downloads=downloads,
        engine_dirs=tuple(dirs),
        llama_cpp_tag=tag if isinstance(tag, str) and tag.strip() else None,
        app_version=version if isinstance(version, str) and version.strip() else __version__,
        root=root,
        built_from=built_from if isinstance(built_from, str) and built_from.strip() else None,
        notes=tuple(notes),
    )


def _frozen() -> bool:
    """Is this a built game (PyInstaller sets ``sys.frozen``)?"""
    return bool(getattr(sys, "frozen", False))


def _frozen_engine_dirs() -> tuple[Path, ...]:
    """Where a built game's engine folder sits when there's no distribution.json to say so."""
    exe = getattr(sys, "executable", "") or ""
    if not exe:
        return ()
    here = Path(exe).parent
    return (here / DEFAULT_ENGINE_DIR, here.parent / "Resources" / DEFAULT_ENGINE_DIR)


def _discover() -> Distribution:
    """Find and read ``distribution.json``, then apply the environment overrides."""
    path = find_distribution_file()
    if path is None:
        if _frozen():
            # A built game without its distribution.json (a packaging slip, a damaged
            # install): it still ships its engine, so keep using that - and a built
            # game never downloads programs, whatever happened to the file.
            note = "No distribution.json found next to the built game; using its engine folder, downloads off."
            _log.warning(note)
            dist = Distribution(engine_downloads=False, engine_dirs=_frozen_engine_dirs(), notes=(note,))
        else:
            dist = Distribution()
    else:
        try:
            dist = _read(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            # A damaged file must never stop the game. Carry on without it, but still
            # look in the engine folder that sits next to the file, so a built game
            # keeps using the engine it ships with. Only a developer copy may then
            # download engines: a built game never downloads programs.
            note = f"Couldn't read {path} ({exc}); carrying on without it."
            _log.warning(note)
            dist = Distribution(root=path.parent, engine_downloads=not _frozen(),
                                engine_dirs=(path.parent / DEFAULT_ENGINE_DIR,), notes=(note,))

    extra = _env_engine_dirs()
    allow = _flag(ENV_ALLOW_DOWNLOAD)
    if extra or allow is not None:
        dirs = extra + tuple(d for d in dist.engine_dirs if d not in extra)
        dist = Distribution(
            channel=dist.channel,
            engine_downloads=dist.engine_downloads if allow is None else allow,
            engine_dirs=dirs,
            llama_cpp_tag=dist.llama_cpp_tag,
            app_version=dist.app_version,
            root=dist.root,
            built_from=dist.built_from,
            notes=dist.notes,
        )
    return dist


_cache: Optional[Distribution] = None
_lock = threading.Lock()


def load(*, refresh: bool = False) -> Distribution:
    """Which copy of the game this is (read once, then remembered). Never raises.

    ``refresh=True`` reads everything again (tests use it after changing the
    environment variables).
    """
    global _cache
    with _lock:
        if _cache is None or refresh:
            try:
                _cache = _discover()
            except Exception as exc:  # belt and braces: this must never stop the game
                note = f"Couldn't work out which copy of the game this is ({exc}); carrying on."
                _log.warning(note)
                # ...and, like every other fallback, a built game keeps its own engine and
                # never downloads programs, whatever went wrong.
                frozen = _frozen()
                _cache = Distribution(engine_downloads=not frozen,
                                      engine_dirs=_frozen_engine_dirs() if frozen else (), notes=(note,))
        return _cache
