#!/usr/bin/env python3
"""Gather the license texts of everything that ships inside the built game (used by CI).

    python packaging/collect_licenses.py --out build/THIRD_PARTY_LICENSES.txt --engine-dir build/engine

A built copy of Get To Work contains other people's software: Python itself,
the Python libraries the game uses, Tcl/Tk (the toolkit behind the game
window), PyInstaller's small start-up program, and the llama.cpp engine.
Their licenses allow that on one condition we must honour: their license
texts travel with the game. This script writes them all into one file,
``THIRD_PARTY_LICENSES.txt``, which ``packaging/assemble.py`` puts next to
the game.

What it collects:

1. every Python distribution the game needs at run time - found by walking
   ``gettowork``'s requirements (and theirs, and so on) with
   :mod:`importlib.metadata`, exactly as ``pip`` resolved them on this
   computer - with each one's license file(s);
2. Python's own license (from the Python that runs this script - the one
   PyInstaller bundles);
3. Tcl/Tk's license (from Tcl's library folder, or a short notice with a link);
4. a note about PyInstaller's bootloader (GPL with an exception that lets it
   ship in any program), plus PyInstaller's license text when installed;
5. the license texts of each engine build in ``--engine-dir``: llama.cpp's
   own MIT license and those of the parts compiled into it (cpp-httplib,
   nlohmann/json, BoringSSL on Windows and macOS, LLVM OpenMP on Windows...),
   which ``packaging/fetch_engine.py`` gathers into each build's
   ``licenses/`` and lists in its ``install.json``. A build without llama.cpp's
   own text, or missing a text its ``install.json`` lists, fails;
6. with ``--app-dir`` (the PyInstaller output folder, e.g. ``dist/GetToWork``):
   every native library PyInstaller copied into the game - OpenSSL, libffi,
   and on Linux the system libraries Tk and Python link against (X11, Xft,
   fontconfig, FreeType, libpng, ...). Each one is matched to its component
   and license; on a Debian/Ubuntu build machine the package's own
   ``/usr/share/doc/<package>/copyright`` text is included, elsewhere a
   maintained notice. A library nobody has mapped to a license fails the
   build, and so does one that must never ship (GNU readline and gdbm are
   GPL-3.0: the spec leaves ``readline`` out for exactly that reason).

Only the standard library is needed (``packaging`` is used when installed,
to evaluate "only on Windows"-style requirement markers exactly).
Exit code 0 = written, 1 = something required was missing.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
GAME = "gettowork"
# Bundled by the build recipe (packaging/gettowork.spec) whenever they are installed,
# even when no requirement names them on this computer.
ALSO_BUNDLED = ("truststore", "certifi")
DEFAULT_OUT = "THIRD_PARTY_LICENSES.txt"
INSTALL_MARKER = "install.json"  # the note next to every engine build (see gettowork.runtime_install)
LICENSE_NAME_RE = re.compile(r"^(?:licen[cs]e|copying|notice|authors?)(?:[-._].*)?$", re.IGNORECASE)
RULE = "=" * 78

TCL_TK_NOTICE = """\
Tcl/Tk is distributed under a BSD-style license by the Regents of the University
of California, Sun Microsystems, Inc., Scriptics Corporation, ActiveState
Corporation and other parties: the software may be used, copied, modified and
distributed for any purpose, provided the copyright notices are kept.
The full license text: https://www.tcl-lang.org/software/tcltk/license.html
"""

PYINSTALLER_NOTE = """\
This program was packaged with PyInstaller (https://pyinstaller.org). The small
start-up program ("bootloader") and the start-up scripts PyInstaller adds are
licensed under the GNU General Public License, version 2 or later, with a special
exception (the "bootloader exception") that allows them to be combined with and
distributed as part of programs under any license, without the GPL applying to
those programs. Runtime hooks from pyinstaller-hooks-contrib, where included, are
licensed under the Apache License 2.0.
"""

PYTHON_EXTRAS_NOTE = """\
Depending on the operating system, the Python runtime inside this build may also
include these libraries, each under its own permissive license: OpenSSL (Apache
License 2.0, https://www.openssl.org/source/license.html), libffi (MIT,
https://github.com/libffi/libffi/blob/master/LICENSE), zlib (zlib license,
https://zlib.net/zlib_license.html), bzip2 (BSD-style, https://sourceware.org/bzip2/),
XZ Utils / liblzma (0BSD / public domain, https://tukaani.org/xz/), Expat (MIT,
https://libexpat.github.io/), mpdecimal (BSD, https://www.bytereef.org/mpdecimal/)
and SQLite (public domain, https://sqlite.org/copyright.html).
"""

LLAMA_CPP_VENDOR_NOTE = """\
llama.cpp is MIT licensed. Its programs also contain code from other projects
(from its vendor/ folder - https://github.com/ggml-org/llama.cpp/tree/{tag}/vendor -
and libraries built into it), each under its own license. The license texts of
every part in these builds follow: {parts}.
"""
VC_RUNTIME_NOTE = """\
The Windows builds also carry Microsoft's Visual C++ runtime
({files}), which Microsoft allows applications
to redistribute under the Visual Studio license terms:
https://visualstudio.microsoft.com/license-terms/
"""
OPENSSL_NOTE = """\
The Linux builds also carry OpenSSL {version} ({files}), copied from the build
machine (Ubuntu 22.04): llama.cpp's official Linux builds need OpenSSL 3, which
Steam's Linux runtime doesn't have. OpenSSL is licensed under the Apache License
2.0 - its text is below (https://www.openssl.org/source/license.html).
"""
# llama.cpp's own MIT license (its copyright line) - every engine build must carry this text.
LLAMA_CPP_NOTICE_RE = re.compile(r"Copyright \(c\) [0-9-]+ The ggml authors", re.IGNORECASE)


@dataclass
class Section:
    """One component and its license text(s)."""

    title: str  # e.g. "rich 15.0.0"
    license: str  # e.g. "MIT"
    source: str = ""  # where to find it
    texts: list[tuple[str, str]] = field(default_factory=list)  # (which file, its text)
    note: str = ""  # anything to say before the texts


# ---------------------------------------------------------------------------
# Python distributions
# ---------------------------------------------------------------------------


def canonical(name: str) -> str:
    """Normalise a distribution name the way pip does ("Foo_Bar" -> "foo-bar")."""
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(requirement: str) -> tuple[str, str]:
    """``'rich>=13 ; python_version < "4"'`` -> ``('rich', 'python_version < "4"')``."""
    spec, _, marker = requirement.partition(";")
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return (match.group(1) if match else spec.strip()), marker.strip()


def marker_applies(marker: str) -> bool:
    """Does a requirement's marker hold on this computer (with no extras selected)?"""
    if not marker:
        return True
    try:
        from packaging.markers import InvalidMarker, Marker
    except ImportError:  # no packaging: extras are the only markers we can be sure about
        return "extra" not in marker
    try:
        return bool(Marker(marker).evaluate({"extra": ""}))
    except InvalidMarker:
        return "extra" not in marker


def game_requirements(root: str = GAME) -> list[str]:
    """The game's own requirements: from its installed metadata, else from pyproject.toml."""
    try:
        return list(md.distribution(root).requires or [])
    except md.PackageNotFoundError:
        pass
    pyproject = ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return []
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    return re.findall(r'"([^"]+)"', block.group(1)) if block else []


def runtime_distributions(root: str = GAME, also: Iterable[str] = ALSO_BUNDLED) -> list[md.Distribution]:
    """Every installed distribution the game needs at run time (not the game itself), sorted by name."""
    found: dict[str, md.Distribution] = {}
    queue = [name for name, marker in map(split_requirement, game_requirements(root)) if marker_applies(marker)]
    queue += list(also)
    while queue:
        name = queue.pop(0)
        key = canonical(name)
        if key in found or key == canonical(root):
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue  # not installed here, so not in the build either
        found[key] = dist
        for requirement in dist.requires or []:
            child, marker = split_requirement(requirement)
            if marker_applies(marker):
                queue.append(child)
    return sorted(found.values(), key=lambda d: canonical(d.metadata["Name"] or ""))


def license_name(dist: md.Distribution) -> str:
    """A short name for a distribution's license, from its metadata."""
    meta = dist.metadata
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    declared = (meta.get("License") or "").strip()
    if declared and "\n" not in declared and len(declared) <= 80:
        return declared
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    classifiers = [c for c in classifiers if c != "OSI Approved"]
    if classifiers:
        return ", ".join(classifiers)
    return "see the license text below"


def project_url(dist: md.Distribution) -> str:
    """The project's home page (or source repository), if its metadata names one."""
    meta = dist.metadata
    urls = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for label in ("homepage", "home", "source", "source code", "repository", "github"):
        if urls.get(label):
            return urls[label]
    home = (meta.get("Home-page") or "").strip()
    return home or next(iter(urls.values()), "")


def read_text(path: object) -> str:
    """A text file's contents, tolerating odd encodings (never raises for bad bytes)."""
    try:
        data = Path(str(path)).read_bytes()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()


def distribution_license_texts(dist: md.Distribution) -> list[tuple[str, str]]:
    """The license-like files (LICENSE, COPYING, NOTICE, AUTHORS) in a distribution's metadata folder."""
    texts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in dist.files or []:
        parts = PurePosixPath(str(item)).parts
        in_metadata = any(part.endswith((".dist-info", ".egg-info")) for part in parts[:-1])
        if not in_metadata or not LICENSE_NAME_RE.match(parts[-1]):
            continue
        text = read_text(dist.locate_file(item))
        if text and text not in seen:
            seen.add(text)
            label = "/".join(parts[1:]) if parts[0].endswith((".dist-info", ".egg-info")) else "/".join(parts)
            texts.append((label, text))
    if not texts:
        declared = (dist.metadata.get("License") or "").strip()
        if "\n" in declared or len(declared) > 80:  # some packages put the whole license in their metadata
            texts.append(("License (from the package metadata)", declared))
    return texts


def distribution_section(dist: md.Distribution) -> Section:
    name, version = dist.metadata["Name"], dist.version
    section = Section(f"{name} {version}", license_name(dist), project_url(dist), distribution_license_texts(dist))
    if not section.texts:
        where = f" - the full text is at {section.source}" if section.source else ""
        section.note = f"This package's installed files include no license text. Its license: {section.license}{where}."
    return section


# ---------------------------------------------------------------------------
# Python, Tcl/Tk, PyInstaller
# ---------------------------------------------------------------------------


def python_license_file(base: Optional[Path] = None) -> Optional[Path]:
    """Python's LICENSE file: next to python.exe on Windows, in lib/pythonX.Y elsewhere."""
    base = Path(sys.base_prefix) if base is None else base
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in (base / "LICENSE.txt", base / "LICENSE", base / "lib" / version / "LICENSE.txt",
                      base / "Lib" / "LICENSE.txt"):
        if candidate.is_file():
            return candidate
    return None


def python_section(base: Optional[Path] = None) -> Section:
    version = ".".join(str(n) for n in sys.version_info[:3])
    section = Section(f"Python {version}", "PSF-2.0 (Python Software Foundation License)",
                      "https://www.python.org/", note=PYTHON_EXTRAS_NOTE)
    path = python_license_file(base)
    text = read_text(path) if path else ""
    if text:
        section.texts.append((path.name, text))
    else:
        section.note = ("Python is distributed under the PSF License Agreement: "
                        "https://docs.python.org/3/license.html\n\n" + PYTHON_EXTRAS_NOTE)
    return section


def tcl_library_dir() -> Optional[Path]:
    """Tcl's library folder (e.g. .../lib/tcl8.6), asked from Tcl itself; None without tkinter."""
    try:
        import tkinter

        return Path(str(tkinter.Tcl().eval("info library")))
    except Exception:  # no tkinter in this Python, or no Tcl library
        return None


def tcl_tk_license_files(library: Optional[Path] = None, base: Optional[Path] = None) -> list[Path]:
    """license.terms files of Tcl and Tk (Tk keeps one in its demos folder), first found first."""
    folders: list[Path] = []
    library = tcl_library_dir() if library is None else library
    if library is not None:
        folders.append(library)
        folders.extend(sorted(library.parent.glob("tk8*")))
    base = Path(sys.base_prefix) if base is None else base
    for pattern in ("lib/tcl8*", "lib/tk8*", "tcl/tcl8*", "tcl/tk8*"):  # (tcl/ is the Windows layout)
        folders.extend(sorted(base.glob(pattern)))
    found: list[Path] = []
    for folder in folders:
        for candidate in (folder / "license.terms", folder / "demos" / "license.terms"):
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
    return found


def tcl_tk_section(library: Optional[Path] = None, base: Optional[Path] = None) -> Section:
    section = Section("Tcl/Tk (the toolkit behind the game window)", "TCL (BSD-style)",
                      "https://www.tcl-lang.org/")
    seen: set[str] = set()
    for path in tcl_tk_license_files(library, base):
        text = read_text(path)
        if text and text not in seen:
            seen.add(text)
            section.texts.append((f"{path.parent.name}/{path.name}", text))
    if not section.texts:
        section.note = TCL_TK_NOTICE
    return section


def pyinstaller_section() -> Section:
    section = Section("PyInstaller bootloader", "GPL-2.0-or-later WITH Bootloader-exception",
                      "https://pyinstaller.org/", note=PYINSTALLER_NOTE)
    try:
        dist = md.distribution("pyinstaller")
    except md.PackageNotFoundError:
        return section
    section.title = f"PyInstaller {dist.version} (bootloader)"
    section.texts = distribution_license_texts(dist)
    return section


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


def engine_builds(engine_dir: Path) -> list[tuple[Path, dict]]:
    """(folder, install.json contents) for every engine build in ``engine_dir``."""
    builds = []
    for folder in sorted(p for p in engine_dir.iterdir() if p.is_dir()) if engine_dir.is_dir() else []:
        try:
            marker = json.loads((folder / INSTALL_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(marker, dict):
            builds.append((folder, marker))
    return builds


def engine_license_files(folder: Path, marker: dict) -> list[Path]:
    """The license files of one engine build: those install.json lists, else any in the folder."""
    listed = [folder / Path(*PurePosixPath(str(rel)).parts) for rel in marker.get("license_files") or []
              if isinstance(rel, str) and ".." not in PurePosixPath(rel).parts]
    found = [path for path in listed if path.is_file()]
    if not found:
        for path in sorted((folder / "licenses").glob("*")) + sorted(folder.glob("*")):
            if path.is_file() and LICENSE_NAME_RE.match(path.name) and path not in found:
                found.append(path)
    return found


def _component_labels(folder: Path, marker: dict) -> dict[Path, str]:
    """``{license file: the part it covers}`` from install.json's ``license_components``."""
    labels: dict[Path, str] = {}
    components = marker.get("license_components")
    for name, rel in (components.items() if isinstance(components, dict) else ()):
        if isinstance(name, str) and isinstance(rel, str) and ".." not in PurePosixPath(rel).parts:
            labels.setdefault(folder / Path(*PurePosixPath(rel).parts), name)
    return labels


def llama_cpp_sections(engine_dir: Path) -> list[Section]:
    """One section per distinct llama.cpp release in the engine folder, with its license texts."""
    by_tag: dict[str, Section] = {}
    variants: dict[str, list[str]] = {}
    parts: dict[str, list[str]] = {}
    vc_files: dict[str, list[str]] = {}
    openssl: dict[str, tuple[list[str], str]] = {}  # tag -> (files, version)
    for folder, marker in engine_builds(engine_dir):
        tag = str(marker.get("tag") or "unknown")
        section = by_tag.get(tag)
        if section is None:
            section = by_tag[tag] = Section(
                f"llama.cpp {tag} (the AI engine)", str(marker.get("license") or "MIT"),
                str(marker.get("source") or "https://github.com/ggml-org/llama.cpp"),
            )
            variants[tag], parts[tag], vc_files[tag] = [], [], []
        variants[tag].append(str(marker.get("variant") or folder.name))
        for name in marker.get("vc_runtime") or []:
            if isinstance(name, str) and name not in vc_files[tag]:
                vc_files[tag].append(name)
        files, version = openssl.setdefault(tag, ([], ""))
        for name in marker.get("openssl") or []:
            if isinstance(name, str) and name not in files:
                files.append(name)
        if isinstance(marker.get("openssl_version"), str) and marker["openssl_version"] and not version:
            openssl[tag] = (files, marker["openssl_version"])
        labels = _component_labels(folder, marker)
        order = {path: index for index, path in enumerate(sorted(labels, key=lambda p: labels[p] != "llama.cpp"))}
        known = {text for _label, text in section.texts}
        # llama.cpp's own license first, then the parts in install.json's order, then anything else.
        for path in sorted(engine_license_files(folder, marker), key=lambda p: order.get(p, len(order))):
            text = read_text(path)
            part = labels.get(path)
            if part and part not in parts[tag]:
                parts[tag].append(part)
            if text and text not in known:
                known.add(text)
                section.texts.append((f"{part} ({path.name})" if part else path.name, text))
    for tag, section in by_tag.items():
        section.title += f" - builds: {', '.join(variants[tag])}"
        others = [part for part in parts[tag] if part != "llama.cpp"]
        if others:
            section.license += f" (llama.cpp), plus the licenses of {', '.join(others)}"
        section.note = LLAMA_CPP_VENDOR_NOTE.format(tag=tag, parts=", ".join(parts[tag]) or "see below")
        if vc_files[tag]:
            section.note += "\n" + VC_RUNTIME_NOTE.format(files=", ".join(vc_files[tag]))
        files, version = openssl.get(tag, ([], ""))
        if files:
            section.note += "\n" + OPENSSL_NOTE.format(version=version or "3", files=", ".join(files))
    return list(by_tag.values())


def llama_cpp_problems(engine_dir: Path) -> list[str]:
    """What's missing from the engine builds' license texts (empty = complete).

    Every build must carry llama.cpp's own MIT license (its copyright line
    names "The ggml authors"), and every license file its ``install.json``
    lists for a part of the build (``license_components``).
    """
    problems: list[str] = []
    for folder, marker in engine_builds(engine_dir):
        texts = [read_text(path) for path in engine_license_files(folder, marker)]
        if texts and not any(LLAMA_CPP_NOTICE_RE.search(text) for text in texts):
            problems.append(f"the engine build {folder.name} has no copy of llama.cpp's own MIT license "
                            "(\"Copyright (c) ... The ggml authors\") - run packaging/fetch_engine.py again")
        for path, part in _component_labels(folder, marker).items():
            if not read_text(path):
                problems.append(f"the engine build {folder.name} is missing the license text of {part} "
                                f"({path.relative_to(folder).as_posix()})")
    return problems


# ---------------------------------------------------------------------------
# Native libraries PyInstaller copied into the game (--app-dir)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeComponent:
    """One component a bundled native library belongs to, and how its license is honoured."""

    title: str
    license: str
    source: str
    notice: str = ""  # shown when no license text is found on the build machine
    covered_by: str = ""  # already covered by another section ("Python", "Tcl/Tk")
    forbidden: str = ""  # why it must never ship (the build fails)


_X11_NOTICE = """\
The X Window System libraries (libX11, libXau, libXdmcp, libXext, libXft,
libXrender, libXss and relatives) are distributed under the MIT/X11 license by
the X.Org Foundation, Keith Packard and many other contributors:

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
Full notices: https://gitlab.freedesktop.org/xorg/lib
"""

# (regular expression for the library's file name, component). First match wins.
NATIVE_COMPONENTS: tuple[tuple[str, NativeComponent], ...] = (
    # Must never ship: GPL-3.0 libraries (the game doesn't need them).
    (r"^libreadline[.\d]", NativeComponent("GNU Readline", "GPL-3.0-or-later", "https://tiswww.case.edu/php/chet/readline/rltop.html",
                                          forbidden="GNU Readline is GPL-3.0: keep 'readline' in the spec's excludes")),
    (r"^libgdbm", NativeComponent("GNU dbm", "GPL-3.0-or-later", "https://www.gnu.org.ua/software/gdbm/",
                                  forbidden="GNU dbm is GPL-3.0: exclude the 'dbm.gnu' module in the spec")),
    (r"^libdb-\d", NativeComponent("Berkeley DB", "AGPL-3.0", "https://www.oracle.com/database/berkeley-db/",
                                  forbidden="Berkeley DB is AGPL-3.0: exclude the 'dbm' modules in the spec")),
    # Already covered by the Python and Tcl/Tk sections.
    (r"^(libpython\d|python\d+\.dll$|python3\.dll$|Python$)", NativeComponent("Python", "PSF-2.0", "https://www.python.org/", covered_by="Python")),
    (r"^(libtcl\d|libtk\d|tcl\d+t?\.dll$|tk\d+t?\.dll$)", NativeComponent("Tcl/Tk", "TCL", "https://www.tcl-lang.org/", covered_by="Tcl/Tk")),
    # Python's own helpers.
    (r"^lib(ssl|crypto)([-.]|$)", NativeComponent(
        "OpenSSL (libssl, libcrypto)", "Apache-2.0", "https://www.openssl.org/source/license.html",
        "OpenSSL 3 is distributed under the Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0\n"
        "Copyright (c) 1998-2025 The OpenSSL Project Authors; Copyright (c) 1995-1998 Eric A. Young, Tim J. Hudson.")),
    (r"^libffi[-.]", NativeComponent(
        "libffi", "MIT", "https://github.com/libffi/libffi/blob/master/LICENSE",
        "libffi - Copyright (c) 1996-2024 Anthony Green, Red Hat, Inc and others. MIT license: permission is "
        "granted free of charge to deal in the software without restriction, provided the copyright notice and "
        "this permission notice are included in all copies. THE SOFTWARE IS PROVIDED \"AS IS\".")),
    (r"^(libz\.|libz\d|zlib1?\.dll$|libzlib)", NativeComponent(
        "zlib", "Zlib", "https://zlib.net/zlib_license.html",
        "zlib - Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler. Provided 'as-is', without any warranty; "
        "anyone may use, alter and redistribute it freely, subject to the notice terms at the link above.")),
    (r"^libbz2[-.]", NativeComponent(
        "bzip2 (libbz2)", "bzip2-1.0.6", "https://sourceware.org/bzip2/",
        "bzip2/libbzip2 - Copyright (C) 1996-2019 Julian R Seward. BSD-style license: redistribution in binary "
        "form is permitted provided the copyright notice, conditions and disclaimer are kept.")),
    (r"^liblzma[-.]", NativeComponent(
        "XZ Utils (liblzma)", "0BSD", "https://tukaani.org/xz/",
        "liblzma from XZ Utils is distributed under the BSD Zero Clause License (0BSD) / public domain.")),
    (r"^libexpat[-.]", NativeComponent(
        "Expat", "MIT", "https://libexpat.github.io/",
        "Expat - Copyright (c) 1998-2000 Thai Open Source Software Center Ltd and Clark Cooper; Copyright (c) "
        "2001-2025 Expat maintainers. MIT license: the copyright notice and permission notice must be included in "
        "all copies. THE SOFTWARE IS PROVIDED \"AS IS\".")),
    (r"^libmpdec[-.]", NativeComponent(
        "mpdecimal", "BSD-2-Clause", "https://www.bytereef.org/mpdecimal/",
        "mpdecimal - Copyright (c) 2008-2024 Stefan Krah. BSD 2-Clause license: redistributions in binary form "
        "must reproduce the copyright notice, conditions and disclaimer.")),
    (r"^(libsqlite3[-.]|sqlite3\.dll$)", NativeComponent(
        "SQLite", "blessing (public domain)", "https://sqlite.org/copyright.html",
        "SQLite is in the public domain.")),
    (r"^libuuid[-.]", NativeComponent(
        "libuuid (util-linux)", "BSD-3-Clause", "https://github.com/util-linux/util-linux",
        "libuuid - Copyright (C) 1996, 1997 Theodore Ts'o. BSD 3-Clause license: redistributions in binary form "
        "must reproduce the copyright notice, conditions and disclaimer.")),
    (r"^lib(tinfo|ncurses|panel|form|menu)w?[-.\d]", NativeComponent(
        "ncurses (libtinfo)", "X11-style (ncurses)", "https://invisible-island.net/ncurses/ncurses-license.html",
        "ncurses - Copyright 2018-2024 Thomas E. Dickey; Copyright 1998-2017 Free Software Foundation, Inc. "
        "Permission is granted free of charge to deal in the software without restriction, provided the copyright "
        "notice and this permission notice are included in all copies.")),
    (r"^libedit[-.]", NativeComponent(
        "libedit (editline)", "BSD-3-Clause", "https://thrysoee.dk/editline/",
        "libedit - Copyright (c) 1992, 1993 The Regents of the University of California. BSD 3-Clause license.")),
    # Tk's own system libraries on Linux.
    (r"^libX(11|au|dmcp|ext|ft|render|ss|inerama|randr|i|cursor|fixes|t|mu|pm|xf86vm)[-.]",
     NativeComponent("X Window System libraries (X11, Xft, Xrender, Xss, Xau, Xdmcp, Xext)", "MIT/X11",
                     "https://gitlab.freedesktop.org/xorg/lib", _X11_NOTICE)),
    (r"^libxcb", NativeComponent("libxcb", "MIT/X11", "https://xcb.freedesktop.org/", _X11_NOTICE)),
    (r"^libfontconfig[-.]", NativeComponent(
        "Fontconfig", "MIT-style (fontconfig)", "https://gitlab.freedesktop.org/fontconfig/fontconfig/-/blob/main/COPYING",
        "Fontconfig - Copyright (C) 2000-2007 Keith Packard; 2005 Patrick Lam; 2009 Roozbeh Pournader; 2008-2009 "
        "Red Hat, Inc.; 2008 Danilo Segan; 2012 Google, Inc. Permission to use, copy, modify, distribute, and sell "
        "this software is granted without fee, provided the copyright notice and this permission notice appear in "
        "all copies.")),
    (r"^libfreetype[-.]", NativeComponent(
        "FreeType", "FTL (FreeType License)", "https://freetype.org/license.html",
        "Portions of this software are copyright (C) 1996-2025 The FreeType Project (www.freetype.org). "
        "All rights reserved. Used under the FreeType License (FTL).")),
    (r"^libpng\d*[-.]", NativeComponent(
        "libpng", "libpng-2.0", "http://www.libpng.org/pub/png/src/libpng-LICENSE.txt",
        "libpng - Copyright (c) 1995-2025 The PNG Reference Library Authors; Copyright (c) 1998-2018 Glenn "
        "Randers-Pehrson and others. PNG Reference Library License version 2.")),
    (r"^libbrotli(common|dec|enc)[-.]", NativeComponent(
        "Brotli", "MIT", "https://github.com/google/brotli/blob/master/LICENSE",
        "Brotli - Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors. MIT license: the copyright notice "
        "and permission notice must be included in all copies. THE SOFTWARE IS PROVIDED \"AS IS\".")),
    (r"^libbsd[-.]", NativeComponent(
        "libbsd", "BSD-3-Clause and others", "https://libbsd.freedesktop.org/",
        "libbsd collects BSD-licensed functions; its copyright file lists each author and license.")),
    (r"^libmd[-.]", NativeComponent(
        "libmd", "BSD-3-Clause / public domain", "https://www.hadrons.org/software/libmd/",
        "libmd provides message digest functions under BSD-3-Clause, BSD-2-Clause and public-domain terms.")),
    # Compiler runtimes.
    (r"^lib(gcc_s|stdc\+\+|gomp|quadmath|atomic)[-.]", NativeComponent(
        "GCC runtime libraries (libgcc_s, libstdc++)", "GPL-3.0-or-later WITH GCC-exception-3.1",
        "https://www.gnu.org/licenses/gcc-exception-3.1.html",
        "The GCC runtime libraries are covered by the GCC Runtime Library Exception, which allows them to be "
        "combined with and distributed as part of programs under any license.")),
    (r"^(vcruntime|msvcp|concrt|vccorlib)\d+(_\d+)?(_\w+)?\.dll$", NativeComponent(
        "Microsoft Visual C++ runtime", "Microsoft Visual C++ Redistributable terms",
        "https://visualstudio.microsoft.com/license-terms/",
        "The Microsoft Visual C++ runtime is redistributable with applications under the Visual Studio "
        "license terms.")),
    (r"^(ucrtbase|api-ms-win-[\w-]+)\.dll$", NativeComponent(
        "Microsoft Universal C Runtime", "Microsoft redistributable terms",
        "https://learn.microsoft.com/cpp/windows/universal-crt-deployment",
        "The Universal C Runtime is redistributable with applications under Microsoft's terms.")),
)

_NATIVE_SUFFIX_RE = re.compile(r"(\.so(\.\d+)*|\.dylib|\.dll)$", re.IGNORECASE)
_PYTHON_EXTENSION_RE = re.compile(r"(\.cpython-|\.abi3\.|\.pyd$|\.pypy)", re.IGNORECASE)
_DEBIAN_LIB_DIRS = ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                    "/lib/aarch64-linux-gnu", "/usr/lib64", "/lib64", "/usr/lib", "/lib")


def native_library_dirs(app_dir: Path) -> list[Path]:
    """Where PyInstaller put the native libraries: ``_internal/`` (and a Mac app's ``Contents/Frameworks``)."""
    found = [app_dir / "_internal", app_dir / "Contents" / "Frameworks"]
    found += sorted(app_dir.glob("*.app/Contents/Frameworks"))
    return [folder for folder in found if folder.is_dir()]


def native_libraries(app_dir: Path) -> list[str]:
    """File names of the native libraries directly in the game's library folder(s).

    Python extension modules (``*.cpython-*.so``, ``*.abi3.so``, ``*.pyd``) and
    everything inside a package's own folder belong to Python distributions,
    whose licenses are collected separately.
    """
    names: set[str] = set()
    for folder in native_library_dirs(app_dir):
        for path in folder.iterdir():
            name = path.name
            if path.is_dir() or _PYTHON_EXTENSION_RE.search(name):
                continue
            if _NATIVE_SUFFIX_RE.search(name) or name == "Python":
                names.add(name)
    return sorted(names, key=str.lower)


def native_component(name: str) -> Optional[NativeComponent]:
    """The component a native library file belongs to (None if nobody has mapped it yet)."""
    for pattern, component in NATIVE_COMPONENTS:
        if re.search(pattern, name, re.IGNORECASE):
            return component
    return None


def debian_copyright(name: str) -> Optional[tuple[str, str]]:
    """``(package, copyright text)`` for a system library on a Debian/Ubuntu machine, else None.

    The library of that name in the usual system folders is looked up with
    ``dpkg-query -S``; the package's ``/usr/share/doc/<package>/copyright``
    holds its license texts.
    """
    if not shutil.which("dpkg-query"):
        return None
    for folder in _DEBIAN_LIB_DIRS:
        path = Path(folder) / name
        if not path.exists():
            continue
        for candidate in dict.fromkeys((str(path), str(path.resolve()))):
            try:
                result = subprocess.run(["dpkg-query", "-S", candidate], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                return None
            if result.returncode != 0 or ":" not in result.stdout:
                continue
            package = result.stdout.split(":", 1)[0].strip().split(",")[0].strip()
            text = read_text(Path("/usr/share/doc") / package / "copyright")
            if text:
                return package, text
    return None


def native_sections(app_dir: Path, *, lookup: Callable[[str], Optional[tuple[str, str]]] = debian_copyright,
                    ) -> tuple[list[Section], list[str]]:
    """One section per component of the bundled native libraries, plus the problems found."""
    problems: list[str] = []
    sections: dict[str, Section] = {}
    libraries: dict[str, list[str]] = {}
    for name in native_libraries(app_dir):
        component = native_component(name)
        if component is None:
            problems.append(f"the game bundles {name}, which has no license entry in packaging/collect_licenses.py "
                            "(add it to NATIVE_COMPONENTS, or exclude it in the spec)")
            continue
        if component.forbidden:
            problems.append(f"the game bundles {name}: {component.forbidden}")
            continue
        if component.covered_by:
            continue
        section = sections.get(component.title)
        if section is None:
            section = sections[component.title] = Section(component.title, component.license, component.source)
            libraries[component.title] = []
        libraries[component.title].append(name)
        found = lookup(name)
        if found is not None:
            package, text = found
            label = f"{package} (/usr/share/doc/{package}/copyright)"
            if all(text != known for _label, known in section.texts):
                section.texts.append((label, text))
    for title, section in sections.items():
        section.title = f"{title} - bundled as {', '.join(libraries[title])}"
        if not section.texts:
            notice = next(c.notice for _p, c in NATIVE_COMPONENTS if c.title == title)
            section.note = notice or f"License: {section.license}. Full text: {section.source}"
    return list(sections.values()), problems


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def render(sections: Sequence[Section]) -> str:
    """The finished THIRD_PARTY_LICENSES.txt text."""
    lines = [
        "Third-party software in Get To Work",
        "===================================",
        "",
        "Get To Work is built with software written by other people - thank you to all",
        "of them! Each part is listed here with its license; the full license texts",
        "follow below.",
        "",
    ]
    width = max((len(s.title) for s in sections), default=0)
    lines += [f"  {s.title.ljust(width)}  {s.license}" for s in sections]
    for section in sections:
        lines += ["", "", RULE, section.title, f"License: {section.license}"]
        if section.source:
            lines.append(f"Source:  {section.source}")
        lines += [RULE, ""]
        if section.note:
            lines += [section.note.rstrip(), ""]
        for label, text in section.texts:
            if len(section.texts) > 1 or section.note:
                lines += [f"--- {label} ---", ""]
            lines += [text, ""]
    return "\n".join(lines).rstrip() + "\n"


def collect(engine_dirs: Sequence[Path] = (), *, root: str = GAME, app_dir: Optional[Path] = None,
            lookup: Callable[[str], Optional[tuple[str, str]]] = debian_copyright) -> tuple[str, list[str]]:
    """Build the license file's text; returns ``(text, problems)``.

    ``problems`` lists anything that would make the file incomplete (e.g. an
    engine build without a license file, or a bundled native library with no
    license entry) - the caller decides whether to fail. ``app_dir`` is the
    built game (PyInstaller's output folder), scanned for native libraries.
    """
    problems: list[str] = []
    sections = [distribution_section(d) for d in runtime_distributions(root)]
    if not sections:
        problems.append(f"found no installed dependencies of {root} (is the game installed? pip install .)")
    sections += [python_section(), tcl_tk_section(), pyinstaller_section()]
    for engine_dir in engine_dirs:
        engine = llama_cpp_sections(Path(engine_dir))
        if not engine:
            problems.append(f"no engine builds (folders with {INSTALL_MARKER}) in {engine_dir}")
        for section in engine:
            if not section.texts:
                problems.append(f"{section.title} has no license file")
        problems += llama_cpp_problems(Path(engine_dir))
        sections += engine
    if app_dir is not None:
        if not Path(app_dir).is_dir():
            problems.append(f"{app_dir} isn't a folder (run PyInstaller first)")
        else:
            native, native_problems = native_sections(Path(app_dir), lookup=lookup)
            sections += native
            problems += native_problems
    return render(sections), problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_licenses.py",
        description="Write THIRD_PARTY_LICENSES.txt for the built game.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"where to write it (default: {DEFAULT_OUT})")
    parser.add_argument("--engine-dir", action="append", default=[],
                        help="a folder of llama.cpp builds made by fetch_engine.py (can be repeated)")
    parser.add_argument("--app-dir", default=None,
                        help="the built game (PyInstaller's dist/GetToWork): its native libraries are listed too")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    text, problems = collect([Path(d) for d in args.engine_dir], app_dir=Path(args.app_dir) if args.app_dir else None)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {out} ({len(text) / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
