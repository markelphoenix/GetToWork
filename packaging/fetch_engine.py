#!/usr/bin/env python3
"""Fetch the official llama.cpp engine builds that ship inside the game (used by CI).

A built copy of Get To Work (Steam, or the double-click test build) carries
the llama.cpp engine in its ``engine`` folder, so players never download
programs while they play - only the AI model, during the guided setup. This
script puts those builds together::

    python packaging/fetch_engine.py --os windows --arch x64 --variants vulkan,cpu --dest build/engine
    python packaging/fetch_engine.py --os macos --arch arm64 --variants metal --dest build/engine --verify
    python packaging/fetch_engine.py --os linux --arch x64 --variants vulkan,cpu --dest build/engine --tag b7000
    python packaging/fetch_engine.py --os linux --arch x64 --variants vulkan,cpu --dest build/engine --tag pinned

1. Finds the newest llama.cpp release that has **all** the requested builds
   for that OS/processor (``--tag auto``, the default), or exactly ``--tag``.
   ``--tag pinned`` means the release named in ``packaging/llama_cpp_tag.txt``:
   the game builds use it, so every OS (and every re-run) ships the same,
   reviewed engine - byte for byte: that file also lists the SHA-256 of every
   archive, and a release whose files changed since (GitHub reporting another
   fingerprint, or other bytes arriving) fails the build.
2. Downloads each archive, checking its size and the SHA-256 fingerprint
   GitHub publishes (the game's own :mod:`gettowork.runtime_install` code) -
   or, with ``--tag pinned``, the one pinned in ``llama_cpp_tag.txt``.
3. Unpacks each build safely into ``<dest>/<tag>-<variant>/`` with the same
   ``install.json`` note the game writes for its own installs, plus
   ``"bundled": true`` - and gathers every license text the build needs into
   ``licenses/``: the archive's own license files, the texts llama.cpp
   embeds in its programs (``License for llama.cpp``, ``cpp-httplib``,
   ``jsonhpp``, ``BoringSSL``... - exactly what ``llama licenses`` prints),
   and, for any part the OS's builds must carry (:data:`REQUIRED_LICENSES`)
   that is still missing, the text from the same release's source on
   GitHub. A build missing one fails: it is never bundled without them.
4. Windows builds also get Microsoft's Visual C++ runtime (``msvcp140.dll``,
   ``vcruntime140.dll``, ``vcruntime140_1.dll``) copied next to
   ``llama-server.exe`` (``--vc-runtime``; by default from this Windows
   computer's System32), so the engine starts on a PC that never installed
   the Visual C++ Redistributable. Then every DLL each program imports must
   be in the build's folder or be part of Windows - checked by reading the
   programs' import tables, so it works on any computer.
5. Linux builds get OpenSSL 3 (``libssl.so.3``, ``libcrypto.so.3``) copied
   next to ``llama-server`` from this Linux computer's system libraries
   (``--linux-openssl``): the official builds link it but don't include it,
   and Steam's Linux runtime (Debian 10/11 based, hiding the computer's own
   ``/usr``) has only OpenSSL 1.1. Then every library each program needs
   (read from its ELF ``NEEDED`` entries) must be in the build's folder or be
   one every player's system provides (:data:`LINUX_RUNTIME_LIBS`: glibc, the
   C++/GCC runtime, OpenMP, the Vulkan loader).
6. Records the model architectures the release knows (from its
   ``src/llama-arch.cpp``) in ``install.json`` as ``architectures``, so a
   built game - which can't update its engine - only offers models it can run.
7. ``--verify`` runs ``llama-server --version`` for every build this computer
   can run, and fails if one doesn't exit cleanly.

The last line printed is ``LLAMA_CPP_TAG=<tag>``, for the build to record in
``distribution.json``. Exit code 0 = success, 1 = failure. A ``GITHUB_TOKEN``
environment variable (raises GitHub's rate limit) is used but never printed.
Only the standard library and ``gettowork.runtime_install`` are needed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Optional, Sequence

try:
    from gettowork import runtime_install as ri
except ImportError:  # run straight from a checkout, without installing the game
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gettowork import runtime_install as ri

OS_NAMES = {"windows": "Windows", "linux": "Linux", "macos": "Darwin"}  # --os -> the names runtime_install knows
ARCHES = ("x64", "arm64")
SEARCH_RELEASES = 30  # how many recent releases --tag auto looks through (the newest is often still uploading)
VERIFY_TIMEOUT_S = 60.0
LICENSE_DIR = "licenses"
# LICENSE, LICENSE.md, LICENSE-curl, COPYING, NOTICE... (one archive can carry several)
LICENSE_NAME_RE = re.compile(r"^(?:licen[cs]e|copying|notice)(?:[-._].*)?$", re.IGNORECASE)
# llama.cpp's own MIT license, from the same release's source - fetched whenever the archive has no copy of it
# (the official Windows zips carry only LICENSE-LLVM-OpenMP, for their libomp.dll).
LICENSE_FALLBACK_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/LICENSE"
_RAW_SOURCE = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/"
# The BoringSSL release a llama.cpp tag builds with is named in this file (set(BORINGSSL_VERSION "0.2026...")).
BORINGSSL_VERSION_URL = _RAW_SOURCE + "vendor/cpp-httplib/CMakeLists.txt"
BORINGSSL_LICENSE_URL = "https://raw.githubusercontent.com/google/boringssl/{version}/LICENSE"
# llama.cpp builds embed the license texts of what they contain (cmake/license.cmake), each as one
# C string "License for <name>\n=====...\n\n<text>" - the list `llama licenses` prints.
EMBEDDED_LICENSE_RE = re.compile(rb"License for ([A-Za-z0-9][A-Za-z0-9 ._+-]{0,60})\r?\n=+\r?\n\r?\n")
MAX_EMBEDDED_LICENSE_BYTES = 256 * 1024
MAX_SCANNED_FILE_BYTES = 1024 * 1024 * 1024
# BoringSSL's source paths (build/_deps/boringssl-src/ssl/...) are compiled into the programs that contain it.
_BORINGSSL_CODE_RE = re.compile(rb"boringssl-src[/\\]|[/\\]boringssl[/\\]", re.IGNORECASE)


class LicenseComponent(NamedTuple):
    """One part of the engine whose license text must ship with it, and how to recognise that text."""

    name: str  # as llama.cpp names it ("License for <name>")
    spdx: str
    aliases: tuple[str, ...]  # found in a license file's name, e.g. LICENSE-httplib
    marker: str = ""  # a phrase only this license's text contains (its copyright line)
    url: str = ""  # where the text is fetched from when missing ({tag} = the llama.cpp release)


LICENSE_COMPONENTS: tuple[LicenseComponent, ...] = (
    LicenseComponent("llama.cpp", "MIT", ("llama.cpp", "llama-cpp", "ggml"), r"The ggml authors", LICENSE_FALLBACK_URL),
    LicenseComponent("cpp-httplib", "MIT", ("cpp-httplib", "httplib"), r"yhirose",
                     _RAW_SOURCE + "vendor/cpp-httplib/LICENSE"),
    LicenseComponent("jsonhpp", "MIT", ("jsonhpp", "nlohmann", "json.hpp"), r"Niels Lohmann",
                     _RAW_SOURCE + "licenses/LICENSE-jsonhpp"),
    LicenseComponent("BoringSSL", "Apache-2.0", ("boringssl",), url=BORINGSSL_LICENSE_URL),
    LicenseComponent("LLVM OpenMP", "Apache-2.0 WITH LLVM-exception", ("llvm-openmp", "openmp", "libomp")),
    # OpenSSL 3's LICENSE.txt is the plain Apache 2.0 text, so it's recognised by name only. {version}: the
    # OpenSSL release the build carries (read from its libcrypto).
    LicenseComponent("OpenSSL", "Apache-2.0", ("openssl",),
                     url="https://raw.githubusercontent.com/openssl/openssl/openssl-{version}/LICENSE.txt"),
)
# The parts each OS's official builds contain (llama.cpp's release.yml builds the Windows and macOS
# archives with -DLLAMA_BUILD_BORINGSSL=ON; the Linux ones link the system's OpenSSL 3, which this script
# copies in - see LINUX_OPENSSL_LIBS). Also required whenever found in a build: BoringSSL (its code is in the
# programs), LLVM OpenMP (libomp) and OpenSSL (libssl / libcrypto).
REQUIRED_LICENSES = {
    "windows": ("llama.cpp", "cpp-httplib", "jsonhpp", "BoringSSL"),
    "macos": ("llama.cpp", "cpp-httplib", "jsonhpp", "BoringSSL"),
    "linux": ("llama.cpp", "cpp-httplib", "jsonhpp"),
}

# Microsoft's Visual C++ runtime, which the official Windows builds need but don't include. Microsoft lets
# applications ship these files next to their programs ("app-local"); the game hides its own copies (in
# _internal) from the engine on purpose, so each engine folder gets its own.
VC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
# DLLs every supported Windows (10/11) provides, or that come with the GPU driver (vulkan-1.dll):
# an engine program may import these without shipping them.
WINDOWS_SYSTEM_DLLS = frozenset({
    "advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "comctl32.dll", "comdlg32.dll", "crypt32.dll", "d3d11.dll",
    "d3d12.dll", "dbghelp.dll", "dwmapi.dll", "dxcore.dll", "dxgi.dll", "gdi32.dll", "imm32.dll", "iphlpapi.dll",
    "kernel32.dll", "kernelbase.dll", "mpr.dll", "msvcrt.dll", "mswsock.dll", "ncrypt.dll", "netapi32.dll",
    "normaliz.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "opengl32.dll", "powrprof.dll", "psapi.dll",
    "rpcrt4.dll", "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "ucrtbase.dll", "user32.dll",
    "userenv.dll", "uxtheme.dll", "version.dll", "vulkan-1.dll", "winhttp.dll", "wininet.dll", "winmm.dll",
    "wintrust.dll", "ws2_32.dll", "wsock32.dll",
})
_WINDOWS_API_SET_RE = re.compile(r"^(api|ext)-ms-win-[a-z0-9-]+\.dll$", re.IGNORECASE)

# Linux: libraries an engine program may need without shipping them - glibc's own, the C++/GCC runtime,
# OpenMP and the Vulkan loader. Every Linux the game supports has them, and so do all of Steam's Linux
# runtimes (1.0 scout/soldier, 3.0 sniper, 4.0); anything else must ship in the build's folder.
LINUX_RUNTIME_LIBS = frozenset({
    "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1", "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0",
    "librt.so.1", "libresolv.so.2", "libgcc_s.so.1", "libstdc++.so.6", "libgomp.so.1", "libvulkan.so.1",
})
# OpenSSL 3, which the official Linux builds link (libllama-common, libllama-server-impl) but don't include.
# Steam's Linux runtimes for native games (1.0 = a Debian 10 "soldier" container, 3.0 "sniper" = Debian 11)
# have only OpenSSL 1.1 and don't show the computer's own /usr to the game, so without its own copy the
# engine couldn't start under Steam (Steam Deck included). The programs look next to themselves first
# (RUNPATH $ORIGIN), and the game puts the engine's folder on LD_LIBRARY_PATH too.
LINUX_OPENSSL_LIBS = ("libssl.so.3", "libcrypto.so.3")
# Where a Linux build machine keeps them, per --arch (Debian/Ubuntu's multiarch folders first).
LINUX_LIBRARY_DIRS = {
    "x64": ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib64", "/lib64", "/usr/lib", "/lib"),
    "arm64": ("/usr/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu", "/usr/lib64", "/lib64", "/usr/lib", "/lib"),
}
ELF_MACHINES = {"x64": 62, "arm64": 183}  # e_machine: EM_X86_64, EM_AARCH64
# libcrypto names its release in a string like "OpenSSL 3.0.2 15 Mar 2022".
_OPENSSL_VERSION_RE = re.compile(rb"OpenSSL (3\.\d+\.\d+[a-z]?) +\d{1,2} [A-Z][a-z]{2} \d{4}")
# OpenSSL's own library files: libssl.so.3, libcrypto.so.3, libssl.3.dylib, libcrypto-3-x64.dll...
_OPENSSL_FILE_RE = re.compile(r"^lib(?:ssl|crypto)(?:\.so(?:\.\d+)*|(?:\.\d+)+\.dylib|-\d+(?:-(?:x64|arm64))?\.dll)$",
                              re.IGNORECASE)

# The model architectures a llama.cpp release knows: the LLM_ARCH_NAMES table in its src/llama-arch.cpp
# ({ LLM_ARCH_QWEN3, "qwen3" }, ...) - exactly the names a GGUF's general.architecture is matched against.
ARCHITECTURES_URL = _RAW_SOURCE + "src/llama-arch.cpp"
_ARCH_TABLE_RE = re.compile(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\s*\};", re.DOTALL)
_ARCH_ENTRY_RE = re.compile(r'\{\s*LLM_ARCH_([A-Z0-9_]+)\s*,\s*"([^"]+)"\s*\}')
MIN_ARCHITECTURES = 20  # far fewer means the table wasn't read properly (llama.cpp knew 100+ by 2025)
_SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PINNED_TAG_FILE = Path(__file__).resolve().parent / "llama_cpp_tag.txt"  # the release the game builds ship


class PinnedRelease(NamedTuple):
    """The llama.cpp release the game builds ship, and the SHA-256 of each of its archives they use."""

    tag: str
    digests: dict[str, str]  # archive name -> lower-case hex SHA-256


_PIN_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})\s+\*?(\S+)$")  # `sha256sum` output works as is


def pinned_release(path: Optional[Path] = None) -> PinnedRelease:
    """The release in ``llama_cpp_tag.txt``: one ``b<N>`` line, then ``<sha256>  <archive name>`` lines.

    Lines starting with # are comments. Every archive line must name an
    archive of that release (``llama-<tag>-...``), once.
    """
    path = PINNED_TAG_FILE if path is None else path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ri.RuntimeInstallError(f"Couldn't read the pinned llama.cpp release from {path} ({exc}).") from exc
    entries = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    tags = [line for line in entries if re.fullmatch(r"b\d+", line)]
    if len(tags) != 1 or not entries or entries[0] != tags[0]:
        raise ri.RuntimeInstallError(f"{path} should name exactly one llama.cpp release, like b7000, on its first "
                                     "line (then one '<sha256>  <archive name>' line per archive).")
    tag = tags[0]
    digests: dict[str, str] = {}
    for line in entries[1:]:
        match = _PIN_DIGEST_RE.match(line)
        if match is None:
            raise ri.RuntimeInstallError(f"{path}: {line!r} isn't a '<sha256>  <archive name>' line.")
        digest, name = match.group(1).lower(), match.group(2)
        if not name.startswith(f"llama-{tag}-"):
            raise ri.RuntimeInstallError(f"{path}: {name} isn't an archive of the pinned release {tag} - update the "
                                         "SHA-256 lines whenever the pin moves.")
        if name in digests:
            raise ri.RuntimeInstallError(f"{path} lists {name} twice.")
        digests[name] = digest
    return PinnedRelease(tag, digests)


def pinned_tag(path: Optional[Path] = None) -> str:
    """The llama.cpp release named in ``llama_cpp_tag.txt``."""
    return pinned_release(path).tag


class PlainProgress:
    """Just enough of :class:`gettowork.ui.UI` for runtime_install's helpers.

    Prints plain lines - no colours, spinners or redrawn bars - so the
    output reads well in a CI log.
    """

    def __init__(self, out: Any = None) -> None:
        self.out = out if out is not None else sys.stdout

    def say(self, text: str = "") -> None:
        try:
            print(text, file=self.out, flush=True)
        except UnicodeEncodeError:  # e.g. a Windows CI log in a legacy code page
            encoding = getattr(self.out, "encoding", None) or "ascii"
            print(text.encode(encoding, "replace").decode(encoding), file=self.out, flush=True)

    def info(self, text: str) -> None:
        self.say(text)

    def success(self, text: str) -> None:
        self.say(text)

    def warn(self, text: str) -> None:
        self.say(f"warning: {text}")

    @contextlib.contextmanager
    def status(self, text: str, **_kwargs: Any) -> Iterator[Callable[[str], None]]:
        self.say(text)
        yield lambda _text: None

    @contextlib.contextmanager
    def download_progress(self, description: str, total_bytes: Optional[int]) -> Iterator[Callable[[int], None]]:
        """Yields ``advance(n)``; prints a line every 10% (or every 10 MB if the size is unknown)."""
        size = f" ({total_bytes / 1e6:,.1f} MB)" if total_bytes else ""
        self.say(f"Downloading {description}{size}...")
        state = {"done": 0, "next": 10.0 if total_bytes else 10e6}

        def advance(n: int) -> None:
            state["done"] += n
            if total_bytes:
                percent = 100.0 * state["done"] / total_bytes
                while percent >= state["next"] and state["next"] <= 100:
                    self.say(f"  {int(state['next'])}%")
                    state["next"] += 10.0
            elif state["done"] >= state["next"]:
                self.say(f"  {state['done'] / 1e6:,.0f} MB")
                state["next"] += 10e6

        yield advance
        self.say(f"  done ({state['done'] / 1e6:,.1f} MB)")


def host_platform() -> tuple[Optional[str], Optional[str]]:
    """This computer as (``--os`` value, ``--arch`` value), e.g. ("linux", "x64"); None if unknown."""
    os_key = {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(platform.system().lower())
    machine = platform.machine().lower()
    arch = "x64" if machine in ("x86_64", "amd64", "x64") else "arm64" if machine in ("arm64", "aarch64") else None
    return os_key, arch


def choose_release(releases: list[dict], variants: Sequence[ri.RuntimeVariant], os_name: str,
                   arch: str) -> Optional[tuple[dict, dict[str, list[dict]]]]:
    """The newest release that has every one of `variants` for this OS/arch.

    Returns ``(release, {variant name: [assets]})`` or None. A release still
    uploading its files (or one missing a build) is skipped, so the builds
    shipped together always come from one and the same release.
    """
    for release in ri.releases_newest_first(releases):
        picked: dict[str, list[dict]] = {}
        for variant in variants:
            assets = ri.select_assets(release.get("assets") or [], variant, os_name, arch)
            if not assets:
                break
            picked[variant.name] = assets
        else:
            return release, picked
    return None


def _has_digest(asset: dict) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-fA-F]{64}", str(asset.get("digest") or "").strip()))


def _license_files(payload: Path) -> list[Path]:
    """License-like files anywhere in an unpacked build (shallowest first)."""
    found = []
    for dirpath, _dirs, files in os.walk(payload):
        for name in files:
            path = Path(dirpath) / name
            if LICENSE_NAME_RE.match(name) and not path.is_symlink():
                found.append(path)
    return sorted(found, key=lambda p: (len(p.relative_to(payload).parts), str(p)))


def _normalised(text: bytes) -> bytes:
    return text.replace(b"\r\n", b"\n").strip()


def _slug(name: str) -> str:
    """ "LLVM OpenMP" -> "LLVM-OpenMP" (safe in a file name)."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", name).strip("-") or "unknown"


def component_named(name: str) -> Optional[LicenseComponent]:
    """The known component llama.cpp calls `name` ("License for <name>"), matched loosely."""
    key = name.strip().lower()
    for component in LICENSE_COMPONENTS:
        if key == component.name.lower() or _slug(key) in component.aliases:
            return component
    return None


def identify_license(file_name: str, text: bytes) -> Optional[str]:
    """Which known component a license file covers: by its copyright line, else by its name."""
    body = text.decode("utf-8", "replace")
    for component in LICENSE_COMPONENTS:
        if component.marker and re.search(component.marker, body):
            return component.name
    lowered = file_name.lower()
    for component in LICENSE_COMPONENTS:
        if any(alias in lowered for alias in component.aliases):
            return component.name
    return None


def _program_files(payload: Path) -> Iterator[Path]:
    """The build's files that may hold compiled code (skips license files, notes and folders)."""
    for dirpath, _dirs, files in os.walk(payload):
        for name in sorted(files):
            path = Path(dirpath) / name
            if path.is_symlink() or LICENSE_NAME_RE.match(name) or name == ri.INSTALL_MARKER:
                continue
            try:
                if 0 < path.stat().st_size <= MAX_SCANNED_FILE_BYTES:
                    yield path
            except OSError:
                continue


def scan_programs(payload: Path) -> tuple[dict[str, bytes], set[str]]:
    """Read the build's programs once: ``(embedded license texts by name, components found in the code)``.

    llama.cpp puts each license text into its programs as a C string that
    starts ``License for <name>`` and ends at the string's closing zero byte.
    """
    embedded: dict[str, bytes] = {}
    found: set[str] = set()
    for path in _program_files(payload):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if re.match(r"^lib(g|i)?omp[\d.-]*\.(dll|dylib|so)", path.name, re.IGNORECASE):
            found.add("LLVM OpenMP")
        if _OPENSSL_FILE_RE.match(path.name):
            found.add("OpenSSL")
        if _BORINGSSL_CODE_RE.search(data):
            found.add("BoringSSL")
        for match in EMBEDDED_LICENSE_RE.finditer(data):
            end = data.find(b"\x00", match.end(), match.end() + MAX_EMBEDDED_LICENSE_BYTES)
            if end < 0:
                continue
            name = match.group(1).decode("ascii", "replace").strip()
            text = data[match.end():end].replace(b"\r\n", b"\n").rstrip().lstrip(b"\n")  # keeps indentation
            if text.strip() and name not in embedded:
                embedded[name] = text + b"\n"
    return embedded, found


def _fetch_text(http: Any, url: str, what: str) -> bytes:
    """GET a small text file (a license) from GitHub; raises RuntimeInstallError on any trouble."""
    try:
        resp = http.request("GET", url, headers={"User-Agent": ri.USER_AGENT}, timeout=30.0)
        with resp:
            status, body = resp.status, resp.read()
    except ri.NETWORK_ERRORS as exc:
        raise ri.RuntimeInstallError(f"The engine build has no copy of {what}, and fetching it failed ({exc}). "
                                     "A build is never bundled without its license text.") from exc
    if status != 200 or not body.strip():
        raise ri.RuntimeInstallError(
            f"The engine build has no copy of {what}, and GitHub answered HTTP {status} for {url}. "
            "A build is never bundled without its license text."
        )
    return body


def openssl_version(payload: Path) -> Optional[str]:
    """The OpenSSL release a build carries ("3.0.2"), read from its libcrypto; None if it has none."""
    for path in sorted(payload.rglob("*")):
        if not (path.is_file() and not path.is_symlink() and _OPENSSL_FILE_RE.match(path.name)
                and "crypto" in path.name.lower()):
            continue
        try:
            found = _OPENSSL_VERSION_RE.search(path.read_bytes())
        except OSError:
            continue
        if found:
            return found.group(1).decode("ascii")
    return None


def fetch_component_license(component: LicenseComponent, tag: str, http: Any, progress: PlainProgress, *,
                            payload: Optional[Path] = None) -> bytes:
    """A component's license text from the same llama.cpp release's source (BoringSSL: from its own release;
    OpenSSL: from the release `payload`'s libcrypto names)."""
    what = f"{component.name}'s license"
    if not component.url:
        raise ri.RuntimeInstallError(f"The engine build has no copy of {what}, and there's nowhere to fetch it "
                                     "from. A build is never bundled without its license text.")
    if component.name == "OpenSSL":
        version = openssl_version(payload) if payload is not None else None
        if version is None or not _SAFE_TAG_RE.match(version):
            raise ri.RuntimeInstallError(f"The engine build has no copy of {what}, and its libcrypto doesn't say "
                                         "which OpenSSL release it is. A build is never bundled without its "
                                         "license text.")
        url = component.url.format(version=version)
        progress.say(f"The build has no copy of {what} - fetching it from {url}")
        return _fetch_text(http, url, what)
    if component.name == "BoringSSL":
        version_url = BORINGSSL_VERSION_URL.format(tag=tag)
        cmake = _fetch_text(http, version_url, what).decode("utf-8", "replace")
        found = re.search(r'set\(\s*BORINGSSL_VERSION\s+"([^"]+)"', cmake)
        if not found or not _SAFE_TAG_RE.match(found.group(1)):
            raise ri.RuntimeInstallError(f"The engine build has no copy of {what}, and {version_url} doesn't name "
                                         "the BoringSSL version it builds with. A build is never bundled without "
                                         "its license text.")
        url = component.url.format(version=found.group(1))
    else:
        url = component.url.format(tag=tag)
    progress.say(f"The build has no copy of {what} - fetching it from {url}")
    return _fetch_text(http, url, what)


def copy_licenses(payload: Path, tag: str, http: Any, progress: PlainProgress, *,
                  os_key: Optional[str] = None, components: Optional[dict[str, str]] = None) -> list[str]:
    """Gather the build's license texts into ``<payload>/licenses/``; return their relative paths.

    1. the archive's own license files (the Linux and macOS tarballs carry
       llama.cpp's LICENSE; the Windows zips only LICENSE-LLVM-OpenMP);
    2. the texts llama.cpp embeds in its programs (see :func:`scan_programs`);
    3. llama.cpp's own MIT license, when no text so far is it; and every
       other part `os_key`'s builds must carry (:data:`REQUIRED_LICENSES`,
       plus BoringSSL / LLVM OpenMP when found in the build) that is still
       missing - fetched from the same release's source on GitHub. If one
       can't be had, this raises: a build is never bundled without them.

    `components`, when given, is filled with ``{component name: relative path}``.
    """
    target_dir = payload / LICENSE_DIR
    sources = _license_files(payload)
    target_dir.mkdir(exist_ok=True)
    copied: list[str] = []
    have: dict[str, str] = {}  # component name -> the license file that covers it
    texts: set[bytes] = set()

    def add(rel_target: str, text: bytes, component: Optional[str]) -> None:
        if rel_target not in copied:
            copied.append(rel_target)
        texts.add(_normalised(text))
        if component and component not in have:
            have[component] = rel_target

    def write_new(name: str, text: bytes) -> str:
        target, number = target_dir / name, 1
        while target.exists() and _normalised(target.read_bytes()) != _normalised(text):
            number += 1
            target = target_dir / f"{name}-{number}"  # never overwrite a different license text
        if not target.exists():
            target.write_bytes(text)
        return target.relative_to(payload).as_posix()

    for src in sources:
        data = src.read_bytes()
        if src.parent == target_dir:  # the archive's own licenses/ folder: already in place
            add(src.relative_to(payload).as_posix(), data, identify_license(src.name, data))
            continue
        rel = src.relative_to(payload)
        name = rel.name if len(rel.parts) == 1 else "-".join(rel.parts)  # "3rdparty/LICENSE" -> "3rdparty-LICENSE"
        target, number = target_dir / name, 1
        while target.exists() and target.read_bytes() != data:
            number += 1
            target = target_dir / f"{name}-{number}"  # never overwrite a different license text
        if not target.exists():
            shutil.copy2(src, target)
        add(target.relative_to(payload).as_posix(), data, identify_license(name, data))

    embedded, found_in_code = scan_programs(payload)
    for label, text in embedded.items():
        component = component_named(label)
        name = component.name if component else label
        if _normalised(text) in texts:  # the archive already carries this exact text
            if component and name not in have:
                have[name] = next(rel for rel in copied
                                  if _normalised((payload / rel).read_bytes()) == _normalised(text))
            continue
        add(write_new(f"LICENSE-{_slug(name)}", text), text, name)

    required = list(REQUIRED_LICENSES.get(os_key or "", ("llama.cpp",)))
    if "llama.cpp" not in required:
        required.insert(0, "llama.cpp")
    required += sorted(found_in_code - set(required))
    for name in required:
        if name in have:
            continue
        component = component_named(name)
        assert component is not None, name
        text = fetch_component_license(component, tag, http, progress, payload=payload)
        add(write_new(f"LICENSE-{_slug(component.name)}", text), text, component.name)

    if components is not None:
        order = {c.name: i for i, c in enumerate(LICENSE_COMPONENTS)}
        components.clear()
        components.update(sorted(have.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))
    return copied


# ---------------------------------------------------------------------------
# Windows: the Visual C++ runtime, and checking what each program imports
# ---------------------------------------------------------------------------


def pe_imports(path: Path) -> Optional[list[str]]:
    """The DLL names a Windows program or DLL imports (normal and delay-loaded); None if it isn't one."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        if data[:2] != b"MZ":
            return None
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return None
        sections_count, = struct.unpack_from("<H", data, pe + 6)
        optional_size, = struct.unpack_from("<H", data, pe + 20)
        optional = pe + 24
        magic, = struct.unpack_from("<H", data, optional)
        directories = optional + {0x10B: 96, 0x20B: 112}[magic]  # PE32 / PE32+
        directories_count, = struct.unpack_from("<I", data, directories - 4)
        table = optional + optional_size
        sections = [struct.unpack_from("<IIII", data, table + 40 * i + 8) for i in range(sections_count)]
    except (struct.error, KeyError):
        return None

    def offset(rva: int) -> Optional[int]:
        for virtual_size, virtual_address, raw_size, raw_pointer in sections:
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return raw_pointer + rva - virtual_address
        return None

    names: list[str] = []
    # (data directory, descriptor size, where the DLL name's address is in it): imports, delay-load imports
    for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
        if index >= directories_count:
            continue
        rva, _size = struct.unpack_from("<II", data, directories + 8 * index)
        at = offset(rva) if rva else None
        while at is not None and at + size <= len(data):
            entry = data[at:at + size]
            if entry == bytes(size):
                break
            name_offset = offset(struct.unpack_from("<I", entry, name_at)[0])
            if name_offset is None:
                break
            end = data.find(b"\0", name_offset, name_offset + 260)
            if end < 0:
                break
            names.append(data[name_offset:end].decode("ascii", "replace"))
            at += size
    return names


def default_vc_runtime_dir() -> Optional[Path]:
    """Where this computer keeps the Visual C++ runtime: Windows' System32 (None on other systems)."""
    if not sys.platform.startswith("win"):
        return None
    return Path(os.environ.get("SystemRoot") or r"C:\Windows") / "System32"


def add_vc_runtime(payload: Path, source: Path, progress: PlainProgress) -> list[str]:
    """Copy the Visual C++ runtime DLLs from `source` next to the build's programs; return their names."""
    present = {p.name.lower(): p for p in payload.iterdir() if p.is_file()}
    added = []
    for name in VC_RUNTIME_DLLS:
        if name in present:  # a future archive may carry its own: keep that one
            added.append(present[name].name)
            continue
        src = next((p for p in (source / name, source / name.upper()) if p.is_file()), None)
        if src is None:
            raise ri.RuntimeInstallError(
                f"The Visual C++ runtime file {name} isn't in {source}, so the Windows engine would need players "
                "to install the Visual C++ Redistributable. (Pass --vc-runtime <folder> with msvcp140.dll, "
                "vcruntime140.dll and vcruntime140_1.dll, or --vc-runtime none to skip it.)"
            )
        shutil.copy2(src, payload / name)
        added.append(name)
    progress.say(f"Added the Visual C++ runtime from {source}: {', '.join(added)}")
    return added


def unresolved_windows_imports(payload: Path, *, allow: Iterable[str] = ()) -> dict[str, list[str]]:
    """``{program: [DLLs it imports that are neither in its folder nor part of Windows]}`` (empty = fine).

    Windows looks for a program's DLLs in its own folder, then System32:
    so whatever isn't a Windows DLL must ship next to it.
    """
    here = {p.name.lower() for p in payload.iterdir() if p.is_file()}
    allowed = {name.lower() for name in allow}
    problems: dict[str, list[str]] = {}
    for path in sorted(payload.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".exe", ".dll"):
            continue
        missing = []
        for name in pe_imports(path) or []:
            key = name.lower()
            if key in here or key in allowed or key in WINDOWS_SYSTEM_DLLS or _WINDOWS_API_SET_RE.match(key):
                continue
            if name not in missing:
                missing.append(name)
        if missing:
            problems[path.name] = missing
    return problems


def check_windows_imports(payload: Path, variant: ri.RuntimeVariant, *, allow: Iterable[str] = ()) -> None:
    """Raise if a program in the build imports a DLL a player's PC may not have."""
    problems = unresolved_windows_imports(payload, allow=allow)
    if problems:
        missing = sorted({name for names in problems.values() for name in names}, key=str.lower)
        programs = sorted(problems, key=lambda name: (name.lower() != "llama-server.exe", name.lower()))
        who = ", ".join(programs[:3]) + (f" and {len(programs) - 3} more" if len(programs) > 3 else "")
        raise ri.RuntimeInstallError(
            f"The {variant.display} build's programs ({who}) need {', '.join(missing)}, which neither ship with it "
            "nor come with Windows. Add them to the build (see VC_RUNTIME_DLLS) or, if Windows provides them, "
            "to WINDOWS_SYSTEM_DLLS in packaging/fetch_engine.py."
        )


# ---------------------------------------------------------------------------
# Linux: OpenSSL 3, and checking what each program needs (ELF NEEDED entries)
# ---------------------------------------------------------------------------


class ElfInfo(NamedTuple):
    machine: int  # e_machine: 62 = x86-64, 183 = AArch64
    needed: tuple[str, ...]  # the DT_NEEDED libraries, in order


def elf_info(path: Path) -> Optional[ElfInfo]:
    """The processor and needed libraries of a Linux program or library; None if it isn't one.

    Reads the ELF program headers, the dynamic segment and its string table
    - no ``readelf`` needed, so it works on any computer. A program with no
    dynamic segment (statically linked) needs nothing.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
            if len(head) < 52 or head[:4] != b"\x7fELF" or head[4] not in (1, 2) or head[5] not in (1, 2):
                return None
            end = "<" if head[5] == 1 else ">"
            machine = struct.unpack_from(end + "H", head, 18)[0]
            if head[4] == 2:  # 64-bit
                if len(head) < 64:
                    return None
                phoff = struct.unpack_from(end + "Q", head, 32)[0]
                phentsize, phnum = struct.unpack_from(end + "HH", head, 54)
                ph_format, dyn_format = end + "IIQQQQQQ", end + "qQ"
                fields = (0, 2, 3, 5)  # p_type, p_offset, p_vaddr, p_filesz
            else:
                phoff = struct.unpack_from(end + "I", head, 28)[0]
                phentsize, phnum = struct.unpack_from(end + "HH", head, 42)
                ph_format, dyn_format = end + "IIIIIIII", end + "iI"
                fields = (0, 1, 2, 4)
            if phnum > 4096 or phentsize < struct.calcsize(ph_format):
                return None

            def read_at(offset: int, size: int) -> bytes:
                fh.seek(offset)
                return fh.read(size)

            loads: list[tuple[int, int, int]] = []  # (vaddr, offset, filesz)
            dynamic: Optional[tuple[int, int]] = None
            table = read_at(phoff, phentsize * phnum)
            for index in range(phnum):
                entry = struct.unpack_from(ph_format, table, index * phentsize)
                p_type, p_offset, p_vaddr, p_filesz = (entry[i] for i in fields)
                if p_type == 1:  # PT_LOAD
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:  # PT_DYNAMIC
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return ElfInfo(machine, ())
            raw = read_at(dynamic[0], min(dynamic[1], 1024 * 1024))
            step = struct.calcsize(dyn_format)
            needed_at: list[int] = []
            strtab = strsz = None
            for at in range(0, len(raw) - step + 1, step):
                d_tag, d_val = struct.unpack_from(dyn_format, raw, at)
                if d_tag == 0:  # DT_NULL
                    break
                if d_tag == 1:  # DT_NEEDED: an offset into the string table
                    needed_at.append(d_val)
                elif d_tag == 5:  # DT_STRTAB (an address)
                    strtab = d_val
                elif d_tag == 10:  # DT_STRSZ
                    strsz = d_val
            if not needed_at:
                return ElfInfo(machine, ())
            if strtab is None:
                return None
            offset = next((off + strtab - vaddr for vaddr, off, size in loads if vaddr <= strtab < vaddr + size), None)
            if offset is None:
                return None
            strings = read_at(offset, min(strsz or 1024 * 1024, 16 * 1024 * 1024))
    except (OSError, struct.error, ValueError):
        return None
    names: list[str] = []
    for at in needed_at:
        stop = strings.find(b"\0", at)
        if at >= len(strings) or stop < 0:
            return None
        names.append(strings[at:stop].decode("utf-8", "replace"))
    return ElfInfo(machine, tuple(names))


def _linux_programs(folder: Path) -> Iterator[tuple[Path, ElfInfo]]:
    """The ELF files directly in `folder` (links to them are skipped: their target is listed)."""
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        info = elf_info(path)
        if info is not None:
            yield path, info


def default_openssl_dir(arch: Optional[str]) -> Optional[Path]:
    """Where this Linux computer keeps OpenSSL 3 for `arch` (None elsewhere, or if it hasn't got it)."""
    if not sys.platform.startswith("linux"):
        return None
    for folder in LINUX_LIBRARY_DIRS.get(arch or "", ()):
        paths = [Path(folder) / name for name in LINUX_OPENSSL_LIBS]
        if all(p.is_file() for p in paths):
            infos = [elf_info(p) for p in paths]
            if all(info is not None and info.machine == ELF_MACHINES.get(arch or "") for info in infos):
                return Path(folder)
    return None


def needs_openssl(folder: Path) -> bool:
    """Does a program in the build's folder need OpenSSL 3's libraries?"""
    wanted = set(LINUX_OPENSSL_LIBS)
    return any(wanted & set(info.needed) for _path, info in _linux_programs(folder))


def add_linux_openssl(folder: Path, source: Path, progress: PlainProgress, *,
                      arch: Optional[str] = None) -> list[str]:
    """Copy OpenSSL 3 from `source` next to the build's programs if they need it; return the file names.

    Nothing is copied for a build that doesn't need it (e.g. one that links
    OpenSSL statically); a copy an archive already carries is kept.
    """
    if not needs_openssl(folder):
        return []
    present = {p.name for p in folder.iterdir() if p.is_file()}
    added = []
    for name in LINUX_OPENSSL_LIBS:
        if name in present:
            added.append(name)
            continue
        src = source / name
        info = elf_info(src) if src.is_file() else None
        if info is None:
            raise ri.RuntimeInstallError(
                f"OpenSSL 3's {name} isn't in {source}, so the Linux engine couldn't start under Steam's Linux "
                "runtime (it has no OpenSSL 3). Install libssl3 on the build machine, or pass --linux-openssl "
                "<folder> with libssl.so.3 and libcrypto.so.3 (or --linux-openssl none to skip it)."
            )
        if arch in ELF_MACHINES and info.machine != ELF_MACHINES[arch]:
            raise ri.RuntimeInstallError(f"{src} isn't made for {arch} processors, so it can't ship with this "
                                         "build. Pass --linux-openssl <folder> with the right libraries.")
        shutil.copyfile(src.resolve(), folder / name)  # the library itself, not a link to it
        os.chmod(folder / name, 0o755)
        added.append(name)
    version = openssl_version(folder)
    progress.say(f"Added OpenSSL {version or '3'} from {source}: {', '.join(added)}")
    return added


def unresolved_linux_libraries(folder: Path, *, allow: Iterable[str] = ()) -> dict[str, list[str]]:
    """``{program: [libraries it needs that are neither in its folder nor on every player's system]}``.

    Empty means fine. The programs look for their libraries next to
    themselves (RUNPATH ``$ORIGIN``, and the game's ``LD_LIBRARY_PATH``),
    then in the system's folders - which, under Steam, are the Linux
    runtime's: so whatever isn't in :data:`LINUX_RUNTIME_LIBS` must ship.
    """
    here = {p.name for p in folder.iterdir() if p.is_file()}  # (links count: they're how sonames resolve)
    allowed = set(allow) | LINUX_RUNTIME_LIBS
    problems: dict[str, list[str]] = {}
    for path, info in _linux_programs(folder):
        missing = [name for name in dict.fromkeys(info.needed) if name not in here and name not in allowed]
        if missing:
            problems[path.name] = missing
    return problems


def check_linux_libraries(folder: Path, variant: ri.RuntimeVariant, *, allow: Iterable[str] = ()) -> None:
    """Raise if a program in the build needs a library a player's system (or Steam's runtime) may not have."""
    problems = unresolved_linux_libraries(folder, allow=allow)
    if problems:
        missing = sorted({name for names in problems.values() for name in names})
        programs = sorted(problems, key=lambda name: (name != "llama-server", name))
        who = ", ".join(programs[:3]) + (f" and {len(programs) - 3} more" if len(programs) > 3 else "")
        raise ri.RuntimeInstallError(
            f"The {variant.display} build's programs ({who}) need {', '.join(missing)}, which neither ship with it "
            "nor come with every Linux (Steam's Linux runtime included). Add them to the build (see "
            "LINUX_OPENSSL_LIBS) or, if every player's system provides them, to LINUX_RUNTIME_LIBS in "
            "packaging/fetch_engine.py."
        )


# ---------------------------------------------------------------------------
# The model architectures the release knows
# ---------------------------------------------------------------------------


def parse_architectures(source: str) -> list[str]:
    """The architecture names in llama.cpp's ``src/llama-arch.cpp`` (LLM_ARCH_NAMES), sorted."""
    table = _ARCH_TABLE_RE.search(source)
    if table is None:
        return []
    names = {name.lower() for key, name in _ARCH_ENTRY_RE.findall(table.group(1)) if key != "UNKNOWN"}
    return sorted(name for name in names if re.fullmatch(r"[a-z0-9][a-z0-9_.\-]*", name))


def fetch_architectures(tag: str, http: Any, progress: PlainProgress) -> list[str]:
    """The model architectures llama.cpp `tag` can load, from that release's source; raises if unreadable."""
    url = ARCHITECTURES_URL.format(tag=tag)
    progress.say(f"Reading the model architectures llama.cpp {tag} knows: {url}")
    try:
        resp = http.request("GET", url, headers={"User-Agent": ri.USER_AGENT}, timeout=30.0)
        with resp:
            status, body = resp.status, resp.read()
    except ri.NETWORK_ERRORS as exc:
        raise ri.RuntimeInstallError(f"Couldn't read the model architectures llama.cpp {tag} knows ({exc}).") from exc
    names = parse_architectures(body.decode("utf-8", "replace")) if status == 200 else []
    if len(names) < MIN_ARCHITECTURES:
        raise ri.RuntimeInstallError(
            f"Couldn't read the model architectures llama.cpp {tag} knows from {url} (HTTP {status}, "
            f"{len(names)} found). A built game needs them to offer only models its engine can run - if "
            "llama.cpp moved its LLM_ARCH_NAMES table, update ARCHITECTURES_URL / parse_architectures."
        )
    progress.say(f"  {len(names)} architectures, e.g. {', '.join(names[:6])}...")
    return names


def unpack_build(work: Path, variant: ri.RuntimeVariant, release: dict, assets: list[dict],
                 archives: list[Path], http: Any, progress: PlainProgress, *, os_key: Optional[str] = None,
                 arch: Optional[str] = None, vc_runtime: Optional[Path] = None,
                 linux_openssl: Optional[Path] = None, architectures: Optional[Sequence[str]] = None) -> Path:
    """Unpack one build's archive(s) into a staging folder with its install.json; return that folder.

    `os_key`/`arch` are the ``--os``/``--arch`` the build is for (which
    licenses it must carry; Windows builds get `vc_runtime`'s DLLs and their
    imports checked, Linux builds `linux_openssl`'s OpenSSL 3 and their
    needed libraries). `architectures` (the model architectures the release
    knows) is recorded in install.json.
    """
    payload = work / f"build-{variant.name}"
    payload.mkdir()
    for index, archive in enumerate(archives):
        ri.unpack_archive(archive, payload, work / f"unpacked-{variant.name}-{index}")
    exe = ri.finish_unpacked(payload)
    tag = str(release.get("tag_name") or "")
    openssl_files: list[str] = []
    if os_key == "linux":
        if linux_openssl is not None:
            openssl_files = add_linux_openssl(exe.parent, linux_openssl, progress, arch=arch)
        elif needs_openssl(exe.parent):
            progress.warn("Not adding OpenSSL 3 (it comes from a Linux build machine's system libraries, or "
                          "--linux-openssl <folder>): under Steam's Linux runtime, which has no OpenSSL 3, this "
                          "engine can't start.")
    components: dict[str, str] = {}
    licenses = copy_licenses(payload, tag, http, progress, os_key=os_key, components=components)
    vc_files: list[str] = []
    if os_key == "windows":
        if vc_runtime is not None:
            vc_files = add_vc_runtime(exe.parent, vc_runtime, progress)
        else:
            progress.warn("Not adding the Visual C++ runtime (it comes from a Windows computer's System32, or "
                          "--vc-runtime <folder>): players without the Visual C++ Redistributable can't start "
                          "this engine.")
        check_windows_imports(exe.parent, variant, allow=() if vc_runtime is not None else VC_RUNTIME_DLLS)
    if os_key == "linux":
        check_linux_libraries(exe.parent, variant, allow=() if linux_openssl is not None else LINUX_OPENSSL_LIBS)
    marker = ri.install_marker(release, assets, variant, exe.relative_to(payload).as_posix(),
                               bundled=True, license_files=licenses)
    marker["license_components"] = components
    if vc_files:
        marker["vc_runtime"] = vc_files
    if openssl_files:
        marker["openssl"] = openssl_files
        marker["openssl_version"] = openssl_version(exe.parent) or ""
    if architectures:
        marker["architectures"] = sorted(set(architectures))
    (payload / ri.INSTALL_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return payload


def _engine_env(exe: Path) -> dict[str, str]:
    """Environment for a ``--version`` check: finds the libraries next to the exe; no tokens."""
    from gettowork import config

    env = config.child_env()
    here = str(exe.parent)
    if sys.platform.startswith("win"):
        env["PATH"] = here + os.pathsep + env.get("PATH", "")
    elif sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = here + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def verify_builds(builds: list[tuple[ri.RuntimeVariant, Path]], *, runner: Callable[..., Any],
                  host: tuple[Optional[str], Optional[str]], target: tuple[str, str],
                  progress: PlainProgress) -> None:
    """Run ``llama-server --version`` for each build this computer can run; raise if one fails."""
    if tuple(host) != tuple(target):
        progress.say(f"Skipping the --version check: these builds are for {target[0]}/{target[1]}, "
                     f"this computer is {host[0]}/{host[1]}.")
        return
    failures = []
    for variant, folder in builds:
        marker = json.loads((folder / ri.INSTALL_MARKER).read_text(encoding="utf-8"))
        # Absolute: with cwd set, Linux and macOS would look for a relative program path inside that folder.
        exe = (folder / Path(*PurePosixPath(marker["exe"]).parts)).absolute()
        progress.say(f"Checking the {variant.display} build: {exe.name} --version")
        try:
            result = runner([str(exe), "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=str(exe.parent), env=_engine_env(exe),
                            timeout=VERIFY_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{variant.name}: couldn't run it ({exc})")
            continue
        raw = getattr(result, "stdout", b"") or b""
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        for line in [ln for ln in text.strip().splitlines() if ln.strip()][-6:]:
            progress.say(f"  {line}")
        code = getattr(result, "returncode", 1)
        if code != 0:
            failures.append(f"{variant.name}: exit code {code}")
    if failures:
        raise ri.RuntimeInstallError("The engine check failed - " + "; ".join(failures) + ".")


def _remove_stale_builds(dest: Path, keep: set[str], variants: set[str], progress: PlainProgress) -> None:
    """Delete bundled builds of the same types from an earlier run (another release), so only one ships."""
    for folder in sorted(p for p in dest.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if folder.name in keep:
            continue
        try:
            marker = json.loads((folder / ri.INSTALL_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # not one of ours: leave it alone
        if isinstance(marker, dict) and marker.get("bundled") is True and marker.get("variant") in variants:
            progress.say(f"Removing an older bundled build: {folder.name}")
            shutil.rmtree(folder, ignore_errors=True)


def pinned_asset(asset: dict, pin: PinnedRelease) -> dict:
    """`asset` with the SHA-256 pinned in ``llama_cpp_tag.txt`` as the one its download must match.

    Raises when the pin has no fingerprint for it, or when GitHub now
    reports a different one: the release's file was replaced after the pin
    was reviewed, and the build never ships bytes nobody reviewed.
    """
    name = str(asset.get("name") or "")
    expected = pin.digests.get(name)
    if expected is None:
        raise ri.RuntimeInstallError(
            f"packaging/llama_cpp_tag.txt pins no SHA-256 for {name}. Add a '<sha256>  {name}' line for it "
            "(sha256sum's output) once you've checked that archive."
        )
    published = str(asset.get("digest") or "").strip().lower()
    if _has_digest(asset) and published != f"sha256:{expected}":
        raise ri.RuntimeInstallError(
            f"GitHub now reports a different SHA-256 for {name} ({published[7:19]}...) than the one pinned in "
            f"packaging/llama_cpp_tag.txt ({expected[:12]}...): the release's file changed after the pin was "
            "reviewed, so it isn't bundled. Check the release before updating the pin."
        )
    return dict(asset, digest=f"sha256:{expected}")


def fetch(args: argparse.Namespace, *, http: Any, runner: Callable[..., Any],
          host: tuple[Optional[str], Optional[str]], progress: PlainProgress) -> str:
    """Do the work described in the module docstring; returns the chosen tag (raises on failure)."""
    os_name = OS_NAMES[args.os]
    variants: list[ri.RuntimeVariant] = []
    for name in [n.strip() for n in str(args.variants).split(",") if n.strip()]:
        variant = ri.variant_by_name(name)
        if variant is None:
            raise ri.RuntimeInstallError(f"Unknown engine build {name!r} (known: {', '.join(sorted(ri.KNOWN_VARIANTS))}).")
        if variant not in variants:
            variants.append(variant)
    if not variants:
        raise ri.RuntimeInstallError("Name at least one engine build with --variants (e.g. vulkan,cpu).")
    wanted = ", ".join(v.name for v in variants)

    tag_arg = (args.tag or "auto").strip()
    pin: Optional[PinnedRelease] = None
    if tag_arg.lower() == "pinned":
        pin = pinned_release()
        tag_arg = pin.tag
    if tag_arg.lower() in ("auto", "latest"):
        progress.say(f"Looking for the newest llama.cpp release with {wanted} builds for {args.os}/{args.arch}...")
        releases = ri.fetch_releases(http=http, limit=SEARCH_RELEASES, ui=progress)
        where = f"the {len(releases)} newest llama.cpp releases"
    else:
        progress.say(f"Fetching llama.cpp release {tag_arg}...")
        releases = [ri.fetch_release(tag_arg, http=http, ui=progress)]
        where = f"llama.cpp release {tag_arg}"
    picked = choose_release(releases, variants, os_name, args.arch)
    if picked is None:
        raise ri.RuntimeInstallError(f"None of {where} has all of these builds for {args.os}/{args.arch}: {wanted}.")
    release, assets_by_variant = picked
    tag = str(release.get("tag_name") or "")
    if not _SAFE_TAG_RE.match(tag):
        raise ri.RuntimeInstallError(f"The release tag {tag!r} doesn't look like a llama.cpp tag, so I stopped.")
    progress.say(f"Using llama.cpp {tag}: {release.get('html_url') or ri.LLAMA_CPP_URL + '/releases/tag/' + tag}")

    if pin is not None:
        assets_by_variant = {name: [pinned_asset(asset, pin) for asset in assets]
                             for name, assets in assets_by_variant.items()}
    for variant in variants:
        for asset in assets_by_variant[variant.name]:
            if not args.allow_missing_digest and not _has_digest(asset):
                raise ri.RuntimeInstallError(
                    f"GitHub published no SHA-256 fingerprint for {asset.get('name')}, so it can't be checked. "
                    "(Pass --allow-missing-digest to accept that.)"
                )

    vc_runtime: Optional[Path] = None
    if args.os == "windows":
        choice = str(args.vc_runtime or "auto").strip()
        if choice.lower() == "auto":
            vc_runtime = default_vc_runtime_dir()
        elif choice.lower() != "none":
            vc_runtime = Path(choice).expanduser()
            if not vc_runtime.is_dir():
                raise ri.RuntimeInstallError(f"--vc-runtime {choice}: no such folder.")
    linux_openssl: Optional[Path] = None
    if args.os == "linux":
        choice = str(args.linux_openssl or "auto").strip()
        if choice.lower() == "auto":
            linux_openssl = default_openssl_dir(args.arch)
        elif choice.lower() != "none":
            linux_openssl = Path(choice).expanduser()
            if not linux_openssl.is_dir():
                raise ri.RuntimeInstallError(f"--linux-openssl {choice}: no such folder.")
    architectures = fetch_architectures(tag, http, progress)

    # Absolute from here on (CI passes a relative --dest; see verify_builds).
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".fetch-engine-", dir=dest))
    try:
        archives: dict[str, Path] = {}  # asset name -> file (Metal and CPU share one macOS archive)
        staged: list[tuple[ri.RuntimeVariant, Path]] = []
        for variant in variants:
            assets = assets_by_variant[variant.name]
            for asset in assets:
                if asset["name"] not in archives:
                    archives[asset["name"]] = ri.download_asset(http, asset, work / "downloads", progress, asset["name"])
            progress.say(f"Unpacking the {variant.display} build...")
            staged.append((variant, unpack_build(work, variant, release, assets,
                                                 [archives[a["name"]] for a in assets], http, progress,
                                                 os_key=args.os, arch=args.arch, vc_runtime=vc_runtime,
                                                 linux_openssl=linux_openssl, architectures=architectures)))
        # Everything is downloaded, checked and unpacked: only now replace what was there.
        builds: list[tuple[ri.RuntimeVariant, Path]] = []
        for variant, folder in staged:
            target = dest / f"{tag}-{variant.name}"
            if target.exists():
                shutil.rmtree(target)
            os.replace(folder, target)
            builds.append((variant, target))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _remove_stale_builds(dest, {folder.name for _v, folder in builds}, {v.name for v in variants}, progress)

    if args.verify:
        verify_builds(builds, runner=runner, host=host, target=(args.os, args.arch), progress=progress)
    for variant, folder in builds:
        progress.say(f"Ready: {folder}")
    return tag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_engine.py",
        description="Download the official llama.cpp builds that ship inside the game.",
    )
    parser.add_argument("--os", required=True, choices=sorted(OS_NAMES), help="the OS the game is built for")
    parser.add_argument("--arch", required=True, choices=ARCHES, help="the processor the game is built for")
    parser.add_argument("--variants", required=True, help="comma-separated engine builds, e.g. vulkan,cpu")
    parser.add_argument("--dest", required=True, help="folder to put the builds in, e.g. build/engine")
    parser.add_argument("--tag", default="auto",
                        help="a llama.cpp release tag (e.g. b7000), pinned (packaging/llama_cpp_tag.txt, what the "
                             "game builds use) or auto (the newest)")
    parser.add_argument("--verify", action="store_true", help="run llama-server --version for builds this computer can run")
    parser.add_argument("--vc-runtime", default="auto",
                        help="Windows builds: the folder to copy msvcp140.dll, vcruntime140.dll and vcruntime140_1.dll "
                             "from; auto (the default) = this Windows computer's System32, none = don't add them")
    parser.add_argument("--linux-openssl", default="auto",
                        help="Linux builds: the folder to copy OpenSSL 3 (libssl.so.3, libcrypto.so.3) from; auto (the "
                             "default) = this Linux computer's system libraries, none = don't add it")
    parser.add_argument("--allow-missing-digest", action="store_true",
                        help="accept archives GitHub published no SHA-256 fingerprint for")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, http: Any = None, runner: Optional[Callable[..., Any]] = None,
         host: Optional[tuple[Optional[str], Optional[str]]] = None, out: Any = None) -> int:
    """Command-line entry point; returns the exit code (0 = success, 1 = failure).

    `http`, `runner` (like ``subprocess.run``) and `host` can be swapped out, which is how the
    tests run without a network or real engines.
    """
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # --help (0) or a usage error (argparse says why)
        return 0 if exc.code in (0, None) else 1
    progress = PlainProgress(out)
    try:
        tag = fetch(args, http=http or ri.UrllibHttp(), runner=runner or subprocess.run,
                    host=host or host_platform(), progress=progress)
    except (ri.RuntimeInstallError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr, flush=True)
        return 1
    progress.say(f"LLAMA_CPP_TAG={tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
