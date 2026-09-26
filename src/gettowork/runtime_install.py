"""Download and unpack the official prebuilt llama.cpp engine (``llama-server``).

The game runs your chosen model with **llama.cpp**, an MIT-licensed C/C++
inference engine. Instead of asking players to compile it, we download the
ready-made build the llama.cpp team publishes on GitHub for every release:

1. :func:`plan_variants` decides which builds could work on this computer,
   best first (e.g. NVIDIA CUDA -> Vulkan -> CPU). CPU always comes last
   because it runs everywhere.
2. :func:`fetch_releases` asks the GitHub API for the most recent releases.
   (llama.cpp publishes its builds as *prereleases*, so the "latest release"
   endpoint would miss them - we list releases instead.)
3. :func:`select_assets` finds the right archive(s) for a variant by matching
   file names with tolerant regular expressions, because names drift a little
   over time.
4. :func:`ensure_llama_server` downloads, verifies (size + SHA-256), safely
   unpacks and records the install, then returns the path to ``llama-server``.

Everything lives in ``config.runtime_dir()/llama.cpp/<tag>-<variant>/``.
Deleting that folder uninstalls it: no admin rights, nothing system-wide.

**The built game (Steam, or a double-clicked download)** ships the engine
*inside* the game instead - one ``<tag>-<variant>/`` folder per build in its
``engine`` folder, found through :mod:`gettowork.distribution` - and never
downloads programs while you play. Those built-in builds are read-only: they
are listed by :func:`installed_runtimes` like any other install, but never
tidied away or changed (a note that one can't run on this computer goes into
``config.runtime_dir()/bundled-unusable.json`` instead), and
:func:`ensure_llama_server` picks from them without touching the network.

The HTTP layer is a tiny wrapper around the standard library
(``urllib.request``) so you can see exactly what is sent; tests swap in a fake.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.client as http_client
import io
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from rich.markup import escape

from . import __version__, config, distribution
from .tls import CERTIFICATE_HELP, https_context, is_certificate_error
from .types import SystemSpecs
from .ui import UI

__all__ = [
    "RuntimeInstallError",
    "RuntimeVariant",
    "NETWORK_ERRORS",
    "CPU",
    "METAL",
    "VULKAN",
    "CUDA12",
    "CUDA13",
    "ROCM",
    "KNOWN_VARIANTS",
    "variant_by_name",
    "is_platform_supported",
    "plan_variants",
    "select_assets",
    "pick_release",
    "releases_newest_first",
    "fetch_releases",
    "fetch_release",
    "download_asset",
    "unpack_archive",
    "finish_unpacked",
    "install_marker",
    "ensure_llama_server",
    "installed_runtimes",
    "install_info",
    "downloads_allowed",
    "is_bundled",
    "find_installed",
    "choose_installed",
    "relocate_engine",
    "own_build_instead",
    "own_builds_first",
    "engine_architectures",
    "other_engines_hint",
    "llama_cpp_python_possible",
    "engine_summary",
    "ENGINE_MISSING_MESSAGE",
    "mark_unusable",
    "unusable_variants",
    "unusable_reasons",
    "unusable_message",
    "usable_plan",
    "available_plan",
    "prune_old_installs",
    "engine_problem",
    "bundled_builds_problem",
    "platform_problem",
    "engine_can_use_gpu",
    "license_text",
    "safe_extract",
    "find_server_executable",
    "UrllibHttp",
    "HttpResponse",
    "RUNTIME_EXPLAINER",
    "RUNTIME_EXPLAINER_BUILT_IN",
    "runtime_explainer",
]

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_LICENSE = "MIT"
CUDA_RUNTIME_LICENSE = "NVIDIA CUDA EULA"
CUDA_EULA_URL = "https://docs.nvidia.com/cuda/eula/"
GITHUB_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

INSTALL_MARKER = "install.json"
# Notes about built-in (bundled) builds that can't run on this computer live in
# the player's own data folder: the game's folder is read-only (and on a Mac,
# writing into the app would break its signature).
BUNDLED_UNUSABLE_FILE = "bundled-unusable.json"
ENGINE_MISSING_MESSAGE = (
    "The game's built-in engine is missing. On Steam: right-click Get To Work → Properties → Installed Files → "
    "Verify integrity. Otherwise re-download the game."
)
SERVER_NAMES = ("llama-server", "llama-server.exe")
PROJECT_URL = "https://github.com/markelphoenix/GetToWork"  # who is making these requests (User-Agent)
USER_AGENT = f"GetToWork/{__version__} (+{PROJECT_URL})"

_CHUNK = 256 * 1024
# Safety limits for unpacking: real llama.cpp archives are well under these.
_MAX_UNPACKED_BYTES = 8 * 1024**3
_MAX_MEMBERS = 20_000
_STALE_STAGING_S = 6 * 3600

# What a flaky network can throw at us: socket/URL errors are OSErrors, while
# http.client raises its own HTTPException for truncated or garbled responses.
NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, http_client.HTTPException)

# True on Python versions that ship tarfile's "data" extraction filter
# (3.12+, and security backports to 3.10.12+ / 3.11.4+).
_HAS_TAR_DATA_FILTER = hasattr(tarfile, "data_filter")


class RuntimeInstallError(RuntimeError):
    """The llama.cpp engine could not be found, downloaded or unpacked.

    The message is always friendly, plain English that can be shown as-is.
    """


# ---------------------------------------------------------------------------
# Variants: which build of llama.cpp to use
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeVariant:
    """One flavour of prebuilt llama.cpp (CPU, CUDA, Vulkan, Metal...).

    ``asset_patterns`` are regular-expression *templates* for the release file
    name, tried in order. They may use these placeholders, which
    :func:`select_assets` fills in for the player's OS and CPU architecture:

    * ``{tag}``  - optional release tag such as ``b7000-``
    * ``{os}``   - ``win``/``windows``, ``ubuntu``/``linux``, ``macos``...
    * ``{arch}`` - ``x64``/``x86_64``/``amd64`` or ``arm64``/``aarch64``

    Every template is anchored at the start and automatically allows harmless
    extra ``-suffix`` tokens before the ``.zip``/``.tar.gz`` extension.
    """

    name: str  # "cpu", "cuda-12", "cuda-13", "vulkan", "metal", "rocm"
    asset_patterns: tuple[str, ...]
    needs_cudart: bool = False  # also needs NVIDIA's CUDA runtime archive
    gpu: bool = False  # uses a graphics card (False = CPU only)
    cudart_patterns: tuple[str, ...] = ()  # templates for the CUDA runtime archive
    label: str = ""  # friendly name for the UI, e.g. "NVIDIA CUDA 12"
    platforms: tuple[str, ...] = ()  # limit to "windows"/"linux"/"darwin"; empty = any

    @property
    def display(self) -> str:
        return self.label or self.name


CPU = RuntimeVariant(
    name="cpu",
    label="CPU",
    asset_patterns=(
        r"llama-{tag}bin-{os}-cpu-{arch}",  # Windows:  llama-b7000-bin-win-cpu-x64.zip
        r"llama-{tag}bin-{os}-{arch}",  # Linux/Mac: llama-b7000-bin-ubuntu-x64.tar.gz
        r"llama-{tag}bin-{os}-(?:avx2|avx|noavx)-{arch}",  # older Windows naming
    ),
)
METAL = RuntimeVariant(
    name="metal",
    label="Apple Metal",
    gpu=True,
    platforms=("darwin",),
    # The Apple Silicon build has Metal built in: llama-b7000-bin-macos-arm64.tar.gz
    asset_patterns=(r"llama-{tag}bin-{os}-{arch}", r"llama-{tag}bin-{os}-metal-{arch}"),
)
VULKAN = RuntimeVariant(
    name="vulkan",
    label="Vulkan",
    gpu=True,
    asset_patterns=(r"llama-{tag}bin-{os}-vulkan-{arch}",),
)
CUDA12 = RuntimeVariant(
    name="cuda-12",
    label="NVIDIA CUDA 12",
    gpu=True,
    needs_cudart=True,
    # llama-b7000-bin-win-cuda-12.4-x64.zip, llama-b7000-bin-ubuntu-cuda-12.8-x64.tar.gz,
    # and the older llama-b4000-bin-win-cuda-cu12.4-x64.zip spelling.
    asset_patterns=(r"llama-{tag}bin-{os}-cuda-?(?:cu)?12(?:\.\d+)*-{arch}",),
    cudart_patterns=(r"cudart-llama-{tag}bin-{os}-(?:cuda-?)?(?:cu)?12(?:\.\d+)*-{arch}",),
)
CUDA13 = RuntimeVariant(
    name="cuda-13",
    label="NVIDIA CUDA 13",
    gpu=True,
    needs_cudart=True,
    asset_patterns=(r"llama-{tag}bin-{os}-cuda-?(?:cu)?13(?:\.\d+)*-{arch}",),
    cudart_patterns=(r"cudart-llama-{tag}bin-{os}-(?:cuda-?)?(?:cu)?13(?:\.\d+)*-{arch}",),
)
ROCM = RuntimeVariant(
    name="rocm",
    label="AMD ROCm",
    gpu=True,
    asset_patterns=(r"llama-{tag}bin-{os}-(?:rocm|hip)(?:-?\d+(?:\.\d+)*)?-{arch}",),
)

KNOWN_VARIANTS: dict[str, RuntimeVariant] = {v.name: v for v in (CUDA13, CUDA12, ROCM, VULKAN, METAL, CPU)}


def variant_by_name(name: Optional[str]) -> Optional[RuntimeVariant]:
    """Look up a built-in variant by its name ("cpu", "cuda-12", ...)."""
    return KNOWN_VARIANTS.get((name or "").strip().lower())


def _os_key(os_name: str) -> Optional[str]:
    s = (os_name or "").strip().lower()
    if s.startswith("win") or s in ("nt", "cygwin", "msys"):
        return "windows"
    if s in ("darwin", "macos", "mac", "osx"):
        return "darwin"
    if s == "linux":
        return "linux"
    return None


def _arch_key(arch: str) -> Optional[str]:
    s = (arch or "").strip().lower()
    if s in ("x86_64", "amd64", "x64", "x86-64", "em64t", "intel64"):
        return "x64"
    if s in ("arm64", "aarch64", "arm64e", "armv8", "armv8l", "aarch64_be"):
        return "arm64"
    return None


def is_platform_supported(os_name: str, arch: str) -> bool:
    """Does llama.cpp publish prebuilt engines for this OS + CPU architecture?"""
    return _os_key(os_name) is not None and _arch_key(arch) is not None


def _nvidia_driver_major(specs: SystemSpecs) -> Optional[int]:
    """Best-effort NVIDIA driver major version (e.g. 580), or None if unknown.

    Reads ``GPUInfo.driver_version`` (filled in from ``nvidia-smi``), and
    falls back to the detection notes (e.g. "NVIDIA driver 581.15").
    CUDA 13 builds need driver 580 or newer; CUDA 12 builds about 525+.
    """
    found: list[int] = []
    for gpu in specs.gpus:
        raw = getattr(gpu, "driver_version", None)
        if raw:
            m = re.match(r"\s*(\d{3,4})", str(raw))
            if m:
                found.append(int(m.group(1)))
    for note in specs.notes:
        low = note.lower()
        if "driver" in low and ("nvidia" in low or "cuda" in low or any(g.vendor == "nvidia" for g in specs.gpus)):
            m = re.search(r"driver[^0-9\n]{0,24}(\d{3,4})(?:\.\d+)*", note, re.IGNORECASE)
            if m:
                found.append(int(m.group(1)))
    return max(found) if found else None


def _system_has_vulkan_loader() -> bool:
    """Live check for the Vulkan loader on *this* computer (best effort)."""
    try:
        import platform

        system = platform.system()
        if system == "Linux":
            import ctypes.util

            return ctypes.util.find_library("vulkan") is not None
        if system == "Windows":
            root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
            return (root / "System32" / "vulkan-1.dll").exists()
    except Exception:
        pass
    return False


def _vulkan_available(specs: SystemSpecs) -> bool:
    """Did hardware detection find a Vulkan loader? (specs.py records "vulkan".)"""
    flags = {f.lower() for f in specs.cpu_flags}
    if "vulkan" in flags:
        return True
    if flags & {"no-vulkan", "novulkan"}:
        return False
    for note in specs.notes:
        low = note.lower()
        if "vulkan" in low:
            negative = re.search(r"\b(no|not|missing|unavailable|without|couldn't|could not)\b", low)
            return negative is None
    return _system_has_vulkan_loader()


# The newest glibc (the core system library) each official Linux build needs,
# from the machines llama.cpp's release workflow builds them on:
# - the CUDA builds, and every arm64 build, are made on Ubuntu 24.04 (with
#   GCC 13/14), so they need glibc 2.38+;
# - the x64 CPU and Vulkan builds are made on Ubuntu 22.04: glibc 2.35+.
# On older systems (Raspberry Pi OS Bookworm, Ubuntu 20.04/22.04 on arm64,
# Debian 11, RHEL/Rocky 8...) such a build can't even start, so it isn't
# offered - and when *no* build can start, the game says so before
# downloading anything. If upstream changes its build machines, the engine
# check right after installing (and the start-up fallback in llamaserver.py)
# still catch it.
CUDA_LINUX_MIN_GLIBC = (2, 38)
LINUX_MIN_GLIBC = {"x64": (2, 35), "arm64": (2, 38)}  # CPU and Vulkan builds, by CPU architecture
# The official macOS builds are made for macOS 13.3 (Ventura) and newer.
MACOS_MIN_VERSION = (13, 3)


def _min_glibc(variant: RuntimeVariant, arch: Optional[str]) -> tuple[int, int]:
    if variant.needs_cudart:
        return CUDA_LINUX_MIN_GLIBC
    return LINUX_MIN_GLIBC.get(arch or "", (2, 35))


def _macos_version() -> Optional[tuple[int, int]]:
    """This Mac's macOS version, e.g. (14, 5), or None (not a Mac / unknown)."""
    try:
        import platform

        if platform.system() != "Darwin":
            return None
        from .specs import macos_release  # sees through the "10.16" compatibility answer

        release = macos_release()
    except Exception:
        return None
    m = re.match(r"(\d+)(?:\.(\d+))?", release or "")
    version = (int(m.group(1)), int(m.group(2) or 0)) if m else None
    return None if version == (10, 16) else version  # 10.16 is never a real version: unknown


def _fmt_version(version: tuple[int, int]) -> str:
    return ".".join(str(x) for x in version)


def platform_problem(os_name: str, arch: str, *, glibc: Optional[tuple[int, int]] = None,
                     macos: Optional[tuple[int, int]] = None, live: bool = True) -> Optional[str]:
    """Why *no* official llama.cpp build can run on this computer, or None if one can.

    Checks the OS and CPU architecture, and the system versions the builds
    need (glibc on Linux, macOS 13.3+). `glibc` / `macos` default to this
    computer's own (read live) unless ``live=False``. The answer comes before
    anything is downloaded, so the game can offer Ollama or llama-cpp-python
    straight away instead of fetching an engine that can never start.
    """
    os_key, arch_key = _os_key(os_name), _arch_key(arch)
    if os_key is None or arch_key is None:
        return f"There's no official prebuilt llama.cpp engine for {os_name} on {arch}. " + other_engines_hint("may still work")
    if os_key == "linux":
        glibc = glibc if glibc is not None else (_glibc_version() if live else None)
        need = LINUX_MIN_GLIBC[arch_key]
        if glibc is not None and glibc < need:
            return (f"The official llama.cpp engine needs a newer Linux than this one (glibc {_fmt_version(need)} "
                    f"or newer; this computer has {_fmt_version(glibc)}), so it couldn't start here. "
                    + other_engines_hint())
    if os_key == "darwin":
        macos = macos if macos is not None else (_macos_version() if live else None)
        if macos == (10, 16):
            macos = None  # Apple's compatibility answer to old-SDK programs, never a real version
        if macos is not None and macos < MACOS_MIN_VERSION:
            return (f"The official llama.cpp engine needs macOS {_fmt_version(MACOS_MIN_VERSION)} or newer "
                    f"(this Mac has {_fmt_version(macos)}). " + other_engines_hint("may still work"))
    return None


def _glibc_version() -> Optional[tuple[int, int]]:
    """This computer's glibc version, e.g. (2, 35), or None (not Linux / not glibc)."""
    try:
        import platform

        if platform.system() != "Linux":
            return None
        lib, version = platform.libc_ver()
    except Exception:
        return None
    m = re.match(r"(\d+)\.(\d+)", version or "")
    if lib != "glibc" or not m:
        return None
    return int(m.group(1)), int(m.group(2))


CUDA13_MIN_COMPUTE_CAPABILITY = 7.5  # Turing (RTX 20xx / GTX 16xx) and newer


def _cuda_variants(specs: SystemSpecs, arch: str) -> list[RuntimeVariant]:
    driver = _nvidia_driver_major(specs)
    nvidia = [g for g in specs.gpus if g.vendor == "nvidia"]
    if driver is None and nvidia and all(g.vram_gb <= 0 for g in nvidia):
        # Seen (e.g. by lspci) but the NVIDIA driver never answered - perhaps
        # the open-source "nouveau" driver. CUDA builds can't work without the
        # real driver, so don't download ~0.6-0.8 GB of CUDA build for nothing.
        return []
    # CUDA 13 dropped Maxwell, Pascal and Volta cards (GTX 9xx/10xx, Titan V:
    # compute capability below 7.5), so its builds contain no code for them.
    # Driver 580 is the last to support those cards, so their owners often have
    # it - offering CUDA 13 there would download ~0.6 GB that can't run.
    caps = [g.compute_capability for g in nvidia if g.compute_capability is not None]
    cuda13_ok = not caps or max(caps) >= CUDA13_MIN_COMPUTE_CAPABILITY
    if arch == "arm64":
        # Only CUDA 13 builds are published for arm64; CUDA 13 needs driver >= 580.
        return [CUDA13] if (driver is None or driver >= 580) and cuda13_ok else []
    if driver is None:
        return [CUDA12]
    if driver >= 580:
        return [CUDA13, CUDA12] if cuda13_ok else [CUDA12]
    if driver >= 525:
        return [CUDA12]
    return []  # too old for CUDA 12: Vulkan or CPU will do


def plan_variants(specs: SystemSpecs) -> list[RuntimeVariant]:
    """Which llama.cpp builds to try on this computer, best first.

    The list always ends with the CPU build, which works everywhere.
    If a GPU build fails to start, the backend moves on to the next entry.
    """
    os_key = _os_key(specs.os_name)
    arch = _arch_key(specs.arch)
    vendors = {g.vendor for g in specs.gpus}
    non_apple_gpu = bool(vendors - {"apple"})
    plan: list[RuntimeVariant] = []

    if os_key == "darwin":
        if arch == "arm64":
            plan.append(METAL)  # Apple Silicon: Metal GPU acceleration is built in
    elif os_key == "windows":
        if arch == "x64":
            if "nvidia" in vendors:
                plan.extend(_cuda_variants(specs, arch))
            if non_apple_gpu:
                plan.append(VULKAN)  # works with NVIDIA, AMD and Intel drivers
        elif arch == "arm64" and "nvidia" in vendors:
            # Windows on ARM with an NVIDIA GPU: a CUDA 13 arm64 build is published
            # (llama-...-bin-win-cuda-13.x-arm64.zip + its cudart); otherwise the CPU build.
            plan.extend(_cuda_variants(specs, arch))
    elif os_key == "linux" and arch is not None:
        glibc = _glibc_version()

        def runs_here(variant: RuntimeVariant) -> bool:
            return glibc is None or glibc >= _min_glibc(variant, arch)

        if "nvidia" in vendors:
            plan.extend(v for v in _cuda_variants(specs, arch) if runs_here(v))
        if non_apple_gpu and _vulkan_available(specs) and runs_here(VULKAN):
            plan.append(VULKAN)

    plan.append(CPU)  # (if even this can't run here, `platform_problem` says so up front)
    return plan


def engine_can_use_gpu(specs: SystemSpecs) -> bool:
    """Could the built-in llama.cpp engine use a graphics card on this computer at all?

    False when :func:`usable_plan` has only the CPU build - for example an
    AMD card on Linux without the Vulkan loader, an Intel Mac, or a CUDA
    build that already failed here for good (see :func:`mark_unusable`). The fit
    engine then plans with the CPU, so it never promises GPU speed that the
    engine can't deliver.
    """
    try:
        plan = usable_plan(specs)  # builds known not to run here don't count
        if not downloads_allowed():
            # A built game can only use the builds it ships with (e.g. Vulkan, not CUDA).
            return any(v.gpu and find_installed(v, specs) is not None for v in plan)
        return any(v.gpu for v in plan)
    except Exception:
        return True  # unsure: don't hide the GPU


def license_text(variant: RuntimeVariant) -> str:
    """Plain-English license line for a build, e.g. for the confirmation screen.

    llama.cpp itself is MIT licensed, but the NVIDIA CUDA builds also bundle
    NVIDIA's CUDA runtime libraries, which come under NVIDIA's own terms.
    """
    if variant.needs_cudart:
        return (
            f"llama.cpp: {LLAMA_CPP_LICENSE} license; the bundled NVIDIA CUDA runtime: NVIDIA's own license "
            f"terms ({CUDA_EULA_URL}, see NOTICE.md)"
        )
    return f"{LLAMA_CPP_LICENSE} license"


# ---------------------------------------------------------------------------
# Matching release assets
# ---------------------------------------------------------------------------

_TAG_RE = r"(?:[a-z0-9.]+-)?"
_OS_RE = {
    "windows": r"(?:win|windows)",
    "linux": r"(?:ubuntu|linux)(?:-?\d+(?:\.\d+)*)?",
    "darwin": r"(?:macos|mac|osx|darwin)(?:-?\d+(?:\.\d+)*)?",
}
_ARCH_RE = {"x64": r"(?:x64|x86_64|x86-64|amd64)", "arm64": r"(?:arm64|aarch64)"}
# Extra "-suffix" tokens we tolerate at the end of a name - unless they name a
# different accelerator (so "...-arm64-snapdragon" never counts as plain CPU).
_ACCEL_WORDS = r"cuda|cu\d|vulkan|rocm|hip|sycl|openvino|opencl|adreno|snapdragon|kleidiai|musa|cann|kompute|hexagon|npu"
_SUFFIX_RE = rf"(?:-(?!(?:{_ACCEL_WORDS}))[a-z0-9._]+)*"
_EXT_RE = r"\.(?:zip|tar\.gz|tgz)"
_VERSION_RE = re.compile(r"(?:cuda-?(?:cu)?|cu|rocm-?|hip-?)(\d+(?:\.\d+)*)", re.IGNORECASE)


def _compile_template(template: str, os_key: str, arch: str) -> re.Pattern[str]:
    body = (
        template.replace("{tag}", _TAG_RE)
        .replace("{os}", _OS_RE[os_key])
        .replace("{arch}", _ARCH_RE[arch])
    )
    return re.compile(rf"^{body}{_SUFFIX_RE}{_EXT_RE}$", re.IGNORECASE)


def _version_of(name: str) -> tuple[int, ...]:
    m = _VERSION_RE.search(name)
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def _usable_assets(assets: list[dict]) -> list[dict]:
    usable = []
    for a in assets or []:
        if not isinstance(a, dict) or not isinstance(a.get("name"), str):
            continue
        if a.get("state") not in (None, "uploaded"):
            continue  # still uploading
        usable.append(a)
    return usable


def _best_match(assets: list[dict], templates: tuple[str, ...], os_key: str, arch: str) -> list[dict]:
    """All assets matching the first template that matches anything, best first."""
    for template in templates:
        rx = _compile_template(template, os_key, arch)
        hits = [a for a in assets if rx.match(a["name"])]
        if hits:
            # Highest CUDA/ROCm version first, then the plainest (shortest) name.
            return sorted(hits, key=lambda a: (_version_of(a["name"]), -len(a["name"])), reverse=True)
    return []


def select_assets(assets: list[dict], variant: RuntimeVariant, os_name: str, arch: str) -> list[dict]:
    """Pick the release file(s) for `variant` on this OS/architecture.

    Pure function. Returns ``[main_archive]``, or ``[main_archive,
    cuda_runtime_archive]`` for CUDA variants, or ``[]`` if this release has
    no suitable build (for example if the CUDA runtime archive is missing).
    """
    os_key, arch_key = _os_key(os_name), _arch_key(arch)
    if os_key is None or arch_key is None:
        return []
    if variant.platforms and os_key not in variant.platforms:
        return []
    usable = _usable_assets(assets)
    mains = _best_match(usable, variant.asset_patterns, os_key, arch_key)
    if not mains:
        return []
    main = mains[0]
    if not variant.needs_cudart:
        return [main]
    runtimes = _best_match(usable, variant.cudart_patterns, os_key, arch_key)
    if not runtimes:
        return []
    # Prefer the CUDA runtime with exactly the same version as the main build.
    wanted = _version_of(main["name"])
    same = [r for r in runtimes if _version_of(r["name"]) == wanted]
    return [main, (same or runtimes)[0]]


def _tag_number(tag: str) -> int:
    m = re.search(r"(\d+)", tag or "")
    return int(m.group(1)) if m else -1


def releases_newest_first(releases: list[dict]) -> list[dict]:
    """The published (non-draft) releases, newest first (by date, then tag number)."""
    candidates = [r for r in releases or [] if isinstance(r, dict) and not r.get("draft")]
    candidates.sort(
        key=lambda r: (str(r.get("published_at") or r.get("created_at") or ""), _tag_number(str(r.get("tag_name", "")))),
        reverse=True,
    )
    return candidates


def pick_release(
    releases: list[dict], variant: RuntimeVariant, os_name: str, arch: str
) -> Optional[tuple[dict, list[dict]]]:
    """The newest release that has a build for `variant`, with its assets."""
    for release in releases_newest_first(releases):
        chosen = select_assets(release.get("assets") or [], variant, os_name, arch)
        if chosen:
            return release, chosen
    return None


# ---------------------------------------------------------------------------
# A tiny HTTP layer (standard library only)
# ---------------------------------------------------------------------------


class HttpResponse:
    """A minimal response: ``status``, lower-cased ``headers``, streaming ``read()``."""

    def __init__(self, status: int, headers: Mapping[str, str], stream: Any) -> None:
        self.status = int(status)
        self.headers = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
        self._stream = stream if stream is not None else io.BytesIO(b"")

    def read(self, n: int = -1) -> bytes:
        return self._stream.read() if n is None or n < 0 else self._stream.read(n)

    def json(self) -> Any:
        raw = self.read()
        return json.loads(raw.decode("utf-8")) if raw else None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.close()

    def __enter__(self) -> "HttpResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Headers that carry credentials: never passed on to a different address.
_CREDENTIAL_HEADERS = ("Authorization", "Proxy-Authorization", "Cookie")


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects without leaking credentials.

    urllib's own handler copies every header - ``Authorization`` (a
    ``GITHUB_TOKEN``) included - to whatever address a redirect names, and
    happily goes from https to plain http. Here a redirect to another host
    (or port, or scheme) loses its credential headers, and an https -> http
    downgrade is refused outright. Same-host redirects (GitHub's answer for a
    renamed repository) keep working.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - urllib's hook
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_parts = urllib.parse.urlsplit(req.full_url)
        new_parts = urllib.parse.urlsplit(new.full_url)
        if old_parts.scheme == "https" and new_parts.scheme != "https":
            raise urllib.error.HTTPError(
                new.full_url, code, "refused a redirect from https to an insecure address", headers, fp
            )
        same_place = (old_parts.scheme, old_parts.hostname, old_parts.port) == (
            new_parts.scheme, new_parts.hostname, new_parts.port)
        if not same_place:
            for name in _CREDENTIAL_HEADERS:
                new.remove_header(name)
        return new


class UrllibHttp:
    """`request(method, url, headers=, body=, timeout=) -> HttpResponse` via urllib.

    HTTP error statuses (404, 503...) are returned as responses rather than
    raised, so callers can look at them. Network failures raise ``OSError``.
    Pass ``use_proxy=False`` for localhost traffic, so a system proxy setting
    can't intercept requests to our own ``llama-server``.

    HTTPS certificates are checked against a trust store that works on every
    computer (see :mod:`gettowork.tls`), and redirects never pass credentials
    on to another address (see :class:`_SafeRedirects`).
    """

    def __init__(self, *, use_proxy: bool = True) -> None:
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=https_context()), _SafeRedirects()]
        if not use_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        req = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
        try:
            resp = self._opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as err:
            hdrs = dict(err.headers.items()) if err.headers is not None else {}
            return HttpResponse(err.code, hdrs, err.fp if err.fp is not None else None)
        return HttpResponse(resp.status, dict(resp.headers.items()), resp)


def _github_headers(*, use_token: bool = True) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip() if use_token else ""
    if token:
        # Optional: a token raises GitHub's hourly rate limit. It is only ever
        # sent to api.github.com, and never printed or logged.
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(http: Any, url: str, ui: Optional[UI]) -> tuple[int, bytes]:
    """GET a GitHub API address; returns ``(status, body)``.

    If GitHub rejects an old or mistyped ``GITHUB_TOKEN`` (HTTP 401 - it does
    that even for public data), we ask once more without it. Network trouble
    and GitHub's rate limit raise RuntimeInstallError with a friendly message.
    """

    def ask(use_token: bool) -> tuple[int, bytes, Optional[str]]:
        try:
            resp = http.request("GET", url, headers=_github_headers(use_token=use_token), timeout=20.0)
            with resp:
                return resp.status, resp.read(), resp.headers.get("x-ratelimit-remaining")
        except NETWORK_ERRORS as exc:
            if is_certificate_error(exc):
                raise RuntimeInstallError(
                    f"I couldn't fetch the llama.cpp engine from GitHub. {CERTIFICATE_HELP}"
                ) from exc
            raise RuntimeInstallError(
                "I couldn't reach GitHub to fetch the llama.cpp engine (are you offline, or is a "
                "firewall blocking github.com?). Please try again later - or install Ollama "
                f"({OLLAMA_DOWNLOAD_URL}) and the game will happily use that instead."
            ) from exc

    headers_sent = _github_headers()
    status, raw, remaining = ask(True)
    if status == 401 and "Authorization" in headers_sent:
        # The releases are public: a stale token shouldn't stop the install.
        message = (
            "GitHub rejected the GITHUB_TOKEN set on this computer (it may have expired), so I ignored it. "
            "You can remove or renew it any time."
        )
        if ui is not None:
            ui.warn(message)
        status, raw, remaining = ask(False)
    if status == 429 or (status == 403 and (remaining == "0" or b"rate limit" in (raw or b"").lower())):
        raise RuntimeInstallError(
            "GitHub says we've asked for the engine too many times this hour (its free rate "
            "limit). Please try again a bit later - or set a GITHUB_TOKEN environment variable "
            f"to raise the limit, or use Ollama ({OLLAMA_DOWNLOAD_URL})."
        )
    return status, raw or b""


def _json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeInstallError("GitHub sent back something I couldn't understand. Please try again later.") from exc


def fetch_releases(*, http: Any = None, limit: int = 8, ui: Optional[UI] = None) -> list[dict]:
    """List the newest llama.cpp releases from the GitHub API (newest first).

    Uses ``GET /repos/ggml-org/llama.cpp/releases?per_page=N`` rather than
    ``/releases/latest``, because llama.cpp publishes builds as prereleases.
    If GitHub rejects an old or mistyped ``GITHUB_TOKEN`` (HTTP 401 - it does
    that even for public data), we ask once more without it.
    Raises RuntimeInstallError with a friendly message on any failure.
    """
    http = http or UrllibHttp()
    url = f"{GITHUB_RELEASES_API}?per_page={max(1, min(int(limit), 100))}"
    status, raw = _github_get(http, url, ui)
    if status != 200:
        raise RuntimeInstallError(
            f"GitHub answered with an unexpected status (HTTP {status}) when I asked for the "
            "llama.cpp releases. Please try again later, or use Ollama instead."
        )
    data = _json_body(raw)
    if not isinstance(data, list):
        raise RuntimeInstallError("GitHub's release list looked unusual. Please try again later.")
    return [r for r in data if isinstance(r, dict)]


def fetch_release(tag: str, *, http: Any = None, ui: Optional[UI] = None) -> dict:
    """One llama.cpp release by its tag (e.g. ``"b7000"``), via ``GET /releases/tags/<tag>``.

    Used by ``packaging/fetch_engine.py`` to bundle an exact, pinned release.
    Raises RuntimeInstallError with a friendly message on any failure.
    """
    http = http or UrllibHttp()
    url = f"{GITHUB_RELEASES_API}/tags/{urllib.parse.quote(str(tag), safe='')}"
    status, raw = _github_get(http, url, ui)
    if status == 404:
        raise RuntimeInstallError(f"There's no llama.cpp release called {tag!r} on GitHub.")
    if status != 200:
        raise RuntimeInstallError(
            f"GitHub answered with an unexpected status (HTTP {status}) when I asked for the "
            f"llama.cpp release {tag}. Please try again later."
        )
    data = _json_body(raw)
    if not isinstance(data, dict):
        raise RuntimeInstallError("GitHub's answer about that release looked unusual. Please try again later.")
    return data


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def _parse_digest(value: Any) -> Optional[str]:
    """GitHub's asset ``digest`` field looks like ``"sha256:<64 hex chars>"``."""
    if isinstance(value, str) and value.lower().startswith("sha256:"):
        hexpart = value.split(":", 1)[1].strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", hexpart):
            return hexpart
    return None


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name.replace("\\", "/")))
    return cleaned.lstrip(".") or "download"


def _download_asset(http: Any, asset: dict, folder: Path, ui: UI, description: str) -> Path:
    """Stream one release file to ``folder``: temp file, verify, then atomic rename."""
    name = _safe_filename(asset["name"])
    url = asset.get("browser_download_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RuntimeInstallError(f"The download link for {name} looked wrong, so I didn't use it.")
    expected = asset.get("size") if isinstance(asset.get("size"), int) and asset.get("size") > 0 else None
    digest = _parse_digest(asset.get("digest"))

    folder.mkdir(parents=True, exist_ok=True)
    final = folder / name
    part = folder / (name + ".part")
    hasher = hashlib.sha256()
    received = 0
    try:
        # No Authorization header here: release files are public, and GitHub
        # redirects them to a storage host that must not receive our token.
        headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream"}
        try:
            resp = http.request("GET", url, headers=headers, timeout=60.0)
        except NETWORK_ERRORS as exc:
            if is_certificate_error(exc):
                raise RuntimeInstallError(f"The download of {name} couldn't start. {CERTIFICATE_HELP}") from exc
            raise RuntimeInstallError(
                f"The download of {name} couldn't start (network problem). Please check your "
                "internet connection and try again."
            ) from exc
        with resp:
            if resp.status != 200:
                raise RuntimeInstallError(
                    f"GitHub answered HTTP {resp.status} while downloading {name}. Please try again later."
                )
            total = expected
            if total is None:
                with contextlib.suppress(TypeError, ValueError):
                    total = int(resp.headers.get("content-length") or 0) or None
            try:
                with open(part, "wb") as fh, ui.download_progress(description, total) as advance:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        hasher.update(chunk)
                        received += len(chunk)
                        advance(len(chunk))
                        if expected is not None and received > expected:
                            break  # more data than promised: stop, the size check below will fail
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise RuntimeInstallError("Your disk filled up while downloading the engine. Free some space and try again.") from exc
                raise RuntimeInstallError(
                    f"The download of {name} was interrupted (network problem). Please try again."
                ) from exc
            except http_client.HTTPException as exc:
                raise RuntimeInstallError(f"The download of {name} was cut short. Please try again.") from exc
        if expected is not None and received != expected:
            raise RuntimeInstallError(
                f"The download of {name} was incomplete ({received:,} of {expected:,} bytes). Please try again."
            )
        if digest and hasher.hexdigest() != digest:
            raise RuntimeInstallError(
                f"{name} didn't match the checksum GitHub published for it, so I threw it away "
                "to keep you safe. Please try again."
            )
        # Real-time antivirus scanners open a freshly closed archive to scan it,
        # which briefly locks it on Windows: wait that out rather than throwing
        # away a complete, verified download.
        _rename_with_retry(part, final)
        return final
    except BaseException:  # includes Ctrl+C: never leave half-downloaded files around
        with contextlib.suppress(OSError):
            part.unlink()
        raise


def download_asset(http: Any, asset: dict, folder: Path, ui: Any, description: str = "llama.cpp engine") -> Path:
    """Download one release file (a GitHub asset dict) into `folder`, checked, and return its path.

    The size and SHA-256 fingerprint GitHub publishes are verified before the
    file appears under its real name; anything that fails is deleted. `ui`
    only needs a ``download_progress(description, total_bytes)`` context
    manager yielding ``advance(n)`` (a :class:`~gettowork.ui.UI` has one).
    Raises RuntimeInstallError with a friendly message.
    """
    return _download_asset(http, asset, Path(folder), ui, description)


# ---------------------------------------------------------------------------
# Safe unpacking (zip-slip / tar-slip protection)
# ---------------------------------------------------------------------------


def _member_parts(name: str) -> list[str]:
    """Validate an archive member name; return its path components.

    Rejects absolute paths (``/x``, ``C:\\x``, ``\\\\server\\x``), any ``..``
    component and drive/stream colons. An empty list means "the archive root".
    """
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise RuntimeInstallError(f"The engine archive contains an unsafe absolute path ({name!r}), so I refused to unpack it.")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    for p in parts:
        if p == ".." or ":" in p or "\x00" in p:
            raise RuntimeInstallError(f"The engine archive contains an unsafe path ({name!r}), so I refused to unpack it.")
    return parts


def _link_target_parts(member_parts: list[str], target: str, *, relative_to_root: bool = False) -> list[str]:
    """Where a link points, as components inside the destination (or raise)."""
    norm = target.replace("\\", "/")
    if not norm or norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise RuntimeInstallError(f"The engine archive contains a link to an absolute path ({target!r}), so I refused to unpack it.")
    stack = [] if relative_to_root else list(member_parts[:-1])
    for p in norm.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if not stack:
                raise RuntimeInstallError(f"The engine archive contains a link pointing outside its folder ({target!r}), so I refused to unpack it.")
            stack.pop()
        else:
            stack.append(p)
    return stack


def _inside(dest: Path, parts: list[str]) -> Path:
    """Join `parts` onto `dest`, double-checking the real path stays inside."""
    path = dest.joinpath(*parts)
    root = os.path.realpath(dest)
    real = os.path.realpath(path)
    if os.path.commonpath([root, real]) != root:
        raise RuntimeInstallError("The engine archive tried to write outside its folder, so I refused to unpack it.")
    return path


def _extract_zip(archive: Path, dest: Path) -> None:
    links: list[tuple[list[str], list[str]]] = []
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_MEMBERS or sum(i.file_size for i in infos) > _MAX_UNPACKED_BYTES:
            raise RuntimeInstallError("The engine archive is suspiciously large, so I refused to unpack it.")
        for info in infos:
            parts = _member_parts(info.filename)
            if not parts:
                continue
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                target = zf.read(info).decode("utf-8", "replace")
                links.append((parts, _link_target_parts(parts, target)))
                continue
            path = _inside(dest, parts)
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(path, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            if os.name != "nt" and unix_mode & 0o111:
                os.chmod(path, (unix_mode & 0o755) | 0o600)
    # zipfile never creates links itself; recreate in-tree ones as copies.
    for parts, target_parts in links:
        _copy_link_target(dest, parts, target_parts)


def _copy_link_target(dest: Path, link_parts: list[str], target_parts: list[str]) -> None:
    src = _inside(dest, target_parts)
    dst = _inside(dest, link_parts)
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _check_tar_members(members: list[tarfile.TarInfo]) -> None:
    if len(members) > _MAX_MEMBERS or sum(max(m.size, 0) for m in members) > _MAX_UNPACKED_BYTES:
        raise RuntimeInstallError("The engine archive is suspiciously large, so I refused to unpack it.")
    for m in members:
        parts = _member_parts(m.name)
        if m.issym():
            _link_target_parts(parts, m.linkname)
        elif m.islnk():
            _link_target_parts(parts, m.linkname, relative_to_root=True)
        elif not (m.isfile() or m.isdir()):
            raise RuntimeInstallError(
                f"The engine archive contains a special file ({m.name!r}), so I refused to unpack it."
            )


def _extract_tar_manually(tf: tarfile.TarFile, members: list[tarfile.TarInfo], dest: Path) -> None:
    """Fallback for Pythons without tarfile's "data" filter (checks already done)."""
    deferred: list[tuple[tarfile.TarInfo, list[str]]] = []
    for m in members:
        parts = _member_parts(m.name)
        if not parts:
            continue
        if m.issym() or m.islnk():
            deferred.append((m, parts))
            continue
        path = _inside(dest, parts)
        if m.isdir():
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(m)
        if src is None:
            continue
        with src, open(path, "wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
        if os.name != "nt":
            # Keep the executable bit, drop setuid/setgid/sticky and group/other write.
            os.chmod(path, (m.mode & 0o755) | 0o600)
    for m, parts in deferred:
        if m.issym():
            target_parts = _link_target_parts(parts, m.linkname)
            link = _inside(dest, parts[:-1]) / parts[-1]
            link.parent.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                with contextlib.suppress(FileNotFoundError):
                    link.unlink()
                os.symlink(m.linkname.replace("\\", "/"), link)
            else:
                _copy_link_target(dest, parts, target_parts)
        else:  # hard link: make a copy of the (already extracted) target
            _copy_link_target(dest, parts, _link_target_parts(parts, m.linkname, relative_to_root=True))


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        _check_tar_members(members)  # our own checks run on every Python version
        if _HAS_TAR_DATA_FILTER:
            try:
                tf.extractall(dest, members=members, filter="data")
            except tarfile.TarError as exc:
                raise RuntimeInstallError(f"The engine archive failed a safety check ({exc}), so I refused to unpack it.") from exc
        else:
            _extract_tar_manually(tf, members, dest)


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a ``.zip`` or ``.tar.gz`` into `dest`, refusing anything unsafe.

    Protects against "zip-slip"/"tar-slip": members with absolute paths, ``..``
    components, links pointing outside `dest`, and device files are rejected
    before anything is written.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lower = archive.name.lower()
    try:
        if lower.endswith(".zip"):
            _extract_zip(archive, dest)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            _extract_tar(archive, dest)
        else:
            raise RuntimeInstallError(f"I don't know how to unpack {archive.name}.")
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        raise RuntimeInstallError(f"{archive.name} seems to be damaged ({exc}). Please try again.") from exc
    except OSError as exc:  # disk full, a locked file...
        raise _friendly_os_error(exc, f"unpacking {archive.name}") from exc


def _single_top_dir(folder: Path) -> Path:
    """Tarballs wrap everything in ``llama-<tag>/``: return that inner folder."""
    entries = list(folder.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        return entries[0]
    return folder


def _merge_tree(src: Path, dst: Path) -> None:
    """Move everything from `src` into `dst` (both on the same disk)."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(src.iterdir()):
        target = dst / child.name
        if child.is_dir() and not child.is_symlink() and target.is_dir() and not target.is_symlink():
            _merge_tree(child, target)
            continue
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        os.replace(child, target)


def _make_executable(folder: Path) -> None:
    """``chmod +x`` the programs and libraries (POSIX only; Windows ignores it)."""
    if os.name == "nt":
        return
    for dirpath, _dirs, files in os.walk(folder):
        for fname in files:
            path = Path(dirpath) / fname
            if path.is_symlink():
                continue
            if fname.startswith("llama") or "." not in fname or ".so" in fname or fname.endswith(".dylib"):
                mode = path.stat().st_mode
                os.chmod(path, mode | stat.S_IRUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def find_server_executable(folder: Path) -> Optional[Path]:
    """Find ``llama-server`` / ``llama-server.exe`` anywhere under `folder` (shallowest wins)."""
    wanted = {n.lower() for n in SERVER_NAMES}
    hits: list[Path] = []
    for dirpath, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower() in wanted:
                hits.append(Path(dirpath) / fname)
    if not hits:
        return None
    return min(hits, key=lambda p: (len(p.relative_to(folder).parts), str(p)))


def unpack_archive(archive: Path, payload: Path, scratch: Path) -> None:
    """Safely unpack one engine archive into `scratch`, then merge it into `payload`.

    Tarballs wrap everything in ``llama-<tag>/``; that wrapper is dropped, so
    every archive of a build (e.g. CUDA + its runtime) ends up side by side
    in one folder, next to ``llama-server``.
    """
    safe_extract(archive, scratch)
    _merge_tree(_single_top_dir(Path(scratch)), Path(payload))


def finish_unpacked(payload: Path) -> Path:
    """Make the unpacked programs executable and return ``llama-server`` (or raise)."""
    _make_executable(payload)
    exe = find_server_executable(payload)
    if exe is None:
        raise RuntimeInstallError(
            "The downloaded engine didn't contain llama-server, which is unusual. "
            "Please try again later (the llama.cpp team may be mid-release)."
        )
    return exe


def install_marker(release: dict, assets: list[dict], variant: RuntimeVariant, rel_exe: str, *,
                   bundled: bool = False, license_files: Optional[list[str]] = None) -> dict:
    """The ``install.json`` note written next to every engine build.

    It records where the build came from (release tag, archives, licenses)
    and where ``llama-server`` is inside the folder. ``bundled=True`` marks a
    build that ships inside the game (read-only, never tidied away).
    """
    tag = str(release.get("tag_name") or "unknown")
    marker: dict[str, Any] = {
        "tag": tag,
        "variant": variant.name,
        "label": variant.display,
        "assets": [a["name"] for a in assets],
        "exe": rel_exe,
        "source": str(release.get("html_url") or f"{LLAMA_CPP_URL}/releases/tag/{tag}"),
        "license": LLAMA_CPP_LICENSE,  # llama.cpp itself
        "licenses": {a["name"]: _asset_license(a) for a in assets},  # per downloaded archive
        "installed_at": int(time.time()),
    }
    if license_files:
        marker["license_files"] = list(license_files)
    if bundled:
        marker["bundled"] = True
    return marker


# ---------------------------------------------------------------------------
# Installed runtimes
# ---------------------------------------------------------------------------


def _llama_root(runtime_root: Optional[Path]) -> Path:
    return Path(runtime_root if runtime_root is not None else config.runtime_dir()) / "llama.cpp"


def _read_marker(folder: Path) -> Optional[dict]:
    try:
        data = json.loads((folder / INSTALL_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def downloads_allowed() -> bool:
    """May the game download llama.cpp builds? False in a built game (see :mod:`gettowork.distribution`)."""
    try:
        return bool(distribution.load().engine_downloads)
    except Exception:
        return True


def llama_cpp_python_possible() -> bool:
    """Could the player switch to the ``llama-cpp-python`` package? Not in a built game.

    A built game (Steam, the double-click builds) has no ``pip``, PyInstaller
    builds only see the packages frozen into them, and the build leaves
    ``llama_cpp`` out - so advice to ``pip install llama-cpp-python`` could
    never work there.
    """
    return not getattr(sys, "frozen", False) and downloads_allowed()


def other_engines_hint(verb: str = "should still work") -> str:
    """What else can run a model when llama.cpp's own engine can't, in one sentence.

    Ollama always; ``pip install llama-cpp-python`` only where that is possible
    (see :func:`llama_cpp_python_possible`).
    """
    if llama_cpp_python_possible():
        return (f"Ollama ({OLLAMA_DOWNLOAD_URL}) or `pip install llama-cpp-python` (which builds the engine on your "
                f"computer) {verb}.")
    return (f"The free Ollama app ({OLLAMA_DOWNLOAD_URL}) {verb}: install it and start it, and the game will use it "
            "the next time you pick a model.")


def _engine_dirs() -> tuple[Path, ...]:
    """The folders holding the builds that ship inside the game (none in a developer copy)."""
    try:
        return tuple(Path(d) for d in distribution.load().engine_dirs)
    except Exception:
        return ()


def _marker_exe(folder: Path, marker: dict) -> Optional[Path]:
    """Where a marker says ``llama-server`` is inside `folder` (None if it names no file that exists)."""
    rel = marker.get("exe")
    if not isinstance(rel, str) or not rel:
        return None
    exe = folder / Path(*PurePosixPath(rel).parts)
    try:
        return exe if exe.is_file() else None
    except OSError:
        return None


def _bundled_builds() -> list[tuple[Path, dict, Path]]:
    """Every engine build shipped inside the game, as ``(folder, marker, exe)``.

    Each engine folder holds one ``<tag>-<variant>/`` sub-folder per build
    (an engine folder that holds an ``install.json`` itself also counts, for
    testing). Builds noted as unusable here are included - see
    :func:`installed_runtimes` for the ones that can run.
    """
    found: list[tuple[Path, dict, Path]] = []
    seen: set[str] = set()
    for root in _engine_dirs():
        try:
            if not root.is_dir():
                continue
            if (root / INSTALL_MARKER).is_file():
                folders = [root]
            else:
                folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            continue
        for folder in folders:
            marker = _read_marker(folder)
            exe = _marker_exe(folder, marker) if marker else None
            if marker is None or exe is None:
                continue  # not a complete build
            key = os.path.normcase(os.path.realpath(exe))
            if key not in seen:
                seen.add(key)
                found.append((folder, marker, exe))
    return found


def engine_architectures() -> Optional[frozenset[str]]:
    """The model architectures a built game's own engine can load; None = not limited (or not known).

    A built game can't update its engine, so a model of an architecture its
    llama.cpp release doesn't know yet ("unknown model architecture") can
    never run there. ``packaging/fetch_engine.py`` records the names that
    release knows in each bundled build's ``install.json``
    (``architectures``: "qwen3", "gpt-oss", ...). None in a copy that
    downloads engines (it gets a newer one for a new architecture) and for a
    build that recorded no list. Never raises.
    """
    try:
        if downloads_allowed():
            return None
        names: set[str] = set()
        for _folder, marker, _exe in _bundled_builds():
            listed = marker.get("architectures")
            if isinstance(listed, list):
                names.update(a.strip().lower() for a in listed if isinstance(a, str) and a.strip())
        return frozenset(names) or None
    except Exception:
        return None


def _inside_engine_dirs(path: Path) -> bool:
    """Is `path` inside one of the game's own (read-only) engine folders?"""
    try:
        real = os.path.normcase(os.path.realpath(path))
        for root in _engine_dirs():
            base = os.path.normcase(os.path.realpath(root))
            if real == base or real.startswith(base.rstrip(os.sep) + os.sep):
                return True
    except (OSError, ValueError):
        pass
    return False


def is_bundled(exe: Path) -> bool:
    """Is `exe` part of the engine that ships inside the game? (Those builds are read-only.)"""
    try:
        info = install_info(Path(exe))
        if info is not None and info.get("bundled") is True:
            return True
        return _inside_engine_dirs(Path(exe))
    except Exception:
        return False


def _bundled_key(tag: Any, variant: Any) -> str:
    """How a built-in build is named in ``bundled-unusable.json``: ``"<tag>-<variant>"``."""
    return f"{tag or ''}-{variant or ''}"


def _bundled_notes_path(runtime_root: Optional[Path]) -> Path:
    return Path(runtime_root if runtime_root is not None else config.runtime_dir()) / BUNDLED_UNUSABLE_FILE


def _file_fingerprint(exe: Path) -> Optional[dict]:
    """Size and modification time of an engine program (None if it can't be read)."""
    try:
        info = Path(exe).stat()
    except OSError:
        return None
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _bundled_note_applies(note: dict, exe: Path) -> bool:
    """Does a "can't run here" note still describe this built-in build's program?

    The note records the program's size and modification time: a repaired
    file (Steam's "Verify integrity of game files", a re-extracted test build)
    or another copy of the game with the same engine release gets a fresh
    check instead of inheriting the verdict. (Notes from before this was
    recorded apply by release and build type alone.)
    """
    recorded = note.get("file")
    if not isinstance(recorded, dict):
        return True
    current = _file_fingerprint(exe)
    return current is not None and all(recorded.get(key) == value for key, value in current.items())


def _bundled_notes(runtime_root: Optional[Path]) -> dict[str, dict]:
    """{"<tag>-<variant>": note} for built-in builds known not to run here ({} if none / unreadable)."""
    try:
        data = json.loads(_bundled_notes_path(runtime_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    builds = data.get("builds") if isinstance(data, dict) else None
    if not isinstance(builds, dict):
        return {}
    return {k: v for k, v in builds.items() if isinstance(k, str) and isinstance(v, dict)}


def installed_runtimes(runtime_root: Optional[Path] = None, *, bundled: bool = True) -> list[tuple[Path, str, str]]:
    """Engines on this computer, newest first, as ``(exe, tag, variant_name)``.

    Lists the builds installed earlier (in ``runtime_root``) and - unless
    ``bundled=False`` - the builds that ship inside the game. Only complete
    builds count: each has an ``install.json`` marker and its ``llama-server``
    executable still exists. Builds marked unusable on this computer (see
    :func:`mark_unusable`) are left out. For the same release and build, the
    game's own copy comes first.
    """
    root = _llama_root(runtime_root)
    found: list[tuple[Path, str, str, bool]] = []
    try:
        folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        folders = []
    for folder in folders:
        marker = _read_marker(folder)
        if not marker or marker.get("unusable"):
            continue  # unfinished, or known not to run on this computer
        exe = _marker_exe(folder, marker)
        if exe is not None:
            found.append((exe, str(marker.get("tag", "")), str(marker.get("variant", "")), False))
    if bundled:
        notes = _bundled_notes(runtime_root)
        known = {os.path.normcase(os.path.realpath(exe)) for exe, *_rest in found}
        for _folder, marker, exe in _bundled_builds():
            note = notes.get(_bundled_key(marker.get("tag"), marker.get("variant")))
            if marker.get("unusable") or (note is not None and _bundled_note_applies(note, exe)):
                continue  # can't run on this computer
            if os.path.normcase(os.path.realpath(exe)) in known:
                continue  # (an engine folder that is also the install folder: listed once)
            found.append((exe, str(marker.get("tag", "")), str(marker.get("variant", "")), True))
    found.sort(key=lambda t: (_tag_number(t[1]), t[2], t[3]), reverse=True)
    return [(exe, tag, name) for exe, tag, name, _bundled in found]


def _serves_as(name: str, assets: Any, variant: RuntimeVariant, os_name: str, arch: str) -> bool:
    """Can an installed build (its variant name + archive names) do the job of `variant`?

    Yes for the same build type - and also when the build was made from the
    very archive(s) `variant` would download here: on an Apple Silicon Mac the
    Metal build *is* the CPU build (llama-server runs on the processor with
    ``--device none``).
    """
    if name == variant.name:
        return True
    names = [a for a in assets if isinstance(a, str)] if isinstance(assets, list) else []
    if not names:
        return False
    chosen = select_assets([{"name": n} for n in names], variant, os_name, arch)
    return bool(chosen) and all(a["name"] in names for a in chosen)


def find_installed(variant: RuntimeVariant, specs: SystemSpecs, *, runtimes: Optional[list] = None,
                   runtime_root: Optional[Path] = None) -> Optional[tuple[Path, str]]:
    """A build already on this computer (installed or built into the game) that can serve as
    `variant`, as ``(exe, tag)`` - the newest one of exactly that build first. None if there isn't one."""
    runtimes = installed_runtimes(runtime_root) if runtimes is None else runtimes
    for exe, tag, name in runtimes:
        if name == variant.name:
            return exe, tag
    for exe, tag, name in runtimes:
        info = install_info(exe) or {}
        if _serves_as(name, info.get("assets"), variant, specs.os_name, specs.arch):
            return exe, tag
    return None


def _made_for(exe: Path, specs: SystemSpecs) -> bool:
    """Was this build made for this computer's OS and processor? (True when its archives aren't recorded.)"""
    names = (install_info(exe) or {}).get("assets")
    if not isinstance(names, list) or not any(isinstance(n, str) for n in names):
        return True
    return any(_serves_as("", names, v, specs.os_name, specs.arch) for v in KNOWN_VARIANTS.values())


def own_builds_first(runtimes: list) -> list[list]:
    """The groups of builds to choose an engine from, in order (empty groups left out).

    In a built game (no engine downloads), this copy's own builds come before
    every other install of any type or release - another copy of the game,
    or the downloads of a developer copy sharing the settings folder (usually
    of a newer llama.cpp, or a CUDA build the game doesn't ship) - so the
    engine that ships is the one that runs, and is tested. Other installs
    only count when none of this copy's own builds will do. A copy that
    downloads engines has one group: everything, newest first.
    """
    runtimes = list(runtimes)
    if downloads_allowed():
        return [runtimes] if runtimes else []
    own = [r for r in runtimes if _inside_engine_dirs(r[0])]
    others = [r for r in runtimes if not _inside_engine_dirs(r[0])]
    return [group for group in (own, others) if group]


def choose_installed(plan: list[RuntimeVariant], specs: SystemSpecs, *, runtimes: Optional[list] = None,
                     runtime_root: Optional[Path] = None) -> Optional[tuple[Path, str, RuntimeVariant]]:
    """The best build in `plan` (best first) that is already here, as ``(exe, tag, variant)``, or None.

    This is the choice a built game makes: if the plan's first builds (say
    NVIDIA CUDA) aren't built in, the best one that is (Vulkan) is used - and
    its own builds win over any other install (see :func:`own_builds_first`).
    """
    runtimes = installed_runtimes(runtime_root) if runtimes is None else runtimes
    for group in own_builds_first(runtimes):
        for variant in plan:
            hit = find_installed(variant, specs, runtimes=group)
            if hit is not None:
                return hit[0], hit[1], variant
    return None


def _build_from_path(path: Path) -> Optional[tuple[str, str]]:
    """``(tag, variant)`` from an engine path's ``<tag>-<variant>`` folder name, if it has one."""
    names = sorted(KNOWN_VARIANTS, key=len, reverse=True)  # "cuda-12" is tried before shorter names
    for folder in list(Path(path).parents)[:3]:
        for name in names:
            if folder.name.endswith("-" + name) and len(folder.name) > len(name) + 1:
                return folder.name[: -len(name) - 1], name
    return None


def relocate_engine(saved_exe: Path, *, runtime_root: Optional[Path] = None) -> Optional[Path]:
    """Find the engine a saved path pointed at, after the game moved. Never raises.

    Settings remember the full path of the engine that worked last time. A
    built game's engine lives inside the game, so that path changes when the
    game moves (another Steam library, the app dragged to a new folder...).
    The same build (``<tag>-<variant>`` folder) is looked up among
    :func:`installed_runtimes` - or else the newest build of the same type.

    A built game (no engine downloads) always prefers its *own* engine: when
    the saved engine still exists but belongs somewhere else - an older copy
    of the game still on disk, or a developer copy sharing the same settings
    folder - this copy's build of the same type is returned instead, so the
    engine that ships is the one that runs (see :func:`own_build_instead`).

    Returns the path unchanged if it still exists (and nothing of this game's
    own should replace it), or None if nothing fits.
    """
    try:
        saved = Path(saved_exe)
        if saved.is_file():
            return own_build_instead(saved, runtime_root=runtime_root) or saved
        wanted = _build_from_path(saved)
        if wanted is None:
            return None
        tag, variant = wanted
        runtimes = [r for group in own_builds_first(installed_runtimes(runtime_root)) for r in group]
        same = [exe for exe, t, v in runtimes if v == variant and t == tag]
        similar = [exe for exe, _t, v in runtimes if v == variant]
        if not downloads_allowed():  # a built game: its own build of that type, before any other copy's
            own = [exe for exe in similar if _inside_engine_dirs(exe)]
            same = [exe for exe in same if _inside_engine_dirs(exe)] or ([] if own else same)
            similar = own or similar
        return (same or similar or [None])[0]
    except Exception:
        return None


def own_build_instead(saved_exe: Path, *, runtime_root: Optional[Path] = None) -> Optional[Path]:
    """In a built game: this copy's own build to use instead of `saved_exe`, or None. Never raises.

    None when the game downloads engines (a developer copy), when `saved_exe`
    is already one of this game's own builds, when it is an engine the player
    set up themselves (no ``install.json``), or when the game has no build of
    that type. Otherwise the same release and build of this game's own, else
    its newest build of the same type.
    """
    try:
        saved = Path(saved_exe)
        if downloads_allowed() or _inside_engine_dirs(saved):
            return None
        info = install_info(saved)
        if not info:
            return None  # the player's own llama-server: never swapped behind their back
        variant, tag = str(info.get("variant") or ""), str(info.get("tag") or "")
        own = [(exe, t, v) for exe, t, v in installed_runtimes(runtime_root) if _inside_engine_dirs(exe)]
        same = [exe for exe, t, v in own if v == variant and t == tag]
        similar = [exe for exe, _t, v in own if v == variant]
        return (same or similar or [None])[0]
    except Exception:
        return None


def install_info(exe: Path) -> Optional[dict]:
    """The ``install.json`` marker for an installed ``llama-server`` (or None).

    Looks in the executable's folder and a couple of parents, so it also
    works if the archive kept its files in a sub-folder.
    """
    folder = Path(exe).parent
    for candidate in (folder, *list(folder.parents)[:2]):
        marker = _read_marker(candidate)
        if marker:
            return marker
    return None


# Start-up failures that will happen every time on this computer, whatever
# the model: remembered so automatic setup doesn't pick that build again.
# "gpu_arch": a CUDA build without code for this graphics card's generation.
PERMANENT_FAILURES = ("glibc", "cpu_unsupported", "cant_execute", "gpu_arch")


def mark_unusable(exe: Path, reason: str, *, runtime_root: Optional[Path] = None) -> bool:
    """Remember that the installed build containing `exe` can't run on this computer.

    Written into its ``install.json`` (so deleting the folder forgets it) -
    except for a build that ships inside the game, which is never changed:
    its note goes into ``runtime_dir()/bundled-unusable.json``, keyed by
    release and build type, so a game update with a newer engine gets a
    fresh chance. Returns True if the note was saved. Never raises.
    """
    try:
        if is_bundled(Path(exe)):
            return _mark_bundled_unusable(Path(exe), reason, runtime_root)
        folder = Path(exe).parent
        for candidate in (folder, *list(folder.parents)[:2]):
            marker = _read_marker(candidate)
            if marker:
                marker["unusable"] = {"reason": str(reason), "at": int(time.time())}
                (candidate / INSTALL_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
                return True
    except Exception:
        pass
    return False


def _mark_bundled_unusable(exe: Path, reason: str, runtime_root: Optional[Path]) -> bool:
    """Note (in the player's data folder) that a built-in build can't run here."""
    marker = install_info(exe) or {}
    tag, variant = marker.get("tag"), marker.get("variant")
    if not isinstance(variant, str) or not variant:
        found = _build_from_path(exe)  # no readable marker: fall back to the "<tag>-<variant>" folder name
        if found is None:
            return False
        tag, variant = found
    notes = _bundled_notes(runtime_root)
    note: dict[str, Any] = {
        "tag": str(tag or ""), "variant": variant, "reason": str(reason), "at": int(time.time()),
        "exe": str(Path(os.path.abspath(exe))),
    }
    fingerprint = _file_fingerprint(exe)
    if fingerprint is not None:
        note["file"] = fingerprint  # a repaired or different copy of the file gets a fresh check
    notes[_bundled_key(tag, variant)] = note
    path = _bundled_notes_path(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps({"schema": 1, "builds": notes}, indent=2), encoding="utf-8")
    os.replace(temp, path)  # all at once: a half-written note can't confuse the next launch
    return True


# A build marked unusable stops being downloaded again - but only for this
# long: a later llama.cpp release may well fix it (a CPU-dispatch bug, say).
UNUSABLE_RETRY_DAYS = 30


def unusable_variants(runtime_root: Optional[Path] = None) -> set[str]:
    """Names of builds ("cuda-12", ...) that can't run here (see :func:`unusable_reasons`)."""
    return set(unusable_reasons(runtime_root))


def unusable_reasons(runtime_root: Optional[Path] = None, *, now: Optional[float] = None) -> dict[str, str]:
    """{build name: why it can't run here ("glibc", "cpu_unsupported"...)}.

    :func:`mark_unusable` notes one *install* (one ``<tag>-<build>`` folder).
    A whole build type only counts as unusable while it has no working
    install left and its note is recent (UNUSABLE_RETRY_DAYS): one failed
    engine update must never block the older install that still works, or
    every later release of that build.
    """
    reasons: dict[str, str] = {}
    root = _llama_root(runtime_root)
    try:
        folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        folders = []  # nothing installed (a built game may still have notes about its own builds)
    bundled_notes = _bundled_notes(runtime_root)
    if not folders and not bundled_notes:
        return reasons
    working = {name for _exe, _tag, name in installed_runtimes(runtime_root)}
    now = time.time() if now is None else now
    notes: list[tuple[int, str, str]] = []
    if bundled_notes:
        # Built-in builds: a note only counts while that very build still ships
        # with the game (an update brings a new one, which gets a fresh chance).
        # When the game can't download engines, there is no newer release to
        # wait for, so the note doesn't expire.
        present: dict[str, list[Path]] = {}
        for _f, m, exe in _bundled_builds():
            present.setdefault(_bundled_key(m.get("tag"), m.get("variant")), []).append(exe)
        expires = downloads_allowed()
        for key, note in bundled_notes.items():
            variant = note.get("variant")
            if not isinstance(variant, str) or variant in working or key not in present:
                continue
            if not any(_bundled_note_applies(note, exe) for exe in present[key]):
                continue  # the program was repaired or replaced since: it gets a fresh chance
            at = note.get("at")
            if expires and isinstance(at, (int, float)) and now - at > UNUSABLE_RETRY_DAYS * 86400:
                continue
            notes.append((_tag_number(str(note.get("tag", ""))), variant, str(note.get("reason") or "unknown")))
    for folder in folders:
        marker = _read_marker(folder)
        reason = _unusable_reason(marker)
        variant = marker.get("variant") if marker else None
        if reason is None or not isinstance(variant, str) or variant in working:
            continue
        note = marker.get("unusable") if marker else None
        at = note.get("at") if isinstance(note, dict) else None
        if isinstance(at, (int, float)) and now - at > UNUSABLE_RETRY_DAYS * 86400:
            continue  # long ago: a newer release may have fixed it
        notes.append((_tag_number(str(marker.get("tag", ""))), variant, reason))
    for _tag, variant, reason in sorted(notes, reverse=True):  # the newest install's reason wins
        reasons.setdefault(variant, reason)
    return reasons


def _marked_install(llama_root: Path, tag: str, variant: RuntimeVariant) -> Optional[str]:
    """Why the install of exactly this release and build was marked unusable here (or None)."""
    return _unusable_reason(_read_marker(llama_root / f"{_safe_filename(tag)}-{variant.name}"))


def _unusable_reason(marker: Optional[dict]) -> Optional[str]:
    if not marker or not marker.get("unusable"):
        return None
    note = marker["unusable"]
    return str(note.get("reason") or "unknown") if isinstance(note, dict) else "unknown"


_UNUSABLE_WORDS = {
    "glibc": "needs a newer Linux (glibc) than this computer has",
    "cpu_unsupported": "needs processor features this computer doesn't have",
    "cant_execute": "can't run on this computer at all (a 32-bit system, or a different kind of processor?)",
    "gpu_arch": "doesn't support this graphics card's generation",
}


def unusable_message(variant_names: Mapping[str, str], *, downloads: Optional[bool] = None) -> str:
    """Plain English for "every build we could use already failed here for good".

    `downloads` says whether this copy of the game downloads engines
    (default: ask :func:`downloads_allowed`); a built game talks about its
    built-in engine instead of downloads.
    """
    parts = [f"the {(variant_by_name(name) or RuntimeVariant(name, ())).display} build "
             f"{_UNUSABLE_WORDS.get(reason, 'already failed to start here')}"
             for name, reason in sorted(variant_names.items())]
    if not (downloads_allowed() if downloads is None else downloads):
        return ("The game's built-in llama.cpp engine can't run on this computer - " + "; ".join(parts) + ". "
                + other_engines_hint())
    return ("The official llama.cpp engine can't run on this computer - " + "; ".join(parts) + ". So I won't "
            "download it again. " + other_engines_hint())


def newer_engine_unusable_message(variant: RuntimeVariant, tag: str, reason: str) -> str:
    """Plain English for "this model needs a newer engine, and the newer engine can't run here"."""
    words = _UNUSABLE_WORDS.get(reason, "doesn't start on this computer")
    tag_text = f" ({tag})" if tag else ""
    return (f"This model needs a newer llama.cpp engine than the one you have, but the newest {variant.display} "
            f"build{tag_text} {words}. Your current engine still works for other models - please pick another "
            "model.")


def usable_plan(specs: SystemSpecs, runtime_root: Optional[Path] = None) -> list[RuntimeVariant]:
    """`plan_variants` minus builds already known not to run here. Empty = nothing left to try."""
    broken = unusable_reasons(runtime_root)
    return [v for v in plan_variants(specs) if v.name not in broken]


def available_plan(specs: SystemSpecs, runtime_root: Optional[Path] = None) -> list[RuntimeVariant]:
    """The builds the game would really try here, best first (for display, e.g. ``--specs``).

    Like :func:`usable_plan`, but a built game only counts the builds it
    ships with (CPU stays last: a CPU build, or CPU mode on a graphics build).
    """
    plan = usable_plan(specs, runtime_root)
    if downloads_allowed():
        return plan
    groups = own_builds_first(installed_runtimes(runtime_root))
    here = groups[0] if groups else []  # its own builds (other installs only when it has none)
    return [v for v in plan if v.name == CPU.name or find_installed(v, specs, runtimes=here) is not None]


def engine_problem(specs: SystemSpecs, runtime_root: Optional[Path] = None) -> Optional[str]:
    """Why the built-in engine can't be used here at all (before any download), or None."""
    problem = platform_problem(specs.os_name, specs.arch)
    if problem:
        return problem
    if not usable_plan(specs, runtime_root):
        return unusable_message(unusable_reasons(runtime_root))
    return bundled_builds_problem(runtime_root)


def bundled_builds_problem(runtime_root: Optional[Path] = None) -> Optional[str]:
    """A built game whose every built-in build is noted as unable to run here: why, else None.

    The builds are all there - they just can't run on this computer (on a Mac
    the one Metal build is also the CPU build) - so "the engine is missing,
    verify the game's files" would be the wrong advice: this says what
    happened and what else can run a model instead.
    """
    if downloads_allowed():
        return None
    try:
        if not _bundled_builds() or installed_runtimes(runtime_root):
            return None
        reasons = unusable_reasons(runtime_root)
    except Exception:
        return None
    return unusable_message(reasons, downloads=False) if reasons else None


def _folder_size(folder: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(folder):
        for fname in files:
            with contextlib.suppress(OSError):
                total += (Path(dirpath) / fname).lstat().st_size
    return total


def _delete_folder(folder: Path) -> bool:
    """Delete a whole install folder, all or nothing: it is first renamed out of
    the way, which fails (and leaves it untouched) while Windows has its files
    open - i.e. while some game is still running that engine."""
    trash = folder.with_name(f".trash-{folder.name}-{os.getpid()}")
    try:
        os.replace(folder, trash)
    except OSError:
        return False
    shutil.rmtree(trash, ignore_errors=True)
    return True


def prune_old_installs(keep_exe: Path, *, runtime_root: Optional[Path] = None,
                       in_use: Optional[set[Path]] = None) -> tuple[int, int]:
    """Reclaim disk space after an engine update or a GPU -> CPU fallback.

    * Older installs of the same build as `keep_exe` (an engine update leaves
      the previous ``<tag>-<variant>`` folder behind - up to ~1 GB each for
      CUDA) are deleted, unless an engine in `in_use` (the ones running
      right now, in any copy of the game) lives there.
    * Builds marked unusable here keep only their ``install.json`` note (so
      they're never downloaded again); the rest of their files are deleted.

    Builds that ship inside the game are never touched. Returns ``(folders
    tidied, bytes freed)``. Never raises.
    """
    try:
        root = _llama_root(runtime_root)
        keep_marker = install_info(Path(keep_exe)) or {}
        keep_variant, keep_tag = keep_marker.get("variant"), _tag_number(str(keep_marker.get("tag", "")))
        if not isinstance(keep_variant, str):
            return 0, 0
        busy = {Path(p).resolve() for p in (in_use or set())} | {Path(keep_exe).resolve()}
        tidied = freed = 0
        for folder in [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]:
            marker = _read_marker(folder)
            if not marker:
                continue
            if marker.get("bundled") is True or _inside_engine_dirs(folder):
                continue  # part of the game itself: read-only, never tidied away
            if any(folder.resolve() in b.parents for b in busy):
                continue  # running right now (or the one we keep)
            if _unusable_reason(marker) is not None:
                size = _folder_size(folder) - len(json.dumps(marker))
                if size <= 64 * 1024:
                    continue  # only the note is left already
                emptied = folder.with_name(f".staging-empty-{folder.name}-{os.getpid()}")
                emptied.mkdir()
                (emptied / INSTALL_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
                if _delete_folder(folder):
                    os.replace(emptied, folder)  # the folder comes back with just its note
                    tidied, freed = tidied + 1, freed + max(0, size)
                else:
                    shutil.rmtree(emptied, ignore_errors=True)
                continue
            if marker.get("variant") == keep_variant and _tag_number(str(marker.get("tag", ""))) < keep_tag:
                size = _folder_size(folder)
                if _delete_folder(folder):
                    tidied, freed = tidied + 1, freed + size
        return tidied, freed
    except Exception:
        return 0, 0


def _find_install_with_assets(llama_root: Path, asset_names: list[str]) -> Optional[tuple[Path, str]]:
    """An existing install made from exactly these archives (e.g. Metal and CPU
    share one macOS build), so we never download the same files twice."""
    for exe, tag, _variant in installed_runtimes(llama_root.parent):
        marker = install_info(exe) or {}
        if sorted(marker.get("assets") or []) == sorted(asset_names):
            return exe, tag
    return None


def _remove_stale_staging(llama_root: Path) -> None:
    """Remove leftovers from an install that was killed half-way (power cut...)."""
    now = time.time()
    with contextlib.suppress(OSError):
        for p in llama_root.glob(".staging-*"):
            with contextlib.suppress(OSError):
                if now - p.stat().st_mtime > _STALE_STAGING_S:
                    shutil.rmtree(p, ignore_errors=True)
        for p in llama_root.glob(".trash-*"):  # an old engine copy whose deletion was cut short
            shutil.rmtree(p, ignore_errors=True)


def _check_disk_space(folder: Path, download_bytes: int) -> None:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    needed = int(download_bytes * 3.5) + 50 * 1024**2  # archive + unpacked copy + slack
    if free < needed:
        raise RuntimeInstallError(
            f"There isn't enough free disk space for the llama.cpp engine: it needs about "
            f"{needed / 1e6:,.0f} MB, and {free / 1e6:,.0f} MB is free. Free up some space and try again."
        )


def _rename_with_retry(src: Path, dst: Path, attempts: int = 30, delay_s: float = 0.5) -> None:
    """``os.replace`` that tolerates a lock of up to ~15 seconds on Windows,
    where antivirus software scans brand-new archives and .exe/.dll files
    (a CUDA build's libraries are hundreds of MB) before letting go."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def _friendly_os_error(exc: OSError, doing: str) -> RuntimeInstallError:
    """Plain English for a file-system error while installing the engine."""
    if exc.errno == errno.ENOSPC:
        return RuntimeInstallError(f"Your disk filled up while {doing}. Free some space and try again.")
    if isinstance(exc, PermissionError) or exc.errno in (errno.EACCES, errno.EPERM, errno.EBUSY):
        return RuntimeInstallError(
            f"A file was locked while {doing} - usually antivirus software checking the new files, or a "
            "permissions problem. Wait a moment and try again (nothing is broken)."
        )
    return RuntimeInstallError(f"Something went wrong on your disk while {doing} ({exc.strerror or exc}). "
                               "Please try again.")


def _describe_asset(asset: dict) -> str:
    return "NVIDIA CUDA runtime" if asset["name"].lower().startswith("cudart") else "llama.cpp engine"


def _asset_license(asset: dict) -> str:
    return CUDA_RUNTIME_LICENSE if asset["name"].lower().startswith("cudart") else LLAMA_CPP_LICENSE


def _existing_install(folder: Path) -> Optional[Path]:
    """The executable of a complete, usable install in `folder` (valid marker + file), or None.

    An install marked unusable here raises instead: downloading the same
    build again would fail the same way.
    """
    existing = _read_marker(folder)
    if not existing or not isinstance(existing.get("exe"), str):
        return None
    reason = _unusable_reason(existing)
    if reason is not None:
        raise RuntimeInstallError(unusable_message({str(existing.get("variant") or folder.name): reason}))
    exe = folder / Path(*PurePosixPath(existing["exe"]).parts)
    return exe if exe.is_file() else None


def _install(ui: UI, http: Any, release: dict, assets: list[dict], variant: RuntimeVariant, llama_root: Path) -> Path:
    """Download + verify + unpack `assets` into ``<llama_root>/<tag>-<variant>/``."""
    tag = str(release.get("tag_name") or "unknown")
    final_dir = llama_root / f"{_safe_filename(tag)}-{variant.name}"
    reason = _marked_install(llama_root, tag, variant)
    if reason is not None:  # this exact build already failed here for good: don't download it again
        raise RuntimeInstallError(unusable_message({variant.name: reason}))
    total = sum(a["size"] for a in assets if isinstance(a.get("size"), int))
    llama_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_staging(llama_root)
    _check_disk_space(llama_root, total)

    size_note = f"~{max(1, round(total / 1e6))} MB" if total else "a small download"
    extra = " - includes NVIDIA's CUDA runtime" if variant.needs_cudart else ""
    ui.info(f"Installing the llama.cpp engine (one-time, {size_note}{extra})...")
    ui.info(
        f"It's the official {escape(variant.display)} build {escape(tag)} from {LLAMA_CPP_URL}/releases "
        f"({escape(license_text(variant))}), unpacked into the game's own folder."
    )

    try:
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=llama_root))
    except OSError as exc:
        raise _friendly_os_error(exc, "preparing the engine's folder") from exc
    try:
        return _install_into(ui, http, release, assets, variant, tag, staging, final_dir)
    except OSError as exc:  # not RuntimeInstallError: that's already friendly
        raise _friendly_os_error(exc, "installing the llama.cpp engine") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _install_into(ui: UI, http: Any, release: dict, assets: list[dict], variant: RuntimeVariant, tag: str,
                  staging: Path, final_dir: Path) -> Path:
    """The body of `_install`: download and unpack into `staging`, then move into `final_dir`."""
    payload = staging / "install"
    payload.mkdir()
    for index, asset in enumerate(assets):
        archive = _download_asset(http, asset, staging / "downloads", ui, _describe_asset(asset))
        with ui.status("Unpacking and checking the files..."):
            unpack_archive(archive, payload, staging / f"unpacked-{index}")
        # Tidy as we go (a CUDA runtime archive is big), but a lock on it -
        # an antivirus scan - doesn't matter: the staging folder goes anyway.
        with contextlib.suppress(OSError):
            archive.unlink()
    exe = finish_unpacked(payload)
    rel_exe = exe.relative_to(payload).as_posix()
    marker = install_marker(dict(release, tag_name=tag), assets, variant, rel_exe)
    (payload / INSTALL_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if final_dir.exists():
        existing = _existing_install(final_dir)
        if existing is not None:
            # Another copy of the game finished the same install first: use it.
            return existing
        shutil.rmtree(final_dir, ignore_errors=True)  # a broken leftover without a valid marker
    try:
        _rename_with_retry(payload, final_dir)  # atomic: the install appears all at once
    except OSError as exc:
        # Another copy of the game may have finished the very same install
        # between our check and our rename: if so, simply use its copy.
        existing = _existing_install(final_dir)
        if existing is not None:
            return existing
        raise RuntimeInstallError(
            f"I couldn't move the finished engine into place ({exc.strerror or exc}). If another copy of the "
            "game is installing it right now, wait a moment and try again."
        ) from exc

    ui.success(f"The llama.cpp engine {escape(tag)} ({escape(variant.display)} build) is installed in {escape(str(final_dir))}")
    return final_dir / Path(*PurePosixPath(rel_exe).parts)


def ensure_llama_server(
    ui: UI,
    specs: SystemSpecs,
    *,
    variant: Optional[RuntimeVariant] = None,
    http: Any = None,
    runtime_root: Optional[Path] = None,
    update: bool = False,
) -> tuple[Path, RuntimeVariant]:
    """Make sure a working ``llama-server`` is installed; return ``(exe, variant)``.

    Without `variant`, walks :func:`plan_variants` best-first and uses the
    first build that is already installed or published for this computer
    (skipping builds that already failed to start here for good, see
    :func:`mark_unusable`). With `variant`, only that build is considered
    (used for GPU -> CPU fallback). Existing installs are reused; if GitHub
    can't be reached, an older install of a lower-ranked build is used rather
    than failing.

    With ``update=True`` the newest published release is fetched and
    installed unless it is already here - used when a model needs a newer
    engine (e.g. "unknown model architecture").

    In a built game (:func:`downloads_allowed` is False) the network is never
    touched: the choice is made from the builds already here - the ones
    shipped inside the game, plus any installed earlier - in plan order. If
    the plan's first builds (e.g. CUDA) aren't built in, the best one that is
    (Vulkan / Metal) is used, and the CPU build - or CPU mode on a graphics
    build - is the fallback.
    Raises RuntimeInstallError with a friendly message.
    """
    llama_root = _llama_root(runtime_root)
    problem = platform_problem(specs.os_name, specs.arch)
    if problem:
        raise RuntimeInstallError(problem)  # say so before downloading anything
    # Builds that already failed here for good (see mark_unusable) are never
    # downloaded again: they'd fail the same way.
    broken = unusable_reasons(llama_root.parent)
    if variant is not None:
        if variant.name in broken:
            raise RuntimeInstallError(unusable_message({variant.name: broken[variant.name]}))
        candidates = [variant]
    else:
        candidates = [v for v in plan_variants(specs) if v.name not in broken]
        if not candidates:
            raise RuntimeInstallError(unusable_message(broken))
    if not downloads_allowed():
        return _use_builds_here(ui, specs, candidates, variant, llama_root.parent, update=update)
    if update:
        return _update_engine(ui, specs, candidates[0], http or UrllibHttp(), llama_root)

    installed: dict[str, tuple[Path, str]] = {}
    for exe, tag, name in installed_runtimes(llama_root.parent):
        installed.setdefault(name, (exe, tag))  # newest first, so keep the first

    def reuse(v: RuntimeVariant) -> tuple[Path, RuntimeVariant]:
        exe, tag = installed[v.name]
        _say_already_here(ui, exe, tag, v)
        return exe, v

    if candidates[0].name in installed:
        return reuse(candidates[0])

    http = http or UrllibHttp()
    try:
        with ui.status("Checking GitHub for the official llama.cpp engine..."):
            releases = fetch_releases(http=http, ui=ui)
    except RuntimeInstallError as exc:
        for v in candidates:
            if v.name in installed:
                ui.warn(f"{escape(str(exc))} Using the engine you installed earlier instead.")
                return reuse(v)
        raise

    for index, cand in enumerate(candidates):
        if cand.name in installed:
            return reuse(cand)
        picked = pick_release(releases, cand, specs.os_name, specs.arch)
        if picked is None:
            if index + 1 < len(candidates):
                ui.info(
                    f"There's no ready-made {escape(cand.display)} build for your computer right now, "
                    f"so let's use the {escape(candidates[index + 1].display)} build."
                )
            continue
        release, assets = picked
        same = _find_install_with_assets(llama_root, [a["name"] for a in assets])
        if same is not None:
            _say_already_here(ui, same[0], same[1], cand)
            return same[0], cand
        return _install(ui, http, release, assets, cand, llama_root), cand

    wanted = " / ".join(v.display for v in candidates)
    raise RuntimeInstallError(
        f"Sorry - I couldn't find an official prebuilt llama.cpp engine ({wanted}) for "
        f"{specs.os_name} on {specs.arch}. " + other_engines_hint("may still work on this computer")
    )


def _say_already_here(ui: UI, exe: Path, tag: str, variant: RuntimeVariant) -> None:
    """Tell the player which engine build is being used, with nothing to download."""
    if not is_bundled(exe):
        ui.success(f"The llama.cpp engine is already installed ({escape(tag)}, {escape(variant.display)}) - "
                   "no download needed.")
        return
    built = variant_by_name((install_info(exe) or {}).get("variant"))
    what = (f"{variant.display} build" if built is None or built.name == variant.name
            else f"{built.display} build, in {variant.display} mode")
    ui.success(f"Using the game's built-in llama.cpp engine ({escape(tag)}, {escape(what)}) - nothing to download.")


def _use_builds_here(ui: UI, specs: SystemSpecs, candidates: list[RuntimeVariant],
                     requested: Optional[RuntimeVariant], runtime_root: Path, *,
                     update: bool = False) -> tuple[Path, RuntimeVariant]:
    """Pick an engine from the builds already on this computer, without touching the network.

    Used when this copy of the game doesn't download engines (a built game
    ships its own). `candidates` are tried in plan order; with ``update=True``
    the newest build of the first candidate that is here is returned, quietly
    (a newer one can only come with a game update): the caller knows whether
    that is the engine that just failed or a newer one worth mentioning.
    """
    runtimes = installed_runtimes(runtime_root)
    if not runtimes:
        raise RuntimeInstallError(bundled_builds_problem(runtime_root) or ENGINE_MISSING_MESSAGE)
    groups = own_builds_first(runtimes)  # this copy's own builds before any other install
    for group in groups:
        for cand in candidates:
            hit = find_installed(cand, specs, runtimes=group)
            if hit is not None:
                exe, tag = hit
                if not update:
                    _say_already_here(ui, exe, tag, cand)
                return exe, cand
    usable_here = [r for group in groups for r in group if _made_for(r[0], specs)]
    if usable_here and any(c.name == CPU.name for c in candidates):
        # No separate CPU build here - but every official build also runs on the
        # processor alone (the backend starts it with --device none -ngl 0).
        order = {v.name: i for i, v in enumerate(plan_variants(specs))}
        own = {id(r) for r in (groups[0] if len(groups) > 1 else [])}
        exe, tag, _name = min(usable_here, key=lambda r: (id(r) not in own, order.get(r[2], len(order))))
        _say_already_here(ui, exe, tag, CPU)
        return exe, CPU
    if not usable_here:
        raise RuntimeInstallError(bundled_builds_problem(runtime_root) or ENGINE_MISSING_MESSAGE)
    wanted = requested or candidates[0]
    raise RuntimeInstallError(
        f"The {wanted.display} build of the llama.cpp engine isn't built into this copy of the game "
        "(and it doesn't download engines)."
    )


def _update_engine(ui: UI, specs: SystemSpecs, variant: RuntimeVariant, http: Any, llama_root: Path) -> tuple[Path, RuntimeVariant]:
    """Install the newest published build of `variant`, unless it's already installed."""
    with ui.status("Checking GitHub for a newer llama.cpp engine..."):
        releases = fetch_releases(http=http, ui=ui)
    picked = pick_release(releases, variant, specs.os_name, specs.arch)
    if picked is None:
        raise RuntimeInstallError(
            f"There's no newer {variant.display} build of the llama.cpp engine for this computer right now."
        )
    release, assets = picked
    tag = str(release.get("tag_name") or "")
    reason = _marked_install(llama_root, tag, variant)
    if reason is not None:
        raise RuntimeInstallError(newer_engine_unusable_message(variant, tag, reason))
    for exe, installed_tag, name in installed_runtimes(llama_root.parent):
        if name == variant.name and installed_tag == tag:
            ui.info(f"You already have the newest llama.cpp engine ({escape(tag)}, {escape(variant.display)}).")
            return exe, variant
    same = _find_install_with_assets(llama_root, [a["name"] for a in assets])
    if same is not None:
        return same[0], variant
    return _install(ui, http, release, assets, variant, llama_root), variant


RUNTIME_EXPLAINER = """\
**llama.cpp** is a free, open-source (MIT) program that runs AI language models
on ordinary computers. Its **llama-server** tool loads a model file (a *GGUF*)
and answers chat requests at a private address on *your own* machine
(`http://127.0.0.1:<port>`). Nothing you type is sent to the internet by
the local model. (Jev, if you switch it on, is the one exception: it's an
online service.)

**Why download a prebuilt copy?** Building llama.cpp from source needs
developer tools. The llama.cpp team publishes ready-made builds on GitHub for
every release, so the game fetches the official one that suits your computer,
checks its size and SHA-256 fingerprint, and unpacks it into the game's own
folder. No admin rights, nothing installed system-wide - delete the folder to
remove it.

**Which build?**
- **CUDA** - for NVIDIA graphics cards. Runs the model on the GPU using
  NVIDIA's CUDA toolkit (usually the fastest). Needs a reasonably recent driver.
- **Vulkan** - a graphics standard supported by AMD, Intel *and* NVIDIA drivers.
- **Metal** - Apple's GPU technology on Apple Silicon Macs (M1, M2, M3...), where
  the GPU shares the Mac's memory ("unified memory").
- **CPU** - works on every computer, just slower.

If a GPU build can't start (for example because of an old driver), the game
quietly falls back to the next option, ending with the CPU build.
"""

RUNTIME_EXPLAINER_BUILT_IN = """\
**llama.cpp** is a free, open-source (MIT) program that runs AI language models
on ordinary computers. Its **llama-server** tool loads a model file (a *GGUF*)
and answers chat requests at a private address on *your own* machine
(`http://127.0.0.1:<port>`). Nothing you type is sent to the internet by
the local model. (Jev, if you switch it on, is the one exception: it's an
online service.)

**Where does it come from?** Building llama.cpp from source needs developer
tools, so the llama.cpp team publishes ready-made builds for every release.
This copy of the game carries the official builds for your kind of computer
inside its own folder: nothing is downloaded or installed to run them, and no
admin rights are needed. The only download is the AI model itself, during
setup. Game updates bring newer engines.

**Which build?**
- **Vulkan** - a graphics standard supported by AMD, Intel *and* NVIDIA drivers
  (Windows and Linux).
- **Metal** - Apple's GPU technology on Apple Silicon Macs (M1, M2, M3...), where
  the GPU shares the Mac's memory ("unified memory").
- **CPU** - works on every computer, just slower.

If the graphics-card build can't start (for example because of an old driver),
the game quietly falls back to the CPU.
"""


def runtime_explainer() -> str:
    """The "how the engine works" Learn page for this copy of the game.

    A built game ships its engine and never downloads one, so it gets its own
    page (no downloads, no CUDA builds); a developer copy gets
    :data:`RUNTIME_EXPLAINER`.
    """
    return RUNTIME_EXPLAINER if downloads_allowed() else RUNTIME_EXPLAINER_BUILT_IN


def engine_summary() -> str:
    """Where this copy's llama.cpp engine comes from, in one line (``--specs``).

    A built game says which release and builds it carries - read from its
    ``distribution.json`` and engine folder - so a build check can confirm it
    really uses its built-in engine (``packaging/smoke_test.sh``)::

        built into the game: llama.cpp b7000 (CPU, Vulkan); engine downloads off
    """
    if downloads_allowed():
        return "downloaded from the official llama.cpp releases when needed; engine downloads on"
    try:
        tag = distribution.load().llama_cpp_tag
    except Exception:
        tag = None
    names: list[str] = []
    for _folder, marker, _exe in _bundled_builds():
        variant = str(marker.get("variant") or "")
        name = (variant_by_name(variant) or RuntimeVariant(variant or "unknown", ())).display
        if name not in names:
            names.append(name)
    release = f"llama.cpp {tag}" if tag else "llama.cpp (release unknown - distribution.json not found)"
    if not names:
        return f"built into the game, but no engine builds were found ({release}); engine downloads off"
    return f"built into the game: {release} ({', '.join(names)}); engine downloads off"
