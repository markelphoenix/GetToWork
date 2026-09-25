"""The local-LLM "engines" the game can talk to, plus a quick detector.

Every backend implements the same small interface (:class:`LLMBackend` in
``base.py``), so the rest of the game never cares which one is running:

* :class:`LlamaServerBackend` - the default. Downloads the official prebuilt
  llama.cpp ``llama-server`` and runs your model with it, fully automatically.
* :class:`OllamaBackend` - uses the Ollama app, if you already have it.
* :class:`LlamaCppBackend` - uses the ``llama-cpp-python`` package (tinkerers).
* :class:`MockBackend` - a scripted, offline pretend model (``--mock``).

The backend classes are imported *lazily* (PEP 562 module ``__getattr__``):
``from gettowork.backends import MockBackend`` only loads ``mock.py``. That
keeps start-up fast, and means one backend with a missing optional
dependency can never stop the others from loading.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
from typing import TYPE_CHECKING, Any, Callable, Optional

from .base import BackendError, LLMBackend

if TYPE_CHECKING:  # for editors and type checkers only; never runs
    from .llamacpp import LlamaCppBackend
    from .llamaserver import LlamaServerBackend
    from .mock import MockBackend
    from .ollama import OllamaBackend

__all__ = [
    "LLMBackend",
    "BackendError",
    "LlamaServerBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "MockBackend",
    "detect_backends",
]

# Public name -> the submodule that defines it.
_LAZY_EXPORTS = {
    "LlamaServerBackend": ".llamaserver",
    "OllamaBackend": ".ollama",
    "LlamaCppBackend": ".llamacpp",
    "MockBackend": ".mock",
}


def __getattr__(name: str) -> Any:
    """Import a backend class the first time someone asks for it (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache it, so this function isn't called again
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# ---------------------------------------------------------------------------
# detect_backends()
# ---------------------------------------------------------------------------

# Operating systems / CPU types that ggml-org/llama.cpp publishes prebuilt
# engines for. Only used if runtime_install.py can't be imported.
_FALLBACK_OS = {"windows": "Windows", "darwin": "macOS", "linux": "Linux"}
_FALLBACK_ARCH = {
    "x86_64": "x64", "amd64": "x64", "x64": "x64",
    "arm64": "arm64", "aarch64": "arm64", "arm64e": "arm64",
}


def _platform_supported(system: str, machine: str) -> bool:
    try:
        from ..runtime_install import is_platform_supported
    except Exception:  # keep working even if that module is broken
        return system.strip().lower() in _FALLBACK_OS and machine.strip().lower() in _FALLBACK_ARCH
    return bool(is_platform_supported(system, machine))


def _check_managed(system: str, machine: str) -> tuple[bool, str]:
    if not _platform_supported(system, machine):
        return False, (
            f"llama.cpp doesn't publish a ready-made engine for {system or 'this system'} on "
            f"{machine or 'this processor'}, so automatic setup isn't possible here. "
            "Ollama or llama-cpp-python may still work."
        )
    try:
        from ..runtime_install import installed_runtimes

        runtimes = installed_runtimes()
    except Exception:
        runtimes = []
    if runtimes:
        _exe, tag, variant = runtimes[0]
        return True, f"The llama.cpp engine is already installed ({tag}, {variant})."
    return True, "The official llama.cpp engine can be downloaded and set up automatically (MIT license)."


def _check_ollama(host: Optional[str], http: Any) -> tuple[bool, str]:
    try:
        from .ollama import OllamaBackend as _Ollama

        return _Ollama("", host=host, http=http).is_available()
    except Exception as exc:
        return False, f"Couldn't check for Ollama ({exc})."


def _check_llamacpp(find_spec: Callable[[str], Any]) -> tuple[bool, str]:
    try:
        found = find_spec("llama_cpp") is not None
    except Exception:  # find_spec can raise for oddly broken installs
        found = False
    if found:
        return True, "llama-cpp-python is installed."
    return False, "llama-cpp-python isn't installed (optional: pip install llama-cpp-python)."


def detect_backends(
    *,
    ollama_host: Optional[str] = None,
    http: Any = None,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    find_spec: Optional[Callable[[str], Any]] = None,
) -> dict[str, tuple[bool, str]]:
    """Which engines could run a model on this computer right now?

    Returns ``{"managed": (ok, why), "ollama": (ok, why), "llamacpp": (ok, why)}``
    where ``why`` is one plain-English sentence for the player.

    * ``managed`` - llama.cpp publishes a prebuilt engine for this OS + CPU
      (Windows/macOS/Linux on x64 or arm64). Nothing is downloaded here.
    * ``ollama`` - an Ollama server answers ``GET /api/version`` within ~1.5 s.
    * ``llamacpp`` - the ``llama_cpp`` package is importable (not imported).

    Fast and never raises. The keyword arguments exist so tests can pretend
    to be another computer without touching the real one.
    """
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    find_spec = find_spec or importlib.util.find_spec

    checks: dict[str, Callable[[], tuple[bool, str]]] = {
        "managed": lambda: _check_managed(system, machine),
        "ollama": lambda: _check_ollama(ollama_host, http),
        "llamacpp": lambda: _check_llamacpp(find_spec),
    }
    results: dict[str, tuple[bool, str]] = {}
    for key, check in checks.items():
        try:
            results[key] = check()
        except Exception as exc:  # belt and braces: detection must never crash the game
            results[key] = (False, f"Couldn't check ({exc}).")
    return results
