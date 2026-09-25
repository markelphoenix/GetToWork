"""Run a GGUF model with the official llama.cpp ``llama-server``, automatically.

This is the game's default backend. The player never installs anything by
hand; :meth:`LlamaServerBackend.prepare` does it all:

1. **Engine**: make sure a prebuilt ``llama-server`` for this computer is
   installed (:func:`gettowork.runtime_install.ensure_llama_server`).
2. **Model**: download the chosen GGUF file from Hugging Face
   (``gettowork.download.download_gguf``) unless it's already on disk.
3. **Start**: launch ``llama-server`` as a child process listening only on
   ``127.0.0.1`` (your own computer) on a free port, then poll ``GET /health``
   until the model has loaded.
4. **Fallbacks**: if a GPU build crashes (old driver, missing CUDA...), read
   its log, explain in one sentence, and try the next build from
   :func:`gettowork.runtime_install.plan_variants` - ending with the CPU build
   running with ``-ngl 0`` (zero layers on the GPU).

After that, :meth:`chat` talks to the server's OpenAI-compatible
``POST /v1/chat/completions`` endpoint, and :meth:`close` stops the process
(also registered with :mod:`atexit`).

**Never leaving the engine behind.** A running engine holds gigabytes of
memory, so it must stop when the game stops - however the game stops:

* Windows: the engine joins a *Job Object* with "kill on close", so Windows
  itself ends it when the game's process goes away for any reason (closing
  the console window included, where no Python cleanup code gets to run).
* Linux: the engine asks the kernel for a "parent-death signal", so it is
  stopped even if the game is killed outright.
* Everywhere: a small record of each running engine is kept next to its log;
  if an earlier game was force-quit and left one running, the next launch
  finds it and stops it.

The engine also runs in its own process group (POSIX), so pressing Ctrl+C in
the terminal only interrupts the game - which then stops the engine itself -
instead of killing the engine behind the game's back.

**Privacy.** The server listens only on ``127.0.0.1`` and requires a random
per-launch API key, so other programs and web pages on this computer can't
use it. Secrets from the player's environment (API tokens) and stray
``LLAMA_*`` settings are not passed to it.

Everything that touches the outside world (HTTP, starting processes, the
clock, installing, downloading) can be swapped out, which is how the tests
run without a network, a GPU or a real model.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import platform
import re
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Optional

from rich.markup import escape

from .. import config
from .. import runtime_install
from ..reasoning import split_reasoning
from ..runtime_install import (
    CPU,
    NETWORK_ERRORS,
    RuntimeInstallError,
    RuntimeVariant,
    UrllibHttp,
    install_info,
    installed_runtimes,
    is_platform_supported,
    plan_variants,
    variant_by_name,
)
from ..types import LLMResult, ModelEntry, SystemSpecs
from ..ui import UI, UserQuit, safe_text
from .base import RETRY_WITHOUT_THINKING_NOTICE, BackendError, EngineStopped, LLMBackend

__all__ = [
    "LlamaServerBackend",
    "build_server_args",
    "classify_server_log",
    "find_free_port",
    "server_env",
    "gpu_offload_from_log",
    "gpu_devices_from_listing",
    "missing_library_name",
    "health_timeout_for",
    "reap_orphaned_servers",
    "WAKE_UP_MESSAGE",
]

HEALTH_TIMEOUT_S = 180.0  # the minimum wait for the model to load
HEALTH_SECONDS_PER_GB = 15.0  # ...plus time to read big files from a slow disk (~70 MB/s)
HEALTH_MAX_TIMEOUT_S = 1800.0  # never wait more than half an hour
HEALTH_GRACE_S = 60.0  # keep waiting while the engine's log still shows it working
HEALTH_POLL_S = 0.5
DISK_PROGRESS_BYTES = 1_000_000  # reading at least this much from disk between checks = still loading
HEALTH_REQUEST_TIMEOUT_S = 2.0
CHAT_TIMEOUT_S = 300.0  # the minimum; scaled up for slow models once we've measured them
CHAT_TIMEOUT_MAX_S = 1800.0
CHAT_PROMPT_ALLOWANCE_S = 60.0  # reading a long prompt on a CPU takes a while too
STOP_TIMEOUT_S = 5.0
LOG_TAIL_BYTES = 8192
LOG_TAIL_LINES = 12
LOG_SCAN_BYTES = 512 * 1024  # how much of the log we read to check GPU use after start-up
LOG_KEEP_DAYS = 7  # old per-port logs are tidied away after this long
OWNER_SUFFIX = ".owner.json"  # "which game started this engine?" records, next to the logs

# Environment variables the engine never needs: credentials for other
# services, plus llama.cpp's own LLAMA_* settings (LLAMA_API_KEY, LLAMA_ARG_*)
# which would silently override what the game asks for.
_SECRET_ENV_VARS = config.SECRET_ENV_VARS

WAKE_UP_MESSAGE = "Waking up the model... big brains take a moment"
BENCHMARK_PROMPT = "Count from 1 to 40, separated by commas. Reply with the numbers only."
# The warm-up reads a prompt about as long as a real story turn (~1,000 tokens),
# so it exercises the same batched GPU work a turn does: some driver or memory
# problems only show up there, not on a one-line prompt.
BENCHMARK_CONTEXT = " ".join(
    f"Step {i}: the goose adjusts its tiny hat, the kettle hums a tune, and the bus waits politely."
    for i in range(1, 46)
)
ENGINE_CHECK_TIMEOUT_S = 15.0  # `llama-server --version` right after installing: does it start at all?
# Signs in the start-up log that the model is actually being loaded (progress
# dots, buffers allocated, the context created...). A graphics-card build that
# goes quiet *before* any of these is stuck setting up the device, not loading.
_LOAD_PROGRESS_RE = re.compile(
    r"\.{4,}|buffer size|llama_context|llama_kv_cache|KV self size|model loaded|all slots are idle",
    re.IGNORECASE,
)

# Windows: start llama-server without popping up a console window.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

# Used when the player points us at their own llama-server (no install.json).
CUSTOM_VARIANT = RuntimeVariant(name="custom", asset_patterns=(), gpu=True, label="your own llama-server")


# ---------------------------------------------------------------------------
# Small pure helpers (easy to test, handy to read)
# ---------------------------------------------------------------------------


def build_server_args(
    exe: Path | str,
    model_path: Path | str,
    *,
    port: int,
    n_ctx: int,
    cpu_only: bool = False,
    minimal: bool = False,
) -> list[str]:
    """The ``llama-server`` command line, as a list (never a shell string).

    * ``--host 127.0.0.1`` keeps the server private to this computer.
    * ``--reasoning-format deepseek`` puts a thinking model's chain-of-thought
      in a separate ``reasoning_content`` field instead of the answer.
    * ``--no-webui`` skips the built-in chat website; ``-np 1`` = one chat slot.
    * GPU layers are left to llama.cpp's automatic "fit" logic, except in CPU
      mode: ``--device none`` stops llama.cpp using any graphics card at all
      (``-ngl 0`` alone still lets it borrow the GPU to read long prompts),
      and ``-ngl 0`` keeps every layer in RAM for builds without ``--device``.
    * ``minimal`` drops the optional flags, for builds that don't know them.

    The per-launch API key is passed in the environment (``LLAMA_API_KEY``,
    see :func:`server_env`), not here, so it never shows up in a process list.
    """
    args = [str(exe), "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port), "-c", str(n_ctx)]
    if not minimal:
        args += ["--reasoning-format", "deepseek", "--no-webui", "-np", "1"]
    if cpu_only:
        if not minimal:
            args += ["--device", "none"]
        args += ["-ngl", "0"]
    return args


def find_free_port() -> int:
    """Ask the operating system for an unused TCP port on 127.0.0.1.

    Binding to port 0 means "any free port"; we read which one we got, then
    release it for llama-server to use a moment later.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_env(exe: Path | str, base: Optional[dict[str, str]] = None, *, api_key: Optional[str] = None) -> dict[str, str]:
    """Environment for the child process.

    * Lets it find the libraries shipped next to ``llama-server`` (e.g.
      ``libggml-cuda.so`` or ``cudart64_12.dll``).
    * Leaves out secrets it doesn't need (API tokens for other services) and
      any ``LLAMA_*`` variables: llama.cpp reads e.g. ``LLAMA_API_KEY`` or
      ``LLAMA_ARG_N_GPU_LAYERS`` from the environment, and a leftover one from
      the player's own llama.cpp experiments would quietly break the game.
    * With `api_key`, sets ``LLAMA_API_KEY`` so the server only answers
      requests that carry it.
    """
    source = os.environ if base is None else base
    env = {
        k: v for k, v in source.items()
        if k.upper() not in _SECRET_ENV_VARS and not k.upper().startswith("LLAMA_")
    }
    if api_key:
        env["LLAMA_API_KEY"] = api_key
    exe_dir = str(Path(exe).parent)
    system = platform.system()
    if system == "Linux":
        old = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = exe_dir + (os.pathsep + old if old else "")
    elif system == "Windows":
        env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
    return env


# llama.cpp's own wording for a flag it doesn't know ("error: invalid argument:
# --no-webui", older builds "error: unknown argument: ..."). Anchored to a flag
# after the colon, so "mmap failed: Invalid argument" (a file-system error) or
# "CUDA error: invalid argument" (a GPU error) never count as a bad flag.
_BAD_ARGS_RE = re.compile(
    r"(?:invalid|unknown|unrecogni[sz]ed) (?:argument|option)s?:\s*['\"]?-"
    r"|unrecognized arguments:\s*-",
    re.IGNORECASE,
)
_PORT_RE = re.compile(
    r"address already in use|(?:couldn't|could not|failed to|unable to) bind|bind(?:ing)? failed"
    r"|failed to listen|only one usage of each socket address",
    re.IGNORECASE,
)
_GLIBC_RE = re.compile(r"GLIBC(?:XX)?_[\d.]+'? not found", re.IGNORECASE)
_GPU_WORD_RE = re.compile(
    r"cuda|cublas|cudart|nvcuda|nvidia|vulkan|vk::|vk_error|\bhip|amdhip|rocblas|rocm|\bmetal|ggml_metal|\bmtl",
    re.IGNORECASE,
)
_FAIL_WORD_RE = re.compile(
    r"error|fail|insufficient|not found|cannot open|can't open|unable|lost|out of memory"
    r"|\bno\b[^\n]*\bdevices?\b|not detected|unsupported|too old",
    re.IGNORECASE,
)
# A CUDA build that has no code for this graphics card's generation (e.g. a
# CUDA 13 build on a GTX 10xx / Titan V: CUDA 13 dropped those cards).
_GPU_ARCH_RE = re.compile(
    r"no kernel image is available|unsupported gpu architecture"
    r"|not compiled for (?:this|the current) (?:gpu|device|architecture)",
    re.IGNORECASE,
)
# "error while loading shared libraries: libgomp.so.1: cannot open shared object file"
_MISSING_LIB_NAME_RE = re.compile(
    r"error while loading shared libraries:\s*([^\s:]+)"
    r"|Library not loaded:\s*(\S+)"
    r"|(?:The code execution cannot proceed because\s+)?(\S+\.dll) was not found",
    re.IGNORECASE,
)
# Linux: the package that usually provides a library the prebuilt engine needs.
_LINUX_LIBRARY_PACKAGES = (
    ("libgomp", "libgomp1"),
    ("libssl", "libssl3"),
    ("libcrypto", "libssl3"),
    ("libvulkan", "libvulkan1"),
    ("libstdc++", "libstdc++6"),
    ("libcurl", "libcurl4"),
)
_MISSING_LIB_RE = re.compile(
    r"error while loading shared libraries|cannot open shared object file|Library not loaded|\.dll was not found",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"out of memory|failed to allocate|bad_alloc|cannot allocate memory|unable to allocate"
    r"|insufficient memory|not enough memory",
    re.IGNORECASE,
)
_MODEL_RE = re.compile(
    r"failed to load model|error loading model|unable to load model|invalid magic"
    r"|unknown model architecture|gguf_init_from_file",
    re.IGNORECASE,
)
# The model uses a design this engine build doesn't know yet: a newer engine fixes it.
_UNSUPPORTED_ARCH_RE = re.compile(r"unknown (?:model )?architecture:?\s*'?([\w.\-]*)'?", re.IGNORECASE)
# Signs that the file itself is damaged or incomplete (re-downloading helps).
_CORRUPT_RE = re.compile(
    r"invalid magic|not within the file bounds|failed to read|unexpected(?:ly)? (?:reached )?end of file"
    r"|truncated|corrupt|tensor data is not|file is too small|incomplete",
    re.IGNORECASE,
)
# After a healthy start: did llama.cpp put any layers on a graphics card?
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.IGNORECASE)
_TENSORS_LOADED_RE = re.compile(r"\bload_tensors:", re.IGNORECASE)
# Where the model's weights went: "load_tensors:   CUDA0 model buffer size = ..."
# (older builds: "llm_load_tensors:      CUDA0 buffer size = ..."). CPU,
# CPU_Mapped, CPU_REPACK... and pinned "CUDA_Host" buffers are system RAM.
_MODEL_BUFFER_RE = re.compile(r"load_tensors:\s+(\S+)\s+(?:model\s+)?buffer size", re.IGNORECASE)
# `llama-server --list-devices` prints "Available devices:" and then one
# "  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)" line per usable
# graphics device - or "  (none)" when this GPU build can't use any.
_DEVICES_HEADER_RE = re.compile(r"^\s*Available devices:\s*$", re.IGNORECASE | re.MULTILINE)
_DEVICE_LINE_RE = re.compile(r"^\s+([A-Za-z][\w.\-]*)\s*:\s*\S")
_NOT_A_GPU_DEVICE_RE = re.compile(r"^(?:CPU|BLAS|RPC|ACCEL)", re.IGNORECASE)
_WINDOWS_MISSING_DLL_CODES = {0xC0000135, -1073741515, 0xC0000139, -1073741511}  # DLL / entry point not found
_ILLEGAL_INSTRUCTION_CODES = {-4, 132, 0xC000001D, -1073741795}
_KILLED_CODES = {-9, 137}  # SIGKILL, usually the Linux out-of-memory killer


def classify_server_log(text: str, returncode: Optional[int] = None) -> str:
    """Guess why llama-server stopped, from the end of its log.

    Returns one of: ``"bad_args"`` (a flag this build doesn't know), ``"port"``
    (port already taken), ``"glibc"`` (Linux too old for the prebuilt binary),
    ``"gpu_arch"`` (a CUDA build with no code for this card's generation),
    ``"gpu"`` (CUDA/Vulkan/Metal/ROCm driver or GPU-memory trouble),
    ``"missing_library"``, ``"memory"``, ``"model_unsupported"`` (a model
    design this engine build doesn't know yet - a newer engine fixes it),
    ``"model"`` (a damaged or unreadable GGUF), ``"cpu_unsupported"``
    (illegal instruction) or ``"unknown"``.
    """
    text = text or ""
    if _BAD_ARGS_RE.search(text):
        return "bad_args"
    if _PORT_RE.search(text):
        return "port"
    if _GLIBC_RE.search(text):
        return "glibc"
    if _GPU_ARCH_RE.search(text):
        return "gpu_arch"
    # A GPU problem = one line that mentions a GPU technology AND a failure.
    # (Healthy logs mention CUDA too, e.g. "found 1 CUDA devices".)
    for line in text.splitlines():
        if _GPU_WORD_RE.search(line) and _FAIL_WORD_RE.search(line):
            return "gpu"
    if _MISSING_LIB_RE.search(text) or returncode in _WINDOWS_MISSING_DLL_CODES:
        return "missing_library"
    if _MEMORY_RE.search(text) or returncode in _KILLED_CODES:
        return "memory"
    if _UNSUPPORTED_ARCH_RE.search(text):
        return "model_unsupported"
    if _MODEL_RE.search(text):
        return "model"
    if "illegal instruction" in text.lower() or returncode in _ILLEGAL_INSTRUCTION_CODES:
        return "cpu_unsupported"
    return "unknown"


def missing_library_name(text: str) -> Optional[str]:
    """The library a failed start couldn't find ("libgomp.so.1", "MSVCP140.dll"...), if the log says."""
    m = _MISSING_LIB_NAME_RE.search(text or "")
    if m is None:
        return None
    name = next((g for g in m.groups() if g), "")
    return Path(name.strip("'\"")).name or None


def missing_library_hint(text: str, system: Optional[str] = None) -> str:
    """Plain English for "a system library is missing", naming it and the usual fix for this OS."""
    system = system or platform.system()
    name = missing_library_name(text)
    named = f" ({name})" if name else ""
    if system == "Windows":
        return (f"The llama.cpp engine needs a system library that isn't installed on this computer{named}. "
                "On Windows this is usually the free Microsoft Visual C++ Redistributable "
                "(https://aka.ms/vs/17/release/vc_redist.x64.exe) - install it and try again.")
    if system == "Linux":
        package = next((pkg for prefix, pkg in _LINUX_LIBRARY_PACKAGES if name and name.startswith(prefix)), None)
        if package:
            how = f"install the '{package}' package (for example 'sudo apt install {package}' on Ubuntu or Debian)"
        else:
            how = "install it with your Linux distribution's package manager"
        return (f"The llama.cpp engine needs a system library that isn't installed on this computer{named}: "
                f"{how}, then try again.")
    if system == "Darwin":
        return (f"The llama.cpp engine needs a system library that isn't on this Mac{named}. Updating macOS "
                "usually fixes this.")
    return f"The llama.cpp engine needs a system library that isn't installed on this computer{named}."


def unsupported_architecture(text: str) -> Optional[str]:
    """The architecture name from an "unknown model architecture: 'x'" log line."""
    m = _UNSUPPORTED_ARCH_RE.search(text or "")
    return (m.group(1) or None) if m else None


def _is_gpu_buffer(name: str) -> bool:
    upper = name.upper()
    return not upper.startswith("CPU") and not upper.endswith("_HOST")


def gpu_offload_from_log(text: str) -> Optional[bool]:
    """Did llama.cpp put any model layers on a graphics card? (From its start-up log.)

    Decided by where the weights went ("CUDA0 model buffer size", "Vulkan0",
    "MTL0", "ROCm0"... = a graphics card; "CPU_Mapped" = system RAM) when the
    log says so. The "offloaded 37/37 layers to GPU" line alone can't be
    trusted: every official build includes the RPC backend, which makes
    llama.cpp report layers "offloaded" even with no graphics card at all.
    Without buffer lines, the offload count is used; a log that loaded the
    model without mentioning a GPU at all means False. None: the log doesn't
    say - current builds print none of this at their normal log level, which
    is why the game also asks the engine which devices it can see
    (``--list-devices``) before loading the model.
    """
    text = text or ""
    buffers = [m.group(1) for m in _MODEL_BUFFER_RE.finditer(text)]
    if buffers:
        return any(_is_gpu_buffer(name) for name in buffers)
    counts = [int(m.group(1)) for m in _OFFLOAD_RE.finditer(text)]
    if counts:
        return any(c > 0 for c in counts)
    if _TENSORS_LOADED_RE.search(text):
        return False
    return None


def gpu_devices_from_listing(text: str) -> Optional[list[str]]:
    """The graphics devices in ``llama-server --list-devices`` output.

    ``[]`` means the build can't see any (a missing or broken driver, a
    CUDA library that wouldn't load...). None means the output isn't a device
    list (an old build that doesn't know the flag, or something went wrong).
    """
    text = text or ""
    header = _DEVICES_HEADER_RE.search(text)
    if header is None:
        return None
    devices: list[str] = []
    for line in text[header.end():].splitlines():
        if not line.strip():
            continue
        m = _DEVICE_LINE_RE.match(line)
        if m is None:
            if line.startswith((" ", "\t")):
                continue  # "  (none)", or a wrapped description
            break  # the listing is over
        if not _NOT_A_GPU_DEVICE_RE.match(m.group(1)):
            devices.append(m.group(1))
    return devices


def health_timeout_for(model_gb: float, minimum: float = HEALTH_TIMEOUT_S) -> float:
    """How long to wait for a model of `model_gb` GB to load before giving up.

    llama-server reads the whole file (and runs a warm-up) before it says
    it's ready. From a hard drive or USB stick that's roughly 15 s per GB on
    a cold start, so a 60 GB model gets 15 minutes instead of a flat 3.
    """
    try:
        size = max(0.0, float(model_gb))
    except (TypeError, ValueError):
        size = 0.0
    return min(HEALTH_MAX_TIMEOUT_S, max(minimum, size * HEALTH_SECONDS_PER_GB))


_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def _file_size(path: Path) -> int:
    """A file's size in bytes (raises OSError if it can't be read).

    The one place model sizes are read, so tests can stand in a 40 GB model
    without writing one (NTFS really allocates "sparse" truncated files).
    """
    return os.stat(path).st_size


def _model_total_bytes(model: Path) -> int:
    """Size of the whole model: all its parts for a split model ("-00001-of-00003.gguf").

    llama-server is handed the first part and reads the rest itself, so the
    wait for it to load must allow for all of them.
    """
    model = Path(model)
    size = _file_size(model)
    m = _SHARD_RE.match(model.name)
    if m is None:
        return size
    total = 0
    for part in range(1, int(m.group("total")) + 1):
        sibling = model.with_name(f"{m.group('stem')}-{part:05d}-of-{m.group('total')}{model.name[m.end('total'):]}")
        with contextlib.suppress(OSError):
            total += _file_size(sibling)
    return max(size, total)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout))


def _server_error_text(raw: bytes) -> str:
    """Pull a readable message out of an error response body."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", "replace").strip()[:300] or "no details"
    if isinstance(data, dict):
        err = data.get("error", data.get("message"))
        if isinstance(err, dict):
            err = err.get("message") or err.get("type")
        if err:
            return str(err)[:300]
    return str(data)[:300]


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):  # OpenAI "content parts" style
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def _extract_answer(data: dict) -> tuple[str, Optional[str]]:
    """(answer, reasoning) from a /v1/chat/completions response.

    With ``--reasoning-format deepseek`` llama-server returns the thinking in
    ``message.reasoning_content``. If a model still leaks ``<think>`` tags into
    the answer, `split_reasoning` catches those too.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise BackendError("The local model server sent an answer without any text in it.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        message = {}
    answer, inline_reasoning = split_reasoning(_message_text(message))
    separate = message.get("reasoning_content") or message.get("reasoning")
    parts = [p.strip() for p in (separate, inline_reasoning) if isinstance(p, str) and p.strip()]
    return answer, ("\n\n".join(parts) or None)


# Chat-template switches that turn visible reasoning off: Qwen3 / SmolLM3 /
# Granite read "enable_thinking"; gpt-oss reads "reasoning_effort"; Seed-OSS
# thinks without limit unless "thinking_budget" is 0. Templates ignore
# switches they don't know.
_NO_THINKING = {"enable_thinking": False, "reasoning_effort": "low", "thinking_budget": 0}
# ...and for models whose template forces thinking on and ignores all of those
# (QwQ, DeepSeek-R1 distills, Qwen3 "Thinking" models...), the engine's own
# per-request thinking budget: 0 makes current llama.cpp close the thinking
# block straight away, so the answer comes first. Older builds ignore it.
_NO_THINKING_BODY = {"reasoning_budget_tokens": 0}


def _without_thinking(body: dict) -> dict:
    """`body` asking the model to answer straight away (every switch we know)."""
    changed = dict(body)
    changed["chat_template_kwargs"] = dict(_NO_THINKING)
    changed.update(_NO_THINKING_BODY)
    return changed


def _finish_reason(data: dict) -> Optional[str]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


def _join_reasoning(*chunks: Optional[str]) -> Optional[str]:
    parts = [c for c in chunks if c]
    return "\n\n".join(parts) or None


def _measured_speed(data: dict, elapsed: float) -> Optional[float]:
    timings = data.get("timings")
    if isinstance(timings, dict):
        speed = timings.get("predicted_per_second")
        if isinstance(speed, (int, float)) and speed > 0:
            return float(speed)
    usage = data.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, (int, float)) and tokens > 0 and elapsed > 0:
        return float(tokens) / elapsed
    return None


def _process_read_bytes(proc: Any) -> Optional[int]:
    """Bytes `proc` has read from disk (psutil; Linux and Windows), or None if unknown.

    Reading a memory-mapped model counts too: its pages are fetched from disk
    on the process's behalf. (macOS doesn't report this; there the game
    falls back to the log.)
    """
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int):
        return None
    try:
        import psutil

        counters = psutil.Process(pid).io_counters()
    except Exception:
        return None
    return int(getattr(counters, "read_bytes", 0) or 0) + int(getattr(counters, "read_chars", 0) or 0)


def _terminate(proc: Any) -> None:
    """Politely stop a process: terminate, wait up to 5 s, then kill."""
    with contextlib.suppress(Exception):
        if proc.poll() is not None:
            return  # already gone
        proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return
    with contextlib.suppress(Exception):
        proc.kill()
        proc.wait(timeout=STOP_TIMEOUT_S)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class LlamaServerBackend(LLMBackend):
    """A managed ``llama-server`` subprocess, installed and started automatically.

    Public attributes you can read after :meth:`prepare` (e.g. to save them
    for next time): ``server_exe``, ``variant``, ``model_path``, ``port``,
    ``cpu_only`` and ``log_path``.

    Test hooks (all optional): ``http`` (object with ``request(method, url,
    headers=, body=, timeout=)``), ``popen`` (like ``subprocess.Popen``),
    ``installer`` (like ``ensure_llama_server``), ``downloader`` (like
    ``download_gguf``), ``runner`` (like ``subprocess.run``, for the engine
    check after installing), ``sleep``/``clock``, ``health_timeout_s`` and
    ``log_dir``.
    """

    name = "llamacpp-server"

    def __init__(
        self,
        entry: Optional[ModelEntry] = None,
        *,
        specs: Optional[SystemSpecs] = None,
        model_path: Optional[Path] = None,
        quant: Optional[str] = None,
        n_ctx: int = 4096,
        server_exe: Optional[Path] = None,
        http: Any = None,
        popen: Optional[Callable[..., Any]] = None,
        port: Optional[int] = None,
        installer: Optional[Callable[..., tuple[Path, RuntimeVariant]]] = None,
        downloader: Optional[Callable[..., Path]] = None,
        runner: Optional[Callable[..., Any]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
        log_dir: Optional[Path] = None,
        io_probe: Optional[Callable[[Any], Optional[int]]] = None,
    ) -> None:
        self.entry = entry
        self.specs = specs
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.quant = quant
        self.n_ctx = int(n_ctx)
        self.server_exe: Optional[Path] = Path(server_exe) if server_exe else None
        self.variant: Optional[RuntimeVariant] = None
        self.port: Optional[int] = port
        self.cpu_only = False
        self.log_path: Optional[Path] = None
        self.tokens_per_s: Optional[float] = None  # measured by benchmark(); scales chat timeouts
        self.gpu_note: Optional[str] = None  # set when a GPU build ended up not using the GPU

        self._fixed_port = port
        self._http = http or UrllibHttp(use_proxy=False)  # localhost: never via a proxy
        self._popen = popen or subprocess.Popen
        self._installer = installer
        self._downloader = downloader
        # Runs `llama-server --version` after an install (like subprocess.run).
        # Tests that fake the engine process skip it unless they pass one.
        self._runner = runner if runner is not None else (subprocess.run if popen is None else None)
        # Bytes the engine process has read from disk so far (a sign it is busy
        # loading, even when its log is quiet). Tests with fake processes skip it.
        self._io_probe = io_probe if io_probe is not None else (_process_read_bytes if popen is None else None)
        self._saw_disk_progress = False  # did the engine read from disk during the last wait?
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._health_timeout_s = float(health_timeout_s)
        self._log_dir = Path(log_dir) if log_dir else None
        self._proc: Any = None
        self._log_fh: Any = None
        self._minimal_args = False
        self._atexit_registered = False
        self._api_key = secrets.token_urlsafe(24)  # only requests carrying it are answered
        self._owner_file: Optional[Path] = None
        self._health_timeout_fixed = health_timeout_s != HEALTH_TIMEOUT_S
        self._restarts = 0  # quiet restarts since the last answer that worked
        self._log_token = f"{os.getpid()}-{secrets.token_hex(3)}"  # this game's own log name, where needed
        self._last_log_growth: Optional[float] = None  # seconds after launch the log last grew (-1: never)
        self._cpu_mode_from_start = False  # the GPU build saw no graphics card and nothing else installed
        self._update_failure: Optional[str] = None  # why the newer engine a model needs can't run here

    # -- LLMBackend API ------------------------------------------------------

    @property
    def model_label(self) -> str:
        if self.entry is not None:
            quant = self.quant or self.entry.quant
            return f"{self.entry.display_name} ({quant})" if quant else self.entry.display_name
        if self.model_path is not None:
            return self.model_path.stem
        return "llama.cpp model"

    def is_available(self) -> tuple[bool, str]:
        """True unless this computer has no official llama.cpp build at all."""
        try:
            if self.server_exe is not None and self.server_exe.is_file():
                return True, f"Using your llama.cpp engine at {self.server_exe}."
            if not is_platform_supported(platform.system(), platform.machine()):
                return False, (
                    f"There's no official prebuilt llama.cpp engine for {platform.system()} on "
                    f"{platform.machine()}. Ollama or llama-cpp-python may still work."
                )
            # Too old a Linux / macOS, or every build already failed here for good:
            # say so now, before the engine and a multi-GB model are downloaded.
            problem = runtime_install.platform_problem(platform.system(), platform.machine())
            if problem is None and self.specs is not None:
                problem = runtime_install.engine_problem(self.specs)
            elif problem is None and runtime_install.CPU.name in runtime_install.unusable_reasons():
                problem = runtime_install.unusable_message(runtime_install.unusable_reasons())
            if problem:
                return False, problem
            runtimes = installed_runtimes()
            if runtimes:
                _exe, tag, variant = runtimes[0]
                return True, f"The llama.cpp engine is already installed ({tag}, {variant})."
            return True, (
                "The official llama.cpp engine will be downloaded automatically (one-time: about 40 MB; "
                "llama.cpp is MIT licensed. NVIDIA CUDA builds are bigger and also include NVIDIA's CUDA "
                "runtime, under NVIDIA's own terms)."
            )
        except Exception as exc:  # must never raise
            return False, f"Couldn't check for the llama.cpp engine ({exc})."

    def prepare(self, ui: UI, entry: Optional[ModelEntry] = None) -> None:
        """Install the engine, download the model, start the server, wait until ready."""
        if entry is not None:
            self.entry = entry
        self.close()  # calling prepare() again restarts cleanly
        self._cpu_mode_from_start = False
        self._reap_orphans(ui)
        self._ensure_engine(ui)
        self._check_gpu_devices(ui)
        model = self._ensure_model(ui)
        if not self._health_timeout_fixed:
            with contextlib.suppress(OSError):
                self._health_timeout_s = health_timeout_for(_model_total_bytes(model) / 1e9)
        ui.info("Starting the llama.cpp engine with your model...")
        try:
            self._start_with_fallbacks(ui, model)
        except BaseException:  # errors *and* Ctrl+C: never leave a half-started server behind
            self._stop_process()
            raise
        variant = self.variant or CUSTOM_VARIANT
        mode = "CPU mode" if self.cpu_only else f"{variant.display} build"
        ui.success(f"Your model is awake and ready! (running privately on your computer, {escape(mode)})")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 700,
        json_mode: bool = False,
        think: Optional[bool] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResult:
        """One chat completion via ``POST /v1/chat/completions``.

        ``think=False`` switches a thinking model's reasoning off (Qwen3's
        ``enable_thinking``, gpt-oss's ``reasoning_effort: low``); ``stop``
        ends the answer at any of those strings. The time limit grows with
        the answer's token budget on slow models (see :meth:`_chat_timeout`).
        """
        self._ensure_running()
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if think is False:
            body = _without_thinking(body)
        if stop:
            body["stop"] = [str(x) for x in stop][:4]  # the OpenAI API allows up to 4

        start = self._clock()
        data = self._post_chat(body, max_tokens)
        answer, reasoning = _extract_answer(data)
        if not answer:
            # Thinking models sometimes spend the whole token budget thinking and
            # never answer. Ask again with thinking switched off: the answer
            # itself is short, so the same budget is plenty (and a model that
            # ignores the switch can't make the wait twice as long).
            self._notice(RETRY_WITHOUT_THINKING_NOTICE)
            retry = _without_thinking(body)
            data = self._post_chat(retry, max_tokens)
            answer, second_reasoning = _extract_answer(data)
            reasoning = _join_reasoning(reasoning, second_reasoning)
        self._restarts = 0  # it answered: this setup works
        elapsed = self._clock() - start
        return LLMResult(
            text=answer,
            reasoning=reasoning,
            model=self.model_label,
            backend=self.name,
            elapsed_s=elapsed,
            messages=[dict(m) for m in messages],
            raw=data,
            truncated=_finish_reason(data) == "length",
        )

    def _post_chat(self, body: dict, max_tokens: int) -> dict:
        """POST a chat request; if the engine crashes on it while using the graphics
        card, switch to a safer setup (CPU mode) and ask once more."""
        try:
            return self._post_json("/v1/chat/completions", body, self._chat_timeout(max_tokens))
        except BackendError as exc:
            kind = self._crash_kind(exc)
            if kind is None or not self._switch_after_crash(None, kind):
                raise
            return self._post_json("/v1/chat/completions", body, self._chat_timeout(max_tokens))

    def _chat_timeout(self, max_tokens: int) -> float:
        """Seconds to wait for one answer: at least CHAT_TIMEOUT_S, more for slow models.

        A non-streamed answer arrives all at once, so the wait must cover the
        whole generation: 1.5 x (tokens / measured speed) plus time to read
        the prompt. Before the speed is measured, CHAT_TIMEOUT_S applies.
        """
        tps = self.tokens_per_s
        if not tps or tps <= 0:
            return CHAT_TIMEOUT_S
        needed = CHAT_PROMPT_ALLOWANCE_S + 1.5 * max(1, int(max_tokens)) / tps
        return float(min(CHAT_TIMEOUT_MAX_S, max(CHAT_TIMEOUT_S, needed)))

    def benchmark(self, ui: Optional[UI] = None) -> Optional[float]:
        """Tokens per second from a short test generation (None if it fails).

        Prefers the server's own measurement (``timings.predicted_per_second``);
        otherwise divides the generated tokens by the wall-clock time.
        """
        if self._proc is None:
            return None
        body = {
            "messages": [
                {"role": "system", "content": "Background notes (ignore them): " + BENCHMARK_CONTEXT},
                {"role": "user", "content": BENCHMARK_PROMPT},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = _without_thinking(body)  # timing the answer, not a model's thinking
        switched = 0
        while True:
            status = ui.status("Timing a quick test sentence to see how fast your model talks...") if ui else contextlib.nullcontext()
            try:
                with status:
                    self._ensure_running()
                    start = self._clock()
                    data = self._post_json("/v1/chat/completions", body, CHAT_TIMEOUT_S)
                    elapsed = self._clock() - start
                break
            except BackendError as exc:
                # The engine started, but broke on its first real piece of work (a
                # graphics-driver crash, running out of GPU memory...). Treat that
                # like a start-up failure: move to the next build or to CPU mode.
                kind = self._crash_kind(exc)
                if kind is None:
                    return None  # it's running, we just couldn't time it (e.g. very slow)
                if switched < 3 and self._switch_after_crash(ui, kind):
                    switched += 1
                    continue
                if self._proc is not None and self._proc.poll() is None:
                    return None  # an error reply, but the engine is still running: the game can retry
                self._stop_process()
                raise EngineStopped(
                    "The model started, but the llama.cpp engine stopped as soon as it was given real work. "
                    + self._explain_failure(kind, None, self._read_log_tail())
                ) from exc
        self._restarts = 0
        speed = _measured_speed(data, elapsed)
        if speed is not None:
            self.tokens_per_s = speed
        return speed

    def close(self) -> None:
        """Stop the server process. Safe to call any number of times."""
        self._stop_process()
        if self._atexit_registered:
            with contextlib.suppress(Exception):
                atexit.unregister(self.close)
            self._atexit_registered = False

    # -- preparing -------------------------------------------------------------

    def _get_specs(self) -> SystemSpecs:
        """The hardware snapshot (detected lazily, only if we need it)."""
        if self.specs is None:
            try:
                from ..specs import detect_specs

                try:
                    self.specs = detect_specs(benchmark=False)
                except TypeError:
                    self.specs = detect_specs()
            except Exception:
                self.specs = SystemSpecs(
                    os_name=platform.system(),
                    os_version=platform.release(),
                    arch=platform.machine(),
                    cpu_name="Unknown CPU",
                    cpu_cores_physical=None,
                    cpu_cores_logical=os.cpu_count(),
                    ram_total_gb=0.0,
                    ram_available_gb=0.0,
                    disk_free_gb=0.0,
                )
        return self.specs

    def _install(self, ui: UI, variant: Optional[RuntimeVariant], *, update: bool = False) -> tuple[Path, RuntimeVariant]:
        installer = self._installer or runtime_install.ensure_llama_server
        try:
            if update:
                exe, chosen = installer(ui, self._get_specs(), variant=variant, update=True)
            else:
                exe, chosen = installer(ui, self._get_specs(), variant=variant)
        except RuntimeInstallError as exc:
            raise BackendError(str(exc)) from exc
        except OSError as exc:  # a locked or full disk: explain, and let the next build be tried
            raise BackendError(f"Something went wrong on your disk while installing the engine ({exc}). "
                               "Please try again.") from exc
        return Path(exe), chosen

    def _ensure_engine(self, ui: UI) -> None:
        if self.server_exe is not None:
            if self.server_exe.is_file():
                info = install_info(self.server_exe) or {}
                self.variant = variant_by_name(info.get("variant")) or CUSTOM_VARIANT
                ui.info(f"Using the llama.cpp engine you already have ({escape(self.variant.display)}).")
                return
            ui.warn("The llama.cpp engine from last time has gone missing - no worries, I'll fetch a fresh copy.")
            self.server_exe = None
        self.server_exe, self.variant = self._install(ui, None)
        self._check_engine_starts(ui)

    # Engine-check results that mean "this build can never run here".
    _CHECK_FAILURES = ("glibc", "cpu_unsupported", "cant_execute", "missing_library")

    def _check_engine_starts(self, ui: UI) -> None:
        """Run ``llama-server --version`` straight after installing - *before* the
        (much bigger) model download - so a build that can't run on this computer
        (too old a Linux, a missing library, the wrong kind of processor...) is
        found now, and the next build is tried instead."""
        remaining: Optional[list[RuntimeVariant]] = None
        while True:
            assert self.server_exe is not None
            failure = self._probe_engine(self.server_exe)
            if failure is None:
                return
            kind, text = failure
            variant = self.variant or CUSTOM_VARIANT
            if kind in runtime_install.PERMANENT_FAILURES and self._installed_exe(self.server_exe):
                runtime_install.mark_unusable(self.server_exe, kind)  # never downloaded again
            if remaining is None:
                remaining = self._fallback_variants(variant) if variant.name != CPU.name else []
            nxt = self._next_build(ui, remaining, variant, kind) if remaining else None
            if nxt is None:
                self._show_log_tail(ui, text)
                raise BackendError(self._explain_failure(kind, None, text))
            self.server_exe, self.variant = nxt

    def _check_gpu_devices(self, ui: UI) -> None:
        """Before loading the model: can this graphics-card build see a graphics card?

        ``llama-server --list-devices`` lists the devices the build can use. An
        empty list means it would quietly run everything on the processor (no
        Vulkan driver, a CUDA library that won't load...), so the next build
        is tried now - and if none can be installed, this one runs in CPU mode,
        so its speed is never mistaken for graphics-card speed.
        """
        remaining: Optional[list[RuntimeVariant]] = None
        while self.server_exe is not None:
            variant = self.variant or CUSTOM_VARIANT
            if not variant.gpu or variant.name == CUSTOM_VARIANT.name:
                return
            devices = self._probe_devices(self.server_exe)
            if devices is None or devices:
                return  # it sees a card (or we can't tell): carry on
            if remaining is None:
                remaining = self._fallback_variants(variant)
            nxt = self._next_build(ui, remaining, variant, "no_device") if remaining else None
            if nxt is None:
                ui.info("Running this build in CPU mode instead: a bit slower, but it works on every computer.")
                self._cpu_mode_from_start = True
                self.gpu_note = "driver"
                return
            self.server_exe, self.variant = nxt
            if not nxt[1].gpu:
                self.gpu_note = "driver"  # no graphics build could see the card

    def _probe_devices(self, exe: Path) -> Optional[list[str]]:
        """The graphics devices ``llama-server --list-devices`` reports, or None if unknown."""
        if self._runner is None:
            return None
        try:
            result = self._runner(
                [str(exe), "--list-devices"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=server_env(exe), cwd=str(Path(exe).parent),
                timeout=ENGINE_CHECK_TIMEOUT_S,
                **({"creationflags": _CREATE_NO_WINDOW} if platform.system() == "Windows" else {}),
            )
        except Exception:
            return None
        if getattr(result, "returncode", 0):
            return None  # an old build that doesn't know the flag, or a crash: let the real start decide
        raw = getattr(result, "stdout", b"") or b""
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        return gpu_devices_from_listing(text)

    def _probe_engine(self, exe: Path) -> Optional[tuple[str, str]]:
        """(kind, output) if the engine can't even print its version here, else None.

        Only clear-cut "can never run here" problems count; anything unclear
        (a slow antivirus scan timing out, an odd exit code) is left for the
        real start-up, which has its own fallbacks.
        """
        if self._runner is None:
            return None
        try:
            result = self._runner(
                [str(exe), "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=server_env(exe), cwd=str(Path(exe).parent), timeout=ENGINE_CHECK_TIMEOUT_S,
                **({"creationflags": _CREATE_NO_WINDOW} if platform.system() == "Windows" else {}),
            )
        except subprocess.TimeoutExpired:
            return None
        except OSError as exc:
            # The file is there but can't be run at all: ENOENT for an existing
            # file means its loader is missing (a 32-bit system reporting a
            # 64-bit processor), ENOEXEC a different kind of processor.
            if getattr(exc, "errno", None) in (2, 8) or isinstance(exc, (FileNotFoundError,)):
                return "cant_execute", str(exc)
            return None
        except Exception:
            return None
        raw = getattr(result, "stdout", b"") or b""
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        code = getattr(result, "returncode", 0)
        if not code:
            return None
        kind = classify_server_log(text, code)
        return (kind, text) if kind in self._CHECK_FAILURES else None

    def _ensure_model(self, ui: UI) -> Path:
        if self.model_path is not None and self.model_path.is_file():
            ui.info(f"Using your downloaded model: {escape(self.model_path.name)}")
            return self.model_path
        if self.entry is None:
            if self.model_path is not None:
                raise BackendError(f"I can't find the model file {self.model_path} any more. Pick a model again and I'll download it.")
            raise BackendError("No model has been chosen yet, so there's nothing to start.")
        downloader = self._downloader
        if downloader is None:
            # Imported here, not at the top: it's only needed the first time.
            from ..download import download_gguf

            downloader = download_gguf
        try:
            path = Path(downloader(self.entry, ui, quant=self.quant))
        except (UserQuit, BackendError):
            raise
        except Exception as exc:  # e.g. download.DownloadError - already friendly
            raise BackendError(f"The model download didn't work: {exc}") from exc
        self.model_path = path
        return path

    # -- starting, with automatic fallbacks ------------------------------------

    def _fallback_variants(self, current: RuntimeVariant) -> list[RuntimeVariant]:
        """Builds to try after `current` failed, ending with CPU."""
        plan = plan_variants(self._get_specs())
        names = [v.name for v in plan]
        if current.name in names:
            rest = plan[names.index(current.name) + 1 :]
        else:
            rest = [v for v in plan if v.name != current.name]
        if current.name != CPU.name and all(v.name != CPU.name for v in rest):
            rest.append(CPU)
        return rest

    def _next_build(self, ui: UI, remaining: list[RuntimeVariant], failed: RuntimeVariant, kind: str) -> Optional[tuple[Path, RuntimeVariant]]:
        """Install the next build in `remaining` that installs successfully."""
        why = {
            "gpu": "it looks like a graphics-driver hiccup",
            "glibc": "that build needs a newer Linux than this one",
            "missing_library": "it needs a system library that isn't installed",
            "cant_execute": "this computer can't run that kind of program",
            "cpu_unsupported": "your processor is missing an instruction it needs",
            "gpu_hang": "the graphics card never finished getting ready",
            "no_device": "it couldn't find your graphics card - updating the graphics driver usually fixes this",
            "gpu_arch": "that build doesn't support your graphics card's generation",
            "memory": "the graphics card ran out of memory",
            "crash": "it stopped as soon as it was given real work",
        }.get(kind, "it stopped while starting up")
        ui.warn(f"The {escape(failed.display)} build couldn't get going on this computer ({why}). No problem - trying the next option.")
        while remaining:
            nxt = remaining.pop(0)
            if nxt.gpu:
                ui.info(f"Trying the {escape(nxt.display)} build...")
            else:
                ui.info("Switching to CPU mode: a bit slower, but it works on every computer.")
            try:
                return self._install(ui, nxt)
            except BackendError as exc:
                ui.warn(f"Couldn't set up the {escape(nxt.display)} build: {escape(str(exc))}")
        return None

    def _start_with_fallbacks(self, ui: UI, model: Path) -> None:
        if self.server_exe is None:
            raise BackendError("The llama.cpp engine isn't installed, so the model can't start.")
        exe, variant = self.server_exe, self.variant or CUSTOM_VARIANT
        cpu_only = not variant.gpu or self._cpu_mode_from_start
        minimal = False
        port_retries = 0
        remaining: Optional[list[RuntimeVariant]] = None
        tried: set[tuple[str, bool, bool, int]] = set()
        update_note: Optional[str] = None  # set once we've tried a newer engine for this model

        while True:
            attempt = (str(exe), cpu_only, minimal, port_retries)
            if attempt in tried:  # safety net: never loop forever
                raise BackendError(self._explain_failure("unknown", None))
            tried.add(attempt)

            outcome, returncode = self._launch_and_wait(ui, exe, model, cpu_only=cpu_only, minimal=minimal)
            while (outcome == "timeout" and variant.gpu and not cpu_only and self._looks_stuck_on_gpu()
                   and self._keep_waiting(ui, variant)):
                outcome, returncode = self._wait_until_healthy(ui)  # same engine, another full wait
            if outcome == "ok":
                self.server_exe, self.variant, self.cpu_only, self._minimal_args = exe, variant, cpu_only, minimal
                if variant.gpu and not cpu_only:
                    self._check_gpu_really_used(ui)
                self._tidy_old_engines(ui)
                return
            if outcome == "timeout" and variant.gpu and not cpu_only and self._looks_stuck_on_gpu():
                # The graphics-card build went quiet before it even began loading the
                # model (e.g. a driver stuck creating the device). A smaller model
                # would hang the same way: try the next build / CPU mode instead.
                self._stop_process()
                if remaining is None:
                    remaining = self._fallback_variants(variant)
                nxt = self._next_build(ui, remaining, variant, "gpu_hang")
                if nxt is not None:
                    exe, variant = nxt
                    cpu_only, minimal = not variant.gpu, False
                    continue
                ui.info("Running this build in CPU mode instead: a bit slower, but it works on every computer.")
                cpu_only, minimal = True, False
                continue
            if outcome == "timeout":
                self._stop_process()
                raise BackendError(
                    f"The model is taking too long to wake up (over {int(self._health_timeout_s)} seconds). "
                    "Big models load slowly from a hard drive or USB stick; trying again often goes faster "
                    "because part of the file is already in memory. If it keeps happening, your computer may be "
                    f"short on memory - a smaller model should help. The engine's log is at {self.log_path}."
                )

            tail = self._read_log_tail()
            kind = classify_server_log(tail, returncode)
            if kind in runtime_install.PERMANENT_FAILURES and exe == self._installed_exe(exe):
                runtime_install.mark_unusable(exe, kind)  # so automatic setup won't pick it again
                if update_note == "updated":
                    # It's the newer engine fetched for this model that can't run here;
                    # only that install is marked, the older one still works.
                    self._update_failure = runtime_install.newer_engine_unusable_message(
                        variant, str((install_info(exe) or {}).get("tag") or ""), kind)
                    update_note = "cant_run_newer"
            if kind == "model_unsupported" and update_note is None:
                newer, update_note = self._newer_engine(ui, exe, variant, unsupported_architecture(tail))
                if newer is not None:
                    exe, variant = newer
                    cpu_only, minimal = cpu_only or not variant.gpu, False
                    continue
            if kind == "bad_args" and not minimal:
                ui.info("This llama.cpp build didn't recognise one of our optional settings - restarting with just the basics.")
                minimal = True
                continue
            if kind == "port" and self._fixed_port is None and port_retries < 2:
                ui.info("That network port was already busy - picking another one.")
                port_retries += 1
                continue
            running_on_gpu = variant.gpu and not cpu_only
            # "glibc": the CUDA builds need a newer Linux than the Vulkan/CPU ones.
            if running_on_gpu and kind in ("gpu", "gpu_arch", "unknown", "missing_library", "cpu_unsupported",
                                           "glibc"):
                if remaining is None:
                    remaining = self._fallback_variants(variant)
                nxt = self._next_build(ui, remaining, variant, kind)
                if nxt is not None:
                    exe, variant = nxt
                    cpu_only, minimal = not variant.gpu, False
                    continue
                # Nothing else could be installed (offline?): run this same build without the GPU.
                ui.info("Running this build in CPU mode instead: a bit slower, but it works on every computer.")
                cpu_only, minimal = True, False
                continue
            self._show_log_tail(ui, tail)
            raise BackendError(self._explain_failure(kind, returncode, tail, update_note))

    def _keep_waiting(self, ui: UI, variant: RuntimeVariant) -> bool:
        """A graphics-card build showed no sign of loading the model before the time
        limit. Ask before switching builds (the switch is remembered for next time)."""
        minutes = max(1, round(self._health_timeout_s / 60))
        ui.warn(f"The {escape(variant.display)} build has been getting ready for about {minutes} "
                f"minute{'s' if minutes != 1 else ''} and hasn't started loading the model yet - the graphics "
                "card may be stuck.")
        choice = ui.choose(
            "What shall I do?",
            [("switch", "Try the next engine build (recommended)"),
             ("wait", "Keep waiting - a big model on a slow disk or USB stick can take a while")],
            default="switch",
            aliases={"next": "switch", "try": "switch", "y": "switch", "yes": "switch", "n": "wait", "no": "wait",
                     "keep": "wait"},
        )
        return choice == "wait"

    def _tidy_old_engines(self, ui: UI) -> None:
        """Now that this engine works, delete copies it replaced (an older version of
        the same build, the files of builds that can't run here). Never raises."""
        if self.server_exe is None or not self._installed_exe(self.server_exe):
            return
        try:
            tidied, freed = runtime_install.prune_old_installs(self.server_exe, in_use=engines_in_use(self._log_folder()))
        except Exception:
            return
        if tidied and freed >= 1_000_000:
            ui.info(f"Tidied away {tidied} engine cop{'y' if tidied == 1 else 'ies'} this computer no longer needs "
                    f"({freed / 1e6:,.0f} MB freed).")

    @staticmethod
    def _installed_exe(exe: Path) -> Optional[Path]:
        """`exe` if it belongs to a build the game installed (has install.json), else None."""
        return exe if install_info(exe) else None

    def _newer_engine(
        self, ui: UI, exe: Path, variant: RuntimeVariant, architecture: Optional[str]
    ) -> tuple[Optional[tuple[Path, RuntimeVariant]], str]:
        """The model needs a newer engine: fetch the newest build of the same kind.

        Returns ``((exe, variant) or None, note)``, where the note explains
        what happened for the final error message if it still doesn't work.
        """
        design = f" ('{escape(architecture)}')" if architecture else ""
        if variant.name == CUSTOM_VARIANT.name or install_info(exe) is None:
            return None, "custom"  # the player's own llama-server: we don't replace it
        ui.info(
            f"This model uses a newer design{design} than your llama.cpp engine knows. No need to download "
            "the model again - let me fetch the newest engine instead..."
        )
        try:
            new_exe, new_variant = self._install(ui, variant, update=True)
        except BackendError as exc:
            if "newer llama.cpp engine than the one you have" in str(exc):
                self._update_failure = str(exc)  # the newer build is known not to run here
                return None, "cant_run_newer"
            ui.warn(f"I couldn't get a newer engine: {escape(str(exc))}")
            return None, "offline"
        if Path(new_exe).resolve() == Path(exe).resolve():
            return None, "newest"
        # Check the new engine starts at all before loading the model with it. If
        # it can't, only *that* install is marked - the current one still works
        # for every other model.
        failure = self._probe_engine(Path(new_exe))
        if failure is not None:
            kind, _text = failure
            if kind in runtime_install.PERMANENT_FAILURES and self._installed_exe(Path(new_exe)):
                runtime_install.mark_unusable(Path(new_exe), kind)
            info = install_info(Path(new_exe)) or {}
            self._update_failure = runtime_install.newer_engine_unusable_message(
                new_variant, str(info.get("tag") or ""), kind)
            return None, "cant_run_newer"
        return (new_exe, new_variant), "updated"

    def _check_gpu_really_used(self, ui: UI) -> None:
        """After a GPU build started: is the graphics card actually doing the work?

        A GPU build whose driver can't be reached doesn't crash - llama.cpp
        quietly runs everything on the processor. We read its start-up log;
        if no layers went to the GPU, we say so (with a driver hint) and treat
        this run as CPU mode, so its speed isn't mistaken for GPU speed.
        """
        text = self._read_log(LOG_SCAN_BYTES)
        used = gpu_offload_from_log(text)
        if used is not False:
            return
        counts = [int(m.group(1)) for m in _OFFLOAD_RE.finditer(text)]
        if counts and all(c == 0 for c in counts):  # it saw the card, but chose not to use it
            self.gpu_note = "too_big"
            ui.warn(
                "The model is too big for your graphics card's free memory, so it's running on the processor "
                "instead (slower, but it works). A smaller model would fit on the graphics card."
            )
        else:
            self.gpu_note = "driver"
            ui.warn(
                f"Heads-up: the {escape((self.variant or CUSTOM_VARIANT).display)} build started, but your graphics "
                "card isn't doing any of the work - the engine couldn't talk to its driver - so the model runs on "
                "the processor. Updating your graphics driver usually fixes this."
            )
        self.cpu_only = True

    def _launch_and_wait(self, ui: Optional[UI], exe: Path, model: Path, *, cpu_only: bool, minimal: bool) -> tuple[str, Optional[int]]:
        self._stop_process()
        self._launch(exe, model, cpu_only=cpu_only, minimal=minimal)
        return self._wait_until_healthy(ui)

    def _log_folder(self) -> Path:
        return self._log_dir or (config.runtime_dir() / "logs")

    def _open_log(self, log_dir: Path, port: int) -> None:
        """Open this launch's log file.

        Normally ``llama-server.log``. If another copy of the game is using
        that file right now (we hold a lock on it while our engine runs), we
        use ``llama-server-<port>.log`` instead, so two games never mix or
        truncate each other's logs.
        """
        self._close_log()
        if not _posix_locks_available():
            # Windows has no advisory lock we can rely on (Python opens files
            # there with "share read/write", so another copy of the game could
            # still open and truncate ours). Each game uses its own log instead.
            self.log_path = log_dir / f"llama-server-{self._log_token}.log"
            self._log_fh = open(self.log_path, "wb")
            return
        main = log_dir / "llama-server.log"
        try:
            fh = open(main, "ab")  # append mode: never truncate a file another game may be using
            if not _lock_exclusively(fh):
                fh.close()
                raise OSError("in use by another copy of the game")
            fh.truncate(0)
            self.log_path, self._log_fh = main, fh
            return
        except OSError:
            pass
        self.log_path = log_dir / f"llama-server-{port}.log"
        self._log_fh = open(self.log_path, "wb")

    def _launch(self, exe: Path, model: Path, *, cpu_only: bool, minimal: bool) -> None:
        exe, model = Path(exe).resolve(), Path(model).resolve()
        port = self._fixed_port or find_free_port()
        self.port = port
        args = build_server_args(exe, model, port=port, n_ctx=self.n_ctx, cpu_only=cpu_only, minimal=minimal)

        log_dir = self._log_folder()
        log_dir.mkdir(parents=True, exist_ok=True)
        _tidy_old_logs(log_dir)
        self._open_log(log_dir, port)

        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": self._log_fh,
            "stderr": subprocess.STDOUT,
            "env": server_env(exe, api_key=self._api_key),
            "cwd": str(exe.parent),
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        else:
            # Its own session (process group): Ctrl+C in the terminal interrupts
            # the game, which then stops the engine cleanly, instead of killing
            # the engine mid-game behind the game's back.
            kwargs["start_new_session"] = True
            death_signal = _parent_death_signal_hook()
            if death_signal is not None:
                kwargs["preexec_fn"] = death_signal
        try:
            self._proc = self._popen(args, **kwargs)
        except PermissionError as exc:
            self._close_log()
            raise BackendError(
                "Your computer wouldn't let the llama.cpp engine start (permission denied). Some systems "
                "block programs in that folder - setting GETTOWORK_HOME to another folder usually fixes it."
            ) from exc
        except OSError as exc:
            self._close_log()
            raise BackendError(f"The llama.cpp engine couldn't be started ({exc}).") from exc
        if platform.system() == "Windows":
            _assign_to_kill_on_close_job(self._proc)
        self._owner_file = _record_owner(log_dir, self._proc, exe)
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def _probe_health(self) -> str:
        """``"ok"`` (ready), ``"loading"`` (503 / not yet) or ``"down"`` (no answer)."""
        try:
            resp = self._http.request(
                "GET",
                f"http://127.0.0.1:{self.port}/health",
                headers={"Accept": "application/json"},
                timeout=HEALTH_REQUEST_TIMEOUT_S,
            )
            with resp:
                status = resp.status
                raw = resp.read()
        except NETWORK_ERRORS:
            return "down"  # not listening yet
        if status != 200:
            return "loading"  # 503 {"error": {"message": "Loading model"}}
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            data = {}
        if isinstance(data, dict) and "loading" in str(data.get("status", "")).lower():
            return "loading"  # very old builds answered 200 {"status": "loading model"}
        return "ok"

    def _wait_until_healthy(self, ui: Optional[UI]) -> tuple[str, Optional[int]]:
        """Poll ``/health`` until the model is loaded, the engine dies, or we give up.

        We give up after the size-based timeout - but not while the engine's
        log is still growing (it's busy loading), up to HEALTH_MAX_TIMEOUT_S.
        """
        start = self._clock()
        deadline = start + self._health_timeout_s
        hard_stop = start + max(self._health_timeout_s, HEALTH_MAX_TIMEOUT_S)
        last_size, last_growth = self._log_size(), float("-inf")  # -inf: no progress seen yet
        last_read = self._read_bytes()
        self._last_log_growth = None
        self._saw_disk_progress = False
        spinner = ui.status(WAKE_UP_MESSAGE) if ui is not None else contextlib.nullcontext()
        with spinner:
            while True:
                returncode = self._proc.poll()
                if returncode is not None:
                    return "died", returncode
                if self._probe_health() == "ok":
                    return "ok", None
                now = self._clock()
                size = self._log_size()
                if size > last_size:
                    last_size, last_growth = size, now
                    self._last_log_growth = now - start
                # Current engines log nothing while they read the model file, so
                # the engine reading from disk counts as "still working" too.
                read = self._read_bytes()
                if read is not None and last_read is not None and read - last_read >= DISK_PROGRESS_BYTES:
                    last_read, last_growth = read, now
                    self._saw_disk_progress = True
                elif last_read is None:
                    last_read = read
                if now >= deadline and (now - last_growth >= HEALTH_GRACE_S or now >= hard_stop):
                    return "timeout", None
                self._sleep(HEALTH_POLL_S)

    def _read_bytes(self) -> Optional[int]:
        """How much the engine process has read from disk so far (None if unknown)."""
        if self._io_probe is None or self._proc is None:
            return None
        try:
            value = self._io_probe(self._proc)
        except Exception:
            return None
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def _log_size(self) -> int:
        try:
            return os.path.getsize(self.log_path) if self.log_path else 0
        except OSError:
            return 0

    # -- running -----------------------------------------------------------------

    def _ensure_running(self) -> None:
        if self._proc is None or self.server_exe is None or self.model_path is None:
            raise BackendError("The local model isn't running yet - it needs to be started first.")
        returncode = self._proc.poll()
        if returncode is None:
            return
        # It stopped since the last message. The first time, restart the same
        # setup; if it keeps stopping on the graphics card, switch to CPU mode
        # rather than reloading the same crash over and over.
        kind = classify_server_log(self._read_log_tail(), returncode)
        if self._restarts >= 1 and self._switch_after_crash(None, kind if kind != "unknown" else "crash"):
            return
        self._restarts += 1
        self._notice("The model stopped unexpectedly - restarting it (this can take a moment)…")
        outcome, _ = self._launch_and_wait(None, self.server_exe, self.model_path, cpu_only=self.cpu_only, minimal=self._minimal_args)
        if outcome != "ok":
            if self._switch_after_crash(None, "crash"):
                return
            self._stop_process()
            raise BackendError(f"The local model stopped unexpectedly and wouldn't restart. The engine's log is at {self.log_path}.")

    def _crash_kind(self, exc: BaseException) -> Optional[str]:
        """After a failed request: why the engine broke ("gpu", "memory", "crash"...),
        or None if it didn't (a timeout on a slow computer, a bad request...)."""
        proc = self._proc
        returncode = proc.poll() if proc is not None else None
        status = getattr(exc, "status", None)
        if proc is not None and returncode is None and not (isinstance(status, int) and status >= 500):
            return None  # still running and no server-side failure: not a crash
        text = self._read_log_tail() + "\n" + str(getattr(exc, "server_text", "") or "")
        kind = classify_server_log(text, returncode)
        return "crash" if kind == "unknown" else kind

    def _switch_after_crash(self, ui: Optional[UI], kind: str) -> bool:
        """The engine broke after it had started. If it was using the graphics card,
        move to a setup that avoids it - the next build (only while setting up,
        with `ui`) or this same build in CPU mode - and start it. True if the new
        setup is up and running; False if there's nothing safer to switch to."""
        variant = self.variant or CUSTOM_VARIANT
        if not (variant.gpu and not self.cpu_only) or kind not in ("gpu", "gpu_arch", "memory", "crash", "unknown"):
            return False
        if self.server_exe is None or self.model_path is None:
            return False
        options: list[tuple[Path, RuntimeVariant, bool]] = []
        if ui is not None:
            remaining = self._fallback_variants(variant)
            nxt = self._next_build(ui, remaining, variant, kind) if remaining else None
            if nxt is not None:
                options.append((nxt[0], nxt[1], not nxt[1].gpu))
        else:
            self._notice("The graphics card stumbled - switching the model to CPU mode (slower, but steadier)…")
        options.append((self.server_exe, variant, True))  # this build, without the graphics card
        for exe, option, cpu_only in options:
            if ui is not None and cpu_only and option is variant:
                ui.info("Running this build in CPU mode instead: a bit slower, but it works on every computer.")
            outcome, _ = self._launch_and_wait(ui, exe, self.model_path, cpu_only=cpu_only, minimal=self._minimal_args)
            if outcome == "ok":
                self.server_exe, self.variant, self.cpu_only = exe, option, cpu_only
                self.tokens_per_s = None  # the old speed no longer applies
                self._restarts = 0
                return True
        self._stop_process()
        return False

    def _looks_stuck_on_gpu(self) -> bool:
        """After a start-up timeout: no sign at all that the model was loading?

        Current llama.cpp builds write nothing to their log between "loading
        model" and "model loaded", so a quiet log alone proves nothing: the
        engine reading the model file from disk also counts as progress.
        """
        if self._saw_disk_progress:
            return False
        return not _LOAD_PROGRESS_RE.search(self._read_log(LOG_SCAN_BYTES))

    def _post_json(self, path: str, body: dict, timeout: float) -> dict:
        url = f"http://127.0.0.1:{self.port}{path}"
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            resp = self._http.request("POST", url, headers=headers, body=payload, timeout=timeout)
            with resp:
                status = resp.status
                raw = resp.read()
        except NETWORK_ERRORS as exc:
            if _is_timeout(exc):
                raise BackendError(
                    f"The model took longer than {int(timeout)} seconds to answer. It may be too big for "
                    "this computer - a smaller model would be much snappier."
                ) from exc
            raise BackendError(
                "I lost contact with the local model server (it may have crashed). "
                f"The engine's log is at {self.log_path}."
            ) from exc
        if status != 200:
            error = BackendError(f"The local model server reported a problem (HTTP {status}): {_server_error_text(raw)}")
            error.status = status  # type: ignore[attr-defined]
            error.server_text = _server_error_text(raw)  # type: ignore[attr-defined]
            raise error
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BackendError("The local model server sent back something that wasn't valid JSON.") from exc
        if not isinstance(data, dict):
            raise BackendError("The local model server sent back an unexpected answer.")
        return data

    # -- process + log housekeeping ----------------------------------------------

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            _terminate(proc)
        self._close_log()
        owner, self._owner_file = self._owner_file, None
        if owner is not None:
            with contextlib.suppress(OSError):
                owner.unlink()

    def _reap_orphans(self, ui: UI) -> None:
        """Stop engines left running by an earlier game that was force-quit."""
        try:
            stopped = reap_orphaned_servers(self._log_folder())
        except Exception:
            return
        if stopped:
            ui.info(
                "I found a llama.cpp engine still running from an earlier game (it was holding memory) "
                "and stopped it."
            )

    def _close_log(self) -> None:
        fh, self._log_fh = self._log_fh, None
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()

    def _read_log_tail(self) -> str:
        return self._read_log(LOG_TAIL_BYTES)

    def _read_log(self, max_bytes: int) -> str:
        """The last `max_bytes` of this launch's log (text; "" if unreadable)."""
        if self.log_path is None:
            return ""
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - max_bytes))
                data = fh.read()
        except OSError:
            return ""
        return data.replace(b"\x00", b"").decode("utf-8", "replace")

    def _show_log_tail(self, ui: UI, tail: str) -> None:
        lines = [line for line in safe_text(tail).strip().splitlines() if line.strip()][-LOG_TAIL_LINES:]
        if lines:
            ui.say("Here's what the engine said just before it stopped:", style="dim")
            for line in lines:
                ui.say("  " + escape(line), style="dim")

    def _explain_failure(
        self, kind: str, returncode: Optional[int], tail: str = "", update_note: Optional[str] = None
    ) -> str:
        where = f" The full log is at {self.log_path}." if self.log_path else ""
        design = unsupported_architecture(tail)
        design_text = f" (it's called '{design}')" if design else ""
        unsupported = {
            "newest": (
                f"This model uses a design{design_text} that even the newest llama.cpp engine doesn't support "
                "yet. The file itself is fine - downloading it again won't help. Please pick another model."
            ),
            "offline": (
                f"This model uses a design{design_text} that your llama.cpp engine is too old for, and I couldn't "
                "fetch a newer engine just now. The file itself is fine - try again when you're online, or pick "
                "another model."
            ),
            "custom": (
                f"This model uses a design{design_text} that your own llama-server doesn't know. The file is fine: "
                "a newer llama.cpp build would run it - or pick another model."
            ),
        }
        if update_note == "cant_run_newer" and self._update_failure:
            return self._update_failure + where  # the newer engine this model needs is what can't run
        if kind == "model_unsupported":
            return unsupported.get(update_note or "newest", unsupported["newest"]) + where
        if kind == "model" and not _CORRUPT_RE.search(tail or ""):
            model_message = (
                "llama.cpp couldn't load this model file. If its download was interrupted, deleting it and "
                "downloading again helps; otherwise another model is the way to go."
            )
        else:
            model_message = (
                "llama.cpp couldn't read the model file - it looks incomplete or damaged. "
                "Deleting it and downloading again (or picking another model) usually fixes this."
            )
        messages = {
            "memory": (
                "Your computer ran out of memory while loading the model. Try closing other apps, "
                "or pick a smaller model."
            ),
            "model": model_message,
            "glibc": (
                "This Linux system is a bit older than the prebuilt llama.cpp engine needs (it wants a newer "
                "glibc). Ollama (https://ollama.com/download) or `pip install llama-cpp-python` should still work."
            ),
            "missing_library": missing_library_hint(tail),
            "gpu_arch": (
                "This llama.cpp build doesn't support your graphics card's generation (the newest CUDA "
                "builds leave out older cards)."
            ),
            "cpu_unsupported": (
                "Your processor is missing an instruction the llama.cpp engine needs. "
                "Ollama or llama-cpp-python may still work."
            ),
            "bad_args": "llama-server rejected its start-up settings, even the basic ones.",
            "port": "llama-server couldn't open a network port on this computer.",
            "gpu": "llama-server had trouble with the graphics card and stopped.",
            "cant_execute": (
                "This computer can't run the prebuilt llama.cpp engine at all (for example a 32-bit system on a "
                "64-bit processor). Ollama or `pip install llama-cpp-python` may still work."
            ),
            "crash": "llama-server stopped as soon as it was given real work.",
        }
        code = f" (exit code {returncode})" if returncode is not None else ""
        default = f"llama-server stopped unexpectedly{code}."
        return messages.get(kind, default) + where


# ---------------------------------------------------------------------------
# Keeping the engine tied to the game's lifetime
# ---------------------------------------------------------------------------

_JOB_HANDLE: Any = None  # Windows: our "kill on close" Job Object (kept open for the game's lifetime)


def _windows_kill_on_close_job() -> Any:
    """Create (once) a Windows Job Object that kills its processes when closed.

    Windows closes the job's handle when the game's process ends - normally,
    by a crash, by Task Manager, or when the console window is closed (the
    case where no Python cleanup code gets to run) - and then ends every
    process in the job. Returns the handle, or None if it can't be created.
    """
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    if not kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    ):
        kernel32.CloseHandle(job)
        return None
    _JOB_HANDLE = job
    return job


def _assign_to_kill_on_close_job(proc: Any) -> bool:
    """Put a freshly started process into the kill-on-close job (Windows). Never raises."""
    try:
        import ctypes
        from ctypes import wintypes

        handle = getattr(proc, "_handle", None)
        if not handle:
            return False
        job = _windows_kill_on_close_job()
        if not job:
            return False
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        return bool(kernel32.AssignProcessToJobObject(job, int(handle)))
    except Exception:
        return False


def _parent_death_signal_hook() -> Optional[Callable[[], None]]:
    """Linux: a pre-exec hook asking the kernel to SIGTERM the engine if the game dies.

    ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` covers the case where the game is
    killed outright (no cleanup code runs). The C function is looked up here,
    in the parent, so the hook itself does nothing but one system call.
    """
    if platform.system() != "Linux":
        return None
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except Exception:
        return None
    pr_set_pdeathsig = 1
    sigterm = int(signal.SIGTERM)

    def hook() -> None:
        prctl(pr_set_pdeathsig, sigterm, 0, 0, 0)

    return hook


def _posix_locks_available() -> bool:
    """True where `flock` exists (Linux, macOS); False on Windows."""
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return False
    return True


def _lock_exclusively(fh: Any) -> bool:
    """Try to take an exclusive lock on an open file (POSIX). True if we hold it.

    The engine inherits the file, so the lock lasts as long as either the game
    or its engine has it open. Without ``flock`` (Windows) we can't lock, so
    the answer is no - and each game uses its own log file there.
    """
    try:
        import fcntl
    except ImportError:
        return False
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _tidy_old_logs(log_dir: Path) -> None:
    """Delete per-port logs (``llama-server-<port>.log``) older than LOG_KEEP_DAYS."""
    cutoff = time.time() - LOG_KEEP_DAYS * 86400
    with contextlib.suppress(OSError):
        for path in log_dir.glob("llama-server-*.log"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()


def _process_started(pid: int) -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _record_owner(log_dir: Path, proc: Any, exe: Path) -> Optional[Path]:
    """Write a small "this game started this engine" record next to the logs."""
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int):
        return None
    record = {
        "pid": pid,
        "started": _process_started(pid),
        "exe": str(exe),
        "game_pid": os.getpid(),
        "game_started": _process_started(os.getpid()),
    }
    path = log_dir / f"llama-server-{os.getpid()}-{pid}{OWNER_SUFFIX}"
    try:
        path.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        return None
    return path


def _same_process(pid: Any, started: Any) -> bool:
    """Is process `pid` still alive *and* the same one that started at `started`?"""
    if not isinstance(pid, int):
        return False
    now_started = _process_started(pid)
    if now_started is None:
        return False
    return not isinstance(started, (int, float)) or abs(now_started - float(started)) < 2.0


def engines_in_use(log_dir: Path) -> set[Path]:
    """The ``llama-server`` programs running right now for some copy of the game
    (from the owner records next to the logs). Their folders must not be deleted."""
    found: set[Path] = set()
    try:
        records = list(Path(log_dir).glob(f"llama-server-*{OWNER_SUFFIX}"))
    except OSError:
        return found
    for path in records:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("exe"), str):
            if _same_process(record.get("pid"), record.get("started")) or record.get("started") is None:
                found.add(Path(record["exe"]))
    return found


def reap_orphaned_servers(log_dir: Path) -> int:
    """Stop engines whose game is gone (e.g. force-quit); returns how many were stopped.

    Each running engine has an owner record (see ``_record_owner``). If the
    game that wrote it is no longer running but its engine still is - and it
    really is a llama-server started at the recorded time, not some other
    program that got the same process number - we stop it. Records of
    engines that have already gone are simply tidied away.
    """
    try:
        import psutil
    except ImportError:
        return 0
    stopped = 0
    try:
        records = list(Path(log_dir).glob(f"llama-server-*{OWNER_SUFFIX}"))
    except OSError:
        return 0
    for path in records:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        if not isinstance(record, dict):
            continue
        if _same_process(record.get("game_pid"), record.get("game_started")):
            continue  # that game is still running (maybe it's us): leave its engine alone
        pid = record.get("pid")
        if _same_process(pid, record.get("started")):
            try:
                child = psutil.Process(pid)
                if "llama-server" in (child.name() or "").lower():
                    child.terminate()
                    try:
                        child.wait(timeout=STOP_TIMEOUT_S)
                    except psutil.TimeoutExpired:
                        child.kill()
                    stopped += 1
            except Exception:
                pass
        with contextlib.suppress(OSError):
            path.unlink()
    return stopped
